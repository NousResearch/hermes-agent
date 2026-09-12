use anyhow::Result;
use crossterm::event::{Event, EventStream, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::SetTitle;
use futures_util::StreamExt;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::Style,
    text::Span,
    widgets::{Block, Borders},
    Terminal,
};
use ratatui_textarea::TextArea;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};
use tokio::time::{interval, Duration, Interval, MissedTickBehavior};

use crate::optimistic;
use crate::rpc::GatewayClient;
use crate::slash::{executable_entries, parse_slash, SlashKind};
use crate::state::{
    parse_gateway_messages, AppState, ArmedKind, BusyMode, HoverKind, PermissionMode, StatusBarMode,
};
use crate::terminal::Backend;
use crate::ui::queue::queue_height;
use crate::ui::{
    AgentDock, AttachPreview, ChatScrollback, CodeHighlighter, CompletePopup, DiffPane, FilesPane,
    Footer, JumpChip, QueuePane, SlashPopup, StreamCache, Theme, TipBar, TracePane, TurnBar,
    ViewsOverlay, WorkPane,
};

mod commands;
use commands::*;
mod keys;
pub(crate) use keys::*;

pub(crate) fn idle_placeholder() -> &'static str {
    crate::ui::copy::idle_placeholder()
}
pub(crate) const TITLE: &str = " ⚕ ";
const ANIM_TICK_MS: u64 = 50;
const IDLE_TICK_MS: u64 = 500;
const BLINK_FRAMES: u64 = 10; // 50ms * 10 = 500ms caret blink while busy

fn new_ticks(ms: u64) -> Interval {
    let mut ticks = interval(Duration::from_millis(ms));
    ticks.set_missed_tick_behavior(MissedTickBehavior::Skip);
    ticks
}

pub async fn run(
    terminal: &mut Terminal<Backend>,
    state: Arc<Mutex<AppState>>,
    client: Arc<GatewayClient>,
    redraw: Arc<Notify>,
) -> Result<()> {
    let highlighter = CodeHighlighter::new();
    let mut stream_cache = StreamCache::new();
    let mut textarea = styled_textarea(false);
    let mut events = EventStream::new();
    let mut tick_ms = ANIM_TICK_MS;
    let mut ticks = new_ticks(tick_ms);
    let mut frame_count: u64 = 0;
    let mut prompt_busy = false;
    let mut prompt_hover = false;
    let mut last_blink: u64 = 0;

    draw(
        terminal,
        &state,
        &mut textarea,
        &highlighter,
        &mut stream_cache,
        frame_count,
    )
    .await?;
    {
        let st = state.clone();
        let cl = client.clone();
        tokio::spawn(async move {
            let resume = st.lock().await.startup_resume.take();
            if let Some(id) = resume {
                resume_session(&st, &cl, &id).await;
            } else {
                maybe_auto_resume(&st, &cl).await;
            }
            refresh_models(&st, &cl).await;
            refresh_catalog(&st, &cl).await;
        });
    }

    loop {
        tokio::select! {
            _ = ticks.tick() => {
                frame_count = frame_count.wrapping_add(1);
                let (generating, hover_composer, animating, placeholder) = {
                    let mut s = state.lock().await;
                    s.advance_boot();
                    s.rotate_tip_if_due();
                    s.release_holds();
                    (
                        s.is_generating,
                        s.hover == HoverKind::Composer,
                        s.needs_animation() || s.dirty,
                        composer_placeholder(&s),
                    )
                };
                let deny = {
                    let mut s = state.lock().await;
                    if s.metrics.permission_mode == PermissionMode::Plan {
                        s.pending_approval
                            .take()
                            .map(|r| (s.session_id.clone(), r.request_id))
                    } else {
                        None
                    }
                };
                if let Some((Some(sid), req_id)) = deny {
                    optimistic::spawn_plan_deny(&state, &client, sid, req_id);
                }
                if generating != prompt_busy || hover_composer != prompt_hover {
                    restyle_prompt(&mut textarea, generating, hover_composer, &placeholder);
                    prompt_busy = generating;
                    prompt_hover = hover_composer;
                    if !generating {
                        let next = state.lock().await.take_queued();
                        if let Some(text) = next {
                            let _ = submit_line(text, &state, &client, &mut textarea).await;
                            prompt_busy = true;
                            restyle_prompt(
                                &mut textarea,
                                true,
                                false,
                                crate::ui::copy::BUSY_PLACEHOLDER,
                            );
                        }
                    }
                }
                let want_ms = if animating { ANIM_TICK_MS } else { IDLE_TICK_MS };
                if want_ms != tick_ms {
                    tick_ms = want_ms;
                    ticks = new_ticks(tick_ms);
                }
                let blink = (frame_count / BLINK_FRAMES) % 2;
                if animating || blink != last_blink {
                    last_blink = blink;
                    set_cursor_blink(&mut textarea, blink == 0);
                    draw(
                        terminal,
                        &state,
                        &mut textarea,
                        &highlighter,
                        &mut stream_cache,
                        frame_count,
                    )
                    .await?;
                }
            }
            _ = redraw.notified() => {
                frame_count = frame_count.wrapping_add(1);
                draw(
                    terminal,
                    &state,
                    &mut textarea,
                    &highlighter,
                    &mut stream_cache,
                    frame_count,
                )
                .await?;
            }
            maybe = events.next() => {
                match maybe {
                    Some(Ok(Event::Resize(cols, _))) => {
                        state.lock().await.mark_dirty();
                        let sid = state.lock().await.session_id.clone();
                        if let Some(sid) = sid {
                            let cl = client.clone();
                            tokio::spawn(async move {
                                let _ = cl.terminal_resize(&sid, cols).await;
                            });
                        }
                        draw(
                            terminal,
                            &state,
                            &mut textarea,
                            &highlighter,
                            &mut stream_cache,
                            frame_count,
                        )
                        .await?;
                    }
                    Some(Ok(Event::Paste(pasted))) => {
                        handle_paste(&pasted, &state, &client, &mut textarea).await;
                        set_cursor_blink(&mut textarea, true);
                        last_blink = 0;
                        draw(
                            terminal,
                            &state,
                            &mut textarea,
                            &highlighter,
                            &mut stream_cache,
                            frame_count,
                        )
                        .await?;
                    }
                    Some(Ok(Event::Mouse(mouse))) => {
                        if handle_mouse(mouse, &state, &client, &mut textarea).await {
                            let s = state.lock().await;
                            restyle_prompt(
                                &mut textarea,
                                s.is_generating,
                                s.hover == HoverKind::Composer,
                                &composer_placeholder(&s),
                            );
                            drop(s);
                            set_cursor_blink(&mut textarea, true);
                            last_blink = 0;
                            draw(
                                terminal,
                                &state,
                                &mut textarea,
                                &highlighter,
                                &mut stream_cache,
                                frame_count,
                            )
                            .await?;
                        }
                    }
                    Some(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => {
                        match handle_key(key, &state, &client, &mut textarea).await? {
                            LoopControl::Quit => break,
                            LoopControl::MouseToggle => {
                                let on = {
                                    let mut s = state.lock().await;
                                    s.mouse_on = !s.mouse_on;
                                    let on = s.mouse_on;
                                    s.set_toast(if on {
                                        "mouse capture on"
                                    } else {
                                        "mouse capture off · terminal select works"
                                    });
                                    on
                                };
                                if on {
                                    let _ = execute!(
                                        std::io::stdout(),
                                        crossterm::event::EnableMouseCapture
                                    );
                                } else {
                                    let _ = execute!(
                                        std::io::stdout(),
                                        crossterm::event::DisableMouseCapture
                                    );
                                }
                                set_cursor_blink(&mut textarea, true);
                                last_blink = 0;
                                draw(
                                    terminal,
                                    &state,
                                    &mut textarea,
                                    &highlighter,
                                    &mut stream_cache,
                                    frame_count,
                                )
                                .await?;
                            }
                            LoopControl::Editor => {
                                let (memory_id, seed) = {
                                    let mut s = state.lock().await;
                                    let id = s.pending_memory_edit.take();
                                    let body = std::mem::take(&mut s.pending_memory_body);
                                    let seed = if id.is_some() {
                                        body
                                    } else {
                                        textarea.lines().join("\n")
                                    };
                                    (id, seed)
                                };
                                match crate::terminal::edit_in_external_editor(terminal, &seed) {
                                    Ok(Some(text)) => {
                                        if let Some(id) = memory_id {
                                            apply_memory_edit(&state, &client, &id, &text).await;
                                        } else {
                                            set_prompt_text(&mut textarea, text.trim_end());
                                            state.lock().await.set_toast("editor · enter to send");
                                        }
                                    }
                                    Ok(None) => {
                                        state.lock().await.set_toast("editor cancelled");
                                    }
                                    Err(e) => {
                                        state.lock().await.set_toast(format!(
                                            "editor failed · {}",
                                            optimistic::brief_err(&e)
                                        ));
                                    }
                                }
                                set_cursor_blink(&mut textarea, true);
                                last_blink = 0;
                                draw(
                                    terminal,
                                    &state,
                                    &mut textarea,
                                    &highlighter,
                                    &mut stream_cache,
                                    frame_count,
                                )
                                .await?;
                            }
                            LoopControl::Continue => {
                                set_cursor_blink(&mut textarea, true);
                                last_blink = 0;
                                draw(
                                    terminal,
                                    &state,
                                    &mut textarea,
                                    &highlighter,
                                    &mut stream_cache,
                                    frame_count,
                                )
                                .await?;
                            }
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => return Err(e.into()),
                    None => break,
                }
            }
        }
    }
    Ok(())
}

pub(crate) enum LoopControl {
    Continue,
    Quit,
    Editor,
    MouseToggle,
}

pub(crate) fn styled_textarea(streaming: bool) -> TextArea<'static> {
    let mut textarea = TextArea::default();
    textarea.set_placeholder_text(idle_placeholder());
    textarea.set_style(
        Style::default()
            .fg(Theme::text_primary())
            .bg(Theme::bg_surface()),
    );
    // tui-textarea underlines the cursor line by default — Grok Build does not.
    textarea.set_cursor_line_style(Style::default());
    textarea.set_placeholder_style(Style::default().fg(Theme::text_muted()));
    restyle_prompt(
        &mut textarea,
        streaming,
        false,
        if streaming {
            crate::ui::copy::BUSY_PLACEHOLDER
        } else {
            idle_placeholder()
        },
    );
    set_cursor_blink(&mut textarea, true);
    textarea
}

pub(crate) fn composer_placeholder(s: &AppState) -> String {
    if s.pending_approval.is_some() {
        "y allow once · a always · n deny".into()
    } else if s.pending_clarify.is_some() {
        "enter to pick · esc dismiss".into()
    } else if s.pending_secret.is_some() {
        "enter submit · esc cancel".into()
    } else if s.is_generating {
        crate::ui::copy::BUSY_PLACEHOLDER.into()
    } else if !s.prompt_queue.is_empty() {
        format!("{} queued · enter send now", s.prompt_queue.len())
    } else {
        idle_placeholder().into()
    }
}

pub(crate) fn restyle_prompt(
    textarea: &mut TextArea<'static>,
    streaming: bool,
    composer_hover: bool,
    placeholder: &str,
) {
    let border = if streaming || composer_hover {
        Theme::brand_gold()
    } else {
        Theme::border_focus()
    };
    textarea.set_style(
        Style::default()
            .fg(Theme::text_primary())
            .bg(Theme::bg_surface()),
    );
    textarea.set_placeholder_text(placeholder);
    textarea.set_placeholder_style(Style::default().fg(Theme::text_muted()));
    textarea.set_cursor_line_style(Style::default());
    textarea.set_block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border))
            .style(Style::default().bg(Theme::bg_surface()))
            .title(Span::styled(
                TITLE,
                Style::default().fg(Theme::brand_gold()),
            )),
    );
}

pub(crate) fn set_cursor_blink(textarea: &mut TextArea<'static>, on: bool) {
    let style = if on {
        Style::default()
            .fg(Theme::bg_base())
            .bg(Theme::brand_gold())
    } else {
        Style::default()
            .fg(Theme::text_primary())
            .bg(Theme::bg_surface())
    };
    textarea.set_cursor_style(style);
}

pub(crate) fn reset_prompt(streaming: bool) -> TextArea<'static> {
    styled_textarea(streaming)
}

pub(crate) fn set_prompt_text(textarea: &mut TextArea<'static>, text: &str) {
    *textarea = reset_prompt(false);
    if !text.is_empty() {
        textarea.insert_str(text);
    }
}

pub(crate) async fn new_session(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    let (old, cwd) = {
        let s = state.lock().await;
        (s.session_id.clone(), s.metrics.cwd.clone())
    };
    if let Some(old) = old {
        let _ = client.close_session(&old).await;
    }
    match client
        .create_session("Hermes TUI", std::path::Path::new(&cwd))
        .await
    {
        Ok(sid) => {
            let mut s = state.lock().await;
            s.reset_session(sid);
            s.set_toast("new session");
        }
        Err(e) => {
            state
                .lock()
                .await
                .add_system(format!("session.create failed: {e}"));
        }
    }
}

pub(crate) async fn resume_session(state: &Mutex<AppState>, client: &Arc<GatewayClient>, id: &str) {
    match client.resume_session(id).await {
        Ok(v) => {
            let live_id = v
                .get("session_id")
                .and_then(|s| s.as_str())
                .unwrap_or(id)
                .to_string();
            let msgs = v
                .get("messages")
                .and_then(|m| m.as_array())
                .cloned()
                .unwrap_or_default();
            let parsed = parse_gateway_messages(&msgs);
            let mut s = state.lock().await;
            s.session_id = Some(live_id);
            s.messages = parsed;
            s.scroll_from_bottom = 0;
            s.is_generating = false;
            if let Some(info) = v.get("info") {
                // session.info shape is applied via events; hydrate what we can.
                if let Some(model) = info.get("model").and_then(|m| m.as_str()) {
                    s.metrics.active_model = model.to_string();
                }
            }
            if let Some(recap) = crate::shell::recap(&s.messages) {
                s.add_system(recap);
            }
            s.set_toast("Session resumed");
        }
        Err(e) => {
            state
                .lock()
                .await
                .add_system(format!("session.resume failed: {e}"));
        }
    }
}

pub(crate) async fn activate_session(
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
    id: &str,
) {
    let current = state.lock().await.session_id.clone();
    if current.as_deref() == Some(id) {
        state.lock().await.set_toast("already this session");
        return;
    }
    match client.session_activate(id).await {
        Ok(v) => {
            let live_id = v
                .get("session_id")
                .and_then(|s| s.as_str())
                .unwrap_or(id)
                .to_string();
            let msgs = v
                .get("messages")
                .and_then(|m| m.as_array())
                .cloned()
                .unwrap_or_default();
            let parsed = parse_gateway_messages(&msgs);
            let running = v.get("running").and_then(|x| x.as_bool()).unwrap_or(false)
                || matches!(
                    v.get("status").and_then(|x| x.as_str()),
                    Some("working") | Some("waiting")
                );
            let mut s = state.lock().await;
            s.session_id = Some(live_id);
            s.messages = parsed;
            s.scroll_from_bottom = 0;
            s.is_generating = running;
            if let Some(info) = v.get("info") {
                if let Some(model) = info.get("model").and_then(|m| m.as_str()) {
                    s.metrics.active_model = model.to_string();
                }
            }
            s.set_toast("switched live session");
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("activate · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn draw(
    terminal: &mut Terminal<Backend>,
    state: &Mutex<AppState>,
    textarea: &mut TextArea<'static>,
    highlighter: &CodeHighlighter,
    stream_cache: &mut StreamCache,
    frame_count: u64,
) -> Result<()> {
    crate::ui::theme::snapshot();
    let mut s = state.lock().await;
    if s.want_attention {
        s.want_attention = false;
        crate::terminal::request_attention("hermes needs you");
    }
    restyle_prompt(
        textarea,
        s.is_generating,
        s.hover == HoverKind::Composer,
        &composer_placeholder(&s),
    );
    s.dirty = false;
    let cols = terminal.size()?.width;
    s.trace_open = s.want_trace(cols);
    let show_diff = s.want_diff(cols);
    let show_files = s.want_files(cols);
    let show_work = s.want_work(cols);
    terminal.draw(|f| {
        let tip_h = if s.tips_open { TipBar::HEIGHT } else { 0 };
        let turn_h = TurnBar::height(&s);
        let scan_h = TurnBar::scan_height(&s);
        let q_h = queue_height(&s);
        let attach_h = AttachPreview::height(&s.pending_images, s.paste_chips.len());
        let dock_h = crate::ui::dock::dock_height(&s);
        let footer_h = Footer::height(&s);
        let composer_h = if s.compact { 2 } else { 3 };
        let jump_h = if s.scrolled_off_tail() {
            JumpChip::HEIGHT
        } else {
            0
        };
        let footer_top = s.status_bar == StatusBarMode::Top && footer_h > 0;
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(tip_h), Constraint::Min(0)])
            .split(f.area());
        TipBar::render(f, outer[0], &mut s);
        let mut constraints = Vec::new();
        if footer_top {
            constraints.push(Constraint::Length(footer_h));
        }
        constraints.extend([
            Constraint::Min(5),
            Constraint::Length(q_h),
            Constraint::Length(attach_h),
            Constraint::Length(jump_h),
            Constraint::Length(turn_h),
            Constraint::Length(dock_h),
            Constraint::Length(composer_h),
        ]);
        if !footer_top {
            constraints.push(Constraint::Length(footer_h));
        }
        constraints.push(Constraint::Length(scan_h));
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(outer[1]);
        let i = usize::from(footer_top);
        let stream = chunks[i];
        match (show_files, show_work, show_diff, s.trace_open) {
            (true, _, _, true) if stream.width >= 110 => {
                let fw = (stream.width / 3).clamp(30, 42);
                let tw = (stream.width / 4).clamp(24, 36);
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Min(20),
                        Constraint::Length(tw),
                        Constraint::Length(fw),
                    ])
                    .split(stream);
                ChatScrollback::render(f, cols[0], &mut s, highlighter, stream_cache, frame_count);
                TracePane::render(f, cols[1], &s, frame_count);
                FilesPane::render(f, cols[2], &mut s);
            }
            (true, _, _, _) => {
                let fw = (stream.width / 3).clamp(30, 44);
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Min(20), Constraint::Length(fw)])
                    .split(stream);
                ChatScrollback::render(f, cols[0], &mut s, highlighter, stream_cache, frame_count);
                FilesPane::render(f, cols[1], &mut s);
            }
            (false, true, _, true) if stream.width >= 110 => {
                let ww = (stream.width / 3).clamp(30, 44);
                let tw = (stream.width / 4).clamp(24, 36);
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Min(20),
                        Constraint::Length(tw),
                        Constraint::Length(ww),
                    ])
                    .split(stream);
                ChatScrollback::render(f, cols[0], &mut s, highlighter, stream_cache, frame_count);
                TracePane::render(f, cols[1], &s, frame_count);
                WorkPane::render(f, cols[2], &mut s, frame_count);
            }
            (false, true, _, _) => {
                let ww = (stream.width / 3).clamp(30, 46);
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Min(20), Constraint::Length(ww)])
                    .split(stream);
                ChatScrollback::render(f, cols[0], &mut s, highlighter, stream_cache, frame_count);
                WorkPane::render(f, cols[1], &mut s, frame_count);
            }
            (false, false, true, true) if stream.width >= 110 => {
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Min(24),
                        Constraint::Length((stream.width / 3).clamp(28, 42)),
                        Constraint::Length((stream.width / 4).clamp(24, 36)),
                    ])
                    .split(stream);
                ChatScrollback::render(f, cols[0], &mut s, highlighter, stream_cache, frame_count);
                DiffPane::render(f, cols[1], &s);
                TracePane::render(f, cols[2], &s, frame_count);
            }
            (false, false, true, _) => {
                let dw = (stream.width / 3).clamp(28, 46);
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Min(20), Constraint::Length(dw)])
                    .split(stream);
                ChatScrollback::render(f, cols[0], &mut s, highlighter, stream_cache, frame_count);
                DiffPane::render(f, cols[1], &s);
            }
            (false, false, false, true) => {
                let trace_w = (stream.width / 3).clamp(28, 42);
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Min(20), Constraint::Length(trace_w)])
                    .split(stream);
                ChatScrollback::render(f, cols[0], &mut s, highlighter, stream_cache, frame_count);
                TracePane::render(f, cols[1], &s, frame_count);
            }
            (false, false, false, false) => {
                ChatScrollback::render(f, stream, &mut s, highlighter, stream_cache, frame_count);
            }
        }
        QueuePane::render(f, chunks[i + 1], &mut s);
        AttachPreview::render(f, chunks[i + 2], &mut s);
        JumpChip::render(f, chunks[i + 3], &mut s);
        TurnBar::render(f, chunks[i + 4], &s, frame_count);
        if s.metrics.is_compacting && chunks[i + 4].height > 0 {
            s.metrics.compaction_painted = true;
        }
        s.queue_area = Some(chunks[i + 1]);
        let dock_area = chunks[i + 5];
        s.composer_area = Some(chunks[i + 6]);
        AgentDock::render(f, dock_area, &mut s, frame_count);
        f.render_widget(&*textarea, chunks[i + 6]);
        let footer_area = if footer_top { chunks[0] } else { chunks[i + 7] };
        Footer::render(f, footer_area, &mut s, frame_count);
        TurnBar::render_scan(f, chunks[chunks.len() - 1], &s, frame_count);

        if s.slash_open {
            let ranked = s.slash_ranked();
            SlashPopup::render(f, chunks[i + 6], s.slash_selected, &ranked);
        }
        if s.complete_open && !s.slash_open {
            CompletePopup::render(f, chunks[i + 6], &s.complete_items, s.complete_selected);
        }
        ViewsOverlay::render(f, f.area(), &mut s, frame_count);
    })?;
    let title = s.tab_title();
    execute!(terminal.backend_mut(), SetTitle(title))?;
    Ok(())
}

pub(crate) async fn sync_composer(
    textarea: &TextArea<'_>,
    state: &Mutex<AppState>,
    client: Option<&Arc<GatewayClient>>,
) {
    let text = textarea.lines().join("\n");
    let mut s = state.lock().await;
    s.refresh_pending_images(&text);
    if crate::slash::looks_like_slash_command(&text) && !text.contains(' ') {
        s.slash_open = true;
        s.slash_query = text.clone();
        s.slash_selected = 0;
        s.close_complete();
        s.mark_dirty();
        drop(s);
        refresh_slash_complete(state, client, &text).await;
        return;
    } else if s.slash_open && !text.starts_with('/') {
        s.slash_open = false;
        s.slash_gateway.clear();
    } else if s.slash_open {
        s.slash_query = text.clone();
        s.close_complete();
        s.mark_dirty();
        drop(s);
        refresh_slash_complete(state, client, &text).await;
        return;
    }
    let trigger = crate::complete::path_trigger(&text);
    let cwd = s.metrics.cwd.clone();
    let sid = s.session_id.clone().unwrap_or_default();
    drop(s);
    let Some(trigger) = trigger else {
        state.lock().await.close_complete();
        return;
    };
    let items = if let Some(c) = client {
        match c.complete_path(&trigger.word, &sid, &cwd).await {
            Ok(v) => {
                let parsed = crate::complete::parse_items(&v);
                if parsed.is_empty() {
                    crate::complete::local_items(&trigger.word, &cwd)
                } else {
                    parsed
                }
            }
            Err(_) => crate::complete::local_items(&trigger.word, &cwd),
        }
    } else {
        crate::complete::local_items(&trigger.word, &cwd)
    };
    state.lock().await.set_complete(items, trigger.replace_from);
}

pub(crate) async fn send_now(
    mut text: String,
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
) -> Result<LoopControl> {
    let mut s = state.lock().await;
    if let Some(orig) = s.take_queue_edit() {
        if text.trim().is_empty() {
            text = orig;
        }
    } else if text.trim().is_empty() {
        if let Some(next) = s.take_queued() {
            text = next;
        }
    }
    let text = text.trim().to_string();
    if text.is_empty() {
        return Ok(LoopControl::Continue);
    }
    if s.is_generating {
        let sid = s.session_id.clone();
        drop(s);
        if let Some(sid) = sid {
            optimistic::spawn_steer(state, client, sid, text);
        }
        *textarea = reset_prompt(true);
        state.lock().await.set_toast("sent now");
        return Ok(LoopControl::Continue);
    }
    drop(s);
    submit_line(text, state, client, textarea).await
}

pub(crate) async fn submit_line(
    text: String,
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
) -> Result<LoopControl> {
    if text == "/exit" || text == "/quit" {
        let mut s = state.lock().await;
        if s.has_unsaved("") && !s.take_armed(ArmedKind::Quit) {
            s.arm(ArmedKind::Quit, "queued prompts · /exit again to quit");
            return Ok(LoopControl::Continue);
        }
        return Ok(LoopControl::Quit);
    }
    if text.starts_with('/') {
        let (name, arg) = parse_slash(&text);
        let full = format!("/{name}");
        let (home, cwd) = {
            let s = state.lock().await;
            (s.hermes_home.clone(), s.metrics.cwd.clone())
        };
        if let Some(line) = crate::skill_md::expand_nested_slash(&full, &arg, &home, &cwd) {
            return gateway_slash(state, client, textarea, &line).await;
        }
        let catalog = state.lock().await.slash_catalog.clone();
        if let Some(cmd) = catalog.iter().find(|c| c.name.eq_ignore_ascii_case(&full)) {
            if cmd.kind == SlashKind::Local {
                return dispatch_slash(&cmd.name, &arg, state, client, textarea).await;
            }
            let line = if arg.is_empty() {
                cmd.name.clone()
            } else {
                format!("{} {arg}", cmd.name)
            };
            return gateway_slash(state, client, textarea, &line).await;
        }
        let matches = executable_entries(&name, &catalog);
        if matches.len() == 1 {
            let cmd = matches[0];
            if cmd.kind == SlashKind::Local {
                return dispatch_slash(&cmd.name, &arg, state, client, textarea).await;
            }
            let line = if arg.is_empty() {
                cmd.name.clone()
            } else {
                format!("{} {arg}", cmd.name)
            };
            return gateway_slash(state, client, textarea, &line).await;
        }
        if matches.len() > 1 {
            let names: Vec<&str> = matches.iter().map(|c| c.name.as_str()).collect();
            state
                .lock()
                .await
                .add_system(format!("ambiguous command: {}", names.join(", ")));
            *textarea = reset_prompt(false);
            return Ok(LoopControl::Continue);
        }
        return dispatch_slash(&full, &arg, state, client, textarea).await;
    }

    if let Some(cmd) = crate::shell::bang_command(&text) {
        return run_bang(cmd.to_string(), state, client, textarea).await;
    }

    send_user_turn(text, state, client, textarea).await
}

pub(crate) async fn send_user_turn(
    text: String,
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
) -> Result<LoopControl> {
    let mut s = state.lock().await;
    let resume = s.resume_step.take();
    let text = if let Some(idx) = resume {
        s.tool_steps()
            .into_iter()
            .find(|t| t.index == idx)
            .map(|step| crate::state::resume_from_step_prompt(&step, &text))
            .unwrap_or(text)
    } else {
        text
    };
    if s.is_generating {
        if resume.is_some() {
            let sid = s.session_id.clone();
            drop(s);
            if let Some(sid) = sid {
                optimistic::spawn_steer(state, client, sid, text);
            }
            *textarea = reset_prompt(true);
            state.lock().await.set_toast("steered from step");
            return Ok(LoopControl::Continue);
        }
        match s.busy_mode {
            BusyMode::Queue => {
                s.enqueue(text);
                drop(s);
                *textarea = reset_prompt(true);
                return Ok(LoopControl::Continue);
            }
            BusyMode::Steer => {
                let sid = s.session_id.clone();
                drop(s);
                if let Some(sid) = sid {
                    optimistic::spawn_steer(state, client, sid, text);
                }
                *textarea = reset_prompt(true);
                state.lock().await.set_toast("steered");
                return Ok(LoopControl::Continue);
            }
            BusyMode::Interrupt => {
                let sid = s.session_id.clone();
                drop(s);
                if let Some(sid) = sid {
                    optimistic::spawn_redirect(state, client, sid, text);
                }
                *textarea = reset_prompt(true);
                state.lock().await.set_toast("redirected live turn");
                return Ok(LoopControl::Continue);
            }
        }
    }
    let chips = s.paste_chips.clone();
    drop(s);
    let visible = text.clone();
    let expanded = crate::paste::expand(&text, &chips);
    let expanded = apply_detect_drop(expanded, state, client).await;
    let gateway = attach_images_for_send(expanded, state, client).await;
    let mut s = state.lock().await;
    s.start_turn(visible);
    let plan = s.metrics.permission_mode == PermissionMode::Plan;
    let ctx = std::mem::take(&mut s.shell_context);
    let sid = s.session_id.clone();
    drop(s);
    let gateway = crate::shell::wrap_prompt(plan, &ctx, &gateway);
    *textarea = reset_prompt(true);
    if let Some(sid) = sid {
        optimistic::spawn_submit(state, client, sid, gateway);
    }
    Ok(LoopControl::Continue)
}

pub(crate) async fn submit_dispatch_message(
    visible: String,
    payload: String,
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
) -> Result<LoopControl> {
    let mut s = state.lock().await;
    if s.is_generating {
        s.enqueue(payload);
        drop(s);
        *textarea = reset_prompt(true);
        state
            .lock()
            .await
            .set_toast("session busy — queued until this turn finishes");
        return Ok(LoopControl::Continue);
    }
    s.start_turn(visible);
    let plan = s.metrics.permission_mode == PermissionMode::Plan;
    let ctx = std::mem::take(&mut s.shell_context);
    let sid = s.session_id.clone();
    drop(s);
    let gateway = crate::shell::wrap_prompt(plan, &ctx, &payload);
    *textarea = reset_prompt(true);
    if let Some(sid) = sid {
        optimistic::spawn_submit(state, client, sid, gateway);
    }
    Ok(LoopControl::Continue)
}

pub(crate) async fn attach_images_for_send(
    text: String,
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
) -> String {
    let (cwd, sid) = {
        let s = state.lock().await;
        (s.metrics.cwd.clone(), s.session_id.clone())
    };
    let (rewritten, paths) = crate::complete::rewrite_image_tokens(&text, &cwd);
    let Some(sid) = sid else {
        return rewritten;
    };
    for path in &paths {
        match client.image_attach(&sid, &path.to_string_lossy()).await {
            Ok(v) if v.get("attached").and_then(|x| x.as_bool()) == Some(true) => {}
            Ok(_) | Err(_) => {
                state.lock().await.set_toast(format!(
                    "image skipped · {} · check the path",
                    path.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default()
                ));
            }
        }
    }
    rewritten
}

pub(crate) fn paste_lead(textarea: &TextArea<'_>) -> String {
    let input = textarea.lines().join("\n");
    if input.is_empty() || input.ends_with(' ') || input.ends_with('\n') {
        String::new()
    } else {
        " ".into()
    }
}

pub(crate) async fn handle_paste(
    pasted: &str,
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
) {
    let trimmed = pasted.trim();
    if crate::complete::looks_like_dropped_image(trimmed) {
        let sid = state.lock().await.session_id.clone();
        if let Some(sid) = sid {
            if let Ok(v) = client.image_attach(&sid, trimmed).await {
                if v.get("attached").and_then(|x| x.as_bool()) == Some(true) {
                    let input = textarea.lines().join("\n");
                    let tok =
                        crate::complete::image_token(crate::complete::next_image_index(&input));
                    textarea.insert_str(format!("{}{tok} ", paste_lead(textarea)));
                    let path = v.get("path").and_then(|s| s.as_str()).unwrap_or(trimmed);
                    let mut s = state.lock().await;
                    s.remember_image(std::path::PathBuf::from(path));
                    s.set_toast(format!(
                        "attached {} · click [[ ]] to preview",
                        v.get("name").and_then(|s| s.as_str()).unwrap_or("image")
                    ));
                    return;
                }
            }
        }
    }
    if crate::complete::looks_like_dropped_path(trimmed) {
        if let Some(inserted) = detect_drop_paste(state, client, trimmed).await {
            textarea.insert_str(format!("{}{inserted} ", paste_lead(textarea)));
            return;
        }
    }
    if crate::paste::should_collapse(pasted) {
        let label = {
            let mut s = state.lock().await;
            s.remember_paste(pasted.to_string())
        };
        textarea.insert_str(format!("{}{label} ", paste_lead(textarea)));
        persist_paste_collapse(state, client, &label, pasted).await;
        state.lock().await.set_toast(format!(
            "pasted {} lines · click [[ ]] to preview",
            pasted.lines().count().max(1)
        ));
        return;
    }
    textarea.insert_str(pasted);
    sync_composer(textarea, state, Some(client)).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ActiveView, PickerStage};
    use ratatui::layout::Rect;

    #[test]
    fn busy_placeholder_is_the_action_only() {
        let mut s = AppState::new();
        s.is_generating = true;
        s.busy_mode = BusyMode::Queue;
        assert_eq!(composer_placeholder(&s), crate::ui::copy::BUSY_PLACEHOLDER);
        s.busy_mode = BusyMode::Steer;
        assert_eq!(composer_placeholder(&s), crate::ui::copy::BUSY_PLACEHOLDER);
        s.busy_mode = BusyMode::Interrupt;
        assert_eq!(composer_placeholder(&s), crate::ui::copy::BUSY_PLACEHOLDER);
    }

    #[test]
    fn click_composer_jumps_cursor() {
        let mut ta = TextArea::from(["hello world"]);
        let area = Rect {
            x: 0,
            y: 10,
            width: 40,
            height: 3,
        };
        assert!(!click_composer(&mut ta, area, 0, 10), "border is not inner");
        assert!(click_composer(&mut ta, area, 7, 11));
        assert_eq!(ta.cursor(), (0, 6));
    }

    #[test]
    fn background_confirm_launch_and_peek() {
        let mut s = AppState::new();
        s.open_background();
        s.picker_filter = "  summarize hn  ".into();
        match background_confirm(&mut s) {
            BgAction::Launch(text) => assert_eq!(text, "summarize hn"),
            other => panic!("expected launch, got {other:?}"),
        }
        assert!(s.picker_filter.is_empty());

        s.start_bg_task("bg_x".into(), "lint".into());
        s.complete_bg_task("bg_x", "clean");
        s.modal_selected = 1;
        match background_confirm(&mut s) {
            BgAction::Peek { title, body } => {
                assert!(title.contains("bg_x"));
                assert!(body.contains("lint"));
                assert!(body.contains("clean"));
            }
            other => panic!("expected peek, got {other:?}"),
        }

        s.modal_selected = 0;
        s.picker_filter.clear();
        assert!(matches!(background_confirm(&mut s), BgAction::None));
    }

    #[test]
    fn picker_confirm_opens_inline_key_for_unconfigured_provider() {
        let mut s = AppState::new();
        s.active_view = ActiveView::ModelPicker;
        s.providers = vec![crate::state::ModelProvider {
            slug: "openrouter".into(),
            name: "OpenRouter".into(),
            models: vec![],
            authenticated: false,
            is_current: false,
            warning: "paste OPENROUTER_API_KEY".into(),
            auth_type: "api_key".into(),
            key_env: "OPENROUTER_API_KEY".into(),
        }];
        s.modal_selected = 0;
        assert!(matches!(picker_confirm(&mut s), PickerAction::None));
        assert_eq!(s.picker_stage, PickerStage::Key);
        assert_eq!(s.picker_provider, 0);
    }

    #[test]
    fn picker_confirm_toasts_oauth_without_inline_key() {
        let mut s = AppState::new();
        s.active_view = ActiveView::ModelPicker;
        s.providers = vec![crate::state::ModelProvider {
            slug: "nous".into(),
            name: "Nous".into(),
            models: vec![],
            authenticated: false,
            is_current: false,
            warning: "run hermes model".into(),
            auth_type: "oauth".into(),
            key_env: String::new(),
        }];
        s.modal_selected = 0;
        match picker_confirm(&mut s) {
            PickerAction::Toast(msg) => assert!(msg.contains("hermes model")),
            other => panic!("expected toast, got {other:?}"),
        }
        assert_eq!(s.picker_stage, PickerStage::Providers);
    }
}
