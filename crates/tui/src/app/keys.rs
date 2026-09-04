//! Keyboard dispatch: composer, overlays, and modal lists.
use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::sync::Arc;
use tokio::sync::Mutex;
use tui_textarea::TextArea;

use crate::optimistic;
use crate::rpc::GatewayClient;
use crate::slash::SlashKind;
use crate::state::{ActiveView, AppState, ArmedKind, PickerStage};

use super::commands::*;
use super::{
    activate_session, new_session, reset_prompt, resume_session, send_now, set_prompt_text,
    submit_line, sync_composer, LoopControl,
};

pub(crate) async fn handle_clarify_key(
    key: KeyEvent,
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
) -> Result<LoopControl> {
    {
        let mut s = state.lock().await;
        let Some(c) = s.pending_clarify.as_mut() else {
            return Ok(LoopControl::Continue);
        };
        match key.code {
            KeyCode::Esc => {
                s.pending_clarify = None;
                s.set_toast("clarify dismissed");
                return Ok(LoopControl::Continue);
            }
            KeyCode::Up => {
                c.selected = c.selected.saturating_sub(1);
                s.mark_dirty();
                return Ok(LoopControl::Continue);
            }
            KeyCode::Down => {
                let n = c.choices.len();
                if n > 0 {
                    c.selected = (c.selected + 1).min(n.saturating_sub(1));
                }
                s.mark_dirty();
                return Ok(LoopControl::Continue);
            }
            KeyCode::Char(ch) if ch.is_ascii_digit() && !c.choices.is_empty() => {
                let i = ch.to_digit(10).unwrap_or(0) as usize;
                if i >= 1 && i <= c.choices.len() {
                    c.selected = i - 1;
                }
            }
            KeyCode::Char(ch)
                if c.choices.is_empty() && !key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                c.typed.push(ch);
                s.mark_dirty();
                return Ok(LoopControl::Continue);
            }
            KeyCode::Backspace if c.choices.is_empty() => {
                c.typed.pop();
                s.mark_dirty();
                return Ok(LoopControl::Continue);
            }
            KeyCode::Enter => {}
            _ => return Ok(LoopControl::Continue),
        }
        let answer = if c.choices.is_empty() {
            c.typed.clone()
        } else {
            c.choices.get(c.selected).cloned().unwrap_or_default()
        };
        if answer.is_empty() {
            return Ok(LoopControl::Continue);
        }
        let rid = c.request_id.clone();
        let sid = s.session_id.clone();
        drop(s);
        if let Some(sid) = sid {
            if let Err(e) = client.clarify_respond(&sid, &rid, &answer).await {
                state
                    .lock()
                    .await
                    .add_system(format!("clarify failed: {e}"));
            } else {
                let mut s = state.lock().await;
                s.pending_clarify = None;
                s.set_toast("answered");
            }
        }
    }
    Ok(LoopControl::Continue)
}

pub(crate) async fn handle_secret_key(
    key: KeyEvent,
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
    _textarea: &mut TextArea<'static>,
) -> Result<LoopControl> {
    let submit = {
        let mut s = state.lock().await;
        let Some(sec) = s.pending_secret.as_mut() else {
            return Ok(LoopControl::Continue);
        };
        match key.code {
            KeyCode::Esc => {
                sec.buffer.clear();
                true
            }
            KeyCode::Enter => true,
            KeyCode::Backspace => {
                sec.buffer.pop();
                s.mark_dirty();
                false
            }
            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                sec.buffer.push(ch);
                s.mark_dirty();
                false
            }
            _ => false,
        }
    };
    if !submit {
        return Ok(LoopControl::Continue);
    }
    let (sid, kind, rid, value) = {
        let s = state.lock().await;
        let sec = s.pending_secret.as_ref();
        (
            s.session_id.clone(),
            sec.map(|x| x.kind),
            sec.map(|x| x.request_id.clone()).unwrap_or_default(),
            sec.map(|x| x.buffer.clone()).unwrap_or_default(),
        )
    };
    let Some(kind) = kind else {
        return Ok(LoopControl::Continue);
    };
    let (method, key_name) = match kind {
        crate::state::SecretKind::Sudo => ("sudo.respond", "password"),
        crate::state::SecretKind::Secret => ("secret.respond", "value"),
    };
    if let Some(sid) = sid {
        let _ = client
            .secret_respond(method, &sid, &rid, key_name, &value)
            .await;
    }
    let mut s = state.lock().await;
    s.pending_secret = None;
    s.set_toast("sent");
    Ok(LoopControl::Continue)
}

pub(crate) async fn handle_trace_key(
    key: KeyEvent,
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
) -> Result<LoopControl> {
    match key.code {
        KeyCode::Esc => {
            let mut s = state.lock().await;
            s.trace_focus = false;
            s.mark_dirty();
        }
        KeyCode::Char('a') => {
            let mut s = state.lock().await;
            s.trace_focus = false;
            s.agents_replay = false;
            s.active_view = ActiveView::Agents;
            s.modal_selected = 0;
            s.picker_filter.clear();
            s.agents_steer = false;
            s.mark_dirty();
            drop(s);
            refresh_agents(state, client).await;
        }
        KeyCode::Up | KeyCode::Char('k') => {
            let mut s = state.lock().await;
            s.trace_follow = false;
            s.trace_selected = s.trace_selected.saturating_sub(1);
            s.mark_dirty();
        }
        KeyCode::Down | KeyCode::Char('j') => {
            let mut s = state.lock().await;
            s.trace_follow = false;
            let n = s.tool_steps().len();
            if n > 0 {
                s.trace_selected = (s.trace_selected + 1).min(n - 1);
            }
            s.mark_dirty();
        }
        KeyCode::Char('y') => {
            let mut s = state.lock().await;
            if let Some(step) = s.selected_step() {
                match crate::platform::copy_to_clipboard(&step.args) {
                    Ok(()) => s.set_toast("copied step args"),
                    Err(e) => s.set_toast(format!("copy failed: {e}")),
                }
            }
        }
        KeyCode::Char('p') => {
            let mut s = state.lock().await;
            let sid = s.session_id.clone();
            s.set_toast("Interrupting…");
            s.trace_focus = false;
            drop(s);
            if let Some(sid) = sid {
                optimistic::spawn_interrupt(state, client, sid);
            }
        }
        KeyCode::Char('e') => {
            let mut s = state.lock().await;
            if let Some(step) = s.selected_step() {
                s.resume_step = Some(step.index);
                s.trace_focus = false;
                s.set_toast(format!("edit step {} · enter to resume", step.index));
                drop(s);
                set_prompt_text(textarea, &step.args);
            }
        }
        KeyCode::Char('r') | KeyCode::Enter => {
            let (sid, generating, prompt) = {
                let s = state.lock().await;
                let prompt = s
                    .selected_step()
                    .map(|step| crate::state::resume_from_step_prompt(&step, ""));
                (s.session_id.clone(), s.is_generating, prompt)
            };
            let Some(prompt) = prompt else {
                return Ok(LoopControl::Continue);
            };
            if generating {
                if let Some(sid) = sid {
                    optimistic::spawn_steer(state, client, sid, prompt.clone());
                }
                state.lock().await.set_toast("steered from step");
            } else {
                return submit_line(prompt, state, client, textarea).await;
            }
            state.lock().await.trace_focus = false;
        }
        KeyCode::Char('g') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            let mut s = state.lock().await;
            s.split_trace = false;
            s.trace_focus = false;
            s.mark_dirty();
        }
        _ => {}
    }
    Ok(LoopControl::Continue)
}

pub(crate) async fn handle_approval_key(
    key: KeyEvent,
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
) -> Result<LoopControl> {
    let choice = match key.code {
        KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter | KeyCode::Char('1') => {
            Some("once")
        }
        KeyCode::Char('a') | KeyCode::Char('A') | KeyCode::Char('2') => Some("always"),
        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc | KeyCode::Char('3') => Some("deny"),
        _ => None,
    };
    let Some(choice) = choice else {
        return Ok(LoopControl::Continue);
    };
    let (sid, req_id, allow_permanent) = {
        let s = state.lock().await;
        let req = s.pending_approval.as_ref();
        (
            s.session_id.clone(),
            req.and_then(|r| r.request_id.clone()),
            req.map(|r| r.allow_permanent).unwrap_or(true),
        )
    };
    let choice = if choice == "always" && !allow_permanent {
        "once"
    } else {
        choice
    };
    if let Some(sid) = sid {
        match client
            .approval_respond(&sid, choice, req_id.as_deref())
            .await
        {
            Ok(_) => {
                let mut s = state.lock().await;
                s.pending_approval = None;
                s.set_toast(format!("approval: {choice}"));
            }
            Err(e) => {
                state
                    .lock()
                    .await
                    .add_system(format!("approval failed: {e}"));
            }
        }
    }
    Ok(LoopControl::Continue)
}

pub(crate) fn is_work_key(key: KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return false;
    }
    matches!(
        key.code,
        KeyCode::Esc
            | KeyCode::Up
            | KeyCode::Down
            | KeyCode::Enter
            | KeyCode::Char('j')
            | KeyCode::Char('k')
            | KeyCode::Char('x')
            | KeyCode::Char('d')
            | KeyCode::Char('g')
            | KeyCode::Char(' ')
            | KeyCode::PageUp
            | KeyCode::PageDown
            | KeyCode::Home
            | KeyCode::End
    )
}

pub(crate) async fn handle_work_key(
    key: KeyEvent,
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
) -> Result<LoopControl> {
    match key.code {
        KeyCode::Esc => {
            let mut s = state.lock().await;
            s.work_focus = false;
            s.mark_dirty();
        }
        KeyCode::Up | KeyCode::Char('k') => {
            state.lock().await.work_move(-1);
        }
        KeyCode::Down | KeyCode::Char('j') => {
            state.lock().await.work_move(1);
        }
        KeyCode::PageUp => {
            state.lock().await.work_scroll_diff(-12);
        }
        KeyCode::PageDown => {
            state.lock().await.work_scroll_diff(12);
        }
        KeyCode::Home => {
            let mut s = state.lock().await;
            if s.work_show_diff {
                s.work_diff_offset = 0;
                s.mark_dirty();
            }
        }
        KeyCode::End => {
            let mut s = state.lock().await;
            if s.work_show_diff {
                let max = s.diff_text.lines().count().saturating_sub(1);
                s.work_diff_offset = max;
                s.mark_dirty();
            }
        }
        KeyCode::Char('d') => {
            let mut s = state.lock().await;
            s.work_show_diff = !s.work_show_diff;
            let diff_on = s.work_show_diff;
            s.refresh_work_chrome();
            s.set_toast(if diff_on {
                "file diffs · ↑↓ pick  pgup/pgdn scroll"
            } else {
                "process output"
            });
        }
        KeyCode::Char('g') => {
            refresh_agents(state, client).await;
            let mut s = state.lock().await;
            s.refresh_work_chrome();
            s.set_toast("work refreshed");
        }
        KeyCode::Char('x') | KeyCode::Char(' ') => {
            if state.lock().await.work_show_diff {
                return Ok(LoopControl::Continue);
            }
            let (proc, sub) = {
                let s = state.lock().await;
                match s.work_selected_row() {
                    Some(a) if a.is_process() => (Some(a.id.clone()), None),
                    Some(a) if a.is_subagent() => (None, Some(a.id.clone())),
                    _ => (None, None),
                }
            };
            if let Some(id) = proc {
                kill_selected_process(state, client, &id).await;
            } else if let Some(id) = sub {
                interrupt_selected_subagent(state, client, &id).await;
            } else {
                state.lock().await.set_toast("select a process or subagent");
            }
        }
        KeyCode::Enter => {
            let mut s = state.lock().await;
            if s.work_show_diff {
                let title = s
                    .work_diff_files
                    .get(s.work_diff_selected)
                    .map(|f| f.rel.clone())
                    .unwrap_or_else(|| "diff".into());
                let body = s.diff_text.clone();
                s.open_peek(title, body, None);
            } else {
                s.agents_replay = false;
                s.active_view = ActiveView::Agents;
                s.modal_selected = s.work_selected;
                s.work_focus = false;
                s.mark_dirty();
                drop(s);
                refresh_agents(state, client).await;
            }
        }
        _ => {}
    }
    Ok(LoopControl::Continue)
}

pub(crate) async fn toggle_work_sidebar(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    let opening = {
        let mut s = state.lock().await;
        s.toggle_work();
        s.split_work
    };
    if opening {
        refresh_agents(state, client).await;
        state.lock().await.refresh_work_chrome();
    }
}

pub(crate) fn is_files_key(key: KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return false;
    }
    matches!(
        key.code,
        KeyCode::Esc
            | KeyCode::Up
            | KeyCode::Down
            | KeyCode::Left
            | KeyCode::Right
            | KeyCode::Enter
            | KeyCode::PageUp
            | KeyCode::PageDown
            | KeyCode::Home
            | KeyCode::End
            | KeyCode::Char('j')
            | KeyCode::Char('k')
            | KeyCode::Char('h')
            | KeyCode::Char('l')
            | KeyCode::Char('o')
            | KeyCode::Char('r')
            | KeyCode::Char('u')
            | KeyCode::Char('g')
            | KeyCode::Char(' ')
    )
}

pub(crate) fn handle_files_key(s: &mut AppState, key: KeyEvent) -> LoopControl {
    match key.code {
        KeyCode::Esc => {
            s.files_focus = false;
            s.mark_dirty();
        }
        KeyCode::Up | KeyCode::Char('k') => s.files_move(-1),
        KeyCode::Down | KeyCode::Char('j') => s.files_move(1),
        KeyCode::PageUp => s.files_move(-10),
        KeyCode::PageDown => s.files_move(10),
        KeyCode::Home => {
            s.files_selected = 0;
            s.load_file_preview();
            s.mark_dirty();
        }
        KeyCode::End => {
            s.files_selected = s.files_rows.len().saturating_sub(1);
            s.load_file_preview();
            s.mark_dirty();
        }
        KeyCode::Left | KeyCode::Char('h') => {
            if let Some(row) = s.files_rows.get(s.files_selected) {
                if row.is_dir && row.expanded {
                    s.files_activate();
                } else if let Some((parent, _)) = row.rel.rsplit_once('/') {
                    if let Some(i) = s.files_rows.iter().position(|r| r.rel == parent) {
                        s.files_selected = i;
                        s.load_file_preview();
                        s.mark_dirty();
                    }
                }
            }
        }
        KeyCode::Right | KeyCode::Enter | KeyCode::Char('l') | KeyCode::Char(' ') => {
            s.files_activate();
        }
        KeyCode::Char('o') => {
            s.files_open_selected();
            s.set_toast("opened");
        }
        KeyCode::Char('r') => {
            let msg = s.files_restore_selected();
            if msg.contains("undo") {
                s.set_toast_for(msg, crate::optimistic::undo_ttl());
            } else {
                s.set_toast(msg);
            }
        }
        KeyCode::Char('u') => {
            let msg = s.apply_undo();
            s.set_toast(msg);
        }
        KeyCode::Char('g') => {
            s.refresh_files();
            s.set_toast("files refreshed");
        }
        _ => {}
    }
    LoopControl::Continue
}

pub(crate) fn is_overview_key(key: KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return false;
    }
    matches!(
        key.code,
        KeyCode::Esc
            | KeyCode::Up
            | KeyCode::Down
            | KeyCode::Enter
            | KeyCode::Char('j')
            | KeyCode::Char('k')
            | KeyCode::Char('p')
            | KeyCode::Char('e')
            | KeyCode::Char('r')
            | KeyCode::Char('y')
            | KeyCode::Char('a')
    )
}

pub(crate) async fn handle_key(
    key: KeyEvent,
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
) -> Result<LoopControl> {
    {
        let mut s = state.lock().await;

        if s.pending_secret.is_some() {
            drop(s);
            return handle_secret_key(key, state, client, textarea).await;
        }
        if s.pending_clarify.is_some() {
            drop(s);
            return handle_clarify_key(key, state, client).await;
        }
        if s.pending_approval.is_some() {
            drop(s);
            return handle_approval_key(key, state, client).await;
        }
        if s.trace_focus && s.active_view == ActiveView::Chat && is_overview_key(key) {
            drop(s);
            return handle_trace_key(key, state, client, textarea).await;
        }
        if s.files_focus && s.active_view == ActiveView::Chat && is_files_key(key) {
            return Ok(handle_files_key(&mut s, key));
        }
        if s.work_focus && s.active_view == ActiveView::Chat && is_work_key(key) {
            drop(s);
            return handle_work_key(key, state, client).await;
        }

        if s.active_view != ActiveView::Chat {
            if s.active_view == ActiveView::ModelPicker && s.picker_stage == PickerStage::Key {
                drop(s);
                return handle_picker_key_input(key, state, client).await;
            }
            if s.active_view == ActiveView::Peek {
                match key.code {
                    KeyCode::Esc | KeyCode::Enter => {
                        s.active_view = ActiveView::Chat;
                        s.mark_dirty();
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        s.peek_offset = s.peek_offset.saturating_sub(1);
                        s.mark_dirty();
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        let n = s.peek_body.lines().count();
                        if n > 0 {
                            s.peek_offset = (s.peek_offset + 1).min(n.saturating_sub(1));
                        }
                        s.mark_dirty();
                    }
                    KeyCode::PageUp => {
                        s.peek_offset = s.peek_offset.saturating_sub(10);
                        s.mark_dirty();
                    }
                    KeyCode::PageDown => {
                        let n = s.peek_body.lines().count();
                        s.peek_offset = (s.peek_offset + 10).min(n.saturating_sub(1));
                        s.mark_dirty();
                    }
                    _ => {}
                }
                return Ok(LoopControl::Continue);
            }
            if key.code == KeyCode::Enter && s.active_view == ActiveView::Context {
                s.active_view = ActiveView::Chat;
                s.mark_dirty();
                let sid = s.session_id.clone();
                drop(s);
                if let Some(sid) = sid {
                    state.lock().await.set_toast("compressing context…");
                    optimistic::spawn_compress(state, client, sid);
                }
                return Ok(LoopControl::Continue);
            }
            if key.code == KeyCode::Enter && s.active_view == ActiveView::ThemePicker {
                close_theme_picker(&mut s, true);
                return Ok(LoopControl::Continue);
            }
            if key.code == KeyCode::Enter && s.active_view == ActiveView::Background {
                let action = background_confirm(&mut s);
                drop(s);
                apply_bg_action(action, state, client).await;
                return Ok(LoopControl::Continue);
            }
            if key.code == KeyCode::Enter && s.active_view == ActiveView::Rollback {
                let idxs = s.filtered_checkpoint_indices();
                let hash = idxs
                    .get(s.modal_selected)
                    .and_then(|i| s.checkpoints.get(*i))
                    .map(|c| c.hash.clone());
                let Some(hash) = hash else {
                    return Ok(LoopControl::Continue);
                };
                if s.is_generating {
                    s.set_toast("session busy — wait or esc interrupt");
                    return Ok(LoopControl::Continue);
                }
                if !s.take_armed(ArmedKind::Rollback) {
                    let short: String = hash.chars().take(10).collect();
                    s.arm(
                        ArmedKind::Rollback,
                        format!("restore {short} · enter again"),
                    );
                    return Ok(LoopControl::Continue);
                }
                drop(s);
                apply_rollback(state, client, &hash).await;
                return Ok(LoopControl::Continue);
            }
            if key.code == KeyCode::Char('d')
                && s.active_view == ActiveView::Rollback
                && s.picker_filter.is_empty()
            {
                let idxs = s.filtered_checkpoint_indices();
                let hash = idxs
                    .get(s.modal_selected)
                    .and_then(|i| s.checkpoints.get(*i))
                    .map(|c| c.hash.clone());
                drop(s);
                if let Some(hash) = hash {
                    load_rollback_diff(state, client, &hash).await;
                }
                return Ok(LoopControl::Continue);
            }
            if key.code == KeyCode::Char('d')
                && s.active_view == ActiveView::Sessions
                && s.picker_filter.is_empty()
            {
                let idxs = s.filtered_session_indices();
                let picked = idxs
                    .get(s.modal_selected)
                    .and_then(|i| s.sessions_list.get(*i).map(|r| (r.id.clone(), r.live)));
                drop(s);
                if let Some((id, live)) = picked {
                    if live {
                        state.lock().await.set_toast("cannot delete a live session");
                    } else {
                        delete_stored_session(state, client, &id).await;
                    }
                }
                return Ok(LoopControl::Continue);
            }
            if key.code == KeyCode::Char('h')
                && s.active_view == ActiveView::Sessions
                && s.picker_filter.is_empty()
            {
                let idxs = s.filtered_session_indices();
                let id = idxs
                    .get(s.modal_selected)
                    .and_then(|i| s.sessions_list.get(*i))
                    .map(|r| r.id.clone());
                drop(s);
                if let Some(id) = id {
                    set_session_hidden(state, client, &id, true).await;
                }
                return Ok(LoopControl::Continue);
            }
            if key.code == KeyCode::Enter && s.active_view == ActiveView::Sessions {
                let idxs = s.filtered_session_indices();
                let picked = idxs
                    .get(s.modal_selected)
                    .and_then(|i| s.sessions_list.get(*i).map(|r| (r.id.clone(), r.live)));
                s.active_view = ActiveView::Chat;
                s.modal_selected = 0;
                s.picker_filter.clear();
                s.mark_dirty();
                drop(s);
                if let Some((id, live)) = picked {
                    if live {
                        activate_session(state, client, &id).await;
                    } else {
                        resume_session(state, client, &id).await;
                    }
                }
                return Ok(LoopControl::Continue);
            }
            if key.code == KeyCode::Char('n')
                && s.active_view == ActiveView::Profiles
                && s.picker_filter.is_empty()
            {
                s.set_toast("usage: /profiles new <slug>");
                return Ok(LoopControl::Continue);
            }
            if key.code == KeyCode::Char('i')
                && s.active_view == ActiveView::Profiles
                && s.picker_filter.is_empty()
            {
                let name = selected_profile_name(&s);
                drop(s);
                if let Some(name) = name {
                    peek_profile(state, client, &name).await;
                }
                return Ok(LoopControl::Continue);
            }
            if key.code == KeyCode::Enter && s.active_view == ActiveView::Profiles {
                let idxs = s.filtered_profile_indices();
                let id = idxs
                    .get(s.modal_selected)
                    .and_then(|i| s.profiles.get(*i))
                    .and_then(|p| p.last_session_id.clone());
                s.active_view = ActiveView::Chat;
                s.modal_selected = 0;
                s.picker_filter.clear();
                s.mark_dirty();
                drop(s);
                if let Some(id) = id {
                    resume_session(state, client, &id).await;
                } else {
                    state
                        .lock()
                        .await
                        .set_toast("no last session on this profile");
                }
                return Ok(LoopControl::Continue);
            }
            if s.active_view == ActiveView::Skills {
                match key.code {
                    KeyCode::Enter => {
                        let name = selected_skill_name(&s);
                        drop(s);
                        if let Some(name) = name {
                            inspect_skill(state, client, &name).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('i') if s.picker_filter.is_empty() => {
                        let name = selected_skill_name(&s);
                        drop(s);
                        if let Some(name) = name {
                            install_skill(state, client, &name).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    _ => {}
                }
            }
            if s.active_view == ActiveView::Memory {
                match key.code {
                    KeyCode::Enter => {
                        let node = selected_memory(&s);
                        drop(s);
                        if let Some((id, fallback)) = node {
                            peek_memory(state, client, &id, &fallback).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('x') if s.picker_filter.is_empty() => {
                        let node = selected_memory(&s);
                        if let Some((id, label)) = node {
                            if !s.take_armed(ArmedKind::MemoryDelete) {
                                s.arm(ArmedKind::MemoryDelete, format!("delete {label} · x again"));
                                return Ok(LoopControl::Continue);
                            }
                            drop(s);
                            delete_memory(state, client, &id).await;
                        } else {
                            drop(s);
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('e') if s.picker_filter.is_empty() => {
                        let node = selected_memory(&s);
                        drop(s);
                        if let Some((id, _)) = node {
                            return begin_memory_edit(state, client, &id).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    _ => {}
                }
            }
            if s.active_view == ActiveView::Replay {
                match key.code {
                    KeyCode::Enter => {
                        let path = s
                            .filtered_spawn_indices()
                            .get(s.modal_selected)
                            .and_then(|i| s.spawn_trees.get(*i).map(|e| e.path.clone()));
                        drop(s);
                        if let Some(path) = path {
                            load_replay(state, client, &path).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('i') if s.picker_filter.is_empty() => {
                        let peek = s
                            .filtered_spawn_indices()
                            .get(s.modal_selected)
                            .and_then(|i| {
                                s.spawn_trees.get(*i).map(|e| {
                                    (
                                        format!("replay  #{}", i + 1),
                                        format!("{}\n{}×\n{}", e.label, e.count, e.path),
                                    )
                                })
                            });
                        drop(s);
                        if let Some((title, body)) = peek {
                            state.lock().await.open_peek(title, body, None);
                        }
                        return Ok(LoopControl::Continue);
                    }
                    _ => {}
                }
            }
            if s.active_view == ActiveView::Projects
                && key.code == KeyCode::Char('s')
                && s.picker_filter.is_empty()
                && s.project_drill.is_none()
            {
                drop(s);
                scan_projects(state, client).await;
                return Ok(LoopControl::Continue);
            }
            if s.active_view == ActiveView::Projects && key.code == KeyCode::Enter {
                if s.project_drill.is_some() {
                    let idxs = s.filtered_project_session_indices();
                    let picked = idxs
                        .get(s.modal_selected)
                        .and_then(|i| s.project_sessions.get(*i).map(|r| r.id.clone()));
                    s.active_view = ActiveView::Chat;
                    s.project_drill = None;
                    s.picker_filter.clear();
                    s.modal_selected = 0;
                    s.mark_dirty();
                    drop(s);
                    if let Some(id) = picked {
                        resume_session(state, client, &id).await;
                    }
                    return Ok(LoopControl::Continue);
                }
                let idxs = s.filtered_project_indices();
                let id = idxs
                    .get(s.modal_selected)
                    .and_then(|i| s.projects_list.get(*i).map(|p| p.id.clone()));
                drop(s);
                if let Some(id) = id {
                    drill_project(state, client, &id).await;
                }
                return Ok(LoopControl::Continue);
            }
            if s.active_view == ActiveView::Agents {
                if s.agents_replay
                    && matches!(
                        key.code,
                        KeyCode::Char('p')
                            | KeyCode::Char('r')
                            | KeyCode::Char('x')
                            | KeyCode::Char('X')
                            | KeyCode::Char('s')
                    )
                    && s.picker_filter.is_empty()
                {
                    s.set_toast("snapshot · live controls disabled");
                    return Ok(LoopControl::Continue);
                }
                if s.agents_steer {
                    match key.code {
                        KeyCode::Esc => {
                            s.agents_steer = false;
                            s.picker_filter.clear();
                            s.mark_dirty();
                        }
                        KeyCode::Enter => {
                            let text = s.picker_filter.trim().to_string();
                            let id = selected_agent_id(&s);
                            s.agents_steer = false;
                            s.picker_filter.clear();
                            drop(s);
                            if text.is_empty() {
                                state
                                    .lock()
                                    .await
                                    .set_toast("usage: type a redirect · enter");
                            } else if let Some(id) = id {
                                steer_selected_subagent(state, client, &id, &text).await;
                            } else {
                                state.lock().await.set_toast("select a live subagent");
                            }
                        }
                        KeyCode::Backspace => {
                            s.picker_filter.pop();
                            s.mark_dirty();
                        }
                        KeyCode::Char(c)
                            if !key.modifiers.contains(KeyModifiers::CONTROL)
                                && is_compose_char(c) =>
                        {
                            s.picker_filter.push(c);
                            s.mark_dirty();
                        }
                        _ => {}
                    }
                    return Ok(LoopControl::Continue);
                }
                match key.code {
                    KeyCode::Char('p') => {
                        drop(s);
                        set_agents_paused(state, client, true).await;
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('r') => {
                        drop(s);
                        set_agents_paused(state, client, false).await;
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('x') => {
                        let proc = selected_agent(&s)
                            .filter(|a| a.is_process())
                            .map(|a| a.id.clone());
                        let sub = selected_agent(&s)
                            .filter(|a| a.is_subagent())
                            .map(|a| a.id.clone());
                        drop(s);
                        if let Some(id) = proc {
                            kill_selected_process(state, client, &id).await;
                        } else if let Some(id) = sub {
                            interrupt_selected_subagent(state, client, &id).await;
                        } else {
                            state
                                .lock()
                                .await
                                .set_toast("select a subagent or process to stop");
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('X') => {
                        let proc = selected_agent(&s)
                            .filter(|a| a.is_process())
                            .map(|a| a.id.clone());
                        let ids = selected_agent(&s)
                            .filter(|a| a.is_subagent())
                            .map(|a| s.descendant_agent_ids(&a.id))
                            .unwrap_or_default();
                        drop(s);
                        if let Some(id) = proc {
                            kill_selected_process(state, client, &id).await;
                        } else if ids.is_empty() {
                            state.lock().await.set_toast("select a subagent to stop");
                        } else {
                            interrupt_subagent_ids(state, client, ids).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('s') => {
                        let live = selected_agent(&s).is_some_and(|a| a.is_live());
                        if live {
                            s.agents_steer = true;
                            s.picker_filter.clear();
                            s.set_toast("type a redirect · enter sends");
                        } else {
                            s.set_toast("select a running subagent to steer");
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Enter => {
                        let peek = selected_agent(&s)
                            .map(|a| (format!("{}  {}", a.title, a.id), agent_peek_body(a)));
                        drop(s);
                        if let Some((title, body)) = peek {
                            state.lock().await.open_peek(title, body, None);
                        }
                        return Ok(LoopControl::Continue);
                    }
                    _ => {}
                }
            }
            if key.code == KeyCode::Enter && s.active_view == ActiveView::Palette {
                let name = s
                    .filtered_palette_entries()
                    .get(s.modal_selected)
                    .map(|e| e.name.clone())
                    .unwrap_or_default();
                s.active_view = ActiveView::Chat;
                s.modal_selected = 0;
                s.picker_filter.clear();
                s.mark_dirty();
                drop(s);
                if !name.is_empty() {
                    return dispatch_slash(&name, "", state, client, textarea).await;
                }
                return Ok(LoopControl::Continue);
            }
            if s.active_view == ActiveView::Mcp {
                if s.mcp_key_name.is_some() {
                    drop(s);
                    return handle_mcp_key_input(key, state, client).await;
                }
                match key.code {
                    KeyCode::Enter => {
                        let idxs = s.filtered_mcp_indices();
                        let peek = idxs.get(s.modal_selected).and_then(|i| {
                            s.mcp_servers.get(*i).map(|m| {
                                let req = if m.requires.is_empty() {
                                    String::new()
                                } else {
                                    format!("\nrequires: {}", m.requires.join(", "))
                                };
                                (
                                    format!("mcp  {}", m.name),
                                    format!(
                                        "{}\n{}  tools {}  {}\n{}{}",
                                        m.description,
                                        m.transport,
                                        m.tools,
                                        if m.enabled { "enabled" } else { "off" },
                                        if m.configured {
                                            "configured"
                                        } else {
                                            "catalog"
                                        },
                                        req
                                    ),
                                )
                            })
                        });
                        drop(s);
                        if let Some((title, body)) = peek {
                            state.lock().await.open_peek(title, body, None);
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('r') if s.picker_filter.is_empty() => {
                        drop(s);
                        reload_mcp(state, client).await;
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('a') | KeyCode::Char('i') if s.picker_filter.is_empty() => {
                        let name = selected_mcp(&s).map(|m| m.0);
                        drop(s);
                        if let Some(name) = name {
                            mcp_add(state, client, &name).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('t') if s.picker_filter.is_empty() => {
                        let name = selected_mcp(&s).map(|m| m.0);
                        drop(s);
                        if let Some(name) = name {
                            mcp_test(state, client, &name).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('o') if s.picker_filter.is_empty() => {
                        let name = selected_mcp(&s).map(|m| m.0);
                        drop(s);
                        if let Some(name) = name {
                            mcp_oauth_login(state, client, &name).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('k') if s.picker_filter.is_empty() => {
                        let name = selected_mcp(&s).map(|m| m.0);
                        if let Some(name) = name {
                            s.mcp_key_name = Some(name.clone());
                            s.picker_key.clear();
                            s.picker_key_error.clear();
                            s.set_toast(format!("mcp key for {name} · paste · enter"));
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('x') if s.picker_filter.is_empty() => {
                        let name =
                            selected_mcp(&s).and_then(|(n, configured)| configured.then_some(n));
                        if let Some(name) = name {
                            if !s.take_armed(ArmedKind::McpRemove) {
                                s.arm(ArmedKind::McpRemove, format!("remove {name} · x again"));
                                return Ok(LoopControl::Continue);
                            }
                            drop(s);
                            mcp_remove(state, client, &name).await;
                        } else {
                            s.set_toast("catalog only · a to add");
                            drop(s);
                        }
                        return Ok(LoopControl::Continue);
                    }
                    _ => {}
                }
            }
            if s.active_view == ActiveView::Tools {
                match key.code {
                    KeyCode::Enter => {
                        let name = selected_toolset_name(&s);
                        drop(s);
                        if let Some(name) = name {
                            peek_toolset(state, client, &name).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char(' ') | KeyCode::Char('x') if s.picker_filter.is_empty() => {
                        let name = selected_toolset_name(&s);
                        let enable = name
                            .as_ref()
                            .and_then(|n| s.toolsets.iter().find(|t| t.name == *n))
                            .map(|t| !t.enabled)
                            .unwrap_or(true);
                        drop(s);
                        if let Some(name) = name {
                            let action = if enable { "enable" } else { "disable" };
                            configure_tools(state, client, action, vec![name]).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    _ => {}
                }
            }
            if s.active_view == ActiveView::Plugins
                && matches!(key.code, KeyCode::Char(' ') | KeyCode::Char('x'))
                && s.picker_filter.is_empty()
            {
                let sel = selected_plugin(&s);
                drop(s);
                if let Some((ident, enabled)) = sel {
                    toggle_plugin(state, client, &ident, !enabled).await;
                }
                return Ok(LoopControl::Continue);
            }
            if s.active_view == ActiveView::Cron {
                match key.code {
                    KeyCode::Enter => {
                        let idxs = s.filtered_cron_indices();
                        let peek = idxs.get(s.modal_selected).and_then(|i| {
                            s.cron_jobs.get(*i).map(|j| {
                                (
                                    format!("cron  {}", j.name),
                                    format!(
                                        "{}\n{}\n{}\n{}\n{}",
                                        j.id,
                                        j.schedule,
                                        j.state,
                                        j.prompt,
                                        if j.enabled { "enabled" } else { "paused" }
                                    ),
                                )
                            })
                        });
                        drop(s);
                        if let Some((title, body)) = peek {
                            state.lock().await.open_peek(title, body, None);
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('p') if s.picker_filter.is_empty() => {
                        let id = selected_cron_id(&s);
                        drop(s);
                        if let Some(id) = id {
                            cron_action(state, client, "pause", &id).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('r') if s.picker_filter.is_empty() => {
                        let id = selected_cron_id(&s);
                        drop(s);
                        if let Some(id) = id {
                            cron_action(state, client, "resume", &id).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    KeyCode::Char('x') if s.picker_filter.is_empty() => {
                        let id = selected_cron_id(&s);
                        drop(s);
                        if let Some(id) = id {
                            cron_action(state, client, "remove", &id).await;
                        }
                        return Ok(LoopControl::Continue);
                    }
                    _ => {}
                }
            }
            if matches!(
                (key.code, s.active_view),
                (KeyCode::Enter | KeyCode::Right, ActiveView::ModelPicker)
                    | (KeyCode::Char(' '), ActiveView::ModelPicker)
                    | (
                        KeyCode::Enter | KeyCode::Char(' '),
                        ActiveView::BranchPicker
                    )
            ) {
                if s.active_view == ActiveView::ModelPicker {
                    let action = picker_confirm(&mut s);
                    drop(s);
                    apply_picker_action(action, state, client).await;
                } else {
                    let action = branch_confirm(&mut s);
                    drop(s);
                    apply_branch_action(action, state, client).await;
                }
                return Ok(LoopControl::Continue);
            }
            if key.code == KeyCode::Left
                && s.active_view == ActiveView::ModelPicker
                && s.picker_stage == PickerStage::Models
            {
                s.picker_stage = PickerStage::Providers;
                s.picker_filter.clear();
                s.modal_selected = s
                    .filtered_provider_indices()
                    .iter()
                    .position(|i| *i == s.picker_provider)
                    .unwrap_or(0);
                s.mark_dirty();
                return Ok(LoopControl::Continue);
            }
            if key.code == KeyCode::Char('x')
                && s.active_view == ActiveView::ModelPicker
                && s.picker_stage == PickerStage::Providers
                && s.picker_filter.is_empty()
            {
                let slug = selected_provider_slug(&s);
                drop(s);
                if let Some(slug) = slug {
                    disconnect_model(state, client, &slug).await;
                }
                return Ok(LoopControl::Continue);
            }
            return Ok(handle_modal_key(&mut s, key));
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('y') => {
                    s.copy_latest_response();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Char('l') => {
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Char('n') => {
                    if s.has_thread() && !s.take_armed(ArmedKind::NewSession) {
                        s.arm(
                            ArmedKind::NewSession,
                            "new session drops this thread · ctrl+n again",
                        );
                        return Ok(LoopControl::Continue);
                    }
                    drop(s);
                    new_session(state, client).await;
                    *textarea = reset_prompt(false);
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Home => {
                    s.scroll_from_bottom = usize::MAX / 4;
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::End => {
                    s.scroll_from_bottom = 0;
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Char('o') => {
                    drop(s);
                    open_model_picker(state, client).await;
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Char('b') => {
                    drop(s);
                    open_branch_picker(state).await;
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Char('g') => {
                    s.split_trace = !s.split_trace;
                    s.trace_focus = false;
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Char('d') => {
                    if s.diff_tool_id.is_some() {
                        s.close_edit_diff();
                    } else {
                        s.split_diff = !s.split_diff;
                        if s.split_diff {
                            s.split_work = false;
                            s.work_focus = false;
                            s.refresh_diff();
                        }
                    }
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Char('e') => {
                    s.split_files = !s.split_files;
                    if s.split_files {
                        s.files_focus = true;
                        s.trace_focus = false;
                        s.split_work = false;
                        s.work_focus = false;
                        s.refresh_files();
                    } else {
                        s.files_focus = false;
                        s.files_list = None;
                    }
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Char('w') => {
                    drop(s);
                    toggle_work_sidebar(state, client).await;
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Char('p') => {
                    s.active_view = ActiveView::Palette;
                    s.modal_selected = 0;
                    s.picker_filter.clear();
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Char('t') => {
                    s.active_view = ActiveView::Tasks;
                    s.modal_selected = 0;
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Char('s') => {
                    s.active_view = ActiveView::Sessions;
                    s.modal_selected = 0;
                    s.picker_filter.clear();
                    s.mark_dirty();
                    drop(s);
                    refresh_sessions(state, client).await;
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Char('k') => {
                    s.active_view = ActiveView::Skills;
                    s.modal_selected = 0;
                    s.picker_filter.clear();
                    s.mark_dirty();
                    drop(s);
                    refresh_skills(state, client).await;
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Char('h') => {
                    s.active_view = ActiveView::Help;
                    s.modal_selected = 0;
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                _ => {}
            }
        }

        if key.modifiers.contains(KeyModifiers::SHIFT) && key.code == KeyCode::BackTab {
            drop(s);
            cycle_permission_mode(state, client).await;
            return Ok(LoopControl::Continue);
        }

        if s.complete_open && !s.slash_open {
            match key.code {
                KeyCode::Esc => {
                    s.close_complete();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Up => {
                    s.complete_selected = s.complete_selected.saturating_sub(1);
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Down => {
                    let n = s.complete_items.len();
                    if n > 0 {
                        s.complete_selected = (s.complete_selected + 1).min(n.saturating_sub(1));
                    }
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Tab | KeyCode::Enter => {
                    let item = s.complete_items.get(s.complete_selected).cloned();
                    let from = s.complete_replace_from;
                    let keep = item.as_ref().is_some_and(|i| i.keep_open());
                    drop(s);
                    if let Some(item) = item {
                        let input = textarea.lines().join("\n");
                        let filled = crate::complete::apply_fill(&input, from, &item);
                        set_prompt_text(textarea, &filled);
                        if !keep {
                            state.lock().await.close_complete();
                        }
                        sync_composer(textarea, state, Some(client)).await;
                    }
                    return Ok(LoopControl::Continue);
                }
                _ => {
                    drop(s);
                    textarea.input(key);
                    sync_composer(textarea, state, Some(client)).await;
                    return Ok(LoopControl::Continue);
                }
            }
        }

        if key.modifiers == KeyModifiers::NONE && key.code == KeyCode::Tab && !s.slash_open {
            s.show_thinking = !s.show_thinking;
            s.mark_dirty();
            return Ok(LoopControl::Continue);
        }

        if s.slash_open {
            match key.code {
                KeyCode::Esc => {
                    s.slash_open = false;
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Up => {
                    s.slash_selected = s.slash_selected.saturating_sub(1);
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Down => {
                    let n = s.slash_ranked().len();
                    if n > 0 {
                        s.slash_selected = (s.slash_selected + 1).min(n.saturating_sub(1));
                    }
                    s.mark_dirty();
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Tab => {
                    let arg = crate::slash::slash_arg_stage(&s.slash_query, s.slash_replace_from);
                    let from = s.slash_replace_from.min(s.slash_query.len());
                    let query = s.slash_query.clone();
                    let fill = s.slash_ranked().get(s.slash_selected).map(|e| {
                        if arg {
                            format!("{}{} ", query.get(..from).unwrap_or(""), e.name)
                        } else {
                            format!("{} ", e.name)
                        }
                    });
                    s.slash_open = false;
                    drop(s);
                    if let Some(fill) = fill {
                        set_prompt_text(textarea, &fill);
                        sync_composer(textarea, state, Some(client)).await;
                    }
                    return Ok(LoopControl::Continue);
                }
                KeyCode::Enter => {
                    let arg = crate::slash::slash_arg_stage(&s.slash_query, s.slash_replace_from);
                    let from = s.slash_replace_from.min(s.slash_query.len());
                    let query = s.slash_query.clone();
                    let picked = s.slash_ranked().get(s.slash_selected).map(|e| (*e).clone());
                    s.slash_open = false;
                    drop(s);
                    if let Some(cmd) = picked {
                        if arg {
                            let fill = format!("{}{} ", query.get(..from).unwrap_or(""), cmd.name);
                            set_prompt_text(textarea, &fill);
                            sync_composer(textarea, state, Some(client)).await;
                            return Ok(LoopControl::Continue);
                        }
                        if cmd.kind == SlashKind::Local {
                            return dispatch_slash(&cmd.name, "", state, client, textarea).await;
                        }
                        return gateway_slash(state, client, textarea, &cmd.name).await;
                    }
                    return Ok(LoopControl::Continue);
                }
                _ => {
                    drop(s);
                    textarea.input(key);
                    sync_composer(textarea, state, Some(client)).await;
                    return Ok(LoopControl::Continue);
                }
            }
        }

        match (key.modifiers, key.code) {
            (KeyModifiers::CONTROL, KeyCode::Char('c')) => {
                let draft = textarea.lines().join("\n");
                if !draft.trim().is_empty() {
                    drop(s);
                    *textarea = reset_prompt(false);
                    return Ok(LoopControl::Continue);
                }
                if s.is_generating {
                    let sid = s.session_id.clone();
                    s.set_toast("Interrupting…");
                    drop(s);
                    if let Some(sid) = sid {
                        optimistic::spawn_interrupt(state, client, sid);
                    }
                    return Ok(LoopControl::Continue);
                }
                if s.has_unsaved("") && !s.take_armed(ArmedKind::Quit) {
                    s.arm(ArmedKind::Quit, "queued prompts · ctrl+c again to quit");
                    return Ok(LoopControl::Continue);
                }
                return Ok(LoopControl::Quit);
            }
            _ if s.vim.is_some() => {
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Enter | KeyCode::Char('\n'))
                {
                    let text = textarea.lines().join("\n").trim().to_string();
                    drop(s);
                    return send_now(text, state, client, textarea).await;
                }
                drop(s);
                apply_vim(textarea, state, key).await;
                return Ok(LoopControl::Continue);
            }
            (KeyModifiers::NONE, KeyCode::Esc) => {
                if s.queue_edit.is_some() {
                    s.cancel_queue_edit();
                    drop(s);
                    *textarea = reset_prompt(false);
                    return Ok(LoopControl::Continue);
                }
                if s.is_generating {
                    let sid = s.session_id.clone();
                    s.set_toast("Interrupting…");
                    drop(s);
                    if let Some(sid) = sid {
                        optimistic::spawn_interrupt(state, client, sid);
                    }
                    return Ok(LoopControl::Continue);
                }
                let draft = textarea.lines().join("\n");
                if s.has_unsaved(&draft) && !s.take_armed(ArmedKind::Quit) {
                    let why = if !draft.trim().is_empty() {
                        "unsaved draft · esc again to quit"
                    } else {
                        "queued prompts · esc again to quit"
                    };
                    s.arm(ArmedKind::Quit, why);
                    return Ok(LoopControl::Continue);
                }
                return Ok(LoopControl::Quit);
            }
            (KeyModifiers::SHIFT, KeyCode::Enter) | (KeyModifiers::ALT, KeyCode::Enter) => {
                drop(s);
                textarea.insert_newline();
                return Ok(LoopControl::Continue);
            }
            (KeyModifiers::CONTROL, KeyCode::Enter)
            | (KeyModifiers::CONTROL, KeyCode::Char('\n')) => {
                let text = textarea.lines().join("\n").trim().to_string();
                drop(s);
                return send_now(text, state, client, textarea).await;
            }
            (KeyModifiers::NONE, KeyCode::Enter) => {
                let text = textarea.lines().join("\n").trim().to_string();
                if s.queue_edit.is_some() {
                    drop(s);
                    return send_now(text, state, client, textarea).await;
                }
                if text.is_empty() {
                    if !s.prompt_queue.is_empty() {
                        drop(s);
                        return send_now(String::new(), state, client, textarea).await;
                    }
                    return Ok(LoopControl::Continue);
                }
                drop(s);
                return submit_line(text, state, client, textarea).await;
            }
            (KeyModifiers::CONTROL, KeyCode::Char('x')) => {
                if s.queue_edit.is_some() {
                    s.drop_queue_edit();
                    drop(s);
                    *textarea = reset_prompt(false);
                }
                return Ok(LoopControl::Continue);
            }
            (KeyModifiers::NONE, KeyCode::PageUp) => {
                s.scroll_older(20);
                return Ok(LoopControl::Continue);
            }
            (KeyModifiers::NONE, KeyCode::PageDown) => {
                s.scroll_newer(20);
                return Ok(LoopControl::Continue);
            }
            (KeyModifiers::NONE, KeyCode::Up) => {
                let (row, _) = textarea.cursor();
                if row == 0 {
                    if !s.prompt_queue.is_empty() {
                        if let Some(text) = s.cycle_queue(1) {
                            drop(s);
                            set_prompt_text(textarea, &text);
                            return Ok(LoopControl::Continue);
                        }
                    }
                    let current = textarea.lines().join("\n");
                    if let Some(prev) = s.prompt_history.prev(&current) {
                        drop(s);
                        set_prompt_text(textarea, &prev);
                        return Ok(LoopControl::Continue);
                    }
                }
                drop(s);
                textarea.input(key);
                return Ok(LoopControl::Continue);
            }
            (KeyModifiers::NONE, KeyCode::Down) => {
                let (row, _) = textarea.cursor();
                let last_row = textarea.lines().len().saturating_sub(1);
                if row >= last_row {
                    if !s.prompt_queue.is_empty() {
                        if let Some(text) = s.cycle_queue(-1) {
                            drop(s);
                            set_prompt_text(textarea, &text);
                            return Ok(LoopControl::Continue);
                        }
                    }
                    if let Some(next) = s.prompt_history.next() {
                        drop(s);
                        set_prompt_text(textarea, &next);
                        return Ok(LoopControl::Continue);
                    }
                }
                drop(s);
                textarea.input(key);
                return Ok(LoopControl::Continue);
            }
            (KeyModifiers::ALT, KeyCode::Backspace) => {
                s.prompt_history.reset_browse();
                textarea.delete_word();
                return Ok(LoopControl::Continue);
            }
            (KeyModifiers::NONE, KeyCode::Char('u'))
                if s.pending_undo.as_ref().is_some_and(|u| u.live())
                    && textarea.lines().iter().all(|l| l.trim().is_empty()) =>
            {
                let msg = s.apply_undo();
                s.set_toast(msg);
                return Ok(LoopControl::Continue);
            }
            _ => {
                s.prompt_history.reset_browse();
                drop(s);
                textarea.input(key);
                sync_composer(textarea, state, Some(client)).await;
            }
        }
    }
    Ok(LoopControl::Continue)
}

pub(crate) fn is_filter_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ':')
}

pub(crate) fn is_compose_char(c: char) -> bool {
    !c.is_control() && c != '\u{7f}'
}

pub(crate) fn handle_modal_key(s: &mut AppState, key: KeyEvent) -> LoopControl {
    let filterable = matches!(
        s.active_view,
        ActiveView::ModelPicker
            | ActiveView::BranchPicker
            | ActiveView::Skills
            | ActiveView::Sessions
            | ActiveView::Profiles
            | ActiveView::Memory
            | ActiveView::Rollback
            | ActiveView::Background
            | ActiveView::Mcp
            | ActiveView::Palette
            | ActiveView::Tools
            | ActiveView::Plugins
            | ActiveView::Cron
            | ActiveView::Replay
            | ActiveView::Projects
    );
    let n = s.picker_len();
    match key.code {
        KeyCode::Esc => {
            if filterable && !s.picker_filter.is_empty() {
                s.picker_filter.clear();
                s.modal_selected = 0;
                s.mark_dirty();
            } else if s.active_view == ActiveView::Projects && s.project_drill.is_some() {
                s.project_drill = None;
                s.project_sessions.clear();
                s.picker_filter.clear();
                s.modal_selected = 0;
                s.mark_dirty();
            } else if s.active_view == ActiveView::Agents && s.agents_replay {
                s.agents_replay = false;
                s.agent_rows.retain(|r| r.kind != "subagent");
                s.active_view = ActiveView::Replay;
                s.modal_selected = 0;
                s.picker_filter.clear();
                s.mark_dirty();
            } else if s.active_view == ActiveView::ModelPicker
                && matches!(s.picker_stage, PickerStage::Models | PickerStage::Key)
            {
                s.picker_stage = PickerStage::Providers;
                s.picker_filter.clear();
                s.clear_picker_key();
                s.modal_selected = s
                    .filtered_provider_indices()
                    .iter()
                    .position(|i| *i == s.picker_provider)
                    .unwrap_or(0);
                s.mark_dirty();
            } else {
                if s.active_view == ActiveView::ThemePicker {
                    close_theme_picker(s, false);
                } else {
                    s.active_view = ActiveView::Chat;
                    s.picker_list = None;
                    s.modal_selected = 0;
                    s.picker_stage = PickerStage::Providers;
                    s.picker_filter.clear();
                    s.clear_picker_key();
                    s.mark_dirty();
                }
            }
        }
        KeyCode::Up => {
            s.modal_selected = s.modal_selected.saturating_sub(1);
            preview_selected_theme(s);
            s.mark_dirty();
        }
        KeyCode::Down => {
            if n > 0 {
                s.modal_selected = (s.modal_selected + 1).min(n - 1);
            }
            preview_selected_theme(s);
            s.mark_dirty();
        }
        KeyCode::PageUp => {
            s.modal_selected = s.modal_selected.saturating_sub(10);
            preview_selected_theme(s);
            s.mark_dirty();
        }
        KeyCode::PageDown => {
            if n > 0 {
                s.modal_selected = (s.modal_selected + 10).min(n - 1);
            }
            preview_selected_theme(s);
            s.mark_dirty();
        }
        KeyCode::Home => {
            s.modal_selected = 0;
            preview_selected_theme(s);
            s.mark_dirty();
        }
        KeyCode::End => {
            s.modal_selected = n.saturating_sub(1);
            preview_selected_theme(s);
            s.mark_dirty();
        }
        KeyCode::Backspace if filterable => {
            s.picker_filter.pop();
            s.modal_selected = 0;
            s.mark_dirty();
        }
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) && filterable => {
            s.picker_filter.clear();
            s.modal_selected = 0;
            s.mark_dirty();
        }
        KeyCode::Char(c)
            if filterable
                && !key.modifiers.contains(KeyModifiers::CONTROL)
                && (if s.active_view == ActiveView::Background {
                    is_compose_char(c)
                } else {
                    is_filter_char(c)
                }) =>
        {
            s.picker_filter.push(c);
            s.modal_selected = 0;
            s.mark_dirty();
        }
        KeyCode::Char(' ') if s.active_view == ActiveView::Tasks => {
            let sel = s.modal_selected;
            if let Some(task) = s.tasks.get_mut(sel) {
                task.status = task.status.cycle();
            }
            s.mark_dirty();
        }
        KeyCode::Enter => {
            s.active_view = ActiveView::Chat;
            s.picker_list = None;
            s.modal_selected = 0;
            s.picker_filter.clear();
            s.mark_dirty();
        }
        _ => {}
    }
    LoopControl::Continue
}

pub(crate) fn open_theme_picker(s: &mut AppState) {
    s.theme_revert = Some(s.theme_id.clone());
    s.active_view = ActiveView::ThemePicker;
    s.modal_selected = crate::ui::theme::catalog()
        .iter()
        .position(|p| p.id == s.theme_id)
        .unwrap_or(0);
    s.picker_filter.clear();
    preview_selected_theme(s);
    s.mark_dirty();
}

pub(crate) fn preview_selected_theme(s: &mut AppState) {
    if s.active_view != ActiveView::ThemePicker {
        return;
    }
    let Some(p) = crate::ui::theme::catalog().get(s.modal_selected).copied() else {
        return;
    };
    crate::ui::theme::apply(p);
    s.theme_id = p.id.to_string();
}

pub(crate) fn revert_theme_preview(s: &mut AppState) {
    let Some(id) = s.theme_revert.take() else {
        return;
    };
    let p = crate::ui::theme::lookup(&id);
    crate::ui::theme::apply(p);
    s.theme_id = p.id.to_string();
}

pub(crate) fn picker_scroll(s: &mut AppState, dir: i32) {
    let n = s.picker_len();
    if n == 0 {
        return;
    }
    if dir < 0 {
        s.modal_selected = s.modal_selected.saturating_sub(1);
    } else {
        s.modal_selected = (s.modal_selected + 1).min(n - 1);
    }
    preview_selected_theme(s);
    s.mark_dirty();
}

pub(crate) fn close_theme_picker(s: &mut AppState, commit: bool) {
    if commit {
        if let Some(id) = crate::ui::theme::catalog()
            .get(s.modal_selected)
            .map(|p| p.id)
        {
            commit_theme(s, id);
        }
    } else {
        revert_theme_preview(s);
    }
    s.active_view = ActiveView::Chat;
    s.picker_list = None;
    s.modal_selected = 0;
    s.picker_filter.clear();
    s.mark_dirty();
}

pub(crate) fn commit_theme(s: &mut AppState, id: &str) {
    let p = crate::ui::theme::lookup(id);
    crate::ui::theme::apply(p);
    s.theme_id = p.id.to_string();
    s.theme_revert = None;
    crate::ui::theme::save(&s.hermes_home, p.id);
    s.set_toast(format!("theme · {}", p.label));
}
