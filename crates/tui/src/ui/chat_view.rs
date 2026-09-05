use std::hash::{Hash, Hasher};

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph},
    Frame,
};
use unicode_width::UnicodeWidthStr;

use super::brand;
use super::highlighter::CodeHighlighter;
use super::stream::{
    clock, cluster_files, cluster_label, is_running, json_str, next_tool_cluster, run_command,
    run_output_style, run_steps, strip_ansi, task_mark, thought_beats, todos_from_content,
    tool_headline, tool_icon, tool_kind, ToolKind,
};
use super::theme::Theme;
use crate::layout::{scroll_y, wrap_chunks, wrap_words};
use crate::state::{AppState, ChatMessage, MessageRole};

pub struct StreamCache {
    width: usize,
    theme_id: &'static str,
    thinking: bool,
    focus: bool,
    epoch: u64,
    expand_epoch: u64,
    motion_epoch: u64,
    blocks: Vec<CachedBlock>,
}

struct CachedBlock {
    hash: u64,
    lines: Vec<Line<'static>>,
    click_id: Option<String>,
}

impl StreamCache {
    pub fn new() -> Self {
        Self {
            width: 0,
            theme_id: "",
            thinking: false,
            focus: false,
            epoch: 0,
            expand_epoch: 0,
            motion_epoch: 0,
            blocks: Vec::new(),
        }
    }

    fn reset_if_stale(
        &mut self,
        width: usize,
        theme_id: &'static str,
        thinking: bool,
        focus: bool,
        epoch: u64,
        expand_epoch: u64,
    ) {
        let motion_epoch = super::motion::epoch();
        if self.width != width
            || self.theme_id != theme_id
            || self.thinking != thinking
            || self.focus != focus
            || self.epoch != epoch
            || self.expand_epoch != expand_epoch
            || self.motion_epoch != motion_epoch
        {
            self.blocks.clear();
            self.width = width;
            self.theme_id = theme_id;
            self.thinking = thinking;
            self.focus = focus;
            self.epoch = epoch;
            self.expand_epoch = expand_epoch;
            self.motion_epoch = motion_epoch;
        }
    }
}

pub struct ChatScrollback;

impl ChatScrollback {
    pub fn render(
        frame: &mut Frame,
        area: Rect,
        state: &mut AppState,
        highlighter: &CodeHighlighter,
        cache: &mut StreamCache,
        frame_count: u64,
    ) {
        let col_width = area.width.max(1) as usize;
        let block = Block::default()
            .borders(Borders::NONE)
            .style(Style::default().bg(Theme::bg_base()));

        let spin = super::motion::think_spinner(frame_count);
        let dots = super::rhythm::ellipsis(frame_count);
        let view_h = area.height as usize;
        let empty = state.messages.is_empty();
        state.stream_area = Some(area);
        state.hit_tools.clear();

        let paragraph = if empty {
            let mut text_lines: Vec<Line> = Vec::new();
            brand::render_intro(&mut text_lines, col_width, view_h, state, frame_count);
            Paragraph::new(Text::from(text_lines)).block(block)
        } else {
            cache.reset_if_stale(
                col_width,
                crate::ui::theme::current().id,
                state.show_thinking,
                state.focus_view,
                crate::ui::theme::epoch(),
                state.expand_epoch,
            );
            let messages = state.visible_messages();
            rebuild_cache(
                cache,
                messages,
                state,
                highlighter,
                col_width,
                spin,
                dots,
                frame_count,
            );
            let total: usize = cache.blocks.iter().map(|b| b.lines.len()).sum();
            let y = scroll_y(total, view_h, state.scroll_from_bottom) as usize;
            state.hit_tools = tool_hits(&cache.blocks, y, view_h, area);
            let mut visible = take_visible(&cache.blocks, y, view_h);
            super::wash::apply(&mut visible, col_width, view_h, frame_count);
            if let crate::state::HoverKind::Tool(id) = &state.hover {
                for (hit, hid) in &state.hit_tools {
                    if hid == id {
                        let idx = hit.y.saturating_sub(area.y) as usize;
                        if let Some(line) = visible.get_mut(idx) {
                            crate::ui::theme::hover_line(line, true);
                        }
                    }
                }
            }
            // Drop the solid canvas Block so gold wash can show through empty cells.
            let para = Paragraph::new(Text::from(visible));
            if super::wash::active() {
                para
            } else {
                para.block(block)
            }
        };

        frame.render_widget(paragraph, area);
    }
}

fn tool_hits(
    blocks: &[CachedBlock],
    mut skip: usize,
    take: usize,
    area: Rect,
) -> Vec<(crate::state::HitRange, String)> {
    let mut out = Vec::new();
    let mut vis = 0usize;
    for block in blocks {
        if vis >= take {
            break;
        }
        let n = block.lines.len();
        if skip >= n {
            skip -= n;
            continue;
        }
        let shown = (n - skip).min(take - vis);
        if let Some(id) = &block.click_id {
            for k in 0..shown {
                out.push((
                    crate::state::HitRange {
                        y: area.y.saturating_add((vis + k) as u16),
                        x0: area.x,
                        x1: area.x.saturating_add(area.width),
                    },
                    id.clone(),
                ));
            }
        }
        vis += shown;
        skip = 0;
    }
    out
}

fn take_visible(blocks: &[CachedBlock], mut skip: usize, take: usize) -> Vec<Line<'static>> {
    let mut out = Vec::with_capacity(take);
    let mut left = take;
    for block in blocks {
        if left == 0 {
            break;
        }
        let n = block.lines.len();
        if skip >= n {
            skip -= n;
            continue;
        }
        let start = skip;
        skip = 0;
        let end = (start + left).min(n);
        out.extend(block.lines[start..end].iter().cloned());
        left -= end - start;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn rebuild_cache(
    cache: &mut StreamCache,
    messages: &[ChatMessage],
    state: &AppState,
    highlighter: &CodeHighlighter,
    col_width: usize,
    spin: &'static str,
    dots: &'static str,
    frame_count: u64,
) {
    let mut plan: Vec<(usize, usize, u64, bool)> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let (end, hash, unstable) = next_span(messages, i);
        plan.push((i, end, hash, unstable));
        i = end;
    }

    let mut old = std::mem::take(&mut cache.blocks);
    let mut next = Vec::with_capacity(plan.len());
    for (k, &(start, end, hash, unstable)) in plan.iter().enumerate() {
        if !unstable && old.get(k).is_some_and(|b| b.hash == hash) {
            next.push(CachedBlock {
                hash,
                lines: std::mem::take(&mut old[k].lines),
                click_id: old[k].click_id.clone(),
            });
        } else {
            let mut lines = Vec::new();
            let click_id = emit_span(
                &mut lines,
                messages,
                start,
                end,
                state,
                highlighter,
                col_width,
                spin,
                dots,
                frame_count,
            );
            next.push(CachedBlock {
                hash,
                lines,
                click_id,
            });
        }
    }
    cache.blocks = next;
}

fn next_span(messages: &[ChatMessage], i: usize) -> (usize, u64, bool) {
    match &messages[i].role {
        MessageRole::User => {
            let with_img = matches!(
                messages.get(i + 1).map(|m| &m.role),
                Some(MessageRole::ImagePreview { .. })
            );
            let end = if with_img { i + 2 } else { i + 1 };
            (end, hash_range(messages, i, end), false)
        }
        MessageRole::Assistant | MessageRole::Reasoning => {
            let unstable = messages[i].is_streaming;
            let hash = if unstable {
                0
            } else {
                hash_range(messages, i, i + 1)
            };
            (i + 1, hash, unstable)
        }
        MessageRole::Tool { status, .. } => {
            if let Some((end, _, _)) = next_tool_cluster(messages, i) {
                (end, hash_range(messages, i, end), false)
            } else {
                let unstable = is_running(status);
                let hash = if unstable {
                    0
                } else {
                    hash_range(messages, i, i + 1)
                };
                (i + 1, hash, unstable)
            }
        }
        _ => (i + 1, hash_range(messages, i, i + 1), false),
    }
}

fn hash_range(messages: &[ChatMessage], start: usize, end: usize) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for m in &messages[start..end] {
        std::mem::discriminant(&m.role).hash(&mut hasher);
        m.is_streaming.hash(&mut hasher);
        m.content.hash(&mut hasher);
        m.output.hash(&mut hasher);
        m.timestamp.hash(&mut hasher);
        if let MessageRole::Tool { name, status, .. } = &m.role {
            name.hash(&mut hasher);
            status.hash(&mut hasher);
        }
        if let MessageRole::ImagePreview { path } = &m.role {
            path.hash(&mut hasher);
        }
    }
    hasher.finish()
}

#[allow(clippy::too_many_arguments)]
fn emit_span(
    lines: &mut Vec<Line<'static>>,
    messages: &[ChatMessage],
    start: usize,
    end: usize,
    state: &AppState,
    highlighter: &CodeHighlighter,
    col_width: usize,
    spin: &'static str,
    dots: &'static str,
    frame_count: u64,
) -> Option<String> {
    let mut click_id = None;
    match &messages[start].role {
        MessageRole::User => {
            if end == start + 2 {
                if let MessageRole::ImagePreview { path } = &messages[start + 1].role {
                    let leaf = path.rsplit(['/', '\\']).next().unwrap_or(path);
                    let mut body = messages[start].content.clone();
                    body.push_str("  [");
                    body.push_str(leaf);
                    body.push(']');
                    push_user(lines, &body, &clock(messages[start].timestamp), col_width);
                    gap(lines);
                    return None;
                }
            }
            push_user(
                lines,
                &messages[start].content,
                &clock(messages[start].timestamp),
                col_width,
            );
            gap(lines);
        }
        MessageRole::Assistant => {
            push_assistant(lines, &messages[start], highlighter, col_width, spin);
            if !messages[start].is_streaming {
                gap(lines);
            }
        }
        MessageRole::Reasoning => {
            push_reasoning(lines, &messages[start], state, col_width, spin, dots);
            click_id = Some(messages[start].id.clone());
        }
        MessageRole::Tool { name, status, .. } => {
            if let Some((end, reads, searches)) = next_tool_cluster(messages, start) {
                lines.push(tool_row(
                    "●",
                    cluster_label(reads, searches),
                    Theme::text_secondary(),
                    false,
                    spin,
                ));
                let files = cluster_files(messages, start, end);
                if !files.is_empty() {
                    lines.push(Line::from(Span::styled(
                        format!("{}{files}", super::rhythm::NEST_STR),
                        Style::default().fg(Theme::text_dim()),
                    )));
                }
                gap(lines);
                return None;
            }
            let running = is_running(status);
            let failed = status.starts_with("failed");
            let kind = tool_kind(name);
            if kind == ToolKind::Todo {
                if state.tasks.is_empty() {
                    push_todo_card(lines, &messages[start], state, running, spin);
                }
                return Some(messages[start].id.clone());
            }
            let mark = if failed {
                crate::ui::stream::tool_done_icon(true)
            } else {
                tool_icon(name)
            };
            let color = if running {
                Theme::brand_gold()
            } else if failed {
                Theme::accent_red()
            } else if kind == ToolKind::Edit || kind == ToolKind::Run {
                Theme::text_primary()
            } else {
                Theme::text_secondary()
            };
            let headline = tool_headline(name, &messages[start].content, status);
            let expanded = kind == ToolKind::Run
                && (running || failed || state.expanded_tools.contains(&messages[start].id));
            let tool_spin = super::motion::tool_spinner(
                frame_count,
                super::motion::salt_id(&messages[start].id),
            );
            let glyph = if running {
                tool_spin
            } else if kind == ToolKind::Run {
                if expanded {
                    "▾"
                } else {
                    "▸"
                }
            } else {
                mark
            };
            if kind == ToolKind::Edit {
                let open = state.diff_tool_id.as_deref() == Some(messages[start].id.as_str());
                let hot = matches!(&state.hover, crate::state::HoverKind::Tool(id) if id == &messages[start].id);
                lines.push(edit_row(
                    &messages[start].content,
                    running,
                    tool_spin,
                    hot || open,
                    open,
                ));
                click_id = Some(messages[start].id.clone());
            } else {
                lines.push(tool_row(glyph, headline, color, running, tool_spin));
            }
            if kind == ToolKind::Run {
                click_id = Some(messages[start].id.clone());
                if expanded {
                    push_run_detail(
                        lines,
                        &messages[start].content,
                        &messages[start].output,
                        col_width,
                        running,
                        failed,
                        spin,
                    );
                }
            } else if kind != ToolKind::Todo && kind != ToolKind::Edit {
                push_tool_context(lines, name, &messages[start].content, col_width);
            }
            gap(lines);
        }
        MessageRole::ImagePreview { path } => {
            let leaf = path.rsplit(['/', '\\']).next().unwrap_or(path);
            lines.extend(super::markdown::image_card(leaf, path, col_width));
            gap(lines);
        }
        MessageRole::Compaction => {
            lines.push(Line::from(vec![
                Span::styled(
                    super::rhythm::FENCE_HEAD,
                    Style::default().fg(Theme::border_subtle()),
                ),
                Span::styled(
                    "fold",
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled(
                    super::rhythm::RAIL,
                    Style::default().fg(Theme::border_subtle()),
                ),
                Span::styled(
                    messages[start].content.clone(),
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]));
            lines.push(Line::from(Span::styled(
                super::rhythm::FENCE_TAIL,
                Style::default().fg(Theme::border_subtle()),
            )));
            gap(lines);
        }
        MessageRole::System => {
            lines.push(Line::from(Span::styled(
                format!("{}{}", super::rhythm::GUTTER_STR, messages[start].content),
                Style::default().fg(Theme::text_muted()),
            )));
            gap(lines);
        }
    }
    click_id
}

fn gap(lines: &mut Vec<Line<'static>>) {
    if !lines.last().is_some_and(super::rhythm::is_blank) {
        lines.push(Line::raw(""));
    }
}

fn push_user(lines: &mut Vec<Line<'static>>, text: &str, time: &str, col_width: usize) {
    let bg = Style::default().bg(Theme::bg_surface());
    let inner = col_width
        .saturating_sub(super::rhythm::GUTTER + 1 + time.width() + super::rhythm::GUTTER)
        .max(8);
    let chunks = wrap_prose(text, inner);
    for (n, chunk) in chunks.iter().enumerate() {
        let left = vec![
            Span::raw(super::rhythm::GUTTER_STR),
            Span::styled(chunk.clone(), Style::default().fg(Theme::text_primary())),
        ];
        if n == 0 {
            lines.push(timed_line(left, time, col_width).style(bg));
        } else {
            lines.push(Line::from(left).style(bg));
        }
    }
}

fn push_assistant(
    lines: &mut Vec<Line<'static>>,
    msg: &ChatMessage,
    highlighter: &CodeHighlighter,
    col_width: usize,
    spin: &'static str,
) {
    let time = clock(msg.timestamp);
    let start = lines.len();
    super::markdown::render(lines, &msg.content, col_width, highlighter);
    if start == lines.len() {
        let spans = if msg.is_streaming {
            vec![
                Span::raw(super::rhythm::GUTTER_STR),
                Span::styled(spin.to_string(), Style::default().fg(Theme::brand_gold())),
            ]
        } else {
            vec![Span::raw(super::rhythm::GUTTER_STR)]
        };
        lines.push(timed_line(spans, &time, col_width));
        return;
    }
    if msg.is_streaming {
        if let Some(first) = lines.get_mut(start) {
            first.spans.push(Span::styled(
                format!("{}{spin}", super::rhythm::GUTTER_STR),
                Style::default().fg(Theme::brand_gold()),
            ));
        }
    }
    let first = std::mem::take(&mut lines[start]);
    lines[start] = timed_line(first.spans, &time, col_width);
}

fn push_reasoning(
    lines: &mut Vec<Line<'static>>,
    msg: &ChatMessage,
    state: &AppState,
    col_width: usize,
    spin: &'static str,
    dots: &'static str,
) {
    let time = clock(msg.timestamp);
    let beats = thought_beats(&msg.content);
    let expanded = state.thought_expanded(&msg.id);
    let header_style = if msg.is_streaming && !expanded {
        Style::default().fg(Theme::brand_gold())
    } else if expanded {
        Style::default()
            .fg(Theme::brand_gold())
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Theme::text_muted())
    };
    let header = if msg.is_streaming && !expanded {
        format!("{}{spin} Thinking", super::rhythm::GUTTER_STR)
    } else if expanded {
        format!("{}▾ Thinking", super::rhythm::GUTTER_STR)
    } else {
        format!("{}▸ Thinking", super::rhythm::GUTTER_STR)
    };
    let _ = dots;
    lines.push(timed_line(
        vec![Span::styled(header, header_style)],
        &time,
        col_width,
    ));

    let show = if expanded {
        0
    } else if msg.is_streaming {
        beats.len().saturating_sub(4)
    } else {
        return;
    };
    let gutter = Style::default().fg(Theme::text_dim());
    let body = Style::default().fg(Theme::text_muted());
    let live = Style::default().fg(Theme::text_secondary());
    for (i, beat) in beats.iter().enumerate().skip(show) {
        let style = if msg.is_streaming && i + 1 == beats.len() {
            live
        } else {
            body
        };
        let inner = col_width
            .saturating_sub(super::rhythm::GUTTER + super::rhythm::NEST + 2)
            .max(8);
        for (k, chunk) in wrap_words(beat, inner).into_iter().enumerate() {
            lines.push(Line::from(vec![
                Span::styled(
                    if k == 0 {
                        super::rhythm::RAIL
                    } else {
                        super::rhythm::RAIL_CONT
                    },
                    gutter,
                ),
                Span::styled(chunk, style),
            ]));
        }
    }
    if expanded && !msg.is_streaming {
        gap(lines);
    }
}

fn push_todo_card(
    lines: &mut Vec<Line<'static>>,
    msg: &ChatMessage,
    state: &AppState,
    running: bool,
    spin: &'static str,
) {
    let mut todos = todos_from_content(&msg.content);
    if todos.is_empty() && !state.tasks.is_empty() {
        todos = state.tasks.clone();
    }
    let n = todos.len();
    let done = todos
        .iter()
        .filter(|t| t.status == crate::state::TaskStatus::Completed)
        .count();
    let all_done = n > 0 && done == n;
    let expanded = running || !all_done || state.expanded_tools.contains(&msg.id);
    let mark = if running {
        spin
    } else if expanded {
        "▾"
    } else {
        "▸"
    };
    let color = if running {
        Theme::brand_gold()
    } else {
        Theme::text_secondary()
    };
    let head = if n == 0 {
        "Todo".into()
    } else {
        format!("Todo  {done}/{n}")
    };
    lines.push(tool_row(mark, head, color, running, spin));
    if !expanded {
        gap(lines);
        return;
    }
    for task in todos.iter().take(16) {
        let (glyph, style) = match task.status {
            crate::state::TaskStatus::InProgress => (
                if running {
                    spin
                } else {
                    task_mark(&task.status)
                },
                Style::default().fg(Theme::brand_gold()),
            ),
            crate::state::TaskStatus::Completed => (
                task_mark(&task.status),
                Style::default().fg(Theme::accent_green()),
            ),
            crate::state::TaskStatus::Failed => (
                task_mark(&task.status),
                Style::default().fg(Theme::accent_red()),
            ),
            crate::state::TaskStatus::Pending => (
                task_mark(&task.status),
                Style::default().fg(Theme::text_muted()),
            ),
        };
        let title = crate::tips::ellipsize(&task.title, 64);
        lines.push(Line::from(vec![
            Span::raw(super::rhythm::NEST_STR),
            Span::styled(format!("{glyph} {title}"), style),
        ]));
    }
    if todos.len() > 16 {
        lines.push(Line::from(Span::styled(
            format!("{}+{} more", super::rhythm::NEST_STR, todos.len() - 16),
            Style::default().fg(Theme::text_dim()),
        )));
    }
    gap(lines);
}

fn push_tool_context(lines: &mut Vec<Line<'static>>, name: &str, content: &str, col_width: usize) {
    let n = name.to_ascii_lowercase();
    let detail = if n.contains("web") || n.contains("extract") || n.contains("fetch") {
        json_str(content, &["url", "uri", "href", "link"])
    } else if n.contains("skill") {
        json_str(content, &["path", "file", "name", "skill"])
    } else if n.contains("search") || n.contains("grep") || n.contains("glob") {
        json_str(content, &["pattern", "query", "glob", "q"])
    } else {
        None
    };
    let Some(detail) = detail else {
        return;
    };
    if tool_headline(name, content, "completed").contains(&detail) {
        return;
    }
    let inner = col_width
        .saturating_sub(super::rhythm::GUTTER + super::rhythm::NEST)
        .max(12);
    let shown = crate::tips::ellipsize(&detail, inner);
    lines.push(Line::from(Span::styled(
        format!("{}{shown}", super::rhythm::NEST_STR),
        Style::default().fg(Theme::text_dim()),
    )));
}

fn push_run_detail(
    lines: &mut Vec<Line<'static>>,
    content: &str,
    output: &str,
    col_width: usize,
    running: bool,
    failed: bool,
    spin: &'static str,
) {
    let inner = col_width
        .saturating_sub(super::rhythm::GUTTER + super::rhythm::NEST)
        .max(12);
    let cmd = run_command(content);
    let steps = run_steps(&cmd);
    if steps.len() > 1 {
        for (i, step) in steps.iter().enumerate() {
            let (mark, color) = if failed && i + 1 == steps.len() {
                ("✗", Theme::accent_red())
            } else if running && i == 0 {
                (spin, Theme::brand_gold())
            } else if running {
                ("○", Theme::text_muted())
            } else {
                ("✓", Theme::accent_green())
            };
            let title = crate::tips::ellipsize(step, inner.saturating_sub(4).max(8));
            lines.push(Line::from(vec![
                Span::raw(super::rhythm::NEST_STR),
                Span::styled(format!("{mark} {title}"), Style::default().fg(color)),
            ]));
        }
    } else if !cmd.is_empty() {
        for chunk in wrap_chunks(&format!("$ {cmd}"), inner) {
            lines.push(Line::from(Span::styled(
                format!("{}{chunk}", super::rhythm::NEST_STR),
                Style::default().fg(Theme::text_primary()),
            )));
        }
    }
    let cleaned = strip_ansi(output);
    let body = if cleaned.trim().is_empty() {
        if cmd.is_empty() {
            strip_ansi(content)
        } else {
            String::new()
        }
    } else {
        cleaned
    };
    let trimmed = body.trim();
    if trimmed.is_empty() {
        lines.push(Line::from(Span::styled(
            format!(
                "{}{}",
                super::rhythm::NEST_STR,
                if running { "running…" } else { "(no output)" }
            ),
            Style::default().fg(Theme::text_dim()),
        )));
        gap(lines);
        return;
    }
    let raw_lines: Vec<&str> = trimmed.lines().collect();
    let cap = if running { 16 } else { 24 };
    let skip = raw_lines.len().saturating_sub(cap);
    if skip > 0 {
        lines.push(Line::from(Span::styled(
            format!("{}… {} earlier", super::rhythm::NEST_STR, skip),
            Style::default().fg(Theme::text_dim()),
        )));
    }
    let mut shown = 0usize;
    for raw in raw_lines.into_iter().skip(skip) {
        if shown >= cap {
            break;
        }
        if raw.is_empty() {
            lines.push(Line::raw(""));
            shown += 1;
            continue;
        }
        let style = Style::default().fg(run_output_style(raw));
        for chunk in wrap_chunks(raw, inner) {
            if shown >= cap {
                break;
            }
            lines.push(Line::from(Span::styled(
                format!("{}{chunk}", super::rhythm::NEST_STR),
                style,
            )));
            shown += 1;
        }
    }
    gap(lines);
}

fn edit_row(
    content: &str,
    running: bool,
    spin: &'static str,
    hot: bool,
    open: bool,
) -> Line<'static> {
    let file = crate::ui::stream::first_path(content).unwrap_or_else(|| "file".into());
    let (plus, minus) = crate::ui::stream::diff_stats(content);
    let glyph = if running { spin } else { "◆" };
    let file_style = if hot || open {
        Style::default()
            .fg(Theme::brand_gold())
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Theme::text_primary())
    };
    let verb_style = if running {
        Style::default()
            .fg(Theme::brand_gold())
            .add_modifier(Modifier::BOLD)
    } else {
        file_style
    };
    let mut spans = vec![
        Span::raw(super::rhythm::GUTTER_STR),
        Span::styled(
            format!("{glyph} "),
            Style::default().fg(if running {
                Theme::brand_gold()
            } else {
                Theme::text_secondary()
            }),
        ),
        Span::styled("Edit ", verb_style),
        Span::styled(file, file_style),
    ];
    if plus > 0 || minus > 0 {
        spans.push(Span::raw(" "));
        spans.push(Span::styled(
            format!("+{plus}"),
            Style::default().fg(Theme::accent_green()),
        ));
        spans.push(Span::styled(
            format!("/-{minus}"),
            Style::default().fg(Theme::accent_red()),
        ));
    }
    if open {
        spans.push(Span::styled(
            "  ▸",
            Style::default().fg(Theme::brand_gold()),
        ));
    }
    Line::from(spans)
}

fn tool_row(
    mark: &'static str,
    text: String,
    color: ratatui::style::Color,
    running: bool,
    spin: &'static str,
) -> Line<'static> {
    let glyph = if running { spin } else { mark };
    let style = if running {
        Style::default()
            .fg(Theme::brand_gold())
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(color)
    };
    Line::from(vec![
        Span::raw(super::rhythm::GUTTER_STR),
        Span::styled(format!("{glyph} {text}"), style),
    ])
}

fn timed_line(mut left: Vec<Span<'static>>, time: &str, width: usize) -> Line<'static> {
    let left_w: usize = left.iter().map(|s| s.content.width()).sum();
    let time_w = time.width();
    let right = super::rhythm::GUTTER;
    let pad = width.saturating_sub(left_w + time_w + right).max(1);
    left.push(Span::raw(" ".repeat(pad)));
    left.push(Span::styled(
        time.to_string(),
        Style::default().fg(Theme::text_dim()),
    ));
    left.push(Span::raw(super::rhythm::GUTTER_STR));
    Line::from(left)
}

fn wrap_prose(text: &str, inner: usize) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.lines() {
        if raw.is_empty() {
            out.push(String::new());
            continue;
        }
        out.extend(wrap_words(raw, inner));
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use chrono::Local;

    #[test]
    fn cache_reuses_stable_assistant() {
        let hl = CodeHighlighter::new();
        let mut cache = StreamCache::new();
        cache.reset_if_stale(80, "gold", false, false, 0, 0);
        let ts = Local::now();
        let messages = vec![
            ChatMessage {
                id: "u".into(),
                role: MessageRole::User,
                content: "hi".into(),
                timestamp: ts,
                output: String::new(),
                is_streaming: false,
            },
            ChatMessage {
                id: "a".into(),
                role: MessageRole::Assistant,
                content: "hello **world**".into(),
                timestamp: ts,
                output: String::new(),
                is_streaming: false,
            },
        ];
        let state = AppState::new();
        rebuild_cache(&mut cache, &messages, &state, &hl, 80, " ", "", 0);
        assert_eq!(cache.blocks.len(), 2);
        let ptr = cache.blocks[1].lines.as_ptr();
        rebuild_cache(&mut cache, &messages, &state, &hl, 80, " ", "", 0);
        assert_eq!(cache.blocks[1].lines.as_ptr(), ptr);
    }

    #[test]
    fn reasoning_collapsed_is_one_header() {
        let hl = CodeHighlighter::new();
        let mut cache = StreamCache::new();
        cache.reset_if_stale(80, "gold", false, false, 0, 0);
        let ts = Local::now();
        let messages = vec![ChatMessage {
            id: "r".into(),
            role: MessageRole::Reasoning,
            content: "**Planning steps***Confirming path****Considering skills**".into(),
            timestamp: ts,
            output: String::new(),
            is_streaming: false,
        }];
        let state = AppState::new();
        rebuild_cache(&mut cache, &messages, &state, &hl, 80, " ", "", 0);
        let nonempty: Vec<_> = cache.blocks[0]
            .lines
            .iter()
            .filter(|l| !crate::ui::rhythm::is_blank(l))
            .collect();
        assert_eq!(nonempty.len(), 1);
        assert_eq!(cache.blocks[0].click_id.as_deref(), Some("r"));
        let blob: String = cache.blocks[0]
            .lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(blob.contains("Thinking"));
        assert!(blob.contains('▸'));
        assert!(!blob.contains("Planning steps"));
        assert!(!blob.contains("**"));
        let mut state = AppState::new();
        state.toggle_tool_expand("r");
        cache.reset_if_stale(80, "gold", false, false, 0, state.expand_epoch);
        rebuild_cache(&mut cache, &messages, &state, &hl, 80, " ", "", 0);
        let blob: String = cache.blocks[0]
            .lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(blob.contains('▾'));
        assert!(blob.contains("Planning steps"));
        assert!(blob.contains("Confirming path"));
        assert!(blob.contains("Considering skills"));
        assert!(state.thought_expanded("r"));
    }

    #[test]
    fn run_terminal_expands_command_and_output() {
        let hl = CodeHighlighter::new();
        let mut cache = StreamCache::new();
        cache.reset_if_stale(80, "gold", false, false, 0, 0);
        let ts = Local::now();
        let messages = vec![ChatMessage {
            id: "term".into(),
            role: MessageRole::Tool {
                name: "terminal".into(),
                status: "completed".into(),
                tool_id: None,
            },
            content: r#"{"command":"cargo test --offline"}"#.into(),
            output: "108 passed".into(),
            timestamp: ts,
            is_streaming: false,
        }];
        let mut state = AppState::new();
        rebuild_cache(&mut cache, &messages, &state, &hl, 80, " ", "", 0);
        let blob: String = cache.blocks[0]
            .lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(blob.contains("Terminal(\"cargo test"));
        assert!(!blob.contains("108 passed"));
        assert_eq!(cache.blocks[0].click_id.as_deref(), Some("term"));
        let mut live = messages.clone();
        if let MessageRole::Tool { status, .. } = &mut live[0].role {
            *status = "running...".into();
        }
        live[0].output = "◆ Python\n  ✓ 3.12".into();
        cache.reset_if_stale(80, "gold", false, false, 1, 0);
        rebuild_cache(&mut cache, &live, &state, &hl, 80, " ", "", 0);
        let blob: String = cache.blocks[0]
            .lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(blob.contains("Python"));
        assert!(blob.contains("3.12"));
        state.toggle_tool_expand("term");
        cache.reset_if_stale(80, "gold", false, false, 0, state.expand_epoch);
        rebuild_cache(&mut cache, &messages, &state, &hl, 80, " ", "", 0);
        let blob: String = cache.blocks[0]
            .lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(blob.contains("cargo test --offline"));
        assert!(blob.contains("108 passed"));
        assert!(blob.contains('▾'));
    }

    #[test]
    fn timed_line_right_gutter_matches_left() {
        let line = timed_line(vec![Span::raw("  hello")], "9:41 AM", 40);
        let widths: Vec<usize> = line.spans.iter().map(|s| s.content.width()).collect();
        let total: usize = widths.iter().sum();
        assert_eq!(total, 40);
        assert_eq!(line.spans[0].content.as_ref(), "  hello");
        assert_eq!(
            line.spans.last().map(|s| s.content.as_ref()),
            Some(crate::ui::rhythm::GUTTER_STR)
        );
        assert!(line.spans.iter().any(|s| s.content.as_ref() == "9:41 AM"));
    }

    #[test]
    fn take_visible_skips_prefix() {
        let blocks = vec![
            CachedBlock {
                hash: 1,
                lines: vec![Line::raw("a"), Line::raw("b")],
                click_id: None,
            },
            CachedBlock {
                hash: 2,
                lines: vec![Line::raw("c"), Line::raw("d"), Line::raw("e")],
                click_id: None,
            },
        ];
        let vis = take_visible(&blocks, 3, 2);
        assert_eq!(vis.len(), 2);
        assert_eq!(vis[0].spans[0].content, "d");
        assert_eq!(vis[1].spans[0].content, "e");
    }
}
