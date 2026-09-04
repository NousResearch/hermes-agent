use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use unicode_width::UnicodeWidthStr;

use super::theme::Theme;
use crate::state::{AppState, HitRange, HoverKind};

pub const QUEUE_WINDOW: usize = 3;
const SEND: &str = "[Send now]";
const EDIT: &str = "[edit]";
const CANCEL: &str = "[cancel]";

pub struct QueuePane;

pub fn queue_window(len: usize, edit: Option<usize>) -> (usize, usize) {
    if len == 0 {
        return (0, 0);
    }
    let start = match edit {
        None => 0,
        Some(i) => i.saturating_sub(1).min(len.saturating_sub(QUEUE_WINDOW)),
    };
    let end = (start + QUEUE_WINDOW).min(len);
    (start, end)
}

pub fn queue_height(state: &AppState) -> u16 {
    let n = state.prompt_queue.len();
    if n == 0 {
        0
    } else {
        let (start, end) = queue_window(n, state.queue_edit);
        (end - start) as u16
    }
}

fn row_hot(hover: &HoverKind, i: usize) -> bool {
    matches!(
        hover,
        HoverKind::Queue(j)
            | HoverKind::QueueSend(j)
            | HoverKind::QueueEdit(j)
            | HoverKind::QueueDrop(j)
            if *j == i
    )
}

fn actions_width() -> usize {
    SEND.width() + EDIT.width() + CANCEL.width()
}

impl QueuePane {
    pub fn render(frame: &mut Frame, area: Rect, state: &mut AppState) {
        state.hit_queue.clear();
        if area.height == 0 {
            return;
        }
        let n = state.prompt_queue.len();
        if n == 0 {
            return;
        }
        let (start, end) = queue_window(n, state.queue_edit);
        let width = area.width as usize;
        let mut lines: Vec<Line> = Vec::new();
        let mut vis = 0u16;
        for (i, text) in state
            .prompt_queue
            .iter()
            .enumerate()
            .skip(start)
            .take(end - start)
        {
            if vis >= area.height {
                break;
            }
            let y = area.y.saturating_add(vis);
            let active = state.queue_edit == Some(i);
            let hot = active || row_hot(&state.hover, i);
            let show_btns = hot && width > actions_width() + 16;
            let num = format!("  #{}  ", i + 1);
            let num_w = num.width();
            let btn_w = if show_btns { actions_width() } else { 0 };
            let preview_w = width.saturating_sub(num_w + btn_w + 1).max(8);
            let preview = compact_preview(text, preview_w);
            let body_style = if active {
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Theme::text_secondary())
            };
            let mut spans = vec![
                Span::styled(
                    num,
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(preview, body_style),
            ];
            let used: usize = spans.iter().map(|s| s.content.width()).sum();
            if used + btn_w < width {
                spans.push(Span::raw(" ".repeat(width - used - btn_w)));
            }
            if show_btns {
                let send_hot = state.hover == HoverKind::QueueSend(i);
                let edit_hot = state.hover == HoverKind::QueueEdit(i);
                let drop_hot = state.hover == HoverKind::QueueDrop(i);
                let x0 = area.x.saturating_add((width - btn_w) as u16);
                let send_w = SEND.width() as u16;
                let edit_w = EDIT.width() as u16;
                let drop_w = CANCEL.width() as u16;
                state.hit_queue.push((
                    HitRange {
                        y,
                        x0,
                        x1: x0.saturating_add(send_w),
                    },
                    HoverKind::QueueSend(i),
                ));
                state.hit_queue.push((
                    HitRange {
                        y,
                        x0: x0.saturating_add(send_w),
                        x1: x0.saturating_add(send_w.saturating_add(edit_w)),
                    },
                    HoverKind::QueueEdit(i),
                ));
                state.hit_queue.push((
                    HitRange {
                        y,
                        x0: x0.saturating_add(send_w.saturating_add(edit_w)),
                        x1: x0.saturating_add(send_w.saturating_add(edit_w).saturating_add(drop_w)),
                    },
                    HoverKind::QueueDrop(i),
                ));
                spans.push(Span::styled(
                    SEND,
                    crate::ui::theme::hover_paint(
                        Style::default().fg(Theme::text_muted()),
                        send_hot,
                    ),
                ));
                spans.push(Span::styled(
                    EDIT,
                    crate::ui::theme::hover_paint(
                        Style::default().fg(Theme::text_secondary()),
                        edit_hot,
                    ),
                ));
                spans.push(Span::styled(
                    CANCEL,
                    crate::ui::theme::hover_paint(
                        Style::default().fg(Theme::accent_red()),
                        drop_hot,
                    ),
                ));
            }
            let mut line = Line::from(spans);
            if matches!(state.hover, HoverKind::Queue(j) if j == i) {
                crate::ui::theme::hover_line(&mut line, true);
            }
            lines.push(line);
            vis = vis.saturating_add(1);
        }
        frame.render_widget(
            Paragraph::new(lines).style(Style::default().bg(Theme::bg_base())),
            area,
        );
    }
}

fn compact_preview(text: &str, max: usize) -> String {
    let one = text.lines().next().unwrap_or("").trim();
    if one.width() <= max {
        return one.to_string();
    }
    let mut out = String::new();
    for c in one.chars() {
        if out.width() + 1 >= max {
            break;
        }
        out.push(c);
    }
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;

    #[test]
    fn queue_window_keeps_edit_visible() {
        assert_eq!(queue_window(1, None), (0, 1));
        assert_eq!(queue_window(5, None), (0, 3));
        assert_eq!(queue_window(5, Some(0)), (0, 3));
        assert_eq!(queue_window(5, Some(4)), (2, 5));
    }

    #[test]
    fn action_chips_match_grok() {
        assert_eq!(SEND, "[Send now]");
        assert_eq!(EDIT, "[edit]");
        assert_eq!(CANCEL, "[cancel]");
        assert!(actions_width() > 20);
        let mut s = AppState::new();
        s.enqueue("test queue".into());
        assert_eq!(queue_height(&s), 1);
    }
}
