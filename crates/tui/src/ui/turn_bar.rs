use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::theme::Theme;
use crate::state::AppState;

/// Scan hairline + wait copy, immediately above the composer.
pub struct TurnBar;

const SCAN_HEAD: usize = 8;
const SCAN_TRAIL: usize = 16;
const SCAN_SPEED: u64 = 3;

impl TurnBar {
    pub fn height(state: &AppState) -> u16 {
        if state.metrics.is_compacting {
            2
        } else if !state.show_turn_bar() {
            0
        } else if !state.compact && state.turn_detail().is_some() {
            2
        } else {
            1
        }
    }

    pub fn scan_height(state: &AppState) -> u16 {
        if state.compact {
            0
        } else {
            u16::from(state.show_turn_bar() || state.metrics.is_compacting)
        }
    }

    pub fn render(frame: &mut Frame, area: Rect, state: &AppState, frame_count: u64) {
        if state.metrics.is_compacting {
            render_compact(frame, area, state, frame_count);
        } else {
            render_status(frame, area, state, frame_count);
        }
    }

    pub fn render_scan(frame: &mut Frame, area: Rect, state: &AppState, frame_count: u64) {
        if area.height == 0 {
            return;
        }
        let spans = if state.metrics.is_compacting {
            fold_spans(area.width as usize, super::motion::frame(frame_count))
        } else {
            scan_spans(area.width as usize, super::motion::frame(frame_count))
        };
        frame.render_widget(
            Paragraph::new(Line::from(spans)).style(Style::default().bg(Theme::bg_base())),
            area,
        );
    }
}

fn render_compact(frame: &mut Frame, area: Rect, state: &AppState, frame_count: u64) {
    if area.height == 0 {
        return;
    }
    let spin = super::motion::spinner_for(state.indicator, frame_count);
    let secs = state
        .metrics
        .compaction_started
        .map(|t| t.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let dots = super::rhythm::ellipsis(frame_count);
    let used = if state.metrics.context_used > 0 {
        state.metrics.context_used
    } else {
        state.metrics.total_tokens
    };
    let window = if state.metrics.context_limit > 0 {
        format!(
            "  {}/{}",
            super::context::fmt_k(used),
            super::context::fmt_k(state.metrics.context_limit)
        )
    } else {
        String::new()
    };
    let left = format!("  {spin}  folding context{dots}");
    let right = format!("{}{}  ", fmt_duration(secs), window);
    let width = area.width as usize;
    let pad = width.saturating_sub(left.width() + right.width()).max(1);
    let header = Line::from(vec![
        Span::styled(
            left,
            Style::default()
                .fg(Theme::brand_gold())
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" ".repeat(pad)),
        Span::styled(right, Style::default().fg(Theme::text_muted())),
    ]);
    if area.height == 1 {
        frame.render_widget(
            Paragraph::new(header).style(Style::default().bg(Theme::bg_surface())),
            area,
        );
        return;
    }
    let rows = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Length(1),
            ratatui::layout::Constraint::Length(1),
        ])
        .split(area);
    frame.render_widget(
        Paragraph::new(header).style(Style::default().bg(Theme::bg_surface())),
        rows[0],
    );
    frame.render_widget(
        Paragraph::new(Line::from(fold_spans(
            rows[1].width as usize,
            super::motion::frame(frame_count),
        )))
        .style(Style::default().bg(Theme::bg_surface())),
        rows[1],
    );
}

fn render_status(frame: &mut Frame, area: Rect, state: &AppState, frame_count: u64) {
    let spin = super::motion::spinner_for(state.indicator, frame_count);
    let elapsed = state.elapsed_secs();
    let label = format!(
        "{}{}",
        turn_label(state).trim_end_matches('.'),
        super::rhythm::ellipsis(frame_count)
    );
    let width = area.width as usize;
    let tokens = token_chip(state);
    let right = if tokens.is_empty() {
        format!("{}  [esc stop]", fmt_duration(elapsed))
    } else {
        format!("{}  {tokens}  [esc stop]", fmt_duration(elapsed))
    };
    let right_w = right.width();
    let budget = width.saturating_sub(spin.width() + right_w + 6);
    let label = truncate_width(&label, budget.max(4));
    let left = format!("  {spin}  {label}");
    let pad = width.saturating_sub(left.width() + right_w + 2);

    let line = Line::from(vec![
        Span::styled(
            left,
            Style::default()
                .fg(Theme::brand_gold())
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" ".repeat(pad.max(1))),
        Span::styled(
            format!("{right}  "),
            Style::default().fg(Theme::text_muted()),
        ),
    ]);
    if area.height >= 2 {
        let rows = ratatui::layout::Layout::default()
            .direction(ratatui::layout::Direction::Vertical)
            .constraints([
                ratatui::layout::Constraint::Length(1),
                ratatui::layout::Constraint::Length(1),
            ])
            .split(area);
        frame.render_widget(
            Paragraph::new(line).style(Style::default().bg(Theme::bg_base())),
            rows[0],
        );
        let detail = state.turn_detail().unwrap_or_default();
        let detail = truncate_width(&detail, width.saturating_sub(4));
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!("     {detail}"),
                Style::default().fg(Theme::text_secondary()),
            )))
            .style(Style::default().bg(Theme::bg_base())),
            rows[1],
        );
        return;
    }
    frame.render_widget(
        Paragraph::new(line).style(Style::default().bg(Theme::bg_base())),
        area,
    );
}

/// Head position in `0 .. width + SCAN_HEAD + SCAN_TRAIL` (enters from the left).
pub fn scan_head(frame_count: u64, width: usize) -> i32 {
    let travel = width.saturating_add(SCAN_HEAD + SCAN_TRAIL).max(1) as u64;
    ((frame_count.wrapping_mul(SCAN_SPEED)) % travel) as i32
}

/// Inward fold: gold mass in the center, edges recede. O(width) run-length spans.
pub fn fold_spans(width: usize, frame_count: u64) -> Vec<Span<'static>> {
    if width == 0 {
        return Vec::new();
    }
    let t = ((frame_count / 2) % 40) as f32 / 40.0;
    let tri = if t < 0.5 { t * 2.0 } else { (1.0 - t) * 2.0 };
    let half = ((0.10 + 0.36 * tri) * width as f32).round() as i32;
    let mid = width as i32 / 2;
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut run = String::new();
    let mut run_color: Option<Color> = None;
    for x in 0..width {
        let d = (x as i32 - mid).abs();
        let color = if d <= half {
            Theme::brand_gold()
        } else if d <= half + 3 {
            Theme::brand_orange()
        } else {
            Theme::border_subtle()
        };
        if run_color == Some(color) {
            run.push('▃');
        } else {
            if !run.is_empty() {
                spans.push(Span::styled(
                    std::mem::take(&mut run),
                    Style::default().fg(run_color.unwrap_or(Theme::border_subtle())),
                ));
            }
            run.push('▃');
            run_color = Some(color);
        }
    }
    if !run.is_empty() {
        spans.push(Span::styled(
            run,
            Style::default().fg(run_color.unwrap_or(Theme::border_subtle())),
        ));
    }
    spans
}

fn scan_spans(width: usize, frame_count: u64) -> Vec<Span<'static>> {
    if width == 0 {
        return Vec::new();
    }
    let head = scan_head(frame_count, width);
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut run = String::new();
    let mut run_color: Option<Color> = None;
    for x in 0..width {
        let color = scan_cell(x as i32, head);
        if run_color == Some(color) {
            run.push('▁');
        } else {
            flush_scan(&mut spans, &mut run, run_color);
            run.push('▁');
            run_color = Some(color);
        }
    }
    flush_scan(&mut spans, &mut run, run_color);
    spans
}

fn scan_cell(x: i32, head: i32) -> Color {
    let d = head - x;
    if d >= 0 && d < SCAN_HEAD as i32 {
        Theme::brand_gold()
    } else if d >= SCAN_HEAD as i32 && d < (SCAN_HEAD + SCAN_TRAIL) as i32 {
        let t = (d - SCAN_HEAD as i32) as f32 / SCAN_TRAIL as f32;
        if t < 0.45 {
            Theme::brand_orange()
        } else {
            Theme::border_subtle()
        }
    } else {
        Theme::bg_surface()
    }
}

fn flush_scan(spans: &mut Vec<Span<'static>>, run: &mut String, color: Option<Color>) {
    if run.is_empty() {
        return;
    }
    let text = std::mem::take(run);
    let fg = color.unwrap_or(Theme::bg_surface());
    spans.push(Span::styled(
        text,
        Style::default().fg(fg).bg(Theme::bg_base()),
    ));
}

fn turn_label(state: &AppState) -> String {
    state.live_status()
}

pub fn fmt_duration(secs: f64) -> String {
    if secs < 10.0 {
        format!("{secs:.1}s")
    } else if secs < 60.0 {
        format!("{:.0}s", secs)
    } else if secs < 3600.0 {
        let m = secs as u64 / 60;
        let s = secs as u64 % 60;
        format!("{m}m{s:02}s")
    } else {
        let h = secs as u64 / 3600;
        let m = (secs as u64 % 3600) / 60;
        format!("{h}h{m:02}m")
    }
}

fn token_chip(state: &AppState) -> String {
    let n = if state.metrics.context_used > 0 {
        state.metrics.context_used
    } else if state.metrics.streaming_tokens_count > 0 {
        state.metrics.streaming_tokens_count
    } else {
        state.metrics.total_tokens
    };
    if n >= 1000 {
        format!("↓{:.0}k", n as f64 / 1000.0)
    } else if n > 0 {
        format!("↓{n}")
    } else {
        String::new()
    }
}

fn truncate_width(s: &str, max: usize) -> String {
    if s.width() <= max {
        return s.to_string();
    }
    let mut out = String::new();
    for c in s.chars() {
        let w = c.width().unwrap_or(1);
        if out.width() + w + 1 > max {
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
    fn duration_formats() {
        assert_eq!(fmt_duration(0.3), "0.3s");
        assert_eq!(fmt_duration(12.0), "12s");
        assert_eq!(fmt_duration(386.0), "6m26s");
        assert_eq!(fmt_duration(503.0), "8m23s");
        assert_eq!(fmt_duration(3600.0 * 23.0 + 54.0 * 60.0), "23h54m");
    }

    #[test]
    fn default_wait_copy() {
        let mut s = AppState::new();
        s.is_generating = true;
        s.metrics.activity = "thinking".into();
        assert_eq!(turn_label(&s), "Thinking");
        s.messages.push(crate::state::ChatMessage {
            id: "t".into(),
            role: crate::state::MessageRole::Tool {
                name: "terminal".into(),
                status: "running...".into(),
                tool_id: None,
            },
            content: r#"{"command":"cargo test"}"#.into(),
            output: String::new(),
            timestamp: chrono::Local::now(),
            is_streaming: false,
        });
        assert_eq!(turn_label(&s), "Running cargo test");
        s.messages.clear();
        s.messages.push(crate::state::ChatMessage {
            id: "a".into(),
            role: crate::state::MessageRole::Assistant,
            content: "hi".into(),
            output: String::new(),
            timestamp: chrono::Local::now(),
            is_streaming: true,
        });
        assert_eq!(turn_label(&s), "Writing");
    }

    #[test]
    fn token_chip_k() {
        let mut s = AppState::new();
        s.metrics.context_used = 382_000;
        assert_eq!(token_chip(&s), "↓382k");
    }

    #[test]
    fn activity_uses_two_rows() {
        let mut s = AppState::new();
        assert_eq!(TurnBar::height(&s), 0);
        s.is_generating = true;
        assert_eq!(TurnBar::height(&s), 1);
        s.metrics.activity = "◆ Security Advisories".into();
        assert_eq!(TurnBar::height(&s), 2);
        assert_eq!(s.turn_detail().as_deref(), Some("◆ Security Advisories"));
        assert_eq!(TurnBar::scan_height(&s), 1);
        s.is_generating = false;
        s.metrics.is_compacting = true;
        assert_eq!(TurnBar::height(&s), 2);
        assert_eq!(TurnBar::scan_height(&s), 1);
    }

    #[test]
    fn wait_verbs_stay_one_row() {
        let mut s = AppState::new();
        s.is_generating = true;
        s.metrics.activity = "mulling".into();
        assert_eq!(turn_label(&s), "Mulling");
        assert_eq!(s.turn_detail(), None);
        assert_eq!(TurnBar::height(&s), 1);
        s.metrics.activity = "thinking".into();
        assert_eq!(turn_label(&s), "Thinking");
        assert_eq!(s.turn_detail(), None);
        assert_eq!(TurnBar::height(&s), 1);
        s.metrics.activity = "contemplating".into();
        assert_eq!(turn_label(&s), "Contemplating");
        assert_eq!(s.turn_detail(), None);
    }

    #[test]
    fn fold_bar_is_centered_gold() {
        crate::ui::theme::apply(crate::ui::theme::lookup("gold"));
        let spans = fold_spans(40, 10);
        assert!(!spans.is_empty());
        let joined: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(joined.chars().count(), 40);
        let gold = Theme::brand_gold();
        assert!(spans.iter().any(|s| s.style.fg == Some(gold)));
    }

    #[test]
    fn scan_travels_left_to_right() {
        let a = scan_head(0, 80);
        let b = scan_head(4, 80);
        assert!(b > a, "head moves right: {a} -> {b}");
        let wrapped = scan_head(10_000, 80);
        assert!(wrapped >= 0);
        assert!(wrapped < 80 + SCAN_HEAD as i32 + SCAN_TRAIL as i32);
    }

    #[test]
    fn scan_covers_full_width() {
        let spans = scan_spans(40, 3);
        let cells: usize = spans.iter().map(|s| s.content.chars().count()).sum();
        assert_eq!(cells, 40);
        assert!(spans.iter().any(|s| s.content.contains('▁')));
    }
}
