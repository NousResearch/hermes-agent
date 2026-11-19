use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use unicode_width::UnicodeWidthStr;

use super::theme::Theme;
use crate::state::{AppState, HitRange, HoverKind};
use crate::tips::{self, TIPS};

pub struct TipBar;

impl TipBar {
    /// One pad row, copy, hairline — same inset above the copy as below it.
    pub const HEIGHT: u16 = 3;

    pub fn render(frame: &mut Frame, area: Rect, state: &mut AppState) {
        if area.height == 0 || !state.tips_open {
            state.hit_tips_close = None;
            state.hit_tips_bar = None;
            return;
        }
        let rows = match area.height {
            1 => Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(1)])
                .split(area),
            2 => Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(1), Constraint::Length(1)])
                .split(area),
            3 => Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),
                    Constraint::Length(1),
                    Constraint::Length(1),
                ])
                .split(area),
            _ => Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),
                    Constraint::Length(1),
                    Constraint::Length(1),
                    Constraint::Length(1),
                ])
                .split(area),
        };
        let (pads, content, rule) = match rows.len() {
            4 => (vec![rows[0], rows[1]], rows[2], Some(rows[3])),
            3 => (vec![rows[0]], rows[1], Some(rows[2])),
            2 => (Vec::new(), rows[0], Some(rows[1])),
            _ => (Vec::new(), rows[0], None),
        };
        for pad_row in pads {
            frame.render_widget(
                Paragraph::new("").style(Style::default().bg(Theme::bg_surface())),
                pad_row,
            );
        }
        let width = content.width as usize;
        let reveal = state.reveal();
        if reveal < 0.04 {
            frame.render_widget(
                Paragraph::new("").style(Style::default().bg(Theme::bg_surface())),
                content,
            );
            if let Some(rule) = rule {
                frame.render_widget(
                    Paragraph::new("").style(Style::default().bg(Theme::bg_base())),
                    rule,
                );
            }
            state.hit_tips_close = None;
            state.hit_tips_bar = None;
            return;
        }

        let close = " × ";
        let close_w = close.width();
        let label = if width > 56 { "did you know" } else { "tip" };
        let prefix = format!("  {label} · ");
        let prefix_w = prefix.width();
        let body_w = width.saturating_sub(prefix_w + close_w).max(8);
        let tip = TIPS[state.tip_index % TIPS.len()];
        let body = tips::ellipsize(tip, body_w);
        let gap = width.saturating_sub(prefix_w + body.width() + close_w);

        let close_hot = state.hover == HoverKind::TipsClose;
        let bar_hot = state.hover == HoverKind::TipsBar;
        let mut spans = vec![
            Span::styled(
                prefix,
                crate::ui::theme::hover_paint(
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                    bar_hot,
                ),
            ),
            Span::styled(
                body,
                crate::ui::theme::hover_paint(
                    Style::default().fg(Theme::text_secondary()),
                    bar_hot,
                ),
            ),
            Span::raw(" ".repeat(gap)),
            Span::styled(
                close,
                crate::ui::theme::hover_paint(Style::default().fg(Theme::text_dim()), close_hot),
            ),
        ];
        if reveal < 0.995 {
            crate::ui::theme::fade_spans(&mut spans, reveal);
        }

        let close_x0 = content
            .x
            .saturating_add(content.width.saturating_sub(close_w as u16));
        state.hit_tips_close = Some(HitRange {
            y: content.y,
            x0: close_x0,
            x1: content.x.saturating_add(content.width),
        });
        state.hit_tips_bar = Some(HitRange {
            y: content.y,
            x0: content.x,
            x1: close_x0,
        });

        frame.render_widget(
            Paragraph::new(Line::from(spans)).style(Style::default().bg(Theme::bg_surface())),
            content,
        );
        if let Some(rule) = rule {
            let line = "─".repeat(rule.width as usize);
            frame.render_widget(
                Paragraph::new(Span::styled(
                    line,
                    Style::default().fg(Theme::border_subtle()),
                ))
                .style(Style::default().bg(Theme::bg_base())),
                rule,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;

    #[test]
    fn height_pads_chrome() {
        assert_eq!(TipBar::HEIGHT, 3);
        let s = AppState::new();
        assert!(s.tips_open);
        assert!(s.tip_index < TIPS.len());
    }
}
