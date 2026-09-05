use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use super::theme::Theme;
use crate::state::AppState;

pub struct DiffPane;

impl DiffPane {
    pub fn render(frame: &mut Frame, area: Rect, state: &AppState) {
        let title = if let Some(id) = &state.diff_tool_id {
            let file = state
                .messages
                .iter()
                .find(|m| m.id == *id)
                .and_then(|m| crate::ui::stream::first_path(&m.content))
                .unwrap_or_else(|| "edit".into());
            format!(" diff  {file}  click again to close ")
        } else {
            " diff  ctrl+d ".into()
        };
        let block = Block::default()
            .title(Span::styled(
                title,
                Style::default().fg(Theme::brand_gold()),
            ))
            .borders(Borders::LEFT)
            .border_style(Style::default().fg(Theme::border_subtle()))
            .style(Style::default().bg(Theme::bg_base()));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.height == 0 {
            return;
        }

        let lines: Vec<Line> = state
            .diff_text
            .lines()
            .map(|raw| {
                let style = if raw.starts_with("+++") || raw.starts_with("---") {
                    Style::default().fg(Theme::text_muted())
                } else if raw.starts_with('+') {
                    Style::default().fg(Theme::accent_green())
                } else if raw.starts_with('-') {
                    Style::default().fg(Theme::accent_red())
                } else if raw.starts_with("@@") {
                    Style::default().fg(Theme::brand_orange())
                } else if raw.starts_with("##") || raw.starts_with(" M") || raw.starts_with("??") {
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Theme::text_secondary())
                };
                Line::from(Span::styled(raw.to_string(), style))
            })
            .collect();

        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .style(Style::default().bg(Theme::bg_base())),
            inner,
        );
    }
}
