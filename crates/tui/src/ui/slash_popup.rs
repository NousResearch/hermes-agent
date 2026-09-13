use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem},
    Frame,
};

use super::theme::Theme;
use crate::slash::{SlashEntry, SlashKind};

pub struct SlashPopup;

impl SlashPopup {
    pub fn render(frame: &mut Frame, area: Rect, selected: usize, ranked: &[&SlashEntry]) {
        if ranked.is_empty() {
            return;
        }

        let height = (ranked.len() + 2).min(12) as u16;
        let popup_area = Rect {
            x: area.x + 2,
            y: area.y.saturating_sub(height),
            width: area.width.saturating_sub(4).min(84),
            height,
        };

        let items: Vec<ListItem> = ranked
            .iter()
            .enumerate()
            .map(|(idx, cmd)| {
                let is_selected = idx == selected;
                let tag = match cmd.kind {
                    SlashKind::Skill => "skill",
                    SlashKind::Command => "cmd",
                    SlashKind::Local => "tui",
                };
                let name = if cmd.name.chars().count() > 22 {
                    let take: String = cmd.name.chars().take(21).collect();
                    format!("{take}…")
                } else {
                    format!("{:<22}", cmd.name)
                };
                let spans = vec![
                    Span::styled(
                        name,
                        if is_selected {
                            Style::default()
                                .fg(Theme::brand_gold())
                                .add_modifier(Modifier::BOLD)
                        } else if cmd.kind == SlashKind::Skill {
                            Style::default().fg(Theme::accent_cyan())
                        } else {
                            Style::default().fg(Theme::text_secondary())
                        },
                    ),
                    Span::styled(
                        format!(" {tag:<5} "),
                        Style::default().fg(Theme::text_dim()),
                    ),
                    Span::styled(
                        if cmd.args_hint.is_empty() {
                            cmd.description.clone()
                        } else {
                            format!("{}  {}", cmd.args_hint, cmd.description)
                        },
                        if is_selected {
                            Style::default().fg(Theme::text_primary())
                        } else {
                            Style::default().fg(Theme::text_secondary())
                        },
                    ),
                ];
                ListItem::new(Line::from(spans)).style(if is_selected {
                    Style::default().bg(Theme::bg_highlight())
                } else {
                    Style::default()
                })
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .title(Span::styled(
                    " /  tab complete  enter run ",
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Theme::border_focus()))
                .style(Style::default().bg(Theme::bg_popup())),
        );

        frame.render_widget(Clear, popup_area);
        frame.render_widget(list, popup_area);
    }
}
