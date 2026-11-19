use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem},
    Frame,
};

use super::theme::Theme;
use crate::complete::CompleteItem;

pub struct CompletePopup;

impl CompletePopup {
    pub fn render(frame: &mut Frame, area: Rect, items: &[CompleteItem], selected: usize) {
        if items.is_empty() || area.width < 20 {
            return;
        }
        let height = (items.len() + 2).min(12) as u16;
        let popup_area = Rect {
            x: area.x + 2,
            y: area.y.saturating_sub(height),
            width: area.width.saturating_sub(4).min(84),
            height,
        };
        let rows: Vec<ListItem> = items
            .iter()
            .enumerate()
            .map(|(idx, it)| {
                let sel = idx == selected;
                let name = if it.display.chars().count() > 28 {
                    let take: String = it.display.chars().take(27).collect();
                    format!("{take}…")
                } else {
                    format!("{:<28}", it.display)
                };
                let spans = vec![
                    Span::styled(
                        name,
                        if sel {
                            Style::default()
                                .fg(Theme::brand_gold())
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Theme::text_secondary())
                        },
                    ),
                    Span::styled(
                        format!(" {}", it.meta),
                        Style::default().fg(Theme::text_dim()),
                    ),
                ];
                ListItem::new(Line::from(spans)).style(if sel {
                    Style::default().bg(Theme::bg_highlight())
                } else {
                    Style::default()
                })
            })
            .collect();
        let list = List::new(rows).block(
            Block::default()
                .title(Span::styled(
                    " @  tab complete  esc ",
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
