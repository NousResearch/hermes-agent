use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use super::theme::Theme;
use crate::state::AppState;

pub struct FilesPane;

impl FilesPane {
    pub fn render(frame: &mut Frame, area: Rect, state: &mut AppState) {
        let border =
            if state.files_focus || matches!(state.hover, crate::state::HoverKind::Files(_)) {
                Theme::brand_gold()
            } else {
                Theme::border_subtle()
            };
        let title = if state.files_focus {
            " files  ↑↓ enter  o open  r restore  u undo  esc "
        } else {
            " files  ctrl+e "
        };
        let block = Block::default()
            .title(Span::styled(
                title,
                Style::default().fg(if state.files_focus {
                    Theme::brand_gold()
                } else {
                    Theme::text_muted()
                }),
            ))
            .borders(Borders::LEFT)
            .border_style(Style::default().fg(border))
            .style(Style::default().bg(Theme::bg_base()));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.height < 4 {
            return;
        }
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(48),
                Constraint::Length(1),
                Constraint::Min(3),
            ])
            .split(inner);

        let items: Vec<ListItem> = if state.files_rows.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no files here · g refresh",
                Style::default().fg(Theme::text_muted()),
            )))]
        } else {
            state
                .files_rows
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    let pad = "  ".repeat(row.depth);
                    let twirl = if !row.is_dir {
                        " "
                    } else if row.expanded {
                        "▾"
                    } else {
                        "▸"
                    };
                    let mark = match row.git {
                        'M' => " M",
                        'A' => " A",
                        'D' => " D",
                        '?' => " ?",
                        _ => "",
                    };
                    let name = if row.is_dir {
                        format!("{pad}{twirl} {}/", row.name)
                    } else {
                        format!("{pad}{twirl} {}", row.name)
                    };
                    let git_style = match row.git {
                        'M' => Theme::brand_gold(),
                        'A' => Theme::accent_green(),
                        'D' => Theme::accent_red(),
                        '?' => Theme::brand_orange(),
                        _ => Theme::text_secondary(),
                    };
                    let hot = matches!(state.hover, crate::state::HoverKind::Files(h) if h == i)
                        && i != state.files_selected;
                    ListItem::new(Line::from(vec![
                        Span::styled(name, Style::default().fg(Theme::text_primary())),
                        Span::styled(
                            mark.to_string(),
                            Style::default().fg(git_style).add_modifier(Modifier::BOLD),
                        ),
                    ]))
                    .style(if hot {
                        Style::default().bg(Theme::bg_highlight())
                    } else {
                        Style::default()
                    })
                })
                .collect()
        };

        let list = List::new(items).highlight_style(
            Style::default()
                .bg(Theme::bg_highlight())
                .fg(Theme::brand_gold())
                .add_modifier(Modifier::BOLD),
        );
        let mut list_state = ListState::default();
        if !state.files_rows.is_empty() {
            list_state.select(Some(state.files_selected));
        }
        state.files_list = Some(split[0]);
        frame.render_stateful_widget(list, split[0], &mut list_state);
        state.files_offset = list_state.offset();

        let rule = "─".repeat(split[1].width as usize);
        frame.render_widget(
            Paragraph::new(Span::styled(
                rule,
                Style::default().fg(Theme::border_subtle()),
            )),
            split[1],
        );

        let preview: Vec<Line> = state
            .files_preview
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
                } else {
                    Style::default().fg(Theme::text_secondary())
                };
                Line::from(Span::styled(raw.to_string(), style))
            })
            .collect();
        frame.render_widget(
            Paragraph::new(preview)
                .wrap(Wrap { trim: false })
                .style(Style::default().bg(Theme::bg_base())),
            split[2],
        );
    }
}
