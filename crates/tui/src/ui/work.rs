use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use super::theme::Theme;
use crate::state::AppState;

/// Right rail: background processes + subagents, with output or git diff-check.
pub struct WorkPane;

impl WorkPane {
    pub fn render(frame: &mut Frame, area: Rect, state: &mut AppState, frame_count: u64) {
        let border = if state.work_focus || matches!(state.hover, crate::state::HoverKind::Work(_))
        {
            Theme::brand_gold()
        } else {
            Theme::border_subtle()
        };
        let dirty = if state.work_dirty.is_empty() {
            String::new()
        } else {
            format!("  {}", state.work_dirty)
        };
        let title = if state.work_focus {
            if state.work_show_diff {
                format!(
                    " work  {n} files{dirty}  ↑↓ file  pgup patch  d back  esc ",
                    n = state.work_diff_files.len()
                )
            } else {
                " work  ↑↓  d diff-check  x stop  esc ".to_string()
            }
        } else {
            " work  ctrl+w ".into()
        };
        let block = Block::default()
            .title(Span::styled(
                title,
                Style::default().fg(if state.work_focus {
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
        let list_pct = if state.work_show_diff { 28 } else { 42 };
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(list_pct),
                Constraint::Length(1),
                Constraint::Min(6),
            ])
            .split(inner);

        let spin = super::motion::spinner_for(state.indicator, frame_count);
        let (items, selected, empty) = if state.work_show_diff {
            if state.work_diff_files.is_empty() {
                (
                    vec![ListItem::new(Line::from(Span::styled(
                        "  working tree clean",
                        Style::default().fg(Theme::text_muted()),
                    )))],
                    0,
                    true,
                )
            } else {
                let items = state
                    .work_diff_files
                    .iter()
                    .map(|f| {
                        let (mark, color) = match f.mark {
                            'M' => ("M", Theme::brand_gold()),
                            'A' => ("A", Theme::accent_green()),
                            'D' => ("D", Theme::accent_red()),
                            '?' => ("?", Theme::brand_orange()),
                            _ => ("·", Theme::text_muted()),
                        };
                        let bang = if f.check.is_empty() { " " } else { "!" };
                        ListItem::new(Line::from(vec![
                            Span::styled(
                                format!("  {mark}{bang} "),
                                Style::default().fg(color).add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(
                                crate::tips::ellipsize(&f.rel, 34),
                                Style::default().fg(Theme::text_primary()),
                            ),
                        ]))
                    })
                    .collect();
                (items, state.work_diff_selected, false)
            }
        } else if state.agent_rows.is_empty() && state.bg_tasks.is_empty() {
            (
                vec![ListItem::new(Line::from(Span::styled(
                    "  no processes · /background or spawn",
                    Style::default().fg(Theme::text_muted()),
                )))],
                0,
                true,
            )
        } else {
            let mut items: Vec<ListItem> = state
                .agent_rows
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let (glyph, color) = row_glyph(a, spin);
                    let kind = if a.is_process() { "proc" } else { "agent" };
                    let extra = if a.is_process() {
                        a.pid.map(|p| format!(" pid {p}")).unwrap_or_default()
                    } else if !a.last_tool.is_empty() {
                        format!("  {}", crate::tips::ellipsize(&a.last_tool, 14))
                    } else {
                        String::new()
                    };
                    let hot = matches!(state.hover, crate::state::HoverKind::Work(h) if h == i)
                        && i != state.work_selected;
                    ListItem::new(Line::from(vec![
                        Span::styled(format!("  {glyph} {kind}  "), Style::default().fg(color)),
                        Span::styled(
                            crate::tips::ellipsize(&a.title, 22),
                            Style::default().fg(Theme::text_primary()),
                        ),
                        Span::styled(extra, Style::default().fg(Theme::text_muted())),
                    ]))
                    .style(if hot {
                        Style::default().bg(Theme::bg_highlight())
                    } else {
                        Style::default()
                    })
                })
                .collect();
            for (j, task) in state.bg_tasks.iter().enumerate() {
                let i = state.agent_rows.len() + j;
                let running = task.status == crate::state::BgStatus::Running;
                let glyph = if running { spin } else { "✓" };
                let color = if running {
                    Theme::brand_gold()
                } else {
                    Theme::accent_green()
                };
                let hot = matches!(state.hover, crate::state::HoverKind::Work(h) if h == i);
                items.push(
                    ListItem::new(Line::from(vec![
                        Span::styled(format!("  {glyph} bg  "), Style::default().fg(color)),
                        Span::styled(
                            crate::tips::ellipsize(&task.prompt, 28),
                            Style::default().fg(Theme::text_primary()),
                        ),
                    ]))
                    .style(if hot {
                        Style::default().bg(Theme::bg_highlight())
                    } else {
                        Style::default()
                    }),
                );
            }
            let selected = if state.agent_rows.is_empty() {
                0
            } else {
                state.work_selected.min(state.agent_rows.len() - 1)
            };
            (items, selected, false)
        };
        let list = List::new(items).highlight_style(
            Style::default()
                .bg(Theme::bg_highlight())
                .fg(Theme::brand_gold())
                .add_modifier(Modifier::BOLD),
        );
        let mut list_state = ListState::default();
        if !empty {
            list_state.select(Some(selected));
        }
        state.work_list = Some(split[0]);
        frame.render_stateful_widget(list, split[0], &mut list_state);
        state.work_offset = list_state.offset();

        let rule = "─".repeat(split[1].width as usize);
        frame.render_widget(
            Paragraph::new(Span::styled(
                rule,
                Style::default().fg(Theme::border_subtle()),
            )),
            split[1],
        );

        let mut preview = if state.work_show_diff {
            diff_lines(&state.diff_text)
        } else {
            detail_lines(state)
        };
        if state.work_show_diff && state.work_diff_offset > 0 {
            let skip = state.work_diff_offset.min(preview.len().saturating_sub(1));
            preview = preview.into_iter().skip(skip).collect();
        }
        let room = split[2].height as usize;
        if preview.len() > room {
            preview.truncate(room);
        }
        frame.render_widget(
            Paragraph::new(preview)
                .wrap(Wrap { trim: false })
                .style(Style::default().bg(Theme::bg_base())),
            split[2],
        );
    }
}

fn row_glyph(
    a: &crate::state::AgentRow,
    spin: &'static str,
) -> (&'static str, ratatui::style::Color) {
    if a.is_process() {
        if a.status == "running" {
            (spin, Theme::accent_cyan())
        } else {
            ("▸", Theme::text_muted())
        }
    } else {
        match a.status.as_str() {
            "running" => (spin, Theme::brand_gold()),
            "queued" => ("○", Theme::text_muted()),
            "completed" => ("✓", Theme::accent_green()),
            "failed" | "error" => ("✗", Theme::accent_red()),
            _ => ("●", Theme::text_secondary()),
        }
    }
}

fn diff_lines(text: &str) -> Vec<Line<'static>> {
    text.lines()
        .map(|raw| {
            let style = if raw.starts_with("##") {
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD)
            } else if raw.starts_with("+++") || raw.starts_with("---") {
                Style::default().fg(Theme::text_muted())
            } else if raw.starts_with('+') {
                Style::default().fg(Theme::accent_green())
            } else if raw.starts_with('-') {
                Style::default().fg(Theme::accent_red())
            } else if raw.starts_with("@@") {
                Style::default().fg(Theme::brand_orange())
            } else if raw.contains("trailing whitespace")
                || raw.contains("conflict")
                || raw.contains("error:")
            {
                Style::default().fg(Theme::accent_red())
            } else {
                Style::default().fg(Theme::text_secondary())
            };
            Line::from(Span::styled(raw.to_string(), style))
        })
        .collect()
}

fn detail_lines(state: &AppState) -> Vec<Line<'static>> {
    let Some(a) = state.work_selected_row() else {
        return vec![Line::from(Span::styled(
            "  d git diff --check   x stop selected",
            Style::default().fg(Theme::text_dim()),
        ))];
    };
    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled(
        format!("  {}  {}", a.status, a.id),
        Style::default().fg(Theme::brand_gold()),
    )));
    if a.is_process() {
        if let Some(pid) = a.pid {
            lines.push(Line::from(Span::styled(
                format!("  pid {pid}"),
                Style::default().fg(Theme::text_secondary()),
            )));
        }
        if !a.cwd.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("  {}", a.cwd),
                Style::default().fg(Theme::text_muted()),
            )));
        }
        lines.push(Line::from(Span::raw("")));
        if a.output.is_empty() {
            lines.push(Line::from(Span::styled(
                "  (no output yet)",
                Style::default().fg(Theme::text_dim()),
            )));
        } else {
            for raw in a
                .output
                .lines()
                .rev()
                .take(24)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
            {
                lines.push(Line::from(Span::styled(
                    raw.to_string(),
                    Style::default().fg(Theme::text_secondary()),
                )));
            }
        }
    } else {
        if !a.model.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("  {}", a.model),
                Style::default().fg(Theme::text_muted()),
            )));
        }
        if !a.last_tool.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("  now  {}", a.last_tool),
                Style::default().fg(Theme::text_secondary()),
            )));
        }
        if !a.summary.is_empty() {
            lines.push(Line::from(Span::raw("")));
            for raw in a.summary.lines().take(12) {
                lines.push(Line::from(Span::styled(
                    raw.to_string(),
                    Style::default().fg(Theme::text_secondary()),
                )));
            }
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_check_header_is_gold() {
        crate::ui::theme::apply(crate::palette::Palette::gold());
        let lines = diff_lines("## diff --check\n+ok\n trailing whitespace\n");
        assert!(lines[0].spans[0].content.contains("diff --check"));
        assert_eq!(lines[0].spans[0].style.fg, Some(Theme::brand_gold()));
        assert_eq!(lines[1].spans[0].style.fg, Some(Theme::accent_green()));
        assert_eq!(lines[2].spans[0].style.fg, Some(Theme::accent_red()));
        let hunk = diff_lines("@@ -1,3 +1,4 @@\n-old\n+new\n");
        assert_eq!(hunk[0].spans[0].style.fg, Some(Theme::brand_orange()));
        assert_eq!(hunk[1].spans[0].style.fg, Some(Theme::accent_red()));
        assert_eq!(hunk[2].spans[0].style.fg, Some(Theme::accent_green()));
    }
}
