use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use crate::state::AppState;
use crate::ui::theme::Theme;

impl crate::ui::modals::ViewsOverlay {
    pub(crate) fn render_sessions_modal(frame: &mut Frame, area: Rect, state: &AppState) {
        let modal_area = Self::centered_rect(70, 65, area);
        frame.render_widget(Clear, modal_area);

        let items: Vec<ListItem> = if state.sessions_list.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no other sessions · ctrl+n starts a new one",
                Style::default().fg(Theme::text_muted()),
            )))]
        } else {
            let idxs = state.filtered_session_indices();
            if idxs.is_empty() {
                vec![ListItem::new(Line::from(Span::styled(
                    "  no matches · backspace clears filter",
                    Style::default().fg(Theme::text_muted()),
                )))]
            } else {
                idxs.iter()
                    .enumerate()
                    .map(|(vis, &si)| {
                        let sess = &state.sessions_list[si];
                        let is_sel = vis == state.modal_selected;
                        let is_current = state.session_id.as_deref() == Some(&sess.id);

                        let line = Line::from(vec![
                            Span::styled(
                                format!("  {:<26}", sess.id),
                                if is_sel {
                                    Style::default()
                                        .fg(Theme::brand_gold())
                                        .add_modifier(Modifier::BOLD)
                                } else {
                                    Style::default().fg(Theme::accent_cyan())
                                },
                            ),
                            Span::styled(
                                format!("{:<30} ", sess.title),
                                Style::default().fg(Theme::text_primary()),
                            ),
                            Span::styled(
                                sess.updated_at.clone(),
                                Style::default().fg(Theme::text_muted()),
                            ),
                            if is_current {
                                Span::styled(
                                    "● CURRENT",
                                    Style::default()
                                        .fg(Theme::accent_green())
                                        .add_modifier(Modifier::BOLD),
                                )
                            } else if sess.live {
                                Span::styled(
                                    format!(" live {}", sess.status),
                                    Style::default().fg(Theme::accent_cyan()),
                                )
                            } else {
                                Span::raw("")
                            },
                        ]);

                        ListItem::new(line).style(if is_sel {
                            Style::default().bg(Theme::bg_highlight())
                        } else {
                            Style::default()
                        })
                    })
                    .collect()
            }
        };

        let list = List::new(items).block(
            Block::default()
                .title(Span::styled(
                    if state.picker_filter.is_empty() {
                        " sessions  enter switch/resume  d delete  h hide  type to filter  esc "
                            .to_string()
                    } else {
                        format!(" sessions  /{} ", state.picker_filter)
                    },
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Theme::border_focus()))
                .style(Style::default().bg(Theme::bg_popup())),
        );

        frame.render_widget(list, modal_area);
    }

    pub(crate) fn render_profiles_modal(frame: &mut Frame, area: Rect, state: &AppState) {
        let modal_area = Self::centered_rect(74, 65, area);
        frame.render_widget(Clear, modal_area);
        let idxs = state.filtered_profile_indices();
        let items: Vec<ListItem> = if state.profiles.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no profiles · check HERMES_HOME",
                Style::default().fg(Theme::text_muted()),
            )))]
        } else if idxs.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no matches · backspace clears filter",
                Style::default().fg(Theme::text_muted()),
            )))]
        } else {
            idxs.iter()
                .enumerate()
                .map(|(vis, &pi)| {
                    let p = &state.profiles[pi];
                    let is_sel = vis == state.modal_selected;
                    let mark = if p.is_default { "●" } else { " " };
                    let worker = if p.worker_active { "  worker" } else { "" };
                    let last = if p.last_title.is_empty() {
                        p.last_preview.clone()
                    } else {
                        p.last_title.clone()
                    };
                    let line1 = Line::from(vec![
                        Span::styled(
                            format!("  {mark} {:<16}", p.display_name),
                            if is_sel {
                                Style::default()
                                    .fg(Theme::brand_gold())
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Theme::text_primary())
                            },
                        ),
                        Span::styled(
                            format!(
                                "{} · {}  {} skills{worker}",
                                p.model, p.provider, p.skill_count
                            ),
                            Style::default().fg(Theme::text_secondary()),
                        ),
                    ]);
                    let line2 = Line::from(vec![Span::styled(
                        format!(
                            "     {}",
                            if last.is_empty() {
                                p.description.as_str()
                            } else {
                                last.as_str()
                            }
                        ),
                        Style::default().fg(Theme::text_muted()),
                    )]);
                    ListItem::new(vec![line1, line2]).style(if is_sel {
                        Style::default().bg(Theme::bg_highlight())
                    } else {
                        Style::default()
                    })
                })
                .collect()
        };
        let list = List::new(items).block(
            Block::default()
                .title(Span::styled(
                    if state.picker_filter.is_empty() {
                        " profiles  ↑↓ enter resume  i inspect  n new via /profiles new  esc "
                            .to_string()
                    } else {
                        format!(" profiles  /{} ", state.picker_filter)
                    },
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Theme::border_focus()))
                .style(Style::default().bg(Theme::bg_popup())),
        );
        frame.render_widget(list, modal_area);
    }

    pub(crate) fn render_memory_modal(frame: &mut Frame, area: Rect, state: &AppState) {
        let modal_area = Self::centered_rect(74, 65, area);
        frame.render_widget(Clear, modal_area);
        let idxs = state.filtered_memory_indices();
        let mut items: Vec<ListItem> = state
            .memory_summary
            .iter()
            .map(|line| {
                ListItem::new(Line::from(Span::styled(
                    format!("  {line}"),
                    Style::default().fg(Theme::text_secondary()),
                )))
            })
            .collect();
        if items.is_empty() && state.memory_nodes.is_empty() {
            items.push(ListItem::new(Line::from(Span::styled(
                "  No learned skills or memories yet. /learn writes here.",
                Style::default().fg(Theme::text_muted()),
            ))));
        } else if idxs.is_empty() && !state.memory_nodes.is_empty() {
            items.push(ListItem::new(Line::from(Span::styled(
                "  no matches · backspace clears filter",
                Style::default().fg(Theme::text_muted()),
            ))));
        } else {
            for (vis, &mi) in idxs.iter().enumerate() {
                let m = &state.memory_nodes[mi];
                let is_sel = vis == state.modal_selected;
                let line1 = Line::from(vec![
                    Span::styled(
                        format!("  {:<8} ", m.kind),
                        Style::default().fg(Theme::accent_cyan()),
                    ),
                    Span::styled(
                        m.label.clone(),
                        if is_sel {
                            Style::default()
                                .fg(Theme::brand_gold())
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Theme::text_primary())
                        },
                    ),
                    Span::styled(
                        format!(
                            "  {}",
                            if m.meta.is_empty() {
                                m.id.as_str()
                            } else {
                                m.meta.as_str()
                            }
                        ),
                        Style::default().fg(Theme::text_muted()),
                    ),
                ]);
                let mut lines = vec![line1];
                if !m.body.is_empty() {
                    lines.push(Line::from(Span::styled(
                        format!("     {}", m.body),
                        Style::default().fg(Theme::text_secondary()),
                    )));
                }
                items.push(ListItem::new(lines).style(if is_sel {
                    Style::default().bg(Theme::bg_highlight())
                } else {
                    Style::default()
                }));
            }
        }
        let list = List::new(items).block(
            Block::default()
                .title(Span::styled(
                    if state.picker_filter.is_empty() {
                        " memory  ↑↓ enter peek  e edit  x delete  type to filter  esc ".to_string()
                    } else {
                        format!(" memory  /{} ", state.picker_filter)
                    },
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Theme::border_focus()))
                .style(Style::default().bg(Theme::bg_popup())),
        );
        frame.render_widget(list, modal_area);
    }

    pub(crate) fn render_rollback_modal(frame: &mut Frame, area: Rect, state: &mut AppState) {
        let modal_area = Self::centered_rect(78, 70, area);
        frame.render_widget(Clear, modal_area);
        let has_diff = !state.rollback_diff.is_empty();
        let split = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(if has_diff {
                [Constraint::Percentage(42), Constraint::Percentage(58)]
            } else {
                [Constraint::Percentage(100), Constraint::Percentage(0)]
            })
            .split(modal_area);

        let idxs = state.filtered_checkpoint_indices();
        let items: Vec<ListItem> = if !state.checkpoints_enabled {
            vec![ListItem::new(Line::from(Span::styled(
                "  checkpoints off · enable in hermes config",
                Style::default().fg(Theme::text_muted()),
            )))]
        } else if state.checkpoints.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no checkpoints · the agent writes these as it edits",
                Style::default().fg(Theme::text_muted()),
            )))]
        } else if idxs.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no matches · backspace clears filter",
                Style::default().fg(Theme::text_muted()),
            )))]
        } else {
            idxs.iter()
                .enumerate()
                .map(|(vis, &ci)| {
                    let c = &state.checkpoints[ci];
                    let sel = vis == state.modal_selected;
                    let short: String = c.hash.chars().take(10).collect();
                    let meta = [&c.timestamp, &c.message]
                        .iter()
                        .filter(|s| !s.is_empty())
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(" · ");
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!("  {short}  "),
                            if sel {
                                Style::default()
                                    .fg(Theme::brand_gold())
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Theme::accent_cyan())
                            },
                        ),
                        Span::styled(
                            crate::tips::ellipsize(&meta, 42),
                            Style::default().fg(Theme::text_secondary()),
                        ),
                    ]))
                    .style(if sel {
                        Style::default().bg(Theme::bg_highlight())
                    } else {
                        Style::default()
                    })
                })
                .collect()
        };
        let title = if state.picker_filter.is_empty() {
            " rollback  ↑↓  enter restore  d diff  esc "
        } else {
            " rollback  filter  esc "
        };
        let block = Block::default()
            .title(Span::styled(
                title,
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Theme::border_focus()))
            .style(Style::default().bg(Theme::bg_popup()));
        let inner = block.inner(split[0]);
        let list = List::new(items).block(block);
        let mut list_state = ListState::default();
        if state.picker_len() > 0 {
            list_state.select(Some(state.modal_selected));
        }
        frame.render_stateful_widget(list, split[0], &mut list_state);
        state.picker_list = Some(inner);
        state.picker_offset = list_state.offset();

        if has_diff && split[1].width > 4 {
            let diff_block = Block::default()
                .title(Span::styled(
                    " diff ",
                    Style::default().fg(Theme::text_muted()),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Theme::border_subtle()))
                .style(Style::default().bg(Theme::bg_popup()));
            let inner = diff_block.inner(split[1]);
            frame.render_widget(diff_block, split[1]);
            let mut lines: Vec<Line> = Vec::new();
            for raw in state.rollback_diff.lines().take(inner.height as usize) {
                let color = if raw.starts_with('+') && !raw.starts_with("+++") {
                    Theme::accent_green()
                } else if raw.starts_with('-') && !raw.starts_with("---") {
                    Theme::accent_red()
                } else if raw.starts_with("@@") {
                    Theme::brand_gold()
                } else {
                    Theme::text_secondary()
                };
                lines.push(Line::from(Span::styled(
                    raw.to_string(),
                    Style::default().fg(color),
                )));
            }
            frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
        }
    }

    pub(crate) fn render_peek_modal(frame: &mut Frame, area: Rect, state: &AppState) {
        let modal_area = Self::centered_rect(72, 70, area);
        frame.render_widget(Clear, modal_area);
        let title = format!(
            " {}  ·  {} lines  ·  esc ",
            if state.peek_title.is_empty() {
                "pasted"
            } else {
                state.peek_title.as_str()
            },
            state.peek_body.lines().count().max(1)
        );
        let block = Block::default()
            .title(Span::styled(
                crate::tips::ellipsize(&title, modal_area.width.saturating_sub(2) as usize),
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Theme::border_focus()))
            .style(Style::default().bg(Theme::bg_popup()));
        let inner = block.inner(modal_area);
        frame.render_widget(block, modal_area);
        let mut lines: Vec<Line> = Vec::new();
        if let Some(path) = &state.peek_image {
            let w = inner.width.saturating_sub(2);
            let h = inner.height.min(8);
            lines.extend(crate::ui::markdown::image_thumb_lines(path, w, h));
        }
        let body_lines: Vec<&str> = state.peek_body.lines().collect();
        let off = state.peek_offset.min(body_lines.len().saturating_sub(1));
        let room = inner.height.saturating_sub(lines.len() as u16).max(1) as usize;
        for line in body_lines.iter().skip(off).take(room) {
            lines.push(Line::from(Span::styled(
                (*line).to_string(),
                Style::default().fg(Theme::text_primary()),
            )));
        }
        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .style(Style::default().bg(Theme::bg_popup())),
            inner,
        );
    }

    pub(crate) fn render_replay_modal(frame: &mut Frame, area: Rect, state: &mut AppState) {
        let modal_area = Self::centered_rect(82, 70, area);
        frame.render_widget(Clear, modal_area);
        let title = if state.picker_filter.is_empty() {
            " replay  enter load snapshot  i peek  /replay save  esc ".to_string()
        } else {
            format!(" replay  /{} ", state.picker_filter)
        };
        let block = Block::default()
            .title(Span::styled(
                title,
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Theme::border_focus()))
            .style(Style::default().bg(Theme::bg_popup()));
        let inner = block.inner(modal_area);
        frame.render_widget(block, modal_area);
        let idxs = state.filtered_spawn_indices();
        let items: Vec<ListItem> = if state.spawn_trees.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no archived spawn trees · /replay save after a spawn",
                Style::default().fg(Theme::text_muted()),
            )))]
        } else if idxs.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no matches",
                Style::default().fg(Theme::text_muted()),
            )))]
        } else {
            idxs.iter()
                .enumerate()
                .map(|(vis, &si)| {
                    let e = &state.spawn_trees[si];
                    let sel = vis == state.modal_selected;
                    let label = if e.label.is_empty() {
                        format!("{}×", e.count)
                    } else {
                        format!("{}×  {}", e.count, e.label)
                    };
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!("  #{}  {label}  ", si + 1),
                            if sel {
                                Style::default()
                                    .fg(Theme::brand_gold())
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Theme::text_primary())
                            },
                        ),
                        Span::styled(
                            crate::tips::ellipsize(&e.path, 42),
                            Style::default().fg(Theme::text_muted()),
                        ),
                    ]))
                    .style(if sel {
                        Style::default().bg(Theme::bg_highlight())
                    } else {
                        Style::default()
                    })
                })
                .collect()
        };
        let list = List::new(items);
        let mut list_state = ListState::default();
        if !idxs.is_empty() {
            list_state.select(Some(state.modal_selected));
        }
        frame.render_stateful_widget(list, inner, &mut list_state);
        state.picker_list = Some(inner);
        state.picker_offset = list_state.offset();
    }

    pub(crate) fn render_projects_modal(frame: &mut Frame, area: Rect, state: &mut AppState) {
        let modal_area = Self::centered_rect(78, 70, area);
        frame.render_widget(Clear, modal_area);
        let drilled = state.project_drill.as_deref().unwrap_or("");
        let title = if !drilled.is_empty() {
            if state.picker_filter.is_empty() {
                format!(" project  {drilled}  enter resume  esc back ")
            } else {
                format!(" project  /{} ", state.picker_filter)
            }
        } else if state.picker_filter.is_empty() {
            " projects  enter sessions  s scan  type to filter  esc ".to_string()
        } else {
            format!(" projects  /{} ", state.picker_filter)
        };
        let block = Block::default()
            .title(Span::styled(
                title,
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Theme::border_focus()))
            .style(Style::default().bg(Theme::bg_popup()));
        let inner = block.inner(modal_area);
        frame.render_widget(block, modal_area);
        let items: Vec<ListItem> = if state.project_drill.is_some() {
            let idxs = state.filtered_project_session_indices();
            if state.project_sessions.is_empty() {
                vec![ListItem::new(Line::from(Span::styled(
                    "  no sessions in this project",
                    Style::default().fg(Theme::text_muted()),
                )))]
            } else if idxs.is_empty() {
                vec![ListItem::new(Line::from(Span::styled(
                    "  no matches",
                    Style::default().fg(Theme::text_muted()),
                )))]
            } else {
                idxs.iter()
                    .enumerate()
                    .map(|(vis, &si)| {
                        let sess = &state.project_sessions[si];
                        let sel = vis == state.modal_selected;
                        ListItem::new(Line::from(vec![
                            Span::styled(
                                format!("  {}  ", sess.title),
                                if sel {
                                    Style::default()
                                        .fg(Theme::brand_gold())
                                        .add_modifier(Modifier::BOLD)
                                } else {
                                    Style::default().fg(Theme::text_primary())
                                },
                            ),
                            Span::styled(
                                format!("{}  {}", sess.id, sess.updated_at),
                                Style::default().fg(Theme::text_muted()),
                            ),
                        ]))
                        .style(if sel {
                            Style::default().bg(Theme::bg_highlight())
                        } else {
                            Style::default()
                        })
                    })
                    .collect()
            }
        } else {
            let idxs = state.filtered_project_indices();
            if state.projects_list.is_empty() {
                vec![ListItem::new(Line::from(Span::styled(
                    "  no projects",
                    Style::default().fg(Theme::text_muted()),
                )))]
            } else if idxs.is_empty() {
                vec![ListItem::new(Line::from(Span::styled(
                    "  no matches",
                    Style::default().fg(Theme::text_muted()),
                )))]
            } else {
                idxs.iter()
                    .enumerate()
                    .map(|(vis, &si)| {
                        let p = &state.projects_list[si];
                        let sel = vis == state.modal_selected;
                        ListItem::new(Line::from(vec![
                            Span::styled(
                                format!("  {}  ", p.name),
                                if sel {
                                    Style::default()
                                        .fg(Theme::brand_gold())
                                        .add_modifier(Modifier::BOLD)
                                } else {
                                    Style::default().fg(Theme::text_primary())
                                },
                            ),
                            Span::styled(
                                format!("{} sessions  {}", p.count, p.id),
                                Style::default().fg(Theme::text_muted()),
                            ),
                        ]))
                        .style(if sel {
                            Style::default().bg(Theme::bg_highlight())
                        } else {
                            Style::default()
                        })
                    })
                    .collect()
            }
        };
        let list = List::new(items);
        let mut list_state = ListState::default();
        if state.picker_len() > 0 {
            list_state.select(Some(state.modal_selected));
        }
        frame.render_stateful_widget(list, inner, &mut list_state);
        state.picker_list = Some(inner);
        state.picker_offset = list_state.offset();
    }

    pub(crate) fn render_help_modal(frame: &mut Frame, area: Rect) {
        let modal_area = Self::centered_rect(65, 65, area);
        frame.render_widget(Clear, modal_area);

        let help_text = vec![
            Line::from(vec![Span::styled(
                "  ⌨ Keyboard Navigation & Shortcuts",
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::raw(""),
            Line::from(vec![
                Span::styled(
                    "  Enter          ",
                    Style::default()
                        .fg(Theme::accent_cyan())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Submit  ·  @file Tab-complete  ·  paste collapses to [[ ]]  ·  click to preview",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Shift + Tab    ",
                    Style::default()
                        .fg(Theme::brand_orange())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Cycle plan → ask → yolo  (plan denies writes)",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Tab            ",
                    Style::default()
                        .fg(Theme::accent_purple())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Toggle Expand/Collapse on Thinking traces",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Ctrl + T       ",
                    Style::default()
                        .fg(Theme::accent_yellow())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Agent overview — now / queue / tasks / recent tools",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Ctrl + D       ",
                    Style::default()
                        .fg(Theme::accent_green())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Git diff split vs HEAD (live working tree)",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Ctrl + E       ",
                    Style::default()
                        .fg(Theme::accent_cyan())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Files + diff preview  o open  r restore  u undo  esc composer",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Ctrl + W       ",
                    Style::default()
                        .fg(Theme::accent_cyan())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Work rail — processes + subagents  d git diff --check  x stop",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Ctrl + P       ",
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Command palette  ·  /theme still switches skins",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /tips          ",
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Did-you-know bar — click × to hide, click the bar for the next tip",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /motion        ",
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Toggle chrome motion — gold wash, shimmer, spinners  (/motion off)",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /background    ",
                    Style::default()
                        .fg(Theme::accent_cyan())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Background tasks — type a prompt, enter launch, enter a row to peek",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /context       ",
                    Style::default()
                        .fg(Theme::accent_cyan())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Context map (click the footer bar)  enter compresses",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Ctrl + G       ",
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Tool trace split: pause, edit, resume a step",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Ctrl + O       ",
                    Style::default()
                        .fg(Theme::accent_cyan())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Model picker — Enter on paste key adds the provider",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Ctrl + B       ",
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Git branch picker (Ctrl+B or click chip)",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Ctrl + S       ",
                    Style::default()
                        .fg(Theme::accent_green())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Open Sessions Manager",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Ctrl + K       ",
                    Style::default()
                        .fg(Theme::accent_purple())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Open Skills & Capabilities Catalog",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Alt + Backspace",
                    Style::default().fg(Theme::text_muted()),
                ),
                Span::styled(
                    "Delete whole word in prompt",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  Esc            ",
                    Style::default()
                        .fg(Theme::accent_red())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Stop the turn, close overlay, or quit — twice if a draft is unsaved",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::raw(""),
            Line::from(vec![Span::styled(
                "  ⚡ Slash Commands (Type /)",
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::raw(""),
            Line::from(vec![
                Span::styled(
                    "  /model         ",
                    Style::default().fg(Theme::accent_cyan()),
                ),
                Span::styled(
                    "Switch active model",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /plan /mcp     ",
                    Style::default().fg(Theme::accent_cyan()),
                ),
                Span::styled(
                    "Plan mode  ·  MCP overlay (r reload)",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /init /export  ",
                    Style::default().fg(Theme::accent_cyan()),
                ),
                Span::styled(
                    "Write AGENTS.md  ·  copy transcript markdown",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /fork /undo    ",
                    Style::default().fg(Theme::accent_cyan()),
                ),
                Span::styled(
                    "Branch this session  ·  undo last exchange",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /editor /focus ",
                    Style::default().fg(Theme::accent_cyan()),
                ),
                Span::styled(
                    "$EDITOR compose  ·  quiet last-turn view",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /goal <task>   ",
                    Style::default().fg(Theme::accent_cyan()),
                ),
                Span::styled(
                    "Set session focus goal",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /tasks         ",
                    Style::default().fg(Theme::accent_cyan()),
                ),
                Span::styled(
                    "View task checklist",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /compress      ",
                    Style::default().fg(Theme::accent_cyan()),
                ),
                Span::styled(
                    "Manual context compaction",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /clear         ",
                    Style::default().fg(Theme::accent_cyan()),
                ),
                Span::styled(
                    "Clear conversation history",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /profiles      ",
                    Style::default().fg(Theme::accent_cyan()),
                ),
                Span::styled(
                    "Bots / profiles — enter resumes last session",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /agents        ",
                    Style::default().fg(Theme::accent_cyan()),
                ),
                Span::styled(
                    "Subagents + processes — x stop  /stop all  s steer",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /memory        ",
                    Style::default().fg(Theme::accent_cyan()),
                ),
                Span::styled(
                    "Learned skills and memories (/journey)",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
            Line::from(vec![
                Span::styled(
                    "  /doctor        ",
                    Style::default().fg(Theme::accent_cyan()),
                ),
                Span::styled(
                    "Run host health diagnostics",
                    Style::default().fg(Theme::text_secondary()),
                ),
            ]),
        ];

        let p = Paragraph::new(help_text)
            .block(
                Block::default()
                    .title(Span::styled(
                        " ❓ Help & Command Reference ",
                        Style::default()
                            .fg(Theme::brand_gold())
                            .add_modifier(Modifier::BOLD),
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Theme::border_focus()))
                    .style(Style::default().bg(Theme::bg_popup())),
            )
            .wrap(Wrap { trim: false });

        frame.render_widget(p, modal_area);
    }
}
