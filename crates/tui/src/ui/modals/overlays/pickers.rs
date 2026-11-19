use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use crate::state::{AppState, PickerStage};
use crate::ui::theme::Theme;

impl crate::ui::modals::ViewsOverlay {
    pub(crate) fn render_theme_modal(frame: &mut Frame, area: Rect, state: &mut AppState) {
        let modal_area = Self::centered_rect(68, 78, area);
        frame.render_widget(Clear, modal_area);
        let palettes = crate::ui::theme::catalog();
        let items: Vec<ListItem> = palettes
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let on = p.id == state.theme_id;
                let mark = if on { "●" } else { "○" };
                let sel = i == state.modal_selected;
                let line = Line::from(vec![
                    Span::styled(
                        format!("  {mark} {:<12}  ", p.id),
                        Style::default()
                            .fg(if sel {
                                Theme::brand_gold()
                            } else {
                                Theme::text_primary()
                            })
                            .add_modifier(if sel {
                                Modifier::BOLD
                            } else {
                                Modifier::empty()
                            }),
                    ),
                    Span::styled(p.blurb, Style::default().fg(Theme::text_muted())),
                ]);
                ListItem::new(line).style(if sel {
                    Style::default().bg(Theme::bg_highlight())
                } else {
                    Style::default()
                })
            })
            .collect();
        let block = Block::default()
            .title(Span::styled(
                " theme  ↑↓ preview  enter keep  esc revert ",
                Style::default().fg(Theme::brand_gold()),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Theme::border_focus()))
            .style(Style::default().bg(Theme::bg_popup()));
        let inner = block.inner(modal_area);
        let list = List::new(items).block(block);
        let mut list_state = ListState::default();
        if state.picker_len() > 0 {
            list_state.select(Some(state.modal_selected));
        }
        frame.render_stateful_widget(list, modal_area, &mut list_state);
        state.picker_list = Some(inner);
        state.picker_offset = list_state.offset();
    }

    pub(crate) fn render_model_modal(frame: &mut Frame, area: Rect, state: &mut AppState) {
        if matches!(state.picker_stage, PickerStage::Key) {
            Self::render_provider_key_modal(frame, area, state);
            return;
        }
        let modal_area = Self::centered_rect(62, 58, area);
        frame.render_widget(Clear, modal_area);

        let filter = state.picker_filter.clone();
        let hint = if filter.is_empty() {
            " ↑↓ enter  x disconnect  type to filter  esc "
        } else {
            ""
        };
        let (title, items) = match state.picker_stage {
            PickerStage::Key => (String::new(), Vec::new()),
            PickerStage::Providers => {
                let idxs = state.filtered_provider_indices();
                let items = if state.providers.is_empty() {
                    vec![ListItem::new(Line::from(Span::styled(
                        "  loading providers…",
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
                            let p = &state.providers[pi];
                            let is_sel = vis == state.modal_selected;
                            let mark = if p.is_current { "●" } else { "·" };
                            let n = p.models.len();
                            let count = if n == 0 {
                                if p.authenticated {
                                    "no models".to_string()
                                } else if p.accepts_inline_key() {
                                    "paste key".to_string()
                                } else {
                                    "needs setup".to_string()
                                }
                            } else {
                                format!("{n} models")
                            };
                            let name_style = if is_sel {
                                Style::default()
                                    .fg(Theme::brand_gold())
                                    .add_modifier(Modifier::BOLD)
                            } else if p.authenticated {
                                Style::default().fg(Theme::text_primary())
                            } else {
                                Style::default().fg(Theme::text_muted())
                            };
                            ListItem::new(Line::from(vec![
                                Span::styled(format!("  {mark} {:<22} ", p.name), name_style),
                                Span::styled(count, Style::default().fg(Theme::text_dim())),
                            ]))
                            .style(if is_sel {
                                Style::default().bg(Theme::bg_highlight())
                            } else {
                                Style::default()
                            })
                        })
                        .collect()
                };
                let title = if filter.is_empty() {
                    format!(" provider{hint}")
                } else {
                    format!(" provider  /{filter} ")
                };
                (title, items)
            }
            PickerStage::Models => {
                let name = state
                    .selected_provider()
                    .map(|p| p.name.clone())
                    .unwrap_or_else(|| "models".into());
                let models = state
                    .selected_provider()
                    .map(|p| p.models.clone())
                    .unwrap_or_default();
                let midx = state.filtered_model_indices();
                let active = state.metrics.active_model.clone();
                let sel = state.modal_selected;
                let title = if filter.is_empty() {
                    format!(" {}  ↑↓ enter  ← back  esc ", name.to_ascii_lowercase())
                } else {
                    format!(" {}  /{filter} ", name.to_ascii_lowercase())
                };
                let items = if models.is_empty() {
                    vec![ListItem::new(Line::from(Span::styled(
                        "  no models on this provider",
                        Style::default().fg(Theme::text_muted()),
                    )))]
                } else if midx.is_empty() {
                    vec![ListItem::new(Line::from(Span::styled(
                        "  no matches · backspace clears filter",
                        Style::default().fg(Theme::text_muted()),
                    )))]
                } else {
                    midx.iter()
                        .enumerate()
                        .map(|(vis, &mi)| {
                            let id = &models[mi];
                            let is_sel = vis == sel;
                            let is_active =
                                active == *id || active.ends_with(id) || id.ends_with(&active);
                            let mark = if is_active { "●" } else { "·" };
                            let style = if is_sel {
                                Style::default()
                                    .fg(Theme::brand_gold())
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Theme::text_primary())
                            };
                            ListItem::new(Line::from(Span::styled(format!("  {mark} {id}"), style)))
                                .style(if is_sel {
                                    Style::default().bg(Theme::bg_highlight())
                                } else {
                                    Style::default()
                                })
                        })
                        .collect()
                };
                (title, items)
            }
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
        let list = List::new(items).block(block);
        let mut list_state = ListState::default();
        if state.picker_len() > 0 {
            list_state.select(Some(state.modal_selected));
        }
        frame.render_stateful_widget(list, modal_area, &mut list_state);
        state.picker_list = Some(inner);
        state.picker_offset = list_state.offset();
    }

    pub(crate) fn render_provider_key_modal(frame: &mut Frame, area: Rect, state: &mut AppState) {
        let modal_area = Self::centered_rect(62, 36, area);
        frame.render_widget(Clear, modal_area);
        let provider = state.selected_provider();
        let name = provider
            .map(|p| p.name.as_str())
            .unwrap_or("provider")
            .to_string();
        let key_env = provider
            .map(|p| p.key_env.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("API key")
            .to_string();
        let mask = if state.picker_key.is_empty() {
            String::new()
        } else {
            "•".repeat(state.picker_key.chars().count().min(48))
        };
        let status = if state.picker_key_saving {
            "saving…"
        } else if !state.picker_key_error.is_empty() {
            state.picker_key_error.as_str()
        } else {
            ""
        };
        let status_style = if state.picker_key_error.is_empty() {
            Style::default().fg(Theme::text_dim())
        } else {
            Style::default().fg(Theme::accent_red())
        };
        let body = vec![
            Line::from(Span::styled(
                format!("  Paste {key_env} — saved to ~/.hermes/.env"),
                Style::default().fg(Theme::text_muted()),
            )),
            Line::raw(""),
            Line::from(Span::styled(
                if mask.is_empty() {
                    "  (empty)_".to_string()
                } else {
                    format!("  {mask}_")
                },
                Style::default().fg(Theme::brand_gold()),
            )),
            Line::raw(""),
            Line::from(Span::styled(format!("  {status}"), status_style)),
            Line::raw(""),
            Line::from(Span::styled(
                "  enter save   ctrl+u clear   esc back",
                Style::default().fg(Theme::text_dim()),
            )),
        ];
        let block = Block::default()
            .title(Span::styled(
                format!(" add {name} "),
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Theme::border_focus()))
            .style(Style::default().bg(Theme::bg_popup()));
        let inner = block.inner(modal_area);
        frame.render_widget(Paragraph::new(body).block(block), modal_area);
        state.picker_list = Some(inner);
        state.picker_offset = 0;
    }

    pub(crate) fn render_branch_modal(frame: &mut Frame, area: Rect, state: &mut AppState) {
        let modal_area = Self::centered_rect(52, 50, area);
        frame.render_widget(Clear, modal_area);
        let cwd = state.metrics.cwd.clone();
        let filter = state.picker_filter.clone();
        let idxs = state.filtered_branch_indices();
        let items: Vec<ListItem> = if state.branches.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no git branches · init a repo in this folder",
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
                .map(|(vis, &bi)| {
                    let b = &state.branches[bi];
                    let is_sel = vis == state.modal_selected;
                    let mark = if b.current { "●" } else { "·" };
                    let other_tree = b
                        .worktree
                        .as_deref()
                        .filter(|p| !crate::platform::same_path(p, &cwd));
                    let style = if is_sel {
                        Style::default()
                            .fg(Theme::brand_gold())
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Theme::text_primary())
                    };
                    let mut spans = vec![Span::styled(format!("  {mark} {}", b.name), style)];
                    if other_tree.is_some() {
                        spans.push(Span::styled(
                            "  worktree",
                            Style::default().fg(Theme::text_dim()),
                        ));
                    }
                    ListItem::new(Line::from(spans)).style(if is_sel {
                        Style::default().bg(Theme::bg_highlight())
                    } else {
                        Style::default()
                    })
                })
                .collect()
        };
        let block = Block::default()
            .title(Span::styled(
                if filter.is_empty() {
                    " branch  ↑↓ enter  type to filter  esc ".to_string()
                } else {
                    format!(" branch  /{filter} ")
                },
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Theme::border_focus()))
            .style(Style::default().bg(Theme::bg_popup()));
        let inner = block.inner(modal_area);
        let list = List::new(items).block(block);
        let mut list_state = ListState::default();
        if !state.branches.is_empty() {
            list_state.select(Some(state.modal_selected));
        }
        frame.render_stateful_widget(list, modal_area, &mut list_state);
        state.picker_list = Some(inner);
        state.picker_offset = list_state.offset();
    }

    pub(crate) fn render_skills_modal(frame: &mut Frame, area: Rect, state: &AppState) {
        let modal_area = Self::centered_rect(86, 72, area);
        frame.render_widget(Clear, modal_area);
        let outer = Block::default()
            .title(Span::styled(
                if state.picker_filter.is_empty() {
                    " skills  ↑↓ enter inspect  i install  type to filter  esc ".to_string()
                } else {
                    format!(" skills  /{} ", state.picker_filter)
                },
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Theme::border_focus()))
            .style(Style::default().bg(Theme::bg_popup()));
        let inner = outer.inner(modal_area);
        frame.render_widget(outer, modal_area);

        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(38), Constraint::Percentage(62)])
            .split(inner);

        let idxs = state.filtered_skill_indices();
        let items: Vec<ListItem> = if state.skills.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no skills loaded · add SKILL.md under ~/.hermes/skills",
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
                .map(|(vis, &si)| {
                    let skill = &state.skills[si];
                    let is_sel = vis == state.modal_selected;
                    let name = crate::tips::ellipsize(&skill.name, 22);
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!(" 📦 {name} "),
                            if is_sel {
                                Style::default()
                                    .fg(Theme::brand_gold())
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Theme::accent_cyan())
                            },
                        ),
                        Span::styled(
                            format!("[{}]", skill.category),
                            Style::default().fg(Theme::brand_orange()),
                        ),
                    ]))
                    .style(if is_sel {
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
        frame.render_stateful_widget(list, cols[0], &mut list_state);

        let selected = idxs
            .get(state.modal_selected)
            .and_then(|i| state.skills.get(*i));
        let mut detail: Vec<Line> = Vec::new();
        if let Some(skill) = selected {
            detail.push(Line::from(Span::styled(
                format!("  {}", skill.name),
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
            )));
            if !skill.category.is_empty() {
                detail.push(Line::from(Span::styled(
                    format!("  [{}]", skill.category),
                    Style::default().fg(Theme::brand_orange()),
                )));
            }
            detail.push(Line::raw(""));
            if skill.description.is_empty() {
                detail.push(Line::from(Span::styled(
                    "  No description in SKILL.md.",
                    Style::default().fg(Theme::text_muted()),
                )));
            } else {
                for line in
                    Self::wrap_detail(&skill.description, cols[1].width.saturating_sub(4) as usize)
                {
                    detail.push(Line::from(Span::styled(
                        format!("  {line}"),
                        Style::default().fg(Theme::text_primary()),
                    )));
                }
            }
            if !skill.preview.is_empty() {
                detail.push(Line::raw(""));
                detail.push(Line::from(Span::styled(
                    "  preview",
                    Style::default()
                        .fg(Theme::text_muted())
                        .add_modifier(Modifier::BOLD),
                )));
                for line in skill.preview.lines().take(18) {
                    detail.push(Line::from(Span::styled(
                        format!("  {line}"),
                        Style::default().fg(Theme::text_secondary()),
                    )));
                }
            }
        } else {
            detail.push(Line::from(Span::styled(
                "  Select a skill to read its description.",
                Style::default().fg(Theme::text_muted()),
            )));
        }
        let pane = Paragraph::new(detail).wrap(Wrap { trim: false }).block(
            Block::default()
                .borders(Borders::LEFT)
                .border_style(Style::default().fg(Theme::border_subtle())),
        );
        frame.render_widget(pane, cols[1]);
    }

    pub(crate) fn render_tools_modal(frame: &mut Frame, area: Rect, state: &mut AppState) {
        let modal_area = Self::centered_rect(78, 72, area);
        frame.render_widget(Clear, modal_area);
        let title = if state.picker_filter.is_empty() {
            " tools  ↑↓  enter peek  space toggle  type to filter  esc ".to_string()
        } else {
            format!(" tools  /{} ", state.picker_filter)
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
        let idxs = state.filtered_toolset_indices();
        let items: Vec<ListItem> = if state.toolsets.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no toolsets loaded",
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
                    let t = &state.toolsets[si];
                    let mark = if t.enabled { "●" } else { "○" };
                    let sel = vis == state.modal_selected;
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!(" {mark} {:<16} ", t.name),
                            if sel {
                                Style::default()
                                    .fg(Theme::brand_gold())
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Theme::text_primary())
                            },
                        ),
                        Span::styled(
                            format!("{} tools  ", t.tool_count),
                            Style::default().fg(Theme::text_muted()),
                        ),
                        Span::styled(
                            crate::tips::ellipsize(&t.description, 40),
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
        let list = List::new(items);
        let mut list_state = ListState::default();
        if !idxs.is_empty() {
            list_state.select(Some(state.modal_selected));
        }
        frame.render_stateful_widget(list, inner, &mut list_state);
        state.picker_list = Some(inner);
        state.picker_offset = list_state.offset();
    }

    pub(crate) fn render_plugins_modal(frame: &mut Frame, area: Rect, state: &mut AppState) {
        let modal_area = Self::centered_rect(72, 64, area);
        frame.render_widget(Clear, modal_area);
        let title = if state.picker_filter.is_empty() {
            " plugins  ↑↓  space toggle  type to filter  esc ".to_string()
        } else {
            format!(" plugins  /{} ", state.picker_filter)
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
        let idxs = state.filtered_plugin_indices();
        let items: Vec<ListItem> = if state.plugins.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no plugins loaded",
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
                    let p = &state.plugins[si];
                    let mark = if p.enabled { "●" } else { "○" };
                    let sel = vis == state.modal_selected;
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!(" {mark} {}  ", p.name),
                            if sel {
                                Style::default()
                                    .fg(Theme::brand_gold())
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Theme::text_primary())
                            },
                        ),
                        Span::styled(
                            format!("v{}  ", p.version),
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

    pub(crate) fn render_palette_modal(frame: &mut Frame, area: Rect, state: &mut AppState) {
        let modal_area = Self::centered_rect(72, 70, area);
        frame.render_widget(Clear, modal_area);
        let title = if state.picker_filter.is_empty() {
            " palette  type to filter  enter run  esc ".to_string()
        } else {
            format!(" palette  /{} ", state.picker_filter)
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
        let entries = state.filtered_palette_entries();
        let items: Vec<ListItem> = if entries.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no matches",
                Style::default().fg(Theme::text_muted()),
            )))]
        } else {
            entries
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let sel = i == state.modal_selected;
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!(" {:<16} ", e.name),
                            if sel {
                                Style::default()
                                    .fg(Theme::brand_gold())
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Theme::accent_cyan())
                            },
                        ),
                        Span::styled(
                            crate::tips::ellipsize(&e.description, 48),
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
        let list = List::new(items);
        let mut list_state = ListState::default();
        if !entries.is_empty() {
            list_state.select(Some(state.modal_selected));
        }
        frame.render_stateful_widget(list, inner, &mut list_state);
        state.picker_list = Some(inner);
        state.picker_offset = list_state.offset();
    }
}
