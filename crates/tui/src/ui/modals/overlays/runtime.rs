use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use crate::state::{AppState, TaskStatus};
use crate::ui::theme::Theme;

impl crate::ui::modals::ViewsOverlay {
    pub(crate) fn render_tasks_modal(
        frame: &mut Frame,
        area: Rect,
        state: &AppState,
        frame_count: u64,
    ) {
        let modal_area = Self::centered_rect(78, 78, area);
        frame.render_widget(Clear, modal_area);

        let active_spinner = crate::ui::motion::spinner(frame_count);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(5),
                Constraint::Min(6),
                Constraint::Length(7),
                Constraint::Length(2),
            ])
            .split(modal_area);

        let goal_text = state.goal.as_deref().unwrap_or("no goal — /goal <task>");
        let goal_p = Paragraph::new(Line::from(vec![
            Span::styled(
                "  goal  ",
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(goal_text, Style::default().fg(Theme::text_primary())),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Theme::border_focus()))
                .style(Style::default().bg(Theme::bg_surface())),
        );
        frame.render_widget(goal_p, chunks[0]);

        let now = if state.is_generating {
            if let Some(tool) = &state.metrics.active_tool {
                format!("{active_spinner} running  {tool}")
            } else {
                format!("{active_spinner} {}", state.metrics.activity)
            }
        } else {
            "idle".into()
        };
        let qn = state.prompt_queue.len();
        let done = state
            .tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Completed)
            .count();
        let run = state
            .tasks
            .iter()
            .filter(|t| t.status == TaskStatus::InProgress)
            .count();
        let pend = state
            .tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Pending)
            .count();
        let now_p = Paragraph::new(vec![
            Line::from(Span::styled(
                format!("  now     {now}"),
                Style::default().fg(Theme::brand_gold()),
            )),
            Line::from(Span::styled(
                format!(
                    "  next    {qn} queued    tasks  {run} running  {pend} next  {done} done    agents  {} live",
                    state.running_agent_count()
                ),
                Style::default().fg(Theme::text_secondary()),
            )),
            Line::from(Span::styled(
                format!(
                    "  queue   {}",
                    state
                        .prompt_queue
                        .iter()
                        .take(2)
                        .map(|s| s.lines().next().unwrap_or("").trim())
                        .collect::<Vec<_>>()
                        .join("  ·  ")
                ),
                Style::default().fg(Theme::text_muted()),
            )),
        ])
        .block(
            Block::default()
                .title(Span::styled(
                    " overview ",
                    Style::default().fg(Theme::brand_gold()),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Theme::border_subtle()))
                .style(Style::default().bg(Theme::bg_popup())),
        );
        frame.render_widget(now_p, chunks[1]);

        let items: Vec<ListItem> = if state.tasks.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no checklist yet — the agent writes these as it works",
                Style::default().fg(Theme::text_muted()),
            )))]
        } else {
            state
                .tasks
                .iter()
                .enumerate()
                .map(|(idx, task)| {
                    let is_sel = idx == state.modal_selected;
                    let (icon, color) = match task.status {
                        TaskStatus::Pending => ("○", Theme::text_muted()),
                        TaskStatus::InProgress => (active_spinner, Theme::accent_yellow()),
                        TaskStatus::Completed => ("✓", Theme::accent_green()),
                        TaskStatus::Failed => ("✗", Theme::accent_red()),
                    };
                    let line = Line::from(vec![
                        Span::styled(
                            format!("  {icon} "),
                            Style::default().fg(color).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            task.title.clone(),
                            if is_sel {
                                Style::default()
                                    .fg(Theme::text_primary())
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Theme::text_secondary())
                            },
                        ),
                    ]);
                    ListItem::new(line).style(if is_sel {
                        Style::default().bg(Theme::bg_highlight())
                    } else {
                        Style::default()
                    })
                })
                .collect()
        };
        let tasks_list = List::new(items).block(
            Block::default()
                .title(Span::styled(
                    " tasks  space cycles status ",
                    Style::default().fg(Theme::brand_gold()),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Theme::border_focus()))
                .style(Style::default().bg(Theme::bg_popup())),
        );
        frame.render_widget(tasks_list, chunks[2]);

        let recent: Vec<Line> = {
            let steps = state.tool_steps();
            let start = steps.len().saturating_sub(5);
            if steps.is_empty() {
                vec![Line::from(Span::styled(
                    "  no tool calls yet",
                    Style::default().fg(Theme::text_dim()),
                ))]
            } else {
                steps[start..]
                    .iter()
                    .map(|st| {
                        let running = st.status.contains("running");
                        let mark = if running { active_spinner } else { "·" };
                        Line::from(Span::styled(
                            format!("  {mark} {}  {}", st.name, st.status),
                            Style::default().fg(if running {
                                Theme::brand_gold()
                            } else {
                                Theme::text_muted()
                            }),
                        ))
                    })
                    .collect()
            }
        };
        frame.render_widget(
            Paragraph::new(recent).block(
                Block::default()
                    .title(Span::styled(
                        " recent tools ",
                        Style::default().fg(Theme::text_muted()),
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Theme::border_subtle()))
                    .style(Style::default().bg(Theme::bg_popup())),
            ),
            chunks[3],
        );

        let footer = Paragraph::new(Line::from(Span::styled(
            " esc close   ↑↓ select   space status   ctrl+d diff   ctrl+p theme ",
            Style::default().fg(Theme::text_muted()),
        )))
        .style(Style::default().bg(Theme::bg_header()));
        frame.render_widget(footer, chunks[4]);
    }

    pub(crate) fn render_agents_modal(
        frame: &mut Frame,
        area: Rect,
        state: &mut AppState,
        frame_count: u64,
    ) {
        let modal_area = Self::centered_rect(80, 74, area);
        frame.render_widget(Clear, modal_area);
        let split = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
            .split(modal_area);

        let paused = if state.agents_paused {
            "spawn PAUSED"
        } else {
            "spawn live"
        };
        let spin = crate::ui::motion::spinner_for(state.indicator, frame_count);
        let idxs = if state.agents_steer {
            (0..state.agent_rows.len()).collect::<Vec<_>>()
        } else {
            state.filtered_agent_indices()
        };

        let items: Vec<ListItem> = if state.agent_rows.is_empty() {
            let empty = if state.agents_replay {
                "  snapshot empty"
            } else {
                "  no subagents yet · ask the agent to spawn one"
            };
            vec![
                ListItem::new(Line::from(Span::styled(
                    empty,
                    Style::default().fg(Theme::text_muted()),
                ))),
                ListItem::new(Line::from(Span::styled(
                    format!("  {paused}  ·  {}", state.agents_caps),
                    Style::default().fg(Theme::text_secondary()),
                ))),
            ]
        } else if idxs.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no matches · backspace clears filter",
                Style::default().fg(Theme::text_muted()),
            )))]
        } else {
            idxs.iter()
                .enumerate()
                .map(|(vis, &ai)| {
                    let a = &state.agent_rows[ai];
                    let sel = vis == state.modal_selected;
                    let (glyph, gcolor) = agent_glyph(a, spin);
                    let indent = "  ".repeat(a.depth.min(4) as usize);
                    let tools = if a.tool_count > 0 {
                        format!(" ·{}t", a.tool_count)
                    } else {
                        String::new()
                    };
                    let trail = if !a.last_tool.is_empty() && a.is_live() {
                        format!("  {}", crate::tips::ellipsize(&a.last_tool, 16))
                    } else {
                        String::new()
                    };
                    let kind_mark = if a.kind == "process" { "proc " } else { "" };
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!("  {indent}{kind_mark}{glyph} "),
                            Style::default().fg(gcolor),
                        ),
                        Span::styled(
                            crate::tips::ellipsize(&a.title, 28),
                            if sel {
                                Style::default()
                                    .fg(Theme::brand_gold())
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Theme::text_primary())
                            },
                        ),
                        Span::styled(
                            format!("{tools}{trail}"),
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

        let title = if state.agents_steer {
            format!(
                " agents  steer → {}  enter send  esc cancel ",
                crate::tips::ellipsize(&state.picker_filter, 24)
            )
        } else if state.agents_replay && state.picker_filter.is_empty() {
            format!(
                " replay snapshot  {} agents  enter peek  esc back  (live x/p/s off) ",
                state.agent_rows.len()
            )
        } else if state.picker_filter.is_empty() {
            format!(
                " agents  {paused}  {} live  {} procs  x stop  /stop all  s steer  esc ",
                state.running_agent_count(),
                state.running_process_count()
            )
        } else {
            format!(" agents  /{} ", state.picker_filter)
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

        if split[1].width > 4 {
            let selected = idxs
                .get(state.modal_selected)
                .and_then(|i| state.agent_rows.get(*i));
            let (detail_title, detail_body) = match selected {
                Some(a) => (
                    format!(" {}  {} ", a.id, a.status),
                    agent_detail(a),
                ),
                None => (
                    " agent ".into(),
                    format!("{paused}\n{}\n\np pause spawn  r resume  x stop selected  X subtree  s steer  enter peek", state.agents_caps),
                ),
            };
            let detail_block = Block::default()
                .title(Span::styled(
                    crate::tips::ellipsize(
                        &detail_title,
                        split[1].width.saturating_sub(2) as usize,
                    ),
                    Style::default().fg(Theme::text_muted()),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Theme::border_subtle()))
                .style(Style::default().bg(Theme::bg_popup()));
            let inner = detail_block.inner(split[1]);
            frame.render_widget(detail_block, split[1]);
            let width = inner.width.saturating_sub(1) as usize;
            let mut lines: Vec<Line> = Vec::new();
            for raw in Self::wrap_detail(&detail_body, width)
                .into_iter()
                .take(inner.height as usize)
            {
                lines.push(Line::from(Span::styled(
                    raw,
                    Style::default().fg(Theme::text_secondary()),
                )));
            }
            frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
        }
    }

    pub(crate) fn render_background_modal(
        frame: &mut Frame,
        area: Rect,
        state: &mut AppState,
        frame_count: u64,
    ) {
        let modal_area = Self::centered_rect(78, 70, area);
        frame.render_widget(Clear, modal_area);
        let idxs = state.filtered_bg_indices();
        let split = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
            .split(modal_area);

        let spinner = crate::ui::motion::spinner(frame_count);
        let mut items: Vec<ListItem> = Vec::new();
        let launch_sel = state.modal_selected == 0;
        let launch_prompt = state.picker_filter.trim();
        let launch_label = if launch_prompt.is_empty() {
            "  +  type a prompt · enter launches".to_string()
        } else {
            format!("  +  launch  {}", crate::tips::ellipsize(launch_prompt, 42))
        };
        items.push(
            ListItem::new(Line::from(Span::styled(
                launch_label,
                if launch_sel {
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Theme::accent_cyan())
                },
            )))
            .style(if launch_sel {
                Style::default().bg(Theme::bg_highlight())
            } else {
                Style::default()
            }),
        );

        if idxs.is_empty() {
            items.push(ListItem::new(Line::from(Span::styled(
                "  no tasks yet · /background <prompt>",
                Style::default().fg(Theme::text_muted()),
            ))));
        } else {
            for (vis, &bi) in idxs.iter().enumerate() {
                let t = &state.bg_tasks[bi];
                let sel = state.modal_selected == vis + 1;
                let (mark, mark_color) = match t.status {
                    crate::state::BgStatus::Running => (spinner, Theme::brand_gold()),
                    crate::state::BgStatus::Done => ("✓", Theme::accent_green()),
                };
                let age = crate::ui::turn_bar::fmt_duration(t.started.elapsed().as_secs_f64());
                let preview = if t.status == crate::state::BgStatus::Done && !t.result.is_empty() {
                    t.result.as_str()
                } else {
                    t.prompt.as_str()
                };
                items.push(
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!("  {mark} {:<10} ", t.id),
                            if sel {
                                Style::default()
                                    .fg(Theme::brand_gold())
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(mark_color)
                            },
                        ),
                        Span::styled(
                            format!("{}  ", crate::tips::ellipsize(preview, 36)),
                            Style::default().fg(Theme::text_secondary()),
                        ),
                        Span::styled(age, Style::default().fg(Theme::text_muted())),
                    ]))
                    .style(if sel {
                        Style::default().bg(Theme::bg_highlight())
                    } else {
                        Style::default()
                    }),
                );
            }
        }

        let title = if state.picker_filter.is_empty() {
            format!(
                " background  ↑↓ peek  enter launch  {} running  esc ",
                state.running_bg_count()
            )
        } else {
            format!(" background  /{} ", state.picker_filter)
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

        if split[1].width > 4 {
            let (detail_title, detail_body) = if state.modal_selected == 0 {
                (
                    " new task ".to_string(),
                    if launch_prompt.is_empty() {
                        "Type a prompt. Enter runs it in another session — this one stays free."
                            .to_string()
                    } else {
                        launch_prompt.to_string()
                    },
                )
            } else if let Some(&bi) = idxs.get(state.modal_selected.saturating_sub(1)) {
                let t = &state.bg_tasks[bi];
                let title = match t.status {
                    crate::state::BgStatus::Running => format!(" {}  running ", t.id),
                    crate::state::BgStatus::Done => format!(" {}  done ", t.id),
                };
                let body = if t.result.is_empty() {
                    t.prompt.clone()
                } else if t.prompt.is_empty() {
                    t.result.clone()
                } else {
                    format!("{}\n\n{}", t.prompt, t.result)
                };
                (title, body)
            } else {
                (" task ".into(), String::new())
            };
            let detail_block = Block::default()
                .title(Span::styled(
                    detail_title,
                    Style::default().fg(Theme::text_muted()),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Theme::border_subtle()))
                .style(Style::default().bg(Theme::bg_popup()));
            let inner = detail_block.inner(split[1]);
            frame.render_widget(detail_block, split[1]);
            let width = inner.width.saturating_sub(1) as usize;
            let mut lines: Vec<Line> = Vec::new();
            for raw in Self::wrap_detail(&detail_body, width)
                .into_iter()
                .take(inner.height as usize)
            {
                lines.push(Line::from(Span::styled(
                    raw,
                    Style::default().fg(Theme::text_secondary()),
                )));
            }
            frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
        }
    }

    pub(crate) fn render_mcp_modal(frame: &mut Frame, area: Rect, state: &mut AppState) {
        let modal_area = Self::centered_rect(78, 72, area);
        frame.render_widget(Clear, modal_area);
        let title = if state.picker_filter.is_empty() {
            if state.mcp_key_name.is_some() {
                format!(
                    " mcp key {}  paste · enter save  esc ",
                    state.mcp_key_name.as_deref().unwrap_or("")
                )
            } else {
                " mcp  ↑↓  a add  t test  k key  o oauth  x remove  r reload  esc ".to_string()
            }
        } else {
            format!(" mcp  /{} ", state.picker_filter)
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
        let idxs = state.filtered_mcp_indices();
        let items: Vec<ListItem> = if state.mcp_servers.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no MCP servers · hermes mcp add, then r reload",
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
                    let m = &state.mcp_servers[si];
                    let mark = if m.connected {
                        "◆"
                    } else if m.enabled {
                        "●"
                    } else if m.installed {
                        "○"
                    } else {
                        "·"
                    };
                    let kind = if m.configured { "cfg" } else { "cat" };
                    let sel = vis == state.modal_selected;
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!(" {mark} {}  ", m.name),
                            if sel {
                                Style::default()
                                    .fg(Theme::brand_gold())
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Theme::text_primary())
                            },
                        ),
                        Span::styled(
                            format!("{kind}  {}  ", m.transport),
                            Style::default().fg(Theme::text_muted()),
                        ),
                        Span::styled(
                            crate::tips::ellipsize(&m.description, 36),
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

    pub(crate) fn render_cron_modal(frame: &mut Frame, area: Rect, state: &mut AppState) {
        let modal_area = Self::centered_rect(82, 72, area);
        frame.render_widget(Clear, modal_area);
        let title = if state.picker_filter.is_empty() {
            " cron  enter peek  p pause  r resume  x remove  esc ".to_string()
        } else {
            format!(" cron  /{} ", state.picker_filter)
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
        let idxs = state.filtered_cron_indices();
        let items: Vec<ListItem> = if state.cron_jobs.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "  no cron jobs",
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
                    let j = &state.cron_jobs[si];
                    let mark = if j.enabled { "●" } else { "○" };
                    let sel = vis == state.modal_selected;
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!(" {mark} {}  ", j.name),
                            if sel {
                                Style::default()
                                    .fg(Theme::brand_gold())
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Theme::text_primary())
                            },
                        ),
                        Span::styled(
                            format!("{}  {}  ", j.schedule, j.state),
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
}

fn agent_glyph(
    a: &crate::state::AgentRow,
    spin: &'static str,
) -> (&'static str, ratatui::style::Color) {
    if a.kind == "process" {
        return if a.status == "running" {
            (spin, Theme::accent_cyan())
        } else {
            ("▸", Theme::text_muted())
        };
    }
    match a.status.as_str() {
        "running" => (spin, Theme::brand_gold()),
        "queued" => ("○", Theme::text_muted()),
        "completed" => ("✓", Theme::accent_green()),
        "failed" | "error" => ("✗", Theme::accent_red()),
        "interrupted" => ("■", Theme::accent_yellow()),
        "timeout" => ("⌛", Theme::accent_yellow()),
        _ => ("●", Theme::text_secondary()),
    }
}

fn agent_detail(a: &crate::state::AgentRow) -> String {
    let mut lines = Vec::new();
    lines.push(a.title.clone());
    lines.push(String::new());
    lines.push(format!("kind    {}", a.kind));
    lines.push(format!("status  {}", a.status));
    if a.depth > 0 || a.is_subagent() {
        lines.push(format!("depth   {}", a.depth));
    }
    if let Some(p) = &a.parent_id {
        lines.push(format!("parent  {p}"));
    }
    if !a.model.is_empty() {
        lines.push(format!("model   {}", a.model));
    }
    if a.tool_count > 0 {
        lines.push(format!("tools   {}", a.tool_count));
    }
    if !a.last_tool.is_empty() {
        lines.push(format!("now     {}", a.last_tool));
    }
    let age = a
        .duration_secs
        .or_else(|| a.started.map(|t| t.elapsed().as_secs_f64()));
    if let Some(secs) = age {
        if secs > 0.05 {
            lines.push(format!(
                "time    {}",
                crate::ui::turn_bar::fmt_duration(secs)
            ));
        }
    }
    if !a.thinking.is_empty() {
        lines.push(String::new());
        lines.push("thinking".into());
        for t in a.thinking.iter().rev().take(4).rev() {
            lines.push(format!("  {t}"));
        }
    }
    if !a.notes.is_empty() {
        lines.push(String::new());
        lines.push("progress".into());
        for n in a.notes.iter().rev().take(6).rev() {
            lines.push(format!("  · {n}"));
        }
    }
    if !a.summary.is_empty() {
        lines.push(String::new());
        lines.push("summary".into());
        lines.push(a.summary.clone());
    }
    if a.is_process() {
        if let Some(pid) = a.pid {
            lines.push(format!("pid     {pid}"));
        }
        if !a.cwd.is_empty() {
            lines.push(format!("cwd     {}", a.cwd));
        }
        if !a.output.is_empty() {
            lines.push(String::new());
            lines.push("output".into());
            let tail: Vec<&str> = a.output.lines().rev().take(18).collect();
            for line in tail.into_iter().rev() {
                lines.push(line.to_string());
            }
        }
        lines.push(String::new());
        lines.push("x kill this process  /stop kills all".into());
    }
    lines.join("\n")
}
