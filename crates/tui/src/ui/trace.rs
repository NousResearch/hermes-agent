use ratatui::{
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use super::stream::tool_icon;
use super::theme::Theme;
use crate::state::{AppState, TaskStatus};

/// Right rail: live agent overview / control — tools live in the feed.
pub struct TracePane;

impl TracePane {
    pub fn render(frame: &mut Frame, area: Rect, state: &AppState, frame_count: u64) {
        let border = if state.trace_focus {
            Theme::brand_gold()
        } else {
            Theme::border_subtle()
        };
        let title = if state.trace_focus {
            " overview  esc composer "
        } else {
            " overview  ctrl+g "
        };
        let block = Block::default()
            .title(Span::styled(
                title,
                Style::default().fg(if state.trace_focus {
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
        if inner.height == 0 {
            return;
        }

        let spin = super::motion::spinner_for(state.indicator, frame_count);
        let now = if state.is_generating {
            if let Some(tool) = &state.metrics.active_tool {
                format!("{spin}  {tool}")
            } else if state.metrics.activity.is_empty() {
                format!("{spin}  waiting")
            } else {
                format!("{spin}  {}", state.metrics.activity)
            }
        } else {
            "idle".into()
        };
        let qn = state.prompt_queue.len();
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
        let done = state
            .tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Completed)
            .count();

        let mut lines = vec![
            Line::from(Span::styled(
                format!("  now   {now}"),
                Style::default().fg(Theme::brand_gold()),
            )),
            Line::from(Span::styled(
                format!("  queue {qn}    {run} run  {pend} next  {done} done"),
                Style::default().fg(Theme::text_secondary()),
            )),
            Line::raw(""),
        ];

        if let Some(goal) = &state.goal {
            lines.push(Line::from(Span::styled(
                format!("  goal  {goal}"),
                Style::default().fg(Theme::text_primary()),
            )));
            lines.push(Line::raw(""));
        }

        if state.tasks.is_empty() {
            lines.push(Line::from(Span::styled(
                "  no tasks yet",
                Style::default().fg(Theme::text_dim()),
            )));
        } else {
            for task in state.tasks.iter().take(8) {
                let (mark, color) = match task.status {
                    TaskStatus::Pending => ("○", Theme::text_muted()),
                    TaskStatus::InProgress => (spin, Theme::accent_yellow()),
                    TaskStatus::Completed => ("✓", Theme::accent_green()),
                    TaskStatus::Failed => ("✗", Theme::accent_red()),
                };
                let title: String = task.title.chars().take(28).collect();
                lines.push(Line::from(vec![
                    Span::styled(format!("  {mark} "), Style::default().fg(color)),
                    Span::styled(title, Style::default().fg(Theme::text_secondary())),
                ]));
            }
        }

        let live_agents: Vec<_> = state
            .agent_rows
            .iter()
            .filter(|r| r.is_live())
            .take(6)
            .collect();
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            format!(
                "  agents  {} live  {}",
                state.running_agent_count(),
                if state.agents_paused {
                    "spawn paused"
                } else {
                    "spawn live"
                }
            ),
            Style::default().fg(Theme::text_muted()),
        )));
        if live_agents.is_empty() {
            lines.push(Line::from(Span::styled(
                "  a  /agents",
                Style::default().fg(Theme::text_dim()),
            )));
        } else {
            for a in live_agents {
                let mark = if a.status == "queued" { "○" } else { spin };
                let title: String = a.title.chars().take(22).collect();
                let tool = if a.last_tool.is_empty() {
                    String::new()
                } else {
                    format!("  {}", crate::tips::ellipsize(&a.last_tool, 14))
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  {mark} "),
                        Style::default().fg(Theme::brand_gold()),
                    ),
                    Span::styled(
                        format!("{title}{tool}"),
                        Style::default().fg(Theme::text_secondary()),
                    ),
                ]));
            }
        }

        let live_procs: Vec<_> = state
            .agent_rows
            .iter()
            .filter(|r| r.is_running_process())
            .take(4)
            .collect();
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            format!("  procs   {} running", state.running_process_count()),
            Style::default().fg(Theme::text_muted()),
        )));
        if live_procs.is_empty() {
            lines.push(Line::from(Span::styled(
                "  /stop kills all  x in /agents kills one",
                Style::default().fg(Theme::text_dim()),
            )));
        } else {
            for p in live_procs {
                let title: String = p.title.chars().take(24).collect();
                let pid = p.pid.map(|n| format!("  pid {n}")).unwrap_or_default();
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  {spin} "),
                        Style::default().fg(Theme::accent_cyan()),
                    ),
                    Span::styled(
                        format!("{title}{pid}"),
                        Style::default().fg(Theme::text_secondary()),
                    ),
                ]));
            }
        }

        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            "  recent",
            Style::default().fg(Theme::text_muted()),
        )));
        let steps = state.tool_steps();
        if steps.is_empty() {
            lines.push(Line::from(Span::styled(
                "  tools show in the feed",
                Style::default().fg(Theme::text_dim()),
            )));
        } else {
            let start = steps.len().saturating_sub(10);
            for st in &steps[start..] {
                let running = st.status.contains("running");
                let icon = tool_icon(&st.name);
                let name: String = st.name.chars().take(16).collect();
                lines.push(Line::from(Span::styled(
                    format!("  {icon} {name}"),
                    Style::default().fg(if running {
                        Theme::brand_gold()
                    } else {
                        Theme::text_muted()
                    }),
                )));
            }
        }

        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .style(Style::default().bg(Theme::bg_base())),
            inner,
        );
    }
}
