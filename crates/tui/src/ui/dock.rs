use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use unicode_width::UnicodeWidthStr;

use super::theme::Theme;
use crate::state::{
    AgentRow, AppState, BgStatus, BgTask, DockEntry, HitRange, HoverKind, TaskStatus,
};

/// Session task rows under the composer — live subagents, processes, /background.
pub struct AgentDock;

pub const MAX_ROWS: usize = 4;
const STOP: &str = " [stop]";

pub fn dock_entries(state: &AppState) -> Vec<DockEntry> {
    let mut live = Vec::new();
    for (i, row) in state.agent_rows.iter().enumerate() {
        if row.is_live() || row.is_running_process() {
            live.push(DockEntry::Agent(i));
        }
    }
    for (i, task) in state.bg_tasks.iter().enumerate() {
        if task.status == BgStatus::Running {
            live.push(DockEntry::Bg(i));
        }
    }
    live.truncate(if state.compact { 2 } else { MAX_ROWS });
    live
}

fn task_rows(state: &AppState) -> usize {
    if state.tasks.is_empty() {
        0
    } else {
        1 + state.tasks.len().min(if state.compact { 3 } else { 6 })
    }
}

fn dock_sep(_state: &AppState, n: usize) -> u16 {
    u16::from(n > 0)
}

pub fn dock_height(state: &AppState) -> u16 {
    let n = dock_entries(state).len();
    let tasks = task_rows(state);
    if n == 0 && tasks == 0 {
        0
    } else {
        dock_sep(state, n) + n as u16 + tasks as u16
    }
}

impl AgentDock {
    pub fn render(frame: &mut Frame, area: Rect, state: &mut AppState, frame_count: u64) {
        state.hit_dock.clear();
        state.hit_dock_stop.clear();
        state.hit_dock_bar = None;
        if area.height == 0 {
            return;
        }
        let entries = dock_entries(state);
        if entries.is_empty() && state.tasks.is_empty() {
            return;
        }
        let width = area.width as usize;
        let spin = super::motion::spinner_for(state.indicator, frame_count);
        let mut lines: Vec<Line> = Vec::new();
        let sep = dock_sep(state, entries.len());
        let (sep_area, body) = if sep > 0 && area.height > sep {
            let split = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(sep), Constraint::Min(0)])
                .split(area);
            (Some(split[0]), split[1])
        } else {
            (None, area)
        };
        if let Some(head) = sep_area {
            let mut head_lines = Vec::new();
            if sep >= 2 {
                head_lines.push(Line::from(Span::raw(" ".repeat(width))));
            }
            if sep >= 1 {
                head_lines.push(Line::from(Span::styled(
                    "─".repeat(width),
                    Style::default().fg(Theme::border_subtle()),
                )));
            }
            frame.render_widget(
                Paragraph::new(head_lines).style(Style::default().bg(Theme::bg_base())),
                head,
            );
        }

        if !entries.is_empty() {
            for (vis, entry) in entries.iter().enumerate() {
                let y = body.y.saturating_add(vis as u16);
                if y >= body.y.saturating_add(body.height) {
                    break;
                }
                let hot = state.hover == HoverKind::Dock(*entry);
                let stop_hot = state.hover == HoverKind::DockStop(*entry);
                let (spans, stop_w) = match *entry {
                    DockEntry::Agent(i) => state
                        .agent_rows
                        .get(i)
                        .map(|row| dock_spans(row, spin, width, hot, stop_hot))
                        .unwrap_or_else(|| (Vec::new(), 0)),
                    DockEntry::Bg(i) => state
                        .bg_tasks
                        .get(i)
                        .map(|t| bg_spans(t, spin, width, hot, stop_hot))
                        .unwrap_or_else(|| (Vec::new(), 0)),
                };
                lines.push(Line::from(spans));
                let row_x1 = area.x.saturating_add(area.width);
                if stop_w > 0 {
                    state.hit_dock_stop.push((
                        HitRange {
                            y,
                            x0: row_x1.saturating_sub(stop_w as u16),
                            x1: row_x1,
                        },
                        *entry,
                    ));
                }
                state.hit_dock.push((
                    HitRange {
                        y,
                        x0: area.x,
                        x1: if stop_w > 0 {
                            row_x1.saturating_sub(stop_w as u16)
                        } else {
                            row_x1
                        },
                    },
                    *entry,
                ));
            }
        }
        let used = lines.len();
        lines.extend(plan_lines(state, spin, width, body.height as usize, used));
        frame.render_widget(
            Paragraph::new(lines).style(Style::default().bg(Theme::bg_header())),
            body,
        );
    }
}

fn dock_title(row: &AgentRow) -> String {
    let title = row.title.trim();
    let id_like = title.is_empty()
        || title == row.id
        || title.starts_with("proc_")
        || (title.len() >= 10
            && title
                .chars()
                .all(|c| c.is_ascii_hexdigit() || c == '-' || c == '_'));
    if !id_like {
        return title.to_string();
    }
    if let Some(line) = row
        .output
        .lines()
        .rev()
        .map(crate::ui::stream::strip_ansi)
        .find(|l| !l.trim().is_empty())
    {
        return crate::tips::ellipsize(line.trim(), 48);
    }
    if !row.last_tool.is_empty() {
        return row.last_tool.clone();
    }
    if row.is_process() {
        "process".into()
    } else {
        crate::tips::ellipsize(&row.id, 12)
    }
}

fn can_stop(row: &AgentRow) -> bool {
    row.is_running_process() || row.is_live()
}

fn dock_spans(
    row: &AgentRow,
    spin: &'static str,
    width: usize,
    hot: bool,
    stop_hot: bool,
) -> (Vec<Span<'static>>, usize) {
    let (glyph, gcolor) = glyph(row, spin);
    let age = row
        .duration_secs
        .or_else(|| row.started.map(|t| t.elapsed().as_secs_f64()))
        .filter(|s| *s > 0.04)
        .map(crate::ui::turn_bar::fmt_duration)
        .unwrap_or_default();
    let mut bits: Vec<(String, Style)> = Vec::new();
    bits.push((
        format!(" {glyph} "),
        Style::default().fg(gcolor).add_modifier(Modifier::BOLD),
    ));
    bits.push((
        format!("{}  ", if row.is_process() { "proc" } else { "agent" }),
        Style::default().fg(if row.is_process() {
            Theme::accent_cyan()
        } else {
            Theme::text_dim()
        }),
    ));
    let title_style = if hot {
        Style::default()
            .fg(Theme::text_primary())
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Theme::text_primary())
    };
    bits.push((String::new(), title_style));
    let mut trail: Vec<(String, Style)> = Vec::new();
    if !row.model.is_empty() && !row.is_process() {
        trail.push((
            format!("  {}  ", crate::tips::ellipsize(&row.model, 16)),
            Style::default().fg(Theme::text_dim()),
        ));
    }
    trail.push((
        format!("{}  ", display_status(&row.status)),
        Style::default().fg(status_color(&row.status)),
    ));
    if !age.is_empty() {
        trail.push((
            format!("· {age}  "),
            Style::default().fg(Theme::text_muted()),
        ));
    }
    let tokens = row.tokens();
    if tokens > 0 {
        trail.push((
            format!("· {} tok  ", crate::ui::context::fmt_k(tokens)),
            Style::default().fg(Theme::text_muted()),
        ));
        if let Some(secs) = row.duration_secs.filter(|s| *s >= 1.0) {
            let rate = tokens as f64 / secs;
            if rate >= 1.0 {
                trail.push((
                    format!("· {:.0}/s  ", rate),
                    Style::default().fg(Theme::text_dim()),
                ));
            }
        }
    }
    let turn = if row.iteration > 0 {
        row.iteration
    } else {
        row.index
    };
    if turn > 0 && row.is_subagent() {
        trail.push((
            format!("· turn {turn}  "),
            Style::default().fg(Theme::text_dim()),
        ));
    }
    if row.cost_usd >= 0.005 {
        trail.push((
            format!("· ~${:.2}  ", row.cost_usd),
            Style::default().fg(Theme::text_muted()),
        ));
    }
    if row.is_process() {
        if let Some(pid) = row.pid {
            trail.push((
                format!("· pid {pid}  "),
                Style::default().fg(Theme::text_muted()),
            ));
        }
    } else if row.tool_count > 0 && tokens == 0 {
        trail.push((
            format!("· {}t  ", row.tool_count),
            Style::default().fg(Theme::text_muted()),
        ));
    }

    let stop_w = if can_stop(row) { STOP.width() } else { 0 };
    let used: usize = bits
        .iter()
        .map(|(t, _)| t.width())
        .chain(trail.iter().map(|(t, _)| t.width()))
        .sum();
    let title_budget = width.saturating_sub(used + stop_w).max(8);
    bits[2].0 = format!(
        "{:<w$}",
        crate::tips::ellipsize(&dock_title(row), title_budget),
        w = title_budget.min(48)
    );

    let mut spans: Vec<Span<'static>> = bits
        .into_iter()
        .map(|(t, st)| Span::styled(t, st))
        .collect();
    spans.extend(trail.into_iter().map(|(t, st)| Span::styled(t, st)));
    let w: usize = spans.iter().map(|s| s.content.width()).sum();
    if w + stop_w < width {
        spans.push(Span::raw(" ".repeat(width - w - stop_w)));
    }
    if stop_w > 0 {
        spans.push(Span::styled(
            STOP,
            crate::ui::theme::hover_paint(
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
                stop_hot,
            ),
        ));
    }
    if hot && !stop_hot {
        for sp in &mut spans {
            if sp.content.as_ref() != STOP {
                sp.style.bg = Some(Theme::bg_highlight());
            }
        }
    }
    (spans, stop_w)
}

fn plan_lines(
    state: &AppState,
    spin: &'static str,
    width: usize,
    height: usize,
    used: usize,
) -> Vec<Line<'static>> {
    if state.tasks.is_empty() {
        return Vec::new();
    }
    let room = height.saturating_sub(used);
    if room == 0 {
        return Vec::new();
    }
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
    let n = state.tasks.len();
    let head = format!(" ☑  plan  {run} running  {done}/{n} done");
    let mut line = vec![Span::styled(
        head.clone(),
        Style::default()
            .fg(Theme::brand_gold())
            .add_modifier(Modifier::BOLD),
    )];
    let w = head.width();
    if w < width {
        line.push(Span::raw(" ".repeat(width - w)));
    }
    let mut lines = vec![Line::from(line)];
    let take = (room.saturating_sub(1))
        .min(if state.compact { 3 } else { 6 })
        .min(n);
    for task in state.tasks.iter().take(take) {
        let (mark, color) = match task.status {
            TaskStatus::Pending => ("○", Theme::text_muted()),
            TaskStatus::InProgress => (spin, Theme::brand_gold()),
            TaskStatus::Completed => ("✓", Theme::accent_green()),
            TaskStatus::Failed => ("✗", Theme::accent_red()),
        };
        let title = crate::tips::ellipsize(&task.title, width.saturating_sub(8).max(8));
        let mut spans = vec![
            Span::styled(
                format!("    {mark} "),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(title, Style::default().fg(Theme::text_primary())),
        ];
        let used_w: usize = spans.iter().map(|s| s.content.width()).sum();
        if used_w < width {
            spans.push(Span::raw(" ".repeat(width - used_w)));
        }
        lines.push(Line::from(spans));
    }
    lines
}

fn bg_spans(
    task: &BgTask,
    spin: &'static str,
    width: usize,
    hot: bool,
    _stop_hot: bool,
) -> (Vec<Span<'static>>, usize) {
    let running = task.status == BgStatus::Running;
    let glyph = if running { spin } else { "✓" };
    let color = if running {
        Theme::brand_gold()
    } else {
        Theme::accent_green()
    };
    let status = if running { "running" } else { "done" };
    let age = crate::ui::turn_bar::fmt_duration(task.started.elapsed().as_secs_f64());
    let mut bits: Vec<(String, Style)> = vec![
        (
            format!(" {glyph} "),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        ("bg  ".into(), Style::default().fg(Theme::accent_cyan())),
    ];
    let trail = vec![
        (
            format!("{status}  "),
            Style::default().fg(if running {
                Theme::brand_gold()
            } else {
                Theme::accent_green()
            }),
        ),
        (
            format!("· {age}  "),
            Style::default().fg(Theme::text_muted()),
        ),
    ];
    let used: usize = bits
        .iter()
        .map(|(t, _)| t.width())
        .chain(trail.iter().map(|(t, _)| t.width()))
        .sum();
    let title_budget = width.saturating_sub(used).max(8);
    let title = if running || task.result.is_empty() {
        task.prompt.as_str()
    } else {
        task.result.as_str()
    };
    bits.push((
        format!(
            "{:<w$}",
            crate::tips::ellipsize(title, title_budget),
            w = title_budget.min(48)
        ),
        Style::default().fg(Theme::text_primary()),
    ));
    let mut spans: Vec<Span<'static>> = bits
        .into_iter()
        .map(|(t, st)| Span::styled(t, st))
        .collect();
    spans.extend(trail.into_iter().map(|(t, st)| Span::styled(t, st)));
    if hot {
        for sp in &mut spans {
            sp.style.bg = Some(Theme::bg_highlight());
        }
    }
    let w: usize = spans.iter().map(|s| s.content.width()).sum();
    if w < width {
        spans.push(Span::raw(" ".repeat(width - w)));
    }
    (spans, 0)
}

fn display_status(status: &str) -> &str {
    match status {
        "completed" => "done",
        other => other,
    }
}

fn glyph(row: &AgentRow, spin: &'static str) -> (&'static str, ratatui::style::Color) {
    if row.is_process() {
        if row.status == "running" {
            (spin, Theme::accent_cyan())
        } else {
            ("▸", Theme::text_muted())
        }
    } else {
        match row.status.as_str() {
            "running" => (spin, Theme::brand_gold()),
            "queued" => ("○", Theme::text_muted()),
            "completed" => ("✓", Theme::accent_green()),
            "failed" | "error" => ("✗", Theme::accent_red()),
            "interrupted" | "timeout" => ("■", Theme::accent_yellow()),
            _ => ("●", Theme::text_secondary()),
        }
    }
}

fn status_color(status: &str) -> ratatui::style::Color {
    match status {
        "running" => Theme::brand_gold(),
        "completed" | "done" => Theme::accent_green(),
        "failed" | "error" => Theme::accent_red(),
        "queued" => Theme::text_muted(),
        _ => Theme::text_secondary(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;

    #[test]
    fn dock_prefers_live_and_caps() {
        let mut s = AppState::new();
        assert_eq!(dock_height(&s), 0);
        for i in 0..6 {
            let mut row = AgentRow::subagent(format!("sa{i}xxxx"));
            row.title = format!("task {i}");
            row.status = "completed".into();
            s.agent_rows.push(row);
        }
        s.agent_rows[5].status = "running".into();
        s.agent_rows[5].input_tokens = 100_000;
        s.agent_rows[5].output_tokens = 18_000;
        s.agent_rows[5].duration_secs = Some(120.0);
        s.agent_rows[5].cost_usd = 0.98;
        s.agent_rows[5].iteration = 28;
        s.agent_rows[5].model = "grok-4".into();
        let entries = dock_entries(&s);
        assert_eq!(entries.len(), 1);
        assert!(matches!(entries[0], DockEntry::Agent(5)));
        assert_eq!(dock_height(&s), 2);
        let (spans, stop_w) = dock_spans(&s.agent_rows[5], "●", 120, false, false);
        let blob: String = spans.iter().map(|sp| sp.content.as_ref()).collect();
        assert!(blob.contains("task 5"));
        assert!(blob.contains("agent"));
        assert!(blob.contains("[stop]"));
        assert!(stop_w > 0);
        assert!(blob.contains("118k tok") || blob.contains("118.0k tok"));
        assert!(blob.contains("turn 28"));
        assert!(blob.contains("$0.98"));
        s.start_bg_task("bg_aa".into(), "summarize hn".into());
        let entries = dock_entries(&s);
        assert!(entries.iter().any(|e| matches!(e, DockEntry::Bg(_))));
        s.tasks.push(crate::state::TaskItem {
            id: "1".into(),
            title: "Inspect repository".into(),
            status: TaskStatus::InProgress,
        });
        s.tasks.push(crate::state::TaskItem {
            id: "2".into(),
            title: "Ship native TUI".into(),
            status: TaskStatus::Pending,
        });
        assert!(dock_height(&s) >= 5);
        let plan = plan_lines(&s, "●", 80, 12, 2);
        let blob: String = plan
            .iter()
            .flat_map(|l| l.spans.iter().map(|sp| sp.content.as_ref()))
            .collect();
        assert!(blob.contains("plan"));
        assert!(blob.contains("Inspect repository"));
        assert!(blob.contains("Ship native TUI"));
        let mut plan_only = AppState::new();
        plan_only.tasks = s.tasks.clone();
        assert!(dock_height(&plan_only) >= 3);
        assert!(dock_entries(&plan_only).is_empty());
        let plan = plan_lines(&plan_only, "●", 80, 8, 0);
        let blob: String = plan
            .iter()
            .flat_map(|l| l.spans.iter().map(|sp| sp.content.as_ref()))
            .collect();
        assert!(blob.contains("Inspect repository"));
        let (bg, _) = bg_spans(&s.bg_tasks[0], "●", 80, false, false);
        let blob: String = bg.iter().map(|sp| sp.content.as_ref()).collect();
        assert!(blob.contains("bg"));
        assert!(blob.contains("summarize hn"));
        let mut proc = AgentRow::subagent("proc_42929e33597b".into());
        proc.kind = "process".into();
        proc.title = "proc_42929e33597b".into();
        proc.status = "running".into();
        proc.output = "hermes doctor\n  ✓ python 3.12\n".into();
        let (spans, stop_w) = dock_spans(&proc, "●", 80, false, true);
        let blob: String = spans.iter().map(|sp| sp.content.as_ref()).collect();
        assert!(blob.contains("proc"));
        assert!(blob.contains("python") || blob.contains("hermes doctor"));
        assert!(!blob.contains("proc_42929e33597b"));
        assert!(blob.contains("[stop]"));
        assert!(stop_w > 0);
    }
}
