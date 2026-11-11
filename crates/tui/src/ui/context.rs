use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use super::theme::Theme;
use crate::state::{AppState, MessageRole};
use serde_json::Value;

pub const MAP_COLS: usize = 20;
pub const MAP_ROWS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceKind {
    System,
    Tools,
    Skills,
    Messages,
    Reasoning,
    Free,
}

impl SliceKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Tools => "tools",
            Self::Skills => "skills",
            Self::Messages => "messages",
            Self::Reasoning => "thought",
            Self::Free => "free",
        }
    }

    pub fn glyph(self) -> &'static str {
        match self {
            Self::Free => "□",
            _ => "▣",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextSlice {
    pub kind: SliceKind,
    pub tokens: u64,
}

#[derive(Debug, Clone)]
pub struct ContextMap {
    pub used: u64,
    pub limit: u64,
    pub slices: Vec<ContextSlice>,
}

impl ContextMap {
    pub fn pct(&self) -> f64 {
        if self.limit == 0 {
            0.0
        } else {
            (self.used as f64 / self.limit as f64) * 100.0
        }
    }
}

pub fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        0
    } else {
        (text.len() as u64 / 4).max(1)
    }
}

pub fn fmt_k(n: u64) -> String {
    if n >= 1_000_000 {
        let v = n as f64 / 1_000_000.0;
        if v >= 10.0 {
            format!("{v:.0}m")
        } else {
            format!("{v:.1}m")
        }
    } else if n >= 1000 {
        let v = n as f64 / 1000.0;
        if n >= 100_000 || (v - v.round()).abs() < 0.05 {
            format!("{:.0}k", v)
        } else {
            format!("{v:.1}k")
        }
    } else {
        n.to_string()
    }
}

pub fn ctx_bar(pct: f64, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let fill = ((pct.clamp(0.0, 100.0) / 100.0) * width as f64).round() as usize;
    let fill = fill.min(width);
    format!("{}{}", "█".repeat(fill), "░".repeat(width - fill))
}

/// Prefer gateway `session.context_breakdown` totals when present.
pub fn apply_breakdown(state: &mut AppState, v: &Value) {
    if let Some(used) = v
        .get("context_used")
        .and_then(|x| x.as_u64())
        .or_else(|| v.get("estimated_total").and_then(|x| x.as_u64()))
    {
        if used > 0 {
            state.metrics.context_used = used;
        }
    }
    if let Some(max) = v.get("context_max").and_then(|x| x.as_u64()) {
        if max > 0 {
            state.metrics.context_limit = max;
        }
    }
}

pub fn build_map(state: &AppState) -> ContextMap {
    let mut system = 0u64;
    let mut tools = 0u64;
    let mut messages = 0u64;
    let mut reasoning = 0u64;
    for m in &state.messages {
        let t = match &m.role {
            MessageRole::ImagePreview { .. } => 800,
            _ => estimate_tokens(&m.content),
        };
        match &m.role {
            MessageRole::System | MessageRole::Compaction => system += t,
            MessageRole::Tool { .. } => tools += t,
            MessageRole::Reasoning => reasoning += t,
            MessageRole::User | MessageRole::Assistant | MessageRole::ImagePreview { .. } => {
                messages += t;
            }
        }
    }
    let skill_n = state.skills.len().max(
        state
            .intro_skills
            .iter()
            .map(|(_, n)| n.len())
            .sum::<usize>(),
    );
    let skills = skill_n as u64 * 160;

    let local = system + tools + skills + messages + reasoning;
    let used = if state.metrics.context_used > 0 {
        state.metrics.context_used
    } else {
        local
    };
    let limit = if state.metrics.context_limit > 0 {
        state.metrics.context_limit
    } else {
        used.max(32_000)
    };

    let mut slices = vec![
        ContextSlice {
            kind: SliceKind::System,
            tokens: system,
        },
        ContextSlice {
            kind: SliceKind::Tools,
            tokens: tools,
        },
        ContextSlice {
            kind: SliceKind::Skills,
            tokens: skills,
        },
        ContextSlice {
            kind: SliceKind::Messages,
            tokens: messages,
        },
        ContextSlice {
            kind: SliceKind::Reasoning,
            tokens: reasoning,
        },
    ];
    let cat_sum: u64 = slices.iter().map(|s| s.tokens).sum();
    if used > cat_sum {
        slices[3].tokens += used - cat_sum;
    } else if used < cat_sum && cat_sum > 0 {
        for s in &mut slices {
            s.tokens = s.tokens * used / cat_sum;
        }
    }
    let filled: u64 = slices.iter().map(|s| s.tokens).sum();
    let free = limit.saturating_sub(filled.max(used));
    slices.push(ContextSlice {
        kind: SliceKind::Free,
        tokens: free,
    });
    ContextMap {
        used,
        limit,
        slices,
    }
}

pub fn grid_cells(map: &ContextMap) -> Vec<SliceKind> {
    let total = (MAP_COLS * MAP_ROWS) as u64;
    let denom = map.limit.max(1);
    let mut cells = Vec::with_capacity(total as usize);
    for slice in &map.slices {
        let n = ((slice.tokens as f64 / denom as f64) * total as f64).round() as usize;
        cells.extend(std::iter::repeat_n(slice.kind, n));
    }
    cells.truncate(total as usize);
    while cells.len() < total as usize {
        cells.push(SliceKind::Free);
    }
    cells
}

fn slice_color(kind: SliceKind) -> ratatui::style::Color {
    match kind {
        SliceKind::System => Theme::text_muted(),
        SliceKind::Tools => Theme::brand_orange(),
        SliceKind::Skills => Theme::brand_gold(),
        SliceKind::Messages => Theme::text_primary(),
        SliceKind::Reasoning => Theme::text_secondary(),
        SliceKind::Free => Theme::text_dim(),
    }
}

pub struct ContextModal;

impl ContextModal {
    pub fn render(frame: &mut Frame, area: Rect, state: &AppState) {
        let popup = centered(72, 72, area);
        frame.render_widget(Clear, popup);
        let map = build_map(state);
        let cells = grid_cells(&map);
        let pct = map.pct();

        let block = Block::default()
            .title(Span::styled(
                " context  ↑↓  enter compress  esc ",
                Style::default().fg(Theme::brand_gold()),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Theme::border_focus()))
            .style(Style::default().bg(Theme::bg_popup()));
        let inner = block.inner(popup);
        frame.render_widget(block, popup);

        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(44), Constraint::Min(24)])
            .split(inner);
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length((MAP_ROWS as u16) + 3),
                Constraint::Min(6),
                Constraint::Length(4),
            ])
            .split(cols[0]);

        let mut map_lines = vec![Line::from(Span::styled(
            "  usage map",
            Style::default()
                .fg(Theme::text_primary())
                .add_modifier(Modifier::BOLD),
        ))];
        for r in 0..MAP_ROWS {
            let mut spans = vec![Span::raw("  ")];
            for c in 0..MAP_COLS {
                let kind = cells[r * MAP_COLS + c];
                spans.push(Span::styled(
                    format!("{} ", kind.glyph()),
                    Style::default().fg(slice_color(kind)),
                ));
            }
            map_lines.push(Line::from(spans));
        }
        frame.render_widget(Paragraph::new(map_lines), rows[0]);

        let model = if state.metrics.active_model.is_empty() {
            "model".into()
        } else {
            state
                .metrics
                .active_model
                .rsplit('/')
                .next()
                .unwrap_or("")
                .to_string()
        };
        let mut legend = vec![
            Line::from(Span::styled(
                format!("  {model}"),
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!("  {}/{}  ({:.0}%)", fmt_k(map.used), fmt_k(map.limit), pct),
                Style::default().fg(Theme::text_secondary()),
            )),
            Line::raw(""),
            Line::from(Span::styled(
                "  by category",
                Style::default().fg(Theme::text_muted()),
            )),
        ];
        for s in &map.slices {
            let share = if map.limit == 0 {
                0.0
            } else {
                s.tokens as f64 / map.limit as f64 * 100.0
            };
            legend.push(Line::from(vec![
                Span::styled(
                    format!("  {} ", s.kind.glyph()),
                    Style::default().fg(slice_color(s.kind)),
                ),
                Span::styled(
                    format!(
                        "{:<9}  {:>6}  ({:.1}%)",
                        s.kind.label(),
                        fmt_k(s.tokens),
                        share
                    ),
                    Style::default().fg(slice_color(s.kind)),
                ),
            ]));
        }
        frame.render_widget(Paragraph::new(legend).wrap(Wrap { trim: false }), cols[1]);

        let tools_n: usize = state.intro_tools.iter().map(|(_, n)| n.len()).sum();
        let skills_n: usize = state
            .skills
            .len()
            .max(state.intro_skills.iter().map(|(_, n)| n.len()).sum());
        let inventory = vec![
            Line::from(Span::styled(
                format!("  mcp     {} connected", state.mcp_connected),
                Style::default().fg(Theme::text_secondary()),
            )),
            Line::from(Span::styled(
                format!("  tools   {tools_n} loaded"),
                Style::default().fg(Theme::text_secondary()),
            )),
            Line::from(Span::styled(
                format!("  skills  {skills_n}"),
                Style::default().fg(Theme::text_secondary()),
            )),
            Line::from(Span::styled(
                "  /compress folds the window   click the footer chip to reopen",
                Style::default().fg(Theme::text_dim()),
            )),
        ];
        frame.render_widget(Paragraph::new(inventory), rows[1]);
    }
}

fn centered(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(v[1])[1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{AppState, ChatMessage, MessageRole};
    use chrono::Local;

    #[test]
    fn apply_breakdown_sets_gateway_totals() {
        let mut s = AppState::new();
        apply_breakdown(
            &mut s,
            &serde_json::json!({
                "context_used": 104_000,
                "context_max": 900_000,
                "estimated_total": 80_000
            }),
        );
        assert_eq!(s.metrics.context_used, 104_000);
        assert_eq!(s.metrics.context_limit, 900_000);
    }

    #[test]
    fn fmt_k_compact() {
        assert_eq!(fmt_k(235_000), "235k");
        assert_eq!(fmt_k(33_300), "33.3k");
        assert_eq!(fmt_k(800), "800");
        assert_eq!(fmt_k(1_000_000), "1.0m");
    }

    #[test]
    fn bar_fills() {
        assert_eq!(ctx_bar(0.0, 4), "░░░░");
        assert_eq!(ctx_bar(50.0, 4), "██░░");
        assert_eq!(ctx_bar(100.0, 4), "████");
    }

    #[test]
    fn map_has_free_space() {
        let mut s = AppState::new();
        s.metrics.context_used = 1000;
        s.metrics.context_limit = 10_000;
        s.messages.push(ChatMessage {
            id: "1".into(),
            role: MessageRole::User,
            content: "hello world this is a reasonably long prompt".into(),
            timestamp: Local::now(),
            output: String::new(),
            is_streaming: false,
        });
        let map = build_map(&s);
        assert_eq!(map.used, 1000);
        assert_eq!(map.limit, 10_000);
        let free = map
            .slices
            .iter()
            .find(|x| x.kind == SliceKind::Free)
            .unwrap();
        assert!(free.tokens > 0);
        let cells = grid_cells(&map);
        assert_eq!(cells.len(), MAP_COLS * MAP_ROWS);
        assert!(cells.contains(&SliceKind::Free));
    }
}
