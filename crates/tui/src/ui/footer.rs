use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use unicode_width::UnicodeWidthStr;

use super::theme::Theme;
use crate::state::{AppState, HoverKind, PermissionMode, SessionMetrics, StatusBarMode};

pub struct Footer;

impl Footer {
    pub const HEIGHT: u16 = 1;

    pub fn height(state: &AppState) -> u16 {
        match state.status_bar {
            StatusBarMode::Off => 0,
            StatusBarMode::Top | StatusBarMode::Bottom => Self::HEIGHT,
        }
    }

    pub fn render(frame: &mut Frame, area: Rect, state: &mut AppState, frame_count: u64) {
        if area.height == 0 {
            return;
        }
        let width = area.width as usize;
        let reveal = state.reveal();
        if reveal < 0.04 {
            frame.render_widget(
                Paragraph::new("").style(Style::default().bg(Theme::bg_base())),
                area,
            );
            state.hit_model = None;
            state.hit_branch = None;
            state.hit_mode = None;
            state.hit_context = None;
            state.hit_session = None;
            state.hit_bg = None;
            state.hit_agents = None;
            state.hit_process = None;
            return;
        }

        let flash = state.flash_kind();
        let built = status_spans(state, &state.hover, flash.as_ref(), frame_count);
        let mut left = built.spans;
        if let Some(toast) = &state.metrics.toast_message {
            if toast.live() {
                left.push(Span::styled(
                    format!("  {}", toast.text),
                    Style::default().fg(Theme::brand_gold()),
                ));
            }
        }

        let hint = hint_text(state);
        let right = Span::styled(hint, Style::default().fg(Theme::text_dim()));
        let left_w = spans_width(&left);
        let right_w = right.content.width();
        let mut spans = left;
        if left_w + right_w + 2 <= width {
            spans.push(Span::raw(
                " ".repeat(width.saturating_sub(left_w + right_w).max(1)),
            ));
            spans.push(right);
        }
        if reveal < 0.995 {
            crate::ui::theme::fade_spans(&mut spans, reveal);
        }

        let row_y = area.y;
        state.hit_mode = built
            .mode
            .map(|(x0, x1)| crate::state::HitRange { y: row_y, x0, x1 });
        state.hit_model = built
            .model
            .map(|(x0, x1)| crate::state::HitRange { y: row_y, x0, x1 });
        state.hit_branch = built
            .branch
            .map(|(x0, x1)| crate::state::HitRange { y: row_y, x0, x1 });
        state.hit_context =
            built
                .context
                .map(|(x0, x1)| crate::state::HitRange { y: row_y, x0, x1 });
        state.hit_session =
            built
                .session
                .map(|(x0, x1)| crate::state::HitRange { y: row_y, x0, x1 });
        state.hit_bg = built
            .bg
            .map(|(x0, x1)| crate::state::HitRange { y: row_y, x0, x1 });
        state.hit_agents = built
            .agents
            .map(|(x0, x1)| crate::state::HitRange { y: row_y, x0, x1 });
        state.hit_process =
            built
                .process
                .map(|(x0, x1)| crate::state::HitRange { y: row_y, x0, x1 });

        let status_area = Rect {
            x: area.x,
            y: area.y,
            width: area.width,
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(Line::from(spans)).style(Style::default().bg(Theme::bg_header())),
            status_area,
        );
    }
}

fn hint_text(state: &AppState) -> String {
    if state.pending_secret.is_some() {
        "enter submit · esc cancel".into()
    } else if state.pending_clarify.is_some() {
        "enter confirm · esc dismiss".into()
    } else if state.pending_approval.is_some() {
        "y once · a always · n deny".into()
    } else if state.queue_edit.is_some() {
        "enter send now · ctrl+x drop · esc cancel".into()
    } else if !state.prompt_queue.is_empty() {
        format!(
            "↑↓ edit · enter send now · queued {}",
            state.prompt_queue.len()
        )
    } else if state.trace_focus {
        "↑↓ step · p pause · e edit · r resume · esc composer".into()
    } else if let Some(u) = state.pending_undo.as_ref().filter(|u| u.live()) {
        u.hint().into()
    } else if state.metrics.is_compacting {
        "folding the window".into()
    } else if state.is_generating {
        match state.busy_mode {
            crate::state::BusyMode::Queue => "esc interrupt · enter queues next".into(),
            crate::state::BusyMode::Steer => "esc interrupt · enter steers this turn".into(),
            crate::state::BusyMode::Interrupt => "esc interrupt · enter redirects".into(),
        }
    } else {
        "enter send · / commands · tab thoughts".into()
    }
}

struct StatusBuilt {
    spans: Vec<Span<'static>>,
    mode: Option<(u16, u16)>,
    model: Option<(u16, u16)>,
    branch: Option<(u16, u16)>,
    context: Option<(u16, u16)>,
    session: Option<(u16, u16)>,
    bg: Option<(u16, u16)>,
    agents: Option<(u16, u16)>,
    process: Option<(u16, u16)>,
}

fn chip_style(
    kind: &HoverKind,
    hover: &HoverKind,
    flash: Option<&HoverKind>,
    idle: Style,
) -> Style {
    if flash == Some(kind) {
        Style::default()
            .fg(Theme::bg_base())
            .bg(Theme::brand_gold())
            .add_modifier(Modifier::BOLD)
    } else if hover == kind {
        Style::default()
            .fg(Theme::text_primary())
            .bg(Theme::bg_highlight())
            .add_modifier(Modifier::BOLD)
    } else {
        idle
    }
}

fn status_spans(
    state: &AppState,
    hover: &HoverKind,
    flash: Option<&HoverKind>,
    frame_count: u64,
) -> StatusBuilt {
    let metrics = &state.metrics;
    let mode_idle = match metrics.permission_mode {
        PermissionMode::Plan => Style::default()
            .fg(Theme::brand_gold())
            .add_modifier(Modifier::BOLD),
        PermissionMode::Manual => Style::default().fg(Theme::text_secondary()),
        PermissionMode::Yolo => Style::default()
            .fg(Theme::accent_red())
            .add_modifier(Modifier::BOLD),
    };

    let mut x = 0u16;
    let mut spans = Vec::new();
    let push = |spans: &mut Vec<Span<'static>>, x: &mut u16, span: Span<'static>| {
        *x = x.saturating_add(span.content.width() as u16);
        spans.push(span);
    };

    let mode_x0 = x;
    push(
        &mut spans,
        &mut x,
        Span::styled(
            format!(" {}  ", metrics.permission_mode.label()),
            chip_style(&HoverKind::Mode, hover, flash, mode_idle),
        ),
    );
    let mode = Some((mode_x0, x));
    if let Some(vim) = state.vim {
        push(
            &mut spans,
            &mut x,
            Span::styled(
                format!(" {}  ", crate::composer_vim::label(vim)),
                Style::default().fg(Theme::accent_cyan()),
            ),
        );
    }
    if state.fast_mode {
        push(
            &mut spans,
            &mut x,
            Span::styled("fast  ", Style::default().fg(Theme::accent_cyan())),
        );
    }
    let glyph = if state.is_generating {
        format!(
            "{} ",
            super::motion::spinner_for(state.indicator, frame_count)
        )
    } else {
        "⚕ ".into()
    };
    push(
        &mut spans,
        &mut x,
        Span::styled(glyph, Style::default().fg(Theme::brand_gold())),
    );
    push(
        &mut spans,
        &mut x,
        Span::styled(
            "hermes  ",
            Style::default()
                .fg(Theme::brand_gold())
                .add_modifier(Modifier::BOLD),
        ),
    );

    let mut session = None;
    if !state.is_generating {
        let title = state.display_title();
        if !title.is_empty() && title != "Hermes TUI" {
            let session_x0 = x;
            push(
                &mut spans,
                &mut x,
                Span::styled(
                    format!("{title}  "),
                    chip_style(
                        &HoverKind::Session,
                        hover,
                        flash,
                        Style::default().fg(Theme::text_primary()),
                    ),
                ),
            );
            session = Some((session_x0, x));
        }
    }

    let bg_hit = None;
    let agents_hit = None;
    let process_hit = None;

    let short_cwd = short_cwd(&metrics.cwd);
    let cwd_leaf = short_cwd.rsplit('/').next().unwrap_or(short_cwd.as_str());
    push(
        &mut spans,
        &mut x,
        Span::styled(
            format!("{short_cwd}  "),
            Style::default().fg(Theme::text_secondary()),
        ),
    );

    if let Some(backend) = sandbox_label(&metrics.terminal_backend) {
        push(
            &mut spans,
            &mut x,
            Span::styled(
                format!("{backend}  "),
                Style::default().fg(Theme::accent_cyan()),
            ),
        );
    }

    let gold = Style::default().fg(Theme::brand_gold());
    let mut branch_hit = None;
    if let Some(branch) = &metrics.git_branch {
        let x0 = x;
        push(
            &mut spans,
            &mut x,
            Span::styled(
                format!("{branch}  "),
                chip_style(&HoverKind::Branch, hover, flash, gold),
            ),
        );
        branch_hit = Some((x0, x));
    } else if let Some(repo) = &metrics.git_repo {
        if repo != cwd_leaf {
            push(
                &mut spans,
                &mut x,
                Span::styled(
                    format!("{repo}  "),
                    Style::default().fg(Theme::text_muted()),
                ),
            );
        }
    }

    let model_label = model_chip(metrics);
    let model_x0 = x;
    push(
        &mut spans,
        &mut x,
        Span::styled(
            model_label,
            chip_style(&HoverKind::Model, hover, flash, gold),
        ),
    );
    let model = Some((model_x0, x));

    let pct = metrics.context_pct();
    let used = if metrics.context_used > 0 {
        metrics.context_used
    } else {
        metrics.total_tokens
    };
    let ctx_x0 = x;
    let ctx_style = if metrics.is_compacting {
        Style::default()
            .fg(Theme::brand_gold())
            .add_modifier(Modifier::BOLD)
    } else if pct > 85.0 {
        Style::default().fg(Theme::accent_red())
    } else if pct > 60.0 {
        Style::default().fg(Theme::accent_yellow())
    } else {
        Style::default().fg(Theme::text_secondary())
    };
    let bar_pct = if metrics.is_compacting {
        let t = ((frame_count / 2) % 24) as f64 / 24.0;
        let tri = if t < 0.5 { t * 2.0 } else { (1.0 - t) * 2.0 };
        (pct * (0.28 + 0.72 * tri)).clamp(6.0, 100.0)
    } else {
        pct
    };
    let ctx_label = if metrics.context_limit > 0 {
        format!(
            "[{}] {}/{}  ",
            crate::ui::context::ctx_bar(bar_pct, 8),
            crate::ui::context::fmt_k(used),
            crate::ui::context::fmt_k(metrics.context_limit)
        )
    } else if used > 0 {
        format!("{} tok  ", crate::ui::context::fmt_k(used))
    } else {
        String::new()
    };
    if !ctx_label.is_empty() {
        push(
            &mut spans,
            &mut x,
            Span::styled(
                ctx_label,
                chip_style(&HoverKind::Context, hover, flash, ctx_style),
            ),
        );
    }
    let context = if ctx_x0 < x { Some((ctx_x0, x)) } else { None };

    if metrics.estimated_cost_usd >= 0.005 {
        push(
            &mut spans,
            &mut x,
            Span::styled(
                format!("${:.2}  ", metrics.estimated_cost_usd),
                Style::default().fg(Theme::text_muted()),
            ),
        );
    }

    if metrics.tokens_per_sec > 0.5 {
        push(
            &mut spans,
            &mut x,
            Span::styled(
                format!("{:.0}/s  ", metrics.tokens_per_sec),
                Style::default().fg(Theme::text_muted()),
            ),
        );
    }

    let _ = x;
    StatusBuilt {
        spans,
        mode,
        model,
        branch: branch_hit,
        context,
        session,
        bg: bg_hit,
        agents: agents_hit,
        process: process_hit,
    }
}

fn sandbox_label(backend: &str) -> Option<&str> {
    let t = backend.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("local") {
        None
    } else {
        Some(t)
    }
}

fn model_chip(metrics: &SessionMetrics) -> String {
    let model = short_model(&metrics.active_model);
    let provider = metrics.active_provider.rsplit('/').next().unwrap_or("");
    if !provider.is_empty() && !model.is_empty() {
        format!("{provider} · {model}  ")
    } else if !model.is_empty() {
        format!("{model}  ")
    } else {
        "model  ".into()
    }
}

fn short_cwd(cwd: &str) -> String {
    let home_dir = std::env::var("HOME").unwrap_or_default();
    let display_cwd = if !home_dir.is_empty() && cwd.starts_with(&home_dir) {
        format!("~{}", &cwd[home_dir.len()..])
    } else {
        cwd.to_string()
    };
    display_cwd
        .rsplit('/')
        .next()
        .filter(|leaf| !leaf.is_empty())
        .unwrap_or(display_cwd.as_str())
        .to_string()
}

fn short_model(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

fn spans_width(spans: &[Span<'_>]) -> usize {
    spans.iter().map(|s| s.content.width()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;

    #[test]
    fn mode_is_leftmost_status() {
        let mut s = AppState::new();
        s.metrics.permission_mode = PermissionMode::Manual;
        s.metrics.active_model = "x-ai/grok-4".into();
        s.metrics.active_provider = "xai".into();
        let built = status_spans(&s, &HoverKind::None, None, 0);
        let blob: String = built.spans.iter().map(|sp| sp.content.as_ref()).collect();
        let mode_at = blob.find("ask").expect("ask mode");
        let hermes_at = blob.find("hermes").expect("hermes");
        let model_at = blob.find("grok-4").expect("model");
        assert!(mode_at < hermes_at, "mode sits left of identity");
        assert!(hermes_at < model_at);
        assert!(blob.contains("xai · grok-4"));
        s.metrics.context_used = 235_000;
        s.metrics.context_limit = 900_000;
        s.session_title = "ship tui".into();
        let built = status_spans(&s, &HoverKind::None, None, 0);
        let blob: String = built.spans.iter().map(|sp| sp.content.as_ref()).collect();
        assert!(blob.contains("ship tui"));
        assert!(blob.contains("235k"));
        assert!(blob.contains("900k"));
        assert!(built.context.is_some());
        assert!(built.session.is_some());
        let (x0, x1) = built.model.expect("model hit range");
        assert!(x1 > x0);
        s.metrics.git_branch = Some("main".into());
        let built = status_spans(&s, &HoverKind::None, None, 0);
        let blob: String = built.spans.iter().map(|sp| sp.content.as_ref()).collect();
        assert!(blob.contains("main"));
        let (bx0, bx1) = built.branch.expect("branch hit range");
        assert!(bx1 > bx0);
    }

    #[test]
    fn yolo_label_is_leftmost() {
        let mut s = AppState::new();
        s.metrics.permission_mode = PermissionMode::Yolo;
        let built = status_spans(&s, &HoverKind::None, None, 0);
        let blob: String = built.spans.iter().map(|sp| sp.content.as_ref()).collect();
        assert!(blob.trim_start().starts_with("yolo"));
        s.metrics.permission_mode = PermissionMode::Plan;
        let built = status_spans(&s, &HoverKind::None, None, 0);
        let blob: String = built.spans.iter().map(|sp| sp.content.as_ref()).collect();
        assert!(blob.trim_start().starts_with("plan"));
    }

    #[test]
    fn sandbox_chip_hides_local() {
        assert_eq!(sandbox_label(""), None);
        assert_eq!(sandbox_label("local"), None);
        assert_eq!(sandbox_label("docker"), Some("docker"));
        let mut s = AppState::new();
        s.metrics.terminal_backend = "docker".into();
        let built = status_spans(&s, &HoverKind::None, None, 0);
        let blob: String = built.spans.iter().map(|sp| sp.content.as_ref()).collect();
        assert!(blob.contains("docker"));
    }

    #[test]
    fn hover_lightens_text_without_moving_hits() {
        crate::ui::theme::apply(crate::palette::Palette::gold());
        let mut s = AppState::new();
        s.metrics.git_branch = Some("main".into());
        s.metrics.active_model = "grok".into();
        let idle = status_spans(&s, &HoverKind::None, None, 0);
        let hot = status_spans(&s, &HoverKind::Model, None, 0);
        assert_eq!(idle.model, hot.model);
        let idle_fg = idle.spans.iter().find(|sp| sp.content.contains("grok"));
        let hot_fg = hot.spans.iter().find(|sp| sp.content.contains("grok"));
        assert_eq!(idle_fg.unwrap().style.fg, Some(Theme::brand_gold()));
        assert_eq!(hot_fg.unwrap().style.fg, Some(Theme::text_primary()));
        assert_eq!(hot_fg.unwrap().style.bg, Some(Theme::bg_highlight()));
        assert_eq!(Footer::HEIGHT, 1);
        let mut off = AppState::new();
        off.status_bar = StatusBarMode::Off;
        assert_eq!(Footer::height(&off), 0);
    }

    #[test]
    fn background_count_chip() {
        let mut s = AppState::new();
        s.start_bg_task("bg_a".into(), "one".into());
        s.start_bg_task("bg_b".into(), "two".into());
        let live = status_spans(&s, &HoverKind::None, None, 0);
        let blob: String = live.spans.iter().map(|sp| sp.content.as_ref()).collect();
        assert!(
            !blob.contains('▶'),
            "live work belongs in the dock, not the footer"
        );
        s.complete_bg_task("bg_a", "ok");
        s.upsert_subagent(
            &serde_json::json!({
                "subagent_id": "sa1",
                "goal": "audit",
                "depth": 0,
                "task_index": 1
            }),
            Some("running"),
            true,
        );
        let live = status_spans(&s, &HoverKind::None, None, 0);
        let blob: String = live.spans.iter().map(|sp| sp.content.as_ref()).collect();
        assert!(!blob.contains('◆'));
        s.merge_agent_snapshot(
            &serde_json::json!({
                "processes": [{
                    "session_id": "proc_1",
                    "command": "sleep 30",
                    "status": "running",
                    "uptime_seconds": 3
                }]
            }),
            &serde_json::json!({}),
        );
        let live = status_spans(&s, &HoverKind::None, None, 0);
        let blob: String = live.spans.iter().map(|sp| sp.content.as_ref()).collect();
        assert!(!blob.contains("▸ 1"));
        assert!(live.process.is_none());
        assert_eq!(s.running_process_count(), 1);
    }
}
