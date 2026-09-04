use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};
use unicode_width::UnicodeWidthStr;

use super::theme::Theme;
use crate::state::AppState;

/// Official Hermes Agent block letters (`ui-tui/src/banner.ts`).
const LOGO_ART: &[&str] = &[
    "██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗",
    "██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝",
    "███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ",
    "██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ",
    "██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ",
    "╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ",
];

/// Official caduceus (`ui-tui/src/banner.ts`).
const CADUCEUS_ART: &[&str] = &[
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⣀⣀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣇⠸⣿⣿⠇⣸⣿⣿⣷⣦⣄⡀⠀⠀⠀⠀⠀⠀",
    "⠀⢀⣠⣴⣶⠿⠋⣩⡿⣿⡿⠻⣿⡇⢠⡄⢸⣿⠟⢿⣿⢿⣍⠙⠿⣶⣦⣄⡀⠀",
    "⠀⠀⠉⠉⠁⠶⠟⠋⠀⠉⠀⢀⣈⣁⡈⢁⣈⣁⡀⠀⠉⠀⠙⠻⠶⠈⠉⠉⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⡿⠛⢁⡈⠛⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⣿⣦⣤⣈⠁⢠⣴⣿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠻⢿⣿⣦⡉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣦⣈⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⠦⠈⠙⠿⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣤⡈⠁⢤⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠷⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠑⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⢰⡆⠈⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⠈⣡⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
];

const TAG_FULL: &str = "Nous Research · Messenger of the Digital Gods";
const TAG_MID: &str = "Messenger of the Digital Gods";
const LOGO_GRADIENT: &[usize] = &[0, 0, 1, 1, 2, 2];

fn logo_color(i: usize) -> ratatui::style::Color {
    match i {
        0 => Theme::brand_gold(),
        1 => Theme::brand_orange(),
        _ => Theme::text_muted(),
    }
}

fn colorize_static(art: &[&str], gradient: &[usize]) -> Vec<Line<'static>> {
    art.iter()
        .enumerate()
        .map(|(i, text)| {
            let color = logo_color(*gradient.get(i).unwrap_or(&2));
            Line::from(vec![Span::styled(
                (*text).to_string(),
                Style::default().fg(color),
            )])
        })
        .collect()
}

fn art_width(art: &[&str]) -> usize {
    art.iter().map(|l| l.width()).max().unwrap_or(0)
}

fn is_blank_glyph(ch: char) -> bool {
    ch == ' ' || ch == '\u{2800}'
}

fn shimmer_color(row: usize, col: usize, frame: u64) -> ratatui::style::Color {
    // Diagonal gold band; ~2s per loop at the 80ms tick.
    let phase = (frame / 2) as i32;
    match (row as i32 * 3 + col as i32 - phase).rem_euclid(22) {
        0..=2 => Theme::brand_gold(),
        3..=5 => Theme::brand_orange(),
        6..=11 => Theme::text_muted(),
        _ => Theme::text_dim(),
    }
}

fn flush_run(
    spans: &mut Vec<Span<'static>>,
    run: &mut String,
    color: Option<ratatui::style::Color>,
    bold: bool,
) {
    if run.is_empty() {
        return;
    }
    let text = std::mem::take(run);
    match color {
        None => spans.push(Span::raw(text)),
        Some(c) => {
            let mut style = Style::default().fg(c);
            if bold {
                style = style.add_modifier(Modifier::BOLD);
            }
            spans.push(Span::styled(text, style));
        }
    }
}

/// Per-glyph gold wave. Empty braille cells stay unlit so the staff reads.
fn colorize_shimmer(art: &[&str], frame: u64, indent: &str) -> Vec<Line<'static>> {
    art.iter()
        .enumerate()
        .map(|(row, text)| {
            let mut spans = vec![Span::raw(indent.to_string())];
            let mut run = String::new();
            let mut run_color: Option<ratatui::style::Color> = None;
            let mut run_bold = false;
            for (col, ch) in text.chars().enumerate() {
                let (color, bold) = if is_blank_glyph(ch) {
                    (None, false)
                } else {
                    let c = shimmer_color(row, col, frame);
                    (Some(c), c == Theme::brand_gold())
                };
                if color == run_color && bold == run_bold {
                    run.push(ch);
                } else {
                    flush_run(&mut spans, &mut run, run_color, run_bold);
                    run.push(ch);
                    run_color = color;
                    run_bold = bold;
                }
            }
            flush_run(&mut spans, &mut run, run_color, run_bold);
            Line::from(spans)
        })
        .collect()
}

fn styled(text: impl Into<String>, color: ratatui::style::Color, bold: bool) -> Line<'static> {
    let mut style = Style::default().fg(color);
    if bold {
        style = style.add_modifier(Modifier::BOLD);
    }
    Line::from(vec![Span::styled(text.into(), style)])
}

fn mixed(parts: Vec<(String, ratatui::style::Color, bool)>) -> Line<'static> {
    let mut spans = Vec::new();
    for (text, color, bold) in parts {
        let mut style = Style::default().fg(color);
        if bold {
            style = style.add_modifier(Modifier::BOLD);
        }
        spans.push(Span::styled(text, style));
    }
    Line::from(spans)
}

fn line_width(line: &Line<'_>) -> usize {
    line.spans.iter().map(|s| s.content.width()).sum()
}

fn center_block(lines: Vec<Line<'static>>, width: usize) -> Vec<Line<'static>> {
    let block_w = lines.iter().map(line_width).max().unwrap_or(0);
    let pad = width.saturating_sub(block_w) / 2;
    if pad == 0 {
        return lines;
    }
    lines
        .into_iter()
        .map(|line| {
            let mut spans = vec![Span::raw(" ".repeat(pad))];
            spans.extend(line.spans);
            Line::from(spans)
        })
        .collect()
}

fn short_model(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

fn home_cwd(cwd: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    if !home.is_empty() && cwd.starts_with(&home) {
        format!("~{}", &cwd[home.len()..])
    } else {
        cwd.to_string()
    }
}

fn clip(s: &str, w: usize) -> String {
    if w == 0 {
        return String::new();
    }
    if s.width() <= w {
        return s.to_string();
    }
    let mut out = String::new();
    for c in s.chars() {
        if out.width() + 1 >= w {
            break;
        }
        out.push(c);
    }
    out.push('…');
    out
}

fn rule_in(label: &str, w: usize) -> String {
    let f = clip(label, w.saturating_sub(4).max(1));
    let slack = w.saturating_sub(f.width() + 2);
    let left = slack / 2;
    format!("{} {f} {}", "─".repeat(left), "─".repeat(slack - left))
}

fn trunc_items(items: &[String], budget: usize) -> String {
    let mut line = String::new();
    for (i, item) in items.iter().enumerate() {
        let next = if line.is_empty() {
            item.clone()
        } else {
            format!("{line}, {item}")
        };
        if next.width() > budget {
            if line.is_empty() {
                return clip(item, budget);
            }
            return format!("{line}, …+{}", items.len() - i);
        }
        line = next;
    }
    line
}

fn compact_banner(width: usize) -> Vec<Line<'static>> {
    let w = width.saturating_sub(4).max(28);
    vec![
        styled(rule_in("hermes", w), Theme::brand_gold(), false),
        styled(clip(TAG_FULL, w), Theme::text_muted(), false),
        styled("─".repeat(w), Theme::brand_gold(), false),
    ]
}

fn banner(width: usize) -> Vec<Line<'static>> {
    let logo_w = art_width(LOGO_ART);
    let lines = if width >= logo_w + 2 {
        let mut lines = colorize_static(LOGO_ART, LOGO_GRADIENT);
        lines.push(mixed(vec![
            ("⚕ ".into(), Theme::brand_gold(), true),
            (TAG_FULL.into(), Theme::text_muted(), false),
        ]));
        lines
    } else if width >= 58 {
        compact_banner(width)
    } else if width >= 34 {
        let tag = if width >= 46 {
            TAG_MID
        } else {
            "Nous Research"
        };
        vec![
            styled("⚕ hermes", Theme::brand_gold(), true),
            styled(format!("⚕ {tag}"), Theme::text_muted(), false),
        ]
    } else {
        vec![styled("⚕ hermes", Theme::brand_gold(), true)]
    };
    center_block(lines, width)
}

fn info_lines(state: &AppState, col_w: usize) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut title = "Hermes Agent".to_string();
    if !state.metrics.hermes_version.is_empty() {
        title.push_str(&format!("  v{}", state.metrics.hermes_version));
    }
    if !state.release_date.is_empty() {
        title.push_str(&format!("  ({})", state.release_date));
    }
    out.push(styled(clip(&title, col_w), Theme::brand_gold(), true));

    let model = short_model(&state.metrics.active_model);
    if model.is_empty() {
        out.push(styled("connecting…  /model", Theme::text_muted(), false));
    } else {
        out.push(mixed(vec![
            (
                clip(model, col_w.saturating_sub(18)),
                Theme::brand_gold(),
                false,
            ),
            (" · Nous Research".into(), Theme::text_muted(), false),
        ]));
    }

    let cwd = home_cwd(&state.metrics.cwd);
    out.push(styled(clip(&cwd, col_w), Theme::text_muted(), false));

    if let Some(sid) = state.session_id.as_deref() {
        let shown = if sid.chars().count() > 12 {
            format!("{}…", sid.chars().take(8).collect::<String>())
        } else {
            sid.to_string()
        };
        out.push(mixed(vec![
            ("Session: ".into(), Theme::text_dim(), false),
            (shown, Theme::text_muted(), false),
        ]));
    }

    if !state.intro_tools.is_empty() {
        out.push(Line::raw(""));
        out.push(styled("tools", Theme::brand_orange(), true));
        for (group, names) in state.intro_tools.iter().take(6) {
            let pfx = format!("{group}: ");
            let rest = trunc_items(names, col_w.saturating_sub(pfx.width()).max(8));
            out.push(mixed(vec![
                (pfx, Theme::text_dim(), false),
                (rest, Theme::text_secondary(), false),
            ]));
        }
        if state.intro_tools.len() > 6 {
            out.push(styled(
                format!("+{} more", state.intro_tools.len() - 6),
                Theme::text_dim(),
                false,
            ));
        }
    }

    out.push(Line::raw(""));
    let tools_n = state
        .intro_tools
        .iter()
        .map(|(_, n)| n.len())
        .sum::<usize>();
    let skills_n = state
        .intro_skills
        .iter()
        .map(|(_, n)| n.len())
        .sum::<usize>();
    let tools_s = if state.intro_tools.is_empty() {
        "… tools".to_string()
    } else {
        format!("{tools_n} tools")
    };
    let skills_s = if state.intro_skills.is_empty() {
        "… skills".to_string()
    } else {
        format!("{skills_n} skills")
    };
    let mut summary = format!("{tools_s} · {skills_s}");
    if state.mcp_connected > 0 {
        summary.push_str(&format!(" · {} MCP", state.mcp_connected));
    }
    summary.push_str(" · /help");
    out.push(styled(
        clip(&summary, col_w),
        Theme::text_secondary(),
        false,
    ));

    if let Some(warn) = &state.intro_warning {
        out.push(styled(clip(warn, col_w), Theme::accent_yellow(), false));
    }

    out.push(Line::raw(""));
    out.push(styled(
        "type a message   / for commands",
        Theme::text_dim(),
        false,
    ));
    out
}

fn pad_line(line: Line<'static>, width: usize) -> Line<'static> {
    let w: usize = line.spans.iter().map(|s| s.content.width()).sum();
    if w >= width {
        return line;
    }
    let mut spans = line.spans;
    spans.push(Span::raw(" ".repeat(width - w)));
    Line::from(spans)
}

fn zip_columns(
    left: Vec<Line<'static>>,
    right: Vec<Line<'static>>,
    left_w: usize,
) -> Vec<Line<'static>> {
    let rows = left.len().max(right.len());
    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        let l = left.get(i).cloned().unwrap_or_else(|| Line::raw(""));
        let r = right.get(i).cloned().unwrap_or_else(|| Line::raw(""));
        let mut spans = pad_line(l, left_w).spans;
        spans.extend(r.spans);
        out.push(Line::from(spans));
    }
    out
}

/// Empty-transcript intro: official logo, then a quieter session panel.
pub fn render_intro(
    lines: &mut Vec<Line<'_>>,
    width: usize,
    height: usize,
    state: &AppState,
    frame: u64,
) {
    let mut body: Vec<Line<'static>> = Vec::new();
    body.push(Line::raw(""));
    body.extend(banner(width));
    body.push(Line::raw(""));

    let cad_w = art_width(CADUCEUS_ART);
    let gap = 4;
    let wide = width >= cad_w + 40;
    let info_w = if wide {
        width.saturating_sub(cad_w + gap).clamp(36, 64)
    } else {
        width.saturating_sub(2).max(20)
    };
    let reveal = state.reveal();
    let cad = if wide || width >= cad_w + 2 {
        colorize_shimmer(CADUCEUS_ART, super::motion::frame(frame), "")
    } else {
        Vec::new()
    };

    let panel = if reveal < 0.04 {
        cad
    } else {
        let mut info = info_lines(state, info_w);
        if reveal < 0.995 {
            for line in &mut info {
                crate::ui::theme::fade_spans(&mut line.spans, reveal);
            }
        }
        if wide {
            zip_columns(cad, info, cad_w + gap)
        } else if !cad.is_empty() {
            let mut p = cad;
            p.push(Line::raw(""));
            p.extend(info);
            p
        } else {
            info
        }
    };
    body.extend(center_block(panel, width));

    let top = height.saturating_sub(body.len()) / 3;
    for _ in 0..top {
        lines.push(Line::raw(""));
    }
    lines.extend(body);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;

    fn plain(line: &Line) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn logo_is_uniform_width() {
        let w = art_width(LOGO_ART);
        assert!(w > 90);
        assert!(LOGO_ART.iter().all(|l| l.width() == w));
        assert_eq!(CADUCEUS_ART.len(), 15);
    }

    #[test]
    fn wide_intro_has_logo_and_caduceus() {
        let mut s = AppState::new();
        s.metrics.active_model = "x-ai/grok-4".into();
        s.metrics.hermes_version = "0.20.5".into();
        s.session_id = Some("sess-12345678".into());
        s.intro_tools = vec![(
            "web".into(),
            vec!["web_search".into(), "web_extract".into()],
        )];
        s.intro_skills = vec![("general".into(), vec!["demo".into()])];
        s.mark_session_ready();
        s.reveal_started = Some(std::time::Instant::now() - std::time::Duration::from_secs(2));
        let mut lines: Vec<Line<'_>> = Vec::new();
        render_intro(&mut lines, 120, 40, &s, 0);
        let blob: String = lines.iter().map(plain).collect();
        assert!(blob.contains("██"), "block-letter logo");
        assert!(blob.contains("Messenger of the Digital Gods"));
        assert!(blob.contains("grok-4"));
        assert!(blob.contains("web_search"));
        assert!(blob.contains("2 tools"));
        assert!(blob.contains("1 skills"));
        assert!(blob.contains("⢀⣀⡀"), "caduceus braille");
    }

    #[test]
    fn narrow_intro_skips_block_logo() {
        let s = AppState::new();
        let mut lines: Vec<Line<'_>> = Vec::new();
        render_intro(&mut lines, 50, 24, &s, 0);
        let blob: String = lines.iter().map(plain).collect();
        assert!(!blob.contains("██"));
        assert!(blob.contains("hermes"));
    }

    #[test]
    fn wide_intro_centers_logo() {
        let s = AppState::new();
        let mut lines: Vec<Line<'_>> = Vec::new();
        render_intro(&mut lines, 140, 40, &s, 0);
        let logo = lines
            .iter()
            .map(plain)
            .find(|t| t.contains("██"))
            .expect("logo line");
        let lead = logo.chars().take_while(|c| *c == ' ').count();
        assert!(lead >= 8, "logo should be centered, lead={lead}");
    }

    #[test]
    fn caduceus_shimmer_changes_with_frame() {
        let a = colorize_shimmer(CADUCEUS_ART, 0, "");
        let b = colorize_shimmer(CADUCEUS_ART, 12, "");
        assert_ne!(format!("{a:?}"), format!("{b:?}"));
        let glyphs: String = a
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(glyphs.contains("⢀⣀⡀"));
    }
}
