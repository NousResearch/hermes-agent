//! Gold-theme canvas wash: slow diagonal gold plus a faint caduceus silhouette.

use ratatui::{
    style::Style,
    text::{Line, Span},
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::brand::CADUCEUS_ART;
use super::theme::Theme;

pub fn active() -> bool {
    crate::ui::theme::current().id == "gold"
}

pub fn animates() -> bool {
    active() && !super::motion::reduced_motion()
}

/// Mix a little antique gold into the canvas. `t` stays under ~12%.
pub fn cell_bg(row: u16, col: u16, frame: u64, caduceus: bool) -> ratatui::style::Color {
    let phase = super::motion::frame(frame) as f32;
    let wave = ((f32::from(row) + f32::from(col) * 0.32) * 0.14 - phase * 0.028).sin() * 0.5 + 0.5;
    let mut t = 0.025 + 0.04 * wave;
    if caduceus {
        let band = (i32::from(row) * 3 + i32::from(col) - (phase as i32 / 2)).rem_euclid(22);
        t += match band {
            0..=2 => 0.07,
            3..=5 => 0.045,
            _ => 0.028,
        };
    }
    super::theme::mix_toward_bg(Theme::brand_gold(), t)
}

fn art_size() -> (usize, usize) {
    let w = CADUCEUS_ART.iter().map(|l| l.width()).max().unwrap_or(0);
    (w, CADUCEUS_ART.len())
}

fn origin(width: usize, height: usize, frame: u64) -> Option<(i32, usize)> {
    let (art_w, art_h) = art_size();
    if art_w == 0 || art_h == 0 || width < art_w || height < art_h {
        return None;
    }
    let bob = if super::motion::reduced_motion() {
        0
    } else {
        (super::motion::frame(frame) / 18 % 3) as i32 - 1
    };
    let top = (height.saturating_sub(art_h) / 2) as i32 + bob;
    let left = width.saturating_sub(art_w) / 2;
    Some((top, left))
}

fn row_on_caduceus(row: u16, width: usize, height: usize, frame: u64) -> bool {
    let Some((top, _)) = origin(width, height, frame) else {
        return false;
    };
    let rr = i32::from(row) - top;
    rr >= 0 && (rr as usize) < CADUCEUS_ART.len()
}

fn caduceus_at(row: u16, col: usize, width: usize, height: usize, frame: u64) -> bool {
    let Some((top, left)) = origin(width, height, frame) else {
        return false;
    };
    let rr = i32::from(row) - top;
    if rr < 0 {
        return false;
    }
    let Some(glyph_line) = CADUCEUS_ART.get(rr as usize) else {
        return false;
    };
    if col < left {
        return false;
    }
    let mut x = left;
    for ch in glyph_line.chars() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if w == 0 {
            continue;
        }
        if col >= x && col < x + w {
            return ch != ' ' && ch != '\u{2800}';
        }
        x += w;
    }
    false
}

fn stamp_bg(line: &mut Line<'static>, bg: ratatui::style::Color) {
    if line.style.bg.is_none() {
        line.style.bg = Some(bg);
    }
    let inherit = line.style.bg.unwrap_or(bg);
    for sp in &mut line.spans {
        if sp.style.bg.is_none() {
            sp.style.bg = Some(inherit);
        }
    }
}

/// Tint transcript rows and pad unused canvas. Gold theme only.
pub fn apply(lines: &mut Vec<Line<'static>>, width: usize, height: usize, frame: u64) {
    if !active() || width == 0 || height == 0 {
        return;
    }
    for (i, line) in lines.iter_mut().enumerate() {
        let row = i as u16;
        if line.style.bg.is_some() {
            // User bubbles keep their surface; still fill any unpainted spans.
            stamp_bg(line, cell_bg(row, 0, frame, false));
            continue;
        }
        if row_on_caduceus(row, width, height, frame) {
            stamp_shaped(line, row, width, height, frame);
        } else {
            stamp_bg(line, cell_bg(row, 0, frame, false));
        }
    }
    while lines.len() < height {
        let row = lines.len() as u16;
        lines.push(watermark_row(row, width, height, frame));
    }
}

fn stamp_shaped(line: &mut Line<'static>, row: u16, width: usize, height: usize, frame: u64) {
    let row_bg = cell_bg(row, 0, frame, false);
    line.style.bg = Some(row_bg);
    let mut col = 0usize;
    let mut out = Vec::with_capacity(line.spans.len() + 4);
    for sp in std::mem::take(&mut line.spans) {
        if sp.style.bg.is_some() {
            col = col.saturating_add(sp.content.width());
            out.push(sp);
            continue;
        }
        let fg = sp.style.fg;
        let mods = sp.style.add_modifier;
        let mut run = String::new();
        let mut run_cad = false;
        let mut started = false;
        for ch in sp.content.chars() {
            let w = UnicodeWidthChar::width(ch).unwrap_or(0);
            let cad = caduceus_at(row, col, width, height, frame);
            if started && cad != run_cad && !run.is_empty() {
                out.push(shaped_span(
                    std::mem::take(&mut run),
                    row,
                    col,
                    run_cad,
                    frame,
                    fg,
                    mods,
                ));
            }
            started = true;
            run_cad = cad;
            run.push(ch);
            col = col.saturating_add(w);
        }
        if !run.is_empty() {
            out.push(shaped_span(run, row, col, run_cad, frame, fg, mods));
        }
    }
    if col < width {
        out.push(pad_span(row, col, width, height, frame));
    }
    line.spans = out;
}

fn shaped_span(
    text: String,
    row: u16,
    col_end: usize,
    cad: bool,
    frame: u64,
    fg: Option<ratatui::style::Color>,
    mods: ratatui::style::Modifier,
) -> Span<'static> {
    let col = col_end.saturating_sub(text.width()) as u16;
    let mut style = Style::default().bg(cell_bg(row, col, frame, cad));
    if let Some(fg) = fg {
        style = style.fg(fg);
    }
    style = style.add_modifier(mods);
    Span::styled(text, style)
}

fn pad_span(row: u16, from: usize, width: usize, height: usize, frame: u64) -> Span<'static> {
    let cad = (from..width).any(|c| caduceus_at(row, c, width, height, frame));
    Span::styled(
        " ".repeat(width - from),
        Style::default().bg(cell_bg(row, from as u16, frame, cad)),
    )
}

fn blank_row(width: usize, bg: ratatui::style::Color) -> Line<'static> {
    Line::from(Span::styled(" ".repeat(width), Style::default().bg(bg)))
        .style(Style::default().bg(bg))
}

fn watermark_row(row: u16, width: usize, height: usize, frame: u64) -> Line<'static> {
    let bg = cell_bg(row, 0, frame, false);
    let Some((top, left)) = origin(width, height, frame) else {
        return blank_row(width, bg);
    };
    let Ok(art_row) = usize::try_from(i32::from(row) - top) else {
        return blank_row(width, bg);
    };
    let Some(glyph_line) = CADUCEUS_ART.get(art_row) else {
        return blank_row(width, bg);
    };
    let fg = super::theme::mix_toward_bg(Theme::brand_gold(), 0.16);
    let mut spans = vec![Span::styled(" ".repeat(left), Style::default().bg(bg))];
    let mut run = String::new();
    let mut lit = false;
    let mut col = left;
    let flush = |spans: &mut Vec<Span<'static>>, run: &mut String, lit: bool, col: usize| {
        if run.is_empty() {
            return;
        }
        let text = std::mem::take(run);
        let style = if lit {
            Style::default()
                .fg(fg)
                .bg(cell_bg(row, col as u16, frame, true))
        } else {
            Style::default().bg(cell_bg(row, col as u16, frame, false))
        };
        spans.push(Span::styled(text, style));
    };
    for ch in glyph_line.chars() {
        let on = ch != ' ' && ch != '\u{2800}';
        if on != lit && !run.is_empty() {
            let start = col.saturating_sub(run.width());
            flush(&mut spans, &mut run, lit, start);
        }
        lit = on;
        run.push(if on { ch } else { ' ' });
        col += UnicodeWidthChar::width(ch).unwrap_or(0);
    }
    let start = col.saturating_sub(run.width());
    flush(&mut spans, &mut run, lit, start);
    if col < width {
        spans.push(Span::styled(
            " ".repeat(width - col),
            Style::default().bg(cell_bg(row, col as u16, frame, false)),
        ));
    }
    Line::from(spans).style(Style::default().bg(bg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::Palette;
    use crate::ui::theme;

    #[test]
    fn gold_only() {
        theme::apply(Palette::midnight());
        assert!(!active());
        theme::apply(Palette::gold());
        assert!(active());
        let c = cell_bg(4, 8, 0, false);
        assert_ne!(c, Theme::brand_gold());
        assert_ne!(c, Theme::bg_base());
    }

    #[test]
    fn caduceus_tints_stronger() {
        theme::apply(Palette::gold());
        let off = cell_bg(8, 20, 0, false);
        let on = cell_bg(8, 20, 0, true);
        assert_ne!(off, on);
    }

    #[test]
    fn pads_and_stamps() {
        theme::apply(Palette::gold());
        let mut lines = vec![Line::raw("hi")];
        apply(&mut lines, 40, 6, 0);
        assert_eq!(lines.len(), 6);
        assert!(lines[0].style.bg.is_some());
        assert!(lines[5].style.bg.is_some());
        let mut tall = Vec::new();
        apply(&mut tall, 48, 24, 0);
        let lit = tall.iter().any(|l| {
            l.spans
                .iter()
                .any(|sp| sp.style.fg == Some(theme::mix_toward_bg(Theme::brand_gold(), 0.16)))
        });
        assert!(
            lit,
            "caduceus watermark should paint on a tall empty canvas"
        );
    }

    #[test]
    fn skips_other_themes() {
        theme::apply(Palette::hermes());
        let mut lines = vec![Line::raw("hi")];
        apply(&mut lines, 20, 4, 3);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].style.bg.is_none());
        theme::apply(Palette::gold());
    }

    #[test]
    fn keeps_user_surface() {
        theme::apply(Palette::gold());
        let surface = Theme::bg_surface();
        let mut lines = vec![
            Line::from(Span::styled("you", Style::default().bg(surface)))
                .style(Style::default().bg(surface)),
        ];
        apply(&mut lines, 24, 4, 0);
        assert_eq!(lines[0].style.bg, Some(surface));
        assert_eq!(lines[0].spans[0].style.bg, Some(surface));
    }
}
