use ratatui::style::Color;
use ratatui::text::Span;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

pub use crate::palette::{catalog, lookup, Palette};

static PALETTE: Mutex<Palette> = Mutex::new(Palette::gold());
static EPOCH: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static SNAP: Cell<Palette> = const { Cell::new(Palette::gold()) };
}

pub fn apply(p: Palette) {
    if let Ok(mut g) = PALETTE.lock() {
        *g = p;
    }
    SNAP.with(|c| c.set(p));
    EPOCH.fetch_add(1, Ordering::Relaxed);
}

pub fn epoch() -> u64 {
    EPOCH.load(Ordering::Relaxed)
}

pub fn current() -> Palette {
    SNAP.with(|c| c.get())
}

/// Copy the mutex palette onto this thread. Call once per frame from the UI task.
pub fn snapshot() {
    let p = PALETTE
        .lock()
        .map(|g| *g)
        .unwrap_or_else(|e| *e.into_inner());
    SNAP.with(|c| c.set(p));
}

/// Mix `c` toward the active canvas. `t=0` is invisible, `t=1` is full.
pub fn mix_toward_bg(c: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let Color::Rgb(r, g, b) = c else {
        return c;
    };
    let Color::Rgb(br, bg, bb) = current().bg_base else {
        return c;
    };
    Color::Rgb(
        (f32::from(br) + (f32::from(r) - f32::from(br)) * t) as u8,
        (f32::from(bg) + (f32::from(g) - f32::from(bg)) * t) as u8,
        (f32::from(bb) + (f32::from(b) - f32::from(bb)) * t) as u8,
    )
}

pub fn hover_paint(idle: ratatui::style::Style, hot: bool) -> ratatui::style::Style {
    use ratatui::style::Modifier;
    if hot {
        idle.fg(Theme::text_primary())
            .bg(Theme::bg_highlight())
            .add_modifier(Modifier::BOLD)
    } else {
        idle
    }
}

pub fn hover_line(line: &mut ratatui::text::Line<'_>, hot: bool) {
    if !hot {
        return;
    }
    for sp in &mut line.spans {
        sp.style.bg = Some(Theme::bg_highlight());
        let dim = matches!(
            sp.style.fg,
            Some(c) if c == Theme::text_dim()
                || c == Theme::text_muted()
                || c == Theme::text_secondary()
        );
        if sp.style.fg.is_none() || dim {
            sp.style.fg = Some(Theme::text_primary());
        }
    }
}

pub fn fade_spans(spans: &mut [Span<'_>], t: f32) {
    if t >= 0.995 {
        return;
    }
    for span in spans {
        if let Some(fg) = span.style.fg {
            span.style.fg = Some(mix_toward_bg(fg, t));
        }
        if let Some(bg) = span.style.bg {
            if bg != current().bg_base {
                span.style.bg = Some(mix_toward_bg(bg, t));
            }
        }
    }
}

/// Color tokens. Call as functions so a `/theme` switch repaints the next frame.
pub struct Theme;

impl Theme {
    pub fn bg_base() -> Color {
        current().bg_base
    }
    pub fn bg_surface() -> Color {
        current().bg_surface
    }
    pub fn bg_header() -> Color {
        current().bg_header
    }
    pub fn bg_popup() -> Color {
        current().bg_popup
    }
    pub fn bg_highlight() -> Color {
        current().bg_highlight
    }
    pub fn border_subtle() -> Color {
        current().border_subtle
    }
    pub fn border_focus() -> Color {
        current().border_focus
    }
    pub fn brand_gold() -> Color {
        current().brand_gold
    }
    pub fn brand_orange() -> Color {
        current().brand_orange
    }
    pub fn accent_cyan() -> Color {
        current().text_secondary
    }
    pub fn accent_purple() -> Color {
        current().brand_orange
    }
    pub fn accent_green() -> Color {
        current().accent_green
    }
    pub fn accent_red() -> Color {
        current().accent_red
    }
    pub fn accent_yellow() -> Color {
        current().accent_yellow
    }
    pub fn text_primary() -> Color {
        current().text_primary
    }
    pub fn text_secondary() -> Color {
        current().text_secondary
    }
    pub fn text_muted() -> Color {
        current().text_muted
    }
    pub fn text_dim() -> Color {
        current().text_dim
    }

    pub fn is_light() -> bool {
        match current().bg_base {
            Color::Rgb(r, g, b) => u16::from(r) + u16::from(g) + u16::from(b) > 480,
            _ => false,
        }
    }

    /// Inline code / links — cool accent, distinct from gold headings.
    pub fn code() -> Color {
        if Self::is_light() {
            current().brand_gold
        } else {
            match current().accent_green {
                Color::Rgb(r, g, b) => Color::Rgb(r.min(170), g.max(168), b.max(168)),
                c => c,
            }
        }
    }

    pub fn link() -> Color {
        Self::code()
    }
}

pub fn persist_path(hermes_home: &std::path::Path) -> std::path::PathBuf {
    hermes_home.join("tui-theme")
}

pub fn load_saved(hermes_home: &std::path::Path) -> Palette {
    let path = persist_path(hermes_home);
    let id = std::fs::read_to_string(&path).unwrap_or_default();
    lookup(&id)
}

pub fn save(hermes_home: &std::path::Path, id: &str) {
    let _ = std::fs::create_dir_all(hermes_home);
    let _ = std::fs::write(persist_path(hermes_home), id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_hermes_and_aliases() {
        assert_eq!(lookup("hermes").id, "hermes");
        assert_eq!(lookup("HERMES SITE").id, "hermes");
        assert_eq!(lookup("nope").id, "gold");
        assert_eq!(lookup("tokyo-night").id, "tokyo");
    }

    #[test]
    fn apply_switches_tokens() {
        apply(Palette::hermes());
        assert_eq!(Theme::bg_base(), Palette::hermes().bg_base);
        apply(Palette::gold());
        assert_eq!(Theme::brand_gold(), Palette::gold().brand_gold);
        apply(Palette::gold());
    }

    #[test]
    fn mix_hides_at_zero() {
        apply(Palette::gold());
        let c = mix_toward_bg(Palette::gold().brand_gold, 0.0);
        assert_eq!(c, Palette::gold().bg_base);
        let full = mix_toward_bg(Palette::gold().brand_gold, 1.0);
        assert_eq!(full, Palette::gold().brand_gold);
    }

    #[test]
    fn catalog_has_website_and_omarchy() {
        let ids: Vec<_> = catalog().iter().map(|p| p.id).collect();
        assert!(ids.contains(&"hermes"));
        assert!(ids.contains(&"gold"));
        assert!(ids.contains(&"tokyo"));
        assert!(ids.contains(&"mocha"));
        assert_eq!(ids.len(), crate::palette::THEME_COUNT);
    }
}
