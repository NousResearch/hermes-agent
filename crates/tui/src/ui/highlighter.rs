use std::sync::OnceLock;

use ratatui::style::{Color, Style};
use ratatui::text::Span;
use syntect::easy::HighlightLines;
use syntect::highlighting::ThemeSet;
use syntect::parsing::{SyntaxReference, SyntaxSet};

use super::theme::Theme;

pub struct CodeHighlighter {
    inner: OnceLock<(SyntaxSet, ThemeSet)>,
}

impl CodeHighlighter {
    pub fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    fn sets(&self) -> &(SyntaxSet, ThemeSet) {
        self.inner.get_or_init(|| {
            (
                SyntaxSet::load_defaults_newlines(),
                ThemeSet::load_defaults(),
            )
        })
    }

    fn syntax_for<'a>(
        &'a self,
        extension: &str,
    ) -> Option<(
        &'a SyntaxSet,
        &'a SyntaxReference,
        &'a syntect::highlighting::Theme,
    )> {
        let (ps, ts) = self.sets();
        let syntax = ps
            .find_syntax_by_token(extension)
            .or_else(|| ps.find_syntax_by_extension(extension))
            .unwrap_or_else(|| ps.find_syntax_plain_text());
        let theme = ts
            .themes
            .get("base16-mocha.dark")
            .or_else(|| ts.themes.get("Solarized (dark)"))
            .or_else(|| ts.themes.get("base16-ocean.dark"))
            .or_else(|| ts.themes.values().next())?;
        Some((ps, syntax, theme))
    }

    pub fn start_block<'a>(&'a self, extension: &str) -> Option<HighlightLines<'a>> {
        if extension.is_empty() {
            return None;
        }
        let (_, syntax, theme) = self.syntax_for(extension)?;
        Some(HighlightLines::new(syntax, theme))
    }

    pub fn paint(&self, h: &mut HighlightLines<'_>, line: &str) -> Vec<Span<'static>> {
        if line.is_empty() || line.len() > 400 {
            return vec![Span::raw(line.to_string())];
        }
        let (ps, _) = self.sets();
        match h.highlight_line(line, ps) {
            Ok(ranges) => ranges
                .into_iter()
                .map(|(style, text)| {
                    Span::styled(
                        text.to_string(),
                        Style::default().fg(syntax_token(
                            style.foreground.r,
                            style.foreground.g,
                            style.foreground.b,
                        )),
                    )
                })
                .collect(),
            Err(_) => vec![Span::raw(line.to_string())],
        }
    }
}

/// Snap syntect's RGB onto the active Caduceus / Omarchy tokens.
pub fn syntax_token(r: u8, g: u8, b: u8) -> Color {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    if max.saturating_sub(min) < 32 {
        return if max > 180 {
            Theme::text_primary()
        } else {
            Theme::text_muted()
        };
    }
    if r >= g && r >= b {
        if g.saturating_sub(b) >= 25 {
            Theme::brand_gold()
        } else if b.saturating_sub(g) >= 25 {
            Theme::brand_orange()
        } else {
            Theme::accent_red()
        }
    } else if g >= r && g >= b {
        Theme::accent_green()
    } else {
        Theme::text_secondary()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::Palette;
    use crate::ui::theme::{apply, current};

    fn is_palette_fg(c: Color) -> bool {
        let p = current();
        [
            p.brand_gold,
            p.brand_orange,
            p.accent_green,
            p.accent_red,
            p.accent_yellow,
            p.text_primary,
            p.text_secondary,
            p.text_muted,
            p.text_dim,
        ]
        .contains(&c)
    }

    #[test]
    fn syntax_token_stays_on_palette() {
        apply(Palette::gold());
        assert_eq!(syntax_token(80, 80, 80), Theme::text_muted());
        assert_eq!(syntax_token(220, 180, 80), Theme::brand_gold());
        assert_eq!(syntax_token(80, 180, 90), Theme::accent_green());
        apply(Palette::hermes());
        assert_eq!(syntax_token(80, 180, 90), Theme::accent_green());
        assert_ne!(Theme::accent_green(), Palette::gold().accent_green);
    }

    #[test]
    fn paint_uses_theme_tokens() {
        apply(Palette::gold());
        let hl = CodeHighlighter::new();
        let mut h = hl.start_block("rs").expect("rust syntax");
        let spans = hl.paint(&mut h, r#"let x = "hi";"#);
        assert!(!spans.is_empty());
        for sp in &spans {
            if let Some(fg) = sp.style.fg {
                assert!(is_palette_fg(fg), "off-palette {fg:?} for {:?}", sp.content);
            }
        }
    }
}
