//! One spacing scale for the transcript.
//!
//! 2-cell gutter, 2-cell nest, at most one blank row between blocks.
//! Headings, body, lists, tools, and thoughts all start on the gutter.

use ratatui::text::Line;

pub const GUTTER: usize = 2;
pub const NEST: usize = 2;
pub const MAX_BLANK: usize = 1;

pub const GUTTER_STR: &str = "  ";
pub const NEST_STR: &str = "    ";
pub const BULLET: &str = "  • ";
pub const QUOTE: &str = "  │ ";
pub const RAIL: &str = "  │ ";
pub const RAIL_CONT: &str = "    ";
pub const FENCE_HEAD: &str = "  ╭─ ";
pub const FENCE_TAIL: &str = "  ╰────────────────────────────────";
pub const RULE: &str = "  ────────";

/// `.` → `..` → `...` for wait / thinking loaders. ~300ms per step at 50ms ticks.
pub fn ellipsis(frame: u64) -> &'static str {
    super::motion::ellipsis_at(frame, super::motion::reduced_motion())
}

pub fn is_blank(line: &Line<'_>) -> bool {
    line.spans.iter().all(|s| s.content.trim().is_empty())
}

/// Collapse stacked blanks to `MAX_BLANK` and drop leading/trailing empties.
pub fn tighten(lines: &mut Vec<Line<'static>>) {
    let mut out = Vec::with_capacity(lines.len());
    let mut blanks = 0usize;
    for line in lines.drain(..) {
        if is_blank(&line) {
            blanks += 1;
            if blanks <= MAX_BLANK && !out.is_empty() {
                out.push(Line::raw(""));
            }
        } else {
            blanks = 0;
            out.push(line);
        }
    }
    while out.last().is_some_and(is_blank) {
        out.pop();
    }
    *lines = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tighten_collapses_and_trims() {
        let mut lines = vec![
            Line::raw(""),
            Line::raw("a"),
            Line::raw(""),
            Line::raw(""),
            Line::raw(""),
            Line::raw("b"),
            Line::raw(""),
        ];
        tighten(&mut lines);
        assert_eq!(lines.len(), 3);
        assert!(!is_blank(&lines[0]));
        assert!(is_blank(&lines[1]));
        assert!(!is_blank(&lines[2]));
    }

    #[test]
    fn ellipsis_cycles() {
        assert_eq!(crate::ui::motion::ellipsis_at(0, false), ".");
        assert_eq!(crate::ui::motion::ellipsis_at(6, false), "..");
        assert_eq!(crate::ui::motion::ellipsis_at(12, false), "...");
        assert_eq!(crate::ui::motion::ellipsis_at(18, false), ".");
    }

    #[test]
    fn prefixes_share_gutter() {
        assert_eq!(GUTTER_STR.len(), GUTTER);
        assert_eq!(NEST_STR.len(), GUTTER + NEST);
        assert_eq!(unicode_width::UnicodeWidthStr::width(BULLET), 4);
        assert_eq!(unicode_width::UnicodeWidthStr::width(RAIL), 4);
    }
}
