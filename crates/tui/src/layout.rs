use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// First visible line when the user has scrolled `from_bottom` lines off the tail.
pub fn scroll_y(total_lines: usize, viewport: usize, from_bottom: usize) -> u16 {
    let max = total_lines.saturating_sub(viewport);
    let fb = clamp_from_bottom(total_lines, viewport, from_bottom);
    (max - fb) as u16
}

pub fn clamp_from_bottom(total_lines: usize, viewport: usize, from_bottom: usize) -> usize {
    let max = total_lines.saturating_sub(viewport);
    from_bottom.min(max)
}

/// Greedy wrap on Unicode display width. Does not split combining sequences
/// specially; good enough for the transcript column.
pub fn wrap_chunks(s: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![s.to_string()];
    }
    if s.is_empty() {
        return vec![String::new()];
    }
    if s.width() <= width {
        return vec![s.to_string()];
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut w = 0usize;
    for ch in s.chars() {
        let cw = UnicodeWidthChar::width(ch).unwrap_or(1).max(1);
        if w + cw > width && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
            w = 0;
        }
        cur.push(ch);
        w += cw;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_is_max_scroll() {
        assert_eq!(scroll_y(100, 20, 0), 80);
        assert_eq!(scroll_y(10, 20, 0), 0);
        assert_eq!(scroll_y(100, 20, 10), 70);
        assert_eq!(scroll_y(100, 20, 999), 0);
    }

    #[test]
    fn wrap_ascii() {
        assert_eq!(wrap_chunks("hello", 10), vec!["hello"]);
        assert_eq!(wrap_chunks("hello world", 5), vec!["hello", " worl", "d"]);
    }

    #[test]
    fn wrap_empty() {
        assert_eq!(wrap_chunks("", 8), vec![""]);
    }
}
