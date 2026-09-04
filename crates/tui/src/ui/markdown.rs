//! Grok-Build-style markdown for assistant output: headings, tables, inline
//! marks, lists, rules, quotes, fenced code, and image cards.

use std::io::Read;
use std::path::Path;

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use super::highlighter::CodeHighlighter;
use super::rhythm::{self, BULLET, FENCE_HEAD, FENCE_TAIL, GUTTER_STR, QUOTE, RAIL, RULE};
use super::theme::Theme;
use crate::layout::wrap_chunks;

pub fn render(
    out: &mut Vec<Line<'static>>,
    text: &str,
    col_width: usize,
    highlighter: &CodeHighlighter,
) {
    let raw: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < raw.len() {
        let trim = raw[i].trim();
        if trim.starts_with("```") {
            i = emit_fence(out, &raw, i, col_width, highlighter);
            continue;
        }
        if is_hr(trim) {
            out.push(Line::from(Span::styled(
                RULE,
                Style::default().fg(Theme::text_dim()),
            )));
            i += 1;
            continue;
        }
        if looks_like_table(&raw, i) {
            i = emit_table(out, &raw, i, col_width);
            continue;
        }
        if let Some((alt, url)) = sole_image(trim) {
            out.extend(image_card(alt, url, col_width));
            i += 1;
            continue;
        }
        if let Some(rest) = heading_text(trim) {
            emit_heading(out, rest, col_width);
            i += 1;
            continue;
        }
        if is_whole_line_bold(trim) {
            let inner = trim.trim_matches('*').trim_matches('_').trim();
            emit_heading(out, inner, col_width);
            i += 1;
            continue;
        }
        if is_list_item(trim) {
            i = emit_list(out, &raw, i, col_width);
            continue;
        }
        if trim.starts_with('>') {
            i = emit_quote(out, &raw, i, col_width);
            continue;
        }
        if trim.is_empty() {
            if out.last().is_some_and(|l| !rhythm::is_blank(l)) {
                out.push(Line::raw(""));
            }
            i += 1;
            continue;
        }
        emit_text_line(out, raw[i], GUTTER_STR, col_width, body_style());
        i += 1;
    }
    rhythm::tighten(out);
}

fn body_style() -> Style {
    Style::default().fg(Theme::text_primary())
}

fn heading_style() -> Style {
    Style::default()
        .fg(Theme::brand_gold())
        .add_modifier(Modifier::BOLD)
}

fn code_style() -> Style {
    Style::default().fg(Theme::code())
}

fn link_style() -> Style {
    Style::default()
        .fg(Theme::link())
        .add_modifier(Modifier::UNDERLINED)
}

fn heading_text(trimmed: &str) -> Option<&str> {
    trimmed
        .strip_prefix("#### ")
        .or_else(|| trimmed.strip_prefix("### "))
        .or_else(|| trimmed.strip_prefix("## "))
        .or_else(|| trimmed.strip_prefix("# "))
}

fn is_whole_line_bold(trim: &str) -> bool {
    let t = trim;
    (t.starts_with("**") && t.ends_with("**") && t.len() > 4 && !t[2..t.len() - 2].contains("**"))
        || (t.starts_with("__") && t.ends_with("__") && t.len() > 4)
}

fn is_hr(trim: &str) -> bool {
    let t = trim.replace(' ', "");
    t.len() >= 3
        && (t.chars().all(|c| c == '-')
            || t.chars().all(|c| c == '*')
            || t.chars().all(|c| c == '_'))
}

fn is_list_item(trim: &str) -> bool {
    trim.starts_with("- ")
        || trim.starts_with("* ")
        || trim.starts_with("+ ")
        || ordered_marker(trim).is_some()
}

fn ordered_marker(trim: &str) -> Option<usize> {
    let digits = trim.bytes().take_while(u8::is_ascii_digit).count();
    if digits > 0 && digits < 4 && trim[digits..].starts_with(". ") {
        Some(digits)
    } else {
        None
    }
}

fn looks_like_table(raw: &[&str], i: usize) -> bool {
    raw.get(i).is_some_and(|l| l.contains('|'))
        && raw.get(i + 1).is_some_and(|l| is_table_sep(l.trim()))
}

fn is_table_sep(trim: &str) -> bool {
    let t = trim.trim_matches('|');
    if t.is_empty() || !t.contains('-') {
        return false;
    }
    t.chars()
        .all(|c| c == '-' || c == ':' || c == '|' || c.is_whitespace())
        && t.chars().filter(|c| *c == '-').count() >= 3
}

fn parse_row(line: &str) -> Vec<String> {
    let mut cells: Vec<&str> = line.split('|').collect();
    if line.trim_start().starts_with('|') && !cells.is_empty() {
        cells.remove(0);
    }
    if line.trim_end().ends_with('|') && !cells.is_empty() {
        cells.pop();
    }
    cells.into_iter().map(|c| c.trim().to_string()).collect()
}

fn emit_heading(out: &mut Vec<Line<'static>>, text: &str, col_width: usize) {
    if out.last().is_some_and(|l| !rhythm::is_blank(l)) {
        out.push(Line::raw(""));
    }
    emit_text_line(out, text, GUTTER_STR, col_width, heading_style());
}

fn emit_text_line(
    out: &mut Vec<Line<'static>>,
    text: &str,
    prefix: &'static str,
    col_width: usize,
    base: Style,
) {
    let inner = col_width.saturating_sub(prefix.width()).max(8);
    let spans = parse_inline(text.trim_end(), base);
    for row in wrap_spans(&spans, inner) {
        let mut line = vec![Span::raw(prefix)];
        line.extend(row);
        out.push(Line::from(line));
    }
}

fn emit_fence(
    out: &mut Vec<Line<'static>>,
    raw: &[&str],
    start: usize,
    col_width: usize,
    highlighter: &CodeHighlighter,
) -> usize {
    let lang = raw[start].trim().trim_start_matches('`').trim();
    let label = if lang.is_empty() { "code" } else { lang };
    out.push(Line::from(vec![
        Span::styled(FENCE_HEAD, Style::default().fg(Theme::border_subtle())),
        Span::styled(
            label.to_string(),
            Style::default()
                .fg(Theme::brand_gold())
                .add_modifier(Modifier::BOLD),
        ),
    ]));
    let mut hl = highlighter.start_block(lang);
    let inner = col_width
        .saturating_sub(rhythm::GUTTER + rhythm::NEST)
        .max(8);
    let mut i = start + 1;
    while i < raw.len() && !raw[i].trim_start().starts_with("```") {
        for chunk in wrap_chunks(raw[i], inner) {
            let painted = if let Some(h) = hl.as_mut() {
                highlighter.paint(h, &chunk)
            } else {
                vec![Span::styled(chunk, code_style())]
            };
            let mut spans = vec![Span::styled(
                RAIL,
                Style::default().fg(Theme::border_subtle()),
            )];
            spans.extend(painted);
            out.push(Line::from(spans));
        }
        i += 1;
    }
    out.push(Line::from(Span::styled(
        FENCE_TAIL,
        Style::default().fg(Theme::border_subtle()),
    )));
    if i < raw.len() {
        i + 1
    } else {
        i
    }
}

fn emit_list(out: &mut Vec<Line<'static>>, raw: &[&str], start: usize, col_width: usize) -> usize {
    let mut i = start;
    while i < raw.len() {
        let trim = raw[i].trim();
        if trim.is_empty() {
            if raw.get(i + 1).is_some_and(|n| is_list_item(n.trim())) {
                i += 1;
                continue;
            }
            break;
        }
        if !is_list_item(trim) {
            break;
        }
        let (mark, rest) = if let Some(n) = ordered_marker(trim) {
            (format!("  {}. ", &trim[..n]), trim[n + 2..].to_string())
        } else if let Some(r) = trim.strip_prefix("- [ ] ") {
            ("  ☐ ".into(), r.to_string())
        } else if let Some(r) = trim
            .strip_prefix("- [x] ")
            .or_else(|| trim.strip_prefix("- [X] "))
        {
            ("  ☑ ".into(), r.to_string())
        } else {
            let body = trim
                .strip_prefix("- ")
                .or_else(|| trim.strip_prefix("* "))
                .or_else(|| trim.strip_prefix("+ "))
                .unwrap_or(trim);
            (BULLET.into(), body.to_string())
        };
        let inner = col_width.saturating_sub(mark.width()).max(8);
        let spans = parse_inline(&rest, body_style());
        for (n, row) in wrap_spans(&spans, inner).into_iter().enumerate() {
            let mut line = vec![Span::styled(
                if n == 0 {
                    mark.clone()
                } else {
                    " ".repeat(mark.width())
                },
                Style::default().fg(Theme::text_secondary()),
            )];
            line.extend(row);
            out.push(Line::from(line));
        }
        i += 1;
    }
    i
}

fn emit_quote(out: &mut Vec<Line<'static>>, raw: &[&str], start: usize, col_width: usize) -> usize {
    let mut i = start;
    let style = Style::default()
        .fg(Theme::text_muted())
        .add_modifier(Modifier::ITALIC);
    while i < raw.len() {
        let trim = raw[i].trim();
        if !trim.starts_with('>') {
            break;
        }
        let body = trim.trim_start_matches('>').trim();
        emit_text_line(out, body, QUOTE, col_width, style);
        i += 1;
    }
    i
}

fn emit_table(out: &mut Vec<Line<'static>>, raw: &[&str], start: usize, col_width: usize) -> usize {
    let header = parse_row(raw[start]);
    let mut rows: Vec<Vec<String>> = vec![header];
    let mut i = start + 2;
    while i < raw.len() && raw[i].contains('|') && !is_hr(raw[i].trim()) {
        if raw[i].trim().is_empty() {
            break;
        }
        if !raw[i].contains('|') {
            break;
        }
        rows.push(parse_row(raw[i]));
        i += 1;
    }
    let cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if cols == 0 {
        return start + 1;
    }
    for row in &mut rows {
        row.resize(cols, String::new());
    }
    let widths = column_widths(&rows, col_width);
    let border = Style::default().fg(Theme::border_subtle());
    out.push(rule_line(&widths, '┌', '┬', '┐', border));
    for (r, row) in rows.iter().enumerate() {
        let wrapped: Vec<Vec<Vec<Span<'static>>>> = row
            .iter()
            .enumerate()
            .map(|(c, cell)| {
                let base = if r == 0 {
                    Style::default()
                        .fg(Theme::text_primary())
                        .add_modifier(Modifier::BOLD)
                } else {
                    body_style()
                };
                wrap_spans(&parse_inline(cell, base), widths[c].max(1))
            })
            .collect();
        let height = wrapped.iter().map(|w| w.len().max(1)).max().unwrap_or(1);
        for y in 0..height {
            let mut spans = vec![Span::styled(format!("{GUTTER_STR}│"), border)];
            for (c, cell_lines) in wrapped.iter().enumerate() {
                let empty: Vec<Span<'static>> = Vec::new();
                let piece = cell_lines.get(y).unwrap_or(&empty);
                let used: usize = piece.iter().map(|s| s.content.width()).sum();
                let pad = widths[c].saturating_sub(used);
                spans.push(Span::raw(" "));
                spans.extend(piece.iter().cloned());
                spans.push(Span::raw(" ".repeat(pad + 1)));
                spans.push(Span::styled("│", border));
            }
            out.push(Line::from(spans));
        }
        if r == 0 {
            out.push(rule_line(&widths, '├', '┼', '┤', border));
        }
    }
    out.push(rule_line(&widths, '└', '┴', '┘', border));
    i
}

fn column_widths(rows: &[Vec<String>], col_width: usize) -> Vec<usize> {
    let cols = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut widths = vec![3usize; cols];
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            let w = plain_width(cell).max(1);
            widths[i] = widths[i].max(w.min(48));
        }
    }
    // "  ┌" + n*("─"*w + "┬") ~ indent 2 + 1 + n + sum(w) + 2n padding
    let chrome = 3 + cols * 3;
    let budget = col_width.saturating_sub(chrome).max(cols);
    let mut total: usize = widths.iter().sum();
    while total > budget {
        if let Some((i, _)) = widths.iter().enumerate().max_by_key(|(_, w)| **w) {
            if widths[i] <= 4 {
                break;
            }
            widths[i] -= 1;
            total -= 1;
        } else {
            break;
        }
    }
    widths
}

fn rule_line(widths: &[usize], left: char, mid: char, right: char, border: Style) -> Line<'static> {
    let mut s = String::from(GUTTER_STR);
    s.push(left);
    for (i, w) in widths.iter().enumerate() {
        s.push_str(&"─".repeat(w + 2));
        if i + 1 < widths.len() {
            s.push(mid);
        }
    }
    s.push(right);
    Line::from(Span::styled(s, border))
}

fn plain_width(md: &str) -> usize {
    parse_inline(md, Style::default())
        .iter()
        .map(|s| s.content.width())
        .sum()
}

fn sole_image(trim: &str) -> Option<(&str, &str)> {
    let t = trim.trim();
    if !t.starts_with("![") {
        return None;
    }
    let alt_end = t.find("](")?;
    let url_end = t.rfind(')')?;
    if url_end != t.len() - 1 {
        return None;
    }
    let alt = &t[2..alt_end];
    let url = &t[alt_end + 2..url_end];
    Some((alt, url))
}

pub fn image_card(alt: &str, url: &str, col_width: usize) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let border = Style::default().fg(Theme::border_subtle());
    out.push(Line::from(vec![
        Span::styled(FENCE_HEAD, border),
        Span::styled(
            "image",
            Style::default()
                .fg(Theme::brand_gold())
                .add_modifier(Modifier::BOLD),
        ),
    ]));
    let inner = col_width
        .saturating_sub(rhythm::GUTTER + rhythm::NEST)
        .max(8);
    if !alt.is_empty() && alt != url {
        for chunk in wrap_chunks(alt, inner) {
            out.push(Line::from(vec![
                Span::styled(RAIL, border),
                Span::styled(chunk, Style::default().fg(Theme::text_primary())),
            ]));
        }
    }
    let leaf = url.rsplit(['/', '\\']).next().unwrap_or(url);
    let mut meta = leaf.to_string();
    if let Some((w, h)) = image_dims(Path::new(url)) {
        meta = format!("{leaf} · {w}×{h}");
    }
    out.push(Line::from(vec![
        Span::styled(RAIL, border),
        Span::styled(meta, code_style()),
    ]));
    for row in image_thumb_lines(Path::new(url), inner as u16, 6) {
        let mut spans = vec![Span::styled(RAIL, border)];
        spans.extend(row.spans);
        out.push(Line::from(spans));
    }
    out.push(Line::from(Span::styled(FENCE_TAIL, border)));
    out
}

/// Half-block thumbnail. Empty when the path is not a local image.
pub fn image_thumb_lines(path: &Path, cols: u16, rows: u16) -> Vec<Line<'static>> {
    if cols < 4 || rows == 0 {
        return Vec::new();
    }
    let Ok(meta) = std::fs::metadata(path) else {
        return Vec::new();
    };
    if meta.len() > 4 * 1024 * 1024 {
        return Vec::new();
    }
    let Ok(img) = image::open(path) else {
        return Vec::new();
    };
    let w = cols as u32;
    let h = (rows as u32).saturating_mul(2).max(2);
    let thumb = img.thumbnail(w, h).into_rgb8();
    let tw = thumb.width();
    let th = thumb.height();
    let mut lines = Vec::new();
    let mut y = 0u32;
    while y < th {
        let mut spans = Vec::new();
        for x in 0..tw {
            let top = thumb.get_pixel(x, y).0;
            let bot = if y + 1 < th {
                thumb.get_pixel(x, y + 1).0
            } else {
                [0, 0, 0]
            };
            spans.push(Span::styled(
                "▀",
                Style::default()
                    .fg(ratatui::style::Color::Rgb(top[0], top[1], top[2]))
                    .bg(ratatui::style::Color::Rgb(bot[0], bot[1], bot[2])),
            ));
        }
        lines.push(Line::from(spans));
        y += 2;
    }
    lines
}

pub fn image_dims(path: &Path) -> Option<(u32, u32)> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf = [0u8; 32];
    let n = f.read(&mut buf).ok()?;
    if n < 24 {
        return None;
    }
    if buf.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        let w = u32::from_be_bytes(buf[16..20].try_into().ok()?);
        let h = u32::from_be_bytes(buf[20..24].try_into().ok()?);
        return Some((w, h));
    }
    if buf.starts_with(b"GIF8") {
        let w = u16::from_le_bytes([buf[6], buf[7]]) as u32;
        let h = u16::from_le_bytes([buf[8], buf[9]]) as u32;
        return Some((w, h));
    }
    if buf.starts_with(&[0xFF, 0xD8]) {
        return jpeg_dims(path);
    }
    None
}

fn jpeg_dims(path: &Path) -> Option<(u32, u32)> {
    let data = std::fs::read(path).ok()?;
    let mut i = 2usize;
    while i + 9 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = data[i + 1];
        if marker == 0xD8 || marker == 0xD9 {
            i += 2;
            continue;
        }
        if i + 4 > data.len() {
            break;
        }
        let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
        if matches!(marker, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF)
            && i + 9 < data.len()
        {
            let h = u16::from_be_bytes([data[i + 5], data[i + 6]]) as u32;
            let w = u16::from_be_bytes([data[i + 7], data[i + 8]]) as u32;
            return Some((w, h));
        }
        i += 2 + len;
    }
    None
}

pub fn parse_inline(input: &str, base: Style) -> Vec<Span<'static>> {
    let chars: Vec<char> = input.chars().collect();
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut i = 0;
    let flush = |buf: &mut String, out: &mut Vec<Span<'static>>, style: Style| {
        if !buf.is_empty() {
            out.push(Span::styled(std::mem::take(buf), style));
        }
    };
    while i < chars.len() {
        if chars[i] == '`' {
            if let Some(end) = find_close(&chars, i + 1, &['`']) {
                flush(&mut buf, &mut out, base);
                let code: String = chars[i + 1..end].iter().collect();
                out.push(Span::styled(code, code_style()));
                i = end + 1;
                continue;
            }
        }
        if chars[i] == '[' {
            if let Some((label, url, end)) = parse_link(&chars, i) {
                flush(&mut buf, &mut out, base);
                out.push(Span::styled(label, link_style()));
                let _ = url;
                i = end;
                continue;
            }
        }
        if chars[i] == '*' && chars.get(i + 1) == Some(&'*') {
            if let Some(end) = find_close(&chars, i + 2, &['*', '*']) {
                flush(&mut buf, &mut out, base);
                let inner: String = chars[i + 2..end].iter().collect();
                out.extend(parse_inline(&inner, base.add_modifier(Modifier::BOLD)));
                i = end + 2;
                continue;
            }
        }
        if chars[i] == '*' {
            if let Some(end) = find_single(&chars, i + 1, '*') {
                flush(&mut buf, &mut out, base);
                let inner: String = chars[i + 1..end].iter().collect();
                out.extend(parse_inline(&inner, base.add_modifier(Modifier::ITALIC)));
                i = end + 1;
                continue;
            }
        }
        if chars[i] == '~' && chars.get(i + 1) == Some(&'~') {
            if let Some(end) = find_close(&chars, i + 2, &['~', '~']) {
                flush(&mut buf, &mut out, base);
                let inner: String = chars[i + 2..end].iter().collect();
                out.push(Span::styled(
                    inner,
                    base.add_modifier(Modifier::CROSSED_OUT),
                ));
                i = end + 2;
                continue;
            }
        }
        buf.push(chars[i]);
        i += 1;
    }
    flush(&mut buf, &mut out, base);
    out
}

fn find_close(chars: &[char], from: usize, delim: &[char]) -> Option<usize> {
    let n = delim.len();
    let mut j = from;
    while j + n <= chars.len() {
        if chars[j..j + n] == *delim {
            return Some(j);
        }
        j += 1;
    }
    None
}

fn find_single(chars: &[char], from: usize, delim: char) -> Option<usize> {
    let mut j = from;
    while j < chars.len() {
        if chars[j] == delim && chars.get(j + 1) != Some(&delim) && j > from {
            return Some(j);
        }
        j += 1;
    }
    None
}

fn parse_link(chars: &[char], i: usize) -> Option<(String, String, usize)> {
    if chars[i] != '[' {
        return None;
    }
    let mut j = i + 1;
    while j < chars.len() && chars[j] != ']' {
        j += 1;
    }
    if j + 1 >= chars.len() || chars[j] != ']' || chars[j + 1] != '(' {
        return None;
    }
    let label: String = chars[i + 1..j].iter().collect();
    let mut k = j + 2;
    while k < chars.len() && chars[k] != ')' {
        k += 1;
    }
    if k >= chars.len() {
        return None;
    }
    let url: String = chars[j + 2..k].iter().collect();
    Some((label, url, k + 1))
}

fn wrap_spans(spans: &[Span<'static>], width: usize) -> Vec<Vec<Span<'static>>> {
    if width == 0 {
        return vec![spans.to_vec()];
    }
    let mut rows: Vec<Vec<Span<'static>>> = vec![Vec::new()];
    let mut used = 0usize;
    for span in spans {
        let mut rest = span.content.as_ref();
        while !rest.is_empty() {
            let avail = width.saturating_sub(used);
            if avail == 0 {
                rows.push(Vec::new());
                used = 0;
                continue;
            }
            let word = rest.split_once(' ').map(|(w, _)| w).unwrap_or(rest);
            let word_w = word.width();
            if used > 0 && word_w > avail && word_w <= width {
                rows.push(Vec::new());
                used = 0;
                continue;
            }
            let mut take = take_prefix(rest, avail);
            if take.len() < rest.len() {
                if let Some(idx) = take.rfind(' ') {
                    if idx > 0 {
                        take = &take[..=idx];
                    }
                }
            }
            if take.is_empty() {
                rows.push(Vec::new());
                used = 0;
                continue;
            }
            rows.last_mut()
                .expect("row")
                .push(Span::styled(take.to_string(), span.style));
            used += take.width();
            rest = rest[take.len()..].trim_start();
        }
    }
    if rows.len() == 1 && rows[0].is_empty() {
        rows[0].push(Span::raw(""));
    }
    rows
}

fn take_prefix(s: &str, max_width: usize) -> &str {
    let mut w = 0usize;
    let mut end = 0usize;
    for (i, ch) in s.char_indices() {
        let cw = unicode_width::UnicodeWidthChar::width(ch)
            .unwrap_or(1)
            .max(1);
        if w + cw > max_width {
            break;
        }
        w += cw;
        end = i + ch.len_utf8();
    }
    if end == 0 && !s.is_empty() {
        let ch = s.chars().next().expect("char");
        return &s[..ch.len_utf8()];
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gold() {
        crate::ui::theme::apply(crate::palette::Palette::gold());
    }

    #[test]
    fn inline_bold_and_code() {
        gold();
        let spans = parse_inline("use `tmutil` and **bold**", body_style());
        let text: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(text, "use tmutil and bold");
        assert!(spans.iter().any(|s| s.content == "tmutil"));
        assert!(spans
            .iter()
            .any(|s| s.content == "bold" && s.style.add_modifier.contains(Modifier::BOLD)));
    }

    #[test]
    fn inline_link() {
        gold();
        let spans = parse_inline("see [Apple](https://apple.com)", body_style());
        assert!(spans
            .iter()
            .any(|s| s.content == "Apple" && s.style.add_modifier.contains(Modifier::UNDERLINED)));
    }

    #[test]
    fn table_has_box_drawing() {
        gold();
        let md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n";
        let mut lines = Vec::new();
        render(&mut lines, md, 40, &CodeHighlighter::new());
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(joined.contains('┌'), "{joined}");
        assert!(joined.contains('┼') || joined.contains('├'), "{joined}");
        assert!(joined.contains("A"));
        assert!(joined.contains("1"));
    }

    #[test]
    fn heading_list_uses_single_blank() {
        gold();
        let md = "## Current limitations\n\n\n\n- Widgets\n- Billing\n";
        let mut lines = Vec::new();
        render(&mut lines, md, 60, &CodeHighlighter::new());
        let blanks = lines.iter().filter(|l| rhythm::is_blank(l)).count();
        assert_eq!(blanks, 1, "stacked blanks after heading");
        assert!(!rhythm::is_blank(&lines[0]));
        assert!(rhythm::is_blank(&lines[1]));
    }

    #[test]
    fn list_skips_blank_between_items() {
        gold();
        let md = "- Alpha\n\n- Bravo\n\n- Charlie\n";
        let mut lines = Vec::new();
        render(&mut lines, md, 40, &CodeHighlighter::new());
        let blanks = lines.iter().filter(|l| rhythm::is_blank(l)).count();
        assert_eq!(blanks, 0, "list items should sit on consecutive rows");
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn heading_strips_hashes() {
        gold();
        let mut lines = Vec::new();
        render(
            &mut lines,
            "## Mapped verdict: aligned",
            40,
            &CodeHighlighter::new(),
        );
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(joined.contains("Mapped verdict"));
        assert!(!joined.contains("##"));
    }

    #[test]
    fn thumb_from_png() {
        gold();
        let dir = std::env::temp_dir().join(format!("ht-thumb-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.png");
        image::RgbImage::from_pixel(8, 8, image::Rgb([200, 80, 20]))
            .save(&path)
            .unwrap();
        let lines = image_thumb_lines(&path, 8, 4);
        assert!(!lines.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn image_syntax_makes_card() {
        gold();
        let mut lines = Vec::new();
        render(
            &mut lines,
            "![shot](https://example.com/a.png)",
            40,
            &CodeHighlighter::new(),
        );
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(joined.contains("image"));
        assert!(joined.contains("a.png"));
    }
}
