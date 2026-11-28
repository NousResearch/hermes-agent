//! Composer image preview strip (Vercel: all states designed).

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use super::markdown::{image_card, image_thumb_lines};
use super::theme::Theme;
use crate::state::{AppState, HitRange, HoverKind, PendingImage};

pub struct AttachPreview;

const SINGLE_IMAGE_ROWS: u16 = 12;
const MULTI_IMAGE_ROWS: u16 = 7;

impl AttachPreview {
    pub fn height(images: &[PendingImage], pastes: usize) -> u16 {
        let img = if images.is_empty() {
            0
        } else {
            let rows = if images.len() == 1 {
                SINGLE_IMAGE_ROWS
            } else {
                MULTI_IMAGE_ROWS.saturating_mul(2)
            };
            1 + rows + images.len().min(2) as u16
        };
        let pst = if pastes == 0 {
            0
        } else {
            1 + pastes.min(4) as u16
        };
        if img > 0 && pst > 0 {
            img + pst.saturating_sub(1)
        } else {
            img + pst
        }
    }

    pub fn render(frame: &mut Frame, area: Rect, state: &mut AppState) {
        state.hit_pastes.clear();
        let images = state.pending_images.clone();
        let pastes: Vec<String> = state
            .paste_chips
            .iter()
            .take(4)
            .map(|c| c.label.clone())
            .collect();
        if area.height == 0 || (images.is_empty() && pastes.is_empty()) {
            return;
        }
        let mut lines: Vec<Line> = vec![Line::from(vec![Span::styled(
            "  attached · click [[ ]] to preview",
            Style::default()
                .fg(Theme::brand_gold())
                .add_modifier(Modifier::BOLD),
        )])];
        let mut row = area.y.saturating_add(1);
        for label in &pastes {
            let shown = crate::tips::ellipsize(label, area.width.saturating_sub(2) as usize);
            let w = unicode_width::UnicodeWidthStr::width(shown.as_str()) as u16;
            state.hit_pastes.push((
                HitRange {
                    y: row,
                    x0: area.x.saturating_add(2),
                    x1: area.x.saturating_add(2).saturating_add(w.max(2)),
                },
                label.clone(),
            ));
            let hot = matches!(&state.hover, HoverKind::Paste(p) if p == label);
            lines.push(Line::from(Span::styled(
                format!("  {shown}"),
                crate::ui::theme::hover_paint(Style::default().fg(Theme::brand_gold()), hot),
            )));
            row = row.saturating_add(1);
        }
        for img in images.iter().take(2) {
            let thumb_h = if images.len() == 1 {
                SINGLE_IMAGE_ROWS
            } else {
                MULTI_IMAGE_ROWS
            };
            let thumb = image_thumb_lines(&img.path, area.width.saturating_sub(4), thumb_h);
            if thumb.is_empty() {
                lines.extend(image_card(
                    &img.name,
                    &img.path.to_string_lossy(),
                    area.width as usize,
                ));
            } else {
                lines.extend(thumb);
                let meta = match (img.width, img.height) {
                    (Some(w), Some(h)) => format!("  {}  ·  {w}×{h}", img.name),
                    _ => format!("  {}", img.name),
                };
                lines.push(Line::from(Span::styled(
                    meta,
                    Style::default().fg(Theme::text_muted()),
                )));
            }
        }
        if images.len() > 2 {
            lines.push(Line::from(Span::styled(
                format!("  +{} more", images.len() - 2),
                Style::default().fg(Theme::text_dim()),
            )));
        }
        frame.render_widget(
            Paragraph::new(lines).style(Style::default().bg(Theme::bg_base())),
            area,
        );
    }
}
