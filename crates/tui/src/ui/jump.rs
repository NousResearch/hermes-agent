use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Clear, Paragraph},
    Frame,
};
use unicode_width::UnicodeWidthStr;

use super::theme::Theme;
use crate::state::{AppState, HitRange, HoverKind};

/// Tiny centered “jump to live tail” chip, painted only while scrolled up.
pub struct JumpChip;

impl JumpChip {
    pub const HEIGHT: u16 = 1;
    const LABEL: &'static str = " ↓ ";

    pub fn render(frame: &mut Frame, area: Rect, state: &mut AppState) {
        if area.height == 0 || !state.scrolled_off_tail() {
            state.hit_jump = None;
            return;
        }
        let label_w = Self::LABEL.width() as u16;
        let width = label_w.min(area.width);
        if width == 0 {
            state.hit_jump = None;
            return;
        }
        let x = area.x.saturating_add(area.width.saturating_sub(width) / 2);
        let chip = Rect {
            x,
            y: area.y,
            width,
            height: 1,
        };
        frame.render_widget(Clear, chip);
        let hot = state.hover == HoverKind::Jump;
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                Self::LABEL,
                crate::ui::theme::hover_paint(
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                    hot,
                ),
            )))
            .style(Style::default().bg(if hot {
                Theme::bg_highlight()
            } else {
                Theme::bg_popup()
            })),
            chip,
        );
        state.hit_jump = Some(HitRange {
            y: chip.y,
            x0: chip.x,
            x1: chip.x.saturating_add(chip.width),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn height_is_one() {
        assert_eq!(JumpChip::HEIGHT, 1);
        assert_eq!(JumpChip::LABEL.width(), 3);
    }
}
