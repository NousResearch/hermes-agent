use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use crate::state::AppState;
use crate::ui::theme::Theme;

impl super::ViewsOverlay {
    pub(super) fn render_clarify_modal(frame: &mut Frame, area: Rect, state: &AppState) {
        let Some(c) = &state.pending_clarify else {
            return;
        };
        let modal_area = Self::centered_rect(70, 50, area);
        frame.render_widget(Clear, modal_area);
        let mut body = vec![
            Line::from(vec![Span::styled(
                " clarify",
                Style::default()
                    .fg(Theme::brand_gold())
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::raw(""),
            Line::from(vec![Span::styled(
                format!("  {}", c.question),
                Style::default().fg(Theme::text_primary()),
            )]),
            Line::raw(""),
        ];
        if c.choices.is_empty() {
            body.push(Line::from(vec![Span::styled(
                format!("  > {}_", c.typed),
                Style::default().fg(Theme::brand_gold()),
            )]));
            body.push(Line::from(vec![Span::styled(
                "  enter submit   esc dismiss",
                Style::default().fg(Theme::text_dim()),
            )]));
        } else {
            for (i, ch) in c.choices.iter().enumerate() {
                let mark = if i == c.selected { "▸" } else { " " };
                let style = if i == c.selected {
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Theme::text_secondary())
                };
                body.push(Line::from(vec![Span::styled(
                    format!("  {mark} {}. {ch}", i + 1),
                    style,
                )]));
            }
            body.push(Line::raw(""));
            body.push(Line::from(vec![Span::styled(
                "  enter confirm   esc dismiss",
                Style::default().fg(Theme::text_dim()),
            )]));
        }
        let p = Paragraph::new(body)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Theme::border_focus()))
                    .style(Style::default().bg(Theme::bg_popup())),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(p, modal_area);
    }

    pub(super) fn render_secret_modal(frame: &mut Frame, area: Rect, state: &AppState) {
        let Some(sec) = &state.pending_secret else {
            return;
        };
        let modal_area = Self::centered_rect(60, 30, area);
        frame.render_widget(Clear, modal_area);
        let title = match sec.kind {
            crate::state::SecretKind::Sudo => " sudo",
            crate::state::SecretKind::Secret => " secret",
        };
        let mask = "•".repeat(sec.buffer.chars().count());
        let body = vec![
            Line::from(vec![Span::styled(
                format!("  {}", sec.prompt),
                Style::default().fg(Theme::text_primary()),
            )]),
            Line::raw(""),
            Line::from(vec![Span::styled(
                format!("  {mask}_"),
                Style::default().fg(Theme::brand_gold()),
            )]),
            Line::raw(""),
            Line::from(vec![Span::styled(
                "  enter submit   esc cancel",
                Style::default().fg(Theme::text_dim()),
            )]),
        ];
        let p = Paragraph::new(body).block(
            Block::default()
                .title(Span::styled(
                    title,
                    Style::default()
                        .fg(Theme::brand_gold())
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Theme::border_focus()))
                .style(Style::default().bg(Theme::bg_popup())),
        );
        frame.render_widget(p, modal_area);
    }

    pub(super) fn render_approval_modal(frame: &mut Frame, area: Rect, state: &AppState) {
        let Some(req) = &state.pending_approval else {
            return;
        };
        let modal_area = Self::centered_rect(70, 40, area);
        frame.render_widget(Clear, modal_area);

        let always = if req.allow_permanent {
            "  [a] always"
        } else {
            ""
        };
        let body = vec![
            Line::from(vec![Span::styled(
                " Approval required",
                Style::default()
                    .fg(Theme::accent_red())
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::raw(""),
            Line::from(vec![Span::styled(
                format!("  {}", req.description),
                Style::default().fg(Theme::text_primary()),
            )]),
            Line::from(vec![Span::styled(
                format!("  {}", req.command),
                Style::default().fg(Theme::text_muted()),
            )]),
            Line::raw(""),
            Line::from(vec![Span::styled(
                format!("  [y/enter] once   [n/esc] deny{always}"),
                Style::default().fg(Theme::text_secondary()),
            )]),
        ];
        let p = Paragraph::new(body)
            .block(
                Block::default()
                    .title(Span::styled(
                        " ⚠ command approval ",
                        Style::default()
                            .fg(Theme::accent_red())
                            .add_modifier(Modifier::BOLD),
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Theme::accent_red()))
                    .style(Style::default().bg(Theme::bg_popup())),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(p, modal_area);
    }
}
