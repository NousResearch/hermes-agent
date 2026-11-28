//! Native rendering of the official Desktop empty-draft intro.
//!
//! Source: components/chat/{intro,wordmark}.tsx and styles.css's .wordmark /
//! [data-slot='aui_intro'] rules. The wordmark uses the original Collapse Bold
//! glyph outlines, including original kerning and the source's inherited
//! 1.12px tracking (.08em computed at the 14px parent before fit-text grows).

use crate::appearance::text;
use super::{Hermes, Message, ui};
use iced::{
    Alignment, Element, Fill,
    widget::{Space, column, container, svg},
};

pub(super) const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Collapse-Bold.ttf");
const WORDMARK: &[u8] = include_bytes!("../assets/hermes-agent-wordmark.svg");

// These are the five neutral English bodies shipped in the official
// components/chat/intro-copy.jsonl. Its editorial headlines are not rendered.
const STOCK_BODY: [&str; 5] = [
    "Ask a question, paste an error, or point me at a repo. I can read code, run tools, and help you ship.",
    "Describe the task in your own words. I'll pick the right tools, explain my plan, and check in before risky steps.",
    "Drop a file path, a traceback, or a rough idea. I'll investigate, suggest next steps, and keep things reversible.",
    "Search the repo, edit files, run tests, open PRs. Tell me the goal and I'll handle the mechanical parts.",
    "Type a task, question, or snippet. I remember the session, cite my sources, and stop to ask when I'm unsure.",
];

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    // Core's splash belongs to a new draft. Selecting an existing, empty chat
    // does not paint another welcome screen over that session.
    if !app.fresh_draft || !app.history.turns.is_empty() {
        return Space::new().width(Fill).height(Fill).into();
    }

    let colors = ui::palette(app.dark);
    let wordmark_color = if app.dark {
        iced::Color {
            a: 0.9,
            ..colors.foreground
        }
    } else {
        colors.accent
    };
    let index = if app.sessions.len() <= 1 {
        0
    } else {
        app.current.as_ref().map_or(0, |session| {
            session.id.bytes().fold(0usize, |seed, byte| {
                seed.wrapping_mul(31).wrapping_add(usize::from(byte))
            }) % STOCK_BODY.len()
        })
    };
    static WORDMARK_HANDLE:std::sync::OnceLock<svg::Handle>=std::sync::OnceLock::new();
    // The original fit-text consumes the thread width after its gutters and
    // hidden measurement span. Keep the glyph outlines proportional on resize.
    let width = ((app.intro_width - 138.0) * 0.9).clamp(92.0, 2200.0);
    let wordmark = container(
        svg(WORDMARK_HANDLE.get_or_init(||svg::Handle::from_memory(WORDMARK)).clone())
            .width(Fill)
            .height(width*90.0274/740.8314)
            .style(move |_, _| svg::Style {
                color: Some(wordmark_color),
            }),
    )
    .width(Fill)
    .max_width(width);
    let body = container(
        text(STOCK_BODY[index])
            .font(crate::appearance::font())
            .size(14)
            .line_height(1.45)
            .color(iced::Color {
                a: 0.54,
                ..colors.foreground
            })
            .width(Fill)
            .align_x(iced::alignment::Horizontal::Center),
    )
    .width(Fill)
    .max_width(544);
    let intro = column![wordmark, body]
        .spacing(20)
        .align_x(Alignment::Center)
        .width(Fill);

    // The official composer overlays its full-height thread. This native
    // surface gives the composer its own layout row, so restore half that
    // compact row's height when centering the intro inside the remaining space.
    iced::widget::sensor(container(column![Space::new().height(48), intro])
        .padding([24, 32])
        .width(Fill)
        .height(Fill)
        .center_x(Fill)
        .center_y(Fill)).on_show(Message::IntroSize).on_resize(Message::IntroSize)
        .into()
}
