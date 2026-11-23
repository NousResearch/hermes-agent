//! First-run "Welcome to Hermes" splash (discovery/first-time-splash.tsx):
//! step 1 lists the places in the nav, step 2 is "Make it yours" and the way in.
//! Shown once per library; reopened from the command palette.
use super::{Hermes, Message, ui};
use crate::appearance::{DISPLAY_FONT, text};
use iced::{Alignment, Color, Element, Fill, Padding, widget::{Space, button, column, container, row}};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Action { Next, Back, Close }

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Step { #[default] Places, Style }

#[derive(Default)]
pub(super) struct State { pub open: bool, step: Step }

const DONE: &str = "onboarding-complete";

impl State {
    /// Open on a library that has never finished the splash.
    pub fn load(root: &Path) -> Self { Self { open: !root.join(DONE).exists(), step: Step::Places } }
}

pub(super) fn update(app: &mut Hermes, action: Action) {
    match action {
        Action::Next => app.onboarding.step = Step::Style,
        Action::Back => app.onboarding.step = Step::Places,
        Action::Close => {
            app.onboarding.open = false; app.onboarding.step = Step::Places;
            if let Some(library) = &app.library { let _ = std::fs::write(library.root().join(DONE), b"1\n"); }
        }
    }
}

/// FIRST_TIME_PLACES with their renamed nav names and copy.lines.
const PLACES: [(&str, &str, &str, &str); 4] = [
    ("comment-discussion", "Chats", "Don’t just ask, assign.", "e.g. “research my 5 biggest competitors, compare their pricing, cite sources”"),
    ("hubot", "Agents", "A team of experts. Each one gets smarter over time.", "e.g. a research agent that knows your industry"),
    ("watch", "Scheduled jobs", "Set it once. Hermes keeps checking.", "e.g. a table for 2 opening up, a sold-out item back in stock"),
    ("files", "Library", "Everything your agents make, in one place.", "e.g. every brief, report and doc it made, from every chat"),
];

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let eyebrow = |label: &'static str| text(label.to_uppercase()).size(11).font(ui::medium()).color(Color { a: 0.6, ..p.foreground });
    let title = |label: &'static str| text(label).size(40).font(DISPLAY_FONT).line_height(1.1);
    let subtitle = |label: &'static str| text(label).size(14).color(p.muted);
    let action = |label: &'static str, message: Message| button(text(label).size(14).font(ui::medium())).padding([10, 18]).on_press(message);
    let card = |content: Element<'static, Message>| container(content).width(Fill).style(move |_| ui::panel(Color { a: 0.04, ..p.foreground }, 4.0, Some(Color { a: 0.06, ..p.foreground })));
    let body: Element<'_, Message> = match app.onboarding.step {
        Step::Places => {
            let mut places = column![].spacing(0);
            for (index, (glyph, name, line, example)) in PLACES.iter().enumerate() {
                if index > 0 { places = places.push(container(Space::new().height(1).width(Fill)).style(move |_| ui::panel(Color { a: 0.05, ..p.foreground }, 0.0, None))); }
                places = places.push(container(row![ui::icon(glyph, 16, p.muted),
                    column![text(*name).size(14).font(ui::medium()), text(*line).size(13).color(p.muted), text(*example).size(12).color(Color { a: 0.5, ..p.foreground })].spacing(2)]
                    .spacing(14).align_y(Alignment::Start)).padding([12, 16]));
            }
            column![
                eyebrow("Hermes Agent"),
                column![title("Welcome to Hermes"), subtitle("Give it a task. It learns how you work and keeps going while you’re away.")].spacing(8),
                card(places.into()),
                column![text("Select your style").size(15).font(ui::medium()), text("Theme and choose your Hermes artwork.").size(13).color(p.muted)].spacing(4),
                row![Space::new().width(Fill), action("Make it yours", Message::Onboarding(Action::Next))],
            ].spacing(20).into()
        }
        Step::Style => column![
            row![eyebrow("Your Hermes"), Space::new().width(Fill),
                button(row![ui::icon("arrow-left", 12, p.muted), text("Back").size(12).color(p.muted)].spacing(4).align_y(Alignment::Center))
                    .padding([2, 4]).style(ui::quiet_button).on_press(Message::Onboarding(Action::Back))].align_y(Alignment::Center),
            column![title("Make it yours"), subtitle("Give it a task. It learns how you work and keeps going while you’re away.")].spacing(8),
            super::appearance::make_it_yours(app),
            column![text("Ready to get started?").size(15).font(ui::medium()), text("Find this again in the command palette.").size(13).color(p.muted)].spacing(4),
            row![Space::new().width(Fill), action("Let’s get to work", Message::Onboarding(Action::Close))],
        ].spacing(20).into(),
    };
    let panel = container(super::settings::quiet_scroll(container(body).padding(Padding { top: 40.0, right: 40.0, bottom: 40.0, left: 40.0 }).max_width(640)))
        .max_width(640).max_height(720)
        .style(move |_| ui::panel(p.background, 8.0, Some(Color { a: 0.08, ..p.foreground })));
    container(panel).center_x(Fill).center_y(Fill).width(Fill).height(Fill)
        .style(|_| iced::widget::container::Style { background: Some(Color { a: 0.45, ..Color::BLACK }.into()), ..Default::default() }).into()
}
