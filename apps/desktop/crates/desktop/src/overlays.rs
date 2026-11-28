//! The two floating, see-through companions (electron/wake-indicator-window.ts,
//! app/pet-overlay):
//! - **Wake indicator**: a glowing pill at the top of the screen while a voice
//!   conversation is listening ("capturing") or waiting ("detected").
//! - **Desktop pet**: the hatchling egg (components/pet/pet-egg-sheet.png) in
//!   its own window, reacting to the agent — bouncing while it works, a bubble
//!   when it needs you or finishes. Drag it anywhere; click to bring Hermes up.
use super::{Hermes, Message, ui};
use crate::appearance::text;
use iced::{Alignment, Color, Element, Fill, Point, Size, Subscription, Task, widget::{column, container, image, mouse_area}, window};

const FRAME: u32 = 32;
const SCALE: u32 = 3;
const SHEET: &[u8] = include_bytes!("../assets/pet-egg-sheet.png");

#[derive(Debug, Clone)]
pub(super) enum Action { TogglePet, PetOpened(window::Id), PetTick, PetDrag, PetClick, WakeOpened(window::Id) }

#[derive(Default)]
pub(super) struct State {
    pub pet: Option<window::Id>, pet_on: bool, frame: usize, frames: Vec<image::Handle>,
    pub wake: Option<window::Id>, wake_opening: bool,
    celebrate_until: Option<std::time::Instant>, last_turns: usize,
}

fn frames() -> Vec<image::Handle> {
    let Ok(sheet) = ::image::load_from_memory_with_format(SHEET, ::image::ImageFormat::Png) else { return vec![] };
    let sheet = sheet.to_rgba8();
    (0..sheet.height() / FRAME).map(|i| {
        let frame = ::image::imageops::crop_imm(&sheet, 0, i * FRAME, FRAME, FRAME).to_image();
        let big = ::image::imageops::resize(&frame, FRAME * SCALE, FRAME * SCALE, ::image::imageops::FilterType::Nearest);
        image::Handle::from_rgba(big.width(), big.height(), big.into_raw())
    }).collect()
}

/// What the pet is doing, from the agent's live state.
pub(super) fn mood(app: &Hermes) -> (&'static str, Option<&'static str>) {
    if app.approval.is_some() { return ("waiting", Some("I need your OK")); }
    if app.running { return ("run", Some("Working on it…")); }
    if app.overlays.celebrate_until.is_some_and(|t| t > std::time::Instant::now()) { return ("jump", Some("Done!")); }
    if app.history.turns.last().is_some_and(|t| matches!(t.status, hermes_engine::Status::Failed(_))) { return ("failed", Some("That didn't work")); }
    ("idle", None)
}

pub(super) fn subscription(app: &Hermes) -> Subscription<Message> {
    if app.overlays.pet.is_none() { return Subscription::none(); }
    let ms = if app.running || app.approval.is_some() || app.overlays.celebrate_until.is_some() { 200 } else { 900 };
    iced::time::every(std::time::Duration::from_millis(ms)).map(|_| Message::Overlay(Action::PetTick))
}

pub(super) fn update(app: &mut Hermes, action: Action) -> Task<Message> {
    let state = &mut app.overlays;
    match action {
        Action::TogglePet if state.pet_on => { state.pet_on = false; state.pet.take().map_or_else(Task::none, window::close) }
        Action::TogglePet => {
            state.pet_on = true;
            if state.frames.is_empty() { state.frames = frames(); }
            let (_, open) = window::open(window::Settings {
                size: Size::new(170.0, 190.0), resizable: false, decorations: false, transparent: true, level: window::Level::AlwaysOnTop,
                position: window::Position::SpecificWith(|w, m| Point::new(m.width - w.width - 40.0, m.height - w.height - 80.0)),
                exit_on_close_request: false, ..Default::default()
            });
            open.map(|id| Message::Overlay(Action::PetOpened(id)))
        }
        Action::PetOpened(id) => {
            state.pet = Some(id);
            if app.pets.has_pet() { let (now, _) = mood(app); let _ = app.pets.frames_for(if now == "jump" { "wave" } else { now }); }
            Task::none()
        }
        Action::PetTick => {
            let turns = app.history.turns.len();
            if !app.running && turns > state.last_turns && app.history.turns.last().is_some_and(|t| t.status == hermes_engine::Status::Complete) {
                state.celebrate_until = Some(std::time::Instant::now() + std::time::Duration::from_secs(3));
            }
            state.last_turns = turns;
            if state.celebrate_until.is_some_and(|t| t <= std::time::Instant::now()) { state.celebrate_until = None; }
            state.frame = state.frame.wrapping_add(1);
            let (now, _) = mood(app);
            if app.pets.has_pet() { let _ = app.pets.frames_for(if now == "jump" { "wave" } else { now }); }
            Task::none()
        }
        Action::PetDrag => state.pet.map_or_else(Task::none, window::drag),
        Action::PetClick => app.webview.window.map_or_else(Task::none, |main| Task::batch([window::set_mode(main, window::Mode::Windowed), window::gain_focus(main)])),
        Action::WakeOpened(id) => { state.wake = Some(id); state.wake_opening = false; Task::none() }
    }
}

/// Open the wake pill while a voice conversation runs, close it after.
pub(super) fn sync_wake(app: &mut Hermes) -> Task<Message> {
    let want = app.voice.conversation;
    let state = &mut app.overlays;
    match (want, state.wake, state.wake_opening) {
        (true, None, false) => {
            state.wake_opening = true;
            let (_, open) = window::open(window::Settings {
                size: Size::new(140.0, 40.0), resizable: false, decorations: false, transparent: true, level: window::Level::AlwaysOnTop,
                position: window::Position::SpecificWith(|w, m| Point::new((m.width - w.width) / 2.0, 6.0)), exit_on_close_request: false, ..Default::default()
            });
            open.map(|id| Message::Overlay(Action::WakeOpened(id)))
        }
        (false, Some(id), _) => { state.wake = None; window::close(id) }
        _ => Task::none(),
    }
}

pub(super) fn pet_view(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let state = &app.overlays;
    let (mood, bubble) = mood(app);
    // Egg sheet: frames 0-5 bounce the intact egg; 6-11 crack it (used for "jump").
    let index = match mood {
        "run" => state.frame % 6,
        "jump" => 6 + state.frame % 6,
        "waiting" => if state.frame % 2 == 0 { 0 } else { 3 },
        "failed" => 7,
        _ => if state.frame % 5 == 0 { 1 } else { 0 },
    };
    let mut col = column![].spacing(4).align_x(Alignment::Center).width(Fill);
    let adopted = app.pets.cached(if mood == "jump" { "wave" } else { mood });
    col = col.push(container(bubble.map_or_else(|| text(""), |b| text(b).size(12).color(p.foreground))).padding([5, 9])
        .style(move |_| if bubble.is_some() { ui::panel(p.background, 10.0, Some(p.border)) } else { container::Style::default() }));
    if !adopted.is_empty() { col = col.push(image(adopted[state.frame % adopted.len()].clone()).width(96).height(104)); }
    else if let Some(frame) = state.frames.get(index) { col = col.push(image(frame.clone()).width((FRAME * SCALE) as f32).height((FRAME * SCALE) as f32)); }
    let body = container(col).center_x(Fill).align_bottom(Fill).padding(8);
    mouse_area(body).on_press(Message::Overlay(Action::PetDrag)).on_double_click(Message::Overlay(Action::PetClick)).into()
}

pub(super) fn wake_view(app: &Hermes) -> Element<'_, Message> {
    let capturing = app.voice.recording;
    let glow = Color::from_rgba8(124, 58, 237, if capturing { 0.85 } else { 0.55 });
    container(container(text(if capturing { "Listening" } else { "Hermes" }).size(12).color(Color::WHITE)).center(Fill)
        .style(move |_| container::Style { background: Some(glow.into()), border: iced::Border { radius: 999.0.into(), width: 1.0, color: Color::from_rgba8(216, 180, 254, 0.82) }, ..Default::default() }))
        .padding(4).width(Fill).height(Fill).into()
}
