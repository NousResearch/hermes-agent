//! HUD mode (the redesign's title-bar "comment-discussion" tool, ⌘⇧H): the
//! chrome-free floating chat band. As in Electron (spawnHudWindow), it is its
//! own frameless, always-on-top window pinned top-centre, and the app window
//! hides while it is up; leaving closes the band and brings the app back.
//! Like Electron's band, it fades out when idle (no typing, no running turn, the
//! cursor elsewhere) and then lets clicks fall through to the apps beneath;
//! bringing the cursor back over it, or a reply arriving, brings it back.
use super::hud_native;
use iced::{Point, Size, Task, window};
use std::time::{Duration, Instant};

pub(super) const WIDTH: f32 = 680.0;
pub(super) const HEIGHT: f32 = 440.0;
/// How long the band holds after the last activity before it fades.
const HOLD: Duration = Duration::from_secs(5);
/// The faded band's opacity: still findable, never in the way.
const FADED: f32 = 0.12;
const STEP: f32 = 0.14;

#[derive(Debug, Clone)]
pub(super) enum Action { Toggle, Opened(window::Id), Drag, Tick, Cursor(Option<bool>) }

#[derive(Default)]
pub(super) struct State {
    pub on: bool,
    pub window: Option<window::Id>,
    active: Option<Instant>,
    alpha: f32,
    passthrough: bool,
}

fn top_centre(window: Size, monitor: Size) -> Point { Point::new(((monitor.width - window.width) / 2.0).max(0.0), 48.0) }

impl State {
    /// Typing, sending or a streamed reply keeps the band up.
    pub fn touch(&mut self) { self.active = Some(Instant::now()); }
    pub fn subscription(&self) -> iced::Subscription<Action> {
        if self.on && self.window.is_some() && hud_native::FADES { iced::time::every(Duration::from_millis(90)).map(|_| Action::Tick) } else { iced::Subscription::none() }
    }
    pub fn update<M: Send + 'static>(&mut self, main: Option<window::Id>, busy: bool, action: Action, wrap: fn(Action) -> M) -> Task<M> {
        match action {
            Action::Toggle if self.on => {
                self.on = false;
                let mut tasks = vec![];
                if let Some(hud) = self.window.take() { tasks.push(window::close(hud)); }
                if let Some(main) = main { tasks.push(window::set_mode(main, window::Mode::Windowed)); tasks.push(window::gain_focus(main)); }
                Task::batch(tasks)
            }
            Action::Toggle => {
                self.on = true;
                let (_, open) = window::open(window::Settings {
                    size: Size::new(WIDTH, HEIGHT), min_size: Some(Size::new(380.0, 160.0)),
                    position: window::Position::SpecificWith(top_centre), level: window::Level::AlwaysOnTop,
                    decorations: false, resizable: true, exit_on_close_request: false, transparent: true, ..Default::default()
                });
                let mut tasks = vec![open.map(move |id| wrap(Action::Opened(id)))];
                if let Some(main) = main { tasks.push(window::set_mode(main, window::Mode::Hidden)); }
                Task::batch(tasks)
            }
            Action::Opened(id) => {
                self.window = Some(id);
                self.alpha = 1.0; self.passthrough = false; self.touch();
                Task::batch([window::run(id, hud_native::frost).discard(), window::gain_focus(id), iced::widget::operation::focus(super::COMPOSER_ID)])
            }
            Action::Drag => { self.touch(); self.window.map_or_else(Task::none, window::drag) }
            Action::Tick => match self.window {
                Some(id) => window::run(id, hud_native::cursor_inside).map(move |inside| wrap(Action::Cursor(inside))),
                None => Task::none(),
            },
            Action::Cursor(inside) => {
                let Some(id) = self.window else { return Task::none() };
                if inside == Some(true) { self.touch(); }
                let engaged = busy || self.active.is_some_and(|at| at.elapsed() < HOLD);
                let (alpha, pass) = step(self.alpha, engaged);
                let mut tasks = vec![];
                if alpha != self.alpha {
                    self.alpha = alpha;
                    tasks.push(window::run(id, move |w| hud_native::alpha(w, alpha)).discard());
                }
                if pass != self.passthrough {
                    self.passthrough = pass;
                    tasks.push(if pass { window::enable_mouse_passthrough(id) } else { window::disable_mouse_passthrough(id) });
                }
                Task::batch(tasks)
            }
        }
    }
}

/// One tick of the band: coming back is immediate, going away is a short fade, and only a
/// fully faded band lets clicks through. Returns (opacity, passes clicks through).
fn step(alpha: f32, engaged: bool) -> (f32, bool) {
    if engaged { return (1.0, false) }
    let alpha = (alpha - STEP).max(FADED);
    (alpha, alpha <= FADED)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_band_fades_then_passes_clicks_and_comes_straight_back() {
        let (mut alpha, mut pass) = (1.0, false);
        let mut ticks = 0;
        while !pass { (alpha, pass) = step(alpha, false); ticks += 1; assert!(ticks < 20); assert!(alpha >= FADED); }
        assert!(ticks > 3, "a fade, not a jump");
        assert_eq!(alpha, FADED);
        assert_eq!(step(alpha, true), (1.0, false), "the cursor or a reply brings it back at once");
        assert!(!step(1.0, false).1, "a band still visible never passes clicks through");
    }
}
