//! Quick Entry (app/quick-entry, electron/quick-entry.ts): the global-hotkey
//! mini window, ⌘⇧Space (Ctrl+Shift+Space elsewhere) from any app. One input
//! plus a session target, nothing else: Enter sends and hides, an empty Enter
//! does nothing, Escape or clicking away dismisses without sending.
use super::{Hermes, Message, ui};
use crate::appearance::text;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState, hotkey::{Code, HotKey, Modifiers}};
use iced::{Alignment, Element, Fill, Size, Subscription, Task, widget::{Space, column, container, pick_list, row, text_input}, window};

pub(super) const INPUT: &str = "quick-entry-input";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Target { NewChat, Current }
impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(match self { Self::NewChat => "New chat", Self::Current => "Current chat" }) }
}

#[derive(Debug, Clone)]
pub(super) enum Action { Hotkey, Opened(window::Id), Text(String), Target(Target), Submit, Dismiss, Unfocused(window::Id) }

#[derive(Default)]
pub(super) struct State {
    pub window: Option<window::Id>,
    manager: Option<GlobalHotKeyManager>,
    hotkey: Option<u32>,
    pub registration_error: Option<String>,
    text: String,
    target: Option<Target>,
    /// Focus settles for a moment after the window opens over another app.
    opened_at: Option<std::time::Instant>,
}

/// The hotkey arrives while another app is in front: bring Hermes forward so
/// the capture window takes keyboard focus.
pub(super) fn activate_app() {
    #[cfg(target_os = "macos")]
    if let Some(mtm) = objc2_foundation::MainThreadMarker::new() {
        #[allow(deprecated)]
        objc2_app_kit::NSApplication::sharedApplication(mtm).activateIgnoringOtherApps(true);
    }
}

impl State {
    /// Register the global shortcut. Must run on the main thread (AppKit).
    pub fn register(&mut self) {
        let modifier = if cfg!(target_os = "macos") { Modifiers::SUPER } else { Modifiers::CONTROL };
        let key = HotKey::new(Some(modifier | Modifiers::SHIFT), Code::Space);
        match GlobalHotKeyManager::new().map_err(|e| e.to_string()).and_then(|m| m.register(key).map(|_| m).map_err(|e| e.to_string())) {
            Ok(manager) => { self.hotkey = Some(key.id()); self.manager = Some(manager); }
            Err(e) => self.registration_error = Some(format!("The Quick Entry shortcut is unavailable: {e}")),
        }
    }
    pub fn subscription(&self) -> Subscription<Message> {
        let Some(id) = self.hotkey else { return Subscription::none() };
        Subscription::run_with(id, |id| {
            let id = *id;
            iced::stream::channel(8, move |mut out: iced::futures::channel::mpsc::Sender<Message>| async move {
                let (tx, rx) = async_channel::unbounded();
                std::thread::spawn(move || while let Ok(event) = GlobalHotKeyEvent::receiver().recv() {
                    if event.id == id && event.state == HotKeyState::Pressed && tx.send_blocking(()).is_err() { break; }
                });
                while rx.recv().await.is_ok() { let _ = iced::futures::SinkExt::send(&mut out, Message::QuickEntry(Action::Hotkey)).await; }
            })
        })
    }
}

fn close(app: &mut Hermes) -> Task<Message> {
    app.quick_entry.text.clear();
    app.quick_entry.window.take().map_or_else(Task::none, window::close)
}

pub(super) fn update(app: &mut Hermes, action: Action) -> Task<Message> {
    match action {
        Action::Hotkey if app.quick_entry.window.is_some() => close(app),
        Action::Hotkey => {
            activate_app();
            app.quick_entry.opened_at = Some(std::time::Instant::now());
            let (_, open) = window::open(window::Settings {
                size: Size::new(620.0, 100.0), resizable: false, decorations: false, level: window::Level::AlwaysOnTop,
                position: window::Position::SpecificWith(|w, m| iced::Point::new(((m.width - w.width) / 2.0).max(0.0), (m.height * 0.28).max(40.0))),
                exit_on_close_request: false, ..Default::default()
            });
            open.map(|id| Message::QuickEntry(Action::Opened(id)))
        }
        Action::Opened(id) => { app.quick_entry.window = Some(id); Task::batch([window::gain_focus(id), iced::widget::operation::focus(INPUT)]) }
        Action::Text(v) => { app.quick_entry.text = v; Task::none() }
        Action::Target(t) => { app.quick_entry.target = Some(t); Task::none() }
        Action::Dismiss => close(app),
        Action::Unfocused(id) if app.quick_entry.window == Some(id)
            && app.quick_entry.opened_at.is_none_or(|t| t.elapsed() > std::time::Duration::from_millis(600)) => close(app),
        Action::Unfocused(_) => Task::none(),
        Action::Submit => {
            let prompt = app.quick_entry.text.trim().to_owned();
            if prompt.is_empty() { return Task::none(); }
            let target = app.quick_entry.target.unwrap_or(if app.current.is_some() { Target::Current } else { Target::NewChat });
            let hide = close(app);
            if app.running { app.note("Hermes is busy with another reply — try again when it finishes."); return hide; }
            if target == Target::NewChat || app.current.is_none() { app.create_chat(); }
            app.screen = super::Screen::Chat;
            app.composer = iced::widget::text_editor::Content::with_text(&prompt);
            Task::batch([hide, app.update(Message::Send)])
        }
    }
}

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let state = &app.quick_entry;
    let target = state.target.unwrap_or(if app.current.is_some() { Target::Current } else { Target::NewChat });
    let input = text_input("Ask Hermes anything…", &state.text).id(INPUT).on_input(|v| Message::QuickEntry(Action::Text(v)))
        .on_submit(Message::QuickEntry(Action::Submit)).size(17).padding([10, 12]).width(Fill);
    let hint = if app.running { "Hermes is replying — wait for it to finish" } else { "Enter to send · Esc to dismiss" };
    container(column![
        row![ui::icon("comment-discussion", 16, p.accent), input].spacing(10).align_y(Alignment::Center),
        row![text(hint).size(11).color(p.muted), Space::new().width(Fill),
            pick_list([Target::NewChat, Target::Current], Some(target), |t| Message::QuickEntry(Action::Target(t))).text_size(12).padding([3, 8])].align_y(Alignment::Center),
    ].spacing(8)).padding(14).width(Fill).height(Fill).style(move |_| ui::panel(p.background, 12.0, Some(p.border))).into()
}
