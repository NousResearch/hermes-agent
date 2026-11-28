//! Cross-client sync: every window and process on one library (secondary
//! "Open in new window" windows, scheduled runs, webhooks, the quick-entry
//! window, a second launch) writes through the same catalog, so a chat made,
//! renamed, archived or advanced elsewhere shows up here. A 2-second tick
//! compares library.json's modification time and reloads the chat list when it
//! moved. An open chat's transcript needs no polling: each conversation is
//! locked to the one window that has it open, so nobody else writes it.
use super::Hermes;
use std::{path::Path, time::SystemTime};

#[derive(Default)]
pub(super) struct State { catalog: Option<SystemTime> }

fn modified(path: &Path) -> Option<SystemTime> { std::fs::metadata(path).and_then(|m| m.modified()).ok() }

pub(super) fn subscription() -> iced::Subscription<super::Message> {
    iced::time::every(std::time::Duration::from_secs(2)).map(|_| super::Message::SyncTick)
}

pub(super) fn tick(app: &mut Hermes) {
    let Some(library) = app.library.clone() else { return };
    let catalog = modified(&library.root().join("library.json"));
    if catalog == app.sync.catalog { return }
    let first = app.sync.catalog.is_none();
    app.sync.catalog = catalog;
    if !first && app.rename.is_none() { app.refresh_sessions(); }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    /// `HERMES_RUST_DATA_DIR=<new empty dir> cargo test -p hermes-desktop -- --ignored another_window`
    #[test]
    #[ignore = "Hermes::new reads the process-wide data dir; needs a new empty HERMES_RUST_DATA_DIR"]
    fn a_chat_made_in_another_window_shows_up_here() {
        let (mut app, _) = Hermes::new();
        let root = app.library.as_ref().unwrap().root().to_owned();
        // This window holds no chat, so the other process may reuse the empty one it made at launch.
        app.current = None; app.session = None;
        app.refresh_sessions();
        super::tick(&mut app);
        // Another process opens the same library and makes a chat.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let other = hermes_library::Library::open(&root).unwrap();
        let made = other.create_session(None, None).unwrap();
        other.rename_session(&made.id, "Made elsewhere").unwrap();
        hermes_engine::Session::open(&made.directory).unwrap().record_turn(hermes_engine::Turn { context: String::new(), attachments: vec![], prompt: "Hi".into(),
            model: "m".into(), entries: vec![hermes_engine::Entry::Assistant { text: "Hello".into(), calls: vec![] }], status: hermes_engine::Status::Complete }).unwrap();
        other.touch_session(&made.id).unwrap();
        assert!(!app.sessions.iter().any(|s| s.id == made.id));
        super::tick(&mut app);
        assert!(app.sessions.iter().any(|s| s.id == made.id && s.title == "Made elsewhere"), "the other window's chat is listed");
    }
}
