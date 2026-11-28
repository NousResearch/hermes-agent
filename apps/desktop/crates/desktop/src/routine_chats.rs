//! Routine chats: every routine has an ordinary chat, "Routine · <name>", in
//! the Sessions list; each finished run is added to it as a turn (the run's
//! own transcript stays in the scheduler's run folder), so a routine reads and
//! continues like any other conversation.
use super::{Hermes, Message, Screen};
use hermes_engine::{History, Session};
use hermes_library::Library;
use std::{collections::BTreeMap, path::Path};

fn index_path(library: &Library) -> std::path::PathBuf { library.root().join("routine-chats.json") }
fn index(library: &Library) -> BTreeMap<String, String> {
    std::fs::read(index_path(library)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

/// The routine's chat, created (and remembered) on first use.
pub(super) fn chat_for(library: &Library, job: &str, title: &str) -> Result<hermes_library::SessionSummary, String> {
    let mut map = index(library);
    if let Some(id) = map.get(job) && let Ok(summary) = library.session(id) && !summary.archived { return Ok(summary); }
    let summary = library.create_session(None, None).map_err(|e| format!("{e:#}"))?;
    let summary = library.rename_session(&summary.id, &format!("Routine · {title}")).map_err(|e| format!("{e:#}"))?;
    map.insert(job.to_owned(), summary.id.clone());
    std::fs::write(index_path(library), serde_json::to_vec_pretty(&map).unwrap_or_default()).map_err(|e| e.to_string())?;
    Ok(summary)
}

/// The routine (job id) whose chat this session is.
pub(super) fn job_of(library: &Library, session: &str) -> Option<String> {
    index(library).into_iter().find(|(_, id)| id == session).map(|(job, _)| job)
}

/// The finished run's turn, read from its transcript folder.
fn run_turn(run_dir: &Path) -> Option<hermes_engine::Turn> {
    let history: History = serde_json::from_slice(&std::fs::read(run_dir.join("conversation.json")).ok()?).ok()?;
    history.turns.last().cloned()
}

/// Add a finished run to its routine's chat.
pub(super) fn record(app: &mut Hermes, job: &str, title: &str, run_dir: &Path) -> iced::Task<Message> {
    let Some(library) = app.library.clone() else { return iced::Task::none() };
    let Some(turn) = run_turn(run_dir) else { return iced::Task::none() };
    let summary = match chat_for(&library, job, title) { Ok(s) => s, Err(e) => { app.error(format!("Could not save the routine's chat: {e}")); return iced::Task::none() } };
    // The chat is open in this window: append through its live session, then show it.
    if app.current.as_ref().is_some_and(|c| c.id == summary.id) && let Some(session) = app.session.clone() {
        if app.running { app.note("A routine finished — its result is added to this chat after the current reply."); }
        return iced::Task::perform(async move {
            let mut session = session.lock().await;
            let result = session.record_turn(turn).map_err(|e| format!("{e:#}"));
            (session.history().clone(), result)
        }, |(history, result)| Message::RoutineRecorded(history, result));
    }
    match append(&library, &summary, turn) {
        Ok(()) => app.refresh_sessions(),
        Err(e) => app.error(format!("Could not save the routine's chat: {e}")),
    }
    iced::Task::none()
}

/// Append a run's turn to a routine chat that is not open anywhere.
fn append(library: &Library, summary: &hermes_library::SessionSummary, turn: hermes_engine::Turn) -> Result<(), String> {
    Session::open(&summary.directory).and_then(|mut s| s.record_turn(turn)).map_err(|e| format!("{e:#}"))?;
    library.touch_session(&summary.id).map_err(|e| format!("{e:#}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn each_run_lands_in_the_routines_own_chat() {
        let dir = tempfile::tempdir().unwrap();
        let library = Library::open(dir.path().join("lib")).unwrap();
        let run_dir = dir.path().join("run-1");
        std::fs::create_dir_all(&run_dir).unwrap();
        let mut history = History::default();
        history.turns.push(hermes_engine::Turn { context: String::new(), attachments: vec![], prompt: "Say hi".into(), model: "m".into(),
            entries: vec![hermes_engine::Entry::Assistant { text: "Hi from the routine".into(), calls: vec![] }], status: hermes_engine::Status::Complete });
        std::fs::write(run_dir.join("conversation.json"), serde_json::to_vec(&history).unwrap()).unwrap();
        for _ in 0..2 {
            let chat = chat_for(&library, "job-1", "Say hi to Hermes").unwrap();
            append(&library, &chat, run_turn(&run_dir).unwrap()).unwrap();
        }
        let chat = chat_for(&library, "job-1", "Say hi to Hermes").unwrap();
        assert_eq!(chat.title, "Routine · Say hi to Hermes");
        assert!(library.sessions(None).unwrap().iter().any(|s| s.id == chat.id), "it is listed with the other chats");
        let saved = Session::open(&chat.directory).unwrap().history().clone();
        assert_eq!(saved.turns.len(), 2, "both runs are in one chat");
        assert_ne!(chat_for(&library, "job-2", "Other").unwrap().id, chat.id, "each routine has its own chat");
    }
}

/// Open a routine's chat (from the sidebar box or the Routines page).
pub(super) fn open(app: &mut Hermes, job: &str) -> iced::Task<Message> {
    let Some(library) = app.library.clone() else { return iced::Task::none() };
    let title = app.scheduled.job_title(job).unwrap_or_else(|| "Routine".into());
    match chat_for(&library, job, &title) {
        Ok(summary) => { app.open_chat(summary); app.screen = Screen::Chat; app.refresh_sessions(); }
        Err(e) => app.error(e),
    }
    iced::Task::none()
}
