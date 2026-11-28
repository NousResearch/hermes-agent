//! Explicit, asynchronous copy of user-selected upstream Hermes conversations.
//! Merely opening this panel never discovers or reads a database.
use crate::appearance::text;
use hermes_library::{
    HermesHistory, HermesSession, ImportReport, Library, Project, SessionSummary,
};
use iced::{
    Element, Font, Length, Padding, Task, Theme,
    widget::{button, column, container, row, scrollable, text_input},
};
use std::{
    fmt,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const VISIBLE_SESSIONS: usize = 150;

#[derive(Clone)]
pub(crate) struct LoadedSource {
    history: Arc<Mutex<HermesHistory>>,
    sessions: Vec<HermesSession>,
    path: PathBuf,
}

#[derive(Clone)]
pub(crate) enum Message {
    PathChanged(String),
    Browse,
    FileChosen(Result<Option<PathBuf>, String>),
    Load,
    Loaded(Result<LoadedSource, String>),
    Search(String),
    Import(String),
    Imported(Box<Result<ImportReport, String>>),
    OpenImported,
    Clear,
}

// Do not log user history, file paths, or transcript contents through diagnostics.
impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PathChanged(_) => "ImportPathChanged",
            Self::Browse => "BrowseHistory",
            Self::FileChosen(_) => "HistoryFileChosen",
            Self::Load => "LoadHistory",
            Self::Loaded(_) => "HistoryLoaded",
            Self::Search(_) => "SearchHistory",
            Self::Import(_) => "ImportConversation",
            Self::Imported(_) => "ConversationImported",
            Self::OpenImported => "OpenImportedConversation",
            Self::Clear => "ClearImportSource",
        })
    }
}

#[derive(Default)]
pub(crate) struct ImportPanel {
    path: String,
    query: String,
    source: Option<LoadedSource>,
    working: Option<&'static str>,
    importing: Option<String>,
    error: Option<String>,
    result: Option<ImportReport>,
}

impl ImportPanel {
    /// The optional summary is returned only for the explicit Open imported chat
    /// action, allowing the result and conversion warnings to be reviewed first.
    pub(crate) fn update(
        &mut self,
        message: Message,
        library: Option<Library>,
        project_id: Option<String>,
    ) -> (Task<Message>, Option<SessionSummary>) {
        match message {
            Message::PathChanged(path) if self.working.is_none() => {
                self.path = path;
                self.source = None;
                self.error = None;
                self.result = None;
            }
            Message::Browse if self.working.is_none() => {
                self.working = Some("Choosing a history file…");
                self.error = None;
                return (
                    Task::perform(
                        async {
                            tokio::task::spawn_blocking(choose_database)
                                .await
                                .map_err(|error| error.to_string())
                                .and_then(|result| result)
                        },
                        Message::FileChosen,
                    ),
                    None,
                );
            }
            Message::FileChosen(result) => {
                self.working = None;
                match result {
                    Ok(Some(path)) => {
                        self.path = path.to_string_lossy().into_owned();
                        self.source = None;
                        self.result = None;
                    }
                    Ok(None) => {}
                    Err(error) => self.error = Some(error),
                }
            }
            Message::Load if self.working.is_none() => {
                if self.path.trim().is_empty() {
                    self.error =
                        Some("Choose your Hermes state.db file or enter its full path.".into());
                    return (Task::none(), None);
                }
                let path = expand_path(self.path.trim());
                self.working = Some("Reading the selected history…");
                self.source = None;
                self.result = None;
                self.error = None;
                return (
                    Task::perform(
                        async move {
                            tokio::task::spawn_blocking(move || load_source(&path))
                                .await
                                .map_err(|error| format!("Could not read history: {error}"))
                                .and_then(|result| result)
                        },
                        Message::Loaded,
                    ),
                    None,
                );
            }
            Message::Loaded(result) => {
                self.working = None;
                match result {
                    Ok(source) => {
                        self.path = source.path.to_string_lossy().into_owned();
                        self.query.clear();
                        self.source = Some(source);
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            Message::Search(query) => self.query = query,
            Message::Import(id) if self.working.is_none() => {
                let Some(library) = library else {
                    self.error =
                        Some("Your chat library is unavailable. Reopen the app and retry.".into());
                    return (Task::none(), None);
                };
                let Some(source) = &self.source else {
                    self.error = Some("Read a selected Hermes history file first.".into());
                    return (Task::none(), None);
                };
                if !source.sessions.iter().any(|session| session.id == id) {
                    self.error = Some("Choose a conversation from the loaded history.".into());
                    return (Task::none(), None);
                }
                let history = Arc::clone(&source.history);
                self.importing = Some(id.clone());
                self.working = Some("Copying the selected conversation…");
                self.error = None;
                self.result = None;
                return (
                    Task::perform(
                        async move {
                            tokio::task::spawn_blocking(move || {
                                let source = history.lock().map_err(|_| "History reader is unavailable. Read the selected file again.".to_owned())?;
                                library.import_hermes(&source, &id, project_id.as_deref())
                                    .map_err(|error| format!("Could not import this conversation: {error:#}"))
                            })
                            .await
                            .map_err(|error| format!("Could not copy history: {error}"))
                            .and_then(|result| result)
                        },
                        |result| Message::Imported(Box::new(result)),
                    ),
                    None,
                );
            }
            Message::Imported(result) => {
                self.working = None;
                self.importing = None;
                match *result {
                    Ok(report) => self.result = Some(report),
                    Err(error) => self.error = Some(error),
                }
            }
            Message::OpenImported => {
                return (
                    Task::none(),
                    self.result.as_ref().map(|report| report.session.clone()),
                );
            }
            Message::Clear if self.working.is_none() => {
                *self = Self::default();
            }
            Message::PathChanged(_)
            | Message::Browse
            | Message::Load
            | Message::Import(_)
            | Message::Clear => {}
        }
        (Task::none(), None)
    }

    pub(crate) fn view<'a>(
        &'a self,
        project: Option<&Project>,
        running: bool,
    ) -> Element<'a, Message> {
        let busy = self.working.is_some();
        let destination = project
            .map(|project| format!("Destination workspace: {}", project.name))
            .unwrap_or_else(|| "Destination: chats without a workspace".into());
        let mut source_actions =
            row![button("Read history").on_press_maybe((!busy).then_some(Message::Load)),]
                .spacing(10);
        {
            source_actions = source_actions
                .push(button("Choose file…").on_press_maybe((!busy).then_some(Message::Browse)));
        }
        if self.source.is_some() {
            source_actions = source_actions.push(
                button("Clear source")
                    .style(button::text)
                    .on_press_maybe((!busy).then_some(Message::Clear)),
            );
        }
        let source_card = column![
            text("History file").size(13).font(semibold()),
            text("Select the state.db file from your Hermes folder or profile. Your original history stays unchanged.").size(12).style(muted),
            text_input("/path/to/.hermes/state.db", &self.path)
                .on_input(Message::PathChanged).on_submit(Message::Load).padding(9).size(13),
            source_actions,
            text(destination).size(12).style(muted),
            text("Choose a project before importing to attach the copied session to it.").size(12).style(muted),
        ].spacing(12);
        let mut content = column![
            text("Import sessions").size(14).font(semibold()),
            text("Copy selected Hermes sessions into this library.")
                .size(12)
                .style(muted),
            container(source_card).width(Length::Fill),
        ]
        .spacing(18);

        if let Some(working) = self.working {
            content = content.push(text(working).size(12).style(muted));
        }
        if let Some(error) = &self.error {
            content =
                content.push(
                    container(text(error).size(12).style(|theme: &Theme| {
                        iced::widget::text::Style {
                            color: Some(theme.palette().danger),
                        }
                    }))
                    .padding([4, 0])
                    .width(Length::Fill),
                );
        }
        if let Some(report) = &self.result {
            let heading = if report.already_imported {
                "This conversation is already in your library"
            } else {
                "Your conversation was copied"
            };
            let mut result = column![
                text(heading).size(14).font(semibold()),
                text(short_label(&report.session.title, 160)).size(13)
            ]
            .spacing(10);
            if !report.already_imported {
                result = result.push(
                    text(format!(
                        "{} conversation messages imported",
                        report.imported_messages
                    ))
                    .size(12)
                    .style(muted),
                );
            }
            for warning in &report.warnings {
                result = result.push(text(format!("• {warning}")).size(12).style(muted));
            }
            result = result.push(
                text(format!("Source archive: {}", report.archive_path.display()))
                    .size(11)
                    .style(muted),
            );
            result = result.push(
                button("Open imported chat")
                    .on_press_maybe((!running && !busy).then_some(Message::OpenImported)),
            );
            if running {
                result = result.push(
                    text("Finish the active response before opening another session.")
                        .size(12)
                        .style(muted),
                );
            }
            content = content.push(container(result).padding([10, 0]).width(Length::Fill));
        }
        if let Some(source) = &self.source {
            let matches: Vec<_> = source
                .sessions
                .iter()
                .filter(|session| matches_search(session, &self.query))
                .collect();
            let mut list = column![
                text("Sessions").size(14).font(semibold()),
                text(format!(
                    "{} matching conversations · {} total",
                    matches.len(),
                    source.sessions.len()
                ))
                .size(12)
                .style(muted),
                text_input("Search sessions…", &self.query)
                    .on_input(Message::Search)
                    .padding(9)
                    .size(13),
            ]
            .spacing(12);
            if matches.is_empty() {
                list = list.push(
                    text(if self.query.trim().is_empty() {
                        "This history file contains no conversations."
                    } else {
                        "No conversations match this search."
                    })
                    .size(12)
                    .style(muted),
                );
            }
            for session in matches.iter().take(VISIBLE_SESSIONS) {
                let title = if session.title.trim().is_empty() {
                    "Untitled Hermes chat"
                } else {
                    &session.title
                };
                let mut details = format!(
                    "{} · {}",
                    short_label(&session.source, 24),
                    short_label(session.model.as_deref().unwrap_or("Unknown model"), 72)
                );
                if session.archived {
                    details.push_str(" · Archived");
                }
                if session.hidden {
                    details.push_str(" · Hidden");
                }
                let metadata = column![
                    text(short_label(title, 150)).size(13),
                    text(details).size(12).style(muted),
                    text(short_label(&session.id, 84)).size(11).style(muted),
                ]
                .spacing(5)
                .width(Length::Fill);
                let label = if self.importing.as_deref() == Some(session.id.as_str()) {
                    "Copying…"
                } else {
                    "Import"
                };
                list = list.push(
                    container(
                        row![
                            metadata,
                            button(label).on_press_maybe(
                                (!busy && !running).then(|| Message::Import(session.id.clone()))
                            ),
                        ]
                        .spacing(16)
                        .align_y(iced::Alignment::Center),
                    )
                    .padding([10, 0])
                    .width(Length::Fill),
                );
            }
            if matches.len() > VISIBLE_SESSIONS {
                list = list.push(text(format!("Showing the first {VISIBLE_SESSIONS} matches. Refine your search to find another session.")).size(12).style(muted));
            }
            if running {
                list = list.push(
                    text("Finish the active response before importing a session.")
                        .size(12)
                        .style(muted),
                );
            }
            content = content.push(container(list).padding(2).width(Length::Fill));
        }
        container(scrollable(content).height(Length::Fill))
            .padding(Padding {
                top: 4.0,
                right: 32.0,
                bottom: 80.0,
                left: 32.0,
            })
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

fn semibold() -> Font {
    Font {
        weight: iced::font::Weight::Semibold,
        ..crate::appearance::font()
    }
}

fn muted(theme: &Theme) -> iced::widget::text::Style {
    let dark = theme.palette().background.r < 0.5;
    iced::widget::text::Style {
        color: Some(super::ui::palette(dark).muted),
    }
}

fn load_source(path: &Path) -> Result<LoadedSource, String> {
    let history = HermesHistory::open(path)
        .map_err(|error| format!("Could not read this Hermes history file: {error:#}"))?;
    let sessions = history
        .sessions()
        .map_err(|error| format!("Could not list this Hermes history: {error:#}"))?;
    let path = history.database().to_owned();
    Ok(LoadedSource {
        history: Arc::new(Mutex::new(history)),
        sessions,
        path,
    })
}

fn expand_path(value: &str) -> PathBuf {
    if (value == "~" || value.starts_with("~/"))
        && let Some(home) = std::env::var_os("HOME")
    {
        return Path::new(&home).join(value.trim_start_matches('~').trim_start_matches('/'));
    }
    PathBuf::from(value)
}

fn short_label(value: &str, limit: usize) -> String {
    let clean = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut characters = clean.chars();
    let result: String = characters.by_ref().take(limit).collect();
    if characters.next().is_some() {
        format!("{result}…")
    } else {
        result
    }
}

fn matches_search(session: &HermesSession, query: &str) -> bool {
    let query = query.trim().to_lowercase();
    query.is_empty()
        || [
            session.title.as_str(),
            session.id.as_str(),
            session.source.as_str(),
            session.model.as_deref().unwrap_or(""),
        ]
        .iter()
        .any(|field| field.to_lowercase().contains(&query))
}

fn choose_database() -> Result<Option<PathBuf>, String> {
    crate::picker::choose("Choose your Hermes state.db history", false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_panel_does_not_select_or_load_any_source() {
        let panel = ImportPanel::default();
        assert!(panel.path.is_empty());
        assert!(panel.source.is_none());
        assert!(panel.result.is_none());
        assert!(panel.working.is_none());
    }

    #[test]
    fn import_messages_do_not_expose_history_in_debug_output() {
        let sensitive = "synthetic-private-history-path-or-id";
        for message in [
            Message::PathChanged(sensitive.into()),
            Message::Search(sensitive.into()),
            Message::Import(sensitive.into()),
            Message::Loaded(Err(sensitive.into())),
        ] {
            assert!(!format!("{message:?}").contains(sensitive));
        }
    }

    #[test]
    fn invalid_import_action_requires_an_explicit_loaded_source() {
        let mut panel = ImportPanel::default();
        let (_, selected) = panel.update(Message::Load, None, None);
        assert!(selected.is_none());
        assert!(panel.error.as_deref().unwrap().contains("Choose"));
        assert!(panel.working.is_none());
        let (_, selected) = panel.update(Message::Import("synthetic-id".into()), None, None);
        assert!(selected.is_none());
        assert!(panel.source.is_none());
        assert!(panel.result.is_none());
    }
}
