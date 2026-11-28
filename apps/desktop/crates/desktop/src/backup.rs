//! Settings > Backup, adapted from stock 0284ce17 backup-settings.tsx.
//! Native snapshots have an explicit, narrower scope than stock whole-home
//! backups and Cloud transfer. No cloud login or service request is made here.
use crate::{Hermes, Message, appearance::text, ui};
use hermes_library::{
    Library, NativeBackupPlan, NativeBackupSummary, NativeRestoreReport, inspect_native_backup,
};
use iced::{
    Alignment, Element, Fill, Font, Padding, Task,
    widget::{Space, button, column, container, row, scrollable, text_input},
};
use std::{fmt, path::PathBuf, sync::Arc};

#[derive(Clone)]
pub(super) enum Action {
    ExportPath(String),
    SourcePath(String),
    BrowseExport,
    BrowseSource,
    ExportChosen(Result<Option<PathBuf>, String>),
    SourceChosen(Result<Option<PathBuf>, String>),
    Export,
    Exported(Result<NativeBackupSummary, String>),
    Inspect,
    Inspected(Result<Arc<NativeBackupPlan>, String>),
    Import,
    Imported(Result<NativeRestoreReport, String>),
}
impl fmt::Debug for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ExportPath(_) => "BackupDestination",
            Self::SourcePath(_) => "BackupSource",
            Self::BrowseExport => "ChooseBackupDestination",
            Self::BrowseSource => "ChooseBackupSource",
            Self::ExportChosen(_) => "BackupDestinationChosen",
            Self::SourceChosen(_) => "BackupSourceChosen",
            Self::Export => "ExportNativeBackup",
            Self::Exported(_) => "NativeBackupExported",
            Self::Inspect => "InspectNativeBackup",
            Self::Inspected(_) => "NativeBackupInspected",
            Self::Import => "ImportNativeBackup",
            Self::Imported(_) => "NativeBackupImported",
        })
    }
}

#[derive(Default)]
pub(super) struct State {
    export_path: String,
    source_path: String,
    busy: bool,
    operation: &'static str,
    plan: Option<Arc<NativeBackupPlan>>,
    error: Option<String>,
    result: Option<String>,
}
impl State {
    pub(super) fn update(
        &mut self,
        action: Action,
        library: Option<Library>,
        running: bool,
    ) -> Task<Action> {
        match action {
            Action::ExportPath(path) if !self.busy => {
                self.export_path = path;
                self.error = None;
            }
            Action::SourcePath(path) if !self.busy => {
                self.source_path = path;
                self.plan = None;
                self.error = None;
            }
            Action::BrowseExport if !self.busy => {
                return Task::perform(
                    async {
                        tokio::task::spawn_blocking(choose_destination)
                            .await
                            .map_err(|_| "Could not open the backup save panel".to_owned())
                            .and_then(|result| result)
                    },
                    Action::ExportChosen,
                );
            }
            Action::BrowseSource if !self.busy => {
                return Task::perform(
                    async {
                        tokio::task::spawn_blocking(choose_source)
                            .await
                            .map_err(|_| "Could not open the backup file picker".to_owned())
                            .and_then(|result| result)
                    },
                    Action::SourceChosen,
                );
            }
            Action::ExportChosen(result) => match result {
                Ok(Some(path)) => match path.to_str() {
                    Some(path) => self.export_path = path.into(),
                    None => self.error = Some("Choose a path with a UTF-8 filename".into()),
                },
                Ok(None) => {}
                Err(error) => self.error = Some(error),
            },
            Action::SourceChosen(result) => match result {
                Ok(Some(path)) => match path.to_str() {
                    Some(path) => {
                        self.source_path = path.into();
                        self.plan = None;
                    }
                    None => self.error = Some("Choose a path with a UTF-8 filename".into()),
                },
                Ok(None) => {}
                Err(error) => self.error = Some(error),
            },
            Action::Export if !self.busy => {
                if running {
                    self.error=Some("Stop the current response and scheduled runs before making a native backup.".into());
                    return Task::none();
                }
                if self.export_path.trim().is_empty() {
                    return self.update(Action::BrowseExport, library, running);
                }
                let Some(library) = library else {
                    self.error = Some("Native profile is unavailable".into());
                    return Task::none();
                };
                let path = PathBuf::from(self.export_path.trim());
                self.begin("Backing up…");
                return Task::perform(
                    async move {
                        tokio::task::spawn_blocking(move || {
                            library
                                .export_native_backup(&path)
                                .map_err(|error| format!("{error:#}"))
                        })
                        .await
                        .map_err(|_| "The backup did not finish".to_owned())
                        .and_then(|result| result)
                    },
                    Action::Exported,
                );
            }
            Action::Exported(result) => {
                self.busy = false;
                match result {
                    Ok(summary) => {
                        self.result = Some(format!(
                            "Saved {} conversations and {} files to {}.",
                            summary.sessions, summary.files, self.export_path
                        ))
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            Action::Inspect if !self.busy => {
                if self.source_path.trim().is_empty() {
                    return self.update(Action::BrowseSource, library, running);
                }
                let path = PathBuf::from(self.source_path.trim());
                self.plan = None;
                self.begin("Inspecting backup…");
                return Task::perform(
                    async move {
                        tokio::task::spawn_blocking(move || {
                            inspect_native_backup(&path)
                                .map(Arc::new)
                                .map_err(|error| format!("{error:#}"))
                        })
                        .await
                        .map_err(|_| "Could not inspect native backup".to_owned())
                        .and_then(|result| result)
                    },
                    Action::Inspected,
                );
            }
            Action::Inspected(result) => {
                self.busy = false;
                match result {
                    Ok(plan) => self.plan = Some(plan),
                    Err(error) => self.error = Some(error),
                }
            }
            Action::Import if !self.busy => {
                if running {
                    self.error = Some(
                        "Stop the current response and scheduled runs before importing a backup."
                            .into(),
                    );
                    return Task::none();
                }
                let (Some(library), Some(plan)) = (library, self.plan.clone()) else {
                    self.error = Some("Inspect a native backup first".into());
                    return Task::none();
                };
                self.begin("Importing backup…");
                return Task::perform(
                    async move {
                        tokio::task::spawn_blocking(move || {
                            library
                                .import_native_backup(&plan)
                                .map_err(|error| format!("{error:#}"))
                        })
                        .await
                        .map_err(|_| "Could not import native backup".to_owned())
                        .and_then(|result| result)
                    },
                    Action::Imported,
                );
            }
            Action::Imported(result) => {
                self.busy = false;
                match result {
                    Ok(report) => {
                        self.plan = None;
                        self.result = Some(format!(
                            "Imported {} conversations as new copies. Recovery files: {}. Your current chat and workspace selection were retained.",
                            report.sessions,
                            report.recovery_directory.display()
                        ));
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            _ => {}
        }
        Task::none()
    }
    fn begin(&mut self, operation: &'static str) {
        self.busy = true;
        self.operation = operation;
        self.result = None;
        self.error = None;
    }
}

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let state = &app.backup;
    let p = ui::palette(app.dark);
    let running = app.running || app.scheduled.has_running_jobs();
    let mut content=column![
        heading("archive","Back up this agent",app.dark),
        row![text("Save the native profile to one private file. Includes conversations, attachment snapshots, imported-history copies, saved drafts, project metadata, and recovery copies of native settings, memory and schedules.").size(12).color(p.muted).width(Fill),
            button(text(if state.busy&&state.operation=="Backing up…"{"Backing up…"}else{"Back up"}).size(12)).on_press_maybe((!state.busy&&!running).then_some(Message::Backup(Action::Export))).padding([7,12]).style(button::secondary)
        ].spacing(16).align_y(Alignment::Start),
        text("Destination").size(12),
        row![text_input("Choose a new .hermes-native.json file",&state.export_path).on_input(|path|Message::Backup(Action::ExportPath(path))).padding(8).size(12),
            button(text("Browse…").size(12)).on_press_maybe((!state.busy).then_some(Message::Backup(Action::BrowseExport))).padding([7,10]).style(button::secondary)
        ].spacing(8),
        text("Saved provider keys, Nous logins, Telegram credentials/inbox and workspace files are excluded. Conversation and recovery content can contain private information; the archive is not encrypted. Existing destination files are retained.").size(12).color(p.muted),
        Space::new().height(14),
        heading("folder-opened","Import a native backup",app.dark),
        text("Inspect the file, then add conversations as new copies. Existing sessions, credentials and settings stay in place. Imported conversations use new empty workspaces; reselect a project manually when needed. Settings, memory, schedule definitions and queued prompts are kept as recovery files and never activated.").size(12).color(p.muted),
        row![text_input("Select a .hermes-native.json backup",&state.source_path).on_input(|path|Message::Backup(Action::SourcePath(path))).padding(8).size(12),
            button(text("Browse…").size(12)).on_press_maybe((!state.busy).then_some(Message::Backup(Action::BrowseSource))).padding([7,10]).style(button::secondary),
            button(text("Inspect").size(12)).on_press_maybe((!state.busy).then_some(Message::Backup(Action::Inspect))).padding([7,10]).style(button::secondary)
        ].spacing(8),
    ].spacing(12);
    if let Some(plan) = &state.plan {
        let summary = plan.summary();
        content = content.push(
            row![
                column![
                    text(format!(
                        "{} conversations · {} files · {:.1} MiB",
                        summary.sessions,
                        summary.files,
                        summary.content_bytes as f64 / (1024.0 * 1024.0)
                    ))
                    .size(13)
                    .font(semibold()),
                    text(format!(
                        "Created {} · {} project references",
                        summary.created_at, summary.projects
                    ))
                    .size(12)
                    .color(p.muted)
                ]
                .spacing(4)
                .width(Fill),
                button(text("Import copies").size(12))
                    .on_press_maybe(
                        (!state.busy && !running).then_some(Message::Backup(Action::Import))
                    )
                    .padding([7, 12])
            ]
            .spacing(16)
            .align_y(Alignment::Center),
        );
    }
    if state.busy {
        content = content.push(text(state.operation).size(12).color(p.muted));
    }
    if let Some(error) = &state.error {
        content = content.push(
            text(error)
                .size(12)
                .color(iced::Color::from_rgb8(190, 70, 70)),
        );
    }
    if let Some(result) = &state.result {
        content = content.push(text(result).size(12).color(p.muted));
    }
    content=content.push(Space::new().height(12)).push(text("Native archive v1 (.hermes-native.json). This covers the files listed above; it is distinct from the stock whole-Hermes-home backup and Hermes Cloud transfer formats.").size(11).color(p.muted));
    scrollable(
        container(content)
            .padding(Padding {
                top: 4.0,
                right: 32.0,
                bottom: 80.0,
                left: 32.0,
            })
            .width(Fill),
    )
    .height(Fill)
    .into()
}
fn heading<'a>(icon: &'static str, label: &'a str, dark: bool) -> Element<'a, Message> {
    let p = ui::palette(dark);
    row![
        ui::icon(icon, 16, p.muted),
        text(label).size(14).font(semibold())
    ]
    .spacing(8)
    .align_y(Alignment::Center)
    .into()
}
fn semibold() -> Font {
    Font {
        weight: iced::font::Weight::Semibold,
        ..crate::appearance::font()
    }
}

fn choose_source() -> Result<Option<PathBuf>, String> {
    crate::picker::choose("Select a native Hermes backup to inspect", false)
}
#[cfg(not(target_os = "macos"))]
fn choose_destination() -> Result<Option<PathBuf>, String> {
    crate::picker::save("Back up Hermes", "Hermes.hermes-native.json")
}
#[cfg(target_os = "macos")]
fn choose_destination() -> Result<Option<PathBuf>, String> {
    use objc2_app_kit::{NSModalResponseOK, NSSavePanel};
    use objc2_foundation::{MainThreadMarker, NSString};
    use std::{
        ffi::{CStr, OsStr},
        os::unix::ffi::OsStrExt,
    };
    fn show(main: MainThreadMarker) -> Result<Option<PathBuf>, String> {
        unsafe {
            let panel = NSSavePanel::savePanel(main);
            panel.setMessage(Some(&NSString::from_str(
                "Save a native Hermes profile backup to a new file",
            )));
            panel.setNameFieldStringValue(&NSString::from_str(&format!(
                "hermes-{}.hermes-native.json",
                chrono::Utc::now().format("%Y-%m-%d")
            )));
            if panel.runModal() != NSModalResponseOK {
                return Ok(None);
            }
            let url = panel.URL().ok_or("The save panel returned no location")?;
            let path = CStr::from_ptr(url.fileSystemRepresentation().as_ptr());
            Ok(Some(PathBuf::from(OsStr::from_bytes(path.to_bytes()))))
        }
    }
    if let Some(main) = MainThreadMarker::new() {
        return show(main);
    }
    dispatch::Queue::main().exec_sync(move || {
        show(MainThreadMarker::new().ok_or("Could not open the save panel on the main thread")?)
    })
}
