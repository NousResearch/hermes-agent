//! Native stock repository discovery: 0284ce17 electron/git-repo-scan.ts,
//! store/projects.ts and settings/constants.ts. Filesystem traversal runs only
//! on a bounded blocking worker after the user opens Projects or chooses Scan.
//! Empty configured roots use registered projects; no implicit home traversal.
use crate::{Hermes, Message, appearance::text, settings::View, ui};
use fs2::FileExt;
use hermes_library::Project;
use iced::{
    Alignment, Element, Fill, Font, Length, Task,
    widget::{Space, button, checkbox, column, container, row, scrollable, text_editor},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeSet, VecDeque},
    fmt,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

const MAX_DEPTH: usize = 3;
const MAX_DIRECTORIES: usize = 5_000;
const MAX_ENTRIES: usize = 50_000;
const MAX_CANDIDATES: usize = 256;
const MAX_DURATION: Duration = Duration::from_secs(6);
const MAX_PATHS: usize = 64;
const JUNK: &[&str] = &[
    "Applications",
    "Library",
    "node_modules",
    "site-packages",
    "vendor",
    "venv",
];

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub(super) struct Policy {
    #[serde(rename = "repo_scan_enabled")]
    enabled: bool,
    #[serde(rename = "repo_scan_roots")]
    roots: Vec<PathBuf>,
    #[serde(rename = "repo_scan_exclude_paths")]
    exclude_paths: Vec<PathBuf>,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            enabled: true,
            roots: vec![],
            exclude_paths: vec![],
        }
    }
}
#[derive(Serialize, Deserialize)]
struct Settings {
    version: u32,
    desktop: Policy,
}
#[derive(Clone)]
struct Candidate {
    path: PathBuf,
    label: String,
}
pub(super) struct Report {
    candidates: Vec<Candidate>,
    directories: usize,
    unreadable: usize,
    roots: usize,
    limited: bool,
}

#[derive(Clone)]
pub(super) enum Action {
    Enabled(bool),
    Roots(text_editor::Action),
    Exclusions(text_editor::Action),
    BrowseRoot,
    RootChosen(Result<Option<PathBuf>, String>),
    Apply,
    Saved(Result<Policy, String>),
    Scan,
    Cancel,
    Scanned(u64, Result<Arc<Report>, String>),
    Select(PathBuf),
}
impl fmt::Debug for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Enabled(_) => "RepositoryDiscoveryEnabled",
            Self::Roots(_) => "RepositoryDiscoveryRoots",
            Self::Exclusions(_) => "RepositoryDiscoveryExclusions",
            Self::BrowseRoot => "ChooseRepositoryDiscoveryRoot",
            Self::RootChosen(_) => "RepositoryDiscoveryRootChosen",
            Self::Apply => "SaveRepositoryDiscovery",
            Self::Saved(_) => "RepositoryDiscoverySaved",
            Self::Scan => "ScanRepositories",
            Self::Cancel => "CancelRepositoryScan",
            Self::Scanned(..) => "RepositoriesScanned",
            Self::Select(_) => "SelectDiscoveredProject",
        })
    }
}

pub(super) struct State {
    path: Option<PathBuf>,
    saved: Policy,
    enabled: bool,
    roots: text_editor::Content,
    exclusions: text_editor::Content,
    dirty: bool,
    saving: bool,
    scanning: bool,
    generation: u64,
    cancel: Option<Arc<AtomicBool>>,
    report: Option<Arc<Report>>,
    signature: Option<String>,
    running_signature: Option<String>,
    error: Option<String>,
    note: Option<String>,
}
impl Default for State {
    fn default() -> Self {
        Self {
            path: None,
            saved: Policy::default(),
            enabled: true,
            roots: text_editor::Content::new(),
            exclusions: text_editor::Content::new(),
            dirty: false,
            saving: false,
            scanning: false,
            generation: 0,
            cancel: None,
            report: None,
            signature: None,
            running_signature: None,
            error: None,
            note: None,
        }
    }
}
impl Drop for State {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
    }
}
impl State {
    pub(super) fn load(root: &Path) -> Self {
        let mut state = Self::default();
        state.path = Some(root.join("project-discovery.json"));
        match load_settings(state.path.as_ref().unwrap()) {
            Ok(Some(policy)) => state.install(policy),
            Ok(None) => {}
            Err(error) => {
                state.saved.enabled = false;
                state.enabled = false;
                state.error = Some(error);
            }
        }
        state
    }
    fn install(&mut self, policy: Policy) {
        self.enabled = policy.enabled;
        self.roots = text_editor::Content::with_text(&paths_text(&policy.roots));
        self.exclusions = text_editor::Content::with_text(&paths_text(&policy.exclude_paths));
        self.saved = policy;
        self.dirty = false;
    }
    pub(super) fn stop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.scanning = false;
        self.generation = self.generation.wrapping_add(1);
        self.running_signature = None;
    }
    pub(super) fn refresh(&mut self, projects: Vec<Project>, force: bool) -> Task<Action> {
        if !self.saved.enabled {
            self.stop();
            self.report = None;
            self.note = Some("Automatic repository discovery is disabled.".into());
            return Task::none();
        }
        if self.saving {
            return Task::none();
        }
        if self.saved.roots.is_empty() && projects.is_empty() {
            self.note=Some("Choose repository discovery roots in Settings → Workspace, or add a project folder.".into());
            return Task::none();
        }
        let paths = projects
            .iter()
            .map(|project| project.path.clone())
            .collect::<Vec<_>>();
        let signature = serde_json::to_string(&(&self.saved, paths)).unwrap_or_default();
        if !force
            && (self.signature.as_ref() == Some(&signature)
                || self.running_signature.as_ref() == Some(&signature))
        {
            return Task::none();
        }
        self.stop();
        let generation = self.generation;
        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel = Some(cancel.clone());
        self.scanning = true;
        self.running_signature = Some(signature);
        self.error = None;
        self.note = None;
        let policy = self.saved.clone();
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || scan(policy, projects, cancel).map(Arc::new))
                    .await
                    .map_err(|_| "Repository discovery worker stopped".to_owned())
                    .and_then(|result| result)
            },
            move |result| Action::Scanned(generation, result),
        )
    }
    pub(super) fn update(
        &mut self,
        action: Action,
        projects: Vec<Project>,
    ) -> (Task<Action>, Option<PathBuf>) {
        match action {
            Action::Enabled(enabled) if !self.saving => {
                self.enabled = enabled;
                self.dirty = true;
            }
            Action::Roots(action) if !self.saving => {
                let edit = action.is_edit();
                self.roots.perform(action);
                self.dirty |= edit;
            }
            Action::Exclusions(action) if !self.saving => {
                let edit = action.is_edit();
                self.exclusions.perform(action);
                self.dirty |= edit;
            }
            Action::BrowseRoot if !self.saving => {
                return (
                    Task::perform(
                        async {
                            tokio::task::spawn_blocking(choose_root)
                                .await
                                .map_err(|_| "Could not open the folder picker".to_owned())
                                .and_then(|result| result)
                        },
                        Action::RootChosen,
                    ),
                    None,
                );
            }
            Action::RootChosen(result) => match result {
                Ok(Some(path)) => {
                    let mut roots = self.roots.text();
                    if !roots.trim().is_empty() {
                        roots.push('\n');
                    }
                    roots.push_str(&path.to_string_lossy());
                    self.roots = text_editor::Content::with_text(&roots);
                    self.dirty = true;
                }
                Ok(None) => {}
                Err(error) => self.error = Some(error),
            },
            Action::Apply if !self.saving => {
                let policy =
                    match parse_policy(self.enabled, &self.roots.text(), &self.exclusions.text()) {
                        Ok(policy) => policy,
                        Err(error) => {
                            self.error = Some(error);
                            return (Task::none(), None);
                        }
                    };
                let Some(path) = self.path.clone() else {
                    self.error = Some("Native profile is unavailable".into());
                    return (Task::none(), None);
                };
                self.stop();
                self.saving = true;
                self.error = None;
                self.note = None;
                return (
                    Task::perform(
                        async move {
                            tokio::task::spawn_blocking(move || {
                                save_settings(&path, &policy)?;
                                Ok(policy)
                            })
                            .await
                            .map_err(|_| "Could not save repository discovery settings".to_owned())
                            .and_then(|result| result)
                        },
                        Action::Saved,
                    ),
                    None,
                );
            }
            Action::Saved(result) => {
                self.saving = false;
                match result {
                    Ok(policy) => {
                        self.install(policy);
                        self.signature = None;
                        self.report = None;
                        self.note = Some("Repository discovery settings saved.".into());
                        return (self.refresh(projects, true), None);
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            Action::Scan => {
                if self.dirty {
                    self.error = Some("Apply discovery settings before scanning.".into());
                } else {
                    return (self.refresh(projects, true), None);
                }
            }
            Action::Cancel => {
                self.stop();
                self.note = Some("Repository discovery stopped.".into());
            }
            Action::Scanned(generation, result) if generation == self.generation => {
                self.scanning = false;
                self.cancel = None;
                match result {
                    Ok(report) => {
                        self.signature = self.running_signature.take();
                        self.report = Some(report);
                    }
                    Err(error) => {
                        self.running_signature = None;
                        self.error = Some(error)
                    }
                }
            }
            Action::Select(path) => {
                if self.report.as_ref().is_some_and(|report| {
                    report
                        .candidates
                        .iter()
                        .any(|candidate| candidate.path == path)
                }) {
                    return (Task::none(), Some(path));
                }
            }
            _ => {}
        }
        (Task::none(), None)
    }
}

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let state = &app.project_discovery;
    let p = ui::palette(app.dark);
    let enabled = row![
        column![
            text("Automatic Repository Discovery").size(13),
            text("Scan local folders for Git repositories to show in Projects.")
                .size(12)
                .color(p.muted)
        ]
        .spacing(3)
        .width(Fill),
        checkbox(state.enabled)
            .on_toggle(|enabled| Message::ProjectDiscovery(Action::Enabled(enabled)))
    ]
    .spacing(16)
    .align_y(Alignment::Center);
    let mut body = column![
        text("Repository discovery").size(16).font(semibold()),
        enabled,
        row![
            column![
                text("Repository Discovery Roots").size(13),
                text("Folders to scan, one per line. Empty uses only registered projects.")
                    .size(12)
                    .color(p.muted)
            ]
            .spacing(3)
            .width(Fill),
            button(text("Browse…").size(12))
                .on_press_maybe(
                    (!state.saving).then_some(Message::ProjectDiscovery(Action::BrowseRoot))
                )
                .padding([5, 10])
                .style(button::secondary)
        ]
        .spacing(16),
        text_editor(&state.roots)
            .on_action(|action| Message::ProjectDiscovery(Action::Roots(action)))
            .height(76)
            .size(12)
            .padding(8),
        column![
            text("Excluded Repository Paths").size(13),
            text(
                "Folders and their descendants to skip during repository discovery, one per line."
            )
            .size(12)
            .color(p.muted)
        ]
        .spacing(3),
        text_editor(&state.exclusions)
            .on_action(|action| Message::ProjectDiscovery(Action::Exclusions(action)))
            .height(64)
            .size(12)
            .padding(8),
        row![
            button(
                text(if state.scanning {
                    "Stop scan"
                } else {
                    "Rescan"
                })
                .size(12)
            )
            .on_press_maybe((!state.saving).then_some(Message::ProjectDiscovery(
                if state.scanning {
                    Action::Cancel
                } else {
                    Action::Scan
                }
            )))
            .padding([6, 10])
            .style(button::secondary),
            Space::new().width(Fill),
            button(text(if state.saving { "Applying…" } else { "Apply" }).size(12))
                .on_press_maybe(
                    (state.dirty && !state.saving)
                        .then_some(Message::ProjectDiscovery(Action::Apply))
                )
                .padding([6, 12])
        ]
        .spacing(8),
    ]
    .spacing(12);
    if let Some(status) = status(state) {
        body = body.push(text(status).size(12).color(p.muted));
    }
    if let Some(error) = &state.error {
        body = body.push(
            text(error)
                .size(12)
                .color(iced::Color::from_rgb8(190, 70, 70)),
        );
    }
    body.into()
}
pub(super) fn suggestions(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let state = &app.project_discovery;
    let header = row![
        text("Repositories").size(12).font(semibold()).width(Fill),
        button(ui::icon(
            if state.scanning { "close" } else { "refresh" },
            13,
            p.muted
        ))
        .on_press(Message::ProjectDiscovery(if state.scanning {
            Action::Cancel
        } else {
            Action::Scan
        }))
        .padding(4)
        .style(button::text)
    ]
    .align_y(Alignment::Center);
    let mut rows = column![].spacing(4);
    let mut count = 0;
    if let Some(report) = &state.report {
        for candidate in &report.candidates {
            if app
                .projects
                .iter()
                .any(|project| project.path == candidate.path)
            {
                continue;
            }
            count += 1;
            let path = candidate.path.clone();
            rows = rows.push(
                button(
                    row![
                        ui::icon("folder", 14, p.muted),
                        column![
                            text(&candidate.label).size(12),
                            text(candidate.path.to_string_lossy().into_owned())
                                .size(10)
                                .color(p.muted)
                        ]
                        .spacing(2)
                        .width(Fill),
                        ui::icon("add", 12, p.muted)
                    ]
                    .spacing(8)
                    .align_y(Alignment::Center),
                )
                .on_press_maybe(
                    (!app.running).then_some(Message::ProjectDiscovery(Action::Select(path))),
                )
                .padding([6, 8])
                .width(Fill)
                .style(button::text),
            );
        }
    }
    let mut body = column![header].spacing(6);
    if count > 0 {
        body = body.push(scrollable(rows).height(Length::Shrink).height(160));
    }
    if let Some(status) = status(state) {
        body = body.push(text(status).size(11).color(p.muted));
    }
    if !state.scanning && count == 0 && state.report.is_some() {
        body = body.push(text("No new repositories found.").size(12).color(p.muted));
    }
    if let Some(error) = &state.error {
        body = body.push(text(error).size(11).color(p.muted));
    }
    body = body.push(
        button(text("Discovery settings").size(11))
            .on_press(Message::SettingsView(View::Workspace))
            .padding([2, 0])
            .style(button::text),
    );
    container(body).width(Fill).into()
}
fn status(state: &State) -> Option<String> {
    if state.scanning {
        return Some("Scanning configured folders…".into());
    }
    if let Some(report) = &state.report {
        return Some(format!(
            "{} repositories · {} folders across {} roots{}{}",
            report.candidates.len(),
            report.directories,
            report.roots,
            if report.limited {
                " · scan limit reached"
            } else {
                ""
            },
            if report.unreadable > 0 {
                format!(" · {} folders unreadable", report.unreadable)
            } else {
                String::new()
            }
        ));
    }
    state.note.clone()
}
fn semibold() -> Font {
    Font {
        weight: iced::font::Weight::Semibold,
        ..crate::appearance::font()
    }
}
fn paths_text(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.to_string_lossy())
        .collect::<Vec<_>>()
        .join("\n")
}
fn parse_policy(enabled: bool, roots: &str, exclusions: &str) -> Result<Policy, String> {
    Ok(Policy {
        enabled,
        roots: parse_paths(roots)?,
        exclude_paths: parse_paths(exclusions)?,
    })
}
fn parse_paths(value: &str) -> Result<Vec<PathBuf>, String> {
    let mut paths = BTreeSet::new();
    for line in value.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let path = normalize(line)?;
        paths.insert(path);
        if paths.len() > MAX_PATHS {
            return Err("Choose up to 64 discovery roots or excluded folders.".into());
        }
    }
    Ok(paths.into_iter().collect())
}
fn normalize(value: &str) -> Result<PathBuf, String> {
    let value = value.trim();
    if value.len() > 4096 || value.chars().any(char::is_control) {
        return Err("Discovery paths must be full folder paths without control characters.".into());
    }
    let path = if value == "~" || value.starts_with("~/") {
        let home = std::env::var_os("HOME").ok_or("Enter an absolute discovery path")?;
        PathBuf::from(home).join(value.trim_start_matches('~').trim_start_matches('/'))
    } else {
        PathBuf::from(value)
    };
    if !path.is_absolute() {
        return Err("Enter full discovery paths, one per line.".into());
    }
    let mut normalized = PathBuf::new();
    for part in path.components() {
        match part {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            part => normalized.push(part.as_os_str()),
        }
    }
    Ok(normalized)
}
fn load_settings(path: &Path) -> Result<Option<Policy>, String> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    if !metadata.file_type().is_file() || metadata.len() > 128 * 1024 {
        return Err(
            "Repository discovery settings must be a regular file smaller than 128 KiB.".into(),
        );
    }
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|file| file.take(128 * 1024 + 1).read_to_end(&mut bytes))
        .map_err(|error| error.to_string())?;
    let settings: Settings = serde_json::from_slice(&bytes)
        .map_err(|_| "Could not read repository discovery settings".to_owned())?;
    if settings.version != 1 {
        return Err("Unsupported repository discovery settings version".into());
    }
    parse_policy(
        settings.desktop.enabled,
        &paths_text(&settings.desktop.roots),
        &paths_text(&settings.desktop.exclude_paths),
    )
    .map(Some)
}
fn save_settings(path: &Path, policy: &Policy) -> Result<(), String> {
    let parent = path.parent().ok_or("Native profile has no folder")?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(parent.join("project-discovery.lock"))
        .map_err(|error| error.to_string())?;
    lock.lock_exclusive().map_err(|error| error.to_string())?;
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
    serde_json::to_writer(
        file.as_file_mut(),
        &Settings {
            version: 1,
            desktop: policy.clone(),
        },
    )
    .map_err(|error| error.to_string())?;
    file.write_all(b"\n")
        .and_then(|_| file.as_file().sync_all())
        .map_err(|error| error.to_string())?;
    file.persist(path)
        .map_err(|_| "Could not save repository discovery settings".to_owned())?;
    #[cfg(unix)]
    File::open(parent)
        .and_then(|folder| folder.sync_all())
        .map_err(|error| error.to_string())?;
    Ok(())
}

/// Filesystem-only detector: never launches Git, reads repository source,
/// follows child symlinks or traverses beyond the explicitly selected roots.
fn scan(policy: Policy, projects: Vec<Project>, cancel: Arc<AtomicBool>) -> Result<Report, String> {
    let mut report = Report {
        candidates: vec![],
        directories: 0,
        unreadable: 0,
        roots: 0,
        limited: false,
    };
    if !policy.enabled {
        return Ok(report);
    }
    let requested = if policy.roots.is_empty() {
        projects
            .iter()
            .map(|project| project.path.clone())
            .collect()
    } else {
        policy.roots
    };
    let mut roots = BTreeSet::new();
    for path in requested {
        if cancel.load(Ordering::Relaxed) {
            return Err("Repository discovery stopped".into());
        }
        match path.canonicalize() {
            Ok(path) if path.is_dir() => {
                roots.insert(path);
            }
            _ => report.unreadable += 1,
        }
    }
    report.roots = roots.len();
    let exclusions = policy
        .exclude_paths
        .into_iter()
        .map(|path| path.canonicalize().unwrap_or(path))
        .collect::<Vec<_>>();
    let registered = projects
        .into_iter()
        .map(|project| project.path.canonicalize().unwrap_or(project.path))
        .collect::<BTreeSet<_>>();
    let mut queue = roots
        .into_iter()
        .map(|root| (root, 0))
        .collect::<VecDeque<_>>();
    let mut visited = BTreeSet::new();
    let mut entries_seen = 0;
    let started = Instant::now();
    while let Some((directory, depth)) = queue.pop_front() {
        if cancel.load(Ordering::Relaxed) {
            return Err("Repository discovery stopped".into());
        }
        if report.directories >= MAX_DIRECTORIES
            || entries_seen >= MAX_ENTRIES
            || started.elapsed() >= MAX_DURATION
            || report.candidates.len() >= MAX_CANDIDATES
        {
            report.limited = true;
            break;
        }
        if depth > MAX_DEPTH
            || exclusions
                .iter()
                .any(|excluded| directory.starts_with(excluded))
            || !visited.insert(directory.clone())
        {
            continue;
        }
        report.directories += 1;
        if !fs::symlink_metadata(&directory).is_ok_and(|metadata| metadata.file_type().is_dir()) {
            report.unreadable += 1;
            continue;
        }
        let git = directory.join(".git");
        if fs::symlink_metadata(&git).is_ok_and(|metadata| metadata.file_type().is_dir())
            && fs::symlink_metadata(git.join("HEAD"))
                .is_ok_and(|metadata| metadata.file_type().is_file())
            && File::open(git.join("HEAD")).is_ok()
        {
            if !registered.contains(&directory) {
                report.candidates.push(Candidate {
                    label: directory
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_else(|| directory.display().to_string()),
                    path: directory,
                });
            }
            continue;
        }
        if depth == MAX_DEPTH {
            continue;
        }
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(_) => {
                report.unreadable += 1;
                continue;
            }
        };
        let mut children = Vec::new();
        for entry in entries {
            if cancel.load(Ordering::Relaxed) {
                return Err("Repository discovery stopped".into());
            }
            entries_seen += 1;
            if entries_seen >= MAX_ENTRIES || started.elapsed() >= MAX_DURATION {
                report.limited = true;
                break;
            }
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => {
                    report.unreadable += 1;
                    continue;
                }
            };
            let kind = match entry.file_type() {
                Ok(kind) => kind,
                Err(_) => {
                    report.unreadable += 1;
                    continue;
                }
            };
            if !kind.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || JUNK.contains(&name.as_ref()) {
                continue;
            }
            #[cfg(target_os = "macos")]
            {
                let lower = name.to_lowercase();
                if depth == 0 && ["Movies", "Music", "Pictures", "Public"].contains(&name.as_ref())
                    || [
                        ".photoslibrary",
                        ".musiclibrary",
                        ".tvlibrary",
                        ".aplibrary",
                    ]
                    .iter()
                    .any(|suffix| lower.ends_with(suffix))
                {
                    continue;
                }
            }
            children.push(entry.path());
        }
        children.sort();
        queue.extend(children.into_iter().map(|child| (child, depth + 1)));
    }
    report.candidates.sort_by(|a, b| {
        a.label
            .to_lowercase()
            .cmp(&b.label.to_lowercase())
            .then_with(|| a.path.cmp(&b.path))
    });
    Ok(report)
}
fn choose_root() -> Result<Option<PathBuf>, String> {
    super::choose_folder()
}
