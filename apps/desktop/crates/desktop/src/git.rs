//! Native, read-only worktree inspection. Git runs on a blocking worker with
//! literal arguments, bounded output and a deadline. No shell is involved.
//!
//! Ported from the upstream right-sidebar Review pane: compact chrome, changed
//! file rows and the selected diff below them. The implemented scope is
//! Uncommitted. Remaining upstream UI gaps are tree/list switching, Branch and
//! Last turn scopes, per-file line counts, syntax highlighting, stage/revert,
//! and the commit/push/PR ship bar. These are absent rather than inert controls.

use crate::appearance::text;
use iced::widget::text as text_widget;
use iced::{
    Color, Element, Font, Length, Task, Theme,
    widget::{button, column, container, row, rule, scrollable, tooltip},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
    time::{Duration, Instant},
};

const MAX_STATUS: usize = 2_097_152;
const MAX_DIFF: usize = 524_288;
const MAX_FILES: usize = 500;
const DEADLINE: Duration = Duration::from_secs(12);

#[cfg(target_os = "macos")]
const GIT: &str = "/usr/bin/git";
#[cfg(not(target_os = "macos"))]
const GIT: &str = "git";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiffMode {
    #[default]
    WorkingTree,
    Staged,
}

#[derive(Debug, Clone)]
pub enum GitMessage {
    Refresh,
    Select(Option<usize>),
    Close,
    Loaded(u64, Result<Arc<Snapshot>, String>),
    DiffLoaded(u64, u64, Result<String, String>),
}

pub struct GitPanel {
    workspace: Option<PathBuf>,
    snapshot: Option<Arc<Snapshot>>,
    selected: Option<usize>,
    mode: DiffMode,
    diff: String,
    error: Option<String>,
    loading: bool,
    diff_loading: bool,
    generation: u64,
    diff_generation: u64,
}

impl GitPanel {
    pub fn new(workspace: PathBuf) -> Self {
        Self {
            workspace: Some(workspace),
            snapshot: None,
            selected: None,
            mode: DiffMode::WorkingTree,
            diff: String::new(),
            error: None,
            loading: false,
            diff_loading: false,
            generation: 0,
            diff_generation: 0,
        }
    }

    /// Discard old worktree state. Refresh is deliberately a separate user action.
    pub fn reset(&mut self, workspace: Option<PathBuf>) -> Task<GitMessage> {
        self.workspace = workspace;
        self.snapshot = None;
        self.selected = None;
        self.diff.clear();
        self.error = None;
        self.loading = false;
        self.diff_loading = false;
        self.generation = self.generation.wrapping_add(1);
        self.diff_generation = self.diff_generation.wrapping_add(1);
        Task::none()
    }

    pub fn update(&mut self, message: GitMessage) -> Task<GitMessage> {
        match message {
            GitMessage::Refresh if !self.loading => {
                let Some(workspace) = self.workspace.clone() else {
                    return Task::none();
                };
                self.generation = self.generation.wrapping_add(1);
                self.diff_generation = self.diff_generation.wrapping_add(1);
                let generation = self.generation;
                self.loading = true;
                self.diff_loading = false;
                self.error = None;
                self.diff.clear();
                self.selected = None;
                return Task::perform(
                    async move {
                        tokio::task::spawn_blocking(move || load_snapshot(&workspace).map(Arc::new))
                            .await
                            .unwrap_or_else(|error| Err(format!("Git worker stopped: {error}")))
                    },
                    move |result| GitMessage::Loaded(generation, result),
                );
            }
            GitMessage::Loaded(generation, result) if generation == self.generation => {
                self.loading = false;
                match result {
                    Ok(snapshot) => {
                        self.snapshot = Some(snapshot);
                        return self.begin_diff();
                    }
                    Err(error) => {
                        self.snapshot = None;
                        self.error = Some(error);
                    }
                }
            }
            GitMessage::Select(selected) if !self.loading => {
                self.selected = selected.filter(|&index| {
                    self.snapshot
                        .as_ref()
                        .is_some_and(|snapshot| index < snapshot.changes.len())
                });
                if let Some(change) = self
                    .selected
                    .and_then(|index| self.snapshot.as_ref()?.changes.get(index))
                {
                    self.mode = if change.staged() {
                        DiffMode::Staged
                    } else {
                        DiffMode::WorkingTree
                    };
                }
                return self.begin_diff();
            }
            GitMessage::DiffLoaded(generation, request, result)
                if generation == self.generation && request == self.diff_generation =>
            {
                self.diff_loading = false;
                match result {
                    Ok(diff) => {
                        self.diff = diff;
                        self.error = None;
                    }
                    Err(error) => {
                        self.diff.clear();
                        self.error = Some(error);
                    }
                }
            }
            _ => {}
        }
        Task::none()
    }

    fn begin_diff(&mut self) -> Task<GitMessage> {
        if self.selected.is_none() {
            self.diff_generation = self.diff_generation.wrapping_add(1);
            self.diff.clear();
            self.diff_loading = false;
            return Task::none();
        }
        let Some(snapshot) = self.snapshot.clone() else {
            return Task::none();
        };
        self.diff_generation = self.diff_generation.wrapping_add(1);
        let (generation, request, mode, selected) = (
            self.generation,
            self.diff_generation,
            self.mode,
            self.selected,
        );
        self.diff_loading = true;
        self.error = None;
        self.diff.clear();
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || load_diff(&snapshot, selected, mode))
                    .await
                    .unwrap_or_else(|error| Err(format!("Git worker stopped: {error}")))
            },
            move |result| GitMessage::DiffLoaded(generation, request, result),
        )
    }

    /// The upstream Review pane: compact header, change list above a selected
    /// file's diff. The shell supplies its right-hand placement and close action.
    pub fn view(&self) -> Element<'_, GitMessage> {
        let refresh = tooltip(
            button(text("⟳").size(14))
                .width(20)
                .height(20)
                .padding(0)
                .style(icon_button)
                .on_press_maybe(
                    (!self.loading && self.workspace.is_some()).then_some(GitMessage::Refresh),
                ),
            text("Refresh tree").size(11),
            tooltip::Position::Bottom,
        );
        let close = tooltip(
            button(text("×").size(15))
                .width(20)
                .height(20)
                .padding(0)
                .style(icon_button)
                .on_press(GitMessage::Close),
            text("Close").size(11),
            tooltip::Position::Bottom,
        );
        let heading = container(
            row![
                text("▦  REVIEW")
                    .size(10)
                    .style(|theme: &Theme| text_widget::Style {
                        color: Some(theme.palette().primary)
                    }),
                iced::widget::Space::new().width(Length::Fill),
                refresh,
                close,
            ]
            .spacing(3)
            .align_y(iced::Alignment::Center),
        )
        .padding([0, 10])
        .height(28);
        let mut content = column![heading].width(Length::Fill).height(Length::Fill);
        if let Some(error) = &self.error {
            content = content.push(container(text(error).size(11)).padding([6, 10]));
        }
        let Some(snapshot) = &self.snapshot else {
            let message = if self.loading {
                "Loading…"
            } else {
                "▦  NO DIFFS"
            };
            content = content.push(
                container(text(message).size(10))
                    .center_x(Length::Fill)
                    .center_y(Length::Fill)
                    .height(Length::Fill),
            );
            return container(content)
                .style(review_surface)
                .width(Length::Fill)
                .height(Length::Fill)
                .into();
        };
        // Branch and last-turn scopes need their upstream data sources. Keep
        // the working-tree label faithful and expose only implemented behavior.
        content = content.push(
            container(text("Uncommitted").size(11))
                .padding([5, 10])
                .width(Length::Fill),
        );
        if snapshot.truncated || snapshot.omitted > 0 {
            content = content.push(
                container(
                    text(format!(
                        "Showing up to {MAX_FILES} files; {} omitted.",
                        snapshot.omitted
                    ))
                    .size(10),
                )
                .padding([3, 10]),
            );
        }
        let selected = self.selected.and_then(|index| snapshot.changes.get(index));
        if snapshot.changes.is_empty() {
            content = content.push(
                container(text("▦  NO DIFFS").size(10))
                    .center_x(Length::Fill)
                    .center_y(Length::Fill)
                    .height(Length::Fill),
            );
        } else {
            let mut files = column![].spacing(0).padding([4, 4]);
            for (index, change) in snapshot.changes.iter().enumerate() {
                let path = Path::new(&change.path);
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                let directory = path
                    .parent()
                    .filter(|path| !path.as_os_str().is_empty())
                    .map(|path| label(&path.to_string_lossy()))
                    .unwrap_or_default();
                let status = change.status.trim();
                let glyph = if change.untracked() || status.contains('A') {
                    "+"
                } else if status.contains('D') {
                    "−"
                } else if status.contains('R') {
                    "↪"
                } else {
                    "◇"
                };
                let color = match glyph {
                    "+" => Color::from_rgb8(46, 160, 67),
                    "−" => Color::from_rgb8(207, 34, 46),
                    "↪" => Color::from_rgb8(56, 139, 253),
                    _ => Color::from_rgb8(191, 135, 0),
                };
                let label = row![
                    text(glyph).size(12).width(13).color(color),
                    text(label(&name))
                        .size(12)
                        .wrapping(text_widget::Wrapping::None),
                    text(directory)
                        .size(10)
                        .wrapping(text_widget::Wrapping::None),
                    iced::widget::Space::new().width(Length::Fill),
                    text(if change.staged() { "S" } else { "" }).size(10),
                ]
                .spacing(5)
                .align_y(iced::Alignment::Center);
                files = files.push(
                    button(label)
                        .height(24)
                        .padding([0, 6])
                        .width(Length::Fill)
                        .on_press_maybe((!self.loading).then_some(GitMessage::Select(Some(index))))
                        .style(move |theme, status| {
                            file_button(theme, status, self.selected == Some(index))
                        }),
                );
            }
            content = content.push(
                scrollable(files)
                    .height(if selected.is_some() {
                        Length::FillPortion(45)
                    } else {
                        Length::Fill
                    })
                    .width(Length::Fill),
            );
        }
        if let Some(selected) = selected {
            let diff_heading = container(
                row![
                    text(label(&selected.path))
                        .size(10.5)
                        .font(Font::MONOSPACE)
                        .width(Length::Fill)
                        .wrapping(text_widget::Wrapping::None),
                    button(text("×").size(14))
                        .width(20)
                        .height(20)
                        .padding(0)
                        .style(icon_button)
                        .on_press(GitMessage::Select(None)),
                ]
                .spacing(4)
                .align_y(iced::Alignment::Center),
            )
            .padding([4, 10]);
            let preview = if self.diff_loading {
                "Loading…"
            } else if self.diff.is_empty() {
                "No diff to show"
            } else {
                &self.diff
            };
            let diff = scrollable(
                text(preview)
                    .font(Font::MONOSPACE)
                    .size(11)
                    .wrapping(text_widget::Wrapping::None),
            )
            .direction(scrollable::Direction::Both {
                vertical: Default::default(),
                horizontal: Default::default(),
            })
            .height(Length::Fill)
            .width(Length::Fill);
            content = content.push(rule::horizontal(1)).push(
                column![diff_heading, container(diff).padding([0, 4])]
                    .height(Length::FillPortion(55)),
            );
        }
        container(content)
            .style(review_surface)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    /// Used by the composer's original coding-context row; no sidebar nav item.
    pub fn branch_label(&self) -> Option<&str> {
        self.snapshot
            .as_ref()
            .map(|snapshot| snapshot.branch.as_str())
    }
}

fn review_surface(theme: &Theme) -> container::Style {
    let dark = theme.palette().background.r < 0.5;
    container::Style {
        background: Some(
            if dark {
                Color::from_rgb8(1, 4, 9)
            } else {
                Color::from_rgb8(246, 248, 250)
            }
            .into(),
        ),
        text_color: Some(Color {
            a: 0.74,
            ..theme.palette().text
        }),
        ..Default::default()
    }
}

fn icon_button(theme: &Theme, status: button::Status) -> button::Style {
    file_button(theme, status, false)
}

fn file_button(theme: &Theme, status: button::Status, selected: bool) -> button::Style {
    let mut style = button::Style {
        text_color: Color {
            a: if matches!(status, button::Status::Disabled) {
                0.36
            } else {
                0.74
            },
            ..theme.palette().text
        },
        border: iced::Border {
            radius: 5.0.into(),
            ..Default::default()
        },
        ..Default::default()
    };
    if selected {
        style.background = Some(
            Color {
                a: 0.10,
                ..theme.palette().primary
            }
            .into(),
        );
    } else if matches!(status, button::Status::Hovered | button::Status::Pressed) {
        style.background = Some(
            Color {
                a: 0.05,
                ..theme.palette().text
            }
            .into(),
        );
    }
    style
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    repo: PathBuf,
    workspace: PathBuf,
    scope: PathBuf,
    branch: String,
    changes: Vec<Change>,
    filter_prefixes: Vec<String>,
    truncated: bool,
    omitted: usize,
}

#[derive(Debug, Clone)]
struct Change {
    status: String,
    path: String,
    original: Option<String>,
}
impl Change {
    fn untracked(&self) -> bool {
        self.status == "??"
    }
    fn staged(&self) -> bool {
        self.status
            .as_bytes()
            .first()
            .is_some_and(|byte| !matches!(byte, b' ' | b'?' | b'!'))
    }
}

fn label(value: &str) -> String {
    value
        .chars()
        .flat_map(|character| {
            if character.is_control() {
                character.escape_default().collect::<Vec<_>>()
            } else {
                vec![character]
            }
        })
        .collect()
}

struct Captured {
    bytes: Vec<u8>,
    truncated: bool,
}
fn capture(mut reader: impl Read, limit: usize) -> io::Result<Captured> {
    let mut result = Captured {
        bytes: Vec::new(),
        truncated: false,
    };
    let mut buffer = [0_u8; 8192];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            return Ok(result);
        }
        let keep = read.min(limit.saturating_sub(result.bytes.len()));
        result.bytes.extend_from_slice(&buffer[..keep]);
        result.truncated |= keep < read;
        // Continue draining so a large output cannot block the child process.
    }
}

struct GitOutput {
    success: bool,
    code: Option<i32>,
    stdout: Captured,
    stderr: Captured,
}
fn git(
    directory: &Path,
    filter_prefixes: &[String],
    args: &[OsString],
    limit: usize,
) -> Result<GitOutput, String> {
    git_input(directory, filter_prefixes, args, limit, None)
}

fn git_input(
    directory: &Path,
    filter_prefixes: &[String],
    args: &[OsString],
    limit: usize,
    input: Option<Vec<u8>>,
) -> Result<GitOutput, String> {
    let mut command = Command::new(GIT);
    for (key, _) in std::env::vars_os() {
        if key
            .to_string_lossy()
            .to_ascii_uppercase()
            .starts_with("GIT_")
        {
            command.env_remove(key);
        }
    }
    #[cfg(windows)]
    let null = "NUL";
    #[cfg(not(windows))]
    let null = "/dev/null";
    command
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_SYSTEM", null)
        .env("GIT_CONFIG_GLOBAL", null)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_NO_LAZY_FETCH", "1")
        .env("LC_ALL", "C")
        .arg("--no-pager");
    // check-ignore reads literal filenames from NUL-delimited stdin; it does
    // not accept pathspec magic, including the global literal-pathspec flag.
    if args
        .first()
        .is_none_or(|argument| argument != "check-ignore")
    {
        command.arg("--literal-pathspecs");
    }
    command.args(["--no-optional-locks", "-C"]).arg(directory);
    for config in [
        "color.ui=false",
        "core.fsmonitor=false",
        "core.untrackedCache=false",
        "core.attributesFile=/dev/null",
        "core.hooksPath=/dev/null",
        "status.relativePaths=false",
        "submodule.recurse=false",
        "fetch.recurseSubmodules=false",
        "maintenance.auto=false",
        "gc.auto=0",
    ] {
        command.args(["-c", config]);
    }
    // A repository's clean/process filters can run during status and diff, even
    // with --no-ext-diff. Neutralize every configured driver before either call.
    for prefix in filter_prefixes {
        for suffix in ["clean=", "smudge=", "process=", "required=false"] {
            command.arg("-c").arg(format!("{prefix}.{suffix}"));
        }
    }
    let mut child = command
        .args(args)
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("Could not start system Git: {error}"))?;
    let stdout = child.stdout.take().ok_or("Git stdout was unavailable")?;
    let stderr = child.stderr.take().ok_or("Git stderr was unavailable")?;
    let input_writer = input.map(|bytes| {
        let stdin = child.stdin.take();
        std::thread::spawn(move || {
            stdin
                .ok_or_else(|| io::Error::other("Git input was unavailable"))
                .and_then(|mut stdin| stdin.write_all(&bytes))
        })
    });
    let out_reader = std::thread::spawn(move || capture(stdout, limit));
    let err_reader = std::thread::spawn(move || capture(stderr, 16_384));
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if started.elapsed() < DEADLINE => {
                std::thread::sleep(Duration::from_millis(20))
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                break Err("Git inspection timed out. Try a smaller workspace.".to_owned());
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(format!("Could not wait for Git: {error}"));
            }
        }
    };
    let stdout = out_reader
        .join()
        .map_err(|_| "Git output reader stopped")?
        .map_err(|error| error.to_string())?;
    let stderr = err_reader
        .join()
        .map_err(|_| "Git error reader stopped")?
        .map_err(|error| error.to_string())?;
    if let Some(writer) = input_writer {
        let _ = writer.join();
    }
    let status = status?;
    Ok(GitOutput {
        success: status.success(),
        code: status.code(),
        stdout,
        stderr,
    })
}

/// Native Files-tree projection of the same local Git authority as Review.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Decoration {
    Added,
    Modified,
    Conflicted,
}

pub(crate) fn tree_decorations(workspace: &Path) -> BTreeMap<PathBuf, Decoration> {
    let Ok(snapshot) = load_snapshot(workspace) else {
        return BTreeMap::new();
    };
    snapshot
        .changes
        .iter()
        .filter_map(|change| {
            let absolute = snapshot.repo.join(&change.path);
            let relative = absolute.strip_prefix(&snapshot.workspace).ok()?.to_owned();
            let kind =
                if change.status.contains('U') || matches!(change.status.as_str(), "AA" | "DD") {
                    Decoration::Conflicted
                } else if change.untracked() || change.status.contains('A') {
                    Decoration::Added
                } else {
                    Decoration::Modified
                };
            Some((relative, kind))
        })
        .collect()
}

pub(crate) fn tree_ignored(workspace: &Path, paths: &[PathBuf]) -> BTreeSet<PathBuf> {
    let mut input = Vec::new();
    for path in paths {
        let Some(path) = path.to_str() else {
            continue;
        };
        if input.len() + path.len() + 1 > MAX_STATUS {
            break;
        }
        input.extend_from_slice(path.as_bytes());
        input.push(0);
    }
    let Ok(result) = git_input(
        workspace,
        &[],
        &args(&["check-ignore", "--no-index", "--stdin", "-z"]),
        MAX_STATUS,
        Some(input),
    ) else {
        return BTreeSet::new();
    };
    if !result.success {
        return BTreeSet::new();
    }
    result
        .stdout
        .bytes
        .split(|&byte| byte == 0)
        .filter(|path| !path.is_empty())
        .filter_map(|path| std::str::from_utf8(path).ok().map(PathBuf::from))
        .collect()
}

fn args(values: &[&str]) -> Vec<OsString> {
    values.iter().map(OsString::from).collect()
}
fn checked(output: GitOutput) -> Result<Captured, String> {
    if output.success {
        Ok(output.stdout)
    } else {
        let reason = String::from_utf8_lossy(&output.stderr.bytes)
            .trim()
            .to_owned();
        Err(if reason.is_empty() {
            format!("Git exited with status {:?}", output.code)
        } else {
            label(&reason)
        })
    }
}

fn filters(directory: &Path) -> Result<Vec<String>, String> {
    let output = git(
        directory,
        &[],
        &args(&[
            "config",
            "--null",
            "--name-only",
            "--get-regexp",
            r"^filter\..*\.(clean|smudge|process|required)$",
        ]),
        65_536,
    )?;
    if output.code == Some(1) {
        return Ok(Vec::new());
    }
    let result = checked(output)?;
    if result.truncated {
        return Err("Repository filter configuration exceeds the inspection limit.".into());
    }
    let mut prefixes = BTreeSet::new();
    for key in result
        .bytes
        .split(|&byte| byte == 0)
        .filter(|key| !key.is_empty())
    {
        let key = std::str::from_utf8(key).map_err(|_| "Repository filter name is not UTF-8")?;
        if let Some((prefix, _)) = key.rsplit_once('.') {
            prefixes.insert(prefix.to_owned());
        }
    }
    if prefixes.len() > 128 {
        return Err("Repository has too many filter drivers to inspect safely.".into());
    }
    Ok(prefixes.into_iter().collect())
}

fn parse_status(bytes: &[u8]) -> (Vec<Change>, usize) {
    let mut records = bytes.split_inclusive(|&byte| byte == 0);
    let mut changes = Vec::new();
    let mut omitted = 0;
    while let Some(record) = records.next() {
        if record.last() != Some(&0) || record.len() < 5 || record[2] != b' ' {
            omitted += 1;
            continue;
        }
        let status = String::from_utf8_lossy(&record[..2]).into_owned();
        let renamed = matches!(record[0], b'R' | b'C') || matches!(record[1], b'R' | b'C');
        let original = if renamed {
            records
                .next()
                .and_then(|record| record.strip_suffix(&[0]))
                .and_then(|bytes| String::from_utf8(bytes.to_vec()).ok())
        } else {
            None
        };
        let Ok(path) = String::from_utf8(record[3..record.len() - 1].to_vec()) else {
            omitted += 1;
            continue;
        };
        if changes.len() >= MAX_FILES {
            omitted += 1;
            continue;
        }
        changes.push(Change {
            status,
            path,
            original,
        });
    }
    (changes, omitted)
}

fn scoped_path(snapshot: &Snapshot, relative: &str) -> Result<PathBuf, String> {
    let path = Path::new(relative);
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err("Git returned a path outside the workspace.".into());
    }
    let absolute = snapshot.repo.join(path);
    if !absolute.starts_with(&snapshot.workspace) {
        return Err("Changed path is outside the selected workspace.".into());
    }
    let mut prefix = snapshot.repo.clone();
    for part in path.components() {
        prefix.push(part);
        match std::fs::symlink_metadata(&prefix) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(format!(
                    "Diff preview is unavailable for a symbolic-link path: {}",
                    label(relative)
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => break,
            Err(error) => return Err(format!("Could not inspect {}: {error}", label(relative))),
        }
    }
    Ok(path.to_owned())
}

fn load_snapshot(workspace: &Path) -> Result<Snapshot, String> {
    let workspace = workspace
        .canonicalize()
        .map_err(|error| format!("Could not open workspace: {error}"))?;
    if !workspace.is_dir() {
        return Err("Workspace must be a directory.".into());
    }
    let root = git(
        &workspace,
        &[],
        &args(&["rev-parse", "--show-toplevel"]),
        16_384,
    )?;
    if !root.success {
        return Err("This workspace is not inside a Git worktree.".into());
    }
    let root =
        String::from_utf8(root.stdout.bytes).map_err(|_| "Git worktree path is not UTF-8")?;
    let repo = Path::new(root.trim_end_matches(['\r', '\n']))
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let scope = workspace
        .strip_prefix(&repo)
        .map_err(|_| "Git worktree is outside this workspace")?
        .to_owned();
    let filter_prefixes = filters(&workspace)?;
    let symbolic = git(
        &workspace,
        &filter_prefixes,
        &args(&["symbolic-ref", "--quiet", "--short", "HEAD"]),
        4096,
    )?;
    let branch = if symbolic.success {
        String::from_utf8_lossy(&symbolic.stdout.bytes)
            .trim()
            .to_owned()
    } else {
        let hash = checked(git(
            &workspace,
            &filter_prefixes,
            &args(&["rev-parse", "--short", "HEAD"]),
            4096,
        )?)?;
        format!(
            "detached at {}",
            String::from_utf8_lossy(&hash.bytes).trim()
        )
    };
    let mut status_args = args(&[
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=all",
        "--",
    ]);
    status_args.push(if scope.as_os_str().is_empty() {
        OsString::from(".")
    } else {
        scope.as_os_str().to_owned()
    });
    let status = checked(git(&repo, &filter_prefixes, &status_args, MAX_STATUS)?)?;
    let (changes, omitted) = parse_status(&status.bytes);
    Ok(Snapshot {
        repo,
        workspace,
        scope,
        branch,
        changes,
        filter_prefixes,
        truncated: status.truncated || omitted > 0,
        omitted,
    })
}

fn diff_part(snapshot: &Snapshot, selected: Option<usize>, staged: bool) -> Result<String, String> {
    let mut diff_args = args(&[
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--ignore-submodules=all",
        "--src-prefix=a/",
        "--dst-prefix=b/",
    ]);
    if staged {
        diff_args.push(OsString::from("--cached"));
    }
    diff_args.push(OsString::from("--"));
    if let Some(index) = selected {
        let change = snapshot
            .changes
            .get(index)
            .ok_or("Changed file no longer exists in this view. Refresh to update it.")?;
        diff_args.push(scoped_path(snapshot, &change.path)?.into_os_string());
        if let Some(original) = &change.original
            && let Ok(original) = scoped_path(snapshot, original)
        {
            diff_args.push(original.into_os_string());
        }
    } else {
        // Git treats tracked symlinks as link text, but refusing them here also
        // prevents a changed parent symlink from widening the preview's scope.
        for change in &snapshot.changes {
            if !change.untracked() {
                scoped_path(snapshot, &change.path)?;
            }
        }
        diff_args.push(if snapshot.scope.as_os_str().is_empty() {
            OsString::from(".")
        } else {
            snapshot.scope.as_os_str().to_owned()
        });
    }
    let output = checked(git(
        &snapshot.repo,
        &snapshot.filter_prefixes,
        &diff_args,
        MAX_DIFF,
    )?)?;
    let mut diff = String::from_utf8_lossy(&output.bytes).into_owned();
    if output.truncated {
        diff.push_str("\n\n[Diff preview truncated at 512 KiB. Select an individual file to inspect a smaller diff.]\n");
    }
    Ok(diff)
}

fn load_diff(
    snapshot: &Snapshot,
    selected: Option<usize>,
    mode: DiffMode,
) -> Result<String, String> {
    // Configuration may have changed since the last refresh. Re-discover and
    // neutralize filters for every preview, not only while listing files.
    let mut current = snapshot.clone();
    current.filter_prefixes = filters(&snapshot.repo)?;
    let snapshot = &current;
    if let Some(index) = selected {
        let change = snapshot
            .changes
            .get(index)
            .ok_or("Changed file is no longer available. Refresh the worktree.")?;
        scoped_path(snapshot, &change.path)?;
        if change.untracked() {
            return untracked_diff(snapshot, &change.path);
        }
    }
    let result = match mode {
        DiffMode::WorkingTree => diff_part(snapshot, selected, false)?,
        DiffMode::Staged => diff_part(snapshot, selected, true)?,
    };
    Ok(if result.is_empty() {
        match mode {
            DiffMode::WorkingTree => "No unstaged changes for this selection.",
            DiffMode::Staged => "No staged changes for this selection.",
        }
        .to_owned()
    } else {
        result
    })
}

fn untracked_diff(snapshot: &Snapshot, relative: &str) -> Result<String, String> {
    scoped_path(snapshot, relative)?;
    let absolute = snapshot.repo.join(relative);
    let inside = absolute
        .strip_prefix(&snapshot.workspace)
        .map_err(|_| "File is outside this workspace")?;
    let directory =
        cap_std::fs::Dir::open_ambient_dir(&snapshot.workspace, cap_std::ambient_authority())
            .map_err(|error| error.to_string())?;
    let metadata = directory
        .symlink_metadata(inside)
        .map_err(|error| error.to_string())?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err("Select a regular file to preview its diff.".into());
    }
    if metadata.len() > (MAX_DIFF / 2) as u64 {
        return Ok("New file exceeds the 256 KiB diff preview limit.".into());
    }
    let file = directory.open(inside).map_err(|error| error.to_string())?;
    if !file
        .metadata()
        .map_err(|error| error.to_string())?
        .is_file()
    {
        return Err("File changed before preview; refresh the worktree.".into());
    }
    let mut bytes = Vec::new();
    file.take((MAX_DIFF / 2 + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > MAX_DIFF / 2 {
        return Ok("New file exceeds the 256 KiB diff preview limit.".into());
    }
    let text = match String::from_utf8(bytes) {
        Ok(text) if !text.contains('\0') => text,
        _ => {
            return Ok(format!(
                "Binary files /dev/null and {} differ\n",
                label(relative)
            ));
        }
    };
    #[cfg(unix)]
    let mode = {
        use cap_std::fs::MetadataExt;
        if metadata.mode() & 0o111 != 0 {
            "100755"
        } else {
            "100644"
        }
    };
    #[cfg(not(unix))]
    let mode = "100644";
    let path = label(relative);
    let mut diff = format!(
        "diff --git a/{path} b/{path}\nnew file mode {mode}\n--- /dev/null\n+++ b/{path}\n"
    );
    if !text.is_empty() {
        diff.push_str(&format!(
            "@@ -0,0 +1,{} @@\n",
            text.split_inclusive('\n').count()
        ));
        for line in text.split_inclusive('\n') {
            diff.push('+');
            diff.push_str(line);
            if !line.ends_with('\n') {
                diff.push_str("\n\\ No newline at end of file\n");
            }
        }
    }
    Ok(diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_REPO: AtomicU64 = AtomicU64::new(1);
    struct TestRepo {
        directory: PathBuf,
    }
    impl TestRepo {
        fn new() -> Self {
            let directory = std::env::temp_dir().join(format!(
                "hermes-git-test-{}-{}",
                std::process::id(),
                NEXT_REPO.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&directory).unwrap();
            let repo = Self { directory };
            repo.command(&["init", "-q", "-b", "main"]);
            repo.command(&["config", "user.name", "Hermes Test"]);
            repo.command(&["config", "user.email", "hermes-test@example.invalid"]);
            repo
        }
        fn command(&self, args: &[&str]) {
            let output = Command::new(GIT)
                .arg("-C")
                .arg(&self.directory)
                .args(args)
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .env("GIT_CONFIG_GLOBAL", "/dev/null")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{:?}: {}",
                args,
                String::from_utf8_lossy(&output.stderr)
            );
        }
        fn write(&self, path: &str, text: &str) {
            let path = self.directory.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        fn commit(&self) {
            self.command(&["add", "--all"]);
            self.command(&["commit", "-q", "-m", "test baseline"]);
        }
    }
    impl Drop for TestRepo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }

    #[test]
    fn real_worktree_status_and_both_diff_sources() {
        let repo = TestRepo::new();
        repo.write("working.txt", "before\n");
        repo.write("staged.txt", "before\n");
        repo.commit();
        repo.write("working.txt", "unstaged content\n");
        repo.write("staged.txt", "staged content\n");
        repo.command(&["add", "--", "staged.txt"]);
        repo.write("--new [file].txt", "new\n");
        let index_before = std::fs::read(repo.directory.join(".git/index")).unwrap();
        let snapshot = load_snapshot(&repo.directory).unwrap();
        assert_eq!(snapshot.branch, "main");
        assert_eq!(snapshot.changes.len(), 3);
        assert_eq!(
            snapshot
                .changes
                .iter()
                .filter(|change| change.staged())
                .count(),
            1
        );
        assert_eq!(
            snapshot
                .changes
                .iter()
                .filter(|change| change.untracked())
                .count(),
            1
        );
        let working = load_diff(&snapshot, None, DiffMode::WorkingTree).unwrap();
        assert!(working.contains("+unstaged content"));
        assert!(!working.contains("+staged content"));
        let staged = load_diff(&snapshot, None, DiffMode::Staged).unwrap();
        assert!(staged.contains("+staged content"));
        assert!(!staged.contains("+unstaged content"));
        let new_file = snapshot.changes.iter().position(Change::untracked).unwrap();
        let new_diff = load_diff(&snapshot, Some(new_file), DiffMode::WorkingTree).unwrap();
        assert!(new_diff.contains("--- /dev/null"));
        assert!(new_diff.contains("+new"));
        assert_eq!(
            std::fs::read(repo.directory.join(".git/index")).unwrap(),
            index_before
        );
        assert!(!repo.directory.join(".git/index.lock").exists());
    }

    #[test]
    fn selected_paths_are_literal_and_nested_workspaces_stay_scoped() {
        let repo = TestRepo::new();
        repo.write("nested/literal[1].txt", "before\n");
        repo.write("nested/literal1.txt", "before\n");
        repo.write("outside.txt", "before\n");
        repo.commit();
        repo.write("nested/literal[1].txt", "exact bracket file\n");
        repo.write("nested/literal1.txt", "other file\n");
        repo.write("outside.txt", "outside workspace\n");
        let snapshot = load_snapshot(&repo.directory.join("nested")).unwrap();
        assert_eq!(snapshot.changes.len(), 2);
        let index = snapshot
            .changes
            .iter()
            .position(|change| change.path == "nested/literal[1].txt")
            .unwrap();
        let diff = load_diff(&snapshot, Some(index), DiffMode::WorkingTree).unwrap();
        assert!(diff.contains("+exact bracket file"));
        assert!(!diff.contains("+other file"));
        assert!(!diff.contains("outside workspace"));
        assert!(scoped_path(&snapshot, "../outside").is_err());
        assert!(scoped_path(&snapshot, "outside.txt").is_err());
    }

    #[test]
    fn linked_worktree_and_detached_branch_are_supported() {
        let repo = TestRepo::new();
        repo.write("note.txt", "before\n");
        repo.commit();
        let linked = repo.directory.join("linked");
        repo.command(&[
            "worktree",
            "add",
            "--quiet",
            "--detach",
            linked.to_str().unwrap(),
        ]);
        std::fs::write(linked.join("note.txt"), "linked change\n").unwrap();
        let snapshot = load_snapshot(&linked).unwrap();
        assert!(snapshot.branch.starts_with("detached at "));
        assert_eq!(snapshot.repo, linked.canonicalize().unwrap());
        assert!(
            load_diff(&snapshot, None, DiffMode::WorkingTree)
                .unwrap()
                .contains("+linked change")
        );
    }

    #[test]
    fn repository_filters_cannot_execute_during_status_or_diff() {
        let repo = TestRepo::new();
        repo.write("note.txt", "before\n");
        repo.commit();
        let marker = repo.directory.join("filter-ran");
        repo.command(&[
            "config",
            "filter.hostile.clean",
            &format!("touch '{}' && cat", marker.display()),
        ]);
        repo.write(".gitattributes", "*.txt filter=hostile\n");
        repo.write("note.txt", "after\n");
        let snapshot = load_snapshot(&repo.directory).unwrap();
        assert_eq!(snapshot.filter_prefixes, vec!["filter.hostile"]);
        assert!(
            load_diff(&snapshot, None, DiffMode::WorkingTree)
                .unwrap()
                .contains("+after")
        );
        assert!(!marker.exists());
        repo.command(&[
            "config",
            "filter.later.clean",
            &format!("touch '{}' && cat", marker.display()),
        ]);
        repo.write(".gitattributes", "*.txt filter=later\n");
        assert!(
            load_diff(&snapshot, None, DiffMode::WorkingTree)
                .unwrap()
                .contains("+after")
        );
        assert!(!marker.exists());
    }

    #[test]
    fn renamed_paths_and_output_bounds_are_parsed() {
        let (changes, omitted) = parse_status(b"R  new name\0old name\0 M next\0?? incomplete");
        assert_eq!(changes.len(), 2);
        assert_eq!(omitted, 1);
        assert_eq!(changes[0].original.as_deref(), Some("old name"));
        let captured = capture(io::Cursor::new(vec![b'x'; 100]), 20).unwrap();
        assert_eq!(captured.bytes.len(), 20);
        assert!(captured.truncated);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_parent_cannot_escape_selected_worktree() {
        use std::os::unix::fs::symlink;
        let repo = TestRepo::new();
        repo.write("nested/note.txt", "before\n");
        repo.commit();
        let outside = repo.directory.join("outside");
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(outside.join("note.txt"), "private\n").unwrap();
        std::fs::remove_file(repo.directory.join("nested/note.txt")).unwrap();
        std::fs::remove_dir(repo.directory.join("nested")).unwrap();
        symlink(&outside, repo.directory.join("nested")).unwrap();
        let snapshot = load_snapshot(&repo.directory).unwrap();
        assert!(
            scoped_path(&snapshot, "nested/note.txt")
                .unwrap_err()
                .contains("symbolic-link")
        );
        assert!(load_diff(&snapshot, None, DiffMode::WorkingTree).is_err());
        assert_eq!(
            std::fs::read_to_string(outside.join("note.txt")).unwrap(),
            "private\n"
        );
    }
}
