//! Native port of the official right-sidebar File system tree.
//! Reference: public v0.23.3, caafe5389ae585ef77387a962aee05b1bcf17029,
//! src/app/right-sidebar/{index.tsx,files/tree.tsx,files/use-project-tree.ts}.
//! The shell owns placement. Files remain a lazy, persistent hierarchy; a file
//! selection never navigates the sidebar, and double-click opens a main preview.
//! Remaining upstream parity gaps: inline rename/context menus, drag-to-composer,
//! keyboard tree navigation, persistent ignore preferences, document/rendered
//! previews and image formats beyond PNG/JPEG. Large trees are bounded here.

use crate::appearance::text;
use crate::{
    git::{self, Decoration},
    ui,
};
use cap_std::{ambient_authority, fs::Dir};
use iced::widget::text as text_widget;
use iced::{
    Alignment, Color, Element, Fill, Font, Padding, Task, Theme,
    widget::{self, Space, button, column, container, mouse_area, row, scrollable, tooltip},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{self, Cursor, Read},
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::Duration,
};

const PREVIEW_BYTES: usize = 512 * 1024;
const IMAGE_BYTES: usize = 16 * 1024 * 1024;
const MAX_IMAGE_PIXELS: u64 = 16_777_216;
const MAX_DIRECTORY_SCAN: usize = 8192;
const MAX_DIRECTORY_ROWS: usize = 4096;
const MAX_VISIBLE_ROWS: usize = 2048;
const MAX_DEPTH: usize = 32;

#[derive(Debug, Clone)]
pub struct Row {
    path: PathBuf,
    title: String,
    directory: bool,
}
#[derive(Debug, Clone)]
pub struct Listing {
    rows: Vec<Row>,
    truncated: bool,
    omitted: usize,
}
#[derive(Debug)]
pub struct Workspace {
    root: PathBuf,
    dir: Dir,
}
impl Workspace {
    fn new(root: &Path) -> io::Result<Self> {
        let root = root.canonicalize()?;
        Ok(Self {
            dir: Dir::open_ambient_dir(&root, ambient_authority())?,
            root,
        })
    }
    fn resolve(&self, relative: &Path) -> io::Result<PathBuf> {
        let path = local_path(relative)?;
        // cap-std resolves links relative to this open directory, refusing any
        // resolution outside it, including parent-directory substitution races.
        self.dir.canonicalize(if path.as_os_str().is_empty() {
            Path::new(".")
        } else {
            &path
        })
    }
}
#[derive(Debug)]
pub struct RootData {
    workspace: Arc<Workspace>,
    listing: Arc<Listing>,
    decorations: BTreeMap<PathBuf, Decoration>,
}
#[derive(Debug, Clone)]
pub struct Preview {
    name: String,
    body: PreviewBody,
    size: u64,
    truncated: bool,
}
#[derive(Debug, Clone)]
enum PreviewBody {
    Text(String),
    Image {
        handle: widget::image::Handle,
        width: u32,
        height: u32,
    },
    Binary,
    Large,
    Unavailable(String),
}
#[derive(Debug, Clone)]
pub enum Action {
    OpenDirectory(PathBuf),
    ToggleDirectory(PathBuf),
    Select(PathBuf),
    OpenFile(PathBuf),
    Refresh,
    ToggleIgnored,
    CollapseAll,
    Hover(Option<PathBuf>),
    OpenFolder,
    ClosePreview,
    PreviewAnyway,
    Ready(u64, Result<Arc<RootData>, String>),
    Listed(u64, PathBuf, Result<Arc<Listing>, String>),
    Read(u64, Result<Arc<Preview>, String>),
    ShowLoading(u64),
}
#[derive(Default)]
struct Branch {
    listing: Option<Arc<Listing>>,
    loading: bool,
    error: Option<String>,
    request: u64,
}

pub struct Files {
    root: Option<PathBuf>,
    workspace: Option<Arc<Workspace>>,
    branches: BTreeMap<PathBuf, Branch>,
    expanded: BTreeSet<PathBuf>,
    selected: Option<PathBuf>,
    hovered: Option<PathBuf>,
    decorations: BTreeMap<PathBuf, Decoration>,
    preferences: BTreeMap<PathBuf, bool>,
    show_ignored: bool,
    preview_path: Option<PathBuf>,
    preview: Option<Arc<Preview>>,
    preview_error: Option<String>,
    preview_loading: bool,
    root_error: Option<String>,
    root_loading: bool,
    show_loading: bool,
    next_request: u64,
    root_request: u64,
    file_version: u64,
}

impl Files {
    pub fn new() -> Self {
        Self {
            root: None,
            workspace: None,
            branches: BTreeMap::new(),
            expanded: BTreeSet::new(),
            selected: None,
            hovered: None,
            decorations: BTreeMap::new(),
            preferences: BTreeMap::new(),
            show_ignored: false,
            preview_path: None,
            preview: None,
            preview_error: None,
            preview_loading: false,
            root_error: None,
            root_loading: false,
            show_loading: false,
            next_request: 0,
            root_request: 0,
            file_version: 0,
        }
    }
    pub fn reset(&mut self, root: Option<PathBuf>) -> Task<Action> {
        if self.root == root && self.workspace.is_some() {
            return Task::none();
        }
        self.root = root;
        self.workspace = None;
        self.branches.clear();
        self.expanded.clear();
        self.decorations.clear();
        self.selected = None;
        self.hovered = None;
        self.root_error = None;
        self.root_loading = false;
        self.close_preview();
        self.show_ignored = self
            .root
            .as_ref()
            .and_then(|root| self.preferences.get(root))
            .copied()
            .unwrap_or(false);
        self.refresh()
    }
    fn ticket(&mut self) -> u64 {
        self.next_request = self.next_request.wrapping_add(1);
        self.next_request
    }
    fn refresh(&mut self) -> Task<Action> {
        let Some(root) = self.root.clone() else {
            self.root_request = self.ticket();
            return Task::none();
        };
        let request = self.ticket();
        self.root_request = request;
        self.root_loading = true;
        self.show_loading = false;
        for branch in self.branches.values_mut() {
            branch.request = 0;
            branch.loading = false;
        }
        let show_ignored = self.show_ignored;
        Task::batch([
            Task::perform(
                async move {
                    tokio::task::spawn_blocking(move || {
                        let workspace =
                            Arc::new(Workspace::new(&root).map_err(|error| error.to_string())?);
                        let listing = Arc::new(
                            list(&workspace, Path::new(""), show_ignored)
                                .map_err(|error| error.to_string())?,
                        );
                        let decorations = git::tree_decorations(&workspace.root);
                        Ok(Arc::new(RootData {
                            workspace,
                            listing,
                            decorations,
                        }))
                    })
                    .await
                    .unwrap_or_else(|error| Err(error.to_string()))
                },
                move |result| Action::Ready(request, result),
            ),
            Task::perform(
                async {
                    tokio::time::sleep(Duration::from_millis(120)).await;
                },
                move |_| Action::ShowLoading(request),
            ),
        ])
    }
    fn load_directory(&mut self, path: PathBuf) -> Task<Action> {
        let Some(workspace) = self.workspace.clone() else {
            return Task::none();
        };
        let Ok(path) = local_path(&path) else {
            return Task::none();
        };
        if path.components().count() > MAX_DEPTH {
            self.branches.entry(path).or_default().error =
                Some("Folder depth limit reached".into());
            return Task::none();
        }
        let request = self.ticket();
        let show_ignored = self.show_ignored;
        let branch = self.branches.entry(path.clone()).or_default();
        branch.loading = true;
        branch.error = None;
        branch.request = request;
        let returned = path.clone();
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || {
                    list(&workspace, &path, show_ignored)
                        .map(Arc::new)
                        .map_err(|error| error.to_string())
                })
                .await
                .unwrap_or_else(|error| Err(error.to_string()))
            },
            move |result| Action::Listed(request, returned.clone(), result),
        )
    }
    fn open_file(&mut self, path: PathBuf, allow_large: bool) -> Task<Action> {
        let Some(workspace) = self.workspace.clone() else {
            return Task::none();
        };
        let Ok(path) = local_path(&path) else {
            return Task::none();
        };
        self.file_version = self.file_version.wrapping_add(1);
        let version = self.file_version;
        self.preview_path = Some(path.clone());
        self.selected = Some(path.clone());
        self.preview = None;
        self.preview_error = None;
        self.preview_loading = true;
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || {
                    preview(&workspace, &path, allow_large)
                        .map(Arc::new)
                        .map_err(|error| error.to_string())
                })
                .await
                .unwrap_or_else(|error| Err(error.to_string()))
            },
            move |result| Action::Read(version, result),
        )
    }
    fn close_preview(&mut self) {
        self.file_version = self.file_version.wrapping_add(1);
        self.preview_path = None;
        self.preview = None;
        self.preview_error = None;
        self.preview_loading = false;
    }
    pub fn update(&mut self, action: Action) -> Task<Action> {
        match action {
            Action::Refresh => return self.refresh(),
            Action::ToggleIgnored => {
                self.show_ignored = !self.show_ignored;
                if let Some(root) = &self.root {
                    self.preferences.insert(root.clone(), self.show_ignored);
                }
                return self.refresh();
            }
            Action::CollapseAll => self.expanded.clear(),
            Action::Hover(path) => self.hovered = path,
            Action::Select(path) => self.selected = Some(path),
            Action::ToggleDirectory(path) => {
                if !self.expanded.remove(&path) {
                    self.expanded.insert(path.clone());
                    if !self
                        .branches
                        .get(&path)
                        .is_some_and(|branch| branch.listing.is_some() || branch.loading)
                    {
                        return self.load_directory(path);
                    }
                }
            }
            Action::OpenDirectory(path) => {
                if path.as_os_str().is_empty() {
                    return self.refresh();
                }
                self.expanded.insert(path.clone());
                if self
                    .branches
                    .get(&path)
                    .is_some_and(|branch| branch.listing.is_some() || branch.loading)
                {
                    return Task::none();
                }
                return self.load_directory(path);
            }
            Action::OpenFile(path) => return self.open_file(path, false),
            Action::PreviewAnyway => {
                if let Some(path) = self.preview_path.clone() {
                    return self.open_file(path, true);
                }
            }
            Action::ClosePreview => self.close_preview(),
            Action::Ready(request, result) if request == self.root_request => {
                self.root_loading = false;
                self.show_loading = false;
                match result {
                    Ok(data) => {
                        self.workspace = Some(data.workspace.clone());
                        self.root_error = None;
                        self.decorations = data.decorations.clone();
                        self.branches.insert(
                            PathBuf::new(),
                            Branch {
                                listing: Some(data.listing.clone()),
                                ..Branch::default()
                            },
                        );
                        let expanded: Vec<_> = self.expanded.iter().cloned().collect();
                        return Task::batch(
                            expanded.into_iter().map(|path| self.load_directory(path)),
                        );
                    }
                    Err(error) => self.root_error = Some(error),
                }
            }
            Action::Listed(request, path, result) => {
                if let Some(branch) = self
                    .branches
                    .get_mut(&path)
                    .filter(|branch| branch.request == request)
                {
                    branch.loading = false;
                    match result {
                        Ok(listing) => {
                            branch.listing = Some(listing);
                            branch.error = None;
                        }
                        Err(error) => branch.error = Some(error),
                    }
                }
            }
            Action::Read(version, result) if version == self.file_version => {
                self.preview_loading = false;
                match result {
                    Ok(preview) => self.preview = Some(preview),
                    Err(error) => self.preview_error = Some(error),
                }
            }
            Action::ShowLoading(request) if request == self.root_request && self.root_loading => {
                self.show_loading = true
            }
            _ => {}
        }
        Task::none()
    }
    fn decoration(&self, path: &Path) -> Option<Decoration> {
        self.decorations
            .iter()
            .filter(|(changed, _)| changed.starts_with(path))
            .map(|(_, kind)| *kind)
            .max_by_key(|kind| match kind {
                Decoration::Added => 0,
                Decoration::Modified => 1,
                Decoration::Conflicted => 2,
            })
    }
    fn visible<'a>(&'a self, path: &Path, depth: usize, result: &mut Vec<Visible<'a>>) {
        if depth > MAX_DEPTH + 1 || result.len() >= MAX_VISIBLE_ROWS {
            return;
        }
        let Some(branch) = self.branches.get(path) else {
            return;
        };
        if let Some(error) = &branch.error {
            result.push(Visible::Placeholder(
                format!("Unable to read ({error})"),
                depth,
                true,
            ));
            return;
        }
        if let Some(listing) = &branch.listing {
            for item in &listing.rows {
                if result.len() >= MAX_VISIBLE_ROWS {
                    break;
                }
                result.push(Visible::Entry(item, depth));
                if item.directory && self.expanded.contains(&item.path) {
                    if self.branches.contains_key(&item.path) {
                        self.visible(&item.path, depth + 1, result);
                    } else {
                        result.push(Visible::Placeholder("Loading…".into(), depth + 1, false));
                    }
                }
            }
            if listing.truncated {
                result.push(Visible::Placeholder(
                    format!("Showing up to {MAX_DIRECTORY_ROWS} entries"),
                    depth,
                    false,
                ));
            }
            if listing.omitted > 0 {
                result.push(Visible::Placeholder(
                    format!("Unable to read ({} entries)", listing.omitted),
                    depth,
                    true,
                ));
            }
        } else if branch.loading {
            result.push(Visible::Placeholder("Loading…".into(), depth, false));
        }
    }
    pub fn view(&self) -> Element<'_, Action> {
        let Some(root) = &self.root else {
            return container(
                column![
                    text("▦  NO PROJECT OPEN").size(10).style(muted_text),
                    button(
                        row![
                            ui::icon("folder-opened", 13, Color::TRANSPARENT).style(muted_text),
                            text("Open folder").size(12)
                        ]
                        .spacing(6)
                        .align_y(Alignment::Center)
                    )
                    .on_press(Action::OpenFolder)
                    .padding([5, 9])
                    .style(button::secondary),
                ]
                .spacing(8)
                .align_x(Alignment::Center),
            )
            .center_x(Fill)
            .center_y(Fill)
            .padding(16)
            .height(Fill)
            .style(surface)
            .into();
        };
        let name = self
            .workspace
            .as_ref()
            .map(|workspace| &workspace.root)
            .unwrap_or(root)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_uppercase();
        let show_ignored_tip = if self.show_ignored {
            "Hide gitignored files"
        } else {
            "Show gitignored files"
        };
        let header = container(
            row![
                text(format!("▦  {name}"))
                    .size(10)
                    .width(Fill)
                    .wrapping(text_widget::Wrapping::None)
                    .style(|theme: &Theme| text_widget::Style {
                        color: Some(theme.palette().primary)
                    }),
                icon_action(
                    if self.show_ignored {
                        "eye"
                    } else {
                        "eye-closed"
                    },
                    show_ignored_tip,
                    Some(Action::ToggleIgnored)
                ),
                icon_action(
                    "refresh",
                    "Refresh tree",
                    (!self.root_loading).then_some(Action::Refresh)
                ),
                icon_action(
                    "three-bars",
                    "Collapse all folders",
                    (!self.expanded.is_empty()).then_some(Action::CollapseAll)
                ),
            ]
            .spacing(1)
            .align_y(Alignment::Center),
        )
        .padding([0, 10])
        .height(28);
        let mut body = column![header].height(Fill);
        if let Some(error) = &self.root_error {
            body = body.push(empty_state(
                "UNREADABLE",
                format!("Could not read this folder ({error})."),
                Some(Action::Refresh),
            ));
        } else {
            let root_listing = self
                .branches
                .get(Path::new(""))
                .and_then(|branch| branch.listing.as_ref());
            if self.root_loading && root_listing.is_none() {
                if self.show_loading {
                    let mut skeleton = column![].spacing(9).padding([10, 17]);
                    for width in [112, 144, 94, 124, 82, 135] {
                        skeleton = skeleton.push(
                            container(Space::new().width(width).height(9)).style(
                                |theme: &Theme| container::Style {
                                    background: Some(
                                        Color {
                                            a: 0.07,
                                            ..theme.palette().text
                                        }
                                        .into(),
                                    ),
                                    ..Default::default()
                                },
                            ),
                        );
                    }
                    body = body.push(skeleton);
                }
            } else if root_listing.is_some_and(|listing| listing.rows.is_empty()) {
                body = body.push(empty_state("EMPTY", "This folder is empty.".into(), None));
            } else {
                let mut visible = Vec::new();
                self.visible(Path::new(""), 0, &mut visible);
                if visible.len() >= MAX_VISIBLE_ROWS {
                    visible.push(Visible::Placeholder(
                        format!("Showing first {MAX_VISIBLE_ROWS} visible entries"),
                        0,
                        false,
                    ));
                }
                let mut tree = column![].spacing(0);
                for item in visible {
                    tree = tree.push(match item {
                        Visible::Entry(item, depth) => self.row_view(item, depth),
                        Visible::Placeholder(message, depth, error) => container(
                            row![
                                ui::icon(
                                    if error { "warning" } else { "loading" },
                                    12,
                                    Color::TRANSPARENT
                                )
                                .style(muted_text),
                                text(message).size(12).style(muted_text),
                            ]
                            .spacing(4),
                        )
                        .padding(Padding {
                            left: 17.0 + depth as f32 * 10.0,
                            ..Padding::ZERO
                        })
                        .height(22)
                        .into(),
                    });
                }
                body = body.push(scrollable(tree).height(Fill).width(Fill));
            }
        }
        container(body)
            .width(Fill)
            .height(Fill)
            .style(surface)
            .into()
    }
    fn row_view<'a>(&'a self, item: &'a Row, depth: usize) -> Element<'a, Action> {
        let selected = self.selected.as_ref() == Some(&item.path);
        let hovered = self.hovered.as_ref() == Some(&item.path);
        let open = self.expanded.contains(&item.path);
        let decoration = self.decoration(&item.path);
        let glyph = if item.directory {
            if open { "folder-opened" } else { "folder" }
        } else {
            "file"
        };
        let name = text(&item.title)
            .size(12)
            .width(Fill)
            .wrapping(text_widget::Wrapping::None)
            .style(move |theme: &Theme| {
                let p = ui::palette(theme.palette().background.r < 0.5);
                text_widget::Style {
                    color: Some(match decoration {
                        Some(Decoration::Added) => theme.palette().success,
                        Some(Decoration::Modified) => theme.palette().warning,
                        Some(Decoration::Conflicted) => theme.palette().danger,
                        None => {
                            if selected || hovered {
                                p.foreground
                            } else {
                                Color {
                                    a: 0.74,
                                    ..p.foreground
                                }
                            }
                        }
                    }),
                }
            });
        let glyph = text(icon_character(glyph).to_string())
            .font(Font::with_name("codicon"))
            .size(14)
            .width(14)
            .style(muted_text);
        let content = container(row![glyph, name].spacing(4).align_y(Alignment::Center))
            .width(Fill)
            .height(22)
            .padding(Padding {
                left: 17.0 + depth as f32 * 10.0,
                right: 8.0,
                top: 0.0,
                bottom: 0.0,
            })
            .clip(true)
            .style(move |theme: &Theme| container::Style {
                background: if selected {
                    Some(
                        Color {
                            a: 0.10,
                            ..theme.palette().primary
                        }
                        .into(),
                    )
                } else if hovered {
                    Some(
                        Color {
                            a: 0.04,
                            ..theme.palette().text
                        }
                        .into(),
                    )
                } else {
                    None
                },
                ..Default::default()
            });
        let action = if item.directory {
            if open {
                Action::ToggleDirectory(item.path.clone())
            } else {
                Action::OpenDirectory(item.path.clone())
            }
        } else {
            Action::Select(item.path.clone())
        };
        let mut hit = mouse_area(content)
            .on_press(action)
            .on_enter(Action::Hover(Some(item.path.clone())))
            .on_exit(Action::Hover(None))
            .interaction(iced::mouse::Interaction::Pointer);
        if !item.directory {
            hit = hit.on_double_click(Action::OpenFile(item.path.clone()));
        }
        tooltip(
            hit,
            text(item.path.to_string_lossy()).size(11),
            tooltip::Position::Left,
        )
        .delay(Duration::from_millis(600))
        .into()
    }
    /// Original preview is a separate main pane, never a sidebar master/detail.
    pub fn preview_view(&self) -> Option<Element<'_, Action>> {
        let path = self.preview_path.as_ref()?;
        let header = container(
            row![
                text(path.file_name().unwrap_or_default().to_string_lossy())
                    .size(12)
                    .width(Fill),
                icon_action("close", "Close preview pane", Some(Action::ClosePreview))
            ]
            .align_y(Alignment::Center),
        )
        .padding([0, 10])
        .height(34);
        let content: Element<'_, Action> = if self.preview_loading {
            empty_state("", "Loading preview".into(), None)
        } else if let Some(error) = &self.preview_error {
            empty_state(
                "PREVIEW UNAVAILABLE",
                error.clone(),
                Some(Action::OpenFile(path.clone())),
            )
        } else if let Some(preview) = &self.preview {
            match &preview.body {
                PreviewBody::Text(body) => {
                    let mut source = column![].spacing(0);
                    if preview.truncated {
                        source = source.push(
                            container(text("Showing first 512 KB.").size(11).style(muted_text))
                                .padding([6, 12]),
                        );
                    }
                    source = source.push(
                        scrollable(
                            text(body)
                                .size(12)
                                .font(Font::MONOSPACE)
                                .wrapping(text_widget::Wrapping::None),
                        )
                        .direction(scrollable::Direction::Both {
                            horizontal: Default::default(),
                            vertical: Default::default(),
                        })
                        .height(Fill)
                        .width(Fill),
                    );
                    source.height(Fill).into()
                }
                PreviewBody::Image {
                    handle,
                    width,
                    height,
                } => column![
                    container(
                        widget::image(handle.clone())
                            .content_fit(iced::ContentFit::Contain)
                            .width(Fill)
                            .height(Fill)
                    )
                    .padding(16)
                    .width(Fill)
                    .height(Fill),
                    container(
                        text(format!("{width} × {height}"))
                            .size(10)
                            .style(muted_text)
                    )
                    .padding([5, 10]),
                ]
                .height(Fill)
                .into(),
                PreviewBody::Binary => empty_state(
                    "THIS LOOKS LIKE A BINARY FILE",
                    format!("Previewing {} may show unreadable text.", preview.name),
                    None,
                ),
                PreviewBody::Large => container(
                    column![
                        text("This file is large").size(15),
                        text(format!(
                            "{} is {} bytes. Hermes will only show the first 512 KB.",
                            preview.name, preview.size
                        ))
                        .size(12),
                        button(text("Preview anyway").size(12))
                            .on_press(Action::PreviewAnyway)
                            .style(button::secondary)
                    ]
                    .spacing(10)
                    .align_x(Alignment::Center),
                )
                .padding(20)
                .center_x(Fill)
                .center_y(Fill)
                .height(Fill)
                .into(),
                PreviewBody::Unavailable(reason) => {
                    empty_state("PREVIEW UNAVAILABLE", reason.clone(), None)
                }
            }
        } else {
            empty_state("", "Loading preview".into(), None)
        };
        Some(
            container(column![header, content].height(Fill))
                .width(Fill)
                .height(Fill)
                .style(|theme: &Theme| container::Style {
                    background: Some(theme.palette().background.into()),
                    text_color: Some(theme.palette().text),
                    ..Default::default()
                })
                .into(),
        )
    }
}

enum Visible<'a> {
    Entry(&'a Row, usize),
    Placeholder(String, usize, bool),
}
fn icon_character(name: &str) -> char {
    match name {
        "folder" => '\u{ea83}',
        "folder-opened" => '\u{eaf7}',
        _ => '\u{ea7b}',
    }
}
fn icon_action<'a>(name: &str, label: &'a str, action: Option<Action>) -> Element<'a, Action> {
    let glyph = match name {
        "eye" => '\u{ea70}',
        "eye-closed" => '\u{eae7}',
        "refresh" => '\u{eb37}',
        "three-bars" => '\u{eb6a}',
        _ => '\u{ea76}',
    };
    tooltip(
        button(
            text(glyph.to_string())
                .font(Font::with_name("codicon"))
                .size(13)
                .style(muted_text),
        )
        .width(20)
        .height(20)
        .padding(0)
        .on_press_maybe(action)
        .style(quiet_button),
        text(label).size(11),
        tooltip::Position::Bottom,
    )
    .delay(Duration::from_millis(200))
    .into()
}
fn quiet_button(theme: &Theme, status: button::Status) -> button::Style {
    button::Style {
        background: matches!(status, button::Status::Hovered | button::Status::Pressed).then_some(
            Color {
                a: 0.05,
                ..theme.palette().text
            }
            .into(),
        ),
        text_color: Color {
            a: if status == button::Status::Disabled {
                0.36
            } else {
                0.7
            },
            ..theme.palette().text
        },
        ..Default::default()
    }
}
fn surface(theme: &Theme) -> container::Style {
    let p = ui::palette(theme.palette().background.r < 0.5);
    container::Style {
        background: Some(p.sidebar.into()),
        text_color: Some(Color {
            a: 0.54,
            ..p.foreground
        }),
        ..Default::default()
    }
}
fn muted_text(theme: &Theme) -> text_widget::Style {
    text_widget::Style {
        color: Some(Color {
            a: 0.54,
            ..theme.palette().text
        }),
    }
}
fn empty_state<'a>(title: &'a str, body: String, retry: Option<Action>) -> Element<'a, Action> {
    let mut content = column![
        text(title).size(11).style(muted_text),
        text(body).size(11).style(muted_text)
    ]
    .spacing(6)
    .align_x(Alignment::Center);
    if let Some(action) = retry {
        content = content.push(
            button(text("Try again").size(11))
                .on_press(action)
                .padding([3, 6])
                .style(quiet_button),
        );
    }
    container(content)
        .padding(16)
        .center_x(Fill)
        .center_y(Fill)
        .height(Fill)
        .width(Fill)
        .into()
}

fn local_path(path: &Path) -> io::Result<PathBuf> {
    let mut result = PathBuf::new();
    for part in path.components() {
        match part {
            Component::Normal(part) => result.push(part),
            Component::CurDir => {}
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Path is outside the selected workspace",
                ));
            }
        }
    }
    if result.as_os_str().len() > 4096 {
        return Err(io::Error::other("Path exceeds the supported length"));
    }
    Ok(result)
}
// Exact reveal floors from public src/lib/excluded-paths.ts.
fn excluded(name: &str, show_ignored: bool) -> bool {
    matches!(
        name,
        ".git"
            | ".hg"
            | ".svn"
            | "node_modules"
            | "bower_components"
            | ".venv"
            | "venv"
            | "__pycache__"
            | ".mypy_cache"
            | ".pytest_cache"
            | ".ruff_cache"
            | ".tox"
            | ".gradle"
            | ".idea"
            | "dist"
            | "build"
            | "target"
            | "Pods"
            | ".next"
            | ".nuxt"
            | ".svelte-kit"
            | ".output"
            | ".turbo"
            | ".parcel-cache"
            | ".cache"
            | ".terraform"
            | ".expo"
            | ".angular"
            | ".DS_Store"
            | "Thumbs.db"
    ) || (!show_ignored && matches!(name, "env" | "out" | "vendor" | "coverage"))
}
fn list(workspace: &Workspace, relative: &Path, show_ignored: bool) -> io::Result<Listing> {
    let relative = local_path(relative)?;
    let resolved = workspace.resolve(&relative)?;
    let directory = workspace.dir.open_dir(if resolved.as_os_str().is_empty() {
        Path::new(".")
    } else {
        &resolved
    })?;
    let mut rows = Vec::new();
    let mut omitted = 0;
    let mut truncated = false;
    for (index, entry) in directory.entries()?.enumerate() {
        if index >= MAX_DIRECTORY_SCAN {
            truncated = true;
            break;
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => {
                omitted += 1;
                continue;
            }
        };
        let name = entry.file_name();
        let Some(title) = name.to_str() else {
            omitted += 1;
            continue;
        };
        if excluded(title, show_ignored) {
            continue;
        }
        let metadata = match directory.symlink_metadata(&name) {
            Ok(metadata) => metadata,
            Err(_) => {
                omitted += 1;
                continue;
            }
        };
        let path = relative.join(&name);
        let symlink = metadata.file_type().is_symlink();
        let is_directory = if symlink {
            workspace
                .dir
                .metadata(&path)
                .is_ok_and(|metadata| metadata.is_dir())
        } else {
            metadata.is_dir()
        };
        rows.push(Row {
            path,
            title: title.into(),
            directory: is_directory,
        });
    }
    if !show_ignored {
        let ignored = git::tree_ignored(
            &workspace.root,
            &rows.iter().map(|row| row.path.clone()).collect::<Vec<_>>(),
        );
        rows.retain(|row| {
            !ignored.contains(&row.path)
                || (row.directory
                    && workspace
                        .dir
                        .symlink_metadata(row.path.join(".git"))
                        .is_ok())
        });
    }
    rows.sort_by_cached_key(|row| (!row.directory, row.title.to_lowercase(), row.title.clone()));
    if rows.len() > MAX_DIRECTORY_ROWS {
        rows.truncate(MAX_DIRECTORY_ROWS);
        truncated = true;
    }
    Ok(Listing {
        rows,
        truncated,
        omitted,
    })
}
fn preview(workspace: &Workspace, relative: &Path, allow_large: bool) -> io::Result<Preview> {
    let relative = local_path(relative)?;
    let resolved = workspace.resolve(&relative)?;
    let metadata = workspace.dir.metadata(&resolved)?;
    if !metadata.is_file() {
        return Err(io::Error::other("Only regular files can be previewed"));
    }
    let mut file = workspace.dir.open(&resolved)?;
    if !file.metadata()?.is_file() {
        return Err(io::Error::other("File changed before preview"));
    }
    let name = relative.to_string_lossy().into_owned();
    let mut bytes = Vec::new();
    file.by_ref().take(32).read_to_end(&mut bytes)?;
    let image_format = image::guess_format(&bytes)
        .ok()
        .filter(|format| matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg));
    let mut result = Preview {
        name,
        body: PreviewBody::Binary,
        size: metadata.len(),
        truncated: false,
    };
    if image_format.is_some() {
        if metadata.len() > IMAGE_BYTES as u64 {
            result.body =
                PreviewBody::Unavailable("Image exceeds the 16 MiB preview limit.".into());
            return Ok(result);
        }
        file.take((IMAGE_BYTES + 1).saturating_sub(bytes.len()) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > IMAGE_BYTES {
            return Err(io::Error::other("Image exceeds the preview limit"));
        }
        let dimensions = image::ImageReader::new(Cursor::new(&bytes))
            .with_guessed_format()?
            .into_dimensions()
            .map_err(io::Error::other)?;
        if u64::from(dimensions.0) * u64::from(dimensions.1) > MAX_IMAGE_PIXELS
            || dimensions.0 > 8192
            || dimensions.1 > 8192
        {
            return Err(io::Error::other(
                "Image dimensions exceed the preview limit",
            ));
        }
        let mut reader = image::ImageReader::new(Cursor::new(&bytes)).with_guessed_format()?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(8192);
        limits.max_image_height = Some(8192);
        limits.max_alloc = Some(64 * 1024 * 1024);
        reader.limits(limits);
        let rgba = reader.decode().map_err(io::Error::other)?.into_rgba8();
        result.body = PreviewBody::Image {
            handle: widget::image::Handle::from_rgba(rgba.width(), rgba.height(), rgba.into_raw()),
            width: dimensions.0,
            height: dimensions.1,
        };
        return Ok(result);
    }
    if metadata.len() > PREVIEW_BYTES as u64 && !allow_large {
        result.body = PreviewBody::Large;
        return Ok(result);
    }
    file.take((PREVIEW_BYTES + 1).saturating_sub(bytes.len()) as u64)
        .read_to_end(&mut bytes)?;
    result.truncated = bytes.len() > PREVIEW_BYTES;
    bytes.truncate(PREVIEW_BYTES);
    if !bytes.contains(&0) {
        result.body = match String::from_utf8(bytes) {
            Ok(text) => PreviewBody::Text(text),
            Err(error) if result.truncated && error.utf8_error().error_len().is_none() => {
                let valid = error.utf8_error().valid_up_to();
                PreviewBody::Text(
                    String::from_utf8_lossy(&error.into_bytes()[..valid]).into_owned(),
                )
            }
            Err(_) => PreviewBody::Binary,
        };
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    struct Tree {
        path: PathBuf,
    }
    impl Tree {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "hermes-files-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self { path }
        }
        fn write(&self, path: &str, body: &[u8]) {
            let path = self.path.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, body).unwrap();
        }
        fn workspace(&self) -> Workspace {
            Workspace::new(&self.path).unwrap()
        }
        fn git(&self, arguments: &[&str]) {
            assert!(
                std::process::Command::new("git")
                    .arg("-C")
                    .arg(&self.path)
                    .args(arguments)
                    .env("GIT_CONFIG_NOSYSTEM", "1")
                    .env("GIT_CONFIG_GLOBAL", "/dev/null")
                    .status()
                    .unwrap()
                    .success()
            );
        }
    }
    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
    #[test]
    fn hierarchy_sorts_folders_first_preserves_names_and_exclusion_floors() {
        let tree = Tree::new();
        tree.write("zebra.txt", b"z");
        tree.write("Alpha.txt", b"a");
        tree.write("folder/child.txt", b"c");
        tree.write("out/result.txt", b"x");
        tree.write("node_modules/hidden.txt", b"x");
        let workspace = tree.workspace();
        let root = list(&workspace, Path::new(""), false).unwrap();
        assert_eq!(
            root.rows
                .iter()
                .map(|row| row.title.as_str())
                .collect::<Vec<_>>(),
            vec!["folder", "Alpha.txt", "zebra.txt"]
        );
        assert_eq!(
            list(&workspace, Path::new("folder"), false).unwrap().rows[0].path,
            Path::new("folder/child.txt")
        );
        let revealed = list(&workspace, Path::new(""), true).unwrap();
        assert!(revealed.rows.iter().any(|row| row.title == "out"));
        assert!(!revealed.rows.iter().any(|row| row.title == "node_modules"));
    }
    #[test]
    fn gitignore_toggle_negation_and_nested_repository_boundaries() {
        let tree = Tree::new();
        tree.git(&["init", "-q"]);
        tree.write(".gitignore", b"*.log\n!keep.log\nnested/\n");
        tree.write("hidden.log", b"x");
        tree.write("keep.log", b"x");
        tree.write("nested/file.txt", b"x");
        tree.git(&["-C", "nested", "init", "-q"]);
        let workspace = tree.workspace();
        let root = list(&workspace, Path::new(""), false).unwrap();
        assert!(!root.rows.iter().any(|row| row.title == "hidden.log"));
        assert!(root.rows.iter().any(|row| row.title == "keep.log"));
        assert!(root.rows.iter().any(|row| row.title == "nested"));
        assert!(
            list(&workspace, Path::new(""), true)
                .unwrap()
                .rows
                .iter()
                .any(|row| row.title == "hidden.log")
        );
    }
    #[test]
    fn nested_expansion_and_stale_results_do_not_replace_new_intent() {
        let tree = Tree::new();
        tree.write("a/note.txt", b"one");
        tree.write("b/note.txt", b"two");
        let workspace = Arc::new(tree.workspace());
        let root = Arc::new(list(&workspace, Path::new(""), true).unwrap());
        let mut files = Files::new();
        files.root = Some(tree.path.clone());
        files.workspace = Some(workspace.clone());
        files.branches.insert(
            PathBuf::new(),
            Branch {
                listing: Some(root),
                ..Branch::default()
            },
        );
        files.expanded.insert("a".into());
        files.branches.insert(
            "a".into(),
            Branch {
                listing: Some(Arc::new(list(&workspace, Path::new("a"), true).unwrap())),
                request: 12,
                ..Branch::default()
            },
        );
        let mut visible = Vec::new();
        files.visible(Path::new(""), 0, &mut visible);
        assert_eq!(visible.len(), 3);
        let _ = files.update(Action::Listed(11, "a".into(), Err("old error".into())));
        assert!(files.branches[Path::new("a")].error.is_none());
        let _ = files.update(Action::CollapseAll);
        let mut visible = Vec::new();
        files.visible(Path::new(""), 0, &mut visible);
        assert_eq!(visible.len(), 2);
        files.file_version = 9;
        let _ = files.update(Action::Read(
            8,
            Ok(Arc::new(
                preview(&workspace, Path::new("a/note.txt"), false).unwrap(),
            )),
        ));
        assert!(files.preview.is_none());
        let _ = files.reset(None);
        assert!(files.workspace.is_none());
        assert!(files.branches.is_empty());
        assert!(files.preview_view().is_none());
    }
    #[test]
    fn text_binary_large_and_png_previews_use_real_bounded_reads() {
        let tree = Tree::new();
        tree.write("text.txt", "native β preview".as_bytes());
        tree.write("binary.bin", &[0, 1, 2]);
        tree.write("large.txt", &vec![b'x'; PREVIEW_BYTES + 50]);
        let workspace = tree.workspace();
        assert!(
            matches!(preview(&workspace, Path::new("text.txt"), false).unwrap().body, PreviewBody::Text(ref text) if text == "native β preview")
        );
        assert!(matches!(
            preview(&workspace, Path::new("binary.bin"), false)
                .unwrap()
                .body,
            PreviewBody::Binary
        ));
        assert!(matches!(
            preview(&workspace, Path::new("large.txt"), false)
                .unwrap()
                .body,
            PreviewBody::Large
        ));
        let large = preview(&workspace, Path::new("large.txt"), true).unwrap();
        assert!(large.truncated);
        assert!(matches!(large.body, PreviewBody::Text(ref text) if text.len() == PREVIEW_BYTES));
        let png = image::RgbaImage::from_pixel(2, 3, image::Rgba([30, 60, 90, 255]));
        png.save(tree.path.join("image.png")).unwrap();
        assert!(matches!(
            preview(&workspace, Path::new("image.png"), false)
                .unwrap()
                .body,
            PreviewBody::Image {
                width: 2,
                height: 3,
                ..
            }
        ));
    }
    #[cfg(unix)]
    #[test]
    fn capability_rejects_parent_and_symlink_escapes_but_supports_internal_links() {
        use std::os::unix::fs::symlink;
        let tree = Tree::new();
        let outside = Tree::new();
        tree.write("real/note.txt", b"inside");
        outside.write("secret.txt", b"private");
        symlink(&outside.path, tree.path.join("escape")).unwrap();
        symlink("real", tree.path.join("alias")).unwrap();
        let workspace = tree.workspace();
        assert!(preview(&workspace, Path::new("../secret.txt"), false).is_err());
        assert!(preview(&workspace, &outside.path.join("secret.txt"), false).is_err());
        assert!(preview(&workspace, Path::new("escape/secret.txt"), false).is_err());
        assert!(list(&workspace, Path::new("escape"), true).is_err());
        assert!(
            matches!(preview(&workspace, Path::new("alias/note.txt"), false).unwrap().body, PreviewBody::Text(ref text) if text == "inside")
        );
        assert!(
            list(&workspace, Path::new(""), true)
                .unwrap()
                .rows
                .iter()
                .any(|row| row.title == "escape" && !row.directory)
        );
    }
}
