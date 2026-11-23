//! Native command palette, ported from official Hermes v0.23.3.
//!
//! Sources: app/command-palette/index.tsx (`scoreItem`, `rankGroups`, root
//! groups, looped selection), app/floating-hud.ts, store/command-palette.ts,
//! lib/keybinds/{actions,combo}.ts, i18n/en.ts and the terminal contribution.
//! Only working native actions are offered. No plugin, gateway restart, paid
//! operation, empty destination or unimplemented appearance picker is invented.

use crate::appearance::text;
use crate::ui;
use iced::{
    Alignment, Border, Color, Element, Fill, Font, Length, Padding, Shadow, Task, Vector, keyboard,
    widget::{
        self, Space, button, column, container, mouse_area, row, scrollable, stack, text_input,
    },
};
use std::fmt;

const INPUT_ID: &str = "hermes-command-palette-input";
const LIST_ID: &str = "hermes-command-palette-list";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    NewSession,
    Settings,
    Capabilities,
    Artifacts,
    OpenFolder,
    SessionsImport,
    Models,
    Accounts,
    About,
    ToggleSidebar,
    ToggleStatusbar,
    ToggleHud,
    TogglePet,
    PetGallery,
    FlipPanes,
    Files,
    Review,
    Terminal,
    NewTerminal,
    Stop,
    QueueCurrent,
    ResumeQueue,
    Shortcuts,
    Welcome,
}

#[derive(Clone)]
pub enum Message {
    Query(String),
    Previous,
    Next,
    Activate,
    Choose(Command),
    Hover(Command),
    Close,
}
// Queries can contain pasted paths or conversation text. Do not log them.
impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Query(_) => "PaletteQuery",
            Self::Previous => "PalettePrevious",
            Self::Next => "PaletteNext",
            Self::Activate => "PaletteActivate",
            Self::Choose(_) => "PaletteChoose",
            Self::Hover(_) => "PaletteHover",
            Self::Close => "PaletteClose",
        })
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Context {
    pub running: bool,
    pub composer_has_text: bool,
    pub queued: bool,
    pub terminal_visible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Group {
    GoTo,
    Projects,
    Commands,
    CommandCenter,
    Settings,
}
impl Group {
    fn label(self) -> &'static str {
        match self {
            Self::GoTo => "Go to",
            Self::Projects => "Projects",
            Self::Commands => "Commands",
            Self::CommandCenter => "Command Center",
            Self::Settings => "Settings",
        }
    }
}
#[derive(Debug, Clone, Copy)]
struct Item {
    command: Command,
    group: Group,
    label: &'static str,
    keywords: &'static [&'static str],
    combo: Option<&'static str>,
    icon: &'static str,
    keep_open: bool,
}
macro_rules! item {
    ($command:ident,$group:ident,$label:literal,$keys:expr,$combo:expr,$icon:literal) => {
        Item {
            command: Command::$command,
            group: Group::$group,
            label: $label,
            keywords: $keys,
            combo: $combo,
            icon: $icon,
            keep_open: false,
        }
    };
}
// Group order and item order are intentional stable tiebreakers, as upstream.
const ITEMS: &[Item] = &[
    item!(
        NewSession,
        GoTo,
        "New session",
        &["chat", "create"],
        Some("mod+n"),
        "add"
    ),
    item!(
        Settings,
        GoTo,
        "Settings",
        &[],
        Some("mod+,"),
        "settings-gear"
    ),
    item!(
        Capabilities,
        GoTo,
        "Capabilities",
        &["skills", "tools", "toolsets", "mcp", "capabilities"],
        None,
        "symbol-misc"
    ),
    item!(Artifacts, GoTo, "Artifacts", &[], None, "files"),
    item!(
        OpenFolder,
        Projects,
        "Open folder as project…",
        &[
            "open",
            "folder",
            "directory",
            "project",
            "add",
            "import",
            "workspace"
        ],
        Some("mod+o"),
        "folder-opened"
    ),
    // Working view actions reuse the official keybind action copy. The terminal
    // toggle preserves the contribution's keepOpen and truthful state detail.
    item!(
        ToggleSidebar,
        Commands,
        "Toggle sessions sidebar",
        &["sidebar", "sessions", "show", "hide"],
        Some("mod+b"),
        "layout-sidebar-left"
    ),
    item!(ToggleStatusbar, Commands, "Toggle status bar", &["status", "bar", "show", "hide"], Some("mod+shift+s"), "layout-panel"),
    item!(PetGallery, Commands, "Pet gallery", &["pet", "adopt", "petdex", "mascot"], None, "smiley"),
    item!(TogglePet, Commands, "Toggle desktop pet", &["pet", "mascot", "egg", "companion", "overlay"], None, "smiley"),
    item!(ToggleHud, Commands, "Toggle HUD mode", &["hud", "floating", "overlay", "spotlight", "band"], Some("mod+shift+h"), "comment-discussion"),
    item!(
        FlipPanes,
        Commands,
        "Swap sidebar sides",
        &["layout", "panes", "swap"],
        Some("mod+\\"),
        "arrow-swap"
    ),
    item!(
        Files,
        Commands,
        "Show file browser",
        &["files", "browser", "workspace"],
        None,
        "files"
    ),
    item!(
        Review,
        Commands,
        "Toggle review pane",
        &["review", "git", "diff", "source control"],
        Some("mod+g"),
        "git-branch"
    ),
    Item {
        command: Command::Terminal,
        group: Group::Commands,
        label: "Toggle terminal",
        keywords: &[
            "terminal", "shell", "console", "pty", "on", "off", "enable", "disable",
        ],
        combo: Some("ctrl+`"),
        icon: "terminal",
        keep_open: true,
    },
    item!(
        NewTerminal,
        Commands,
        "New terminal",
        &["terminal", "shell", "console", "pty"],
        Some("ctrl+shift+`"),
        "add"
    ),
    item!(
        Stop,
        Commands,
        "Stop",
        &["stop", "interrupt", "cancel"],
        Some("escape"),
        "debug-stop"
    ),
    item!(
        QueueCurrent,
        Commands,
        "Queue message",
        &["queue", "message"],
        Some("mod+enter"),
        "list-ordered"
    ),
    item!(
        ResumeQueue,
        Commands,
        "Send next queued turn",
        &["queue", "resume", "send", "paused"],
        Some("mod+shift+k"),
        "play"
    ),
    item!(
        SessionsImport,
        CommandCenter,
        "Import session",
        &["import", "claude", "codex", "conversation"],
        None,
        "cloud-download"
    ),
    item!(
        Models,
        Settings,
        "Model",
        &["settings", "model"],
        None,
        "symbol-misc"
    ),
    item!(
        Accounts,
        Settings,
        "Accounts",
        &[
            "settings",
            "accounts",
            "sign in",
            "oauth",
            "login",
            "subscription",
            "models",
            "anthropic",
            "openai"
        ],
        None,
        "account"
    ),
    item!(
        Welcome,
        Settings,
        "Welcome to Hermes",
        &["onboarding", "welcome", "make it yours", "artwork", "accent", "theme"],
        None,
        "home"
    ),
    item!(
        Shortcuts,
        Settings,
        "Keyboard shortcuts",
        &["keybinds", "shortcuts", "hotkeys", "keyboard"],
        Some("mod+/"),
        "keyboard"
    ),
    item!(
        About,
        Settings,
        "About",
        &["settings", "version", "about"],
        None,
        "info"
    ),
];

#[derive(Default)]
pub struct Commands {
    open: bool,
    query: String,
    selected: usize,
    context: Context,
    bindings: Vec<(Command, Vec<String>)>,
}
impl Commands {
    /// Fresh opening resets search/highlight, matching the remounted TS body.
    pub fn open(&mut self) -> Task<Message> {
        self.open = true;
        self.query.clear();
        self.selected = 0;
        widget::operation::focus(INPUT_ID)
    }
    pub fn close(&mut self) {
        self.open = false;
        self.query.clear();
        self.selected = 0;
    }
    pub fn is_open(&self) -> bool {
        self.open
    }
    pub fn focus(&self) -> Task<Message> {
        widget::operation::focus(INPUT_ID)
    }
    pub fn set_bindings(&mut self, bindings: Vec<(Command, Vec<String>)>) {
        self.bindings = bindings;
    }
    pub fn set_context(&mut self, context: Context) {
        if self.context == context {
            return;
        }
        let selected = self.selected_command();
        self.context = context;
        let items = self.visible();
        self.selected = selected
            .and_then(|command| items.iter().position(|item| item.command == command))
            .unwrap_or(0);
    }
    pub fn selected_command(&self) -> Option<Command> {
        self.visible().get(self.selected).map(|item| item.command)
    }
    pub fn update(&mut self, message: Message) -> (Task<Message>, Option<Command>) {
        if !self.open {
            return (Task::none(), None);
        }
        match message {
            Message::Query(query) => {
                self.query = query;
                self.selected = 0;
                return (
                    widget::operation::snap_to(LIST_ID, scrollable::RelativeOffset::START),
                    None,
                );
            }
            Message::Previous => {
                let count = self.visible().len();
                if count > 0 {
                    self.selected = if self.selected == 0 {
                        count - 1
                    } else {
                        self.selected - 1
                    };
                }
            }
            Message::Next => {
                let count = self.visible().len();
                if count > 0 {
                    self.selected = (self.selected + 1) % count;
                }
            }
            Message::Hover(command) => {
                if let Some(index) = self
                    .visible()
                    .iter()
                    .position(|item| item.command == command)
                {
                    self.selected = index;
                }
                return (Task::none(), None);
            }
            Message::Close => {
                self.close();
                return (Task::none(), None);
            }
            Message::Activate => return self.select(self.selected_command()),
            Message::Choose(command) => return self.select(Some(command)),
        }
        let count = self.visible().len();
        let y = if count > 1 {
            self.selected as f32 / (count - 1) as f32
        } else {
            0.0
        };
        (
            widget::operation::snap_to(LIST_ID, scrollable::RelativeOffset { x: 0.0, y }),
            None,
        )
    }
    fn select(&mut self, command: Option<Command>) -> (Task<Message>, Option<Command>) {
        let Some(item) = command.and_then(|command| {
            self.visible()
                .into_iter()
                .find(|item| item.command == command)
        }) else {
            return (Task::none(), None);
        };
        if !item.keep_open {
            self.close();
        }
        (Task::none(), Some(item.command))
    }
    fn visible(&self) -> Vec<&'static Item> {
        let allowed = |item: &&Item| match item.command {
            Command::NewSession | Command::OpenFolder => !self.context.running,
            Command::Stop => self.context.running,
            Command::QueueCurrent => self.context.composer_has_text,
            Command::ResumeQueue => self.context.queued && !self.context.running,
            _ => true,
        };
        let needle = self.query.trim().to_lowercase();
        let mut groups: Vec<Vec<(&Item, u8)>> = Vec::new();
        for group in [
            Group::GoTo,
            Group::Projects,
            Group::Commands,
            Group::CommandCenter,
            Group::Settings,
        ] {
            let mut scored: Vec<_> = ITEMS
                .iter()
                .filter(allowed)
                .filter(|item| item.group == group)
                .map(|item| {
                    (
                        item,
                        if needle.is_empty() {
                            100
                        } else {
                            score(item.label, item.keywords, &needle)
                        },
                    )
                })
                .filter(|(_, grade)| *grade > 0)
                .collect();
            // sort_by is stable: curated order breaks equal scores.
            scored.sort_by_key(|entry| std::cmp::Reverse(entry.1));
            if !scored.is_empty() {
                groups.push(scored);
            }
        }
        if !needle.is_empty() {
            groups.sort_by(|a, b| b[0].1.cmp(&a[0].1));
        }
        groups.into_iter().flatten().map(|(item, _)| item).collect()
    }
    /// Full-size transparent click-away layer, with the official compact HUD.
    /// Parent stacks this above its regular shell and intercepts its key events.
    pub fn view(&self, dark: bool) -> Element<'_, Message> {
        if !self.open {
            return Space::new().into();
        }
        widget::responsive(move |size| {
            let top = if size.width < 704.0 { 40.0 } else { 12.0 };
            stack![
                mouse_area(container(Space::new()).width(Fill).height(Fill))
                    .on_press(Message::Close),
                container(self.card(dark))
                    .width(Fill)
                    .height(Fill)
                    .align_x(Alignment::Center)
                    .padding(Padding {
                        top,
                        right: 16.0,
                        bottom: 16.0,
                        left: 16.0
                    }),
            ]
            .width(Fill)
            .height(Fill)
            .into()
        })
        .into()
    }
    fn card(&self, dark: bool) -> Element<'_, Message> {
        let p = ui::palette(dark);
        let input = row![
            ui::icon("search", 14, p.muted),
            text_input("Search sessions, views, and actions", &self.query)
                .id(INPUT_ID)
                .on_input(Message::Query)
                .on_submit(Message::Activate)
                .padding([10, 0])
                .size(12),
            button(text("esc").size(10).color(p.muted))
                .on_press(Message::Close)
                .padding([2, 5])
                .style(button::text)
        ]
        .spacing(9)
        .align_y(Alignment::Center);
        let mut list = column![].spacing(1);
        let mut previous = None;
        let items = self.visible();
        if items.is_empty() {
            list = list.push(
                container(text("No matching results found.").size(12).color(p.muted))
                    .padding([24, 12])
                    .width(Fill)
                    .center_x(Fill),
            );
        }
        for (index, item) in items.iter().enumerate() {
            if previous != Some(item.group) {
                list = list.push(
                    container(
                        text(item.group.label().to_uppercase())
                            .size(10)
                            .color(p.accent)
                            .font(Font {
                                weight: iced::font::Weight::Semibold,
                                ..Font::DEFAULT
                            }),
                    )
                    .padding(Padding {
                        top: 10.0,
                        right: 10.0,
                        bottom: 3.0,
                        left: 10.0,
                    }),
                );
                previous = Some(item.group);
            }
            let selected = index == self.selected;
            let command = item.command;
            let mut contents = row![
                ui::icon(item.icon, 14, if selected { p.foreground } else { p.muted }),
                text(item.label).size(12)
            ]
            .spacing(8)
            .align_y(Alignment::Center);
            if item.command == Command::Terminal {
                contents = contents.push(
                    text(if self.context.terminal_visible {
                        "on"
                    } else {
                        "off"
                    })
                    .size(11)
                    .color(p.muted),
                );
            }
            contents = contents.push(Space::new().width(Fill));
            let rebound = self
                .bindings
                .iter()
                .find(|(command, _)| *command == item.command);
            let combo = rebound
                .map(|(_, combos)| combos.first().map(String::as_str))
                .unwrap_or(item.combo);
            if let Some(combo) = combo {
                contents = contents.push(text(shortcut_label(combo)).size(10).color(p.muted));
            }
            let action = button(contents)
                .on_press(Message::Choose(command))
                .padding([5, 8])
                .width(Fill)
                .style(move |_, status| {
                    let background = if selected
                        || matches!(status, button::Status::Hovered | button::Status::Pressed)
                    {
                        Some(p.secondary.into())
                    } else {
                        None
                    };
                    button::Style {
                        background,
                        text_color: p.foreground,
                        border: Border {
                            radius: 5.0.into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                });
            // Pointer motion, rather than mere entry on opening/reflow, changes
            // highlight. A parked cursor cannot steal the keyboard selection.
            list = list.push(mouse_area(action).on_move(move |_| Message::Hover(command)));
        }
        let listing = scrollable(container(list).padding([0, 4]))
            .id(LIST_ID)
            .height(Length::Shrink);
        let listing = container(listing).max_height(400);
        container(column![
            container(input).padding([0, 12]),
            widget::rule::horizontal(1),
            listing,
            Space::new().height(6)
        ])
        .max_width(544)
        .style(move |_| container::Style {
            background: Some(p.background.into()),
            text_color: Some(p.foreground),
            border: Border {
                radius: 12.0.into(),
                width: 1.0,
                color: p.border,
            },
            shadow: Shadow {
                color: Color {
                    a: 0.18,
                    ..Color::BLACK
                },
                offset: Vector::new(0.0, 8.0),
                blur_radius: 24.0,
            },
            ..Default::default()
        })
        .into()
    }
}

/// Exact integer-percent equivalent of official scoreItem's 1/.9/.85/.8/.7/.6/.4.
/// Every word must match label OR keywords. Label matches outrank keyword-only.
fn score(label: &str, keywords: &[&str], needle: &str) -> u8 {
    let label = label.to_lowercase();
    let keys = keywords.join(" ").to_lowercase();
    let terms: Vec<_> = needle.split_whitespace().collect();
    if terms
        .iter()
        .any(|term| !label.contains(term) && !keys.contains(term))
    {
        return 0;
    }
    if label == needle {
        return 100;
    }
    if label.starts_with(needle) {
        return 90;
    }
    let words: Vec<_> = label
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    if words.contains(&needle) {
        return 85;
    }
    if words.iter().any(|word| word.starts_with(needle)) {
        return 80;
    }
    if label.contains(needle) {
        return 70;
    }
    if terms.iter().all(|term| label.contains(term)) {
        return 60;
    }
    40
}

fn shortcut_label(combo: &str) -> String {
    let platform = if cfg!(target_os = "macos") {
        "⌘"
    } else {
        "Ctrl"
    };
    combo
        .split('+')
        .map(|part| match part {
            "mod" => platform,
            "ctrl" => "Ctrl",
            "shift" => "⇧",
            "alt" => "Alt",
            "enter" => "↵",
            "escape" => "Esc",
            other => other,
        })
        .collect::<Vec<_>>()
        .join(" ")
}


/// Route these only while the palette is open, before composer/global shortcuts.
pub fn key_message(event: &keyboard::Event) -> Option<Message> {
    let keyboard::Event::KeyPressed { key, modifiers, .. } = event else {
        return None;
    };
    if !modifiers.is_empty() {
        return None;
    }
    match key.as_ref() {
        keyboard::Key::Named(keyboard::key::Named::ArrowUp) => Some(Message::Previous),
        keyboard::Key::Named(keyboard::key::Named::ArrowDown) => Some(Message::Next),
        keyboard::Key::Named(keyboard::key::Named::Enter) => Some(Message::Activate),
        keyboard::Key::Named(keyboard::key::Named::Escape) => Some(Message::Close),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_score_grades_and_all_terms_match() {
        assert_eq!(score("Tools", &[], "tools"), 100);
        assert_eq!(score("Tools and skills", &[], "tools"), 90);
        assert_eq!(score("Workspace Tools", &[], "tools"), 85);
        assert_eq!(score("Workspace Toolsets", &[], "tools"), 80);
        assert_eq!(score("Toolsets", &[], "ool"), 70);
        assert_eq!(
            score("Workspace file browser", &[], "browser workspace"),
            60
        );
        assert_eq!(score("Capabilities", &["tools"], "tools"), 40);
        assert_eq!(
            score("Capabilities", &["tools", "skills"], "tools missing"),
            0
        );
    }
    #[test]
    fn search_highlights_best_match_and_stable_groups_break_ties() {
        let mut palette = Commands::default();
        let _ = palette.open();
        assert_eq!(palette.selected_command(), Some(Command::NewSession));
        let _ = palette.update(Message::Query("  terminal  ".into()));
        assert_eq!(palette.selected_command(), Some(Command::Terminal));
        let visible = palette.visible();
        assert_eq!(visible[0].label, "Toggle terminal");
        assert_eq!(visible[1].label, "New terminal");
        let _ = palette.update(Message::Query("model".into()));
        assert_eq!(palette.selected_command(), Some(Command::Models));
        let _ = palette.update(Message::Query("model login".into()));
        assert_eq!(palette.selected_command(), Some(Command::Accounts));
        let _ = palette.update(Message::Query("model unavailable".into()));
        assert!(palette.selected_command().is_none());
    }
    #[test]
    fn arrows_wrap_and_enter_runs_exact_highlight_then_closes() {
        let mut palette = Commands::default();
        let _ = palette.open();
        let first = palette.selected_command();
        let _ = palette.update(Message::Previous);
        assert_eq!(palette.selected_command(), Some(Command::About));
        let _ = palette.update(Message::Next);
        assert_eq!(palette.selected_command(), first);
        let _ = palette.update(Message::Query("import".into()));
        assert_eq!(palette.selected_command(), Some(Command::SessionsImport));
        let (_, action) = palette.update(Message::Activate);
        assert_eq!(action, Some(Command::SessionsImport));
        assert!(!palette.is_open());
        let (_, action) = palette.update(Message::Activate);
        assert!(action.is_none());
    }
    #[test]
    fn unsupported_or_unavailable_actions_never_select_and_toggle_stays_open() {
        let mut palette = Commands::default();
        let _ = palette.open();
        let (_, action) = palette.update(Message::Choose(Command::Stop));
        assert!(action.is_none());
        assert!(palette.is_open());
        palette.set_context(Context {
            running: true,
            composer_has_text: true,
            queued: true,
            ..Context::default()
        });
        assert!(
            !palette
                .visible()
                .iter()
                .any(|item| item.command == Command::NewSession)
        );
        assert!(
            palette
                .visible()
                .iter()
                .any(|item| item.command == Command::Stop)
        );
        assert!(
            !palette
                .visible()
                .iter()
                .any(|item| item.command == Command::ResumeQueue)
        );
        let (_, action) = palette.update(Message::Choose(Command::Terminal));
        assert_eq!(action, Some(Command::Terminal));
        assert!(palette.is_open());
        let _ = palette.update(Message::Query("no matching command".into()));
        let _ = palette.update(Message::Next);
        let (_, action) = palette.update(Message::Activate);
        assert!(action.is_none());
        let _ = palette.update(Message::Close);
        assert!(!palette.is_open());
        let _ = palette.open();
        assert!(palette.query.is_empty());
        assert!(
            format!("{:?}", Message::Query("synthetic-secret".into()))
                .find("synthetic-secret")
                .is_none()
        );
    }
}
