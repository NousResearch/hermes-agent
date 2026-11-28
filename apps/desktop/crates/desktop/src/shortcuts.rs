//! Native keyboard-shortcut settings and dispatch, ported from clean official
//! Hermes 0284ce17: store/keybinds.ts, lib/keybinds/{actions,combo}.ts,
//! app/settings/keybind-settings.tsx, app/hooks/use-keybinds.ts and i18n/en.ts.
//! Only implemented native actions are registered. The reference is read-only.

use crate::appearance::text;
use crate::{Hermes, Message as AppMessage, commands, ui};
use iced::{
    Alignment, Border, Element, Fill, Padding, Task, keyboard,
    widget::{self, Space, button, column, container, row, scrollable, text_input},
};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    FocusComposer,
    Models,
    NewTab, NextSession, PreviousSession, CloseTab, ReopenTab,
    NewSession,
    SearchSessions,
    PinSession,
    ArchiveSession,
    OpenFolder,
    Palette,
    Settings,
    Capabilities,
    Artifacts,
    Sidebar,
    Review,
    Files,
    Terminal,
    NewTerminal,
    ToggleStatusbar,
    FlipPanes,
    Shortcuts,
    Hud,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Category {
    Composer,
    Session,
    Navigation,
    View,
}
impl Category {
    fn label(self) -> &'static str {
        match self {
            Self::Composer => "Composer",
            Self::Session => "Session",
            Self::Navigation => "Navigation",
            Self::View => "View",
        }
    }
}
#[derive(Clone, Copy)]
struct Spec {
    action: Action,
    id: &'static str,
    label: &'static str,
    category: Category,
    defaults: &'static [&'static str],
}
macro_rules! spec {
    ($action:ident,$id:literal,$label:literal,$category:ident,$defaults:expr) => {
        Spec {
            action: Action::$action,
            id: $id,
            label: $label,
            category: Category::$category,
            defaults: $defaults,
        }
    };
}
const ACTIONS: &[Spec] = &[
    spec!(
        FocusComposer,
        "composer.focus",
        "Focus composer",
        Composer,
        &["/", "enter"]
    ),
    spec!(
        Models,
        "composer.modelPicker",
        "Open model picker",
        Composer,
        &["mod+shift+m"]
    ),
    spec!(NewTab,"session.newTab","New tab",Session,&["mod+t"]),
    spec!(NextSession,"session.next","Next session tab",Session,&["ctrl+tab","ctrl+pagedown"]),
    spec!(PreviousSession,"session.prev","Previous session tab",Session,&["ctrl+shift+tab","ctrl+pageup"]),
    spec!(CloseTab,"view.closeTab","Close tab",View,&["mod+w"]),
    spec!(ReopenTab,"view.reopenTab","Reopen closed tab",View,&["mod+shift+t"]),
    spec!(
        NewSession,
        "session.new",
        "New session",
        Session,
        &["mod+n"]
    ),
    spec!(
        SearchSessions,
        "session.focusSearch",
        "Search sessions",
        Session,
        &["mod+shift+f"]
    ),
    spec!(
        PinSession,
        "session.togglePin",
        "Pin / unpin current session",
        Session,
        &[]
    ),
    spec!(
        ArchiveSession,
        "session.archive",
        "Archive current session",
        Session,
        &[]
    ),
    spec!(
        OpenFolder,
        "workspace.openFolder",
        "Open folder as project",
        Session,
        &["mod+o"]
    ),
    spec!(
        Palette,
        "nav.commandPalette",
        "Open command palette",
        Navigation,
        &["mod+k", "mod+p"]
    ),
    spec!(
        Settings,
        "nav.settings",
        "Open settings",
        Navigation,
        &["mod+,"]
    ),
    spec!(
        Capabilities,
        "nav.capabilities",
        "Open skills",
        Navigation,
        &[]
    ),
    spec!(
        Artifacts,
        "nav.artifacts",
        "Open artifacts",
        Navigation,
        &[]
    ),
    spec!(
        Sidebar,
        "view.toggleSidebar",
        "Toggle sessions sidebar",
        View,
        &["mod+b"]
    ),
    spec!(
        Review,
        "view.toggleReview",
        "Toggle review pane",
        View,
        &["mod+g"]
    ),
    spec!(Files, "view.showFiles", "Show file browser", View, &[]),
    spec!(
        Terminal,
        "view.showTerminal",
        "Toggle terminal",
        View,
        &["ctrl+`"]
    ),
    spec!(
        NewTerminal,
        "view.newTerminal",
        "New terminal",
        View,
        &["ctrl+shift+`"]
    ),
    spec!(ToggleStatusbar, "view.toggleStatusbar", "Toggle status bar", View, &["mod+shift+s"]),
    spec!(Hud, "view.toggleHud", "Toggle HUD mode", View, &["mod+shift+h"]),
    spec!(
        FlipPanes,
        "view.flipPanes",
        "Swap sidebar sides",
        View,
        &["mod+\\"]
    ),
    spec!(
        Shortcuts,
        "keybinds.openPanel",
        "Open keyboard shortcuts",
        View,
        &["mod+/"]
    ),
];

#[derive(Clone)]
pub enum Message {
    Query(String),
    Begin(Action),
    Capture(keyboard::Event),
    Cancel,
    Clear(Action),
    Reset(Action),
    ResetAll,
    ToggleCategory(Category),
}
impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Query(_) => "ShortcutSearch",
            Self::Begin(_) => "ShortcutCapture",
            Self::Capture(_) => "ShortcutKey",
            Self::Cancel => "ShortcutCaptureCancelled",
            Self::Clear(_) => "ShortcutCleared",
            Self::Reset(_) => "ShortcutReset",
            Self::ResetAll => "ShortcutsReset",
            Self::ToggleCategory(_) => "ShortcutCategory",
        })
    }
}

#[derive(Clone, Default)]
pub struct Bindings {
    overrides: BTreeMap<String, Vec<String>>,
}
impl Bindings {
    pub fn combos(&self, action: Action) -> Vec<String> {
        let spec = spec_for(action);
        self.overrides
            .get(spec.id)
            .cloned()
            .unwrap_or_else(|| spec.defaults.iter().map(|s| (*s).into()).collect())
    }
    pub fn action_for_key(
        &self,
        key: &keyboard::Key,
        physical: keyboard::key::Physical,
        modifiers: keyboard::Modifiers,
        editable: bool,
    ) -> Option<Action> {
        let combo = combo_from_key(key, physical, modifiers)?;
        for spec in ACTIONS {
            if self
                .combos(spec.action)
                .iter()
                .any(|binding| canonicalize(binding) == combo)
            {
                // Same first-registration-wins index and editable-target safety.
                return if !editable || allowed_in_input(spec.action, &combo) {
                    Some(spec.action)
                } else {
                    None
                };
            }
        }
        None
    }
    pub fn action_for_event(&self, event: &keyboard::Event, editable: bool) -> Option<Action> {
        let keyboard::Event::KeyPressed {
            key,
            physical_key,
            modifiers,
            ..
        } = event
        else {
            return None;
        };
        self.action_for_key(key, *physical_key, *modifiers, editable)
    }
}
#[derive(Default)]
pub struct State {
    bindings: Bindings,
    path: Option<PathBuf>,
    query: String,
    capture: Option<Action>,
    collapsed: BTreeSet<Category>,
    error: Option<String>,
}
impl State {
    pub fn load(root: &Path) -> Self {
        let mut state = Self {
            path: Some(root.join("desktop-keybinds.json")),
            ..Self::default()
        };
        let path = state.path.as_ref().expect("assigned path");
        match fs::read(path) {
            Ok(bytes) if bytes.len() <= 262144 => match serde_json::from_slice::<Value>(&bytes) {
                Ok(Value::Object(values)) => {
                    for (id, value) in values {
                        if let Some(values) = value.as_array() {
                            state.bindings.overrides.insert(
                                id,
                                values
                                    .iter()
                                    .filter_map(Value::as_str)
                                    .filter(|s| s.len() <= 128)
                                    .take(32)
                                    .map(str::to_owned)
                                    .collect(),
                            );
                        }
                    }
                }
                _ => {
                    state.error =
                        Some("Could not read keyboard shortcuts. Defaults are active.".into())
                }
            },
            Ok(_) => {
                state.error =
                    Some("The keyboard shortcut file is too large. Defaults are active.".into())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => state.error = Some(format!("Could not read keyboard shortcuts: {error}")),
        }
        state
    }
    pub fn bindings(&self) -> &Bindings {
        &self.bindings
    }
    pub fn is_capturing(&self) -> bool {
        self.capture.is_some()
    }
    pub fn cancel_capture(&mut self) {
        self.capture = None;
    }
    pub fn combos(&self, action: Action) -> Vec<String> {
        self.bindings.combos(action)
    }
    pub fn palette_bindings(&self) -> Vec<(commands::Command, Vec<String>)> {
        use commands::Command;
        [
            (Command::NewSession, Action::NewSession),
            (Command::Settings, Action::Settings),
            (Command::Capabilities, Action::Capabilities),
            (Command::Artifacts, Action::Artifacts),
            (Command::OpenFolder, Action::OpenFolder),
            (Command::Models, Action::Models),
            (Command::ToggleSidebar, Action::Sidebar),
            (Command::ToggleStatusbar, Action::ToggleStatusbar),
            (Command::ToggleHud, Action::Hud),
            (Command::FlipPanes, Action::FlipPanes),
            (Command::Files, Action::Files),
            (Command::Review, Action::Review),
            (Command::Terminal, Action::Terminal),
            (Command::NewTerminal, Action::NewTerminal),
            (Command::Shortcuts, Action::Shortcuts),
        ]
        .into_iter()
        .map(|(command, action)| (command, self.combos(action)))
        .collect()
    }
    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Query(query) => self.query = query,
            Message::Begin(action) => {
                self.error = None;
                self.capture = if self.capture == Some(action) {
                    None
                } else {
                    Some(action)
                };
                return widget::operation::focus("hermes-shortcut-capture");
            }
            Message::Cancel => self.capture = None,
            Message::Capture(event) => {
                let Some(action) = self.capture else {
                    return Task::none();
                };
                let keyboard::Event::KeyPressed {
                    key,
                    physical_key,
                    modifiers,
                    ..
                } = event
                else {
                    return Task::none();
                };
                if key == keyboard::Key::Named(keyboard::key::Named::Escape) {
                    self.capture = None;
                } else if matches!(
                    key,
                    keyboard::Key::Named(
                        keyboard::key::Named::Backspace | keyboard::key::Named::Delete
                    )
                ) {
                    self.capture = None;
                    self.set(action, vec![]);
                } else if let Some(combo) = combo_from_key(&key, physical_key, modifiers) {
                    self.capture = None;
                    self.set(action, vec![combo]);
                }
            }
            Message::Clear(action) => self.set(action, vec![]),
            Message::Reset(action) => {
                let mut next = self.bindings.clone();
                next.overrides.remove(spec_for(action).id);
                self.commit(next);
            }
            Message::ResetAll => {
                let mut next = self.bindings.clone();
                next.overrides
                    .retain(|id, _| !ACTIONS.iter().any(|spec| spec.id == id));
                self.capture = None;
                self.commit(next);
            }
            Message::ToggleCategory(category) => {
                if !self.collapsed.remove(&category) {
                    self.collapsed.insert(category);
                }
            }
        }
        Task::none()
    }
    fn set(&mut self, action: Action, combos: Vec<String>) {
        let spec = spec_for(action);
        let mut next = self.bindings.clone();
        if combos
            == spec
                .defaults
                .iter()
                .map(|s| (*s).to_owned())
                .collect::<Vec<_>>()
        {
            next.overrides.remove(spec.id);
        } else {
            next.overrides.insert(spec.id.into(), combos);
        }
        self.commit(next);
    }
    fn commit(&mut self, next: Bindings) {
        let result = (|| -> Result<(), String> {
            let path = self
                .path
                .as_ref()
                .ok_or("Open the native library before saving shortcuts")?;
            let parent = path.parent().ok_or("Missing shortcut directory")?;
            let mut file = NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
            serde_json::to_writer_pretty(&mut file, &next.overrides).map_err(|e| e.to_string())?;
            file.write_all(b"\n").map_err(|e| e.to_string())?;
            file.as_file().sync_all().map_err(|e| e.to_string())?;
            file.persist(path).map_err(|e| e.to_string())?;
            #[cfg(unix)]
            File::open(parent)
                .and_then(|file| file.sync_all())
                .map_err(|e| e.to_string())?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.bindings = next;
                self.error = None;
            }
            Err(error) => self.error = Some(format!("Could not save keyboard shortcuts: {error}")),
        }
    }
    fn conflict(&self, action: Action) -> Option<&'static str> {
        let mine = self.combos(action);
        ACTIONS
            .iter()
            .find(|other| {
                other.action != action
                    && self.combos(other.action).iter().any(|combo| {
                        mine.iter()
                            .any(|own| canonicalize(own) == canonicalize(combo))
                    })
            })
            .map(|spec| spec.label)
    }
}
fn spec_for(action: Action) -> &'static Spec {
    ACTIONS
        .iter()
        .find(|spec| spec.action == action)
        .expect("registered native action")
}

/// Original combo.ts platform fold: literal Control stays distinct on macOS.
pub fn canonicalize(combo: &str) -> String {
    combo
        .split('+')
        .map(|part| {
            if part == "ctrl" && !cfg!(target_os = "macos") {
                "mod"
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join("+")
}
pub fn combo_from_key(
    key: &keyboard::Key,
    physical: keyboard::key::Physical,
    modifiers: keyboard::Modifiers,
) -> Option<String> {
    use keyboard::{
        Key,
        key::{Code, Named, Physical},
    };
    if matches!(
        key,
        Key::Named(
            Named::Process
                | Named::Alt
                | Named::AltGraph
                | Named::Control
                | Named::Meta
                | Named::Super
                | Named::Shift
        )
    ) {
        return None;
    }
    let physical_base = || -> Option<String> {
        let Physical::Code(code) = physical else {
            return None;
        };
        let name = format!("{code:?}");
        if let Some(value) = name.strip_prefix("Key").filter(|s| s.len() == 1) {
            return Some(value.to_lowercase());
        }
        if let Some(value) = name
            .strip_prefix("Digit")
            .or_else(|| name.strip_prefix("Numpad"))
            .filter(|s| s.len() == 1 && s.bytes().all(|b| b.is_ascii_digit()))
        {
            return Some(value.into());
        }
        if name.starts_with('F') && name[1..].bytes().all(|b| b.is_ascii_digit()) {
            return Some(name.to_lowercase());
        }
        Some(
            match code {
                Code::Backquote => "`",
                Code::Backslash => "\\",
                Code::BracketLeft => "[",
                Code::BracketRight => "]",
                Code::Comma => ",",
                Code::Equal => "=",
                Code::Minus => "-",
                Code::Period => ".",
                Code::Quote => "'",
                Code::Semicolon => ";",
                Code::Slash => "/",
                Code::Space => "space",
                Code::Enter | Code::NumpadEnter => "enter",
                Code::Escape => "escape",
                Code::Backspace => "backspace",
                Code::Delete => "delete",
                Code::CapsLock => "capslock",
                Code::Tab => "tab",
                Code::PageUp => "pageup",
                Code::PageDown => "pagedown",
                Code::ArrowUp => "up",
                Code::ArrowDown => "down",
                Code::ArrowLeft => "left",
                Code::ArrowRight => "right",
                _ => return None,
            }
            .into(),
        )
    };
    let base = match key.as_ref() {
        Key::Character(value)
            if value.len() == 1 && value.chars().all(|c| c.is_ascii_alphabetic()) =>
        {
            Some(value.to_ascii_lowercase())
        }
        Key::Character(value)
            if !modifiers.shift() && value.len() == 1 && "`\\[]=-.,';/".contains(value) =>
        {
            Some(value.into())
        }
        Key::Named(named) => Some(
            match named {
                Named::Enter => "enter",
                Named::Escape => "escape",
                Named::Backspace => "backspace",
                Named::Delete => "delete",
                Named::Tab => "tab",
                Named::Space => "space",
                Named::ArrowUp => "up",
                Named::ArrowDown => "down",
                Named::ArrowLeft => "left",
                Named::ArrowRight => "right",
                Named::Home => "home",
                Named::End => "end",
                Named::PageUp => "pageup",
                Named::PageDown => "pagedown",
                _ => return physical_base().map(|base| with_modifiers(&base, modifiers)),
            }
            .into(),
        ),
        _ => physical_base(),
    }?;
    Some(with_modifiers(&base, modifiers))
}
fn with_modifiers(base: &str, modifiers: keyboard::Modifiers) -> String {
    let mut parts = Vec::new();
    if modifiers.logo() || (!cfg!(target_os = "macos") && modifiers.control()) {
        parts.push("mod");
    }
    if cfg!(target_os = "macos") && modifiers.control() {
        parts.push("ctrl");
    }
    if modifiers.alt() {
        parts.push("alt");
    }
    if modifiers.shift() {
        parts.push("shift");
    }
    parts.push(base);
    parts.join("+")
}
fn allowed_in_input(action: Action, combo: &str) -> bool {
    let mut parts = combo.split('+').collect::<Vec<_>>();
    let Some(base) = parts.pop() else {
        return false;
    };
    if base.is_empty() || matches!(base, "mod" | "ctrl") {
        return false;
    }
    let primary = parts.contains(&"mod") || parts.contains(&"ctrl");
    if matches!(
        base,
        "up" | "down" | "left" | "right" | "home" | "end" | "pageup" | "pagedown"
    ) && !(primary && parts.contains(&"alt"))
    {
        return false;
    }
    combo.starts_with("mod+")
        || combo.starts_with("ctrl+")
        || matches!(action, Action::Models | Action::Shortcuts | Action::Palette)
}
pub fn format_combo(combo: &str) -> String {
    let mut parts = combo.split('+').collect::<Vec<_>>();
    if cfg!(target_os = "macos") && parts.starts_with(&["mod", "ctrl"]) {
        parts.swap(0, 1);
    }
    parts
        .into_iter()
        .map(|part| match part {
            "mod" => {
                if cfg!(target_os = "macos") {
                    "⌘".into()
                } else {
                    "Ctrl".into()
                }
            }
            "ctrl" => {
                if cfg!(target_os = "macos") {
                    "⌃".into()
                } else {
                    "Ctrl".into()
                }
            }
            "alt" => {
                if cfg!(target_os = "macos") {
                    "⌥".into()
                } else {
                    "Alt".into()
                }
            }
            "shift" => {
                if cfg!(target_os = "macos") {
                    "⇧".into()
                } else {
                    "Shift".into()
                }
            }
            "enter" => "↵".into(),
            "escape" => "Esc".into(),
            "backspace" => "⌫".into(),
            "delete" => "Del".into(),
            "tab" => "⇥".into(),
            "space" => "Space".into(),
            "up" => "↑".into(),
            "down" => "↓".into(),
            "left" => "←".into(),
            "right" => "→".into(),
            "pageup" => "PgUp".into(),
            "pagedown" => "PgDn".into(),
            other => {
                if other.len() == 1 || other.starts_with('f') {
                    other.to_uppercase()
                } else {
                    other.into()
                }
            }
        })
        .collect::<Vec<String>>()
        .join(if cfg!(target_os = "macos") { "" } else { "+" })
}

pub fn view(app: &Hermes) -> Element<'_, AppMessage> {
    let state = &app.shortcuts;
    let p = ui::palette(app.dark);
    let opener = state
        .combos(Action::Shortcuts)
        .first()
        .map(|combo| format_combo(combo))
        .unwrap_or_default();
    let header = row![
        column![
            text(format!(
                "Click a shortcut to rebind it · {opener} reopens this panel."
            ))
            .size(11)
            .color(p.muted)
        ]
        .spacing(5),
        Space::new().width(Fill),
        button(text("Reset all").size(12))
            .on_press(AppMessage::Shortcuts(Message::ResetAll))
            .style(button::text)
    ]
    .align_y(Alignment::Center);
    let search = text_input("Search shortcuts…", &state.query)
        .on_input(|query| AppMessage::Shortcuts(Message::Query(query)))
        .size(13)
        .padding(8);
    let query = state.query.to_lowercase();
    let searching = !query.trim().is_empty();
    let mut body = column![header, search].spacing(14);
    if let Some(error) = &state.error {
        body = body.push(
            text(error)
                .size(12)
                .color(iced::Color::from_rgb8(190, 65, 65)),
        );
    }
    for category in [
        Category::Composer,
        Category::Session,
        Category::Navigation,
        Category::View,
    ] {
        let specs = ACTIONS
            .iter()
            .filter(|spec| {
                spec.category == category
                    && spec.action != Action::Shortcuts
                    && (!searching
                        || spec.label.to_lowercase().contains(&query)
                        || spec.id.contains(&query))
            })
            .collect::<Vec<_>>();
        if specs.is_empty() {
            continue;
        }
        let expanded = searching || !state.collapsed.contains(&category);
        if !searching {
            body = body.push(
                button(
                    row![
                        text(if expanded { "⌄" } else { "›" }).size(12),
                        text(category.label().to_uppercase())
                            .size(10)
                            .color(p.muted)
                    ]
                    .spacing(7),
                )
                .on_press(AppMessage::Shortcuts(Message::ToggleCategory(category)))
                .padding([4, 0])
                .style(button::text),
            );
        }
        if !expanded {
            continue;
        }
        for spec in specs {
            let combos = state.combos(spec.action);
            let capturing = state.capture == Some(spec.action);
            let defaults = spec
                .defaults
                .iter()
                .map(|s| (*s).into())
                .collect::<Vec<String>>();
            let is_default = combos == defaults;
            let label = if capturing {
                "Press a key…".into()
            } else if combos.is_empty() {
                "set".into()
            } else {
                combos
                    .iter()
                    .map(|combo| format_combo(combo))
                    .collect::<Vec<_>>()
                    .join("  ")
            };
            let shortcut = button(text(label).size(12))
                .on_press(AppMessage::Shortcuts(Message::Begin(spec.action)))
                .padding([4, 8])
                .style(move |_, status| button::Style {
                    background: Some(if capturing { p.secondary } else { p.card }.into()),
                    text_color: if capturing { p.accent } else { p.foreground },
                    border: Border {
                        radius: 5.0.into(),
                        width: 1.0,
                        color: if capturing { p.accent } else { p.border },
                    },
                    ..button::text(&ui::theme_for(app.dark), status)
                });
            let mut line = row![text(spec.label).size(13), Space::new().width(Fill)];
            if let Some(conflict) = state.conflict(spec.action) {
                line = line.push(
                    text(format!("Also bound to “{conflict}”"))
                        .size(10)
                        .color(p.muted),
                );
            }
            line = line.push(shortcut);
            if is_default && !combos.is_empty() {
                line = line.push(
                    button(text("Clear").size(11))
                        .on_press(AppMessage::Shortcuts(Message::Clear(spec.action)))
                        .style(button::text),
                );
            } else if !is_default {
                line = line.push(
                    button(text("Reset to default").size(11))
                        .on_press(AppMessage::Shortcuts(Message::Reset(spec.action)))
                        .style(button::text),
                );
            } else {
                line = line.push(Space::new().width(40));
            }
            body =
                body.push(container(line.spacing(10).align_y(Alignment::Center)).padding([4, 8]));
        }
    }
    let readonly = [
        ("Send message", "enter"),
        ("Insert newline", "shift+enter"),
        ("Queue message", "mod+enter"),
        ("Send next queued turn", "mod+shift+k"),
        ("Close popover · cancel run", "escape"),
    ];
    let rows = readonly
        .into_iter()
        .filter(|(label, _)| !searching || label.to_lowercase().contains(&query))
        .collect::<Vec<_>>();
    if !rows.is_empty() {
        body = body.push(text("COMPOSER · FIXED").size(10).color(p.muted));
        for (label, combo) in rows {
            body = body.push(
                container(
                    row![
                        text(label).size(13).color(p.muted),
                        Space::new().width(Fill),
                        text(format_combo(combo)).size(12).color(p.muted)
                    ]
                    .spacing(10),
                )
                .padding([4, 8]),
            );
        }
    }
    container(scrollable(body.padding(Padding {
        top: 24.0,
        right: 28.0,
        bottom: 28.0,
        left: 28.0,
    })))
    .width(Fill)
    .height(Fill)
    .into()
}
