//! Functional native Appearance settings, derived from stock 0284ce17
//! appearance-settings.tsx, chat-font.ts and chat-text-scale.ts. The palette
//! asset is the stock shared theme table, exported without hand-edited colors.
use super::{Hermes, Message, ui};
use iced::{
    Alignment, Border, Color, Element, Fill, Font, Task,
    widget::{
        self, Space, button, column, combo_box, container, responsive, row, rule, scrollable,
        text_input,
    },
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt,
    path::Path,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

pub(super) const MONO_FONT_BYTES: &[u8] =
    include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf");
const TEXT_SCALES: [u16; 6] = [90, 100, 110, 125, 150, 175];
const UI_SCALES: [u16; 6] = [90, 100, 110, 125, 150, 175];
const FONT_SUGGESTIONS: [&str; 9] = [
    "OpenDyslexic",
    "Atkinson Hyperlegible",
    "Lexend",
    "Inter",
    "IBM Plex Sans",
    "Source Sans 3",
    "Noto Sans",
    "Segoe UI",
    "SF Pro Text",
];
static ACTIVE_THEME: AtomicUsize = AtomicUsize::new(0);
static ACTIVE_SCALE: AtomicUsize = AtomicUsize::new(110);
static APP_FONT: OnceLock<Mutex<Font>> = OnceLock::new();
/// The redesign's UI face (--dt-font-sans): Mona Sans, bundled as static weights.
pub(super) const UI_FONT: Font = Font::with_name("Mona Sans");
/// The redesign's display face (--portal-font-display): Oswald Light, for page titles.
pub(super) const DISPLAY_FONT: Font = Font { family: iced::font::Family::Name("Oswald"), weight: iced::font::Weight::Light, ..Font::DEFAULT };
pub(super) const UI_FONT_FILES: [&[u8]; 4] = [include_bytes!("../assets/fonts/MonaSans-Regular.ttf"), include_bytes!("../assets/fonts/MonaSans-Medium.ttf"),
    include_bytes!("../assets/fonts/MonaSans-SemiBold.ttf"), include_bytes!("../assets/fonts/Oswald-Light.ttf")];
static TERMINAL_FONT: OnceLock<Mutex<Font>> = OnceLock::new();

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Mode {
    #[default]
    System,
    Light,
    Dark,
}
impl Mode {
    pub fn dark(self, system: bool) -> bool {
        match self {
            Self::System => system,
            Self::Light => false,
            Self::Dark => true,
        }
    }
}
impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::System => "System",
            Self::Light => "Light",
            Self::Dark => "Dark",
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Skin {
    #[default]
    Nous,
    Github,
    Catppuccin,
    Everforest,
    Solarized,
    NousAlt,
    Midnight,
    Ember,
    Mono,
    Cyberpunk,
    Slate,
}
impl Skin {
    pub const ALL: [Self; 11] = [
        Self::Nous,
        Self::Github,
        Self::Catppuccin,
        Self::Everforest,
        Self::Solarized,
        Self::NousAlt,
        Self::Midnight,
        Self::Ember,
        Self::Mono,
        Self::Cyberpunk,
        Self::Slate,
    ];
    fn id(self) -> &'static str {
        match self {
            Self::Nous => "nous",
            Self::Github => "github",
            Self::Catppuccin => "catppuccin",
            Self::Everforest => "everforest",
            Self::Solarized => "solarized",
            Self::NousAlt => "nous-alt",
            Self::Midnight => "midnight",
            Self::Ember => "ember",
            Self::Mono => "mono",
            Self::Cyberpunk => "cyberpunk",
            Self::Slate => "slate",
        }
    }
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Nous => "Nous",
            Self::Github => "GitHub",
            Self::Catppuccin => "Catppuccin",
            Self::Everforest => "Everforest",
            Self::Solarized => "Solarized",
            Self::NousAlt => "Nous Alt",
            Self::Midnight => "Midnight",
            Self::Ember => "Ember",
            Self::Mono => "Mono",
            Self::Cyberpunk => "Cyberpunk",
            Self::Slate => "Slate",
        }
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub(super) struct Preferences {
    pub mode: Mode,
    pub skin: Skin,
    pub font_family: String,
    pub terminal_font_family: String,
    pub chat_scale: u16,
    pub ui_scale: u16,
    pub reopen_last_chat: bool,
}
impl Default for Preferences {
    fn default() -> Self {
        Self {
            mode: Mode::System,
            skin: Skin::Nous,
            font_family: String::new(),
            terminal_font_family: String::new(),
            chat_scale: 110,
            ui_scale: 100,
            reopen_last_chat: true,
        }
    }
}
impl Preferences {
    pub fn dark(&self, system: bool) -> bool {
        let background = palette_for(self.skin, self.mode.dark(system)).background;
        background.r * 0.2126 + background.g * 0.7152 + background.b * 0.0722 < 0.4
    }
}

#[derive(Clone)]
pub(super) enum Action {
    Mode(Mode),
    Skin(Skin),
    Search(String),
    FontFamily(String),
    TerminalFont(String),
    ResetFont,
    ResetTerminalFont,
    ChatScale(u16),
    UiScale(u16),
    ReopenLastChat(bool),
    Persist(u64),
}
impl fmt::Debug for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Appearance action")
    }
}

pub(super) struct State {
    pub preferences: Preferences,
    saved: Preferences,
    pub query: String,
    font_options: combo_box::State<String>,
    terminal_options: combo_box::State<String>,
    revision: u64,
}
impl Default for State {
    fn default() -> Self {
        let preferences = Preferences::default();
        Self {
            saved: preferences.clone(),
            preferences,
            query: String::new(),
            font_options: combo_box::State::new(
                FONT_SUGGESTIONS.into_iter().map(str::to_owned).collect(),
            ),
            terminal_options: combo_box::State::new(
                [
                    "JetBrains Mono",
                    "Menlo",
                    "Monaco",
                    "SF Mono",
                    "Cascadia Code",
                    "Consolas",
                    "MesloLGS NF",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            ),
            revision: 0,
        }
    }
}
impl State {
    pub fn load(root: &Path) -> Self {
        let mut state = Self::default();
        if let Ok(bytes) = std::fs::read(root.join("appearance-preferences.json"))
            && let Ok(mut prefs) = serde_json::from_slice::<Preferences>(&bytes)
        {
            if !TEXT_SCALES.contains(&prefs.chat_scale) {
                prefs.chat_scale = 110;
            }
            if !UI_SCALES.contains(&prefs.ui_scale) {
                prefs.ui_scale = 100;
            }
            limit_family(&mut prefs.font_family);
            limit_family(&mut prefs.terminal_font_family);
            state.saved = prefs.clone();
            state.preferences = prefs;
        }
        state.activate();
        state
    }
    fn save(&mut self, root: &Path) -> Result<(), String> {
        let temp = root.join("appearance-preferences.tmp");
        std::fs::write(
            &temp,
            serde_json::to_vec_pretty(&self.preferences).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        std::fs::rename(temp, root.join("appearance-preferences.json"))
            .map_err(|e| e.to_string())?;
        self.saved = self.preferences.clone();
        Ok(())
    }
    pub fn flush(&mut self, root: &Path) -> Result<(), String> {
        if self.preferences != self.saved {
            self.save(root)
        } else {
            Ok(())
        }
    }
    pub fn activate(&self) {
        ACTIVE_THEME.store(
            Skin::ALL
                .iter()
                .position(|skin| *skin == self.preferences.skin)
                .unwrap_or(0),
            Ordering::Relaxed,
        );
        ACTIVE_SCALE.store(self.preferences.chat_scale.into(), Ordering::Relaxed);
        if let Ok(mut value) = APP_FONT.get_or_init(|| Mutex::new(UI_FONT)).lock() {
            *value = family(&self.preferences.font_family, UI_FONT);
        }
        if let Ok(mut value) = TERMINAL_FONT
            .get_or_init(|| Mutex::new(Font::with_name("JetBrains Mono")))
            .lock()
        {
            *value = family(
                &self.preferences.terminal_font_family,
                Font::with_name("JetBrains Mono"),
            );
        }
    }
    pub fn update(&mut self, action: Action, root: &Path) -> Result<Task<Action>, String> {
        let delayed = match action {
            Action::Search(query) => {
                self.query = query;
                return Ok(Task::none());
            }
            Action::Mode(mode) => {
                self.preferences.mode = mode;
                false
            }
            Action::Skin(skin) => {
                self.preferences.skin = skin;
                false
            }
            Action::FontFamily(value) => {
                if value.len() > 256 {
                    return Ok(Task::none());
                }
                self.preferences.font_family = value;
                true
            }
            Action::TerminalFont(value) => {
                if value.len() > 256 {
                    return Ok(Task::none());
                }
                self.preferences.terminal_font_family = value;
                true
            }
            Action::ResetFont => {
                self.preferences.font_family.clear();
                false
            }
            Action::ResetTerminalFont => {
                self.preferences.terminal_font_family.clear();
                false
            }
            Action::ChatScale(value) => {
                if TEXT_SCALES.contains(&value) {
                    self.preferences.chat_scale = value;
                }
                false
            }
            Action::UiScale(value) => {
                if UI_SCALES.contains(&value) {
                    self.preferences.ui_scale = value;
                }
                false
            }
            Action::ReopenLastChat(value) => {
                self.preferences.reopen_last_chat = value;
                false
            }
            Action::Persist(revision) => {
                if revision != self.revision {
                    return Ok(Task::none());
                }
                false
            }
        };
        self.activate();
        self.revision = self.revision.wrapping_add(1);
        if delayed {
            let revision = self.revision;
            return Ok(Task::perform(
                async move {
                    tokio::time::sleep(std::time::Duration::from_millis(550)).await;
                    revision
                },
                Action::Persist,
            ));
        }
        if let Err(error) = self.save(root) {
            self.preferences = self.saved.clone();
            self.activate();
            return Err(error);
        }
        Ok(Task::none())
    }
}

pub(super) fn font() -> Font {
    APP_FONT
        .get_or_init(|| Mutex::new(UI_FONT))
        .lock()
        .map(|font| *font)
        .unwrap_or(UI_FONT)
}
pub(super) fn terminal_font() -> Font {
    TERMINAL_FONT
        .get_or_init(|| Mutex::new(Font::with_name("JetBrains Mono")))
        .lock()
        .map(|font| *font)
        .unwrap_or(Font::MONOSPACE)
}
pub(super) fn chat_size(base: f32) -> f32 {
    base * ACTIVE_SCALE.load(Ordering::Relaxed) as f32 / 100.0
}
pub(super) fn text<'a>(value: impl widget::text::IntoFragment<'a>) -> widget::Text<'a> {
    widget::text(value).font(font())
}
fn family(value: &str, fallback: Font) -> Font {
    let name = value
        .split(',')
        .next()
        .unwrap_or_default()
        .trim()
        .trim_matches(['\'', '"']);
    if name.is_empty() {
        return fallback;
    }
    match name.to_ascii_lowercase().as_str() {
        "sans-serif" | "system-ui" | "mona sans" => return UI_FONT,
        "serif" => {
            return Font {
                family: iced::font::Family::Serif,
                ..Font::DEFAULT
            };
        }
        "monospace" => return Font::MONOSPACE,
        _ => {}
    }
    static NAMES: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let Ok(mut names) = NAMES.get_or_init(|| Mutex::new(HashMap::new())).lock() else {
        return fallback;
    };
    if let Some(name) = names.get(name) {
        return Font::with_name(*name);
    }
    if names.len() >= 512 {
        return fallback;
    }
    let permanent: &'static str = Box::leak(name.to_owned().into_boxed_str());
    names.insert(name.to_owned(), permanent);
    Font::with_name(permanent)
}
fn limit_family(value: &mut String) {
    let mut end = value.len().min(256);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value.truncate(end);
}

pub(super) fn palette(dark: bool) -> ui::Palette {
    let skin = palette_for(
        Skin::ALL[ACTIVE_THEME
            .load(Ordering::Relaxed)
            .min(Skin::ALL.len() - 1)],
        dark,
    );
    if dark { portal_dark(skin) } else { portal_light(skin) }
}
/// Light portal roles (color.css :not(.dark)): white stage and sidebar over the
/// #f2f2f2 canvas, brand blue ink, the 80/72/64 text ladder.
fn portal_light(skin: ui::Palette) -> ui::Palette {
    
    let ink = |a: f32| Color { a, ..skin.foreground };
    ui::Palette {
        background: Color::from_rgb8(250, 250, 250), sidebar: Color::from_rgb8(254, 254, 254), card: Color::WHITE, accent: super::shell_art::accent().ink(false),
        foreground: skin.foreground, muted: ink(0.72), secondary: ink(0.05), border: ink(0.10),
    }
}
/// Dark mode is the redesign's portal shell: the navy roles over the artwork,
/// with the skin supplying only its foreground (--ui-base).
fn portal_dark(skin: ui::Palette) -> ui::Palette {
    use super::shell_art::roles;
    let ink = |a: f32| Color { a, ..skin.foreground };
    ui::Palette {
        background: roles::STAGE, sidebar: roles::SIDEBAR, card: roles::COMPOSER, accent: super::shell_art::accent().ink(true),
        foreground: skin.foreground, muted: ink(0.6), secondary: ink(0.05), border: ink(0.08),
    }
}
pub(super) fn palette_for(skin: Skin, dark: bool) -> ui::Palette {
    static TABLE: OnceLock<serde_json::Value> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        serde_json::from_str(include_str!("../assets/theme-presets.json"))
            .expect("stock theme table")
    });
    let entry = &table[skin.id()];
    let values = if dark && entry.get("darkColors").is_some() {
        &entry["darkColors"]
    } else {
        &entry["colors"]
    };
    let get =
        |name: &str, fallback: Color| values[name].as_str().and_then(color).unwrap_or(fallback);
    let background = get("background", if dark { Color::BLACK } else { Color::WHITE });
    let foreground = get("foreground", if dark { Color::WHITE } else { Color::BLACK });
    ui::Palette {
        background,
        foreground,
        sidebar: get("sidebarBackground", get("card", background)),
        secondary: get("secondary", background),
        muted: get("mutedForeground", foreground),
        accent: get("midground", get("primary", foreground)),
        border: get("sidebarBorder", get("border", foreground)),
        card: get("card", background),
    }
}
fn color(value: &str) -> Option<Color> {
    let value = value.trim();
    if value == "transparent" {
        return Some(Color::TRANSPARENT);
    }
    if let Some(hex) = value.strip_prefix('#') {
        let n = u32::from_str_radix(hex, 16).ok()?;
        return match hex.len() {
            6 => Some(Color::from_rgb8((n >> 16) as u8, (n >> 8) as u8, n as u8)),
            3 => Some(Color::from_rgb8(
                ((n >> 8) & 15) as u8 * 17,
                ((n >> 4) & 15) as u8 * 17,
                (n & 15) as u8 * 17,
            )),
            _ => None,
        };
    }
    let inner = value
        .strip_prefix("color-mix(in srgb,")?
        .strip_suffix(')')?;
    let mut depth = 0usize;
    let split = inner.char_indices().find_map(|(i, c)| {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => return Some(i),
            _ => {}
        }
        None
    })?;
    let part = |part: &str| -> Option<(Color, Option<f32>)> {
        let part = part.trim();
        if let Some((c, p)) = part.rsplit_once(' ')
            && let Some(p) = p.strip_suffix('%')
        {
            Some((color(c)?, Some(p.parse::<f32>().ok()? / 100.0)))
        } else {
            Some((color(part)?, None))
        }
    };
    let (a, wa) = part(&inner[..split])?;
    let (b, wb) = part(&inner[split + 1..])?;
    let wa = wa.unwrap_or_else(|| 1.0 - wb.unwrap_or(0.5));
    let wb = wb.unwrap_or(1.0 - wa);
    let sum = wa + wb;
    if sum <= 0.0 {
        return None;
    }
    let (wa, wb) = (wa / sum, wb / sum);
    let alpha = a.a * wa + b.a * wb;
    if alpha <= 0.0 {
        return Some(Color::TRANSPARENT);
    }
    Some(Color {
        r: (a.r * a.a * wa + b.r * b.a * wb) / alpha,
        g: (a.g * a.a * wa + b.g * b.a * wb) / alpha,
        b: (a.b * a.a * wa + b.b * b.a * wb) / alpha,
        a: alpha,
    })
}

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let prefs = &app.appearance.preferences;
    let modes = row![
        mode_button("Light", Mode::Light, prefs.mode),
        mode_button("Dark", Mode::Dark, prefs.mode),
        mode_button("System", Mode::System, prefs.mode)
    ]
    .spacing(2);
    let query = app.appearance.query.trim().to_lowercase();
    let themes: Vec<_> = Skin::ALL
        .into_iter()
        .filter(|skin| query.is_empty() || skin.label().to_lowercase().contains(&query))
        .collect();
    let gallery = responsive(move |size| {
        let per_row = if size.width > 900.0 { 3 } else { 2 };
        let mut grid = column![].spacing(12);
        for group in themes.chunks(per_row) {
            let mut items = row![].spacing(12);
            for skin in group {
                items = items.push(theme_card(*skin, prefs.skin, app.dark));
            }
            if group.len() < per_row {
                items = items.push(Space::new().width(Fill));
            }
            grid = grid.push(items);
        }
        grid.into()
    });
    let make_it_yours = make_it_yours(app);
    let mut content=column![
        make_it_yours,
        rule::horizontal(1),
        text("Appearance").size(14).font(Font{weight:iced::font::Weight::Semibold,..font()}),
        text("Desktop-only. Mode is brightness; theme is palette and chat chrome.").size(12).color(p.muted),
        row![column![text("Theme").size(13),text("Desktop palettes only. The selected mode is applied on top.").size(11).color(p.muted)].spacing(3),Space::new().width(Fill),modes].spacing(16).align_y(Alignment::Center),
        text_input("Search your themes…",&app.appearance.query).on_input(|value|Message::Appearance(Action::Search(value))).padding(8).size(12),gallery,
        rule::horizontal(1),scale_row("UI Scale",&format!("Scales text and controls across the whole app. Current: {}%.",prefs.ui_scale),&UI_SCALES,prefs.ui_scale,false,app.dark),
        scale_row("Chat Text Size","Scales conversation text and the message editor relative to UI Scale. Sidebars and controls stay the same size.",&TEXT_SCALES,prefs.chat_scale,true,app.dark),
        rule::horizontal(1),font_row(app,false),font_row(app,true),rule::horizontal(1),
        row![column![text("Chat Backdrop").size(13),text("The faint statue image behind the conversation.").size(11).color(p.muted)].spacing(3),Space::new().width(Fill),widget::toggler(app.backdrop).on_toggle(|_|Message::ToggleBackdrop).size(18)].spacing(16).align_y(Alignment::Center),
        Space::new().height(8),row![column![text("Reopen Last Chat on Launch").size(13),text("When enabled, the app reopens your most recent chat on cold start. Turn off to always start with a fresh new chat.").size(11).color(p.muted)].spacing(3),Space::new().width(Fill),widget::toggler(prefs.reopen_last_chat).on_toggle(|value|Message::Appearance(Action::ReopenLastChat(value))).size(18)].spacing(16).align_y(Alignment::Center),
    ].spacing(18);
    if !query.is_empty()
        && Skin::ALL
            .into_iter()
            .all(|skin| !skin.label().to_lowercase().contains(&query))
    {
        content = content.push(
            text(format!(
                "No installed themes match \"{}\".",
                app.appearance.query.trim()
            ))
            .size(12)
            .color(p.muted),
        );
    }
    scrollable(
        container(content)
            .padding(iced::Padding {
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
fn mode_button(label: &'static str, mode: Mode, selected: Mode) -> Element<'static, Message> {
    button(text(label).size(12))
        .on_press(Message::Appearance(Action::Mode(mode)))
        .padding([6, 10])
        .style(move |theme, status| {
            let mut s = button::text(theme, status);
            if mode == selected {
                s.background = Some(
                    Color {
                        a: 0.1,
                        ..theme.palette().primary
                    }
                    .into(),
                );
                s.text_color = theme.palette().text;
            }
            s.border.radius = 4.0.into();
            s
        })
        .into()
}
fn theme_card(skin: Skin, selected: Skin, dark: bool) -> Element<'static, Message> {
    let p = palette_for(skin, dark);
    button(
        column![
            container(row![
                container(Space::new().height(45))
                    .width(28)
                    .style(move |_| container::Style {
                        background: Some(p.sidebar.into()),
                        ..Default::default()
                    }),
                container(
                    column![
                        container(Space::new().height(4)).width(52).style(move |_| {
                            container::Style {
                                background: Some(p.accent.into()),
                                ..Default::default()
                            }
                        }),
                        container(Space::new().height(3)).width(80).style(move |_| {
                            container::Style {
                                background: Some(
                                    Color {
                                        a: 0.2,
                                        ..p.foreground
                                    }
                                    .into(),
                                ),
                                ..Default::default()
                            }
                        })
                    ]
                    .spacing(6)
                )
                .padding(9)
                .width(Fill)
            ])
            .width(Fill)
            .style(move |_| container::Style {
                background: Some(p.background.into()),
                ..Default::default()
            }),
            text(skin.label()).size(12)
        ]
        .spacing(8),
    )
    .on_press(Message::Appearance(Action::Skin(skin)))
    .padding(10)
    .width(Fill)
    .style(move |_, _| button::Style {
        background: Some(p.background.into()),
        text_color: p.foreground,
        border: Border {
            color: if skin == selected { p.accent } else { p.border },
            width: if skin == selected { 2.0 } else { 1.0 },
            radius: 7.0.into(),
        },
        ..Default::default()
    })
    .into()
}
fn scale_row<'a>(
    title: &'a str,
    description: &str,
    values: &[u16],
    selected: u16,
    chat: bool,
    dark: bool,
) -> Element<'a, Message> {
    let mut options = row![].spacing(2);
    for value in values {
        let v = *value;
        options = options.push(
            button(text(format!("{v}%")).size(11))
                .on_press(Message::Appearance(if chat {
                    Action::ChatScale(v)
                } else {
                    Action::UiScale(v)
                }))
                .padding([6, 7])
                .style(move |theme, status| {
                    let mut s = button::text(theme, status);
                    if v == selected {
                        s.background = Some(
                            Color {
                                a: 0.1,
                                ..theme.palette().primary
                            }
                            .into(),
                        );
                    }
                    s
                }),
        );
    }
    row![
        column![
            text(title).size(13),
            text(description.to_owned())
                .size(11)
                .color(ui::palette(dark).muted)
        ]
        .spacing(3),
        Space::new().width(Fill),
        options
    ]
    .spacing(16)
    .align_y(Alignment::Center)
    .into()
}
fn font_row(app: &Hermes, terminal: bool) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let prefs = &app.appearance.preferences;
    let (value, options, title, description, placeholder, reset) = if terminal {
        (
            &prefs.terminal_font_family,
            &app.appearance.terminal_options,
            "Terminal Font",
            "Choose an installed font for Desktop terminals. Nerd Fonts render Powerlevel10k and shell icons; leave blank to use bundled JetBrains Mono.",
            "MesloLGS NF or a CSS font stack",
            "Use default",
        )
    } else {
        (
            &prefs.font_family,
            &app.appearance.font_options,
            "Chat Font",
            "Choose an installed font for chat and the rest of the app. Handy for readability faces such as OpenDyslexic; leave blank to use the theme's font.",
            "OpenDyslexic or a CSS font stack",
            "Use theme font",
        )
    };
    let selected = (!value.is_empty()).then_some(value);
    let choose = move |value| {
        Message::Appearance(if terminal {
            Action::TerminalFont(value)
        } else {
            Action::FontFamily(value)
        })
    };
    let mut reset_button = button(text(reset).size(12))
        .padding([6, 8])
        .style(button::text);
    if !value.is_empty() {
        reset_button = reset_button.on_press(Message::Appearance(if terminal {
            Action::ResetTerminalFont
        } else {
            Action::ResetFont
        }));
    }
    column![
        text(title).size(13),
        text(description).size(11).color(p.muted),
        row![
            widget::combo_box(options, placeholder, selected, choose)
                .on_input(choose)
                .size(13)
                .padding(8)
                .width(Fill),
            reset_button
        ]
        .spacing(12)
        .align_y(Alignment::Center),
        row![
            text("Preview").size(11).color(p.muted),
            text("The quick brown fox jumps over the lazy dog. 0123456789")
                .size(14)
                .font(if terminal { terminal_font() } else { font() })
        ]
        .spacing(8)
    ]
    .spacing(9)
    .into()
}

/// "Make it yours" (discovery/make-it-yours.tsx): accent, light or dark, artwork.
/// Shared by Settings › Appearance and the first-run splash.
pub(super) fn make_it_yours(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let prefs = &app.appearance.preferences;
    // "Make it yours" (discovery/make-it-yours.tsx): accent, light or dark, artwork.
    let accent = super::shell_art::accent();
    let mut swatches = row![].spacing(10);
    for choice in super::shell_art::ACCENTS {
        let selected = choice == accent; let fill = choice.fill(); let ring = p.foreground;
        swatches = swatches.push(widget::tooltip(button(Space::new().width(22).height(22)).padding(0).on_press(Message::PortalAccent(choice))
            .style(move |_, _| button::Style { background: Some(fill.into()), border: iced::Border { color: if selected { ring } else { Color::TRANSPARENT }, width: 2.0, radius: 12.0.into() }, ..Default::default() }),
            text(choice.label()).size(11), widget::tooltip::Position::Top));
    }
    let painted = app.artwork.map(|a| super::shell_art::for_mode(a, app.dark));
    let mut tiles = column![].spacing(10);
    for group in super::shell_art::artworks_for(app.dark).chunks(5) {
        let mut line = row![].spacing(10);
        for artwork in group {
            let selected = painted == Some(*artwork); let accent_ink = p.accent;
            let mut tile = column![super::shell_art::thumbnail::<Message>(*artwork, app.dark),
                row![text(artwork.id()[..1].to_uppercase() + &artwork.id()[1..]).size(12), Space::new().width(Fill),
                    text(if selected { "Selected" } else { "" }).size(11).color(accent_ink)].padding([6, 8])].spacing(0);
            tile = tile.width(Fill);
            line = line.push(button(tile).padding(0).width(Fill).on_press(Message::PortalArtwork(*artwork)).style(move |_, status| button::Style {
                background: Some(Color { a: 0.03, ..p.foreground }.into()),
                border: iced::Border { color: if selected { accent_ink } else if status == button::Status::Hovered { Color { a: 0.2, ..p.foreground } } else { Color { a: 0.06, ..p.foreground } }, width: if selected { 2.0 } else { 1.0 }, radius: 4.0.into() },
                ..Default::default() }));
        }
        for _ in group.len()..5 { line = line.push(Space::new().width(Fill)); }
        tiles = tiles.push(line);
    }
    column![
        super::settings::section_title("Make it yours", app.dark),
        super::settings::card(vec![
            super::settings::card_row("Color", "The accent for buttons, switches, links and selection.", swatches.into(), app.dark),
            super::settings::card_row("Light or dark", "", row![mode_button("Light", Mode::Light, prefs.mode), mode_button("Dark", Mode::Dark, prefs.mode), mode_button("System", Mode::System, prefs.mode)].spacing(2).into(), app.dark),
        ], app.dark),
        Space::new().height(6),
        super::settings::section_title("Make it beautiful", app.dark),
        tiles,
    ].spacing(10).into()
}
