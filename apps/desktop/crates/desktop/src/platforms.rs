//! Messaging platform catalog (hermes_cli/web_server_messaging.py + every
//! plugins/platforms/*/plugin.yaml): name, description, required and optional
//! fields. Settings are kept per library; secret fields go to a 0600 file and
//! are never shown again. Telegram has its own live native adapter
//! (messaging.rs); the other adapters run in the Hermes backend.
use super::messaging::Message;
use crate::appearance::text;
use iced::{Alignment, Border, Color, Element, Fill, Padding, widget::{Space, button, column, container, row, rule, scrollable, text_input}};
use serde::{Deserialize, Serialize};
use std::{collections::{BTreeMap, BTreeSet}, path::{Path, PathBuf}, sync::LazyLock};

#[derive(Clone, Debug, Deserialize)]
pub(super) struct Field { pub name: String, pub label: String, #[serde(default)] pub description: String, #[serde(default)] pub secret: bool, #[serde(default)] pub url: String, #[serde(default)] pub required: bool }
#[derive(Clone, Debug, Deserialize)]
pub(super) struct Platform { pub id: String, pub name: String, #[serde(default)] pub description: String, #[serde(default)] pub fields: Vec<Field> }

pub(super) static CATALOG: LazyLock<Vec<Platform>> = LazyLock::new(|| {
    let mut all: Vec<Platform> = serde_json::from_str(include_str!("../assets/messaging-platforms.json")).unwrap_or_default();
    all.retain(|p| p.id != "telegram"); // Telegram is the native live adapter.
    all
});

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Saved { enabled: BTreeSet<String>, values: BTreeMap<String, BTreeMap<String, String>> }

#[derive(Clone, Debug)]
pub(super) enum Action { Select(String), Field(String, String), Save, Toggle }

#[derive(Default)]
pub(super) struct State {
    pub selected: Option<String>, saved: Saved, secrets: BTreeMap<String, BTreeSet<String>>,
    draft: BTreeMap<String, String>, dirty: bool, note: Option<Result<String, String>>, root: Option<PathBuf>,
}

fn values_path(root: &Path) -> PathBuf { root.join("messaging-platforms.json") }
fn secrets_path(root: &Path) -> PathBuf { root.join("messaging-secrets.json") }

impl State {
    pub fn load(&mut self, root: &Path) {
        self.root = Some(root.to_owned());
        self.saved = std::fs::read(values_path(root)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        let secrets: BTreeMap<String, BTreeMap<String, String>> = std::fs::read(secrets_path(root)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        self.secrets = secrets.into_iter().map(|(p, m)| (p, m.into_keys().collect())).collect();
    }
    fn platform(&self) -> Option<&'static Platform> { let id = self.selected.as_deref()?; CATALOG.iter().find(|p| p.id == id) }
    /// Every required field has a saved value (or a saved secret).
    pub fn configured(&self, platform: &Platform) -> bool {
        platform.fields.iter().filter(|f| f.required).all(|f| self.has_value(&platform.id, f))
    }
    fn has_value(&self, platform: &str, field: &Field) -> bool {
        if field.secret { self.secrets.get(platform).is_some_and(|s| s.contains(&field.name)) }
        else { self.saved.values.get(platform).and_then(|m| m.get(&field.name)).is_some_and(|v| !v.trim().is_empty()) }
    }
    pub fn enabled(&self, id: &str) -> bool { self.saved.enabled.contains(id) }
    fn persist(&mut self) -> Result<(), String> {
        let Some(root) = self.root.clone() else { return Err("The chat library is unavailable.".into()) };
        let Some(platform) = self.platform() else { return Ok(()) };
        let mut secrets: BTreeMap<String, BTreeMap<String, String>> = std::fs::read(secrets_path(&root)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        for field in &platform.fields {
            let Some(value) = self.draft.get(&field.name) else { continue };
            if field.secret {
                if !value.trim().is_empty() { secrets.entry(platform.id.clone()).or_default().insert(field.name.clone(), value.trim().to_owned()); }
            } else {
                self.saved.values.entry(platform.id.clone()).or_default().insert(field.name.clone(), value.trim().to_owned());
            }
        }
        std::fs::write(values_path(&root), serde_json::to_vec_pretty(&self.saved).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let file = secrets_path(&root);
        std::fs::write(&file, serde_json::to_vec(&secrets).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        #[cfg(unix)] { use std::os::unix::fs::PermissionsExt; let _ = std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)); }
        self.secrets = secrets.into_iter().map(|(p, m)| (p, m.into_keys().collect())).collect();
        self.draft.clear(); self.dirty = false;
        Ok(())
    }
    pub fn update(&mut self, action: Action) {
        match action {
            Action::Select(id) => { self.selected = Some(id); self.draft.clear(); self.dirty = false; self.note = None; }
            Action::Field(name, value) => { self.draft.insert(name, value); self.dirty = true; self.note = None; }
            Action::Save => self.note = Some(self.persist().map(|_| "Saved.".into())),
            Action::Toggle => {
                let Some(platform) = self.platform() else { return };
                if self.dirty && let Err(e) = self.persist() { self.note = Some(Err(e)); return; }
                if !self.configured(platform) { self.note = Some(Err("Fill in the required fields first.".into())); return; }
                let id = platform.id.clone();
                let on = !self.saved.enabled.contains(&id);
                if on { self.saved.enabled.insert(id); } else { self.saved.enabled.remove(&id); }
                self.note = Some(self.persist().map(|_| if on { "Enabled. The Hermes backend starts this channel on its next run.".into() } else { "Disabled.".into() }));
            }
        }
    }
}

fn status_label(state: &State, platform: &Platform) -> &'static str {
    if state.enabled(&platform.id) { "Enabled" } else if state.configured(platform) { "Ready" } else { "Needs setup" }
}

/// Rail rows for every catalog platform matching the search.
pub(super) fn rail_rows<'a>(state: &'a State, query: &str, p: super::ui::Palette) -> Vec<Element<'a, Message>> {
    let query = query.trim().to_lowercase();
    CATALOG.iter().filter(|pl| query.is_empty() || pl.name.to_lowercase().contains(&query) || pl.id.contains(&query)).map(|platform| {
        let selected = state.selected.as_deref() == Some(platform.id.as_str());
        let dot = if state.enabled(&platform.id) { p.accent } else { p.muted };
        let initial = platform.name.chars().next().unwrap_or('?').to_string();
        button(row![
            container(text(initial).size(11).font(super::ui::medium())).width(18).height(18).center_x(18).center_y(18)
                .style(move |_| super::ui::panel(Color { a: 0.12, ..p.accent }, 9.0, None)),
            text(platform.name.clone()).size(13).width(Fill), text("●").size(9).color(dot)].spacing(8).align_y(Alignment::Center))
            .on_press(Message::Platform(Action::Select(platform.id.clone()))).padding([7, 8]).width(Fill)
            .style(move |_, status| button::Style { background: (selected || status == button::Status::Hovered).then(|| p.card.into()), text_color: p.foreground,
                border: Border { radius: 6.0.into(), ..Default::default() }, ..Default::default() }).into()
    }).collect()
}

/// The setup form for the selected platform, as the original's platform card.
pub(super) fn detail<'a>(state: &'a State, dark: bool) -> Element<'a, Message> {
    let p = super::ui::palette(dark);
    let Some(platform) = state.platform() else { return Space::new().into() };
    let chip = |label: &'static str| container(text(label).size(11).color(p.muted)).padding([3, 7]).style(move |_| super::ui::panel(p.card, 6.0, None));
    let heading = |label: &'static str| text(label).size(11).font(super::ui::medium()).color(p.muted);
    let mut body = column![
        row![text(platform.name.clone()).size(15).font(super::ui::medium()), chip(status_label(state, platform))].spacing(10).align_y(Alignment::Center),
        text(platform.description.clone()).size(12).color(p.muted),
    ].spacing(14);
    if let Some(guide) = platform.fields.iter().find(|f| !f.url.is_empty()) {
        body = body.push(heading("GET YOUR CREDENTIALS")).push(button(row![text("Open setup guide").size(12), super::ui::icon("link-external", 11, p.muted)].spacing(6))
            .padding(0).style(button::text).on_press(Message::OpenLink(guide.url.clone())));
    }
    for (title, required) in [("REQUIRED", true), ("OPTIONAL", false)] {
        let fields: Vec<&Field> = platform.fields.iter().filter(|f| f.required == required).collect();
        if fields.is_empty() { continue; }
        body = body.push(heading(title));
        for field in fields {
            let saved = state.saved.values.get(&platform.id).and_then(|m| m.get(&field.name)).cloned().unwrap_or_default();
            let value = state.draft.get(&field.name).cloned().unwrap_or(if field.secret { String::new() } else { saved });
            let placeholder = if field.secret && state.has_value(&platform.id, field) { "Saved · paste to replace".to_owned() } else { field.label.clone() };
            let name = field.name.clone();
            let mut input = text_input(&placeholder, &value).size(13).padding([7, 10]).on_input(move |v| Message::Platform(Action::Field(name.clone(), v)));
            if field.secret { input = input.secure(true); }
            let mut block = column![text(field.label.clone()).size(13)].spacing(5);
            if !field.description.is_empty() && field.description != field.label { block = block.push(text(field.description.clone()).size(12).color(p.muted)); }
            body = body.push(block.push(input));
        }
    }
    body = body.push(text("This platform's adapter runs in the Hermes backend; settings saved here are what it reads. Telegram also has a live native adapter in this build.").size(12).color(p.muted));
    if let Some(note) = &state.note {
        body = body.push(text(match note { Ok(s) | Err(s) => s.as_str() }).size(12).color(if note.is_ok() { p.muted } else { Color::from_rgb8(232, 98, 98) }));
    }
    let enabled = state.enabled(&platform.id);
    let actions = row![text(status_label(state, platform)).size(12).color(p.muted), Space::new().width(Fill),
        button(text("Save changes").size(12)).padding([7, 12]).on_press_maybe(state.dirty.then_some(Message::Platform(Action::Save))),
        button(text(if enabled { "Disable" } else { "Enable" }).size(12)).padding([7, 12]).on_press(Message::Platform(Action::Toggle))].spacing(10).align_y(Alignment::Center);
    column![
        scrollable(container(body).padding(Padding { top: 16.0, right: 24.0, bottom: 24.0, left: 24.0 }).max_width(720).width(Fill)).height(Fill),
        rule::horizontal(1), container(actions).padding([12, 20]),
    ].height(Fill).width(Fill).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn catalog_lists_the_hermes_platforms_and_saves_privately() {
        assert!(CATALOG.len() >= 25 && CATALOG.iter().any(|p| p.id == "discord") && CATALOG.iter().any(|p| p.id == "slack"));
        let dir = tempfile::tempdir().unwrap();
        let mut state = State::default(); state.load(dir.path());
        state.update(Action::Select("discord".into()));
        state.update(Action::Toggle);
        assert!(!state.enabled("discord"), "cannot enable without the required token");
        state.update(Action::Field("DISCORD_BOT_TOKEN".into(), "secret-token".into()));
        state.update(Action::Toggle);
        assert!(state.enabled("discord"));
        assert!(!std::fs::read_to_string(dir.path().join("messaging-platforms.json")).unwrap().contains("secret-token"));
        let mut reopened = State::default(); reopened.load(dir.path());
        assert!(reopened.enabled("discord") && reopened.configured(CATALOG.iter().find(|p| p.id == "discord").unwrap()));
    }
}
