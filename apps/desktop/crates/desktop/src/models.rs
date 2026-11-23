//! Native model/account flows from stock0284ce17 model-picker and model settings.
use crate::appearance::text;
use super::{Config, Hermes, Message, Preferences, Screen, Session, settings, ui};
use hermes_engine::{
    credentials::{EndpointProfile, EndpointStore, scope},
    models::{GenerationOptions, Model},
    portal,
};
use iced::{
    Alignment, Border, Element, Fill, Font, Task,
    widget::{Space, button, column, container, row, scrollable, text_input},
};
use serde::{Deserialize, Serialize};
use std::{io::Write, path::PathBuf};

#[derive(Clone)]
pub(super) enum Action {
    Open,
    Close,
    Query(String),
    Refresh,
    Loaded(u64, String, Result<Vec<Model>, String>),
    Select(String),
    Preset(Preset),
    Name(String),
    SaveEndpoint,
    UseEndpoint(String),
    RemoveEndpoint(String),
    ApplyProvider(bool),
    Disconnect,
    Disconnected(Result<(), String>),
    Temperature(String),
    MaxOutput(String),
    Reasoning(String),
    SaveOptions,
    ResetOptions,
}
impl std::fmt::Debug for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProviderAction")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Preset {
    Nous,
    OpenRouter,
    OpenAi,
    Anthropic,
    Local,
}
impl Preset {
    pub fn name(self) -> &'static str {
        match self {
            Self::Nous => "Nous Portal",
            Self::OpenRouter => "OpenRouter",
            Self::OpenAi => "OpenAI",
            Self::Anthropic => "Anthropic",
            Self::Local => "Local / custom endpoint",
        }
    }
    fn base(self) -> &'static str {
        match self {
            Self::Nous => portal::INFERENCE,
            Self::OpenRouter => hermes_engine::OPENROUTER,
            Self::OpenAi => hermes_engine::OPENAI,
            Self::Anthropic => hermes_engine::ANTHROPIC,
            Self::Local => "http://localhost:11434/v1",
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
struct SavedOptions {
    #[serde(default)]
    entries: Vec<OptionEntry>,
}
#[derive(Serialize, Deserialize)]
struct OptionEntry {
    scope: String,
    model: String,
    options: GenerationOptions,
}

pub(super) struct State {
    pub picker: bool,
    pub query: String,
    pub catalog: Vec<Model>,
    pub loading: bool,
    pub error: Option<String>,
    pub endpoints: Vec<EndpointProfile>,
    pub endpoint_name: String,
    pub selected_endpoint: Option<String>,
    pub temperature: String,
    pub max_output: String,
    pub reasoning: String,
    pub options: GenerationOptions,
    pub disconnecting: bool,
    generation: u64,
    catalog_scope: Option<String>,
    vault: Option<EndpointStore>,
    dir: Option<PathBuf>,
    saved: SavedOptions,
    options_readable: bool,
}
impl Default for State {
    fn default() -> Self {
        Self {
            picker: false,
            query: String::new(),
            catalog: Vec::new(),
            loading: false,
            error: None,
            endpoints: Vec::new(),
            endpoint_name: String::new(),
            selected_endpoint: None,
            temperature: String::new(),
            max_output: String::new(),
            reasoning: String::new(),
            options: GenerationOptions::default(),
            disconnecting: false,
            generation: 0,
            catalog_scope: None,
            vault: None,
            dir: None,
            saved: SavedOptions::default(),
            options_readable: true,
        }
    }
}
impl State {
    pub fn open(config: &mut Config) -> Self {
        let mut state = Self::default();
        let result = (|| -> anyhow::Result<()> {
            let dir = Session::default_dir()?.join("providers");
            let vault = EndpointStore::open(&dir)?;
            state.endpoints = vault.list()?;
            if !portal::is_portal(&config.base_url) && config.api_key.is_empty() {
                if let Some(key) = vault.credential(&config.base_url)? {
                    config.api_key = key;
                }
            }
            if let Some(profile) = state
                .endpoints
                .iter()
                .find(|entry| scope(&entry.base_url).ok() == scope(&config.base_url).ok())
            {
                state.selected_endpoint = Some(profile.id.clone());
                state.endpoint_name = profile.name.clone();
            }
            let options = dir.join("model-options.json");
            match std::fs::read(&options) {
                Ok(bytes) => {
                    anyhow::ensure!(
                        bytes.len() <= 1_048_576,
                        "Saved model options are too large"
                    );
                    state.saved = serde_json::from_slice(&bytes)?;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            state.dir = Some(dir);
            state.vault = Some(vault);
            Ok(())
        })();
        if let Err(error) = result {
            state.error = Some(format!("Could not load provider settings: {error}"));
            state.options_readable = false;
        }
        state.load_options(config);
        state
    }
    pub fn current_model<'a>(&'a self, config: &Config) -> Option<&'a Model> {
        self.catalog.iter().find(|model| model.id == config.model)
    }
    pub fn model_label<'a>(&'a self, config: &'a Config) -> &'a str {
        self.current_model(config)
            .map(|model| model.name.as_str())
            .unwrap_or(if config.model == portal::AUTO_MODEL {
                "Auto"
            } else {
                config.model.rsplit('/').next().unwrap_or(&config.model)
            })
    }
    pub fn reasoning_label(&self) -> &str {
        super::reasoning::short(self.options.reasoning_effort.as_deref())
    }
    pub fn invalidate(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.loading = false;
        self.catalog.clear();
        self.catalog_scope = None;
        self.error = None;
    }
    pub fn load_options(&mut self, config: &Config) {
        self.options = scope(&config.base_url)
            .ok()
            .and_then(|wanted| {
                self.saved
                    .entries
                    .iter()
                    .find(|entry| entry.scope == wanted && entry.model == config.model)
                    .map(|entry| entry.options.clone())
            })
            .unwrap_or_default();
        self.temperature = self
            .options
            .temperature
            .map(|value| value.to_string())
            .unwrap_or_default();
        self.max_output = self
            .options
            .max_output_tokens
            .map(|value| value.to_string())
            .unwrap_or_default();
        self.reasoning = self.options.reasoning_effort.clone().unwrap_or_default();
    }
    pub(super) fn save_options(&mut self, config: &Config) -> Result<(), String> {
        if !self.options_readable {
            return Err(
                "Saved model options could not be read; existing values were left unchanged."
                    .into(),
            );
        }
        let wanted = scope(&config.base_url).map_err(|e| e.to_string())?;
        self.options.validate().map_err(|e| e.to_string())?;
        if let Some(entry) = self
            .saved
            .entries
            .iter_mut()
            .find(|entry| entry.scope == wanted && entry.model == config.model)
        {
            entry.options = self.options.clone();
        } else {
            if self.saved.entries.len() >= 512 {
                return Err("Model settings storage limit reached".into());
            }
            self.saved.entries.push(OptionEntry {
                scope: wanted,
                model: config.model.clone(),
                options: self.options.clone(),
            });
        }
        let dir = self
            .dir
            .as_ref()
            .ok_or("Private provider storage is unavailable")?;
        let path = dir.join("model-options.json");
        if std::fs::symlink_metadata(&path).is_ok_and(|meta| meta.file_type().is_symlink()) {
            return Err("Model settings file must not be a link".into());
        }
        let mut file = tempfile::NamedTempFile::new_in(dir).map_err(|e| e.to_string())?;
        serde_json::to_writer_pretty(&mut file, &self.saved).map_err(|e| e.to_string())?;
        file.flush().map_err(|e| e.to_string())?;
        file.as_file().sync_all().map_err(|e| e.to_string())?;
        file.persist(path).map_err(|e| e.error.to_string())?;
        Ok(())
    }
    fn reload_endpoints(&mut self) {
        if let Some(vault) = &self.vault {
            match vault.list() {
                Ok(endpoints) => self.endpoints = endpoints,
                Err(error) => self.error = Some(error.to_string()),
            }
        }
    }
}

impl Hermes {
    pub(super) fn discover_models(&mut self) -> Task<Message> {
        let key = match scope(&self.config.base_url) {
            Ok(value) => value,
            Err(error) => {
                self.models.error = Some(error.to_string());
                return Task::none();
            }
        };
        self.models.generation = self.models.generation.wrapping_add(1);
        let generation = self.models.generation;
        self.models.loading = true;
        self.models.error = None;
        let config = self.config.clone();
        Task::perform(
            async move {
                hermes_engine::models::discover(config)
                    .await
                    .map_err(|e| format!("{e:#}"))
            },
            move |result| Message::Models(Action::Loaded(generation, key.clone(), result)),
        )
    }
    pub(super) fn models_update(&mut self, action: Action) -> Task<Message> {
        match action {
            Action::Open => {
                self.models.picker = true;
                self.models.query.clear();
                if self.models.catalog.is_empty() {
                    return self.discover_models();
                }
            }
            Action::Close => self.models.picker = false,
            Action::Query(value) => self.models.query = value,
            Action::Refresh => return self.discover_models(),
            Action::Loaded(generation, source, result) => {
                if generation != self.models.generation
                    || scope(&self.config.base_url).ok().as_ref() != Some(&source)
                {
                    return Task::none();
                }
                self.models.loading = false;
                match result {
                    Ok(catalog) => {
                        self.models.catalog = catalog;
                        self.models.catalog_scope = Some(source);
                        self.models.error = None;
                        self.portal_connected = portal::is_available();
                    }
                    Err(error) => {
                        if self.is_portal()
                            && (error.contains("HTTP 401") || error.contains("HTTP 403"))
                        {
                            self.portal_connected = false;
                        }
                        self.models.error = Some(error);
                    }
                }
            }
            Action::Select(id) if !self.running => {
                if self.models.catalog.iter().any(|model| model.id == id)
                    || (self.is_portal() && id == portal::AUTO_MODEL)
                {
                    self.config.model = id;
                    self.models.load_options(&self.config);
                    self.models.picker = false;
                    if let Err(error) = self.persist_provider(false) {
                        self.models.error = Some(error);
                    } else {
                        self.note("Model selected");
                    }
                }
            }
            Action::Preset(preset) if !self.running => {
                self.settings_view = match preset {
                    Preset::Nous => settings::View::Accounts,
                    Preset::Local => settings::View::CustomEndpoints,
                    _ => settings::View::ApiKeys,
                };
                self.screen = Screen::Provider;
                self.config = Config {
                    base_url: preset.base().into(),
                    model: if preset == Preset::Nous {
                        portal::AUTO_MODEL.into()
                    } else if preset == Preset::OpenRouter {
                        hermes_engine::DEFAULT_MODEL.into()
                    } else {
                        String::new()
                    },
                    api_key: String::new(),
                };
                self.models.selected_endpoint = None;
                self.models.endpoint_name = preset.name().into();
                if let Some(profile) = self
                    .models
                    .endpoints
                    .iter()
                    .find(|profile| {
                        scope(&profile.base_url).ok() == scope(&self.config.base_url).ok()
                    })
                    .cloned()
                {
                    return self.models_update(Action::UseEndpoint(profile.id));
                }
                self.models.invalidate();
                self.models.load_options(&self.config);
                if preset == Preset::Nous && self.portal_connected {
                    return self.discover_models();
                }
            }
            Action::Name(value) => self.models.endpoint_name = value,
            Action::UseEndpoint(id) if !self.running => {
                match self
                    .models
                    .vault
                    .as_ref()
                    .ok_or_else(|| "Private provider storage is unavailable".to_owned())
                    .and_then(|vault| vault.config(&id).map_err(|e| e.to_string()))
                {
                    Ok(config) => {
                        self.config = config;
                        self.models.selected_endpoint = Some(id.clone());
                        self.models.endpoint_name = self
                            .models
                            .endpoints
                            .iter()
                            .find(|entry| entry.id == id)
                            .map(|entry| entry.name.clone())
                            .unwrap_or_default();
                        self.models.invalidate();
                        self.models.load_options(&self.config);
                        if let Err(error) = Preferences::save(&self.config) {
                            self.models.error = Some(error);
                        }
                        return self.discover_models();
                    }
                    Err(error) => self.models.error = Some(error),
                }
            }
            Action::RemoveEndpoint(id) if !self.running => {
                let result = self
                    .models
                    .vault
                    .as_ref()
                    .ok_or_else(|| "Private provider storage is unavailable".to_owned())
                    .and_then(|vault| vault.remove(&id).map_err(|e| e.to_string()));
                match result {
                    Ok(()) => {
                        if self.models.selected_endpoint.as_deref() == Some(&id) {
                            self.config.api_key.clear();
                            self.models.selected_endpoint = None;
                            self.models.invalidate();
                        }
                        self.models.reload_endpoints();
                        self.note("Endpoint disconnected");
                    }
                    Err(error) => self.models.error = Some(error),
                }
            }
            Action::SaveEndpoint | Action::ApplyProvider(_) if !self.running => {
                let close = matches!(action, Action::ApplyProvider(true));
                match self.persist_provider(close) {
                    Ok(()) => {
                        self.note("Provider settings saved");
                        if !close {
                            return self.discover_models();
                        }
                    }
                    Err(error) => self.models.error = Some(error),
                }
            }
            Action::Disconnect if !self.running && !self.models.disconnecting => {
                if let Some(cancel) = self.portal_cancel.take() {
                    cancel.cancel();
                }
                self.portal_id = self.portal_id.wrapping_add(1);
                self.portal_login = None;
                self.portal_starting = false;
                self.models.disconnecting = true;
                return Task::perform(
                    async { portal::disconnect().await.map_err(|e| e.to_string()) },
                    |result| Message::Models(Action::Disconnected(result)),
                );
            }
            Action::Disconnected(result) => {
                self.models.disconnecting = false;
                match result {
                    Ok(()) => {
                        self.portal_connected = false;
                        if self.is_portal() {
                            self.config.api_key.clear();
                            self.models.invalidate();
                        }
                        self.note("Nous Portal disconnected");
                    }
                    Err(error) => self.models.error = Some(error),
                }
            }
            Action::Temperature(value) => self.models.temperature = value,
            Action::MaxOutput(value) => self.models.max_output = value,
            Action::Reasoning(value) => self.models.reasoning = value,
            Action::ResetOptions if !self.running => {
                self.models.options = GenerationOptions::default();
                self.models.temperature.clear();
                self.models.max_output.clear();
                self.models.reasoning.clear();
                if let Err(error) = self.models.save_options(&self.config) {
                    self.models.error = Some(error);
                } else {
                    self.note("Model defaults restored");
                }
            }
            Action::SaveOptions if !self.running => {
                let result = (|| -> Result<(), String> {
                    let model = self
                        .models
                        .current_model(&self.config)
                        .ok_or("Refresh the model catalog before changing model options")?;
                    let parse = |text: &str| -> Result<Option<f64>, String> {
                        if text.trim().is_empty() {
                            Ok(None)
                        } else {
                            text.trim()
                                .parse()
                                .map(Some)
                                .map_err(|_| "Enter a valid temperature".into())
                        }
                    };
                    let max = if self.models.max_output.trim().is_empty() {
                        None
                    } else {
                        Some(
                            self.models
                                .max_output
                                .trim()
                                .parse::<u32>()
                                .map_err(|_| "Enter a whole number of output tokens")?,
                        )
                    };
                    if let (Some(max), Some(limit)) = (max, model.max_output_tokens) {
                        if u64::from(max) > limit {
                            return Err(format!(
                                "This model supports at most {limit} output tokens"
                            ));
                        }
                    }
                    let supported_max =
                        model.supports("max_tokens") || model.supports("max_completion_tokens");
                    self.models.options = GenerationOptions {
                        temperature: if model.supports("temperature") {
                            parse(&self.models.temperature)?
                        } else {
                            None
                        },
                        max_output_tokens: if supported_max { max } else { None },
                        fast: model.supports("service_tier") && self.models.options.fast,
                        reasoning_effort: if (model.supports("reasoning")
                            || model.supports("reasoning_effort"))
                            && !self.models.reasoning.is_empty()
                        {
                            Some(self.models.reasoning.clone())
                        } else {
                            None
                        },
                        reasoning_object: model.supports("reasoning"),
                        max_completion_tokens: model.supports("max_completion_tokens")
                            && !model.supports("max_tokens"),
                    };
                    self.models.save_options(&self.config)
                })();
                match result {
                    Ok(()) => self.note("Model settings saved"),
                    Err(error) => self.models.error = Some(error),
                }
            }
            _ => {}
        }
        Task::none()
    }
    fn persist_provider(&mut self, close: bool) -> Result<(), String> {
        self.config.validate().map_err(|e| e.to_string())?;
        if !self.is_portal() {
            let name = if self.models.endpoint_name.trim().is_empty() {
                "Custom endpoint"
            } else {
                &self.models.endpoint_name
            };
            let vault = self
                .models
                .vault
                .as_ref()
                .ok_or("Private provider storage is unavailable")?;
            let saved = vault
                .save(self.models.selected_endpoint.as_deref(), name, &self.config)
                .map_err(|e| e.to_string())?;
            self.models.selected_endpoint = Some(saved.id);
            self.models.reload_endpoints();
        }
        Preferences::save(&self.config)?;
        if close {
            self.screen = Screen::Chat;
            self.models.picker = false;
        }
        Ok(())
    }
}

pub(super) fn picker(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let head = row![
        text("Select a model").size(16).font(medium()),
        Space::new().width(Fill),
        button(text("×").size(20))
            .on_press(Message::Models(Action::Close))
            .padding(4)
            .style(button::text)
    ]
    .align_y(Alignment::Center);
    let search = text_input("Search models…", &app.models.query)
        .on_input(|value| Message::Models(Action::Query(value)))
        .size(13)
        .padding(9);
    let header = row![
        search,
        button(
            text(if app.models.loading {
                "Loading…"
            } else {
                "Refresh"
            })
            .size(12)
        )
        .on_press_maybe((!app.models.loading).then_some(Message::Models(Action::Refresh)))
        .padding([8, 10])
        .style(button::text)
    ]
    .spacing(6);
    let mut list = column![].spacing(2);
    if app.is_portal() {
        list = list.push(model_button(app, portal::AUTO_MODEL, "Auto", None));
    }
    let query = normalize(&app.models.query);
    let mut count = 0;
    for model in app
        .models
        .catalog
        .iter()
        .filter(|model| {
            query.is_empty() || normalize(&format!("{} {}", model.id, model.name)).contains(&query)
        })
        .take(150)
    {
        count += 1;
        list = list.push(model_button(app, &model.id, &model.name, Some(model)));
    }
    if let Some(error) = &app.models.error {
        list = list.push(text(error).size(12).color(p.muted));
    } else if app.models.loading && app.models.catalog.is_empty() {
        list = list.push(text("Loading models…").size(12).color(p.muted));
    } else if count == 0 {
        list = list.push(text("No matching models.").size(12).color(p.muted));
    }
    let footer = row![
        text(provider_label(app)).size(11).color(p.muted),
        Space::new().width(Fill),
        button(text("Manage providers").size(11))
            .on_press(Message::SettingsView(settings::View::Accounts))
            .padding(5)
            .style(button::text)
    ]
    .align_y(Alignment::Center);
    container(column![head, header, scrollable(list).height(360), footer].spacing(12))
        .padding(18)
        .width(680)
        .max_width(680)
        .style(move |_| container::Style {
            background: Some(p.background.into()),
            text_color: Some(p.foreground),
            border: Border {
                color: p.border,
                width: 1.0,
                radius: 10.0.into(),
            },
            ..Default::default()
        })
        .into()
}

pub(super) fn model_rows(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let provider = row![
        text("Provider").size(13),
        Space::new().width(Fill),
        button(row![text(provider_label(app)).size(13).width(Fill), ui::icon("chevron-right", 11, p.muted)].align_y(Alignment::Center))
            .on_press(Message::SettingsView(settings::View::Accounts))
            .padding([7, 10])
            .width(288)
            .style(move |_, status| button::Style { background: Some(iced::Color { a: if status == button::Status::Hovered { 0.07 } else { 0.04 }, ..p.foreground }.into()),
                text_color: p.foreground, border: iced::Border { color: iced::Color { a: 0.1, ..p.foreground }, width: 1.0, radius: 3.0.into() }, ..Default::default() })
    ]
    .align_y(Alignment::Center);
    // A dropdown of the provider's catalog (model-select.tsx). Until the catalog
    // loads it shows the current model and offers a refresh.
    let options: Vec<ModelOption> = app.models.catalog.iter().map(|m| ModelOption { id: m.id.clone(), label: if m.name.is_empty() { m.id.clone() } else { m.name.clone() } }).collect();
    let selected = options.iter().find(|o| o.id == app.config.model).cloned()
        .or_else(|| Some(ModelOption { id: app.config.model.clone(), label: app.models.model_label(&app.config).to_owned() }));
    let picker: Element<'_, Message> = if options.is_empty() {
        button(text(if app.models.loading { "Loading models…".to_owned() } else { format!("{} · Load models", app.models.model_label(&app.config)) }).size(13))
            .on_press_maybe((!app.running && !app.models.loading).then_some(Message::Models(Action::Refresh))).padding([7, 10]).width(288).style(button::secondary).into()
    } else {
        iced::widget::pick_list(options, selected, |o: ModelOption| Message::Models(Action::Select(o.id)))
            .placeholder("Choose a model").text_size(13).padding([7, 10]).width(288).into()
    };
    let model = row![text("Model").size(13), Space::new().width(Fill), picker].align_y(Alignment::Center);
    let manual = text_input("Custom model ID", &app.config.model)
        .on_input_maybe((!app.running).then_some(Message::Model))
        .size(12)
        .padding(8);
    let mut content = column![
        provider,
        model,
        Space::new().height(8),
        manual,
        row![
            Space::new().width(Fill),
            button(text("Apply").size(12))
                .on_press_maybe(
                    (!app.running).then_some(Message::Models(Action::ApplyProvider(false)))
                )
                .padding([7, 14])
        ]
        .spacing(6)
    ]
    .spacing(8);
    if let Some(model) = app.models.current_model(&app.config) {
        if let Some(limit) = model.context_length {
            content = content.push(
                text(format!("Context: {limit} tokens"))
                    .size(11)
                    .color(p.muted),
            );
        }
        if model.supports("temperature") {
            content = content.push(setting_input(
                "Temperature",
                "Default",
                &app.models.temperature,
                Action::Temperature,
                app.running,
            ));
        }
        if model.supports("max_tokens") || model.supports("max_completion_tokens") {
            content = content.push(setting_input(
                "Maximum output tokens",
                "Default",
                &app.models.max_output,
                Action::MaxOutput,
                app.running,
            ));
        }
        if model.supports("reasoning") || model.supports("reasoning_effort") {
            let mut choices = row![text("Reasoning effort").size(12), Space::new().width(Fill)]
                .spacing(4)
                .align_y(Alignment::Center);
            for (id, label) in [
                ("", "Default"),
                ("low", "Low"),
                ("medium", "Med"),
                ("high", "High"),
            ] {
                let active = app.models.reasoning == id;
                choices = choices.push(
                    button(text(label).size(11))
                        .on_press_maybe(
                            (!app.running).then_some(Message::Models(Action::Reasoning(id.into()))),
                        )
                        .padding([5, 8])
                        .style(move |theme, status| {
                            if active {
                                button::primary(theme, status)
                            } else {
                                button::text(theme, status)
                            }
                        }),
                );
            }
            content = content.push(choices);
        }
        if model.supports("temperature")
            || model.supports("max_tokens")
            || model.supports("max_completion_tokens")
            || model.supports("reasoning")
            || model.supports("reasoning_effort")
        {
            content = content.push(
                row![
                    button(text("Reset to defaults").size(11))
                        .on_press_maybe(
                            (!app.running).then_some(Message::Models(Action::ResetOptions))
                        )
                        .style(button::text),
                    Space::new().width(Fill),
                    button(text("Save model settings").size(12))
                        .on_press_maybe(
                            (!app.running).then_some(Message::Models(Action::SaveOptions))
                        )
                        .padding([6, 10])
                ]
                .spacing(8),
            );
        }
    }
    if let Some(error) = &app.models.error {
        content = content.push(text(error).size(12).color(p.muted));
    }
    content.into()
}
pub(super) fn endpoint_rows(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let mut list = column![].spacing(8);
    for endpoint in &app.models.endpoints {
        list =
            list.push(
                row![
                    button(
                        column![
                            text(&endpoint.name).size(13),
                            text(&endpoint.base_url).size(11).color(p.muted)
                        ]
                        .spacing(4)
                    )
                    .on_press_maybe(
                        (!app.running)
                            .then_some(Message::Models(Action::UseEndpoint(endpoint.id.clone())))
                    )
                    .padding([8, 0])
                    .width(Fill)
                    .style(button::text),
                    text(
                        if app.models.selected_endpoint.as_deref() == Some(&endpoint.id) {
                            "Selected"
                        } else if endpoint.has_key {
                            "Connected"
                        } else {
                            "Local"
                        }
                    )
                    .size(11)
                    .color(p.muted),
                    button(text("Disconnect").size(11))
                        .on_press_maybe((!app.running).then_some(Message::Models(
                            Action::RemoveEndpoint(endpoint.id.clone())
                        )))
                        .padding(6)
                        .style(button::text)
                ]
                .spacing(8)
                .align_y(Alignment::Center),
            );
    }
    list.into()
}
pub(super) fn provider_label(app: &Hermes) -> &str {
    app.models
        .endpoints
        .iter()
        .find(|endpoint| app.models.selected_endpoint.as_deref() == Some(&endpoint.id))
        .map(|endpoint| endpoint.name.as_str())
        .unwrap_or(if app.is_portal() {
            "Nous Portal"
        } else if app.config.base_url.contains("openrouter.ai") {
            "OpenRouter"
        } else if app.config.base_url.contains("api.openai.com") {
            "OpenAI"
        } else if app.config.base_url.contains("api.anthropic.com") {
            "Anthropic"
        } else {
            "Custom endpoint"
        })
}
fn model_button<'a>(
    app: &Hermes,
    id: &str,
    name: &str,
    model: Option<&Model>,
) -> Element<'a, Message> {
    let p = ui::palette(app.dark);
    let selected = app.config.model == id;
    let mut title = row![text(name.to_owned()).size(13), Space::new().width(Fill)].spacing(8);
    if model.is_some_and(|model| model.free) {
        title = title.push(text("Free").size(10).color(p.accent));
    }
    if selected {
        title = title.push(text("✓").size(13).color(p.accent));
    }
    let details = model
        .map(|model| {
            format!(
                "{}{}",
                model.id,
                model
                    .context_length
                    .map(|limit| format!(" · {limit} context"))
                    .unwrap_or_default()
            )
        })
        .unwrap_or_else(|| id.to_owned());
    button(column![title, text(details).size(10).color(p.muted)].spacing(4))
        .on_press_maybe((!app.running).then_some(Message::Models(Action::Select(id.to_owned()))))
        .padding([9, 10])
        .width(Fill)
        .style(move |_, status| button::Style {
            background: (selected || status == button::Status::Hovered)
                .then_some(if selected { p.secondary } else { p.card }.into()),
            text_color: p.foreground,
            border: Border {
                radius: 4.0.into(),
                ..Default::default()
            },
            ..Default::default()
        })
        .into()
}
fn setting_input<'a>(
    label: &'static str,
    placeholder: &'static str,
    value: &'a str,
    action: fn(String) -> Action,
    running: bool,
) -> Element<'a, Message> {
    let mut field = text_input(placeholder, value)
        .size(12)
        .padding(7)
        .width(150);
    if !running {
        field = field.on_input(move |value| Message::Models(action(value)));
    }
    row![text(label).size(12), Space::new().width(Fill), field]
        .spacing(12)
        .align_y(Alignment::Center)
        .into()
}
fn normalize(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}
fn medium() -> Font {
    Font {
        weight: iced::font::Weight::Medium,
        ..Font::DEFAULT
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ModelOption { id: String, label: String }
impl std::fmt::Display for ModelOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.label) }
}
