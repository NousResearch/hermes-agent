//! Native Scheduled jobs overlay, projected from stock 0284ce17 app/cron.
//! The in-app ticker serves this native profile while Hermes is open. External
//! delivery, blueprints, scripts and monitor jobs remain outside this port.
use crate::appearance::text;
use crate::ui;
use chrono::Utc;
use hermes_engine::{
    CancellationToken, Config, Event, ToolCall,
    cron::{self, Claim, Draft, Job, Outcome, Run, Store},
};
use iced::widget::text as text_widget;
use iced::{
    Alignment, Border, Color, Element, Fill, Subscription, Task, Theme,
    widget::{
        Space, button, checkbox, column, container, pick_list, row, scrollable, text_editor,
        text_input,
    },
};
use std::{collections::BTreeMap, fmt, path::PathBuf, time::Duration};

#[derive(Clone, Debug)]
pub enum Action {
    /// Repaint the running timers.
    Redraw,
    Close,
    Refresh,
    Tick,
    Search(String),
    Select(String),
    New,
    Scratch,
    TemplatesClose,
    TemplatePick(usize),
    TemplateSlot(String, String),
    TemplateUse,
    Edit(String),
    EditorClose,
    Save,
    Name(String),
    Prompt(text_editor::Action),
    Schedule(String),
    Timezone(String),
    Model(String),
    Frequency(Frequency),
    CatchUp(CatchUp),
    Pause(String),
    Resume(String),
    Trigger(String),
    Delete(String),
    DeleteCancel,
    DeleteConfirm,
    Stop(String),
    Approve(String, bool),
    ShowRun(String),
    Opened(u64, Result<Store, String>),
    Loaded(u64, Result<Data, String>),
    Saved(u64, Result<Job, String>),
    Changed(u64, Result<(), String>),
    Claimed(u64, bool, Result<Vec<Claim>, String>),
    Engine(u64, String, Event),
    Finished(u64, String, Result<Outcome, String>),
}
#[derive(Clone, Debug)]
pub struct Data {
    jobs: Vec<Job>,
    runs: Vec<Run>,
    selected: Option<String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frequency {
    Daily,
    Weekdays,
    Weekly,
    Monthly,
    Hourly,
    Every15,
    Custom,
}
impl Frequency {
    const ALL: [Self; 7] = [
        Self::Daily,
        Self::Weekdays,
        Self::Weekly,
        Self::Monthly,
        Self::Hourly,
        Self::Every15,
        Self::Custom,
    ];
    fn expression(self) -> Option<&'static str> {
        match self {
            Self::Daily => Some("0 9 * * *"),
            Self::Weekdays => Some("0 9 * * 1-5"),
            Self::Weekly => Some("0 9 * * 1"),
            Self::Monthly => Some("0 9 1 * *"),
            Self::Hourly => Some("0 * * * *"),
            Self::Every15 => Some("*/15 * * * *"),
            Self::Custom => None,
        }
    }
    fn for_expression(expr: &str) -> Self {
        Self::ALL
            .into_iter()
            .find(|frequency| frequency.expression() == Some(expr))
            .unwrap_or(Self::Custom)
    }
}
impl fmt::Display for Frequency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Daily => "Daily",
            Self::Weekdays => "Weekdays",
            Self::Weekly => "Weekly",
            Self::Monthly => "Monthly",
            Self::Hourly => "Hourly",
            Self::Every15 => "Every 15 minutes",
            Self::Custom => "Custom",
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatchUp {
    Default,
    Run,
    Skip,
}
impl fmt::Display for CatchUp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Default => "Default",
            Self::Run => "Run once",
            Self::Skip => "Skip",
        })
    }
}
struct Editor {
    id: Option<String>,
    draft: Draft,
    prompt: text_editor::Content,
    frequency: Frequency,
    model: String,
}
struct Pending {
    call: ToolCall,
    answer: async_channel::Sender<bool>,
}
struct Active {
    session_dir: PathBuf,
    job: String,
    title: String,
    started: std::time::Instant,
    cancel: CancellationToken,
    manual: bool,
    approval: Option<Pending>,
    text: String,
}
pub struct Scheduled {
    profile: Option<PathBuf>,
    workspace: Option<PathBuf>,
    store: Option<Store>,
    jobs: Vec<Job>,
    runs: Vec<Run>,
    selected: Option<String>,
    search: String,
    editor: Option<Editor>,
    delete: Option<String>,
    shown_run: Option<String>,
    error: Option<String>,
    loading: bool,
    busy: bool,
    ticking: bool,
    scope: u64,
    request: u64,
    active: BTreeMap<String, Active>,
    /// Routines gallery: open, and the template being filled with its blanks.
    templates: bool,
    template: Option<(usize, BTreeMap<String, String>)>,
}
impl Scheduled {
    pub fn new() -> Self {
        Self {
            profile: None,
            workspace: None,
            store: None,
            jobs: vec![],
            runs: vec![],
            selected: None,
            search: String::new(),
            editor: None,
            delete: None,
            shown_run: None,
            error: None,
            loading: false,
            busy: false,
            ticking: false,
            scope: 0,
            request: 0,
            active: BTreeMap::new(),
            templates: false,
            template: None,
        }
    }
    pub fn reset(&mut self, profile: Option<PathBuf>, workspace: Option<PathBuf>) -> Task<Action> {
        self.workspace = workspace;
        if self.profile == profile && self.store.is_some() {
            return Task::none();
        }
        self.scope = self.scope.wrapping_add(1);
        self.request = self.request.wrapping_add(1);
        self.profile = profile.clone();
        self.store = None;
        self.jobs.clear();
        self.runs.clear();
        self.selected = None;
        self.editor = None;
        self.error = None;
        self.busy = false;
        self.ticking = false;
        let Some(profile) = profile else {
            self.loading = false;
            return Task::none();
        };
        self.loading = true;
        let scope = self.scope;
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || {
                    Store::open(&profile).map_err(|error| format!("{error:#}"))
                })
                .await
                .unwrap_or_else(|error| Err(error.to_string()))
            },
            move |result| Action::Opened(scope, result),
        )
    }
    pub fn subscription(&self) -> Subscription<Action> {
        if self.store.is_some() {
            let ticks = iced::time::every(Duration::from_secs(15)).map(|_| Action::Tick);
            if self.active.is_empty() { ticks } else { Subscription::batch([ticks, iced::time::every(Duration::from_secs(1)).map(|_| Action::Redraw)]) }
        } else {
            Subscription::none()
        }
    }
    fn reload(&mut self) -> Task<Action> {
        let Some(store) = self.store.clone() else {
            return Task::none();
        };
        self.request = self.request.wrapping_add(1);
        let request = self.request;
        let selected = self.selected.clone();
        self.loading = true;
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || {
                    let jobs = store.jobs().map_err(|error| format!("{error:#}"))?;
                    let selected = selected
                        .filter(|id| jobs.iter().any(|job| job.id == *id))
                        .or_else(|| jobs.first().map(|job| job.id.clone()));
                    let runs = if let Some(id) = &selected {
                        store.runs(id).map_err(|error| format!("{error:#}"))?
                    } else {
                        vec![]
                    };
                    Ok(Data {
                        jobs,
                        runs,
                        selected,
                    })
                })
                .await
                .unwrap_or_else(|error| Err(error.to_string()))
            },
            move |result| Action::Loaded(request, result),
        )
    }
    fn mutate(
        &mut self,
        operation: impl FnOnce(Store) -> Result<(), String> + Send + 'static,
    ) -> Task<Action> {
        let Some(store) = self.store.clone() else {
            return Task::none();
        };
        self.busy = true;
        self.error = None;
        let scope = self.scope;
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || operation(store))
                    .await
                    .unwrap_or_else(|error| Err(error.to_string()))
            },
            move |result| Action::Changed(scope, result),
        )
    }
    fn start(&mut self, claim: Claim, manual: bool, config: Config, scope: u64) -> Task<Action> {
        let Some(store) = self.store.clone() else {
            return Task::none();
        };
        let id = claim.run.id.clone();
        let cancel = CancellationToken::new();
        self.active.insert(
            id.clone(),
            Active {
                session_dir: claim.run.session_dir.clone(),
                job: claim.job.id.clone(),
                title: claim.job.title().to_owned(),
                started: std::time::Instant::now(),
                cancel: cancel.clone(),
                manual,
                approval: None,
                text: String::new(),
            },
        );
        let (send, receive) = async_channel::unbounded();
        let messages = iced::futures::stream::unfold(receive, |receive| async {
            receive.recv().await.ok().map(|event| (event, receive))
        });
        let event_id = id.clone();
        Task::batch([
            Task::run(messages, move |event| {
                Action::Engine(scope, event_id.clone(), event)
            }),
            Task::perform(
                async move {
                    cron::execute(store, claim, config, send, cancel)
                        .await
                        .map_err(|error| format!("{error:#}"))
                },
                move |result| Action::Finished(scope, id.clone(), result),
            ),
        ])
    }
    pub fn update(&mut self, action: Action, config: Config) -> Task<Action> {
        match action {
            Action::Refresh => return self.reload(),
            Action::Opened(scope, result) if scope == self.scope => {
                self.loading = false;
                match result {
                    Ok(store) => {
                        self.store = Some(store);
                        return self.reload();
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            Action::Loaded(request, result) if request == self.request => {
                self.loading = false;
                match result {
                    Ok(data) => {
                        self.jobs = data.jobs;
                        self.runs = data.runs;
                        self.selected = data.selected;
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            Action::Search(search) => self.search = search,
            Action::Select(id) => {
                self.selected = Some(id);
                self.shown_run = None;
                return self.reload();
            }
            Action::Redraw => {}
            Action::New => { self.templates = true; self.template = None; self.error = None; }
            Action::TemplatesClose => { self.templates = false; self.template = None; }
            Action::TemplatePick(i) => if let Some(t) = super::routines::TEMPLATES.get(i) { self.template = Some((i, super::routines::defaults(t))); },
            Action::TemplateSlot(name, value) => if let Some((_, values)) = &mut self.template { values.insert(name, value); },
            Action::TemplateUse => {
                let Some((i, values)) = self.template.clone() else { return Task::none() };
                let Some(t) = super::routines::TEMPLATES.get(i) else { return Task::none() };
                match super::routines::fill(t, &values) {
                    Ok((name, schedule, prompt)) => {
                        let _ = self.update(Action::Scratch, config.clone());
                        if let Some(editor) = &mut self.editor {
                            editor.draft.name = name; editor.draft.schedule = schedule.clone(); editor.draft.prompt = prompt.clone();
                            editor.prompt = text_editor::Content::with_text(&prompt);
                            editor.frequency = Frequency::for_expression(&schedule);
                        }
                    }
                    Err(e) => self.error = Some(e),
                }
            }
            Action::Scratch => {
                self.templates = false; self.template = None;
                let mut draft = Draft {
                    workdir: self.workspace.clone(),
                    ..Draft::default()
                };
                if let Some(profile) = &self.profile
                    && let Ok(settings) = hermes_engine::settings::ProfileSettings::load(profile)
                    && !settings.timezone.is_empty()
                {
                    draft.timezone = settings.timezone;
                }
                self.editor = Some(Editor {
                    id: None,
                    prompt: text_editor::Content::new(),
                    frequency: Frequency::Daily,
                    model: String::new(),
                    draft,
                });
                self.error = None;
            }
            Action::Edit(id) => {
                if let Some(job) = self.jobs.iter().find(|job| job.id == id) {
                    let draft = job.draft();
                    self.editor = Some(Editor {
                        id: Some(id),
                        prompt: text_editor::Content::with_text(&draft.prompt),
                        frequency: Frequency::for_expression(&draft.schedule),
                        model: draft.model.clone().unwrap_or_default(),
                        draft,
                    });
                    self.error = None;
                }
            }
            Action::EditorClose if !self.busy => self.editor = None,
            Action::Name(value) => {
                if let Some(editor) = &mut self.editor {
                    editor.draft.name = value;
                }
            }
            Action::Prompt(action) => {
                if let Some(editor) = &mut self.editor {
                    editor.prompt.perform(action);
                }
            }
            Action::Schedule(value) => {
                if let Some(editor) = &mut self.editor {
                    editor.draft.schedule = value;
                    editor.frequency = Frequency::Custom;
                }
            }
            Action::Timezone(value) => {
                if let Some(editor) = &mut self.editor {
                    editor.draft.timezone = value;
                }
            }
            Action::Model(value) => {
                if let Some(editor) = &mut self.editor {
                    editor.model = value;
                }
            }
            Action::Frequency(frequency) => {
                if let Some(editor) = &mut self.editor {
                    editor.frequency = frequency;
                    if let Some(expr) = frequency.expression() {
                        editor.draft.schedule = expr.into();
                    }
                }
            }
            Action::CatchUp(choice) => {
                if let Some(editor) = &mut self.editor {
                    editor.draft.catch_up = match choice {
                        CatchUp::Default => None,
                        CatchUp::Run => Some(true),
                        CatchUp::Skip => Some(false),
                    };
                }
            }
            Action::Save if !self.busy => {
                let (Some(store), Some(editor)) = (self.store.clone(), self.editor.as_ref()) else {
                    return Task::none();
                };
                let mut draft = editor.draft.clone();
                draft.prompt = editor.prompt.text();
                draft.model =
                    (!editor.model.trim().is_empty()).then(|| editor.model.trim().to_owned());
                if draft.model.is_none() {
                    draft.provider = None;
                }
                let id = editor.id.clone();
                let scope = self.scope;
                self.busy = true;
                self.error = None;
                return Task::perform(
                    async move {
                        tokio::task::spawn_blocking(move || {
                            if let Some(id) = id {
                                store.update(&id, draft, Utc::now())
                            } else {
                                store.create(draft, Utc::now())
                            }
                            .map_err(|error| format!("{error:#}"))
                        })
                        .await
                        .unwrap_or_else(|error| Err(error.to_string()))
                    },
                    move |result| Action::Saved(scope, result),
                );
            }
            Action::Saved(scope, result) if scope == self.scope => {
                self.busy = false;
                match result {
                    Ok(job) => {
                        self.editor = None;
                        self.selected = Some(job.id);
                        return self.reload();
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            Action::Pause(id) => {
                return self.mutate(move |store| {
                    store
                        .pause(&id, Utc::now())
                        .map_err(|error| format!("{error:#}"))
                });
            }
            Action::Resume(id) => {
                return self.mutate(move |store| {
                    store
                        .resume(&id, Utc::now())
                        .map_err(|error| format!("{error:#}"))
                });
            }
            Action::Delete(id) => self.delete = Some(id),
            Action::DeleteCancel => self.delete = None,
            Action::DeleteConfirm => {
                if let Some(id) = self.delete.take() {
                    return self.mutate(move |store| {
                        store.remove(&id).map_err(|error| format!("{error:#}"))
                    });
                }
            }
            Action::Changed(scope, result) if scope == self.scope => {
                self.busy = false;
                match result {
                    Ok(()) => return self.reload(),
                    Err(error) => self.error = Some(error),
                }
            }
            Action::Tick if !self.ticking => {
                if config.validate().is_err()
                    || (config.base_url.trim_end_matches('/')
                        == hermes_engine::portal::INFERENCE.trim_end_matches('/')
                        && !hermes_engine::portal::is_available())
                {
                    return self.reload();
                }
                let Some(store) = self.store.clone() else {
                    return Task::none();
                };
                self.ticking = true;
                let scope = self.scope;
                return Task::perform(
                    async move {
                        tokio::task::spawn_blocking(move || {
                            store
                                .claim_due(Utc::now())
                                .map_err(|error| format!("{error:#}"))
                        })
                        .await
                        .unwrap_or_else(|error| Err(error.to_string()))
                    },
                    move |result| Action::Claimed(scope, false, result),
                );
            }
            Action::Trigger(id) if !self.busy => {
                let Some(store) = self.store.clone() else {
                    return Task::none();
                };
                if let Err(error) = config.validate() {
                    self.error = Some(error.to_string());
                    return Task::none();
                }
                self.busy = true;
                self.error = None;
                let scope = self.scope;
                return Task::perform(
                    async move {
                        tokio::task::spawn_blocking(move || {
                            store
                                .claim_now(&id, Utc::now())
                                .map(|claim| vec![claim])
                                .map_err(|error| format!("{error:#}"))
                        })
                        .await
                        .unwrap_or_else(|error| Err(error.to_string()))
                    },
                    move |result| Action::Claimed(scope, true, result),
                );
            }
            Action::Claimed(scope, manual, result) if scope == self.scope => {
                if manual {
                    self.busy = false;
                } else {
                    self.ticking = false;
                }
                match result {
                    Ok(claims) => {
                        let mut tasks = Vec::new();
                        for claim in claims {
                            tasks.push(self.start(claim, manual, config.clone(), scope));
                        }
                        tasks.push(self.reload());
                        return Task::batch(tasks);
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            Action::Engine(scope, id, event) => {
                let Some(active) = self.active.get_mut(&id) else {
                    if let Event::Approval { answer, .. } = event {
                        let _ = answer.try_send(false);
                    }
                    return Task::none();
                };
                match event {
                    Event::Text(delta) => {
                        if active.text.len() < 262144 {
                            active.text.push_str(&delta);
                        }
                    }
                    Event::Approval { call, answer } => {
                        if active.manual && scope == self.scope {
                            active.approval = Some(Pending { call, answer });
                        } else {
                            let _ = answer.try_send(false);
                        }
                    }
                    Event::Snapshot(_) | Event::Subagent(_) => {}
                }
            }
            Action::Approve(id, approved) => {
                if let Some(active) = self.active.get_mut(&id)
                    && let Some(approval) = active.approval.take()
                {
                    let _ = approval.answer.try_send(approved);
                }
            }
            Action::Stop(id) => {
                if let Some(active) = self.active.get_mut(&id) {
                    active.cancel.cancel();
                    if let Some(approval) = active.approval.take() {
                        let _ = approval.answer.try_send(false);
                    }
                }
                if let Some(store) = self.store.clone() {
                    return self
                        .mutate(move |_| store.cancel(&id).map_err(|error| format!("{error:#}")));
                }
            }
            Action::Finished(scope, id, result) => {
                self.active.remove(&id);
                if scope == self.scope {
                    if let Err(error) = result {
                        self.error = Some(error);
                    }
                    return self.reload();
                }
            }
            Action::ShowRun(id) => {
                self.shown_run = if self.shown_run.as_ref() == Some(&id) {
                    None
                } else {
                    Some(id)
                }
            }
            _ => {}
        }
        Task::none()
    }
    /// (job id, title, the run's transcript folder) for a run still tracked as active.
    pub fn run_info(&self, run: &str) -> Option<(String, String, PathBuf)> {
        self.active.get(run).map(|a| (a.job.clone(), a.title.clone(), a.session_dir.clone()))
    }
    /// Every job, for the sidebar's Cron jobs section.
    pub fn jobs(&self) -> &[Job] { &self.jobs }
    /// Seconds the job has been running, if a run of it is active.
    pub fn running_for(&self, job: &str) -> Option<u64> { self.active.values().filter(|a| a.job == job).map(|a| a.started.elapsed().as_secs()).max() }
    /// The job's newest active run as it streams: (seconds running, text so far, waiting on approval).
    pub fn live(&self, job: &str) -> Option<(u64, &str, bool)> {
        self.active.values().filter(|a| a.job == job).min_by_key(|a| a.started.elapsed()).map(|a| (a.started.elapsed().as_secs(), a.text.as_str(), a.approval.is_some()))
    }
    /// The job's most recent runs, newest first: (run id, when it started).
    pub fn recent_runs(&self, job: &str, limit: usize) -> Vec<(String, String)> {
        let mut runs: Vec<&Run> = self.runs.iter().filter(|r| r.job_id == job).collect();
        runs.sort_by(|a, b| b.claimed_at.cmp(&a.claimed_at));
        runs.into_iter().take(limit).map(|r| (r.id.clone(), r.started_at.clone().unwrap_or_else(|| r.claimed_at.clone()))).collect()
    }
    /// Job id and title of a routine by run id or job id (for opening its chat).
    pub fn job_title(&self, job: &str) -> Option<String> { self.jobs.iter().find(|j| j.id == job).map(|j| j.title().to_owned()) }


    /// Routines › New: the curated templates (cron/blueprint_catalog.py), then their blanks.
    fn templates_view(&self) -> Element<'_, Action> {
        use super::routines::TEMPLATES;
        if let Some((i, values)) = &self.template && let Some(t) = TEMPLATES.get(*i) {
            let mut form = column![text(t.title.clone()).size(15).font(semibold()), text(t.description.clone()).size(12).style(muted)].spacing(10);
            for slot in t.slots.iter().filter(|s| s.name != "deliver") {
                let value = values.get(&slot.name).cloned().unwrap_or_default();
                let name = slot.name.clone();
                let control: Element<'_, Action> = if slot.kind == "enum" && !slot.options.is_empty() {
                    pick_list(slot.options.clone(), Some(value), move |v| Action::TemplateSlot(name.clone(), v)).text_size(12).padding([5, 8]).into()
                } else if slot.kind == "weekdays" {
                    pick_list(["everyday", "weekdays", "weekends", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"].map(str::to_owned).to_vec(), Some(value), move |v| Action::TemplateSlot(name.clone(), v)).text_size(12).padding([5, 8]).into()
                } else {
                    text_input(&slot.default, &value).on_input(move |v| Action::TemplateSlot(name.clone(), v)).size(12).padding([6, 8]).width(if slot.kind == "time" { 90.0 } else { 360.0 }).into()
                };
                let mut label = column![text(slot.label.clone()).size(12)].spacing(2).width(220);
                if !slot.help.is_empty() { label = label.push(text(slot.help.clone()).size(10).style(muted)); }
                form = form.push(row![label, control].spacing(14).align_y(Alignment::Center));
            }
            form = form.push(row![button(text("Back").size(12)).on_press(Action::New).style(quiet), Space::new().width(Fill),
                button(text("Continue").size(12)).on_press(Action::TemplateUse).padding([6, 14])].spacing(8));
            return scrollable(container(form).padding(16).max_width(680)).height(Fill).into();
        }
        let mut col = column![row![text("Start from a template").size(15).font(semibold()), Space::new().width(Fill),
            button(text("Start from scratch").size(12)).on_press(Action::Scratch).style(quiet), button(text("Cancel").size(12)).on_press(Action::TemplatesClose).style(quiet)].spacing(8).align_y(Alignment::Center)].spacing(10);
        let mut categories: Vec<&str> = TEMPLATES.iter().map(|t| t.category.as_str()).collect();
        categories.sort(); categories.dedup();
        for category in categories {
            col = col.push(text(category.to_uppercase()).size(10).font(semibold()).style(muted));
            let items: Vec<(usize, &super::routines::Template)> = TEMPLATES.iter().enumerate().filter(|(_, t)| t.category == category).collect();
            for chunk in items.chunks(2) {
                let mut line = row![].spacing(10);
                for (i, t) in chunk {
                    line = line.push(button(column![text(t.title.clone()).size(13).font(semibold()), text(t.description.clone()).size(11).style(muted)].spacing(4))
                        .width(Fill).padding(12).on_press(Action::TemplatePick(*i)).style(move |theme, status| list_row(theme, status, false)));
                }
                if chunk.len() == 1 { line = line.push(Space::new().width(Fill)); }
                col = col.push(line);
            }
        }
        scrollable(col).height(Fill).into()
    }

    pub fn view(&self) -> Element<'_, Action> {
        let header = row![
            column![
                text("Routines").size(14).font(semibold()),
                text(format!(
                    "{} {}",
                    self.jobs.len(),
                    if self.jobs.len() == 1 { "routine" } else { "routines" }
                ))
                .size(12)
                .style(muted)
            ]
            .spacing(3)
            .width(Fill),
            button(text("⟳").size(15))
                .on_press_maybe((!self.loading).then_some(Action::Refresh))
                .padding([3, 6])
                .style(quiet),
            button(text("×").size(19))
                .on_press(Action::Close)
                .padding([1, 5])
                .style(quiet)
        ]
        .spacing(8)
        .align_y(Alignment::Center);
        let mut panel = column![header].spacing(12).height(Fill);
        if let Some(error) = &self.error {
            panel = panel.push(
                container(
                    text(error)
                        .size(12)
                        .style(|theme: &Theme| text_widget::Style {
                            color: Some(theme.palette().danger),
                        }),
                )
                .padding([6, 8]),
            );
        }
        let body: Element<'_, Action> = if self.templates && self.editor.is_none() {
            self.templates_view()
        } else if let Some(editor) = &self.editor {
            self.editor_view(editor)
        } else if let Some(id) = &self.delete {
            let name = self
                .jobs
                .iter()
                .find(|job| job.id == *id)
                .map_or(id.as_str(), |job| job.title());
            container(
                column![
                    text("Delete cron job?").size(15).font(semibold()),
                    text(format!(
                        "This will remove {name} permanently. It will stop firing immediately."
                    ))
                    .size(12),
                    row![
                        button(text("Cancel").size(12))
                            .on_press(Action::DeleteCancel)
                            .style(quiet),
                        button(text("Delete").size(12))
                            .on_press(Action::DeleteConfirm)
                            .style(button::danger)
                    ]
                    .spacing(8)
                ]
                .spacing(14),
            )
            .padding(20)
            .center_x(Fill)
            .center_y(Fill)
            .height(Fill)
            .into()
        } else {
            let search = text_input("Search routines...", &self.search)
                .on_input(Action::Search)
                .size(12)
                .padding([6, 4]);
            let mut list = column![search].spacing(5);
            let query = self.search.to_lowercase();
            let mut visible = 0;
            let now = Utc::now();
            let mut shown: Vec<(&Job, super::routines::Group, &'static str)> = self.jobs.iter().filter(|job| {
                query.is_empty()
                    || job.title().to_lowercase().contains(&query)
                    || job.prompt.to_lowercase().contains(&query)
            }).map(|job| { let (g, chip) = super::routines::classify(job, self.active.values().any(|a| a.job == job.id), now); (job, g, chip) }).collect();
            shown.sort_by_key(|(_, g, _)| *g);
            let mut last_group = None;
            for (job, group, chip) in shown {
                if last_group != Some(group) {
                    last_group = Some(group);
                    list = list.push(container(text(group.label()).size(9).font(semibold()).style(muted)).padding(iced::Padding { top: 8.0, right: 0.0, bottom: 2.0, left: 6.0 }));
                }
                visible += 1;
                let selected = self.selected.as_ref() == Some(&job.id);
                let state = job.state.clone();
                let title = short(job.title(), 38);
                let meta = short(job.schedule.display(), 38);
                list = list.push(
                    button(
                        row![
                            text("●")
                                .size(7)
                                .style(move |theme: &Theme| text_widget::Style {
                                    color: Some(state_color(theme, &state))
                                }),
                            column![text(title).size(12), row![text(meta).size(10).style(muted), Space::new().width(Fill), text(chip).size(9).style(move |theme: &Theme| text_widget::Style {
                                    color: Some(match chip { "Failed" | "Overdue" => theme.palette().danger, "Paused" => theme.palette().warning, "Running" | "On" => theme.palette().primary, _ => Color { a: 0.5, ..theme.palette().text } }) })]]
                                .spacing(3)
                        ]
                        .spacing(7)
                        .align_y(Alignment::Center),
                    )
                    .width(Fill)
                    .padding([7, 6])
                    .on_press(Action::Select(job.id.clone()))
                    .style(move |theme, status| list_row(theme, status, selected)),
                );
            }
            list = list.push(
                button(text("+  New routine").size(12))
                    .on_press_maybe((self.store.is_some() && !self.busy).then_some(Action::New))
                    .width(Fill)
                    .padding([8, 6])
                    .style(quiet),
            );
            let detail = if let Some(job) = self
                .jobs
                .iter()
                .find(|job| self.selected.as_ref() == Some(&job.id))
            {
                self.detail(job)
            } else {
                let title = if self.loading {
                    "Loading cron jobs..."
                } else if visible == 0 && !query.is_empty() {
                    "No matches"
                } else {
                    "No scheduled jobs yet"
                };
                container(column![text(title).size(14),text(if query.is_empty(){"Schedule a prompt to run on a cron expression. Hermes will run it and deliver results to the destination you pick."}else{"Try a broader search query."}).size(12).style(muted)].spacing(7).max_width(400)).center_x(Fill).center_y(Fill).padding(24).height(Fill).into()
            };
            row![scrollable(list).width(208).height(Fill), detail]
                .spacing(20)
                .height(Fill)
                .into()
        };
        panel = panel.push(body).push(
            text("Runs while Hermes is open · This desktop · Unattended tool requests are denied")
                .size(10)
                .style(muted),
        );
        container(panel)
            .padding(20)
            .width(Fill)
            .height(Fill)
            .style(|theme: &Theme| container::Style {
                background: Some(theme.palette().background.into()),
                text_color: Some(theme.palette().text),
                border: Border {
                    radius: 8.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            })
            .into()
    }
    fn detail<'a>(&'a self, job: &'a Job) -> Element<'a, Action> {
        let live = self.runs.iter().find(|run| run.status.live());
        let mut actions = row![
            button(text(if job.enabled { "Pause" } else { "Resume" }).size(12))
                .on_press_maybe((!self.busy).then(|| if job.enabled {
                    Action::Pause(job.id.clone())
                } else {
                    Action::Resume(job.id.clone())
                }))
                .style(quiet),
            button(text("Edit cron").size(12))
                .on_press_maybe((!self.busy).then(|| Action::Edit(job.id.clone())))
                .style(quiet),
            button(text("Delete").size(12))
                .on_press_maybe((!self.busy).then(|| Action::Delete(job.id.clone())))
                .style(quiet)
        ]
        .spacing(6);
        actions = if let Some(run) = live {
            actions.push(
                button(text("Stop").size(12))
                    .on_press(Action::Stop(run.id.clone()))
                    .style(button::danger),
            )
        } else {
            actions.push(
                button(
                    text(if job.state == "completed" {
                        "Run again"
                    } else {
                        "Trigger now"
                    })
                    .size(12),
                )
                .on_press_maybe((!self.busy).then(|| Action::Trigger(job.id.clone())))
                .style(quiet),
            )
        };
        let next_label = if job
            .next_run_at
            .as_deref()
            .and_then(|at| cron::instant(at).ok())
            .is_some_and(|at| Utc::now().signed_duration_since(at).num_minutes() > 15)
        {
            "Overdue since"
        } else {
            "Next"
        };
        let mut detail = column![
            row![
                text(job.title()).size(16).font(semibold()).width(Fill),
                text(&job.state)
                    .size(11)
                    .style(move |theme: &Theme| text_widget::Style {
                        color: Some(state_color(theme, &job.state))
                    })
            ]
            .spacing(8),
            actions,
            meta("Frequency", job.schedule.display()),
            meta("Last", job.last_run_at.as_deref().unwrap_or("—")),
            meta(next_label, job.next_run_at.as_deref().unwrap_or("—")),
            meta("Deliver to", "This desktop"),
            meta("Timezone", job.timezone()),
            meta(
                "Model",
                job.model.as_deref().unwrap_or("Default (global model)")
            ),
            text("PROMPT").size(10).style(muted),
            text(&job.prompt).size(13),
            text("RUN HISTORY").size(10).style(muted)
        ]
        .spacing(12);
        if let Some(error) = &job.last_error {
            detail = detail.push(text(format!("Last run failed: {error}")).size(12).style(
                |theme: &Theme| text_widget::Style {
                    color: Some(theme.palette().danger),
                },
            ));
        }
        if self.runs.is_empty() {
            detail = detail.push(text("No runs yet").size(12).style(muted));
        }
        for run in &self.runs {
            detail = detail.push(
                button(
                    row![
                        text(format!("{:?}", run.status).to_lowercase()).size(11),
                        text(&run.claimed_at).size(11).style(muted),
                        Space::new().width(Fill),
                        text(if run.manual { "manual" } else { "scheduled" })
                            .size(10)
                            .style(muted)
                    ]
                    .spacing(10),
                )
                .on_press(Action::ShowRun(run.id.clone()))
                .padding([6, 4])
                .width(Fill)
                .style(quiet),
            );
            if let Some(active) = self.active.get(&run.id) {
                if let Some(approval) = &active.approval {
                    detail = detail.push(
                        container(
                            column![
                                text("Approval required").size(13).font(semibold()),
                                text(&approval.call.name).size(12),
                                text(&approval.call.arguments)
                                    .size(11)
                                    .font(iced::Font::MONOSPACE),
                                row![
                                    button(text("Deny").size(12))
                                        .on_press(Action::Approve(run.id.clone(), false))
                                        .style(quiet),
                                    button(text("Allow once").size(12))
                                        .on_press(Action::Approve(run.id.clone(), true))
                                ]
                                .spacing(8)
                            ]
                            .spacing(8),
                        )
                        .padding(12)
                        .style(container::bordered_box),
                    );
                }
                if !active.text.is_empty() {
                    detail = detail.push(text(&active.text).size(12));
                }
            }
            if self.shown_run.as_ref() == Some(&run.id) {
                if let Some(error) = &run.error {
                    detail = detail.push(text(error).size(12));
                }
                if !run.output.is_empty() {
                    detail = detail.push(text(&run.output).size(12));
                }
            }
        }
        scrollable(detail).width(Fill).height(Fill).into()
    }
    fn editor_view<'a>(&'a self, editor: &'a Editor) -> Element<'a, Action> {
        let label = if editor.id.is_some() {
            "Edit cron job"
        } else {
            "New cron job"
        };
        let catch_up = match editor.draft.catch_up {
            None => CatchUp::Default,
            Some(true) => CatchUp::Run,
            Some(false) => CatchUp::Skip,
        };
        let mut fields=column![text(label).size(15).font(semibold()),text(if editor.id.is_some(){"Update the schedule, prompt, or delivery target. Changes apply on next run."}else{"Schedule a prompt to run automatically. Use cron syntax or a natural phrase like every 15 minutes."}).size(12).style(muted),
            text("Name").size(12),text_input("Morning briefing",&editor.draft.name).on_input(Action::Name).size(12).padding(8),
            text("Prompt").size(12),text_editor(&editor.prompt).on_action(Action::Prompt).height(120).padding(8),
            text("Frequency").size(12),pick_list(Frequency::ALL,Some(editor.frequency),Action::Frequency).text_size(12).padding(8),
        ].spacing(9).max_width(600);
        if editor.frequency == Frequency::Custom {
            fields = fields
                .push(text("Custom schedule").size(12))
                .push(
                    text_input("0 9 * * * or weekdays at 9am", &editor.draft.schedule)
                        .on_input(Action::Schedule)
                        .size(12)
                        .padding(8),
                )
                .push(
                    text("Cron expression, or phrases like every hour or weekdays at 9am.")
                        .size(11)
                        .style(muted),
                );
        }
        fields = fields
            .push(text("Timezone").size(12))
            .push(
                text_input("UTC or Australia/Sydney", &editor.draft.timezone)
                    .on_input(Action::Timezone)
                    .size(12)
                    .padding(8),
            )
            .push(text("Deliver to").size(12))
            .push(checkbox(true).label("This desktop"))
            .push(text("Model").size(12))
            .push(
                text_input("Default (global model)", &editor.model)
                    .on_input(Action::Model)
                    .size(12)
                    .padding(8),
            )
            .push(text("If a run is missed").size(12))
            .push(
                pick_list(
                    [CatchUp::Default, CatchUp::Run, CatchUp::Skip],
                    Some(catch_up),
                    Action::CatchUp,
                )
                .text_size(12)
                .padding(8),
            )
            .push(
                row![
                    Space::new().width(Fill),
                    button(text("Cancel").size(12))
                        .on_press_maybe((!self.busy).then_some(Action::EditorClose))
                        .style(quiet),
                    button(
                        text(if self.busy {
                            "Saving..."
                        } else if editor.id.is_some() {
                            "Save changes"
                        } else {
                            "Create cron"
                        })
                        .size(12)
                    )
                    .on_press_maybe((!self.busy).then_some(Action::Save))
                ]
                .spacing(8),
            );
        scrollable(
            container(fields)
                .width(Fill)
                .center_x(Fill)
                .padding([4, 16]),
        )
        .height(Fill)
        .into()
    }
    pub fn has_running_jobs(&self) -> bool {
        !self.active.is_empty()
    }
    pub fn stop_all(&mut self) {
        for active in self.active.values_mut() {
            active.cancel.cancel();
            if let Some(approval) = active.approval.take() {
                let _ = approval.answer.try_send(false);
            }
        }
    }
}
fn semibold() -> iced::Font {
    iced::Font {
        weight: iced::font::Weight::Semibold,
        ..Default::default()
    }
}
fn short(value: &str, max: usize) -> String {
    let mut result: String = value.chars().take(max).collect();
    if value.chars().count() > max {
        result.push('…');
    }
    result
}
fn muted(theme: &Theme) -> text_widget::Style {
    text_widget::Style {
        color: Some(ui::palette(theme.palette().background.r < 0.5).muted),
    }
}
fn state_color(theme: &Theme, state: &str) -> Color {
    match state {
        "paused" => theme.palette().warning,
        "error" => theme.palette().danger,
        "running" | "scheduled" | "enabled" => theme.palette().primary,
        _ => Color {
            a: 0.36,
            ..theme.palette().text
        },
    }
}
fn quiet(theme: &Theme, status: button::Status) -> button::Style {
    list_row(theme, status, false)
}
fn list_row(theme: &Theme, status: button::Status, selected: bool) -> button::Style {
    button::Style {
        background: if selected {
            Some(
                Color {
                    a: 0.10,
                    ..theme.palette().primary
                }
                .into(),
            )
        } else if matches!(status, button::Status::Hovered | button::Status::Pressed) {
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
        text_color: Color {
            a: if status == button::Status::Disabled {
                0.36
            } else {
                0.85
            },
            ..theme.palette().text
        },
        ..Default::default()
    }
}
fn meta<'a>(label: &'a str, value: &'a str) -> Element<'a, Action> {
    row![
        text(label).size(12).style(muted).width(100),
        text(value).size(12)
    ]
    .spacing(8)
    .into()
}
