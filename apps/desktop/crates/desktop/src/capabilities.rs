//! Capabilities page: port of app/capabilities (Skills · Plugins · Connectors · Tools).
//! The skills catalog is the published snapshot (`docs/api/skills.json`, ~104k rows),
//! decoded off the UI thread on first open. Installing, removing and switching a skill
//! writes the profile's skill library, which the agent reads on its next tool call.
use super::{Hermes, Message, ui};
use crate::appearance::text;
use hermes_engine::{connector, skill_catalog::{self, Entry}, tools, settings::ProfileSettings};
use iced::{Alignment, Border, Color, ContentFit, Element, Fill, Length, Task,
    widget::{Space, button, column, container, image, pick_list, row, rule, stack, text_input, toggler}};
use serde_json::{Value, json};
use std::{collections::BTreeSet, sync::LazyLock};

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Page { #[default] Skills, Plugins, Connectors, Toolsets }

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum Sort { #[default] Featured, Name, Installed }
impl std::fmt::Display for Sort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self { Sort::Featured => "Featured", Sort::Name => "Alphabetical", Sort::Installed => "Installed first" })
    }
}

#[derive(Clone)]
pub(super) enum Action {
    Show(Page), Search(String), Loaded(Result<usize, String>),
    Source(Facet), Category(Facet), InstalledOnly(bool), SortBy(Sort), ClearFilters, More,
    Hero(isize), Open(String), CloseDetail, Install(String), Installed(String, Result<skill_catalog::Status, String>),
    Enable(String, bool), Remove(String), EnableAll(bool),
    PluginCategory(Facet), Toolset(&'static str, bool),
    PluginInstall(String), PluginInstalled(String, Result<String, String>), PluginEnable(String, bool), PluginRemove(String), PluginsInstalledOnly(bool),
    PluginOpen(String), PluginClose, PluginField(String, String), PluginSaveConfig, PluginSignIn(String), PluginSignedIn(String, Result<usize, String>),
    DiscoverSkills, SkillsReady(u64, Result<Value, String>),
    Url(String), Token(String), Save(bool), DiscoverTools, ToolsReady(u64, Result<Vec<String>, String>),
}

/// A filter value shown in a pick list; `None` is the "All …" row.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Facet { pub value: Option<String>, label: String }
impl std::fmt::Display for Facet { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.label) } }
fn all(label: &str) -> Facet { Facet { value: None, label: label.into() } }
fn facet(value: &str, count: usize) -> Facet { Facet { value: Some(value.into()), label: format!("{value} · {count}") } }

const PAGE: usize = 60;
const SHELF: usize = 6;

#[derive(Default)]
pub(super) struct State {
    page: Page, search: String, loaded: bool, loading: bool, error: Option<String>, notice: Option<String>,
    source: Option<String>, category: Option<String>, installed_only: bool, sort: Sort, limit: usize, hero: usize,
    status: skill_catalog::Status, installing: BTreeSet<String>, detail: Option<String>, document: Option<String>,
    sources: Vec<Facet>, categories: Vec<Facet>, plugin_category: Option<String>,
    plugins_installed: Vec<hermes_engine::plugins::Installed>, plugin_busy: BTreeSet<String>, plugins_installed_only: bool,
    plugin_detail: Option<String>, plugin_form: std::collections::BTreeMap<String, String>,
    local: Vec<Value>, url: String, token: String, has_token: bool, environment: bool, tools: Vec<String>, busy: bool, revision: u64,
}

// ---------------------------------------------------------------- plugins snapshot
struct Plugin { name: String, description: String, category: String, url: String, title: String, repo: String, sha: String, subdir: String }
static PLUGINS: LazyLock<Vec<Plugin>> = LazyLock::new(|| {
    let rows: Vec<Vec<Value>> = hermes_engine::skill_catalog::plugin_rows();
    let s = |v: Option<&Value>| v.and_then(Value::as_str).unwrap_or_default().to_owned();
    rows.into_iter().map(|r| Plugin { name: s(r.first()), description: s(r.get(1)), category: s(r.get(2)), url: s(r.get(3)), title: s(r.get(4)), repo: s(r.get(5)), sha: s(r.get(6)), subdir: s(r.get(7)) }).collect()
});

// ---------------------------------------------------------------- artwork
const ART_ONE: &[u8] = include_bytes!("../assets/catalog/official-art-card-one.jpg");
const ART_FEATURE: &[u8] = include_bytes!("../assets/catalog/official-art-feature.jpg");
const ART_MAIN: &[u8] = include_bytes!("../assets/catalog/official-art.jpg");
const ART_TWO: &[u8] = include_bytes!("../assets/catalog/official-art-card-two.jpg");
/// `catalog-curation.json` artwork, keyed by `category/name` of the built-in skill.
fn artwork(entry: &Entry) -> Option<image::Handle> {
    // Pre-cropped to the banner's aspect: the renderer does not clip an
    // overflowing `ContentFit::Cover` image to its container.
    static HANDLES: LazyLock<[image::Handle; 4]> = LazyLock::new(|| [ART_ONE, ART_FEATURE, ART_MAIN, ART_TWO].map(|bytes| crop(bytes, HERO_ASPECT)));
    let key = entry.id.strip_prefix("NousResearch/hermes-agent/skills/")?;
    Some(HANDLES[match key {
        "autonomous-ai-agents/hermes-agent" => 0, "autonomous-ai-agents/computer-use" => 1,
        "autonomous-ai-agents/codex" => 2, "autonomous-ai-agents/claude-code" => 3, _ => return None,
    }].clone())
}

const HERO_ASPECT: f32 = 3.6;
fn crop(bytes: &'static [u8], aspect: f32) -> image::Handle {
    let Ok(decoded) = ::image::load_from_memory(bytes) else { return image::Handle::from_bytes(bytes) };
    let (w, h) = (decoded.width(), decoded.height());
    let (cw, ch) = if w as f32 / h as f32 > aspect { ((h as f32 * aspect) as u32, h) } else { (w, (w as f32 / aspect) as u32) };
    let rgba = decoded.crop_imm((w - cw) / 2, (h - ch) / 2, cw, ch).to_rgba8();
    image::Handle::from_rgba(cw, ch, rgba.into_raw())
}

// ---------------------------------------------------------------- update
pub(super) fn open(app: &mut Hermes) -> Task<Message> {
    let root = app.library.as_ref().map(|lib| lib.root().to_owned());
    let state = &mut app.capabilities;
    state.detail = None; state.document = None; state.error = None; state.token.clear(); state.limit = PAGE;
    if let Some(root) = &root {
        match skill_catalog::status(root) { Ok(status) => state.status = status, Err(e) => state.error = Some(format!("Could not read installed skills: {e:#}")) }
        match connector::summary(root) {
            Ok(info) => { state.has_token = info.has_token; state.environment = info.environment; state.url = info.url; }
            Err(e) => state.error = Some(e.to_string()),
        }
    }
    let page = state.page;
    let mut tasks = vec![update(app, Action::Show(page))];
    if !app.capabilities.loaded && !app.capabilities.loading {
        app.capabilities.loading = true;
        tasks.push(Task::perform(async { tokio::task::spawn_blocking(|| skill_catalog::entries().len()).await.map_err(|e| e.to_string()) },
            |r| Message::Capability(Action::Loaded(r))));
    }
    Task::batch(tasks)
}

fn reset_paging(state: &mut State) { state.limit = PAGE; state.detail = None; state.document = None; }

pub(super) fn update(app: &mut Hermes, action: Action) -> Task<Message> {
    let root = app.library.as_ref().map(|lib| lib.root().to_owned());
    let workspace = matches!(action, Action::DiscoverSkills).then(|| app.workspace()).flatten();
    let state = &mut app.capabilities;
    match action {
        Action::Show(page) => {
            state.revision = state.revision.wrapping_add(1); state.busy = false; state.page = page; state.error = None; state.notice = None;
            reset_paging(state);
            if page == Page::Skills { return update(app, Action::DiscoverSkills); }
            if page == Page::Plugins && let Some(root) = &root { state.plugins_installed = hermes_engine::plugins::installed(root).unwrap_or_default(); }
        }
        Action::Search(s) => { state.search = s; reset_paging(state); }
        Action::Loaded(result) => {
            state.loading = false;
            match result {
                Ok(_) => {
                    state.loaded = true;
                    let mut sources = std::collections::BTreeMap::<&str, usize>::new();
                    let mut categories = std::collections::BTreeMap::<&str, usize>::new();
                    for e in skill_catalog::entries() { *sources.entry(&e.source).or_default() += 1; *categories.entry(&e.category).or_default() += 1; }
                    let mut sources: Vec<_> = sources.into_iter().collect(); sources.sort_by(|a, b| b.1.cmp(&a.1));
                    let mut categories: Vec<_> = categories.into_iter().collect(); categories.sort_by(|a, b| b.1.cmp(&a.1));
                    let official = sources.iter().filter(|(s, _)| skill_catalog::is_official(s)).map(|(_, n)| n).sum();
                    state.sources = std::iter::once(all("All sources")).chain(std::iter::once(facet("Official", official)))
                        .chain(sources.iter().map(|(s, n)| facet(s, *n))).collect();
                    state.categories = std::iter::once(all("All categories")).chain(categories.iter().map(|(c, n)| facet(c, *n))).collect();
                }
                Err(e) => state.error = Some(format!("Could not load the skills catalog: {e}")),
            }
        }
        Action::Source(f) => { state.source = f.value; reset_paging(state); }
        Action::Category(f) => { state.category = f.value; reset_paging(state); }
        Action::InstalledOnly(on) => { state.installed_only = on; reset_paging(state); }
        Action::SortBy(sort) => { state.sort = sort; reset_paging(state); }
        Action::ClearFilters => { state.source = None; state.category = None; state.installed_only = false; state.search.clear(); state.sort = Sort::Featured; reset_paging(state); }
        Action::More => state.limit += PAGE,
        Action::Hero(step) => {
            let n = featured().len().max(1) as isize;
            state.hero = (((state.hero as isize + step) % n + n) % n) as usize;
        }
        Action::Open(id) => {
            state.document = root.as_ref().and_then(|root| skill_catalog::document(root, &id).ok().flatten());
            state.detail = Some(id);
        }
        Action::CloseDetail => { state.detail = None; state.document = None; }
        Action::Install(id) => {
            let Some(root) = root else { return Task::none() };
            if !state.installing.insert(id.clone()) { return Task::none(); }
            state.notice = None; state.error = None;
            let key = id.clone();
            return Task::perform(async move { skill_catalog::install(&root, &key).await.map_err(|e| format!("{e:#}")) },
                move |r| Message::Capability(Action::Installed(id.clone(), r)));
        }
        Action::Installed(id, result) => {
            state.installing.remove(&id);
            let name = id.rsplit('/').next().unwrap_or(&id).to_owned();
            match result {
                Ok(status) => { state.status = status; state.notice = Some(format!("Installed {name}. Hermes can use it in your next message."));
                    if state.detail.as_deref() == Some(id.as_str()) { state.document = root.as_ref().and_then(|r| skill_catalog::document(r, &id).ok().flatten()); } }
                Err(e) => state.error = Some(format!("Could not install {name}: {e}")),
            }
        }
        Action::Enable(id, on) => {
            let Some(root) = root else { return Task::none() };
            match skill_catalog::enable(&root, &id, on) { Ok(status) => state.status = status, Err(e) => state.error = Some(format!("{e:#}")) }
        }
        Action::EnableAll(on) => {
            let Some(root) = root else { return Task::none() };
            let ids: Vec<String> = state.status.installed.iter().filter(|id| state.status.disabled.contains(*id) == on).cloned().collect();
            for id in ids { match skill_catalog::enable(&root, &id, on) { Ok(status) => state.status = status, Err(e) => { state.error = Some(format!("{e:#}")); break; } } }
        }
        Action::Remove(id) => {
            let Some(root) = root else { return Task::none() };
            match skill_catalog::remove(&root, &id) { Ok(status) => { state.status = status; state.document = None; } Err(e) => state.error = Some(format!("{e:#}")) }
        }
        Action::PluginCategory(f) => { state.plugin_category = f.value; state.limit = PAGE; }
        Action::PluginsInstalledOnly(on) => { state.plugins_installed_only = on; state.limit = PAGE; }
        Action::PluginOpen(name) => if let Some(root) = &root {
            let values = hermes_engine::plugins::config(root, &name);
            state.plugin_form = hermes_engine::plugins::config_schema(root, &name).into_iter()
                .map(|f| { let v = values.get(&f.key).map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned)).unwrap_or(f.default); (f.key, v) }).collect();
            state.plugin_detail = Some(name); state.error = None; state.notice = None;
        },
        Action::PluginClose => { state.plugin_detail = None; state.plugin_form.clear(); }
        Action::PluginField(key, value) => { state.plugin_form.insert(key, value); }
        Action::PluginSaveConfig => if let (Some(root), Some(name)) = (&root, &state.plugin_detail) {
            let values = state.plugin_form.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect();
            match hermes_engine::plugins::set_config(root, name, &values) { Ok(()) => state.notice = Some("Plugin settings saved. They apply from the next message.".into()), Err(e) => state.error = Some(format!("{e:#}")) }
        },
        Action::PluginSignIn(name) => {
            let Some(root) = root else { return Task::none() };
            if !state.plugin_busy.insert(name.clone()) { return Task::none(); }
            state.notice = Some("Finish signing in in your browser…".into()); state.error = None;
            let target = name.clone();
            return Task::perform(async move {
                hermes_engine::plugins::sign_in(&root, &target, |url: &str| { let _ = super::open_url(url); }).await.map_err(|e| format!("{e:#}"))
            }, move |r| pressed(Action::PluginSignedIn(name.clone(), r)));
        }
        Action::PluginSignedIn(name, result) => {
            state.plugin_busy.remove(&name);
            match result { Ok(n) => { state.notice = Some(format!("Signed in to {n} server{}.", if n == 1 { "" } else { "s" })); state.error = None; } Err(e) => { state.notice = None; state.error = Some(e); } }
        }
        Action::PluginInstall(name) => {
            let (Some(root), Some(plugin)) = (root, PLUGINS.iter().find(|p| p.name == name)) else { return Task::none() };
            if !state.plugin_busy.insert(name.clone()) { return Task::none(); }
            state.error = None; state.notice = Some(format!("Installing {}…", display(plugin)));
            let entry = hermes_engine::plugins::CatalogEntry { name: plugin.name.clone(), title: plugin.title.clone(), repo: plugin.repo.clone(), sha: plugin.sha.clone(), subdir: plugin.subdir.clone() };
            return Task::perform(async move {
                tokio::task::spawn_blocking(move || hermes_engine::plugins::install(&root, &entry).map(|p| {
                    let parts = [(!p.servers.is_empty()).then(|| format!("{} MCP server{}", p.servers.len(), if p.servers.len() == 1 { "" } else { "s" })),
                        (!p.skills.is_empty()).then(|| format!("{} skill{}", p.skills.len(), if p.skills.len() == 1 { "" } else { "s" }))];
                    let parts: Vec<String> = parts.into_iter().flatten().collect();
                    format!("Installed {} at {}{}", p.title, &p.sha[..8], if parts.is_empty() { String::new() } else { format!(" · {}", parts.join(" · ")) })
                }).map_err(|e| format!("{e:#}"))).await.unwrap_or_else(|e| Err(e.to_string()))
            }, move |result| pressed(Action::PluginInstalled(name.clone(), result)));
        }
        Action::PluginInstalled(name, result) => {
            state.plugin_busy.remove(&name);
            match result { Ok(note) => { state.notice = Some(note); state.error = None; } Err(e) => { state.notice = None; state.error = Some(e); } }
            if let Some(root) = &root { state.plugins_installed = hermes_engine::plugins::installed(root).unwrap_or_default(); }
        }
        Action::PluginEnable(name, on) => if let Some(root) = &root {
            if let Err(e) = hermes_engine::plugins::set_enabled(root, &name, on) { state.error = Some(format!("{e:#}")); }
            state.plugins_installed = hermes_engine::plugins::installed(root).unwrap_or_default();
        },
        Action::PluginRemove(name) => if let Some(root) = &root {
            match hermes_engine::plugins::uninstall(root, &name) { Ok(()) => state.notice = Some("Plugin removed.".into()), Err(e) => state.error = Some(format!("{e:#}")) }
            state.plugins_installed = hermes_engine::plugins::installed(root).unwrap_or_default();
        },
        Action::Toolset(key, on) => {
            let Some(root) = root else { return Task::none() };
            let mut changes = serde_json::Map::new(); changes.insert(key.into(), json!(on));
            match ProfileSettings::update(&root, &changes) { Ok(saved) => app.profile_settings.saved = saved, Err(e) => state.error = Some(format!("{e:#}")) }
        }
        Action::DiscoverSkills => {
            let Some(path) = workspace else { state.local.clear(); return Task::none() };
            state.revision = state.revision.wrapping_add(1); let rev = state.revision;
            return Task::perform(async move { tokio::task::spawn_blocking(move || {
                let ctx = tools::Context::new(path)?; tools::invoke("skills_list", &json!({}), &ctx)
            }).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string()) }, move |r| Message::Capability(Action::SkillsReady(rev, r)));
        }
        Action::SkillsReady(rev, result) if rev == state.revision => {
            if let Ok(value) = result { state.local = value["skills"].as_array().cloned().unwrap_or_default(); }
        }
        Action::Url(s) => { state.url = s; state.tools.clear(); state.revision = state.revision.wrapping_add(1); state.busy = false; }
        Action::Token(s) => { state.token = s; state.revision = state.revision.wrapping_add(1); state.busy = false; }
        Action::Save(clear) => {
            let Some(root) = root else { return Task::none() };
            let url = if clear { "" } else { &state.url };
            match connector::save(&root, url, &state.token, clear) {
                Ok(info) => { state.url = info.url; state.token.clear(); state.environment = info.environment; state.has_token = info.has_token; state.tools.clear(); state.error = None;
                    state.notice = Some(if clear { "Disconnected the MCP server.".into() } else { "Saved. New sessions will offer this server's tools.".into() }); }
                Err(e) => state.error = Some(e.to_string()),
            }
            state.revision = state.revision.wrapping_add(1); state.busy = false;
        }
        Action::DiscoverTools => {
            let Some(root) = root else { return Task::none() };
            state.revision = state.revision.wrapping_add(1); let rev = state.revision; state.busy = true; state.error = None;
            return Task::perform(async move { connector::discover(&root).await.map_err(|e| e.to_string()) }, move |r| Message::Capability(Action::ToolsReady(rev, r)));
        }
        Action::ToolsReady(rev, r) if rev == state.revision => { state.busy = false; match r { Ok(tools) => state.tools = tools, Err(e) => state.error = Some(e) } }
        _ => {}
    }
    Task::none()
}

// ---------------------------------------------------------------- queries
fn featured() -> Vec<&'static Entry> {
    let mut picks: Vec<_> = skill_catalog::entries().iter().filter(|e| e.featured > 0).collect();
    picks.sort_by_key(|e| e.featured); picks
}
fn installed(state: &State, e: &Entry) -> bool { state.status.installed.contains(&e.id) }
fn enabled(state: &State, e: &Entry) -> bool { installed(state, e) && !state.status.disabled.contains(&e.id) }
fn source_label(source: &str) -> &str { match source { "built-in" => "Built-in", "optional" => "Official", other => other } }

/// Discover view = no search, no facet. Anything else is a flat filtered grid.
fn narrowed(state: &State) -> bool {
    !state.search.trim().is_empty() || state.source.is_some() || state.category.is_some() || state.installed_only || state.sort == Sort::Installed
}
fn filtered(state: &State) -> Vec<&'static Entry> {
    let terms: Vec<String> = state.search.to_lowercase().split_whitespace().map(str::to_owned).collect();
    let mut rows: Vec<&Entry> = skill_catalog::entries().iter().filter(|e| {
        (match state.source.as_deref() { None => true, Some("Official") => skill_catalog::is_official(&e.source), Some(s) => e.source == s })
            && state.category.as_deref().is_none_or(|c| e.category == c)
            && (!state.installed_only || installed(state, e))
            && terms.iter().all(|t| e.name.to_lowercase().contains(t) || e.description.to_lowercase().contains(t) || e.category.to_lowercase().contains(t) || e.author.to_lowercase().contains(t))
    }).collect();
    let rank = |e: &Entry| (e.featured == 0, !skill_catalog::is_official(&e.source));
    match state.sort {
        Sort::Featured => rows.sort_by_key(|e| rank(e)),
        Sort::Name => rows.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase())),
        Sort::Installed => rows.sort_by_key(|e| (!installed(state, e), rank(e))),
    }
    rows
}

// ---------------------------------------------------------------- view
fn pressed(action: Action) -> Message { Message::Capability(action) }

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let state = &app.capabilities; let p = ui::palette(app.dark);
    let skills_count = state.status.installed.len();
    let tools_count = tools::tool_specs().iter().filter(|s| app.profile_settings.saved.tool_enabled(s["function"]["name"].as_str().unwrap_or_default())).count();
    let mut tabs = row![].spacing(4).align_y(Alignment::Center);
    for (page, name, meta) in [(Page::Skills, "Skills", Some(skills_count)), (Page::Plugins, "Plugins", None), (Page::Connectors, "Connectors", None), (Page::Toolsets, "Tools", Some(tools_count))] {
        let active = state.page == page;
        let mut label = row![text(name).size(13).font(ui::medium()).color(if active { p.foreground } else { p.muted })].spacing(6).align_y(Alignment::Center);
        if let Some(n) = meta { label = label.push(text(n.to_string()).size(11).color(p.muted)); }
        tabs = tabs.push(button(label).padding([5, 11]).on_press(pressed(Action::Show(page)))
            .style(move |_, status| button::Style { background: (active || status == button::Status::Hovered).then_some(Color { a: if active { 0.09 } else { 0.05 }, ..p.foreground }.into()),
                text_color: p.foreground, border: Border { radius: 6.0.into(), ..Default::default() }, ..Default::default() }));
    }
    // PageHeader: the H1 in the display face, then the tab bar under it.
    let mut body = column![text("Capabilities").size(40).font(crate::appearance::DISPLAY_FONT).line_height(1.1), tabs].spacing(20);
    if let Some(err) = &state.error { body = body.push(banner(err, p.accent, p)); }
    if let Some(note) = &state.notice { body = body.push(banner(note, p.muted, p)); }
    body = body.push(match state.page {
        Page::Skills => skills(app),
        Page::Plugins => plugins(app),
        Page::Connectors => connectors(app),
        Page::Toolsets => toolsets(app),
    });
    super::settings::quiet_scroll(container(body).width(Fill).max_width(1080).padding(iced::Padding { top: 60.0, right: 36.0, bottom: 32.0, left: 36.0 })).height(Fill).into()
}

fn banner<'a>(message: &'a str, accent: Color, p: ui::Palette) -> Element<'a, Message> {
    container(text(message).size(12).color(p.foreground)).padding([8, 12]).width(Fill)
        .style(move |_| ui::panel(Color { a: 0.07, ..accent }, 6.0, Some(Color { a: 0.25, ..accent }))).into()
}

fn search_bar<'a>(placeholder: &'a str, value: &'a str, p: ui::Palette, trailing: Element<'a, Message>) -> Element<'a, Message> {
    container(row![ui::icon("search", 14, p.muted),
        text_input(placeholder, value).on_input(|s| pressed(Action::Search(s))).size(14).padding([6, 0])
            .style(move |_, _| text_input::Style { background: Color::TRANSPARENT.into(), border: Border::default(), icon: p.muted, placeholder: p.muted, value: p.foreground, selection: Color { a: 0.25, ..p.accent } }),
        trailing].spacing(8).align_y(Alignment::Center))
        .padding([4, 12]).width(Fill).style(move |_| ui::panel(p.card, 8.0, Some(p.border))).into()
}

fn skills(app: &Hermes) -> Element<'_, Message> {
    let state = &app.capabilities; let p = ui::palette(app.dark);
    if let Some(id) = &state.detail { return detail(app, id); }
    let all_on = !state.status.installed.is_empty() && state.status.disabled.is_empty();
    let bulk = row![text("All").size(12).color(p.muted), toggler(all_on).size(15).on_toggle(|on| pressed(Action::EnableAll(on)))].spacing(6).align_y(Alignment::Center);
    let mut col = column![search_bar("Search skills…", &state.search, p, bulk.into())].spacing(14);
    if !state.loaded {
        return col.push(container(text(if state.loading { "Loading the skills catalog…" } else { "The skills catalog is unavailable." }).size(13).color(p.muted)).padding(30).center_x(Fill)).into();
    }
    let sort_options = [Sort::Featured, Sort::Name, Sort::Installed];
    let selected_source = state.sources.iter().find(|f| f.value.as_deref() == state.source.as_deref() || (f.value.is_none() && state.source.is_none())).cloned();
    let selected_category = state.categories.iter().find(|f| f.value == state.category).cloned();
    let filters = row![
        pick_list(state.sources.clone(), selected_source, |f| pressed(Action::Source(f))).text_size(12).padding([5, 9]),
        pick_list(state.categories.clone(), selected_category, |f| pressed(Action::Category(f))).text_size(12).padding([5, 9]),
        row![toggler(state.installed_only).size(15).on_toggle(|on| pressed(Action::InstalledOnly(on))), text("Installed").size(12)].spacing(6).align_y(Alignment::Center),
        Space::new().width(Fill),
        pick_list(sort_options, Some(state.sort), |s| pressed(Action::SortBy(s))).text_size(12).padding([5, 9]),
    ].spacing(10).align_y(Alignment::Center);
    col = col.push(filters);
    if narrowed(state) {
        let rows = filtered(state);
        if rows.is_empty() {
            return col.push(column![text("No skills match").size(15), text("Try another search or clear the filters.").size(12).color(p.muted),
                button(text("Clear filters").size(12)).on_press(pressed(Action::ClearFilters))].spacing(8).align_x(Alignment::Center).width(Fill).padding(30)).into();
        }
        col = col.push(text(format!("{} skills", fmt_count(rows.len()))).size(12).color(p.muted));
        col = col.push(grid(app, rows.iter().take(state.limit).copied().collect()));
        if rows.len() > state.limit { col = col.push(container(button(text("More").size(12)).on_press(pressed(Action::More)).style(ui::quiet_button)).center_x(Fill)); }
        return col.into();
    }
    // Discover: featured carousel, project skills, installed, then official shelves.
    let picks = featured();
    if !picks.is_empty() {
        col = col.push(text("Discover Skills").size(17).font(ui::medium())).push(hero(app, picks[state.hero % picks.len()], picks.len()));
    }
    if !state.local.is_empty() {
        let mut list = column![section("In this project", "Skills found in the open project folder.", None, p)].spacing(8);
        for skill in &state.local {
            list = list.push(container(column![text(skill["name"].as_str().unwrap_or_default().to_owned()).size(14).font(ui::medium()),
                text(skill["description"].as_str().unwrap_or_default().to_owned()).size(12).color(p.muted)].spacing(3)).padding(12).width(Fill)
                .style(move |_| ui::panel(p.card, 10.0, Some(p.border))));
        }
        col = col.push(list);
    }
    let installed_rows: Vec<&Entry> = skill_catalog::entries().iter().filter(|e| installed(state, e) && e.featured == 0).collect();
    if !installed_rows.is_empty() {
        col = col.push(section("Installed", "Skills Hermes can use now. Switch one off to hide it from the agent.", Some(Action::InstalledOnly(true)), p))
            .push(grid(app, installed_rows.into_iter().take(SHELF).collect()));
    }
    let mut shelves: Vec<(&str, Vec<&Entry>)> = Vec::new();
    for e in skill_catalog::entries().iter().filter(|e| e.source == "optional") {
        match shelves.iter_mut().find(|(c, _)| *c == e.category) { Some((_, v)) => v.push(e), None => shelves.push((&e.category, vec![e])) }
    }
    shelves.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    for (category, items) in shelves.into_iter().take(6) {
        let facet = Facet { value: Some(category.to_owned()), label: category.to_owned() };
        col = col.push(section(category, &format!("Official skills for {}.", category.to_lowercase()), Some(Action::Category(facet)), p))
            .push(grid(app, items.into_iter().take(SHELF).collect()));
    }
    col = col.push(container(button(text(format!("Browse all {} skills", fmt_count(skill_catalog::entries().len()))).size(13)).padding([8, 14])
        .on_press(pressed(Action::SortBy(Sort::Name)))).center_x(Fill));
    col.into()
}

fn fmt_count(n: usize) -> String {
    let s = n.to_string(); let mut out = String::new();
    for (i, c) in s.chars().enumerate() { if i > 0 && (s.len() - i) % 3 == 0 { out.push(','); } out.push(c); }
    out
}

fn section<'a>(title: &str, blurb: &str, see_all: Option<Action>, p: ui::Palette) -> Element<'a, Message> {
    let mut head = row![column![text(title.to_owned()).size(16).font(ui::medium()), text(blurb.to_owned()).size(12).color(p.muted)].spacing(2), Space::new().width(Fill)].align_y(Alignment::End);
    if let Some(action) = see_all {
        head = head.push(button(row![text("See all").size(12).color(p.accent), ui::icon("chevron-right", 11, p.accent)].spacing(3).align_y(Alignment::Center))
            .on_press(pressed(action)).style(ui::quiet_button));
    }
    container(head).padding(iced::Padding { top: 10.0, ..Default::default() }).into()
}

fn hero(app: &Hermes, entry: &'static Entry, count: usize) -> Element<'static, Message> {
    let state = &app.capabilities; let p = ui::palette(app.dark);
    let backdrop: Element<'static, Message> = match artwork(entry) {
        Some(handle) => image(handle).content_fit(ContentFit::Fill).width(Fill).height(260).into(),
        None => container(Space::new().width(Fill).height(260)).style(move |_| ui::panel(Color { a: 0.85, ..p.accent }, 0.0, None)).into(),
    };
    let shade = container(Space::new().width(Fill).height(260)).style(|_| container::Style {
        background: Some(iced::Background::Gradient(iced::Gradient::Linear(iced::gradient::Linear::new(0.0)
            .add_stop(0.0, Color { a: 0.78, ..Color::BLACK }).add_stop(0.65, Color::TRANSPARENT)))), ..Default::default() });
    let action: Element<'static, Message> = install_control(state, entry, true, p);
    let info = column![
        text("FEATURED").size(10).font(ui::medium()).color(Color { a: 0.8, ..Color::WHITE }),
        text(entry.name.clone()).size(26).font(ui::medium()).color(Color::WHITE),
        text(clip(&entry.description, 140)).size(13).color(Color { a: 0.9, ..Color::WHITE }),
        row![action, button(text("Details").size(12).color(Color::WHITE)).padding([5, 12]).on_press(pressed(Action::Open(entry.id.clone())))
            .style(|_, _| button::Style { background: Some(Color { a: 0.18, ..Color::WHITE }.into()), text_color: Color::WHITE, border: Border { radius: 6.0.into(), ..Default::default() }, ..Default::default() })].spacing(8),
    ].spacing(6).max_width(520);
    let arrow = |glyph: &'static str, step: isize| button(ui::icon(glyph, 14, Color::WHITE)).padding(6).on_press(pressed(Action::Hero(step)))
        .style(|_, _| button::Style { background: Some(Color { a: 0.25, ..Color::BLACK }.into()), text_color: Color::WHITE, border: Border { radius: 14.0.into(), ..Default::default() }, ..Default::default() });
    let mut dots = row![].spacing(5);
    for i in 0..count { let on = i == state.hero % count; dots = dots.push(container(Space::new().width(if on { 16 } else { 6 }).height(6))
        .style(move |_| ui::panel(Color { a: if on { 1.0 } else { 0.5 }, ..Color::WHITE }, 3.0, None))); }
    let overlay = column![
        row![Space::new().width(Fill), arrow("chevron-left", -1), arrow("chevron-right", 1)].spacing(6),
        Space::new().height(Fill),
        row![info, Space::new().width(Fill), dots].align_y(Alignment::End),
    ].padding(20).height(260);
    container(stack![backdrop, shade, overlay]).width(Fill).height(260).clip(true)
        .style(move |_| container::Style { border: Border { radius: 12.0.into(), width: 1.0, color: p.border }, ..Default::default() }).into()
}

fn clip(value: &str, max: usize) -> String {
    let flat = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max { flat } else { format!("{}…", flat.chars().take(max - 1).collect::<String>().trim_end()) }
}

fn install_control(state: &State, entry: &'static Entry, prominent: bool, p: ui::Palette) -> Element<'static, Message> {
    if installed(state, entry) {
        let on = enabled(state, entry); let id = entry.id.clone();
        return row![text(if on { "On" } else { "Off" }).size(11).color(if prominent { Color::WHITE } else { p.muted }),
            toggler(on).size(15).on_toggle(move |v| pressed(Action::Enable(id.clone(), v)))].spacing(5).align_y(Alignment::Center).into();
    }
    if state.installing.contains(&entry.id) { return text("Installing…").size(12).color(if prominent { Color::WHITE } else { p.muted }).into(); }
    if !skill_catalog::can_install(&entry.id) {
        return button(row![text("Get").size(12), ui::icon("link-external", 11, p.accent)].spacing(4)).padding([4, 10])
            .on_press_maybe((!entry.docs.is_empty()).then(|| Message::Link(entry.docs.clone()))).style(ui::quiet_button).into();
    }
    button(row![ui::icon("add", 12, Color::WHITE), text("Install").size(12).color(Color::WHITE)].spacing(4).align_y(Alignment::Center)).padding([4, 10])
        .on_press(pressed(Action::Install(entry.id.clone())))
        .style(move |_, status| button::Style { background: Some(Color { a: if status == button::Status::Hovered { 0.85 } else { 1.0 }, ..p.accent }.into()),
            text_color: Color::WHITE, border: Border { radius: 6.0.into(), ..Default::default() }, ..Default::default() }).into()
}

fn card(app: &Hermes, entry: &'static Entry) -> Element<'static, Message> {
    let state = &app.capabilities; let p = ui::palette(app.dark);
    let chip = |label: String| container(text(label).size(10).color(p.muted)).padding([2, 6]).style(move |_| ui::panel(p.secondary, 4.0, None));
    let mut tags = row![chip(source_label(&entry.source).to_owned())].spacing(5);
    if !entry.category.is_empty() && entry.category != "Other" { tags = tags.push(chip(entry.category.clone())); }
    let byline = if entry.author.is_empty() { String::new() } else { format!("{}{}", clip(&entry.author, 28), if entry.version.is_empty() { String::new() } else { format!(" · v{}", entry.version) }) };
    let content = column![
        tags,
        text(entry.name.clone()).size(14).font(ui::medium()),
        text(clip(&entry.description, 120)).size(12).color(p.muted).height(Length::Fixed(48.0)),
        row![text(byline).size(11).color(p.muted), Space::new().width(Fill), install_control(state, entry, false, p)].align_y(Alignment::Center),
    ].spacing(7);
    button(content).padding(14).width(Fill).on_press(pressed(Action::Open(entry.id.clone())))
        .style(move |_, status| button::Style { background: Some(p.card.into()), text_color: p.foreground,
            border: Border { radius: 10.0.into(), width: 1.0, color: if status == button::Status::Hovered { Color { a: 0.5, ..p.accent } } else { p.border } }, ..Default::default() }).into()
}

fn grid(app: &Hermes, entries: Vec<&'static Entry>) -> Element<'static, Message> {
    let mut rows = column![].spacing(12);
    for chunk in entries.chunks(3) {
        let mut line = row![].spacing(12);
        for entry in chunk { line = line.push(container(card(app, entry)).width(Fill)); }
        for _ in chunk.len()..3 { line = line.push(Space::new().width(Fill)); }
        rows = rows.push(line);
    }
    rows.into()
}

fn detail<'a>(app: &'a Hermes, id: &'a str) -> Element<'a, Message> {
    let state = &app.capabilities; let p = ui::palette(app.dark);
    let back = button(row![ui::icon("chevron-left", 12, p.muted), text("Skills").size(12).color(p.muted)].spacing(4).align_y(Alignment::Center))
        .on_press(pressed(Action::CloseDetail)).style(ui::quiet_button);
    let Some(entry) = skill_catalog::entries().iter().find(|e| e.id == id) else {
        return column![back, text("This skill is no longer in the catalog.").size(13)].spacing(12).into();
    };
    let mut meta = column![].spacing(4);
    for (label, value) in [("Source", source_label(&entry.source).to_owned()), ("Category", entry.category.clone()), ("Author", entry.author.clone()),
        ("Version", entry.version.clone()), ("Platforms", entry.platforms.join(", "))] {
        if !value.is_empty() { meta = meta.push(row![text(label).size(12).color(p.muted).width(90), text(value).size(12)]); }
    }
    let mut actions = row![install_control(state, entry, false, p)].spacing(8).align_y(Alignment::Center);
    if installed(state, entry) && entry.source != "built-in" {
        actions = actions.push(button(text("Remove").size(12)).padding([4, 10]).on_press(pressed(Action::Remove(entry.id.clone()))).style(ui::quiet_button));
    }
    if !entry.docs.is_empty() {
        actions = actions.push(button(row![text("Documentation").size(12).color(p.accent), ui::icon("link-external", 11, p.accent)].spacing(4))
            .on_press(Message::Link(entry.docs.clone())).style(ui::quiet_button));
    }
    let doc: Element<'a, Message> = match &state.document {
        Some(doc) => container(text(doc.as_str()).size(12).font(iced::Font::MONOSPACE)).padding(14).width(Fill)
            .style(move |_| ui::panel(p.secondary, 8.0, Some(p.border))).into(),
        None => text(if installed(state, entry) { "The skill document is not stored locally." } else { "Install this skill to read its full instructions here." }).size(12).color(p.muted).into(),
    };
    let mut col = column![back];
    if let Some(art) = artwork(entry) { col = col.push(container(image(art).content_fit(ContentFit::Fill).width(Fill).height(220)).clip(true)
        .style(|_| container::Style { border: Border { radius: 10.0.into(), ..Default::default() }, ..Default::default() })); }
    col.push(text(entry.name.clone()).size(24).font(ui::medium()))
        .push(text(entry.description.clone()).size(14).color(p.muted))
        .push(actions).push(rule::horizontal(1)).push(meta).push(doc).spacing(14).into()
}

fn display(plugin: &Plugin) -> String { if plugin.title.is_empty() { plugin.name.clone() } else { plugin.title.clone() } }

fn plugin_detail<'a>(app: &'a Hermes, name: &'a str) -> Element<'a, Message> {
    let state = &app.capabilities; let p = ui::palette(app.dark);
    let root = app.library.as_ref().map(|l| l.root().to_owned()).unwrap_or_default();
    let Some(record) = state.plugins_installed.iter().find(|i| i.name == name) else { return Space::new().into() };
    let catalog = PLUGINS.iter().find(|p| p.name == name);
    let busy = state.plugin_busy.contains(name);
    let mut actions = row![].spacing(8).align_y(Alignment::Center);
    if let Some(entry) = catalog && hermes_engine::plugins::update_available(record, &entry.sha) {
        actions = actions.push(button(text(if busy { "Updating…" } else { "Update" }).size(12)).padding([5, 12]).on_press_maybe((!busy).then(|| pressed(Action::PluginInstall(name.to_owned())))));
    }
    let http = hermes_engine::plugins::http_servers(&root, name).unwrap_or_default();
    if !http.is_empty() {
        let signed = hermes_engine::plugins::signed_in(&root, name);
        actions = actions.push(button(text(if busy { "Signing in…" } else if signed { "Sign in again" } else { "Sign in" }).size(12)).padding([5, 12]).style(ui::quiet_button)
            .on_press_maybe((!busy).then(|| pressed(Action::PluginSignIn(name.to_owned())))));
    }
    actions = actions.push(Space::new().width(Fill)).push(button(text("Close").size(12)).padding([5, 10]).style(ui::quiet_button).on_press(pressed(Action::PluginClose)));
    let facts = format!("{} · version {} · pinned {} · {} server{} · {} skill{}", if record.enabled { "On" } else { "Off" }, if record.version.is_empty() { "—" } else { &record.version },
        &record.sha[..record.sha.len().min(8)], record.servers.len(), if record.servers.len() == 1 { "" } else { "s" }, record.skills.len(), if record.skills.len() == 1 { "" } else { "s" });
    let mut body = column![text(record.title.clone()).size(18).font(ui::medium()), text(facts).size(12).color(p.muted)].spacing(8);
    if !record.description.is_empty() { body = body.push(text(record.description.clone()).size(13).color(p.muted)); }
    if !record.skills.is_empty() { body = body.push(text(format!("Skills: {}", record.skills.join(", "))).size(12).color(p.muted)); }
    let schema = hermes_engine::plugins::config_schema(&root, name);
    if !schema.is_empty() {
        body = body.push(text("Settings").size(13).font(ui::medium()));
        for field in schema {
            let value = state.plugin_form.get(&field.key).cloned().unwrap_or_default();
            let key = field.key.clone();
            let input = text_input(if field.default.is_empty() { &field.kind } else { &field.default }, &value).secure(field.secret).size(13).padding([6, 9]).width(320)
                .on_input(move |v| pressed(Action::PluginField(key.clone(), v)));
            body = body.push(row![column![text(field.key.clone()).size(13), text(field.description.clone()).size(11).color(p.muted)].spacing(2).width(Fill), input].spacing(16).align_y(Alignment::Center));
        }
        body = body.push(row![Space::new().width(Fill), button(text("Save settings").size(12)).padding([5, 12]).on_press(pressed(Action::PluginSaveConfig))]);
    }
    body = body.push(actions);
    container(body).padding(16).width(Fill).style(move |_| ui::panel(p.card, 10.0, Some(Color { a: 0.4, ..p.accent }))).into()
}

fn plugins(app: &Hermes) -> Element<'_, Message> {
    let state = &app.capabilities; let p = ui::palette(app.dark);
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for plugin in PLUGINS.iter() { *counts.entry(&plugin.category).or_default() += 1; }
    let options: Vec<Facet> = std::iter::once(all("All categories")).chain(counts.iter().map(|(c, n)| facet(c, *n))).collect();
    let selected = options.iter().find(|f| f.value == state.plugin_category).cloned();
    let terms: Vec<String> = state.search.to_lowercase().split_whitespace().map(str::to_owned).collect();
    let installed = |name: &str| state.plugins_installed.iter().find(|i| i.name == name);
    let rows: Vec<&Plugin> = PLUGINS.iter().filter(|plugin| state.plugin_category.as_deref().is_none_or(|c| plugin.category == c)
        && (!state.plugins_installed_only || installed(&plugin.name).is_some())
        && terms.iter().all(|t| plugin.name.to_lowercase().contains(t) || plugin.description.to_lowercase().contains(t))).collect();
    let mut col = column![
        search_bar("Search plugins…", &state.search, p, pick_list(options, selected, |f| pressed(Action::PluginCategory(f))).text_size(12).padding([4, 8]).into()),
        row![text(format!("{} plugins · {} installed · Agent Plugins packages run here; Python plugins run through your installed Hermes agent.", fmt_count(rows.len()), state.plugins_installed.len())).size(12).color(p.muted).width(Fill),
            toggler(state.plugins_installed_only).label("Installed only").text_size(12).size(16).on_toggle(|v| pressed(Action::PluginsInstalledOnly(v)))].align_y(Alignment::Center),
    ].spacing(14);
    if let Some(error) = &state.error { col = col.push(text(error.clone()).size(12).color(Color::from_rgb8(0xd1, 0x43, 0x43))); }
    else if let Some(notice) = &state.notice { col = col.push(text(notice.clone()).size(12).color(p.muted)); }
    if let Some(name) = &state.plugin_detail { col = col.push(plugin_detail(app, name)); }
    let mut grid = column![].spacing(12);
    for chunk in rows.iter().take(state.limit).collect::<Vec<_>>().chunks(3) {
        let mut line = row![].spacing(12);
        for plugin in chunk {
            let busy = state.plugin_busy.contains(&plugin.name);
            let control: Element<'_, Message> = match installed(&plugin.name) {
                Some(record) => row![
                    toggler(record.enabled).size(14).on_toggle({ let n = plugin.name.clone(); move |v| pressed(Action::PluginEnable(n.clone(), v)) }),
                    button(text("Remove").size(11)).padding([2, 6]).style(ui::quiet_button).on_press(pressed(Action::PluginRemove(plugin.name.clone()))),
                ].spacing(4).align_y(Alignment::Center).into(),
                None if busy => text("Installing…").size(11).color(p.muted).into(),
                None if plugin.sha.is_empty() => Space::new().into(),
                None => button(row![ui::icon("add", 11, p.foreground), text("Install").size(11)].spacing(3).align_y(Alignment::Center)).padding([2, 7])
                    .style(ui::quiet_button).on_press(pressed(Action::PluginInstall(plugin.name.clone()))).into(),
            };
            let body = column![
                row![container(text(plugin.category.clone()).size(10).color(p.muted)).padding([2, 6]).style(move |_| ui::panel(p.secondary, 4.0, None)), Space::new().width(Fill), control].align_y(Alignment::Center),
                text(display(plugin)).size(14).font(ui::medium()),
                text(clip(&plugin.description, 130)).size(12).color(p.muted).height(Length::Fixed(48.0)),
            ].spacing(7);
            let open = if installed(&plugin.name).is_some() { Some(pressed(Action::PluginOpen(plugin.name.clone()))) } else { (!plugin.url.is_empty()).then(|| Message::Link(plugin.url.clone())) };
            line = line.push(button(body).padding(14).width(Fill).on_press_maybe(open)
                .style(move |_, status| button::Style { background: Some(p.card.into()), text_color: p.foreground,
                    border: Border { radius: 10.0.into(), width: 1.0, color: if status == button::Status::Hovered { Color { a: 0.5, ..p.accent } } else { p.border } }, ..Default::default() }));
        }
        for _ in chunk.len()..3 { line = line.push(Space::new().width(Fill)); }
        grid = grid.push(line);
    }
    col = col.push(grid);
    if rows.len() > state.limit { col = col.push(container(button(text("More").size(12)).on_press(pressed(Action::More)).style(ui::quiet_button)).center_x(Fill)); }
    col.into()
}

fn connectors(app: &Hermes) -> Element<'_, Message> {
    let state = &app.capabilities; let p = ui::palette(app.dark);
    let configured = app.library.is_some() && !state.url.is_empty();
    let token_hint = if state.has_token { "Saved token · enter a replacement" } else { "Bearer token (optional)" };
    let input = |placeholder: &str, value: &str| text_input(placeholder, value).size(13).padding([7, 10]);
    let mut card = column![
        row![ui::icon("plug", 16, p.accent), column![text("MCP server").size(15).font(ui::medium()),
            text("Connect a Streamable HTTP MCP server. Its tools are offered to Hermes in new sessions.").size(12).color(p.muted)].spacing(2)].spacing(10),
        input("https://example.com/mcp", &state.url).on_input(|s| pressed(Action::Url(s))),
        input(token_hint, &state.token).secure(true).on_input(|s| pressed(Action::Token(s))),
        row![button(text("Save").size(12)).padding([5, 12]).on_press(pressed(Action::Save(false))),
            button(text("Discover tools").size(12)).padding([5, 12]).on_press_maybe((configured && !state.busy).then_some(pressed(Action::DiscoverTools))),
            Space::new().width(Fill),
            button(text("Disconnect").size(12)).padding([5, 12]).style(ui::quiet_button).on_press_maybe(configured.then_some(pressed(Action::Save(true))))].spacing(8),
    ].spacing(12);
    if state.environment && configured { card = card.push(text("Using the endpoint from HERMES_RUST_MCP_URL.").size(12).color(p.muted)); }
    if state.busy { card = card.push(text("Discovering tools…").size(12).color(p.muted)); }
    if !state.tools.is_empty() {
        let mut list = column![text(format!("{} tools", state.tools.len())).size(12).color(p.muted)].spacing(4);
        for name in &state.tools { list = list.push(row![ui::icon("tools", 12, p.muted), text(name.as_str()).size(13)].spacing(6)); }
        card = card.push(list);
    }
    let enabled = app.profile_settings.saved.mcp_tools;
    column![container(card).padding(18).width(Fill).style(move |_| ui::panel(p.card, 10.0, Some(p.border))),
        row![toggler(enabled).size(15).on_toggle(|on| pressed(Action::Toolset("mcp_tools", on))), text("Offer MCP tools to Hermes").size(13)].spacing(8).align_y(Alignment::Center),
    ].spacing(14).into()
}

const TOOLSETS: [(&str, &str, &str); 5] = [
    ("clock_tools", "System clock", "Read the local date and time."),
    ("file_tools", "Files", "List, search, read and write files in the open project."),
    ("skill_tools", "Skills", "Discover and read installed skills."),
    ("memory_tool", "Memory", "Save and update persistent memory."),
    ("mcp_tools", "MCP", "Tools from the connected MCP server."),
];

fn toolsets(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark); let saved = &app.profile_settings.saved;
    let specs = tools::tool_specs();
    let mut col = column![text("Switch a toolset off to stop Hermes from requesting its tools. Changes apply to the next message.").size(12).color(p.muted)].spacing(12);
    for (key, title, blurb) in TOOLSETS {
        let on = serde_json::to_value(saved).ok().and_then(|v| v[key].as_bool()).unwrap_or(false);
        let members: Vec<String> = specs.iter().filter_map(|s| s["function"]["name"].as_str()).filter(|name| {
            let mut probe = serde_json::to_value(saved).unwrap_or_default();
            for (k, _, _) in TOOLSETS { probe[k] = json!(k == key); }
            serde_json::from_value::<ProfileSettings>(probe).is_ok_and(|s| s.tool_enabled(name))
        }).map(str::to_owned).collect();
        let tools_line = if members.is_empty() { String::new() } else { members.join(" · ") };
        col = col.push(container(row![
            column![text(title).size(14).font(ui::medium()), text(blurb).size(12).color(p.muted), text(tools_line).size(11).color(p.muted).font(iced::Font::MONOSPACE)].spacing(3).width(Fill),
            toggler(on).size(16).on_toggle(move |v| pressed(Action::Toolset(key, v))),
        ].align_y(Alignment::Center)).padding(14).width(Fill).style(move |_| ui::panel(p.card, 10.0, Some(p.border))));
    }
    col.into()
}
