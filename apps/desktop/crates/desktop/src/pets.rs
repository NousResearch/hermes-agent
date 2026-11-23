//! Pet gallery (agent/pet, app/pet-generate): adopt pets from the public
//! petdex gallery (https://petdex.dev/api/manifest, read-only, no account).
//! A pet is a `pet.json` plus a sprite sheet of 192×208 cells — the 9-row Codex
//! atlas or the older 8-row one — stored at `<library>/pets/<slug>/`. The
//! active pet animates in the desktop-pet window by agent state.
use super::{Hermes, Message, ui};
use crate::appearance::text;
use iced::{Alignment, Color, Element, Fill, Task, widget::{Space, button, column, container, image, row, scrollable, text_input}};
use std::path::{Path, PathBuf};

const MANIFEST: &str = "https://petdex.dev/api/manifest";
const FRAME_W: u32 = 192;
const FRAME_H: u32 = 208;
const FRAMES: u32 = 6;
const CODEX_ROWS: [&str; 9] = ["idle", "running-right", "running-left", "waving", "jumping", "failed", "waiting", "running", "review"];
const LEGACY_ROWS: [&str; 8] = ["idle", "wave", "run", "failed", "review", "jump", "extra1", "extra2"];

#[derive(Debug, Clone)]
pub(super) struct Entry { pub slug: String, pub display_name: String, pub kind: String, pub submitted_by: String, pub spritesheet_url: String, pub pet_json_url: String }

/// The manifest's pets, skipping any entry without a slug or sprite sheet (fields may be null).
fn entries(doc: &serde_json::Value) -> Vec<Entry> {
    let s = |v: &serde_json::Value, k: &str| v[k].as_str().unwrap_or_default().to_owned();
    doc["pets"].as_array().into_iter().flatten().filter_map(|v| {
        let e = Entry { slug: s(v, "slug"), display_name: s(v, "displayName"), kind: s(v, "kind"), submitted_by: s(v, "submittedBy"), spritesheet_url: s(v, "spritesheetUrl"), pet_json_url: s(v, "petJsonUrl") };
        (safe(&e.slug) && !e.spritesheet_url.is_empty()).then(|| Entry { display_name: if e.display_name.is_empty() { e.slug.clone() } else { e.display_name.clone() }, ..e })
    }).collect()
}

#[derive(Debug, Clone)]
pub(super) enum Action {
    Open, Loaded(Result<Vec<Entry>, String>), Search(String), Adopt(String), Adopted(String, Result<(), String>), Use(Option<String>), Remove(String),
    // AI generation (agent/pet/generate): describe → 4 drafts → pick → hatch.
    GenConcept(String), GenStyle(String), GenName(String), GenDraft, Drafted(u64, Result<Vec<Vec<u8>>, String>), GenPick(usize), Hatch, HatchProgress(u64, usize, usize), Hatched(u64, Result<String, String>), GenReset,
}

#[derive(Default)]
pub(super) struct State {
    manifest: Vec<Entry>, loading: bool, search: String, busy: Option<String>, error: Option<String>, notice: Option<String>,
    installed: Vec<String>, thumbs: std::collections::BTreeMap<String, image::Handle>,
    /// The active pet's decoded sheet: (slug, rgba, row names).
    sheet: Option<(String, ::image::RgbaImage, Vec<&'static str>)>,
    frames: std::collections::BTreeMap<&'static str, Vec<image::Handle>>,
    hatchery: Gen,
}
#[derive(Default)]
struct Gen { concept: String, style: String, name: String, drafts: Vec<(Vec<u8>, image::Handle)>, pick: Option<usize>, busy: Option<String>, run: u64 }

fn slugify(name: &str) -> String {
    let base: String = name.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    let base = base.split('-').filter(|s| !s.is_empty()).collect::<Vec<_>>().join("-");
    let base: String = base.chars().take(40).collect();
    format!("{}-{:06x}", if base.is_empty() { "pet" } else { &base }, chrono::Utc::now().timestamp_subsec_micros() & 0xffffff)
}

fn dir(root: &Path) -> PathBuf { root.join("pets") }
fn safe(slug: &str) -> bool { !slug.is_empty() && slug.len() <= 120 && slug.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') }
fn active(root: &Path) -> Option<String> { std::fs::read_to_string(dir(root).join("active")).ok().map(|s| s.trim().to_owned()).filter(|s| safe(s)) }
fn sheet_path(root: &Path, slug: &str) -> Option<PathBuf> {
    ["sprite.webp", "spritesheet.webp", "sprite.png", "spritesheet.png"].iter().map(|n| dir(root).join(slug).join(n)).find(|p| p.is_file())
}
fn installed(root: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(dir(root)).map(|d| d.flatten().filter(|e| e.path().is_dir()).filter_map(|e| e.file_name().into_string().ok()).filter(|n| safe(n) && sheet_path(root, n).is_some()).collect()).unwrap_or_default();
    out.sort(); out
}
fn decode(path: &Path) -> Option<(::image::RgbaImage, Vec<&'static str>)> {
    let img = ::image::open(path).ok()?.to_rgba8();
    let rows = img.height() / FRAME_H;
    (img.width() >= FRAME_W && rows >= 1).then(|| (img, if rows as usize >= CODEX_ROWS.len() { CODEX_ROWS.to_vec() } else { LEGACY_ROWS.to_vec() }))
}
fn crop(sheet: &::image::RgbaImage, row: u32, col: u32, scale: f32) -> image::Handle {
    let cell = ::image::imageops::crop_imm(sheet, col * FRAME_W, row * FRAME_H, FRAME_W, FRAME_H).to_image();
    let (w, h) = (((FRAME_W as f32) * scale) as u32, ((FRAME_H as f32) * scale) as u32);
    let small = ::image::imageops::resize(&cell, w, h, ::image::imageops::FilterType::Triangle);
    image::Handle::from_rgba(w, h, small.into_raw())
}

impl State {
    /// Load the active pet (call when the library opens or the choice changes).
    pub fn load_active(&mut self, root: &Path) {
        self.installed = installed(root);
        self.frames.clear();
        self.sheet = active(root).and_then(|slug| { let (img, rows) = decode(&sheet_path(root, &slug)?)?; Some((slug, img, rows)) });
    }
    pub fn has_pet(&self) -> bool { self.sheet.is_some() }
    /// Frames already prepared for `state` (the pet window's tick prepares them).
    pub fn cached(&self, state: &str) -> &[image::Handle] { self.frames.get(state).map(Vec::as_slice).unwrap_or(&[]) }
    /// Frames for a Hermes pet state (idle, wave, run, failed, review, jump, waiting), cached.
    pub fn frames_for(&mut self, state: &'static str) -> &[image::Handle] {
        let Some((_, sheet, rows)) = &self.sheet else { return &[] };
        if !self.frames.contains_key(state) {
            let aliases: &[&str] = match state { "wave" => &["waving", "wave"], "jump" => &["jumping", "jump"], "run" => &["running", "run"],
                "failed" => &["failed"], "review" => &["review"], "waiting" => &["waiting"], _ => &["idle"] };
            let row = aliases.iter().find_map(|a| rows.iter().position(|r| r == a)).unwrap_or(0) as u32;
            let cols = (sheet.width() / FRAME_W).min(FRAMES).max(1);
            let frames = (0..cols).map(|c| crop(sheet, row.min(sheet.height() / FRAME_H - 1), c, 0.5)).collect();
            self.frames.insert(state, frames);
        }
        self.frames.get(state).map(Vec::as_slice).unwrap_or(&[])
    }
}

async fn download(url: &str, max: usize) -> Result<Vec<u8>, String> {
    let parsed = reqwest::Url::parse(url).map_err(|_| "bad URL".to_owned())?;
    if parsed.scheme() != "https" { return Err("pets download over HTTPS only".into()); }
    let mut response = reqwest::get(parsed).await.map_err(|e| e.to_string())?.error_for_status().map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? { bytes.extend_from_slice(&chunk); if bytes.len() > max { return Err("download too large".into()); } }
    Ok(bytes)
}

pub(super) fn update(app: &mut Hermes, action: Action) -> Task<Message> {
    let Some(root) = app.library.as_ref().map(|l| l.root().to_owned()) else { return Task::none() };
    let state = &mut app.pets;
    match action {
        Action::Open => {
            state.installed = installed(&root);
            for slug in state.installed.clone() {
                if !state.thumbs.contains_key(&slug) && let Some((img, _)) = sheet_path(&root, &slug).and_then(|p| decode(&p)) { state.thumbs.insert(slug, crop(&img, 0, 0, 0.4)); }
            }
            if state.manifest.is_empty() && !state.loading {
                state.loading = true; state.error = None;
                return Task::perform(async {
                    let bytes = download(MANIFEST, 20 * 1024 * 1024).await?;
                    let doc: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
                    Ok(entries(&doc))
                }, |r| Message::Pet(Action::Loaded(r)));
            }
        }
        Action::Loaded(result) => { state.loading = false; match result { Ok(list) => state.manifest = list, Err(e) => state.error = Some(format!("Could not load the pet gallery: {e}")) } }
        Action::Search(v) => state.search = v,
        Action::Adopt(slug) => {
            let Some(entry) = state.manifest.iter().find(|e| e.slug == slug).cloned() else { return Task::none() };
            if !safe(&slug) { state.error = Some("That pet has an invalid name".into()); return Task::none(); }
            state.busy = Some(slug.clone()); state.error = None;
            let target = dir(&root).join(&slug);
            return Task::perform(async move {
                let sheet = download(&entry.spritesheet_url, 40 * 1024 * 1024).await?;
                let ext = if entry.spritesheet_url.ends_with(".png") { "png" } else { "webp" };
                let decoded = ::image::load_from_memory(&sheet).map_err(|_| "The sprite sheet could not be read".to_owned())?;
                if decoded.width() < FRAME_W || decoded.height() < FRAME_H { return Err("The sprite sheet is too small".into()); }
                std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
                std::fs::write(target.join(format!("sprite.{ext}")), &sheet).map_err(|e| e.to_string())?;
                if !entry.pet_json_url.is_empty() && let Ok(meta) = download(&entry.pet_json_url, 256 * 1024).await { let _ = std::fs::write(target.join("pet.json"), meta); }
                Ok(())
            }, move |r| Message::Pet(Action::Adopted(slug.clone(), r)));
        }
        Action::Adopted(slug, result) => {
            state.busy = None;
            match result { Ok(()) => { state.notice = Some("Adopted. It's now your desktop pet.".into()); return update(app, Action::Use(Some(slug))); } Err(e) => state.error = Some(e) }
        }
        Action::Use(slug) => {
            let _ = std::fs::create_dir_all(dir(&root));
            let _ = match &slug { Some(s) if safe(s) => std::fs::write(dir(&root).join("active"), s), _ => std::fs::remove_file(dir(&root).join("active")) };
            state.load_active(&root);
            let _ = update(app, Action::Open);
            if app.overlays.pet.is_none() { return app.update(Message::Overlay(super::overlays::Action::TogglePet)); }
        }
        Action::GenConcept(v) => state.hatchery.concept = v,
        Action::GenStyle(v) => state.hatchery.style = v,
        Action::GenName(v) => state.hatchery.name = v.chars().take(40).collect(),
        Action::GenReset => { state.hatchery = Gen { style: state.hatchery.style.clone(), ..Gen::default() }; }
        Action::GenDraft => {
            if state.hatchery.concept.trim().is_empty() { state.error = Some("Describe your pet first.".into()); return Task::none(); }
            state.hatchery.run += 1; state.hatchery.busy = Some("Drawing four looks… (about a minute)".into()); state.hatchery.drafts.clear(); state.hatchery.pick = None; state.error = None; state.notice = None;
            let (run, concept, style) = (state.hatchery.run, state.hatchery.concept.clone(), if state.hatchery.style.is_empty() { "auto".to_owned() } else { state.hatchery.style.clone() });
            return Task::perform(async move {
                let config = hermes_engine::petgen::portal_config().await.map_err(|_| "Sign in to Nous Portal (Settings › Accounts) to generate pets.".to_owned())?;
                hermes_engine::petgen::drafts(&config, hermes_engine::petgen::DEFAULT_MODEL, &concept, &style, 4).await.map_err(|e| format!("{e:#}"))
            }, move |r| Message::Pet(Action::Drafted(run, r)));
        }
        Action::Drafted(run, result) if run == state.hatchery.run => {
            state.hatchery.busy = None;
            match result {
                Ok(pngs) => {
                    state.hatchery.drafts = pngs.into_iter().filter_map(|png| { let img = ::image::load_from_memory(&png).ok()?.to_rgba8(); let (w, h) = img.dimensions(); Some((png, image::Handle::from_rgba(w, h, img.into_raw()))) }).collect();
                    state.hatchery.pick = (!state.hatchery.drafts.is_empty()).then_some(0);
                    if state.hatchery.name.is_empty() { state.hatchery.name = state.hatchery.concept.split_whitespace().rev().find(|w| w.len() > 2).unwrap_or("Pet").chars().take(20).collect(); }
                }
                Err(e) => state.error = Some(e),
            }
        }
        Action::GenPick(i) => state.hatchery.pick = Some(i),
        Action::Hatch => {
            let Some(base) = state.hatchery.pick.and_then(|i| state.hatchery.drafts.get(i)).map(|d| d.0.clone()) else { return Task::none() };
            state.hatchery.run += 1; state.hatchery.busy = Some("Hatching… drawing the animations (a few minutes)".into()); state.error = None;
            let (run, concept, style) = (state.hatchery.run, state.hatchery.concept.clone(), if state.hatchery.style.is_empty() { "auto".to_owned() } else { state.hatchery.style.clone() });
            let display = if state.hatchery.name.trim().is_empty() { "Pet".to_owned() } else { state.hatchery.name.trim().to_owned() };
            let slug = slugify(&display);
            let target = dir(&root).join(&slug);
            return Task::run(iced::stream::channel(32, move |mut out: iced::futures::channel::mpsc::Sender<Message>| async move {
                use iced::futures::SinkExt;
                let (tx, rx) = async_channel::unbounded::<(usize, usize)>();
                let mut forward = out.clone();
                let relay = tokio::spawn(async move { while let Ok((d, t)) = rx.recv().await { let _ = forward.send(Message::Pet(Action::HatchProgress(run, d, t))).await; } });
                let result = async {
                    let config = hermes_engine::petgen::portal_config().await.map_err(|_| "Sign in to Nous Portal (Settings › Accounts) to generate pets.".to_owned())?;
                    let sheet = hermes_engine::petgen::hatch(&config, hermes_engine::petgen::DEFAULT_MODEL, &base, &concept, &style, move |d, t| { let _ = tx.try_send((d, t)); }).await.map_err(|e| format!("{e:#}"))?;
                    std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
                    sheet.save(target.join("sprite.png")).map_err(|e| e.to_string())?;
                    let meta = serde_json::json!({"id": slug, "displayName": display, "description": concept, "spritesheetPath": "sprite.png", "generated": true});
                    std::fs::write(target.join("pet.json"), serde_json::to_vec_pretty(&meta).unwrap_or_default()).map_err(|e| e.to_string())?;
                    Ok(slug.clone())
                }.await;
                relay.abort();
                let _ = out.send(Message::Pet(Action::Hatched(run, result))).await;
            }), |m| m);
        }
        Action::HatchProgress(run, d, t) if run == state.hatchery.run => state.hatchery.busy = Some(format!("Hatching… animation {d} of {t}")),
        Action::Hatched(run, result) if run == state.hatchery.run => {
            state.hatchery.busy = None;
            match result {
                Ok(slug) => { state.hatchery = Gen { style: state.hatchery.style.clone(), ..Gen::default() }; state.notice = Some("Your pet hatched! It's now your desktop pet.".into()); return update(app, Action::Use(Some(slug))); }
                Err(e) => state.error = Some(e),
            }
        }
        Action::Drafted(..) | Action::HatchProgress(..) | Action::Hatched(..) => {}
        Action::Remove(slug) => if safe(&slug) {
            let _ = std::fs::remove_dir_all(dir(&root).join(&slug));
            if active(&root).as_deref() == Some(slug.as_str()) { let _ = std::fs::remove_file(dir(&root).join("active")); }
            state.thumbs.remove(&slug);
            state.load_active(&root);
        },
    }
    Task::none()
}

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let state = &app.pets; let p = ui::palette(app.dark);
    let current = state.sheet.as_ref().map(|s| s.0.clone());
    let mut col = column![
        row![text("Adopt a pet from petdex — it lives on your desktop and reacts while Hermes works.").size(12).color(p.muted).width(Fill),
            button(text(if current.is_some() { "Use the egg" } else { "Using the egg" }).size(11)).style(ui::quiet_button).on_press_maybe(current.is_some().then(|| Message::Pet(Action::Use(None))))].align_y(Alignment::Center),
    ].spacing(12);
    if let Some(e) = &state.error { col = col.push(text(e.clone()).size(12).color(Color::from_rgb8(0xd1, 0x43, 0x43))); }
    else if let Some(n) = &state.notice { col = col.push(text(n.clone()).size(12).color(p.muted)); }
    if !state.installed.is_empty() {
        col = col.push(text("YOUR PETS").size(10).font(ui::medium()).color(p.muted));
        let mut line = row![].spacing(10);
        for slug in &state.installed {
            let chosen = current.as_deref() == Some(slug.as_str());
            let mut card = column![].spacing(4).align_x(Alignment::Center).width(110);
            if let Some(thumb) = state.thumbs.get(slug) { card = card.push(image(thumb.clone()).width(76).height(83)); }
            card = card.push(text(slug.split('-').next().unwrap_or(slug).to_owned()).size(11));
            card = card.push(row![button(text(if chosen { "In use" } else { "Use" }).size(10)).padding([2, 6]).on_press_maybe((!chosen).then(|| Message::Pet(Action::Use(Some(slug.clone()))))),
                button(text("Remove").size(10)).padding([2, 6]).style(ui::quiet_button).on_press(Message::Pet(Action::Remove(slug.clone())))].spacing(4));
            line = line.push(container(card).padding(8).style(move |_| ui::panel(p.card, 8.0, Some(if chosen { p.accent } else { p.border }))));
        }
        col = col.push(scrollable(line).direction(iced::widget::scrollable::Direction::Horizontal(Default::default())));
    }
    // Generate
    let hatch = &state.hatchery;
    let busy = hatch.busy.is_some();
    let mut make = column![text("GENERATE A PET").size(10).font(ui::medium()).color(p.muted),
        row![text_input("Describe your pet — e.g. a sleepy fox astronaut with a red scarf", &hatch.concept).on_input(|v| Message::Pet(Action::GenConcept(v))).on_submit(Message::Pet(Action::GenDraft)).size(13).padding([7, 10]).width(Fill),
            iced::widget::pick_list(hermes_engine::petgen::STYLES.map(str::to_owned).to_vec(), Some(if hatch.style.is_empty() { "auto".to_owned() } else { hatch.style.clone() }), |v| Message::Pet(Action::GenStyle(v))).text_size(12).padding([6, 8]),
            button(text("Draw looks").size(12)).padding([7, 12]).on_press_maybe((!busy).then_some(Message::Pet(Action::GenDraft)))].spacing(8).align_y(Alignment::Center),
    ].spacing(8);
    if let Some(b) = &hatch.busy { make = make.push(text(b.clone()).size(12).color(p.accent)); }
    if !hatch.drafts.is_empty() {
        let mut looks = row![].spacing(10);
        for (i, (_, handle)) in hatch.drafts.iter().enumerate() {
            let chosen = hatch.pick == Some(i);
            looks = looks.push(button(image(handle.clone()).width(110).height(110)).padding(6).on_press_maybe((!busy).then_some(Message::Pet(Action::GenPick(i))))
                .style(move |_, _| button::Style { background: Some(p.card.into()), border: iced::Border { radius: 8.0.into(), width: if chosen { 2.0 } else { 1.0 }, color: if chosen { p.accent } else { p.border } }, ..Default::default() }));
        }
        make = make.push(looks).push(row![text("Name").size(12), text_input("Name", &hatch.name).on_input(|v| Message::Pet(Action::GenName(v))).size(12).padding([5, 8]).width(200), Space::new().width(Fill),
            button(text("Start over").size(11)).style(ui::quiet_button).on_press_maybe((!busy).then_some(Message::Pet(Action::GenReset))),
            button(text("Hatch this one").size(12)).padding([6, 12]).on_press_maybe((!busy && hatch.pick.is_some()).then_some(Message::Pet(Action::Hatch)))].spacing(8).align_y(Alignment::Center))
            .push(text("Hatching draws nine animations with your Nous Portal account; it takes a few minutes.").size(11).color(p.muted));
    }
    col = col.push(container(make).padding(14).width(Fill).style(move |_| ui::panel(p.card, 10.0, Some(p.border))));
    col = col.push(row![text("GALLERY").size(10).font(ui::medium()).color(p.muted), Space::new().width(Fill),
        text_input("Search pets…", &state.search).on_input(|v| Message::Pet(Action::Search(v))).size(12).padding([5, 8]).width(240)].align_y(Alignment::Center));
    if state.loading { col = col.push(text("Loading the gallery…").size(12).color(p.muted)); }
    let q = state.search.to_lowercase();
    let matches: Vec<&Entry> = state.manifest.iter().filter(|e| q.is_empty() || e.display_name.to_lowercase().contains(&q) || e.kind.to_lowercase().contains(&q)).take(60).collect();
    if !state.manifest.is_empty() { col = col.push(text(format!("{} pets · showing {}", state.manifest.len(), matches.len())).size(11).color(p.muted)); }
    for chunk in matches.chunks(3) {
        let mut line = row![].spacing(10);
        for e in chunk {
            let have = state.installed.contains(&e.slug);
            let busy = state.busy.as_deref() == Some(e.slug.as_str());
            line = line.push(container(row![column![text(e.display_name.clone()).size(13), text(format!("{}{}", e.kind, if e.submitted_by.is_empty() { String::new() } else { format!(" · by {}", e.submitted_by) })).size(10).color(p.muted)].spacing(2).width(Fill),
                button(text(if have { "Adopted" } else if busy { "Adopting…" } else { "Adopt" }).size(11)).padding([4, 10]).on_press_maybe((!have && state.busy.is_none()).then(|| Message::Pet(Action::Adopt(e.slug.clone()))))].align_y(Alignment::Center))
                .padding(10).width(Fill).style(move |_| ui::panel(p.card, 8.0, Some(p.border))));
        }
        for _ in chunk.len()..3 { line = line.push(Space::new().width(Fill)); }
        col = col.push(line);
    }
    scrollable(container(col).padding([8, 32]).max_width(980).center_x(Fill)).height(Fill).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Network: fetch the gallery, adopt one pet's sheet, cut its frames.
    #[tokio::test] #[ignore]
    async fn live_adopt_a_petdex_pet() {
        let bytes = download(MANIFEST, 20 * 1024 * 1024).await.unwrap();
        let doc: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let list = entries(&doc);
        assert!(list.len() > 100);
        let entry = &list[0];
        let sheet = download(&entry.spritesheet_url, 40 * 1024 * 1024).await.unwrap();
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir(root.path()).join(&entry.slug)).unwrap();
        std::fs::write(dir(root.path()).join(&entry.slug).join("sprite.webp"), &sheet).unwrap();
        std::fs::write(dir(root.path()).join("active"), &entry.slug).unwrap();
        let mut state = State::default();
        state.load_active(root.path());
        assert!(state.has_pet(), "the sheet decodes");
        let (_, img, rows) = state.sheet.as_ref().unwrap();
        eprintln!("{}: {}x{} rows={}", entry.slug, img.width(), img.height(), rows.len());
        for mood in ["idle", "run", "waiting", "failed", "wave"] { assert!(!state.frames_for(mood).is_empty(), "{mood}"); }
    }
}
