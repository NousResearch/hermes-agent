//! Profiles page (app/profiles): "Each one is its own Hermes — own keys, own
//! memory, own sessions." List, create, rename, delete, edit SOUL.md,
//! export/import, and open a profile in its own window.
use super::{Hermes, Message, ui};
use crate::appearance::text;
use hermes_library::profiles::Profile;
use iced::{Alignment, Color, Element, Fill, Task, widget::{Space, button, column, container, row, scrollable, text_editor, text_input}};

#[derive(Debug, Clone)]
pub(super) enum Action {
    Refresh, Select(String), New, NewName(String), Create, Cancel, Rename, RenameValue(String), RenameSave, Delete, ConfirmDelete,
    Soul(text_editor::Action), SaveSoul, Export, Import, Open(String), Search(String),
}

#[derive(Default)]
pub(super) struct State {
    list: Vec<Profile>, selected: Option<String>, creating: Option<String>, renaming: Option<String>, deleting: bool,
    soul: text_editor::Content, soul_saved: String, error: Option<String>, notice: Option<String>, search: String,
}

/// The profile this window runs (its data directory is `<root>/profiles/<name>`), else "default".
pub(super) fn current_name() -> String {
    hermes_engine::Session::default_dir().ok().and_then(|p| {
        let parent = p.parent()?.file_name()?.to_str()?.to_owned();
        (parent == "profiles").then(|| p.file_name()?.to_str().map(str::to_owned)).flatten()
    }).unwrap_or_else(|| "default".into())
}

pub(super) fn update(app: &mut Hermes, action: Action) -> Task<Message> {
    let Some(library) = app.library.clone() else { return Task::none() };
    let state = &mut app.profiles;
    let select = |state: &mut State, name: String| {
        let soul = library.profile_soul(&name).unwrap_or_default();
        state.soul = text_editor::Content::with_text(&soul); state.soul_saved = soul;
        state.selected = Some(name); state.renaming = None; state.deleting = false;
    };
    match action {
        Action::Refresh => {}
        Action::Search(v) => state.search = v,
        Action::Select(name) => { select(state, name); state.error = None; state.notice = None; }
        Action::New => state.creating = Some(String::new()),
        Action::NewName(v) => state.creating = Some(v.to_lowercase()),
        Action::Cancel => { state.creating = None; state.renaming = None; state.deleting = false; }
        Action::Create => if let Some(name) = state.creating.clone() {
            match library.create_profile(name.trim()) { Ok(p) => { state.creating = None; select(state, p.name); state.notice = Some("Profile created.".into()); } Err(e) => state.error = Some(format!("{e:#}")) }
        },
        Action::Rename => state.renaming = state.selected.clone(),
        Action::RenameValue(v) => state.renaming = Some(v.to_lowercase()),
        Action::RenameSave => if let (Some(from), Some(to)) = (state.selected.clone(), state.renaming.clone()) {
            match library.rename_profile(&from, to.trim()) { Ok(()) => { select(state, to.trim().to_owned()); state.notice = Some("Renamed.".into()); } Err(e) => state.error = Some(format!("{e:#}")) }
        },
        Action::Delete => state.deleting = true,
        Action::ConfirmDelete => if let Some(name) = state.selected.clone() {
            match library.delete_profile(&name) { Ok(()) => { state.selected = None; state.deleting = false; state.notice = Some(format!("Deleted {name}.")); } Err(e) => { state.deleting = false; state.error = Some(format!("{e:#}")) } }
        },
        Action::Soul(edit) => state.soul.perform(edit),
        Action::SaveSoul => if let Some(name) = &state.selected {
            let text = state.soul.text();
            match library.set_profile_soul(name, &text) { Ok(()) => { state.soul_saved = text; state.notice = Some("SOUL.md saved. It applies from the next message.".into()); } Err(e) => state.error = Some(format!("{e:#}")) }
        },
        Action::Export => if let Some(name) = state.selected.clone() {
            match super::picker::choose("Choose a folder for the profile archive", true) {
                Ok(Some(dir)) => { let dest = dir.join(format!("{name}.hermes-profile.tar.gz"));
                    match library.export_profile(&name, &dest) { Ok(()) => state.notice = Some(format!("Exported to {}", dest.display())), Err(e) => state.error = Some(format!("{e:#}")) } }
                Ok(None) => {}
                Err(e) => state.error = Some(e),
            }
        },
        Action::Import => match super::picker::choose("Choose a .hermes-profile.tar.gz archive", false) {
            Ok(Some(file)) => match library.import_profile(&file) { Ok(name) => { select(state, name.clone()); state.notice = Some(format!("Imported {name}.")); } Err(e) => state.error = Some(format!("{e:#}")) },
            Ok(None) => {}
            Err(e) => state.error = Some(e),
        },
        Action::Open(name) => match library.profile_path(&name) {
            Ok(path) => match std::env::current_exe().and_then(|exe| std::process::Command::new(exe).env("HERMES_RUST_DATA_DIR", &path).spawn()) {
                Ok(_) => state.notice = Some(format!("Opened {name} in a new window.")),
                Err(e) => state.error = Some(format!("Could not open a window: {e}")),
            },
            Err(e) => state.error = Some(format!("{e:#}")),
        },
    }
    state.list = library.profiles().unwrap_or_default();
    if state.selected.is_none() && let Some(first) = state.list.first().map(|p| p.name.clone()) { select(state, first); }
    Task::none()
}

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let state = &app.profiles; let p = ui::palette(app.dark);
    let act = Message::Profile;
    let current = current_name();
    let q = state.search.to_lowercase();
    let mut list = column![
        row![text(format!("{} profile{}", state.list.len(), if state.list.len() == 1 { "" } else { "s" })).size(12).color(p.muted), Space::new().width(Fill),
            button(text("Import…").size(11)).style(ui::quiet_button).on_press(act(Action::Import))].align_y(Alignment::Center),
        text_input("Search profiles...", &state.search).on_input(move |v| act(Action::Search(v))).size(13).padding([6, 9]),
    ].spacing(8);
    for profile in state.list.iter().filter(|p| q.is_empty() || p.name.contains(&q)) {
        let selected = state.selected.as_deref() == Some(profile.name.as_str());
        let mut label = row![ui::icon(if profile.is_bot { "hubot" } else { "account" }, 14, p.muted), text(profile.name.clone()).size(13)].spacing(8).align_y(Alignment::Center);
        if profile.name == current { label = label.push(text("this window").size(10).color(p.accent)); }
        list = list.push(button(label).width(Fill).padding([6, 8]).on_press(act(Action::Select(profile.name.clone())))
            .style(move |t, s| { let mut st = ui::quiet_button(t, s); if selected { st.background = Some(Color { a: 0.08, ..p.accent }.into()); } st }));
    }
    list = match &state.creating {
        Some(name) => list.push(column![text_input("new-profile", name).on_input(move |v| act(Action::NewName(v))).on_submit(act(Action::Create)).size(13).padding([6, 9]),
            text("Lowercase letters, digits, hyphens, and underscores.").size(11).color(p.muted),
            row![Space::new().width(Fill), button(text("Cancel").size(11)).style(ui::quiet_button).on_press(act(Action::Cancel)), button(text("Create").size(11)).padding([4, 10]).on_press(act(Action::Create))].spacing(6)].spacing(6)),
        None => list.push(button(row![ui::icon("add", 12, p.muted), text("New profile").size(12)].spacing(6).align_y(Alignment::Center)).style(ui::quiet_button).on_press(act(Action::New))),
    };
    let detail: Element<'_, Message> = match state.selected.as_ref().and_then(|n| state.list.iter().find(|p| &p.name == n)) {
        None => container(text("Select a profile").size(13).color(p.muted)).center(Fill).into(),
        Some(profile) => {
            let mut head = row![text(profile.name.clone()).size(20).font(ui::medium())].spacing(8).align_y(Alignment::Center);
            if profile.is_default { head = head.push(container(text("Default").size(10).color(p.accent)).padding([2, 6]).style(move |_| ui::panel(Color { a: 0.1, ..p.accent }, 4.0, None))); }
            if profile.is_bot { head = head.push(text("Bot").size(11).color(p.muted)); }
            head = head.push(Space::new().width(Fill));
            if profile.name != current { head = head.push(button(text(format!("Switch to {}", profile.name)).size(12)).padding([6, 12]).on_press(act(Action::Open(profile.name.clone())))); }
            let facts = column![
                row![text("Model").size(12).color(p.muted).width(120), text(if profile.model.is_empty() { "Not set".to_owned() } else { profile.model.clone() }).size(12)],
                row![text("Skills").size(12).color(p.muted).width(120), text(profile.skill_count.to_string()).size(12)],
                row![text("Sessions").size(12).color(p.muted).width(120), text(profile.sessions.to_string()).size(12)],
                row![text("Folder").size(12).color(p.muted).width(120), text(profile.path.display().to_string()).size(11).font(iced::Font::MONOSPACE)],
            ].spacing(6);
            let dirty = state.soul.text() != state.soul_saved;
            let soul = column![
                row![column![text("SOUL.md").size(13).font(ui::medium()), text(if state.soul_saved.is_empty() { "No SOUL.md yet — write the persona this profile always uses." } else { "The persona this profile always uses." }).size(11).color(p.muted)].spacing(2).width(Fill),
                    text(if dirty { "Unsaved changes" } else { "" }).size(11).color(p.muted),
                    button(text("Save SOUL.md").size(12)).padding([5, 12]).on_press_maybe(dirty.then(|| act(Action::SaveSoul)))].spacing(10).align_y(Alignment::Center),
                text_editor(&state.soul).on_action(move |a| act(Action::Soul(a))).height(220).size(13).padding(10),
            ].spacing(8);
            let mut actions = row![].spacing(8).align_y(Alignment::Center);
            if !profile.is_default {
                actions = actions.push(button(text("Export…").size(12)).style(ui::quiet_button).on_press(act(Action::Export)));
                if !profile.is_bot {
                    actions = match &state.renaming {
                        Some(v) => actions.push(text_input("new-name", v).on_input(move |v| act(Action::RenameValue(v))).on_submit(act(Action::RenameSave)).size(12).padding([4, 8]).width(180))
                            .push(button(text("Rename").size(12)).padding([4, 10]).on_press(act(Action::RenameSave))).push(button(text("Cancel").size(12)).style(ui::quiet_button).on_press(act(Action::Cancel))),
                        None => actions.push(button(text("Rename…").size(12)).style(ui::quiet_button).on_press(act(Action::Rename))),
                    };
                    actions = actions.push(Space::new().width(Fill));
                    actions = if state.deleting { actions.push(text(format!("Delete {} and everything in it?", profile.name)).size(12).color(p.muted)).push(button(text("Delete").size(12)).padding([4, 10]).on_press(act(Action::ConfirmDelete))).push(button(text("Cancel").size(12)).style(ui::quiet_button).on_press(act(Action::Cancel))) }
                        else { actions.push(button(text("Delete…").size(12)).style(ui::quiet_button).on_press(act(Action::Delete))) };
                }
            }
            let mut body = column![head, facts, soul, actions].spacing(18);
            if let Some(e) = &state.error { body = body.push(text(e.clone()).size(12).color(Color::from_rgb8(0xd1, 0x43, 0x43))); }
            else if let Some(n) = &state.notice { body = body.push(text(n.clone()).size(12).color(p.muted)); }
            scrollable(container(body).padding([8, 24])).height(Fill).into()
        }
    };
    row![container(scrollable(list)).width(260).padding([8, 16]), container(Space::new().width(1).height(Fill)).style(move |_| iced::widget::container::Style { background: Some(p.border.into()), ..Default::default() }), container(detail).width(Fill)].height(Fill).into()
}
