//! Settings › Providers › Accounts, beyond the featured Nous Portal card: the
//! Electron OAuthPicker's provider rows ("Connected" / "Other providers"
//! disclosure). Device-code providers sign in here; external ones run their
//! CLI in the Terminal pane, as Electron does.
use super::{Hermes, Message, Screen, settings, ui};
use crate::appearance::text;
use hermes_engine::{CancellationToken, Config, accounts::{self, DeviceLogin, Flow}};
use iced::{Alignment, Border, Color, Element, Fill, Font, Task, widget::{Space, button, column, container, row}};

#[derive(Debug, Clone)]
pub(super) enum Action {
    Start(&'static str), Started(u64, Result<DeviceLogin, String>), Finished(u64, &'static str, Result<(), String>),
    Cancel, Use(&'static str), Disconnect(&'static str), External(&'static str), ToggleAll,
}

#[derive(Default)]
pub(super) struct State {
    login: Option<DeviceLogin>, starting: Option<&'static str>, cancel: Option<CancellationToken>, run: u64,
    error: Option<String>, notice: Option<String>, show_all: bool,
}

pub(super) fn update(app: &mut Hermes, action: Action) -> Task<Message> {
    let state = &mut app.accounts;
    match action {
        Action::Start(id) => {
            if let Some(cancel) = state.cancel.take() { cancel.cancel(); }
            state.run += 1; state.starting = Some(id); state.login = None; state.error = None; state.notice = None;
            let run = state.run;
            return Task::perform(async move { accounts::begin(id).await.map_err(|e| format!("{e:#}")) }, move |r| Message::Account(Action::Started(run, r)));
        }
        Action::Started(run, result) if run == state.run => {
            let id = state.starting.take().unwrap_or_default();
            match result {
                Ok(login) => {
                    let _ = super::open_url(&login.verification_url);
                    let cancel = CancellationToken::new();
                    state.cancel = Some(cancel.clone()); state.login = Some(login.clone());
                    return Task::perform(async move { login.finish(cancel).await.map_err(|e| format!("{e:#}")) }, move |r| Message::Account(Action::Finished(run, id, r)));
                }
                Err(e) => state.error = Some(e),
            }
        }
        Action::Finished(run, id, result) if run == state.run => {
            state.login = None; state.cancel = None;
            match result {
                Ok(()) => { state.notice = Some(format!("{} connected.", accounts::provider(id).map_or(id, |p| p.name))); return update(app, Action::Use(id)); }
                Err(e) if e.contains("cancelled") => {}
                Err(e) => state.error = Some(e),
            }
        }
        Action::Started(..) | Action::Finished(..) => {}
        Action::Cancel => { if let Some(c) = state.cancel.take() { c.cancel(); } state.login = None; state.starting = None; state.run += 1; }
        Action::Use(id) if !app.running => {
            let Some(provider) = accounts::provider(id) else { return Task::none() };
            app.config = Config { base_url: provider.base_url.into(), model: provider.default_model.into(), api_key: String::new() };
            app.models.selected_endpoint = None; app.models.endpoint_name = provider.name.into();
            app.models.invalidate(); app.models.load_options(&app.config);
            if let Err(e) = super::Preferences::save(&app.config) { app.accounts.error = Some(e); return Task::none(); }
            app.accounts.notice = Some(format!("Chats now use {} · {}.", provider.name, provider.default_model));
        }
        Action::Use(_) => {}
        Action::Disconnect(id) => match accounts::disconnect(id) {
            Ok(true) => state.notice = Some(format!("{} was removed.", accounts::provider(id).map_or(id, |p| p.name))),
            Ok(false) => state.error = Some(format!("{} is signed in through its own CLI — sign out there.", accounts::provider(id).map_or(id, |p| p.name))),
            Err(e) => state.error = Some(format!("{e:#}")),
        },
        Action::External(id) => {
            let Some(provider) = accounts::provider(id) else { return Task::none() };
            state.notice = Some(format!("Running `{}` in the terminal. Finish the sign-in there, then come back.", provider.cli_command));
            let open = app.update(Message::Navigate(Screen::Terminal));
            let line = format!("{}\r", provider.cli_command).into_bytes();
            return Task::batch([open, Task::done(Message::Terminal(super::terminal::Action::Input(line)))]);
        }
        Action::ToggleAll => state.show_all = !state.show_all,
    }
    Task::none()
}

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let state = &app.accounts; let p = ui::palette(app.dark);
    let mut col = column![].spacing(6);
    if let Some(login) = &state.login {
        let name = accounts::provider(login.provider).map_or("", |p| p.name);
        col = col.push(column![
            text(format!("Sign in with {name}")).size(14).font(settings::semibold()),
            text("We opened the sign-in page in your browser. Enter this code there:").size(13).color(p.muted),
            button(text(login.user_code.clone()).size(22).font(Font::MONOSPACE)).padding([6, 12]).style(button::text).on_press(Message::Copy(login.user_code.clone())),
            row![button(text("Re-open sign-in page").size(12)).style(button::text).on_press(Message::Link(login.verification_url.clone())), Space::new().width(Fill),
                text("Waiting for you to authorize…").size(12).color(p.muted),
                button(text("Cancel").size(12)).style(button::text).on_press(Message::Account(Action::Cancel))].spacing(12).align_y(Alignment::Center),
        ].spacing(8)).push(Space::new().height(8));
    } else if let Some(id) = state.starting {
        col = col.push(text(format!("Starting sign-in for {}…", accounts::provider(id).map_or(id, |p| p.name))).size(13).color(p.muted));
    }
    let rows: Vec<_> = accounts::CATALOG.iter().map(|provider| (provider, accounts::status(provider.id))).collect();
    let (connected, others): (Vec<_>, Vec<_>) = rows.into_iter().partition(|(_, s)| s.logged_in);
    let active = |id: &str| accounts::for_base_url(&app.config.base_url).is_some_and(|p| p.id == id);
    if !connected.is_empty() {
        col = col.push(group_label("Connected", app.dark));
        for (provider, status) in &connected {
            let mut actions = row![].spacing(6).align_y(Alignment::Center);
            if active(provider.id) { actions = actions.push(text("In use").size(12).color(p.accent)); }
            else if provider.flow == Flow::DeviceCode || provider.id == "qwen-oauth" {
                actions = actions.push(button(text("Use").size(12)).padding([5, 12]).on_press_maybe((!app.running).then_some(Message::Account(Action::Use(provider.id)))));
            }
            actions = actions.push(button(text("Disconnect").size(12)).padding([5, 8]).style(button::text).on_press(Message::Account(Action::Disconnect(provider.id))));
            col = col.push(provider_row(provider.name, format!("{} · {}", provider.description, status.source), Some(actions.into()), None, app.dark));
        }
    }
    let collapsible = !others.is_empty();
    if !collapsible || state.show_all {
        if !connected.is_empty() { col = col.push(group_label("Other providers", app.dark)); }
        for (provider, _) in &others {
            let press = match provider.flow { Flow::DeviceCode => Action::Start(provider.id), Flow::External => Action::External(provider.id) };
            let hint = if provider.flow == Flow::External { format!("{} · runs `{}` in the terminal", provider.description, provider.cli_command) } else { provider.description.into() };
            col = col.push(provider_row(provider.name, hint, None, (state.login.is_none() && state.starting.is_none()).then_some(Message::Account(press)), app.dark));
        }
        for (name, hint) in [("Fireworks", "Fast open models with one API key"), ("OpenRouter", "One key, hundreds of models — a solid default")] {
            col = col.push(provider_row(name, hint.into(), None, Some(Message::SettingsView(settings::View::ApiKeys)), app.dark));
        }
    }
    if collapsible {
        let label = if state.show_all { "Collapse" } else if connected.is_empty() { "Other providers" } else { "Connect another provider" };
        col = col.push(button(row![text(label).size(12), ui::icon(if state.show_all { "chevron-up" } else { "chevron-down" }, 12, p.muted)].spacing(4).align_y(Alignment::Center))
            .padding([4, 0]).style(button::text).on_press(Message::Account(Action::ToggleAll)));
    }
    if let Some(e) = &state.error { col = col.push(text(e.clone()).size(12).color(Color::from_rgb8(192, 71, 72))); }
    else if let Some(n) = &state.notice { col = col.push(text(n.clone()).size(12).color(p.muted)); }
    col.into()
}

fn group_label(label: &str, dark: bool) -> Element<'_, Message> {
    let p = ui::palette(dark);
    container(text(label.to_uppercase()).size(11).font(settings::semibold()).color(Color { a: 0.55, ..p.foreground })).padding([8, 2]).into()
}

fn provider_row<'a>(name: &'a str, hint: String, actions: Option<Element<'a, Message>>, press: Option<Message>, dark: bool) -> Element<'a, Message> {
    let p = ui::palette(dark);
    let body = row![column![text(name).size(14).font(settings::semibold()), text(hint).size(12).color(p.muted).line_height(1.5)].spacing(4).width(Fill)]
        .spacing(12).align_y(Alignment::Center);
    match actions {
        Some(actions) => container(body.push(actions)).padding([10, 12]).width(Fill)
            .style(move |_| container::Style { border: Border { color: Color { a: 0.08, ..p.foreground }, width: 1.0, radius: 8.0.into() }, ..Default::default() }).into(),
        None => button(body.push(settings::icon("chevron", 16.0, p.muted))).padding([10, 12]).width(Fill).on_press_maybe(press)
            .style(move |_, status| button::Style {
                background: (status == button::Status::Hovered).then(|| Color { a: 0.04, ..p.foreground }.into()),
                text_color: p.foreground, border: Border { radius: 8.0.into(), ..Default::default() }, ..Default::default() }).into(),
    }
}
