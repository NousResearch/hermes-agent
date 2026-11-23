//! Webhooks page (app/webhooks): subscriptions the Rust gateway serves at
//! `POST /webhooks/<name>`. Create one (the secret is shown once), switch it
//! on or off, see its recent deliveries, open the chat its runs go to.
use super::{Hermes, Message, ui};
use crate::appearance::text;
use hermes_library::webhooks::Webhook;
use iced::{Alignment, Color, Element, Fill, Task, widget::{Space, button, column, container, row, scrollable, text_input, toggler}};

#[derive(Debug, Clone)]
pub(super) enum Action {
    New, Cancel, Name(String), Description(String), Prompt(String), Events(String), Skills(String), DeliverOnly(bool), Create,
    Enable(String, bool), Delete(String), ConfirmDelete, OpenChat(String), Copy(String), Search(String), Refresh, DismissCreated,
}

#[derive(Default)]
pub(super) struct State {
    form: Option<Form>, created: Option<(String, String)>, deleting: Option<String>, error: Option<String>, search: String, list: Vec<Webhook>,
}
#[derive(Default, Clone)]
struct Form { name: String, description: String, prompt: String, events: String, skills: String, deliver_only: bool }

fn split(list: &str) -> Vec<String> { list.split([',', ' ']).map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned).collect() }

pub(super) fn update(app: &mut Hermes, action: Action) -> Task<Message> {
    let Some(library) = app.library.clone() else { return Task::none() };
    let state = &mut app.webhooks;
    match action {
        Action::Refresh => {}
        Action::New => { state.form = Some(Form::default()); state.error = None; }
        Action::Cancel => state.form = None,
        Action::Name(v) => if let Some(f) = &mut state.form { f.name = v },
        Action::Description(v) => if let Some(f) = &mut state.form { f.description = v },
        Action::Prompt(v) => if let Some(f) = &mut state.form { f.prompt = v },
        Action::Events(v) => if let Some(f) = &mut state.form { f.events = v },
        Action::Skills(v) => if let Some(f) = &mut state.form { f.skills = v },
        Action::DeliverOnly(v) => if let Some(f) = &mut state.form { f.deliver_only = v },
        Action::Create => if let Some(f) = state.form.clone() {
            let hook = Webhook { name: f.name, description: f.description, prompt: f.prompt, events: split(&f.events), skills: split(&f.skills), deliver_only: f.deliver_only, ..Default::default() };
            match library.create_webhook(hook) {
                Ok(hook) => { state.form = None; state.created = Some((hook.name, hook.secret)); state.error = None; }
                Err(e) => state.error = Some(format!("{e:#}")),
            }
        },
        Action::DismissCreated => state.created = None,
        Action::Enable(name, on) => if let Err(e) = library.set_webhook_enabled(&name, on) { state.error = Some(format!("{e:#}")) },
        Action::Delete(name) => state.deleting = Some(name),
        Action::ConfirmDelete => if let Some(name) = state.deleting.take() && let Err(e) = library.delete_webhook(&name) { state.error = Some(format!("{e:#}")) },
        Action::OpenChat(id) => { if let Ok(summary) = library.session(&id) { app.open_chat(summary); app.screen = super::Screen::Chat; } return Task::none(); }
        Action::Copy(v) => return iced::clipboard::write(v),
        Action::Search(v) => state.search = v,
    }
    state.list = library.webhooks().unwrap_or_default();
    Task::none()
}

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let state = &app.webhooks; let p = ui::palette(app.dark);
    let base = app.local_gateway.http_url().map(str::to_owned);
    let url = |name: &str| base.as_ref().map_or_else(|| format!("(start the local gateway) /webhooks/{name}"), |b| format!("{b}/webhooks/{name}"));
    let act = Message::Webhook;
    let mut col = column![
        row![text(format!("{} subscription{}", state.list.len(), if state.list.len() == 1 { "" } else { "s" })).size(13).color(p.muted), Space::new().width(Fill),
            text_input("Search webhooks…", &state.search).on_input(move |v| act(Action::Search(v))).size(13).padding([6, 9]).width(240),
            button(row![ui::icon("add", 12, p.background), text("New webhook").size(12)].spacing(5).align_y(Alignment::Center)).padding([6, 12]).on_press(act(Action::New))].spacing(10).align_y(Alignment::Center),
        text("Each subscription is an HTTPS-style endpoint on the local gateway. Senders sign the body with HMAC-SHA256 (X-Hub-Signature-256: sha256=…). Expose the gateway through a tunnel to receive from the internet.").size(12).color(p.muted),
    ].spacing(12);
    if base.is_none() { col = col.push(text("The local gateway is stopped, so deliveries can't arrive. Start it in Settings › Gateways.").size(12).color(Color::from_rgb8(0xc2, 0x7c, 0x0e))); }
    if let Some(e) = &state.error { col = col.push(text(e.clone()).size(12).color(Color::from_rgb8(0xd1, 0x43, 0x43))); }
    if let Some((name, secret)) = &state.created {
        col = col.push(container(column![
            text(format!("Created “{name}”")).size(14).font(ui::medium()),
            row![text("URL").size(12).width(80), text(url(name)).size(12).width(Fill), button(text("Copy").size(11)).style(ui::quiet_button).on_press(act(Action::Copy(url(name))))].align_y(Alignment::Center),
            row![text("Secret").size(12).width(80), text(secret.clone()).size(12).font(iced::Font::MONOSPACE).width(Fill), button(text("Copy").size(11)).style(ui::quiet_button).on_press(act(Action::Copy(secret.clone())))].align_y(Alignment::Center),
            row![text("The secret is shown only now. Store it with the sender.").size(11).color(p.muted).width(Fill), button(text("Done").size(12)).padding([5, 12]).on_press(act(Action::DismissCreated))].align_y(Alignment::Center),
        ].spacing(8)).padding(14).style(move |_| ui::panel(p.card, 8.0, Some(Color { a: 0.5, ..p.accent }))));
    }
    if let Some(form) = &state.form {
        let field = |label: &'static str, hint: &'static str, value: &str, f: fn(String) -> Action| row![column![text(label).size(13), text(hint).size(11).color(p.muted)].spacing(2).width(220),
            text_input("", value).on_input(move |v| act(f(v))).size(13).padding([6, 9])].spacing(14).align_y(Alignment::Center);
        col = col.push(container(column![
            text("New webhook").size(14).font(ui::medium()),
            field("Name", "lowercase, - or _", &form.name, Action::Name),
            field("Description", "optional", &form.description, Action::Description),
            field("Prompt", "what the agent should do", &form.prompt, Action::Prompt),
            field("Events", "comma-separated; empty = all", &form.events, Action::Events),
            field("Skills", "comma-separated, optional", &form.skills, Action::Skills),
            row![column![text("Log only").size(13), text("record deliveries without running the agent").size(11).color(p.muted)].spacing(2).width(Fill),
                toggler(form.deliver_only).on_toggle(move |v| act(Action::DeliverOnly(v))).size(16)].align_y(Alignment::Center),
            row![Space::new().width(Fill), button(text("Cancel").size(12)).padding([6, 12]).style(ui::quiet_button).on_press(act(Action::Cancel)),
                button(text("Create").size(12)).padding([6, 14]).on_press_maybe((!form.name.trim().is_empty()).then(|| act(Action::Create)))].spacing(8),
        ].spacing(10)).padding(14).style(move |_| ui::panel(p.card, 8.0, Some(p.border))));
    }
    let q = state.search.to_lowercase();
    let shown: Vec<&Webhook> = state.list.iter().filter(|w| q.is_empty() || [&w.name, &w.description].iter().any(|v| v.to_lowercase().contains(&q)) || w.events.iter().any(|e| e.contains(&q))).collect();
    if shown.is_empty() && state.form.is_none() {
        col = col.push(container(column![ui::icon("plug", 22, p.muted), text("No webhooks yet").size(14), text("Create one to let GitHub, Stripe or any service start the agent.").size(12).color(p.muted)].spacing(6).align_x(Alignment::Center)).padding(40).center_x(Fill));
    }
    for hook in shown {
        let events = if hook.events.is_empty() { "all events".to_owned() } else { hook.events.join(", ") };
        let mut card = column![
            row![text(hook.name.clone()).size(14).font(ui::medium()), text(if hook.deliver_only { "log only" } else { "runs agent" }).size(11).color(p.muted), Space::new().width(Fill),
                toggler(hook.enabled).on_toggle({ let n = hook.name.clone(); move |v| act(Action::Enable(n.clone(), v)) }).size(16)].spacing(10).align_y(Alignment::Center),
            text(url(&hook.name)).size(12).font(iced::Font::MONOSPACE).color(p.muted),
            text(format!("Events: {events}{}", if hook.skills.is_empty() { String::new() } else { format!(" · Skills: {}", hook.skills.join(", ")) })).size(12).color(p.muted),
        ].spacing(6);
        if !hook.description.is_empty() { card = card.push(text(hook.description.clone()).size(12)); }
        if !hook.prompt.is_empty() { card = card.push(container(text(hook.prompt.clone()).size(12)).padding([6, 9]).width(Fill).style(move |_| ui::panel(p.secondary, 4.0, None))); }
        for d in hook.deliveries.iter().take(3) { card = card.push(text(format!("{} · {} · {}", d.at.get(..19).unwrap_or(&d.at).replace('T', " "), if d.event.is_empty() { "—" } else { &d.event }, d.status)).size(11).color(p.muted)); }
        let mut actions = row![button(text("Copy URL").size(11)).style(ui::quiet_button).on_press(act(Action::Copy(url(&hook.name))))].spacing(6).align_y(Alignment::Center);
        if let Some(id) = &hook.session_id { actions = actions.push(button(text("Open chat").size(11)).style(ui::quiet_button).on_press(act(Action::OpenChat(id.clone())))); }
        actions = actions.push(Space::new().width(Fill));
        actions = if state.deleting.as_deref() == Some(&hook.name) {
            actions.push(text("Delete this webhook?").size(11).color(p.muted)).push(button(text("Delete").size(11)).padding([4, 10]).on_press(act(Action::ConfirmDelete)))
        } else { actions.push(button(text("Delete").size(11)).style(ui::quiet_button).on_press(act(Action::Delete(hook.name.clone())))) };
        card = card.push(actions);
        col = col.push(container(card).padding(14).width(Fill).style(move |_| ui::panel(p.card, 8.0, Some(p.border))));
    }
    scrollable(container(col).padding([8, 32]).max_width(920).center_x(Fill)).height(Fill).into()
}
