//! Bot Mode screens: port of plugins/hermes-bots roster pane, screen hero,
//! create dialog and edit-profile dialog. A bot row always opens that bot's
//! canonical "Bot Chat" (resolved by name in the library, every time).
use super::{Hermes, Message, Screen, ui};
use crate::appearance::text;
use hermes_library::Bot;
use iced::{Alignment, Border, Color, Element, Fill, Task,
    widget::{Space, button, column, container, row, rule, scrollable, text_editor, text_input}};

#[derive(Clone, Debug)]
pub(super) enum Action {
    Roster(bool), Open(String), New, Edit(String), Cancel,
    Name(String), Description(String), Emoji(String), Soul(text_editor::Action),
    Save, Delete(String), ConfirmDelete, Menu, Toasts,
    // Roster sections (user-sections.ts)
    NewSection(Option<String>), RenameSection(String), SectionName(String), SaveSection, DeleteSection(String),
    MoveSection(String, i32), RowMenu(String), MoveTo(String, Option<String>), SectionMenu(String),
    // Group chats
    NewGroup, GroupName(String), GroupMember(String, bool), SaveGroup, OpenGroup(String), DeleteGroup(String),
}

/// A user-made roster section; bots are assigned by name.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(super) struct Section { pub id: String, pub name: String }

#[derive(Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Preferences { activity_toasts: Option<bool>, sections: Vec<Section>, assigned: std::collections::BTreeMap<String, String> }

/// The New / Rename section dialog: (section being renamed, name, bot to file into it).
struct SectionDialog { renaming: Option<String>, name: String, bot: Option<String> }

#[derive(Default)]
pub(super) struct State {
    /// The sidebar shows the bot roster instead of sessions.
    pub roster: bool,
    pub bots: Vec<Bot>,
    form: Option<Form>,
    deleting: Option<String>,
    error: Option<String>,
    /// The "+" dropdown in the roster header.
    menu: bool,
    /// Bot activity notifications (the bell toggle). Persisted per library.
    pub toasts: bool,
    sections: Vec<Section>,
    assigned: std::collections::BTreeMap<String, String>,
    dialog: Option<SectionDialog>,
    row_menu: Option<String>,
    section_menu: Option<String>,
    pub groups: Vec<hermes_library::Group>,
    group_form: Option<(String, std::collections::BTreeSet<String>)>,
}

struct Form { editing: Option<String>, name: String, description: String, emoji: String, soul: text_editor::Content }

impl State {
    pub fn refresh(&mut self, library: Option<&hermes_library::Library>) {
        if let Some(library) = library {
            let prefs: Preferences = std::fs::read(library.root().join("bots-preferences.json")).ok()
                .and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
            self.toasts = prefs.activity_toasts.unwrap_or(true);
            self.sections = prefs.sections; self.assigned = prefs.assigned;
            match library.bots() { Ok(bots) => self.bots = bots, Err(e) => self.error = Some(format!("Could not read your bots: {e:#}")) }
            match library.groups() { Ok(groups) => self.groups = groups, Err(e) => self.error = Some(format!("Could not read your group chats: {e:#}")) }
        }
    }
    fn save(&self, library: &hermes_library::Library) -> Result<(), String> {
        let prefs = Preferences { activity_toasts: Some(self.toasts), sections: self.sections.clone(), assigned: self.assigned.clone() };
        std::fs::write(library.root().join("bots-preferences.json"), serde_json::to_vec_pretty(&prefs).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
    }
    /// The bot whose canonical chat is open, if any.
    pub fn active<'a>(&'a self, app: &'a Hermes) -> Option<&'a Bot> {
        let name = app.current.as_ref()?.bot.as_deref()?;
        self.bots.iter().find(|bot| bot.name == name)
    }
}

pub(super) fn update(app: &mut Hermes, action: Action) -> Task<Message> {
    let library = app.library.clone();
    let state = &mut app.bots;
    match action {
        Action::Roster(on) => {
            state.roster = on; state.menu = false; state.refresh(library.as_ref());
            if app.screen == Screen::Bots { app.screen = Screen::Chat; }
            if !on && app.current.as_ref().is_some_and(|c| c.bot.is_some()) { return app.update(Message::NewChat); }
        }
        Action::Open(name) => {
            let Some(library) = library else { return Task::none() };
            if app.running { state.error = Some("Stop the current response before switching chats.".into()); return Task::none(); }
            match library.bot_chat(&name) {
                Ok(summary) => { state.error = None; app.banner = None; app.open_chat(summary); app.screen = Screen::Chat;
                    return iced::widget::operation::focus(super::COMPOSER_ID); }
                Err(e) => state.error = Some(format!("Could not open this bot's chat: {e:#}")),
            }
        }
        Action::Menu => state.menu = !state.menu,
        Action::Toasts => {
            state.toasts = !state.toasts;
            if let Some(library) = &library && let Err(e) = state.save(library) { state.error = Some(format!("Could not save the notification setting: {e}")); }
        }
        Action::NewSection(bot) => { state.menu = false; state.row_menu = None; state.dialog = Some(SectionDialog { renaming: None, name: String::new(), bot }); }
        Action::RenameSection(id) => {
            state.section_menu = None;
            let name = state.sections.iter().find(|s| s.id == id).map(|s| s.name.clone()).unwrap_or_default();
            state.dialog = Some(SectionDialog { renaming: Some(id), name, bot: None });
        }
        Action::SectionName(v) => if let Some(d) = &mut state.dialog { d.name = v.chars().take(40).collect() },
        Action::SaveSection => {
            let Some(dialog) = state.dialog.take() else { return Task::none() };
            let name = dialog.name.trim().to_owned();
            if name.is_empty() { state.dialog = Some(dialog); return Task::none(); }
            match dialog.renaming {
                Some(id) => if let Some(section) = state.sections.iter_mut().find(|s| s.id == id) { section.name = name; },
                None => {
                    let id = format!("sec-{}", chrono::Utc::now().timestamp_millis());
                    if let Some(bot) = dialog.bot { state.assigned.insert(bot, id.clone()); }
                    state.sections.push(Section { id, name });
                }
            }
            if let Some(library) = &library && let Err(e) = state.save(library) { state.error = Some(e); }
        }
        Action::DeleteSection(id) => {
            state.section_menu = None;
            state.sections.retain(|s| s.id != id); state.assigned.retain(|_, s| *s != id);
            if let Some(library) = &library && let Err(e) = state.save(library) { state.error = Some(e); }
        }
        Action::MoveSection(id, step) => {
            state.section_menu = None;
            if let Some(i) = state.sections.iter().position(|s| s.id == id) {
                let j = (i as i32 + step).clamp(0, state.sections.len() as i32 - 1) as usize;
                state.sections.swap(i, j);
                if let Some(library) = &library && let Err(e) = state.save(library) { state.error = Some(e); }
            }
        }
        Action::RowMenu(name) => { state.section_menu = None; state.row_menu = if state.row_menu.as_deref() == Some(name.as_str()) { None } else { Some(name) }; }
        Action::SectionMenu(id) => { state.row_menu = None; state.section_menu = if state.section_menu.as_deref() == Some(id.as_str()) { None } else { Some(id) }; }
        Action::MoveTo(bot, section) => {
            state.row_menu = None;
            match section { Some(id) => { state.assigned.insert(bot, id); } None => { state.assigned.remove(&bot); } }
            if let Some(library) = &library && let Err(e) = state.save(library) { state.error = Some(e); }
        }
        Action::NewGroup => { state.menu = false; state.form = None; state.deleting = None; state.error = None; state.group_form = Some((String::new(), Default::default())); app.screen = Screen::Bots; }
        Action::GroupName(v) => if let Some(f) = &mut state.group_form { f.0 = v.chars().take(60).collect() },
        Action::GroupMember(name, on) => if let Some(f) = &mut state.group_form { if on { f.1.insert(name); } else { f.1.remove(&name); } },
        Action::SaveGroup => {
            let (Some(library), Some((name, members))) = (library.clone(), state.group_form.clone()) else { return Task::none() };
            match library.create_group(&name, &members.into_iter().collect::<Vec<_>>()) {
                Ok(group) => { state.group_form = None; state.refresh(Some(&library)); return update(app, Action::OpenGroup(group.id)); }
                Err(e) => state.error = Some(format!("{e:#}")),
            }
        }
        Action::OpenGroup(id) => {
            let Some(library) = library else { return Task::none() };
            if app.running { state.error = Some("Stop the current response before switching chats.".into()); return Task::none(); }
            match library.group_chat(&id) {
                Ok(summary) => { state.error = None; app.banner = None; app.open_chat(summary); app.screen = Screen::Chat; return iced::widget::operation::focus(super::COMPOSER_ID); }
                Err(e) => state.error = Some(format!("Could not open this group chat: {e:#}")),
            }
        }
        Action::DeleteGroup(id) => { state.deleting = Some(format!("group:{id}")); state.form = None; app.screen = Screen::Bots; }
        Action::New => {
            app.screen = Screen::Bots; state.menu = false; state.deleting = None; state.error = None;
            state.form = Some(Form { editing: None, name: String::new(), description: String::new(), emoji: "🤖".into(), soul: text_editor::Content::new() });
        }
        Action::Edit(name) => {
            app.screen = Screen::Bots; state.error = None;
            let Some(bot) = state.bots.iter().find(|b| b.name == name).cloned() else { return Task::none() };
            let soul = library.as_ref().and_then(|l| l.bot_soul(&name).ok()).unwrap_or_default();
            state.form = Some(Form { editing: Some(name), name: bot.display_name, description: bot.description, emoji: bot.emoji, soul: text_editor::Content::with_text(&soul) });
        }
        Action::Cancel => {
            if state.dialog.take().is_some() { return Task::none(); }
            state.group_form = None;
            state.form = None; state.deleting = None; state.error = None; state.menu = false; app.screen = Screen::Chat;
        }
        Action::Name(v) => if let Some(f) = &mut state.form { f.name = v },
        Action::Description(v) => if let Some(f) = &mut state.form { f.description = v },
        Action::Emoji(v) => if let Some(f) = &mut state.form { f.emoji = v.chars().take(4).collect() },
        Action::Soul(a) => if let Some(f) = &mut state.form { f.soul.perform(a) },
        Action::Save => {
            let (Some(library), Some(form)) = (library, &state.form) else { return Task::none() };
            let soul = form.soul.text();
            let result = match &form.editing {
                Some(name) => library.update_bot(name, &form.name, &form.description, &form.emoji)
                    .and_then(|bot| library.set_bot_soul(&bot.name, &soul).map(|_| bot)),
                None => library.create_bot(&form.name, &form.description, &soul, &form.emoji),
            };
            match result {
                Ok(bot) => { let created = form.editing.is_none(); state.form = None; state.error = None; state.refresh(Some(&library));
                    if created { return update(app, Action::Open(bot.name)); } app.screen = Screen::Chat; }
                Err(e) => state.error = Some(format!("{e:#}")),
            }
        }
        Action::Delete(name) => { state.deleting = Some(name); state.form = None; app.screen = Screen::Bots; }
        Action::ConfirmDelete => {
            let (Some(library), Some(name)) = (library, state.deleting.take()) else { return Task::none() };
            let open = app.current.as_ref().is_some_and(|c| c.bot.as_deref() == Some(name.as_str()));
            if open && app.running { state.error = Some("Stop the current response before deleting this bot.".into()); return Task::none(); }
            let result = match name.strip_prefix("group:") { Some(id) => library.delete_group(id), None => library.delete_bot(&name) };
            match result {
                Ok(()) => { state.refresh(Some(&library)); if open { app.screen = Screen::Bots; let _ = app.update(Message::NewChat); app.screen = Screen::Bots; } }
                Err(e) => state.error = Some(format!("{e:#}")),
            }
        }
    }
    Task::none()
}

fn avatar(bot: &Bot, size: u16, p: ui::Palette) -> Element<'static, Message> {
    let glyph = if bot.emoji.is_empty() { bot.display_name.chars().next().map(|c| c.to_uppercase().to_string()).unwrap_or_default() } else { bot.emoji.clone() };
    container(text(glyph).size(size as f32 * 0.55)).width(size as f32).height(size as f32).center_x(size as f32).center_y(size as f32)
        .style(move |_| ui::panel(Color { a: 0.12, ..p.accent }, size as f32 / 2.0, None)).into()
}

/// The roster that replaces the session list while the BOTS tab is active
/// (roster-pane-toolbar.tsx + roster-pane-content.tsx).
pub(super) fn sidebar(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark); let state = &app.bots;
    let active = state.active(app).map(|b| b.name.clone());
    let tool = |glyph: &'static str, message: Message| button(ui::icon(glyph, 14, p.muted)).padding(4).style(ui::quiet_button).on_press(message);
    let header = row![
        text("BOTS").size(11).font(ui::medium()).color(Color { a: 0.55, ..p.foreground }),
        Space::new().width(Fill),
        iced::widget::tooltip(tool(if state.toasts { "bell" } else { "bell-slash" }, Message::Bot(Action::Toasts)),
            text(if state.toasts { "Bot activity notifications on" } else { "Bot activity notifications off" }).size(11), iced::widget::tooltip::Position::Bottom),
        iced::widget::tooltip(tool("add", Message::Bot(Action::Menu)), text("New bot or group chat").size(11), iced::widget::tooltip::Position::Bottom),
    ].spacing(2).align_y(Alignment::Center).padding(iced::Padding { top: 10.0, right: 10.0, bottom: 6.0, left: 10.0 });
    let mut col = column![header];
    if state.menu {
        let item = |glyph: &'static str, label: &'static str, message: Option<Message>| button(row![ui::icon(glyph, 13, p.muted), text(label).size(12)].spacing(8).align_y(Alignment::Center))
            .padding([6, 8]).width(Fill).style(ui::quiet_button).on_press_maybe(message);
        col = col.push(container(column![
            item("hubot", "New bot", Some(Message::Bot(Action::New))),
            item("organization", "New group chat", (state.bots.len() >= 2).then_some(Message::Bot(Action::NewGroup))),
            rule::horizontal(1),
            item("new-folder", "New section", Some(Message::Bot(Action::NewSection(None)))),
        ].spacing(1)).padding(4).width(Fill).style(move |_| ui::panel(p.card, 6.0, Some(p.border))));
    }
    if state.bots.is_empty() {
        col = col.push(container(column![ui::icon("hubot", 22, Color { a: 0.4, ..p.foreground }), text("No bots yet").size(13).font(ui::medium()),
            text("Create your first bot.").size(12).color(p.muted)].spacing(6).align_x(Alignment::Center)).center_x(Fill).center_y(Fill).height(Fill));
        return col.into();
    }
    if let Some(dialog) = &state.dialog {
        col = col.push(container(column![
            text(if dialog.renaming.is_some() { "Rename section" } else { "New section" }).size(12).font(ui::medium()),
            text_input("e.g. Clients", &dialog.name).size(13).padding([6, 8]).on_input(|v| Message::Bot(Action::SectionName(v))).on_submit(Message::Bot(Action::SaveSection)),
            row![Space::new().width(Fill), button(text("Cancel").size(12)).padding([4, 10]).style(ui::quiet_button).on_press(Message::Bot(Action::Cancel)),
                button(text(if dialog.renaming.is_some() { "Rename" } else { "Create" }).size(12)).padding([4, 10]).on_press_maybe((!dialog.name.trim().is_empty()).then_some(Message::Bot(Action::SaveSection)))].spacing(6),
        ].spacing(8)).padding(8).style(move |_| ui::panel(p.card, 6.0, Some(p.border))));
    }
    let section_of = |bot: &Bot| state.assigned.get(&bot.name).filter(|id| state.sections.iter().any(|s| &s.id == *id)).cloned();
    let mut groups: Vec<(Option<&Section>, Vec<&Bot>)> = state.sections.iter().map(|s| (Some(s), state.bots.iter().filter(|b| section_of(b).as_deref() == Some(s.id.as_str())).collect())).collect();
    groups.push((None, state.bots.iter().filter(|b| section_of(b).is_none()).collect()));
    let mut list = column![].spacing(1);
    for (section, bots) in groups {
        if !state.sections.is_empty() {
            let label = section.map_or("Unassigned".to_owned(), |s| s.name.clone());
            let mut head = row![text(label.to_uppercase()).size(11).font(ui::medium()).color(Color { a: 0.55, ..p.foreground }), Space::new().width(Fill)].align_y(Alignment::Center);
            if let Some(section) = section { head = head.push(button(ui::icon("ellipsis", 12, p.muted)).padding(2).style(ui::quiet_button).on_press(Message::Bot(Action::SectionMenu(section.id.clone())))); }
            list = list.push(container(head).padding(iced::Padding { top: 10.0, right: 4.0, bottom: 4.0, left: 8.0 }));
            if let Some(section) = section && state.section_menu.as_deref() == Some(section.id.as_str()) {
                let id = section.id.clone();
                let item = |label: &'static str, message: Message| button(text(label).size(12)).padding([5, 8]).width(Fill).style(ui::quiet_button).on_press(message);
                list = list.push(container(column![item("Rename…", Message::Bot(Action::RenameSection(id.clone()))), item("Move up", Message::Bot(Action::MoveSection(id.clone(), -1))),
                    item("Move down", Message::Bot(Action::MoveSection(id.clone(), 1))), item("Delete section", Message::Bot(Action::DeleteSection(id)))].spacing(1))
                    .padding(4).style(move |_| ui::panel(p.card, 6.0, Some(p.border))));
            }
            if section.is_some() && bots.is_empty() { list = list.push(container(text("Right-click a bot to move it here").size(11).color(p.muted)).padding([2, 8])); }
        }
        for bot in bots {
        let selected = active.as_deref() == Some(bot.name.as_str()) && app.screen == Screen::Chat;
        let subtitle = if bot.description.is_empty() { format!("@{}", bot.name) } else { bot.description.clone() };
        let row_content = row![avatar(bot, 24, p), column![text(bot.display_name.clone()).size(13), text(clip(&subtitle, 30)).size(11).color(p.muted)].spacing(1)]
            .spacing(9).align_y(Alignment::Center);
        let bot_row = button(row_content).padding([5, 8]).width(Fill).on_press(Message::Bot(Action::Open(bot.name.clone())))
            .style(move |t, s| { let mut style = ui::quiet_button(t, s); if selected { style.background = Some(Color { a: 0.08, ..p.accent }.into()); } style });
        list = list.push(iced::widget::mouse_area(bot_row).on_right_press(Message::Bot(Action::RowMenu(bot.name.clone()))));
        if state.row_menu.as_deref() == Some(bot.name.as_str()) {
            let name = bot.name.clone();
            let item = |label: String, message: Message| button(text(label).size(12)).padding([5, 8]).width(Fill).style(ui::quiet_button).on_press(message);
            let mut menu = column![item("Edit…".into(), Message::Bot(Action::Edit(name.clone())))].spacing(1);
            menu = menu.push(container(text("Move to section").size(11).color(p.muted)).padding([4, 8]));
            for section in &state.sections { menu = menu.push(item(format!("  {}", section.name), Message::Bot(Action::MoveTo(name.clone(), Some(section.id.clone()))))); }
            menu = menu.push(item("  New section…".into(), Message::Bot(Action::NewSection(Some(name.clone())))));
            if section_of(bot).is_some() { menu = menu.push(item("Remove from section".into(), Message::Bot(Action::MoveTo(name.clone(), None)))); }
            menu = menu.push(rule::horizontal(1)).push(item("Delete…".into(), Message::Bot(Action::Delete(name))));
            list = list.push(container(menu).padding(4).style(move |_| ui::panel(p.card, 6.0, Some(p.border))));
        }
        }
    }
    if !state.groups.is_empty() {
        list = list.push(container(text("GROUP CHATS").size(11).font(ui::medium()).color(Color { a: 0.55, ..p.foreground })).padding(iced::Padding { top: 12.0, right: 4.0, bottom: 4.0, left: 8.0 }));
        let open = app.current.as_ref().and_then(|c| c.bot.clone());
        for group in &state.groups {
            let selected = open.as_deref() == Some(format!("group:{}", group.id).as_str()) && app.screen == Screen::Chat;
            let names: Vec<String> = group.members.iter().map(|m| state.bots.iter().find(|b| &b.name == m).map_or(m.clone(), |b| b.display_name.clone())).collect();
            let row_content = row![ui::icon("organization", 14, p.muted), column![text(group.name.clone()).size(13), text(clip(&names.join(", "), 30)).size(11).color(p.muted)].spacing(1)].spacing(9).align_y(Alignment::Center);
            let button = button(row_content).padding([5, 8]).width(Fill).on_press(Message::Bot(Action::OpenGroup(group.id.clone())))
                .style(move |t, s| { let mut style = ui::quiet_button(t, s); if selected { style.background = Some(Color { a: 0.08, ..p.accent }.into()); } style });
            list = list.push(iced::widget::mouse_area(button).on_right_press(Message::Bot(Action::DeleteGroup(group.id.clone()))));
        }
    }
    col.push(scrollable(container(list).padding([0, 6])).height(Fill)).into()
}

fn clip(value: &str, max: usize) -> String {
    if value.chars().count() <= max { value.into() } else { format!("{}…", value.chars().take(max - 1).collect::<String>()) }
}

/// Main area for the BOTS tab: hero, roster cards, and the create/edit/delete dialogs.
pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark); let state = &app.bots;
    let mut body = column![].spacing(18).max_width(820);
    if let Some(error) = &state.error {
        body = body.push(container(text(error.as_str()).size(12)).padding([8, 12]).width(Fill).style(move |_| ui::panel(Color { a: 0.08, ..p.accent }, 6.0, Some(p.border))));
    }
    if let Some((name, members)) = &state.group_form {
        body = body.push(text("New group chat").size(22).font(ui::medium()))
            .push(column![text("Name").size(12).font(ui::medium()), text_input("Research team", name).on_input(|v| Message::Bot(Action::GroupName(v))).size(13).padding(8)].spacing(6))
            .push(text("Members · each one replies in turn, in its own voice").size(12).font(ui::medium()));
        for bot in &state.bots {
            let on = members.contains(&bot.name); let id = bot.name.clone();
            body = body.push(iced::widget::checkbox(on).label(bot.display_name.clone()).on_toggle(move |v| Message::Bot(Action::GroupMember(id.clone(), v))).text_size(13));
        }
        body = body.push(row![Space::new().width(Fill), button(text("Cancel").size(13)).padding([7, 14]).style(ui::quiet_button).on_press(Message::Bot(Action::Cancel)),
            button(text("Create group chat").size(13)).padding([7, 14]).on_press_maybe((!name.trim().is_empty() && members.len() >= 2).then_some(Message::Bot(Action::SaveGroup)))].spacing(8));
        return scrollable(container(body).padding([28, 40]).center_x(Fill)).height(Fill).into();
    }
    if let Some(form) = &state.form {
        let title = if form.editing.is_some() { "Edit bot" } else { "New bot" };
        let field = |label: &'static str, hint: &'static str| column![text(label).size(12).font(ui::medium()), text(hint).size(11).color(p.muted)].spacing(2);
        let can_save = !form.name.trim().is_empty();
        body = body.push(text(title).size(22).font(ui::medium()))
            .push(row![
                column![field("Icon", "An emoji."), text_input("🤖", &form.emoji).on_input(|v| Message::Bot(Action::Emoji(v))).size(16).padding(8).width(70)].spacing(6),
                column![field("Name", "How you and other agents address it."), text_input("Research Helper", &form.name).on_input(|v| Message::Bot(Action::Name(v))).size(13).padding(8)].spacing(6).width(Fill),
            ].spacing(14))
            .push(column![field("Description", "One line shown in the roster."), text_input("Finds and summarizes papers", &form.description).on_input(|v| Message::Bot(Action::Description(v))).size(13).padding(8)].spacing(6))
            .push(column![field(if form.editing.is_some() { "SOUL.md" } else { "Instructions" }, "Its personality and how it should work. Saved to the bot's SOUL.md."),
                text_editor(&form.soul).on_action(|a| Message::Bot(Action::Soul(a))).height(200).size(13).padding(10)].spacing(6))
            .push(row![Space::new().width(Fill), button(text("Cancel").size(13)).padding([7, 14]).style(ui::quiet_button).on_press(Message::Bot(Action::Cancel)),
                button(text(if form.editing.is_some() { "Save" } else { "Create bot" }).size(13)).padding([7, 14]).on_press_maybe(can_save.then_some(Message::Bot(Action::Save)))].spacing(8));
        return scrollable(container(body).padding([28, 40]).center_x(Fill)).height(Fill).into();
    }
    if let Some(name) = &state.deleting {
        let label = state.bots.iter().find(|b| &b.name == name).map_or(name.clone(), |b| b.display_name.clone());
        body = body.push(text(format!("Delete {label}?")).size(20).font(ui::medium()))
            .push(text("Its profile, personality, memory and settings are removed. Its chat is moved to Archived sessions.").size(13).color(p.muted))
            .push(row![button(text("Cancel").size(13)).padding([7, 14]).style(ui::quiet_button).on_press(Message::Bot(Action::Cancel)),
                button(text("Delete bot").size(13)).padding([7, 14]).on_press(Message::Bot(Action::ConfirmDelete))].spacing(8));
        return container(body).padding([28, 40]).center_x(Fill).into();
    }
    body = body.push(row![column![text("Bots").size(40).font(crate::appearance::DISPLAY_FONT).line_height(1.1),
            text("Named agents with their own personality, memory and settings. Each bot keeps one ongoing chat.").size(13).color(p.muted)].spacing(4).width(Fill),
        button(row![ui::icon("add", 13, Color::WHITE), text("New bot").size(13).color(Color::WHITE)].spacing(6).align_y(Alignment::Center)).padding([7, 14])
            .on_press(Message::Bot(Action::New)).style(move |_, s| button::Style { background: Some(Color { a: if s == button::Status::Hovered { 0.85 } else { 1.0 }, ..p.accent }.into()),
                text_color: Color::WHITE, border: Border { radius: 6.0.into(), ..Default::default() }, ..Default::default() })].align_y(Alignment::Center));
    if state.bots.is_empty() {
        body = body.push(container(column![ui::icon("robot", 28, p.muted), text("No bots yet").size(15).font(ui::medium()),
            text("Create a bot to give an agent a name, a personality and a chat of its own.").size(13).color(p.muted)].spacing(8).align_x(Alignment::Center))
            .padding(40).center_x(Fill).style(move |_| ui::panel(p.card, 10.0, Some(p.border))));
    }
    for chunk in state.bots.chunks(2) {
        let mut line = row![].spacing(12);
        for bot in chunk {
            let card = column![
                row![avatar(bot, 36, p), column![text(bot.display_name.clone()).size(15).font(ui::medium()), text(format!("@{}", bot.name)).size(11).color(p.muted)].spacing(2)].spacing(10).align_y(Alignment::Center),
                text(if bot.description.is_empty() { "No description".to_owned() } else { bot.description.clone() }).size(12).color(p.muted),
                rule::horizontal(1),
                row![button(text("Chat").size(12)).padding([4, 12]).on_press(Message::Bot(Action::Open(bot.name.clone()))), Space::new().width(Fill),
                    button(text("Edit").size(12)).padding([4, 10]).style(ui::quiet_button).on_press(Message::Bot(Action::Edit(bot.name.clone()))),
                    button(text("Delete").size(12)).padding([4, 10]).style(ui::quiet_button).on_press(Message::Bot(Action::Delete(bot.name.clone())))].spacing(4).align_y(Alignment::Center),
            ].spacing(10);
            line = line.push(container(card).padding(14).width(Fill).style(move |_| ui::panel(p.card, 10.0, Some(p.border))));
        }
        if chunk.len() == 1 { line = line.push(Space::new().width(Fill)); }
        body = body.push(line);
    }
    scrollable(container(body).padding([28, 40]).center_x(Fill)).height(Fill).into()
}
