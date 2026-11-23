//! Native projection of the official desktop shell (apps/desktop/src).
//! Palette: shared/src/theme-presets.ts; geometry: shell/titlebar.ts and styles.css.
use super::*;
use crate::appearance::text;
use hermes_engine::Status;
use iced::{Alignment, Background, Border, Color, Fill, Font, Length, Padding,
    widget::{button, column, container, mouse_area, row, rule, scrollable, stack, text_input, tooltip, Space}};

#[derive(Clone, Copy)]
pub(super) struct Palette {
    pub background: Color, pub sidebar: Color, pub foreground: Color, pub secondary: Color,
    pub muted: Color, pub accent: Color, pub border: Color, pub card: Color,
}
const fn hex(value: u32) -> Color {
    Color::from_rgb(((value >> 16) & 255) as f32 / 255.0, ((value >> 8) & 255) as f32 / 255.0, (value & 255) as f32 / 255.0)
}
pub(super) fn palette(dark: bool) -> Palette {
    appearance::palette(dark)
}
pub(super) fn theme_for(dark: bool) -> Theme {
    let p = palette(dark);
    Theme::custom(if dark { "Nous Dark" } else { "Nous Light" }, iced::theme::Palette {
        background: p.background, text: p.foreground, primary: shell_art::accent().fill(),
        success: if dark { hex(0x3fb950) } else { hex(0x1f8a65) }, danger: if dark { hex(0xf85149) } else { hex(0xcf222e) }, warning: hex(0xc08532),
    })
}

impl Hermes {
    pub(super) fn view_window(&self, window: iced::window::Id) -> Element<'_, Message> {
        if self.overlays.pet == Some(window) { return super::overlays::pet_view(self); }
        if self.overlays.wake == Some(window) { return super::overlays::wake_view(self); }
        // The HUD paints its own (translucent) band over the native frost.
        if self.hud.window == Some(window) && super::hud_native::FADES { return self.hud_view(); }
        let p = palette(self.dark);
        let page = if self.quick_entry.window == Some(window) { super::quick_entry::view(self) }
            else if self.hud.window == Some(window) { self.hud_view() } else { self.view() };
        container(page).width(Fill).height(Fill).style(move |_| container::Style { background: Some(p.background.into()), ..Default::default() }).into()
    }
    pub(super) fn view(&self) -> Element<'_, Message> {
        let p = palette(self.dark);
        let overlay = matches!(self.screen, Screen::Provider | Screen::Import | Screen::Workspaces | Screen::About | Screen::CommandCenter | Screen::Agents | Screen::Cron);
        let body: Element<'_, _> = match self.screen {
            Screen::Tools if !overlay => self.capabilities(),
            Screen::Messaging => column![page_title("Messaging"), self.messaging.view(self.dark).map(Message::Messaging)].height(Fill).into(),
            Screen::Webhooks => column![page_title("Webhooks"), super::webhooks::view(self)].height(Fill).into(),
            Screen::Profiles => column![page_title("Profiles"), super::profiles::view(self)].height(Fill).into(),
            Screen::Starmap => column![page_title("Star map"), super::starmap::view(self)].height(Fill).into(),
            Screen::Pets => column![page_title("Pets"), super::pets::view(self)].height(Fill).into(),
            Screen::Artifacts => self.artifacts(),
            Screen::Bots => bots::view(self),
            _ => self.chat(),
        };
        let body = if self.backdrop && let Some(handle) = &self.backdrop_image {
            let background=widget::image(handle.clone()).width(Fill).height(Fill).content_fit(iced::ContentFit::Cover).opacity(0.025_f32);
            stack![background,body].width(Fill).height(Fill).into()
        } else {body};
        let portal = self.artwork.is_some(); let dark = self.dark;
        // The stage is a veil over the artwork (--portal-surface-stage-veil-conversation).
        let chat: Element<'_, Message> = container(column![self.header(overlay), body].width(Fill).height(Fill))
            .style(move |_| container::Style { background: portal.then(|| shell_art::roles::stage_veil(dark).into()), ..Default::default() }).into();
        let main: Element<'_, Message> = if let Some(preview) = self.files.preview_view() {
            row![container(chat).width(iced::Length::FillPortion(1)), hairline(p),
                container(preview.map(Message::Files)).width(iced::Length::FillPortion(1))].height(Fill).into()
        } else { chat };
        // Settings is a page in the redesign: its nav panel and column replace the chat columns.
        let settings_page = matches!(self.screen, Screen::Provider | Screen::Import);
        let mut panes = row![].height(Fill);
        if settings_page {
            panes = panes.push(if self.screen == Screen::Import { settings::imports(self) } else { settings::provider(self) });
        } else {
        if !self.flipped && self.sidebar_open { panes = panes.push(self.sidebar()).push(hairline(p)); }
        if self.flipped && let Some(pane) = self.right_pane { panes = panes.push(self.right_sidebar(pane)).push(hairline(p)); }
        panes = panes.push(main);
        if !self.flipped && let Some(pane) = self.right_pane { panes = panes.push(hairline(p)).push(self.right_sidebar(pane)); }
        if self.flipped && self.sidebar_open { panes = panes.push(hairline(p)).push(self.sidebar()); }
        }
        let base = container(column![panes, self.statusbar()]).width(Fill).height(Fill)
            .style(move |_| container::Style { background: (!portal).then(|| p.background.into()), text_color: Some(p.foreground), ..Default::default() });
        let mut layers = stack![];
        if portal && let Some(artwork) = self.artwork { layers = layers.push(shell_art::layer(shell_art::for_mode(artwork, dark), dark)); }
        let mut layers = layers.push(base);
        if overlay && !settings_page {
            let card = match self.screen {
                Screen::Provider => settings::provider(self), Screen::Import => settings::imports(self),
                Screen::CommandCenter => command_center::view(self), Screen::Agents => command_center::agents(self),
                Screen::Cron => self.scheduled.view().map(Message::Scheduled),
                Screen::Workspaces => self.workspaces(), _ => self.about(),
            };
            layers = layers.push(container(Space::new().width(Fill).height(Fill)).style(|_| container::Style {
                background: Some(Color { a: 0.22, ..Color::BLACK }.into()), ..Default::default()
            })).push(container(card).width(Fill).height(Fill).padding(Padding { top: 32.0, right: 46.0, bottom: 46.0, left: 46.0 }));
        } else if self.layout_menu {
            let menu = container(column![text("Layout").size(12).font(medium()),
                menu_action("layout-sidebar-left", "Sessions", Message::ToggleSidebar, p),
                menu_action("files", "Files", Message::Navigate(Screen::Files), p),
                menu_action("git-branch", "Review", Message::Navigate(Screen::Git), p),
                menu_action("terminal", "Terminal", Message::Navigate(Screen::Terminal), p),
                menu_action("arrow-swap", "Flip panes", Message::FlipPanes, p),
            ].spacing(4)).padding(8).width(200).style(move |_| panel(p.background, 6.0, Some(p.border)));
            layers = layers.push(container(menu).width(Fill).height(Fill).align_right(Fill).padding(Padding { top: 34.0, right: 14.0, bottom: 0.0, left: 0.0 }));
        }
        if !overlay && let Some(menu) = self.statusbar.menu {
            layers = layers.push(mouse_area(container(self.statusbar_popup(menu)).width(Fill).height(Fill).align_right(Fill).align_bottom(Fill)
                .padding(Padding { top: 0.0, right: 6.0, bottom: 22.0, left: 6.0 })).on_press(Message::Statusbar(statusbar::Action::Close)));
        }
        if self.onboarding.open { layers = layers.push(onboarding::view(self)); }
        if self.commands.is_open() { layers = layers.push(self.commands.view(self.dark).map(Message::Commands)); }
        if self.models.picker {
            layers=layers.push(container(Space::new().width(Fill).height(Fill)).style(|_|container::Style {background:Some(Color {a:0.22,..Color::BLACK}.into()),..Default::default()}))
                .push(container(models::picker(self)).width(Fill).height(Fill).center_x(Fill).center_y(Fill).padding(24));
        }
        if let Some(anchor)=self.reasoning.anchor && self.screen==Screen::Chat {layers=layers.push(reasoning::overlay(self,anchor,false));}
        layers.width(Fill).height(Fill).into()
    }

    /// Sidebar › Cron jobs (app/chat/sidebar/cron-jobs-section.tsx): soonest
    /// first, a state dot and countdown per job, Run now / Manage on hover, the
    /// full menu on right-click, and a peek at the last five runs.
    fn cron_section(&self) -> Element<'_, Message> {
        let p = palette(self.dark);
        let ui_state = &self.cron_ui;
        let now = chrono::Utc::now();
        let next = |j: &hermes_engine::cron::Job| j.next_run_at.as_deref().and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok()).map(|t| t.with_timezone(&chrono::Utc));
        let mut jobs: Vec<&hermes_engine::cron::Job> = self.scheduled.jobs().iter().collect();
        jobs.sort_by(|a, b| match (next(a), next(b)) { (Some(x), Some(y)) if x != y => x.cmp(&y), (None, Some(_)) => std::cmp::Ordering::Greater, (Some(_), None) => std::cmp::Ordering::Less, _ => a.title().cmp(b.title()) });
        let header = button(row![text(spaced("CRON JOBS")).size(10).font(medium()).color(p.accent), text(jobs.len().to_string()).size(10).color(p.muted),
            icon(if ui_state.open { "chevron-down" } else { "chevron-right" }, 10, p.muted)].spacing(6).align_y(Alignment::Center))
            .padding([6, 4]).style(button::text).on_press(Message::Cron(CronUi::Toggle));
        let mut col = column![header].spacing(1);
        if !ui_state.open { return col.into(); }
        let relative = |secs: i64| { let a = secs.abs(); if a < 60 { "now".to_owned() } else if a < 3600 { format!("{}m", a / 60) } else if a < 86_400 { format!("{}h", a / 3600) } else { format!("{}d", a / 86_400) } };
        for job in jobs.iter().take(ui_state.visible) {
            let id = job.id.clone();
            let running = self.scheduled.running_for(&id);
            let paused = job.state == "paused" || !job.enabled;
            let failed = job.last_status.as_deref().is_some_and(|s| matches!(s, "error" | "failed")) || job.state == "error";
            let (dot, meta) = if let Some(secs) = running { (p.accent, format!("Running · {}", if secs >= 60 { format!("{}m", secs / 60) } else { format!("{secs}s") })) }
                else if job.state == "completed" { (p.muted, "Completed".into()) }
                else if paused { (Color::from_rgb8(0xc2, 0x7c, 0x0e), "Paused".into()) }
                else if failed { (Color::from_rgb8(0xd1, 0x43, 0x43), "Failed".into()) }
                else if let Some(t) = next(job) { let d = (t - now).num_seconds(); if d < -60 { (Color::from_rgb8(0xd1, 0x43, 0x43), format!("Overdue {}", relative(d))) } else { (Color::from_rgb8(0x1f, 0x8a, 0x65), format!("in {}", relative(d))) } }
                else { (p.muted, "—".into()) };
            let hovered = ui_state.hover.as_deref() == Some(id.as_str());
            let peeking = ui_state.peek.as_deref() == Some(id.as_str());
            let trailing: Element<'_, Message> = if hovered {
                row![tooltip(button(icon("zap", 11, p.muted)).padding(3).style(quiet_button).on_press_maybe(running.is_none().then(|| Message::Cron(CronUi::Trigger(id.clone())))), text("Trigger now").size(11), tooltip::Position::Top),
                    tooltip(button(icon("watch", 11, p.muted)).padding(3).style(quiet_button).on_press(Message::Cron(CronUi::Manage(id.clone()))), text("Manage").size(11), tooltip::Position::Top)].spacing(1).into()
            } else { text(meta).size(10).color(p.muted).into() };
            let dot_size = if running.is_some() { 7.0 } else { 5.0 };
            let body = button(row![container(Space::new().width(dot_size).height(dot_size)).style(move |_| panel(dot, 4.0, None)),
                text(short(job.title(), 22)).size(13).width(Fill),
                icon(if peeking { "chevron-down" } else { "chevron-right" }, 10, if hovered || peeking { p.muted } else { Color::TRANSPARENT })].spacing(8).align_y(Alignment::Center))
                .padding([4, 6]).width(Fill).style(button::text).on_press(Message::Cron(CronUi::Peek(id.clone())));
            let line = row![body, trailing].spacing(4).align_y(Alignment::Center).height(27);
            col = col.push(mouse_area(container(line).style(move |_| panel(if hovered { Color { a: 0.05, ..p.foreground } } else { Color::TRANSPARENT }, 5.0, None)))
                .on_enter(Message::Cron(CronUi::Hover(Some(id.clone())))).on_exit(Message::Cron(CronUi::Hover(None))).on_right_press(Message::Cron(CronUi::Menu(id.clone()))));
            if ui_state.menu.as_deref() == Some(id.as_str()) {
                col = col.push(container(column![
                    menu_action("zap", "Trigger now", Message::Cron(CronUi::Trigger(id.clone())), p),
                    menu_action(if paused { "play" } else { "debug-pause" }, if paused { "Resume" } else { "Pause" }, Message::Cron(CronUi::Pause(id.clone(), !paused)), p),
                    menu_action("watch", "Manage", Message::Cron(CronUi::Manage(id.clone())), p),
                    menu_action("trash", "Delete", Message::Cron(CronUi::Delete(id.clone())), p),
                ].spacing(1)).padding(5).width(Fill).style(move |_| panel(p.background, 5.0, Some(p.border))));
            }
            if peeking {
                let runs = self.scheduled.recent_runs(&id, 5);
                let mut peek = column![].spacing(1).padding(Padding { top: 0.0, right: 0.0, bottom: 4.0, left: 22.0 });
                if runs.is_empty() { peek = peek.push(text("No runs yet").size(11).color(p.muted)); }
                for (_, at) in runs {
                    let when = chrono::DateTime::parse_from_rfc3339(&at).map(|t| t.with_timezone(&chrono::Local).format("%a %b %-d, %H:%M").to_string()).unwrap_or(at);
                    peek = peek.push(button(text(when).size(11).color(p.muted)).padding([2, 6]).width(Fill).style(quiet_button).on_press(Message::OpenRoutineChat(id.clone())));
                }
                col = col.push(peek);
            }
        }
        if jobs.len() > ui_state.visible {
            col = col.push(button(text(format!("Show {} more", (jobs.len() - ui_state.visible).min(10))).size(11).color(p.muted)).padding([3, 6]).style(button::text).on_press(Message::Cron(CronUi::More)));
        }
        container(col).padding(Padding { top: 10.0, right: 0.0, bottom: 6.0, left: 0.0 }).into()
    }

    fn sidebar(&self) -> Element<'_, Message> {
        let p = palette(self.dark);
        let top = row![Space::new().width(75), icon_button("layout-sidebar-left", "Hide sidebar", Message::ToggleSidebar, p), Space::new().width(Fill)].height(34).align_y(Alignment::Center);
        let mut new_session = button(row![icon("robot", 16, p.muted), text("New session").size(13), Space::new().width(Fill), keycap("⌘", p), keycap("N", p)].spacing(8).align_y(Alignment::Center))
            .height(28).padding([0, 8]).width(Fill).style(quiet_button);
        if !self.running { new_session = new_session.on_press(Message::NewChat); }
        let nav = column![new_session,
            self.nav("symbol-misc", "Capabilities", Screen::Tools), self.nav("comment", "Messaging", Screen::Messaging),
            self.nav("files", "Artifacts", Screen::Artifacts), self.nav("watch", "Routines", Screen::Cron), self.nav("plug", "Webhooks", Screen::Webhooks), self.nav("account", "Profiles", Screen::Profiles), self.nav("sparkle", "Star map", Screen::Starmap),
        ].spacing(1);
        let search = row![icon("search", 14, p.muted), text_input("Search sessions…", &self.search).id("hermes-session-search").on_input(Message::Search).size(13).padding([5, 0]).style(search_input)]
            .spacing(7).align_y(Alignment::Center);
        let mut pinned = column![section_label("PINNED", None, p)].spacing(2);
        let mut chats = column![].spacing(1);
        let query = self.search.to_lowercase();
        let visible: Vec<_> = self.sessions.iter().filter(|s| {
            !(self.fresh_draft && self.current.as_ref().is_some_and(|current| current.id == s.id))
        }).collect();
        if visible.is_empty() { chats = chats.push(container(text(if query.is_empty() { "No sessions yet" } else { "No sessions match this search." }).size(12).color(p.muted)).padding([10, 8])); }
        // Date dividers apply to the flat recents list, not to pins or search results.
        let recents: Vec<_> = visible.iter().filter(|s| !s.pinned).copied().collect();
        let labels = if query.is_empty() {
            let now = chrono::Utc::now().timestamp_millis();
            date_groups::dividers(&recents.iter().map(|s| date_groups::millis(s.recency())).collect::<Vec<_>>(), now)
        } else { vec![None; recents.len()] };
        for summary in visible.iter().filter(|s| s.pinned) { pinned = pinned.push(self.session_row(summary)); }
        for (summary, label) in recents.into_iter().zip(labels) {
            if let Some(label) = label {
                chats = chats.push(container(text(label).size(11).color(p.muted)).padding(Padding { top: 10.0, right: 8.0, bottom: 3.0, left: 8.0 }));
            }
            chats = chats.push(self.session_row(summary));
        }
        if !self.sessions.iter().any(|summary| summary.pinned) {
            pinned = pinned.push(container(row![icon("pinned", 11, p.muted), text("Shift-click a chat to pin").size(12).color(p.muted)].spacing(7)).padding([4, 8]));
        }
        let count = format!("{}", self.sessions.iter().filter(|s| !s.pinned && !(self.fresh_draft && self.current.as_ref().is_some_and(|current| current.id == s.id))).count());
        let sessions_header = row![section_label(if self.search.is_empty() { "SESSIONS" } else { "RESULTS" }, Some(count), p),
            Space::new().width(Fill), icon_button("add", "New project", Message::Navigate(Screen::Workspaces), p),
            icon_button("list-filter", "Session filters", Message::ToggleSessionFilter, p)].align_y(Alignment::Center);
        let rail = row![
            button(icon("home", 14, p.foreground)).on_press(Message::SelectWorkspace(None)).padding(4).style(move |theme, status| nav_button(theme, status, true)),
            Space::new().width(Fill),
            icon_button("add", "New project", Message::Navigate(Screen::Workspaces), p),
            icon_button("cloud", "Gateways", Message::SettingsView(settings::View::Gateways), p),
            icon_button("ellipsis", "Import sessions", Message::Navigate(Screen::Import), p),
            icon_button("plug", "Backend", Message::Statusbar(statusbar::Action::Backend), p),
        ].spacing(2).align_y(Alignment::Center).height(29);
        let roster = self.bots.roster;
        let tab = |label: &'static str, active: bool, message: Message| button(column![
                container(text(spaced(label)).size(11).font(medium()).color(if active { p.foreground } else { p.muted })).padding([0, 8]).height(26).center_y(26),
                container(Space::new().height(2)).width(Fill).style(move |_| panel(if active { p.accent } else { Color::TRANSPARENT }, 0.0, None))].spacing(0))
            .on_press(message).padding(0).width(Length::Shrink).style(quiet_button);
        let tabs = row![tab("SESSIONS", !roster, Message::Bot(bots::Action::Roster(false))), tab("BOTS", roster, Message::Bot(bots::Action::Roster(true)))]
            .height(28).spacing(0).align_y(Alignment::Start);
        if roster {
            let body = column![top, tabs, bots::sidebar(self), container(rail).padding([2, 12])];
            return container(body).width(SIDEBAR_WIDTH).height(Fill).style(move |_| panel(p.sidebar, 0.0, None)).into();
        }
        let no_sessions = self.search.is_empty() && self.archive_filter == ArchiveFilter::Active && self.sessions.iter().all(|s| self.fresh_draft && self.current.as_ref().is_some_and(|current| current.id == s.id));
        let mut body = column![top, tabs, container(nav).padding(Padding { top: 4.0, right: 10.0, bottom: 8.0, left: 10.0 })];
        if no_sessions {
            body = body.push(container(column![icon("root-folder", 20, Color { a: 0.36, ..p.foreground }),
                text("No sessions yet").size(12).color(Color { a: 0.54, ..p.foreground }),
                button(row![icon("add", 12, p.muted), text("New project").size(12).color(p.muted)].spacing(5).align_y(Alignment::Center))
                    .on_press(Message::Navigate(Screen::Workspaces)).padding([6, 9]).style(quiet_button),
            ].spacing(8).align_x(Alignment::Center)).center_x(Fill).center_y(Fill).height(Fill));
        } else {
            body = body.push(container(search).padding([6, 18])).push(container(pinned).padding([4, 10])).push(Space::new().height(8))
                .push(container(sessions_header).padding([0, 10]));
            if self.session_filter_menu {
                body = body.push(container(column![
                    menu_action("circle-filled", "Active sessions", Message::ArchiveFilter(ArchiveFilter::Active), p),
                    menu_action("archive", "Archived sessions", Message::ArchiveFilter(ArchiveFilter::Archived), p),
                    menu_action("list-unordered", "All sessions", Message::ArchiveFilter(ArchiveFilter::All), p),
                ].spacing(1)).padding([3, 10]));
            }
            // Cron jobs sit at the bottom of the list, as in the redesign; hidden while searching.
            let chats = if self.search.trim().is_empty() && !self.scheduled.jobs().is_empty() { chats.push(self.cron_section()) } else { chats };
            body = body.push(scrollable(container(chats).padding([0, 10])).height(Fill));
        }
        container(body.push(container(rail).padding([2, 12]))).width(SIDEBAR_WIDTH).height(Fill).style(move |_| panel(p.sidebar, 0.0, None)).into()
    }

    fn session_row<'a>(&self, summary: &'a SessionSummary) -> Element<'a, Message> {
        let p = palette(self.dark);
        let selected = self.current.as_ref().is_some_and(|current| current.id == summary.id) && self.screen == Screen::Chat;
        let mut chat = button(row![icon(if summary.pinned { "pinned" } else { "circle-filled" }, if summary.pinned { 11 } else { 5 }, if selected { p.accent } else { p.muted }),
            text(short(&summary.title, 28)).size(13), Space::new().width(Fill)].spacing(8).align_y(Alignment::Center))
            .height(27).padding([0, 8]).width(Fill).style(move |theme, status| nav_button(theme, status, selected));
        if !self.running || selected { chat = chat.on_press(Message::SelectChat(summary.id.clone())); }
        let mut row = column![mouse_area(chat).on_right_press(Message::SessionMenu(summary.id.clone()))];
        if self.session_menu.as_ref() == Some(&summary.id) {
            row = row.push(container(column![
                menu_action("pinned", if summary.pinned { "Unpin" } else { "Pin" }, Message::TogglePin(summary.id.clone()), p),
                menu_action("edit", "Rename", Message::RenameFor(summary.id.clone()), p),
                menu_action("multiple-windows", "Open in new window", Message::OpenInWindow(summary.id.clone()), p),
                menu_action("archive", if summary.archived { "Unarchive" } else { "Archive" }, Message::ArchiveSession(summary.id.clone(), !summary.archived), p),
            ].spacing(1)).padding(5).width(Fill).style(move |_| panel(p.background, 5.0, Some(p.border))));
        }
        row.into()
    }

    fn nav<'a>(&self, glyph: &'a str, label: &'a str, screen: Screen) -> Element<'a, Message> {
        let p = palette(self.dark); let selected = self.screen == screen;
        button(row![icon(glyph, 16, p.muted), text(label).size(13)].spacing(8).align_y(Alignment::Center)).on_press(Message::Navigate(screen))
            .height(28).padding([0, 8]).width(Fill).style(move |theme, status| nav_button(theme, status, selected)).into()
    }

    fn header(&self, overlay: bool) -> Element<'_, Message> {
        let p = palette(self.dark);
        let title:Element<'_,Message>=if self.screen==Screen::Tools {Space::new().width(Fill).into()}
        else if let Some(rename)=&self.rename {
            row![text_input("Session title",rename).on_input(Message::RenameChanged).on_submit(Message::RenameSave).width(180),button("Save").on_press(Message::RenameSave)].into()
        }else{
            let mut strip=row![].align_y(Alignment::Center);
            for tab in &self.tabs.entries {
                let active=self.tabs.active.as_ref()==Some(&tab.id);
                let label=spaced(&if tab.draft{"NEW SESSION".to_owned()}else{short(&tab.title,19).to_uppercase()});
                let mut line=row![button(row![tab_dot(active, p), text(label).size(11).font(medium()).color(if active{p.foreground}else{p.muted})].spacing(6).align_y(Alignment::Center)).width(Fill).padding([0,5]).style(quiet_button).on_press(Message::Tab(tabs::Action::Select(tab.id.clone())))].height(30).align_y(Alignment::Center);
                if self.tabs.entries.len()>1 {line=line.push(icon_button("close","Close tab",Message::Tab(tabs::Action::Close(tab.id.clone())),p));}
                let underline=container(Space::new().width(Fill).height(2)).style(move |_|panel(if active{p.accent}else{p.background},0.,None));
                strip=strip.push(mouse_area(container(column![line,underline]).width(tabs::tab_width(tab))).on_right_press(Message::Tab(tabs::Action::Menu(tab.id.clone()))));
            }
            let visible=self.tabs.entries.iter().map(tabs::tab_width).sum::<f32>().clamp(112.,640.);
            row![scrollable(strip).id("hermes-session-tabs").direction(scrollable::Direction::Horizontal(scrollable::Scrollbar::new().width(1))).width(visible),
                icon_button("add","New session",Message::NewChat,p)].align_y(Alignment::Center).into()
        };
        let mut controls = row![].spacing(0).align_y(Alignment::Center);
        if !overlay {
            if !self.sidebar_open { controls = controls.push(icon_button("layout-sidebar-left", "Show sidebar", Message::ToggleSidebar, p)); }
            controls = controls.push(icon_button("settings-gear", "Settings", Message::Navigate(Screen::Provider), p))
                .push(icon_button("layout", "Layout", Message::ToggleLayout, p))
                .push(icon_button("arrow-swap", "Flip panes", Message::FlipPanes, p))
                .push(icon_button("layout-sidebar-right", "Files", Message::Navigate(Screen::Files), p))
                .push(icon_button("comment-discussion", "Enter HUD mode · ⌘⇧H", Message::Hud(super::hud::Action::Toggle), p));
        }
        mouse_area(container(row![title, Space::new().width(Fill), controls].align_y(Alignment::Center)).padding([0, 12]).height(34).width(Fill))
            .on_press(Message::DragWindow).into()
    }

    fn right_sidebar(&self, pane: RightPane) -> Element<'_, Message> {
        let p = palette(self.dark);
        let content = match pane { RightPane::Files => self.files.view().map(Message::Files), RightPane::Review => self.git.view().map(Message::Git), RightPane::Terminal => self.terminal.view(self.shortcuts.bindings(), self.shortcuts.is_capturing()).map(Message::Terminal) };
        container(column![Space::new().height(34), content].height(Fill)).width(237).height(Fill).style(move |_| panel(p.sidebar, 0.0, None)).into()
    }

    fn statusbar(&self) -> Element<'_, Message> {
        let p = palette(self.dark);
        if !self.statusbar.preferences.visible { return Space::new().height(0).width(Fill).into(); }
        let sb = &self.statusbar;
        let backend_ready = self.library.is_some() && self.session.is_some();
        // Left: command center, backend health, workspace, agents, scheduled jobs.
        let command = tooltip(button(container(text("⌘").size(12).font(Font::DEFAULT).color(p.muted)).center_x(Fill).center_y(Fill))
                .on_press(Message::Navigate(Screen::CommandCenter)).padding(0).width(28).height(20)
                .style(move |t, s| bar_button(t, s, self.screen == Screen::CommandCenter)),
            text("Open Command Center").size(11), tooltip::Position::Top);
        let mut left = row![command].spacing(2).height(20).align_y(Alignment::Center);
        if sb.shows("gateway-health") {
            left = left.push(bar_item(if backend_ready { "pulse" } else { "warning" }, if backend_ready { "Backend ready" } else { "Backend unavailable" }.into(), "Backend", Message::Statusbar(statusbar::Action::Backend), false, p));
        }
        if sb.shows("workspace-cwd") && let Some(project) = self.chat_project() {
            left = left.push(bar_item("folder-opened", project.name.clone(), "Change workspace", Message::Navigate(Screen::Workspaces), false, p));
        }
        if sb.shows("agents") { left = left.push(bar_item("robot", "Agents".into(), "Open agents", Message::Navigate(Screen::Agents), self.screen == Screen::Agents, p)); }
        if sb.shows("cron") { left = left.push(bar_item("watch", "Scheduled".into(), "Scheduled jobs", Message::Navigate(Screen::Cron), self.screen == Screen::Cron, p)); }
        // Right: live turn metrics, resources, approval mode, terminal, version.
        let mut right = row![].spacing(2).height(20).align_y(Alignment::Center);
        if sb.shows("running-timer") && let Some(started) = sb.turn_started {
            right = right.push(bar_item("loading", format!("Running {}", statusbar::duration(started)), "Turn running", Message::Statusbar(statusbar::Action::Close), false, p));
        }
        if sb.shows("context-usage") {
            let used = self.history.turns.iter().map(|turn| turn.prompt.len() + turn.entries.iter().map(|entry| match entry {
                Entry::Assistant { text, .. } => text.len(), Entry::Tool { result, .. } => result.len() }).sum::<usize>()).sum::<usize>() as u64 / 4;
            let label = match self.models.current_model(&self.config).and_then(|m| m.context_length) {
                Some(window) if window > 0 => format!("{} / {} · {}%", statusbar::tokens(used), statusbar::tokens(window), (used * 100 / window).min(100)),
                _ if used > 0 => statusbar::tokens(used),
                _ => "—".into(),
            };
            right = right.push(bar_item("layout", label, "Estimated context in use", Message::Statusbar(statusbar::Action::Close), false, p));
        }
        if sb.shows("cache-hit-rate") {
            right = right.push(bar_item("layers", "—".into(), "Prompt cache hit rate (not reported by this provider)", Message::Statusbar(statusbar::Action::Close), false, p));
        }
        if sb.shows("tokens-per-second") {
            let label = sb.tokens_per_second.map_or("—".into(), |tps| format!("{tps:.0} tok/s"));
            right = right.push(bar_item("zap", label, "Output speed of the latest reply (estimated)", Message::Statusbar(statusbar::Action::Close), false, p));
        }
        if sb.shows("session-timer") && let Some(opened) = sb.opened {
            right = right.push(bar_item("watch", format!("Focused {}", statusbar::duration(opened)), "Time since this window opened", Message::Statusbar(statusbar::Action::Close), false, p));
        }
        if sb.shows("system-resources") {
            let label = sb.memory.map_or("RAM —".into(), |(used, total)| format!("RAM {}/{}", statusbar::gigabytes(used), statusbar::gigabytes(total)));
            right = right.push(bar_item("pulse", label, "System memory in use", Message::Statusbar(statusbar::Action::Close), false, p));
        }
        if backend_ready && sb.shows("approval-mode") {
            let mode = self.profile_settings.saved.approval_mode.as_str();
            let (_, label, _) = statusbar::APPROVAL_MODES.iter().find(|(v, _, _)| *v == mode).copied().unwrap_or(statusbar::APPROVAL_MODES[1]);
            right = right.push(bar_item("zap", label.into(), "Approval mode", Message::Statusbar(statusbar::Action::Approval), mode == "off" || sb.menu == Some(statusbar::Menu::Approval), p));
        }
        if sb.shows("terminal") && self.screen == Screen::Chat {
            right = right.push(bar_item("terminal", String::new(), "Show terminal", Message::Navigate(Screen::Terminal), false, p));
        }
        if let Some((label, title)) = statusbar::build_label() {
            right = right.push(bar_item("hash", label, &title, Message::Navigate(Screen::About), false, p));
        }
        let bar = container(row![left, Space::new().width(Fill), right].height(20).align_y(Alignment::Center))
            .height(20).padding([0, 4]).width(Fill).style(move |_| if self.artwork.is_some() { container::Style::default() } else { panel(p.sidebar, 0.0, None) });
        mouse_area(bar).on_right_press(Message::Statusbar(statusbar::Action::Customize)).into()
    }

    fn statusbar_popup(&self, menu: statusbar::Menu) -> Element<'_, Message> {
        let p = palette(self.dark);
        let mut content = column![].spacing(2);
        let heading = |label: &'static str| container(text(label).size(11).font(medium()).color(p.muted)).padding([5, 8]);
        match menu {
            statusbar::Menu::Backend => {
                content = content.push(heading("Backend")).push(rule::horizontal(1)).push(container(row![text("Local Rust backend").size(12), Space::new().width(Fill),
                        text(if self.session.is_some() { "Ready" } else { "Unavailable" }).size(12).color(p.muted)].spacing(10)).padding(8))
                    .push(menu_action("settings-gear", "Provider settings", Message::SettingsView(settings::View::Models), p))
                    .push(menu_action("folder", "Change workspace", Message::Navigate(Screen::Workspaces), p));
            }
            statusbar::Menu::Approval => {
                let current = self.profile_settings.saved.approval_mode.as_str();
                content = content.push(heading("Approval mode")).push(rule::horizontal(1));
                for (value, label, description) in statusbar::APPROVAL_MODES {
                    let selected = current == value;
                    content = content.push(button(row![
                        container(if selected { icon("circle-filled", 7, p.foreground) } else { text("") }).width(14).center_x(14).padding(iced::Padding { top: 4.0, ..Default::default() }),
                        column![text(label).size(12), text(description).size(11).color(p.muted)].spacing(2)].spacing(6).align_y(Alignment::Start))
                        .on_press(Message::Statusbar(statusbar::Action::SetApproval(value))).padding([6, 8]).width(Fill)
                        .style(move |t, s| bar_button(t, s, false)));
                }
            }
            statusbar::Menu::Customize => {
                content = content.push(heading("Customize status bar")).push(rule::horizontal(1));
                for (id, label) in statusbar::ITEMS {
                    let on = self.statusbar.shows(id);
                    content = content.push(button(row![container(if on { icon("check", 12, p.foreground) } else { text("") }).width(16), text(label).size(12)].spacing(6).align_y(Alignment::Center))
                        .on_press(Message::Statusbar(statusbar::Action::ToggleItem(id))).padding([5, 8]).width(Fill).style(move |t, s| bar_button(t, s, false)));
                }
                content = content.push(rule::horizontal(1)).push(menu_action("refresh", "Reset to default", Message::Statusbar(statusbar::Action::Reset), p))
                    .push(menu_action("eye-closed", "Hide status bar", Message::Statusbar(statusbar::Action::ToggleVisible), p));
            }
        }
        let width = if menu == statusbar::Menu::Customize { 208 } else { 288 };
        container(content).padding(4).width(width).style(move |_| panel(p.card, 6.0, Some(p.border))).into()
    }

    fn chat(&self) -> Element<'_, Message> {
        let mut content = column![].height(Fill);
        if let Some(banner) = &self.banner { content = content.push(self.banner_view(banner)); }
        content = content.push(if self.history.turns.is_empty() { welcome::view(self) } else { self.transcript() });
        if self.attach_menu {
            let p = palette(self.dark);
            content = content.push(container(column![
                menu_action("files", "Attach file…", Message::Attachment(attachments::Action::Choose), p),
                menu_action("folder", "Folders", Message::Navigate(Screen::Workspaces), p),
                menu_action("files", "Files", Message::Navigate(Screen::Files), p),
                menu_action("symbol-misc", "Skills & Tools", Message::Navigate(Screen::Tools), p),
            ].spacing(3)).padding(6).width(180).style(move |_| panel(p.background, 8.0, Some(p.border))));
        }
        if !self.follow_stream && self.running {
            content = content.push(container(button(icon("arrow-down", 14, palette(self.dark).muted)).on_press(Message::JumpToLatest).padding(6).style(quiet_button)).center_x(Fill));
        }
        if !self.draft_files.files.is_empty() || self.draft_files.pending>0 {
            content=content.push(container(self.draft_files.view().map(Message::Attachment)).padding([2,16]));
        }
        content.push(self.composer_view()).into()
    }

    fn banner_view<'a>(&self, banner: &'a Banner) -> Element<'a, Message> {
        let p = palette(self.dark);
        container(row![text(&banner.text).size(12).color(if banner.error { hex(0xcf222e) } else { p.muted }).width(Fill),
            icon_button("close", "Dismiss", Message::Dismiss, p)].spacing(10).align_y(Alignment::Center)).padding([8, 20]).width(Fill)
            .style(move |_| panel(p.background, 0.0, Some(p.border))).into()
    }

    /// HUD mode: a drag strip, the composer on top, the transcript hanging below.
    pub(super) fn hud_view(&self) -> Element<'_, Message> {
        let p = palette(self.dark);
        let title = self.current.as_ref().map_or("New chat".to_owned(), |c| c.title.clone());
        let strip = mouse_area(container(row![Space::new().width(10), icon("comment-discussion", 12, p.muted), text(short(&title, 48)).size(11).color(p.muted),
            Space::new().width(Fill), icon_button("screen-normal", "Leave HUD mode · ⌘⇧H", Message::Hud(super::hud::Action::Toggle), p)]
            .spacing(6).align_y(Alignment::Center)).padding([0, 8]).height(30).width(Fill)).on_press(Message::Hud(super::hud::Action::Drag));
        let mut band = column![strip].spacing(4);
        if let Some(banner) = &self.banner { band = band.push(self.banner_view(banner)); }
        band = band.push(container(self.composer_view()).padding([0, 10]));
        // The composer's + menu opens under it, as the main window's opens above it.
        if self.attach_menu {
            band = band.push(container(container(column![
                menu_action("files", "Attach file…", Message::Attachment(attachments::Action::Choose), p),
                menu_action("folder", "Folders", Message::Navigate(Screen::Workspaces), p),
                menu_action("files", "Files", Message::Navigate(Screen::Files), p),
                menu_action("symbol-misc", "Skills & Tools", Message::Navigate(Screen::Tools), p),
            ].spacing(3)).padding(6).width(180).style(move |_| panel(p.background, 8.0, Some(p.border)))).padding([0, 10]));
        }
        if !self.draft_files.files.is_empty() || self.draft_files.pending > 0 {
            band = band.push(container(self.draft_files.view().map(Message::Attachment)).padding([2, 10]));
        }
        if let Some(approval) = &self.approval { band = band.push(container(self.approval_view(approval)).padding([0, 10])); }
        if !self.history.turns.is_empty() || self.running { band = band.push(container(self.transcript()).height(Fill)); }
        // Over the native frost the band is a uniform tint, not a slab (Electron's glass rule).
        let tint = if super::hud_native::FADES { Color { a: 0.78, ..p.background } } else { p.background };
        let band = container(band.padding(iced::Padding { bottom: 8.0, ..Default::default() })).width(Fill).height(Fill)
            .style(move |_| panel(tint, 12.0, Some(p.border)));
        // The composer's pickers live in this window while the HUD is up (the app window is hidden).
        let mut layers = stack![band];
        if self.commands.is_open() { layers = layers.push(self.commands.view(self.dark).map(Message::Commands)); }
        if self.models.picker {
            layers = layers.push(container(Space::new().width(Fill).height(Fill)).style(|_| container::Style { background: Some(Color { a: 0.22, ..Color::BLACK }.into()), border: iced::Border { radius: 12.0.into(), ..Default::default() }, ..Default::default() }))
                .push(container(models::picker(self)).width(Fill).height(Fill).center_x(Fill).padding(10));
        }
        if let Some(anchor) = self.reasoning.anchor { layers = layers.push(reasoning::overlay(self, anchor, true)); }
        layers.width(Fill).height(Fill).into()
    }

    fn transcript(&self) -> Element<'_, Message> {
        let p = palette(self.dark);
        let imported = self.current.as_ref().is_some_and(|s| matches!(s.source, hermes_library::SessionSource::Hermes { .. }));
        let mut messages = column![].spacing(18).padding([20, 24]);
        if imported { messages = messages.push(text("Imported session · Historical tool records do not establish approval in this app.").size(11).color(p.muted)); }
        for (turn_index, turn) in self.history.turns.iter().enumerate() {
            let (visible, speaker) = super::group_prompt(&turn.prompt);
            if !visible.is_empty() || !turn.attachments.is_empty() {
                let mut user_content=column![text(visible).size(appearance::chat_size(13.0)).line_height(1.4)].spacing(3);
                for file in &turn.attachments { user_content=user_content.push(text(&file.name).size(12).color(p.muted)); }
                let user = container(user_content).padding([9, 12]).width(Fill)
                    .style(move |_| panel(p.card, 4.0, Some(Color { a: 0.5, ..p.border })));
                messages = messages.push(user);
            }
            if let Some(name) = speaker {
                messages = messages.push(row![icon("hubot", 12, p.accent), text(name).size(12).font(medium()).color(p.accent)].spacing(6).align_y(Alignment::Center));
            }
            for (entry_index, entry) in turn.entries.iter().enumerate() {
                match entry {
                    Entry::Assistant { text: body, calls } => {
                        let mut assistant = column![].spacing(12);
                        if body.is_empty() && calls.is_empty() {
                            assistant = assistant.push(text(if turn.status == Status::Running { "Thinking…" } else { "No text response." }).size(12).color(p.muted));
                        } else if let Some(Some(content)) = self.rendered.get(turn_index).and_then(|r| r.get(entry_index)) {
                            let mut settings = markdown::Settings::with_text_size(appearance::chat_size(13.0), theme_for(self.dark));
                            settings.style.font = appearance::font();
                            settings.style.inline_code_font = Font::with_name("JetBrains Mono");
                            settings.style.code_block_font = Font::with_name("JetBrains Mono");
                            assistant = assistant.push(markdown::view(content.items(), settings).map(Message::Link));
                        }
                        for call in calls {
                            assistant = assistant.push(row![icon("chevron-right", 11, p.muted), icon("tools", 12, p.muted), text(tool_label(&call.name)).size(11).color(p.muted)].spacing(5).align_y(Alignment::Center));
                        }
                        if !body.is_empty() && turn.status != Status::Running {
                            let mut actions=row![tooltip(button(icon("copy", 12, p.muted)).on_press(Message::Copy(body.clone())).padding(3).style(quiet_button), text("Copy").size(11), tooltip::Position::Bottom)].spacing(5);
                            if self.voice.speech_unavailable().is_none() {
                                let id=format!("{turn_index}:{entry_index}");
                                let speaking=self.voice.speaking.as_ref()==Some(&id);
                                actions=actions.push(icon_button(if speaking {"primitive-square"} else {"unmute"},if speaking {"Stop speaking"} else {"Read aloud"},Message::Voice(if speaking {voice::Action::StopSpeaking} else {voice::Action::Speak(id,body.clone())}),p));
                            }
                            assistant = assistant.push(actions);
                        }
                        messages = messages.push(container(assistant).padding([0, 12]));
                    }
                    Entry::Tool { name, approved, result, .. } => {
                        let failed = serde_json::from_str::<serde_json::Value>(result).ok().is_some_and(|value| value.get("error").is_some());
                        let state = if failed && *approved { "Failed" } else if *approved { "" } else if imported { "Historical record" } else { "Denied / not run" };
                        messages = messages.push(container(column![row![icon("chevron-right", 11, p.muted), icon(if *approved { "check" } else { "close" }, 11, p.muted),
                            text(tool_label(name)).size(11).color(p.muted), text(state).size(10).color(p.muted), Space::new().width(Fill),
                            button(icon("copy", 11, p.muted)).on_press(Message::Copy(result.clone())).padding(2).style(quiet_button)].spacing(5).align_y(Alignment::Center),
                            text(short(result, 1600)).size(11).font(Font::MONOSPACE).color(p.muted)].spacing(5)).padding([4, 12]));
                    }
                }
            }
            match &turn.status {
                Status::Interrupted => messages = messages.push(text("Stopped. Partial response saved.").size(11).color(p.muted)),
                Status::Failed(reason) => messages = messages.push(container(column![text(reason).size(12).color(hex(0xcf222e)), text("Send a follow-up to try again.").size(11).color(p.muted)].spacing(5)).padding([8, 12])),
                _ => {},
            }
        }
        // A routine's chat shows its run live while the scheduler executes it; the finished turn replaces this.
        if let (Some(current), Some(library)) = (&self.current, &self.library) && current.title.starts_with("Routine · ")
            && let Some(job) = super::routine_chats::job_of(library, &current.id) && let Some((secs, live, waiting)) = self.scheduled.live(&job) {
            let status = if waiting { "Waiting for approval on the Routines page".to_owned() } else { format!("Running now · {}:{:02}", secs / 60, secs % 60) };
            let body: Element<'_, Message> = if live.is_empty() { text("Thinking…").size(12).color(p.muted).into() }
                else { text(live.to_owned()).size(appearance::chat_size(13.0)).line_height(1.4).into() };
            messages = messages.push(container(column![row![icon("watch", 12, p.accent), text(status).size(11).font(medium()).color(p.accent)].spacing(6).align_y(Alignment::Center), body].spacing(8)).padding([0, 12]));
        }
        if let Some(approval) = &self.approval { messages = messages.push(self.approval_view(approval)); }
        scrollable(container(messages).width(Fill)).id(SCROLL_ID).height(Fill)
            .on_scroll(|viewport| Message::Scrolled(viewport.absolute_offset_reversed().y < 48.0)).into()
    }

    fn approval_view<'a>(&self, approval: &'a Approval) -> Element<'a, Message> {
        let p = palette(self.dark);
        let args = serde_json::from_str::<serde_json::Value>(&approval.call.arguments).ok().and_then(|v| serde_json::to_string_pretty(&v).ok()).unwrap_or_else(|| approval.call.arguments.clone());
        container(column![row![icon("shield", 15, p.accent), text("Approval required").size(13).font(medium())].spacing(8).align_y(Alignment::Center),
            text(tool_label(&approval.call.name)).size(12),
            scrollable(text(args).size(11).font(Font::MONOSPACE)).height(130),
            row![Space::new().width(Fill), button(text("Deny").size(12)).on_press(Message::Approve(false)).padding([6, 12]).style(outline_button),
                button(text("Allow once").size(12)).on_press(Message::Approve(true)).padding([6, 12]).style(primary_button)].spacing(8),
        ].spacing(10)).padding(14).width(Fill).style(move |_| panel(p.background, 8.0, Some(p.border))).into()
    }

    fn composer_view(&self) -> Element<'_, Message> {
        let p = palette(self.dark);
        let bindings = self.shortcuts.bindings();
        let slash_open = !slash::items(self).is_empty();
        let capturing_shortcut = self.shortcuts.is_capturing();
        let lines = self.composer.line_count().clamp(1, 7) as f32;
        let height = (lines * appearance::chat_size(18.0) + 8.0).clamp(26.0, appearance::chat_size(150.0));
        let editor = widget::text_editor(&self.composer).id(COMPOSER_ID).on_action(Message::Edit)
            .placeholder(composer_placeholder(self.current.as_ref().map(|c| c.id.as_str()), self.history.turns.is_empty())).font(appearance::font()).size(appearance::chat_size(13.0)).line_height(1.38).padding([4, 3]).height(height)
            .key_binding(move |press| {
                use keyboard::key::Named;
                // While the slash popup is open it owns the arrows, Tab, Enter and Escape.
                if slash_open && !press.modifiers.command() && !press.modifiers.shift() {
                    match press.key {
                        keyboard::Key::Named(Named::ArrowUp) => return Some(text_editor::Binding::Custom(Message::Slash(slash::Action::Up))),
                        keyboard::Key::Named(Named::ArrowDown) => return Some(text_editor::Binding::Custom(Message::Slash(slash::Action::Down))),
                        keyboard::Key::Named(Named::Tab | Named::Enter) => return Some(text_editor::Binding::Custom(Message::Slash(slash::Action::Accept))),
                        keyboard::Key::Named(Named::Escape) => return Some(text_editor::Binding::Custom(Message::Slash(slash::Action::Dismiss))),
                        _ => {}
                    }
                }
                if capturing_shortcut || bindings.action_for_key(&press.key, press.physical_key, press.modifiers, true).is_some() { None }
                else if press.key == keyboard::Key::Named(keyboard::key::Named::Enter) && press.modifiers.command() && press.modifiers.shift() { Some(text_editor::Binding::Custom(Message::QueueResume)) }
                else if press.key == keyboard::Key::Named(keyboard::key::Named::Enter) && press.modifiers.command() { Some(text_editor::Binding::Custom(Message::QueueCurrent)) }
                else if press.key == keyboard::Key::Named(keyboard::key::Named::Enter) && !press.modifiers.shift() { Some(text_editor::Binding::Custom(Message::Send)) }
                else { text_editor::Binding::from_key_press(press) }
            }).style(move |_, _| text_editor::Style { background: Background::Color(Color::TRANSPARENT), border: Border::default(), placeholder: p.muted, value: p.foreground, selection: p.secondary });
        let has_payload = !self.composer.text().trim().is_empty();
        let show_stop = self.running && !has_payload;
        let mut action = button(icon(if show_stop { "primitive-square" } else { "arrow-up" }, 14, p.background)).width(26).height(26).padding(6).style(send_button);
        if show_stop { action = action.on_press(Message::Stop); }
        else if has_payload { action = action.on_press(Message::Send); }
        let model = self.models.model_label(&self.config);
        let mut controls = row![button(row![text(short(model, 24)).size(12).color(p.muted), icon("chevron-down", 11, p.muted)].spacing(4).align_y(Alignment::Center))
            .on_press_maybe((!self.running).then_some(Message::Models(models::Action::Open))).padding([3, 7]).style(quiet_button)].spacing(4).align_y(Alignment::Center);
        if self.models.current_model(&self.config).is_some_and(|model|model.supports("reasoning") || model.supports("reasoning_effort")) {
            controls=controls.push(container(button(row![text(self.models.reasoning_label()).size(12).color(p.muted),icon("chevron-down",11,p.muted)].spacing(4)).on_press_maybe((!self.running).then_some(Message::Reasoning(reasoning::Action::Open))).padding([3,7]).style(quiet_button)).id(reasoning::TRIGGER));
        }
        if self.running && has_payload { controls = controls.push(icon_button("layers", "Queue message · ⌘ Enter", Message::QueueCurrent, p)); }
        // Wake-word listening keeps the mic open but is not dictation.
        let dictation_busy=(self.voice.recording && !self.voice.wake_listening) || (self.voice.authorizing && !self.voice.wake_listening) || self.voice.finishing;
        let dictation_label=if self.voice.recording {"Finish dictation"} else if self.voice.authorizing || self.voice.finishing {"Cancel dictation"} else {self.voice.dictation_unavailable().unwrap_or("Dictate")};
        let dictation_action=if self.voice.recording && !self.voice.wake_listening {voice::Action::StopRecord} else if dictation_busy {voice::Action::CancelRecord} else {voice::Action::Record};
        let microphone=button(icon(if self.voice.recording {"primitive-square"} else {"mic"},13,if dictation_busy {p.accent} else {p.muted})).padding(5).style(quiet_button)
            .on_press_maybe((dictation_busy || self.voice.dictation_unavailable().is_none()).then_some(Message::Voice(dictation_action)));
        controls=controls.push(tooltip(microphone,text(dictation_label).size(11),tooltip::Position::Top));
        // Empty and idle, the action is the voice-conversation button (the round waveform).
        if !has_payload && !self.running && self.voice.dictation_unavailable().is_none() {
            let on = self.voice.conversation;
            let voice = button(icon("pulse", 14, if on { shell_art::accent().on() } else { p.background })).width(26).height(26).padding(6)
                .on_press(Message::Voice(voice::Action::Conversation(!on)))
                .style(move |theme, status| { let mut s = send_button(theme, status); if on { s.background = Some(shell_art::accent().fill().into()); } s });
            controls = controls.push(tooltip(voice, text(if on { "End voice conversation" } else { "Start voice conversation" }).size(11), tooltip::Position::Top));
        } else {
            controls = controls.push(tooltip(action, text(if show_stop { "Stop" } else if self.running { "Redirect the current run" } else { "Send" }).size(11), tooltip::Position::Top));
        }
        let surface = container(row![
            icon_button("add", "Attach", Message::ToggleAttach, p), editor, controls,
        ].spacing(4).align_y(Alignment::Center)).padding([8, 10]).width(Fill).style(move |_| panel(p.card, 4.0,
            Some(if has_payload || self.running { p.accent } else if self.dark { Color { a: 0.4, ..p.accent } } else { p.border })));
        let mut dock = column![].spacing(6);
        if self.voice.conversation {
            let state = if self.voice.speaking.is_some() { "Speaking…" } else if self.running { "Thinking…" } else if self.voice.recording { "Listening… pause to send" } else { "Voice conversation" };
            dock = dock.push(row![icon("pulse", 12, p.accent), text(state).size(12).color(p.muted), Space::new().width(Fill),
                button(text("End").size(11)).on_press(Message::Voice(voice::Action::Conversation(false))).padding([3, 8]).style(quiet_button)].spacing(6).align_y(Alignment::Center));
        }
        if dictation_busy {
            let label=if self.voice.authorizing {"Waiting for microphone and speech permission…"} else if self.voice.finishing {"Finishing dictation…"} else if self.voice.transcript.is_empty() {"Listening…"} else {&self.voice.transcript};
            dock=dock.push(row![icon("mic",12,p.accent),text(label).size(12).color(p.muted),Space::new().width(Fill),button(text("Cancel").size(11)).on_press(Message::Voice(voice::Action::CancelRecord)).padding([3,6]).style(quiet_button)].spacing(6).align_y(Alignment::Center));
        }
        if let Some(error)=&self.voice.error {dock=dock.push(text(error).size(11).color(p.muted));}
        if !self.queue.entries.is_empty() { dock = dock.push(self.queue_view()); }
        if self.chat_project().is_some() {
            let branch = self.git.branch_label().unwrap_or(self.chat_project().map(|p| p.name.as_str()).unwrap_or("No workspace"));
            dock = dock.push(button(row![icon("git-branch", 12, hex(0x1f8a65)), text(branch).size(11).color(p.muted)].spacing(5).align_y(Alignment::Center))
                .on_press(Message::Navigate(Screen::Git)).padding([2, 6]).style(quiet_button));
        }
        if let Some(popup) = slash::popup(self) { dock = dock.push(popup); }
        dock = dock.push(surface);
        container(dock).padding(Padding { top: 8.0, right: 16.0, bottom: 10.0, left: 16.0 }).width(Fill).into()
    }

    fn queue_view(&self) -> Element<'_, Message> {
        let p = palette(self.dark);
        let mut header = row![icon("layers", 12, p.muted), text(format!("{} Queued{}", self.queue.entries.len(), if self.queue.parked { " — paused" } else { "" })).size(11).color(p.muted), Space::new().width(Fill)].spacing(6).align_y(Alignment::Center);
        if self.queue.parked { header = header.push(button(text("Resume").size(11)).on_press(Message::QueueResume).padding([3, 7]).style(quiet_button)); }
        else { header = header.push(icon_button("debug-pause", "Pause queue", Message::QueuePause, p)); }
        let mut entries = column![].spacing(3);
        for entry in &self.queue.entries {
            if let Some((id, value)) = &self.queue_edit && *id == entry.id {
                entries = entries.push(column![text_input("Queued message", value).on_input(Message::QueueEditChanged).on_submit(Message::QueueEditSave).size(12).padding(6).style(field_input),
                    row![Space::new().width(Fill), button(text("Cancel").size(11)).on_press(Message::QueueEditCancel).padding([3, 7]).style(quiet_button),
                        button(text("Save").size(11)).on_press(Message::QueueEditSave).padding([3, 7]).style(quiet_button)].spacing(5)].spacing(4));
            } else {
                entries = entries.push(row![text(short(&entry.text.replace('\n', " "), 150)).size(12).color(p.muted).width(Fill),
                    icon_button("edit", "Edit", Message::QueueEdit(entry.id), p),
                    button(text(if self.running { "Next" } else { "Send" }).size(11)).on_press(Message::QueueSendNow(entry.id)).padding([3, 6]).style(quiet_button),
                    icon_button("close", "Delete", Message::QueueRemove(entry.id), p)].spacing(4).align_y(Alignment::Center));
            }
        }
        let height = ((self.queue.entries.len() as f32 * 31.0) + if self.queue_edit.is_some() { 35.0 } else { 0.0 }).min(180.0);
        container(column![header, scrollable(entries).height(height)].spacing(4)).padding([6, 10]).width(Fill).style(move |_| panel(p.background, 7.0, Some(p.border))).into()
    }

    fn capabilities(&self) -> Element<'_, Message> {
        capabilities::view(self)
    }

    fn artifacts(&self) -> Element<'_, Message> {
        column![page_title("Artifacts"), self.artifacts.view(self.dark, self.running, self.webview.offered_link()).map(Message::Artifacts)].height(Fill).into()
    }


    fn workspaces(&self) -> Element<'_, Message> {
        let p = palette(self.dark);
        let mut list = column![].spacing(6);
        for project in &self.projects {
            let mut item = button(row![icon("folder", 15, p.muted), column![text(&project.name).size(13), text(project.path.to_string_lossy().into_owned()).size(11).color(p.muted)].spacing(3)].spacing(10).align_y(Alignment::Center))
                .padding(10).width(Fill).style(quiet_button);
            if !self.running { item = item.on_press(Message::SelectWorkspace(Some(project.id.clone()))); }
            list = list.push(item);
        }
        let mut path = text_input("Project folder", &self.workspace_path).padding(9).size(13).style(field_input);
        let mut browse = button(text("Browse…").size(12)).padding([8, 12]).style(outline_button);
        let mut add = button(text("Add project").size(12)).padding([8, 12]).style(primary_button);
        if !self.running { path = path.on_input(Message::WorkspacePath).on_submit(Message::AddWorkspace); browse = browse.on_press(Message::BrowseWorkspace); if !self.workspace_path.trim().is_empty() { add = add.on_press(Message::AddWorkspace); } }
        let content = container(scrollable(column![row![text("Projects").size(18).font(medium()), Space::new().width(Fill), icon_button("close", "Close", Message::Navigate(Screen::Chat), p)].align_y(Alignment::Center),
            list, text("Project folder").size(12), path, row![browse, Space::new().width(Fill), add].spacing(8),
            super::project_discovery::suggestions(self),
        ].spacing(16))).padding(24).width(560).max_height(640).style(move |_| panel(p.background, 9.0, Some(p.border)));
        container(content).center_x(Fill).center_y(Fill).height(Fill).into()
    }

    fn about(&self) -> Element<'_, Message> {
        let p = palette(self.dark);
        let data = Session::default_dir().map(|p| p.display().to_string()).unwrap_or_default();
        container(container(column![row![text("About Hermes").size(20).font(medium()), Space::new().width(Fill), icon_button("close", "Close", Message::Navigate(Screen::Chat), p)].align_y(Alignment::Center),
            text(format!("Hermes Desktop {}", env!("CARGO_PKG_VERSION"))).size(13), text("Local profile: default").size(13),
            text(data).size(11).font(Font::MONOSPACE).color(p.muted),
            button(text("Import sessions").size(12)).on_press(Message::Navigate(Screen::Import)).padding([7, 10]).style(outline_button),
        ].spacing(18)).padding(26).width(600).style(move |_| panel(p.background, 9.0, Some(p.border)))).center_x(Fill).center_y(Fill).height(Fill).into()
    }
}

pub(super) fn medium() -> Font { Font { weight: iced::font::Weight::Medium, ..appearance::font() } }
fn short(value: &str, limit: usize) -> String { if value.chars().count() > limit { format!("{}…", value.chars().take(limit.saturating_sub(1)).collect::<String>()) } else { value.into() } }
fn tool_label(name: &str) -> &str { match name { "get_current_time" => "Read system clock", "read_file" => "Read file", "list_directory" => "List directory", "search_files" => "Search files", "write_file" => "Write file", "skills_list" => "Skills", "skill_view" => "Read skill", _ => name } }
fn section_label(label: &str, count: Option<String>, p: Palette) -> Element<'_, Message> {
    // sidebar-label.tsx: an 8px dithered square in the theme primary, not a glyph.
    let dot = container(Space::new().width(8).height(8)).style(move |_| container::Style {
        background: Some(Color { a: 0.55, ..p.accent }.into()), border: Border { radius: 1.0.into(), width: 1.0, color: p.accent }, ..Default::default() });
    row![dot, text(label).size(10).font(medium()).color(p.accent), text(count.unwrap_or_default()).size(11).color(p.muted)].spacing(7).align_y(Alignment::Center).padding([7, 8]).into()
}
pub(super) fn icon<'a>(name: &str, size: u16, color: Color) -> widget::Text<'a> {
    let glyph = match name { "robot" => '\u{ec20}', "symbol-misc" => '\u{eb63}', "comment" => '\u{ea6b}', "files" => '\u{eaf0}', "watch" => '\u{eb7c}',
        "search" => '\u{ea6d}', "pinned" => '\u{eba0}', "list-filter" => '\u{eb83}', "settings-gear" => '\u{eb51}',
        "layout-sidebar-left" => '\u{ebf3}', "layout-sidebar-right" => '\u{ebf4}', "arrow-swap" => '\u{ebcb}', "layout" => '\u{ebeb}',
        "multiple-windows" => '\u{eb78}', "comment-discussion" => '\u{eac7}', "screen-full" => '\u{eb4c}', "screen-normal" => '\u{eb4d}', "close" => '\u{ea76}', "add" => '\u{ea60}', "mic" => '\u{ec12}', "unmute" => '\u{eb75}', "arrow-up" => '\u{eaa1}',
        "arrow-down" => '\u{ea9a}', "git-branch" => '\u{ec6f}', "folder" => '\u{ea83}', "chevron-up" => '\u{eab7}', "chevron-down" => '\u{eab4}', "chevron-right" => '\u{eab6}', "chevron-left" => '\u{eab5}', "cloud" => '\u{ebaa}', "bell-slash" => '\u{ec08}', "arrow-left" => '\u{ea9b}', "hubot" => '\u{eb08}', "organization" => '\u{ea7e}', "new-folder" => '\u{ea80}', "link-external" => '\u{eb14}',
        "home" => '\u{eb06}', "account" => '\u{eb99}', "circle-filled" => '\u{ea71}', "bell" => '\u{eaa2}', "terminal" => '\u{ea85}', "refresh" => '\u{eb37}',
        "folder-opened" => '\u{eaf7}', "file" => '\u{ea7b}', "eye" => '\u{ea70}', "eye-closed" => '\u{eae7}', "three-bars" => '\u{eb6a}', "warning" => '\u{ea6c}', "loading" => '\u{eb19}',
        "layers" => '\u{ebd2}', "archive" => '\u{ea98}', "edit" => '\u{ea73}', "list-unordered" => '\u{eb17}', "debug-pause" => '\u{ead1}', "play" => '\u{eb2c}', "save" => '\u{eb4b}',
        "zap" => '\u{ea86}', "pulse" => '\u{eb31}', "error" => '\u{ea87}', "smiley" => '\u{eb54}', "trash" => '\u{ea81}', "sparkle" => '\u{ec10}', "plug" => '\u{eb2d}', "dash" => '\u{eacc}', "hash" => '\u{ea90}',
        "copy" => '\u{ebcc}', "ellipsis" => '\u{ea7c}', "check" => '\u{eab2}', "tools" => '\u{eb6d}', "shield" => '\u{eb53}', "remote" => '\u{eb3a}', "root-folder" => '\u{eb46}', "primitive-square" => '\u{ea72}', _ => '\u{ea71}' };
    text(glyph.to_string()).font(Font::with_name("codicon")).size(u32::from(size)).color(color)
}
fn icon_button<'a>(name: &str, label: &'a str, message: Message, p: Palette) -> Element<'a, Message> {
    tooltip(button(icon(name, 14, p.muted)).on_press(message).width(24).height(24).padding(5).style(quiet_button), text(label).size(11), tooltip::Position::Bottom).into()
}
fn menu_action<'a>(name: &str, label: &'a str, message: Message, p: Palette) -> Element<'a, Message> {
    button(row![icon(name, 14, p.muted), text(label).size(12)].spacing(8).align_y(Alignment::Center)).on_press(message).padding([7, 8]).width(Fill).style(quiet_button).into()
}
pub(super) fn panel(background: Color, radius: f32, border: Option<Color>) -> container::Style {
    container::Style { background: Some(background.into()), border: Border { color: border.unwrap_or(Color::TRANSPARENT), width: if border.is_some() { 1.0 } else { 0.0 }, radius: radius.into() }, ..Default::default() }
}
/// STATUSBAR_ACTION_CLASS: full-height, 11px, tertiary text, hover tint; `active` is the bg-accent/55 state.
fn bar_button(theme: &Theme, status: button::Status, active: bool) -> button::Style {
    let p = theme.palette();
    let hover = matches!(status, button::Status::Hovered | button::Status::Pressed);
    button::Style { background: (active || hover).then_some(Color { a: if active { 0.10 } else { 0.06 }, ..p.text }.into()),
        text_color: p.text, border: Border::default(), ..Default::default() }
}
fn bar_item<'a>(glyph: &str, label: String, title: &str, message: Message, active: bool, p: Palette) -> Element<'a, Message> {
    let tint = if active { p.foreground } else { p.muted };
    let mut content = row![icon(glyph, 12, tint)].spacing(4).height(20).align_y(Alignment::Center);
    if !label.is_empty() { content = content.push(text(label).size(11).color(tint)); }
    tooltip(button(content).on_press(message).height(20).padding([0, 6]).style(move |t, s| bar_button(t, s, active)),
        text(title.to_owned()).size(11), tooltip::Position::Top).into()
}
/// use-composer-placeholder.ts: a starter (new chat) or continuation (existing
/// chat), picked once per conversation and stable while it stays open.
fn composer_placeholder(conversation: Option<&str>, fresh: bool) -> &'static str {
    const STARTERS: [&str; 7] = ["What are we building?", "Give Hermes a task", "What's on your mind?", "Describe what you need", "What should we tackle?", "Ask anything", "Start with a goal"];
    const FOLLOW_UPS: [&str; 7] = ["Send a follow-up", "Add more context", "Refine the request", "What's next?", "Keep it going", "Push it further", "Adjust or continue"];
    let list = if fresh { &STARTERS } else { &FOLLOW_UPS };
    let seed = conversation.unwrap_or_default().bytes().fold(5381u32, |h, b| h.wrapping_mul(33) ^ b as u32);
    list[seed as usize % list.len()]
}
/// PageHeader H1: the display face at 40px on the page's 60px top.
fn page_title<'a>(title: &'a str) -> Element<'a, Message> {
    container(text(title).size(40).font(appearance::DISPLAY_FONT).line_height(1.1)).padding(Padding { top: 60.0, right: 40.0, bottom: 12.0, left: 40.0 }).into()
}
/// The session status dot before a tab label: a hollow ring while idle.
fn tab_dot<'a>(active: bool, p: Palette) -> Element<'a, Message> {
    container(Space::new().width(6).height(6)).style(move |_| container::Style {
        border: Border { color: if active { p.foreground } else { p.muted }, width: 1.0, radius: 3.0.into() }, ..Default::default() }).into()
}
/// Left sidebar width of the original layout (260 logical px).
const SIDEBAR_WIDTH: f32 = 260.0;
/// Uppercase tab labels. Iced has no letter-spacing, and the UI font lacks
/// hair spaces, so labels are drawn as-is.
fn spaced(label: &str) -> String { label.to_owned() }
fn keycap<'a>(label: &'a str, p: Palette) -> Element<'a, Message> {
    container(text(label).size(10).font(Font::DEFAULT).color(p.muted)).width(18).height(18).center_x(18).center_y(18)
        .style(move |_| panel(Color { a: 0.5, ..p.secondary }, 4.0, Some(p.border))).into()
}
/// Pane divider: a single hairline in the border ink (flat, not boxed).
fn hairline<'a>(p: Palette) -> rule::Rule<'a, Theme> {
    rule::vertical(1).style(move |theme| rule::Style { color: p.border, ..rule::default(theme) })
}
pub(super) fn quiet_button(theme: &Theme, status: button::Status) -> button::Style {
    let p = theme.palette(); button::Style { background: (status == button::Status::Hovered).then_some(Color { a: 0.06, ..p.primary }.into()), text_color: p.text,
        border: Border { radius: 3.0.into(), ..Default::default() }, ..Default::default() }
}
fn nav_button(theme: &Theme, status: button::Status, selected: bool) -> button::Style {
    let p = theme.palette(); button::Style { background: (selected || status == button::Status::Hovered).then_some(Color { a: if selected { 0.08 } else { 0.04 }, ..p.primary }.into()),
        text_color: p.text, border: Border { radius: 3.0.into(), ..Default::default() }, ..Default::default() }
}
fn primary_button(theme: &Theme, status: button::Status) -> button::Style {
    button::Style { background: Some(Color { a: if status == button::Status::Disabled { 0.4 } else { 1.0 }, ..theme.palette().primary }.into()), text_color: Color::WHITE,
        border: Border { radius: 4.0.into(), ..Default::default() }, ..Default::default() }
}
fn send_button(theme: &Theme, status: button::Status) -> button::Style {
    button::Style { background: Some(Color { a: if status == button::Status::Disabled { 0.3 } else { 1.0 }, ..theme.palette().text }.into()), text_color: theme.palette().background,
        border: Border { radius: 99.0.into(), ..Default::default() }, ..Default::default() }
}
fn outline_button(theme: &Theme, status: button::Status) -> button::Style {
    let mut style = quiet_button(theme, status); style.border.width = 1.0; style.border.color = Color { a: 0.18, ..theme.palette().text }; style
}
fn field_input(theme: &Theme, status: text_input::Status) -> text_input::Style {
    let mut style = text_input::default(theme, status); style.background = theme.palette().background.into(); style.border.radius = 4.0.into(); style
}
fn search_input(theme: &Theme, status: text_input::Status) -> text_input::Style {
    let mut style = field_input(theme, status); style.background = Color::TRANSPARENT.into(); style.border = Border::default(); style
}

/// Image and SVG handles built once per source. A handle created inside `view` gets a fresh id on
/// every redraw, so iced would decode and upload the same picture again each frame.
pub(super) fn static_image(bytes: &'static [u8]) -> iced::widget::image::Handle {
    static CACHE: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<usize, iced::widget::image::Handle>>> = std::sync::LazyLock::new(Default::default);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    cache.entry(bytes.as_ptr() as usize).or_insert_with(|| iced::widget::image::Handle::from_bytes(bytes)).clone()
}
pub(super) fn cached_svg(source: String) -> iced::widget::svg::Handle {
    static CACHE: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<String, iced::widget::svg::Handle>>> = std::sync::LazyLock::new(Default::default);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(handle) = cache.get(&source) { return handle.clone() }
    let handle = iced::widget::svg::Handle::from_memory(source.clone().into_bytes());
    cache.insert(source, handle.clone());
    handle
}
