//! Opt-in headless screen captures for visual parity review.
//! Run: HERMES_RUST_DATA_DIR=<empty dir> HERMES_HOME=<empty dir> HERMES_SHOTS=<out dir>
//!      cargo test -p hermes-desktop screens:: -- --ignored --nocapture
use super::*;
use iced_core::renderer::Headless;

async fn shoot(app: &Hermes, renderer: &mut iced::Renderer, name: &str) { shoot_element(app.view(), app, renderer, name).await }
async fn shoot_element(element: Element<'_, Message>, app: &Hermes, renderer: &mut iced::Renderer, name: &str) {
    let out = PathBuf::from(std::env::var_os("HERMES_SHOTS").expect("Set HERMES_SHOTS"));
    let size = iced::Size::new(1220.0, 800.0);
    let theme = ui::theme_for(app.dark);
    let cache = iced_runtime::user_interface::Cache::new();
    let mut tree = iced_runtime::UserInterface::build(element, size, cache, renderer);
    tree.draw(renderer, &theme, &iced_core::renderer::Style { text_color: theme.palette().text }, iced::mouse::Cursor::Unavailable);
    let pixels = Headless::screenshot(renderer, iced::Size::new(1220, 800), 1.0, iced::Color::WHITE);
    image::save_buffer(out.join(format!("{name}.png")), &pixels, 1220, 800, image::ColorType::Rgba8).unwrap();
}

#[tokio::test]
#[ignore = "writes PNG captures of the main screens"]
async fn capture_screens() {
    {
        let mut fonts = iced::advanced::graphics::text::font_system().write().unwrap();
        for bytes in [include_bytes!("../assets/fonts/codicon.ttf").as_slice(), welcome::FONT_BYTES, appearance::MONO_FONT_BYTES, appearance::UI_FONT_FILES[0], appearance::UI_FONT_FILES[1], appearance::UI_FONT_FILES[2], appearance::UI_FONT_FILES[3]] {
            fonts.load_font(std::borrow::Cow::Borrowed(bytes));
        }
    }
    let (mut app, _) = Hermes::new();
    let mut renderer = <iced::Renderer as Headless>::new(appearance::UI_FONT, 16.into(), None).await.unwrap();
    assert!(app.onboarding.open, "a new library opens the welcome splash");
    shoot(&app, &mut renderer, "00-welcome").await;
    let _ = app.update(Message::Onboarding(onboarding::Action::Next));
    shoot(&app, &mut renderer, "00-welcome-style").await;
    let _ = app.update(Message::Onboarding(onboarding::Action::Close));
    assert!(!onboarding::State::load(app.library.as_ref().unwrap().root()).open, "finishing the splash is remembered");
    shoot(&app, &mut renderer, "01-chat").await;
    if let Some(library) = app.library.clone() {
        for prompt in ["hi", "second chat", "third chat"] {
            let session = library.create_session(None, None).unwrap();
            let mut history = History::default();
            history.turns.push(hermes_engine::Turn { context: String::new(), attachments: vec![], prompt: prompt.into(), model: "m".into(), entries: vec![], status: hermes_engine::Status::Complete });
            std::fs::write(session.directory.join("conversation.json"), serde_json::to_vec(&history).unwrap()).unwrap();
            library.touch_session(&session.id).unwrap();
        }
        let first = library.sessions(None).unwrap().into_iter().find(|s| s.title == "hi").unwrap();
        library.rename_session(&first.id, "hi renamed").unwrap();
    }
    let _ = app.update(Message::NewChat);
    let _ = app.update(Message::NewChat);
    app.refresh_sessions();
    shoot(&app, &mut renderer, "02-sidebar").await;
    let _ = app.update(Message::Navigate(Screen::Tools));
    let count = hermes_engine::skill_catalog::entries().len();
    let _ = app.update(Message::Capability(capabilities::Action::Loaded(Ok(count))));
    shoot(&app, &mut renderer, "03-capabilities").await;
    let notes = "NousResearch/hermes-agent/skills/apple/apple-notes".to_owned();
    let _ = app.update(Message::Capability(capabilities::Action::Enable(notes.clone(), false)));
    let root = app.library.as_ref().unwrap().root().to_owned();
    assert!(hermes_engine::skill_catalog::status(&root).unwrap().disabled.contains(&notes), "switching a skill off is saved for the agent");
    let _ = app.update(Message::Capability(capabilities::Action::Search("apple".into())));
    shoot(&app, &mut renderer, "03b-capabilities-search").await;
    let _ = app.update(Message::Capability(capabilities::Action::Open(notes.clone())));
    shoot(&app, &mut renderer, "03c-capabilities-detail").await;
    let _ = app.update(Message::Capability(capabilities::Action::Show(capabilities::Page::Plugins)));
    shoot(&app, &mut renderer, "03d-capabilities-plugins").await;
    let _ = app.update(Message::Capability(capabilities::Action::Show(capabilities::Page::Toolsets)));
    let _ = app.update(Message::Capability(capabilities::Action::Toolset("file_tools", false)));
    assert!(!hermes_engine::settings::ProfileSettings::load(&root).unwrap().file_tools, "toolset switch is saved");
    shoot(&app, &mut renderer, "03e-capabilities-tools").await;
    let _ = app.update(Message::Capability(capabilities::Action::Toolset("file_tools", true)));
    let _ = app.update(Message::Capability(capabilities::Action::Show(capabilities::Page::Connectors)));
    shoot(&app, &mut renderer, "03f-capabilities-connectors").await;
    let _ = app.update(Message::Navigate(Screen::Provider));
    shoot(&app, &mut renderer, "04-settings").await;
    let _ = app.update(Message::Navigate(Screen::Chat));
    // Reasoning picker: a catalog model with an effort control shows the pill;
    // a pick is saved and survives a restart.
    let reasoning_model = hermes_engine::models::Model { id: app.config.model.clone(), name: "Reasoner".into(), context_length: None,
        max_output_tokens: None, supported_parameters: vec!["reasoning".into()], free: true };
    app.models.catalog = vec![reasoning_model.clone()];
    shoot(&app, &mut renderer, "05-reasoning-pill").await;
    let pill = iced::Rectangle::new(iced::Point::new(1050.0, 738.0), iced::Size::new(60.0, 24.0));
    let _ = app.update(Message::Reasoning(reasoning::Action::Anchor(format!("{}\n{}", app.config.base_url, app.config.model), Some(pill))));
    assert!(app.reasoning.anchor.is_some(), "the picker opens on its trigger");
    let _ = app.update(Message::Reasoning(reasoning::Action::Pick("high")));
    assert_eq!(app.models.options.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(app.models.reasoning_label(), "High");
    shoot(&app, &mut renderer, "06-reasoning-open").await;
    let _ = app.update(Message::Reasoning(reasoning::Action::Close));
    drop(app);
    let (mut app, _) = Hermes::new();
    app.models.catalog = vec![reasoning_model];
    assert_eq!(app.models.options.reasoning_effort.as_deref(), Some("high"), "effort survives restart");
    shoot(&app, &mut renderer, "07-reasoning-restart").await;

    // Statusbar: every Electron item on, and the approval mode switched to Off.
    for id in ["cache-hit-rate", "tokens-per-second", "system-resources", "context-usage"] {
        let _ = app.update(Message::Statusbar(statusbar::Action::ToggleItem(id)));
    }
    let _ = app.update(Message::Statusbar(statusbar::Action::Memory(Some(statusbar::sample_memory().unwrap_or((12 << 30, 16 << 30))))));
    let _ = app.update(Message::Statusbar(statusbar::Action::Approval));
    shoot(&app, &mut renderer, "08-statusbar-approval").await;
    let _ = app.update(Message::Statusbar(statusbar::Action::SetApproval("off")));
    let root = app.library.as_ref().unwrap().root().to_owned();
    assert_eq!(hermes_engine::settings::ProfileSettings::load(&root).unwrap().approval_mode, "off", "approval choice is saved for the agent");
    let _ = app.update(Message::Statusbar(statusbar::Action::Customize));
    shoot(&app, &mut renderer, "09-statusbar-customize").await;
    let _ = app.update(Message::Statusbar(statusbar::Action::Close));

    // Bots: create one, open its canonical chat, and come back to the roster.
    let _ = app.update(Message::Bot(bots::Action::Roster(true)));
    shoot(&app, &mut renderer, "10-bots-empty").await;
    let _ = app.update(Message::Bot(bots::Action::New));
    let _ = app.update(Message::Bot(bots::Action::Name("Research Helper".into())));
    let _ = app.update(Message::Bot(bots::Action::Description("Finds and summarizes papers".into())));
    shoot(&app, &mut renderer, "11-bots-new").await;
    let _ = app.update(Message::Bot(bots::Action::Save));
    assert_eq!(app.current.as_ref().and_then(|c| c.bot.as_deref()), Some("research-helper"), "creating a bot opens its chat");
    shoot(&app, &mut renderer, "12-bots-chat").await;
    let chat = app.current.as_ref().unwrap().id.clone();
    let _ = app.update(Message::Bot(bots::Action::Roster(false)));
    assert!(app.sessions.iter().all(|s| s.bot.is_none()), "bot chats stay out of Sessions");
    let _ = app.update(Message::Bot(bots::Action::Roster(true)));
    let _ = app.update(Message::Bot(bots::Action::Open("research-helper".into())));
    assert_eq!(app.current.as_ref().unwrap().id, chat, "a bot row always opens the same chat");
    app.screen = Screen::Bots;
    shoot(&app, &mut renderer, "13-bots-roster").await;
    // Roster sections: create one, file the bot into it, reload.
    app.screen = Screen::Chat;
    let _ = app.update(Message::Bot(bots::Action::NewSection(Some("research-helper".into()))));
    let _ = app.update(Message::Bot(bots::Action::SectionName("Clients".into())));
    let _ = app.update(Message::Bot(bots::Action::SaveSection));
    let prefs = std::fs::read_to_string(app.library.as_ref().unwrap().root().join("bots-preferences.json")).unwrap();
    assert!(prefs.contains("Clients") && prefs.contains("research-helper"), "sections persist with their bots");
    let _ = app.update(Message::Bot(bots::Action::RowMenu("research-helper".into())));
    shoot(&app, &mut renderer, "13b-bots-sections").await;
    let _ = app.update(Message::Bot(bots::Action::RowMenu("research-helper".into())));
    // Slash commands: the popup lists and filters; commands run on send.
    let _ = app.update(Message::Bot(bots::Action::Roster(false)));
    app.composer = text_editor::Content::with_text("/");
    assert!(slash::items(&app).len() > 15, "typing / lists the commands");
    shoot(&app, &mut renderer, "15-slash-popup").await;
    app.composer = text_editor::Content::with_text("/appr");
    assert_eq!(slash::items(&app).first().map(|i| i.name.as_str()), Some("/approvals"));
    app.composer = text_editor::Content::with_text("/approvals manual");
    let _ = app.update(Message::Send);
    assert_eq!(hermes_engine::settings::ProfileSettings::load(app.library.as_ref().unwrap().root()).unwrap().approval_mode, "manual", "/approvals sets the mode");
    assert!(app.composer.text().trim().is_empty(), "a command clears the composer");
    app.composer = text_editor::Content::with_text("/nonsense");
    let _ = app.update(Message::Send);
    assert!(app.banner.as_ref().is_some_and(|b| b.error), "unknown commands explain themselves");
    let _ = app.update(Message::SettingsView(settings::View::About));
    shoot(&app, &mut renderer, "14-about").await;

    // The redesign's dark portal shell over two of its artworks.
    let _ = app.update(Message::Navigate(Screen::Chat));
    let _ = app.update(Message::Bot(bots::Action::Roster(false)));
    app.dark = true;
    for artwork in [shell_art::Artwork::Ascend, shell_art::Artwork::Angel, shell_art::Artwork::Wing] {
        app.artwork = Some(artwork);
        shoot(&app, &mut renderer, &format!("20-dark-{}", artwork.id())).await;
    }
    let _ = app.update(Message::NewChat);
    shoot(&app, &mut renderer, "21-dark-new").await;
    let _ = app.update(Message::PortalAccent(shell_art::Accent::Green));
    let _ = app.update(Message::SettingsView(settings::View::Appearance));
    shoot(&app, &mut renderer, "21b-dark-make-it-yours-green").await;
    let _ = app.update(Message::PortalAccent(shell_art::Accent::Blue));
    for (view, name) in [(settings::View::Models, "model"), (settings::View::Chat, "chat"), (settings::View::Appearance, "appearance"), (settings::View::About, "about"),
        (settings::View::Workspace, "workspace"), (settings::View::Safety, "safety"), (settings::View::Memory, "memory"), (settings::View::Voice, "voice"),
        (settings::View::Advanced, "advanced"), (settings::View::Vault, "vault"), (settings::View::Notifications, "notifications"), (settings::View::Billing, "billing"),
        (settings::View::Accounts, "accounts"), (settings::View::ApiKeys, "apikeys"), (settings::View::CustomEndpoints, "endpoints"), (settings::View::Gateways, "gateways"),
        (settings::View::Shortcuts, "shortcuts"), (settings::View::Sessions, "sessions"), (settings::View::Backup, "backup")] {
        let _ = app.update(Message::SettingsView(view));
        shoot(&app, &mut renderer, &format!("22-dark-settings-{name}")).await;
    }
    let _ = app.update(Message::Connection(connections::Action::Mode(connections::Mode::Ssh)));
    shoot(&app, &mut renderer, "22-dark-settings-gateways-ssh").await;
    let _ = app.update(Message::Connection(connections::Action::Mode(connections::Mode::Remote)));
    shoot(&app, &mut renderer, "22-dark-settings-gateways-remote").await;
    app.dark = false;
    let _ = app.update(Message::SettingsView(settings::View::Safety));
    shoot(&app, &mut renderer, "23-light-settings-safety").await;
    let _ = app.update(Message::Navigate(Screen::Chat));
    shoot(&app, &mut renderer, "24-light-chat").await;
    app.dark = true;
    for (screen, name) in [(Screen::Tools, "capabilities"), (Screen::Artifacts, "artifacts"), (Screen::Messaging, "messaging")] {
        let _ = app.update(Message::Navigate(screen));
        shoot(&app, &mut renderer, &format!("25-dark-{name}")).await;
    }
}

/// Seeds HERMES_RUST_DATA_DIR with one chat whose reply produced an HTML page,
/// for verifying the native preview on a real window (webview.rs).
#[test]
#[ignore = "writes a demo library for manual webview verification"]
fn seed_html_artifact() {
    let data = PathBuf::from(std::env::var_os("HERMES_RUST_DATA_DIR").expect("Set HERMES_RUST_DATA_DIR"));
    let site = data.join("site"); std::fs::create_dir_all(&site).unwrap();
    std::fs::write(data.join("outside.html"), "<h1>OUTSIDE — should never load</h1>").unwrap();
    std::fs::write(site.join("report.html"), r#"<!doctype html><html><body style="background:#0e0e23;color:#e8e8f0;font:16px -apple-system;padding:32px">
<h1 style="color:#8585ff">Hermes HTML preview</h1><p id="js">JavaScript: not run</p>
<p><a href="https://example.com/">External link (offered, not loaded)</a></p>
<p><a href="../outside.html">Link outside the folder (blocked)</a></p>
<script>document.getElementById('js').textContent='JavaScript: running inside the sandboxed preview';
setTimeout(()=>{ try { window.open('https://popup.example/'); } catch(e) {} }, 300);</script></body></html>"#).unwrap();
    let library = Library::open(&data).unwrap();
    let session = library.create_session(None, Some("HTML report")).unwrap();
    let mut history = History::default();
    history.turns.push(hermes_engine::Turn { context: String::new(), attachments: vec![], prompt: "make a report".into(), model: "m".into(),
        entries: vec![Entry::Assistant { text: format!("Here it is: [report.html]({})", site.join("report.html").display()), calls: vec![] }], status: hermes_engine::Status::Complete });
    std::fs::write(session.directory.join("conversation.json"), serde_json::to_vec(&history).unwrap()).unwrap();
    library.touch_session(&session.id).unwrap();
}

/// This round's new surfaces: Accounts providers, HUD, plugin cards, group chats.
#[tokio::test]
#[ignore = "writes PNG captures of the round-two screens"]
async fn capture_round_two() {
    {
        let mut fonts = iced::advanced::graphics::text::font_system().write().unwrap();
        for bytes in [include_bytes!("../assets/fonts/codicon.ttf").as_slice(), welcome::FONT_BYTES, appearance::MONO_FONT_BYTES, appearance::UI_FONT_FILES[0], appearance::UI_FONT_FILES[1], appearance::UI_FONT_FILES[2], appearance::UI_FONT_FILES[3]] {
            fonts.load_font(std::borrow::Cow::Borrowed(bytes));
        }
    }
    let (mut app, _) = Hermes::new();
    let mut renderer = <iced::Renderer as Headless>::new(appearance::UI_FONT, 16.into(), None).await.unwrap();
    let _ = app.update(Message::Onboarding(onboarding::Action::Close));
    app.screen = Screen::Provider; app.settings_view = settings::View::Accounts;
    shoot(&app, &mut renderer, "30-accounts").await;
    let _ = app.update(Message::Account(accounts::Action::ToggleAll));
    shoot(&app, &mut renderer, "31-accounts-all").await;
    // Group chats: two bots, a group, its chat.
    app.screen = Screen::Chat;
    for name in ["Ada", "Grace"] { let _ = app.update(Message::Bot(bots::Action::New)); let _ = app.update(Message::Bot(bots::Action::Name(name.into()))); let _ = app.update(Message::Bot(bots::Action::Save)); }
    let _ = app.update(Message::Bot(bots::Action::NewGroup));
    let _ = app.update(Message::Bot(bots::Action::GroupName("Pair review".into())));
    let _ = app.update(Message::Bot(bots::Action::GroupMember("ada".into(), true)));
    let _ = app.update(Message::Bot(bots::Action::GroupMember("grace".into(), true)));
    shoot(&app, &mut renderer, "32-group-new").await;
    let _ = app.update(Message::Bot(bots::Action::SaveGroup));
    assert!(app.current.as_ref().and_then(|c| c.bot.as_deref()).is_some_and(|b| b.starts_with("group:")), "creating a group opens its chat");
    let _ = app.update(Message::Bot(bots::Action::Roster(true)));
    shoot(&app, &mut renderer, "33-group-chat").await;
    let _ = app.update(Message::Bot(bots::Action::Roster(false)));
    // HUD
    app.screen = Screen::Chat;
    shoot_element(app.hud_view(), &app, &mut renderer, "34-hud").await;
    // Plugins
    let _ = app.update(Message::Navigate(Screen::Tools));
    let _ = app.update(Message::Capability(capabilities::Action::Show(capabilities::Page::Plugins)));
    let _ = app.update(Message::Capability(capabilities::Action::Search("aave".into())));
    shoot(&app, &mut renderer, "35-plugins").await;
}

/// Round three: Webhooks, Profiles, Star map.
#[tokio::test]
#[ignore = "writes PNG captures of the round-three screens"]
async fn capture_round_three() {
    {
        let mut fonts = iced::advanced::graphics::text::font_system().write().unwrap();
        for bytes in [include_bytes!("../assets/fonts/codicon.ttf").as_slice(), welcome::FONT_BYTES, appearance::MONO_FONT_BYTES, appearance::UI_FONT_FILES[0], appearance::UI_FONT_FILES[1], appearance::UI_FONT_FILES[2], appearance::UI_FONT_FILES[3]] {
            fonts.load_font(std::borrow::Cow::Borrowed(bytes));
        }
    }
    let (mut app, _) = Hermes::new();
    let mut renderer = <iced::Renderer as Headless>::new(appearance::UI_FONT, 16.into(), None).await.unwrap();
    let _ = app.update(Message::Onboarding(onboarding::Action::Close));
    let _ = app.update(Message::Navigate(Screen::Webhooks));
    let _ = app.update(Message::Webhook(webhooks::Action::New));
    let _ = app.update(Message::Webhook(webhooks::Action::Name("github-push".into())));
    let _ = app.update(Message::Webhook(webhooks::Action::Prompt("Summarize the pushed commits".into())));
    let _ = app.update(Message::Webhook(webhooks::Action::Events("push".into())));
    let _ = app.update(Message::Webhook(webhooks::Action::Create));
    shoot(&app, &mut renderer, "40-webhooks").await;
    let _ = app.update(Message::Navigate(Screen::Profiles));
    let _ = app.update(Message::Profile(profiles::Action::New));
    let _ = app.update(Message::Profile(profiles::Action::NewName("work".into())));
    let _ = app.update(Message::Profile(profiles::Action::Create));
    shoot(&app, &mut renderer, "41-profiles").await;
    let root = app.library.as_ref().unwrap().root().to_owned();
    std::fs::create_dir_all(root.join("memories")).unwrap();
    std::fs::write(root.join("memories/MEMORY.md"), "Prefers concise answers\n§\nBuilding the Rust Hermes desktop").unwrap();
    let _ = app.update(Message::Navigate(Screen::Starmap));
    let first = app.starmap_first();
    let _ = app.update(Message::Starmap(starmap::Action::Select(first)));
    shoot(&app, &mut renderer, "42-starmap").await;
}

/// Routines: template gallery and a template's blanks.
#[tokio::test]
#[ignore = "writes PNG captures of the Routines page"]
async fn capture_routines() {
    {
        let mut fonts = iced::advanced::graphics::text::font_system().write().unwrap();
        for bytes in [include_bytes!("../assets/fonts/codicon.ttf").as_slice(), welcome::FONT_BYTES, appearance::MONO_FONT_BYTES, appearance::UI_FONT_FILES[0], appearance::UI_FONT_FILES[1], appearance::UI_FONT_FILES[2], appearance::UI_FONT_FILES[3]] {
            fonts.load_font(std::borrow::Cow::Borrowed(bytes));
        }
    }
    let (mut app, _) = Hermes::new();
    let mut renderer = <iced::Renderer as Headless>::new(appearance::UI_FONT, 16.into(), None).await.unwrap();
    let _ = app.update(Message::Onboarding(onboarding::Action::Close));
    app.screen = Screen::Cron;
    let _ = app.scheduled.update(scheduled::Action::New, app.config.clone());
    shoot(&app, &mut renderer, "50-routines-templates").await;
    let i = routines::TEMPLATES.iter().position(|t| t.key == "news-digest").unwrap();
    let _ = app.scheduled.update(scheduled::Action::TemplatePick(i), app.config.clone());
    shoot(&app, &mut renderer, "51-routines-template-form").await;
}

/// The HUD band takes clicks and typing like the main window's composer.
#[tokio::test]
#[ignore = "Hermes::new reads the process-wide data dir; needs a new empty HERMES_RUST_DATA_DIR"]
async fn hud_band_takes_clicks_and_typing() {
    use iced::{Point, keyboard, mouse, Event as E};
    let (mut app, _) = Hermes::new();
    app.onboarding.open = false;
    app.hud.on = true;
    let id = iced::window::Id::unique();
    app.hud.window = Some(id);
    let mut renderer = <iced::Renderer as Headless>::new(appearance::UI_FONT, 16.into(), None).await.unwrap();
    let size = iced::Size::new(super::hud::WIDTH, super::hud::HEIGHT);
    let mut cache = iced_runtime::user_interface::Cache::new();
    let mut clipboard = iced_core::clipboard::Null;
    let mut send = |app: &mut Hermes, cache: &mut iced_runtime::user_interface::Cache, renderer: &mut iced::Renderer, at: Point, events: Vec<E>| -> Vec<Message> {
        let mut ui = iced_runtime::UserInterface::build(app.view_window(id), size, std::mem::take(cache), renderer);
        let mut out = vec![];
        let _ = ui.update(&events, mouse::Cursor::Available(at), renderer, &mut clipboard, &mut out);
        *cache = ui.into_cache();
        for m in &out { let _ = app.update(m.clone()); }
        out
    };
    let press = |at| vec![E::Mouse(mouse::Event::CursorMoved { position: at }), E::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)), E::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left))];
    let composer = Point::new(300.0, 52.0);
    let clicked = send(&mut app, &mut cache, &mut renderer, composer, press(composer));
    eprintln!("click composer -> {:?}", clicked);
    let key = |c: &str| E::Keyboard(keyboard::Event::KeyPressed { key: keyboard::Key::Character(c.into()), modified_key: keyboard::Key::Character(c.into()), physical_key: keyboard::key::Physical::Unidentified(keyboard::key::NativeCode::Unidentified), location: keyboard::Location::Standard, modifiers: keyboard::Modifiers::empty(), text: Some(c.into()), repeat: false });
    let typed = send(&mut app, &mut cache, &mut renderer, composer, vec![key("h"), key("i")]);
    eprintln!("typed -> {:?} composer={:?}", typed, app.composer.text());
    assert_eq!(app.composer.text().trim(), "hi", "typing lands in the HUD composer");
    // Find the + button and the model control by pressing along the composer row.
    let shots = std::env::var_os("HERMES_SHOTS").map(PathBuf::from);
    let shot = |app: &Hermes, renderer: &mut iced::Renderer, name: &str| {
        let Some(out) = &shots else { return };
        let theme = ui::theme_for(app.dark);
        let mut ui = iced_runtime::UserInterface::build(app.view_window(id), size, iced_runtime::user_interface::Cache::new(), renderer);
        ui.draw(renderer, &theme, &iced_core::renderer::Style { text_color: theme.palette().text }, mouse::Cursor::Unavailable);
        let pixels = Headless::screenshot(renderer, iced::Size::new(size.width as u32, size.height as u32), 1.0, iced::Color::WHITE);
        image::save_buffer(out.join(format!("{name}.png")), &pixels, size.width as u32, size.height as u32, image::ColorType::Rgba8).unwrap();
    };
    let mut plus = None;
    'find: for y in [44.0, 52.0, 60.0] { for x in (12..120).step_by(4) {
        let at = Point::new(x as f32, y);
        let out = send(&mut app, &mut cache, &mut renderer, at, press(at));
        if out.iter().any(|m| matches!(m, Message::ToggleAttach)) { plus = Some(at); break 'find; }
        if !out.is_empty() { app.attach_menu = false; }
    } }
    assert!(plus.is_some(), "the + button is pressable in the HUD");
    assert!(app.attach_menu, "the + menu is open");
    shot(&app, &mut renderer, "hud-plus-menu");
    let mut model = None;
    'model: for y in [60.0, 64.0, 68.0] { for x in (500..600).rev().step_by(3) {
        let at = Point::new(x as f32, y);
        let out = send(&mut app, &mut cache, &mut renderer, at, press(at));
        if out.iter().any(|m| matches!(m, Message::Models(_) | Message::Reasoning(_))) { model = Some((at, format!("{out:?}"))); break 'model; }
    } }
    eprintln!("model control -> {model:?} picker={} anchor={:?}", app.models.picker, app.reasoning.anchor);
    assert!(model.is_some(), "the model control is pressable in the HUD");
    if app.reasoning.anchor.is_none() && !app.models.picker { app.reasoning.anchor = Some(iced::Rectangle { x: 520.0, y: 40.0, width: 60.0, height: 22.0 }); }
    shot(&app, &mut renderer, "hud-model-menu");
    app.reasoning.anchor = None;
    app.models.picker = true;
    shot(&app, &mut renderer, "hud-model-picker");
    app.models.picker = false;
    app.attach_menu = false;
    let leave = Point::new(super::hud::WIDTH - 22.0, 15.0);
    let left = send(&mut app, &mut cache, &mut renderer, leave, press(leave));
    eprintln!("leave -> {:?}", left);
    assert!(left.iter().any(|m| matches!(m, Message::Hud(super::hud::Action::Toggle))), "the leave button works");
}
