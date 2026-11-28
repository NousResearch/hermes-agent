//! Native port of official Settings/Providers geometry and copy.
//!
//! References at Hermes 0284ce17: app/settings/index.tsx, primitives.tsx,
//! app/overlays/overlay-{view,split-layout,breadcrumb-header}.tsx,
//! components/onboarding/{providers,flow}.tsx and i18n/en{,_onboarding}.ts.
//! Tabler outline icon paths are from tabler/tabler-icons (MIT).
use crate::appearance::text;
use super::{Hermes, Message, Screen, ui};
use iced::{
    Alignment, Border, Color, Element, Fill, Font, Length, Padding,
    widget::{
        Space, button, column, container, image, row, scrollable, svg, text_input,
    },
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum View {
    #[default]
    Models,
    Accounts,
    ApiKeys,
    CustomEndpoints, Chat, Workspace, Safety, Memory, Advanced, Sessions, Appearance, Shortcuts, Notifications, Gateways, Backup, Voice, About, Vault, Billing,
}

pub(super) fn provider(app: &Hermes) -> Element<'_, Message> {
    let view = if app.portal_starting || app.portal_login.is_some() {
        View::Accounts
    } else {
        app.settings_view
    };
    let (crumb, content) = match view {
        View::Models => ("Model", models(app)),
        View::About=>("About",about(app)),
        View::Billing=>("Billing",super::billing::view(app)),
        View::Vault=>("Credential Vault",super::vault::view(app)),
        View::Voice=>("Voice",super::voice::view(app)),
        View::Notifications=>("Notifications",super::notifications::view(app)),
        View::Gateways=>("Gateways",super::connections::view(app)),
        View::Backup=>("Backup",super::backup::view(app)),
        View::Appearance=>("Appearance",super::appearance::view(app)),
        View::Shortcuts=>("Keyboard Shortcuts",super::shortcuts::view(app)),
        page @ (View::Chat|View::Workspace|View::Safety|View::Memory|View::Advanced|View::Sessions) =>{
            let title=match page{View::Chat=>"Chat",View::Workspace=>"Workspace",View::Safety=>"Safety",View::Memory=>"Memory & Context",View::Advanced=>"Advanced",_=>"Sessions"};
            (title, super::profile_settings::view(app,page))
        }
        View::Accounts => ("Providers  ›  Accounts", accounts(app)),
        View::ApiKeys => ("Providers  ›  API keys", api_keys(app)),
        View::CustomEndpoints => ("Providers  ›  Custom Endpoints", endpoints(app)),
    };
    shell(app, crumb, content, false)
}

pub(super) fn imports(app: &Hermes) -> Element<'_, Message> {
    shell(
        app,
        "Sessions  ›  Import",
        app.imports
            .view(app.project.as_ref(), app.running)
            .map(Message::Import),
        true,
    )
}

/// The redesign's Settings page (settings/index.tsx in the portal shell,
/// portal-pages.css, portal-layout.css): a page nav panel with the "Settings"
/// title in the display face and 40px rows, then a 640px column with the page H1.
fn shell<'a>(
    app: &'a Hermes,
    crumb: &'static str,
    content: Element<'a, Message>,
    importing: bool,
) -> Element<'a, Message> {
    let p = ui::palette(app.dark);
    let view = if app.portal_starting || app.portal_login.is_some() { View::Accounts } else { app.settings_view };
    let query = app.settings_query.trim().to_lowercase();
    let visible = |label: &str| query.is_empty() || label.to_lowercase().contains(&query);
    let mut nav = column![].spacing(4);
    let push = |nav: iced::widget::Column<'a, Message>, glyph: &'static str, name: &'static str, keywords: &str, active: bool, nested: bool, message: Message| {
        if visible(&format!("{name} {keywords}")) { nav.push(nav_item(glyph, name, active, nested, message, app.dark)) } else { nav }
    };
    // Section order of SECTIONS + navGroups in settings/index.tsx.
    nav = push(nav, "cpu", "Model", "models provider", !importing && view == View::Models, false, Message::SettingsView(View::Models));
    for (page, name, glyph, keywords) in [
        (View::Chat, "Chat", "comment", "behavior personality timezone"),
        (View::Appearance, "Appearance", "palette", "display color light dark system theme typeface backdrop font size zoom reopen"),
        (View::Workspace, "Workspace", "folder", "files projects working directory"),
        (View::Safety, "Safety", "shield", "approval timeout permissions"),
        (View::Memory, "Memory & Context", "brain", "agent memory user profile character limit"),
        (View::Voice, "Voice", "mic", "dictation speech microphone read aloud"),
        (View::Advanced, "Advanced", "settings", "runtime output keep awake"),
        (View::Vault, "Credential Vault", "key", "password login credential domain"),
        (View::Notifications, "Notifications", "bell", "approval needed response ready turn failed mute sounds"),
        (View::Billing, "Billing", "card", "account credit balance spending payment method"),
    ] { nav = push(nav, glyph, name, keywords, !importing && view == page, false, Message::SettingsView(page)); }
    let providers = matches!(view, View::Accounts | View::ApiKeys | View::CustomEndpoints);
    nav = push(nav, "bolt", "Providers", "accounts api keys custom endpoints", !importing && providers, false, Message::SettingsView(View::Accounts));
    if providers || !query.is_empty() {
        for (glyph, name, page) in [("account", "Accounts", View::Accounts), ("key", "API keys", View::ApiKeys), ("globe", "Custom Endpoints", View::CustomEndpoints)] {
            nav = push(nav, glyph, name, "providers", !importing && view == page, true, Message::SettingsView(page));
        }
    }
    nav = push(nav, "globe", "Gateways", "local gateway connection", !importing && view == View::Gateways, false, Message::SettingsView(View::Gateways));
    nav = push(nav, "keyboard", "Keyboard Shortcuts", "keybinds hotkeys", !importing && view == View::Shortcuts, false, Message::SettingsView(View::Shortcuts));
    nav = push(nav, "archive", "Sessions", "archived import", !importing && view == View::Sessions, false, Message::SettingsView(View::Sessions));
    if importing || view == View::Sessions || query.contains("import") {
        nav = push(nav, "archive", "Import", "sessions", importing, true, Message::Navigate(Screen::Import));
    }
    nav = push(nav, "archive", "Backup", "export restore data transfer", !importing && view == View::Backup, false, Message::SettingsView(View::Backup));
    nav = push(nav, "info", "About", "version updates", !importing && view == View::About, false, Message::SettingsView(View::About));
    let search = row![
        icon("search", 14.0, p.muted),
        text_input("Search settings...", &app.settings_query).on_input(Message::SettingsSearch).size(13).padding([6, 0])
            .style(move |theme, status| { let mut style = text_input::default(theme, status); style.background = Color::TRANSPARENT.into();
                style.border.width = 0.0; style.value = p.foreground; style.placeholder = p.muted; style }),
    ].spacing(8).align_y(Alignment::Center);
    // Title 30px in and down, rows on the 20px inset, 56px from the title row.
    let head = column![
        container(text("Settings").size(30).font(super::appearance::DISPLAY_FONT).line_height(1.2)).padding(Padding { left: 10.0, ..Padding::ZERO }),
        container(search).padding([0, 12]),
    ].spacing(14);
    let nav = scrollable(nav).height(Fill).direction(scrollable::Direction::Vertical(scrollable::Scrollbar::new().width(3).scroller_width(3).margin(0)))
        .style(move |theme, status| { let mut style = scrollable::default(theme, status);
            style.vertical_rail.scroller.background = Color { a: 0.12, ..p.foreground }.into(); style.vertical_rail.background = None; style });
    let panel = container(column![head, Space::new().height(20), nav])
        .padding(Padding { top: 30.0, right: 20.0, bottom: 16.0, left: 20.0 }).width(260).height(Fill)
        .style(move |_| container::Style { background: Some(p.sidebar.into()), ..Default::default() });
    let hairline = container(Space::new().width(1).height(Fill)).style(move |_| container::Style { background: Some(p.border.into()), ..Default::default() });
    let close = button(icon("x", 16.0, p.muted)).on_press(Message::Navigate(Screen::Chat)).padding(6).style(button::text);
    let title = crumb.rsplit("  ›  ").next().unwrap_or(crumb);
    let column_body = column![
        container(text(title).size(40).font(super::appearance::DISPLAY_FONT).line_height(1.1)).padding([0, 32]),
        Space::new().height(32),
        content,
    ].height(Fill).max_width(640);
    let main = column![
        row![Space::new().width(Fill), close].padding(Padding { top: 10.0, right: 12.0, ..Padding::ZERO }),
        container(column_body).center_x(Fill).padding(Padding { top: 60.0, ..Padding::ZERO }).height(Fill),
    ].width(Fill).height(Fill);
    let portal = app.artwork.is_some();
    container(row![panel, hairline, main]).width(Fill).height(Fill)
        .style(move |_| container::Style { background: Some(if portal { super::shell_art::roles::stage_veil(app.dark) } else { p.background }.into()),
            text_color: Some(p.foreground), ..Default::default() }).into()
}
fn models(app: &Hermes) -> Element<'_, Message> {
    settings_content(super::models::model_rows(app))
}

fn accounts(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let connected = app.portal_connected;
    let mut content = column![
        row![section_heading("key", "Connect an account", app.dark), Space::new().width(Fill), button(text("Have an API key instead?").size(12)).on_press(Message::SettingsView(View::ApiKeys)).style(button::text)].align_y(Alignment::Center),
        text("Sign in with a subscription — no API key to copy. Hermes runs the browser sign-in for you, right here in the app.").size(12).color(p.muted).line_height(1.5),
        Space::new().height(1),
    ].spacing(10);
    if let Some(banner)=&app.banner{content=content.push(text(banner.text.as_str()).size(13).color(if banner.error{Color::from_rgb8(192,71,72)}else{p.muted}));}
    let badge = container(
        text(if connected {
            "✓ Connected"
        } else {
            "RECOMMENDED"
        })
        .size(if connected { 11 } else { 10 })
        .font(semibold())
        .color(if connected { p.accent } else { p.background }),
    )
    .padding([3, 7])
    .style(move |_| container::Style {
        background: Some(
            if connected {
                Color {
                    a: 0.10,
                    ..p.accent
                }
            } else {
                p.accent
            }
            .into(),
        ),
        ..Default::default()
    });
    let featured = button(
        row![
            column![
                row![
                    image(ui::static_image(
                        include_bytes!("../assets/apple-touch-icon.png").as_slice()
                    ))
                    .width(20)
                    .height(20),
                    text("Nous Portal").size(14).font(semibold()),
                    badge
                ]
                .spacing(8)
                .align_y(Alignment::Center),
                text("One subscription, 300+ frontier models — the recommended way to run Hermes")
                    .size(12)
                    .color(p.muted)
                    .line_height(1.5),
            ]
            .spacing(5)
            .width(Fill),
            icon("chevron", 16.0, p.accent),
        ]
        .spacing(16)
        .align_y(Alignment::Center),
    )
    .on_press_maybe(
        (!app.running && !app.portal_starting && app.portal_login.is_none()).then_some(
            if connected {
                Message::ChoosePortal(true)
            } else {
                Message::PortalStart
            },
        ),
    )
    .padding([10, 12])
    .width(Fill)
    .style(move |_, status| button::Style {
        background: Some(
            Color {
                a: if status == button::Status::Hovered {
                    0.10
                } else {
                    0.06
                },
                ..p.accent
            }
            .into(),
        ),
        text_color: p.foreground,
        border: Border {
            color: Color {
                a: 0.30,
                ..p.accent
            },
            width: 1.0,
            radius: 8.0.into(),
        },
        ..Default::default()
    });
    content = content.push(featured);
    if let Some(login) = &app.portal_login {
        content = content.push(Space::new().height(15));
        content = content.push(text("Sign in with Nous Portal").size(14).font(semibold()));
        content = content.push(
            text("We opened Nous Portal in your browser. Enter this code there:")
                .size(13)
                .color(p.muted),
        );
        let mut cells = row![].spacing(6).align_y(Alignment::Center);
        for character in login.user_code.chars() {
            if character == '-' || character == ' ' {
                cells = cells.push(container(text("–").size(18).color(p.muted)).width(6));
            } else {
                cells = cells.push(
                    container(text(character.to_string()).size(20).font(Font::MONOSPACE))
                        .width(40)
                        .height(40)
                        .center_x(40)
                        .center_y(40)
                        .style(move |_| container::Style {
                            border: Border {
                                color: p.border,
                                width: 1.0,
                                radius: 6.0.into(),
                            },
                            ..Default::default()
                        }),
                );
            }
        }
        content = content.push(
            container(
                button(cells)
                    .on_press(Message::Copy(login.user_code.clone()))
                    .style(button::text),
            )
            .width(Fill)
            .center_x(Fill),
        );
        content = content.push(
            row![
                button(text("Re-open verification page").size(12))
                    .on_press(Message::Link(login.verification_url.clone()))
                    .style(button::text),
                Space::new().width(Fill),
                text("Waiting for you to authorize...")
                    .size(12)
                    .color(p.muted),
                button(text("Cancel").size(12))
                    .on_press(Message::PortalCancel)
                    .style(button::text),
            ]
            .spacing(12)
            .align_y(Alignment::Center),
        );
    } else if app.portal_starting {
        content = content.push(
            text("Starting sign-in for Nous Portal...")
                .size(13)
                .color(p.muted),
        );
    } else if connected && app.is_portal() {
        content = content.push(Space::new().height(12));
        content = content.push(text("Nous Portal connected").size(14).font(semibold()));
        content = content.push(text("Default model").size(12).color(p.muted));
        content = content.push(text(&app.config.model).size(14).font(Font::MONOSPACE));
        content = content.push(
            row![
                button(text("Change").size(12))
                    .on_press(Message::SettingsView(View::Models))
                    .style(button::text),
                button(text("Begin").size(13))
                    .on_press(Message::SaveProvider)
                    .padding([7, 16])
            ]
            .spacing(12),
        );
    }
    content = content.push(Space::new().height(10));
    content = content.push(super::accounts::view(app));
    if connected {
        content=content.push(button(text(if app.models.disconnecting {"Disconnecting…"} else {"Disconnect Nous Portal"}).size(12)).on_press_maybe((!app.running && !app.models.disconnecting).then_some(Message::Models(super::models::Action::Disconnect))).padding([7,10]).style(button::text));
    }
    if let Some(error)=&app.models.error {content=content.push(text(error).size(12).color(p.muted));}
    settings_content(content.into())
}

fn api_keys(app: &Hermes) -> Element<'_, Message> {
    use super::models::{Action, Preset};
    let p=ui::palette(app.dark);
    let mut content=column![section_heading("key","API keys",app.dark)].spacing(12);
    for preset in [Preset::OpenRouter,Preset::OpenAi,Preset::Anthropic] {
        content=content.push(button(row![text(preset.name()).size(14).font(semibold()),Space::new().width(Fill),icon("chevron",16.0,p.muted)].spacing(10)).on_press_maybe((!app.running).then_some(Message::Models(Action::Preset(preset)))).padding([10,12]).width(Fill).style(button::text));
    }
    content=content.push(super::models::endpoint_rows(app));
    if !app.is_portal() {
        let mut key=text_input("Paste API key",&app.config.api_key).secure(true).padding(9).size(13);
        let mut model=text_input("Model ID",&app.config.model).padding(9).size(13);
        let mut name=text_input("Endpoint name",&app.models.endpoint_name).padding(9).size(13);
        if !app.running {key=key.on_input(Message::ApiKey);model=model.on_input(Message::Model);name=name.on_input(|value|Message::Models(Action::Name(value)));}
        content=content.push(Space::new().height(8)).push(text(super::models::provider_label(app)).size(14).font(semibold())).push(name).push(key).push(model)
            .push(text("Saved privately for this endpoint. Changing the endpoint never forwards this key to another host.").size(12).color(p.muted))
            .push(row![button(text("Discover models").size(12)).on_press_maybe((!app.running && !app.models.loading).then_some(Message::Models(Action::Refresh))).padding([7,10]).style(button::secondary),
                button(text("Choose model").size(12)).on_press_maybe((!app.running).then_some(Message::Models(Action::Open))).padding([7,10]).style(button::secondary),Space::new().width(Fill),
                button(text("Save").size(13)).on_press_maybe((!app.running).then_some(Message::Models(Action::ApplyProvider(false)))).padding([7,14])].spacing(8));
    }
    if let Some(error)=&app.models.error {content=content.push(text(error).size(12).color(p.muted));}
    if app.models.loading {content=content.push(text("Loading models…").size(12).color(p.muted));}
    settings_content(content.into())
}

fn endpoints(app: &Hermes) -> Element<'_, Message> {
    use super::models::{Action,Preset};
    let p=ui::palette(app.dark);
    let mut endpoint=text_input("https://example.com/v1",&app.config.base_url).padding(9).size(13);
    let mut model=text_input("Model ID",&app.config.model).padding(9).size(13);
    let mut key=text_input("API key (optional)",&app.config.api_key).secure(true).padding(9).size(13);
    let mut name=text_input("Endpoint name",&app.models.endpoint_name).padding(9).size(13);
    if !app.running {endpoint=endpoint.on_input(Message::BaseUrl);model=model.on_input(Message::Model);key=key.on_input(Message::ApiKey);name=name.on_input(|value|Message::Models(Action::Name(value)));}
    let mut content=column![section_heading("globe","Local / custom endpoint",app.dark),
        text("Point Hermes at an OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama) or a native Anthropic Messages endpoint.").size(12).color(p.muted),
        super::models::endpoint_rows(app),
        button(text("Add local endpoint").size(12)).on_press_maybe((!app.running).then_some(Message::Models(Action::Preset(Preset::Local)))).padding([6,10]).style(button::secondary),
        Space::new().height(8),text("Name").size(12),name,text("Base URL").size(12),endpoint,text("Model").size(12),model,text("API key").size(12),key,
        row![button(text("Discover models").size(12)).on_press_maybe((!app.running && !app.models.loading).then_some(Message::Models(Action::Refresh))).padding([7,10]).style(button::secondary),
            button(text("Choose model").size(12)).on_press_maybe((!app.running).then_some(Message::Models(Action::Open))).padding([7,10]).style(button::secondary),Space::new().width(Fill),
            button(text("Save").size(13)).on_press_maybe((!app.running).then_some(Message::Models(Action::SaveEndpoint))).padding([7,14])].spacing(8),
    ].spacing(12);
    if let Some(error)=&app.models.error {content=content.push(text(error).size(12).color(p.muted));}
    settings_content(content.into())
}

/// SettingsCard: the card role (2% fill, 5% hairline, r4) holding 64px rows on a
/// 16px inset, divided by 3% lines.
pub(super) fn card<'a>(rows: Vec<Element<'a, Message>>, dark: bool) -> Element<'a, Message> {
    let p = ui::palette(dark);
    let mut body = column![];
    for (index, row_element) in rows.into_iter().enumerate() {
        if index > 0 { body = body.push(container(Space::new().height(1).width(Fill)).style(move |_| container::Style { background: Some(Color { a: 0.03, ..p.foreground }.into()), ..Default::default() })); }
        body = body.push(row_element);
    }
    container(body).width(Fill).style(move |_| container::Style {
        background: Some(Color { a: 0.02, ..p.foreground }.into()),
        border: Border { color: Color { a: 0.05, ..p.foreground }, width: 1.0, radius: 4.0.into() }, ..Default::default() }).into()
}
/// One settings row: 14px Medium title, 14px tertiary description, the control at the end.
pub(super) fn card_row<'a>(title: &'a str, description: &'a str, control: Element<'a, Message>, dark: bool) -> Element<'a, Message> {
    let p = ui::palette(dark);
    let mut label = column![text(title).size(14).font(medium())].spacing(2).width(Fill);
    if !description.is_empty() { label = label.push(text(description).size(13).color(Color { a: 0.6, ..p.foreground }).line_height(1.4)); }
    container(row![label, control].spacing(24).align_y(Alignment::Center)).padding([12, 16]).width(Fill).height(Length::Shrink)
        .style(|_| container::Style::default()).into()
}
/// `card_row` for titles built at runtime (saved connection names and details).
pub(super) fn card_row_owned<'a>(title: String, description: String, control: Element<'a, Message>, dark: bool) -> Element<'a, Message> {
    let p = ui::palette(dark);
    let mut label = column![text(title).size(14).font(medium())].spacing(2).width(Fill);
    if !description.is_empty() { label = label.push(text(description).size(13).color(Color { a: 0.6, ..p.foreground })); }
    container(row![label, control].spacing(24).align_y(Alignment::Center)).padding([12, 16]).width(Fill).into()
}
/// Section head above a card: 14px at .8 ink.
pub(super) fn section_title<'a>(title: &'a str, dark: bool) -> Element<'a, Message> {
    let p = ui::palette(dark);
    container(text(title).size(14).font(medium()).color(Color { a: 0.8, ..p.foreground })).padding(Padding { bottom: 4.0, ..Padding::ZERO }).into()
}
/// Portal switch: the brand fill when on.
pub(super) fn switch<'a>(on: bool, message: impl Fn(bool) -> Message + 'a, dark: bool) -> Element<'a, Message> {
    let p = ui::palette(dark);
    iced::widget::toggler(on).on_toggle(message).size(18).style(move |theme, status| {
        let mut style = iced::widget::toggler::default(theme, status);
        let toggled = matches!(status, iced::widget::toggler::Status::Active { is_toggled: true } | iced::widget::toggler::Status::Hovered { is_toggled: true });
        style.background = if toggled || on { super::shell_art::accent().fill().into() } else { Color { a: 0.12, ..p.foreground }.into() };
        style.foreground = if toggled || on { super::shell_art::accent().on().into() } else { Color::WHITE.into() };
        style
    }).into()
}
/// Portal text field: a faint fill, hairline border, r3.
pub(super) fn field_style(dark: bool) -> impl Fn(&iced::Theme, text_input::Status) -> text_input::Style {
    let p = ui::palette(dark);
    move |theme, status| {
        let mut style = text_input::default(theme, status);
        style.background = Color { a: 0.04, ..p.foreground }.into();
        style.border = Border { color: if matches!(status, text_input::Status::Focused { .. }) { p.accent } else { Color { a: 0.1, ..p.foreground } }, width: 1.0, radius: 3.0.into() };
        style.value = p.foreground; style.placeholder = Color { a: 0.4, ..p.foreground };
        style
    }
}
fn medium() -> Font { Font { weight: iced::font::Weight::Medium, ..super::appearance::font() } }
pub(super) fn settings_content(content: Element<'_, Message>) -> Element<'_, Message> {
    quiet_scroll(
        container(content)
            .padding(Padding {
                top: 4.0,
                right: 32.0,
                bottom: 80.0,
                left: 32.0,
            })
            .width(Fill),
    )
    .height(Fill)
    .into()
}

fn section_heading<'a>(name: &'static str, title: &'a str, dark: bool) -> Element<'a, Message> {
    let p = ui::palette(dark);
    row![
        icon(name, 16.0, p.muted),
        text(title).size(14).font(semibold())
    ]
    .spacing(8)
    .align_y(Alignment::Center)
    .into()
}

fn nav_item<'a>(
    name: &'static str,
    title: &'a str,
    active: bool,
    nested: bool,
    message: Message,
    dark: bool,
) -> Element<'a, Message> {
    let p = ui::palette(dark);
    // Top-level rows read at full ink with icons at .4; nested sub-pages stay quieter.
    let glyph = Color { a: if active { 1.0 } else { 0.4 }, ..p.foreground };
    let label = if nested && !active { p.muted } else { p.foreground };
    button(
        row![icon(name, if nested { 14.0 } else { 16.0 }, glyph), text(title).size(if nested { 13 } else { 14 }).font(semibold()).color(label)]
            .spacing(12).align_y(Alignment::Center),
    )
    .on_press(message)
    .height(if nested { 34 } else { 40 })
    .width(Fill)
    .padding([0, if nested { 24 } else { 12 }])
    .style(move |_, status| button::Style {
        background: if active { Some(Color { a: 0.05, ..p.foreground }.into()) }
            else if status == button::Status::Hovered { Some(Color { a: 0.03, ..p.foreground }.into()) } else { None },
        text_color: label,
        border: Border { radius: 3.0.into(), ..Default::default() },
        ..Default::default()
    })
    .into()
}


pub(super) fn semibold() -> Font {
    Font {
        weight: iced::font::Weight::Semibold,
        ..crate::appearance::font()
    }
}

// Original Tabler outline paths, as used by the official icons.ts aliases.
pub(super) fn icon(name: &str, size: f32, color: Color) -> Element<'static, Message> {
    if name == "account" {
        // Original codicon-account from the bundled source font (U+EB99).
        return text("\u{eb99}")
            .font(Font::with_name("codicon"))
            .size(size)
            .color(color)
            .into();
    }
    let body = match name {
        "palette"=>"<path d='M12 21a9 9 0 0 1 0 -18c4.97 0 9 3.582 9 8c0 1.06 -.474 2.078 -1.318 2.828c-.844 .75 -1.989 1.172 -3.182 1.172h-2.5a2 2 0 0 0 -1 3.75a1.3 1.3 0 0 1 -1 2.25'/><path d='M7.5 10.5a1 1 0 1 0 2 0a1 1 0 1 0 -2 0'/><path d='M11.5 7.5a1 1 0 1 0 2 0a1 1 0 1 0 -2 0'/><path d='M15.5 10.5a1 1 0 1 0 2 0a1 1 0 1 0 -2 0'/>",
        "folder"=>"<path d='M5 4h4l3 3h7a2 2 0 0 1 2 2v8a2 2 0 0 1 -2 2h-14a2 2 0 0 1 -2 -2v-11a2 2 0 0 1 2 -2'/>",
        "comment"=>"<path d='M3 20l1.3 -3.9c-2.324 -3.437 -1.426 -7.872 2.1 -10.374c3.526 -2.501 8.59 -2.296 11.845 .48c3.255 2.777 3.695 7.266 1.029 10.501c-2.666 3.235 -7.615 4.215 -11.574 2.293l-4.7 1'/>",
        "mic"=>"<path d='M9 5a3 3 0 0 1 3 -3a3 3 0 0 1 3 3v5a3 3 0 0 1 -3 3a3 3 0 0 1 -3 -3l0 -5'/><path d='M5 10a7 7 0 0 0 14 0'/><path d='M8 21l8 0'/><path d='M12 17l0 4'/>",
        "shield"=>"<path d='M5 13a2 2 0 0 1 2 -2h10a2 2 0 0 1 2 2v6a2 2 0 0 1 -2 2h-10a2 2 0 0 1 -2 -2v-6'/><path d='M11 16a1 1 0 1 0 2 0a1 1 0 0 0 -2 0'/><path d='M8 11v-4a4 4 0 1 1 8 0v4'/>",
        "brain"=>"<path d='M15.5 13a3.5 3.5 0 0 0 -3.5 3.5v1a3.5 3.5 0 0 0 7 0v-1.8'/><path d='M8.5 13a3.5 3.5 0 0 1 3.5 3.5v1a3.5 3.5 0 0 1 -7 0v-1.8'/><path d='M17.5 16a3.5 3.5 0 0 0 0 -7h-.5'/><path d='M19 9.3v-2.8a3.5 3.5 0 0 0 -7 0'/><path d='M6.5 16a3.5 3.5 0 0 1 0 -7h.5'/><path d='M5 9.3v-2.8a3.5 3.5 0 0 1 7 0v10'/>",
        "settings"=>"<path d='M7 10h3v-3l-3.5 -3.5a6 6 0 0 1 8 8l6 6a2 2 0 0 1 -3 3l-6 -6a6 6 0 0 1 -8 -8l3.5 3.5'/>",
        "keyboard"=>"<path d='M2 8a2 2 0 0 1 2 -2h16a2 2 0 0 1 2 2v8a2 2 0 0 1 -2 2h-16a2 2 0 0 1 -2 -2l0 -8'/><path d='M6 10l0 .01'/><path d='M10 10l0 .01'/><path d='M14 10l0 .01'/><path d='M18 10l0 .01'/><path d='M6 14l0 .01'/><path d='M18 14l0 .01'/><path d='M10 14l4 .01'/>",
        "bell"=>"<path d='M10 5a2 2 0 1 1 4 0a7 7 0 0 1 4 6v3a4 4 0 0 0 2 3h-16a4 4 0 0 0 2 -3v-3a7 7 0 0 1 4 -6'/><path d='M9 17v1a3 3 0 0 0 6 0v-1'/>",
        "cpu" => {
            "<path d='M5 6a1 1 0 0 1 1 -1h12a1 1 0 0 1 1 1v12a1 1 0 0 1 -1 1h-12a1 1 0 0 1 -1 -1l0 -12'/><path d='M9 9h6v6h-6l0 -6M3 10h2M3 14h2M10 3v2M14 3v2M21 10h-2M21 14h-2M14 21v-2M10 21v-2'/>"
        }
        "bolt" => "<path d='M13 3l0 7l6 0l-8 11l0 -7l-6 0l8 -11'/>",
        "key" => {
            "<path d='M16.555 3.843l3.602 3.602a2.877 2.877 0 0 1 0 4.069l-2.643 2.643a2.877 2.877 0 0 1 -4.069 0l-.301 -.301l-6.558 6.558a2 2 0 0 1 -1.239 .578l-.175 .008h-1.172a1 1 0 0 1 -.993 -.883l-.007 -.117v-1.172a2 2 0 0 1 .467 -1.284l.119 -.13l.414 -.414h2v-2h2v-2l2.144 -2.144l-.301 -.301a2.877 2.877 0 0 1 0 -4.069l2.643 -2.643a2.877 2.877 0 0 1 4.069 0'/><path d='M15 9h.01'/>"
        }
        "globe" => {
            "<path d='M7 9a4 4 0 1 0 8 0a4 4 0 0 0 -8 0M5.75 15a8.015 8.015 0 1 0 9.25 -13M11 17v4M7 21h8'/>"
        }
        "archive" => {
            "<path d='M3 6a2 2 0 0 1 2 -2h14a2 2 0 0 1 2 2a2 2 0 0 1 -2 2h-14a2 2 0 0 1 -2 -2M5 8v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2 -2v-10M10 12l4 0'/>"
        }
        "search" => "<path d='M3 10a7 7 0 1 0 14 0a7 7 0 1 0 -14 0M21 21l-6 -6'/>",
        "x" => "<path d='M18 6l-12 12M6 6l12 12'/>",
        "chevron" => "<path d='M9 6l6 6l-6 6'/>",
        "refresh" => "<path d='M20 11a8.1 8.1 0 0 0 -15.5 -2m-.5 -4v4h4'/><path d='M4 13a8.1 8.1 0 0 0 15.5 2m.5 4v-4h-4'/>",
        "copy" => "<path d='M7 9.667a2.667 2.667 0 0 1 2.667 -2.667h8.666a2.667 2.667 0 0 1 2.667 2.667v8.666a2.667 2.667 0 0 1 -2.667 2.667h-8.666a2.667 2.667 0 0 1 -2.667 -2.667z'/><path d='M4.012 16.737a2 2 0 0 1 -1.012 -1.737v-10c0 -1.1 .9 -2 2 -2h10c.75 0 1.158 .385 1.5 1'/>",
        "chevron-down" => "<path d='M6 9l6 6l6 -6'/>",
        _ => "<path d='M3 12a9 9 0 1 0 18 0a9 9 0 0 0 -18 0M12 9h.01M11 12h1v4h1'/>",
    };
    let stroke = format!(
        "#{:02x}{:02x}{:02x}",
        (color.r * 255.0) as u8,
        (color.g * 255.0) as u8,
        (color.b * 255.0) as u8
    );
    let bytes = format!("<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='{stroke}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>{body}</svg>").into_bytes();
    svg(ui::cached_svg(String::from_utf8(bytes).unwrap_or_default()))
        .width(Length::Fixed(size))
        .height(Length::Fixed(size))
        .into()
}

/// Port of about-settings.tsx: VersionHero (BrandMark, app name, version ·
/// channel), the Updates section, and the VersionDetails card.
fn about(app:&Hermes)->Element<'_,Message>{
    let p=ui::palette(app.dark);
    let revision=option_env!("HERMES_DESKTOP_SOURCE_REVISION").unwrap_or("0284ce17d9bf9ed2f1832ad8950dfac2394d45ab");
    let version=env!("CARGO_PKG_VERSION");
    let brand=iced::widget::image(ui::static_image(if app.dark {include_bytes!("../assets/nous-girl-dark.png").as_slice()} else {include_bytes!("../assets/nous-girl.png").as_slice()}))
        .width(Length::Fixed(64.0)).height(Length::Fixed(64.0));
    let hero=column![brand,
        column![text("Hermes").size(18).font(ui::medium()),text(format!("Version {version} · Native (Rust)")).size(12).color(p.muted)].spacing(4).align_x(Alignment::Center)]
        .spacing(12).align_x(Alignment::Center).width(Fill).padding(Padding{top:24.0,bottom:8.0,..Padding::ZERO});
    let card=|content:Element<'static,Message>|container(content).padding([12,12]).width(Fill)
        .style(move|_|container::Style{background:Some(Color{a:0.35,..p.secondary}.into()),border:Border{radius:8.0.into(),width:1.0,color:Color{a:0.7,..p.border}},..Default::default()});
    let heading=|label:&'static str,glyph:&'static str|row![icon(glyph,16.0,p.muted),text(label).size(14).font(ui::medium())].spacing(8).align_y(Alignment::Center);
    // Updates: this build has no self-updater, so say so and point to the source.
    let updates=card(row![
        column![text("This version of Hermes can’t update itself from inside the app.").size(13),
            text("Rebuild from source to update this native build.").size(12).color(p.muted)].spacing(3).width(Fill),
        button(text("Release notes").size(12)).padding([5,10]).style(ui::quiet_button).on_press(Message::Link("https://github.com/NousResearch/hermes-agent/releases".into())),
    ].spacing(12).align_y(Alignment::Center).into());
    let row_item=|label:&'static str,value:Element<'static,Message>|row![text(label).size(13).color(p.muted),Space::new().width(Fill),value].spacing(16).align_y(Alignment::Center);
    let plain=|value:String|->Element<'static,Message>{text(value).size(13).into()};
    let commit:Element<'static,Message>=button(text(revision[..revision.len().min(14)].to_owned()).size(12).font(iced::Font::MONOSPACE).color(p.accent))
        .padding(0).style(ui::quiet_button).on_press(Message::Link(format!("https://github.com/NousResearch/hermes-agent/commit/{revision}"))).into();
    let mut details=column![
        row_item("Version",plain(format!("v{version}"))),
        row_item("Commit",commit),
        row_item("Channel",plain("Development".into())),
        row_item("Build Origin",plain("Local".into())),
        row_item("Distribution",plain(if cfg!(target_os="macos"){"Desktop app (macOS)".into()}else{"Desktop app".into()})),
        row_item("Runtime",plain("Embedded runtime (Rust)".into())),
        row_item("Platform",plain(format!("{} · {}",std::env::consts::OS,std::env::consts::ARCH))),
    ].spacing(8);
    let mut copy=format!("Hermes Desktop v{version} (native Rust)\nCommit {revision}\n{} / {}",std::env::consts::OS,std::env::consts::ARCH);
    if let Some(library)=&app.library{
        let folder=library.root().display().to_string();
        copy.push_str(&format!("\nData {folder}"));
        details=details.push(row_item("Data folder",text(folder).size(12).color(p.muted).into()));
    }
    let body=column![
        hero,
        column![
            heading("Updates","refresh"),updates,
            row![heading("Version details","info"),Space::new().width(Fill),
                button(row![icon("copy",13.0,p.muted),text("Copy").size(12)].spacing(5).align_y(Alignment::Center)).padding([4,8]).style(ui::quiet_button).on_press(Message::Copy(copy))].align_y(Alignment::Center),
            card(details.into()),
        ].spacing(12).max_width(672),
    ].spacing(16).align_x(Alignment::Center).width(Fill);
    settings_content(body.into())
}

/// The redesign's thin scrollbar: a 3px faint thumb and no track.
pub(super) fn quiet_scroll<'a>(content: impl Into<Element<'a, Message>>) -> scrollable::Scrollable<'a, Message> {
    scrollable(content).direction(scrollable::Direction::Vertical(scrollable::Scrollbar::new().width(3).scroller_width(3).margin(2)))
        .style(|theme: &iced::Theme, status| { let mut style = scrollable::default(theme, status);
            style.vertical_rail.scroller.background = Color { a: 0.14, ..theme.palette().text }.into(); style.vertical_rail.background = None; style })
}
