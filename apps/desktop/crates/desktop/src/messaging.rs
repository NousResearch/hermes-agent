//! Native manual Telegram setup, adapted from the stock desktop Messaging
//! master/detail route and AllowlistField. Only the implemented adapter is listed.
mod store;
mod telegram;

use crate::appearance::text;
use hermes_library::Library;
use iced::{
    Alignment, Border, Color, Element, Fill, Font, Length, Padding, Task,
    widget::{Space, button, column, container, row, rule, scrollable, svg, text_input},
};
use std::{collections::BTreeMap, fmt, fs::File, sync::Arc, time::Duration};
use store::{Inbox, Settings, Store};
pub(super) use telegram::Incoming;
use telegram::{Client, Error, ErrorKind, Identity, Token};
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
pub(super) struct Loaded {
    settings: Option<Settings>,
    inbox: Option<Arc<Inbox>>,
}
#[derive(Clone)]
pub(super) struct Ready {
    client: Client,
    identity: Identity,
    inbox: Arc<Inbox>,
    lease: Arc<File>,
}
#[derive(Clone)]
pub(super) struct Received {
    inbox: Arc<Inbox>,
    messages: Vec<Incoming>,
    ignored: usize,
}

#[derive(Clone)]
pub(super) enum Message {
    Search(String),
    Reload,
    SelectTelegram,
    Token(String),
    Allowed(usize, String),
    AddAllowed,
    RemoveAllowed(usize),
    Save,
    ClearToken,
    Loaded(u64, Result<Arc<Loaded>, String>),
    Saved(u64, Result<Arc<Settings>, String>),
    Connect,
    Disconnect,
    Connected(u64, Result<Arc<Ready>, String>),
    Polled(u64, Result<Arc<Received>, Error>),
    SelectUser(i64),
    Reply(String),
    Send,
    Sent(u64, Result<Arc<Inbox>, String>),
    OpenLink(String),
    Platform(super::platforms::Action),
}
impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Search(_) => "SearchMessaging",
            Self::Reload => "ReloadMessaging",
            Self::SelectTelegram => "SelectTelegram", Self::Platform(_) => "MessagingPlatform([redacted])",
            Self::Token(_) => "TelegramToken([redacted])",
            Self::Allowed(..) | Self::AddAllowed | Self::RemoveAllowed(_) => "TelegramAllowlist",
            Self::Save => "SaveTelegram",
            Self::ClearToken => "ClearTelegramToken",
            Self::Loaded(..) => "TelegramSettingsLoaded",
            Self::Saved(..) => "TelegramSettingsSaved",
            Self::Connect => "ConnectTelegram",
            Self::Disconnect => "DisconnectTelegram",
            Self::Connected(..) => "TelegramConnected",
            Self::Polled(..) => "TelegramUpdates",
            Self::SelectUser(_) => "SelectTelegramConversation",
            Self::Reply(_) => "TelegramReply",
            Self::Send => "SendTelegramReply",
            Self::Sent(..) => "TelegramReplySent",
            Self::OpenLink(_) => "OpenMessagingGuide",
        })
    }
}
pub(super) enum Effect {
    Incoming(Incoming),
    OpenLink(String),
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Status {
    #[default]
    NeedsSetup,
    Disabled,
    Connecting,
    Connected,
    Retrying,
}
impl Status {
    fn label(self) -> &'static str {
        match self {
            Self::NeedsSetup => "Needs setup",
            Self::Disabled => "Disabled",
            Self::Connecting => "Connecting",
            Self::Connected => "Connected",
            Self::Retrying => "Retrying",
        }
    }
}

pub(super) struct Messaging {
    pub platforms: super::platforms::State,
    store: Option<Store>,
    loaded: bool,
    saved: Option<Settings>,
    inbox: Option<Arc<Inbox>>,
    query: String,
    token: String,
    allowed: Vec<String>,
    dirty: bool,
    saving: bool,
    status: Status,
    client: Option<Client>,
    bot: Option<Identity>,
    lease: Option<Arc<File>>,
    cancel: Option<CancellationToken>,
    generation: u64,
    selected_user: Option<i64>,
    reply: String,
    sending: bool,
    error: Option<String>,
    note: Option<String>,
    ignored: usize,
}
impl Default for Messaging {
    fn default() -> Self {
        Self {
            platforms: super::platforms::State::default(),
            store: None,
            loaded: false,
            saved: None,
            inbox: None,
            query: String::new(),
            token: String::new(),
            allowed: vec![String::new()],
            dirty: false,
            saving: false,
            status: Status::NeedsSetup,
            client: None,
            bot: None,
            lease: None,
            cancel: None,
            generation: 0,
            selected_user: None,
            reply: String::new(),
            sending: false,
            error: None,
            note: None,
            ignored: 0,
        }
    }
}
impl Drop for Messaging {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.cancel();
        }
    }
}

impl Messaging {
    pub(super) fn load(&mut self, library: Option<Library>) -> Task<Message> {
        if self.loaded || self.store.is_some() {
            return Task::none();
        }
        let Some(library) = library else {
            self.error = Some("Chat library is unavailable".into());
            return Task::none();
        };
        self.platforms.load(library.root());
        let store = Store::new(library.root());
        self.store = Some(store.clone());
        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || {
                    let settings = store.load_settings()?;
                    let inbox = settings
                        .as_ref()
                        .and_then(|settings| settings.bot.as_ref())
                        .map(|bot| store.load_inbox(bot))
                        .transpose()?;
                    Ok(Arc::new(Loaded { settings, inbox }))
                })
                .await
                .map_err(|_| "Could not load native Telegram settings".to_owned())
                .and_then(|result| result)
            },
            move |result| Message::Loaded(generation, result),
        )
    }
    pub(super) fn shutdown(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.cancel();
        }
        self.client = None;
        self.lease = None;
        self.generation = self.generation.wrapping_add(1);
        self.sending = false;
        self.status = if self
            .saved
            .as_ref()
            .is_some_and(|settings| settings.token.is_some())
        {
            Status::Disabled
        } else {
            Status::NeedsSetup
        };
    }
    fn apply_settings(&mut self, settings: Settings) {
        self.shutdown();
        if self
            .selected_user
            .is_some_and(|user| !settings.allowed.contains(&user))
        {
            self.selected_user = None;
            self.reply.clear();
        }
        self.allowed = settings.allowed.iter().map(i64::to_string).collect();
        if self.allowed.is_empty() {
            self.allowed.push(String::new());
        }
        self.bot = settings.bot.clone();
        if self
            .inbox
            .as_ref()
            .is_some_and(|inbox| Some(inbox.bot.id) != self.bot.as_ref().map(|bot| bot.id))
        {
            self.inbox = None;
            self.selected_user = None;
        }
        self.saved = Some(settings);
        self.token.clear();
        self.dirty = false;
        self.saving = false;
        self.error = None;
        self.status = if self
            .saved
            .as_ref()
            .is_some_and(|settings| settings.token.is_some())
        {
            Status::Disabled
        } else {
            Status::NeedsSetup
        };
    }
    fn apply_inbox(&mut self, inbox: Arc<Inbox>) {
        if self.inbox.as_ref().is_none_or(|current| {
            current.bot.id != inbox.bot.id || inbox.revision >= current.revision
        }) {
            self.inbox = Some(inbox);
        }
        if self.selected_user.is_none() {
            self.selected_user = self.inbox.as_ref().and_then(|inbox| {
                inbox
                    .messages
                    .iter()
                    .rev()
                    .find(|message| {
                        self.saved
                            .as_ref()
                            .is_some_and(|settings| settings.allowed.contains(&message.user_id))
                    })
                    .map(|message| message.user_id)
            });
        }
    }
    fn poll(&self, delay: Duration) -> Task<Message> {
        let (Some(client), Some(bot), Some(store), Some(cancel)) = (
            self.client.clone(),
            self.bot.clone(),
            self.store.clone(),
            self.cancel.clone(),
        ) else {
            return Task::none();
        };
        let generation = self.generation;
        let cursor = self.inbox.as_ref().map_or(0, |inbox| inbox.cursor);
        Task::perform(
            async move {
                if !delay.is_zero() {
                    tokio::select! {_=cancel.cancelled()=>return Err(Error{kind:ErrorKind::Cancelled,message:"Telegram connection stopped".into(),retry_after:None}),_=tokio::time::sleep(delay)=>{}}
                }
                let batch = client.poll(bot.id, cursor, &cancel).await?;
                let ignored = batch.ignored;
                let (inbox, messages) = tokio::task::spawn_blocking(move || {
                    store.commit_poll(
                        &bot,
                        client.credential(),
                        &batch.messages,
                        batch.next_offset,
                    )
                })
                .await
                .map_err(|_| Error::invalid("Could not save Telegram updates"))?
                .map_err(Error::invalid)?;
                Ok(Arc::new(Received {
                    inbox,
                    messages,
                    ignored,
                }))
            },
            move |result| Message::Polled(generation, result),
        )
    }
    pub(super) fn update(
        &mut self,
        message: Message,
        library: Option<Library>,
    ) -> (Task<Message>, Vec<Effect>) {
        if self.store.is_none() && !matches!(message, Message::Loaded(..)) {
            let task = self.load(library.clone());
            if !matches!(message, Message::Search(_) | Message::SelectTelegram | Message::Platform(_)) {
                return (task, vec![]);
            }
        }
        match message {
            Message::Reload if !self.saving => {
                self.shutdown();
                self.store = None;
                self.loaded = false;
                self.saved = None;
                self.inbox = None;
                self.bot = None;
                self.selected_user = None;
                self.reply.clear();
                self.token.clear();
                self.allowed = vec![String::new()];
                self.dirty = false;
                return (self.load(library), vec![]);
            }
            Message::Search(query) => self.query = query,
            Message::SelectTelegram => self.platforms.selected = None,
            Message::Platform(action) => self.platforms.update(action),
            Message::Token(token) if !self.saving => {
                self.token = token;
                self.dirty = true;
                self.note = None;
            }
            Message::Allowed(index, value) if !self.saving => {
                if index < self.allowed.len() {
                    let values: Vec<_> = value
                        .split([',', '\n'])
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_owned)
                        .collect();
                    if values.len() > 1 {
                        self.allowed.splice(index..=index, values);
                    } else {
                        self.allowed[index] = value;
                    }
                    self.dirty = true;
                    self.note = None;
                }
            }
            Message::AddAllowed if !self.saving => self.allowed.push(String::new()),
            Message::RemoveAllowed(index) if !self.saving => {
                if index < self.allowed.len() {
                    if self.allowed.len() == 1 {
                        self.allowed[0].clear();
                    } else {
                        self.allowed.remove(index);
                    }
                    self.dirty = true;
                }
            }
            Message::Loaded(generation, result) if generation == self.generation => {
                self.loaded = true;
                match result {
                    Ok(loaded) => {
                        self.inbox = loaded.inbox.clone();
                        if let Some(settings) = &loaded.settings {
                            self.apply_settings(settings.clone());
                        }
                        self.apply_inbox_if_present();
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            Message::Save if !self.saving => {
                let token = if self.token.trim().is_empty() {
                    self.saved
                        .as_ref()
                        .and_then(|settings| settings.token.clone())
                        .ok_or_else(|| Error::invalid("Paste a Telegram bot token first"))
                } else {
                    Token::parse(&self.token)
                };
                let allowed = telegram::parse_allowlist(&self.allowed);
                let (token, allowed) = match (token, allowed) {
                    (Ok(token), Ok(allowed)) => (token, allowed),
                    (Err(error), _) | (_, Err(error)) => {
                        self.error = Some(error.to_string());
                        return (Task::none(), vec![]);
                    }
                };
                let Some(store) = self.store.clone() else {
                    return (Task::none(), vec![]);
                };
                let settings = Settings {
                    version: 1,
                    token: Some(token),
                    allowed,
                    bot: if self.token.trim().is_empty() {
                        self.bot.clone()
                    } else {
                        None
                    },
                };
                self.shutdown();
                self.saving = true;
                self.error = None;
                let generation = self.generation;
                return (
                    Task::perform(
                        async move {
                            tokio::task::spawn_blocking(move || {
                                store.save_settings(&settings)?;
                                Ok(Arc::new(settings))
                            })
                            .await
                            .map_err(|_| "Could not save Telegram setup".to_owned())
                            .and_then(|result| result)
                        },
                        move |result| Message::Saved(generation, result),
                    ),
                    vec![],
                );
            }
            Message::ClearToken if !self.saving => {
                self.shutdown();
                let generation = self.generation;
                self.saving = true;
                let Some(store) = self.store.clone() else {
                    return (Task::none(), vec![]);
                };
                return (
                    Task::perform(
                        async move {
                            tokio::task::spawn_blocking(move || {
                                store
                                    .clear_token()?
                                    .map(Arc::new)
                                    .ok_or_else(|| "No saved Telegram credential".to_owned())
                            })
                            .await
                            .map_err(|_| "Could not clear Telegram credential".to_owned())
                            .and_then(|result| result)
                        },
                        move |result| Message::Saved(generation, result),
                    ),
                    vec![],
                );
            }
            Message::Saved(generation, result) if generation == self.generation => {
                self.saving = false;
                match result {
                    Ok(settings) => {
                        self.apply_settings((*settings).clone());
                        self.note =
                            Some("Telegram setup saved. Enable Telegram to connect.".into());
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            Message::Connect
                if !self.saving
                    && !matches!(
                        self.status,
                        Status::Connecting | Status::Connected | Status::Retrying
                    ) =>
            {
                if self.dirty {
                    self.error = Some("Save changes before enabling Telegram.".into());
                    return (Task::none(), vec![]);
                }
                let Some(settings) = self.saved.clone() else {
                    self.error = Some("Save the bot token and allowlist first.".into());
                    return (Task::none(), vec![]);
                };
                let Some(token) = settings.token else {
                    self.error = Some("Paste and save a Telegram bot token first.".into());
                    return (Task::none(), vec![]);
                };
                let client = match Client::new(token, settings.allowed) {
                    Ok(client) => client,
                    Err(error) => {
                        self.error = Some(error.to_string());
                        return (Task::none(), vec![]);
                    }
                };
                let Some(store) = self.store.clone() else {
                    return (Task::none(), vec![]);
                };
                self.shutdown();
                let generation = self.generation;
                let cancel = CancellationToken::new();
                self.cancel = Some(cancel.clone());
                self.status = Status::Connecting;
                self.error = None;
                self.note = None;
                return (
                    Task::perform(
                        async move {
                            let identity = client
                                .verify(&cancel)
                                .await
                                .map_err(|error| error.to_string())?;
                            let bot = identity.clone();
                            let credential = client.credential().clone();
                            let (inbox, lease) = tokio::task::spawn_blocking(move || {
                                let lease = store.claim(bot.id)?;
                                let inbox = store.load_inbox(&bot)?;
                                store.remember_bot(&bot, &credential)?;
                                Ok::<_, String>((inbox, lease))
                            })
                            .await
                            .map_err(|_| "Could not open native Telegram inbox".to_owned())??;
                            Ok(Arc::new(Ready {
                                client,
                                identity,
                                inbox,
                                lease,
                            }))
                        },
                        move |result| Message::Connected(generation, result),
                    ),
                    vec![],
                );
            }
            Message::Disconnect => {
                self.shutdown();
                self.note = Some("Telegram disabled".into());
            }
            Message::Connected(generation, result) if generation == self.generation => match result
            {
                Ok(ready) => {
                    self.client = Some(ready.client.clone());
                    self.bot = Some(ready.identity.clone());
                    self.lease = Some(ready.lease.clone());
                    self.apply_inbox(ready.inbox.clone());
                    self.status = Status::Connected;
                    return (self.poll(Duration::ZERO), vec![]);
                }
                Err(error) => {
                    self.shutdown();
                    self.error = Some(error);
                }
            },
            Message::Polled(generation, result) if generation == self.generation => match result {
                Ok(received) => {
                    self.status = Status::Connected;
                    self.error = None;
                    self.ignored = self.ignored.saturating_add(received.ignored);
                    self.apply_inbox(received.inbox.clone());
                    let effects = received
                        .messages
                        .iter()
                        .cloned()
                        .map(Effect::Incoming)
                        .collect();
                    return (self.poll(Duration::from_millis(200)), effects);
                }
                Err(error) if error.kind == ErrorKind::Cancelled => {}
                Err(error) if error.kind == ErrorKind::Retryable => {
                    self.status = Status::Retrying;
                    self.error = Some(error.message);
                    return (
                        self.poll(Duration::from_secs(
                            error.retry_after.unwrap_or(3).clamp(1, 60),
                        )),
                        vec![],
                    );
                }
                Err(error) => {
                    self.shutdown();
                    self.error = Some(error.message);
                }
            },
            Message::SelectUser(user) => {
                if self
                    .saved
                    .as_ref()
                    .is_some_and(|settings| settings.allowed.contains(&user))
                {
                    self.selected_user = Some(user);
                    self.reply.clear();
                }
            }
            Message::Reply(reply) if !self.sending => self.reply = reply,
            Message::Send if !self.sending && !self.saving && self.status == Status::Connected => {
                let (Some(client), Some(bot), Some(cancel), Some(store), Some(user)) = (
                    self.client.clone(),
                    self.bot.clone(),
                    self.cancel.clone(),
                    self.store.clone(),
                    self.selected_user,
                ) else {
                    return (Task::none(), vec![]);
                };
                let text = self.reply.trim().to_owned();
                if text.is_empty() || text.chars().count() > 4096 {
                    self.error = Some("Enter a reply of 1–4096 characters.".into());
                    return (Task::none(), vec![]);
                }
                self.sending = true;
                self.error = None;
                let generation = self.generation;
                return (
                    Task::perform(
                        async move {
                            let policy_store = store.clone();
                            let policy_bot = bot.clone();
                            let credential = client.credential().clone();
                            tokio::task::spawn_blocking(move || {
                                policy_store.authorize_reply(&policy_bot, &credential, user)
                            })
                            .await
                            .map_err(|_| {
                                "Could not check the saved Telegram allowlist".to_owned()
                            })??;
                            let receipt = client
                                .send(user, &text, &cancel)
                                .await
                                .map_err(|error| error.to_string())?;
                            tokio::task::spawn_blocking(move || {
                                store.record_sent(&bot, user, text, receipt)
                            })
                            .await
                            .map_err(|_| {
                                "Reply delivered, but the native inbox could not be saved"
                                    .to_owned()
                            })?
                        },
                        move |result| Message::Sent(generation, result),
                    ),
                    vec![],
                );
            }
            Message::Sent(generation, result) if generation == self.generation => {
                self.sending = false;
                match result {
                    Ok(inbox) => {
                        self.apply_inbox(inbox);
                        self.reply.clear();
                        self.note = Some("Reply sent".into());
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            Message::OpenLink(url) => return (Task::none(), vec![Effect::OpenLink(url)]),
            _ => {}
        }
        (Task::none(), vec![])
    }
    fn apply_inbox_if_present(&mut self) {
        if let Some(inbox) = self.inbox.clone() {
            self.apply_inbox(inbox);
        }
    }

    pub(super) fn view(&self, dark: bool) -> Element<'_, Message> {
        let p = super::ui::palette(dark);
        let search = row![
            super::ui::icon("search", 14, p.muted),
            text_input("Search messaging...", &self.query)
                .on_input(Message::Search)
                .size(13)
                .padding([5, 0])
                .style(move |theme, status| {
                    let mut style = text_input::default(theme, status);
                    style.background = Color::TRANSPARENT.into();
                    style.border.width = 0.0;
                    style.value = p.foreground;
                    style.placeholder = p.muted;
                    style
                })
                .width(Fill),
            button(super::ui::icon("refresh", 14, p.muted))
                .on_press_maybe((!self.saving).then_some(Message::Reload))
                .padding(5)
                .style(button::text)
        ]
        .spacing(7)
        .align_y(Alignment::Center);
        let header = container(search).padding(Padding {
            top: 8.0,
            right: 12.0,
            bottom: 8.0,
            left: 12.0,
        });
        let matched = self.query.trim().is_empty()
            || format!("telegram bot {}", self.status.label())
                .to_lowercase()
                .contains(&self.query.trim().to_lowercase());
        let status_color = if self.status == Status::Connected {
            p.accent
        } else {
            p.muted
        };
        let mut list = column![].spacing(4);
        if matched {
            list = list.push(
                button(
                    row![
                        telegram_icon(18),
                        text("Telegram").size(13).width(Fill),
                        text("●").size(9).color(status_color)
                    ]
                    .spacing(8)
                    .align_y(Alignment::Center),
                )
                .on_press(Message::SelectTelegram)
                .padding([7, 8])
                .width(Fill)
                .style(move |_, _| button::Style {
                    background: Some(p.card.into()),
                    text_color: p.foreground,
                    border: Border {
                        radius: 6.0.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                }),
            );
        }
        for row in super::platforms::rail_rows(&self.platforms, &self.query, p) { list = list.push(row); }
        let rail = container(scrollable(list))
            .padding(8)
            .width(224)
            .height(Fill);
        let mut detail = column![
            row![
                telegram_icon(24),
                text("Telegram").size(15).font(semibold()),
                container(text(self.status.label()).size(11).color(status_color))
                    .padding([3, 7])
                    .style(move |_| container::Style {
                        background: Some(p.card.into()),
                        border: Border {
                            radius: 6.0.into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    })
            ]
            .spacing(10)
            .align_y(Alignment::Center),
            text("Connect your Telegram bot for private messages from allowed users.")
                .size(12)
                .color(p.muted),
        ]
        .spacing(14);
        if let Some(error) = &self.error {
            detail = detail.push(text(error).size(12).style(|theme: &iced::Theme| {
                iced::widget::text::Style {
                    color: Some(theme.palette().danger),
                }
            }));
        }
        if let Some(note) = &self.note {
            detail = detail.push(text(note).size(12).color(p.muted));
        }
        if let Some(bot) = &self.bot {
            detail = detail.push(
                text(if bot.username.is_empty() {
                    bot.name.clone()
                } else {
                    format!("@{}", bot.username)
                })
                .size(12)
                .color(p.muted),
            );
        }
        detail = detail.push(section("GET YOUR CREDENTIALS", p.muted));
        detail = detail.push(
            text("Create a bot with @BotFather, then paste the token it gives you.")
                .size(12)
                .color(p.muted),
        );
        detail = detail.push(
            button(
                row![
                    text("Open setup guide").size(12),
                    super::ui::icon("link-external", 12, p.muted)
                ]
                .spacing(6),
            )
            .on_press(Message::OpenLink("https://t.me/BotFather".into()))
            .padding([3, 0])
            .style(button::text),
        );
        detail = detail.push(section("REQUIRED", p.muted));
        let has_token = self
            .saved
            .as_ref()
            .is_some_and(|settings| settings.token.is_some());
        let token_field = text_input(
            if has_token {
                "Replace current value"
            } else {
                "Paste Telegram bot token"
            },
            &self.token,
        )
        .secure(true)
        .on_input(Message::Token)
        .padding(9)
        .size(13)
        .width(Fill);
        let mut token_controls = column![token_field].spacing(6);
        if has_token {
            token_controls = token_controls.push(row![
                text("Credentials set").size(11).color(p.muted),
                Space::new().width(Fill),
                button(text("Clear").size(11))
                    .on_press_maybe((!self.saving).then_some(Message::ClearToken))
                    .padding([2, 4])
                    .style(button::text)
            ]);
        }
        detail = detail.push(column![text("Bot token").size(13), token_controls].spacing(7));
        let mut allow=column![text("Allowed Telegram user IDs").size(13),text("Numeric IDs from @userinfobot, one per box. Only these users can message this connection.").size(12).color(p.muted)].spacing(7);
        for (index, value) in self.allowed.iter().enumerate() {
            allow = allow.push(
                row![
                    text_input("Enter an ID", value)
                        .on_input(move |value| Message::Allowed(index, value))
                        .on_submit(Message::AddAllowed)
                        .padding(9)
                        .size(13),
                    button(super::ui::icon("close", 14, p.muted))
                        .on_press_maybe((!self.saving).then_some(Message::RemoveAllowed(index)))
                        .padding(7)
                        .style(button::text)
                ]
                .spacing(8)
                .align_y(Alignment::Center),
            );
        }
        allow = allow.push(row![
            button(
                row![
                    super::ui::icon("add", 12, p.muted),
                    text("Add another").size(12)
                ]
                .spacing(4)
            )
            .on_press_maybe((!self.saving).then_some(Message::AddAllowed))
            .padding([3, 0])
            .style(button::text),
            Space::new().width(Fill),
            button(text("Open docs").size(12))
                .on_press(Message::OpenLink("https://t.me/userinfobot".into()))
                .style(button::text)
        ]);
        detail = detail.push(allow);
        if let Some(inbox) = &self.inbox {
            let allowed = self.saved.as_ref().map(|settings| &settings.allowed);
            let mut users = BTreeMap::new();
            for message in &inbox.messages {
                if allowed.is_some_and(|ids| ids.contains(&message.user_id)) {
                    users.insert(message.user_id, message.sender.clone());
                }
            }
            detail = detail.push(section("INBOX", p.muted));
            if users.is_empty() {
                detail = detail.push(
                    text("Messages from allowed users will appear here.")
                        .size(12)
                        .color(p.muted),
                );
            }
            let mut people = row![].spacing(6);
            for (user, name) in users {
                people = people.push(
                    button(text(format!("{name} · {user}")).size(11))
                        .on_press(Message::SelectUser(user))
                        .padding([5, 8])
                        .style(if self.selected_user == Some(user) {
                            button::secondary
                        } else {
                            button::text
                        }),
                );
            }
            detail = detail.push(people);
            if let Some(user) = self.selected_user {
                let mut conversation = column![].spacing(10);
                for message in inbox
                    .messages
                    .iter()
                    .filter(|message| message.user_id == user)
                    .rev()
                    .take(100)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                {
                    conversation = conversation.push(
                        column![
                            text(if message.outgoing {
                                "You"
                            } else {
                                &message.sender
                            })
                            .size(11)
                            .color(p.muted),
                            text(&message.text).size(13)
                        ]
                        .spacing(3),
                    );
                }
                detail = detail.push(
                    container(scrollable(conversation).height(240))
                        .padding(12)
                        .width(Fill)
                        .style(move |_| container::Style {
                            border: Border {
                                color: p.border,
                                width: 1.0,
                                radius: 6.0.into(),
                            },
                            ..Default::default()
                        }),
                );
                detail = detail.push(
                    row![
                        text_input("Reply…", &self.reply)
                            .on_input(Message::Reply)
                            .on_submit(Message::Send)
                            .padding(9)
                            .size(13),
                        button(
                            text(if self.sending {
                                "Sending…"
                            } else {
                                "Send reply"
                            })
                            .size(12)
                        )
                        .on_press_maybe(
                            (self.status == Status::Connected
                                && !self.sending
                                && !self.reply.trim().is_empty())
                            .then_some(Message::Send)
                        )
                        .padding([8, 12])
                    ]
                    .spacing(8),
                );
            }
        }
        detail=detail.push(text("Replies are sent only when you choose Send reply. Telegram is the available native adapter in this build.").size(11).color(p.muted));
        let working = matches!(
            self.status,
            Status::Connecting | Status::Connected | Status::Retrying
        );
        let actions = row![
            text(if self.dirty {
                "Unsaved changes"
            } else {
                self.status.label()
            })
            .size(12)
            .color(p.muted),
            Space::new().width(Fill),
            button(
                text(if self.saving {
                    "Saving..."
                } else {
                    "Save changes"
                })
                .size(12)
            )
            .on_press_maybe((self.dirty && !self.saving).then_some(Message::Save))
            .padding([7, 12])
            .style(button::secondary),
            button(text(if working { "Disable" } else { "Enable" }).size(12))
                .on_press_maybe((!self.saving).then_some(if working {
                    Message::Disconnect
                } else {
                    Message::Connect
                }))
                .padding([7, 12])
        ]
        .spacing(10)
        .align_y(Alignment::Center);
        if self.platforms.selected.is_some() {
            return column![header, row![rail, rule::vertical(1), super::platforms::detail(&self.platforms, dark)].height(Fill)].height(Fill).into();
        }
        let main = column![
            scrollable(
                container(detail)
                    .padding(Padding {
                        top: 16.0,
                        right: 24.0,
                        bottom: 24.0,
                        left: 24.0
                    })
                    .max_width(720)
                    .width(Fill)
            )
            .height(Fill),
            rule::horizontal(1),
            container(actions).padding([12, 20])
        ]
        .height(Fill)
        .width(Fill);
        column![header, row![rail, rule::vertical(1), main].height(Fill)]
            .height(Fill)
            .into()
    }
}
fn section(label: &str, color: Color) -> Element<'_, Message> {
    container(text(label).size(11).font(semibold()).color(color))
        .padding(Padding {
            top: 9.0,
            right: 0.0,
            bottom: 0.0,
            left: 0.0,
        })
        .into()
}
fn semibold() -> Font {
    Font {
        weight: iced::font::Weight::Semibold,
        ..crate::appearance::font()
    }
}
fn telegram_icon(size: u16) -> Element<'static, Message> {
    // Original Messaging platform icon: Simple Icons Telegram (#26A5E4).
    let source = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='#26A5E4'><path d='M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.59 7.205c.1-.002.321.023.465.14.12.099.153.232.17.326.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.202-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.008-.032.014-.15-.056-.212-.07-.062-.174-.041-.25-.024-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.481-.428-.009-1.252-.242-1.865-.44-.751-.245-1.349-.374-1.297-.79.027-.215.323-.435.887-.66 3.499-1.525 5.832-2.53 7-3.015 3.333-1.386 4.026-1.627 4.478-1.635z'/></svg>";
    svg(super::ui::cached_svg(source.to_owned()))
        .width(Length::Fixed(size as f32))
        .height(Length::Fixed(size as f32))
        .into()
}
