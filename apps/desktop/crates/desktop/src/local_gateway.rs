//! Settings → Gateways, local in-process Rust listener only.
//! Stock reference: 0284ce17 settings/gateway-settings and connections-registry.
//! No listener is started until the user presses Start. No remote/SSH runtime
//! or dashboard hosting is claimed by this local control.
use super::{Hermes, Message, ui};
use crate::appearance::text;
use hermes_engine::Config;
use hermes_gateway::{
    Gateway, GatewayConfig, RunningGateway,
    client::{ClientOptions, GatewayClient},
};
use iced::{
    Alignment, Element, Fill, Font, Task,
    widget::{Space, button, column, container, row},
};
use serde_json::{Value, json};
use std::{
    fmt,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::sync::Mutex;

#[derive(Clone)]
pub enum Action {
    Start,
    Stop,
    Restart,
    Discover,
    CopyAddress,
    CopyToken,
    Started(u64, Result<Info, String>),
    Stopped(u64, bool, Result<(), String>),
    Discovered(u64, Result<Value, String>),
    Token(Result<String, String>),
}
impl fmt::Debug for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Start => "Start",
            Self::Stop => "Stop",
            Self::Restart => "Restart",
            Self::Discover => "Discover",
            Self::CopyAddress => "CopyAddress",
            Self::CopyToken => "CopyToken",
            Self::Started(..) => "Started",
            Self::Stopped(..) => "Stopped",
            Self::Discovered(..) => "Discovered",
            Self::Token(_) => "Token([redacted])",
        })
    }
}
#[derive(Clone, Debug)]
pub struct Info {
    pub http_url: String,
    pub websocket_url: String,
    pub profile_root: PathBuf,
    pub model: String,
}
pub struct State {
    handle: Arc<Mutex<Option<RunningGateway>>>,
    generation: Arc<AtomicU64>,
    info: Option<Info>,
    details: Option<Value>,
    error: Option<String>,
    busy: bool,
    stopping: bool,
}
impl State {
    /// The running gateway's HTTP base URL, if it is up.
    pub fn http_url(&self) -> Option<&str> { self.info.as_ref().map(|i| i.http_url.as_str()) }
}
impl Default for State {
    fn default() -> Self {
        Self {
            handle: Arc::new(Mutex::new(None)),
            generation: Arc::new(AtomicU64::new(0)),
            info: None,
            details: None,
            error: None,
            busy: false,
            stopping: false,
        }
    }
}
impl State {
    pub fn has_service(&self) -> bool {
        self.info.is_some() || self.busy
    }
    pub fn update(
        &mut self,
        action: Action,
        profile_root: Option<PathBuf>,
        provider: Config,
    ) -> Task<Action> {
        match action {
            Action::Start if !self.busy && self.info.is_none() => {
                let Some(profile_root) = profile_root else {
                    self.error = Some("Open a local profile before starting its gateway.".into());
                    return Task::none();
                };
                let request = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
                let generation = self.generation.clone();
                let handle = self.handle.clone();
                self.busy = true;
                self.stopping = false;
                self.error = None;
                let model = provider.model.clone();
                return Task::perform(
                    async move {
                        let service = Gateway::start(GatewayConfig::new(&profile_root, provider))
                            .await
                            .map_err(|error| format!("{error:#}"))?;
                        if generation.load(Ordering::SeqCst) != request {
                            let _ = service.shutdown().await;
                            return Err("Gateway start was cancelled.".into());
                        }
                        let info = Info {
                            http_url: service.http_url(),
                            websocket_url: service.websocket_url(),
                            profile_root,
                            model,
                        };
                        *handle.lock().await = Some(service);
                        Ok(info)
                    },
                    move |result| Action::Started(request, result),
                );
            }
            Action::Stop | Action::Restart => {
                let restart = matches!(action, Action::Restart);
                let request = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
                let handle = self.handle.clone();
                self.busy = true;
                self.stopping = true;
                self.error = None;
                return Task::perform(
                    async move {
                        let service = handle.lock().await.take();
                        if let Some(service) = service {
                            service
                                .shutdown()
                                .await
                                .map_err(|error| format!("{error:#}"))
                        } else {
                            Ok(())
                        }
                    },
                    move |result| Action::Stopped(request, restart, result),
                );
            }
            Action::Started(request, result)
                if request == self.generation.load(Ordering::SeqCst) =>
            {
                self.busy = false;
                self.stopping = false;
                match result {
                    Ok(info) => {
                        self.info = Some(info);
                        self.details = None;
                    }
                    Err(error) => self.error = Some(error),
                }
            }
            Action::Stopped(request, restart, result)
                if request == self.generation.load(Ordering::SeqCst) =>
            {
                self.busy = false;
                self.stopping = false;
                self.info = None;
                self.details = None;
                match result {
                    Ok(()) if restart => return Task::done(Action::Start),
                    Ok(()) => {}
                    Err(error) => self.error = Some(error),
                }
            }
            Action::Discover if !self.busy && self.info.is_some() => {
                let request = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
                let handle = self.handle.clone();
                self.busy = true;
                self.error = None;
                return Task::perform(
                    async move {
                        let (origin, token) = {
                            let locked = handle.lock().await;
                            let service = locked
                                .as_ref()
                                .ok_or_else(|| "The local gateway is stopped.".to_owned())?;
                            (service.http_url(), service.token().to_owned())
                        };
                        let client = GatewayClient::new(ClientOptions {
                            connect_timeout: Duration::from_secs(5),
                            request_timeout: Duration::from_secs(10),
                            ..ClientOptions::default()
                        });
                        let result = async {
                            client
                                .connect_native(&origin, &token)
                                .await
                                .map_err(|error| error.to_string())?;
                            let methods = client
                                .request("rpc.discover", json!({}))
                                .await
                                .map_err(|error| error.to_string())?;
                            let runtime = client
                                .request("runtime.describe", json!({}))
                                .await
                                .map_err(|error| error.to_string())?;
                            Ok(json!({"runtime":runtime,"discovery":methods}))
                        }
                        .await;
                        client.close();
                        result
                    },
                    move |result| Action::Discovered(request, result),
                );
            }
            Action::Discovered(request, result)
                if request == self.generation.load(Ordering::SeqCst) =>
            {
                self.busy = false;
                match result {
                    Ok(details) => self.details = Some(details),
                    Err(error) => self.error = Some(error),
                }
            }
            Action::CopyAddress => {
                if let Some(info) = &self.info {
                    return iced::clipboard::write(info.websocket_url.clone());
                }
            }
            Action::CopyToken if self.info.is_some() => {
                let handle = self.handle.clone();
                return Task::perform(
                    async move {
                        handle
                            .lock()
                            .await
                            .as_ref()
                            .map(|service| service.token().to_owned())
                            .ok_or_else(|| "The local gateway is stopped.".into())
                    },
                    Action::Token,
                );
            }
            Action::Token(result) => match result {
                Ok(token) => return iced::clipboard::write(token),
                Err(error) => self.error = Some(error),
            },
            _ => {}
        }
        Task::none()
    }
}

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let gateway = &app.local_gateway;
    let state = if gateway.busy {
        if gateway.stopping {
            "Stopping"
        } else if gateway.info.is_some() {
            "Checking"
        } else {
            "Starting"
        }
    } else if gateway.info.is_some() {
        "Running"
    } else {
        "Stopped"
    };
    let mut content = column![
        row![
            ui::icon("home", 16, p.muted),
            column![
                text("Local").size(14),
                text("This computer").size(11).color(p.muted)
            ]
            .spacing(3)
            .width(Fill),
            text(state).size(12).color(if gateway.info.is_some() {
                p.accent
            } else {
                p.muted
            })
        ]
        .spacing(10)
        .align_y(Alignment::Center),
    ]
    .spacing(10)
    .max_width(720);
    let mut controls = row![].spacing(7);
    if gateway.info.is_some() {
        controls = controls.push(
            button(text("Stop").size(12))
                .on_press_maybe((!gateway.busy).then_some(Message::LocalGateway(Action::Stop)))
                .padding([6, 12])
                .style(button::secondary),
        );
        controls = controls.push(
            button(text("Restart gateway").size(12))
                .on_press_maybe((!gateway.busy).then_some(Message::LocalGateway(Action::Restart)))
                .padding([6, 12])
                .style(button::secondary),
        );
        controls = controls.push(
            button(text("Check connection").size(12))
                .on_press_maybe((!gateway.busy).then_some(Message::LocalGateway(Action::Discover)))
                .padding([6, 12])
                .style(button::secondary),
        );
    } else {
        controls = controls.push(
            button(text(if gateway.busy { "Starting..." } else { "Start" }).size(12))
                .on_press_maybe((!gateway.busy).then_some(Message::LocalGateway(Action::Start)))
                .padding([6, 12]),
        );
    }
    content = content.push(controls);
    if let Some(error) = &gateway.error {
        content = content.push(
            text(error)
                .size(12)
                .color(iced::Color::from_rgb8(207, 34, 46)),
        );
    }
    if let Some(info) = &gateway.info {
        content=content.push(Space::new().height(12)).push(text("Connection details").size(13))
            .push(detail("Address",&info.http_url,app.dark)).push(detail("WebSocket",&info.websocket_url,app.dark))
            .push(detail("Profile",&info.profile_root.to_string_lossy(),app.dark)).push(detail("Model",&info.model,app.dark))
            .push(detail("Access","Only this computer",app.dark))
            .push(row![button(text("Copy WebSocket address").size(12)).on_press(Message::LocalGateway(Action::CopyAddress)).style(button::text),button(text("Copy authentication token").size(12)).on_press(Message::LocalGateway(Action::CopyToken)).style(button::text)].spacing(12))
            .push(text("The authentication token is generated for this running gateway. Restarting creates a new token.").size(11).color(p.muted));
    } else if let Some(library) = &app.library {
        content = content.push(detail(
            "Profile",
            &library.root().to_string_lossy(),
            app.dark,
        ));
    }
    if let Some(details) = &gateway.details {
        let formatted =
            serde_json::to_string_pretty(details).unwrap_or_else(|_| details.to_string());
        content = content
            .push(Space::new().height(10))
            .push(text("Local gateway details").size(13))
            .push(text(formatted).size(11).font(Font::MONOSPACE));
    }
    // Embedded in Settings › Gateways (connections.rs), which owns the page scroll.
    container(content).width(Fill).into()
}
fn detail<'a>(label: &'a str, value: &str, dark: bool) -> Element<'a, Message> {
    let p = ui::palette(dark);
    row![
        text(label).size(12).color(p.muted).width(100),
        text(value.to_owned()).size(12)
    ]
    .spacing(12)
    .into()
}
