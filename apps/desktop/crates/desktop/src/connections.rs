//! Settings › Gateways (settings/gateway-settings.tsx): connection mode
//! (Local · Remote · Hermes Cloud · SSH), the remote URL and session token,
//! SSH host details with ~/.ssh/config aliases, saved connections, tests and
//! diagnostics. Tests are real: the remote test calls the gateway's
//! `/api/status`, the SSH test runs `ssh` and asks the remote `hermes --version`.
use super::{Hermes, Message, settings, ui};
use crate::appearance::text;
use iced::{Alignment, Color, Element, Fill, Length, Task, widget::{Space, button, column, container, pick_list, row, text_input}};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Mode { #[default] Local, Remote, Cloud, Ssh }

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct Ssh { pub host: String, pub user: String, pub port: String, pub key: String, pub hermes_path: String }

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct Saved { pub name: String, pub mode: Mode, pub url: String, pub ssh: Ssh }

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct Config { pub mode: Mode, pub url: String, pub ssh: Ssh, pub saved: Vec<Saved> }

#[derive(Clone, Debug)]
pub(super) enum Action {
    Mode(Mode), Url(String), Token(String), Host(String), User(String), Port(String), Key(String), HermesPath(String),
    Save, TestRemote, TestSsh, ConnectSsh, SshConnected(u64, Result<super::ssh_tunnel::Tunnel, String>), DisconnectSsh, Tested(u64, Result<String, String>), SaveAs(String), SaveNamed, UseSaved(usize), RemoveSaved(usize), OpenLogs,
    GatewaySignIn(String), GatewaySignedIn(String, Result<(), String>), GatewaySignOut,
    CloudDiscover(Option<String>), CloudLoaded(u64, Result<hermes_engine::portal::CloudDiscovery, String>), CloudConnect(String),
}

#[derive(Default)]
pub(super) struct State {
    pub config: Config, token: String, has_token: bool, name: String, dirty: bool,
    testing: bool, revision: u64, result: Option<Result<String, String>>, aliases: Vec<String>,
    /// The live SSH tunnel; chats run through it while it is up.
    pub tunnel: Option<super::ssh_tunnel::Tunnel>, connecting: bool,
    signing_in: bool,
    cloud: Option<Result<hermes_engine::portal::CloudDiscovery, String>>, cloud_loading: bool, cloud_org: Option<String>,
}

fn path(root: &Path) -> PathBuf { root.join("connection.json") }
fn token_path(root: &Path) -> PathBuf { root.join("connection-token") }

impl State {
    pub fn load(root: &Path) -> Self {
        let config = std::fs::read(path(root)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        Self { config, has_token: token_path(root).is_file(), aliases: ssh_aliases(), ..Default::default() }
    }
    fn save(&mut self, root: &Path) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(&self.config).map_err(|e| e.to_string())?;
        std::fs::write(path(root), bytes).map_err(|e| e.to_string())?;
        if !self.token.trim().is_empty() {
            let file = token_path(root);
            std::fs::write(&file, self.token.trim()).map_err(|e| e.to_string())?;
            #[cfg(unix)] { use std::os::unix::fs::PermissionsExt; let _ = std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)); }
            self.token.clear(); self.has_token = true;
        }
        self.dirty = false;
        Ok(())
    }
}

/// `Host` aliases from ~/.ssh/config, skipping wildcard patterns.
fn ssh_aliases() -> Vec<String> {
    let Some(home) = std::env::var_os("HOME") else { return vec![] };
    let Ok(text) = std::fs::read_to_string(Path::new(&home).join(".ssh/config")) else { return vec![] };
    let mut out: Vec<String> = text.lines().filter_map(|line| {
        let line = line.trim();
        let rest = line.strip_prefix("Host ").or_else(|| line.strip_prefix("Host\t"))?;
        Some(rest.split_whitespace().filter(|h| !h.contains(['*', '?', '!'])).map(str::to_owned).collect::<Vec<_>>())
    }).flatten().collect();
    out.sort(); out.dedup(); out
}

/// GET {url}/api/status with the saved session token (fetchConnectionStatus).
async fn test_remote(url: String, token: String) -> Result<String, String> {
    let base = url.trim().trim_end_matches('/').to_owned();
    if base.is_empty() { return Err("Enter a remote URL first.".into()); }
    let parsed = reqwest::Url::parse(&base).map_err(|_| "That URL is not valid.".to_owned())?;
    if !matches!(parsed.scheme(), "http" | "https") { return Err("Use an http:// or https:// URL.".into()); }
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(8)).build().map_err(|e| e.to_string())?;
    let mut request = client.get(format!("{base}/api/status"));
    if !token.trim().is_empty() { request = request.bearer_auth(token.trim()); }
    let response = request.send().await.map_err(|e| if e.is_timeout() { "The gateway did not answer within 8 seconds.".to_owned() } else { "Could not reach that gateway.".to_owned() })?;
    match response.status().as_u16() {
        401 | 403 => return Err("The gateway refused this token. Sign in again or paste a new session token.".into()),
        s if s >= 400 => return Err(format!("The gateway answered HTTP {s}.")),
        _ => {}
    }
    let body: serde_json::Value = response.json().await.unwrap_or_default();
    let version = body["version"].as_str().or_else(|| body["hermes_version"].as_str()).unwrap_or_default();
    Ok(format!("Connected to {base}{}", if version.is_empty() { String::new() } else { format!(" · Hermes {version}") }))
}

/// ssh into the host and look for Hermes, like the desktop's SSH probe.
fn test_ssh(ssh: Ssh) -> Result<String, String> {
    let host = ssh.host.trim();
    if host.is_empty() { return Err("Enter an SSH host before connecting.".into()); }
    if host.starts_with('-') || ssh.user.starts_with('-') { return Err("That host is not valid.".into()); }
    let target = if ssh.user.trim().is_empty() { host.to_owned() } else { format!("{}@{host}", ssh.user.trim()) };
    let hermes = if ssh.hermes_path.trim().is_empty() { "$(command -v hermes || echo ~/.local/bin/hermes)".to_owned() } else { ssh.hermes_path.trim().replace('\'', "") };
    let mut command = std::process::Command::new("ssh");
    command.args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new"]);
    if !ssh.port.trim().is_empty() { command.args(["-p", ssh.port.trim()]); }
    if !ssh.key.trim().is_empty() { command.args(["-i", ssh.key.trim()]); }
    command.arg("--").arg(&target).arg(format!("uname -sm; {hermes} --version 2>&1 | head -1"));
    let output = command.output().map_err(|_| "ssh is not available on this computer.".to_owned())?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() && stdout.trim().is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
        return Err(if stderr.contains("timed out") { "SSH connection timed out. The host may be unreachable or asleep.".into() }
            else if stderr.contains("permission denied") { "The host refused this key. Check the user and identity file.".into() }
            else { "Could not reach that host over SSH. Check the host, port, and your network.".into() });
    }
    let mut lines = stdout.lines();
    let platform = lines.next().unwrap_or("unknown").trim().to_owned();
    let version = lines.next().unwrap_or_default().trim();
    if version.is_empty() || version.contains("not found") || version.contains("No such file") {
        return Err(format!("Reachable: {host} ({platform}), but Hermes was not found. Set the Hermes path."));
    }
    Ok(format!("Reachable: {host} ({platform}) — Hermes found · {version}"))
}

pub(super) fn update(app: &mut Hermes, action: Action) -> Task<Message> {
    let Some(root) = app.library.as_ref().map(|l| l.root().to_owned()) else { return Task::none() };
    let state = &mut app.connections;
    let mut changed = true;
    match action {
        Action::Mode(mode) => {
            state.config.mode = mode; state.result = None;
            if mode == Mode::Cloud && app.portal_connected && state.cloud.is_none() && !state.cloud_loading { let org = state.cloud_org.clone(); return update(app, Action::CloudDiscover(org)); }
        }
        Action::CloudDiscover(org) => {
            state.revision = state.revision.wrapping_add(1); state.cloud_loading = true; state.cloud_org = org.clone();
            let revision = state.revision;
            return Task::perform(async move { hermes_engine::portal::cloud_agents(org.as_deref()).await.map_err(|e| format!("{e:#}")) },
                move |result| Message::Connection(Action::CloudLoaded(revision, result)));
        }
        Action::CloudLoaded(revision, result) if revision == state.revision => {
            state.cloud_loading = false;
            if let Ok(hermes_engine::portal::CloudDiscovery::Agents { org: Some(org), .. }) = &result { state.cloud_org = Some(org.clone()); }
            state.cloud = Some(result);
        }
        Action::CloudLoaded(..) => {}
        Action::CloudConnect(url) | Action::GatewaySignIn(url) => {
            // An agent's gateway is a remote gateway: point Remote at it and sign in
            // with its native OAuth flow (one browser approval, then it is silent).
            state.config.mode = Mode::Remote; state.config.url = url.clone(); state.signing_in = true;
            state.result = Some(Ok("Approve the sign-in in your browser…".into()));
            let (root, target) = (root.clone(), url.clone());
            return Task::perform(async move {
                if !super::gateway_oauth::supports_native(&target).await? { return Err("NO_NATIVE".to_owned()); }
                super::gateway_oauth::sign_in(root, target).await
            }, move |r| Message::Connection(Action::GatewaySignedIn(url.clone(), r)));
        }
        Action::GatewaySignedIn(url, result) => {
            state.signing_in = false; changed = false;
            state.result = Some(match result {
                Ok(()) => { let _ = std::fs::remove_file(root.join("connection-token")); state.has_token = false;
                    state.save(&root).map(|_| format!("Signed in. Chats now run on {url}.")) }
                Err(e) if e == "NO_NATIVE" => { state.dirty = true; let _ = super::open_url(&url);
                    Err("This gateway has no browser sign-in. Paste the session token from its dashboard, then Save.".into()) }
                Err(e) => Err(e),
            });
        }
        Action::GatewaySignOut => {
            changed = false;
            state.result = Some(super::gateway_oauth::sign_out(&root, &state.config.url).map(|_| "Signed out of this gateway.".into()));
        }
        Action::Url(v) => state.config.url = v,
        Action::Token(v) => state.token = v,
        Action::Host(v) => state.config.ssh.host = if v == "Custom (enter manually)…" { String::new() } else { v },
        Action::User(v) => state.config.ssh.user = v,
        Action::Port(v) => state.config.ssh.port = v.chars().filter(char::is_ascii_digit).take(5).collect(),
        Action::Key(v) => state.config.ssh.key = v,
        Action::HermesPath(v) => state.config.ssh.hermes_path = v,
        Action::SaveAs(v) => { state.name = v; changed = false; }
        Action::Save => { changed = false; state.result = Some(state.save(&root).map(|_| "Saved for the next restart.".into())); }
        Action::SaveNamed => {
            changed = false;
            let name = state.name.trim().to_owned();
            if name.is_empty() { state.result = Some(Err("Name this connection first.".into())); return Task::none(); }
            let entry = Saved { name: name.clone(), mode: state.config.mode, url: state.config.url.clone(), ssh: state.config.ssh.clone() };
            state.config.saved.retain(|s| s.name != name); state.config.saved.push(entry); state.name.clear();
            state.result = Some(state.save(&root).map(|_| format!("Saved “{name}”.")));
        }
        Action::UseSaved(i) => if let Some(saved) = state.config.saved.get(i).cloned() {
            state.config.mode = saved.mode; state.config.url = saved.url; state.config.ssh = saved.ssh;
            state.result = Some(state.save(&root).map(|_| format!("Using “{}”.", saved.name)));
            changed = false;
        },
        Action::RemoveSaved(i) => { if i < state.config.saved.len() { state.config.saved.remove(i); } changed = false; state.result = Some(state.save(&root).map(|_| "Removed.".into())); }
        Action::TestRemote => {
            let token = if state.token.trim().is_empty() { std::fs::read_to_string(token_path(&root)).unwrap_or_default() } else { state.token.clone() };
            state.revision += 1; let rev = state.revision; state.testing = true; state.result = None;
            let url = state.config.url.clone();
            return Task::perform(test_remote(url, token), move |r| Message::Connection(Action::Tested(rev, r)));
        }
        Action::TestSsh => {
            state.revision += 1; let rev = state.revision; state.testing = true; state.result = None;
            let ssh = state.config.ssh.clone();
            return Task::perform(async move { tokio::task::spawn_blocking(move || test_ssh(ssh)).await.map_err(|e| e.to_string()).and_then(|r| r) },
                move |r| Message::Connection(Action::Tested(rev, r)));
        }
        Action::ConnectSsh => {
            if state.dirty && let Err(e) = state.save(&root) { state.result = Some(Err(e)); return Task::none(); }
            state.revision += 1; let rev = state.revision; state.connecting = true; state.result = Some(Ok("Starting Hermes on the remote machine…".into()));
            let ssh = state.config.ssh.clone(); let root = root.clone();
            return Task::perform(async move { tokio::task::spawn_blocking(move || super::ssh_tunnel::connect(&root, ssh)).await.map_err(|e| e.to_string()).and_then(|r| r) },
                move |r| Message::Connection(Action::SshConnected(rev, r)));
        }
        Action::SshConnected(rev, result) => {
            changed = false;
            if rev == state.revision {
                state.connecting = false;
                match result {
                    Ok(tunnel) => { state.result = Some(Ok(format!("Connected to {}. New messages run there.", tunnel.host))); state.tunnel = Some(tunnel); }
                    Err(e) => state.result = Some(Err(e)),
                }
            } else if let Ok(tunnel) = result { tunnel.close(); }
        }
        Action::DisconnectSsh => {
            changed = false;
            if let Some(tunnel) = state.tunnel.take() { std::thread::spawn(move || tunnel.close()); }
            state.result = Some(Ok("Disconnected. Chats run on this computer again.".into()));
        }
        Action::Tested(rev, result) => { changed = false; if rev == state.revision { state.testing = false; state.result = Some(result); } }
        Action::OpenLogs => {
            changed = false;
            let opener = if cfg!(target_os = "macos") { "open" } else if cfg!(windows) { "explorer" } else { "xdg-open" };
            if let Err(e) = std::process::Command::new(opener).arg(&root).spawn() { state.result = Some(Err(format!("Could not open the folder: {e}"))); }
        }
    }
    if changed { state.dirty = true; }
    Task::none()
}

fn segment<'a>(label: &'static str, active: bool, message: Message, p: ui::Palette) -> Element<'a, Message> {
    button(container(text(label).size(13)).center_x(Fill)).width(Fill).padding([7, 10]).on_press(message)
        .style(move |_, status| button::Style {
            background: Some(if active { Color { a: 0.10, ..p.foreground } } else if status == button::Status::Hovered { Color { a: 0.04, ..p.foreground } } else { Color::TRANSPARENT }.into()),
            text_color: if active { p.foreground } else { p.muted }, border: iced::Border { radius: 3.0.into(), ..Default::default() }, ..Default::default() }).into()
}

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark); let dark = app.dark; let state = &app.connections; let mode = state.config.mode;
    let input = |placeholder: &'static str, value: &str, on: fn(String) -> Action| text_input(placeholder, value).size(13).padding([7, 10]).width(280)
        .style(settings::field_style(dark)).on_input(move |v| Message::Connection(on(v)));
    let modes = container(row![
        segment("Local", mode == Mode::Local, Message::Connection(Action::Mode(Mode::Local)), p),
        segment("Remote", mode == Mode::Remote, Message::Connection(Action::Mode(Mode::Remote)), p),
        segment("Hermes Cloud", mode == Mode::Cloud, Message::Connection(Action::Mode(Mode::Cloud)), p),
        segment("SSH", mode == Mode::Ssh, Message::Connection(Action::Mode(Mode::Ssh)), p),
    ].spacing(2)).padding(2).style(move |_| ui::panel(Color { a: 0.03, ..p.foreground }, 4.0, Some(Color { a: 0.06, ..p.foreground })));
    let description = match mode {
        Mode::Local => "Start a private Hermes backend on localhost. This is the default and works offline.",
        Mode::Remote => "Connect this desktop shell to a remote Hermes backend.",
        Mode::Cloud => "Sign in once to Hermes Cloud and pick from the agents on your account — no URL to paste.",
        Mode::Ssh => "Reach a Hermes install on another machine over SSH.",
    };
    let mut body = column![settings::section_title("Connection mode", dark), modes, text(description).size(13).color(p.muted)].spacing(10);
    match mode {
        Mode::Local => { body = body.push(Space::new().height(8)).push(super::local_gateway::view(app)); }
        Mode::Remote => {
            let token_hint = if state.has_token { "saved · paste to replace" } else { "Paste session token" };
            body = body.push(Space::new().height(8)).push(settings::card(vec![
                settings::card_row("Remote URL", "Base URL for the remote dashboard backend. Path prefixes are supported, for example /hermes.",
                    input("https://hermes.example.com", &state.config.url, Action::Url).into(), dark),
                settings::card_row("Session token", "The dashboard session token used for REST and WebSocket access. Leave blank to keep the saved token.",
                    text_input(token_hint, &state.token).secure(true).size(13).padding([7, 10]).width(280).style(settings::field_style(dark))
                        .on_input(|v| Message::Connection(Action::Token(v))).into(), dark),
            ], dark));
            let url = state.config.url.trim().to_owned();
            let signed = app.library.as_ref().is_some_and(|l| super::gateway_oauth::signed_in(l.root(), &url));
            let mut oauth = row![text(if signed { "Signed in with the gateway's browser sign-in." } else { "Hosted gateways sign in with your browser; self-hosted ones may use a session token." }).size(12).color(p.muted).width(Fill)].spacing(8).align_y(Alignment::Center);
            if signed { oauth = oauth.push(button(text("Sign out").size(12)).padding([5, 10]).style(ui::quiet_button).on_press(Message::Connection(Action::GatewaySignOut))); }
            oauth = oauth.push(button(text(if state.signing_in { "Waiting for browser…" } else if signed { "Sign in again" } else { "Sign in with browser" }).size(12)).padding([5, 12])
                .on_press_maybe((!state.signing_in && !url.is_empty()).then(|| Message::Connection(Action::GatewaySignIn(url.clone())))));
            body = body.push(Space::new().height(6)).push(oauth);
        }
        Mode::Cloud => {
            let signed_in = app.portal_connected;
            body = body.push(Space::new().height(8)).push(settings::card(vec![settings::card_row(
                if signed_in { "Signed in to Hermes Cloud" } else { "Hermes Cloud" },
                if signed_in { "Your Nous account is signed in. Cloud agents on this account appear here once Hermes Cloud is enabled for it." }
                    else { "Sign in to Hermes Cloud to discover the agents on your account." },
                button(text(if signed_in { "Account" } else { "Sign in to Hermes Cloud" }).size(13)).padding([7, 12])
                    .on_press(Message::SettingsView(settings::View::Accounts)).into(), dark)], dark));
            if signed_in { body = body.push(Space::new().height(12)).push(cloud_agents(state, dark)); }
        }
        Mode::Ssh => {
            let mut hosts: Vec<String> = state.aliases.clone(); hosts.push("Custom (enter manually)…".into());
            let selected = state.aliases.iter().find(|a| **a == state.config.ssh.host).cloned();
            let host_control: Element<'_, Message> = if state.aliases.is_empty() || (selected.is_none() && !state.config.ssh.host.is_empty()) {
                input("user@host or alias", &state.config.ssh.host, Action::Host).into()
            } else {
                pick_list(hosts, selected, |h| Message::Connection(Action::Host(h))).placeholder("Select a host…").text_size(13).padding([7, 10]).width(280).into()
            };
            body = body.push(Space::new().height(8)).push(settings::card(vec![
                settings::card_row("Host", "A Host alias from ~/.ssh/config, or Custom to type one.", host_control, dark),
                settings::card_row("User", "Blank = ~/.ssh/config or your current user.", input("from ~/.ssh/config", &state.config.ssh.user, Action::User).into(), dark),
                settings::card_row("Port", "Blank = 22 or the ~/.ssh/config port.", input("22", &state.config.ssh.port, Action::Port).into(), dark),
                settings::card_row("Identity file", "Private key path. Blank = ssh-agent or ~/.ssh/config.", input("~/.ssh/id_ed25519", &state.config.ssh.key, Action::Key).into(), dark),
                settings::card_row("Hermes path (optional)", "Full path to the remote hermes binary. Blank = auto-detect.", input("auto-detect", &state.config.ssh.hermes_path, Action::HermesPath).into(), dark),
            ], dark)).push(text("The first presented host key is trusted and pinned; later changes fail closed.").size(12).color(p.muted));
        }
    }
    if let Some(result) = &state.result {
        body = body.push(text(match result { Ok(s) | Err(s) => s.as_str() }).size(13).color(if result.is_ok() { p.accent } else { Color::from_rgb8(232, 98, 98) }));
    }
    if mode != Mode::Cloud && mode != Mode::Local {
        let test: Option<(&str, Action)> = match mode { Mode::Remote => Some(("Test remote", Action::TestRemote)), Mode::Ssh => Some(("Test SSH", Action::TestSsh)), _ => None };
        let mut actions = row![].spacing(12).align_y(Alignment::Center);
        if let Some((label, action)) = test {
            actions = actions.push(button(text(if state.testing { "Testing…" } else { label }).size(13)).padding([7, 12]).style(ui::quiet_button)
                .on_press_maybe((!state.testing).then_some(Message::Connection(action))));
        }
        if mode == Mode::Ssh {
            actions = actions.push(if state.tunnel.is_some() {
                button(text("Disconnect").size(13)).padding([7, 14]).style(ui::quiet_button).on_press(Message::Connection(Action::DisconnectSsh))
            } else {
                button(text(if state.connecting { "Connecting…" } else { "Connect" }).size(13)).padding([7, 14])
                    .on_press_maybe((!state.connecting && !state.config.ssh.host.trim().is_empty()).then_some(Message::Connection(Action::ConnectSsh)))
            });
        }
        actions = actions.push(Space::new().width(Fill)).push(button(text("Save for next restart").size(13)).padding([7, 14])
            .on_press_maybe(state.dirty.then_some(Message::Connection(Action::Save))));
        body = body.push(actions);
        body = body.push(text(match mode {
            Mode::Remote => "Once saved with a URL and session token, new messages run on this gateway and their replies are kept in your local chat.",
            _ => "Connect starts Hermes on that machine over SSH and tunnels to it; while connected, new messages run there and their replies are kept in your local chat.",
        }).size(12).color(p.muted));
    }
    // Saved connections (connections-registry.tsx).
    let mut saved_rows: Vec<Element<'_, Message>> = state.config.saved.iter().enumerate().map(|(i, s)| {
        let detail = match s.mode { Mode::Remote => s.url.clone(), Mode::Ssh => s.ssh.host.clone(), Mode::Cloud => "Hermes Cloud".into(), Mode::Local => "This computer".into() };
        settings::card_row_owned(s.name.clone(), detail,
            row![button(text("Use").size(12)).padding([5, 10]).on_press(Message::Connection(Action::UseSaved(i))),
                button(text("Remove").size(12)).padding([5, 10]).style(ui::quiet_button).on_press(Message::Connection(Action::RemoveSaved(i)))].spacing(6).into(), dark)
    }).collect();
    saved_rows.push(container(row![
        text_input("Name this connection", &state.name).size(13).padding([7, 10]).width(Length::Fill).style(settings::field_style(dark)).on_input(|v| Message::Connection(Action::SaveAs(v))),
        button(text("Save connection").size(13)).padding([7, 12]).on_press(Message::Connection(Action::SaveNamed)),
    ].spacing(8).align_y(Alignment::Center)).padding([12, 16]).into());
    body = body.push(Space::new().height(16)).push(settings::section_title("Saved connections", dark)).push(settings::card(saved_rows, dark));
    body = body.push(Space::new().height(16)).push(settings::section_title("Diagnostics", dark)).push(settings::card(vec![settings::card_row(
        "Open logs", "Reveal the desktop data folder in your file manager — useful when the gateway fails to start.",
        button(text("Open logs").size(13)).padding([7, 12]).on_press(Message::Connection(Action::OpenLogs)).into(), dark)], dark));
    settings::settings_content(body.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn config_round_trips_and_token_is_private() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = State::load(dir.path());
        assert_eq!(state.config.mode, Mode::Local);
        state.config.mode = Mode::Ssh; state.config.ssh.host = "box".into(); state.token = "secret".into();
        state.save(dir.path()).unwrap();
        let reopened = State::load(dir.path());
        assert_eq!(reopened.config.mode, Mode::Ssh);
        assert_eq!(reopened.config.ssh.host, "box");
        assert!(reopened.has_token);
        assert!(!std::fs::read_to_string(dir.path().join("connection.json")).unwrap().contains("secret"), "the token is not in the config file");
    }
    #[test]
    fn ssh_rejects_option_injection() {
        assert!(test_ssh(Ssh { host: "-oProxyCommand=x".into(), ..Default::default() }).is_err());
        assert!(test_ssh(Ssh::default()).is_err());
    }
}

/// Settings › Gateways › Hermes Cloud: "Your agents".
fn cloud_agents(state: &State, dark: bool) -> Element<'_, Message> {
    use hermes_engine::portal::CloudDiscovery;
    let p = super::ui::palette(dark);
    let mut header = row![text("Your agents").size(13).font(super::ui::medium()), Space::new().width(Fill)].spacing(6).align_y(iced::Alignment::Center);
    if state.cloud_org.is_some() { header = header.push(button(text("Change org").size(12)).padding([3, 8]).style(super::ui::quiet_button).on_press(Message::Connection(Action::CloudDiscover(None)))); }
    header = header.push(button(text(if state.cloud_loading { "Loading…" } else { "Refresh" }).size(12)).padding([3, 8]).style(super::ui::quiet_button)
        .on_press_maybe((!state.cloud_loading).then(|| Message::Connection(Action::CloudDiscover(state.cloud_org.clone())))));
    let mut col = column![header].spacing(8);
    match &state.cloud {
        _ if state.cloud_loading => col = col.push(text("Loading your agents…").size(12).color(p.muted)),
        None => {}
        Some(Err(e)) => col = col.push(text(e.clone()).size(12).color(p.muted)),
        Some(Ok(CloudDiscovery::PickOrg(orgs))) => {
            col = col.push(text("Choose an organization").size(12).color(p.muted));
            let rows = orgs.iter().map(|o| settings::card_row_owned(o.name.clone(), format!("Role: {}", o.role),
                button(text("Select").size(12)).padding([5, 12]).on_press(Message::Connection(Action::CloudDiscover(Some(o.slug.clone().unwrap_or(o.id.clone()))))).into(), dark)).collect();
            col = col.push(settings::card(rows, dark));
        }
        Some(Ok(CloudDiscovery::Agents { agents, .. })) if agents.is_empty() => {
            col = col.push(row![text("No agents found on this account. Create one in the ").size(12).color(p.muted),
                button(text("Nous portal").size(12)).padding(0).style(super::ui::quiet_button).on_press(Message::Link("https://portal.nousresearch.com/agents".into())),
                text(", then refresh.").size(12).color(p.muted)].align_y(iced::Alignment::Center));
        }
        Some(Ok(CloudDiscovery::Agents { agents, .. })) => {
            let rows = agents.iter().map(|a| {
                let active = a.dashboard_url.as_deref().is_some_and(|u| u.trim_end_matches('/') == state.config.url.trim().trim_end_matches('/') && state.config.mode == Mode::Remote);
                let action: Element<'_, Message> = if active { text("Active in this window").size(12).color(p.accent).into() } else {
                    button(text(if a.dashboard_url.is_some() { "Connect" } else { "Provisioning…" }).size(12)).padding([5, 12])
                        .on_press_maybe(a.dashboard_url.clone().map(|u| Message::Connection(Action::CloudConnect(u)))).into()
                };
                let known = !a.gateway_state.trim().is_empty() && !a.gateway_state.eq_ignore_ascii_case("unknown");
                settings::card_row_owned(a.name.clone(), if known { format!("Status: {}", a.gateway_state) } else { String::new() }, action, dark)
            }).collect();
            col = col.push(settings::card(rows, dark));
        }
    }
    col.into()
}
