//! Settings › Providers › Accounts: the subscription sign-ins beyond Nous
//! Portal, ported from hermes_cli/web_server_oauth.py `_OAUTH_PROVIDER_CATALOG`
//! and the device-code flows in hermes_cli/auth_{codex,xai,minimax}.py.
//! Device-code providers sign in inside the app; "external" ones (Qwen CLI,
//! Anthropic, Claude Code) run their own CLI in the terminal, exactly as the
//! Electron card does, and the app reads the credential they leave behind.
//! Tokens are stored in the desktop's own `accounts-oauth.json` (0600) and
//! refreshed before each turn; a chat whose base URL is an account's inference
//! URL with no API key is authenticated with that account.
use crate::{CancellationToken, Config, Session};
use anyhow::{Context, Result, bail, ensure};
use base64::Engine;
use reqwest::Client;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf, time::Duration};
use tokio::time::{Instant, sleep};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow { DeviceCode, External }

#[derive(Clone, Debug)]
pub struct Provider {
    pub id: &'static str, pub name: &'static str, pub flow: Flow, pub cli_command: &'static str, pub docs_url: &'static str,
    /// Where chats run once signed in, and a sensible first model.
    pub base_url: &'static str, pub default_model: &'static str, pub description: &'static str,
}

const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CODEX_ISSUER: &str = "https://auth.openai.com";
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const XAI_ISSUER: &str = "https://auth.x.ai";
const XAI_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const XAI_SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
const MINIMAX_CLIENT_ID: &str = "78257093-7e40-4613-99e0-527b14b39113";
const MINIMAX_SCOPE: &str = "group_id profile model.completion";
const MINIMAX_PORTAL: &str = "https://api.minimax.io";
const QWEN_CLIENT_ID: &str = "f0304373b74a44d2b584a3fb70ca9e56";
const QWEN_TOKEN_URL: &str = "https://chat.qwen.ai/api/v1/oauth2/token";

/// The Electron Accounts list, after Nous Portal (which has its own featured card).
pub const CATALOG: &[Provider] = &[
    Provider { id: "openai-codex", name: "ChatGPT or Codex Subscription", flow: Flow::DeviceCode, cli_command: "hermes auth add openai-codex",
        docs_url: "https://platform.openai.com/docs", base_url: "https://chatgpt.com/backend-api/codex", default_model: "gpt-5.5",
        description: "Use your ChatGPT Plus, Pro or Business plan" },
    Provider { id: "qwen-oauth", name: "Qwen (via Qwen CLI)", flow: Flow::External, cli_command: "qwen",
        docs_url: "https://github.com/QwenLM/qwen-code", base_url: "https://portal.qwen.ai/v1", default_model: "qwen3-coder-plus",
        description: "Signs in through the Qwen CLI, then Hermes reuses it" },
    Provider { id: "minimax-oauth", name: "MiniMax (OAuth)", flow: Flow::DeviceCode, cli_command: "hermes auth add minimax-oauth",
        docs_url: "https://www.minimax.io", base_url: "https://api.minimax.io/anthropic", default_model: "MiniMax-M2",
        description: "MiniMax coding plan" },
    Provider { id: "xai-oauth", name: "xAI Grok OAuth (SuperGrok / Premium+)", flow: Flow::DeviceCode, cli_command: "hermes auth add xai-oauth",
        docs_url: "https://hermes-agent.nousresearch.com/docs/guides/xai-grok-oauth", base_url: "https://api.x.ai/v1", default_model: "grok-4",
        description: "Use your SuperGrok or X Premium+ subscription" },
    Provider { id: "anthropic", name: "Anthropic Account", flow: Flow::External, cli_command: "hermes auth add anthropic",
        docs_url: "https://docs.claude.com/en/api/getting-started", base_url: "https://api.anthropic.com/v1", default_model: "claude-sonnet-4-5",
        description: "Sign in from the terminal, or use an API key" },
    Provider { id: "claude-code", name: "Anthropic OAuth: Required Extra Usage Credits to Use Subscription", flow: Flow::External, cli_command: "claude setup-token",
        docs_url: "https://docs.claude.com/en/docs/claude-code", base_url: "https://api.anthropic.com/v1", default_model: "claude-sonnet-4-5",
        description: "Set up a Claude Code token from the terminal" },
];
pub fn provider(id: &str) -> Option<&'static Provider> { CATALOG.iter().find(|p| p.id == id) }
/// The account whose inference URL a chat's base URL names.
pub fn for_base_url(base: &str) -> Option<&'static Provider> {
    let base = base.trim().trim_end_matches('/');
    CATALOG.iter().find(|p| p.flow == Flow::DeviceCode && p.base_url == base).or_else(|| (base == "https://portal.qwen.ai/v1").then(|| provider("qwen-oauth")).flatten())
}

fn client() -> Result<Client> {
    Ok(Client::builder().user_agent(concat!("HermesRust/", env!("CARGO_PKG_VERSION"))).timeout(Duration::from_secs(20)).redirect(reqwest::redirect::Policy::none()).build()?)
}
/// MiniMax moved its OAuth endpoints to account.minimax.io behind 307s; follow
/// those (method and body kept) only within https *.minimax.io.
fn minimax_client() -> Result<Client> {
    let policy = reqwest::redirect::Policy::custom(|attempt| {
        let ok = attempt.previous().len() < 3 && attempt.url().scheme() == "https"
            && attempt.url().host_str().is_some_and(|h| h == "minimax.io" || h.ends_with(".minimax.io"));
        if ok { attempt.follow() } else { attempt.stop() }
    });
    Ok(Client::builder().user_agent(concat!("HermesRust/", env!("CARGO_PKG_VERSION"))).timeout(Duration::from_secs(20)).redirect(policy).build()?)
}
fn now() -> i64 { chrono::Utc::now().timestamp() }

// ---------------------------------------------------------------- store
fn store_path() -> Result<PathBuf> { Ok(Session::default_dir()?.join("accounts-oauth.json")) }
fn read_store() -> BTreeMap<String, Value> {
    store_path().ok().and_then(|p| std::fs::read(p).ok()).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}
fn write_store(store: &BTreeMap<String, Value>) -> Result<()> {
    let path = store_path()?;
    let dir = path.parent().context("No store directory")?;
    std::fs::create_dir_all(dir)?;
    let temp = tempfile::NamedTempFile::new_in(dir)?;
    #[cfg(unix)] { use std::os::unix::fs::PermissionsExt; std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o600))?; }
    serde_json::to_writer(temp.as_file(), store)?;
    temp.as_file().sync_all()?;
    temp.persist(path)?;
    Ok(())
}
fn save(id: &str, state: Value) -> Result<()> { let mut s = read_store(); s.insert(id.into(), state); write_store(&s) }

fn hermes_home() -> Option<PathBuf> {
    std::env::var_os("HERMES_HOME").map(PathBuf::from).or_else(|| directories::BaseDirs::new().map(|d| d.home_dir().join(".hermes")))
}
/// A provider the Hermes CLI signed in to (~/.hermes/auth.json), read-only.
fn cli_state(id: &str) -> Option<Value> {
    let store: Value = serde_json::from_slice(&std::fs::read(hermes_home()?.join("auth.json")).ok()?).ok()?;
    let state = store.pointer(&format!("/providers/{id}"))?.clone();
    (state.get("access_token").and_then(Value::as_str).is_some_and(|t| !t.is_empty())
        || state.pointer("/tokens/access_token").and_then(Value::as_str).is_some_and(|t| !t.is_empty())).then_some(state)
}
fn qwen_creds_path() -> Option<PathBuf> { directories::BaseDirs::new().map(|d| d.home_dir().join(".qwen/oauth_creds.json")) }

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Status { pub logged_in: bool, pub source: String }
pub fn status(id: &str) -> Status {
    if read_store().get(id).and_then(|s| s["access_token"].as_str()).is_some_and(|t| !t.is_empty()) {
        return Status { logged_in: true, source: "Signed in here".into() };
    }
    match id {
        "qwen-oauth" => if qwen_creds_path().is_some_and(|p| p.is_file()) { return Status { logged_in: true, source: "Qwen CLI".into() } },
        "claude-code" => if directories::BaseDirs::new().is_some_and(|d| d.home_dir().join(".claude/.credentials.json").is_file()) { return Status { logged_in: true, source: "Claude Code".into() } },
        _ => {}
    }
    if cli_state(id).is_some() { return Status { logged_in: true, source: "Hermes CLI".into() }; }
    Status::default()
}
pub fn disconnect(id: &str) -> Result<bool> {
    let mut s = read_store();
    let removed = s.remove(id).is_some();
    write_store(&s)?;
    Ok(removed)
}

// ---------------------------------------------------------------- device login
#[derive(Clone)]
pub struct DeviceLogin {
    pub provider: &'static str, pub verification_url: String, pub user_code: String,
    secret: Value, expires: Instant, interval: u64,
}
impl std::fmt::Debug for DeviceLogin { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("DeviceLogin([private])") } }

fn random_url_safe(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut buf).expect("system RNG");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(buf)
}
fn s256(verifier: &str) -> String { base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(ring::digest::digest(&ring::digest::SHA256, verifier.as_bytes())) }
fn https_on(url: &str, host_suffix: &str) -> Result<String> {
    let parsed = reqwest::Url::parse(url).context("Invalid verification URL")?;
    ensure!(parsed.scheme() == "https" && parsed.host_str().is_some_and(|h| h == host_suffix || h.ends_with(&format!(".{host_suffix}"))), "Unexpected sign-in page origin");
    Ok(parsed.to_string())
}

async fn xai_token_endpoint(client: &Client) -> Result<String> {
    let body: Value = client.get(format!("{XAI_ISSUER}/.well-known/openid-configuration")).send().await?.json().await?;
    let endpoint = body["token_endpoint"].as_str().context("xAI discovery returned no token endpoint")?;
    https_on(endpoint, "x.ai")
}

pub async fn begin(id: &str) -> Result<DeviceLogin> {
    let provider = provider(id).filter(|p| p.flow == Flow::DeviceCode).context("This provider signs in from its own CLI")?;
    let client = client()?;
    let (url, code, secret, ttl, interval) = match id {
        "openai-codex" => {
            let response = client.post(format!("{CODEX_ISSUER}/api/accounts/deviceauth/usercode")).json(&json!({"client_id":CODEX_CLIENT_ID})).send().await?;
            if response.status().as_u16() == 429 { bail!("OpenAI is rate-limiting sign-in requests. Wait a minute and try again."); }
            ensure!(response.status().is_success(), "OpenAI sign-in returned HTTP {}", response.status());
            let body: Value = response.json().await?;
            let code = body["user_code"].as_str().context("Missing user code")?.to_owned();
            let interval = body["interval"].as_u64().or_else(|| body["interval"].as_str()?.parse().ok()).unwrap_or(5).max(3);
            (format!("{CODEX_ISSUER}/codex/device"), code.clone(), json!({"device_auth_id":body["device_auth_id"],"user_code":code}), 15 * 60, interval)
        }
        "xai-oauth" => {
            let response = client.post(format!("{XAI_ISSUER}/oauth2/device/code")).header("Accept", "application/json").form(&[("client_id", XAI_CLIENT_ID), ("scope", XAI_SCOPE)]).send().await?;
            ensure!(response.status().is_success(), "xAI sign-in returned HTTP {}", response.status());
            let body: Value = response.json().await?;
            let url = https_on(body["verification_uri_complete"].as_str().or(body["verification_uri"].as_str()).context("Missing verification URL")?, "x.ai")?;
            (url, body["user_code"].as_str().context("Missing user code")?.into(), json!({"device_code":body["device_code"],"token_endpoint":xai_token_endpoint(&client).await?}),
                body["expires_in"].as_u64().unwrap_or(600), body["interval"].as_u64().unwrap_or(5).max(1))
        }
        "minimax-oauth" => {
            let client = minimax_client()?;
            let (verifier, state) = (random_url_safe(32), random_url_safe(16));
            let response = client.post(format!("{MINIMAX_PORTAL}/oauth/code")).header("Accept", "application/json").header("x-request-id", uuid_v4())
                .form(&[("response_type", "code"), ("client_id", MINIMAX_CLIENT_ID), ("scope", MINIMAX_SCOPE), ("code_challenge", &s256(&verifier)), ("code_challenge_method", "S256"), ("state", &state)]).send().await?;
            ensure!(response.status().is_success(), "MiniMax sign-in returned HTTP {}", response.status());
            let body: Value = response.json().await?;
            ensure!(body["state"].as_str() == Some(state.as_str()), "MiniMax sign-in state mismatch");
            let url = https_on(body["verification_uri"].as_str().context("Missing verification URL")?, "minimax.io")?;
            let code: String = body["user_code"].as_str().context("Missing user code")?.into();
            let ttl = expiry_unix(body["expired_in"].as_i64().unwrap_or(600)).saturating_sub(now()).clamp(60, 3600) as u64;
            (url, code.clone(), json!({"user_code":code,"verifier":verifier}), ttl, body["interval"].as_u64().map(|ms| if ms > 100 { ms / 1000 } else { ms }).unwrap_or(2).max(1))
        }
        _ => bail!("Unknown provider"),
    };
    Ok(DeviceLogin { provider: provider.id, verification_url: url, user_code: code, secret, expires: Instant::now() + Duration::from_secs(ttl.min(3600)), interval })
}

fn uuid_v4() -> String { let r = random_url_safe(16); r }
/// MiniMax `expired_in`: unix ms, unix seconds, or a TTL.
fn expiry_unix(value: i64) -> i64 { if value > 1_000_000_000_000 { value / 1000 } else if value > 1_000_000_000 { value } else { now() + value } }

impl DeviceLogin {
    pub async fn finish(self, cancel: CancellationToken) -> Result<()> {
        tokio::select! {
            _ = cancel.cancelled() => bail!("Sign-in cancelled"),
            result = tokio::time::timeout_at(self.expires, self.poll()) => result.context("The sign-in code expired")?,
        }
    }
    async fn poll(&self) -> Result<()> {
        let client = client()?;
        let mut interval = self.interval;
        loop {
            sleep(Duration::from_secs(interval)).await;
            match self.provider {
                "openai-codex" => {
                    let response = client.post(format!("{CODEX_ISSUER}/api/accounts/deviceauth/token")).json(&self.secret).send().await?;
                    if matches!(response.status().as_u16(), 403 | 404) { continue; }
                    ensure!(response.status().is_success(), "OpenAI sign-in returned HTTP {}", response.status());
                    let code: Value = response.json().await?;
                    let tokens: Value = client.post(CODEX_TOKEN_URL).form(&[("grant_type", "authorization_code"), ("code", code["authorization_code"].as_str().context("Missing authorization code")?),
                        ("redirect_uri", &format!("{CODEX_ISSUER}/deviceauth/callback")), ("client_id", CODEX_CLIENT_ID), ("code_verifier", code["code_verifier"].as_str().context("Missing code verifier")?)])
                        .send().await?.error_for_status().context("OpenAI token exchange failed")?.json().await?;
                    return save(self.provider, token_state(&tokens, None, json!({}))?);
                }
                "xai-oauth" => {
                    let endpoint = self.secret["token_endpoint"].as_str().context("Missing token endpoint")?;
                    let response = client.post(endpoint).header("Accept", "application/json").form(&[("grant_type", "urn:ietf:params:oauth:grant-type:device_code"), ("client_id", XAI_CLIENT_ID), ("device_code", self.secret["device_code"].as_str().unwrap_or_default())]).send().await?;
                    let status = response.status();
                    let body: Value = response.json().await.unwrap_or(Value::Null);
                    if status.is_success() { return save(self.provider, token_state(&body, None, json!({"token_endpoint":endpoint}))?); }
                    match body["error"].as_str() {
                        Some("authorization_pending") => {}
                        Some("slow_down") => interval = (interval + 5).min(60),
                        Some("access_denied") => bail!("xAI sign-in was declined"),
                        Some("expired_token") => bail!("The xAI sign-in code expired"),
                        _ => bail!("xAI sign-in returned HTTP {status}"),
                    }
                }
                "minimax-oauth" => {
                    let response = minimax_client()?.post(format!("{MINIMAX_PORTAL}/oauth/token")).header("Accept", "application/json").form(&[("grant_type", "urn:ietf:params:oauth:grant-type:user_code"), ("client_id", MINIMAX_CLIENT_ID),
                        ("user_code", self.secret["user_code"].as_str().unwrap_or_default()), ("code_verifier", self.secret["verifier"].as_str().unwrap_or_default())]).send().await?;
                    ensure!(response.status().is_success(), "MiniMax sign-in returned HTTP {}", response.status());
                    let body: Value = response.json().await.unwrap_or(Value::Null);
                    match body["status"].as_str() {
                        Some("success") => {
                            let expires = expiry_unix(body["expired_in"].as_i64().context("Missing expiry")?);
                            return save(self.provider, json!({"access_token":body["access_token"],"refresh_token":body["refresh_token"],"expires_at":expires}));
                        }
                        Some("error") => bail!("MiniMax reported an error. Try again later."),
                        _ => {}
                    }
                }
                _ => bail!("Unknown provider"),
            }
        }
    }
}

fn token_state(body: &Value, previous: Option<&Value>, extra: Value) -> Result<Value> {
    let access = body["access_token"].as_str().filter(|t| !t.is_empty()).context("The provider returned no access token")?;
    let refresh = body["refresh_token"].as_str().or(previous.and_then(|p| p["refresh_token"].as_str())).unwrap_or_default();
    let expires = body["expires_in"].as_i64().map(|s| now() + s).or_else(|| jwt_exp(access)).unwrap_or(now() + 3600);
    let mut state = json!({"access_token":access,"refresh_token":refresh,"expires_at":expires});
    if let (Some(map), Some(extra)) = (state.as_object_mut(), extra.as_object()) { map.extend(extra.clone()); }
    if let (Some(prev), Some(map)) = (previous.and_then(Value::as_object), state.as_object_mut()) {
        for (k, v) in prev { map.entry(k.clone()).or_insert(v.clone()); }
    }
    Ok(state)
}
fn jwt_exp(token: &str) -> Option<i64> {
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(token.split('.').nth(1)?.trim_end_matches('=')).ok()?;
    serde_json::from_slice::<Value>(&payload).ok()?["exp"].as_i64()
}

// ---------------------------------------------------------------- runtime
/// A fresh access token for `id`, refreshing (and saving) it when near expiry.
pub async fn access_token(id: &str) -> Result<String> {
    if id == "qwen-oauth" { return qwen_token().await; }
    let mut state = read_store().get(id).cloned().or_else(|| cli_state(id).map(|s| if s.get("tokens").is_some() { s["tokens"].clone() } else { s }))
        .with_context(|| format!("Sign in to {} in Settings › Accounts", provider(id).map_or(id, |p| p.name)))?;
    let expires = state["expires_at"].as_i64().or_else(|| state["access_token"].as_str().and_then(jwt_exp)).unwrap_or(i64::MAX);
    if expires > now() + 120 { return state["access_token"].as_str().map(str::to_owned).context("Missing access token"); }
    let refresh = state["refresh_token"].as_str().filter(|t| !t.is_empty()).context("The session expired. Sign in again in Settings › Accounts")?.to_owned();
    let client = client()?;
    let (url, form): (String, Vec<(&str, String)>) = match id {
        "openai-codex" => (CODEX_TOKEN_URL.into(), vec![("grant_type", "refresh_token".into()), ("refresh_token", refresh), ("client_id", CODEX_CLIENT_ID.into())]),
        "xai-oauth" => (match state["token_endpoint"].as_str() { Some(e) => https_on(e, "x.ai")?, None => xai_token_endpoint(&client).await? },
            vec![("grant_type", "refresh_token".into()), ("client_id", XAI_CLIENT_ID.into()), ("refresh_token", refresh)]),
        "minimax-oauth" => (format!("{MINIMAX_PORTAL}/oauth/token"), vec![("grant_type", "refresh_token".into()), ("client_id", MINIMAX_CLIENT_ID.into()), ("refresh_token", refresh)]),
        _ => bail!("This account cannot be refreshed here"),
    };
    let client = if id == "minimax-oauth" { minimax_client()? } else { client };
    let response = client.post(url).header("Accept", "application/json").form(&form).send().await?;
    ensure!(response.status().is_success(), "Token refresh returned HTTP {}. Sign in again in Settings › Accounts", response.status());
    let body: Value = response.json().await?;
    state = if id == "minimax-oauth" {
        json!({"access_token":body["access_token"],"refresh_token":body["refresh_token"].as_str().or(state["refresh_token"].as_str()),"expires_at":expiry_unix(body["expired_in"].as_i64().unwrap_or(3600))})
    } else { token_state(&body, Some(&state), json!({}))? };
    save(id, state.clone())?;
    state["access_token"].as_str().map(str::to_owned).context("Missing access token")
}

async fn qwen_token() -> Result<String> {
    let path = qwen_creds_path().context("No home directory")?;
    let mut creds: Value = serde_json::from_slice(&std::fs::read(&path).context("Sign in with the Qwen CLI first (run `qwen` in the terminal)")?)?;
    let expiry_ms = creds["expiry_date"].as_i64().unwrap_or(0);
    if expiry_ms / 1000 > now() + 120 { return creds["access_token"].as_str().map(str::to_owned).context("Missing Qwen access token"); }
    let refresh = creds["refresh_token"].as_str().context("The Qwen session expired. Run `qwen` to sign in again")?.to_owned();
    let body: Value = client()?.post(QWEN_TOKEN_URL).header("Accept", "application/json").form(&[("grant_type", "refresh_token"), ("refresh_token", refresh.as_str()), ("client_id", QWEN_CLIENT_ID)])
        .send().await?.error_for_status().context("Qwen token refresh failed")?.json().await?;
    let access = body["access_token"].as_str().context("Qwen returned no access token")?.to_owned();
    creds["access_token"] = json!(access);
    if let Some(r) = body["refresh_token"].as_str() { creds["refresh_token"] = json!(r); }
    creds["expiry_date"] = json!((now() + body["expires_in"].as_i64().unwrap_or(3600)) * 1000);
    std::fs::write(&path, serde_json::to_vec_pretty(&creds)?)?;
    Ok(access)
}

/// Fill in a signed-in account's token for a chat pointed at its inference URL.
pub(crate) async fn resolve(mut config: Config) -> Result<Config> {
    if !config.api_key.trim().is_empty() { return Ok(config); }
    if let Some(provider) = for_base_url(&config.base_url) { config.api_key = access_token(provider.id).await?; }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn catalog_matches_the_electron_accounts_list() {
        let ids: Vec<_> = CATALOG.iter().map(|p| p.id).collect();
        assert_eq!(ids, ["openai-codex", "qwen-oauth", "minimax-oauth", "xai-oauth", "anthropic", "claude-code"]);
        assert_eq!(for_base_url("https://api.x.ai/v1/").map(|p| p.id), Some("xai-oauth"));
        assert_eq!(for_base_url("https://chatgpt.com/backend-api/codex").map(|p| p.id), Some("openai-codex"));
        assert!(for_base_url("https://api.anthropic.com/v1").is_none(), "Anthropic stays an API-key provider here");
        assert_eq!(s256("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"), "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM", "RFC 7636 vector");
        assert!(https_on("https://accounts.x.ai/device?c=1", "x.ai").is_ok() && https_on("https://x.ai.evil.com/", "x.ai").is_err() && https_on("http://x.ai/", "x.ai").is_err());
        assert_eq!(expiry_unix(1_900_000_000_000), 1_900_000_000);
        let state = token_state(&json!({"access_token":"a","expires_in":60}), Some(&json!({"refresh_token":"r","token_endpoint":"e"})), json!({})).unwrap();
        assert_eq!((state["refresh_token"].as_str(), state["token_endpoint"].as_str()), (Some("r"), Some("e")));
    }
    /// Network: each device-code provider issues a user code. `cargo test -- --ignored live_begin`
    #[tokio::test] #[ignore]
    async fn live_begin_each_device_login() {
        for id in ["openai-codex", "xai-oauth", "minimax-oauth"] {
            match begin(id).await { Ok(l) => eprintln!("{id}: {} code={}", l.verification_url, l.user_code.len()), Err(e) => eprintln!("{id}: ERROR {e:#}") }
        }
    }
}
