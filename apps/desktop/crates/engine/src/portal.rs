//! Nous Portal device authorization and inference credentials, independent of the Python CLI.
use crate::{CancellationToken, Config, Session};
use anyhow::{Context, Result, bail, ensure};
use chrono::Utc;
use fs2::FileExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs::File, io::Write, path::{Path, PathBuf}, time::Duration};
use tempfile::NamedTempFile;
use tokio::time::{Instant, sleep};

pub const INFERENCE: &str = "https://inference-api.nousresearch.com/v1";
pub const WELCOME: &str = "https://welcome-api.nousresearch.com/v1";
pub const AUTO_MODEL: &str = "auto";
const PORTAL: &str = "https://portal.nousresearch.com";
const CLIENT_ID: &str = "hermes-cli";
const SCOPE: &str = "inference:invoke";

fn client() -> Result<Client> { Ok(Client::builder().user_agent(concat!("HermesRust/",env!("CARGO_PKG_VERSION"))).timeout(Duration::from_secs(20)).redirect(reqwest::redirect::Policy::none()).build()?) }

fn inference_url(value: &str) -> Result<&str> {
    let value = value.trim_end_matches('/');
    ensure!([INFERENCE, WELCOME].contains(&value), "Unrecognized Nous inference endpoint");
    Ok(value)
}

pub fn is_portal(base: &str) -> bool { inference_url(base).is_ok() }

#[derive(Serialize, Deserialize)]
struct Tokens {
    access_token: String,
    #[serde(default)] refresh_token: Option<String>,
    inference_base_url: String,
    expires_at: i64,
    scope: String,
}
impl Tokens {
    fn usable(&self) -> bool {
        !self.access_token.is_empty() && self.expires_at > Utc::now().timestamp() + 60
            && self.scope.split_whitespace().any(|s| s == SCOPE) && is_portal(&self.inference_base_url)
    }
}

fn store_path() -> Result<PathBuf> { Ok(Session::default_dir()?.join("portal-oauth.json")) }
fn read_own(path: &Path) -> Result<Option<Tokens>> {
    match File::open(path) {
        Ok(file) => Ok(Some(serde_json::from_reader(file).context("Could not read the desktop Portal session")?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into())
    }
}
fn cli_session() -> Option<Tokens> {
    // A desktop disconnect never signs another installation out or silently
    // borrows that installation's account again.
    if store_path().ok()?.with_file_name("portal-disconnected").exists() { return None; }
    let home = std::env::var_os("HERMES_HOME").map(PathBuf::from).or_else(|| directories::BaseDirs::new().map(|dirs| dirs.home_dir().join(".hermes")))?;
    let store: Value = serde_json::from_reader(File::open(home.join("auth.json")).ok()?).ok()?;
    let state = store.pointer("/providers/nous")?;
    if state.get("portal_base_url").and_then(Value::as_str).is_some_and(|url| url.trim_end_matches('/') != PORTAL) { return None; }
    // Only borrow a currently valid access token. The CLI retains ownership of its rotating refresh token.
    let token = Tokens { access_token:state["access_token"].as_str()?.into(), refresh_token:None,
        inference_base_url:inference_url(state["inference_base_url"].as_str()?).ok()?.into(),
        expires_at:chrono::DateTime::parse_from_rfc3339(state["expires_at"].as_str()?).ok()?.timestamp(),
        scope:state["scope"].as_str()?.into() };
    token.usable().then_some(token)
}
pub fn is_available() -> bool {
    match store_path().and_then(|path|read_own(&path)) {
        Ok(Some(t)) => t.usable() || (t.refresh_token.as_ref().is_some_and(|value|!value.is_empty()) && t.scope.split_whitespace().any(|scope|scope==SCOPE) && is_portal(&t.inference_base_url)),
        Ok(None) => cli_session().is_some(),
        Err(_) => false,
    }
}

async fn lock(path: &Path) -> Result<File> {
    let parent = path.parent().context("Missing OAuth store directory")?;
    std::fs::create_dir_all(parent)?;
    let file = File::options().read(true).write(true).create(true).truncate(false).open(path.with_extension("lock"))?;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match file.try_lock_exclusive() {
            Ok(()) => return Ok(file),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline => sleep(Duration::from_millis(100)).await,
            Err(e) => return Err(e).context("Could not lock the Portal session")
        }
    }
}
fn save(path: &Path, tokens: &Tokens) -> Result<()> {
    let mut out = NamedTempFile::new_in(path.parent().context("Missing OAuth store directory")?)?;
    serde_json::to_writer(&mut out,tokens)?;
    out.flush()?;
    out.as_file().sync_all()?;
    out.persist(path).context("Could not save the Portal session")?;
    let marker=path.with_file_name("portal-disconnected");
    if marker.exists() { std::fs::remove_file(marker)?; }
    Ok(())
}

pub async fn disconnect() -> Result<()> {
    let path=store_path()?;let _guard=lock(&path).await?;
    let mut marker=NamedTempFile::new_in(path.parent().context("Missing OAuth store directory")?)?;
    marker.write_all(b"Desktop account disconnected\n")?;marker.as_file().sync_all()?;
    marker.persist(path.with_file_name("portal-disconnected"))?;
    match std::fs::remove_file(path) { Ok(())=>Ok(()),Err(error) if error.kind()==std::io::ErrorKind::NotFound=>Ok(()),Err(error)=>Err(error.into()) }
}

pub(crate) async fn catalog_credentials(mut config:Config)->Result<Config> {
    if is_portal(&config.base_url) && config.api_key.is_empty() {
        let token=credentials().await?;config.base_url=token.inference_base_url;config.api_key=token.access_token;
    }
    Ok(config)
}

fn decode(response: &Value, previous: Option<&Tokens>) -> Result<Tokens> {
    let scope = response["scope"].as_str().or(previous.map(|t| t.scope.as_str())).unwrap_or(SCOPE);
    ensure!(scope.split_whitespace().any(|s| s == SCOPE), "Nous has not granted inference permission");
    let token = Tokens {
        access_token:response["access_token"].as_str().filter(|s|!s.is_empty()).context("Portal returned no access token")?.into(),
        refresh_token:response["refresh_token"].as_str().map(str::to_owned).or_else(||previous.and_then(|p|p.refresh_token.clone())),
        scope:scope.into(),
        expires_at:Utc::now().timestamp()+response["expires_in"].as_i64().filter(|n| (1..=31_536_000).contains(n)).context("Invalid Portal token lifetime")?,
        inference_base_url:inference_url(response["inference_base_url"].as_str().or(previous.map(|p| p.inference_base_url.as_str())).unwrap_or(INFERENCE))?.into()
    };
    Ok(token)
}

async fn credentials() -> Result<Tokens> {
    let path = store_path()?;
    let _guard = lock(&path).await?;
    let Some(prev) = read_own(&path)? else {
        return cli_session().context("Sign in with Nous Portal in Provider settings");
    };
    if prev.usable() { return Ok(prev); }
    let refresh = prev.refresh_token.as_deref().context("The Portal session has expired. Sign in again")?;
    let response = client()?.post(format!("{PORTAL}/api/oauth/token")).form(&[
        ("grant_type","refresh_token"),("client_id",CLIENT_ID),("refresh_token",refresh)
    ]).send().await.context("Could not refresh the Portal session")?;
    ensure!(response.status().is_success(),"Nous token refresh returned HTTP {}. Sign in again if the session expired",response.status());
    let token = decode(&response.json().await?,Some(&prev))?;
    save(&path,&token)?;
    Ok(token)
}

#[derive(Clone)]
pub struct DeviceLogin {
    pub verification_url: String,
    pub user_code: String,
    device_code: String,
    expires: Instant,
    interval: u64,
}
impl std::fmt::Debug for DeviceLogin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("DeviceLogin([private])") }
}

pub async fn begin_login() -> Result<DeviceLogin> {
    let response = client()?.post(format!("{PORTAL}/api/oauth/device/code")).form(&[("client_id",CLIENT_ID),("scope",SCOPE)]).send().await?;
    ensure!(response.status().is_success(),"Nous sign-in returned HTTP {}",response.status());
    let body: Value = response.json().await?;
    let url = reqwest::Url::parse(body["verification_uri_complete"].as_str().or_else(||body["verification_uri"].as_str()).context("Missing Nous verification URL")?)?;
    ensure!(url.origin().ascii_serialization() == PORTAL && url.username().is_empty() && url.password().is_none(),"Unexpected OAuth verification origin");
    Ok(DeviceLogin { verification_url:url.to_string(), user_code:body["user_code"].as_str().context("Missing user code")?.into(),
        device_code:body["device_code"].as_str().context("Missing device code")?.into(), expires:Instant::now()+Duration::from_secs(body["expires_in"].as_u64().unwrap_or(600).min(3600)),
        interval:body["interval"].as_u64().unwrap_or(5).clamp(1,60) })
}

impl DeviceLogin {
    pub async fn finish(self, cancel: CancellationToken) -> Result<()> {
        tokio::select! {
            _ = cancel.cancelled() => bail!("Sign-in cancelled"),
            result = tokio::time::timeout_at(self.expires,self.poll()) => result.context("Nous sign-in code expired")?,
        }
    }
    async fn poll(&self) -> Result<()> {
        let client = client()?;
        let mut interval = self.interval;
        loop {
            sleep(Duration::from_secs(interval)).await;
            let response = client.post(format!("{PORTAL}/api/oauth/token")).form(&[("grant_type","urn:ietf:params:oauth:grant-type:device_code"),("client_id",CLIENT_ID),("device_code",self.device_code.as_str())]).send().await?;
            let status = response.status();
            if status.as_u16() == 429 || status.is_server_error() {
                interval = (interval + 5).min(60); continue;
            }
            let body: Value = response.json().await?;
            if status.is_success() {
                let token = decode(&body,None)?;
                let path = store_path()?;
                let _guard = lock(&path).await?;
                return save(&path,&token);
            }
            match body["error"].as_str() {
                Some("authorization_pending") => (),
                Some("slow_down") => interval = (interval + 5).min(60),
                Some("access_denied") => bail!("Nous sign-in was declined"),
                Some("expired_token") => bail!("Nous sign-in code expired"),
                _ => bail!("Nous sign-in returned HTTP {status}"),
            }
        }
    }
}

pub(crate) async fn resolve(mut config: Config) -> Result<Config> {
    if !is_portal(&config.base_url) || !config.api_key.is_empty() { return Ok(config); }
    let token = credentials().await?;
    config.base_url = token.inference_base_url;
    config.api_key = token.access_token;
    if config.model == AUTO_MODEL {
        let response = client()?.get(format!("{}/models",config.base_url)).bearer_auth(&config.api_key).send().await?;
        ensure!(response.status().is_success(),"Nous model discovery returned HTTP {}", response.status());
        let body: Value = response.json().await?;
        let models = body["data"].as_array().context("Nous returned no model catalog")?;
        let ids: Vec<_> = models.iter().filter_map(|m| m["id"].as_str()).collect();
        config.model = [if config.base_url == WELCOME { "nous/welcome" } else { "deepseek/deepseek-v4.1-flash" }, "deepseek/deepseek-v4-flash","openai/gpt-4.1-mini","nous/welcome"].into_iter()
            .find(|id| ids.contains(id)).or_else(||ids.iter().copied().find(|id| !id.to_lowercase().contains("hermes"))).context("No usable model in the Nous catalog")?.to_owned();
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test] fn verifies_credential_destination_and_scope() {
        let good=json!({"access_token":"sample","refresh_token":"refresh","expires_in":3600,"scope":"inference:invoke","inference_base_url":WELCOME});
        let original=decode(&good,None).unwrap();
        assert!(original.usable());
        let mut bad=good.clone(); bad["inference_base_url"]=json!("https://welcome-api.nousresearch.com.example.com/v1");
        assert!(decode(&bad,None).is_err());
        let mut bad=good; bad["scope"]=json!("profile"); assert!(decode(&bad,None).is_err());
        let rotated=decode(&json!({"access_token":"new","expires_in":3600}),Some(&original)).unwrap();
        assert_eq!(rotated.refresh_token.as_deref(),Some("refresh"));
        assert_eq!(rotated.inference_base_url,WELCOME);
    }
}

/// The stock desktop resolves billing on the Portal origin, independently of
/// the chosen inference provider. Only summarized display fields leave here.
pub async fn billing_overview() -> Result<Vec<(String,String)>> {
    let token=credentials().await?;
    let mut response=client()?.get(format!("{PORTAL}/api/billing/state"))
        .bearer_auth(&token.access_token).header("Accept","application/json").send().await?;
    ensure!(response.status().is_success(),"Billing is unavailable (HTTP {}).",response.status());
    let mut bytes=Vec::new();
    while let Some(chunk)=response.chunk().await?{
        ensure!(bytes.len()+chunk.len()<=256*1024,"Billing response exceeded the size limit.");
        bytes.extend_from_slice(&chunk);
    }
    let data:Value=serde_json::from_slice(&bytes).context("Invalid billing response")?;
    ensure!(data.is_object(),"Invalid billing response");
    let mut rows=Vec::new();
    for (label,path) in [("Organization","/org/name"),("Role","/org/role"),("Credit balance (USD)","/balanceUsd"),
        ("Monthly spending cap (USD)","/monthlyCap/limitUsd"),("Auto-reload","/autoReload/enabled"),
        ("Reload threshold (USD)","/autoReload/thresholdUsd"),("Reload to (USD)","/autoReload/reloadToUsd"),
        ("Keep going on credits","/keepGoingOnCredits/enabled")]{
        let display=match data.pointer(path) {
            Some(Value::String(s)) if !s.trim().is_empty()=>s.chars().take(120).collect(),
            Some(Value::Number(n))=>n.to_string(),
            Some(Value::Bool(b))=>if *b{"On".into()}else{"Off".into()},
            _=>continue,
        }; rows.push((label.to_owned(),display));
    }
    if let Some(last)=data.pointer("/card/last4").and_then(Value::as_str).filter(|s|s.len()==4&&s.bytes().all(|b|b.is_ascii_digit())){
        let brand=data.pointer("/card/brand").and_then(Value::as_str).unwrap_or("Card").chars().take(30).collect::<String>();
        rows.push(("Payment method".into(),format!("{brand} •••• {last}")));
    }
    Ok(rows)
}

/// A hosted Hermes Cloud agent (the trimmed DTO of the portal's GET /api/agents).
#[derive(Clone, Debug, PartialEq)]
pub struct CloudAgent { pub id: String, pub name: String, pub status: String, pub dashboard_url: Option<String>, pub gateway_state: String }
/// An organization offered when a multi-org account must pick one (HTTP 409).
#[derive(Clone, Debug, PartialEq)]
pub struct CloudOrg { pub id: String, pub slug: Option<String>, pub name: String, pub role: String }
#[derive(Clone, Debug, PartialEq)]
pub enum CloudDiscovery { Agents { agents: Vec<CloudAgent>, org: Option<String> }, PickOrg(Vec<CloudOrg>) }

fn cloud_orgs(body: &Value) -> Vec<CloudOrg> {
    body["orgs"].as_array().into_iter().flatten().filter_map(|o| Some(CloudOrg { id: o["id"].as_str()?.into(), slug: o["slug"].as_str().map(str::to_owned),
        name: o["name"].as_str().or(o["id"].as_str())?.into(), role: o["role"].as_str().unwrap_or("MEMBER").into() })).collect()
}
fn cloud_agents_from(body: &Value) -> Vec<CloudAgent> {
    let list = body["agents"].as_array().or(body.as_array());
    list.into_iter().flatten().filter_map(|a| Some(CloudAgent { id: a["id"].as_str()?.into(), name: a["name"].as_str().or(a["id"].as_str())?.into(),
        status: a["status"].as_str().unwrap_or_default().into(),
        dashboard_url: a["dashboardUrl"].as_str().filter(|u| u.starts_with("https://")).map(str::to_owned),
        gateway_state: a["dashboardGatewayState"].as_str().unwrap_or("unknown").into() })).collect()
}

/// Discover the Hermes Cloud agents on the signed-in Nous account, scoped to `org` when given.
pub async fn cloud_agents(org: Option<&str>) -> Result<CloudDiscovery> {
    let token = credentials().await.context("Sign in to Hermes Cloud first")?;
    let mut url = reqwest::Url::parse(&format!("{PORTAL}/api/agents"))?;
    if let Some(org) = org { url.query_pairs_mut().append_pair("org", org); }
    let response = client()?.get(url).bearer_auth(&token.access_token).header("Accept", "application/json").send().await?;
    let status = response.status();
    let bytes = response.bytes().await?;
    ensure!(bytes.len() <= 512 * 1024, "The agent list exceeded the size limit.");
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    if status.as_u16() == 409 && body["error"] == "org_selection_required" { return Ok(CloudDiscovery::PickOrg(cloud_orgs(&body))); }
    if matches!(status.as_u16(), 401 | 403) { anyhow::bail!("Hermes Cloud did not accept this sign-in (HTTP {status}). Sign in again from Settings › Accounts."); }
    ensure!(status.is_success(), "Could not load your Hermes Cloud agents (HTTP {status}).");
    Ok(CloudDiscovery::Agents { agents: cloud_agents_from(&body), org: body.pointer("/org/slug").or(body.pointer("/org/id")).and_then(Value::as_str).map(str::to_owned) })
}

#[cfg(test)]
mod cloud_tests {
    use super::*;
    use serde_json::json;
    /// Network: list Portal models that can output images. `cargo test -- --ignored live_image_models`
    #[tokio::test] #[ignore]
    async fn live_image_models() {
        let token = credentials().await.unwrap();
        let body: Value = client().unwrap().get(format!("{}/models", token.inference_base_url)).bearer_auth(&token.access_token).send().await.unwrap().json().await.unwrap();
        let models = body["data"].as_array().unwrap();
        eprintln!("PORTAL models: {}", models.len());
        for m in models {
            let id = m["id"].as_str().unwrap_or_default();
            let out = m.pointer("/architecture/output_modalities").map(|v| v.to_string()).unwrap_or_default();
            if id.contains("image") || out.contains("image") { eprintln!("PORTAL image-capable: {id} out={out} in={}", m.pointer("/architecture/input_modalities").map(|v| v.to_string()).unwrap_or_default()); }
        }
    }
    /// Network, with this machine's real sign-in: `cargo test -- --ignored live_cloud`
    #[tokio::test] #[ignore]
    async fn live_cloud_discovery() {
        match cloud_agents(None).await {
            Ok(CloudDiscovery::Agents { agents, .. }) => {
                eprintln!("CLOUD: ok, {} agents", agents.len());
                for agent in agents.iter().filter_map(|a| a.dashboard_url.clone()).take(3) {
                    let flows = match client().unwrap().get(format!("{}/api/status", agent.trim_end_matches('/'))).send().await {
                        Ok(r) => { let status = r.status(); let v: Value = r.json().await.unwrap_or_default(); format!("{status} flows={}", v["auth_flows"]) }
                        Err(e) => format!("unreachable {e}"),
                    };
                    eprintln!("CLOUD agent: {flows}");
                }
            }
            Ok(CloudDiscovery::PickOrg(orgs)) => eprintln!("CLOUD: pick org ({})", orgs.len()),
            Err(e) => eprintln!("CLOUD: error {e:#}"),
        }
    }
    #[test]
    fn trims_agents_and_orgs() {
        let body = json!({"agents":[{"id":"a1","name":"Research","status":"running","dashboardUrl":"https://a1.agents.nousresearch.com","dashboardGatewayState":"active"},
            {"id":"a2","name":"New","status":"provisioning","dashboardUrl":null},{"id":"a3","dashboardUrl":"http://insecure"},{"name":"no id"}],"org":{"id":"o1","slug":"team"}});
        let agents = cloud_agents_from(&body);
        assert_eq!(agents.len(), 3);
        assert_eq!(agents[0].dashboard_url.as_deref(), Some("https://a1.agents.nousresearch.com"));
        assert_eq!(agents[1].dashboard_url, None, "provisioning");
        assert_eq!(agents[2].dashboard_url, None, "only https dashboards are offered");
        let orgs = cloud_orgs(&json!({"error":"org_selection_required","orgs":[{"id":"o1","slug":"team","name":"Team","role":"OWNER"},{"bad":1}]}));
        assert_eq!(orgs, vec![CloudOrg { id: "o1".into(), slug: Some("team".into()), name: "Team".into(), role: "OWNER".into() }]);
    }
}
