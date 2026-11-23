//! Native sign-in to an OAuth-mode remote or Hermes Cloud gateway
//! (electron/native-oauth.ts, hermes_cli/gateway_connections_oauth.py): PKCE
//! with a loopback redirect to the gateway's `/auth/native/authorize`, the code
//! exchanged at `/auth/native/token`, refreshed at `/auth/native/refresh`. The
//! access token is the bearer for `/api/auth/ws-ticket`.
use hermes_engine::mcp_oauth::{random, s256, wait_for_code};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, time::Duration};

const SKEW: i64 = 60;
fn store(root: &Path) -> std::path::PathBuf { root.join("gateway-oauth.json") }
fn read(root: &Path) -> BTreeMap<String, Value> { std::fs::read(store(root)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default() }
fn write(root: &Path, all: &BTreeMap<String, Value>) -> Result<(), String> {
    let temp = tempfile::NamedTempFile::new_in(root).map_err(|e| e.to_string())?;
    #[cfg(unix)] { use std::os::unix::fs::PermissionsExt; let _ = std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o600)); }
    serde_json::to_writer(temp.as_file(), all).map_err(|e| e.to_string())?;
    temp.persist(store(root)).map_err(|e| e.to_string())?;
    Ok(())
}
fn key(base: &str) -> String { base.trim().trim_end_matches('/').to_owned() }
fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder().timeout(Duration::from_secs(15)).redirect(reqwest::redirect::Policy::none()).build().map_err(|e| e.to_string())
}
fn now() -> i64 { chrono::Utc::now().timestamp() }

pub(super) fn signed_in(root: &Path, base: &str) -> bool { read(root).get(&key(base)).is_some_and(|t| t["access_token"].as_str().is_some_and(|a| !a.is_empty())) }
pub(super) fn sign_out(root: &Path, base: &str) -> Result<(), String> { let mut all = read(root); all.remove(&key(base)); write(root, &all) }

/// Whether the gateway offers the native PKCE flow (`/api/status` → auth_flows).
pub(super) async fn supports_native(base: &str) -> Result<bool, String> {
    let response = client()?.get(format!("{}/api/status", key(base))).send().await
        .map_err(|_| "The gateway is not answering. A Hermes Cloud agent may be waking up — try again in a minute.".to_owned())?;
    Ok(response.json::<Value>().await.ok().and_then(|s| s["auth_flows"].as_array().map(|f| f.iter().any(|v| v == "native_pkce"))).unwrap_or(false))
}

/// Browser sign-in; returns once the gateway has issued tokens.
pub(super) async fn sign_in(root: std::path::PathBuf, base: String) -> Result<(), String> { sign_in_with(root, base, super::open_url).await }
async fn sign_in_with(root: std::path::PathBuf, base: String, open: impl Fn(&str) -> Result<(), String>) -> Result<(), String> {
    let base = key(&base);
    let url = reqwest::Url::parse(&base).map_err(|_| "Invalid gateway URL".to_owned())?;
    if url.scheme() != "https" && !matches!(url.host_str(), Some("127.0.0.1" | "localhost")) { return Err("Gateway sign-in needs an HTTPS address.".into()); }
    let listener = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let redirect = format!("http://127.0.0.1:{}/callback", listener.local_addr().map_err(|e| e.to_string())?.port());
    let (verifier, state) = (random(32), random(16));
    let mut authorize = reqwest::Url::parse(&format!("{base}/auth/native/authorize")).map_err(|e| e.to_string())?;
    authorize.query_pairs_mut().append_pair("code_challenge", &s256(&verifier)).append_pair("code_challenge_method", "S256").append_pair("redirect_uri", &redirect).append_pair("state", &state);
    open(authorize.as_str())?;
    let expected = state.clone();
    let code = tokio::task::spawn_blocking(move || wait_for_code(listener, &expected)).await.map_err(|e| e.to_string())?.map_err(|e| format!("{e:#}"))?;
    let response = client()?.post(format!("{base}/auth/native/token")).json(&json!({"code": code, "code_verifier": verifier})).send().await.map_err(|e| format!("Could not reach {base}: {e}"))?;
    if !response.status().is_success() { return Err(format!("{base} refused the sign-in code ({}).", response.status())); }
    let body: Value = response.json().await.map_err(|e| e.to_string())?;
    let tokens = parse(&body)?;
    let mut all = read(&root); all.insert(base, tokens); write(&root, &all)
}
fn parse(body: &Value) -> Result<Value, String> {
    let access = body["access_token"].as_str().filter(|a| !a.is_empty()).ok_or("The gateway's token response has no access token.")?;
    Ok(json!({"access_token": access, "refresh_token": body["refresh_token"].as_str().unwrap_or_default(), "expires_at": body["expires_at"].as_f64().map(|f| f as i64).unwrap_or(0),
        "provider": body["provider"].as_str().unwrap_or_default()}))
}

/// A current access token, refreshed through `/auth/native/refresh` when near expiry.
pub(super) async fn access_token(root: &Path, base: &str) -> Result<String, String> {
    let base = key(base);
    let mut all = read(root);
    let tokens = all.get(&base).cloned().ok_or("Sign in to this gateway in Settings › Gateways.")?;
    let expires = tokens["expires_at"].as_i64().unwrap_or(0);
    if expires > now() + SKEW { return tokens["access_token"].as_str().map(str::to_owned).ok_or_else(|| "No access token".into()); }
    let refresh = tokens["refresh_token"].as_str().filter(|r| !r.is_empty()).ok_or("The gateway session expired. Sign in again in Settings › Gateways.")?;
    let response = client()?.post(format!("{base}/auth/native/refresh")).json(&json!({"refresh_token": refresh, "provider": tokens["provider"]})).send().await.map_err(|e| e.to_string())?;
    if response.status().as_u16() == 401 { all.remove(&base); let _ = write(root, &all); return Err("The gateway session expired. Sign in again in Settings › Gateways.".into()); }
    if !response.status().is_success() { return Err(format!("Could not renew the gateway session ({}).", response.status())); }
    let mut fresh = parse(&response.json::<Value>().await.map_err(|e| e.to_string())?)?;
    if fresh["refresh_token"].as_str().is_some_and(str::is_empty) { fresh["refresh_token"] = tokens["refresh_token"].clone(); }
    let access = fresh["access_token"].as_str().unwrap_or_default().to_owned();
    all.insert(base, fresh); write(root, &all)?;
    Ok(access)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, routing::{get, post}};
    #[tokio::test]
    async fn native_sign_in_then_refresh_supplies_the_ws_ticket_bearer() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let app = Router::new()
            .route("/api/status", get(|| async { Json(json!({"auth_flows":["native_pkce"]})) }))
            .route("/auth/native/token", post(|Json(b): Json<Value>| async move { assert_eq!(b["code"], "c0de"); assert!(b["code_verifier"].as_str().unwrap().len() > 40);
                Json(json!({"access_token":"a1","refresh_token":"r1","expires_at":0,"provider":"nous"})) }))
            .route("/auth/native/refresh", post(|Json(b): Json<Value>| async move { assert_eq!(b["refresh_token"], "r1");
                Json(json!({"access_token":"a2","expires_at": chrono::Utc::now().timestamp() + 3600})) }));
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        let root = tempfile::tempdir().unwrap();
        assert!(supports_native(&base).await.unwrap());
        let open = |authorize: &str| {
            let url = reqwest::Url::parse(authorize).unwrap();
            let q = |k: &str| url.query_pairs().find(|(n, _)| n == k).unwrap().1.into_owned();
            assert_eq!(q("code_challenge_method"), "S256");
            let callback = reqwest::Url::parse(&format!("{}?code=c0de&state={}", q("redirect_uri"), q("state"))).unwrap();
            std::thread::spawn(move || {
                use std::io::{Read, Write};
                let mut s = std::net::TcpStream::connect(("127.0.0.1", callback.port().unwrap())).unwrap();
                write!(s, "GET {}?{} HTTP/1.1\r\nHost: x\r\n\r\n", callback.path(), callback.query().unwrap()).unwrap();
                let mut out = String::new(); let _ = s.read_to_string(&mut out);
            });
            Ok(())
        };
        sign_in_with(root.path().to_owned(), base.clone(), open).await.unwrap();
        assert!(signed_in(root.path(), &base));
        assert_eq!(access_token(root.path(), &base).await.unwrap(), "a2", "an expired token is renewed");
        assert_eq!(access_token(root.path(), &base).await.unwrap(), "a2", "and reused while fresh");
        sign_out(root.path(), &base).unwrap();
        assert!(!signed_in(root.path(), &base));
    }
}
