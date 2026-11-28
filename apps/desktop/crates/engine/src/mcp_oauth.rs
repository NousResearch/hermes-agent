//! MCP OAuth 2.1 for plugin servers that require sign-in (tools/mcp_oauth.py):
//! protected-resource and authorization-server discovery, dynamic client
//! registration, an authorization-code + PKCE browser sign-in with a loopback
//! callback, and refresh. Tokens live beside the plugin's data (0600).
use anyhow::{Context, Result, bail, ensure};
use base64::Engine;
use reqwest::{Client, Url};
use serde_json::{Value, json};
use std::{io::{Read, Write}, net::TcpListener, path::{Path, PathBuf}, time::Duration};

fn client() -> Result<Client> {
    Ok(Client::builder().timeout(Duration::from_secs(20)).redirect(reqwest::redirect::Policy::none())
        .user_agent(concat!("HermesRust/", env!("CARGO_PKG_VERSION"))).build()?)
}
fn now() -> i64 { chrono::Utc::now().timestamp() }
pub fn random(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut buf).expect("system RNG");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(buf)
}
pub fn s256(v: &str) -> String { base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(ring::digest::digest(&ring::digest::SHA256, v.as_bytes())) }
fn https(url: &str) -> Result<Url> {
    let url = Url::parse(url).context("Invalid OAuth URL")?;
    ensure!(url.scheme() == "https" || matches!(url.host_str(), Some("127.0.0.1" | "localhost")), "OAuth endpoints must use HTTPS");
    Ok(url)
}

pub(crate) fn token_path(data: &Path, server: &str) -> PathBuf {
    let safe: String = server.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
    data.join(format!("oauth-{safe}.json"))
}
fn save(path: &Path, state: &Value) -> Result<()> {
    let dir = path.parent().context("No data directory")?;
    std::fs::create_dir_all(dir)?;
    let temp = tempfile::NamedTempFile::new_in(dir)?;
    #[cfg(unix)] { use std::os::unix::fs::PermissionsExt; std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o600))?; }
    serde_json::to_writer(temp.as_file(), state)?;
    temp.persist(path)?;
    Ok(())
}

/// Authorization-server metadata for an MCP server URL (RFC 9728 → RFC 8414).
async fn discover(client: &Client, server: &Url) -> Result<Value> {
    let origin = server.origin().ascii_serialization();
    let path = server.path().trim_end_matches('/');
    let mut issuer = origin.clone();
    for candidate in [format!("{origin}/.well-known/oauth-protected-resource{path}"), format!("{origin}/.well-known/oauth-protected-resource")] {
        if let Ok(r) = client.get(&candidate).send().await && r.status().is_success() && let Ok(meta) = r.json::<Value>().await
            && let Some(first) = meta["authorization_servers"].as_array().and_then(|a| a.first()).and_then(Value::as_str) { issuer = first.trim_end_matches('/').to_owned(); break; }
    }
    let base = https(&issuer)?;
    let (o, p) = (base.origin().ascii_serialization(), base.path().trim_end_matches('/').to_owned());
    for candidate in [format!("{o}/.well-known/oauth-authorization-server{p}"), format!("{o}/.well-known/openid-configuration{p}"), format!("{issuer}/.well-known/openid-configuration")] {
        if let Ok(r) = client.get(&candidate).send().await && r.status().is_success() && let Ok(meta) = r.json::<Value>().await && meta["token_endpoint"].is_string() { return Ok(meta); }
    }
    // Legacy servers: endpoints at the server's own origin.
    Ok(json!({"authorization_endpoint": format!("{origin}/authorize"), "token_endpoint": format!("{origin}/token"), "registration_endpoint": format!("{origin}/register")}))
}

/// Browser sign-in for one server. Blocks until the callback arrives (5 min) or fails.
pub async fn sign_in(data: &Path, server: &str, url: &str, open: impl Fn(&str)) -> Result<()> {
    let client = client()?;
    let server_url = https(url)?;
    let meta = discover(&client, &server_url).await?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let redirect = format!("http://127.0.0.1:{}/callback", listener.local_addr()?.port());
    let registration = meta["registration_endpoint"].as_str().context("This server does not allow app registration (no registration_endpoint)")?;
    let registered: Value = client.post(https(registration)?.as_str()).json(&json!({"client_name":"Hermes Desktop","redirect_uris":[redirect],
        "grant_types":["authorization_code","refresh_token"],"response_types":["code"],"token_endpoint_auth_method":"none"})).send().await?
        .error_for_status().context("Client registration was refused")?.json().await?;
    let client_id = registered["client_id"].as_str().context("Registration returned no client_id")?.to_owned();
    let (verifier, state) = (random(32), random(16));
    let mut authorize = https(meta["authorization_endpoint"].as_str().context("No authorization endpoint")?)?;
    authorize.query_pairs_mut().append_pair("response_type", "code").append_pair("client_id", &client_id).append_pair("redirect_uri", &redirect)
        .append_pair("code_challenge", &s256(&verifier)).append_pair("code_challenge_method", "S256").append_pair("state", &state).append_pair("resource", server_url.as_str());
    if let Some(scopes) = meta["scopes_supported"].as_array().filter(|s| !s.is_empty()) {
        authorize.query_pairs_mut().append_pair("scope", &scopes.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" "));
    }
    open(authorize.as_str());
    let expected = state.clone();
    let code = tokio::task::spawn_blocking(move || wait_for_code(listener, &expected)).await??;
    let token_endpoint = https(meta["token_endpoint"].as_str().context("No token endpoint")?)?;
    let tokens: Value = client.post(token_endpoint.as_str()).form(&[("grant_type", "authorization_code"), ("code", code.as_str()), ("redirect_uri", redirect.as_str()),
        ("client_id", client_id.as_str()), ("code_verifier", verifier.as_str()), ("resource", server_url.as_str())]).send().await?.error_for_status().context("Token exchange failed")?.json().await?;
    let access = tokens["access_token"].as_str().context("No access token")?;
    save(&token_path(data, server), &json!({"access_token":access,"refresh_token":tokens["refresh_token"],"expires_at":now() + tokens["expires_in"].as_i64().unwrap_or(3600),
        "client_id":client_id,"token_endpoint":token_endpoint.as_str(),"resource":server_url.as_str()}))
}

/// Wait (up to 5 min) on a loopback listener for `/callback?codefn wait_for_code(listener: TcpListener, state: &str) -> Result<String> {state` and return the code.
pub fn wait_for_code(listener: TcpListener, state: &str) -> Result<String> {
    listener.set_nonblocking(false)?;
    let deadline = std::time::Instant::now() + Duration::from_secs(300);
    loop {
        ensure!(std::time::Instant::now() < deadline, "Sign-in timed out");
        let (mut stream, _) = listener.accept()?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        // Read until the request line is complete: it can arrive in several segments.
        let mut buf = Vec::with_capacity(1024);
        let mut chunk = [0u8; 2048];
        while !buf.windows(2).any(|w| w == b"\r\n") && buf.len() < 16384 {
            match stream.read(&mut chunk) { Ok(0) | Err(_) => break, Ok(n) => buf.extend_from_slice(&chunk[..n]) }
        }
        let request = String::from_utf8_lossy(&buf);
        let target = request.lines().next().and_then(|l| l.split_whitespace().nth(1)).unwrap_or("/");
        let Ok(url) = Url::parse(&format!("http://127.0.0.1{target}")) else { continue };
        if url.path() != "/callback" { let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"); continue; }
        let get = |k: &str| url.query_pairs().find(|(n, _)| n == k).map(|(_, v)| v.into_owned());
        let body = |msg: &str| format!("HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\n\r\n<html><body style=\"font-family:system-ui;padding:40px\"><h2>{msg}</h2><p>You can close this tab and return to Hermes.</p></body></html>");
        if let Some(error) = get("error") { let _ = stream.write_all(body("Sign-in was not completed").as_bytes()); bail!("Sign-in failed: {error}"); }
        if get("state").as_deref() != Some(state) { let _ = stream.write_all(body("Sign-in state did not match").as_bytes()); bail!("Sign-in state mismatch"); }
        let code = get("code").context("No authorization code")?;
        let _ = stream.write_all(body("Signed in to Hermes").as_bytes());
        return Ok(code);
    }
}

/// A usable bearer token for this server, refreshed when close to expiry; None when never signed in.
pub(crate) async fn token(data: &Path, server: &str) -> Result<Option<String>> {
    let path = token_path(data, server);
    let Ok(bytes) = std::fs::read(&path) else { return Ok(None) };
    let mut state: Value = serde_json::from_slice(&bytes)?;
    if state["expires_at"].as_i64().unwrap_or(0) > now() + 60 { return Ok(state["access_token"].as_str().map(str::to_owned)); }
    let Some(refresh) = state["refresh_token"].as_str().map(str::to_owned) else { return Ok(None) };
    let endpoint = https(state["token_endpoint"].as_str().context("No token endpoint")?)?;
    let response = client()?.post(endpoint.as_str()).form(&[("grant_type", "refresh_token"), ("refresh_token", refresh.as_str()),
        ("client_id", state["client_id"].as_str().unwrap_or_default()), ("resource", state["resource"].as_str().unwrap_or_default())]).send().await?;
    if !response.status().is_success() { return Ok(None); }
    let tokens: Value = response.json().await?;
    state["access_token"] = tokens["access_token"].clone();
    if tokens["refresh_token"].is_string() { state["refresh_token"] = tokens["refresh_token"].clone(); }
    state["expires_at"] = json!(now() + tokens["expires_in"].as_i64().unwrap_or(3600));
    save(&path, &state)?;
    Ok(state["access_token"].as_str().map(str::to_owned))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, extract::Query, routing::{get, post}};
    #[tokio::test]
    async fn discovers_registers_signs_in_and_refreshes() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let issuer = base.clone();
        let app = Router::new()
            .route("/.well-known/oauth-protected-resource/mcp", get({ let i = issuer.clone(); move || async move { Json(json!({"authorization_servers":[i]})) } }))
            .route("/.well-known/oauth-authorization-server", get({ let i = issuer.clone(); move || async move { Json(json!({"authorization_endpoint":format!("{i}/authorize"),"token_endpoint":format!("{i}/token"),"registration_endpoint":format!("{i}/register")})) } }))
            .route("/register", post(|| async { Json(json!({"client_id":"c-1"})) }))
            .route("/token", post(|body: String| async move {
                let refresh = body.contains("grant_type=refresh_token");
                Json(json!({"access_token": if refresh { "fresh" } else { "first" },"refresh_token":"r-1","expires_in": if refresh { 3600 } else { 0 }}))
            }))
            .route("/authorize", get(|Query(q): Query<std::collections::HashMap<String, String>>| async move { assert_eq!(q["code_challenge_method"], "S256"); "ok" }));
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        let data = tempfile::tempdir().unwrap();
        // The "browser": follow the authorize URL's redirect_uri with a code and the same state.
        let open = |authorize: &str| {
            let url = Url::parse(authorize).unwrap();
            let q = |k: &str| url.query_pairs().find(|(n, _)| n == k).unwrap().1.into_owned();
            let callback = format!("{}?code=abc&state={}", q("redirect_uri"), q("state"));
            std::thread::spawn(move || { std::thread::sleep(Duration::from_millis(100)); let _ = reqwest::blocking_get(&callback); });
        };
        sign_in(data.path(), "srv", &format!("{base}/mcp"), open).await.unwrap();
        assert_eq!(token(data.path(), "srv").await.unwrap().as_deref(), Some("fresh"), "an expired token is refreshed");
        assert_eq!(token(data.path(), "other").await.unwrap(), None);
    }
    mod reqwest { pub fn blocking_get(url: &str) -> std::io::Result<()> {
        use std::io::{Read, Write};
        let u = ::reqwest::Url::parse(url).unwrap();
        let mut s = std::net::TcpStream::connect((u.host_str().unwrap(), u.port().unwrap()))?;
        write!(s, "GET {}?{} HTTP/1.1\r\nHost: x\r\n\r\n", u.path(), u.query().unwrap_or(""))?;
        let mut out = String::new(); let _ = s.read_to_string(&mut out); Ok(())
    } }
}
