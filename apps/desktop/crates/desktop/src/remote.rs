//! Running a chat turn on a remote Hermes gateway (Settings › Gateways ›
//! Remote). Uses the shared JSON-RPC transport (hermes-gateway's client): a
//! one-use ticket, `session.create`, `prompt.submit`, then the session's
//! `message.delta` / `tool.*` / `approval.request` / `message.complete` events.
//! The streamed text and approvals reach the desktop as ordinary engine events,
//! and the finished turn is saved into the local transcript.
use hermes_engine::{Entry, Event, Status, ToolCall, Turn};
use hermes_gateway::client::{ClientMessage, ClientOptions, GatewayClient};
use serde_json::{Value, json};
use std::path::Path;

/// The saved remote gateway, if Settings › Gateways is on Remote with a URL and token.
pub(super) fn configured(root: &Path, state: &super::connections::State) -> Option<(String, String)> {
    let config = &state.config;
    if config.mode == super::connections::Mode::Ssh { return state.tunnel.as_ref().map(|t| (t.url.clone(), t.token.clone())); }
    if config.mode != super::connections::Mode::Remote || config.url.trim().is_empty() { return None; }
    let url = config.url.trim().trim_end_matches('/').to_owned();
    let token = std::fs::read_to_string(root.join("connection-token")).unwrap_or_default().trim().to_owned();
    // An empty token means the gateway's native OAuth sign-in supplies the bearer at turn time.
    if token.is_empty() && !super::gateway_oauth::signed_in(root, &url) { return None; }
    Some((url, token))
}

/// The remote session this local chat continues, remembered beside its transcript.
fn remote_session(dir: &Path, url: &str) -> Option<String> {
    let saved: Value = serde_json::from_slice(&std::fs::read(dir.join("remote-session.json")).ok()?).ok()?;
    (saved["url"].as_str() == Some(url)).then(|| saved["session_id"].as_str().map(str::to_owned)).flatten()
}

pub(super) async fn run_turn(root: std::path::PathBuf, url: String, token: String, dir: std::path::PathBuf, title: String, prompt: String,
    events: async_channel::Sender<Event>, cancel: tokio_util::sync::CancellationToken) -> Result<Turn, String> {
    let token = if token.is_empty() { super::gateway_oauth::access_token(&root, &url).await? } else { token };
    let client = GatewayClient::new(ClientOptions::default());
    client.connect_remote(&url, &token).await.map_err(|e| format!("Could not connect to the remote gateway: {e}"))?;
    let mut stream = client.subscribe();
    let session_id = match remote_session(&dir, &url) {
        Some(id) => id,
        None => {
            let created = client.request("session.create", json!({"source":"gui","title":title})).await.map_err(|e| format!("The remote gateway refused a new session: {e}"))?;
            let id = created["session_id"].as_str().ok_or("The remote gateway returned no session id")?.to_owned();
            let _ = std::fs::write(dir.join("remote-session.json"), json!({"url":url,"session_id":id}).to_string());
            id
        }
    };
    let _ = client.request("session.resume", json!({"session_id":session_id})).await;
    client.request("prompt.submit", json!({"session_id":session_id,"text":prompt})).await.map_err(|e| format!("The remote gateway refused the message: {e}"))?;
    let mut text = String::new();
    let mut entries: Vec<Entry> = Vec::new();
    let status = loop {
        let message = tokio::select! {
            _ = cancel.cancelled() => { let _ = client.request("session.interrupt", json!({"session_id":session_id})).await; break Status::Interrupted; }
            message = stream.recv() => message,
        };
        let Ok(ClientMessage::Event(frame)) = message else { if message.is_err() { break Status::Failed("The remote gateway connection closed".into()); } continue };
        let event = if frame.get("params").is_some() { frame["params"].clone() } else { frame };
        if event["session_id"].as_str().is_some_and(|id| id != session_id) { continue; }
        let payload = &event["payload"];
        match event["type"].as_str().unwrap_or_default() {
            "message.delta" => if let Some(delta) = payload["text"].as_str() { text.push_str(delta); let _ = events.send(Event::Text(delta.to_owned())).await; },
            "tool.start" => {
                if !text.is_empty() { entries.push(Entry::Assistant { text: std::mem::take(&mut text), calls: vec![] }); }
            }
            "tool.complete" => entries.push(Entry::Tool { id: payload["tool_id"].as_str().unwrap_or_default().to_owned(), name: payload["name"].as_str().unwrap_or_default().to_owned(),
                approved: true, result: payload["result_text"].as_str().map(str::to_owned).unwrap_or_else(|| payload["result"].to_string()) }),
            "approval.request" => {
                let request = payload["request_id"].as_str().or(payload["prompt_id"].as_str()).unwrap_or_default().to_owned();
                let call = ToolCall { id: request.clone(), name: payload["tool_name"].as_str().unwrap_or("remote tool").to_owned(),
                    arguments: payload["command"].as_str().map(str::to_owned).unwrap_or_else(|| payload["description"].to_string()) };
                let (answer, decision) = async_channel::bounded(1);
                let _ = events.send(Event::Approval { call, answer }).await;
                let approved = decision.recv().await.unwrap_or(false);
                let _ = client.request("approval.respond", json!({"session_id":session_id,"request_id":request,"prompt_id":request,
                    "execution_generation":event["execution_generation"],"choice": if approved { "once" } else { "deny" }})).await;
            }
            "message.complete" => {
                if let Some(final_text) = payload["text"].as_str() && text.is_empty() { text = final_text.to_owned(); }
                break Status::Complete;
            }
            "subagent.progress" => if let Ok(progress) = serde_json::from_value(payload.clone()) { let _ = events.send(Event::Subagent(progress)).await; },
            "error" | "message.error" => break Status::Failed(payload["message"].as_str().unwrap_or("The remote turn failed").to_owned()),
            _ => {}
        }
    };
    if !text.is_empty() { entries.push(Entry::Assistant { text, calls: vec![] }); }
    client.close();
    Ok(Turn { context: String::new(), attachments: vec![], prompt, model: format!("remote · {url}"), entries, status })
}
