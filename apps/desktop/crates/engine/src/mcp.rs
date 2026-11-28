//! MCP client for an explicitly configured tool server: Streamable HTTP, or a
//! local stdio process (Agent Plugins v1 `"type": "stdio"` servers).
use anyhow::{Context, Result, bail, ensure};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest::{Client, Url, header::ACCEPT};
use serde_json::{Value, json};
use std::{collections::BTreeMap, time::Duration};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

const VERSION: &str = "2025-06-18";
const MAX_BYTES: usize = 2 * 1024 * 1024;

enum Transport {
    Http { client: Client, url: Url, token: Option<String>, headers: Vec<(String, String)>, session: Option<String> },
    Stdio { child: tokio::process::Child, stdin: tokio::process::ChildStdin, lines: tokio::io::Lines<BufReader<tokio::process::ChildStdout>> },
}

/// How to start a stdio server.
pub(crate) struct Stdio { pub command: String, pub args: Vec<String>, pub env: Vec<(String, String)>, pub cwd: std::path::PathBuf }

pub(crate) struct Mcp {
    transport: Transport,
    version: String,
    next_id: u32,
    /// Exposed name → (server-side tool name, definition).
    tools: BTreeMap<String, Value>,
    prefix: String,
}

impl Mcp {
    pub(crate) async fn connect(url:&str,token:Option<String>)->Result<Self>{ Self::connect_http(url, token, vec![], "mcp").await }
    pub(crate) async fn connect_http(url:&str,token:Option<String>,headers:Vec<(String,String)>,prefix:&str)->Result<Self>{
        let url=crate::connector::validated(url)?;
        let client = Client::builder().timeout(Duration::from_secs(45)).redirect(reqwest::redirect::Policy::none()).build()?;
        Self { transport: Transport::Http { client, url, token, headers, session: None }, version: VERSION.into(), next_id: 0, tools: BTreeMap::new(), prefix: prefix.into() }.handshake().await
    }
    pub(crate) async fn connect_stdio(spec: Stdio, prefix: &str) -> Result<Self> {
        let mut command = tokio::process::Command::new(&spec.command);
        command.args(&spec.args).envs(spec.env.iter().map(|(k, v)| (k, v))).current_dir(&spec.cwd)
            .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null()).kill_on_drop(true);
        let mut child = command.spawn().with_context(|| format!("Could not start {}", spec.command))?;
        let stdin = child.stdin.take().context("No stdin")?;
        let lines = BufReader::new(child.stdout.take().context("No stdout")?).lines();
        Self { transport: Transport::Stdio { child, stdin, lines }, version: VERSION.into(), next_id: 0, tools: BTreeMap::new(), prefix: prefix.into() }.handshake().await
    }

    async fn handshake(mut self) -> Result<Self> {
        let hello = tokio::time::timeout(Duration::from_secs(30), self.request("initialize", json!({"protocolVersion":VERSION, "capabilities":{}, "clientInfo":{"name":"hermes-desktop-rust","version":env!("CARGO_PKG_VERSION")}}))).await.context("MCP server did not answer initialize")??;
        let protocol = hello["protocolVersion"].as_str().context("No MCP protocol version")?;
        ensure!([VERSION, "2025-03-26", "2024-11-05"].contains(&protocol), "Unsupported MCP protocol version {protocol}");
        ensure!(hello.pointer("/capabilities/tools").is_some(), "MCP server does not offer tools");
        self.version = protocol.to_owned();
        self.notify(json!({"jsonrpc":"2.0","method":"notifications/initialized"})).await?;
        let mut cursor = None;
        let mut seen_cursors = std::collections::HashSet::<String>::new();
        loop {
            let params = cursor.take().map_or(json!({}), |c: String| json!({"cursor":c}));
            let page = self.request("tools/list", params).await?;
            let tools = page["tools"].as_array().context("Invalid MCP tools/list response")?;
            for tool in tools {
                let name = tool["name"].as_str().context("MCP tool without name")?;
                let exposed = format!("{}__{name}", self.prefix);
                ensure!(!name.is_empty() && exposed.len() <= 64 && name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b,b'_'|b'-')), "MCP tool name cannot be exposed to this model");
                ensure!(tool["inputSchema"].is_object(), "Missing MCP tool input schema");
                ensure!(self.tools.insert(exposed, tool.clone()).is_none(), "Duplicate MCP tool name");
            }
            ensure!(self.tools.len() <= 128, "MCP server exposes more than 128 tools");
            cursor = page.get("nextCursor").and_then(Value::as_str).map(str::to_owned);
            if let Some(c) = &cursor {
                ensure!(seen_cursors.len() < 32 && seen_cursors.insert(c.clone()), "Invalid MCP pagination");
            } else { break; }
        }
        Ok(self)
    }

    fn http(&self, body: Value) -> Option<reqwest::RequestBuilder> {
        let Transport::Http { client, url, token, headers, session } = &self.transport else { return None };
        let mut req = client.post(url.clone()).header(ACCEPT,"application/json, text/event-stream").json(&body);
        if body["method"] != "initialize" { req = req.header("MCP-Protocol-Version", &self.version); }
        for (name, value) in headers { req = req.header(name.as_str(), value.as_str()); }
        if let Some(session) = session { req = req.header("Mcp-Session-Id",session); }
        if let Some(token) = token { req = req.bearer_auth(token); }
        Some(req)
    }

    async fn notify(&mut self, body: Value) -> Result<()> {
        if let Transport::Stdio { stdin, .. } = &mut self.transport {
            stdin.write_all(format!("{body}\n").as_bytes()).await?; stdin.flush().await?; return Ok(());
        }
        let response = self.http(body).context("No transport")?.send().await?;
        ensure!(response.status().is_success(), "MCP initialization notification failed ({})", response.status());
        Ok(())
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        let body = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        let msg: Value = if let Transport::Stdio { stdin, lines, .. } = &mut self.transport {
            stdin.write_all(format!("{body}\n").as_bytes()).await?; stdin.flush().await?;
            loop {
                let line = lines.next_line().await?.context("MCP server exited")?;
                ensure!(line.len() <= MAX_BYTES, "MCP response too large");
                let Ok(value) = serde_json::from_str::<Value>(&line) else { continue };
                if value.get("id") == Some(&json!(id)) && value.get("method").is_none() { break value; }
            }
        } else {
            let response = self.http(body).context("No transport")?.send().await?;
            ensure!(response.status().is_success(), "MCP {method} returned HTTP {}", response.status());
            if method == "initialize" && let Transport::Http { session, .. } = &mut self.transport {
                *session = response.headers().get("Mcp-Session-Id").map(|h| h.to_str().map(str::to_owned)).transpose()?;
            }
            let sse = response.headers().get("Content-Type").and_then(|h| h.to_str().ok()).is_some_and(|v| v.starts_with("text/event-stream"));
            if sse {
                let mut events = response.bytes_stream().eventsource();
                let mut bytes = 0;
                loop {
                    let event = events.next().await.context("MCP stream closed without response")??;
                    bytes += event.data.len();
                    ensure!(bytes <= MAX_BYTES, "MCP response too large");
                    let value: Value = serde_json::from_str(&event.data)?;
                    if value.get("id") == Some(&json!(id)) && value.get("method").is_none() { break value; }
                }
            } else {
                let mut data = Vec::new();
                let mut chunks = response.bytes_stream();
                while let Some(chunk) = chunks.next().await {
                    let chunk = chunk?;
                    ensure!(data.len() + chunk.len() <= MAX_BYTES, "MCP response too large");
                    data.extend_from_slice(&chunk);
                }
                serde_json::from_slice(&data).context("Invalid MCP JSON response")?
            }
        };
        ensure!(msg["jsonrpc"] == "2.0" && msg["id"] == id, "Mismatched MCP response");
        if let Some(error) = msg.get("error") { bail!("MCP {method} failed (code {})", error["code"]); }
        msg.get("result").cloned().context("Missing MCP result")
    }

    pub fn specs(&self) -> Vec<Value> {
        self.tools.iter().map(|(name,tool)| json!({"type":"function","function":{"name":name,"description":tool["description"].as_str().unwrap_or("External MCP tool"),"parameters":tool["inputSchema"]}})).collect()
    }
    pub fn contains(&self, name: &str) -> bool { self.tools.contains_key(name) }

    pub async fn call(&mut self, name: &str, args: &Value) -> Result<Value> {
        let tool = self.tools.get(name).context("Unknown MCP tool")?;
        ensure!(args.is_object(), "MCP arguments must be an object");
        self.request("tools/call", json!({"name":tool["name"],"arguments":args})).await
    }

    pub async fn close(&mut self) {
        match &mut self.transport {
            Transport::Http { client, url, token, session: Some(id), .. } => {
                let mut req = client.delete(url.clone()).header("Mcp-Session-Id",id.as_str()).header("MCP-Protocol-Version",&self.version).timeout(Duration::from_secs(2));
                if let Some(token) = token { req = req.bearer_auth(token); }
                let _ = req.send().await;
            }
            Transport::Stdio { child, .. } => { let _ = child.kill().await; }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, http::{HeaderMap, StatusCode}, response::{IntoResponse, Response}, routing::post};

    async fn endpoint(headers: HeaderMap, Json(msg): Json<Value>) -> Response {
        assert_eq!(headers.get("authorization").unwrap(),"Bearer fixture-token");
        let id = &msg["id"];
        let ok = |value:Value| Json(json!({"jsonrpc":"2.0","id":id,"result":value}));
        if msg["method"] == "initialize" {
            assert_eq!(msg["params"]["protocolVersion"],VERSION);
            return ([("Mcp-Session-Id","session-a")],ok(json!({"protocolVersion":VERSION,"capabilities":{"tools":{}}}))).into_response();
        }
        assert_eq!(headers.get("mcp-session-id").unwrap(),"session-a");
        assert_eq!(headers.get("mcp-protocol-version").unwrap(),VERSION);
        match msg["method"].as_str().unwrap() {
            "notifications/initialized" => StatusCode::ACCEPTED.into_response(),
            "tools/list" => ok(json!({"tools":[{"name":"echo","description":"Echo a value","inputSchema":{"type":"object"}}]})).into_response(),
            "tools/call" => {
                let reply = json!({"jsonrpc":"2.0","id":id,"result":{"content":[{"type":"text","text":msg["params"]["arguments"]["value"]}]}});
                ([("content-type","text/event-stream")], format!("event: message\ndata: {reply}\n\n")).into_response()
            }
            _ => StatusCode::BAD_REQUEST.into_response(),
        }
    }

    #[tokio::test] async fn streamable_http_negotiates_then_executes_discovered_tool() {
        let listener=tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr=listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener,Router::new().route("/mcp",post(endpoint))).await.unwrap(); });
        let mut client=Mcp::connect(&format!("http://{addr}/mcp"),Some("fixture-token".into())).await.unwrap();
        assert!(client.contains("mcp__echo"));
        assert_eq!(client.specs()[0]["function"]["name"],"mcp__echo");
        assert_eq!(client.call("mcp__echo",&json!({"value":"worked"})).await.unwrap()["content"][0]["text"],"worked");
        assert!(client.call("mcp__missing",&json!({})).await.is_err());
    }
}
