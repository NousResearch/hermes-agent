use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::Path;
#[cfg(test)]
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
#[cfg(test)]
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Child;
#[cfg(test)]
use tokio::process::{ChildStdin, Command};
use tokio::sync::{broadcast, mpsc, oneshot, Mutex};
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;
#[cfg(test)]
use tokio_util::codec::{FramedRead, LinesCodec};

use super::attach::attach_gateway;
use super::canonical;
use super::types::{EventParams, JsonRpcEvent, JsonRpcRequest, JsonRpcResponse};
use super::wire;
use crate::config::LaunchConfig;

#[cfg(test)]
const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

struct ServerAsk {
    id: Value,
    method: String,
    question_count: usize,
    answers: Map<String, Value>,
}

/// Latest execution generation per session, and the generation a prompt was opened on.
struct GenerationBook {
    sessions: HashMap<String, i64>,
    prompts: HashMap<String, i64>,
}

impl GenerationBook {
    fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            prompts: HashMap::new(),
        }
    }

    fn note_session(&mut self, session_id: &str, generation: i64) {
        let stale = self
            .sessions
            .get(session_id)
            .is_some_and(|current| *current > generation);
        if !stale {
            self.sessions.insert(session_id.to_string(), generation);
        }
    }

    fn note_prompt(&mut self, prompt_id: &str, generation: i64) {
        self.prompts.insert(prompt_id.to_string(), generation);
    }
}

enum ClarifyStep {
    Hold,
    Reply(Value, Value),
    Legacy,
}

enum Outbound {
    #[cfg(test)]
    Child(ChildStdin),
    Socket(mpsc::Sender<String>),
    Closed,
}

pub struct GatewayClient {
    next_id: AtomicU64,
    outbound: Arc<Mutex<Outbound>>,
    pending_requests: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
    server_requests: Arc<Mutex<HashMap<String, ServerAsk>>>,
    event_tx: broadcast::Sender<JsonRpcEvent>,
    child: Arc<Mutex<Option<Child>>>,
    subscriptions: Arc<Mutex<HashMap<String, String>>>,
    generations: Arc<Mutex<GenerationBook>>,
    rpc_timeout: Duration,
}

impl GatewayClient {
    /// Attach to the profile's shared gateway (`hermes-gateway-v1`).
    pub async fn spawn(cfg: &LaunchConfig) -> Result<Self> {
        let attached = attach_gateway(cfg).await?;
        let mut request = attached
            .url
            .into_client_request()
            .context("gateway websocket url")?;
        let joined = attached.protocols.join(", ");
        request.headers_mut().insert(
            "Sec-WebSocket-Protocol",
            HeaderValue::from_str(&joined).context("gateway subprotocol header")?,
        );
        let (ws, response) = timeout(
            cfg.startup_timeout.max(Duration::from_secs(20)),
            tokio_tungstenite::connect_async(request),
        )
        .await
        .context("timed out dialing the profile gateway")?
        .context("profile gateway websocket failed")?;
        let selected = response
            .headers()
            .get("sec-websocket-protocol")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !selected.split(',').any(|p| p.trim() == "hermes-gateway-v1") {
            anyhow::bail!("gateway selected protocol {selected:?}, expected hermes-gateway-v1");
        }

        let (event_tx, mut ready_rx) = broadcast::channel(4096);
        let pending_requests: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let server_requests = Arc::new(Mutex::new(HashMap::new()));
        let generations = Arc::new(Mutex::new(GenerationBook::new()));
        let (tx, rx) = mpsc::channel(64);
        let pending_clone = pending_requests.clone();
        let server_clone = server_requests.clone();
        let generations_clone = generations.clone();
        let exit_tx = event_tx.clone();
        tokio::spawn(async move {
            pump_socket(
                ws,
                rx,
                pending_clone,
                server_clone,
                generations_clone,
                exit_tx,
            )
            .await;
        });

        let client = Self {
            next_id: AtomicU64::new(1),
            outbound: Arc::new(Mutex::new(Outbound::Socket(tx))),
            pending_requests,
            server_requests,
            event_tx,
            child: Arc::new(Mutex::new(None)),
            subscriptions: Arc::new(Mutex::new(HashMap::new())),
            generations,
            rpc_timeout: cfg.rpc_timeout,
        };
        wait_until_gateway_ready(&mut ready_rx, cfg.startup_timeout)
            .await
            .context("waiting for gateway.ready on the profile gateway")?;
        if let Err(err) = client
            .call("client.capabilities", json!({ "server_requests": true }))
            .await
        {
            tracing::debug!("client.capabilities: {err}");
        }
        Ok(client)
    }

    /// Stdio child used by unit tests. Production `spawn` attaches instead.
    #[cfg(test)]
    pub async fn spawn_stdio(cfg: &LaunchConfig) -> Result<Self> {
        let mut cmd = Command::new(&cfg.python);
        cmd.args(["-m", "tui_gateway.entry"])
            .current_dir(&cfg.src_root)
            .env("HERMES_HOME", &cfg.hermes_home)
            .env("HERMES_PYTHON_SRC_ROOT", &cfg.src_root)
            .env("PYTHONUNBUFFERED", "1")
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let pythonpath = match std::env::var("PYTHONPATH") {
            Ok(existing) if !existing.is_empty() => {
                format!("{}{}{}", cfg.src_root.display(), path_sep(), existing)
            }
            _ => cfg.src_root.display().to_string(),
        };
        cmd.env("PYTHONPATH", pythonpath);

        let mut child = cmd.spawn().with_context(|| {
            format!(
                "Failed to spawn `{} -m tui_gateway.entry` (cwd {})",
                cfg.python.display(),
                cfg.src_root.display()
            )
        })?;

        let stdin = child.stdin.take().context("Failed to open child stdin")?;
        let stdout = child.stdout.take().context("Failed to open child stdout")?;
        let stderr = child.stderr.take().context("Failed to open child stderr")?;
        let outbound = Arc::new(Mutex::new(Outbound::Child(stdin)));

        let (event_tx, _) = broadcast::channel(4096);
        let mut ready_rx = event_tx.subscribe();
        let pending_requests: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let server_requests = Arc::new(Mutex::new(HashMap::new()));
        let generations = Arc::new(Mutex::new(GenerationBook::new()));

        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(target: "tui_gateway", "{}", redact_gateway_line(&line));
            }
        });

        let pending_clone = pending_requests.clone();
        let server_clone = server_requests.clone();
        let generations_clone = generations.clone();
        let event_tx_clone = event_tx.clone();
        let outbound_reader = outbound.clone();

        tokio::spawn(async move {
            let codec = LinesCodec::new_with_max_length(MAX_LINE_BYTES);
            let mut lines = FramedRead::new(stdout, codec);
            while let Some(item) = lines.next().await {
                let line = match item {
                    Ok(l) => l,
                    Err(e) => {
                        tracing::warn!("gateway stdout codec error: {e}");
                        continue;
                    }
                };
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Value>(&line) {
                    Ok(val) => {
                        if let Some(reply) = dispatch_frame(
                            val,
                            &pending_clone,
                            &server_clone,
                            &generations_clone,
                            &event_tx_clone,
                        )
                        .await
                        {
                            write_child(&outbound_reader, &reply).await;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to parse gateway line: {e} (line: {})",
                            redact_gateway_line(&line)
                        );
                    }
                }
            }
        });

        let child = Arc::new(Mutex::new(Some(child)));
        let watch = child.clone();
        let exit_tx = event_tx.clone();
        tokio::spawn(async move {
            loop {
                let dead = {
                    let mut slot = watch.lock().await;
                    let Some(ch) = slot.as_mut() else {
                        return;
                    };
                    match ch.try_wait() {
                        Ok(Some(st)) => Some(st.code()),
                        Ok(None) => None,
                        Err(_) => return,
                    }
                };
                if let Some(code) = dead {
                    let _ = exit_tx.send(JsonRpcEvent::synthetic(
                        "gateway.exit",
                        json!({ "code": code }),
                    ));
                    return;
                }
                tokio::time::sleep(Duration::from_millis(400)).await;
            }
        });

        wait_until_gateway_ready(&mut ready_rx, cfg.startup_timeout)
            .await
            .with_context(|| {
                format!(
                    "waiting for gateway.ready (python={}, cwd={})",
                    cfg.python.display(),
                    cfg.src_root.display()
                )
            })?;

        Ok(Self {
            next_id: AtomicU64::new(1),
            outbound,
            pending_requests,
            server_requests,
            event_tx,
            child,
            subscriptions: Arc::new(Mutex::new(HashMap::new())),
            generations,
            rpc_timeout: cfg.rpc_timeout,
        })
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<JsonRpcEvent> {
        self.event_tx.subscribe()
    }

    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        self.call_timeout(method, params, self.rpc_timeout).await
    }

    pub async fn call_timeout(&self, method: &str, params: Value, dur: Duration) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let req = JsonRpcRequest::new(id, method, params);
        let req_str = serde_json::to_string(&req)?;

        let (tx, rx) = oneshot::channel();
        {
            let mut map = self.pending_requests.lock().await;
            map.insert(id, tx);
        }

        let write_res = self.write_frame(&req_str).await;
        if let Err(e) = write_res {
            self.pending_requests.lock().await.remove(&id);
            return Err(e).context("writing JSON-RPC request to gateway");
        }

        match timeout(dur, rx).await {
            Ok(Ok(resp)) => {
                if let Some(err) = resp.error {
                    anyhow::bail!("{method} error: {} ({})", err.message, err.code);
                }
                resp.result.context(format!("empty {method} result"))
            }
            Ok(Err(_)) => {
                self.pending_requests.lock().await.remove(&id);
                anyhow::bail!("{method}: gateway dropped the response channel");
            }
            Err(_) => {
                self.pending_requests.lock().await.remove(&id);
                anyhow::bail!("{method}: timed out after {}ms", dur.as_millis());
            }
        }
    }

    async fn write_frame(&self, body: &str) -> Result<()> {
        let mut outbound = self.outbound.lock().await;
        match &mut *outbound {
            #[cfg(test)]
            Outbound::Child(stdin) => {
                stdin
                    .write_all(body.as_bytes())
                    .await
                    .context("gateway stdin")?;
                stdin.write_all(b"\n").await.context("gateway stdin")?;
                stdin.flush().await.context("gateway stdin")?;
            }
            Outbound::Socket(tx) => {
                tx.send(body.to_string())
                    .await
                    .context("gateway socket closed")?;
            }
            Outbound::Closed => anyhow::bail!("gateway connection is closed"),
        }
        Ok(())
    }

    async fn respond_server(&self, id: &Value, result: Value) -> Result<Value> {
        let frame = json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        });
        let text = serde_json::to_string(&frame)?;
        self.write_frame(&text).await?;
        frame.get("result").cloned().context("server response")
    }

    async fn remember_snapshot(&self, snapshot: &Value) {
        let sid = if let Some((sid, sub)) = canonical::remember_fields(snapshot) {
            self.subscriptions.lock().await.insert(sid.clone(), sub);
            Some(sid)
        } else {
            snapshot
                .get("session_id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
        };
        if let Some(sid) = sid {
            if let Some(gen) = canonical::generation_of(snapshot) {
                self.generations.lock().await.note_session(&sid, gen);
            }
        }
    }

    async fn generation(&self, session_id: &str) -> Result<i64> {
        self.generations
            .lock()
            .await
            .sessions
            .get(session_id)
            .copied()
            .with_context(|| format!("session {session_id} has no execution generation"))
    }

    async fn prompt_generation(&self, session_id: &str, prompt_id: &str) -> Result<i64> {
        let book = self.generations.lock().await;
        if let Some(gen) = book.prompts.get(prompt_id) {
            return Ok(*gen);
        }
        book.sessions
            .get(session_id)
            .copied()
            .with_context(|| format!("prompt {prompt_id} has no execution generation"))
    }

    async fn resume_for_cas(&self, session_id: &str) -> Result<Value> {
        let snap = self
            .call(
                "session.resume",
                json!({
                    "session_id": session_id,
                    "display_projection": true,
                }),
            )
            .await?;
        self.remember_snapshot(&snap).await;
        Ok(snap)
    }

    async fn mutate(
        &self,
        session_id: &str,
        snapshot: &Value,
        operation: &str,
        payload: Value,
        with_generation: bool,
        budget: Duration,
    ) -> Result<Value> {
        let params = canonical::mutate_params(
            session_id,
            &fresh_request_id(),
            snapshot,
            operation,
            payload,
            with_generation,
        )
        .with_context(|| format!("session {session_id} has no revision for {operation}"))?;
        self.call_timeout("session.mutate", params, budget).await
    }

    async fn detach_subscription(&self, session_id: &str, end_reason: Option<&str>) -> Result<()> {
        let subscription_id = self.subscriptions.lock().await.get(session_id).cloned();
        let Some(subscription_id) = subscription_id else {
            return Ok(());
        };
        let mut params = json!({
            "session_id": session_id,
            "subscription_id": subscription_id,
        });
        if let Some(reason) = end_reason {
            params["end_reason"] = json!(reason);
        }
        self.call("session.detach", params).await?;
        self.subscriptions.lock().await.remove(session_id);
        Ok(())
    }

    pub async fn create_session(&self, title: &str, cwd: &Path) -> Result<String> {
        let params = json!({
            "title": title,
            "cwd": cwd.display().to_string(),
            "source": "tui",
            "client": {
                "name": "hermes-tui-native",
                "version": env!("CARGO_PKG_VERSION"),
            },
        });
        let res = self.call("session.create", params).await?;
        self.remember_snapshot(&res).await;
        res.get("session_id")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .context("Missing session_id")
    }

    pub async fn submit_prompt(&self, session_id: &str, text: &str) -> Result<()> {
        let params = json!({
            "session_id": session_id,
            "text": text,
            "submission_id": fresh_request_id(),
        });
        let res = self.call("prompt.submit", params).await?;
        if let Some(gen) = canonical::generation_of(&res) {
            self.generations.lock().await.note_session(session_id, gen);
        }
        Ok(())
    }

    pub async fn prompt_background(&self, session_id: &str, text: &str) -> Result<Value> {
        self.call(
            "prompt.background",
            json!({ "session_id": session_id, "text": text }),
        )
        .await
    }

    pub async fn interrupt(&self, session_id: &str) -> Result<()> {
        let generation = self.generation(session_id).await?;
        self.call(
            "session.interrupt",
            json!({
                "session_id": session_id,
                "execution_generation": generation,
            }),
        )
        .await?;
        Ok(())
    }

    pub async fn steer(&self, session_id: &str, text: &str) -> Result<String> {
        let generation = self.generation(session_id).await?;
        let res = self
            .call(
                "session.steer",
                json!({
                    "session_id": session_id,
                    "text": text,
                    "execution_generation": generation,
                }),
            )
            .await?;
        Ok(res
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("ok")
            .to_string())
    }

    pub async fn close_session(&self, session_id: &str) -> Result<()> {
        if !self.subscriptions.lock().await.contains_key(session_id) {
            let _ = self.resume_for_cas(session_id).await;
        }
        let _ = self
            .detach_subscription(session_id, Some("viewer_close"))
            .await;
        Ok(())
    }

    pub async fn compress(&self, session_id: &str) -> Result<Value> {
        let snap = self.resume_for_cas(session_id).await?;
        self.mutate(
            session_id,
            &snap,
            "compress",
            json!({}),
            true,
            Duration::from_secs(660),
        )
        .await
    }

    pub async fn context_breakdown(&self, session_id: &str) -> Result<Value> {
        self.call(
            "session.context_breakdown",
            json!({ "session_id": session_id }),
        )
        .await
    }

    pub async fn set_config(&self, params: Value) -> Result<Value> {
        let key = params.get("key").and_then(|v| v.as_str());
        let session_id = params
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        if key == Some("model") {
            if let Some(session_id) = session_id {
                let model = params
                    .get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let snap = self.resume_for_cas(&session_id).await?;
                return self
                    .mutate(
                        &session_id,
                        &snap,
                        "model",
                        json!({ "model": model }),
                        true,
                        self.rpc_timeout,
                    )
                    .await;
            }
        }
        self.call("config.set", params).await
    }

    pub async fn session_title(&self, session_id: &str, title: Option<&str>) -> Result<Value> {
        let Some(title) = title else {
            let info = self
                .call("session.info", json!({ "session_id": session_id }))
                .await?;
            return Ok(canonical::title_view(&info, ""));
        };
        let snap = self.resume_for_cas(session_id).await?;
        let result = self
            .mutate(
                session_id,
                &snap,
                "rename",
                json!({ "title": title }),
                false,
                self.rpc_timeout,
            )
            .await?;
        Ok(canonical::title_view(&result, title))
    }

    pub async fn session_status(&self, session_id: &str) -> Result<Value> {
        self.call("session.status", json!({ "session_id": session_id }))
            .await
    }

    pub async fn get_config(&self, key: &str, session_id: Option<&str>) -> Result<Value> {
        let mut params = json!({ "key": key });
        if let Some(sid) = session_id {
            params["session_id"] = json!(sid);
        }
        self.call("config.get", params).await
    }

    pub async fn set_cwd(&self, session_id: &str, cwd: &str) -> Result<Value> {
        let snap = self.resume_for_cas(session_id).await?;
        self.mutate(
            session_id,
            &snap,
            "workspace",
            json!({ "cwd": cwd }),
            true,
            self.rpc_timeout,
        )
        .await
    }

    pub async fn slash_exec(&self, session_id: &str, command: &str) -> Result<Value> {
        let params = json!({ "session_id": session_id, "command": command });
        self.call("slash.exec", params).await
    }

    pub async fn command_dispatch(&self, session_id: &str, name: &str, arg: &str) -> Result<Value> {
        self.call(
            "command.dispatch",
            json!({
                "session_id": session_id,
                "name": name.trim_start_matches('/'),
                "arg": arg,
            }),
        )
        .await
    }

    pub async fn image_attach_bytes(
        &self,
        session_id: &str,
        content_base64: &str,
        filename: Option<&str>,
    ) -> Result<Value> {
        let mut params = json!({
            "session_id": session_id,
            "content_base64": content_base64,
        });
        if let Some(name) = filename {
            params["filename"] = json!(name);
        }
        self.call("image.attach_bytes", params).await
    }

    pub async fn projects_project_sessions(&self, project_id: &str) -> Result<Value> {
        self.call(
            "projects.project_sessions",
            json!({ "project_id": project_id }),
        )
        .await
    }

    pub async fn list_sessions(&self) -> Result<Vec<Value>> {
        let mut params = json!({ "scope": "stored", "limit": 50, "picker": true });
        let mut out = Vec::new();
        let mut fell_back = false;
        loop {
            let res = match self.call("session.list", params.clone()).await {
                Ok(v) => v,
                Err(err) if !fell_back && out.is_empty() && list_params_rejected(&err) => {
                    fell_back = true;
                    params = json!({ "limit": 50 });
                    continue;
                }
                Err(err) => return Err(err),
            };
            if let Some(rows) = res.get("sessions").and_then(|s| s.as_array()) {
                out.extend(rows.iter().cloned());
            }
            let cursor = res
                .get("next_cursor")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());
            match cursor {
                Some(cursor) if !fell_back => {
                    params = json!({
                        "scope": "stored",
                        "limit": 50,
                        "picker": true,
                        "cursor": cursor,
                    });
                }
                _ => break,
            }
            if out.len() > 5_000 {
                break;
            }
        }
        Ok(out)
    }

    pub async fn list_skills(&self) -> Result<Value> {
        self.call("skills.manage", json!({ "action": "list" }))
            .await
    }

    pub async fn skills_manage(&self, action: &str, query: &str) -> Result<Value> {
        self.call("skills.manage", json!({ "action": action, "query": query }))
            .await
    }

    pub async fn handoff_request(&self, session_id: &str, platform: &str) -> Result<Value> {
        self.call(
            "handoff.request",
            json!({ "session_id": session_id, "platform": platform }),
        )
        .await
    }

    pub async fn handoff_state(&self, session_id: &str) -> Result<Value> {
        self.call("handoff.state", json!({ "session_id": session_id }))
            .await
    }

    pub async fn handoff_fail(&self, session_id: &str, error: &str) -> Result<Value> {
        self.call(
            "handoff.fail",
            json!({ "session_id": session_id, "error": error }),
        )
        .await
    }

    pub async fn profiles_create(&self, name: &str, clone_from: Option<&str>) -> Result<Value> {
        let mut params = json!({ "name": name, "mirror_credentials": true });
        if let Some(src) = clone_from {
            params["clone_from"] = json!(src);
        }
        self.call("profiles.create", params).await
    }

    pub async fn llm_oneshot(
        &self,
        session_id: &str,
        instructions: &str,
        input: &str,
    ) -> Result<Value> {
        self.call(
            "llm.oneshot",
            json!({
                "session_id": session_id,
                "instructions": instructions,
                "input": input,
                "task": "title_generation",
                "max_tokens": 256,
            }),
        )
        .await
    }

    pub async fn commands_catalog(&self) -> Result<Value> {
        self.call("commands.catalog", json!({})).await
    }

    pub async fn list_profiles(&self) -> Result<Value> {
        self.call("profiles.list", json!({ "include_sessions": true }))
            .await
    }

    pub async fn list_agents(&self) -> Result<Value> {
        Ok(json!({ "processes": [] }))
    }

    pub async fn process_list(&self, session_id: &str) -> Result<Value> {
        self.call("process.list", json!({ "session_id": session_id }))
            .await
    }

    pub async fn process_kill(&self, session_id: &str, process_id: &str) -> Result<Value> {
        self.call(
            "process.kill",
            json!({ "session_id": session_id, "process_id": process_id }),
        )
        .await
    }

    pub async fn process_stop(&self, session_id: &str) -> Result<Value> {
        self.call("process.stop", json!({ "session_id": session_id }))
            .await
    }

    pub async fn delegation_status(&self) -> Result<Value> {
        self.call("delegation.status", json!({})).await
    }

    pub async fn delegation_pause(&self, paused: bool) -> Result<Value> {
        self.call("delegation.pause", json!({ "paused": paused }))
            .await
    }

    pub async fn interrupt_subagent(&self, session_id: &str, subagent_id: &str) -> Result<Value> {
        self.call(
            "subagent.interrupt",
            json!({ "session_id": session_id, "subagent_id": subagent_id }),
        )
        .await
    }

    pub async fn steer_subagent(
        &self,
        session_id: &str,
        subagent_id: &str,
        text: &str,
    ) -> Result<Value> {
        self.call(
            "subagent.steer",
            json!({
                "session_id": session_id,
                "subagent_id": subagent_id,
                "text": text,
            }),
        )
        .await
    }

    pub async fn learning_frames(&self) -> Result<Value> {
        self.call(
            "learning.frames",
            json!({ "frames": 2, "cols": 80, "rows": 24 }),
        )
        .await
    }

    pub async fn complete_slash(&self, text: &str) -> Result<Value> {
        self.call("complete.slash", json!({ "text": text })).await
    }

    pub async fn paste_collapse(&self, text: &str) -> Result<Value> {
        self.call("paste.collapse", json!({ "text": text })).await
    }

    pub async fn input_detect_drop(&self, session_id: &str, text: &str) -> Result<Value> {
        self.call(
            "input.detect_drop",
            json!({ "session_id": session_id, "text": text }),
        )
        .await
    }

    pub async fn session_active_list(&self, current_session_id: &str) -> Result<Value> {
        self.call(
            "session.active_list",
            json!({ "current_session_id": current_session_id }),
        )
        .await
    }

    pub async fn session_activate(&self, session_id: &str) -> Result<Value> {
        self.call("session.activate", json!({ "session_id": session_id }))
            .await
    }

    pub async fn complete_path(&self, word: &str, session_id: &str, cwd: &str) -> Result<Value> {
        self.call(
            "complete.path",
            json!({
                "word": word,
                "session_id": session_id,
                "cwd": cwd,
            }),
        )
        .await
    }

    pub async fn rollback_list(&self, session_id: &str) -> Result<Value> {
        self.call("rollback.list", json!({ "session_id": session_id }))
            .await
    }

    pub async fn rollback_diff(&self, session_id: &str, hash: &str) -> Result<Value> {
        self.call(
            "rollback.diff",
            json!({ "session_id": session_id, "hash": hash }),
        )
        .await
    }

    pub async fn rollback_restore(
        &self,
        session_id: &str,
        hash: &str,
        file_path: Option<&str>,
    ) -> Result<Value> {
        let mut params = json!({ "session_id": session_id, "hash": hash });
        if let Some(fp) = file_path {
            params["file_path"] = json!(fp);
        }
        self.call("rollback.restore", params).await
    }

    pub async fn image_attach(&self, session_id: &str, path: &str) -> Result<Value> {
        self.call(
            "image.attach",
            json!({ "session_id": session_id, "path": path }),
        )
        .await
    }

    pub async fn model_options(&self, session_id: &str) -> Result<Value> {
        self.call(
            "model.options",
            json!({
                "session_id": session_id,
                "include_unconfigured": true,
            }),
        )
        .await
    }

    pub async fn model_save_key(
        &self,
        session_id: &str,
        slug: &str,
        api_key: &str,
    ) -> Result<Value> {
        self.call(
            "model.save_key",
            json!({
                "session_id": session_id,
                "slug": slug,
                "api_key": api_key,
            }),
        )
        .await
    }

    pub async fn resume_session(&self, session_id: &str) -> Result<Value> {
        let snap = self.resume_for_cas(session_id).await?;
        self.replay_open(&snap).await;
        Ok(snap)
    }

    /// Re-show prompts and server requests a reconnecting viewer still has to answer.
    async fn replay_open(&self, snap: &Value) {
        let session_id = snap
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        if let Some(rows) = snap.get("open_requests").and_then(|v| v.as_array()) {
            for row in rows {
                let Some(id) = row.get("id").cloned() else {
                    continue;
                };
                let Some(method) = row.get("method").and_then(|m| m.as_str()) else {
                    continue;
                };
                let params = row.get("params").cloned().unwrap_or(Value::Null);
                let key = wire::id_key(&id);
                let Some((event, payload)) = wire::server_event(method, &key, &params) else {
                    continue;
                };
                let question_count = params
                    .get("questions")
                    .and_then(|q| q.as_array())
                    .map(|items| items.len())
                    .filter(|n| *n > 0)
                    .unwrap_or(1);
                self.server_requests.lock().await.insert(
                    key,
                    ServerAsk {
                        id,
                        method: method.to_string(),
                        question_count,
                        answers: Map::new(),
                    },
                );
                let _ = self.event_tx.send(JsonRpcEvent {
                    jsonrpc: "2.0".into(),
                    method: "event".into(),
                    params: EventParams {
                        event_type: event.into(),
                        session_id: session_id.clone(),
                        payload: Some(payload),
                    },
                });
            }
        }
        let Some(prompts) = snap.get("prompts").and_then(|v| v.as_array()) else {
            return;
        };
        for prompt in prompts {
            let event = match prompt.get("kind").and_then(|k| k.as_str()) {
                Some("approval") => "approval.request",
                Some("clarify") => "clarify.request",
                _ => continue,
            };
            if let Some(gen) = canonical::generation_of(prompt) {
                if let Some(sid) = session_id.as_deref() {
                    self.generations.lock().await.note_session(sid, gen);
                }
                if let Some(pid) = prompt.get("prompt_id").and_then(|v| v.as_str()) {
                    self.generations.lock().await.note_prompt(pid, gen);
                }
            }
            let mut payload = prompt.clone();
            if let Some(obj) = payload.as_object_mut() {
                if let Some(pid) = prompt.get("prompt_id").cloned() {
                    obj.entry("request_id").or_insert(pid);
                }
            }
            let _ = self.event_tx.send(JsonRpcEvent {
                jsonrpc: "2.0".into(),
                method: "event".into(),
                params: EventParams {
                    event_type: event.into(),
                    session_id: session_id.clone(),
                    payload: Some(payload),
                },
            });
        }
    }

    pub async fn clarify_respond(
        &self,
        session_id: &str,
        request_id: &str,
        answer: &str,
        question_id: Option<&str>,
    ) -> Result<Value> {
        let step = {
            let mut map = self.server_requests.lock().await;
            match map.get_mut(request_id) {
                Some(ask) if ask.method == "clarify" => {
                    if answer.is_empty() {
                        let id = ask.id.clone();
                        map.remove(request_id);
                        ClarifyStep::Reply(id, json!({ "outcome": "cancelled" }))
                    } else {
                        let qid = question_id.unwrap_or("answer").to_string();
                        ask.answers.insert(qid, Value::String(answer.to_string()));
                        let done = question_id.is_none() || ask.answers.len() >= ask.question_count;
                        if done {
                            let id = ask.id.clone();
                            let answers = Value::Object(ask.answers.clone());
                            map.remove(request_id);
                            ClarifyStep::Reply(
                                id,
                                json!({ "answers": answers, "outcome": "submitted" }),
                            )
                        } else {
                            ClarifyStep::Hold
                        }
                    }
                }
                _ => ClarifyStep::Legacy,
            }
        };
        match step {
            ClarifyStep::Hold => Ok(json!({ "remaining": ["pending"] })),
            ClarifyStep::Reply(id, result) => self.respond_server(&id, result).await,
            ClarifyStep::Legacy => {
                if request_id.is_empty() {
                    anyhow::bail!("clarify needs a prompt id");
                }
                let generation = self.prompt_generation(session_id, request_id).await?;
                self.call(
                    "clarify.respond",
                    json!({
                        "session_id": session_id,
                        "execution_generation": generation,
                        "prompt_id": request_id,
                        "answer": answer,
                    }),
                )
                .await
            }
        }
    }

    pub async fn secret_respond(
        &self,
        method: &str,
        session_id: &str,
        request_id: &str,
        key: &str,
        value: &str,
    ) -> Result<Value> {
        let want = match method {
            "sudo.respond" => "sudo",
            "secret.respond" => "secret",
            other => other,
        };
        let rpc_id = {
            let mut map = self.server_requests.lock().await;
            match map.get(request_id) {
                Some(ask) if ask.method == want => {
                    let id = ask.id.clone();
                    map.remove(request_id);
                    Some(id)
                }
                _ => None,
            }
        };
        if let Some(id) = rpc_id {
            return self.respond_server(&id, json!({ "value": value })).await;
        }
        let mut params = json!({
            "session_id": session_id,
            "request_id": request_id,
        });
        params[key] = json!(value);
        self.call(method, params).await
    }

    pub async fn approval_respond(
        &self,
        session_id: &str,
        choice: &str,
        request_id: Option<&str>,
    ) -> Result<Value> {
        if let Some(rid) = request_id {
            let rpc_id = {
                let mut map = self.server_requests.lock().await;
                match map.get(rid) {
                    Some(ask) if ask.method == "approval" => {
                        let id = ask.id.clone();
                        map.remove(rid);
                        Some(id)
                    }
                    _ => None,
                }
            };
            if let Some(id) = rpc_id {
                return self.respond_server(&id, json!({ "choice": choice })).await;
            }
        }
        let Some(prompt_id) = request_id.filter(|id| !id.is_empty()) else {
            anyhow::bail!("approval needs a prompt id");
        };
        let generation = self.prompt_generation(session_id, prompt_id).await?;
        self.call(
            "approval.respond",
            json!({
                "session_id": session_id,
                "execution_generation": generation,
                "prompt_id": prompt_id,
                "choice": choice,
            }),
        )
        .await
    }

    pub async fn mcp_servers_list(&self) -> Result<Value> {
        self.call("mcp.servers.list", json!({})).await
    }

    pub async fn mcp_catalog(&self) -> Result<Value> {
        self.call("mcp.catalog", json!({})).await
    }

    pub async fn mcp_servers_add(&self, name: &str, preset: Option<&str>) -> Result<Value> {
        let mut params = json!({ "name": name });
        if let Some(p) = preset {
            params["preset"] = json!(p);
        }
        self.call("mcp.servers.add", params).await
    }

    pub async fn mcp_servers_remove(&self, name: &str) -> Result<Value> {
        self.call("mcp.servers.remove", json!({ "name": name }))
            .await
    }

    pub async fn mcp_servers_test(&self, name: &str) -> Result<Value> {
        self.call("mcp.servers.test", json!({ "name": name })).await
    }

    pub async fn tools_show(&self, session_id: &str) -> Result<Value> {
        self.call("tools.show", json!({ "session_id": session_id }))
            .await
    }

    pub async fn learning_detail(&self, id: &str) -> Result<Value> {
        self.call("learning.detail", json!({ "id": id })).await
    }

    pub async fn learning_delete(&self, id: &str) -> Result<Value> {
        self.call("learning.delete", json!({ "id": id })).await
    }

    pub async fn image_detach(&self, session_id: &str, path: &str) -> Result<Value> {
        self.call(
            "image.detach",
            json!({ "session_id": session_id, "path": path }),
        )
        .await
    }

    pub async fn profiles_describe(&self, name: &str) -> Result<Value> {
        self.call("profiles.describe", json!({ "name": name }))
            .await
    }

    pub async fn spawn_tree_save(
        &self,
        session_id: &str,
        label: &str,
        subagents: Vec<Value>,
    ) -> Result<Value> {
        self.call(
            "spawn_tree.save",
            json!({
                "session_id": session_id,
                "label": label,
                "subagents": subagents,
            }),
        )
        .await
    }

    pub async fn reload_mcp(&self, session_id: &str, confirm: bool) -> Result<Value> {
        self.call(
            "reload.mcp",
            json!({ "session_id": session_id, "confirm": confirm }),
        )
        .await
    }

    pub async fn session_branch(&self, session_id: &str, name: &str) -> Result<Value> {
        let snap = self.resume_for_cas(session_id).await?;
        let payload = if name.is_empty() {
            json!({})
        } else {
            json!({ "title": name })
        };
        let result = self
            .mutate(session_id, &snap, "branch", payload, true, self.rpc_timeout)
            .await?;
        let branched = result
            .get("branched_session_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .context("session.mutate branch omitted branched_session_id")?
            .to_string();
        let child = self.resume_for_cas(&branched).await.ok();
        Ok(canonical::branch_view(name, &branched, child.as_ref()))
    }

    pub async fn session_undo(&self, session_id: &str) -> Result<Value> {
        let snap = self.resume_for_cas(session_id).await?;
        let Some(row) = canonical::last_user_row_id(&snap) else {
            return Ok(json!({ "removed": 0, "rewound_count": 0 }));
        };
        let result = self
            .mutate(
                session_id,
                &snap,
                "rewind",
                json!({ "target_message_id": row }),
                true,
                self.rpc_timeout,
            )
            .await?;
        Ok(canonical::undo_view(&result))
    }

    pub async fn session_save(&self, session_id: &str) -> Result<Value> {
        self.call("session.save", json!({ "session_id": session_id }))
            .await
    }

    pub async fn tools_list(&self, session_id: &str) -> Result<Value> {
        self.call("tools.list", json!({ "session_id": session_id }))
            .await
    }

    pub async fn plugins_list(&self) -> Result<Value> {
        self.call("plugins.list", json!({})).await
    }

    pub async fn browser_manage(
        &self,
        action: &str,
        session_id: Option<&str>,
        url: Option<&str>,
    ) -> Result<Value> {
        let mut params = json!({ "action": action });
        if let Some(sid) = session_id {
            params["session_id"] = json!(sid);
        }
        if let Some(url) = url {
            params["url"] = json!(url);
        }
        self.call("browser.manage", params).await
    }

    pub async fn clipboard_paste(&self, session_id: &str) -> Result<Value> {
        self.call("clipboard.paste", json!({ "session_id": session_id }))
            .await
    }

    pub async fn reload_env(&self, session_id: &str) -> Result<Value> {
        self.call("reload.env", json!({ "session_id": session_id }))
            .await
    }

    pub async fn skills_reload(&self) -> Result<Value> {
        self.call("skills.reload", json!({})).await
    }

    pub async fn system_battery(&self) -> Result<Value> {
        self.call("system.battery", json!({})).await
    }

    pub async fn cron_manage(
        &self,
        action: &str,
        name: Option<&str>,
        include_disabled: bool,
    ) -> Result<Value> {
        let mut params = json!({
            "action": action,
            "include_disabled": include_disabled,
        });
        if let Some(n) = name {
            params["name"] = json!(n);
        }
        self.call("cron.manage", params).await
    }

    pub async fn setup_status(&self) -> Result<Value> {
        self.call("setup.status", json!({})).await
    }

    pub async fn setup_runtime_check(&self) -> Result<Value> {
        self.call("setup.runtime_check", json!({})).await
    }

    pub async fn config_show(&self) -> Result<Value> {
        self.call("config.show", json!({})).await
    }

    pub async fn project_facts(&self, cwd: &str) -> Result<Value> {
        self.call("project.facts", json!({ "cwd": cwd })).await
    }

    pub async fn verification_status(&self, session_id: &str, cwd: &str) -> Result<Value> {
        self.call(
            "verification.status",
            json!({ "session_id": session_id, "cwd": cwd }),
        )
        .await
    }

    pub async fn spawn_tree_list(&self, session_id: &str) -> Result<Value> {
        self.call(
            "spawn_tree.list",
            json!({ "session_id": session_id, "limit": 30 }),
        )
        .await
    }

    pub async fn file_attach(&self, session_id: &str, path: &str) -> Result<Value> {
        self.call(
            "file.attach",
            json!({ "session_id": session_id, "path": path }),
        )
        .await
    }

    pub async fn pdf_attach(&self, session_id: &str, path: &str) -> Result<Value> {
        self.call(
            "pdf.attach",
            json!({ "session_id": session_id, "path": path }),
        )
        .await
    }

    pub async fn session_delete(&self, session_id: &str) -> Result<Value> {
        let snap = self.resume_for_cas(session_id).await?;
        let result = self
            .mutate(
                session_id,
                &snap,
                "delete",
                json!({}),
                true,
                self.rpc_timeout,
            )
            .await?;
        let _ = self.detach_subscription(session_id, None).await;
        Ok(result)
    }

    pub async fn spawn_tree_load(&self, path: &str) -> Result<Value> {
        self.call("spawn_tree.load", json!({ "path": path })).await
    }

    pub async fn tools_configure(
        &self,
        session_id: &str,
        action: &str,
        names: &[String],
    ) -> Result<Value> {
        self.call(
            "tools.configure",
            json!({
                "session_id": session_id,
                "action": action,
                "names": names,
            }),
        )
        .await
    }

    pub async fn plugins_manage(&self, action: &str, extra: Value) -> Result<Value> {
        let mut params = if extra.is_object() { extra } else { json!({}) };
        params["action"] = json!(action);
        self.call("plugins.manage", params).await
    }

    pub async fn model_disconnect(&self, slug: &str) -> Result<Value> {
        self.call("model.disconnect", json!({ "slug": slug })).await
    }

    pub async fn session_set_hidden(&self, session_id: &str, hidden: bool) -> Result<Value> {
        let snap = self.resume_for_cas(session_id).await?;
        self.mutate(
            session_id,
            &snap,
            "sidebar",
            json!({ "hidden": hidden }),
            false,
            self.rpc_timeout,
        )
        .await
    }

    pub async fn message_react(
        &self,
        session_id: &str,
        newest_role: &str,
        emoji: Option<&str>,
    ) -> Result<Value> {
        let mut params = json!({
            "session_id": session_id,
            "newest_role": newest_role,
            "author": "user",
        });
        match emoji {
            Some(e) => params["emoji"] = json!(e),
            None => params["emoji"] = Value::Null,
        }
        self.call("message.react", params).await
    }

    pub async fn image_generate(&self, prompt: Option<&str>, probe: bool) -> Result<Value> {
        let mut params = json!({ "probe": probe });
        if let Some(p) = prompt {
            params["prompt"] = json!(p);
            params["aspect_ratio"] = json!("square");
        }
        self.call("image.generate", params).await
    }

    pub async fn terminal_resize(&self, session_id: &str, cols: u16) -> Result<Value> {
        self.call(
            "terminal.resize",
            json!({ "session_id": session_id, "cols": cols }),
        )
        .await
    }

    pub async fn session_most_recent(&self) -> Result<Value> {
        self.call("session.most_recent", json!({})).await
    }

    pub async fn session_history(&self, session_id: &str) -> Result<Value> {
        self.call("session.history", json!({ "session_id": session_id }))
            .await
    }

    pub async fn session_redirect(&self, session_id: &str, text: &str) -> Result<Value> {
        let generation = self.generation(session_id).await?;
        self.call(
            "session.redirect",
            json!({
                "session_id": session_id,
                "text": text,
                "execution_generation": generation,
            }),
        )
        .await
    }

    pub async fn session_workspace_move(&self, session_key: &str, cwd: &str) -> Result<Value> {
        self.set_cwd(session_key, cwd).await
    }

    pub async fn learning_edit(&self, id: &str, content: &str) -> Result<Value> {
        self.call("learning.edit", json!({ "id": id, "content": content }))
            .await
    }

    pub async fn mcp_servers_oauth_start(&self, name: &str) -> Result<Value> {
        self.call("mcp.servers.oauth.start", json!({ "name": name }))
            .await
    }

    pub async fn mcp_servers_oauth_poll(
        &self,
        name: &str,
        oauth_session_id: &str,
    ) -> Result<Value> {
        self.call(
            "mcp.servers.oauth.poll",
            json!({ "name": name, "session_id": oauth_session_id }),
        )
        .await
    }

    pub async fn mcp_servers_set_api_key(
        &self,
        name: &str,
        value: &str,
        env_var: Option<&str>,
    ) -> Result<Value> {
        let mut params = json!({ "name": name, "value": value });
        if let Some(ev) = env_var {
            params["env_var"] = json!(ev);
        }
        self.call("mcp.servers.set_api_key", params).await
    }

    pub async fn projects_tree(&self) -> Result<Value> {
        self.call("projects.tree", json!({ "preview_limit": 3 }))
            .await
    }

    pub async fn projects_discover_repos(&self, scan: bool) -> Result<Value> {
        self.call("projects.discover_repos", json!({ "scan": scan }))
            .await
    }

    pub async fn toolsets_list(&self, session_id: &str) -> Result<Value> {
        self.call("toolsets.list", json!({ "session_id": session_id }))
            .await
    }

    pub async fn cli_exec(&self, argv: &[String]) -> Result<Value> {
        self.call_timeout(
            "cli.exec",
            json!({ "argv": argv }),
            Duration::from_secs(240),
        )
        .await
    }

    pub async fn insights_get(&self) -> Result<Value> {
        self.call("insights.get", json!({ "days": 30 })).await
    }

    pub async fn session_usage(&self, session_id: &str) -> Result<Value> {
        let params = json!({ "session_id": session_id });
        let account = match self.call("session.account_usage", params.clone()).await {
            Ok(v) => Some(v),
            Err(err) if canonical::method_missing(&err) => None,
            Err(err) => return Err(err),
        };
        let stats = match self.call("session.stats", params.clone()).await {
            Ok(v) => Some(v),
            Err(err) if canonical::method_missing(&err) => None,
            Err(err) => return Err(err),
        };
        // Current main serves both. The third call is only for a gateway that
        // still answers the old combined `session.usage` verb.
        let legacy = if account.is_none() || stats.is_none() {
            match self.call("session.usage", params).await {
                Ok(v) => Some(v),
                Err(err) if canonical::method_missing(&err) => None,
                Err(err) if account.is_none() && stats.is_none() => return Err(err),
                Err(_) => None,
            }
        } else {
            None
        };
        canonical::merge_usage(account.as_ref(), stats.as_ref(), legacy.as_ref())
            .ok_or_else(|| anyhow::anyhow!("session usage is not available"))
    }

    pub async fn usage_bars(&self) -> Result<Value> {
        self.call("usage.bars", json!({})).await
    }

    pub async fn shell_exec(&self, command: &str) -> Result<Value> {
        self.call_timeout(
            "shell.exec",
            json!({ "command": command }),
            Duration::from_secs(35),
        )
        .await
    }

    pub async fn shutdown(&self) {
        {
            let mut outbound = self.outbound.lock().await;
            let _ = std::mem::replace(&mut *outbound, Outbound::Closed);
        }
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
    }
}

/// Block until the gateway emits `gateway.ready`, dies, or the startup budget elapses.
pub async fn wait_until_gateway_ready(
    rx: &mut broadcast::Receiver<JsonRpcEvent>,
    dur: Duration,
) -> Result<()> {
    match timeout(dur, async {
        loop {
            match rx.recv().await {
                Ok(evt) if evt.params.event_type == "gateway.ready" => return Ok(()),
                Ok(evt) if evt.params.event_type == "gateway.exit" => {
                    let code = evt
                        .params
                        .payload
                        .as_ref()
                        .and_then(|p| p.get("code"))
                        .cloned()
                        .unwrap_or(Value::Null);
                    anyhow::bail!("tui_gateway exited before ready (code {code})");
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => {
                    anyhow::bail!("tui_gateway event channel closed before ready");
                }
            }
        }
    })
    .await
    {
        Ok(inner) => inner,
        Err(_) => anyhow::bail!(
            "timed out waiting for gateway.ready after {}ms",
            dur.as_millis()
        ),
    }
}

#[cfg(test)]
const STDERR_LINE_CHARS: usize = 240;

/// Cap and mask secrets before gateway stderr hits the log file.
#[cfg(test)]
pub fn redact_gateway_line(raw: &str) -> String {
    let clipped: String = raw.chars().take(STDERR_LINE_CHARS).collect();
    const PREFIXES: &[&str] = &[
        "github_pat_",
        "sk-ant-",
        "Bearer ",
        "bearer ",
        "ghp_",
        "gho_",
        "sk-",
    ];
    let mut out = String::new();
    let mut rest = clipped.as_str();
    while !rest.is_empty() {
        let mut hit: Option<(&str, usize)> = None;
        for prefix in PREFIXES {
            if let Some(i) = rest.find(prefix) {
                match hit {
                    None => hit = Some((prefix, i)),
                    Some((prev, j)) if i < j || (i == j && prefix.len() > prev.len()) => {
                        hit = Some((prefix, i));
                    }
                    _ => {}
                }
            }
        }
        let Some((prefix, i)) = hit else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..i]);
        out.push_str(prefix);
        out.push_str("***");
        rest = &rest[i + prefix.len()..];
        let skip = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')))
            .unwrap_or(rest.len());
        rest = &rest[skip..];
    }
    out
}

#[cfg(test)]
fn path_sep() -> &'static str {
    if cfg!(windows) {
        ";"
    } else {
        ":"
    }
}

fn fresh_request_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn list_params_rejected(err: &anyhow::Error) -> bool {
    let text = err.to_string();
    text.contains("(4001)") || text.contains("invalid_params")
}

#[cfg(test)]
async fn write_child(outbound: &Mutex<Outbound>, body: &str) {
    let mut slot = outbound.lock().await;
    if let Outbound::Child(stdin) = &mut *slot {
        if stdin.write_all(body.as_bytes()).await.is_ok() {
            let _ = stdin.write_all(b"\n").await;
            let _ = stdin.flush().await;
        }
    }
}

async fn pump_socket<S>(
    ws: tokio_tungstenite::WebSocketStream<S>,
    mut outbound: mpsc::Receiver<String>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
    server_requests: Arc<Mutex<HashMap<String, ServerAsk>>>,
    generations: Arc<Mutex<GenerationBook>>,
    event_tx: broadcast::Sender<JsonRpcEvent>,
) where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let (mut sink, mut stream) = ws.split();
    'pump: loop {
        tokio::select! {
            biased;
            outgoing = outbound.recv() => {
                let Some(text) = outgoing else { break; };
                if sink.send(Message::Text(text.into())).await.is_err() {
                    break;
                }
            }
            incoming = stream.next() => {
                let Some(frame) = incoming else { break; };
                let Ok(frame) = frame else { break; };
                match frame {
                    Message::Text(text) => {
                        for line in text.lines() {
                            let line = line.trim();
                            if line.is_empty() {
                                continue;
                            }
                            let Ok(val) = serde_json::from_str::<Value>(line) else {
                                continue;
                            };
                            if let Some(reply) = dispatch_frame(
                                val,
                                &pending,
                                &server_requests,
                                &generations,
                                &event_tx,
                            )
                            .await
                            {
                                if sink.send(Message::Text(reply.into())).await.is_err() {
                                    break 'pump;
                                }
                            }
                        }
                    }
                    Message::Ping(payload) => {
                        if sink.send(Message::Pong(payload)).await.is_err() {
                            break;
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
        }
    }
    let _ = event_tx.send(JsonRpcEvent::synthetic(
        "gateway.exit",
        json!({ "code": Value::Null }),
    ));
}

async fn dispatch_frame(
    val: Value,
    pending: &Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>,
    server_requests: &Mutex<HashMap<String, ServerAsk>>,
    generations: &Mutex<GenerationBook>,
    event_tx: &broadcast::Sender<JsonRpcEvent>,
) -> Option<String> {
    match wire::classify(val) {
        wire::Inbound::Event(evt) => {
            if let Some(sid) = evt.params.session_id.as_deref() {
                if let Some(gen) = evt
                    .params
                    .payload
                    .as_ref()
                    .and_then(canonical::generation_of)
                {
                    generations.lock().await.note_session(sid, gen);
                }
            }
            if matches!(
                evt.params.event_type.as_str(),
                "approval.request" | "clarify.request"
            ) {
                if let Some(payload) = evt.params.payload.as_ref() {
                    if let (Some(pid), Some(gen)) = (
                        payload.get("prompt_id").and_then(|v| v.as_str()),
                        canonical::generation_of(payload),
                    ) {
                        generations.lock().await.note_prompt(pid, gen);
                    }
                }
            }
            let _ = event_tx.send(evt);
            None
        }
        wire::Inbound::Response(resp) => {
            if let Some(id_num) = resp.id.as_u64() {
                let mut map = pending.lock().await;
                if let Some(sender) = map.remove(&id_num) {
                    let _ = sender.send(resp);
                }
            }
            None
        }
        wire::Inbound::Ignore => None,
        wire::Inbound::Server {
            key,
            id,
            method,
            params,
        } => {
            if let Some((event, payload)) = wire::server_event(&method, &key, &params) {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let question_count = params
                    .get("questions")
                    .and_then(|q| q.as_array())
                    .map(|rows| rows.len())
                    .filter(|n| *n > 0)
                    .unwrap_or(1);
                server_requests.lock().await.insert(
                    key,
                    ServerAsk {
                        id,
                        method,
                        question_count,
                        answers: Map::new(),
                    },
                );
                let _ = event_tx.send(JsonRpcEvent {
                    jsonrpc: "2.0".into(),
                    method: "event".into(),
                    params: EventParams {
                        event_type: event.into(),
                        session_id,
                        payload: Some(payload),
                    },
                });
                None
            } else {
                Some(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32601, "message": "method not found" },
                    })
                    .to_string(),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn ready_event_unblocks_startup() {
        let (tx, mut rx) = broadcast::channel(8);
        let wait = wait_until_gateway_ready(&mut rx, Duration::from_millis(200));
        tx.send(JsonRpcEvent::synthetic("gateway.ready", json!({})))
            .unwrap();
        wait.await.expect("ready");
    }

    #[tokio::test]
    async fn exit_before_ready_fails() {
        let (tx, mut rx) = broadcast::channel(8);
        let wait = wait_until_gateway_ready(&mut rx, Duration::from_millis(200));
        tx.send(JsonRpcEvent::synthetic(
            "gateway.exit",
            json!({ "code": 1 }),
        ))
        .unwrap();
        let err = wait.await.expect_err("exit");
        assert!(err.to_string().contains("exited before ready"), "{err}");
    }

    #[tokio::test]
    async fn approval_server_request_is_an_event() {
        let (tx, mut rx) = broadcast::channel(4);
        let pending = Mutex::new(HashMap::new());
        let server = Mutex::new(HashMap::new());
        let generations = Mutex::new(GenerationBook::new());
        let reply = dispatch_frame(
            json!({
                "jsonrpc": "2.0",
                "id": "srq-1",
                "method": "approval",
                "params": {"command": "rm", "session_id": "s"}
            }),
            &pending,
            &server,
            &generations,
            &tx,
        )
        .await;
        assert!(reply.is_none());
        let evt = rx.try_recv().expect("event");
        assert_eq!(evt.params.event_type, "approval.request");
        let payload = evt.params.payload.expect("payload");
        assert_eq!(payload["request_id"], "s:srq-1");
        assert_eq!(payload["command"], "rm");
        assert!(server.lock().await.contains_key("s:srq-1"));
    }

    #[tokio::test]
    async fn unknown_server_request_replies_method_not_found() {
        let (tx, _rx) = broadcast::channel(4);
        let pending = Mutex::new(HashMap::new());
        let server = Mutex::new(HashMap::new());
        let generations = Mutex::new(GenerationBook::new());
        let reply = dispatch_frame(
            json!({
                "jsonrpc": "2.0",
                "id": "srq-2",
                "method": "tour",
                "params": {}
            }),
            &pending,
            &server,
            &generations,
            &tx,
        )
        .await
        .expect("error frame");
        assert!(reply.contains("-32601"), "{reply}");
        assert!(server.lock().await.is_empty());
    }

    #[tokio::test]
    async fn startup_timeout_fails() {
        let (_tx, mut rx) = broadcast::channel::<JsonRpcEvent>(8);
        let err = wait_until_gateway_ready(&mut rx, Duration::from_millis(20))
            .await
            .expect_err("timeout");
        assert!(err.to_string().contains("timed out"), "{err}");
    }

    #[test]
    fn redact_masks_keys_and_caps() {
        let gh = ["ghp", "_", "abcdefghijklmnopqrstuvwxyz0123"].concat();
        let line = redact_gateway_line(&format!("auth Bearer {gh} done"));
        assert!(line.contains("Bearer ***"), "{line}");
        assert!(!line.contains("abcdefghijklmnopqrstuvwxyz"), "{line}");
        let ant = ["sk-", "ant-", "secretvalue999"].concat();
        let sk = redact_gateway_line(&format!("OPENAI_API_KEY={ant}"));
        assert!(sk.contains("sk-ant-***"), "{sk}");
        assert!(!sk.contains("secretvalue999"), "{sk}");
        let long = "x".repeat(400);
        assert_eq!(
            redact_gateway_line(&long).chars().count(),
            STDERR_LINE_CHARS
        );
        assert_eq!(redact_gateway_line("gateway starting"), "gateway starting");
    }

    const FAKE_GATEWAY: &str = r#"
import json, sys
sys.stdout.write(json.dumps({
    "jsonrpc": "2.0",
    "method": "event",
    "params": {"type": "gateway.ready", "payload": {"version": "test"}}
}) + "\n")
sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    method = req.get("method")
    rid = req.get("id")
    if method == "session.create":
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0",
            "id": rid,
            "result": {"session_id": "s-test", "execution_generation": 3}
        }) + "\n")
    elif method == "session.interrupt":
        params = req.get("params") or {}
        sid = params.get("session_id")
        if params.get("execution_generation") != 3:
            sys.stdout.write(json.dumps({
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": 4001, "message": "invalid_params"}
            }) + "\n")
        else:
            sys.stdout.write(json.dumps({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {"ok": True, "session_id": sid}
            }) + "\n")
    else:
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0",
            "id": rid,
            "error": {"code": -32601, "message": method}
        }) + "\n")
    sys.stdout.flush()
"#;

    async fn spawn_fake_gateway() -> Option<(GatewayClient, std::path::PathBuf)> {
        let python = std::process::Command::new("python3")
            .arg("-c")
            .arg("print(1)")
            .output();
        if python.map(|o| !o.status.success()).unwrap_or(true) {
            return None;
        }
        let dir = std::env::temp_dir().join(format!("hermes-tui-gw-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("tui_gateway")).expect("tmpdir");
        std::fs::write(dir.join("tui_gateway/__init__.py"), "").expect("init");
        std::fs::write(dir.join("tui_gateway/entry.py"), FAKE_GATEWAY).expect("entry");
        let cfg = LaunchConfig {
            python: "python3".into(),
            src_root: dir.clone(),
            hermes_home: dir.join("home"),
            cwd: dir.clone(),
            title: "t".into(),
            resume: None,
            rpc_timeout: Duration::from_secs(5),
            startup_timeout: Duration::from_secs(5),
        };
        let client = GatewayClient::spawn_stdio(&cfg)
            .await
            .expect("spawn fake gateway");
        Some((client, dir))
    }

    #[tokio::test]
    async fn fake_gateway_ready_and_session_create() {
        let Some((client, dir)) = spawn_fake_gateway().await else {
            return;
        };
        let sid = client
            .create_session("t", &dir)
            .await
            .expect("session.create");
        assert_eq!(sid, "s-test");
        client.shutdown().await;
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn fake_gateway_interrupt_is_rpc_not_kill() {
        let Some((client, dir)) = spawn_fake_gateway().await else {
            return;
        };
        let sid = client
            .create_session("t", &dir)
            .await
            .expect("session.create");
        client.interrupt(&sid).await.expect("session.interrupt");
        // Child still answers after interrupt — we did not restart Python.
        client.interrupt(&sid).await.expect("second interrupt");
        client.shutdown().await;
        let _ = std::fs::remove_dir_all(&dir);
    }
}
