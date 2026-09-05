use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{broadcast, oneshot, Mutex};
use tokio::time::timeout;
use tokio_util::codec::{FramedRead, LinesCodec};

use super::types::{JsonRpcEvent, JsonRpcRequest, JsonRpcResponse};
use crate::config::LaunchConfig;

const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

pub struct GatewayClient {
    next_id: AtomicU64,
    stdin: Arc<Mutex<ChildStdin>>,
    pending_requests: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
    event_tx: broadcast::Sender<JsonRpcEvent>,
    child: Arc<Mutex<Child>>,
    rpc_timeout: Duration,
}

impl GatewayClient {
    pub async fn spawn(cfg: &LaunchConfig) -> Result<Self> {
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

        let (event_tx, _) = broadcast::channel(4096);
        let mut ready_rx = event_tx.subscribe();
        let pending_requests: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(target: "tui_gateway", "{}", redact_gateway_line(&line));
            }
        });

        let pending_clone = pending_requests.clone();
        let event_tx_clone = event_tx.clone();

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
                    Ok(val) => dispatch_frame(val, &pending_clone, &event_tx_clone).await,
                    Err(e) => {
                        tracing::warn!(
                            "Failed to parse gateway line: {e} (line: {})",
                            redact_gateway_line(&line)
                        );
                    }
                }
            }
        });

        let child = Arc::new(Mutex::new(child));
        let watch = child.clone();
        let exit_tx = event_tx.clone();
        tokio::spawn(async move {
            loop {
                let dead = {
                    let mut ch = watch.lock().await;
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
            stdin: Arc::new(Mutex::new(stdin)),
            pending_requests,
            event_tx,
            child,
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
        let req_str = serde_json::to_string(&req)? + "\n";

        let (tx, rx) = oneshot::channel();
        {
            let mut map = self.pending_requests.lock().await;
            map.insert(id, tx);
        }

        let write_res = {
            let mut stdin = self.stdin.lock().await;
            stdin
                .write_all(req_str.as_bytes())
                .await
                .and(stdin.flush().await)
        };
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

    pub async fn create_session(&self, title: &str, cwd: &Path) -> Result<String> {
        let params = json!({
            "title": title,
            "cwd": cwd.display().to_string(),
            "source": "tui",
        });
        let res = self.call("session.create", params).await?;
        res.get("session_id")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .context("Missing session_id")
    }

    pub async fn submit_prompt(&self, session_id: &str, text: &str) -> Result<()> {
        let params = json!({ "session_id": session_id, "text": text });
        let _ = self.call("prompt.submit", params).await?;
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
        let params = json!({ "session_id": session_id });
        self.call("session.interrupt", params).await?;
        Ok(())
    }

    pub async fn steer(&self, session_id: &str, text: &str) -> Result<String> {
        let res = self
            .call(
                "session.steer",
                json!({ "session_id": session_id, "text": text }),
            )
            .await?;
        Ok(res
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("ok")
            .to_string())
    }

    pub async fn close_session(&self, session_id: &str) -> Result<()> {
        let params = json!({ "session_id": session_id });
        let _ = self.call("session.close", params).await;
        Ok(())
    }

    pub async fn compress(&self, session_id: &str) -> Result<Value> {
        let params = json!({ "session_id": session_id });
        self.call_timeout("session.compress", params, Duration::from_secs(120))
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
        self.call("config.set", params).await
    }

    pub async fn session_title(&self, session_id: &str, title: Option<&str>) -> Result<Value> {
        let mut params = json!({ "session_id": session_id });
        if let Some(t) = title {
            params["title"] = json!(t);
        }
        self.call("session.title", params).await
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
        self.call(
            "session.cwd.set",
            json!({ "session_id": session_id, "cwd": cwd }),
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
        let res = self.call("session.list", json!({ "limit": 50 })).await?;
        Ok(res
            .get("sessions")
            .and_then(|s| s.as_array())
            .cloned()
            .unwrap_or_default())
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
        self.call("agents.list", json!({})).await
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

    pub async fn process_stop(&self) -> Result<Value> {
        self.call("process.stop", json!({})).await
    }

    pub async fn delegation_status(&self) -> Result<Value> {
        self.call("delegation.status", json!({})).await
    }

    pub async fn delegation_pause(&self, paused: bool) -> Result<Value> {
        self.call("delegation.pause", json!({ "paused": paused }))
            .await
    }

    pub async fn interrupt_subagent(&self, subagent_id: &str) -> Result<Value> {
        self.call("subagent.interrupt", json!({ "subagent_id": subagent_id }))
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
        self.call("session.resume", json!({ "session_id": session_id }))
            .await
    }

    pub async fn clarify_respond(
        &self,
        session_id: &str,
        request_id: &str,
        answer: &str,
        question_id: Option<&str>,
    ) -> Result<Value> {
        let mut params = json!({
            "session_id": session_id,
            "request_id": request_id,
            "answer": answer,
        });
        if let Some(qid) = question_id {
            params["question_id"] = json!(qid);
        }
        self.call("clarify.respond", params).await
    }

    pub async fn secret_respond(
        &self,
        method: &str,
        session_id: &str,
        request_id: &str,
        key: &str,
        value: &str,
    ) -> Result<Value> {
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
        let mut params = json!({
            "session_id": session_id,
            "choice": choice,
        });
        if let Some(id) = request_id {
            params["request_id"] = json!(id);
        }
        self.call("approval.respond", params).await
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
        let mut params = json!({ "session_id": session_id });
        if !name.is_empty() {
            params["name"] = json!(name);
        }
        self.call("session.branch", params).await
    }

    pub async fn session_undo(&self, session_id: &str) -> Result<Value> {
        self.call("session.undo", json!({ "session_id": session_id }))
            .await
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

    pub async fn reload_env(&self) -> Result<Value> {
        self.call("reload.env", json!({})).await
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
        self.call("session.delete", json!({ "session_id": session_id }))
            .await
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
        self.call(
            "session.set_hidden",
            json!({ "session_id": session_id, "hidden": hidden }),
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
        self.call(
            "session.redirect",
            json!({ "session_id": session_id, "text": text }),
        )
        .await
    }

    pub async fn session_workspace_move(&self, session_key: &str, cwd: &str) -> Result<Value> {
        self.call(
            "session.workspace.move",
            json!({ "session_key": session_key, "cwd": cwd }),
        )
        .await
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
        self.call("session.usage", json!({ "session_id": session_id }))
            .await
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
        let mut child = self.child.lock().await;
        let _ = child.kill().await;
        let _ = child.wait().await;
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

const STDERR_LINE_CHARS: usize = 240;

/// Cap and mask secrets before gateway stderr hits the log file.
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

fn path_sep() -> &'static str {
    if cfg!(windows) {
        ";"
    } else {
        ":"
    }
}

async fn dispatch_frame(
    val: Value,
    pending: &Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>,
    event_tx: &broadcast::Sender<JsonRpcEvent>,
) {
    if val.get("method").and_then(|m| m.as_str()) == Some("event") {
        if let Ok(evt) = serde_json::from_value::<JsonRpcEvent>(val.clone()) {
            let _ = event_tx.send(evt);
            return;
        }
    }
    if let Ok(resp) = serde_json::from_value::<JsonRpcResponse>(val) {
        if let Some(id_num) = resp.id.as_u64() {
            let mut map = pending.lock().await;
            if let Some(sender) = map.remove(&id_num) {
                let _ = sender.send(resp);
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
            "result": {"session_id": "s-test"}
        }) + "\n")
    elif method == "session.interrupt":
        sid = (req.get("params") or {}).get("session_id")
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
        let client = GatewayClient::spawn(&cfg)
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
