//! Show/config/project overlay command handlers.
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::optimistic;
use crate::rpc::GatewayClient;
use crate::state::{
    format_usage_bars, match_project_id, parse_gateway_messages, parse_project_rows,
    parse_project_session_records, parse_spawn_entries, parse_spawn_tree_agents, ActiveView,
    AppState,
};

pub(crate) fn config_value_text(v: &serde_json::Value) -> String {
    match v.get("value") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Bool(b)) => {
            if *b {
                "on".into()
            } else {
                "off".into()
            }
        }
        Some(other) => other.to_string(),
        None => v
            .get("output")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
    }
}

pub(crate) async fn show_stored_history(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>) {
    let sid = state.lock().await.session_id.clone().unwrap_or_default();
    match client.session_history(&sid).await {
        Ok(v) => {
            let msgs = v
                .get("messages")
                .and_then(|m| m.as_array())
                .cloned()
                .unwrap_or_default();
            let parsed = parse_gateway_messages(&msgs);
            match crate::shell::history_text(&parsed, 400) {
                Some(body) => state
                    .lock()
                    .await
                    .open_peek("history stored".into(), body, None),
                None => state.lock().await.set_toast("no stored history"),
            }
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("history · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn show_projects(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    arg: &str,
) {
    if arg.trim().eq_ignore_ascii_case("scan") {
        scan_projects(state, client).await;
        return;
    }
    if arg.trim().is_empty() {
        open_projects_overlay(state, client).await;
        return;
    }
    match client.projects_tree().await {
        Ok(v) => {
            let rows = parse_project_rows(&v);
            let q = arg.trim();
            let drill = match_project_id(&v, q).or_else(|| Some(q.to_string()));
            let mut s = state.lock().await;
            s.projects_list = rows;
            s.project_sessions.clear();
            s.project_drill = None;
            s.active_view = ActiveView::Projects;
            s.modal_selected = 0;
            s.picker_filter.clear();
            s.mark_dirty();
            drop(s);
            if let Some(id) = drill {
                drill_project(state, client, &id).await;
            }
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("projects · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn open_projects_overlay(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
) {
    match client.projects_tree().await {
        Ok(v) => {
            let rows = parse_project_rows(&v);
            let mut s = state.lock().await;
            s.projects_list = rows;
            s.project_sessions.clear();
            s.project_drill = None;
            s.active_view = ActiveView::Projects;
            s.modal_selected = 0;
            s.picker_filter.clear();
            let n = s.picker_len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("projects · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn scan_projects(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>) {
    state.lock().await.set_toast("scanning repos…");
    match client.projects_discover_repos(true).await {
        Ok(v) => {
            let n = v
                .get("repos")
                .and_then(|x| x.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            let enabled = v
                .get("discovery_policy")
                .and_then(|p| p.get("enabled"))
                .and_then(|x| x.as_bool())
                .unwrap_or(true);
            if !enabled {
                state.lock().await.set_toast("repo discovery off in config");
                return;
            }
            state
                .lock()
                .await
                .set_toast(format!("discovered {n} repos"));
            open_projects_overlay(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("scan · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn drill_project(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    id: &str,
) {
    match client.projects_project_sessions(id).await {
        Ok(v) => {
            let sessions = parse_project_session_records(&v);
            let name = v
                .get("project")
                .and_then(|p| p.get("label").or_else(|| p.get("name")))
                .and_then(|x| x.as_str())
                .unwrap_or(id)
                .to_string();
            if v.get("project").map(|p| p.is_null()).unwrap_or(true) && sessions.is_empty() {
                state
                    .lock()
                    .await
                    .set_toast(format!("project not found · {id}"));
                return;
            }
            let mut s = state.lock().await;
            s.project_sessions = sessions;
            s.project_drill = Some(name);
            s.active_view = ActiveView::Projects;
            s.modal_selected = 0;
            s.picker_filter.clear();
            let n = s.picker_len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("projects · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn show_logs(state: &Arc<Mutex<AppState>>) {
    let home = state.lock().await.hermes_home.clone();
    let path = home.join("logs").join("tui.log");
    let body = match std::fs::read_to_string(&path) {
        Ok(raw) => {
            let lines: Vec<&str> = raw.lines().collect();
            let take = lines.len().saturating_sub(80);
            lines[take..].join("\n")
        }
        Err(_) => format!(
            "no log at {}\nset HERMES_TUI_LOG or RUST_LOG to enable",
            path.display()
        ),
    };
    state.lock().await.open_peek("logs".into(), body, None);
}

pub(crate) async fn load_replay(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    path: &str,
) {
    if path.is_empty() {
        state.lock().await.set_toast("usage: /replay load <path>");
        return;
    }
    match client.spawn_tree_load(path).await {
        Ok(v) => {
            let rows = parse_spawn_tree_agents(&v);
            if rows.is_empty() {
                state.lock().await.set_toast("snapshot empty or unreadable");
                return;
            }
            let label = v.get("label").and_then(|x| x.as_str()).unwrap_or(path);
            let n = rows.len();
            let mut s = state.lock().await;
            s.agents_replay = true;
            s.agents_steer = false;
            s.agent_rows.retain(|r| r.kind != "subagent");
            s.agent_rows.extend(rows);
            s.active_view = ActiveView::Agents;
            s.modal_selected = 0;
            s.picker_filter.clear();
            let len = s.picker_len();
            s.clamp_modal(len);
            s.set_toast(format!("snapshot {label} · {n} agents · live controls off"));
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("replay load · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn show_credits(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>) {
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session");
        return;
    };
    match client.session_usage(&sid).await {
        Ok(v) => {
            let calls = v.get("calls").and_then(|x| x.as_u64()).unwrap_or(0);
            let input = v.get("input").and_then(|x| x.as_u64()).unwrap_or(0);
            let output = v.get("output").and_then(|x| x.as_u64()).unwrap_or(0);
            let total = v.get("total").and_then(|x| x.as_u64()).unwrap_or(0);
            let mut body = format!("calls {calls}\ninput {input}\noutput {output}\ntotal {total}");
            if let Some(lines) = v.get("credits_lines").and_then(|x| x.as_array()) {
                for line in lines {
                    if let Some(s) = line.as_str() {
                        body.push('\n');
                        body.push_str(s);
                    }
                }
            }
            if let Ok(bars) = client.usage_bars().await {
                if let Some(extra) = format_usage_bars(&bars) {
                    body.push_str("\n\n");
                    body.push_str(&extra);
                }
            }
            state.lock().await.open_peek("usage".into(), body, None);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("credits · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn show_setup(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>) {
    let status = client.setup_status().await;
    let runtime = client.setup_runtime_check().await;
    let mut body = String::new();
    match status {
        Ok(v) => {
            let ok = v
                .get("provider_configured")
                .and_then(|x| x.as_bool())
                .unwrap_or(false);
            body.push_str(if ok {
                "provider configured: yes\n"
            } else {
                "provider configured: no\n"
            });
        }
        Err(e) => body.push_str(&format!("setup.status: {e}\n")),
    }
    match runtime {
        Ok(v) => {
            let ok = v.get("ok").and_then(|x| x.as_bool()).unwrap_or(false);
            let provider = v.get("provider").and_then(|x| x.as_str()).unwrap_or("");
            let model = v.get("model").and_then(|x| x.as_str()).unwrap_or("");
            let err = v.get("error").and_then(|x| x.as_str()).unwrap_or("");
            body.push_str(&format!(
                "runtime: {}  {provider}  {model}\n{err}",
                if ok { "ok" } else { "blocked" }
            ));
        }
        Err(e) => body.push_str(&format!("runtime_check: {e}")),
    }
    state.lock().await.open_peek("setup".into(), body, None);
}

pub(crate) async fn show_config(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>) {
    match client.config_show().await {
        Ok(v) => {
            let mut body = String::new();
            if let Some(sections) = v.get("sections").and_then(|x| x.as_array()) {
                for sec in sections {
                    let title = sec.get("title").and_then(|x| x.as_str()).unwrap_or("");
                    if !title.is_empty() {
                        body.push_str(title);
                        body.push('\n');
                    }
                    if let Some(rows) = sec.get("rows").and_then(|x| x.as_array()) {
                        for row in rows {
                            if let Some(pair) = row.as_array() {
                                let k = pair.first().and_then(|x| x.as_str()).unwrap_or("");
                                let val = pair.get(1).and_then(|x| x.as_str()).unwrap_or("");
                                body.push_str(&format!("  {k}: {val}\n"));
                            }
                        }
                    }
                    body.push('\n');
                }
            }
            if body.trim().is_empty() {
                body = v.to_string();
            }
            state.lock().await.open_peek("config".into(), body, None);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("config · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn show_facts(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>) {
    let cwd = state.lock().await.metrics.cwd.clone();
    match client.project_facts(&cwd).await {
        Ok(v) => {
            let body = match v.get("facts") {
                Some(serde_json::Value::Null) | None => {
                    "not a code workspace (or facts unavailable)".to_string()
                }
                Some(facts) => {
                    serde_json::to_string_pretty(facts).unwrap_or_else(|_| facts.to_string())
                }
            };
            state
                .lock()
                .await
                .open_peek("project facts".into(), body, None);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("facts · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn show_verify(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>) {
    let (sid, cwd) = {
        let s = state.lock().await;
        (
            s.session_id.clone().unwrap_or_default(),
            s.metrics.cwd.clone(),
        )
    };
    match client.verification_status(&sid, &cwd).await {
        Ok(v) => {
            let body = serde_json::to_string_pretty(v.get("verification").unwrap_or(&v))
                .unwrap_or_else(|_| v.to_string());
            state.lock().await.open_peek("verify".into(), body, None);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("verify · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn show_replay(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>) {
    let sid = state.lock().await.session_id.clone().unwrap_or_default();
    match client.spawn_tree_list(&sid).await {
        Ok(v) => {
            let entries = parse_spawn_entries(&v);
            let mut s = state.lock().await;
            s.spawn_trees = entries;
            s.active_view = ActiveView::Replay;
            s.modal_selected = 0;
            s.picker_filter.clear();
            let n = s.picker_len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("replay · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn show_insights(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>) {
    match client.insights_get().await {
        Ok(v) => {
            let days = v.get("days").and_then(|x| x.as_u64()).unwrap_or(30);
            let sessions = v.get("sessions").and_then(|x| x.as_u64()).unwrap_or(0);
            let messages = v.get("messages").and_then(|x| x.as_u64()).unwrap_or(0);
            state.lock().await.set_toast(format!(
                "{days}d · {sessions} sessions · {messages} messages"
            ));
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("insights · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn browser_command(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    arg: &str,
) {
    let mut parts = arg.split_whitespace();
    let action = parts.next().unwrap_or("status").to_ascii_lowercase();
    if !matches!(action.as_str(), "status" | "connect" | "disconnect") {
        state
            .lock()
            .await
            .set_toast("usage: /browser [connect|disconnect|status] [url]");
        return;
    }
    let rest = parts.collect::<Vec<_>>().join(" ");
    let url = if action == "connect" {
        Some(if rest.is_empty() {
            "http://127.0.0.1:9222".to_string()
        } else {
            rest
        })
    } else {
        None
    };
    let sid = state.lock().await.session_id.clone();
    match client
        .browser_manage(&action, sid.as_deref(), url.as_deref())
        .await
    {
        Ok(v) => {
            let connected = v
                .get("connected")
                .and_then(|x| x.as_bool())
                .unwrap_or(false);
            let url = v.get("url").and_then(|x| x.as_str()).unwrap_or("");
            let msg = v
                .get("message")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    if connected {
                        format!("browser connected · {url}")
                    } else {
                        "browser disconnected".into()
                    }
                });
            state.lock().await.set_toast(msg);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("browser · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn config_key(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    key: &str,
    arg: &str,
) {
    let sid = state.lock().await.session_id.clone();
    let mode = arg.trim().to_ascii_lowercase();
    if mode.is_empty() || mode == "status" || mode == "show" || mode == "?" {
        match client.get_config(key, sid.as_deref()).await {
            Ok(v) => {
                let val = config_value_text(&v);
                let mut s = state.lock().await;
                s.apply_config_value(key, &val);
                s.set_toast(format!("{key}: {val}"));
            }
            Err(e) => {
                state
                    .lock()
                    .await
                    .set_toast(format!("{key} · {}", optimistic::brief_err(&e)));
            }
        }
        return;
    }
    let mut params = json!({ "key": key, "value": arg.trim() });
    if let Some(sid) = sid {
        params["session_id"] = json!(sid);
    }
    match client.set_config(params).await {
        Ok(v) => {
            let val = config_value_text(&v);
            let shown = if val.is_empty() {
                arg.trim().to_string()
            } else {
                val
            };
            let mut s = state.lock().await;
            s.apply_config_value(key, &shown);
            s.set_toast(format!("{key}: {shown}"));
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("{key} · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn set_or_show_title(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    arg: &str,
) {
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session");
        return;
    };
    let title = arg.trim();
    if title.is_empty() {
        match client.session_title(&sid, None).await {
            Ok(v) => {
                let t = v.get("title").and_then(|x| x.as_str()).unwrap_or("");
                state.lock().await.set_toast(if t.is_empty() {
                    "no title set".into()
                } else {
                    format!("title: {t}")
                });
            }
            Err(e) => {
                state
                    .lock()
                    .await
                    .set_toast(format!("title · {}", optimistic::brief_err(&e)));
            }
        }
        return;
    }
    match client.session_title(&sid, Some(title)).await {
        Ok(v) => {
            let next = v
                .get("title")
                .and_then(|x| x.as_str())
                .unwrap_or(title)
                .to_string();
            let mut s = state.lock().await;
            s.session_title = next.clone();
            s.set_toast(format!("title: {next}"));
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("title · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn change_cwd(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    arg: &str,
) {
    let raw = arg.trim();
    if raw.is_empty() {
        state.lock().await.set_toast("usage: /cd <path>");
        return;
    }
    let (sid, busy) = {
        let s = state.lock().await;
        (s.session_id.clone(), s.is_generating)
    };
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session");
        return;
    };
    if busy {
        state.lock().await.set_toast("session busy");
        return;
    }
    match client.set_cwd(&sid, raw).await {
        Ok(v) => {
            let cwd = v
                .get("cwd")
                .and_then(|x| x.as_str())
                .unwrap_or(raw)
                .to_string();
            let mut s = state.lock().await;
            s.metrics.cwd = cwd.clone();
            let (repo, branch) = crate::platform::probe_git_repo_branch(&cwd);
            s.metrics.git_repo = repo;
            s.metrics.git_branch = branch;
            s.set_toast(cwd);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("cd · {}", optimistic::brief_err(&e)));
        }
    }
}
