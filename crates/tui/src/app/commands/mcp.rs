//! MCP, memory, and cron command handlers.
use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use crate::optimistic;
use crate::rpc::GatewayClient;
use crate::state::{
    format_mcp_test, format_tools_show, parse_cron_jobs, parse_mcp_servers, ActiveView, AppState,
};

use super::super::LoopControl;
use super::refresh_memory;

pub(crate) async fn refresh_mcp(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    let list = client.mcp_servers_list().await;
    let catalog = client.mcp_catalog().await;
    let mut s = state.lock().await;
    match (list, catalog) {
        (Ok(list), Ok(catalog)) => {
            s.mcp_servers = parse_mcp_servers(&list, &catalog);
            let n = s.mcp_servers.len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        (Ok(list), Err(_)) => {
            s.mcp_servers = parse_mcp_servers(&list, &json!({}));
            let n = s.mcp_servers.len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        (Err(e), _) => {
            s.set_toast(format!("mcp list failed · {e}"));
        }
    }
}

pub(crate) fn selected_plugin(s: &AppState) -> Option<(String, bool)> {
    let idxs = s.filtered_plugin_indices();
    idxs.get(s.modal_selected)
        .and_then(|i| s.plugins.get(*i))
        .map(|p| {
            let ident = if p.key.is_empty() {
                p.name.clone()
            } else {
                p.key.clone()
            };
            (ident, p.enabled)
        })
}

pub(crate) fn selected_provider_slug(s: &AppState) -> Option<String> {
    let idxs = s.filtered_provider_indices();
    idxs.get(s.modal_selected)
        .and_then(|i| s.providers.get(*i))
        .map(|p| p.slug.clone())
}

pub(crate) fn selected_mcp(s: &AppState) -> Option<(String, bool)> {
    let idxs = s.filtered_mcp_indices();
    idxs.get(s.modal_selected)
        .and_then(|i| s.mcp_servers.get(*i))
        .map(|m| (m.name.clone(), m.configured))
}

pub(crate) fn selected_memory(s: &AppState) -> Option<(String, String)> {
    let idxs = s.filtered_memory_indices();
    idxs.get(s.modal_selected)
        .and_then(|i| s.memory_nodes.get(*i))
        .map(|m| {
            let id = if m.id.is_empty() {
                m.label.clone()
            } else {
                m.id.clone()
            };
            let fallback = if m.body.is_empty() {
                format!("{}  {}", m.label, m.meta)
            } else {
                format!("{}\n{}", m.label, m.body)
            };
            (id, fallback)
        })
}

pub(crate) fn selected_profile_name(s: &AppState) -> Option<String> {
    let idxs = s.filtered_profile_indices();
    idxs.get(s.modal_selected)
        .and_then(|i| s.profiles.get(*i))
        .map(|p| p.name.clone())
}

pub(crate) async fn mcp_add(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>, name: &str) {
    match client.mcp_servers_add(name, Some(name)).await {
        Ok(_) => {
            state
                .lock()
                .await
                .set_toast(format!("added {name} · r reloads live session"));
            refresh_mcp(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("mcp add · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn mcp_remove(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    name: &str,
) {
    match client.mcp_servers_remove(name).await {
        Ok(_) => {
            state.lock().await.set_toast(format!("removed {name}"));
            refresh_mcp(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("mcp remove · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn mcp_oauth_login(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    name: &str,
) {
    state.lock().await.set_toast(format!("oauth start {name}…"));
    let started = match client.mcp_servers_oauth_start(name).await {
        Ok(v) => v,
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("oauth · {}", optimistic::brief_err(&e)));
            return;
        }
    };
    let url = started
        .get("auth_url")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let oauth_sid = started
        .get("session_id")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    if url.is_empty() || oauth_sid.is_empty() {
        state.lock().await.set_toast("oauth: no auth_url");
        return;
    }
    match crate::platform::open_http_url(&url) {
        Ok(()) => state.lock().await.set_toast(format!("browser · {name}")),
        Err(_) => {
            state
                .lock()
                .await
                .open_peek(format!("oauth  {name}"), url.clone(), None);
        }
    }
    let st = Arc::clone(state);
    let cl = Arc::clone(client);
    let server = name.to_string();
    tokio::spawn(async move {
        for _ in 0..90 {
            tokio::time::sleep(Duration::from_secs(2)).await;
            match cl.mcp_servers_oauth_poll(&server, &oauth_sid).await {
                Ok(v) => {
                    let status = v.get("status").and_then(|x| x.as_str()).unwrap_or("");
                    if status == "approved" {
                        st.lock()
                            .await
                            .set_toast(format!("oauth approved · {server}"));
                        refresh_mcp(&st, &cl).await;
                        return;
                    }
                    if status == "error" {
                        let err = v
                            .get("error_message")
                            .or_else(|| v.get("error"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("oauth failed");
                        st.lock().await.set_toast(err);
                        return;
                    }
                }
                Err(e) => {
                    st.lock()
                        .await
                        .set_toast(format!("oauth poll · {}", optimistic::brief_err(&e)));
                    return;
                }
            }
        }
        st.lock()
            .await
            .set_toast(format!("oauth timeout · {server}"));
    });
}

pub(crate) async fn mcp_test(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    name: &str,
) {
    state.lock().await.set_toast(format!("testing {name}…"));
    match client.mcp_servers_test(name).await {
        Ok(v) => {
            let msg = format_mcp_test(&v, name);
            if v.get("ok").and_then(|x| x.as_bool()) == Some(true) {
                let mut body = msg.clone();
                if let Some(tools) = v.get("tools").and_then(|x| x.as_array()) {
                    for t in tools.iter().take(24) {
                        let tn = t.get("name").and_then(|x| x.as_str()).unwrap_or("?");
                        body.push_str(&format!("\n  {tn}"));
                    }
                }
                state
                    .lock()
                    .await
                    .open_peek(format!("mcp test  {name}"), body, None);
            } else {
                state.lock().await.set_toast(msg);
            }
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("mcp test · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn peek_toolset(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    name: &str,
) {
    let sid = state.lock().await.session_id.clone().unwrap_or_default();
    let local = state
        .lock()
        .await
        .toolsets
        .iter()
        .find(|t| t.name == name)
        .map(|t| {
            format!(
                "{}\n{} tools · {}",
                t.description,
                t.tool_count,
                if t.enabled { "enabled" } else { "disabled" }
            )
        })
        .unwrap_or_default();
    match client.tools_show(&sid).await {
        Ok(v) => {
            let shown = format_tools_show(&v, Some(name));
            let body = if shown.starts_with("no tools") {
                local
            } else {
                format!("{local}\n\n{shown}")
            };
            state
                .lock()
                .await
                .open_peek(format!("toolset  {name}"), body, None);
        }
        Err(_) => {
            state
                .lock()
                .await
                .open_peek(format!("toolset  {name}"), local, None);
        }
    }
}

pub(crate) async fn begin_memory_edit(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    id: &str,
) -> Result<LoopControl> {
    let body = match client.learning_detail(id).await {
        Ok(v) => v
            .get("content")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("memory edit · {}", optimistic::brief_err(&e)));
            return Ok(LoopControl::Continue);
        }
    };
    let mut s = state.lock().await;
    s.pending_memory_edit = Some(id.to_string());
    s.pending_memory_body = body;
    s.active_view = ActiveView::Chat;
    Ok(LoopControl::Editor)
}

pub(crate) async fn apply_memory_edit(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    id: &str,
    content: &str,
) {
    match client.learning_edit(id, content).await {
        Ok(v) => {
            let msg = v.get("message").and_then(|x| x.as_str()).unwrap_or("saved");
            state.lock().await.set_toast(msg);
            refresh_memory(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("memory edit · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn handle_mcp_key_input(
    key: KeyEvent,
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
) -> Result<LoopControl> {
    match key.code {
        KeyCode::Esc => {
            let mut s = state.lock().await;
            s.mcp_key_name = None;
            s.picker_key.clear();
            s.set_toast("mcp key cancelled");
        }
        KeyCode::Enter => {
            let (name, value) = {
                let s = state.lock().await;
                (
                    s.mcp_key_name.clone().unwrap_or_default(),
                    s.picker_key.trim().to_string(),
                )
            };
            if value.is_empty() {
                let mut s = state.lock().await;
                s.picker_key_error = "paste a key".into();
                s.mark_dirty();
                return Ok(LoopControl::Continue);
            }
            match client.mcp_servers_set_api_key(&name, &value, None).await {
                Ok(_) => {
                    let mut s = state.lock().await;
                    s.mcp_key_name = None;
                    s.picker_key.clear();
                    s.set_toast(format!("key saved for {name}"));
                    drop(s);
                    refresh_mcp(state, client).await;
                }
                Err(e) => {
                    let mut s = state.lock().await;
                    s.picker_key_error = optimistic::brief_err(&e);
                    s.mark_dirty();
                }
            }
        }
        KeyCode::Backspace => {
            let mut s = state.lock().await;
            s.picker_key.pop();
            s.mark_dirty();
        }
        KeyCode::Char(c)
            if !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT) =>
        {
            let mut s = state.lock().await;
            s.picker_key.push(c);
            s.picker_key_error.clear();
            s.mark_dirty();
        }
        _ => {}
    }
    Ok(LoopControl::Continue)
}

pub(crate) async fn peek_memory(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    id: &str,
    fallback: &str,
) {
    match client.learning_detail(id).await {
        Ok(v) => {
            if v.get("ok").and_then(|x| x.as_bool()) == Some(false) {
                let msg = v
                    .get("message")
                    .and_then(|x| x.as_str())
                    .unwrap_or(fallback);
                state
                    .lock()
                    .await
                    .open_peek("memory".into(), msg.to_string(), None);
                return;
            }
            let label = v.get("label").and_then(|x| x.as_str()).unwrap_or(id);
            let content = v
                .get("content")
                .and_then(|x| x.as_str())
                .unwrap_or(fallback);
            state
                .lock()
                .await
                .open_peek(format!("memory  {label}"), content.to_string(), None);
        }
        Err(_) => {
            state
                .lock()
                .await
                .open_peek("memory".into(), fallback.to_string(), None);
        }
    }
}

pub(crate) async fn delete_memory(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    id: &str,
) {
    match client.learning_delete(id).await {
        Ok(v) => {
            let msg = v
                .get("message")
                .and_then(|x| x.as_str())
                .unwrap_or("deleted");
            state.lock().await.set_toast(msg);
            refresh_memory(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("memory · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) fn selected_cron_id(s: &AppState) -> Option<String> {
    let idxs = s.filtered_cron_indices();
    idxs.get(s.modal_selected)
        .and_then(|i| s.cron_jobs.get(*i))
        .map(|j| j.id.clone())
}

pub(crate) async fn refresh_cron(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    match client.cron_manage("list", None, true).await {
        Ok(v) => {
            let mut s = state.lock().await;
            s.cron_jobs = parse_cron_jobs(&v);
            let n = s.cron_jobs.len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("cron · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn cron_action(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    action: &str,
    id: &str,
) {
    match client.cron_manage(action, Some(id), true).await {
        Ok(_) => {
            state.lock().await.set_toast(format!("cron {action}"));
            refresh_cron(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("cron {action} · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn reload_mcp(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>) {
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session");
        return;
    };
    match client.reload_mcp(&sid, true).await {
        Ok(v) => {
            let status = v
                .get("status")
                .and_then(|x| x.as_str())
                .unwrap_or("reloaded");
            let msg = v.get("message").and_then(|x| x.as_str()).unwrap_or(status);
            state.lock().await.set_toast(msg);
            refresh_mcp(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("reload-mcp failed · {e}"));
        }
    }
}
