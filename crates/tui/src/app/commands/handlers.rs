//! Per-command implementations called from `dispatch_slash`.
use anyhow::Result;
use crossterm::event::KeyEvent;
use ratatui_textarea::TextArea;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use crate::optimistic;
use crate::rpc::GatewayClient;
use crate::state::{
    config_flag_on, format_profile_describe, format_skill_inspect, format_tools_configure,
    parse_gateway_messages, parse_plugins, parse_toolsets, spawn_subagents_from_rows, AppState,
};

use super::super::{reset_prompt, resume_session, LoopControl};
use super::{
    config_value_text, refresh_agents, refresh_models, refresh_profiles, refresh_sessions,
    refresh_skills,
};

pub(crate) async fn cycle_permission_mode(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
) {
    let mut s = state.lock().await;
    let (prev, next, epoch) = optimistic::apply_mode_cycle(&mut s);
    let sid = s.session_id.clone();
    drop(s);
    if let Some(sid) = sid {
        optimistic::spawn_mode_reconcile(state, client, sid, epoch, prev, next);
    }
}

pub(crate) async fn run_bang(
    cmd: String,
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
) -> Result<LoopControl> {
    {
        let mut s = state.lock().await;
        s.add_user_message(format!("!{cmd}"));
        s.set_toast("running…");
    }
    *textarea = reset_prompt(false);
    match client.shell_exec(&cmd).await {
        Ok(v) => {
            let stdout = v.get("stdout").and_then(|x| x.as_str()).unwrap_or("");
            let stderr = v.get("stderr").and_then(|x| x.as_str()).unwrap_or("");
            let code = v.get("code").and_then(|x| x.as_i64()).unwrap_or(-1);
            let ctx = crate::shell::format_shell_context(&cmd, stdout, stderr, code);
            let mut s = state.lock().await;
            s.add_system(ctx.clone());
            s.shell_context = ctx;
            s.set_toast("shell output attached to next send");
        }
        Err(e) => {
            state.lock().await.add_system(format!("!{cmd} failed: {e}"));
        }
    }
    Ok(LoopControl::Continue)
}

pub(crate) async fn set_personality(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    arg: &str,
) {
    let sid = state.lock().await.session_id.clone();
    if arg.trim().is_empty() {
        match client.get_config("personality", sid.as_deref()).await {
            Ok(v) => {
                let val = config_value_text(&v);
                state.lock().await.set_toast(format!(
                    "personality: {}",
                    if val.is_empty() { "default" } else { &val }
                ));
            }
            Err(e) => {
                state
                    .lock()
                    .await
                    .set_toast(format!("personality · {}", optimistic::brief_err(&e)));
            }
        }
        return;
    }
    let mut params = json!({ "key": "personality", "value": arg.trim() });
    if let Some(sid) = &sid {
        params["session_id"] = json!(sid);
    }
    match client.set_config(params).await {
        Ok(v) => {
            let val = config_value_text(&v);
            let reset = v.get("history_reset").and_then(|x| x.as_bool()) == Some(true);
            let name = if val.is_empty() {
                arg.trim()
            } else {
                val.as_str()
            };
            let mut s = state.lock().await;
            s.set_toast(if reset {
                format!("personality: {name} · transcript reset")
            } else {
                format!("personality: {name}")
            });
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("personality · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn delete_stored_session(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    id: &str,
) {
    let live = state.lock().await.session_id.clone();
    if live.as_deref() == Some(id) {
        state
            .lock()
            .await
            .set_toast("cannot delete the live session");
        return;
    }
    match client.session_delete(id).await {
        Ok(_) => {
            state.lock().await.set_toast(format!("deleted {id}"));
            refresh_sessions(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("delete · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) fn selected_toolset_name(s: &AppState) -> Option<String> {
    let idxs = s.filtered_toolset_indices();
    idxs.get(s.modal_selected)
        .and_then(|i| s.toolsets.get(*i))
        .map(|t| t.name.clone())
}

pub(crate) fn selected_skill_name(s: &AppState) -> Option<String> {
    let idxs = s.filtered_skill_indices();
    idxs.get(s.modal_selected)
        .and_then(|i| s.skills.get(*i))
        .map(|t| t.name.clone())
}

pub(crate) async fn apply_vim(
    textarea: &mut TextArea<'static>,
    state: &Mutex<AppState>,
    key: KeyEvent,
) {
    let mut vim = match state.lock().await.vim {
        Some(v) => v,
        None => return,
    };
    let action = crate::composer_vim::handle(&mut vim, key, textarea);
    let mut s = state.lock().await;
    match action {
        crate::composer_vim::VimAction::Leave => {
            s.vim = None;
            s.set_toast("vim off · esc is interrupt");
        }
        crate::composer_vim::VimAction::Stay => {
            s.vim = Some(vim);
            s.mark_dirty();
        }
    }
}

pub(crate) async fn inspect_skill(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    name: &str,
) {
    match client.skills_manage("inspect", name).await {
        Ok(v) => {
            let body = format_skill_inspect(&v);
            state
                .lock()
                .await
                .open_peek(format!("skill  {name}"), body, None);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("skill · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn install_skill(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    name: &str,
) {
    state.lock().await.set_toast(format!("installing {name}…"));
    match client.skills_manage("install", name).await {
        Ok(_) => {
            state.lock().await.set_toast(format!("installed {name}"));
            refresh_skills(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("install · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn search_skills(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    query: &str,
) {
    match client.skills_manage("search", query).await {
        Ok(v) => {
            let mut body = String::new();
            if let Some(arr) = v.get("results").and_then(|x| x.as_array()) {
                for r in arr {
                    let n = r.get("name").and_then(|x| x.as_str()).unwrap_or("?");
                    let d = r.get("description").and_then(|x| x.as_str()).unwrap_or("");
                    body.push_str(&format!("{n}\n  {d}\n"));
                }
            }
            if body.is_empty() {
                body = "no matches".into();
            }
            state
                .lock()
                .await
                .open_peek(format!("skills  {query}"), body, None);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("search · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn draft_commit_message(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
) {
    let (sid, cwd) = {
        let s = state.lock().await;
        (
            s.session_id.clone().unwrap_or_default(),
            s.metrics.cwd.clone(),
        )
    };
    let dur = std::time::Duration::from_secs(8);
    let staged = crate::platform::git_probe(&cwd, &["diff", "--cached"], dur).unwrap_or_default();
    let unstaged = crate::platform::git_probe(&cwd, &["diff"], dur).unwrap_or_default();
    let diff = if staged.trim().is_empty() {
        unstaged
    } else {
        staged
    };
    if diff.trim().is_empty() {
        state.lock().await.set_toast("no git diff to describe");
        return;
    }
    let clip: String = diff.chars().take(12_000).collect();
    state.lock().await.set_toast("drafting commit message…");
    match client
        .llm_oneshot(
            &sid,
            "Write a conventional commit message for this diff. Subject line only first, then an optional body. No fences.",
            &clip,
        )
        .await
    {
        Ok(v) => {
            let text = v.get("text").and_then(|x| x.as_str()).unwrap_or("").trim();
            if text.is_empty() {
                state.lock().await.set_toast("oneshot returned empty");
            } else {
                state
                    .lock()
                    .await
                    .open_peek("commit draft".into(), text.to_string(), None);
            }
        }
        Err(e) => {
            state.lock().await.set_toast(format!(
                "commit · {}",
                optimistic::brief_err(&e)
            ));
        }
    }
}

pub(crate) async fn maybe_auto_resume(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    let on = match client
        .get_config("display.tui_auto_resume_recent", None)
        .await
    {
        Ok(v) => config_flag_on(&v),
        Err(_) => false,
    };
    if !on {
        return;
    }
    let id = match client.session_most_recent().await {
        Ok(v) => v
            .get("session_id")
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string()),
        Err(_) => None,
    };
    let Some(id) = id else {
        return;
    };
    let created = state.lock().await.session_id.clone();
    if created.as_deref() == Some(id.as_str()) {
        resume_session(state, client, &id).await;
        return;
    }
    if let Some(old) = created {
        let _ = client.close_session(&old).await;
    }
    resume_session(state, client, &id).await;
    state.lock().await.set_toast("auto-resumed recent session");
}

pub(crate) async fn redirect_turn(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    arg: &str,
) {
    let text = arg.trim();
    if text.is_empty() {
        state.lock().await.set_toast("usage: /redirect <text>");
        return;
    }
    let (sid, generating) = {
        let s = state.lock().await;
        (s.session_id.clone(), s.is_generating)
    };
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session");
        return;
    };
    if !generating {
        state
            .lock()
            .await
            .set_toast("no live turn · /steer queues instead");
        return;
    }
    match client.session_redirect(&sid, text).await {
        Ok(v) => {
            let status = v
                .get("status")
                .and_then(|x| x.as_str())
                .unwrap_or("redirected");
            state.lock().await.set_toast(format!("redirect {status}"));
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("redirect · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn move_workspace(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    arg: &str,
) {
    let cwd = arg.trim();
    if cwd.is_empty() {
        state.lock().await.set_toast("usage: /workspace <cwd>");
        return;
    }
    let key = {
        let s = state.lock().await;
        if !s.session_key.is_empty() {
            s.session_key.clone()
        } else {
            s.session_id.clone().unwrap_or_default()
        }
    };
    if key.is_empty() {
        state.lock().await.set_toast("no session");
        return;
    }
    match client.session_workspace_move(&key, cwd).await {
        Ok(v) => {
            let resolved = v
                .get("cwd")
                .and_then(|x| x.as_str())
                .unwrap_or(cwd)
                .to_string();
            let mut s = state.lock().await;
            s.metrics.cwd = resolved.clone();
            let (repo, branch) = crate::platform::probe_git_repo_branch(&resolved);
            s.metrics.git_repo = repo;
            s.metrics.git_branch = branch;
            s.set_toast(format!("workspace {resolved}"));
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("workspace · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn run_cli(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>, arg: &str) {
    let argv: Vec<String> = arg.split_whitespace().map(|s| s.to_string()).collect();
    if argv.is_empty() {
        state.lock().await.set_toast("usage: /cli <argv…>");
        return;
    }
    state.lock().await.set_toast("cli.exec…");
    match client.cli_exec(&argv).await {
        Ok(v) => {
            if v.get("blocked").and_then(|x| x.as_bool()) == Some(true) {
                let hint = v.get("hint").and_then(|x| x.as_str()).unwrap_or("blocked");
                state.lock().await.set_toast(hint);
                return;
            }
            let out = v
                .get("output")
                .and_then(|x| x.as_str())
                .unwrap_or("(no output)");
            let code = v.get("code").and_then(|x| x.as_i64()).unwrap_or(0);
            state
                .lock()
                .await
                .open_peek(format!("cli  {code}"), out.to_string(), None);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("cli · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn request_handoff(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    arg: &str,
) {
    let platform = arg.trim().to_ascii_lowercase();
    if platform.is_empty() {
        state
            .lock()
            .await
            .set_toast("usage: /handoff <telegram|discord|slack|whatsapp|…>");
        return;
    }
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session");
        return;
    };
    if state.lock().await.is_generating {
        state
            .lock()
            .await
            .set_toast("session busy — wait, then /handoff");
        return;
    }
    match client.handoff_request(&sid, &platform).await {
        Ok(v) => {
            let home = v.get("home_name").and_then(|x| x.as_str()).unwrap_or("");
            state.lock().await.set_toast(if home.is_empty() {
                format!("handoff queued · {platform}")
            } else {
                format!("handoff queued · {platform} · {home}")
            });
            let st = Arc::clone(state);
            let cl = Arc::clone(client);
            tokio::spawn(async move {
                for _ in 0..45 {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    match cl.handoff_state(&sid).await {
                        Ok(v) => {
                            let status = v.get("state").and_then(|x| x.as_str()).unwrap_or("");
                            if status == "completed" {
                                st.lock().await.set_toast("handoff completed");
                                return;
                            }
                            if status == "failed" {
                                let err = v
                                    .get("error")
                                    .and_then(|x| x.as_str())
                                    .unwrap_or("handoff failed");
                                st.lock().await.set_toast(err);
                                return;
                            }
                        }
                        Err(e) => {
                            st.lock()
                                .await
                                .set_toast(format!("handoff · {}", optimistic::brief_err(&e)));
                            return;
                        }
                    }
                }
                let _ = cl.handoff_fail(&sid, "poll timed out").await;
                st.lock()
                    .await
                    .set_toast("handoff timeout · gateway watcher may still finish");
            });
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("handoff · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn create_profile(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    name: &str,
    clone_from: Option<&str>,
) {
    state.lock().await.set_toast(format!("creating {name}…"));
    match client.profiles_create(name, clone_from).await {
        Ok(v) => {
            let shown = v.get("name").and_then(|x| x.as_str()).unwrap_or(name);
            state.lock().await.set_toast(format!("profile {shown}"));
            refresh_profiles(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("profile · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn peek_profile(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    name: &str,
) {
    match client.profiles_describe(name).await {
        Ok(v) => {
            let body = format_profile_describe(&v);
            state
                .lock()
                .await
                .open_peek(format!("profile  {name}"), body, None);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("profile · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn save_replay(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    label: &str,
) {
    refresh_agents(state, client).await;
    let (sid, rows, default_label) = {
        let s = state.lock().await;
        (
            s.session_id.clone().unwrap_or_default(),
            s.agent_rows.clone(),
            s.agent_rows
                .iter()
                .filter(|r| r.kind == "subagent")
                .take(2)
                .map(|r| r.title.as_str())
                .collect::<Vec<_>>()
                .join(" · "),
        )
    };
    let subagents = spawn_subagents_from_rows(&rows);
    if subagents.is_empty() {
        state
            .lock()
            .await
            .set_toast("no subagents this turn to save");
        return;
    }
    let label = if label.is_empty() {
        default_label
    } else {
        label.to_string()
    };
    match client.spawn_tree_save(&sid, &label, subagents).await {
        Ok(v) => {
            let path = v.get("path").and_then(|x| x.as_str()).unwrap_or("");
            state.lock().await.set_toast(if path.is_empty() {
                "spawn tree saved".into()
            } else {
                format!("saved {path}")
            });
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("replay save · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn configure_tools(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    action: &str,
    names: Vec<String>,
) {
    let sid = state.lock().await.session_id.clone().unwrap_or_default();
    match client.tools_configure(&sid, action, &names).await {
        Ok(v) => {
            let msg = format_tools_configure(&v, action);
            state.lock().await.set_toast(msg);
            refresh_tools(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("tools · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn toggle_plugin(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    ident: &str,
    enable: bool,
) {
    match client
        .plugins_manage("toggle", json!({ "key": ident, "enable": enable }))
        .await
    {
        Ok(v) => {
            let unchanged = v.get("unchanged").and_then(|x| x.as_bool()) == Some(true);
            let verb = if enable { "enabled" } else { "disabled" };
            state.lock().await.set_toast(if unchanged {
                format!("{ident} already {verb}")
            } else {
                format!("{verb} {ident}")
            });
            refresh_plugins(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("plugins · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn disconnect_model(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    slug: &str,
) {
    match client.model_disconnect(slug).await {
        Ok(v) => {
            let name = v.get("name").and_then(|x| x.as_str()).unwrap_or(slug);
            state.lock().await.set_toast(format!("disconnected {name}"));
            refresh_models(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("disconnect · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn set_session_hidden(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    id: &str,
    hidden: bool,
) {
    match client.session_set_hidden(id, hidden).await {
        Ok(_) => {
            state.lock().await.set_toast(if hidden {
                format!("hidden {id}")
            } else {
                format!("unhidden {id}")
            });
            refresh_sessions(state, client).await;
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("hide · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn react_last(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    arg: &str,
) {
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session");
        return;
    };
    let raw = arg.trim();
    let lower = raw.to_ascii_lowercase();
    let emoji = match lower.as_str() {
        "" => Some("👍"),
        "clear" | "none" | "off" | "remove" => None,
        _ => Some(raw),
    };
    match client.message_react(&sid, "assistant", emoji).await {
        Ok(_) => {
            state.lock().await.set_toast(match emoji {
                Some(e) => format!("reacted {e}"),
                None => "reaction cleared".into(),
            });
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("react · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn steer_or_queue(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    arg: &str,
) {
    let text = arg.trim();
    if text.is_empty() {
        state.lock().await.set_toast("usage: /steer <text>");
        return;
    }
    let (sid, busy) = {
        let s = state.lock().await;
        (s.session_id.clone(), s.is_generating)
    };
    if busy {
        if let Some(sid) = sid {
            optimistic::spawn_steer(state, client, sid, text.to_string());
            state.lock().await.set_toast("steer queued");
        }
        return;
    }
    let mut s = state.lock().await;
    s.enqueue(text.to_string());
    s.set_toast("no active turn — queued");
}

pub(crate) async fn fork_session(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    name: &str,
) {
    let (sid, busy) = {
        let s = state.lock().await;
        (s.session_id.clone(), s.is_generating)
    };
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session to fork");
        return;
    };
    if busy {
        state
            .lock()
            .await
            .set_toast("session busy — wait or esc interrupt");
        return;
    }
    match client.session_branch(&sid, name.trim()).await {
        Ok(v) => {
            let new_id = v
                .get("session_id")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            if new_id.is_empty() {
                state.lock().await.set_toast("fork returned no session");
                return;
            }
            let title = v
                .get("title")
                .and_then(|s| s.as_str())
                .unwrap_or("branch")
                .to_string();
            let msgs = v
                .get("messages")
                .and_then(|m| m.as_array())
                .cloned()
                .unwrap_or_default();
            let parsed = parse_gateway_messages(&msgs);
            let _ = client.close_session(&sid).await;
            let mut s = state.lock().await;
            s.session_id = Some(new_id);
            s.messages = parsed;
            s.session_title = title.clone();
            s.scroll_from_bottom = 0;
            s.is_generating = false;
            s.set_toast(format!("forked → {title}"));
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("fork failed · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn undo_last_exchange(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>) {
    let (sid, busy) = {
        let s = state.lock().await;
        (s.session_id.clone(), s.is_generating)
    };
    let Some(sid) = sid else {
        state.lock().await.set_toast("nothing to undo");
        return;
    };
    if busy {
        state
            .lock()
            .await
            .set_toast("session busy — /interrupt then /undo");
        return;
    }
    match client.session_undo(&sid).await {
        Ok(v) => {
            let removed = v.get("removed").and_then(|x| x.as_u64()).unwrap_or(0);
            let mut s = state.lock().await;
            if removed > 0 {
                s.trim_last_user_turn();
                s.set_toast(format!("undid {removed} messages"));
            } else {
                s.set_toast("nothing to undo");
            }
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("undo failed · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn save_session(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>) {
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session to save");
        return;
    };
    match client.session_save(&sid).await {
        Ok(v) => {
            let file = v.get("file").and_then(|s| s.as_str()).unwrap_or("");
            let mut s = state.lock().await;
            if file.is_empty() {
                s.set_toast("save failed");
            } else {
                s.set_toast(format!("saved {file}"));
            }
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("save failed · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn refresh_tools(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    let sid = state.lock().await.session_id.clone().unwrap_or_default();
    let listed = match client.tools_list(&sid).await {
        Ok(v) => Ok(v),
        Err(_) => client.toolsets_list(&sid).await,
    };
    match listed {
        Ok(v) => {
            let mut s = state.lock().await;
            s.toolsets = parse_toolsets(&v);
            let n = s.toolsets.len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("tools list failed · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn refresh_plugins(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    let listed = match client.plugins_manage("list", json!({})).await {
        Ok(v) => Ok(v),
        Err(_) => client.plugins_list().await,
    };
    match listed {
        Ok(v) => {
            let mut s = state.lock().await;
            s.plugins = parse_plugins(&v);
            let n = s.plugins.len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        Err(e) => {
            state.lock().await.set_toast(format!(
                "plugins list failed · {}",
                optimistic::brief_err(&e)
            ));
        }
    }
}
