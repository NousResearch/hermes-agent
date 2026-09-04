//! Gateway list/refresh helpers used by slash commands and overlays.
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::optimistic;
use crate::rpc::GatewayClient;
use crate::slash::{merge_entries, parse_catalog, parse_complete_slash};
use crate::state::{
    merge_session_lists, parse_checkpoints, parse_live_sessions, parse_memory_payload,
    parse_profiles, parse_sessions, parse_skills_payload, ActiveView, AppState, BgStatus,
    DockEntry, SkillCard,
};

pub(crate) async fn refresh_context(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        return;
    };
    match client.context_breakdown(&sid).await {
        Ok(v) => {
            let mut s = state.lock().await;
            crate::ui::context::apply_breakdown(&mut s, &v);
            s.mark_dirty();
        }
        Err(e) => {
            tracing::debug!("session.context_breakdown unavailable: {e}");
        }
    }
}

pub(crate) async fn refresh_skills(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    match client.list_skills().await {
        Ok(v) => {
            let mut skills = parse_skills_payload(&v);
            let mut s = state.lock().await;
            if skills.is_empty() && !s.intro_skills.is_empty() {
                skills = s
                    .intro_skills
                    .iter()
                    .flat_map(|(cat, names)| {
                        names.iter().map(|n| SkillCard {
                            name: n.clone(),
                            category: cat.clone(),
                            description: String::new(),
                            preview: String::new(),
                        })
                    })
                    .collect();
            }
            crate::skill_md::enrich_skill_cards(&mut skills, &s.hermes_home, &s.metrics.cwd);
            s.skills = skills;
            let n = s.skills.len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        Err(e) => {
            let mut s = state.lock().await;
            if !s.intro_skills.is_empty() {
                let mut skills: Vec<SkillCard> = s
                    .intro_skills
                    .iter()
                    .flat_map(|(cat, names)| {
                        names.iter().map(|n| SkillCard {
                            name: n.clone(),
                            category: cat.clone(),
                            description: String::new(),
                            preview: String::new(),
                        })
                    })
                    .collect();
                crate::skill_md::enrich_skill_cards(&mut skills, &s.hermes_home, &s.metrics.cwd);
                s.skills = skills;
                let n = s.skills.len();
                s.clamp_modal(n);
                s.mark_dirty();
            } else {
                s.add_system(format!("skills.manage failed: {e}"));
            }
        }
    }
}

pub(crate) async fn refresh_slash_complete(
    state: &Mutex<AppState>,
    client: Option<&Arc<GatewayClient>>,
    text: &str,
) {
    let Some(c) = client else {
        return;
    };
    match c.complete_slash(text).await {
        Ok(v) => {
            let (items, from) = parse_complete_slash(&v);
            let mut s = state.lock().await;
            s.slash_gateway = items;
            s.slash_replace_from = from;
            let n = s.slash_ranked().len();
            if s.slash_selected >= n {
                s.slash_selected = n.saturating_sub(1);
            }
            s.mark_dirty();
        }
        Err(_) => {
            let mut s = state.lock().await;
            s.slash_gateway.clear();
        }
    }
}

pub(crate) async fn persist_paste_collapse(
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
    label: &str,
    text: &str,
) {
    if let Ok(v) = client.paste_collapse(text).await {
        if let Some(path) = v.get("path").and_then(|x| x.as_str()) {
            let mut s = state.lock().await;
            if let Some(chip) = s.paste_chips.iter_mut().find(|c| c.label == label) {
                chip.path = Some(path.to_string());
            }
        }
    }
}

pub(crate) async fn detect_drop_paste(
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
    text: &str,
) -> Option<String> {
    let sid = state.lock().await.session_id.clone()?;
    let v = client.input_detect_drop(&sid, text).await.ok()?;
    if v.get("matched").and_then(|x| x.as_bool()) != Some(true) {
        return None;
    }
    if v.get("is_image").and_then(|x| x.as_bool()) == Some(true) {
        if let Some(path) = v.get("path").and_then(|x| x.as_str()) {
            state
                .lock()
                .await
                .remember_image(std::path::PathBuf::from(path));
        }
    }
    let inserted = v.get("text").and_then(|x| x.as_str()).unwrap_or(text);
    state.lock().await.set_toast("dropped file attached");
    Some(inserted.to_string())
}

pub(crate) async fn apply_detect_drop(
    text: String,
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
) -> String {
    if !crate::complete::looks_like_dropped_path(text.trim()) {
        return text;
    }
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        return text;
    };
    match client.input_detect_drop(&sid, &text).await {
        Ok(v) if v.get("matched").and_then(|x| x.as_bool()) == Some(true) => {
            if v.get("is_image").and_then(|x| x.as_bool()) == Some(true) {
                if let Some(path) = v.get("path").and_then(|x| x.as_str()) {
                    state
                        .lock()
                        .await
                        .remember_image(std::path::PathBuf::from(path));
                }
            }
            v.get("text")
                .and_then(|x| x.as_str())
                .unwrap_or(&text)
                .to_string()
        }
        _ => text,
    }
}

pub(crate) async fn refresh_catalog(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    match client.commands_catalog().await {
        Ok(v) => {
            let merged = merge_entries(parse_catalog(&v));
            let mut s = state.lock().await;
            s.slash_catalog = merged;
            s.mark_dirty();
        }
        Err(e) => {
            tracing::debug!("commands.catalog unavailable: {e}");
        }
    }
}

pub(crate) async fn refresh_profiles(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    match client.list_profiles().await {
        Ok(v) => {
            let rows = parse_profiles(&v);
            let mut s = state.lock().await;
            s.profiles = rows;
            let n = s.picker_len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        Err(e) => {
            state
                .lock()
                .await
                .add_system(format!("profiles.list failed: {e}"));
        }
    }
}

pub(crate) async fn load_processes(
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
) -> serde_json::Value {
    let sid = state.lock().await.session_id.clone();
    if let Some(sid) = sid {
        if let Ok(v) = client.process_list(&sid).await {
            return v;
        }
    }
    client.list_agents().await.unwrap_or(json!({}))
}

pub(crate) async fn refresh_agents(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    if state.lock().await.agents_replay {
        return;
    }
    let processes = load_processes(state, client).await;
    match client.delegation_status().await {
        Ok(status) => {
            let paused = status
                .get("paused")
                .and_then(|p| p.as_bool())
                .unwrap_or(false);
            let depth = status
                .get("max_spawn_depth")
                .and_then(|x| x.as_u64())
                .unwrap_or(0);
            let conc = status
                .get("max_concurrent_children")
                .and_then(|x| x.as_u64())
                .unwrap_or(0);
            let mut s = state.lock().await;
            s.merge_agent_snapshot(&processes, &status);
            s.agents_paused = paused;
            s.agents_caps = format!("depth {depth} · concurrent {conc}");
            let n = s.picker_len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        Err(e) => {
            let mut s = state.lock().await;
            s.merge_agent_snapshot(&processes, &json!({}));
            if s.agent_rows.is_empty() {
                s.add_system(format!("delegation.status failed: {e}"));
            } else {
                let n = s.picker_len();
                s.clamp_modal(n);
                s.mark_dirty();
            }
        }
    }
}

pub(crate) async fn refresh_memory(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    match client.learning_frames().await {
        Ok(v) => {
            let (summary, nodes) = parse_memory_payload(&v);
            let mut s = state.lock().await;
            s.memory_summary = summary;
            s.memory_nodes = nodes;
            let n = s.picker_len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        Err(e) => {
            state
                .lock()
                .await
                .add_system(format!("learning.frames failed: {e}"));
        }
    }
}

pub(crate) async fn set_agents_paused(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    paused: bool,
) {
    {
        let mut s = state.lock().await;
        s.agents_paused = paused;
        s.set_toast(if paused {
            "subagent spawn paused"
        } else {
            "subagent spawn live"
        });
    }
    optimistic::spawn_agents_pause(state, client, paused);
}

pub(crate) fn selected_agent(s: &AppState) -> Option<&crate::state::AgentRow> {
    let idxs = if s.agents_steer {
        (0..s.agent_rows.len()).collect::<Vec<_>>()
    } else {
        s.filtered_agent_indices()
    };
    idxs.get(s.modal_selected)
        .and_then(|i| s.agent_rows.get(*i))
}

pub(crate) fn selected_agent_id(s: &AppState) -> Option<String> {
    selected_agent(s).map(|a| a.id.clone())
}

pub(crate) fn agent_peek_body(a: &crate::state::AgentRow) -> String {
    let mut parts = vec![a.title.clone(), format!("{} · {}", a.status, a.id)];
    if !a.model.is_empty() {
        parts.push(format!("model {}", a.model));
    }
    if !a.last_tool.is_empty() {
        parts.push(format!("now {}", a.last_tool));
    }
    if !a.notes.is_empty() {
        parts.push(String::new());
        parts.extend(a.notes.iter().cloned());
    }
    if !a.summary.is_empty() {
        parts.push(String::new());
        parts.push(a.summary.clone());
    }
    parts.join("\n")
}

pub(crate) async fn interrupt_subagent_ids(
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
    ids: Vec<String>,
) {
    let n = ids.len();
    state.lock().await.set_toast(format!("stopping {n}…"));
    let mut stopped = 0usize;
    for id in &ids {
        if let Ok(v) = client.interrupt_subagent(id).await {
            if v.get("found").and_then(|x| x.as_bool()).unwrap_or(true) {
                stopped += 1;
                if let Some(row) = state
                    .lock()
                    .await
                    .agent_rows
                    .iter_mut()
                    .find(|r| r.id == *id)
                {
                    if !crate::state::is_terminal_agent_status(&row.status) {
                        row.status = "interrupted".into();
                    }
                }
            }
        }
    }
    state
        .lock()
        .await
        .set_toast(format!("stopped {stopped}/{n}"));
    refresh_agents(state, client).await;
}

pub(crate) async fn steer_selected_subagent(
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
    id: &str,
    text: &str,
) {
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no active session");
        return;
    };
    state.lock().await.set_toast(format!("steering {id}…"));
    match client.steer_subagent(&sid, id, text).await {
        Ok(_) => {
            let mut s = state.lock().await;
            s.push_agent_note(id, &format!("steer: {text}"));
            s.set_toast(format!("steered {id}"));
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("steer failed · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn stop_dock_entry(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    entry: DockEntry,
) {
    let action = {
        let s = state.lock().await;
        match entry {
            DockEntry::Agent(i) => s.agent_rows.get(i).and_then(|a| {
                if a.is_running_process() {
                    Some(("proc", a.id.clone()))
                } else if a.is_live() {
                    Some(("agent", a.id.clone()))
                } else {
                    None
                }
            }),
            DockEntry::Bg(_) => None,
        }
    };
    match action {
        Some(("proc", id)) => kill_selected_process(state, client, &id).await,
        Some(("agent", id)) => interrupt_selected_subagent(state, client, &id).await,
        _ => {
            state.lock().await.set_toast("nothing to stop on that row");
        }
    }
}

pub(crate) async fn kill_selected_process(
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
    id: &str,
) {
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no active session");
        return;
    };
    state.lock().await.set_toast(format!("killing {id}…"));
    match client.process_kill(&sid, id).await {
        Ok(v) => {
            let st = v.get("status").and_then(|x| x.as_str()).unwrap_or("killed");
            let mut s = state.lock().await;
            if let Some(row) = s
                .agent_rows
                .iter_mut()
                .find(|r| r.id == id && r.is_process())
            {
                row.status = "exited".into();
                if let Some(out) = v.get("output").and_then(|x| x.as_str()) {
                    if !out.is_empty() {
                        row.output = out.to_string();
                    }
                }
            }
            s.set_toast(format!("{id} {st}"));
        }
        Err(e) => {
            state.lock().await.set_toast(format!(
                "still running · {} · x retries",
                optimistic::brief_err(&e)
            ));
        }
    }
    refresh_agents(state, client).await;
}

pub(crate) async fn stop_background_processes(
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
) {
    state.lock().await.set_toast("stopping processes…");
    match client.process_stop().await {
        Ok(v) => {
            let n = v.get("killed").and_then(|x| x.as_u64()).unwrap_or(0);
            let noun = if n == 1 { "process" } else { "processes" };
            let mut s = state.lock().await;
            for row in s.agent_rows.iter_mut().filter(|r| r.is_running_process()) {
                row.status = "exited".into();
            }
            s.add_system(format!("stopped {n} background {noun}"));
            s.set_toast(format!("stopped {n} {noun}"));
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("stop failed · {}", optimistic::brief_err(&e)));
        }
    }
    refresh_agents(state, client).await;
}

pub(crate) async fn interrupt_selected_subagent(
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
    id: &str,
) {
    state.lock().await.set_toast(format!("stopping {id}…"));
    match client.interrupt_subagent(id).await {
        Ok(v) => {
            let found = v.get("found").and_then(|x| x.as_bool()).unwrap_or(true);
            let mut s = state.lock().await;
            if found {
                if let Some(row) = s.agent_rows.iter_mut().find(|r| r.id == id) {
                    if !crate::state::is_terminal_agent_status(&row.status) {
                        row.status = "interrupted".into();
                    }
                }
            }
            s.set_toast(if found {
                format!("stopped {id}")
            } else {
                format!("{id} not running")
            });
        }
        Err(e) => {
            state.lock().await.set_toast(format!(
                "still running · {} · x retries",
                optimistic::brief_err(&e)
            ));
        }
    }
    refresh_agents(state, client).await;
}

#[derive(Debug)]
pub(crate) enum BgAction {
    None,
    Launch(String),
    Peek { title: String, body: String },
}

pub(crate) fn background_confirm(s: &mut AppState) -> BgAction {
    if s.modal_selected == 0 {
        let prompt = s.picker_filter.trim().to_string();
        if prompt.is_empty() {
            s.set_toast("usage: type a prompt · enter");
            return BgAction::None;
        }
        s.picker_filter.clear();
        s.mark_dirty();
        return BgAction::Launch(prompt);
    }
    let idxs = s.filtered_bg_indices();
    let Some(&i) = idxs.get(s.modal_selected.saturating_sub(1)) else {
        return BgAction::None;
    };
    let Some(t) = s.bg_tasks.get(i) else {
        return BgAction::None;
    };
    let title = format!("bg {}", t.id);
    let body = if t.result.is_empty() {
        if t.status == BgStatus::Running {
            format!("{}\n\n(still running)", t.prompt)
        } else {
            t.prompt.clone()
        }
    } else if t.prompt.is_empty() {
        t.result.clone()
    } else {
        format!("{}\n\n{}", t.prompt, t.result)
    };
    BgAction::Peek { title, body }
}

pub(crate) async fn apply_bg_action(
    action: BgAction,
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
) {
    match action {
        BgAction::None => {}
        BgAction::Launch(text) => {
            start_background(state, client, &text).await;
            let mut s = state.lock().await;
            s.active_view = ActiveView::Background;
            s.picker_filter.clear();
            s.modal_selected = if s.bg_tasks.is_empty() { 0 } else { 1 };
            s.mark_dirty();
        }
        BgAction::Peek { title, body } => {
            state.lock().await.open_peek(title, body, None);
        }
    }
}

pub(crate) async fn start_background(
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
    text: &str,
) {
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no active session");
        return;
    };
    match client.prompt_background(&sid, text).await {
        Ok(v) => {
            let id = v
                .get("task_id")
                .and_then(|x| x.as_str())
                .unwrap_or("bg")
                .to_string();
            let mut s = state.lock().await;
            s.start_bg_task(id.clone(), text.to_string());
            s.add_system(format!("bg {id} started"));
            s.set_toast(format!("bg {id} started"));
            if s.active_view == ActiveView::Background && !s.bg_tasks.is_empty() {
                s.modal_selected = 1;
            }
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("background failed · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn refresh_rollback(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no active session");
        return;
    };
    match client.rollback_list(&sid).await {
        Ok(v) => {
            let (enabled, rows) = parse_checkpoints(&v);
            let mut s = state.lock().await;
            s.checkpoints_enabled = enabled;
            s.checkpoints = rows;
            let n = s.picker_len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        Err(e) => {
            state.lock().await.set_toast(format!(
                "rollback list failed · {}",
                optimistic::brief_err(&e)
            ));
        }
    }
}

pub(crate) async fn load_rollback_diff(
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
    hash: &str,
) {
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        return;
    };
    match client.rollback_diff(&sid, hash).await {
        Ok(v) => {
            let body = v
                .get("rendered")
                .and_then(|x| x.as_str())
                .or_else(|| v.get("diff").and_then(|x| x.as_str()))
                .unwrap_or("")
                .to_string();
            let stat = v
                .get("stat")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let mut s = state.lock().await;
            s.rollback_diff = if stat.is_empty() {
                body
            } else {
                format!("{stat}\n\n{body}")
            };
            if s.rollback_diff.trim().is_empty() {
                s.set_toast("no changes since this checkpoint");
            }
            s.mark_dirty();
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("diff failed · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn apply_rollback(
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
    hash: &str,
) {
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        return;
    };
    match client.rollback_restore(&sid, hash, None).await {
        Ok(v) => {
            let ok = v.get("success").and_then(|x| x.as_bool()).unwrap_or(false);
            let mut s = state.lock().await;
            if !ok {
                let err = v
                    .get("error")
                    .or_else(|| v.get("message"))
                    .and_then(|x| x.as_str())
                    .unwrap_or("unknown error");
                s.set_toast(format!("rollback failed · {err}"));
                return;
            }
            let removed = v
                .get("history_removed")
                .and_then(|x| x.as_u64())
                .unwrap_or(0);
            if removed > 0 {
                s.trim_last_user_turn();
            }
            s.refresh_files();
            s.active_view = ActiveView::Chat;
            let short: String = hash.chars().take(10).collect();
            s.set_toast(format!("restored {short}"));
        }
        Err(e) => {
            state.lock().await.set_toast(format!(
                "rollback failed · {} · /interrupt if busy",
                optimistic::brief_err(&e)
            ));
        }
    }
}

pub(crate) async fn refresh_sessions(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    let current = state.lock().await.session_id.clone().unwrap_or_default();
    let live = match client.session_active_list(&current).await {
        Ok(v) => parse_live_sessions(&v),
        Err(_) => Vec::new(),
    };
    match client.list_sessions().await {
        Ok(v) => {
            let stored = parse_sessions(&v);
            let mut s = state.lock().await;
            s.sessions_list = merge_session_lists(live, stored);
            let n = s.sessions_list.len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        Err(e) => {
            if live.is_empty() {
                state
                    .lock()
                    .await
                    .add_system(format!("session.list failed: {e}"));
            } else {
                let mut s = state.lock().await;
                s.sessions_list = live;
                let n = s.sessions_list.len();
                s.clamp_modal(n);
                s.mark_dirty();
            }
        }
    }
}
