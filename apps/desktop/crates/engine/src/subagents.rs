//! `delegate_task` (tools/delegate_tool.py): the agent spawns child agents in
//! isolated contexts, each with its own saved session under
//! `<profile>/subagents/`, running in parallel (at most 3 at once). Children
//! cannot delegate further. Their tool approvals reach the user like the
//! parent's, and every change streams as an `Event::Subagent` for the Agents
//! page's spawn tree. The parent gets each child's final answer.
use crate::{CancellationToken, Config, Entry, Event, Session, Status};
use async_channel::Sender;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{future::Future, path::{Path, PathBuf}, pin::Pin};

pub const MAX_TASKS: usize = 6;
pub const MAX_CONCURRENT: usize = 3;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Progress {
    pub id: String, pub delegation_id: String, pub goal: String,
    /// queued | running | completed | failed | interrupted
    pub status: String,
    pub task_index: usize, pub task_count: usize, pub started_at: i64, pub updated_at: i64,
    pub summary: Option<String>, pub current_tool: Option<String>, pub tool_count: usize,
    pub preview: String, pub session_dir: PathBuf, pub model: String,
}

pub fn spec() -> Value {
    json!({"type":"function","function":{"name":"delegate_task","description":format!("Spawn up to {MAX_TASKS} subagents in isolated contexts to work in parallel ({MAX_CONCURRENT} at a time). Each child sees only its own goal and context, can use the same tools (not delegation), and returns a final answer. Use for independent, self-contained pieces of work."),
        "parameters":{"type":"object","properties":{"tasks":{"type":"array","minItems":1,"maxItems":MAX_TASKS,"items":{"type":"object","properties":{
            "goal":{"type":"string","description":"What this subagent should accomplish. Be specific and self-contained."},
            "context":{"type":"string","description":"Background THIS child needs: file paths, error messages, constraints."}},"required":["goal"]}}},"required":["tasks"]}}})
}

fn now() -> i64 { chrono::Utc::now().timestamp() }

/// Run the delegation and return the JSON result for the parent model.
/// Boxed with an explicit `Send` bound: a child turn can reach this tool again
/// (it does not, children never get it), and the erased type breaks that cycle.
pub(crate) fn delegate<'a>(args: &'a Value, config: Config, options: crate::models::GenerationOptions, profile_root: &'a Path, workspace: &'a Path,
    events: &'a Sender<Event>, cancel: CancellationToken) -> Pin<Box<dyn Future<Output = Value> + Send + 'a>> {
    Box::pin(delegate_inner(args, config, options, profile_root, workspace, events, cancel))
}
async fn delegate_inner(args: &Value, config: Config, options: crate::models::GenerationOptions, profile_root: &Path, workspace: &Path,
    events: &Sender<Event>, cancel: CancellationToken) -> Value {
    let tasks: Vec<(String, String)> = args["tasks"].as_array().into_iter().flatten()
        .filter_map(|t| Some((t["goal"].as_str()?.trim().to_owned(), t["context"].as_str().unwrap_or_default().trim().to_owned())))
        .filter(|(g, _)| !g.is_empty()).collect();
    if tasks.is_empty() { return json!({"error":"Give at least one task with a goal"}); }
    if tasks.len() > MAX_TASKS { return json!({"error":format!("At most {MAX_TASKS} tasks per delegation")}); }
    let delegation = uuid_like();
    let count = tasks.len();
    let mut progress: Vec<Progress> = tasks.iter().enumerate().map(|(i, (goal, _))| Progress {
        id: format!("{delegation}-{i}"), delegation_id: delegation.clone(), goal: goal.clone(), status: "queued".into(), task_index: i, task_count: count,
        started_at: now(), updated_at: now(), session_dir: profile_root.join("subagents").join(format!("{delegation}-{i}")), model: config.model.clone(), ..Default::default()
    }).collect();
    for p in &progress { let _ = events.send(Event::Subagent(p.clone())).await; }
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT));
    let runs = tasks.into_iter().zip(progress.iter_mut()).map(|((goal, context), progress)| {
        let (config, options, semaphore, cancel) = (config.clone(), options.clone(), semaphore.clone(), cancel.clone());
        async move {
            let _permit = semaphore.acquire().await;
            run_child(goal, context, progress, config, options, profile_root, workspace, events, cancel).await;
        }
    });
    futures_util::future::join_all(runs).await;
    json!({"delegation_id": delegation, "results": progress.iter().map(|p| json!({"task_index":p.task_index,"goal":p.goal,"status":p.status,"summary":p.summary,"tool_calls":p.tool_count})).collect::<Vec<_>>()})
}

fn uuid_like() -> String {
    let mut b = [0u8; 6];
    ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut b).expect("system RNG");
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[allow(clippy::too_many_arguments)]
async fn run_child(goal: String, context: String, progress: &mut Progress, config: Config, options: crate::models::GenerationOptions, profile_root: &Path, workspace: &Path,
    events: &Sender<Event>, cancel: CancellationToken) {
    let fail = |p: &mut Progress, why: String| { p.status = "failed".into(); p.summary = Some(why); p.updated_at = now(); };
    if let Err(e) = std::fs::create_dir_all(&progress.session_dir) { fail(progress, e.to_string()); let _ = events.send(Event::Subagent(progress.clone())).await; return; }
    let mut child = match Session::open_for_profile(&progress.session_dir, Some(workspace), profile_root) { Ok(s) => s, Err(e) => { fail(progress, format!("{e:#}")); let _ = events.send(Event::Subagent(progress.clone())).await; return; } };
    child.child = true;
    progress.status = "running".into(); progress.updated_at = now();
    let _ = events.send(Event::Subagent(progress.clone())).await;
    let prompt = format!("You are a subagent working for Hermes on one task. Work independently with your tools, then end with a concise final answer for the parent agent.\n\nGoal: {goal}{}",
        if context.is_empty() { String::new() } else { format!("\n\nContext:\n{context}") });
    let (tx, rx) = async_channel::unbounded();
    let parent = events.clone();
    let mut live = progress.clone();
    let forward = tokio::spawn(async move {
        while let Ok(event) = rx.recv().await {
            match event {
                Event::Approval { call, answer } => { let _ = parent.send(Event::Approval { call, answer }).await; }
                Event::Snapshot(history) => if let Some(turn) = history.turns.last() {
                    let calls: Vec<_> = turn.entries.iter().filter_map(|e| if let Entry::Assistant { calls, .. } = e { Some(calls) } else { None }).flatten().collect();
                    let done = turn.entries.iter().filter(|e| matches!(e, Entry::Tool { .. })).count();
                    live.tool_count = calls.len();
                    live.current_tool = calls.get(done).map(|c| c.name.clone());
                    if let Some(Entry::Assistant { text, .. }) = turn.entries.iter().rev().find(|e| matches!(e, Entry::Assistant { text, .. } if !text.is_empty())) {
                        live.preview = text.chars().rev().take(220).collect::<Vec<_>>().into_iter().rev().collect();
                    }
                    live.updated_at = now();
                    let _ = parent.send(Event::Subagent(live.clone())).await;
                },
                Event::Text(_) | Event::Subagent(_) => {}
            }
        }
    });
    let result = child.turn_with_options(config, options, prompt, vec![], tx, cancel).await;
    let _ = forward.await;
    let turn = child.history().turns.last().cloned();
    progress.tool_count = turn.as_ref().map_or(0, |t| t.entries.iter().filter(|e| matches!(e, Entry::Tool { .. })).count());
    progress.current_tool = None; progress.updated_at = now();
    let answer = turn.as_ref().and_then(|t| t.entries.iter().rev().find_map(|e| match e { Entry::Assistant { text, .. } if !text.trim().is_empty() => Some(text.clone()), _ => None }));
    match (result, turn.map(|t| t.status)) {
        (Ok(()), Some(Status::Complete)) => { progress.status = "completed".into(); progress.summary = answer.or(Some("(no answer)".into())); }
        (_, Some(Status::Interrupted)) => { progress.status = "interrupted".into(); progress.summary = answer; }
        (Err(e), _) => fail(progress, format!("{e:#}")),
        (_, Some(Status::Failed(why))) => fail(progress, why),
        _ => fail(progress, "The subagent stopped without an answer".into()),
    }
    let _ = events.send(Event::Subagent(progress.clone())).await;
}
