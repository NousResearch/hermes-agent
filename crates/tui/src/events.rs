use crate::platform::probe_git_repo_branch;
use crate::rpc::types::JsonRpcEvent;
use crate::state::{
    activity_from_thinking, count_mcp_connected, parse_grouped_names, parse_todos, tasks_blob,
    AppState, ChatMessage, ClarifyQuestion, ClarifyRequest, MessageRole, TaskItem,
};

/// Apply a `tui_gateway` event onto `AppState`.
///
/// Event names follow the official Ink client (`createGatewayEventHandler.ts`).
pub fn apply_event(state: &mut AppState, evt: &JsonRpcEvent) {
    if let Some(sid) = evt.params.session_id.as_deref() {
        if let Some(ours) = state.session_id.as_deref() {
            if sid != ours {
                return;
            }
        }
    }

    match evt.params.event_type.as_str() {
        "message.delta" => {
            let text = payload_str(&evt.params.payload, "text")
                .or_else(|| payload_str(&evt.params.payload, "rendered"));
            if let Some(text) = text {
                state.append_assistant_delta(text);
            }
        }
        "thinking.delta" => {
            // Spinner/status only — kaomoji verbs stay on the turn bar, not in chat.
            if let Some(text) = payload_str(&evt.params.payload, "text") {
                if state.is_generating && !text.is_empty() {
                    state.metrics.activity = activity_from_thinking(text);
                    state.mark_dirty();
                }
            }
        }
        "reasoning.delta" => {
            if let Some(text) = payload_str(&evt.params.payload, "text") {
                if !text.is_empty() {
                    state.append_reasoning_delta(text);
                }
            }
        }
        "reasoning.available" => {
            if let Some(text) = payload_str(&evt.params.payload, "text") {
                if !text.is_empty() {
                    state.append_reasoning_delta(text);
                }
            }
        }
        "message.start" => {
            state.is_generating = true;
            if state.metrics.activity.is_empty() {
                state.metrics.activity = "writing".into();
            }
            state.mark_dirty();
        }
        "status.update" => {
            if let Some(text) = payload_str(&evt.params.payload, "text") {
                let kind = payload_str(&evt.params.payload, "kind").unwrap_or("status");
                if kind == "error" {
                    state.add_system(text);
                }
                let brief: String = text.chars().take(48).collect();
                if !brief.is_empty() {
                    state.metrics.activity = brief;
                }
                state.mark_dirty();
            }
        }
        "tool.progress" => {
            let preview = payload_str(&evt.params.payload, "preview")
                .or_else(|| payload_str(&evt.params.payload, "text"))
                .or_else(|| payload_str(&evt.params.payload, "chunk"));
            if let Some(preview) = preview {
                let name = payload_str(&evt.params.payload, "name");
                let tool_id = payload_str(&evt.params.payload, "tool_id");
                state.append_running_tool_output(name, tool_id, preview);
                let brief: String = crate::ui::stream::strip_ansi(preview)
                    .lines()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("")
                    .chars()
                    .take(48)
                    .collect();
                if !brief.trim().is_empty() {
                    state.metrics.activity = brief;
                }
            }
        }
        "tool.start" | "tool.started" => {
            let payload = evt.params.payload.as_ref();
            let name = payload
                .and_then(|p| p.get("name").and_then(|n| n.as_str()))
                .unwrap_or("tool");
            let tool_id = payload
                .and_then(|p| p.get("tool_id").and_then(|n| n.as_str()))
                .map(|s| s.to_string());
            state.metrics.active_tool = Some(name.to_string());

            let structured = payload
                .and_then(|p| p.get("args"))
                .filter(|v| v.is_object() || v.is_array());
            let mut args = if let Some(v) = structured {
                v.to_string()
            } else {
                payload
                    .and_then(|p| {
                        p.get("args_text")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or_default()
            };
            let todo_tool = name.to_ascii_lowercase().contains("todo");
            let cap = if todo_tool || structured.is_some() {
                16 * 1024
            } else {
                500
            };
            crate::tips::truncate_utf8(&mut args, cap);

            state.messages.push(ChatMessage {
                id: uuid::Uuid::new_v4().to_string(),
                role: MessageRole::Tool {
                    name: name.to_string(),
                    status: "running...".to_string(),
                    tool_id,
                },
                content: args,
                timestamp: chrono::Local::now(),
                output: String::new(),
                is_streaming: false,
            });
            apply_todo_payload(state, payload);
            state.freeze_thought();
            state.follow_latest_tool();
            state.mark_dirty();
        }
        "tool.complete" => {
            let payload = evt.params.payload.as_ref();
            let name = payload.and_then(|p| p.get("name").and_then(|n| n.as_str()));
            let tool_id = payload.and_then(|p| p.get("tool_id").and_then(|n| n.as_str()));
            let error = payload
                .and_then(|p| p.get("error"))
                .map(|e| !e.is_null() && e.as_bool() != Some(false) && e.as_str() != Some(""))
                .unwrap_or(false);
            apply_todo_payload(state, payload);
            if let Some(args) = payload
                .and_then(|p| p.get("args"))
                .filter(|v| v.is_object())
            {
                if let Some(last) = find_running_tool_mut(state, name, tool_id) {
                    if last.content.is_empty() || last.content.ends_with('…') {
                        last.content = args.to_string();
                    }
                }
            }
            if let Some(diff) = payload.and_then(|p| p.get("inline_diff").and_then(|v| v.as_str()))
            {
                if let Some(last) = find_running_tool_mut(state, name, tool_id) {
                    if !diff.trim().is_empty() {
                        last.output = diff.to_string();
                    }
                }
            }
            if let Some(result) = payload.and_then(|p| {
                p.get("result_text")
                    .or_else(|| p.get("summary"))
                    .and_then(|v| v.as_str())
            }) {
                if let Some(last) = find_running_tool_mut(state, name, tool_id) {
                    if last.output.is_empty() {
                        last.output = result.to_string();
                    }
                    if last.content.is_empty() {
                        last.content = result.to_string();
                    }
                }
            }
            let duration_s = payload.and_then(|p| {
                p.get("duration_s")
                    .or_else(|| p.get("duration"))
                    .and_then(|v| v.as_f64())
            });
            state.complete_tool(name, tool_id, error, duration_s);
        }
        "session.info" => apply_session_info(state, evt.params.payload.as_ref()),
        "compression:start" | "compression.start" => {
            state.begin_compaction();
        }
        "compression:end" | "compression.end" => {
            state.end_compaction();
        }
        "message.complete" | "turn.complete" => {
            state.finish_streaming();
            if let Some(usage) = evt.params.payload.as_ref().and_then(|p| p.get("usage")) {
                apply_usage(state, usage);
            }
        }
        "approval.request" => {
            let p = evt.params.payload.as_ref();
            let description = p
                .and_then(|v| v.get("description").and_then(|d| d.as_str()))
                .unwrap_or("dangerous command")
                .to_string();
            let command = p
                .and_then(|v| v.get("command").and_then(|d| d.as_str()))
                .unwrap_or("")
                .to_string();
            let allow_permanent = p
                .and_then(|v| v.get("allow_permanent").and_then(|d| d.as_bool()))
                .unwrap_or(true);
            let request_id = p
                .and_then(|v| v.get("request_id").and_then(|d| d.as_str()))
                .map(|s| s.to_string());
            state.pending_approval = Some(crate::state::ApprovalRequest {
                description: description.clone(),
                command,
                allow_permanent,
                request_id,
            });
            state.want_attention = true;
            state.set_toast("Approval needed");
            state.mark_dirty();
        }
        "clarify.request" => {
            let p = evt.params.payload.as_ref();
            let request_id = p
                .and_then(|v| v.get("request_id").and_then(|d| d.as_str()))
                .unwrap_or("")
                .to_string();
            let answered = p.and_then(|v| v.get("answers")).and_then(|v| v.as_object());
            let mut questions = p
                .and_then(|v| v.get("questions").and_then(|q| q.as_array()))
                .map(|rows| {
                    rows.iter()
                        .filter_map(|row| {
                            let qid = row.get("qid")?.as_str()?.trim();
                            let question = row.get("question")?.as_str()?.trim();
                            if qid.is_empty()
                                || question.is_empty()
                                || answered.is_some_and(|a| a.contains_key(qid))
                            {
                                return None;
                            }
                            Some(clarify_question(row, Some(qid.to_string()), question))
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if questions.is_empty() && p.and_then(|v| v.get("questions")).is_none() {
                let value = p.unwrap_or(&serde_json::Value::Null);
                let question = value
                    .get("question")
                    .and_then(|v| v.as_str())
                    .unwrap_or("clarify")
                    .trim();
                questions.push(clarify_question(value, None, question));
            }
            if !questions.is_empty() {
                state.pending_clarify = Some(ClarifyRequest {
                    request_id,
                    questions,
                    active: 0,
                });
                state.set_toast("Clarify");
            }
        }
        "sudo.request" => {
            let p = evt.params.payload.as_ref();
            state.pending_secret = Some(crate::state::SecretRequest {
                kind: crate::state::SecretKind::Sudo,
                request_id: p
                    .and_then(|v| v.get("request_id").and_then(|d| d.as_str()))
                    .unwrap_or("")
                    .to_string(),
                prompt: "sudo password".into(),
                buffer: String::new(),
            });
            state.set_toast("sudo");
        }
        "secret.request" => {
            let p = evt.params.payload.as_ref();
            state.pending_secret = Some(crate::state::SecretRequest {
                kind: crate::state::SecretKind::Secret,
                request_id: p
                    .and_then(|v| v.get("request_id").and_then(|d| d.as_str()))
                    .unwrap_or("")
                    .to_string(),
                prompt: p
                    .and_then(|v| {
                        v.get("prompt")
                            .or_else(|| v.get("env_var"))
                            .and_then(|d| d.as_str())
                    })
                    .unwrap_or("secret")
                    .to_string(),
                buffer: String::new(),
            });
            state.set_toast("secret");
        }
        "gateway.exit" => {
            state.add_system("gateway process exited");
            state.finish_streaming();
        }
        "gateway.stderr" => {
            if let Some(line) = payload_str(&evt.params.payload, "line") {
                tracing::debug!(target: "tui_gateway", "{line}");
            }
        }
        "agent.terminal.output" => {
            let id = payload_str(&evt.params.payload, "process_id").unwrap_or("");
            let chunk = payload_str(&evt.params.payload, "chunk").unwrap_or("");
            if !id.is_empty() && !chunk.is_empty() {
                state.append_process_output(id, chunk);
            }
        }
        "terminal.close" => {
            if let Some(id) = payload_str(&evt.params.payload, "process_id") {
                state.close_process(id);
            }
        }
        "subagent.spawn_requested" => {
            if let Some(p) = evt.params.payload.as_ref() {
                state.upsert_subagent(p, Some("queued"), true);
                state.nudge_agents();
            }
        }
        "subagent.start" => {
            if let Some(p) = evt.params.payload.as_ref() {
                state.upsert_subagent(p, Some("running"), true);
                state.nudge_agents();
            }
        }
        "subagent.thinking" => {
            if let Some(p) = evt.params.payload.as_ref() {
                if state.upsert_subagent(p, Some("running"), false) {
                    if let Some(text) = p.get("text").and_then(|x| x.as_str()) {
                        let id = AppState::subagent_payload_id(p);
                        state.push_agent_thought(&id, text);
                    }
                }
            }
        }
        "subagent.tool" => {
            if let Some(p) = evt.params.payload.as_ref() {
                if state.upsert_subagent(p, Some("running"), false) {
                    let name = p
                        .get("tool_name")
                        .and_then(|x| x.as_str())
                        .unwrap_or("tool");
                    let preview = p
                        .get("tool_preview")
                        .or_else(|| p.get("text"))
                        .and_then(|x| x.as_str())
                        .unwrap_or("");
                    let line = if preview.is_empty() {
                        name.to_string()
                    } else {
                        format!("{name}({preview})")
                    };
                    let id = AppState::subagent_payload_id(p);
                    state.push_agent_tool(&id, &line);
                }
            }
        }
        "subagent.progress" => {
            if let Some(p) = evt.params.payload.as_ref() {
                if state.upsert_subagent(p, Some("running"), false) {
                    if let Some(text) = p.get("text").and_then(|x| x.as_str()) {
                        let id = AppState::subagent_payload_id(p);
                        state.push_agent_note(&id, text);
                    }
                }
            }
        }
        "subagent.complete" => {
            if let Some(p) = evt.params.payload.as_ref() {
                let status = p
                    .get("status")
                    .and_then(|x| x.as_str())
                    .unwrap_or("completed");
                state.upsert_subagent(p, Some(status), false);
            }
        }
        "background.complete" => {
            let id = payload_str(&evt.params.payload, "task_id").unwrap_or("bg");
            let text = payload_str(&evt.params.payload, "text").unwrap_or("");
            state.complete_bg_task(id, text);
            let preview: String = text.chars().take(80).collect();
            state.add_system(format!("bg {id} done\n{text}"));
            state.set_toast(if preview.is_empty() {
                format!("bg {id} done")
            } else {
                format!("bg {id} · {preview}")
            });
        }
        "gateway.ready" => {
            if let Some(ver) = payload_str(&evt.params.payload, "version") {
                state.metrics.hermes_version = ver.to_string();
            }
            state.mark_dirty();
        }
        "session.usage" => {
            if let Some(usage) = evt.params.payload.as_ref().and_then(|p| p.get("usage")) {
                apply_usage(state, usage);
            } else if let Some(p) = evt.params.payload.as_ref() {
                apply_usage(state, p);
            }
            state.mark_dirty();
        }
        "notification.show" => {
            if let Some(text) = payload_str(&evt.params.payload, "text") {
                if !text.is_empty() {
                    state.set_toast(text);
                    if !state.is_generating {
                        state.add_system(text);
                    }
                }
            }
        }
        "notification.clear" => {
            if let Some(toast) = state.metrics.toast_message.as_ref() {
                let key = payload_str(&evt.params.payload, "key").unwrap_or("");
                if key.is_empty() || toast.text.contains(key) {
                    state.metrics.toast_message = None;
                    state.mark_dirty();
                }
            }
        }
        "secret.expire" | "sudo.expire" => {
            let rid = payload_str(&evt.params.payload, "request_id").unwrap_or("");
            if let Some(sec) = state.pending_secret.as_ref() {
                if rid.is_empty() || sec.request_id == rid {
                    state.pending_secret = None;
                    state.set_toast("prompt expired");
                }
            }
        }
        "review.summary" => {
            if let Some(text) = payload_str(&evt.params.payload, "text") {
                let t = text.trim();
                if !t.is_empty() {
                    state.add_system(t);
                }
            }
        }
        "error" => {
            let text = payload_str(&evt.params.payload, "message")
                .or_else(|| payload_str(&evt.params.payload, "error"))
                .unwrap_or("error");
            state.add_system(text);
            state.set_toast(crate::optimistic::brief_err(&text));
        }
        "gateway.protocol_error" | "gateway.start_timeout" => {
            if !state.protocol_warned {
                state.protocol_warned = true;
                state.add_system("protocol noise detected · /logs to inspect");
            }
            if let Some(preview) = payload_str(&evt.params.payload, "preview")
                .or_else(|| payload_str(&evt.params.payload, "reason"))
            {
                let brief: String = preview.chars().take(120).collect();
                if !brief.is_empty() {
                    state.set_toast(format!("protocol · {brief}"));
                }
            }
        }
        "skin.changed" => {
            let name = payload_str(&evt.params.payload, "name")
                .or_else(|| payload_str(&evt.params.payload, "id"))
                .or_else(|| payload_str(&evt.params.payload, "skin"));
            if let Some(name) = name {
                let p = crate::ui::theme::lookup(name);
                if p.id.eq_ignore_ascii_case(name) || p.label.eq_ignore_ascii_case(name) {
                    crate::ui::theme::apply(p);
                    state.theme_id = p.id.to_string();
                    state.set_toast(format!("theme · {}", p.label));
                } else {
                    state.set_toast(format!("skin · {name} (native palette unchanged)"));
                }
            }
        }
        _ => {}
    }
}

fn apply_todo_payload(state: &mut AppState, payload: Option<&serde_json::Value>) {
    let Some(parsed) = collect_todos(payload) else {
        return;
    };
    let blob = tasks_blob(&parsed);
    state.tasks = parsed;
    if let Some(last) = state.messages.iter_mut().rev().find(|m| {
        matches!(&m.role, MessageRole::Tool { name, .. } if name.to_ascii_lowercase().contains("todo"))
    }) {
        last.content = blob;
    }
}

fn collect_todos(payload: Option<&serde_json::Value>) -> Option<Vec<TaskItem>> {
    let p = payload?;
    if p.get("todos").is_some() {
        return Some(parse_todos(p));
    }
    for key in ["args", "result"] {
        if let Some(v) = p.get(key) {
            if let Some(parsed) = collect_todos(Some(v)) {
                return Some(parsed);
            }
        }
    }
    for key in ["args_text", "result_text", "summary"] {
        if let Some(s) = p.get(key).and_then(|v| v.as_str()) {
            let parsed = crate::ui::stream::todos_from_content(s);
            if !parsed.is_empty() {
                return Some(parsed);
            }
        }
    }
    None
}

fn clarify_question(
    value: &serde_json::Value,
    qid: Option<String>,
    question: &str,
) -> ClarifyQuestion {
    let choices = value
        .get("choices")
        .and_then(|c| c.as_array())
        .map(|rows| {
            rows.iter()
                .filter_map(|choice| choice.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    ClarifyQuestion {
        qid,
        question: question.to_string(),
        multi_select: !choices.is_empty()
            && value.get("multi_select").and_then(|v| v.as_bool()) == Some(true),
        choices,
        selected: 0,
        selected_indices: std::collections::HashSet::new(),
        typed: String::new(),
    }
}

fn find_running_tool_mut<'a>(
    state: &'a mut AppState,
    name: Option<&str>,
    tool_id: Option<&str>,
) -> Option<&'a mut ChatMessage> {
    state
        .messages
        .iter_mut()
        .rev()
        .find(|message| match &message.role {
            MessageRole::Tool {
                name: candidate_name,
                status,
                tool_id: candidate_id,
            } if status.contains("running") => {
                let id_matches = match (tool_id, candidate_id.as_deref()) {
                    (Some(expected), Some(actual)) => expected == actual,
                    (Some(_), None) => false,
                    (None, _) => true,
                };
                let name_matches = name.is_none_or(|expected| expected == candidate_name);
                id_matches && name_matches
            }
            _ => false,
        })
}

fn payload_str<'a>(payload: &'a Option<serde_json::Value>, key: &str) -> Option<&'a str> {
    payload.as_ref()?.get(key)?.as_str()
}

fn apply_session_info(state: &mut AppState, payload: Option<&serde_json::Value>) {
    let Some(p) = payload else { return };
    if let Some(model) = p.get("model").and_then(|m| m.as_str()) {
        if !model.is_empty() {
            state.metrics.active_model = model.to_string();
        }
    }
    if let Some(provider) = p.get("provider").and_then(|m| m.as_str()) {
        if !provider.is_empty() {
            state.metrics.active_provider = provider.to_string();
        }
    }
    if let Some(plan) = p.get("plan").and_then(|v| v.as_bool()) {
        state.apply_session_plan(plan);
    }
    if let Some(yolo) = p.get("yolo").and_then(|v| v.as_bool()) {
        state.apply_session_yolo(yolo);
    }
    if let Some(backend) = p.get("terminal_backend").and_then(|v| v.as_str()) {
        state.metrics.terminal_backend = backend.to_string();
    }
    if let Some(mode) = p.get("approval_mode").and_then(|v| v.as_str()) {
        state.metrics.approval_mode = mode.to_string();
    }
    if let Some(cwd) = p.get("cwd").and_then(|v| v.as_str()) {
        if cwd != state.metrics.cwd {
            state.metrics.cwd = cwd.to_string();
            let (repo, branch) = probe_git_repo_branch(cwd);
            state.metrics.git_repo = repo;
            state.metrics.git_branch = branch;
        }
    }
    if let Some(branch) = p.get("branch").and_then(|v| v.as_str()) {
        if !branch.is_empty() {
            state.metrics.git_branch = Some(branch.to_string());
        }
    }
    if let Some(ver) = p.get("version").and_then(|v| v.as_str()) {
        if !ver.is_empty() {
            state.metrics.hermes_version = ver.to_string();
        }
    }
    if let Some(date) = p.get("release_date").and_then(|v| v.as_str()) {
        state.release_date = date.to_string();
    }
    if let Some(title) = p.get("title").and_then(|v| v.as_str()) {
        if !title.is_empty() {
            state.session_title = title.to_string();
        }
    }
    if let Some(key) = p.get("session_key").and_then(|v| v.as_str()) {
        if !key.is_empty() {
            state.session_key = key.to_string();
        }
    }
    if let Some(tools) = p.get("tools") {
        state.intro_tools = parse_grouped_names(tools);
    }
    if let Some(skills) = p.get("skills") {
        state.intro_skills = parse_grouped_names(skills);
    }
    if let Some(mcp) = p.get("mcp_servers") {
        state.mcp_connected = count_mcp_connected(mcp);
    }
    let warn = p
        .get("install_warning")
        .or_else(|| p.get("credential_warning"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    if warn.is_some() {
        state.intro_warning = warn;
    }
    if let Some(usage) = p.get("usage") {
        apply_usage(state, usage);
    }
    if let Some(fast) = p.get("fast") {
        match fast {
            serde_json::Value::Bool(b) => {
                let _ = state.apply_config_value("fast", if *b { "on" } else { "off" });
            }
            serde_json::Value::String(s) => {
                let _ = state.apply_config_value("fast", s);
            }
            _ => {}
        }
    }
    if let Some(v) = p
        .get("indicator")
        .or_else(|| p.get("tui_status_indicator"))
        .and_then(|x| x.as_str())
    {
        let _ = state.apply_config_value("indicator", v);
    }
    if let Some(v) = p
        .get("statusbar")
        .or_else(|| p.get("tui_statusbar"))
        .and_then(|x| x.as_str())
    {
        let _ = state.apply_config_value("statusbar", v);
    }
    if let Some(v) = p
        .get("busy_input_mode")
        .or_else(|| p.get("busy"))
        .and_then(|x| x.as_str())
    {
        let _ = state.apply_config_value("busy", v);
    }
    if let Some(v) = p.get("density").and_then(|x| x.as_str()) {
        let _ = state.apply_config_value("density", v);
    }
    // Older/wrong field names the previous client looked for.
    if let Some(tokens) = p.get("total_tokens").and_then(|t| t.as_u64()) {
        state.metrics.total_tokens = tokens;
    }
    if let Some(cost) = p.get("estimated_cost_usd").and_then(|c| c.as_f64()) {
        state.metrics.estimated_cost_usd = cost;
    }
    state.mark_session_ready();
    state.mark_dirty();
}

fn apply_usage(state: &mut AppState, usage: &serde_json::Value) {
    if let Some(total) = usage.get("total").and_then(|t| t.as_u64()) {
        state.metrics.total_tokens = total;
    }
    if let Some(used) = usage.get("context_used").and_then(|t| t.as_u64()) {
        state.metrics.context_used = used;
    }
    if let Some(max) = usage.get("context_max").and_then(|t| t.as_u64()) {
        state.metrics.context_limit = max;
    }
    if let Some(cost) = usage
        .get("cost_usd")
        .or_else(|| usage.get("estimated_cost_usd"))
        .and_then(|c| c.as_f64())
    {
        state.metrics.estimated_cost_usd = cost;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rpc::types::{EventParams, JsonRpcEvent};
    use crate::state::PermissionMode;
    use serde_json::json;

    fn evt(ty: &str, payload: serde_json::Value) -> JsonRpcEvent {
        JsonRpcEvent {
            jsonrpc: "2.0".into(),
            method: "event".into(),
            params: EventParams {
                event_type: ty.into(),
                session_id: Some("abc".into()),
                payload: Some(payload),
            },
        }
    }

    #[test]
    fn session_info_reads_usage_not_legacy_fields() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        apply_event(
            &mut s,
            &evt(
                "session.info",
                json!({
                    "model": "x-ai/grok-4",
                    "yolo": true,
                    "terminal_backend": "docker",
                    "approval_mode": "manual",
                    "usage": { "total": 1200, "context_used": 800, "context_max": 128000, "cost_usd": 0.04 },
                    "fast": true,
                    "indicator": "ascii",
                    "busy_input_mode": "steer"
                }),
            ),
        );
        assert_eq!(s.metrics.active_model, "x-ai/grok-4");
        assert!(s.fast_mode);
        assert_eq!(s.indicator, crate::state::IndicatorStyle::Ascii);
        assert_eq!(s.busy_mode, crate::state::BusyMode::Steer);
        assert_eq!(s.metrics.permission_mode, PermissionMode::Yolo);
        assert_eq!(s.metrics.terminal_backend, "docker");
        assert_eq!(s.metrics.approval_mode, "manual");
        s.metrics.permission_mode = PermissionMode::Plan;
        apply_event(
            &mut s,
            &evt(
                "session.info",
                json!({ "yolo": false, "model": "x-ai/grok-4" }),
            ),
        );
        assert_eq!(s.metrics.permission_mode, PermissionMode::Plan);
        apply_event(
            &mut s,
            &evt("session.info", json!({ "plan": true, "yolo": false })),
        );
        assert_eq!(s.metrics.permission_mode, PermissionMode::Plan);
        apply_event(
            &mut s,
            &evt("session.info", json!({ "plan": false, "yolo": false })),
        );
        assert_eq!(s.metrics.permission_mode, PermissionMode::Manual);
        assert_eq!(s.metrics.total_tokens, 1200);
        assert_eq!(s.metrics.context_used, 800);
        assert_eq!(s.metrics.context_limit, 128000);
        assert!((s.metrics.estimated_cost_usd - 0.04).abs() < f64::EPSILON);
    }

    #[test]
    fn session_info_fills_intro_panel() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        apply_event(
            &mut s,
            &evt(
                "session.info",
                json!({
                    "model": "x-ai/grok-4",
                    "version": "0.20.5",
                    "release_date": "2026-04-01",
                    "tools": { "web_tools": ["web_search"], "files": ["read_file"] },
                    "skills": { "general": ["demo"] },
                    "mcp_servers": [
                        { "name": "git", "connected": true, "tools": 3, "transport": "stdio" },
                        { "name": "off", "connected": false, "tools": 0, "transport": "stdio" }
                    ]
                }),
            ),
        );
        assert_eq!(s.metrics.hermes_version, "0.20.5");
        assert_eq!(s.release_date, "2026-04-01");
        assert_eq!(s.mcp_connected, 1);
        assert!(s
            .intro_tools
            .iter()
            .any(|(k, v)| k == "web" && v.contains(&"web_search".into())));
        assert_eq!(s.intro_skills.len(), 1);
        assert!(s.session_ready);
    }

    #[test]
    fn ignores_other_session() {
        let mut s = AppState::new();
        s.session_id = Some("ours".into());
        apply_event(&mut s, &evt("message.delta", json!({ "text": "nope" })));
        // event helper stamps session abc
        assert!(s.messages.is_empty());
    }

    #[test]
    fn message_delta_appends() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        apply_event(&mut s, &evt("message.delta", json!({ "text": "hi" })));
        apply_event(&mut s, &evt("message.complete", json!({})));
        assert_eq!(s.messages.len(), 1);
        assert!(!s.is_generating);
    }

    #[test]
    fn tool_start_and_complete_update_transcript() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        s.start_turn("list files".into());
        apply_event(
            &mut s,
            &evt(
                "tool.start",
                json!({
                    "name": "list_dir",
                    "tool_id": "t1",
                    "args_text": "/var/tmp/hermes-tui-src"
                }),
            ),
        );
        assert_eq!(s.metrics.active_tool.as_deref(), Some("list_dir"));
        let tool = s
            .messages
            .iter()
            .find(|m| matches!(&m.role, MessageRole::Tool { name, .. } if name == "list_dir"))
            .expect("tool row");
        assert!(tool.content.contains("hermes-tui-src"));
        match &tool.role {
            MessageRole::Tool {
                status, tool_id, ..
            } => {
                assert!(status.contains("running"));
                assert_eq!(tool_id.as_deref(), Some("t1"));
            }
            other => panic!("{other:?}"),
        }
        apply_event(
            &mut s,
            &evt(
                "tool.complete",
                json!({
                    "name": "list_dir",
                    "tool_id": "t1",
                    "result_text": "src/"
                }),
            ),
        );
        let tool = s
            .messages
            .iter()
            .rev()
            .find(|m| matches!(&m.role, MessageRole::Tool { name, .. } if name == "list_dir"))
            .expect("completed tool");
        match &tool.role {
            MessageRole::Tool { status, .. } => assert!(!status.contains("running")),
            other => panic!("{other:?}"),
        }
        assert!(tool.output.contains("src/") || tool.content.contains("src/"));
    }

    #[test]
    fn tool_progress_appends_running_output() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        apply_event(
            &mut s,
            &evt(
                "tool.start",
                json!({
                    "name": "terminal",
                    "tool_id": "t-doc",
                    "args": { "command": "hermes doctor; printf hi" }
                }),
            ),
        );
        apply_event(
            &mut s,
            &evt(
                "tool.progress",
                json!({
                    "name": "terminal",
                    "tool_id": "t-doc",
                    "preview": "◆ Security Advisories\n  ✓ none"
                }),
            ),
        );
        let tool = s
            .messages
            .iter()
            .rev()
            .find(|m| matches!(&m.role, MessageRole::Tool { name, .. } if name == "terminal"))
            .expect("terminal");
        assert!(tool.output.contains("Security Advisories"));
        assert!(s.metrics.activity.contains("none") || s.metrics.activity.contains("Security"));
    }

    #[test]
    fn todo_tool_keeps_plan_items() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        apply_event(
            &mut s,
            &evt(
                "tool.start",
                json!({
                    "name": "todo",
                    "tool_id": "td1",
                    "args": {
                        "todos": [
                            {"id": "1", "content": "Inspect repository", "status": "in_progress"},
                            {"id": "2", "content": "Ship native TUI", "status": "pending"}
                        ]
                    }
                }),
            ),
        );
        assert_eq!(s.tasks.len(), 2);
        assert_eq!(s.tasks[0].title, "Inspect repository");
        let tool = s
            .messages
            .iter()
            .rev()
            .find(|m| matches!(&m.role, MessageRole::Tool { name, .. } if name == "todo"))
            .expect("todo tool");
        assert!(tool.content.contains("Inspect repository"));
        apply_event(
            &mut s,
            &evt(
                "tool.complete",
                json!({
                    "name": "todo",
                    "tool_id": "td1",
                    "todos": [
                        {"id": "1", "content": "Inspect repository", "status": "completed"},
                        {"id": "2", "content": "Ship native TUI", "status": "in_progress"}
                    ]
                }),
            ),
        );
        assert_eq!(s.tasks[0].status, crate::state::TaskStatus::Completed);
        assert_eq!(s.tasks[1].status, crate::state::TaskStatus::InProgress);
        let tool = s
            .messages
            .iter()
            .rev()
            .find(|m| matches!(&m.role, MessageRole::Tool { name, .. } if name == "todo"))
            .expect("todo tool");
        assert!(crate::ui::stream::todos_from_content(&tool.content).len() >= 2);
    }

    #[test]
    fn explicit_empty_todo_payload_clears_plan() {
        let mut s = AppState::new();
        s.tasks.push(TaskItem {
            id: "stale".into(),
            title: "stale".into(),
            status: crate::state::TaskStatus::Pending,
        });
        apply_event(
            &mut s,
            &evt(
                "tool.complete",
                json!({"name": "todo", "tool_id": "td", "todos": []}),
            ),
        );
        assert!(s.tasks.is_empty());
    }

    #[test]
    fn tool_completion_updates_the_matching_running_tool() {
        let mut s = AppState::new();
        for (name, id) in [("read_file", "one"), ("terminal", "two")] {
            apply_event(
                &mut s,
                &evt("tool.start", json!({"name": name, "tool_id": id})),
            );
        }
        apply_event(
            &mut s,
            &evt(
                "tool.complete",
                json!({"name": "read_file", "tool_id": "one", "result_text": "first result"}),
            ),
        );
        let first = s
            .messages
            .iter()
            .find(|m| matches!(&m.role, MessageRole::Tool { tool_id, .. } if tool_id.as_deref() == Some("one")))
            .expect("first tool");
        let second = s
            .messages
            .iter()
            .find(|m| matches!(&m.role, MessageRole::Tool { tool_id, .. } if tool_id.as_deref() == Some("two")))
            .expect("second tool");
        assert_eq!(first.output, "first result");
        assert!(second.output.is_empty());
    }

    #[test]
    fn approval_request_sets_pending() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        apply_event(
            &mut s,
            &evt(
                "approval.request",
                json!({
                    "description": "run shell",
                    "command": "rm -rf /tmp/x",
                    "allow_permanent": true
                }),
            ),
        );
        let req = s.pending_approval.expect("pending");
        assert_eq!(req.description, "run shell");
        assert!(s.want_attention);
        assert_eq!(req.command, "rm -rf /tmp/x");
    }

    #[test]
    fn thinking_delta_is_turn_bar_not_chat() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        s.start_turn("q".into());
        apply_event(
            &mut s,
            &evt(
                "thinking.delta",
                json!({ "text": "( •_•)>⌐■-■ contemplating...Online and ready." }),
            ),
        );
        assert_eq!(s.metrics.activity, "contemplating");
        assert!(
            !s.messages.iter().any(|m| m.role == MessageRole::Reasoning),
            "kaomoji wait status must not become a thought row"
        );
        apply_event(
            &mut s,
            &evt("reasoning.delta", json!({ "text": "considering the repo" })),
        );
        assert!(s
            .messages
            .iter()
            .any(|m| m.role == MessageRole::Reasoning && m.content.contains("considering")));
    }

    #[test]
    fn status_update_sets_activity() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        apply_event(
            &mut s,
            &evt("status.update", json!({ "text": "compiling tools…" })),
        );
        assert!(s.metrics.activity.contains("compiling"));
    }

    #[test]
    fn message_delta_accepts_rendered() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        apply_event(
            &mut s,
            &evt("message.delta", json!({ "rendered": "hello" })),
        );
        assert_eq!(s.messages[0].content, "hello");
    }

    #[test]
    fn clarify_request_sets_pending() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        apply_event(
            &mut s,
            &evt(
                "clarify.request",
                json!({
                    "request_id": "r1",
                    "question": "which file?",
                    "choices": ["a.rs", "b.rs"]
                }),
            ),
        );
        let c = s.pending_clarify.expect("clarify");
        let question = c.current().expect("current question");
        assert_eq!(question.choices.len(), 2);
        assert_eq!(question.question, "which file?");
        assert!(!question.multi_select);
    }

    #[test]
    fn batch_clarify_preserves_questions_and_multi_select() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        apply_event(
            &mut s,
            &evt(
                "clarify.request",
                json!({
                    "request_id": "r2",
                    "answers": {"q0": "done"},
                    "questions": [
                        {"qid": "q0", "question": "answered?", "choices": ["yes"]},
                        {"qid": "q1", "question": "targets?", "choices": ["a", "b"], "multi_select": true},
                        {"qid": "q2", "question": "notes?", "choices": []}
                    ]
                }),
            ),
        );
        let c = s.pending_clarify.expect("clarify batch");
        assert!(c.is_batch());
        assert_eq!(c.questions.len(), 2, "replayed answers are skipped");
        assert_eq!(c.questions[0].qid.as_deref(), Some("q1"));
        assert!(c.questions[0].multi_select);
        assert_eq!(c.questions[1].qid.as_deref(), Some("q2"));
    }

    #[test]
    fn background_complete_drops_task() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        s.start_bg_task("bg_abc".into(), "summarize hn".into());
        apply_event(
            &mut s,
            &evt(
                "background.complete",
                json!({ "task_id": "bg_abc", "text": "done here" }),
            ),
        );
        assert_eq!(s.running_bg_count(), 0);
        assert_eq!(s.bg_tasks[0].status, crate::state::BgStatus::Done);
        assert!(s.bg_tasks[0].result.contains("done here"));
        assert!(s
            .messages
            .iter()
            .any(|m| m.content.contains("bg_abc") && m.content.contains("done here")));
    }

    #[test]
    fn gateway_exit_stops_generation() {
        let mut s = AppState::new();
        s.is_generating = true;
        apply_event(&mut s, &evt("gateway.exit", json!({ "code": 1 })));
        assert!(!s.is_generating);
    }

    #[test]
    fn ink_leftover_events_surface() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        s.pending_secret = Some(crate::state::SecretRequest {
            kind: crate::state::SecretKind::Sudo,
            request_id: "su1".into(),
            prompt: "sudo".into(),
            buffer: String::new(),
        });
        apply_event(&mut s, &evt("sudo.expire", json!({ "request_id": "su1" })));
        assert!(s.pending_secret.is_none());
        apply_event(
            &mut s,
            &evt(
                "session.usage",
                json!({ "usage": { "total": 42, "context_used": 10, "context_max": 100 } }),
            ),
        );
        assert_eq!(s.metrics.total_tokens, 42);
        apply_event(
            &mut s,
            &evt(
                "review.summary",
                json!({ "text": "💾 Self-improvement review: saved a note" }),
            ),
        );
        assert!(s
            .messages
            .iter()
            .any(|m| m.content.contains("Self-improvement")));
        apply_event(
            &mut s,
            &evt("gateway.protocol_error", json!({ "preview": "bad frame" })),
        );
        assert!(s.protocol_warned);
        apply_event(
            &mut s,
            &evt(
                "notification.show",
                json!({ "text": "low credits", "key": "credits" }),
            ),
        );
        assert!(s
            .metrics
            .toast_message
            .as_ref()
            .unwrap()
            .text
            .contains("low credits"));
    }

    #[test]
    fn compression_end_same_drain_folds_without_hold() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        apply_event(&mut s, &evt("compression.start", json!({})));
        assert!(s.metrics.is_compacting);
        apply_event(&mut s, &evt("compression.end", json!({})));
        assert!(!s.metrics.is_compacting);
        assert_eq!(s.messages.len(), 1);
        assert_eq!(s.messages[0].role, MessageRole::Compaction);
        assert!(s.messages[0].content.starts_with("context folded"));
    }

    #[test]
    fn subagent_events_build_tree() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        apply_event(
            &mut s,
            &evt(
                "subagent.spawn_requested",
                json!({
                    "subagent_id": "sa1",
                    "goal": "audit",
                    "depth": 0,
                    "task_index": 1
                }),
            ),
        );
        assert_eq!(s.running_agent_count(), 1);
        assert_eq!(s.agent_rows[0].status, "queued");
        apply_event(
            &mut s,
            &evt(
                "subagent.start",
                json!({ "subagent_id": "sa1", "goal": "audit", "depth": 0, "task_index": 1 }),
            ),
        );
        assert_eq!(s.agent_rows[0].status, "running");
        apply_event(
            &mut s,
            &evt(
                "subagent.tool",
                json!({
                    "subagent_id": "sa1",
                    "tool_name": "read_file",
                    "tool_preview": "src/app.rs"
                }),
            ),
        );
        assert!(s.agent_rows[0].last_tool.contains("read_file"));
        apply_event(
            &mut s,
            &evt(
                "subagent.complete",
                json!({
                    "subagent_id": "sa1",
                    "status": "completed",
                    "summary": "clean",
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "cost_usd": 0.02,
                    "iteration": 4
                }),
            ),
        );
        assert_eq!(s.running_agent_count(), 0);
        assert_eq!(s.agent_rows[0].status, "completed");
        assert_eq!(s.agent_rows[0].summary, "clean");
        assert_eq!(s.agent_rows[0].tokens(), 1200);
        assert_eq!(s.agent_rows[0].iteration, 4);
        assert!((s.agent_rows[0].cost_usd - 0.02).abs() < f64::EPSILON);
    }

    #[test]
    fn process_output_events_append_and_close() {
        let mut s = AppState::new();
        s.session_id = Some("abc".into());
        apply_event(
            &mut s,
            &evt(
                "agent.terminal.output",
                json!({ "process_id": "proc_1", "chunk": "boot\n" }),
            ),
        );
        assert_eq!(s.running_process_count(), 1);
        assert!(s.agent_rows[0].output.contains("boot"));
        apply_event(
            &mut s,
            &evt("terminal.close", json!({ "process_id": "proc_1" })),
        );
        assert_eq!(s.agent_rows[0].status, "exited");
        assert_eq!(s.running_process_count(), 0);
    }
}
