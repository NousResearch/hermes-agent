//! Gateway JSON to TUI structs. Re-exported from `crate::state`.
use super::*;
use chrono::Local;

pub fn parse_gateway_messages(values: &[serde_json::Value]) -> Vec<ChatMessage> {
    values
        .iter()
        .map(|v| {
            let role = v
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("assistant");
            let text = v
                .get("text")
                .or_else(|| v.get("content"))
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            let role = match role {
                "user" => MessageRole::User,
                "system" => MessageRole::System,
                "tool" => MessageRole::Tool {
                    name: v
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("tool")
                        .to_string(),
                    status: "completed".to_string(),
                    tool_id: None,
                },
                "reasoning" => MessageRole::Reasoning,
                _ => MessageRole::Assistant,
            };
            ChatMessage {
                id: uuid::Uuid::new_v4().to_string(),
                role,
                content: text,
                timestamp: Local::now(),
                output: String::new(),
                is_streaming: false,
            }
        })
        .filter(|m| !m.content.is_empty() || matches!(m.role, MessageRole::Tool { .. }))
        .collect()
}

pub fn parse_todos(value: &serde_json::Value) -> Vec<TaskItem> {
    let Some(arr) = value
        .as_array()
        .or_else(|| value.get("todos").and_then(|x| x.as_array()))
        .or_else(|| value.get("items").and_then(|x| x.as_array()))
        .or_else(|| value.get("tasks").and_then(|x| x.as_array()))
        .or_else(|| {
            value
                .get("args")
                .and_then(|a| a.get("todos").or_else(|| a.get("items")))
                .and_then(|x| x.as_array())
        })
    else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|item| {
            let title = item
                .get("content")
                .or_else(|| item.get("title"))
                .and_then(|v| v.as_str())?
                .to_string();
            let id = item
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            let status = item
                .get("status")
                .and_then(|v| v.as_str())
                .map(TaskStatus::from_gateway)
                .unwrap_or(TaskStatus::Pending);
            Some(TaskItem { id, title, status })
        })
        .collect()
}

pub fn tasks_blob(tasks: &[TaskItem]) -> String {
    let items: Vec<serde_json::Value> = tasks
        .iter()
        .map(|t| {
            serde_json::json!({
                "id": t.id,
                "content": t.title,
                "status": t.status.as_str(),
            })
        })
        .collect();
    serde_json::json!({ "todos": items }).to_string()
}

pub fn parse_skills(values: &[serde_json::Value]) -> Vec<SkillCard> {
    values
        .iter()
        .filter_map(|v| {
            if let Some(name) = v.as_str() {
                return Some(SkillCard {
                    name: name.to_string(),
                    category: String::new(),
                    description: String::new(),
                    preview: String::new(),
                });
            }
            let name = v
                .get("name")
                .or_else(|| v.get("id"))
                .and_then(|x| x.as_str())?
                .to_string();
            let category = v
                .get("category")
                .or_else(|| v.get("source"))
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let description = v
                .get("description")
                .or_else(|| v.get("desc"))
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            Some(SkillCard {
                name,
                category,
                description,
                preview: String::new(),
            })
        })
        .collect()
}

/// `skills.manage list` returns `{ skills: { category: [name, …] } }`, not an array.
pub fn parse_skills_payload(v: &serde_json::Value) -> Vec<SkillCard> {
    let Some(skills) = v.get("skills") else {
        return v
            .as_array()
            .map(|arr| parse_skills(arr))
            .unwrap_or_default();
    };
    if let Some(arr) = skills.as_array() {
        return parse_skills(arr);
    }
    let Some(obj) = skills.as_object() else {
        return Vec::new();
    };
    let mut cards = Vec::new();
    for (category, names) in obj {
        if let serde_json::Value::Array(arr) = names {
            for item in arr {
                if let Some(name) = item.as_str() {
                    cards.push(SkillCard {
                        name: name.to_string(),
                        category: category.clone(),
                        description: String::new(),
                        preview: String::new(),
                    });
                } else {
                    for mut card in parse_skills(std::slice::from_ref(item)) {
                        if card.category.is_empty() {
                            card.category = category.clone();
                        }
                        cards.push(card);
                    }
                }
            }
        }
    }
    cards.sort_by(|a, b| a.category.cmp(&b.category).then(a.name.cmp(&b.name)));
    cards
}

pub fn parse_profiles(v: &serde_json::Value) -> Vec<ProfileCard> {
    let Some(arr) = v.get("profiles").and_then(|p| p.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|p| {
            let name = p.get("name").and_then(|x| x.as_str())?.to_string();
            let last = p.get("last_session");
            let last_session_id = last
                .and_then(|s| s.get("id"))
                .and_then(|x| x.as_str())
                .map(|s| s.to_string());
            Some(ProfileCard {
                display_name: p
                    .get("display_name")
                    .and_then(|x| x.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or(&name)
                    .to_string(),
                model: p
                    .get("model")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                provider: p
                    .get("provider")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                description: p
                    .get("description")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                skill_count: p.get("skill_count").and_then(|x| x.as_u64()).unwrap_or(0),
                is_default: p
                    .get("is_default")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(false),
                last_title: last
                    .and_then(|s| s.get("title"))
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                last_preview: last
                    .and_then(|s| s.get("preview"))
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                worker_active: p
                    .get("worker_session")
                    .map(|w| !w.is_null())
                    .unwrap_or(false),
                last_session_id,
                name,
            })
        })
        .collect()
}

pub(crate) fn json_u64(v: &serde_json::Value, key: &str) -> Option<u64> {
    v.get(key)
        .and_then(|x| x.as_u64().or_else(|| x.as_f64().map(|f| f as u64)))
}

pub(crate) fn json_f64(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key)
        .and_then(|x| x.as_f64().or_else(|| x.as_u64().map(|n| n as f64)))
}

pub fn parse_agent_rows(
    processes: &serde_json::Value,
    status: &serde_json::Value,
) -> Vec<AgentRow> {
    let mut rows = Vec::new();
    if let Some(arr) = status.get("active").and_then(|a| a.as_array()) {
        for a in arr {
            let id = a
                .get("subagent_id")
                .or_else(|| a.get("id"))
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            if id.is_empty() {
                continue;
            }
            let goal = a
                .get("goal")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let model = a
                .get("model")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let depth = a.get("depth").and_then(|x| x.as_u64()).unwrap_or(0);
            let parent_id = a
                .get("parent_id")
                .and_then(|x| x.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());
            let status = a
                .get("status")
                .and_then(|x| x.as_str())
                .unwrap_or("running")
                .to_string();
            let tool_count = a.get("tool_count").and_then(|x| x.as_u64()).unwrap_or(0);
            rows.push(AgentRow {
                id,
                kind: "subagent".into(),
                title: if goal.is_empty() {
                    "subagent".into()
                } else {
                    goal
                },
                status,
                extra: format!("d{depth} {model}").trim().to_string(),
                depth: depth as u32,
                parent_id,
                model,
                tool_count,
                last_tool: String::new(),
                notes: Vec::new(),
                thinking: Vec::new(),
                summary: String::new(),
                started: Some(Instant::now()),
                duration_secs: None,
                index: a.get("task_index").and_then(|x| x.as_u64()).unwrap_or(0),
                pid: None,
                cwd: String::new(),
                output: String::new(),
                input_tokens: json_u64(a, "input_tokens").unwrap_or(0),
                output_tokens: json_u64(a, "output_tokens").unwrap_or(0),
                cost_usd: json_f64(a, "cost_usd").unwrap_or(0.0),
                iteration: json_u64(a, "iteration").unwrap_or(0),
                api_calls: json_u64(a, "api_calls").unwrap_or(0),
            });
        }
    }
    if let Some(arr) = processes
        .get("processes")
        .and_then(|p| p.as_array())
        .or_else(|| processes.as_array())
    {
        for p in arr {
            let id = p
                .get("session_id")
                .or_else(|| p.get("id"))
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            if id.is_empty() || rows.iter().any(|r| r.id == id) {
                continue;
            }
            let cmd = p
                .get("command")
                .and_then(|x| x.as_str())
                .unwrap_or("process")
                .to_string();
            let uptime = p
                .get("uptime_seconds")
                .or_else(|| p.get("uptime"))
                .and_then(|x| x.as_u64())
                .unwrap_or(0);
            let output = p
                .get("output_tail")
                .or_else(|| p.get("output_preview"))
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let pid = p.get("pid").and_then(|x| x.as_u64());
            let cwd = p
                .get("cwd")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let started = (uptime > 0)
                .then(|| Instant::now().checked_sub(Duration::from_secs(uptime)))
                .flatten();
            rows.push(AgentRow {
                id,
                kind: "process".into(),
                title: cmd,
                status: p
                    .get("status")
                    .and_then(|x| x.as_str())
                    .unwrap_or("running")
                    .to_string(),
                extra: if uptime > 0 {
                    format!("{uptime}s")
                } else {
                    String::new()
                },
                depth: 0,
                parent_id: None,
                model: String::new(),
                tool_count: 0,
                last_tool: String::new(),
                notes: Vec::new(),
                thinking: Vec::new(),
                summary: String::new(),
                started,
                duration_secs: Some(uptime as f64),
                index: 0,
                pid,
                cwd,
                output,
                input_tokens: 0,
                output_tokens: 0,
                cost_usd: 0.0,
                iteration: 0,
                api_calls: 0,
            });
        }
    }
    rows
}

pub fn parse_memory_payload(v: &serde_json::Value) -> (Vec<String>, Vec<MemoryRow>) {
    let summary = v
        .get("summary")
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let mut nodes = Vec::new();
    if let Some(buckets) = v.get("buckets").and_then(|b| b.as_array()) {
        for bucket in buckets {
            let date = bucket
                .get("date")
                .or_else(|| bucket.get("label"))
                .and_then(|x| x.as_str())
                .unwrap_or("");
            if let Some(arr) = bucket.get("nodes").and_then(|n| n.as_array()) {
                for n in arr {
                    let id = n
                        .get("id")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                    let label = n
                        .get("label")
                        .or_else(|| n.get("fullLabel"))
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                    if label.is_empty() && id.is_empty() {
                        continue;
                    }
                    let kind = n
                        .get("style")
                        .and_then(|x| x.as_str())
                        .or_else(|| n.get("kind").and_then(|x| x.as_str()))
                        .unwrap_or("skill")
                        .to_string();
                    let meta = n
                        .get("meta")
                        .and_then(|x| x.as_str())
                        .unwrap_or(date)
                        .to_string();
                    nodes.push(MemoryRow {
                        id,
                        kind,
                        label,
                        meta,
                        body: n
                            .get("body")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string(),
                    });
                }
            }
        }
    }
    (summary, nodes)
}

pub fn parse_checkpoints(v: &serde_json::Value) -> (bool, Vec<Checkpoint>) {
    let enabled = v.get("enabled").and_then(|x| x.as_bool()).unwrap_or(true);
    let rows = v
        .get("checkpoints")
        .and_then(|a| a.as_array())
        .into_iter()
        .flatten()
        .filter_map(|c| {
            let hash = c.get("hash").and_then(|x| x.as_str())?.to_string();
            if hash.is_empty() {
                return None;
            }
            Some(Checkpoint {
                hash,
                timestamp: c
                    .get("timestamp")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                message: c
                    .get("message")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
            })
        })
        .collect();
    (enabled, rows)
}

pub fn parse_sessions(values: &[serde_json::Value]) -> Vec<SessionRecord> {
    values
        .iter()
        .filter_map(|v| {
            let id = v.get("id").and_then(|x| x.as_str())?.to_string();
            let title = v
                .get("title")
                .and_then(|x| x.as_str())
                .unwrap_or("(untitled)")
                .to_string();
            let updated_at = v
                .get("preview")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
                .or_else(|| {
                    v.get("started_at")
                        .and_then(|x| x.as_u64())
                        .map(|n| n.to_string())
                })
                .unwrap_or_default();
            Some(SessionRecord {
                id,
                title,
                updated_at,
                live: false,
                status: String::new(),
            })
        })
        .collect()
}

pub fn parse_live_sessions(value: &serde_json::Value) -> Vec<SessionRecord> {
    let Some(arr) = value.get("sessions").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|v| {
            let id = v.get("id").and_then(|x| x.as_str())?.to_string();
            if id.is_empty() {
                return None;
            }
            let title = v
                .get("title")
                .and_then(|x| x.as_str())
                .unwrap_or("(untitled)")
                .to_string();
            let preview = v
                .get("preview")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let status = v
                .get("status")
                .and_then(|x| x.as_str())
                .unwrap_or("live")
                .to_string();
            Some(SessionRecord {
                id,
                title,
                updated_at: if preview.is_empty() {
                    status.clone()
                } else {
                    preview
                },
                live: true,
                status,
            })
        })
        .collect()
}

pub fn merge_session_lists(
    live: Vec<SessionRecord>,
    stored: Vec<SessionRecord>,
) -> Vec<SessionRecord> {
    let mut out = live;
    let seen: HashSet<String> = out.iter().map(|s| s.id.clone()).collect();
    for row in stored {
        if !seen.contains(&row.id) {
            out.push(row);
        }
    }
    out
}

pub fn parse_model_provider(provider: &serde_json::Value) -> Option<ModelProvider> {
    let slug = provider
        .get("slug")
        .or_else(|| provider.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if slug.is_empty() {
        return None;
    }
    let name = provider
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or(&slug)
        .to_string();
    let models = provider
        .get("models")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    m.as_str()
                        .or_else(|| m.get("id").and_then(|v| v.as_str()))
                        .map(|s| s.to_string())
                })
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    Some(ModelProvider {
        slug,
        name,
        models,
        authenticated: provider
            .get("authenticated")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        is_current: provider
            .get("is_current")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        warning: provider
            .get("warning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        auth_type: provider
            .get("auth_type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        key_env: provider
            .get("key_env")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
    })
}

pub fn parse_saved_provider(value: &serde_json::Value) -> Option<ModelProvider> {
    value.get("provider").and_then(parse_model_provider)
}

pub fn parse_model_providers(value: &serde_json::Value) -> Vec<ModelProvider> {
    let Some(providers) = value.get("providers").and_then(|p| p.as_array()) else {
        return Vec::new();
    };
    providers.iter().filter_map(parse_model_provider).collect()
}

pub fn parse_grouped_names(value: &serde_json::Value) -> Vec<(String, Vec<String>)> {
    let Some(obj) = value.as_object() else {
        return Vec::new();
    };
    let mut out: Vec<(String, Vec<String>)> = obj
        .iter()
        .map(|(key, val)| {
            let label = key.strip_suffix("_tools").unwrap_or(key).to_string();
            let names = val
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            (label, names)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

pub(crate) fn tool_live_status(msg: &ChatMessage) -> String {
    let MessageRole::Tool { name, .. } = &msg.role else {
        return "Working".into();
    };
    let n = name.to_ascii_lowercase();
    let file = json_field(
        &msg.content,
        &["path", "file", "target_file", "filename", "file_path"],
    )
    .map(|p| p.rsplit(['/', '\\']).next().unwrap_or("file").to_string());
    let cmd = json_field(&msg.content, &["command", "cmd", "script"]);
    if n.contains("skill") {
        match file {
            Some(f) => format!("Reading skill {f}"),
            None => "Reading skill".into(),
        }
    } else if n.contains("todo") {
        "Updating plan".into()
    } else if n.contains("read") || n.contains("view") || n == "cat" {
        match file {
            Some(f) => format!("Reading {f}"),
            None => "Reading".into(),
        }
    } else if n.contains("search") || n.contains("grep") || n.contains("glob") || n.contains("find")
    {
        match json_field(&msg.content, &["pattern", "query", "glob", "q"]) {
            Some(q) => format!("Searching {}", crate::tips::ellipsize(&q, 36)),
            None => "Searching".into(),
        }
    } else if n.contains("replace")
        || n.contains("edit")
        || n.contains("write")
        || n.contains("patch")
    {
        match file {
            Some(f) => format!("Editing {f}"),
            None => "Editing".into(),
        }
    } else if n.contains("terminal")
        || n.contains("bash")
        || n.contains("shell")
        || n.contains("exec")
        || n == "run"
        || n.contains("command")
    {
        match cmd {
            Some(c) => format!("Running {}", crate::tips::ellipsize(&c, 36)),
            None => "Running command".into(),
        }
    } else if n.contains("web") || n.contains("extract") || n.contains("fetch") {
        if let Some(url) = json_field(&msg.content, &["url", "uri", "href"]) {
            format!("Fetching {}", crate::tips::ellipsize(&url, 36))
        } else {
            format!("Using {}", name.rsplit(['/', ':']).next().unwrap_or(name))
        }
    } else {
        let short = name.rsplit(['/', ':']).next().unwrap_or(name);
        format!("Using {short}")
    }
}

fn json_field(content: &str, keys: &[&str]) -> Option<String> {
    crate::ui::stream::json_str(content, keys).or_else(|| {
        let head = content.lines().next().unwrap_or(content).trim();
        let v: serde_json::Value = serde_json::from_str(head).ok()?;
        if let Some(arr) = v.get("argv").and_then(|a| a.as_array()) {
            let parts: Vec<&str> = arr.iter().filter_map(|x| x.as_str()).collect();
            if !parts.is_empty() {
                return Some(parts.join(" "));
            }
        }
        None
    })
}

pub(crate) fn sanitize_tab_title(s: &str) -> String {
    s.chars()
        .filter(|c| *c != '\x1b' && *c != '\x07' && !c.is_control())
        .take(96)
        .collect()
}

pub fn parse_toolsets(value: &serde_json::Value) -> Vec<ToolsetRow> {
    let Some(arr) = value.get("toolsets").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|s| {
            let name = s.get("name").and_then(|v| v.as_str())?.trim().to_string();
            if name.is_empty() {
                return None;
            }
            Some(ToolsetRow {
                name,
                description: s
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                enabled: s.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true),
                tool_count: s.get("tool_count").and_then(|v| v.as_u64()).unwrap_or(0),
            })
        })
        .collect()
}

pub fn parse_plugins(value: &serde_json::Value) -> Vec<PluginRow> {
    let Some(arr) = value.get("plugins").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|s| {
            let name = s.get("name").and_then(|v| v.as_str())?.trim().to_string();
            if name.is_empty() {
                return None;
            }
            let key = s
                .get("key")
                .and_then(|v| v.as_str())
                .unwrap_or(&name)
                .to_string();
            let enabled = if let Some(b) = s.get("enabled").and_then(|v| v.as_bool()) {
                b
            } else if let Some(st) = s.get("status").and_then(|v| v.as_str()) {
                st.eq_ignore_ascii_case("enabled")
            } else {
                true
            };
            Some(PluginRow {
                name,
                key,
                version: s
                    .get("version")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string(),
                enabled,
            })
        })
        .collect()
}

pub fn parse_project_rows(value: &serde_json::Value) -> Vec<ProjectRow> {
    let Some(arr) = value.get("projects").and_then(|x| x.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|p| {
            let id = p.get("id").and_then(|x| x.as_str())?.to_string();
            if id.is_empty() {
                return None;
            }
            let name = p
                .get("label")
                .or_else(|| p.get("name"))
                .and_then(|x| x.as_str())
                .unwrap_or(&id)
                .to_string();
            Some(ProjectRow {
                id,
                name,
                count: project_session_count(p),
            })
        })
        .collect()
}

pub fn parse_project_session_records(value: &serde_json::Value) -> Vec<SessionRecord> {
    let Some(p) = value.get("project") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut push = |s: &serde_json::Value| {
        let Some(id) = s.get("id").and_then(|x| x.as_str()) else {
            return;
        };
        if id.is_empty() || out.iter().any(|r: &SessionRecord| r.id == id) {
            return;
        }
        let title = s
            .get("title")
            .or_else(|| s.get("preview"))
            .and_then(|x| x.as_str())
            .unwrap_or("(untitled)")
            .to_string();
        let preview = s
            .get("preview")
            .or_else(|| s.get("cwd"))
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        out.push(SessionRecord {
            id: id.to_string(),
            title,
            updated_at: preview,
            live: false,
            status: s
                .get("status")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
        });
    };
    if let Some(repos) = p.get("repos").and_then(|x| x.as_array()) {
        for repo in repos {
            if let Some(groups) = repo
                .get("groups")
                .or_else(|| repo.get("lanes"))
                .and_then(|x| x.as_array())
            {
                for g in groups {
                    if let Some(sessions) = g.get("sessions").and_then(|x| x.as_array()) {
                        for s in sessions {
                            push(s);
                        }
                    }
                }
            }
        }
    }
    if let Some(sessions) = p.get("sessions").and_then(|x| x.as_array()) {
        for s in sessions {
            push(s);
        }
    }
    if let Some(preview) = p.get("previewSessions").and_then(|x| x.as_array()) {
        for s in preview {
            push(s);
        }
    }
    out
}

pub fn parse_spawn_tree_agents(value: &serde_json::Value) -> Vec<AgentRow> {
    let Some(arr) = value.get("subagents").and_then(|x| x.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .enumerate()
        .map(|(i, a)| {
            let id = a
                .get("id")
                .or_else(|| a.get("subagent_id"))
                .and_then(|x| x.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("snap-{i}"));
            let title = a
                .get("goal")
                .or_else(|| a.get("title"))
                .or_else(|| a.get("name"))
                .and_then(|x| x.as_str())
                .unwrap_or("subagent")
                .to_string();
            let status = a
                .get("status")
                .and_then(|x| x.as_str())
                .unwrap_or("completed")
                .to_string();
            let depth = a.get("depth").and_then(|x| x.as_u64()).unwrap_or(0);
            let parent_id = a
                .get("parentId")
                .or_else(|| a.get("parent_id"))
                .and_then(|x| x.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());
            let model = a
                .get("model")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let tool_count = a
                .get("toolCount")
                .or_else(|| a.get("tool_count"))
                .and_then(|x| x.as_u64())
                .unwrap_or(0);
            AgentRow {
                id,
                kind: "subagent".into(),
                title,
                status,
                extra: String::new(),
                depth: depth as u32,
                parent_id,
                model,
                tool_count,
                last_tool: String::new(),
                notes: a
                    .get("notes")
                    .and_then(|x| x.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|n| n.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default(),
                thinking: Vec::new(),
                summary: a
                    .get("summary")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                started: None,
                duration_secs: None,
                index: a.get("index").and_then(|x| x.as_u64()).unwrap_or(i as u64),
                pid: None,
                cwd: String::new(),
                output: String::new(),
                input_tokens: json_u64(a, "inputTokens")
                    .or_else(|| json_u64(a, "input_tokens"))
                    .unwrap_or(0),
                output_tokens: json_u64(a, "outputTokens")
                    .or_else(|| json_u64(a, "output_tokens"))
                    .unwrap_or(0),
                cost_usd: json_f64(a, "costUsd")
                    .or_else(|| json_f64(a, "cost_usd"))
                    .unwrap_or(0.0),
                iteration: json_u64(a, "iteration").unwrap_or(0),
                api_calls: json_u64(a, "apiCalls")
                    .or_else(|| json_u64(a, "api_calls"))
                    .unwrap_or(0),
            }
        })
        .collect()
}

pub fn parse_spawn_entries(value: &serde_json::Value) -> Vec<SpawnTreeEntry> {
    let Some(arr) = value.get("entries").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|e| {
            let path = e.get("path").and_then(|x| x.as_str())?.to_string();
            if path.is_empty() {
                return None;
            }
            Some(SpawnTreeEntry {
                label: e
                    .get("label")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                path,
                count: e.get("count").and_then(|x| x.as_u64()).unwrap_or(0),
            })
        })
        .collect()
}

pub fn resolve_spawn_entry<'a>(
    token: &str,
    entries: &'a [SpawnTreeEntry],
) -> Option<&'a SpawnTreeEntry> {
    let t = token.trim();
    if t.is_empty() {
        return None;
    }
    if let Ok(n) = t.parse::<usize>() {
        if n >= 1 {
            return entries.get(n - 1);
        }
    }
    entries.iter().find(|e| e.path == t || e.path.ends_with(t))
}

fn spawn_goals(v: &serde_json::Value) -> Vec<String> {
    let Some(arr) = v.get("subagents").and_then(|x| x.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|s| {
            s.get("goal")
                .or_else(|| s.get("name"))
                .or_else(|| s.get("id"))
                .and_then(|x| x.as_str())
                .map(|g| g.to_string())
        })
        .collect()
}

fn spawn_tool_count(v: &serde_json::Value) -> u64 {
    let Some(arr) = v.get("subagents").and_then(|x| x.as_array()) else {
        return 0;
    };
    arr.iter()
        .map(|s| {
            s.get("toolCount")
                .or_else(|| s.get("tool_count"))
                .and_then(|x| x.as_u64())
                .unwrap_or(0)
        })
        .sum()
}

pub fn format_spawn_diff(
    a: &serde_json::Value,
    b: &serde_json::Value,
    a_ref: &str,
    b_ref: &str,
) -> String {
    let a_n = a
        .get("subagents")
        .and_then(|x| x.as_array())
        .map(|x| x.len())
        .unwrap_or(0);
    let b_n = b
        .get("subagents")
        .and_then(|x| x.as_array())
        .map(|x| x.len())
        .unwrap_or(0);
    let a_tools = spawn_tool_count(a);
    let b_tools = spawn_tool_count(b);
    let a_label = a.get("label").and_then(|x| x.as_str()).unwrap_or(a_ref);
    let b_label = b.get("label").and_then(|x| x.as_str()).unwrap_or(b_ref);
    let a_goals = spawn_goals(a);
    let b_goals = spawn_goals(b);
    let dn = b_n as i64 - a_n as i64;
    let dt = b_tools as i64 - a_tools as i64;
    format!(
        "A  {a_ref}\n   {a_label} · {a_n} agents · {a_tools} tools\n   {}\n\nB  {b_ref}\n   {b_label} · {b_n} agents · {b_tools} tools\n   {}\n\nΔ  agents {dn:+}  tools {dt:+}",
        if a_goals.is_empty() {
            "(no goals)".into()
        } else {
            a_goals.join(" · ")
        },
        if b_goals.is_empty() {
            "(no goals)".into()
        } else {
            b_goals.join(" · ")
        },
    )
}

pub fn format_tools_configure(v: &serde_json::Value, action: &str) -> String {
    let join = |key: &str| -> Vec<String> {
        v.get(key)
            .and_then(|x| x.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    };
    let changed = join("changed");
    let unknown = join("unknown");
    let missing = join("missing_servers");
    let verb = if action == "disable" {
        "disabled"
    } else {
        "enabled"
    };
    let mut parts = Vec::new();
    if !changed.is_empty() {
        parts.push(format!("{verb}: {}", changed.join(", ")));
    }
    if !unknown.is_empty() {
        parts.push(format!("unknown: {}", unknown.join(", ")));
    }
    if !missing.is_empty() {
        parts.push(format!("missing MCP: {}", missing.join(", ")));
    }
    if v.get("reset").and_then(|x| x.as_bool()) == Some(true) {
        parts.push("session reset".into());
    }
    if parts.is_empty() {
        format!("tools {action}: nothing changed")
    } else {
        parts.join(" · ")
    }
}

pub fn format_tools_show(v: &serde_json::Value, toolset: Option<&str>) -> String {
    let Some(sections) = v.get("sections").and_then(|x| x.as_array()) else {
        return v.to_string();
    };
    let want = toolset.map(|t| t.to_ascii_lowercase());
    let mut body = String::new();
    for sec in sections {
        let name = sec.get("name").and_then(|x| x.as_str()).unwrap_or("");
        if let Some(w) = want.as_deref() {
            if !name.eq_ignore_ascii_case(w) {
                continue;
            }
        }
        body.push_str(&format!("[{name}]\n"));
        if let Some(tools) = sec.get("tools").and_then(|x| x.as_array()) {
            for t in tools {
                let tn = t.get("name").and_then(|x| x.as_str()).unwrap_or("?");
                let td = t.get("description").and_then(|x| x.as_str()).unwrap_or("");
                if td.is_empty() {
                    body.push_str(&format!("  {tn}\n"));
                } else {
                    body.push_str(&format!("  {tn}  {td}\n"));
                }
            }
        }
    }
    if body.is_empty() {
        format!("no tools in {}", toolset.unwrap_or("show"))
    } else {
        body
    }
}

pub fn format_mcp_test(v: &serde_json::Value, name: &str) -> String {
    if v.get("ok").and_then(|x| x.as_bool()) == Some(true) {
        let n = v
            .get("tools")
            .and_then(|x| x.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let prompts = v.get("prompts").and_then(|x| x.as_u64()).unwrap_or(0);
        let resources = v.get("resources").and_then(|x| x.as_u64()).unwrap_or(0);
        format!("{name}  ok  {n} tools  {prompts} prompts  {resources} resources")
    } else {
        let err = v
            .get("error")
            .and_then(|x| x.as_str())
            .unwrap_or("probe failed");
        if v.get("oauth_needed").and_then(|x| x.as_bool()) == Some(true) {
            format!("{name}  {err}  ·  hermes mcp login {name}")
        } else {
            format!("{name}  {err}")
        }
    }
}

pub fn format_profile_describe(v: &serde_json::Value) -> String {
    let name = v.get("name").and_then(|x| x.as_str()).unwrap_or("profile");
    let desc = v.get("description").and_then(|x| x.as_str()).unwrap_or("");
    let soul = v.get("soul").and_then(|x| x.as_str()).unwrap_or("");
    let model = v
        .get("model")
        .and_then(|m| m.get("default"))
        .and_then(|x| x.as_str())
        .unwrap_or("");
    let provider = v
        .get("model")
        .and_then(|m| m.get("provider"))
        .and_then(|x| x.as_str())
        .unwrap_or("");
    let skills = v
        .get("skills")
        .and_then(|x| x.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let toolsets = v
        .get("toolsets")
        .and_then(|x| x.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let mut body =
        format!("{name}\n{desc}\n{provider} · {model}\n{skills} skills  {toolsets} toolsets");
    if !soul.is_empty() {
        body.push('\n');
        body.push_str(soul);
    }
    body
}

pub fn format_skill_inspect(v: &serde_json::Value) -> String {
    let info = v.get("info").unwrap_or(v);
    if let Some(s) = info.as_str() {
        return s.to_string();
    }
    let name = info.get("name").and_then(|x| x.as_str()).unwrap_or("skill");
    let desc = info
        .get("description")
        .or_else(|| info.get("desc"))
        .and_then(|x| x.as_str())
        .unwrap_or("");
    let origin = info
        .get("origin")
        .or_else(|| info.get("source"))
        .and_then(|x| x.as_str())
        .unwrap_or("");
    let path = info.get("path").and_then(|x| x.as_str()).unwrap_or("");
    let mut body = format!("{name}\n{desc}");
    if !origin.is_empty() {
        body.push_str(&format!("\n{origin}"));
    }
    if !path.is_empty() {
        body.push_str(&format!("\n{path}"));
    }
    if body.trim() == name {
        serde_json::to_string_pretty(info).unwrap_or_else(|_| info.to_string())
    } else {
        body
    }
}

#[cfg(test)]
pub fn format_projects_tree(v: &serde_json::Value) -> String {
    let Some(projects) = v.get("projects").and_then(|x| x.as_array()) else {
        return "no projects".into();
    };
    if projects.is_empty() {
        return "no projects".into();
    }
    let active = v.get("active_id").and_then(|x| x.as_str()).unwrap_or("");
    let mut body = String::new();
    for p in projects {
        let id = p.get("id").and_then(|x| x.as_str()).unwrap_or("");
        let name = p
            .get("name")
            .or_else(|| p.get("label"))
            .and_then(|x| x.as_str())
            .unwrap_or(id);
        let n = project_session_count(p);
        let mark = if !active.is_empty() && id == active {
            "●"
        } else {
            "·"
        };
        body.push_str(&format!("{mark} {name}  {n} sessions\n"));
    }
    body
}

fn project_session_count(p: &serde_json::Value) -> u64 {
    p.get("sessionCount")
        .or_else(|| p.get("session_count"))
        .or_else(|| p.get("count"))
        .and_then(|x| x.as_u64())
        .unwrap_or(0)
}

pub fn match_project_id(tree: &serde_json::Value, query: &str) -> Option<String> {
    let q = query.trim();
    if q.is_empty() {
        return None;
    }
    let projects = tree.get("projects")?.as_array()?;
    let ql = q.to_ascii_lowercase();
    let exact = projects.iter().find_map(|p| {
        let id = p.get("id").and_then(|x| x.as_str()).unwrap_or("");
        let name = p
            .get("name")
            .or_else(|| p.get("label"))
            .and_then(|x| x.as_str())
            .unwrap_or("");
        if id == q || name.eq_ignore_ascii_case(q) {
            Some(id.to_string())
        } else {
            None
        }
    });
    if exact.is_some() {
        return exact;
    }
    let hits: Vec<&serde_json::Value> = projects
        .iter()
        .filter(|p| {
            let id = p.get("id").and_then(|x| x.as_str()).unwrap_or("");
            let name = p
                .get("name")
                .or_else(|| p.get("label"))
                .and_then(|x| x.as_str())
                .unwrap_or("");
            id.to_ascii_lowercase().contains(&ql) || name.to_ascii_lowercase().contains(&ql)
        })
        .collect();
    if hits.len() == 1 {
        hits[0]
            .get("id")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
    } else {
        None
    }
}

#[cfg(test)]
pub fn format_project_sessions(v: &serde_json::Value) -> String {
    let Some(p) = v.get("project") else {
        return "project not found".into();
    };
    if p.is_null() {
        return "project not found".into();
    }
    let name = p
        .get("label")
        .or_else(|| p.get("name"))
        .and_then(|x| x.as_str())
        .unwrap_or("project");
    let n = project_session_count(p);
    let mut body = format!("{name}  {n} sessions\n");
    let Some(repos) = p.get("repos").and_then(|x| x.as_array()) else {
        return body;
    };
    for repo in repos {
        let rname = repo
            .get("label")
            .or_else(|| repo.get("path"))
            .and_then(|x| x.as_str())
            .unwrap_or("repo");
        body.push_str(&format!("  {rname}\n"));
        let Some(groups) = repo
            .get("groups")
            .or_else(|| repo.get("lanes"))
            .and_then(|x| x.as_array())
        else {
            continue;
        };
        for g in groups {
            let lname = g
                .get("label")
                .or_else(|| g.get("id"))
                .and_then(|x| x.as_str())
                .unwrap_or("lane");
            body.push_str(&format!("    {lname}\n"));
            let Some(sessions) = g.get("sessions").and_then(|x| x.as_array()) else {
                continue;
            };
            for s in sessions {
                let title = s
                    .get("title")
                    .or_else(|| s.get("preview"))
                    .or_else(|| s.get("id"))
                    .and_then(|x| x.as_str())
                    .unwrap_or("session");
                let id = s.get("id").and_then(|x| x.as_str()).unwrap_or("");
                if id.is_empty() {
                    body.push_str(&format!("      · {title}\n"));
                } else {
                    body.push_str(&format!("      · {title}  {id}\n"));
                }
            }
        }
    }
    body
}

pub fn format_usage_bars(v: &serde_json::Value) -> Option<String> {
    if v.get("available").and_then(|x| x.as_bool()) != Some(true) {
        return None;
    }
    let mut body = String::new();
    if let Some(plan) = v.get("plan_name").and_then(|x| x.as_str()) {
        if !plan.is_empty() {
            body.push_str(plan);
            body.push('\n');
        }
    }
    if let Some(st) = v.get("status").and_then(|x| x.as_str()) {
        body.push_str(st);
        body.push('\n');
    }
    if let Some(sub) = v
        .get("subscription_remaining_display")
        .and_then(|x| x.as_str())
    {
        body.push_str(&format!("plan remaining  {sub}\n"));
    }
    if let Some(top) = v.get("topup_remaining_display").and_then(|x| x.as_str()) {
        body.push_str(&format!("top-up remaining  {top}\n"));
    }
    if let Some(tot) = v.get("total_spendable_display").and_then(|x| x.as_str()) {
        body.push_str(&format!("spendable  {tot}\n"));
    }
    if let Some(renew) = v.get("renews_display").and_then(|x| x.as_str()) {
        if !renew.is_empty() {
            body.push_str(&format!("renews  {renew}\n"));
        }
    }
    if body.trim().is_empty() {
        None
    } else {
        Some(body.trim_end().to_string())
    }
}

pub fn config_flag_on(v: &serde_json::Value) -> bool {
    match v.get("value") {
        Some(serde_json::Value::Bool(b)) => *b,
        Some(serde_json::Value::String(s)) => {
            let t = s.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "on" | "true" | "yes")
        }
        Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(0) != 0,
        _ => false,
    }
}

pub fn spawn_subagents_from_rows(rows: &[AgentRow]) -> Vec<serde_json::Value> {
    rows.iter()
        .filter(|r| r.kind == "subagent")
        .map(|r| {
            serde_json::json!({
                "id": r.id,
                "goal": r.title,
                "status": r.status,
                "parentId": r.parent_id,
                "depth": r.depth,
                "model": r.model,
                "toolCount": r.tool_count,
                "notes": r.notes,
                "thinking": r.thinking,
                "summary": r.summary,
                "index": r.index,
                "inputTokens": r.input_tokens,
                "outputTokens": r.output_tokens,
                "costUsd": r.cost_usd,
                "iteration": r.iteration,
                "apiCalls": r.api_calls,
            })
        })
        .collect()
}

pub fn parse_cron_jobs(value: &serde_json::Value) -> Vec<CronJobRow> {
    let arr = value
        .get("jobs")
        .and_then(|v| v.as_array())
        .or_else(|| value.as_array());
    let Some(arr) = arr else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|s| {
            let id = s
                .get("job_id")
                .or_else(|| s.get("id"))
                .and_then(|v| v.as_str())?
                .trim()
                .to_string();
            if id.is_empty() {
                return None;
            }
            Some(CronJobRow {
                name: s
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&id)
                    .to_string(),
                schedule: s
                    .get("schedule")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string(),
                enabled: s.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true),
                state: s
                    .get("state")
                    .and_then(|v| v.as_str())
                    .unwrap_or(
                        if s.get("enabled").and_then(|v| v.as_bool()) == Some(false) {
                            "paused"
                        } else {
                            "active"
                        },
                    )
                    .to_string(),
                prompt: s
                    .get("prompt_preview")
                    .or_else(|| s.get("prompt"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                id,
            })
        })
        .collect()
}

pub fn parse_mcp_servers(
    list: &serde_json::Value,
    catalog: &serde_json::Value,
) -> Vec<McpServerRow> {
    let mut rows: Vec<McpServerRow> = Vec::new();
    let mut seen = HashSet::new();
    if let Some(arr) = list.get("servers").and_then(|v| v.as_array()) {
        for s in arr {
            let name = s
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if name.is_empty() {
                continue;
            }
            seen.insert(name.to_ascii_lowercase());
            let tools = s
                .get("tools")
                .and_then(|v| v.as_array().map(|a| a.len() as u64).or_else(|| v.as_u64()))
                .unwrap_or(0);
            rows.push(McpServerRow {
                name,
                transport: s
                    .get("transport")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                description: s
                    .get("url")
                    .and_then(|v| v.as_str())
                    .or_else(|| s.get("command").and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .to_string(),
                enabled: s.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true),
                installed: true,
                connected: s
                    .get("connected")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                tools,
                requires: Vec::new(),
                configured: true,
            });
        }
    }
    if let Some(arr) = catalog.get("servers").and_then(|v| v.as_array()) {
        for s in arr {
            let name = s
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if name.is_empty() || seen.contains(&name.to_ascii_lowercase()) {
                continue;
            }
            let requires = s
                .get("requires")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|t| t.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            rows.push(McpServerRow {
                name,
                transport: s
                    .get("transport")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                description: s
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                enabled: s.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false),
                installed: s
                    .get("installed")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                connected: false,
                tools: 0,
                requires,
                configured: false,
            });
        }
    }
    rows
}

pub fn count_mcp_connected(value: &serde_json::Value) -> usize {
    value
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter(|s| s.get("connected").and_then(|v| v.as_bool()) == Some(true))
                .count()
        })
        .unwrap_or(0)
}
