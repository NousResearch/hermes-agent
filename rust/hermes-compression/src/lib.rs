use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const SUMMARY_PREFIX: &str = "[CONTEXT COMPACTION] Earlier turns in this conversation were compacted to save context space. The summary below describes work that was already completed, and the current session state may still reflect that work (for example, files may already be changed). Use the summary and the current state to continue from where things left off, and avoid repeating work:";
const PRUNED_TOOL_PLACEHOLDER: &str = "[Old tool output cleared to save context space]";
const CHARS_PER_TOKEN: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompressionPlan {
    pub messages: Vec<Value>,
    pub compress_start: usize,
    pub compress_end: usize,
    pub needs_compression: bool,
}

pub fn plan(messages: &[Value], protect_first_n: usize, protect_last_n: usize) -> CompressionPlan {
    let n_messages = messages.len();
    if n_messages <= protect_first_n + protect_last_n + 1 {
        return CompressionPlan {
            messages: messages.to_vec(),
            compress_start: 0,
            compress_end: 0,
            needs_compression: false,
        };
    }

    let pruned = prune_old_tool_results(messages, protect_last_n * 3);
    let compress_start = align_boundary_forward(&pruned, protect_first_n);
    let compress_end = find_tail_cut_by_tokens(&pruned, compress_start, protect_last_n);

    CompressionPlan {
        messages: pruned,
        compress_start,
        compress_end,
        needs_compression: compress_start < compress_end,
    }
}

pub fn apply(
    messages: &[Value],
    compress_start: usize,
    compress_end: usize,
    summary: Option<&str>,
    compression_count: usize,
) -> Vec<Value> {
    let n_messages = messages.len();
    if compress_start >= compress_end || compress_end > n_messages {
        return messages.to_vec();
    }

    let mut compressed: Vec<Value> = Vec::new();
    for i in 0..compress_start {
        let mut msg = messages[i].clone();
        if i == 0 && role_of(&msg) == "system" && compression_count == 0 {
            let original = content_of(&msg);
            set_content(
                &mut msg,
                format!(
                    "{original}\n\n[Note: Some earlier conversation turns have been compacted into a handoff summary to preserve context space. The current session state may still reflect earlier work, so build on that summary and state rather than re-doing work.]"
                ),
            );
        }
        compressed.push(msg);
    }

    let mut merge_summary_into_tail = false;
    if let Some(summary) = summary {
        let normalized_summary = with_summary_prefix(summary);
        let last_head_role = if compress_start > 0 {
            role_of(&messages[compress_start - 1])
        } else {
            "user".to_string()
        };
        let first_tail_role = if compress_end < n_messages {
            role_of(&messages[compress_end])
        } else {
            "user".to_string()
        };

        let mut summary_role = if last_head_role == "assistant" || last_head_role == "tool" {
            "user".to_string()
        } else {
            "assistant".to_string()
        };

        if summary_role == first_tail_role {
            let flipped = if summary_role == "user" { "assistant" } else { "user" }.to_string();
            if flipped != last_head_role {
                summary_role = flipped;
            } else {
                merge_summary_into_tail = true;
            }
        }

        if !merge_summary_into_tail {
            compressed.push(json_message(&summary_role, normalized_summary));
        }
    }

    for i in compress_end..n_messages {
        let mut msg = messages[i].clone();
        if merge_summary_into_tail {
            if let Some(summary_text) = summary {
                let merged = format!("{}\n\n{}", with_summary_prefix(summary_text), content_of(&msg));
                set_content(&mut msg, merged);
            }
            merge_summary_into_tail = false;
        }
        compressed.push(msg);
    }

    sanitize_tool_pairs(&compressed)
}

fn prune_old_tool_results(messages: &[Value], protect_tail_count: usize) -> Vec<Value> {
    if messages.is_empty() {
        return Vec::new();
    }

    let mut result = messages.to_vec();
    let prune_boundary = result.len().saturating_sub(protect_tail_count);
    for msg in result.iter_mut().take(prune_boundary) {
        if role_of(msg) != "tool" {
            continue;
        }
        let content = content_of(msg);
        if content.len() > 200 && content != PRUNED_TOOL_PLACEHOLDER {
            set_content(msg, PRUNED_TOOL_PLACEHOLDER.to_string());
        }
    }
    result
}

fn align_boundary_forward(messages: &[Value], mut idx: usize) -> usize {
    while idx < messages.len() && role_of(&messages[idx]) == "tool" {
        idx += 1;
    }
    idx
}

fn align_boundary_backward(messages: &[Value], mut idx: usize) -> usize {
    if idx == 0 || idx >= messages.len() {
        return idx;
    }
    let mut check = idx - 1;
    while role_of(&messages[check]) == "tool" {
        if check == 0 {
            return idx;
        }
        check -= 1;
    }
    if role_of(&messages[check]) == "assistant" && has_tool_calls(&messages[check]) {
        idx = check;
    }
    idx
}

fn find_tail_cut_by_tokens(messages: &[Value], head_end: usize, protect_last_n: usize) -> usize {
    let n = messages.len();
    let mut accumulated = 0usize;
    let mut cut_idx = n;
    let token_budget = protect_last_n.max(1) * 4000;

    for i in (head_end..n).rev() {
        let msg = &messages[i];
        let mut msg_tokens = content_of(msg).len() / CHARS_PER_TOKEN + 10;
        if let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) {
            for tc in tool_calls {
                let args = tc
                    .get("function")
                    .and_then(Value::as_object)
                    .and_then(|obj| obj.get("arguments"))
                    .map(stringify_value)
                    .unwrap_or_default();
                msg_tokens += args.len() / CHARS_PER_TOKEN;
            }
        }
        if accumulated + msg_tokens > token_budget && (n - i) >= protect_last_n {
            break;
        }
        accumulated += msg_tokens;
        cut_idx = i;
    }

    let fallback_cut = n.saturating_sub(protect_last_n);
    if cut_idx > fallback_cut {
        cut_idx = fallback_cut;
    }
    if cut_idx <= head_end {
        cut_idx = fallback_cut;
    }
    let cut_idx = align_boundary_backward(messages, cut_idx);
    cut_idx.max(head_end + 1)
}

fn sanitize_tool_pairs(messages: &[Value]) -> Vec<Value> {
    let mut surviving_call_ids: Vec<String> = Vec::new();
    for msg in messages {
        if role_of(msg) != "assistant" {
            continue;
        }
        if let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) {
            for tc in tool_calls {
                let call_id = tc.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                if !call_id.is_empty() {
                    surviving_call_ids.push(call_id);
                }
            }
        }
    }

    let mut filtered: Vec<Value> = messages
        .iter()
        .filter(|msg| {
            if role_of(msg) != "tool" {
                return true;
            }
            let call_id = msg.get("tool_call_id").and_then(Value::as_str).unwrap_or("");
            call_id.is_empty() || surviving_call_ids.iter().any(|id| id == call_id)
        })
        .cloned()
        .collect();

    let mut result_call_ids: Vec<String> = filtered
        .iter()
        .filter(|msg| role_of(msg) == "tool")
        .filter_map(|msg| msg.get("tool_call_id").and_then(Value::as_str).map(str::to_string))
        .collect();

    let mut patched: Vec<Value> = Vec::new();
    for msg in filtered.drain(..) {
        let role = role_of(&msg);
        patched.push(msg.clone());
        if role != "assistant" {
            continue;
        }
        if let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) {
            for tc in tool_calls {
                let call_id = tc.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                if call_id.is_empty() || result_call_ids.iter().any(|id| id == &call_id) {
                    continue;
                }
                patched.push(tool_stub_message(&call_id));
                result_call_ids.push(call_id);
            }
        }
    }

    patched
}

fn json_message(role: &str, content: String) -> Value {
    let mut map = Map::new();
    map.insert("role".to_string(), Value::String(role.to_string()));
    map.insert("content".to_string(), Value::String(content));
    Value::Object(map)
}

fn tool_stub_message(call_id: &str) -> Value {
    let mut map = Map::new();
    map.insert("role".to_string(), Value::String("tool".to_string()));
    map.insert(
        "content".to_string(),
        Value::String("[Result from earlier conversation — see context summary above]".to_string()),
    );
    map.insert("tool_call_id".to_string(), Value::String(call_id.to_string()));
    Value::Object(map)
}

fn role_of(message: &Value) -> String {
    message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("user")
        .to_string()
}

fn content_of(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => stringify_value(other),
    }
}

fn set_content(message: &mut Value, content: String) {
    if let Some(obj) = message.as_object_mut() {
        obj.insert("content".to_string(), Value::String(content));
    }
}

fn has_tool_calls(message: &Value) -> bool {
    message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|value| !value.is_empty())
        .unwrap_or(false)
}

fn stringify_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn with_summary_prefix(summary: &str) -> String {
    let mut text = summary.trim().to_string();
    for prefix in ["[CONTEXT SUMMARY]:", SUMMARY_PREFIX] {
        if text.starts_with(prefix) {
            text = text[prefix.len()..].trim_start().to_string();
            break;
        }
    }
    if text.is_empty() {
        SUMMARY_PREFIX.to_string()
    } else {
        format!("{SUMMARY_PREFIX}\n{text}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn plan_returns_compression_window() {
        let messages = vec![
            json!({"role": "user", "content": "m0"}),
            json!({"role": "assistant", "content": "m1"}),
            json!({"role": "user", "content": "m2"}),
            json!({"role": "assistant", "content": "m3"}),
            json!({"role": "user", "content": "m4"}),
            json!({"role": "assistant", "content": "m5"}),
        ];

        let plan = plan(&messages, 2, 2);
        assert!(plan.needs_compression);
        assert!(plan.compress_start < plan.compress_end);
    }

    #[test]
    fn apply_inserts_summary_without_breaking_roles() {
        let messages = vec![
            json!({"role": "user", "content": "m0"}),
            json!({"role": "assistant", "content": "m1"}),
            json!({"role": "user", "content": "m2"}),
            json!({"role": "assistant", "content": "m3"}),
            json!({"role": "user", "content": "m4"}),
            json!({"role": "assistant", "content": "m5"}),
        ];

        let applied = apply(&messages, 2, 4, Some("summary text"), 0);
        assert!(applied.iter().any(|msg| content_of(msg).starts_with(SUMMARY_PREFIX)));
    }

    #[test]
    fn sanitize_keeps_tool_pairs_consistent() {
        let messages = vec![
            json!({"role": "user", "content": "m0"}),
            json!({"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": {"name": "tool", "arguments": "{}"}}]}),
            json!({"role": "user", "content": "tail"}),
        ];

        let applied = sanitize_tool_pairs(&messages);
        let tool_results = applied
            .iter()
            .filter(|msg| role_of(msg) == "tool")
            .count();
        assert_eq!(tool_results, 1);
    }
}
