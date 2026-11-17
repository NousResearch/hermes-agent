//! Shapes the profile gateway expects, mapped back onto the fields this screen
//! already reads. Session edits go through `session.mutate`.

use serde_json::{json, Value};

const COMPACTION_MARKERS: [&str; 3] = [
    "[CONTEXT COMPACTION",
    "[CONTEXT SUMMARY]:",
    "[END OF PRIOR CONTEXT",
];

pub fn remember_fields(snapshot: &Value) -> Option<(String, String)> {
    let session_id = snapshot
        .get("session_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())?;
    let subscription_id = snapshot
        .get("subscription_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())?;
    Some((session_id.to_string(), subscription_id.to_string()))
}

pub fn mutate_params(
    session_id: &str,
    request_id: &str,
    snapshot: &Value,
    operation: &str,
    payload: Value,
    with_generation: bool,
) -> Option<Value> {
    let revision = snapshot.get("revision").and_then(as_i64)?;
    let mut params = json!({
        "session_id": session_id,
        "request_id": request_id,
        "expected_revision": revision,
        "operation": operation,
        "payload": payload,
    });
    if with_generation {
        let generation = snapshot.get("execution_generation").and_then(as_i64)?;
        params["expected_generation"] = json!(generation);
    }
    Some(params)
}

pub fn last_user_row_id(snapshot: &Value) -> Option<i64> {
    let rows = snapshot
        .get("stored_messages")
        .or_else(|| snapshot.get("display_messages"))
        .or_else(|| snapshot.get("messages"))?
        .as_array()?;
    rows.iter().rev().find_map(user_turn_row)
}

fn user_turn_row(row: &Value) -> Option<i64> {
    if row.get("role").and_then(|v| v.as_str()) != Some("user") {
        return None;
    }
    if row.get("archived").and_then(|v| v.as_bool()) == Some(true) {
        return None;
    }
    match row.get("display_kind").and_then(|v| v.as_str()) {
        None | Some("steer") => {}
        Some(_) => return None,
    }
    let text = row
        .get("content")
        .and_then(|v| v.as_str())
        .or_else(|| row.get("text").and_then(|v| v.as_str()))
        .unwrap_or("");
    if COMPACTION_MARKERS.iter().any(|mark| text.contains(mark)) {
        return None;
    }
    row.get("row_id").and_then(as_i64)
}

pub fn undo_view(result: &Value) -> Value {
    let removed = result
        .get("rewound_count")
        .and_then(as_i64)
        .unwrap_or(0)
        .max(0) as u64;
    json!({"removed": removed, "rewound_count": removed})
}

pub fn branch_view(name: &str, branched_session_id: &str, snapshot: Option<&Value>) -> Value {
    let title = snapshot
        .and_then(|snap| snap.pointer("/info/title").and_then(|v| v.as_str()))
        .filter(|s| !s.is_empty())
        .unwrap_or(if name.is_empty() { "branch" } else { name });
    let messages = snapshot
        .and_then(|snap| {
            snap.get("display_messages")
                .or_else(|| snap.get("messages"))
        })
        .cloned()
        .unwrap_or_else(|| json!([]));
    json!({
        "session_id": branched_session_id,
        "title": title,
        "messages": messages,
    })
}

pub fn title_view(result: &Value, fallback: &str) -> Value {
    let title = result
        .get("title")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(fallback);
    json!({"title": title})
}

pub fn usage_view(account: &Value) -> Value {
    let mut lines = Vec::new();
    push_lines(&mut lines, account.get("lines"));
    push_lines(&mut lines, account.get("nous_lines"));
    json!({
        "calls": 0,
        "input": 0,
        "output": 0,
        "total": 0,
        "credits_lines": lines,
    })
}

fn push_lines(out: &mut Vec<Value>, rows: Option<&Value>) {
    let Some(rows) = rows.and_then(|v| v.as_array()) else {
        return;
    };
    for row in rows {
        if let Some(text) = row.as_str() {
            if !text.is_empty() {
                out.push(Value::String(text.to_string()));
            }
        }
    }
}

fn as_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|n| i64::try_from(n).ok()))
        .or_else(|| value.as_f64().map(|n| n as i64))
}

/// Map whatever usage verb answered onto the fields `/credits` already reads.
pub fn present_usage(value: &Value) -> Value {
    if value.get("calls").is_some() || value.get("credits_lines").is_some() {
        return value.clone();
    }
    if value.get("lines").is_some() || value.get("nous_lines").is_some() {
        return usage_view(value);
    }
    if value.get("api_calls").is_some() || value.get("input_tokens").is_some() {
        return json!({
            "calls": value.get("api_calls").cloned().unwrap_or(json!(0)),
            "input": value.get("input_tokens").cloned().unwrap_or(json!(0)),
            "output": value.get("output_tokens").cloned().unwrap_or(json!(0)),
            "total": value.get("total_tokens").cloned().unwrap_or(json!(0)),
        });
    }
    value.clone()
}

pub fn method_missing(err: &anyhow::Error) -> bool {
    let text = err.to_string();
    text.contains("(-32601)") || text.contains("method not found")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rewind_skips_compaction_and_archived_rows() {
        let snap = json!({
            "messages": [
                {"role": "user", "row_id": 1, "content": "keep"},
                {"role": "assistant", "row_id": 2, "content": "done"},
                {"role": "user", "row_id": 3, "content": "[CONTEXT COMPACTION] old", "display_kind": null},
                {"role": "user", "row_id": 4, "archived": true, "content": "gone"}
            ]
        });
        assert_eq!(last_user_row_id(&snap), Some(1));
    }

    #[test]
    fn mutate_rename_omits_generation() {
        let snap = json!({"revision": 8, "execution_generation": 2});
        let params =
            mutate_params("s", "r", &snap, "rename", json!({"title": "A"}), false).unwrap();
        assert_eq!(params["expected_revision"], 8);
        assert!(params.get("expected_generation").is_none());
    }

    #[test]
    fn mutate_rewind_requires_generation() {
        let snap = json!({"revision": 8});
        assert!(mutate_params("s", "r", &snap, "rewind", json!({}), true).is_none());
    }

    #[test]
    fn usage_puts_provider_lines_before_nous() {
        let view = usage_view(&json!({
            "lines": ["plan 10"],
            "nous_lines": ["balance 2"],
        }));
        assert_eq!(view["credits_lines"], json!(["plan 10", "balance 2"]));
    }
}
