//! One inbound JSON-RPC frame from the profile gateway.

use serde_json::Value;

use super::types::{JsonRpcEvent, JsonRpcResponse};

#[derive(Debug)]
pub enum Inbound {
    Event(JsonRpcEvent),
    Response(JsonRpcResponse),
    /// Server→client request. `key` is what the UI sends back as `request_id`.
    Server {
        key: String,
        id: Value,
        method: String,
        params: Value,
    },
    Ignore,
}

pub fn classify(val: Value) -> Inbound {
    let method = val
        .get("method")
        .and_then(|m| m.as_str())
        .map(|s| s.to_string());
    if method.as_deref() == Some("event") {
        return match serde_json::from_value::<JsonRpcEvent>(val) {
            Ok(evt) => Inbound::Event(evt),
            Err(_) => Inbound::Ignore,
        };
    }
    let has_result = val.get("result").is_some() || val.get("error").is_some();
    if method.is_none() || has_result {
        return match serde_json::from_value::<JsonRpcResponse>(val) {
            Ok(resp) => Inbound::Response(resp),
            Err(_) => Inbound::Ignore,
        };
    }
    let Some(method) = method else {
        return Inbound::Ignore;
    };
    let Some(id) = val.get("id").cloned() else {
        return Inbound::Ignore;
    };
    let key = id_key(&id);
    let params = val.get("params").cloned().unwrap_or(Value::Null);
    Inbound::Server {
        key,
        id,
        method,
        params,
    }
}

pub fn id_key(id: &Value) -> String {
    match id {
        Value::String(s) => format!("s:{s}"),
        Value::Number(n) => format!("n:{n}"),
        other => format!("o:{other}"),
    }
}

/// Event name the existing screen already paints, plus params carrying `request_id`.
pub fn server_event(method: &str, key: &str, params: &Value) -> Option<(&'static str, Value)> {
    let event = match method {
        "approval" => "approval.request",
        "clarify" => "clarify.request",
        "sudo" => "sudo.request",
        "secret" => "secret.request",
        _ => return None,
    };
    let mut payload = match params {
        Value::Object(map) => Value::Object(map.clone()),
        _ => serde_json::json!({}),
    };
    if let Some(obj) = payload.as_object_mut() {
        obj.insert("request_id".into(), Value::String(key.to_string()));
    }
    Some((event, payload))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn event_frame_stays_an_event() {
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "event",
            "params": {"type": "gateway.ready", "payload": {}}
        });
        assert!(matches!(classify(frame), Inbound::Event(_)));
    }

    #[test]
    fn numeric_response_is_not_a_server_request() {
        let frame = json!({"jsonrpc": "2.0", "id": 4, "result": {"ok": true}});
        assert!(matches!(classify(frame), Inbound::Response(_)));
    }

    #[test]
    fn approval_request_keeps_its_string_id() {
        let frame = json!({
            "jsonrpc": "2.0",
            "id": "srq-7",
            "method": "approval",
            "params": {"session_id": "s", "command": "rm", "request_id": "inner"}
        });
        match classify(frame) {
            Inbound::Server {
                key,
                method,
                params,
                ..
            } => {
                assert_eq!(key, "s:srq-7");
                assert_eq!(method, "approval");
                let (event, payload) = server_event(&method, &key, &params).expect("event");
                assert_eq!(event, "approval.request");
                assert_eq!(payload["request_id"], "s:srq-7");
                assert_eq!(payload["command"], "rm");
            }
            other => panic!("expected server request, got {other:?}"),
        }
    }
}
