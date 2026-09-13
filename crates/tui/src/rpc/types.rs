use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Value,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(default)]
    pub data: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcEvent {
    pub jsonrpc: String,
    pub method: String,
    pub params: EventParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventParams {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub payload: Option<Value>,
}

impl JsonRpcEvent {
    pub fn synthetic(event_type: &str, payload: Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            method: "event".into(),
            params: EventParams {
                event_type: event_type.into(),
                session_id: None,
                payload: Some(payload),
            },
        }
    }
}

impl JsonRpcRequest {
    pub fn new(id: u64, method: impl Into<String>, params: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Value::from(id),
            method: method.into(),
            params,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn event_type_field_renames() {
        let raw = json!({
            "jsonrpc": "2.0",
            "method": "event",
            "params": { "type": "message.delta", "payload": { "text": "hi" } }
        });
        let evt: JsonRpcEvent = serde_json::from_value(raw).unwrap();
        assert_eq!(evt.params.event_type, "message.delta");
        assert_eq!(
            evt.params.payload.unwrap().get("text").unwrap().as_str(),
            Some("hi")
        );
    }

    #[test]
    fn response_error_roundtrip() {
        let raw = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": { "code": 4001, "message": "no session" }
        });
        let resp: JsonRpcResponse = serde_json::from_value(raw).unwrap();
        assert_eq!(resp.error.unwrap().code, 4001);
    }
}
