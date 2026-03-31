use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PROTOCOL_VERSION: &str = "0.1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RpcRequest {
    pub version: String,
    pub id: String,
    pub method: String,
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RpcSuccess<T> {
    pub version: String,
    pub id: String,
    pub ok: bool,
    pub result: T,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RpcError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RpcFailure {
    pub version: String,
    pub id: String,
    pub ok: bool,
    pub error: RpcError,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HealthInfo {
    pub service: String,
    pub status: String,
    pub protocol_version: String,
    pub fallback_supported: bool,
}

pub fn health_response(id: impl Into<String>) -> RpcSuccess<HealthInfo> {
    RpcSuccess {
        version: PROTOCOL_VERSION.to_string(),
        id: id.into(),
        ok: true,
        result: HealthInfo {
            service: "hermes-sidecar".to_string(),
            status: "ok".to_string(),
            protocol_version: PROTOCOL_VERSION.to_string(),
            fallback_supported: true,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rpc_request_round_trip() {
        let request = RpcRequest {
            version: PROTOCOL_VERSION.to_string(),
            id: "req-1".to_string(),
            method: "health".to_string(),
            params: json!({"ping": true}),
        };

        let encoded = serde_json::to_string(&request).unwrap();
        let decoded: RpcRequest = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, request);
    }

    #[test]
    fn rpc_failure_round_trip() {
        let failure = RpcFailure {
            version: PROTOCOL_VERSION.to_string(),
            id: "req-2".to_string(),
            ok: false,
            error: RpcError {
                code: "bad_request".to_string(),
                message: "missing params".to_string(),
                retryable: false,
            },
        };

        let encoded = serde_json::to_string(&failure).unwrap();
        let decoded: RpcFailure = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, failure);
    }

    #[test]
    fn health_response_shape_is_stable() {
        let response = health_response("health-check");

        assert!(response.ok);
        assert_eq!(response.id, "health-check");
        assert_eq!(response.result.service, "hermes-sidecar");
        assert_eq!(response.result.status, "ok");
        assert_eq!(response.result.protocol_version, PROTOCOL_VERSION);
        assert!(response.result.fallback_supported);
    }
}

