use serde_json::{json, Value};
use std::env;
use std::io::{self, BufRead, Write};
use std::process;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let wants_health = args.iter().any(|arg| arg == "--health");
    let wants_serve = args.iter().any(|arg| arg == "--serve");

    if wants_health {
        let payload = hermes_proto::health_response("health-check");
        println!("{}", serde_json::to_string(&payload).unwrap());
        return;
    }

    if wants_serve {
        let db_path = value_after(&args, "--db-path");
        let max_size = value_after(&args, "--max-size")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(100);

        let store = match hermes_store::ResponseStoreBackend::new(max_size, db_path.as_deref()) {
            Ok(store) => store,
            Err(err) => {
                eprintln!("failed to initialize response store: {err}");
                process::exit(1);
            }
        };

        serve_loop(store);
        return;
    }

    eprintln!("usage: hermes-sidecar --health | --serve [--db-path PATH] [--max-size N]");
    process::exit(2);
}

fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn serve_loop(store: hermes_store::ResponseStoreBackend) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(err) => {
                write_response(
                    &mut stdout,
                    error_response("unknown", "read_error", &err.to_string(), true),
                );
                break;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let request: hermes_proto::RpcRequest = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(err) => {
                write_response(
                    &mut stdout,
                    error_response("unknown", "invalid_json", &err.to_string(), false),
                );
                continue;
            }
        };

        let response = handle_request(&store, request);
        write_response(&mut stdout, response);
    }
}

fn write_response(stdout: &mut io::Stdout, payload: Value) {
    writeln!(stdout, "{}", payload).unwrap();
    stdout.flush().unwrap();
}

fn handle_request(store: &hermes_store::ResponseStoreBackend, request: hermes_proto::RpcRequest) -> Value {
    match request.method.as_str() {
        "health" => json!(hermes_proto::health_response(request.id)),
        "store.get" => {
            let Some(response_id) = request.params.get("response_id").and_then(Value::as_str) else {
                return error_response(request.id, "bad_request", "missing response_id", false);
            };

            match store.get(response_id) {
                Ok(value) => success_response(request.id, json!({ "value": value })),
                Err(err) => error_response(request.id, "store_error", &err.to_string(), true),
            }
        }
        "store.put" => {
            let Some(response_id) = request.params.get("response_id").and_then(Value::as_str) else {
                return error_response(request.id, "bad_request", "missing response_id", false);
            };
            let Some(data) = request.params.get("data") else {
                return error_response(request.id, "bad_request", "missing data", false);
            };

            match store.put(response_id, data) {
                Ok(()) => success_response(request.id, json!({ "stored": true })),
                Err(err) => error_response(request.id, "store_error", &err.to_string(), true),
            }
        }
        "store.delete" => {
            let Some(response_id) = request.params.get("response_id").and_then(Value::as_str) else {
                return error_response(request.id, "bad_request", "missing response_id", false);
            };

            match store.delete(response_id) {
                Ok(deleted) => success_response(request.id, json!({ "deleted": deleted })),
                Err(err) => error_response(request.id, "store_error", &err.to_string(), true),
            }
        }
        "conversation.get" => {
            let Some(name) = request.params.get("name").and_then(Value::as_str) else {
                return error_response(request.id, "bad_request", "missing name", false);
            };

            match store.get_conversation(name) {
                Ok(value) => success_response(request.id, json!({ "value": value })),
                Err(err) => error_response(request.id, "store_error", &err.to_string(), true),
            }
        }
        "conversation.set" => {
            let Some(name) = request.params.get("name").and_then(Value::as_str) else {
                return error_response(request.id, "bad_request", "missing name", false);
            };
            let Some(response_id) = request.params.get("response_id").and_then(Value::as_str) else {
                return error_response(request.id, "bad_request", "missing response_id", false);
            };

            match store.set_conversation(name, response_id) {
                Ok(()) => success_response(request.id, json!({ "stored": true })),
                Err(err) => error_response(request.id, "store_error", &err.to_string(), true),
            }
        }
        "store.len" => match store.len() {
            Ok(value) => success_response(request.id, json!({ "value": value })),
            Err(err) => error_response(request.id, "store_error", &err.to_string(), true),
        },
        _ => error_response(request.id, "unknown_method", "unsupported method", false),
    }
}

fn success_response(id: impl Into<String>, result: Value) -> Value {
    json!({
        "version": hermes_proto::PROTOCOL_VERSION,
        "id": id.into(),
        "ok": true,
        "result": result,
    })
}

fn error_response(id: impl Into<String>, code: &str, message: &str, retryable: bool) -> Value {
    json!({
        "version": hermes_proto::PROTOCOL_VERSION,
        "id": id.into(),
        "ok": false,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        }
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn health_response_is_serializable() {
        let payload = hermes_proto::health_response("unit-test");
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("\"service\":\"hermes-sidecar\""));
        assert!(json.contains("\"status\":\"ok\""));
    }

    #[test]
    fn success_response_shape_is_stable() {
        let payload = super::success_response("req-1", json!({"stored": true}));
        assert_eq!(payload["ok"], json!(true));
        assert_eq!(payload["result"]["stored"], json!(true));
    }
}
