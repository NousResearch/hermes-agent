use super::*;
use axum::{
    Json, Router,
    body::{Body, Bytes},
    http::{HeaderMap, StatusCode},
    response::Response,
    routing::post,
};
use futures_util::{Stream, stream};
use std::{
    convert::Infallible,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
};
use tokio::sync::mpsc;

const OPENAI_TOOLS: &str = include_str!("fixtures/openai_parallel_tools.sse");
const ANTHROPIC_TOOLS: &str = include_str!("fixtures/anthropic_tools.sse");
const ANTHROPIC_TEXT: &str = include_str!("fixtures/anthropic_text.sse");
const OPENAI_TEXT: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

fn config(base_url: impl Into<String>) -> Config {
    Config {
        base_url: base_url.into(),
        model: "fixture-model".into(),
        api_key: String::new(),
    }
}

fn sse(data: &'static str) -> Response {
    // Split inside JSON, SSE delimiters, and UTF-8 code points to exercise the
    // actual network/SSE decoder rather than splitting fixtures on newlines.
    let chunks = data
        .as_bytes()
        .chunks(7)
        .map(|part| Ok::<_, Infallible>(Bytes::copy_from_slice(part)))
        .collect::<Vec<_>>();
    Response::builder()
        .header("content-type", "text/event-stream; charset=utf-8")
        .body(Body::from_stream(stream::iter(chunks)))
        .unwrap()
}

async fn serve(router: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (format!("http://{address}"), task)
}

fn user_messages() -> Vec<Value> {
    vec![json!({"role":"user","content":"Hello"})]
}

#[test]
fn endpoint_detection_preserves_explicit_paths_and_supports_ipv6_loopback() {
    for (base, protocol, endpoint) in [
        (
            OPENROUTER,
            Protocol::OpenAi,
            "https://openrouter.ai/api/v1/chat/completions",
        ),
        (
            OPENAI,
            Protocol::OpenAi,
            "https://api.openai.com/v1/chat/completions",
        ),
        (
            "https://api.openai.com",
            Protocol::OpenAi,
            "https://api.openai.com/v1/chat/completions",
        ),
        (
            ANTHROPIC,
            Protocol::Anthropic,
            "https://api.anthropic.com/v1/messages",
        ),
        (
            "https://api.anthropic.com",
            Protocol::Anthropic,
            "https://api.anthropic.com/v1/messages",
        ),
        (
            "https://proxy.example/custom/messages",
            Protocol::Anthropic,
            "https://proxy.example/custom/messages",
        ),
        (
            "http://localhost:11434/v1",
            Protocol::OpenAi,
            "http://localhost:11434/v1/chat/completions",
        ),
        (
            "http://[::1]:8080/v1/chat/completions",
            Protocol::OpenAi,
            "http://[::1]:8080/v1/chat/completions",
        ),
        (
            "http://localhost:1234/v1/chat/completions/",
            Protocol::OpenAi,
            "http://localhost:1234/v1/chat/completions",
        ),
        (
            crate::portal::INFERENCE,
            Protocol::OpenAi,
            "https://inference-api.nousresearch.com/v1/chat/completions",
        ),
    ] {
        let mut config = config(base);
        config.api_key = "unit-test-key".into();
        config.validate().unwrap();
        let actual = config.endpoint().unwrap();
        assert_eq!(actual.0, protocol);
        assert_eq!(actual.1.as_str(), endpoint);
    }
}

#[test]
fn credential_fallback_is_limited_to_exact_official_origins_and_paths() {
    assert_eq!(official_key_env(OPENROUTER), Some("OPENROUTER_API_KEY"));
    assert_eq!(
        official_key_env("https://api.openai.com/v1/"),
        Some("OPENAI_API_KEY")
    );
    assert_eq!(official_key_env(ANTHROPIC), Some("ANTHROPIC_API_KEY"));
    for base in [
        "http://localhost:8000/v1",
        "https://evil.example/v1",
        "https://api.openai.com.evil.example/v1",
        "https://api.openai.com/custom",
        "https://api.openai.com:444/v1",
        "https://user@api.openai.com/v1",
        "https://api.openai.com/v1?api_key=wrong",
        crate::portal::INFERENCE,
    ] {
        assert_eq!(official_key_env(base), None);
    }
}

#[test]
fn config_rejects_remote_http_embedded_credentials_and_invalid_keys() {
    for base in [
        "http://remote.example/v1",
        "https://user:password@proxy.example/v1",
        "https://proxy.example/v1?secret=hidden",
        "https://proxy.example/v1#fragment",
    ] {
        assert!(config(base).validate().is_err());
    }
    let mut invalid = config("http://127.0.0.1:1234/v1");
    invalid.api_key = "test\r\nheader".into();
    assert!(invalid.validate().is_err());
    assert!(
        config(crate::portal::INFERENCE).validate().is_ok(),
        "Portal configuration can be validated before sign-in"
    );
}

#[test]
fn anthropic_request_translates_parallel_results_denials_and_tool_schemas() {
    let tools = vec![
        json!({"type":"function","function":{"name":"read_file","description":"Read a file","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}),
    ];
    let messages = vec![
        json!({"role":"system","content":"System instructions"}),
        json!({"role":"user","content":"Read the files"}),
        json!({"role":"assistant","content":"I'll read both.","tool_calls":[
            {"id":"call_one","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"one.txt\"}"}},
            {"id":"call_two","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"two.txt\"}"}}
        ]}),
        json!({"role":"tool","tool_call_id":"call_one","content":"file contents"}),
        json!({"role":"tool","tool_call_id":"call_two","content":"{\"error\":\"User denied this tool call\"}"}),
        json!({"role":"user","content":"Use what you have"}),
    ];
    let body = anthropic_body("claude-fixture", messages, &tools).unwrap();
    assert_eq!(body["system"], "System instructions");
    assert_eq!(body["max_tokens"], 8192);
    assert_eq!(body["messages"].as_array().unwrap().len(), 3);
    assert_eq!(
        body["messages"][1]["content"][1],
        json!({"type":"tool_use","id":"call_one","name":"read_file","input":{"path":"one.txt"}})
    );
    let results = &body["messages"][2]["content"];
    assert_eq!(results[0]["tool_use_id"], "call_one");
    assert_eq!(results[0]["is_error"], false);
    assert_eq!(results[1]["tool_use_id"], "call_two");
    assert_eq!(results[1]["is_error"], true);
    assert_eq!(
        results[2],
        json!({"type":"text","text":"Use what you have"})
    );
    assert_eq!(
        body["tools"][0]["input_schema"],
        tools[0]["function"]["parameters"]
    );
    assert!(body["tools"][0].get("function").is_none());
}

#[tokio::test]
async fn openai_stream_assembles_parallel_tool_deltas_over_real_http() {
    let (captured_tx, mut captured_rx) = mpsc::channel(1);
    let router = Router::new().route(
        "/v1/chat/completions",
        post(
            move |headers: HeaderMap, Json(body): Json<Value>| async move {
                captured_tx.send((headers, body)).await.unwrap();
                sse(OPENAI_TOOLS)
            },
        ),
    );
    let (base, server) = serve(router).await;
    let mut config = config(format!("{base}/v1"));
    config.api_key = "unit-test-key".into();
    let provider = Provider::new(config).unwrap();
    let mut text = String::new();
    let custom_tool = json!({"type":"function","function":{"name":"custom","parameters":{"type":"object","properties":{}}}});
    let calls = provider
        .stream_with_tools(
            user_messages(),
            std::slice::from_ref(&custom_tool),
            |delta| {
                text.push_str(delta);
                Ok(())
            },
        )
        .await
        .unwrap();
    assert_eq!(text, "I'll inspect the workspace.");
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls[0],
        ToolCall {
            id: "call_read".into(),
            name: "read_file".into(),
            arguments: "{\"path\":\"notes/hello.txt\"}".into()
        }
    );
    assert_eq!(calls[1].arguments, "{}");
    let (headers, body) = captured_rx.recv().await.unwrap();
    assert_eq!(headers["authorization"], "Bearer unit-test-key");
    assert_eq!(
        headers["user-agent"],
        concat!("HermesRust/", env!("CARGO_PKG_VERSION"))
    );
    assert!(!headers.contains_key("x-api-key"));
    assert_eq!(body["tools"], json!([custom_tool]));
    server.abort();
}

#[tokio::test]
async fn native_anthropic_stream_assembles_tools_and_sends_native_headers() {
    let (captured_tx, mut captured_rx) = mpsc::channel(1);
    let router = Router::new().route(
        "/v1/messages",
        post(
            move |headers: HeaderMap, Json(body): Json<Value>| async move {
                captured_tx.send((headers, body)).await.unwrap();
                sse(ANTHROPIC_TOOLS)
            },
        ),
    );
    let (base, server) = serve(router).await;
    let mut config = config(format!("{base}/v1/messages"));
    config.api_key = "unit-test-key".into();
    let provider = Provider::new(config).unwrap();
    let mut text = String::new();
    let calls = provider
        .stream(user_messages(), |delta| {
            text.push_str(delta);
            Ok(())
        })
        .await
        .unwrap();
    assert_eq!(text, "I'll read the note.");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id, "toolu_read");
    assert_eq!(calls[0].arguments, "{\"path\":\"notes/hello.txt\"}");
    assert_eq!(calls[1].arguments, "{}");
    let (headers, body) = captured_rx.recv().await.unwrap();
    assert_eq!(headers["x-api-key"], "unit-test-key");
    assert_eq!(headers["anthropic-version"], "2023-06-01");
    assert!(!headers.contains_key("authorization"));
    assert!(body["tools"][0]["input_schema"].is_object());
    server.abort();
}

#[tokio::test]
async fn native_anthropic_text_handles_split_utf8_and_terminal_message_stop() {
    let (base, server) =
        serve(Router::new().route("/v1/messages", post(|| async { sse(ANTHROPIC_TEXT) }))).await;
    let provider = Provider::new(config(format!("{base}/v1/messages"))).unwrap();
    let mut text = String::new();
    let calls = provider
        .stream(user_messages(), |delta| {
            text.push_str(delta);
            Ok(())
        })
        .await
        .unwrap();
    assert_eq!(text, "Hello 🌏");
    assert!(calls.is_empty());
    server.abort();
}

#[tokio::test]
async fn local_provider_needs_no_auth_and_empty_tool_registry_is_supported() {
    let (captured_tx, mut captured_rx) = mpsc::channel(1);
    let router = Router::new().route(
        "/v1/chat/completions",
        post(
            move |headers: HeaderMap, Json(body): Json<Value>| async move {
                captured_tx.send((headers, body)).await.unwrap();
                sse(OPENAI_TEXT)
            },
        ),
    );
    let (base, server) = serve(router).await;
    let provider = Provider::new(config(base)).unwrap();
    let calls = provider
        .stream_with_tools(user_messages(), &[], |_| Ok(()))
        .await
        .unwrap();
    assert!(calls.is_empty());
    let (headers, body) = captured_rx.recv().await.unwrap();
    assert!(!headers.contains_key("authorization"));
    assert!(body.get("tools").is_none());
    server.abort();
}

#[test]
fn openai_finish_reasons_never_dispatch_truncated_tools() {
    for reason in ["length", "content_filter", "stop", "function_call", "error"] {
        let mut state = OpenAiStream::default();
        state.accept(&json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_test","function":{"name":"write_file","arguments":"{\"path\":"}}]},"finish_reason":reason}]}).to_string(), &mut |_| Ok(())).unwrap();
        assert!(
            state.finish().is_err(),
            "Unexpected usable result for {reason}"
        );
    }
    assert!(OpenAiStream::default().finish().is_err());
}

#[test]
fn invalid_duplicate_or_excessive_tools_are_rejected() {
    let call = ToolCall {
        id: "same".into(),
        name: "read_file".into(),
        arguments: "{}".into(),
    };
    assert!(valid_calls(vec![call.clone(), call]).is_err());
    for arguments in ["[]", "null", "{\"unfinished\":"] {
        assert!(
            valid_calls(vec![ToolCall {
                id: "call".into(),
                name: "read_file".into(),
                arguments: arguments.into()
            }])
            .is_err()
        );
    }
    let mut state = OpenAiStream::default();
    assert!(state.accept(&json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":8,"id":"call","function":{"name":"read_file","arguments":"{}"}}]},"finish_reason":null}]}).to_string(), &mut |_| Ok(())).is_err());
}

#[test]
fn anthropic_disconnect_and_limit_never_dispatch_tools() {
    let mut state = AnthropicStream::default();
    for data in ANTHROPIC_TOOLS
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
    {
        if data.contains("\"type\":\"message_stop\"") {
            break;
        }
        state.accept(data, &mut |_| Ok(())).unwrap();
    }
    assert!(state.finish().is_err());
    let mut limited = AnthropicStream::default();
    for data in ANTHROPIC_TOOLS
        .replace(
            "\"stop_reason\":\"tool_use\"",
            "\"stop_reason\":\"max_tokens\"",
        )
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
    {
        limited.accept(data, &mut |_| Ok(())).unwrap();
    }
    assert!(
        limited
            .finish()
            .unwrap_err()
            .to_string()
            .contains("token limit")
    );
}

#[tokio::test]
async fn http_and_in_stream_errors_redact_configured_credentials() {
    for (path, router) in [
        ("/v1/chat/completions", Router::new().route("/v1/chat/completions", post(|| async { (StatusCode::UNAUTHORIZED, Json(json!({"error":{"message":"Invalid unit-test-secret"}}))) }))),
        ("/v1/chat/completions", Router::new().route("/v1/chat/completions", post(|| async { sse("data: {\"error\":{\"code\":429,\"message\":\"Limited unit-test-secret\"}}\n\n") }))),
        ("/v1/chat/completions", Router::new().route("/v1/chat/completions", post(|| async { sse("event: error\ndata: {\"message\":\"Failed unit-test-secret\"}\n\n") }))),
        ("/v1/messages", Router::new().route("/v1/messages", post(|| async { sse("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded unit-test-secret\"}}\n\n") }))),
    ] {
        let (base, server) = serve(router).await;
        let mut config = config(format!("{base}{path}"));
        config.api_key = "unit-test-secret".into();
        let error = Provider::new(config).unwrap().stream(user_messages(), |_| Ok(())).await.unwrap_err().to_string();
        assert!(error.contains("[redacted]"));
        assert!(!error.contains("unit-test-secret"));
        server.abort();
    }
}

#[tokio::test]
async fn http_error_supports_top_level_detail_and_non_json_access_hints() {
    let router = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            (
                StatusCode::FORBIDDEN,
                Json(json!({"detail":"Inference scope missing"})),
            )
        }),
    );
    let (base, server) = serve(router).await;
    let error = Provider::new(config(format!("{base}/v1")))
        .unwrap()
        .stream(user_messages(), |_| Ok(()))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("Inference scope missing"));
    server.abort();
    let router = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            (
                StatusCode::FORBIDDEN,
                [("content-type", "text/html")],
                "<html>gateway response</html>",
            )
        }),
    );
    let (base, server) = serve(router).await;
    let error = Provider::new(config(format!("{base}/v1")))
        .unwrap()
        .stream(user_messages(), |_| Ok(()))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("Inference access denied"));
    assert!(error.contains("non-JSON error; content type: text/html"));
    assert!(!error.contains("<html>"));
    server.abort();
}

#[test]
fn refusals_complete_without_tools_and_unknown_anthropic_events_are_ignored() {
    let mut state = OpenAiStream::default();
    let mut text = String::new();
    state.accept(r#"{"choices":[{"index":0,"delta":{"refusal":"I cannot help with that."},"finish_reason":"stop"}]}"#, &mut |delta| { text.push_str(delta); Ok(()) }).unwrap();
    assert!(state.finish().unwrap().is_empty());
    assert_eq!(text, "I cannot help with that.");
    let mut state = AnthropicStream::default();
    state
        .accept(r#"{"type":"future_event","data":"ignored"}"#, &mut |_| {
            Ok(())
        })
        .unwrap();
    for data in ANTHROPIC_TEXT
        .replace(
            "\"stop_reason\":\"end_turn\"",
            "\"stop_reason\":\"refusal\"",
        )
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
    {
        state.accept(data, &mut |_| Ok(())).unwrap();
    }
    assert!(state.finish().unwrap().is_empty());
}

#[test]
fn malformed_events_and_closed_ui_abort_before_tools_can_execute() {
    let mut state = OpenAiStream::default();
    assert!(state.accept("invalid JSON", &mut |_| Ok(())).is_err());
    assert!(
        state
            .accept(
                r#"{"choices":[{"index":0,"delta":{"content":"text"},"finish_reason":"stop"}]}"#,
                &mut |_| Err(anyhow!("UI closed"))
            )
            .is_err()
    );
    let mut state = AnthropicStream::default();
    assert!(state.accept(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"no start"}}"#, &mut |_| Ok(())).is_err());
}

struct PendingBody {
    sent: bool,
    dropped: Arc<AtomicBool>,
}
impl Stream for PendingBody {
    type Item = std::result::Result<Bytes, Infallible>;
    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.sent {
            return Poll::Pending;
        }
        self.sent = true;
        Poll::Ready(Some(Ok(Bytes::from_static(b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Started\"},\"finish_reason\":null}]}\n\n"))))
    }
}
impl Drop for PendingBody {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn cancelling_stream_drops_http_response_and_server_body() {
    let dropped = Arc::new(AtomicBool::new(false));
    let server_dropped = dropped.clone();
    let router = Router::new().route(
        "/v1/chat/completions",
        post(move || async move {
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from_stream(PendingBody {
                    sent: false,
                    dropped: server_dropped,
                }))
                .unwrap()
        }),
    );
    let (base, server) = serve(router).await;
    let provider = Provider::new(config(format!("{base}/v1"))).unwrap();
    let (seen_tx, mut seen_rx) = mpsc::unbounded_channel();
    {
        let request = provider.stream(user_messages(), |text| {
            seen_tx.send(text.to_owned()).unwrap();
            Ok(())
        });
        tokio::pin!(request);
        tokio::select! {
            result = &mut request => panic!("Pending stream ended unexpectedly: {result:?}"),
            text = seen_rx.recv() => assert_eq!(text.as_deref(), Some("Started")),
            _ = tokio::time::sleep(Duration::from_secs(3)) => panic!("No initial stream text"),
        }
        // Leaving the scope drops the future exactly as Session cancellation does.
    }
    tokio::time::timeout(Duration::from_secs(3), async {
        while !dropped.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("Cancellation did not close the server's response stream");
    server.abort();
}

#[test]
fn permits_empty_usage_trailer_after_finish_but_rejects_late_tokens() {
    let mut state = OpenAiStream::default();
    let mut text = String::new();
    let mut emit = |delta: &str| {text.push_str(delta); Ok(())};
    state.accept(r#"{"choices":[{"index":0,"delta":{"content":"finished"},"finish_reason":"stop"}]}"#, &mut emit).unwrap();
    state.accept(r#"{"choices":[{"index":0,"delta":{},"finish_reason":null}],"usage":{"completion_tokens":1}}"#, &mut emit).unwrap();
    assert!(state.accept(r#"{"choices":[{"index":0,"delta":{"content":"untrusted late text"},"finish_reason":null}]}"#, &mut emit).is_err());
    assert_eq!(text,"finished");
}

#[test]
fn codex_subscription_uses_the_responses_protocol() {
    let config = Config { base_url: "https://chatgpt.com/backend-api/codex".into(), model: "gpt-5.5".into(), api_key: "t".into() };
    let (protocol, url) = config.endpoint().unwrap();
    assert_eq!(protocol, Protocol::Responses);
    assert_eq!(url.as_str(), "https://chatgpt.com/backend-api/codex/responses");
    let minimax = Config { base_url: "https://api.minimax.io/anthropic".into(), model: "MiniMax-M2".into(), api_key: "t".into() };
    let (protocol, url) = minimax.endpoint().unwrap();
    assert_eq!((protocol, url.as_str()), (Protocol::Anthropic, "https://api.minimax.io/anthropic/v1/messages"));
    let body = responses_body("gpt-5.5", vec![
        json!({"role":"system","content":"Be brief."}), json!({"role":"user","content":"hi"}),
        json!({"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{}"}}]}),
        json!({"role":"tool","tool_call_id":"c1","content":"ok"}),
    ], &[json!({"type":"function","function":{"name":"read_file","description":"Read","parameters":{"type":"object"}}})]);
    assert_eq!(body["instructions"], "Be brief.");
    assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(body["input"][1]["type"], "function_call");
    assert_eq!(body["input"][2]["type"], "function_call_output");
    assert_eq!(body["tools"][0]["name"], "read_file");
    assert_eq!(body["store"], false);
    let mut stream = ResponsesStream::default();
    let mut text = String::new();
    let mut emit = |t: &str| { text.push_str(t); Ok(()) };
    assert!(!stream.accept(r#"{"type":"response.output_text.delta","delta":"Hel"}"#, &mut emit).unwrap());
    assert!(!stream.accept(r#"{"type":"response.output_text.delta","delta":"lo"}"#, &mut emit).unwrap());
    assert!(!stream.accept(r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"c2","name":"list_directory","arguments":"{\"path\":\".\"}"}}"#, &mut emit).unwrap());
    assert!(stream.accept(r#"{"type":"response.completed","response":{}}"#, &mut emit).unwrap());
    assert_eq!(text, "Hello");
    let calls = stream.finish().unwrap();
    assert_eq!((calls[0].id.as_str(), calls[0].name.as_str()), ("c2", "list_directory"));
    // The account id from the JWT becomes ChatGPT-Account-ID.
    use base64::Engine;
    let claims = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct_1"}}"#);
    assert_eq!(codex_account_headers(&format!("h.{claims}.s")), vec![("ChatGPT-Account-ID", "acct_1".to_owned())]);
}
