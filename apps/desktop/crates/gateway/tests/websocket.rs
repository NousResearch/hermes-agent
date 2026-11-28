use axum::{
    Json, Router,
    body::{Body, Bytes},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::post,
};
use futures_util::{SinkExt, StreamExt, stream};
use hermes_engine::Config;
use hermes_gateway::{Gateway, GatewayConfig, RunningGateway};
use serde_json::{Value, json};
use std::{convert::Infallible, net::SocketAddr, time::Duration};
use tempfile::TempDir;
use tokio::{
    net::{TcpListener, TcpStream},
    task::JoinHandle,
    time::timeout,
};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async,
    tungstenite::{Message, client::IntoClientRequest, http::HeaderValue},
};

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

struct Client {
    socket: Socket,
    events: Vec<Value>,
    id: u64,
}
impl Client {
    async fn native(gateway: &RunningGateway) -> Self {
        let http = reqwest::Client::new();
        let ticket: Value = http
            .post(format!("{}/api/auth/ws-ticket", gateway.http_url()))
            .bearer_auth(gateway.token())
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap();
        let mut request = format!("{}?native_dial=1", gateway.websocket_url())
            .into_client_request()
            .unwrap();
        request.headers_mut().insert(
            "sec-websocket-protocol",
            HeaderValue::from_str(&format!(
                "hermes-gateway-v1, hermes-gateway-ticket.{}",
                ticket["ticket"].as_str().unwrap()
            ))
            .unwrap(),
        );
        let (socket, response) = connect_async(request).await.unwrap();
        assert_eq!(
            response.headers()["sec-websocket-protocol"],
            "hermes-gateway-v1"
        );
        let mut client = Self {
            socket,
            events: vec![],
            id: 0,
        };
        let ready = client.frame().await;
        assert_eq!(ready["params"]["type"], "gateway.ready");
        assert_eq!(ready["params"]["payload"]["session_authority"], true);
        client
    }
    async fn legacy(gateway: &RunningGateway) -> Self {
        let (socket, _) = connect_async(format!(
            "{}?token={}",
            gateway.websocket_url(),
            gateway.token()
        ))
        .await
        .unwrap();
        let mut client = Self {
            socket,
            events: vec![],
            id: 0,
        };
        assert_eq!(
            client.frame().await["params"]["payload"]["session_authority"],
            false
        );
        client
    }
    async fn frame(&mut self) -> Value {
        loop {
            let frame = timeout(Duration::from_secs(5), self.socket.next())
                .await
                .expect("gateway timed out")
                .expect("gateway closed")
                .unwrap();
            if let Message::Text(text) = frame {
                return serde_json::from_str(&text).unwrap();
            }
        }
    }
    async fn call(&mut self, method: &str, params: Value) -> Value {
        self.id += 1;
        let id = self.id;
        self.socket
            .send(Message::Text(
                json!({"jsonrpc":"2.0","id":id,"method":method,"params":params})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        loop {
            let frame = self.frame().await;
            if frame["id"] == id {
                return frame;
            }
            self.events.push(frame["params"].clone());
        }
    }
    async fn ok(&mut self, method: &str, params: Value) -> Value {
        let frame = self.call(method, params).await;
        assert!(frame.get("error").is_none(), "{method}: {frame}");
        frame["result"].clone()
    }
    async fn event(&mut self, name: &str) -> Value {
        if let Some(index) = self.events.iter().position(|e| e["type"] == name) {
            return self.events.remove(index);
        }
        loop {
            let frame = self.frame().await;
            assert_eq!(frame["method"], "event", "unexpected frame: {frame}");
            let event = frame["params"].clone();
            if event["type"] == name {
                return event;
            }
            self.events.push(event);
        }
    }
}
struct Provider {
    address: SocketAddr,
    task: JoinHandle<()>,
}
impl Drop for Provider {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Provider {
    async fn start() -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new().route("/v1/chat/completions", post(model)),
            )
            .await
            .unwrap();
        });
        Self { address, task }
    }
    fn config(&self) -> Config {
        Config {
            base_url: format!("http://{}/v1", self.address),
            model: "fixture-model".into(),
            api_key: String::new(),
        }
    }
    async fn gateway(&self, root: &TempDir) -> RunningGateway {
        // These tests drive the approval prompt, so they pin the Manual approval mode.
        let mut changes = serde_json::Map::new();
        changes.insert("approval_mode".into(), serde_json::json!("manual"));
        hermes_engine::settings::ProfileSettings::update(root.path(), &changes).unwrap();
        Gateway::start(GatewayConfig::new(root.path(), self.config()))
            .await
            .unwrap()
    }
}
fn event(delta: Value, finish: Option<&str>) -> String {
    format!(
        "data: {}\n\n",
        json!({"choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
    )
}
async fn model(Json(request): Json<Value>) -> Response {
    let messages = request["messages"].as_array().unwrap();
    let last = messages.last().unwrap();
    if last["content"] == "provider-error" {
        return (
            [(header::CONTENT_TYPE, "text/event-stream")],
            "data: {\"error\":{\"message\":\"fixture rejected\"}}\n\n",
        )
            .into_response();
    }
    if last["content"] == "slow" {
        let text = event(json!({"content":"Waiting"}), None);
        let body = stream::once(async move { Ok::<_, Infallible>(Bytes::from(text)) })
            .chain(stream::pending());
        return (
            [(header::CONTENT_TYPE, "text/event-stream")],
            Body::from_stream(body),
        )
            .into_response();
    }
    let body = if last["role"] == "tool" {
        let content = last["content"].as_str().unwrap();
        let reply = if content.contains("denied") {
            "Tool denied."
        } else {
            "Clock read successfully."
        };
        event(json!({"content":reply}), Some("stop"))
    } else if last["content"]
        .as_str()
        .is_some_and(|s| s.contains("clock"))
    {
        format!(
            "{}{}",
            event(json!({"content":"The clock needs approval."}), None),
            event(
                json!({"tool_calls":[{"index":0,"id":"clock_call","function":{"name":"get_current_time","arguments":"{}"}}]}),
                Some("tool_calls")
            )
        )
    } else {
        format!(
            "{}{}",
            event(json!({"content":"Hello 😀 "}), None),
            event(json!({"content":"world"}), Some("stop"))
        )
    };
    (
        [(header::CONTENT_TYPE, "text/event-stream")],
        format!("{body}data: [DONE]\n\n"),
    )
        .into_response()
}

#[tokio::test]
async fn ticket_auth_is_single_use_and_boundaries_fail_closed() {
    let provider = Provider::start().await;
    let root = TempDir::new().unwrap();
    let gateway = provider.gateway(&root).await;
    let http = reqwest::Client::new();
    let mint = format!("{}/api/auth/ws-ticket", gateway.http_url());
    assert_eq!(
        http.post(&mint).send().await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        http.post(&mint)
            .bearer_auth("incorrect")
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        http.post(&mint)
            .bearer_auth(gateway.token())
            .header("Origin", "https://evil.example")
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        http.post(&mint)
            .bearer_auth(gateway.token())
            .header("Host", format!("evil.example:{}", gateway.address().port()))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::FORBIDDEN
    );
    assert!(connect_async(gateway.websocket_url()).await.is_err());
    let ticket: Value = http
        .post(&mint)
        .bearer_auth(gateway.token())
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let url = format!(
        "{}?ticket={}",
        gateway.websocket_url(),
        ticket["ticket"].as_str().unwrap()
    );
    let (mut ws, _) = connect_async(&url).await.unwrap();
    assert!(
        connect_async(&url).await.is_err(),
        "a redeemed ticket must never work twice"
    );
    ws.close(None).await.unwrap();
    let mut native = Client::native(&gateway).await;
    assert_eq!(
        native.ok("gateway.ping", json!({})).await,
        json!({"ok":true})
    );
    let description = native.ok("runtime.describe", json!({})).await;
    assert_eq!(description["profile_id"], "default");
    assert!(
        description["session_create"]["parameters"]
            .as_array()
            .unwrap()
            .contains(&json!("model"))
    );
    assert_eq!(
        native.call("python.exec", json!({})).await["error"]["code"],
        -32601
    );
    assert_eq!(
        native
            .call("session.list", json!({"profile":"another"}))
            .await["error"]["data"]["reason"],
        "unsupported_profile"
    );
    native.socket.close(None).await.unwrap();
    gateway.shutdown().await.unwrap();
    let mut unsafe_config = GatewayConfig::new(root.path(), provider.config());
    unsafe_config.bind = "0.0.0.0:0".parse().unwrap();
    assert!(Gateway::start(unsafe_config).await.is_err());
}

#[tokio::test]
async fn native_turn_survives_reconnect_approval_and_restart_with_rust_history() {
    let provider = Provider::start().await;
    let root = TempDir::new().unwrap();
    let workspace = TempDir::new().unwrap();
    let gateway = provider.gateway(&root).await;
    let mut client = Client::native(&gateway).await;
    let create = json!({"request_id":"create-once","source":"gui","title":"Clock test","cwd":workspace.path(),"model":"fixture-model"});
    let session = client.ok("session.create", create.clone()).await;
    let sid = session["session_id"].as_str().unwrap().to_owned();
    assert_eq!(
        client.ok("session.create", create.clone()).await["session_id"],
        sid
    );
    assert_eq!(session["messages"], json!([]));
    assert_eq!(session["execution_generation"], 0);
    let receipt = client
        .ok(
            "prompt.submit",
            json!({"session_id":sid,"submission_id":"send-once","text":"Read the clock"}),
        )
        .await;
    assert_eq!(receipt["ref"]["session_id"], sid);
    assert_eq!(receipt["status"], "started");
    let approval = client.event("approval.request").await;
    let prompt = approval["payload"]["prompt_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let generation = approval["execution_generation"].as_u64().unwrap();
    assert_eq!(approval["payload"]["choices"], json!(["once", "deny"]));
    assert_eq!(
        client.ok("gateway.ping", json!({})).await["ok"],
        true,
        "heartbeats must work while tools wait"
    );
    assert_eq!(client.call("approval.respond",json!({"session_id":sid,"prompt_id":prompt,"execution_generation":0,"choice":"once"})).await["error"]["data"]["reason"],"stale_generation");
    client.socket.close(None).await.unwrap();
    let mut resumed = Client::native(&gateway).await;
    assert_eq!(resumed.call("approval.respond",json!({"session_id":sid,"prompt_id":prompt,"execution_generation":generation,"choice":"once"})).await["error"]["data"]["reason"],"not_attached");
    let snapshot = resumed
        .ok("session.resume", json!({"session_id":sid}))
        .await;
    assert_eq!(snapshot["running"], true);
    assert_eq!(snapshot["prompts"][0]["prompt_id"], prompt);
    assert_eq!(
        resumed
            .ok(
                "prompt.submit",
                json!({"session_id":sid,"submission_id":"send-once","text":"Read the clock"})
            )
            .await["admission_id"],
        receipt["admission_id"],
        "retry must not run a second tool call"
    );
    assert_eq!(resumed.ok("approval.respond",json!({"session_id":sid,"prompt_id":prompt,"execution_generation":generation,"choice":"once"})).await["status"],"resolved");
    let complete = resumed.event("message.complete").await;
    assert_eq!(complete["payload"]["status"], "complete");
    assert_eq!(complete["payload"]["text"], "Clock read successfully.");
    let history = resumed
        .ok("session.history", json!({"session_id":sid}))
        .await;
    assert_eq!(history["count"], 4);
    assert_eq!(history["messages"][2]["name"], "get_current_time");
    let clock: Value =
        serde_json::from_str(history["messages"][2]["content"].as_str().unwrap()).unwrap();
    assert!(clock["utc"].as_str().unwrap().ends_with('Z'));
    let terminal = resumed
        .ok(
            "prompt.receipt",
            json!({"session_id":sid,"admission_id":receipt["admission_id"],"include_result":true}),
        )
        .await;
    assert_eq!(terminal["status"], "terminal");
    assert_eq!(terminal["outcome"], "complete");
    let replay = resumed
        .ok(
            "session.events.since",
            json!({"session_id":sid,"last_seen":0,"replay_epoch":snapshot["replay_epoch"]}),
        )
        .await;
    assert_eq!(replay["truncated"], false);
    assert!(
        replay["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["type"] == "message.complete")
    );
    let mismatch = resumed
        .ok(
            "session.events.since",
            json!({"session_id":sid,"last_seen":100,"replay_epoch":"old-process"}),
        )
        .await;
    assert_eq!(mismatch["snapshot_required"], true);
    let snapshot = resumed
        .ok(
            "session.resume",
            json!({"session_id":sid,"omit_messages":true}),
        )
        .await;
    assert_eq!(snapshot["messages"], json!([]));
    assert_eq!(snapshot["message_count"], 4);
    assert_eq!(snapshot["running"], false);
    let mutation = json!({"session_id":sid,"request_id":"rename-once","operation":"rename","expected_revision":snapshot["revision"],"payload":{"title":"Renamed clock"}});
    let renamed = resumed.ok("session.mutate", mutation.clone()).await;
    assert_eq!(resumed.ok("session.mutate", mutation).await, renamed);
    assert_eq!(resumed.call("session.mutate",json!({"session_id":sid,"request_id":"rename-stale","operation":"rename","expected_revision":0,"payload":{"title":"Wrong"}})).await["error"]["data"]["reason"],"revision_conflict");
    resumed.socket.close(None).await.unwrap();
    gateway.shutdown().await.unwrap();
    // The new listener opens the exact native JSON transcript; no Python process is involved.
    let gateway = provider.gateway(&root).await;
    let mut restarted = Client::native(&gateway).await;
    let restored = restarted
        .ok("session.resume", json!({"session_id":sid}))
        .await;
    assert_eq!(restored["message_count"], 4);
    assert_eq!(restored["info"]["title"], "Renamed clock");
    assert_eq!(
        restarted.ok("session.create", create).await["session_id"],
        sid,
        "creation identity persists across restart"
    );
    assert_eq!(
        restarted
            .ok("session.list", json!({"scope":"stored"}))
            .await["sessions"][0]["id"],
        sid
    );
    restarted.socket.close(None).await.unwrap();
    gateway.shutdown().await.unwrap();
}

#[tokio::test]
async fn legacy_stream_offsets_busy_and_native_interrupt_are_real_transport_operations() {
    let provider = Provider::start().await;
    let root = TempDir::new().unwrap();
    let gateway = provider.gateway(&root).await;
    let mut legacy = Client::legacy(&gateway).await;
    let sid = legacy
        .ok("session.create", json!({"title":"Streaming"}))
        .await["session_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(
        legacy
            .ok("prompt.submit", json!({"session_id":sid,"text":"hello"}))
            .await["status"],
        "streaming"
    );
    let delta = legacy.event("message.delta").await;
    assert_eq!(delta["payload"]["offset"], 0);
    let delta = legacy.event("message.delta").await;
    assert_eq!(
        delta["payload"]["offset"], 9,
        "offsets are UTF-16 code units"
    );
    legacy.event("message.complete").await;
    let mut native = Client::native(&gateway).await;
    native.ok("session.resume", json!({"session_id":sid})).await;
    native
        .ok("prompt.submit", json!({"session_id":sid,"text":"slow"}))
        .await;
    native.event("message.delta").await;
    assert_eq!(
        native
            .call("prompt.submit", json!({"session_id":sid,"text":"another"}))
            .await["error"]["data"]["reason"],
        "session_busy"
    );
    assert_eq!(
        native
            .call(
                "prompt.submit",
                json!({"session_id":sid,"text":"another","queued":true})
            )
            .await["error"]["data"]["reason"],
        "unsupported_control"
    );
    assert_eq!(
        native
            .call(
                "session.interrupt",
                json!({"session_id":sid,"execution_generation":1})
            )
            .await["error"]["data"]["reason"],
        "stale_generation"
    );
    let interrupted = native
        .ok(
            "session.interrupt",
            json!({"session_id":sid,"execution_generation":2}),
        )
        .await;
    assert_eq!(interrupted["execution_generation"], 2);
    let complete = native.event("message.complete").await;
    assert_eq!(complete["payload"]["status"], "interrupted");
    assert_eq!(complete["payload"]["text"], "Waiting");
    let snapshot = native.ok("session.resume", json!({"session_id":sid})).await;
    assert_eq!(snapshot["running"], false);
    native.socket.close(None).await.unwrap();
    legacy.socket.close(None).await.unwrap();
    gateway.shutdown().await.unwrap();
    let summaries = hermes_library::Library::open(root.path())
        .unwrap()
        .sessions(None)
        .unwrap();
    let history: Value = serde_json::from_reader(
        std::fs::File::open(summaries[0].directory.join("conversation.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(history["turns"][1]["status"]["state"], "interrupted");
}

#[tokio::test]
async fn denial_provider_errors_and_shutdown_persist_their_real_outcomes() {
    let provider = Provider::start().await;
    let root = TempDir::new().unwrap();
    let gateway = provider.gateway(&root).await;
    let mut client = Client::native(&gateway).await;
    client.socket.send(Message::Text("{".into())).await.unwrap();
    assert_eq!(client.frame().await["error"]["code"], -32700);
    let sid = client.ok("session.create", json!({"title":"Denial"})).await["session_id"]
        .as_str()
        .unwrap()
        .to_owned();
    client
        .ok(
            "prompt.submit",
            json!({"session_id":sid,"text":"Read the clock"}),
        )
        .await;
    let approval = client.event("approval.request").await;
    let response = json!({"session_id":sid,"prompt_id":approval["payload"]["prompt_id"],"execution_generation":approval["execution_generation"],"choice":"deny"});
    let mut standing = response.clone();
    standing["choice"] = json!("session");
    assert_eq!(
        client.call("approval.respond", standing).await["error"]["code"],
        -32602
    );
    client.ok("approval.respond", response).await;
    let complete = client.event("message.complete").await;
    assert_eq!(complete["payload"]["text"], "Tool denied.");
    let history = client
        .ok("session.history", json!({"session_id":sid}))
        .await;
    assert_eq!(history["messages"][2]["approved"], false);
    assert!(
        history["messages"][2]["content"]
            .as_str()
            .unwrap()
            .contains("denied")
    );
    client
        .ok(
            "prompt.submit",
            json!({"session_id":sid,"text":"provider-error"}),
        )
        .await;
    let failed = client.event("message.complete").await;
    assert_eq!(failed["payload"]["status"], "error");
    assert!(
        failed["payload"]["error"]
            .as_str()
            .unwrap()
            .contains("fixture rejected")
    );
    client
        .ok("prompt.submit", json!({"session_id":sid,"text":"slow"}))
        .await;
    client.event("message.delta").await;
    gateway.shutdown().await.unwrap();
    let library = hermes_library::Library::open(root.path()).unwrap();
    let history: Value = serde_json::from_reader(
        std::fs::File::open(library.session_dir(&sid).unwrap().join("conversation.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(history["turns"][1]["status"]["state"], "failed");
    assert_eq!(history["turns"][2]["status"]["state"], "interrupted");
}

#[tokio::test]
async fn ported_rust_client_replays_native_completion_after_a_real_disconnect() {
    use hermes_gateway::client::{
        ClientError, ClientMessage, ClientOptions, ConnectionState, GatewayClient,
    };
    let provider = Provider::start().await;
    let root = TempDir::new().unwrap();
    let gateway = provider.gateway(&root).await;
    let client = GatewayClient::new(ClientOptions {
        heartbeat_interval: Duration::ZERO,
        heartbeat_deadline: Duration::ZERO,
        ..ClientOptions::default()
    });
    let mut messages = client.subscribe();
    client
        .connect_native(&gateway.http_url(), gateway.token())
        .await
        .unwrap();
    assert_eq!(client.connection_state(), ConnectionState::Open);
    let sid = client
        .request("session.create", json!({"title":"Ported client"}))
        .await
        .unwrap()["session_id"]
        .as_str()
        .unwrap()
        .to_owned();
    client
        .request(
            "prompt.submit",
            json!({"session_id":sid,"submission_id":"rust-client-turn","text":"Read the clock"}),
        )
        .await
        .unwrap();
    let approval = loop {
        if let ClientMessage::Event(event) = timeout(Duration::from_secs(5), messages.recv())
            .await
            .unwrap()
            .unwrap()
            && event["type"] == "approval.request"
        {
            break event;
        }
    };
    let watermark = client.watermarks()[&sid].as_u64().unwrap();
    assert!(watermark > 0);
    client.close();
    assert!(
        !client.session_replay_barrier(&sid).unwrap().wait().await,
        "disconnected history reads must be discarded"
    );
    // Another attached native viewer completes the actual engine approval while
    // the ported client has no socket. Its old cursor now has a real replay gap.
    let mut other = Client::native(&gateway).await;
    other.ok("session.resume", json!({"session_id":sid})).await;
    other.ok("approval.respond",json!({"session_id":sid,"prompt_id":approval["payload"]["prompt_id"],"execution_generation":approval["execution_generation"],"choice":"once"})).await;
    other.event("message.complete").await;
    client
        .connect_native(&gateway.http_url(), gateway.token())
        .await
        .unwrap();
    if let Some(barrier) = client.session_replay_barrier(&sid) {
        assert!(
            timeout(Duration::from_secs(5), barrier.wait())
                .await
                .unwrap()
        );
    }
    let mut completions = vec![];
    while let Ok(message) = messages.try_recv() {
        if let ClientMessage::Event(event) = message
            && event["type"] == "message.complete"
        {
            completions.push(event);
        }
    }
    assert_eq!(
        completions.len(),
        1,
        "replayed completion must dispatch exactly once"
    );
    assert_eq!(completions[0]["replayed"], true);
    assert_eq!(
        completions[0]["payload"]["text"],
        "Clock read successfully."
    );
    assert!(client.watermarks()[&sid].as_u64().unwrap() > watermark);
    let (ping, history) = tokio::join!(
        client.request("ping", json!({})),
        client.request("session.history", json!({"session_id":sid}))
    );
    assert_eq!(ping.unwrap()["pong"], true);
    assert_eq!(
        history.unwrap()["count"],
        4,
        "RPCs multiplex on the same socket"
    );
    let cancel = hermes_engine::CancellationToken::new();
    cancel.cancel();
    assert!(matches!(
        client
            .request_with_cancel("ping", json!({}), Duration::from_secs(1), Some(&cancel))
            .await,
        Err(ClientError::Cancelled)
    ));
    client.close();
    other.socket.close(None).await.unwrap();
    gateway.shutdown().await.unwrap();
}

/// Settings › Gateways › Remote: the desktop's remote path (connect_remote,
/// session.create, prompt.submit, approval.respond, message.complete) against
/// a real gateway on this computer.
#[tokio::test]
async fn remote_connect_runs_a_whole_turn_with_approval() {
    use hermes_gateway::client::{ClientMessage, ClientOptions, GatewayClient};
    let provider = Provider::start().await;
    let root = TempDir::new().unwrap();
    let gateway = provider.gateway(&root).await;
    let client = GatewayClient::new(ClientOptions { heartbeat_interval: Duration::ZERO, heartbeat_deadline: Duration::ZERO, ..ClientOptions::default() });
    let mut messages = client.subscribe();
    // A non-HTTPS remote host is refused before any token is sent.
    assert!(GatewayClient::new(ClientOptions::default()).connect_remote("http://example.com", gateway.token()).await.is_err());
    client.connect_remote(&gateway.http_url(), gateway.token()).await.unwrap();
    let sid = client.request("session.create", json!({"source":"gui","title":"Remote"})).await.unwrap()["session_id"].as_str().unwrap().to_owned();
    client.request("prompt.submit", json!({"session_id":sid,"text":"Read the clock"})).await.unwrap();
    let mut streamed = String::new();
    loop {
        let Ok(Ok(ClientMessage::Event(event))) = timeout(Duration::from_secs(10), messages.recv()).await else { continue };
        match event["type"].as_str().unwrap_or_default() {
            "message.delta" => streamed.push_str(event["payload"]["text"].as_str().unwrap_or_default()),
            "approval.request" => { client.request("approval.respond", json!({"session_id":sid,"prompt_id":event["payload"]["prompt_id"],
                "execution_generation":event["execution_generation"],"choice":"once"})).await.unwrap(); }
            "message.complete" => break,
            _ => {}
        }
    }
    assert!(!streamed.is_empty(), "the reply streamed back over the remote connection");
    client.close();
}

#[tokio::test]
async fn signed_webhook_runs_the_agent_into_its_own_chat() {
    let provider = Provider::start().await;
    let root = TempDir::new().unwrap();
    let gateway = Gateway::start(GatewayConfig::new(root.path(), provider.config())).await.unwrap();
    let library = hermes_library::Library::open(root.path()).unwrap();
    let hook = library.create_webhook(hermes_library::webhooks::Webhook { name: "deploys".into(), prompt: "Say hello".into(), events: vec!["push".into()], ..Default::default() }).unwrap();
    let body = br#"{"ref":"refs/heads/main"}"#.to_vec();
    let tag = ring::hmac::sign(&ring::hmac::Key::new(ring::hmac::HMAC_SHA256, hook.secret.as_bytes()), &body);
    let signature = format!("sha256={}", tag.as_ref().iter().map(|b| format!("{b:02x}")).collect::<String>());
    let http = reqwest::Client::new();
    let url = format!("{}/webhooks/deploys", gateway.http_url());
    let post = |sig: String, event: &'static str| http.post(&url).header("x-hub-signature-256", sig).header("x-github-event", event).body(body.clone()).send();
    assert_eq!(post("sha256=00".into(), "push").await.unwrap().status(), 401, "a bad signature is refused");
    assert_eq!(post(signature.clone(), "issues").await.unwrap().status(), 200, "a filtered event is ignored");
    assert_eq!(post(signature, "push").await.unwrap().status(), 202);
    let hook = timeout(Duration::from_secs(20), async {
        loop {
            let hook = library.webhooks().unwrap().remove(0);
            if hook.deliveries.first().is_some_and(|d| d.status.starts_with("ran") || d.status.starts_with("failed")) { break hook; }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }).await.unwrap();
    assert_eq!(hook.deliveries[0].status, "ran (complete)", "{:?}", hook.deliveries);
    assert_eq!(hook.deliveries[1].status, "ignored (event filter)");
    let chat = library.session(hook.session_id.as_deref().unwrap()).unwrap();
    assert_eq!(chat.title, "Webhook · deploys");
    let history = hermes_engine::Session::open(&chat.directory).unwrap().history().clone();
    let turn = &history.turns[0];
    assert!(turn.prompt.contains("Say hello") && turn.prompt.contains("refs/heads/main"));
    drop(gateway);
}
