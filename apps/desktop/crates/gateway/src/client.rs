//! Native asynchronous port of the shared JSON-RPC gateway transport.
//!
//! Replay semantics live in [`crate::replay`] and are checked against fixtures
//! executed by the original TypeScript client. This module supplies real Rust
//! WebSockets, concurrent request IDs/deadlines, request cancellation, heartbeat,
//! open-request redelivery and per-session history barriers. No UI is invented.

use crate::replay::{BarrierState, REPLAY_REQUEST_TIMEOUT_MS, ReplayAction, ReplayMachine};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fmt,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, mpsc, oneshot, watch},
    time::timeout,
};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{Message, client::IntoClientRequest, http::HeaderValue},
};
use tokio_util::sync::CancellationToken;

pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
pub const APPROVAL_RESPOND_TIMEOUT: Duration = Duration::from_secs(300);
pub const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
pub const DEFAULT_HEARTBEAT_DEADLINE: Duration = Duration::from_secs(45);

type PendingReply = oneshot::Sender<Result<Value, ClientError>>;
type BarrierKey = (u64, String);
type BarrierSender = watch::Sender<Option<bool>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionState {
    Idle,
    Connecting,
    Open,
    Closed,
    Error,
}
#[derive(Debug, Clone)]
pub enum ClientMessage {
    Event(Value),
    ServerRequest {
        id: String,
        method: String,
        params: Value,
        replayed: bool,
        response: ServerResponse,
    },
    State(ConnectionState),
}
/// First response wins, including across clones of the same delivered card.
/// Like the original channel, answers use this client's current transport.
#[derive(Clone)]
pub struct ServerResponse {
    inner: Weak<Inner>,
    id: String,
    settled: Arc<AtomicBool>,
}
impl fmt::Debug for ServerResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ServerResponse")
    }
}
impl ServerResponse {
    pub fn respond(&self, result: Value) -> Result<(), ClientError> {
        self.send(json!({"result":result}))
    }
    pub fn fail(&self, code: i64, message: &str) -> Result<(), ClientError> {
        self.send(json!({"error":{"code":code,"message":message}}))
    }
    fn send(&self, mut frame: Value) -> Result<(), ClientError> {
        if self.settled.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let inner = self.inner.upgrade().ok_or(ClientError::NotConnected)?;
        let tx = lock(&inner.connection)
            .as_ref()
            .map(|connection| connection.tx.clone())
            .ok_or(ClientError::NotConnected)?;
        frame["jsonrpc"] = json!("2.0");
        frame["id"] = json!(self.id);
        tx.try_send(frame)
            .map_err(|_| ClientError::Transport("server response could not be queued"))
    }
}

#[derive(Debug, Clone)]
pub enum ClientError {
    NotConnected,
    Cancelled,
    Timeout,
    Transport(&'static str),
    Rpc {
        code: Option<i64>,
        message: String,
        data: Value,
    },
}
impl ClientError {
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Rpc { data, .. } => data["reason"].as_str(),
            _ => None,
        }
    }
}
impl fmt::Display for ClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotConnected => f.write_str("gateway not connected"),
            Self::Cancelled => f.write_str("request cancelled"),
            Self::Timeout => f.write_str("gateway request timed out"),
            Self::Transport(message) => f.write_str(message),
            Self::Rpc { message, .. } => f.write_str(message),
        }
    }
}
impl std::error::Error for ClientError {}

#[derive(Clone)]
pub struct ClientOptions {
    pub replay: bool,
    /// Set only when the subscriber handles and answers ServerRequest messages.
    pub server_requests: bool,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub heartbeat_interval: Duration,
    pub heartbeat_deadline: Duration,
}
impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            replay: true,
            server_requests: false,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            heartbeat_deadline: DEFAULT_HEARTBEAT_DEADLINE,
        }
    }
}

/// A history read must wait until replay and its parked live frames finish.
/// `false` means the owning socket died, so that history read must be discarded.
pub struct ReplayBarrier {
    receiver: watch::Receiver<Option<bool>>,
}
impl ReplayBarrier {
    pub async fn wait(mut self) -> bool {
        loop {
            if let Some(valid) = *self.receiver.borrow_and_update() {
                return valid;
            }
            if self.receiver.changed().await.is_err() {
                return false;
            }
        }
    }
}

/// One multiplexed connection owner. Explicitly reconnect with a fresh ticket;
/// like upstream, the outer owner decides retry timing after invalidation.
pub struct GatewayClient {
    inner: Arc<Inner>,
}
struct Connection {
    generation: u64,
    tx: mpsc::Sender<Value>,
    stop: CancellationToken,
}
struct Inner {
    options: ClientOptions,
    connection: Mutex<Option<Connection>>,
    connect_guard: tokio::sync::Mutex<()>,
    processing: Mutex<()>,
    socket_generation: AtomicU64,
    next_id: AtomicU64,
    pending: Mutex<HashMap<String, PendingReply>>,
    replay: Mutex<ReplayMachine>,
    barriers: Mutex<HashMap<BarrierKey, BarrierSender>>,
    events: broadcast::Sender<ClientMessage>,
    state: watch::Sender<ConnectionState>,
}
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|p| p.into_inner())
}
impl Default for GatewayClient {
    fn default() -> Self {
        Self::new(ClientOptions::default())
    }
}
impl GatewayClient {
    pub fn new(options: ClientOptions) -> Self {
        let (events, _) = broadcast::channel(2048);
        let (state, _) = watch::channel(ConnectionState::Idle);
        Self {
            inner: Arc::new(Inner {
                replay: Mutex::new(ReplayMachine::new(options.replay)),
                options,
                connection: Mutex::new(None),
                connect_guard: tokio::sync::Mutex::new(()),
                processing: Mutex::new(()),
                socket_generation: AtomicU64::new(0),
                next_id: AtomicU64::new(0),
                pending: Mutex::new(HashMap::new()),
                barriers: Mutex::new(HashMap::new()),
                events,
                state,
            }),
        }
    }
    pub fn subscribe(&self) -> broadcast::Receiver<ClientMessage> {
        self.inner.events.subscribe()
    }
    pub fn connection_state(&self) -> ConnectionState {
        self.inner.state.borrow().clone()
    }
    pub fn watch_state(&self) -> watch::Receiver<ConnectionState> {
        self.inner.state.subscribe()
    }
    pub fn watermarks(&self) -> Value {
        lock(&self.inner.replay).watermarks()
    }
    pub fn session_replay_barrier(&self, sid: &str) -> Option<ReplayBarrier> {
        let machine = lock(&self.inner.replay);
        match machine.barrier(sid) {
            BarrierState::Absent => None,
            BarrierState::Invalid => {
                let (_, receiver) = watch::channel(Some(false));
                Some(ReplayBarrier { receiver })
            }
            BarrierState::Pending(generation) => {
                let mut barriers = lock(&self.inner.barriers);
                let sender = barriers
                    .entry((generation, sid.into()))
                    .or_insert_with(|| watch::channel(None).0);
                Some(ReplayBarrier {
                    receiver: sender.subscribe(),
                })
            }
        }
    }
    /// A public/legacy ws:// or wss:// URL. Errors never include a credential URL.
    pub async fn connect(&self, url: &str) -> Result<(), ClientError> {
        let parsed = reqwest::Url::parse(url).map_err(|_| {
            ClientError::Transport("gateway connect requires a ws:// or wss:// URL")
        })?;
        if !matches!(parsed.scheme(), "ws" | "wss") {
            return Err(ClientError::Transport(
                "gateway connect requires a ws:// or wss:// URL",
            ));
        }
        let request = url
            .into_client_request()
            .map_err(|_| ClientError::Transport("invalid gateway WebSocket request"))?;
        self.open(request).await
    }
    /// Mint the actual upstream 30-second ticket and remove it from the URL;
    /// it travels only in the native WebSocket subprotocol. Loopback only.
    pub async fn connect_native(&self, origin: &str, token: &str) -> Result<(), ClientError> {
        self.connect_ticketed(origin, token, false).await
    }
    /// A remote Hermes gateway (Settings › Gateways › Remote): the same one-use
    /// ticket handshake over HTTPS to any host, with an optional path prefix
    /// such as `/hermes`. Plain HTTP stays loopback-only so a session token
    /// never crosses a network unencrypted.
    pub async fn connect_remote(&self, origin: &str, token: &str) -> Result<(), ClientError> {
        self.connect_ticketed(origin, token, true).await
    }
    async fn connect_ticketed(&self, origin: &str, token: &str, remote: bool) -> Result<(), ClientError> {
        if lock(&self.inner.connection).is_some() {
            return Ok(());
        }
        let mut base = reqwest::Url::parse(origin)
            .map_err(|_| ClientError::Transport("invalid gateway origin"))?;
        let loopback = matches!(base.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
        let scheme_ok = if remote { base.scheme() == "https" || (base.scheme() == "http" && loopback) } else { base.scheme() == "http" && loopback };
        if !scheme_ok
            || !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
            || (!remote && !matches!(base.path(), "" | "/"))
        {
            return Err(ClientError::Transport(if remote {
                "remote gateway URL must be HTTPS (or HTTP on this computer)"
            } else {
                "native gateway origin must be an HTTP loopback origin"
            }));
        }
        let prefix = base.path().trim_end_matches('/').to_owned();
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(self.inner.options.connect_timeout)
            .build()
            .map_err(|_| ClientError::Transport("could not create native ticket client"))?;
        let response = http
            .post(
                base.join(&format!("{prefix}/api/auth/ws-ticket"))
                    .map_err(|_| ClientError::Transport("invalid ticket endpoint"))?,
            )
            .bearer_auth(token)
            .send()
            .await
            .map_err(|_| ClientError::Transport("native gateway ticket request failed"))?;
        if !response.status().is_success() {
            return Err(ClientError::Transport("native gateway ticket was refused"));
        }
        let ticket: Value = response
            .json()
            .await
            .map_err(|_| ClientError::Transport("native gateway returned an invalid ticket"))?;
        let ticket = ticket["ticket"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or(ClientError::Transport("native gateway returned no ticket"))?;
        base.set_scheme(if base.scheme() == "https" { "wss" } else { "ws" })
            .map_err(|_| ClientError::Transport("invalid native WebSocket origin"))?;
        base.set_path(&format!("{prefix}/api/ws"));
        base.set_query(Some("native_dial=1"));
        let mut request = base
            .as_str()
            .into_client_request()
            .map_err(|_| ClientError::Transport("invalid native WebSocket request"))?;
        request.headers_mut().insert(
            "sec-websocket-protocol",
            HeaderValue::from_str(&format!(
                "hermes-gateway-v1, hermes-gateway-ticket.{ticket}"
            ))
            .map_err(|_| ClientError::Transport("invalid native ticket"))?,
        );
        self.open(request).await
    }
    async fn open(
        &self,
        request: tokio_tungstenite::tungstenite::http::Request<()>,
    ) -> Result<(), ClientError> {
        let _guard = self.inner.connect_guard.lock().await;
        if lock(&self.inner.connection).is_some() {
            return Ok(());
        }
        let generation = self.inner.socket_generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.inner.set_state(ConnectionState::Connecting);
        let opening = timeout(self.inner.options.connect_timeout, connect_async(request)).await;
        let (socket, _) = match opening {
            Ok(Ok(socket)) => socket,
            Ok(Err(_)) => {
                self.inner.set_state(ConnectionState::Error);
                return Err(ClientError::Transport("WebSocket error before open"));
            }
            Err(_) => {
                self.inner.set_state(ConnectionState::Error);
                return Err(ClientError::Transport("WebSocket open timed out"));
            }
        };
        if generation != self.inner.socket_generation.load(Ordering::Acquire) {
            return Err(ClientError::Cancelled);
        }
        let (tx, mut rx) = mpsc::channel::<Value>(256);
        let stop = CancellationToken::new();
        let _processing = lock(&self.inner.processing);
        *lock(&self.inner.connection) = Some(Connection {
            generation,
            tx,
            stop: stop.clone(),
        });
        let actions = {
            let mut replay = lock(&self.inner.replay);
            let actions = replay.opened();
            self.inner.install_barriers(&replay, &actions);
            actions
        };
        self.inner.apply(actions);
        self.inner.set_state(ConnectionState::Open);
        drop(_processing);
        let inner = self.inner.clone();
        tokio::spawn(async move {
            let (mut sink, mut source) = socket.split();
            let mut last_inbound = Instant::now();
            let mut last_tick = Instant::now();
            let mut ping = 0u64;
            let mut heartbeat = false;
            let interval = inner.options.heartbeat_interval;
            let mut ticks = tokio::time::interval(if interval.is_zero() {
                Duration::from_secs(86400)
            } else {
                interval
            });
            ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                let outbound = tokio::select! {
                    _=stop.cancelled()=>break,
                    value=rx.recv()=>value,
                    frame=source.next()=>{
                        match frame {
                            Some(Ok(Message::Text(text)))=>{
                                last_inbound=Instant::now();
                                if let Ok(frame)=serde_json::from_str::<Value>(&text) {
                                    if frame["method"]=="event" && frame["params"]["type"]=="gateway.ready" && frame["params"]["payload"]["heartbeat"]==true {heartbeat=true;}
                                    inner.frame(generation,frame);
                                }
                            }
                            Some(Ok(_))=>{last_inbound=Instant::now();}
                            Some(Err(_))|None=>break,
                        }
                        continue;
                    }
                    _=ticks.tick()=>{
                        let now=Instant::now();let late=!interval.is_zero()&&now.duration_since(last_tick).as_secs_f64()>=interval.as_secs_f64()*1.5;last_tick=now;
                        if !heartbeat||interval.is_zero()||inner.options.heartbeat_deadline.is_zero() {continue;}
                        // As upstream: any inbound frame is life; suspend/throttle
                        // gets one fresh probe rather than an immediate disconnect.
                        if late {last_inbound=now;}
                        if now.duration_since(last_inbound)>=inner.options.heartbeat_deadline {break;}
                        ping+=1;Some(json!({"jsonrpc":"2.0","id":format!("heartbeat-{ping}"),"method":"gateway.ping","params":{}}))
                    }
                };
                let Some(frame) = outbound else {
                    break;
                };
                if !matches!(
                    timeout(
                        Duration::from_secs(10),
                        sink.send(Message::Text(frame.to_string().into()))
                    )
                    .await,
                    Ok(Ok(()))
                ) {
                    break;
                }
            }
            let _ = sink.close().await;
            inner.disconnect_generation(generation);
        });
        Ok(())
    }
    pub async fn request(&self, method: &str, params: Value) -> Result<Value, ClientError> {
        self.request_with_cancel(method, params, self.inner.options.request_timeout, None)
            .await
    }
    pub async fn request_with_cancel(
        &self,
        method: &str,
        params: Value,
        deadline: Duration,
        cancel: Option<&CancellationToken>,
    ) -> Result<Value, ClientError> {
        self.inner.request(method, params, deadline, cancel).await
    }
    pub fn close(&self) {
        self.inner.disconnect();
    }
}
impl Drop for GatewayClient {
    fn drop(&mut self) {
        self.inner.disconnect();
    }
}

impl Inner {
    fn set_state(&self, state: ConnectionState) {
        if *self.state.borrow() != state {
            self.state.send_replace(state.clone());
            let _ = self.events.send(ClientMessage::State(state));
        }
    }
    fn install_barriers(&self, _replay: &ReplayMachine, actions: &[ReplayAction]) {
        let mut barriers = lock(&self.barriers);
        for action in actions {
            if let ReplayAction::Request {
                session_id,
                generation,
                ..
            } = action
            {
                barriers
                    .entry((*generation, session_id.clone()))
                    .or_insert_with(|| watch::channel(None).0);
            }
        }
    }
    fn apply(self: &Arc<Self>, actions: Vec<ReplayAction>) {
        let mut barriers = Vec::new();
        for action in actions {
            match action {
                ReplayAction::Dispatch { event } => {
                    let _ = self.events.send(ClientMessage::Event(event));
                }
                ReplayAction::Barrier {
                    session_id,
                    generation,
                    valid,
                } => {
                    barriers.push((generation, session_id, valid));
                }
                ReplayAction::Request {
                    session_id,
                    params,
                    generation,
                } => {
                    let inner = self.clone();
                    tokio::spawn(async move {
                        let result = inner
                            .request(
                                "session.events.since",
                                params,
                                Duration::from_millis(REPLAY_REQUEST_TIMEOUT_MS),
                                None,
                            )
                            .await;
                        let _processing = lock(&inner.processing);
                        let actions = match result {
                            Ok(result) => {
                                lock(&inner.replay).completed(&session_id, generation, result)
                            }
                            Err(error) => {
                                lock(&inner.replay).failed(&session_id, generation, error.reason())
                            }
                        };
                        inner.apply(actions);
                    });
                }
            }
        }
        for (generation, session_id, valid) in barriers {
            if let Some(sender) = lock(&self.barriers).remove(&(generation, session_id)) {
                sender.send_replace(Some(valid));
            }
        }
    }
    fn disconnect_generation(self: &Arc<Self>, generation: u64) {
        let _processing = lock(&self.processing);
        let mut connection = lock(&self.connection);
        if connection
            .as_ref()
            .is_none_or(|connection| connection.generation != generation)
        {
            return;
        }
        if let Some(connection) = connection.take() {
            connection.stop.cancel();
        }
        self.finish_disconnect();
    }
    fn disconnect(self: &Arc<Self>) {
        let _processing = lock(&self.processing);
        let mut connection = lock(&self.connection);
        if let Some(connection) = connection.take() {
            connection.stop.cancel();
        }
        self.finish_disconnect();
    }
    fn finish_disconnect(self: &Arc<Self>) {
        self.socket_generation.fetch_add(1, Ordering::AcqRel);
        let actions = lock(&self.replay).closed();
        self.apply(actions);
        for (_, reply) in lock(&self.pending).drain() {
            let _ = reply.send(Err(ClientError::NotConnected));
        }
        self.set_state(ConnectionState::Closed);
    }
    async fn request(
        &self,
        method: &str,
        params: Value,
        deadline: Duration,
        cancel: Option<&CancellationToken>,
    ) -> Result<Value, ClientError> {
        if cancel.is_some_and(CancellationToken::is_cancelled) {
            return Err(ClientError::Cancelled);
        }
        let tx = lock(&self.connection)
            .as_ref()
            .map(|connection| connection.tx.clone())
            .ok_or(ClientError::NotConnected)?;
        let id = format!("r{}", self.next_id.fetch_add(1, Ordering::Relaxed) + 1);
        let (reply, result) = oneshot::channel();
        lock(&self.pending).insert(id.clone(), reply);
        let _registration = PendingRegistration {
            inner: self,
            id: id.clone(),
        };
        let response = async {
            tx.send(json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))
                .await
                .map_err(|_| ClientError::NotConnected)?;
            result.await.map_err(|_| ClientError::NotConnected)?
        };
        let never = CancellationToken::new();
        let cancellation = cancel.unwrap_or(&never);
        let result = tokio::select! {
            _=cancellation.cancelled()=>Err(ClientError::Cancelled),
            result=async {if deadline.is_zero(){response.await}else{timeout(deadline,response).await.map_err(|_|ClientError::Timeout)?}}=>result,
        };
        result
    }
    fn deliver_request(
        self: &Arc<Self>,
        id: &str,
        method: &str,
        params: Option<&Value>,
        replayed: bool,
    ) {
        let response = ServerResponse {
            inner: Arc::downgrade(self),
            id: id.into(),
            settled: Arc::new(AtomicBool::new(false)),
        };
        if !self.options.server_requests || self.events.receiver_count() == 0 {
            let _ = response.fail(-32601, "no handler for server request");
            return;
        }
        let _ = self.events.send(ClientMessage::ServerRequest {
            id: id.into(),
            method: method.into(),
            params: params
                .filter(|p| p.is_object())
                .cloned()
                .unwrap_or_else(|| json!({})),
            replayed,
            response,
        });
    }
    fn frame(self: &Arc<Self>, generation: u64, frame: Value) {
        let _processing = lock(&self.processing);
        if lock(&self.connection)
            .as_ref()
            .is_none_or(|connection| connection.generation != generation)
        {
            return;
        }
        if frame["method"] == "event" && frame["params"]["type"].is_string() {
            if frame["params"]["type"] == "gateway.ready" && self.options.server_requests {
                let inner = self.clone();
                tokio::spawn(async move {
                    let _ = inner
                        .request(
                            "client.capabilities",
                            json!({"server_requests":true}),
                            inner.options.request_timeout,
                            None,
                        )
                        .await;
                });
            }
            let actions = lock(&self.replay).live(frame["params"].clone());
            self.apply(actions);
            return;
        }
        if let (Some(id), Some(method)) = (frame["id"].as_str(), frame["method"].as_str()) {
            self.deliver_request(id, method, frame.get("params"), false);
            return;
        }
        let Some(id) = frame["id"].as_str() else {
            return;
        };
        let Some(reply) = lock(&self.pending).remove(id) else {
            return;
        };
        let result = if let Some(error) = frame.get("error") {
            Err(ClientError::Rpc {
                code: error["code"].as_i64(),
                message: error["message"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("Hermes RPC failed")
                    .into(),
                data: error.get("data").cloned().unwrap_or(Value::Null),
            })
        } else {
            let result = frame.get("result").cloned().unwrap_or(Value::Null);
            for request in result["open_requests"].as_array().into_iter().flatten() {
                if let (Some(id), Some(method)) =
                    (request["id"].as_str(), request["method"].as_str())
                {
                    self.deliver_request(id, method, request.get("params"), true);
                }
            }
            Ok(result)
        };
        let _ = reply.send(result);
    }
}

struct PendingRegistration<'a> {
    inner: &'a Inner,
    id: String,
}
impl Drop for PendingRegistration<'_> {
    fn drop(&mut self) {
        lock(&self.inner.pending).remove(&self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn synthetic_connection() -> (GatewayClient, mpsc::Receiver<Value>) {
        let client = GatewayClient::default();
        let (tx, rx) = mpsc::channel(16);
        *lock(&client.inner.connection) = Some(Connection {
            generation: 1,
            tx,
            stop: CancellationToken::new(),
        });
        client.inner.set_state(ConnectionState::Open);
        (client, rx)
    }
    #[tokio::test]
    async fn response_handles_send_once_and_have_no_secret_debug_fields() {
        let (client, mut outgoing) = synthetic_connection();
        let response = ServerResponse {
            inner: Arc::downgrade(&client.inner),
            id: "synthetic-request".into(),
            settled: Arc::new(AtomicBool::new(false)),
        };
        let duplicate = response.clone();
        response.respond(json!({"choice":"once"})).unwrap();
        duplicate.fail(-32603, "ignored second answer").unwrap();
        assert_eq!(
            outgoing.recv().await.unwrap(),
            json!({"jsonrpc":"2.0","id":"synthetic-request","result":{"choice":"once"}})
        );
        assert!(outgoing.try_recv().is_err());
        assert_eq!(format!("{response:?}"), "ServerResponse");
    }
    #[tokio::test]
    async fn deadline_cancel_and_dropped_future_retire_pending_requests() {
        let (client, mut outgoing) = synthetic_connection();
        let result = client
            .request_with_cancel("synthetic.never", json!({}), Duration::from_millis(1), None)
            .await;
        assert!(matches!(result, Err(ClientError::Timeout)));
        assert!(lock(&client.inner.pending).is_empty());
        let timed_out = outgoing.recv().await.unwrap();
        client.inner.frame(
            1,
            json!({"jsonrpc":"2.0","id":timed_out["id"],"result":{"late":true}}),
        );
        assert_eq!(client.watermarks(), json!({}));
        let cancel = CancellationToken::new();
        let mut future = Box::pin(client.request_with_cancel(
            "synthetic.cancel",
            json!({}),
            Duration::from_secs(30),
            Some(&cancel),
        ));
        tokio::select! {result=&mut future=>panic!("request settled before cancellation: {result:?}"),frame=outgoing.recv()=>assert_eq!(frame.unwrap()["method"],"synthetic.cancel")}
        cancel.cancel();
        assert!(matches!(future.await, Err(ClientError::Cancelled)));
        assert!(lock(&client.inner.pending).is_empty());
        let mut dropped = Box::pin(client.request_with_cancel(
            "synthetic.dropped",
            json!({}),
            Duration::from_secs(30),
            None,
        ));
        tokio::select! {result=&mut dropped=>panic!("request settled before drop: {result:?}"),frame=outgoing.recv()=>assert_eq!(frame.unwrap()["method"],"synthetic.dropped")}
        drop(dropped);
        assert!(lock(&client.inner.pending).is_empty());
    }
    #[tokio::test]
    async fn invalid_connection_errors_do_not_echo_credential_urls() {
        let client = GatewayClient::default();
        let error = client
            .connect("https://synthetic.invalid/?token=synthetic-secret")
            .await
            .unwrap_err();
        assert!(!error.to_string().contains("synthetic-secret"));
        assert!(
            client
                .connect_native("https://synthetic.invalid", "synthetic-secret")
                .await
                .is_err()
        );
    }
}
