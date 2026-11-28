//! Authenticated loopback WebSocket gateway for the native Rust engine.
//!
//! This is a documented subset of Hermes' legacy and `hermes-gateway-v1` wires,
//! not an implementation of the entire upstream dashboard or session authority.

pub mod client;
pub mod replay;

mod webhooks;
use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    extract::{
        ConnectInfo, Query, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures_util::{SinkExt, StreamExt};
use hermes_engine::{CancellationToken, Config, Entry, Event, History, Session, Status};
use hermes_library::{Library, SessionSummary};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fs::{self, File},
    io::Write,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use tempfile::NamedTempFile;
use tokio::{
    net::TcpListener,
    sync::{Mutex, mpsc},
    task::JoinHandle,
};
use uuid::Uuid;

const PROTOCOL: &str = "hermes-gateway-v1";
const TICKET_PREFIX: &str = "hermes-gateway-ticket.";
const TICKET_TTL: Duration = Duration::from_secs(30);
const MAX_SESSIONS: usize = 64;
const MAX_CONNECTIONS: usize = 32;
const MAX_REPLAY_EVENTS: usize = 512;
const MAX_REPLAY_BYTES: usize = 524_288;
const MAX_RECEIPTS: usize = 256;
const MAX_PROMPT_BYTES: usize = 262_144;

/// Methods implemented by this crate. Other methods return JSON-RPC -32601.
pub const METHODS: &[&str] = &[
    "ping",
    "gateway.ping",
    "rpc.discover",
    "runtime.describe",
    "gateway.capabilities",
    "client.capabilities",
    "session.create",
    "session.resume",
    "session.activate",
    "session.list",
    "session.active_list",
    "session.history",
    "session.info",
    "session.status",
    "session.events.since",
    "session.interrupt",
    "session.close",
    "session.title",
    "session.mutate",
    "prompt.submit",
    "prompt.receipt",
    "prompt.cancel",
    "approval.pending",
    "approval.received",
    "approval.respond",
];

/// Gateway settings. The provider key stays in memory and is never sent to clients.
pub struct GatewayConfig {
    pub bind: SocketAddr,
    pub library_root: PathBuf,
    /// Empty generates a random 256-bit token. Supply a saved token for native clients.
    pub token: String,
    pub provider: Config,
}

impl GatewayConfig {
    pub fn new(library_root: impl Into<PathBuf>, provider: Config) -> Self {
        Self {
            bind: ([127, 0, 0, 1], 0).into(),
            library_root: library_root.into(),
            token: String::new(),
            provider,
        }
    }
}

pub struct Gateway;

/// A running listener. Drop cancels its listener and active turns.
pub struct RunningGateway {
    address: SocketAddr,
    shared: Arc<Shared>,
    task: Option<JoinHandle<Result<()>>>,
}

impl RunningGateway {
    pub fn address(&self) -> SocketAddr {
        self.address
    }
    pub fn token(&self) -> &str {
        &self.shared.token
    }
    pub fn http_url(&self) -> String {
        format!("http://{}", self.address)
    }
    pub fn websocket_url(&self) -> String {
        format!("ws://{}/api/ws", self.address)
    }
    pub async fn shutdown(mut self) -> Result<()> {
        self.shared.shutdown.cancel();
        self.shared.stop_turns().await;
        if let Some(task) = self.task.take() {
            task.await.context("Gateway task stopped unexpectedly")??;
        }
        Ok(())
    }
}

impl Drop for RunningGateway {
    fn drop(&mut self) {
        self.shared.shutdown.cancel();
    }
}

impl Gateway {
    pub async fn start(mut config: GatewayConfig) -> Result<RunningGateway> {
        ensure!(
            config.bind.ip().is_loopback(),
            "The Rust gateway only binds to a loopback address"
        );
        if config.token.is_empty() {
            config.token = format!("{}{}", new_id(), new_id());
        }
        ensure!(
            config.token.len() >= 32
                && config.token.len() <= 512
                && config.token.bytes().all(|b| b.is_ascii_graphic()),
            "Gateway token must be 32–512 printable ASCII characters without spaces"
        );
        let library = Library::open(&config.library_root)?;
        let listener = TcpListener::bind(config.bind)
            .await
            .context("Could not bind gateway")?;
        let address = listener.local_addr()?;
        let shared = Arc::new(Shared {
            library,
            provider: config.provider,
            token: config.token,
            address,
            epoch: new_id(),
            instance_id: new_id(),
            shutdown: CancellationToken::new(),
            sessions: Mutex::new(HashMap::new()),
            tickets: StdMutex::new(HashMap::new()),
            creates: Mutex::new(HashMap::new()),
            turns: Mutex::new(Vec::new()),
            connections: AtomicUsize::new(0),
        });
        let router = Router::new()
            .route("/health", get(health))
            .route("/api/auth/ws-ticket", post(mint_ticket))
            .route("/api/ws", get(upgrade))
            .route("/webhooks/{name}", post(webhooks::receive))
            .with_state(shared.clone());
        let stop = shared.shutdown.clone();
        let on_stop = shared.clone();
        let task = tokio::spawn(async move {
            let result = axum::serve(
                listener,
                router.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .with_graceful_shutdown(stop.cancelled_owned())
            .await
            .context("Gateway listener failed");
            on_stop.stop_turns().await;
            result
        });
        Ok(RunningGateway {
            address,
            shared,
            task: Some(task),
        })
    }
}

fn new_id() -> String {
    Uuid::new_v4().simple().to_string()
}
fn lock<T>(mutex: &StdMutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|p| p.into_inner())
}
fn now() -> f64 {
    chrono::Utc::now().timestamp_millis() as f64 / 1000.0
}
fn epoch_time(value: &str) -> f64 {
    chrono::DateTime::parse_from_rfc3339(value)
        .map(|v| v.timestamp_millis() as f64 / 1000.0)
        .unwrap_or(0.0)
}
fn same_secret(left: &str, right: &str) -> bool {
    let mut difference = left.len() ^ right.len();
    for (a, b) in left.bytes().zip(right.bytes()) {
        difference |= (a ^ b) as usize;
    }
    difference == 0
}

struct Shared {
    library: Library,
    provider: Config,
    token: String,
    address: SocketAddr,
    epoch: String,
    instance_id: String,
    shutdown: CancellationToken,
    sessions: Mutex<HashMap<String, Arc<LiveSession>>>,
    tickets: StdMutex<HashMap<String, Instant>>,
    creates: Mutex<HashMap<String, (String, String)>>,
    turns: Mutex<Vec<JoinHandle<()>>>,
    connections: AtomicUsize,
}

#[derive(Clone)]
struct Peer {
    id: String,
    tx: mpsc::Sender<Value>,
    stop: CancellationToken,
    canonical: bool,
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct Settings {
    #[serde(default)]
    source: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    creation_key: Option<String>,
    #[serde(default)]
    creation_fingerprint: Option<String>,
    #[serde(default)]
    metadata_revision: u64,
    #[serde(default)]
    execution_generation: u64,
}

impl Settings {
    fn read(dir: &Path) -> Result<Self> {
        match File::open(dir.join("gateway.json")) {
            Ok(file) => serde_json::from_reader(file).context("Invalid gateway session settings"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }
    fn save(&self, dir: &Path) -> Result<()> {
        let mut file = NamedTempFile::new_in(dir)?;
        serde_json::to_writer_pretty(&mut file, self)?;
        file.write_all(b"\n")?;
        file.as_file().sync_all()?;
        file.persist(dir.join("gateway.json"))
            .context("Could not save gateway settings")?;
        Ok(())
    }
}

struct LiveSession {
    summary: StdMutex<SessionSummary>,
    dir: PathBuf,
    settings: StdMutex<Settings>,
    config: StdMutex<Config>,
    engine: Mutex<Session>,
    view: StdMutex<View>,
    peers: StdMutex<HashMap<String, Peer>>,
}

struct View {
    history: History,
    running: bool,
    generation: u64,
    revision: u64,
    started: Option<f64>,
    cancel: Option<CancellationToken>,
    current_admission: Option<String>,
    approvals: BTreeMap<String, Approval>,
    receipts: VecDeque<Receipt>,
    sequence: u64,
    replay: VecDeque<Value>,
    replay_bytes: usize,
    evicted_through: u64,
    mutations: VecDeque<(String, String, Value)>,
}

struct Approval {
    payload: Value,
    answer: async_channel::Sender<bool>,
    timeout: CancellationToken,
}

#[derive(Clone)]
struct Receipt {
    id: String,
    input_id: Option<String>,
    text: String,
    generation: u64,
    status: String,
    outcome: Option<String>,
    result: Option<Value>,
}

impl Receipt {
    fn value(&self, sid: &str, include_result: bool) -> Value {
        let mut result = json!({ "admission_id":self.id, "ref":{"profile_id":"default","session_id":sid},
            "sequence":self.generation, "status":self.status, "outcome":self.outcome,
            "authority_epoch":1, "execution_generation":self.generation });
        if include_result {
            result["result"] = self.result.clone().unwrap_or(Value::Null);
        }
        result
    }
}

#[derive(Debug)]
struct RpcError {
    code: i64,
    message: String,
    data: Value,
}
type RpcResult = std::result::Result<Value, RpcError>;
impl RpcError {
    fn new(code: i64, reason: &str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: json!({"reason":reason}),
        }
    }
    fn invalid(message: impl Into<String>) -> Self {
        Self::new(-32602, "invalid_params", message)
    }
    fn unsupported(message: impl Into<String>) -> Self {
        Self::new(4001, "unsupported_control", message)
    }
    fn internal(error: impl std::fmt::Display) -> Self {
        Self::new(-32603, "internal_error", error.to_string())
    }
    fn fenced(reason: &str, message: &str, view: &View) -> Self {
        Self {
            code: 4001,
            message: message.into(),
            data: json!({"reason":reason,"revision":view.revision,"execution_generation":view.generation}),
        }
    }
    fn value(self) -> Value {
        json!({"code":self.code,"message":self.message,"data":self.data})
    }
}

impl Shared {
    async fn stop_turns(&self) {
        for session in self.sessions.lock().await.values() {
            if let Some(cancel) = &lock(&session.view).cancel {
                cancel.cancel();
            }
            for peer in lock(&session.peers).values() {
                peer.stop.cancel();
            }
        }
        let tasks = std::mem::take(&mut *self.turns.lock().await);
        for task in tasks {
            let _ = task.await;
        }
    }

    async fn session(&self, id: &str) -> std::result::Result<Arc<LiveSession>, RpcError> {
        let mut sessions = self.sessions.lock().await;
        if let Some(session) = sessions.get(id) {
            return Ok(session.clone());
        }
        if sessions.len() >= MAX_SESSIONS {
            return Err(RpcError::new(
                4001,
                "session_limit",
                "Close an idle session before opening another (limit 64)",
            ));
        }
        let summary = self
            .library
            .sessions(None)
            .map_err(RpcError::internal)?
            .into_iter()
            .find(|s| s.id == id)
            .ok_or_else(|| {
                RpcError::new(4004, "not_found", "Session not found in the Rust library")
            })?;
        let dir = self.library.session_dir(id).map_err(RpcError::internal)?;
        let mut settings = Settings::read(&dir).map_err(RpcError::internal)?;
        let workspace = self
            .library
            .workspace_for_session(id)
            .map_err(RpcError::internal)?
            .or_else(|| settings.cwd.clone());
        let actual_workspace = workspace.clone().unwrap_or_else(|| dir.join("workspace"));
        let engine = Session::open_for_profile(&dir, workspace.as_deref(), self.library.root())
            .map_err(|e| {
                RpcError::new(
                    4001,
                    "session_busy",
                    format!("Could not open session: {e:#}"),
                )
            })?;
        settings.cwd = Some(actual_workspace);
        if settings.source.is_empty() {
            settings.source = "rust".into();
        }
        let history = engine.history().clone();
        let generation = (history.turns.len() as u64).max(settings.execution_generation);
        let mut config = self.provider.clone();
        if let Some(model) = &settings.model {
            config.model.clone_from(model);
        }
        let session = Arc::new(LiveSession {
            summary: StdMutex::new(summary),
            dir,
            settings: StdMutex::new(settings.clone()),
            config: StdMutex::new(config),
            engine: Mutex::new(engine),
            peers: StdMutex::new(HashMap::new()),
            view: StdMutex::new(View {
                history,
                running: false,
                generation,
                revision: generation * 2 + settings.metadata_revision,
                started: None,
                cancel: None,
                current_admission: None,
                approvals: BTreeMap::new(),
                receipts: VecDeque::new(),
                sequence: 0,
                replay: VecDeque::new(),
                replay_bytes: 0,
                evicted_through: 0,
                mutations: VecDeque::new(),
            }),
        });
        sessions.insert(id.into(), session.clone());
        Ok(session)
    }

    fn emit(&self, session: &LiveSession, kind: &str, payload: Value) {
        let id = lock(&session.summary).id.clone();
        let mut view = lock(&session.view);
        view.sequence += 1;
        let mut params = json!({"type":kind,"session_id":id,"payload":payload,
            "seq":view.sequence,"replay_epoch":self.epoch,"authority_epoch":1,"execution_generation":view.generation});
        if let Some(admission) = &view.current_admission {
            params["admission_id"] = json!(admission);
        }
        let size = params.to_string().len();
        if size <= MAX_REPLAY_BYTES {
            view.replay_bytes += size;
            view.replay.push_back(params.clone());
        } else {
            view.evicted_through = view.sequence;
        }
        while view.replay.len() > MAX_REPLAY_EVENTS || view.replay_bytes > MAX_REPLAY_BYTES {
            if let Some(event) = view.replay.pop_front() {
                view.replay_bytes -= event.to_string().len();
                view.evicted_through = view.evicted_through.max(event["seq"].as_u64().unwrap_or(0));
            }
        }
        let frame = json!({"jsonrpc":"2.0","method":"event","params":params});
        // Keep the view lock until queued, preserving sequence order across concurrent controls.
        lock(&session.peers).retain(|_, peer| {
            if peer.tx.try_send(frame.clone()).is_err() {
                peer.stop.cancel();
                false
            } else {
                true
            }
        });
    }

    fn info(&self, session: &LiveSession) -> Value {
        let summary = lock(&session.summary);
        let settings = lock(&session.settings);
        let config = lock(&session.config);
        let view = lock(&session.view);
        json!({ "desktop_protocol":PROTOCOL,"version":concat!(env!("CARGO_PKG_VERSION"),"-rust"),
            "model":config.model,"provider":provider_label(&config),"profile_id":"default","profile_name":"default",
            "title":summary.title,"source":settings.source,"stored_session_id":summary.id,
            "cwd":settings.cwd.as_ref().map(|p|p.to_string_lossy().to_string()).unwrap_or_default(),
            "running":view.running,"execution_generation":view.generation,"revision":view.revision,
            "turn_started_at":view.started,"started_at":epoch_time(&summary.created_at),
            "approval_mode":"manual","yolo":false,"tools":{"rust":tool_names()},"skills":{},"mcp_servers":[],
            "terminal_backend":"rust","replay_epoch":self.epoch,"last_sequence":view.sequence,
            "credential_warning": if config.api_key.is_empty() && config.base_url.contains("openrouter.ai") { Some("Set HERMES_RUST_API_KEY or OPENROUTER_API_KEY before starting the gateway") } else { None }
        })
    }

    fn snapshot(&self, session: &LiveSession, peer: &Peer, omit: bool) -> Value {
        let info = self.info(session);
        let id = lock(&session.summary).id.clone();
        let view = lock(&session.view);
        let messages = transcript(&view.history);
        let prompts: Vec<_> = view.approvals.values().map(|p| p.payload.clone()).collect();
        let mut result = json!({"session_id":id,"stored_session_id":id,"message_count":messages.len(),
            "messages":if omit {vec![]} else {messages},"info":info,"running":view.running,
            "messages_omitted":omit,"open_requests":[],"pending_approval":prompts.first(),
            "turn_started_at":view.started});
        if peer.canonical {
            result["authority_epoch"] = json!(1);
            result["replay_epoch"] = json!(self.epoch);
            result["last_sequence"] = json!(view.sequence);
            result["subscription_id"] = json!(format!("{}:{id}", peer.id));
            result["revision"] = json!(view.revision);
            result["execution_generation"] = json!(view.generation);
            result["pending"] = json!([]);
            result["prompts"] = json!(prompts);
        }
        result
    }

    async fn dispatch(self: &Arc<Self>, method: &str, params: &Value, peer: &Peer) -> RpcResult {
        params
            .as_object()
            .ok_or_else(|| RpcError::invalid("params must be an object"))?;
        if let Some(profile) = optional_string(params, "profile")?
            && profile != "default"
        {
            return Err(RpcError::new(
                4001,
                "unsupported_profile",
                "This gateway serves only the Rust library's default profile",
            ));
        }
        match method {
            "ping" => Ok(json!({"pong":true})),
            "gateway.ping" => Ok(json!({"ok":true})),
            "runtime.describe" => Ok(
                json!({"instance_id":self.instance_id,"profile_id":"default","authority_epoch":1,"capabilities":["rust-engine-subset-v1","event-replay-v1"],"session_create":{"sources":["cli","tui","gui","acp","tool","oneshot"],"parameters":["request_id","source","cwd","model","provider","title","cols"]},"skin":{}}),
            ),
            "rpc.discover" => Ok(
                json!({"implementation":"hermes-rust","version":env!("CARGO_PKG_VERSION"),"protocols":["legacy",PROTOCOL],"methods":METHODS,"limitations":["one active turn per session; busy submissions are refused","queue, steer, redirect, multimodal attachments, profile routing, dashboard REST and non-Rust tools are not implemented","receipts and replay are bounded and retained only until process restart"]}),
            ),
            "gateway.capabilities" => Ok(json!({"per_session_exclusive_submit":true})),
            "client.capabilities" => Ok(
                json!({"server_requests":[],"declines_not_shown":false,"canonical_clarify_batch":false}),
            ),
            "session.create" => self.create(params, peer).await,
            "session.list" | "session.active_list" => {
                self.list(params, peer, method == "session.active_list")
                    .await
            }
            "session.resume" | "session.activate" => {
                let session = self.resolve(params).await?;
                lock(&session.peers).insert(peer.id.clone(), peer.clone());
                Ok(self.snapshot(
                    &session,
                    peer,
                    params["omit_messages"].as_bool().unwrap_or(false),
                ))
            }
            "session.history" => {
                let session = self.resolve(params).await?;
                let messages = transcript(&lock(&session.view).history);
                Ok(json!({"count":messages.len(),"messages":messages}))
            }
            "session.info" => {
                let session = self.resolve(params).await?;
                Ok(self.info(&session))
            }
            "session.status" => {
                let session = self.resolve(params).await?;
                let view = lock(&session.view);
                Ok(json!({"output":if view.running {"Running"} else {"Idle"}}))
            }
            "session.events.since" => self.replay(params, peer).await,
            "prompt.submit" => self.submit(params, peer).await,
            "prompt.receipt" => {
                let session = self.attached(params, peer).await?;
                let id = required_string(params, "admission_id")?;
                let sid = lock(&session.summary).id.clone();
                let view = lock(&session.view);
                let receipt = view.receipts.iter().find(|r| r.id == id).ok_or_else(|| {
                    RpcError::new(
                        4004,
                        "not_found",
                        "Admission not retained; refetch session history",
                    )
                })?;
                Ok(receipt.value(&sid, params["include_result"].as_bool().unwrap_or(false)))
            }
            "session.interrupt" | "prompt.cancel" => {
                self.interrupt(params, peer, method == "prompt.cancel")
                    .await
            }
            "approval.pending" => {
                let session = self.attached(params, peer).await?;
                let approvals: Vec<_> = lock(&session.view)
                    .approvals
                    .values()
                    .map(|a| a.payload.clone())
                    .collect();
                Ok(json!({"approvals":approvals}))
            }
            "approval.received" => {
                let session = self.attached(params, peer).await?;
                let id = required_string(params, "request_id")?;
                Ok(json!({"acknowledged":lock(&session.view).approvals.contains_key(id)}))
            }
            "approval.respond" => self.respond(params, peer).await,
            "session.title" | "session.mutate" => {
                self.mutate(params, peer, method == "session.mutate").await
            }
            "session.close" => {
                let id = required_string(params, "session_id")?;
                let session = self.attached(params, peer).await?;
                if lock(&session.view).running {
                    return Err(RpcError::fenced(
                        "session_busy",
                        "Interrupt and wait for the turn to finish before closing",
                        &lock(&session.view),
                    ));
                }
                let mut sessions = self.sessions.lock().await;
                lock(&session.peers).remove(&peer.id);
                if lock(&session.peers).is_empty() {
                    sessions.remove(id);
                }
                Ok(json!({"closed":true}))
            }
            _ => Err(RpcError::new(
                -32601,
                "method_not_found",
                format!("Method not implemented by the Rust gateway: {method}"),
            )),
        }
    }

    async fn resolve(&self, params: &Value) -> std::result::Result<Arc<LiveSession>, RpcError> {
        let id = required_string(params, "session_id")?;
        self.session(id).await
    }
    async fn attached(
        &self,
        params: &Value,
        peer: &Peer,
    ) -> std::result::Result<Arc<LiveSession>, RpcError> {
        let session = self.resolve(params).await?;
        if peer.canonical && !lock(&session.peers).contains_key(&peer.id) {
            return Err(RpcError::new(
                4001,
                "not_attached",
                "Resume the session on this socket before controlling it",
            ));
        }
        Ok(session)
    }

    async fn create(self: &Arc<Self>, params: &Value, peer: &Peer) -> RpcResult {
        reject_options(
            params,
            &[
                "messages",
                "parent_session_id",
                "reasoning_effort",
                "reasoning",
                "toolsets",
                "skills",
                "base_url",
                "api_key",
                "max_turns",
                "client_env",
                "editor",
                "run_budget",
                "row_cwd",
            ],
        )?;
        for name in [
            "hidden",
            "room_plumbing",
            "follow_profile_config",
            "yolo",
            "safe_mode",
            "ignore_rules",
            "ignore_user_config",
            "checkpoints",
            "accept_hooks",
            "pass_session_id",
            "close_on_disconnect",
        ] {
            if params[name].as_bool() == Some(true) {
                return Err(RpcError::unsupported(format!(
                    "session.create option {name} is not supported"
                )));
            }
        }
        if params["fast"].as_bool() == Some(true)
            || optional_string(params, "service_tier")?.is_some()
        {
            return Err(RpcError::unsupported(
                "Service tier selection is not implemented by this provider runtime",
            ));
        }
        if let Some(provider) = optional_string(params, "provider")?
            && !provider.is_empty()
            && provider != provider_label(&self.provider)
        {
            return Err(RpcError::unsupported(
                "Only the gateway's configured provider is available; select its model instead",
            ));
        }
        let title = optional_string(params, "title")?
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("New conversation");
        if title.chars().count() > 200 {
            return Err(RpcError::invalid("title is limited to 200 characters"));
        }
        let model = optional_string(params, "model")?.map(str::to_owned);
        if matches!(&model,Some(s) if s.trim().is_empty() || s.len()>256) {
            return Err(RpcError::invalid("model must contain 1–256 bytes"));
        }
        let cwd = optional_string(params, "cwd")?
            .filter(|s| !s.is_empty())
            .map(|s| {
                let path = fs::canonicalize(s)
                    .map_err(|e| RpcError::invalid(format!("cwd cannot be opened: {e}")))?;
                if !path.is_dir() {
                    return Err(RpcError::invalid("cwd must be an existing directory"));
                }
                Ok(path)
            })
            .transpose()?;
        let source = optional_string(params, "source")?.unwrap_or(if peer.canonical {
            "gui"
        } else {
            "desktop"
        });
        if source.len() > 64 {
            return Err(RpcError::invalid("source is limited to 64 bytes"));
        }
        let key = optional_string(params, "request_id")?
            .or(optional_string(params, "idempotency_key")?)
            .map(str::to_owned);
        if matches!(&key,Some(k) if k.is_empty() || k.len()>256) {
            return Err(RpcError::invalid(
                "request identity must contain 1–256 bytes",
            ));
        }
        let fingerprint =
            json!({"title":title,"model":model,"cwd":cwd,"source":source}).to_string();
        // One admission lock covers durable lookup and creation, including reconnect retries.
        let mut creates = self.creates.lock().await;
        let mut existing = None;
        if let Some(key) = &key {
            if let Some((old_fingerprint, id)) = creates.get(key) {
                if old_fingerprint != &fingerprint {
                    return Err(RpcError::new(
                        4001,
                        "request_conflict",
                        "Creation request identity was reused with different options",
                    ));
                }
                existing = Some(id.clone());
            } else {
                for summary in self.library.sessions(None).map_err(RpcError::internal)? {
                    let settings =
                        Settings::read(&summary.directory).map_err(RpcError::internal)?;
                    if settings.creation_key.as_ref() == Some(key) {
                        if settings.creation_fingerprint.as_ref() != Some(&fingerprint) {
                            return Err(RpcError::new(
                                4001,
                                "request_conflict",
                                "Creation request identity was reused with different options",
                            ));
                        }
                        existing = Some(summary.id);
                        break;
                    }
                }
            }
        }
        let id = if let Some(id) = existing {
            id
        } else {
            if self.sessions.lock().await.len() >= MAX_SESSIONS {
                return Err(RpcError::new(
                    4001,
                    "session_limit",
                    "Close an idle session before creating another",
                ));
            }
            let summary = self
                .library
                .create_session(None, Some(title))
                .map_err(RpcError::internal)?;
            Settings {
                source: source.into(),
                model,
                cwd,
                creation_key: key.clone(),
                creation_fingerprint: Some(fingerprint.clone()),
                metadata_revision: 0,
                execution_generation: 0,
            }
            .save(&summary.directory)
            .map_err(RpcError::internal)?;
            summary.id
        };
        if let Some(key) = key {
            if creates.len() >= 1024 {
                creates.clear();
            }
            creates.insert(key, (fingerprint, id.clone()));
        }
        let session = self.session(&id).await?;
        lock(&session.peers).insert(peer.id.clone(), peer.clone());
        Ok(self.snapshot(&session, peer, false))
    }

    async fn list(&self, params: &Value, peer: &Peer, active: bool) -> RpcResult {
        if params["cursor"].is_string() {
            return Err(RpcError::unsupported(
                "Cursor pagination is not implemented; use limit",
            ));
        }
        let limit = params["limit"].as_u64().unwrap_or(200).min(1000) as usize;
        let title = optional_string(params, "title")?;
        let query = optional_string(params, "query")?.map(str::to_lowercase);
        let workspace = optional_string(params, "workspace")?.map(PathBuf::from);
        let scope = optional_string(params, "scope")?.unwrap_or(if peer.canonical {
            "live"
        } else {
            "stored"
        });
        if !matches!(scope, "live" | "stored") {
            return Err(RpcError::invalid("scope must be live or stored"));
        }
        let mut rows = Vec::new();
        let sessions = self.sessions.lock().await;
        for summary in self.library.sessions(None).map_err(RpcError::internal)? {
            if rows.len() >= limit {
                break;
            }
            if (active || (peer.canonical && scope == "live"))
                && !sessions.contains_key(&summary.id)
            {
                continue;
            }
            if title.is_some_and(|t| t != summary.title) {
                continue;
            }
            if query.as_ref().is_some_and(|q| {
                !summary.title.to_lowercase().contains(q) && !summary.id.contains(q)
            }) {
                continue;
            }
            let settings = Settings::read(&summary.directory).map_err(RpcError::internal)?;
            let cwd = self
                .library
                .workspace_for_session(&summary.id)
                .map_err(RpcError::internal)?
                .or(settings.cwd);
            if workspace
                .as_ref()
                .is_some_and(|w| cwd.as_ref().is_none_or(|c| !c.starts_with(w)))
            {
                continue;
            }
            let (history, running) = if let Some(session) = sessions.get(&summary.id) {
                let view = lock(&session.view);
                (view.history.clone(), view.running)
            } else {
                (
                    read_history(&summary.directory).map_err(RpcError::internal)?,
                    false,
                )
            };
            let messages = transcript(&history);
            let model = settings
                .model
                .unwrap_or_else(|| self.provider.model.clone());
            let preview = history
                .turns
                .first()
                .map(|t| shorten(&t.prompt, 160))
                .unwrap_or_default();
            let last_reply = history.turns.last().and_then(|t| {
                t.entries.iter().rev().find_map(|e| match e {
                    Entry::Assistant { text, .. } => Some(shorten(text, 160)),
                    _ => None,
                })
            });
            if active && !peer.canonical {
                rows.push(json!({"current":false,"id":summary.id,"last_active":epoch_time(&summary.updated_at),
                    "message_count":messages.len(),"model":model,"preview":preview,"session_key":summary.id,
                    "source":settings.source,"started_at":epoch_time(&summary.created_at),"status":if running {"running"} else {"idle"},"title":summary.title}));
            } else {
                let mut row = json!({"id":summary.id,"title":summary.title,"preview":preview,"started_at":epoch_time(&summary.created_at),"message_count":messages.len(),"source":settings.source});
                if peer.canonical {
                    row["session_id"] = json!(summary.id);
                    row["running"] = json!(running);
                    row["model"] = json!(model);
                    row["last_reply"] = json!(last_reply);
                    row["cwd"] = json!(cwd);
                    row["workspace"] = row["cwd"].clone();
                    row["last_active_at"] = json!(epoch_time(&summary.updated_at));
                }
                rows.push(row);
            }
        }
        if peer.canonical {
            Ok(json!({"sessions":rows,"scope":scope,"next_cursor":null}))
        } else {
            Ok(json!({"sessions":rows}))
        }
    }

    async fn replay(&self, params: &Value, _peer: &Peer) -> RpcResult {
        // Like the upstream shared client, reconnect replay can precede resume.
        let session = self.resolve(params).await?;
        let cursor = params["last_sequence"]
            .as_u64()
            .or_else(|| params["last_seen"].as_u64())
            .unwrap_or(0);
        let mismatch =
            optional_string(params, "replay_epoch")?.is_some_and(|epoch| epoch != self.epoch);
        let view = lock(&session.view);
        let first = view
            .replay
            .front()
            .and_then(|e| e["seq"].as_u64())
            .unwrap_or(view.sequence + 1);
        let truncated = mismatch
            || cursor > view.sequence
            || cursor < view.evicted_through
            || (cursor < view.sequence && cursor + 1 < first);
        let events: Vec<_> = if mismatch {
            vec![]
        } else {
            view.replay
                .iter()
                .filter(|e| e["seq"].as_u64().unwrap_or(0) > cursor)
                .cloned()
                .collect()
        };
        Ok(
            json!({"count":events.len(),"events":events,"latest_seq":view.sequence,"last_sequence":view.sequence,
            "truncated":truncated,"snapshot_required":truncated,"epoch":self.epoch,"replay_epoch":self.epoch,"open_requests":[]}),
        )
    }

    async fn submit(self: &Arc<Self>, params: &Value, peer: &Peer) -> RpcResult {
        reject_options(
            params,
            &[
                "attachments",
                "api_text",
                "model_override",
                "turn_author",
                "voice_context",
                "voice_turn",
                "truncate_before_row_id",
                "truncate_before_user_ordinal",
                "truncate_before_message_id",
                "confirm_truncate",
                "display_kind",
            ],
        )?;
        if params["queued"].as_bool() == Some(true) {
            return Err(RpcError::unsupported(
                "This gateway does not queue turns; send after the current turn completes",
            ));
        }
        let text = required_string(params, "text")?.trim().to_owned();
        if text.is_empty() || text.len() > MAX_PROMPT_BYTES {
            return Err(RpcError::invalid("text must contain 1–262144 bytes"));
        }
        let input = optional_string(params, "submission_id")?
            .or(optional_string(params, "input_id")?)
            .map(str::to_owned);
        if matches!(&input,Some(s) if s.is_empty() || s.len()>256) {
            return Err(RpcError::invalid(
                "submission identity must contain 1–256 bytes",
            ));
        }
        let session = self.attached(params, peer).await?;
        let config = lock(&session.config).clone();
        config
            .validate()
            .map_err(|e| RpcError::new(4001, "provider_configuration", format!("{e:#}")))?;
        let id = lock(&session.summary).id.clone();
        let cancel = self.shutdown.child_token();
        let receipt = {
            let mut settings = lock(&session.settings);
            let mut view = lock(&session.view);
            if let Some(input) = &input
                && let Some(receipt) = view
                    .receipts
                    .iter()
                    .find(|r| r.input_id.as_ref() == Some(input))
            {
                if receipt.text != text {
                    return Err(RpcError::new(
                        4001,
                        "admission_conflict",
                        "Submission identity was reused with different text",
                    ));
                }
                return Ok(receipt.value(&id, false));
            }
            if view.running {
                return Err(RpcError::fenced(
                    "session_busy",
                    "A turn is already running; interrupt it or wait for completion",
                    &view,
                ));
            }
            let mut saved = settings.clone();
            saved.execution_generation = view.generation + 1;
            saved.save(&session.dir).map_err(RpcError::internal)?;
            *settings = saved;
            view.generation += 1;
            view.revision += 1;
            view.running = true;
            view.started = Some(now());
            view.cancel = Some(cancel.clone());
            let receipt = Receipt {
                id: new_id(),
                input_id: input,
                text: text.clone(),
                generation: view.generation,
                status: "started".into(),
                outcome: None,
                result: None,
            };
            view.current_admission = Some(receipt.id.clone());
            view.receipts.push_back(receipt.clone());
            while view.receipts.len() > MAX_RECEIPTS {
                view.receipts.pop_front();
            }
            receipt
        };
        // Legacy connections subscribe when submitting even when created by another client.
        lock(&session.peers).insert(peer.id.clone(), peer.clone());
        let shared = self.clone();
        let task = tokio::spawn(async move {
            shared.run_turn(session, config, text, cancel).await;
        });
        let mut turns = self.turns.lock().await;
        turns.retain(|task| !task.is_finished());
        turns.push(task);
        if peer.canonical {
            Ok(receipt.value(&id, false))
        } else {
            Ok(json!({"status":"streaming","admission_id":receipt.id}))
        }
    }

    async fn interrupt(&self, params: &Value, peer: &Peer, admission: bool) -> RpcResult {
        let session = self.attached(params, peer).await?;
        let id = lock(&session.summary).id.clone();
        let view = lock(&session.view);
        if admission {
            let admission_id = required_string(params, "admission_id")?;
            let receipt = view
                .receipts
                .iter()
                .find(|r| r.id == admission_id)
                .ok_or_else(|| RpcError::new(4004, "not_found", "Admission is not retained"))?;
            if receipt.status == "terminal" {
                return Ok(receipt.value(&id, false));
            }
            if view.current_admission.as_deref() != Some(admission_id) {
                return Err(RpcError::new(
                    4001,
                    "admission_started",
                    "The requested admission is not the active turn",
                ));
            }
        } else if peer.canonical {
            let generation = params["execution_generation"].as_u64().ok_or_else(|| {
                RpcError::invalid("execution_generation is required on native controls")
            })?;
            if generation != view.generation {
                return Err(RpcError::fenced(
                    "stale_generation",
                    "This control refers to an older turn",
                    &view,
                ));
            }
        }
        let interrupted = view.running;
        if let Some(cancel) = &view.cancel {
            cancel.cancel();
        }
        if admission {
            let receipt = view
                .receipts
                .iter()
                .find(|r| Some(&r.id) == view.current_admission.as_ref())
                .expect("active admission retained");
            return Ok(receipt.value(&id, false));
        }
        if peer.canonical {
            Ok(
                json!({"ref":{"profile_id":"default","session_id":id},"instance_id":self.instance_id,"authority_epoch":1,
                "revision":view.revision,"execution_generation":view.generation,"execution_state":if view.running {"running"} else {"idle"}}),
            )
        } else {
            Ok(
                json!({"status":if interrupted {"interrupted"} else {"not_interrupted"},"interrupted":interrupted}),
            )
        }
    }

    async fn respond(&self, params: &Value, peer: &Peer) -> RpcResult {
        let session = self.attached(params, peer).await?;
        let choice = optional_string(params, "choice")?.unwrap_or("deny");
        if !matches!(choice, "once" | "deny") {
            return Err(RpcError::invalid(
                "Only the offered once and deny choices are supported",
            ));
        }
        let requested =
            optional_string(params, "prompt_id")?.or(optional_string(params, "request_id")?);
        if peer.canonical && requested.is_none() {
            return Err(RpcError::invalid("prompt_id is required"));
        }
        let mut view = lock(&session.view);
        if peer.canonical {
            let generation = params["execution_generation"]
                .as_u64()
                .ok_or_else(|| RpcError::invalid("execution_generation is required"))?;
            if generation != view.generation {
                return Err(RpcError::fenced(
                    "stale_generation",
                    "This approval belongs to an older turn",
                    &view,
                ));
            }
        }
        let ids: Vec<_> = if params["all"].as_bool() == Some(true) {
            view.approvals.keys().cloned().collect()
        } else if let Some(id) = requested {
            vec![id.to_owned()]
        } else {
            view.approvals.keys().next().cloned().into_iter().collect()
        };
        let mut resolved = Vec::new();
        for id in ids {
            if let Some(approval) = view.approvals.remove(&id) {
                approval.timeout.cancel();
                if approval.answer.try_send(choice == "once").is_ok() {
                    resolved.push(id);
                }
            }
        }
        if peer.canonical && resolved.is_empty() {
            return Err(RpcError::new(
                4001,
                "prompt_stale",
                "Approval is no longer waiting; resume to refresh",
            ));
        }
        drop(view);
        for id in &resolved {
            self.emit(
                &session,
                "approval.settled",
                json!({"prompt_id":id,"request_id":id,"status":"resolved"}),
            );
        }
        if peer.canonical {
            Ok(json!({"status":"resolved"}))
        } else {
            Ok(json!({"resolved":resolved.len()}))
        }
    }

    async fn mutate(&self, params: &Value, peer: &Peer, canonical_method: bool) -> RpcResult {
        let session = self.attached(params, peer).await?;
        let operation = if canonical_method {
            required_string(params, "operation")?
        } else {
            "rename"
        };
        if !matches!(operation, "rename" | "model") {
            return Err(RpcError::unsupported(format!(
                "session.mutate operation {operation} is not implemented"
            )));
        }
        let payload = if canonical_method {
            &params["payload"]
        } else {
            params
        };
        if !payload.is_object() {
            return Err(RpcError::invalid("payload must be an object"));
        }
        let request_id = if canonical_method {
            Some(required_string(params, "request_id")?)
        } else {
            None
        };
        if request_id.is_some_and(|s| s.is_empty() || s.len() > 256) {
            return Err(RpcError::invalid("request_id must contain 1–256 bytes"));
        }
        let fingerprint=json!({"operation":operation,"payload":payload,"expected_revision":params["expected_revision"],"expected_generation":params["expected_generation"]}).to_string();
        // The lock order is shared with info(): summary, settings, config, view.
        let mut summary = lock(&session.summary);
        let mut settings = lock(&session.settings);
        let mut config = lock(&session.config);
        let mut view = lock(&session.view);
        let id = summary.id.clone();
        if let Some(request_id) = request_id {
            if let Some((_, old, value)) =
                view.mutations.iter().find(|(key, _, _)| key == request_id)
            {
                if old != &fingerprint {
                    return Err(RpcError::new(
                        4001,
                        "request_conflict",
                        "Mutation request identity was reused with different options",
                    ));
                }
                return Ok(value.clone());
            }
            let expected = params["expected_revision"]
                .as_u64()
                .ok_or_else(|| RpcError::invalid("expected_revision is required"))?;
            if expected != view.revision {
                return Err(RpcError::fenced(
                    "revision_conflict",
                    "Session changed; resume before editing",
                    &view,
                ));
            }
        }
        if operation == "model" {
            if view.running {
                return Err(RpcError::fenced(
                    "session_busy",
                    "Change the model after the turn finishes",
                    &view,
                ));
            }
            let expected = params["expected_generation"]
                .as_u64()
                .ok_or_else(|| RpcError::invalid("expected_generation is required"))?;
            if expected != view.generation {
                return Err(RpcError::fenced(
                    "stale_generation",
                    "Session execution changed",
                    &view,
                ));
            }
            if let Some(provider) = optional_string(payload, "provider")?
                && provider != provider_label(&self.provider)
            {
                return Err(RpcError::unsupported(
                    "Only the gateway's configured provider is available",
                ));
            }
            reject_options(payload, &["reasoning", "confirm"])?;
            let model = required_string(payload, "model")?;
            if model.trim().is_empty() || model.len() > 256 {
                return Err(RpcError::invalid("model must contain 1–256 bytes"));
            }
            let mut saved = settings.clone();
            saved.model = Some(model.into());
            saved.metadata_revision += 1;
            saved.save(&session.dir).map_err(RpcError::internal)?;
            *settings = saved;
            config.model = model.into();
        } else {
            let title = required_string(payload, "title")?;
            if title.trim().is_empty() || title.chars().count() > 200 {
                return Err(RpcError::invalid("title must contain 1–200 characters"));
            }
            let mut saved = settings.clone();
            saved.metadata_revision += 1;
            saved.save(&session.dir).map_err(RpcError::internal)?;
            *settings = saved;
            *summary = self
                .library
                .rename_session(&id, title)
                .map_err(RpcError::internal)?;
        }
        view.revision += 1;
        let receipt = json!({"session_id":id,"operation":operation,"revision":view.revision,"execution_generation":view.generation,"ok":true});
        if let Some(request_id) = request_id {
            view.mutations
                .push_back((request_id.into(), fingerprint, receipt.clone()));
            while view.mutations.len() > 256 {
                view.mutations.pop_front();
            }
        }
        drop(view);
        drop(config);
        drop(settings);
        drop(summary);
        self.emit(&session, "session.updated", receipt.clone());
        self.emit(&session, "session.info", self.info(&session));
        Ok(receipt)
    }

    async fn run_turn(
        self: Arc<Self>,
        session: Arc<LiveSession>,
        config: Config,
        text: String,
        cancel: CancellationToken,
    ) {
        let id = lock(&session.summary).id.clone();
        let mut stream = TurnStream {
            id: new_id(),
            segment: 0,
            offset: 0,
            sealed: std::collections::HashSet::new(),
            tools: std::collections::HashSet::new(),
            results: std::collections::HashSet::new(),
        };
        self.emit(&session, "message.start", json!({"turn_id":stream.id}));
        self.emit(&session, "session.info", self.info(&session));
        let (tx, rx) = async_channel::bounded(1024);
        let mut engine = session.engine.lock().await;
        let initial_count = engine.history().turns.len();
        let result = {
            let future = engine.turn(config, text, tx, cancel.clone());
            tokio::pin!(future);
            loop {
                tokio::select! {
                    result=&mut future => break result,
                    event=rx.recv() => if let Ok(event)=event {self.handle_event(&session,event,&mut stream,&cancel);},
                }
            }
        };
        while let Ok(event) = rx.try_recv() {
            self.handle_event(&session, event, &mut stream, &cancel);
        }
        let history = engine.history().clone();
        drop(engine);
        let (status, error) = match result {
            Err(error) => ("error", Some(format!("{error:#}"))),
            Ok(()) => match history.turns.get(initial_count).map(|t| &t.status) {
                None if cancel.is_cancelled() => ("interrupted", None),
                Some(Status::Complete) => ("complete", None),
                Some(Status::Interrupted) => ("interrupted", None),
                Some(Status::Failed(message)) => ("error", Some(message.clone())),
                _ => (
                    "error",
                    Some("The Rust runtime ended without a terminal status".into()),
                ),
            },
        };
        let text = history
            .turns
            .get(initial_count)
            .and_then(|turn| {
                turn.entries.iter().rev().find_map(|entry| match entry {
                    Entry::Assistant { text, .. } => Some(text.clone()),
                    _ => None,
                })
            })
            .unwrap_or_default();
        let payload =
            json!({"text":text,"status":status,"error":error,"partial":status!="complete"});
        let cancelled = {
            let mut view = lock(&session.view);
            view.history = history;
            view.revision += 1;
            let current = view.current_admission.clone();
            if let Some(receipt) = view
                .receipts
                .iter_mut()
                .find(|r| Some(&r.id) == current.as_ref())
            {
                receipt.status = "terminal".into();
                receipt.outcome = Some(status.into());
                receipt.result = Some(payload.clone());
            }
            let approvals = std::mem::take(&mut view.approvals);
            for approval in approvals.values() {
                approval.timeout.cancel();
                let _ = approval.answer.try_send(false);
            }
            approvals.into_keys().collect::<Vec<_>>()
        };
        if !cancelled.is_empty() {
            self.emit(&session,"approval.cancelled",json!({"session_id":id,"stored_session_id":id,"reason":status,"cancelled_count":cancelled.len(),"request_ids":cancelled}));
        }
        // Complete is the terminal fence. History and the receipt already reflect it.
        self.emit(&session, "message.complete", payload);
        {
            let mut view = lock(&session.view);
            view.running = false;
            view.cancel = None;
            view.started = None;
            view.current_admission = None;
        }
        match self.library.touch_session(&id) {
            Ok(summary)=>*lock(&session.summary)=summary,
            Err(error)=>self.emit(&session,"error",json!({"message":format!("Conversation saved, but catalog update failed: {error:#}")})),
        }
        self.emit(&session, "session.info", self.info(&session));
    }

    fn handle_event(
        self: &Arc<Self>,
        session: &Arc<LiveSession>,
        event: Event,
        stream: &mut TurnStream,
        cancel: &CancellationToken,
    ) {
        match event {
            Event::Subagent(progress) => self.emit(session, "subagent.progress", serde_json::to_value(&progress).unwrap_or_default()),
            Event::Text(text) => {
                self.emit(session,"message.delta",json!({"text":text,"turn_id":stream.id,"segment":stream.segment,"offset":stream.offset}));
                stream.offset += text.encode_utf16().count();
            }
            Event::Snapshot(history) => {
                if let Some(turn) = history.turns.last() {
                    for (index, entry) in turn.entries.iter().enumerate() {
                        match entry {
                            Entry::Assistant { text, calls } => {
                                if !calls.is_empty()
                                    && !text.is_empty()
                                    && stream.sealed.insert(index)
                                {
                                    self.emit(session,"message.interim",json!({"text":text,"already_streamed":true,"turn_id":stream.id,"segment":stream.segment}));
                                    stream.segment += 1;
                                    stream.offset = 0;
                                }
                                for call in calls {
                                    if stream.tools.insert(call.id.clone()) {
                                        self.emit(session,"tool.start",json!({"tool_id":call.id,"name":call.name,"args":serde_json::from_str::<Value>(&call.arguments).unwrap_or(Value::Null),"args_text":call.arguments}));
                                    }
                                }
                            }
                            Entry::Tool {
                                id, name, result, ..
                            } => {
                                if stream.results.insert(id.clone()) {
                                    self.emit(session,"tool.complete",json!({"tool_id":id,"name":name,"result":serde_json::from_str::<Value>(result).unwrap_or_else(|_|json!(result)),"result_text":result}));
                                }
                            }
                        }
                    }
                }
                lock(&session.view).history = history;
            }
            Event::Approval { call, answer } => {
                let id = new_id();
                let timeout = CancellationToken::new();
                let mut command = format!("{}({})", call.name, call.arguments);
                let key = lock(&session.config).api_key.clone();
                if !key.is_empty() {
                    command = command.replace(&key, "[redacted]");
                }
                let generation = lock(&session.view).generation;
                let payload = json!({"kind":"approval","prompt_id":id,"request_id":id,"execution_generation":generation,
                    "choices":["once","deny"],"allow_session":false,"allow_permanent":false,"tool_name":call.name,
                    "command":command,"description":"Approve this Rust tool call once?","timeout_s":300});
                lock(&session.view).approvals.insert(
                    id.clone(),
                    Approval {
                        payload: payload.clone(),
                        answer,
                        timeout: timeout.clone(),
                    },
                );
                self.emit(session, "approval.request", payload);
                let shared = self.clone();
                let session = session.clone();
                let cancel = cancel.clone();
                tokio::spawn(async move {
                    tokio::select! {
                        _=timeout.cancelled()=>return,
                        _=cancel.cancelled()=>return,
                        _=tokio::time::sleep(Duration::from_secs(300))=>{},
                    }
                    let approval = lock(&session.view).approvals.remove(&id);
                    if let Some(approval) = approval {
                        let _ = approval.answer.try_send(false);
                        shared.emit(
                            &session,
                            "approval.settled",
                            json!({"prompt_id":id,"request_id":id,"status":"expired"}),
                        );
                    }
                });
            }
        }
    }
}

struct TurnStream {
    id: String,
    segment: u64,
    offset: usize,
    sealed: std::collections::HashSet<usize>,
    tools: std::collections::HashSet<String>,
    results: std::collections::HashSet<String>,
}

fn required_string<'a>(params: &'a Value, name: &str) -> std::result::Result<&'a str, RpcError> {
    params[name]
        .as_str()
        .ok_or_else(|| RpcError::invalid(format!("{name} must be a string")))
}
fn optional_string<'a>(
    params: &'a Value,
    name: &str,
) -> std::result::Result<Option<&'a str>, RpcError> {
    if params[name].is_null() {
        Ok(None)
    } else {
        required_string(params, name).map(Some)
    }
}
fn reject_options(params: &Value, names: &[&str]) -> std::result::Result<(), RpcError> {
    for name in names {
        if let Some(value) = params.get(name)
            && !value.is_null()
            && value != &json!(false)
            && value != &json!([])
            && value != &json!("")
        {
            return Err(RpcError::unsupported(format!(
                "Explicit option {name} is not implemented by the Rust gateway"
            )));
        }
    }
    Ok(())
}

fn provider_label(config: &Config) -> &'static str {
    let base = config.base_url.trim_end_matches('/');
    if hermes_engine::portal::is_portal(base) {
        "nous"
    } else if base == hermes_engine::OPENROUTER {
        "openrouter"
    } else if base.starts_with("https://api.anthropic.com/") || base.ends_with("/messages") {
        "anthropic"
    } else if base.starts_with("https://api.openai.com/") {
        "openai"
    } else {
        "openai-compatible"
    }
}
fn tool_names() -> Vec<String> {
    hermes_engine::tools::tool_specs()
        .iter()
        .filter_map(|tool| {
            tool.pointer("/function/name")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}

fn shorten(value: &str, maximum: usize) -> String {
    value.chars().take(maximum).collect()
}
fn read_history(dir: &Path) -> Result<History> {
    match File::open(dir.join("conversation.json")) {
        Ok(file) => serde_json::from_reader(file).context("Could not read conversation history"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(History::default()),
        Err(e) => Err(e.into()),
    }
}
fn transcript(history: &History) -> Vec<Value> {
    let mut messages = Vec::new();
    for turn in &history.turns {
        messages.push(json!({"role":"user","text":turn.prompt,"content":turn.prompt,"row_id":messages.len()+1}));
        for entry in &turn.entries {
            match entry {
                Entry::Assistant{text,calls}=>{
                    let mut message=json!({"role":"assistant","text":text,"content":text,"row_id":messages.len()+1});
                    if !calls.is_empty() {message["tool_calls"]=json!(calls.iter().map(|call|json!({"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.arguments}})).collect::<Vec<_>>());}
                    messages.push(message);
                }
                Entry::Tool{id,name,result,approved}=>messages.push(json!({"role":"tool","text":result,"content":result,"tool_call_id":id,"name":name,"approved":approved,"row_id":messages.len()+1})),
            }
        }
    }
    messages
}

// HTTP/upgrade auth. No proxy or forwarded headers can widen the loopback boundary.
fn boundary(
    shared: &Shared,
    headers: &HeaderMap,
    remote: SocketAddr,
) -> std::result::Result<(), StatusCode> {
    if !remote.ip().is_loopback() {
        return Err(StatusCode::FORBIDDEN);
    }
    let host = headers
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .ok_or(StatusCode::FORBIDDEN)?;
    let authority = host
        .parse::<axum::http::uri::Authority>()
        .map_err(|_| StatusCode::FORBIDDEN)?;
    if !matches!(
        authority.host(),
        "localhost" | "127.0.0.1" | "[::1]" | "::1"
    ) || authority.port_u16() != Some(shared.address.port())
    {
        return Err(StatusCode::FORBIDDEN);
    }
    if let Some(origin) = headers.get(header::ORIGIN) {
        let origin = origin.to_str().map_err(|_| StatusCode::FORBIDDEN)?;
        let allowed = [
            format!("http://127.0.0.1:{}", shared.address.port()),
            format!("http://localhost:{}", shared.address.port()),
            format!("http://[::1]:{}", shared.address.port()),
        ];
        if !allowed.iter().any(|s| s == origin) && !matches!(origin, "file://" | "app://hermes") {
            return Err(StatusCode::FORBIDDEN);
        }
    }
    Ok(())
}
fn header_token(headers: &HeaderMap) -> Option<&str> {
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "));
    let session = headers
        .get("x-hermes-session-token")
        .and_then(|h| h.to_str().ok());
    match (bearer, session) {
        (Some(_), Some(_)) => None,
        (Some(token), None) | (None, Some(token)) => Some(token),
        _ => None,
    }
}
async fn health(
    State(shared): State<Arc<Shared>>,
    ConnectInfo(remote): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> Response {
    if let Err(status) = boundary(&shared, &headers, remote) {
        return status.into_response();
    }
    Json(json!({"status":"ok","implementation":"hermes-rust","version":env!("CARGO_PKG_VERSION")}))
        .into_response()
}
async fn mint_ticket(
    State(shared): State<Arc<Shared>>,
    ConnectInfo(remote): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> Response {
    if let Err(status) = boundary(&shared, &headers, remote) {
        return status.into_response();
    }
    if !header_token(&headers).is_some_and(|t| same_secret(t, &shared.token)) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let ticket = new_id();
    let mut tickets = lock(&shared.tickets);
    tickets.retain(|_, created| created.elapsed() < TICKET_TTL);
    if tickets.len() >= 256 {
        return StatusCode::TOO_MANY_REQUESTS.into_response();
    }
    tickets.insert(ticket.clone(), Instant::now());
    let mut response =
        Json(json!({"ticket":ticket,"ttl_seconds":TICKET_TTL.as_secs()})).into_response();
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store"),
    );
    response
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct WsQuery {
    token: Option<String>,
    ticket: Option<String>,
    native_dial: Option<String>,
}
async fn upgrade(
    State(shared): State<Arc<Shared>>,
    ConnectInfo(remote): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Query(query): Query<WsQuery>,
    ws: WebSocketUpgrade,
) -> Response {
    if let Err(status) = boundary(&shared, &headers, remote) {
        return status.into_response();
    }
    let protocols: Vec<_> = headers
        .get(header::SEC_WEBSOCKET_PROTOCOL)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    let protocol_tickets: Vec<_> = protocols
        .iter()
        .filter_map(|p| p.strip_prefix(TICKET_PREFIX))
        .collect();
    let canonical = query.native_dial.is_some() || protocols.contains(&PROTOCOL);
    if !protocol_tickets.is_empty()
        && (protocol_tickets.len() != 1 || !protocols.contains(&PROTOCOL))
    {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let sources = usize::from(query.token.is_some())
        + usize::from(query.ticket.is_some())
        + headers.get_all(header::AUTHORIZATION).iter().count()
        + headers.get_all("x-hermes-session-token").iter().count()
        + protocol_tickets.len();
    if sources != 1 {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let authenticated = if let Some(ticket) = query
        .ticket
        .as_deref()
        .or_else(|| protocol_tickets.first().copied())
    {
        lock(&shared.tickets)
            .remove(ticket)
            .is_some_and(|created| created.elapsed() < TICKET_TTL)
    } else if canonical {
        false
    } else {
        query
            .token
            .as_deref()
            .or_else(|| header_token(&headers))
            .is_some_and(|t| same_secret(t, &shared.token))
    };
    if !authenticated {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    if shared.connections.fetch_add(1, Ordering::AcqRel) >= MAX_CONNECTIONS {
        shared.connections.fetch_sub(1, Ordering::AcqRel);
        return StatusCode::TOO_MANY_REQUESTS.into_response();
    }
    let upgraded = ws.max_message_size(1_048_576).max_frame_size(1_048_576);
    let upgraded = if protocols.contains(&PROTOCOL) {
        upgraded.protocols([PROTOCOL])
    } else {
        upgraded
    };
    let failed = shared.clone();
    upgraded
        .on_failed_upgrade(move |_| {
            failed.connections.fetch_sub(1, Ordering::AcqRel);
        })
        .on_upgrade(move |socket| serve_socket(shared, socket, canonical))
        .into_response()
}

async fn serve_socket(shared: Arc<Shared>, socket: WebSocket, canonical: bool) {
    let (mut sink, mut source) = socket.split();
    let (tx, mut rx) = mpsc::channel::<Value>(256);
    let stop = shared.shutdown.child_token();
    let peer = Peer {
        id: new_id(),
        tx,
        stop: stop.clone(),
        canonical,
    };
    let ready = json!({"jsonrpc":"2.0","method":"event","params":{"type":"gateway.ready","payload":{
        "skin":{},"change_events":false,"heartbeat":true,"replay_epoch":shared.epoch,"session_authority":canonical}}});
    let _ = peer.tx.try_send(ready);
    let writer_stop = stop.clone();
    let writer = tokio::spawn(async move {
        loop {
            tokio::select! {
                _=writer_stop.cancelled()=>break,
                frame=rx.recv()=>match frame {
                    Some(frame)=>if !matches!(tokio::time::timeout(Duration::from_secs(10),sink.send(Message::Text(frame.to_string().into()))).await,Ok(Ok(()))) {break;},
                    None=>break,
                },
            }
        }
        let _ = sink.close().await;
        writer_stop.cancel();
    });
    loop {
        let incoming = tokio::select! {_=stop.cancelled()=>break,frame=source.next()=>frame};
        match incoming {
            Some(Ok(Message::Text(raw))) => {
                let parsed = serde_json::from_str::<Value>(&raw);
                let (id, result) = match parsed {
                    Err(_) => (
                        Value::Null,
                        Err(RpcError::new(-32700, "parse_error", "Invalid JSON")),
                    ),
                    Ok(frame) => {
                        let id = frame.get("id").cloned().unwrap_or(Value::Null);
                        if frame.get("id").is_none() && frame.get("method").is_none() {
                            continue;
                        }
                        if frame["jsonrpc"] != "2.0"
                            || !frame["method"].is_string()
                            || !(id.is_null() || id.is_string() || id.is_number())
                        {
                            (
                                id,
                                Err(RpcError::new(
                                    -32600,
                                    "invalid_request",
                                    "Expected a JSON-RPC 2.0 method request",
                                )),
                            )
                        } else {
                            let method = frame["method"].as_str().unwrap_or("");
                            let params = frame.get("params").cloned().unwrap_or_else(|| json!({}));
                            let result = shared.dispatch(method, &params, &peer).await;
                            if frame.get("id").is_none() {
                                continue;
                            }
                            (id, result)
                        }
                    }
                };
                let response = match result {
                    Ok(result) => json!({"jsonrpc":"2.0","id":id,"result":result}),
                    Err(error) => json!({"jsonrpc":"2.0","id":id,"error":error.value()}),
                };
                if peer.tx.send(response).await.is_err() {
                    break;
                }
            }
            Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
            Some(Ok(_)) => {}
        }
    }
    stop.cancel();
    for session in shared.sessions.lock().await.values() {
        lock(&session.peers).remove(&peer.id);
    }
    let _ = writer.await;
    shared.connections.fetch_sub(1, Ordering::AcqRel);
}
