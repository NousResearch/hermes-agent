use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::complete::CompleteItem;

use chrono::{DateTime, Local};

use crate::fs_tree::FileRow;
use crate::platform::probe_git_repo_branch;
use crate::slash::{local_entries, SlashEntry};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
    Tool {
        name: String,
        status: String,
        tool_id: Option<String>,
    },
    Reasoning,
    ImagePreview {
        path: String,
    },
    System,
    Compaction,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    #[allow(dead_code)]
    pub id: String,
    pub role: MessageRole,
    pub content: String,
    pub output: String,
    pub timestamp: DateTime<Local>,
    pub is_streaming: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmedKind {
    Quit,
    NewSession,
    Rollback,
    McpRemove,
    MemoryDelete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    Plan,
    Manual,
    Yolo,
}

impl PermissionMode {
    pub fn toggle(&self) -> Self {
        match self {
            Self::Yolo => Self::Manual,
            Self::Plan | Self::Manual => Self::Yolo,
        }
    }

    pub fn cycle(&self) -> Self {
        match self {
            Self::Plan => Self::Manual,
            Self::Manual => Self::Yolo,
            Self::Yolo => Self::Plan,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Plan => "plan",
            Self::Manual => "ask",
            Self::Yolo => "yolo",
        }
    }

    pub fn from_session_info(yolo: bool) -> Self {
        if yolo {
            Self::Yolo
        } else {
            Self::Manual
        }
    }

    pub fn needs_yolo_rpc(from: Self, to: Self) -> bool {
        matches!((from, to), (Self::Yolo, _) | (_, Self::Yolo))
    }

    pub fn needs_plan_rpc(from: Self, to: Self) -> bool {
        matches!((from, to), (Self::Plan, _) | (_, Self::Plan))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndicatorStyle {
    Unicode,
    Ascii,
    Emoji,
    Kaomoji,
}

impl IndicatorStyle {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "unicode" | "braille" => Some(Self::Unicode),
            "ascii" => Some(Self::Ascii),
            "emoji" => Some(Self::Emoji),
            "kaomoji" | "faces" => Some(Self::Kaomoji),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusBarMode {
    Top,
    Bottom,
    Off,
}

impl StatusBarMode {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "top" | "on" => Some(Self::Top),
            "bottom" => Some(Self::Bottom),
            "off" => Some(Self::Off),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusyMode {
    Queue,
    Steer,
    Interrupt,
}

impl BusyMode {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "queue" => Some(Self::Queue),
            "steer" => Some(Self::Steer),
            "interrupt" | "redirect" => Some(Self::Interrupt),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveView {
    Chat,
    Tasks,
    ModelPicker,
    BranchPicker,
    Skills,
    Sessions,
    ThemePicker,
    Context,
    Help,
    Profiles,
    Agents,
    Memory,
    Peek,
    Rollback,
    Background,
    Mcp,
    Palette,
    Tools,
    Plugins,
    Cron,
    Replay,
    Projects,
}

#[derive(Debug, Clone)]
pub struct ToolsetRow {
    pub name: String,
    pub description: String,
    pub enabled: bool,
    pub tool_count: u64,
}

#[derive(Debug, Clone)]
pub struct CronJobRow {
    pub id: String,
    pub name: String,
    pub schedule: String,
    pub enabled: bool,
    pub state: String,
    pub prompt: String,
}

#[derive(Debug, Clone)]
pub struct PluginRow {
    pub name: String,
    pub key: String,
    pub version: String,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct SpawnTreeEntry {
    pub label: String,
    pub path: String,
    pub count: u64,
}

#[derive(Debug, Clone)]
pub struct ProjectRow {
    pub id: String,
    pub name: String,
    pub count: u64,
}

#[derive(Debug, Clone)]
pub struct McpServerRow {
    pub name: String,
    pub transport: String,
    pub description: String,
    pub enabled: bool,
    pub installed: bool,
    pub connected: bool,
    pub tools: u64,
    pub requires: Vec<String>,
    pub configured: bool,
}

#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub description: String,
    pub command: String,
    pub allow_permanent: bool,
    pub request_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ClarifyQuestion {
    pub qid: Option<String>,
    pub question: String,
    pub choices: Vec<String>,
    pub multi_select: bool,
    pub selected: usize,
    pub selected_indices: HashSet<usize>,
    pub typed: String,
}

#[derive(Debug, Clone)]
pub struct ClarifyRequest {
    pub request_id: String,
    pub questions: Vec<ClarifyQuestion>,
    pub active: usize,
}

impl ClarifyRequest {
    pub fn current(&self) -> Option<&ClarifyQuestion> {
        self.questions.get(self.active)
    }

    pub fn current_mut(&mut self) -> Option<&mut ClarifyQuestion> {
        self.questions.get_mut(self.active)
    }

    pub fn is_batch(&self) -> bool {
        self.questions.iter().any(|q| q.qid.is_some())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretKind {
    Sudo,
    Secret,
}

#[derive(Debug, Clone)]
pub struct SecretRequest {
    pub kind: SecretKind,
    pub request_id: String,
    pub prompt: String,
    pub buffer: String,
}

#[derive(Debug, Clone, Default)]
pub struct PromptHistory {
    entries: Vec<String>,
    cursor: Option<usize>,
    stash: String,
}

impl PromptHistory {
    pub fn push(&mut self, text: String) {
        let text = text.trim_end().to_string();
        if text.is_empty() {
            return;
        }
        if self.entries.last() == Some(&text) {
            self.cursor = None;
            self.stash.clear();
            return;
        }
        self.entries.push(text);
        if self.entries.len() > 100 {
            self.entries.remove(0);
        }
        self.cursor = None;
        self.stash.clear();
    }

    pub fn prev(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        match self.cursor {
            None => {
                self.stash = current.to_string();
                self.cursor = Some(self.entries.len() - 1);
            }
            Some(0) => {}
            Some(i) => self.cursor = Some(i - 1),
        }
        self.entries.get(self.cursor?).cloned()
    }

    pub fn next(&mut self) -> Option<String> {
        match self.cursor {
            None => None,
            Some(i) if i + 1 >= self.entries.len() => {
                self.cursor = None;
                let stash = self.stash.clone();
                self.stash.clear();
                Some(stash)
            }
            Some(i) => {
                self.cursor = Some(i + 1);
                Some(self.entries[i + 1].clone())
            }
        }
    }

    pub fn reset_browse(&mut self) {
        self.cursor = None;
        self.stash.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

impl TaskStatus {
    pub fn from_gateway(s: &str) -> Self {
        match s {
            "in_progress" | "running" => Self::InProgress,
            "completed" | "done" => Self::Completed,
            "failed" | "error" | "cancelled" => Self::Failed,
            _ => Self::Pending,
        }
    }

    pub fn cycle(&self) -> Self {
        match self {
            Self::Pending => Self::InProgress,
            Self::InProgress => Self::Completed,
            Self::Completed => Self::Failed,
            Self::Failed => Self::Pending,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TaskItem {
    #[allow(dead_code)]
    pub id: String,
    pub title: String,
    pub status: TaskStatus,
}

#[derive(Debug, Clone)]
pub struct SessionRecord {
    pub id: String,
    pub title: String,
    pub updated_at: String,
    pub live: bool,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct SkillCard {
    pub name: String,
    pub category: String,
    pub description: String,
    pub preview: String,
}

#[derive(Debug, Clone)]
pub struct ProfileCard {
    pub name: String,
    pub display_name: String,
    pub model: String,
    pub provider: String,
    pub description: String,
    pub skill_count: u64,
    pub is_default: bool,
    pub last_session_id: Option<String>,
    pub last_title: String,
    pub last_preview: String,
    pub worker_active: bool,
}

#[derive(Debug, Clone)]
pub struct AgentRow {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub status: String,
    pub extra: String,
    pub depth: u32,
    pub parent_id: Option<String>,
    pub model: String,
    pub tool_count: u64,
    pub last_tool: String,
    pub notes: Vec<String>,
    pub thinking: Vec<String>,
    pub summary: String,
    pub started: Option<Instant>,
    pub duration_secs: Option<f64>,
    pub index: u64,
    pub pid: Option<u64>,
    pub cwd: String,
    pub output: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    pub iteration: u64,
    pub api_calls: u64,
}

impl AgentRow {
    pub(crate) fn subagent(id: String) -> Self {
        Self {
            id,
            kind: "subagent".into(),
            title: "subagent".into(),
            status: "queued".into(),
            extra: String::new(),
            depth: 0,
            parent_id: None,
            model: String::new(),
            tool_count: 0,
            last_tool: String::new(),
            notes: Vec::new(),
            thinking: Vec::new(),
            summary: String::new(),
            started: Some(Instant::now()),
            duration_secs: None,
            index: 0,
            pid: None,
            cwd: String::new(),
            output: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            cost_usd: 0.0,
            iteration: 0,
            api_calls: 0,
        }
    }

    pub fn tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }

    pub fn is_live(&self) -> bool {
        self.kind == "subagent" && matches!(self.status.as_str(), "running" | "queued")
    }

    pub fn is_subagent(&self) -> bool {
        self.kind == "subagent"
    }

    pub fn is_process(&self) -> bool {
        self.kind == "process"
    }

    pub fn is_running_process(&self) -> bool {
        self.is_process() && self.status == "running"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DockEntry {
    Agent(usize),
    Bg(usize),
}

pub fn is_terminal_agent_status(status: &str) -> bool {
    matches!(
        status,
        "completed" | "failed" | "error" | "interrupted" | "timeout"
    )
}

fn push_capped(buf: &mut Vec<String>, line: String) {
    buf.push(line);
    const CAP: usize = 12;
    if buf.len() > CAP {
        let drop_n = buf.len() - CAP;
        buf.drain(0..drop_n);
    }
}

#[derive(Debug, Clone)]
pub struct MemoryRow {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub meta: String,
    pub body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickerStage {
    Providers,
    Models,
    Key,
}

#[derive(Debug, Clone)]
pub struct ModelProvider {
    pub slug: String,
    pub name: String,
    pub models: Vec<String>,
    pub authenticated: bool,
    pub is_current: bool,
    pub warning: String,
    pub auth_type: String,
    pub key_env: String,
}

impl ModelProvider {
    pub fn accepts_inline_key(&self) -> bool {
        !self.authenticated
            && !self.key_env.is_empty()
            && (self.auth_type.is_empty() || self.auth_type == "api_key")
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HitRange {
    pub y: u16,
    pub x0: u16,
    pub x1: u16,
}

impl HitRange {
    pub fn contains(self, col: u16, row: u16) -> bool {
        row == self.y && col >= self.x0 && col < self.x1
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum HoverKind {
    #[default]
    None,
    Mode,
    Branch,
    Model,
    Context,
    Session,
    Background,
    Agents,
    Process,
    Dock(DockEntry),
    DockStop(DockEntry),
    Composer,
    TipsClose,
    TipsBar,
    Jump,
    Tool(String),
    Paste(String),
    Queue(usize),
    QueueSend(usize),
    QueueEdit(usize),
    QueueDrop(usize),
    Files(usize),
    Work(usize),
    Picker(usize),
}

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub hash: String,
    pub timestamp: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BgStatus {
    Running,
    Done,
}

#[derive(Debug, Clone)]
pub struct BgTask {
    pub id: String,
    pub prompt: String,
    pub status: BgStatus,
    pub result: String,
    pub started: Instant,
}

#[derive(Debug, Clone)]
pub struct PendingImage {
    pub path: PathBuf,
    pub name: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Toast {
    pub text: String,
    pub created: Instant,
    pub ttl: Duration,
}

impl Toast {
    pub fn live(&self) -> bool {
        self.created.elapsed() < self.ttl
    }
}

/// Snapshot so `u` can reverse a likely-success mutation.
#[derive(Debug, Clone)]
pub enum UndoKind {
    File {
        rel: String,
        previous: Option<Vec<u8>>,
    },
    Transcript {
        messages: Vec<ChatMessage>,
    },
}

#[derive(Debug, Clone)]
pub struct PendingUndo {
    pub kind: UndoKind,
    pub created: Instant,
}

impl PendingUndo {
    pub fn live(&self) -> bool {
        self.created.elapsed().as_secs_f64() < crate::optimistic::UNDO_SECS
    }

    pub fn hint(&self) -> &'static str {
        match self.kind {
            UndoKind::File { .. } => "u undo restore  ",
            UndoKind::Transcript { .. } => "u undo clear  ",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SessionMetrics {
    pub total_tokens: u64,
    pub context_used: u64,
    pub context_limit: u64,
    pub tokens_per_sec: f64,
    pub estimated_cost_usd: f64,
    pub active_model: String,
    pub active_provider: String,
    pub is_compacting: bool,
    pub compaction_status: String,
    pub compaction_started: Option<Instant>,
    pub compaction_hide_at: Option<Instant>,
    pub compaction_painted: bool,
    pub active_tool: Option<String>,
    pub turn_start_time: Option<Instant>,
    pub streaming_tokens_count: u64,
    pub cwd: String,
    pub git_branch: Option<String>,
    pub git_repo: Option<String>,
    pub permission_mode: PermissionMode,
    pub terminal_backend: String,
    pub approval_mode: String,
    pub toast_message: Option<Toast>,
    pub hermes_version: String,
    pub activity: String,
}

impl Default for SessionMetrics {
    fn default() -> Self {
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string());
        let (git_repo, git_branch) = probe_git_repo_branch(&cwd);

        Self {
            total_tokens: 0,
            context_used: 0,
            context_limit: 0,
            tokens_per_sec: 0.0,
            estimated_cost_usd: 0.0,
            active_model: String::new(),
            active_provider: String::new(),
            is_compacting: false,
            compaction_status: String::new(),
            compaction_started: None,
            compaction_hide_at: None,
            compaction_painted: false,
            active_tool: None,
            turn_start_time: None,
            streaming_tokens_count: 0,
            cwd,
            git_branch,
            git_repo,
            permission_mode: PermissionMode::Manual,
            terminal_backend: String::new(),
            approval_mode: String::new(),
            toast_message: None,
            hermes_version: String::new(),
            activity: String::new(),
        }
    }
}

impl SessionMetrics {
    pub fn context_pct(&self) -> f64 {
        if self.context_limit == 0 {
            return 0.0;
        }
        let used = if self.context_used > 0 {
            self.context_used
        } else {
            self.total_tokens
        };
        (used as f64 / self.context_limit as f64) * 100.0
    }
}

const THINKING_VERBS: &[&str] = &[
    "pondering",
    "contemplating",
    "musing",
    "cogitating",
    "ruminating",
    "deliberating",
    "mulling",
    "reflecting",
    "processing",
    "reasoning",
    "analyzing",
    "computing",
    "synthesizing",
    "formulating",
    "brainstorming",
];

/// Kaomoji + verb lines from `thinking_callback` are wait-status, not CoT.
pub fn is_thinking_status(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }
    // Faces + verb are short; real CoT is longer prose.
    if t.chars().count() > 96 {
        return false;
    }
    let lower = t.to_ascii_lowercase();
    THINKING_VERBS.iter().any(|verb| {
        lower == *verb
            || lower == format!("{verb}...")
            || lower == format!("{verb}…")
            || lower.contains(&format!("{verb}..."))
            || lower.contains(&format!("{verb}…"))
    })
}

pub fn activity_from_thinking(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    for verb in THINKING_VERBS {
        if lower.contains(verb) {
            return (*verb).to_string();
        }
    }
    let cleaned: String = text
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == ' ' || *c == '-' || *c == '.')
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if cleaned.is_empty() {
        "thinking".into()
    } else {
        cleaned.chars().take(32).collect()
    }
}

fn wait_activity_stem(act: &str) -> Option<&'static str> {
    let low = act.trim().trim_end_matches(['.', '…']).to_ascii_lowercase();
    if low == "thinking" {
        return Some("thinking");
    }
    if low == "writing" {
        return Some("writing");
    }
    THINKING_VERBS.iter().copied().find(|verb| *verb == low)
}

/// Kaomoji / wait verbs that repeat `Thinking`, not extra turn-bar detail.
pub fn is_wait_activity(act: &str) -> bool {
    wait_activity_stem(act).is_some()
}

/// Ink FaceTicker verb (`mulling`, `thinking`, `compacting`). Lowercase.
pub fn wait_status_label(act: &str) -> String {
    match wait_activity_stem(act) {
        Some("writing") => "writing".into(),
        Some(verb) => verb.to_string(),
        None => "thinking".into(),
    }
}

#[derive(Debug, Clone)]
pub struct AppState {
    pub session_id: Option<String>,
    pub vim: Option<crate::composer_vim::VimState>,
    pub session_key: String,
    pub startup_resume: Option<String>,
    pub pending_memory_edit: Option<String>,
    pub pending_memory_body: String,
    pub mcp_key_name: Option<String>,
    pub bg_tasks: Vec<BgTask>,
    pub session_title: String,
    pub session_started: Instant,
    pub messages: Vec<ChatMessage>,
    pub metrics: SessionMetrics,
    pub is_generating: bool,
    pub show_thinking: bool,
    pub active_view: ActiveView,
    pub modal_selected: usize,
    pub goal: Option<String>,
    pub tasks: Vec<TaskItem>,
    pub sessions_list: Vec<SessionRecord>,
    pub skills: Vec<SkillCard>,
    pub profiles: Vec<ProfileCard>,
    pub agent_rows: Vec<AgentRow>,
    pub agents_paused: bool,
    pub agents_caps: String,
    pub agents_steer: bool,
    pub agents_replay: bool,
    pub agents_nudged: bool,
    pub spawn_trees: Vec<SpawnTreeEntry>,
    pub projects_list: Vec<ProjectRow>,
    pub project_sessions: Vec<SessionRecord>,
    pub project_drill: Option<String>,
    pub compact: bool,
    pub indicator: IndicatorStyle,
    pub status_bar: StatusBarMode,
    pub fast_mode: bool,
    pub busy_mode: BusyMode,
    pub memory_summary: Vec<String>,
    pub memory_nodes: Vec<MemoryRow>,
    pub providers: Vec<ModelProvider>,
    pub picker_stage: PickerStage,
    pub picker_provider: usize,
    pub picker_list: Option<ratatui::layout::Rect>,
    pub picker_offset: usize,
    pub picker_filter: String,
    pub picker_key: String,
    pub picker_key_error: String,
    pub picker_key_saving: bool,
    pub hit_model: Option<HitRange>,
    pub hit_branch: Option<HitRange>,
    pub hit_mode: Option<HitRange>,
    pub hit_context: Option<HitRange>,
    pub hit_session: Option<HitRange>,
    pub hit_bg: Option<HitRange>,
    pub hit_agents: Option<HitRange>,
    pub hit_process: Option<HitRange>,
    pub hit_dock: Vec<(HitRange, DockEntry)>,
    pub hit_dock_stop: Vec<(HitRange, DockEntry)>,
    pub hit_dock_bar: Option<HitRange>,
    pub hover: HoverKind,
    pub click_flash: Option<(HoverKind, Instant)>,
    pub composer_area: Option<ratatui::layout::Rect>,
    pub queue_area: Option<ratatui::layout::Rect>,
    pub hit_queue: Vec<(HitRange, HoverKind)>,
    pub stream_area: Option<ratatui::layout::Rect>,
    pub hit_tools: Vec<(HitRange, String)>,
    pub expanded_tools: HashSet<String>,
    pub expand_epoch: u64,
    pub branches: Vec<crate::platform::GitBranch>,
    pub slash_open: bool,
    pub slash_query: String,
    pub slash_selected: usize,
    pub slash_catalog: Vec<SlashEntry>,
    pub slash_gateway: Vec<SlashEntry>,
    pub slash_replace_from: usize,
    pub complete_open: bool,
    pub complete_items: Vec<CompleteItem>,
    pub complete_selected: usize,
    pub complete_replace_from: usize,
    pub pending_images: Vec<PendingImage>,
    pub paste_chips: Vec<crate::paste::PasteChip>,
    pub peek_title: String,
    pub peek_body: String,
    pub peek_image: Option<PathBuf>,
    pub peek_offset: usize,
    pub hit_pastes: Vec<(HitRange, String)>,
    pub checkpoints: Vec<Checkpoint>,
    pub checkpoints_enabled: bool,
    pub rollback_diff: String,
    pub dirty: bool,
    /// Lines the viewport is raised above the tail. 0 = follow new output.
    pub scroll_from_bottom: usize,
    pub prompt_history: PromptHistory,
    pub pending_approval: Option<ApprovalRequest>,
    pub pending_clarify: Option<ClarifyRequest>,
    pub pending_secret: Option<SecretRequest>,
    pub protocol_warned: bool,
    pub prompt_queue: VecDeque<String>,
    pub queue_edit: Option<usize>,
    pub split_trace: bool,
    pub split_diff: bool,
    pub split_files: bool,
    pub split_work: bool,
    pub work_focus: bool,
    pub work_selected: usize,
    pub work_offset: usize,
    pub work_list: Option<ratatui::layout::Rect>,
    pub work_show_diff: bool,
    pub work_dirty: String,
    pub work_diff_files: Vec<crate::platform::DirtyFile>,
    pub work_diff_selected: usize,
    pub work_diff_offset: usize,
    pub files_focus: bool,
    pub files_rows: Vec<FileRow>,
    pub files_selected: usize,
    pub files_offset: usize,
    pub files_expanded: HashSet<String>,
    pub files_status: HashMap<String, char>,
    pub files_preview: String,
    pub files_list: Option<ratatui::layout::Rect>,
    pub diff_text: String,
    pub diff_tool_id: Option<String>,
    pub theme_id: String,
    pub theme_revert: Option<String>,
    pub hermes_home: std::path::PathBuf,
    pub trace_open: bool,
    pub trace_focus: bool,
    pub trace_follow: bool,
    pub trace_selected: usize,
    pub resume_step: Option<usize>,
    pub intro_tools: Vec<(String, Vec<String>)>,
    pub intro_skills: Vec<(String, Vec<String>)>,
    pub mcp_connected: usize,
    pub mcp_servers: Vec<McpServerRow>,
    pub toolsets: Vec<ToolsetRow>,
    pub plugins: Vec<PluginRow>,
    pub cron_jobs: Vec<CronJobRow>,
    pub shell_context: String,
    pub focus_view: bool,
    pub want_attention: bool,
    pub mouse_on: bool,
    pub release_date: String,
    pub intro_warning: Option<String>,
    pub session_ready: bool,
    pub reveal_started: Option<Instant>,
    pub tips_open: bool,
    pub tip_index: usize,
    pub tip_shown_at: Instant,
    pub hit_tips_close: Option<HitRange>,
    pub hit_tips_bar: Option<HitRange>,
    pub hit_jump: Option<HitRange>,
    pub yolo_epoch: u64,
    pub model_epoch: u64,
    pub pending_undo: Option<PendingUndo>,
    pub pending_fold: Option<String>,
    pub armed: Option<(ArmedKind, Instant)>,
}

mod agents;
mod chrome;
mod parse;
mod session;
pub use parse::*;

#[derive(Debug, Clone)]
pub struct ToolStep {
    pub index: usize,
    pub name: String,
    pub status: String,
    pub args: String,
    #[allow(dead_code)]
    pub tool_id: Option<String>,
    #[allow(dead_code)]
    pub msg_index: usize,
}

pub fn resume_from_step_prompt(step: &ToolStep, edited_args: &str) -> String {
    let args = if edited_args.trim().is_empty() {
        step.args.as_str()
    } else {
        edited_args.trim()
    };
    format!(
        "Resume from tool step {} (`{}`, was {}).\nUse these arguments:\n```\n{}\n```\nContinue from this step. Do not restart the whole task unless this step requires it.",
        step.index, step.name, step.status, args
    )
}

pub(crate) fn pending_from_path(path: PathBuf) -> PendingImage {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string());
    let dims = crate::ui::markdown::image_dims(&path);
    PendingImage {
        path,
        name,
        width: dims.map(|d| d.0),
        height: dims.map(|d| d.1),
    }
}

fn maybe_attach_images(state: &mut AppState, user_text: &str) {
    for path in crate::complete::image_refs_in(user_text, &state.metrics.cwd) {
        let shown = path.to_string_lossy().to_string();
        let already = state
            .messages
            .iter()
            .any(|m| matches!(&m.role, MessageRole::ImagePreview { path: p } if p == &shown));
        if already {
            continue;
        }
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| shown.clone());
        state.messages.push(ChatMessage {
            id: uuid::Uuid::new_v4().to_string(),
            role: MessageRole::ImagePreview { path: shown },
            content: format!("Image: {name}"),
            timestamp: Local::now(),
            output: String::new(),
            is_streaming: false,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn want_work_excludes_files() {
        let mut s = AppState::new();
        s.split_work = true;
        s.split_files = false;
        assert!(s.want_work(80));
        s.split_files = true;
        assert!(!s.want_work(80));
        s.split_files = false;
        s.split_work = true;
        s.split_diff = true;
        assert!(!s.want_diff(80));
        s.work_move(1);
        assert_eq!(s.work_selected, 0);
        s.agent_rows
            .push(crate::state::AgentRow::subagent("a".into()));
        s.agent_rows
            .push(crate::state::AgentRow::subagent("b".into()));
        s.work_move(1);
        assert_eq!(s.work_selected, 1);
        s.work_move(5);
        assert_eq!(s.work_selected, 1);
        s.work_show_diff = true;
        s.work_diff_files = crate::platform::list_dirty_files(" M a.rs\n M b.rs\n", "");
        s.work_diff_selected = 0;
        s.work_move(1);
        assert_eq!(s.work_diff_selected, 1);
        s.diff_text.clear();
        s.work_diff_offset = 0;
        s.work_scroll_diff(3);
        assert_eq!(s.work_diff_offset, 0);
    }

    #[test]
    fn toggle_edit_diff_opens_and_closes() {
        let mut s = AppState::new();
        s.messages.push(ChatMessage {
            id: "ed1".into(),
            role: MessageRole::Tool {
                name: "search_replace".into(),
                status: "completed".into(),
                tool_id: None,
            },
            content: r#"{"path":"src/ui/footer.rs","old_string":"a","new_string":"b"}"#.into(),
            output: String::new(),
            timestamp: chrono::Local::now(),
            is_streaming: false,
        });
        s.toggle_edit_diff("ed1");
        assert!(s.split_diff);
        assert_eq!(s.diff_tool_id.as_deref(), Some("ed1"));
        assert!(s.diff_text.contains("footer.rs"));
        assert!(s.diff_text.contains("+b"));
        s.toggle_edit_diff("ed1");
        assert!(!s.split_diff);
        assert!(s.diff_tool_id.is_none());
    }

    #[test]
    fn jump_to_tail_clears_scroll() {
        let mut s = AppState::new();
        assert!(!s.scrolled_off_tail());
        s.scroll_older(12);
        assert!(s.scrolled_off_tail());
        s.jump_to_tail();
        assert!(!s.scrolled_off_tail());
        assert_eq!(s.scroll_from_bottom, 0);
    }

    #[test]
    fn tab_title_tracks_activity_and_session() {
        let mut s = AppState::new();
        s.session_title = "ship the tui".into();
        let idle = s.tab_title();
        assert!(idle.starts_with("ship the tui"), "{idle}");
        assert!(idle.ends_with("hermes"), "{idle}");
        assert!(!idle.contains('\x1b'));

        s.is_generating = true;
        s.messages.push(ChatMessage {
            id: "t".into(),
            role: MessageRole::Tool {
                name: "read_file".into(),
                status: "running...".into(),
                tool_id: None,
            },
            content: r#"{"path":"src/app.rs"}"#.into(),
            output: String::new(),
            timestamp: Local::now(),
            is_streaming: false,
        });
        let busy = s.tab_title();
        assert!(busy.starts_with("thinking…"), "{busy}");
        assert!(busy.contains("ship the tui"));
        assert!(busy.ends_with("hermes"));

        s.messages.clear();
        let think = s.tab_title();
        assert!(think.starts_with("thinking…"), "{think}");

        s.pending_approval = Some(ApprovalRequest {
            description: "run".into(),
            command: "ls".into(),
            allow_permanent: false,
            request_id: None,
        });
        let ask = s.tab_title();
        assert!(ask.starts_with("approval needed"), "{ask}");
    }

    #[test]
    fn start_turn_does_not_insert_thought() {
        let mut s = AppState::new();
        s.start_turn("hi".into());
        assert!(s.is_generating);
        assert!(s.show_turn_bar());
        assert_eq!(s.metrics.activity, "thinking");
        assert!(!s.messages.iter().any(|m| m.role == MessageRole::Reasoning));
        s.finish_streaming();
        assert!(!s.show_turn_bar());
        assert!(!s.messages.iter().any(|m| m.role == MessageRole::Reasoning));
    }

    #[test]
    fn empty_intro_keeps_animating() {
        let s = AppState::new();
        assert!(s.messages.is_empty());
        assert!(s.needs_animation());
        assert_eq!(s.reveal(), 0.0);
        let mut s = AppState::new();
        s.start_turn("hi".into());
        s.reveal_started = Some(Instant::now() - std::time::Duration::from_secs(2));
        s.finish_streaming();
        // Gold wash keeps the idle canvas ticking.
        assert!(s.needs_animation());
        crate::ui::theme::apply(crate::palette::Palette::midnight());
        assert!(!s.needs_animation());
        crate::ui::theme::apply(crate::palette::Palette::gold());
    }

    #[test]
    fn boot_holds_then_reveals() {
        let mut s = AppState::new();
        assert!(!s.session_ready);
        assert_eq!(s.reveal(), 0.0);
        s.mark_session_ready();
        assert!(s.session_ready);
        assert!(s.reveal() < 1.0);
        s.reveal_started = Some(Instant::now() - std::time::Duration::from_secs(2));
        assert!((s.reveal() - 1.0).abs() < 0.01);
    }

    #[test]
    fn activity_from_thinking_extracts_verb() {
        assert_eq!(
            activity_from_thinking("( •_•)>⌐■-■ contemplating...Online and ready."),
            "contemplating"
        );
        assert_eq!(activity_from_thinking("looking around"), "looking around");
        assert!(is_thinking_status(
            "( •_•)>⌐■-■ contemplating...Online and ready."
        ));
        assert!(!is_thinking_status("considering the repo layout"));
        assert!(is_wait_activity("mulling"));
        assert!(is_wait_activity("Thinking..."));
        assert!(!is_wait_activity("◆ Security Advisories"));
        assert_eq!(wait_status_label("mulling"), "mulling");
        assert_eq!(wait_status_label("thinking"), "thinking");
    }

    #[test]
    fn thinking_delta_fills_placeholder() {
        let mut s = AppState::new();
        s.start_turn("hi".into());
        s.append_reasoning_delta("considering the repo");
        assert_eq!(
            s.messages
                .iter()
                .filter(|m| m.role == MessageRole::Reasoning)
                .count(),
            1
        );
        assert!(s.messages.iter().any(|m| m.content.contains("considering")));
        s.finish_streaming();
        assert!(s.messages.iter().any(|m| m.role == MessageRole::Reasoning));
    }

    #[test]
    fn thinking_after_tools_starts_a_new_row() {
        let mut s = AppState::new();
        s.start_turn("hi".into());
        s.append_reasoning_delta("Planning task checklist");
        s.freeze_thought();
        s.messages.push(ChatMessage {
            id: "t".into(),
            role: MessageRole::Tool {
                name: "read_file".into(),
                status: "completed".into(),
                tool_id: Some("1".into()),
            },
            content: "Cargo.toml".into(),
            timestamp: Local::now(),
            output: String::new(),
            is_streaming: false,
        });
        s.append_reasoning_delta("Inspecting SKILL.md next");
        let roles: Vec<_> = s.messages.iter().map(|m| format!("{:?}", m.role)).collect();
        assert!(
            matches!(s.messages[1].role, MessageRole::Reasoning),
            "{roles:?}"
        );
        assert!(
            matches!(s.messages[2].role, MessageRole::Tool { .. }),
            "{roles:?}"
        );
        assert!(
            matches!(s.messages[3].role, MessageRole::Reasoning),
            "{roles:?}"
        );
        assert!(s.messages[1].content.contains("Planning"));
        assert!(s.messages[3].content.contains("SKILL"));
        assert!(!s.messages[1].content.contains("SKILL"));
        assert!(s.messages[3].is_streaming);
        assert!(!s.messages[1].is_streaming);
        s.finish_streaming();
        assert_eq!(
            s.messages
                .iter()
                .filter(|m| m.role == MessageRole::Reasoning)
                .count(),
            2,
            "later thoughts stay after the tools they followed"
        );
    }

    #[test]
    fn thinking_after_assistant_starts_a_new_row() {
        let mut s = AppState::new();
        s.start_turn("hi".into());
        s.append_reasoning_delta("first");
        s.append_assistant_delta("hello");
        s.append_reasoning_delta("second");
        assert!(matches!(s.messages[1].role, MessageRole::Reasoning));
        assert!(matches!(s.messages[2].role, MessageRole::Assistant));
        assert!(matches!(s.messages[3].role, MessageRole::Reasoning));
        assert!(s.messages[1].content.contains("first"));
        assert!(s.messages[3].content.contains("second"));
        assert!(!s.messages[1].is_streaming);
        assert!(s.messages[3].is_streaming);
        s.finish_streaming();
        assert_eq!(
            s.messages
                .iter()
                .filter(|m| m.role == MessageRole::Reasoning)
                .count(),
            2
        );
    }

    #[test]
    fn reasoning_snapshot_does_not_duplicate() {
        let mut s = AppState::new();
        s.start_turn("hi".into());
        s.append_reasoning_delta("foo");
        s.append_reasoning_delta("foobar");
        let thought = s
            .messages
            .iter()
            .find(|m| m.role == MessageRole::Reasoning)
            .unwrap();
        assert_eq!(thought.content, "foobar");
    }

    #[test]
    fn assistant_survives_tool_in_between() {
        let mut s = AppState::new();
        s.start_turn("hi".into());
        s.append_assistant_delta("Hel");
        s.messages.push(ChatMessage {
            id: "t".into(),
            role: MessageRole::Tool {
                name: "shell".into(),
                status: "running...".into(),
                tool_id: Some("1".into()),
            },
            content: String::new(),
            timestamp: Local::now(),
            output: String::new(),
            is_streaming: false,
        });
        s.append_assistant_delta("lo");
        let assistants: Vec<_> = s
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::Assistant)
            .collect();
        assert_eq!(assistants.len(), 1);
        assert_eq!(assistants[0].content, "Hello");
    }

    #[test]
    fn assistant_deltas_coalesce() {
        let mut s = AppState::new();
        s.append_assistant_delta("Hel");
        s.append_assistant_delta("lo");
        assert_eq!(s.messages.len(), 1);
        assert_eq!(s.messages[0].content, "Hello");
        assert!(s.messages[0].is_streaming);
        s.metrics.tokens_per_sec = 12.0;
        s.finish_streaming();
        assert!(!s.messages[0].is_streaming);
        assert!(!s.is_generating);
        assert_eq!(s.metrics.tokens_per_sec, 0.0);
    }

    #[test]
    fn reasoning_and_assistant_both_finish() {
        let mut s = AppState::new();
        s.append_reasoning_delta("think");
        s.append_assistant_delta("say");
        s.finish_streaming();
        assert!(s.messages.iter().all(|m| !m.is_streaming));
    }

    #[test]
    fn tool_complete_matches_running_by_id() {
        let mut s = AppState::new();
        s.messages.push(ChatMessage {
            id: "1".into(),
            role: MessageRole::Tool {
                name: "shell".into(),
                status: "running...".into(),
                tool_id: Some("t1".into()),
            },
            content: String::new(),
            timestamp: Local::now(),
            output: String::new(),
            is_streaming: false,
        });
        s.messages.push(ChatMessage {
            id: "2".into(),
            role: MessageRole::Tool {
                name: "read".into(),
                status: "running...".into(),
                tool_id: Some("t2".into()),
            },
            content: String::new(),
            timestamp: Local::now(),
            output: String::new(),
            is_streaming: false,
        });
        s.complete_tool(Some("shell"), Some("t1"), false, None);
        match &s.messages[0].role {
            MessageRole::Tool { status, .. } => assert_eq!(status, "completed"),
            _ => panic!("expected tool"),
        }
        match &s.messages[1].role {
            MessageRole::Tool { status, .. } => assert_eq!(status, "running..."),
            _ => panic!("expected tool"),
        }
    }

    #[test]
    fn parse_todos_from_gateway() {
        let v = json!([
            { "id": "a", "content": "Write tests", "status": "in_progress" },
            { "id": "b", "title": "Ship", "status": "pending" }
        ]);
        let todos = parse_todos(&v);
        assert_eq!(todos.len(), 2);
        assert_eq!(todos[0].status, TaskStatus::InProgress);
        assert_eq!(todos[1].title, "Ship");
        let wrapped = parse_todos(&json!({
            "todos": [{ "content": "Inspect repo", "status": "completed" }]
        }));
        assert_eq!(wrapped[0].title, "Inspect repo");
        assert_eq!(wrapped[0].status, TaskStatus::Completed);
        let blob = tasks_blob(&wrapped);
        let round = parse_todos(&serde_json::from_str(&blob).unwrap());
        assert_eq!(round[0].title, "Inspect repo");
        assert_eq!(round[0].status, TaskStatus::Completed);
    }

    #[test]
    fn arm_quit_expires_and_consumes() {
        let mut s = AppState::new();
        assert!(!s.take_armed(ArmedKind::Quit));
        s.arm(ArmedKind::Quit, "bye");
        assert!(!s.take_armed(ArmedKind::NewSession));
        assert!(s.take_armed(ArmedKind::Quit));
        assert!(!s.take_armed(ArmedKind::Quit));
        s.arm(ArmedKind::Quit, "bye");
        s.armed = Some((ArmedKind::Quit, Instant::now() - Duration::from_secs(1)));
        assert!(!s.take_armed(ArmedKind::Quit));
        assert!(s.armed.is_none());
    }

    #[test]
    fn has_unsaved_draft_or_queue() {
        let mut s = AppState::new();
        assert!(!s.has_unsaved(""));
        assert!(!s.has_thread());
        assert!(s.has_unsaved("hi"));
        s.enqueue("later".into());
        assert!(s.has_unsaved(""));
        assert!(s.has_thread());
    }

    #[test]
    fn lagged_stops_generating() {
        let mut s = AppState::new();
        s.is_generating = true;
        s.note_lagged(12);
        assert!(!s.is_generating);
        let toast = s.metrics.toast_message.as_ref().unwrap();
        assert!(toast.text.contains("dropped 12"));
        s.note_lagged(0);
        assert!(s
            .metrics
            .toast_message
            .as_ref()
            .unwrap()
            .text
            .contains("dropped 12"));
    }

    #[test]
    fn parse_checkpoints_reads_hash() {
        let v = serde_json::json!({
            "enabled": true,
            "checkpoints": [
                {"hash": "abc123def", "timestamp": "2026-09-03", "message": "edit app.rs"}
            ]
        });
        let (on, rows) = parse_checkpoints(&v);
        assert!(on);
        assert_eq!(rows[0].hash, "abc123def");
        assert_eq!(rows[0].message, "edit app.rs");
    }

    #[test]
    fn trim_last_user_turn_drops_exchange() {
        let mut s = AppState::new();
        s.add_user_message("one".into());
        s.append_assistant_delta("a");
        s.add_user_message("two".into());
        s.append_assistant_delta("b");
        s.trim_last_user_turn();
        assert_eq!(
            s.messages
                .iter()
                .filter(|m| m.role == MessageRole::User)
                .count(),
            1
        );
    }

    #[test]
    fn yolo_toggle() {
        assert_eq!(PermissionMode::Manual.toggle(), PermissionMode::Yolo);
        assert_eq!(PermissionMode::Yolo.toggle(), PermissionMode::Manual);
        assert_eq!(PermissionMode::Plan.toggle(), PermissionMode::Yolo);
        assert_eq!(
            PermissionMode::from_session_info(true),
            PermissionMode::Yolo
        );
        assert_eq!(PermissionMode::Plan.cycle(), PermissionMode::Manual);
        assert_eq!(PermissionMode::Manual.cycle(), PermissionMode::Yolo);
        assert_eq!(PermissionMode::Yolo.cycle(), PermissionMode::Plan);
        assert!(PermissionMode::needs_yolo_rpc(
            PermissionMode::Manual,
            PermissionMode::Yolo
        ));
        assert!(!PermissionMode::needs_yolo_rpc(
            PermissionMode::Plan,
            PermissionMode::Manual
        ));
        assert!(PermissionMode::needs_plan_rpc(
            PermissionMode::Manual,
            PermissionMode::Plan
        ));
        assert!(PermissionMode::needs_plan_rpc(
            PermissionMode::Plan,
            PermissionMode::Manual
        ));
        let mut s = AppState::new();
        s.metrics.permission_mode = PermissionMode::Plan;
        s.apply_session_yolo(false);
        assert_eq!(s.metrics.permission_mode, PermissionMode::Plan);
        s.apply_session_plan(false);
        assert_eq!(s.metrics.permission_mode, PermissionMode::Manual);
        s.apply_session_plan(true);
        assert_eq!(s.metrics.permission_mode, PermissionMode::Plan);
        s.apply_session_yolo(true);
        assert_eq!(s.metrics.permission_mode, PermissionMode::Yolo);
        let mut chrome = AppState::new();
        assert!(chrome.apply_config_value("density", "on"));
        assert!(chrome.compact);
        assert!(chrome.apply_config_value("indicator", "kaomoji"));
        assert_eq!(chrome.indicator, IndicatorStyle::Kaomoji);
        assert!(chrome.apply_config_value("statusbar", "off"));
        assert_eq!(chrome.status_bar, StatusBarMode::Off);
        assert!(chrome.apply_config_value("statusbar", "toggle"));
        assert_eq!(chrome.status_bar, StatusBarMode::Top);
        assert!(chrome.apply_config_value("fast", "on"));
        assert!(chrome.fast_mode);
        assert!(chrome.apply_config_value("busy", "steer"));
        assert_eq!(chrome.busy_mode, BusyMode::Steer);
        assert!(chrome.apply_config_value("reasoning", "show"));
        assert!(chrome.show_thinking);
        let mut f = AppState::new();
        f.add_user_message("one".into());
        f.append_assistant_delta("a");
        f.finish_streaming();
        f.add_user_message("two".into());
        f.focus_view = true;
        assert_eq!(f.visible_messages().len(), 1);
        assert!(matches!(f.visible_messages()[0].role, MessageRole::User));
        assert_eq!(f.last_user_text().as_deref(), Some("two"));
        f.append_assistant_delta("alpha");
        f.finish_streaming();
        assert_eq!(f.assistant_text(None), Some("alpha"));
        assert_eq!(f.assistant_text(Some(1)), Some("a"));
        assert_eq!(f.assistant_text(Some(99)), Some("alpha"));
        f.messages.push(ChatMessage {
            id: "t".into(),
            role: MessageRole::Tool {
                name: "read".into(),
                status: "completed".into(),
                tool_id: None,
            },
            content: "x".into(),
            output: String::new(),
            timestamp: chrono::Local::now(),
            is_streaming: false,
        });
        f.toggle_details();
        assert!(f.expanded_tools.contains("t"));
        f.toggle_details();
        assert!(f.expanded_tools.is_empty());
    }

    #[test]
    fn toast_live_uses_ttl() {
        let mut s = AppState::new();
        s.set_toast_for("hi", Duration::from_millis(50));
        assert!(s.metrics.toast_message.as_ref().unwrap().live());
        std::thread::sleep(Duration::from_millis(70));
        assert!(!s.metrics.toast_message.as_ref().unwrap().live());
    }

    #[test]
    fn file_undo_expires() {
        let mut s = AppState::new();
        s.pending_undo = Some(PendingUndo {
            kind: UndoKind::File {
                rel: "a.rs".into(),
                previous: Some(b"old".to_vec()),
            },
            created: Instant::now() - Duration::from_secs(30),
        });
        assert!(s.take_live_undo().is_none());
        s.pending_undo = Some(PendingUndo {
            kind: UndoKind::File {
                rel: "a.rs".into(),
                previous: Some(b"old".to_vec()),
            },
            created: Instant::now(),
        });
        let u = s.take_live_undo().expect("live");
        match u.kind {
            UndoKind::File { rel, .. } => assert_eq!(rel, "a.rs"),
            _ => panic!("expected file undo"),
        }
        assert!(s.pending_undo.is_none());
    }

    #[test]
    fn file_undo_rewrites_snapshot() {
        let dir = std::env::temp_dir().join(format!("hermes-tui-undo-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let rel = "snap.txt";
        std::fs::write(dir.join(rel), b"head").unwrap();
        let mut s = AppState::new();
        s.metrics.cwd = dir.to_string_lossy().to_string();
        s.pending_undo = Some(PendingUndo {
            kind: UndoKind::File {
                rel: rel.into(),
                previous: Some(b"dirty".to_vec()),
            },
            created: Instant::now(),
        });
        let msg = s.apply_undo();
        assert!(msg.starts_with("undid restore"), "{msg}");
        assert_eq!(std::fs::read(dir.join(rel)).unwrap(), b"dirty");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clear_transcript_undoes() {
        let mut s = AppState::new();
        s.add_system("keep me");
        assert_eq!(s.clear_transcript(), "cleared · u undo");
        assert!(s.messages.is_empty());
        assert_eq!(s.apply_undo(), "undid clear");
        assert_eq!(s.messages.len(), 1);
        assert_eq!(s.clear_transcript(), "cleared · u undo");
        assert_eq!(s.clear_transcript(), "already empty");
    }

    #[test]
    fn compaction_same_drain_skips_loader() {
        let mut s = AppState::new();
        s.begin_compaction();
        assert!(s.metrics.is_compacting);
        s.end_compaction();
        assert!(!s.metrics.is_compacting);
        assert!(s.pending_fold.is_none());
        assert!(s.messages.iter().any(|m| m.role == MessageRole::Compaction));
    }

    #[test]
    fn compaction_holds_min_visible_once_painted() {
        let mut s = AppState::new();
        s.begin_compaction();
        s.metrics.compaction_painted = true;
        s.metrics.compaction_started = Some(Instant::now() - Duration::from_millis(200));
        s.end_compaction();
        assert!(s.metrics.is_compacting);
        assert!(s.pending_fold.is_some());
        s.metrics.compaction_hide_at = Some(Instant::now() - Duration::from_millis(1));
        s.release_holds();
        assert!(!s.metrics.is_compacting);
        assert!(s.pending_fold.is_none());
        assert_eq!(
            s.messages
                .iter()
                .filter(|m| m.role == MessageRole::Compaction)
                .count(),
            1
        );
    }

    #[test]
    fn thinking_status_accepts_unicode_ellipsis() {
        assert!(is_thinking_status("pondering…"));
        assert!(is_thinking_status("pondering..."));
    }

    #[test]
    fn context_pct_zero_limit() {
        let m = SessionMetrics::default();
        assert_eq!(m.context_pct(), 0.0);
    }

    #[test]
    fn hover_and_flash() {
        let mut s = AppState::new();
        s.hit_model = Some(HitRange {
            y: 10,
            x0: 20,
            x1: 40,
        });
        assert_eq!(s.hover_at(25, 10), HoverKind::Model);
        assert_eq!(s.hover_at(25, 9), HoverKind::None);
        assert!(s.set_hover(HoverKind::Model));
        assert!(!s.set_hover(HoverKind::Model));
        s.ping_click(HoverKind::Model);
        assert_eq!(s.flash_kind(), Some(HoverKind::Model));
        assert!(s.needs_animation());
        s.hit_tips_bar = Some(HitRange {
            y: 1,
            x0: 0,
            x1: 20,
        });
        assert_eq!(s.hover_at(4, 1), HoverKind::TipsBar);
        s.hit_tools.push((
            HitRange {
                y: 5,
                x0: 0,
                x1: 40,
            },
            "tool-1".into(),
        ));
        assert_eq!(s.hover_at(2, 5), HoverKind::Tool("tool-1".into()));
        s.hit_dock.push((
            HitRange {
                y: 12,
                x0: 0,
                x1: 80,
            },
            DockEntry::Agent(0),
        ));
        assert_eq!(s.hover_at(10, 12), HoverKind::Dock(DockEntry::Agent(0)));
    }

    #[test]
    fn tips_rotate_and_toggle() {
        let mut s = AppState::new();
        let first = s.tip_index;
        s.next_tip();
        assert_eq!(s.tip_index, (first + 1) % crate::tips::COUNT);
        s.set_tips_open(false);
        assert!(!s.tips_open);
        s.rotate_tip_if_due();
        assert_eq!(s.tip_index, (first + 1) % crate::tips::COUNT);
        s.set_tips_open(true);
        assert!(s.tips_open);
    }

    #[test]
    fn prompt_history_walk() {
        let mut h = PromptHistory::default();
        h.push("one".into());
        h.push("two".into());
        assert_eq!(h.prev("draft").as_deref(), Some("two"));
        assert_eq!(h.prev("two").as_deref(), Some("one"));
        assert_eq!(h.prev("one").as_deref(), Some("one"));
        assert_eq!(h.next().as_deref(), Some("two"));
        assert_eq!(h.next().as_deref(), Some("draft"));
        assert_eq!(h.next(), None);
    }

    #[test]
    fn queue_fifo() {
        let mut s = AppState::new();
        s.enqueue("first".into());
        s.enqueue("second".into());
        assert_eq!(s.take_queued().as_deref(), Some("first"));
        assert_eq!(s.take_queued().as_deref(), Some("second"));
        assert_eq!(s.take_queued(), None);
    }

    #[test]
    fn queue_edit_cycle_and_drop() {
        let mut s = AppState::new();
        s.enqueue("a".into());
        s.enqueue("b".into());
        s.enqueue("c".into());
        assert_eq!(s.cycle_queue(1).as_deref(), Some("a"));
        assert_eq!(s.queue_edit, Some(0));
        assert_eq!(s.cycle_queue(1).as_deref(), Some("b"));
        assert!(s.drop_queue_edit());
        assert_eq!(s.prompt_queue.len(), 2);
        assert_eq!(s.take_queue_edit(), None);
        s.queue_edit = Some(0);
        assert_eq!(s.take_queue_edit().as_deref(), Some("a"));
        assert_eq!(s.prompt_queue.len(), 1);
        s.enqueue("z".into());
        assert!(s.drop_queued(0));
        assert_eq!(s.prompt_queue.len(), 1);
    }

    #[test]
    fn parse_gateway_transcript() {
        let v = json!([
            { "role": "user", "text": "hi" },
            { "role": "assistant", "content": "hello" }
        ]);
        let msgs = parse_gateway_messages(v.as_array().unwrap());
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, MessageRole::User);
        assert_eq!(msgs[1].content, "hello");
    }

    #[test]
    fn tool_steps_are_numbered() {
        let mut s = AppState::new();
        s.start_turn("hi".into());
        s.messages.push(ChatMessage {
            id: "t".into(),
            role: MessageRole::Tool {
                name: "read_file".into(),
                status: "completed".into(),
                tool_id: Some("1".into()),
            },
            content: "src/main.rs".into(),
            timestamp: Local::now(),
            output: String::new(),
            is_streaming: false,
        });
        let steps = s.tool_steps();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].index, 1);
        assert_eq!(steps[0].name, "read_file");
        let prompt = resume_from_step_prompt(&steps[0], "src/app.rs");
        assert!(prompt.contains("step 1"));
        assert!(prompt.contains("src/app.rs"));
        assert!(prompt.contains("read_file"));
        s.split_trace = true;
        s.trace_focus = false;
        assert!(!s.want_trace(40), "narrow, unfocused, still hide");
        assert!(s.want_trace(100), "wide with tools");
        s.split_trace = false;
        assert!(!s.want_trace(100));
    }

    #[test]
    fn picker_filter_matches_provider_or_model() {
        let mut s = AppState::new();
        s.active_view = ActiveView::ModelPicker;
        s.providers = vec![
            ModelProvider {
                slug: "xai".into(),
                name: "xAI".into(),
                models: vec!["grok-4".into()],
                authenticated: true,
                is_current: true,
                warning: String::new(),
                auth_type: "api_key".into(),
                key_env: "XAI_API_KEY".into(),
            },
            ModelProvider {
                slug: "openai".into(),
                name: "OpenAI".into(),
                models: vec!["gpt-4.1".into()],
                authenticated: true,
                is_current: false,
                warning: String::new(),
                auth_type: "api_key".into(),
                key_env: "OPENAI_API_KEY".into(),
            },
        ];
        s.picker_filter = "grok".into();
        assert_eq!(s.filtered_provider_indices(), vec![0]);
        s.picker_filter = "openai".into();
        assert_eq!(s.filtered_provider_indices(), vec![1]);
        s.picker_provider = 1;
        s.picker_stage = PickerStage::Models;
        s.picker_filter = "gpt".into();
        assert_eq!(s.filtered_model_indices(), vec![0]);
        s.picker_filter = "nope".into();
        assert!(s.filtered_model_indices().is_empty());
    }

    #[test]
    fn background_picker_keeps_launch_row() {
        let mut s = AppState::new();
        s.open_background();
        assert_eq!(s.active_view, ActiveView::Background);
        assert_eq!(s.picker_len(), 1);
        s.start_bg_task("bg_a".into(), "summarize hn".into());
        s.start_bg_task("bg_b".into(), "lint the crate".into());
        assert_eq!(s.running_bg_count(), 2);
        assert_eq!(s.picker_len(), 3);
        s.picker_filter = "hn".into();
        assert_eq!(s.filtered_bg_indices(), vec![1]);
        assert_eq!(s.picker_len(), 2);
        s.complete_bg_task("bg_a", "top stories…");
        assert_eq!(s.running_bg_count(), 1);
        assert_eq!(s.bg_tasks[1].status, BgStatus::Done);
        assert!(s.bg_tasks[1].result.contains("top stories"));
        s.complete_bg_task("bg_b", "ok");
        assert_eq!(s.running_bg_count(), 0);
        s.picker_filter.clear();
        for i in 0..30 {
            s.start_bg_task(format!("bg_{i}"), "x".into());
            s.complete_bg_task(&format!("bg_{i}"), "ok");
        }
        assert!(s.bg_tasks.len() <= 24);
        assert_eq!(s.running_bg_count(), 0);
    }

    #[test]
    fn parse_model_providers_reads_string_ids() {
        let v = json!({
            "model": "grok-4",
            "provider": "xai",
            "providers": [
                {
                    "slug": "xai",
                    "name": "xAI",
                    "is_current": true,
                    "authenticated": true,
                    "models": ["grok-4", "grok-3"]
                },
                {
                    "slug": "openrouter",
                    "name": "OpenRouter",
                    "authenticated": false,
                    "auth_type": "api_key",
                    "key_env": "OPENROUTER_API_KEY",
                    "warning": "paste OPENROUTER_API_KEY",
                    "models": []
                }
            ]
        });
        let rows = parse_model_providers(&v);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].slug, "xai");
        assert_eq!(rows[0].models, vec!["grok-4", "grok-3"]);
        assert!(rows[0].is_current);
        assert!(!rows[1].authenticated);
        assert!(rows[1].warning.contains("OPENROUTER"));
        assert_eq!(rows[1].auth_type, "api_key");
        assert_eq!(rows[1].key_env, "OPENROUTER_API_KEY");
        assert!(rows[1].accepts_inline_key());
        assert!(!rows[0].accepts_inline_key());
    }

    #[test]
    fn apply_saved_provider_merges_and_opens_models() {
        let mut s = AppState::new();
        s.providers = vec![ModelProvider {
            slug: "openrouter".into(),
            name: "OpenRouter".into(),
            models: vec![],
            authenticated: false,
            is_current: false,
            warning: "paste OPENROUTER_API_KEY".into(),
            auth_type: "api_key".into(),
            key_env: "OPENROUTER_API_KEY".into(),
        }];
        s.open_provider_key(0);
        assert_eq!(s.picker_stage, PickerStage::Key);
        let saved = parse_saved_provider(&json!({
            "provider": {
                "slug": "openrouter",
                "name": "OpenRouter",
                "authenticated": true,
                "models": ["anthropic/claude-sonnet-4.6"]
            }
        }))
        .expect("saved provider");
        s.apply_saved_provider(saved);
        assert_eq!(s.picker_stage, PickerStage::Models);
        assert!(s.providers[0].authenticated);
        assert_eq!(s.providers[0].models.len(), 1);
        assert!(s.picker_key.is_empty());
        assert!(!s.picker_key_saving);
    }

    #[test]
    fn oauth_provider_rejects_inline_key() {
        let p = ModelProvider {
            slug: "nous".into(),
            name: "Nous".into(),
            models: vec![],
            authenticated: false,
            is_current: false,
            warning: "run hermes model".into(),
            auth_type: "oauth".into(),
            key_env: String::new(),
        };
        assert!(!p.accepts_inline_key());
    }

    #[test]
    fn parse_grouped_names_strips_tools_suffix() {
        let v = json!({
            "web_tools": ["web_search", "web_extract"],
            "files": ["read_file"]
        });
        let g = parse_grouped_names(&v);
        assert_eq!(g[0].0, "files");
        assert_eq!(g[1].0, "web");
        assert_eq!(g[1].1.len(), 2);
        assert_eq!(
            count_mcp_connected(&json!([
                { "connected": true },
                { "connected": false }
            ])),
            1
        );
        let mcp = parse_mcp_servers(
            &json!({
                "servers": [{
                    "name": "figma",
                    "transport": "sse",
                    "enabled": true,
                    "tools": ["a", "b"],
                    "url": "https://mcp.figma"
                }]
            }),
            &json!({
                "servers": [
                    { "name": "figma", "description": "dup" },
                    {
                        "name": "exa",
                        "description": "search",
                        "installed": false,
                        "enabled": false,
                        "transport": "http",
                        "requires": ["EXA_API_KEY"]
                    }
                ]
            }),
        );
        assert_eq!(mcp.len(), 2);
        assert!(mcp[0].configured);
        assert_eq!(mcp[0].tools, 2);
        assert_eq!(mcp[1].name, "exa");
        assert!(!mcp[1].configured);
        assert_eq!(mcp[1].requires, vec!["EXA_API_KEY"]);
        let tools = parse_toolsets(&json!({
            "toolsets": [{ "name": "web", "description": "search", "enabled": true, "tool_count": 3 }]
        }));
        assert_eq!(tools[0].name, "web");
        assert_eq!(tools[0].tool_count, 3);
        let plugs = parse_plugins(&json!({
            "plugins": [
                { "name": "demo", "version": "1.0", "enabled": false },
                { "name": "fal", "key": "image_gen/fal", "version": "2", "status": "enabled" }
            ]
        }));
        assert_eq!(plugs[0].name, "demo");
        assert_eq!(plugs[0].key, "demo");
        assert!(!plugs[0].enabled);
        assert_eq!(plugs[1].key, "image_gen/fal");
        assert!(plugs[1].enabled);
        let trees = parse_spawn_entries(&json!({
            "entries": [
                { "label": "research", "path": "/tmp/a.json", "count": 2 },
                { "label": "fix", "path": "/tmp/b.json", "count": 1 }
            ]
        }));
        assert_eq!(trees.len(), 2);
        assert_eq!(resolve_spawn_entry("1", &trees).unwrap().label, "research");
        assert_eq!(resolve_spawn_entry("b.json", &trees).unwrap().label, "fix");
        let snap = parse_spawn_tree_agents(&json!({
            "subagents": [
                { "id": "a1", "goal": "search", "status": "completed", "toolCount": 3, "parentId": "root" }
            ]
        }));
        assert_eq!(snap[0].title, "search");
        assert_eq!(snap[0].tool_count, 3);
        assert!(!snap[0].is_live());
        let prows = parse_project_rows(&json!({
            "projects": [{ "id": "p_ab", "label": "hermes-tui", "sessionCount": 2 }]
        }));
        assert_eq!(prows[0].count, 2);
        let sess = parse_project_session_records(&json!({
            "project": {
                "repos": [{ "groups": [{ "sessions": [{ "id": "s1", "title": "fix vim" }] }] }]
            }
        }));
        assert_eq!(sess[0].id, "s1");
        let diff = format_spawn_diff(
            &json!({
                "label": "research",
                "subagents": [
                    { "goal": "search", "toolCount": 3 },
                    { "goal": "read", "tool_count": 1 }
                ]
            }),
            &json!({
                "label": "fix",
                "subagents": [{ "goal": "patch", "toolCount": 2 }]
            }),
            "#1",
            "#2",
        );
        assert!(diff.contains("Δ  agents -1  tools -2"));
        assert!(diff.contains("search · read"));
        let cfg = format_tools_configure(
            &json!({
                "changed": ["web"],
                "unknown": ["nope"],
                "missing_servers": [],
                "reset": true
            }),
            "enable",
        );
        assert!(cfg.contains("enabled: web"));
        assert!(cfg.contains("session reset"));
        assert!(cfg.contains("unknown: nope"));
        let shown = format_tools_show(
            &json!({
                "sections": [
                    { "name": "web", "tools": [{ "name": "web_search", "description": "search" }] },
                    { "name": "files", "tools": [{ "name": "read_file" }] }
                ]
            }),
            Some("web"),
        );
        assert!(shown.contains("web_search"));
        assert!(!shown.contains("read_file"));
        let probe = format_mcp_test(
            &json!({ "ok": false, "error": "OAuth required", "oauth_needed": true }),
            "github",
        );
        assert!(probe.contains("hermes mcp login github"));
        let desc = format_profile_describe(&json!({
            "name": "work",
            "description": "day job",
            "soul": "be terse",
            "model": { "provider": "openai", "default": "gpt-5" },
            "skills": [1, 2],
            "toolsets": [1]
        }));
        assert!(desc.contains("work"));
        assert!(desc.contains("be terse"));
        let mut row = AgentRow::subagent("sa-1".into());
        row.title = "research".into();
        row.kind = "subagent".into();
        let payload = spawn_subagents_from_rows(&[row]);
        assert_eq!(payload[0]["goal"], "research");
        let cron = parse_cron_jobs(&json!({
            "jobs": [{
                "job_id": "abc",
                "name": "nightly",
                "schedule": "0 3 * * *",
                "enabled": true,
                "state": "active",
                "prompt_preview": "run tests"
            }]
        }));
        assert_eq!(cron[0].id, "abc");
        assert_eq!(cron[0].schedule, "0 3 * * *");
        let live = parse_live_sessions(&json!({
            "sessions": [{ "id": "live-1", "title": "now", "status": "working", "preview": "hi" }]
        }));
        assert!(live[0].live);
        let stored = parse_sessions(&[
            json!({ "id": "live-1", "title": "old" }),
            json!({ "id": "stored" }),
        ]);
        let merged = merge_session_lists(live, stored);
        assert_eq!(merged.len(), 2);
        assert!(merged[0].live);
        assert!(!merged[1].live);
        let tree = format_projects_tree(&json!({
            "active_id": "p1",
            "projects": [
                { "id": "p1", "name": "hermes-tui", "session_count": 4 },
                { "id": "p2", "label": "other", "count": 1 }
            ]
        }));
        assert!(tree.contains("● hermes-tui"));
        assert!(tree.contains("· other"));
        let camel = format_projects_tree(&json!({
            "projects": [{ "id": "p1", "label": "camel", "sessionCount": 2 }]
        }));
        assert!(camel.contains("camel  2 sessions"));
        assert_eq!(
            match_project_id(
                &json!({ "projects": [{ "id": "p_ab", "label": "hermes-tui" }] }),
                "hermes"
            )
            .as_deref(),
            Some("p_ab")
        );
        let drilled = format_project_sessions(&json!({
            "project": {
                "id": "p_ab",
                "label": "hermes-tui",
                "sessionCount": 1,
                "repos": [{
                    "label": "hermes-tui",
                    "groups": [{
                        "label": "main",
                        "sessions": [{ "id": "s1", "title": "fix vim" }]
                    }]
                }]
            }
        }));
        assert!(drilled.contains("fix vim  s1"));
        assert!(config_flag_on(&json!({ "value": true })));
        let bars = format_usage_bars(&json!({
            "available": true,
            "plan_name": "Pro",
            "status": "ok",
            "subscription_remaining_display": "$12.00",
            "total_spendable_display": "$12.00"
        }))
        .unwrap();
        assert!(bars.contains("Pro"));
        assert!(bars.contains("$12.00"));
        assert!(format_usage_bars(&json!({ "available": false })).is_none());
        assert!(!config_flag_on(&json!({ "value": "off" })));
        let skill = format_skill_inspect(&json!({
            "info": { "name": "x-content", "description": "write for X", "path": "/tmp/SKILL.md" }
        }));
        assert!(skill.contains("x-content"));
        assert!(skill.contains("write for X"));
    }

    #[test]
    fn parse_skills_flattens_category_map() {
        let cards = parse_skills_payload(&json!({
            "skills": {
                "general": ["x-content", "devgod"],
                "research": [{ "name": "xint", "description": "search X" }]
            }
        }));
        assert_eq!(cards.len(), 3);
        assert!(cards
            .iter()
            .any(|c| c.name == "x-content" && c.category == "general"));
        assert!(cards
            .iter()
            .any(|c| c.name == "xint" && c.description.contains("search")));
    }

    #[test]
    fn parse_profiles_reads_last_session() {
        let rows = parse_profiles(&json!({
            "profiles": [{
                "name": "default",
                "display_name": "Hermes",
                "model": "gpt-5.6-sol",
                "provider": "nous",
                "is_default": true,
                "skill_count": 45,
                "last_session": { "id": "abc", "title": "tui", "preview": "hi" },
                "worker_session": null
            }]
        }));
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].last_session_id.as_deref(), Some("abc"));
        assert!(!rows[0].worker_active);
    }

    #[test]
    fn parse_agents_and_memory_nodes() {
        let rows = parse_agent_rows(
            &json!({ "processes": [{ "session_id": "p1", "command": "sleep", "status": "running", "uptime": 12 }] }),
            &json!({ "active": [{ "subagent_id": "s1", "goal": "audit", "status": "running", "depth": 1, "model": "gpt" }] }),
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].kind, "subagent");
        assert_eq!(rows[0].depth, 1);
        assert_eq!(rows[1].kind, "process");
        assert_eq!(rows[1].id, "p1");
        let (sum, nodes) = parse_memory_payload(&json!({
            "summary": ["2 learned skills · 1 memories"],
            "buckets": [{
                "date": "2026-09-01",
                "nodes": [{ "id": "n1", "label": "note", "style": "memory", "meta": "today", "body": "keep" }]
            }]
        }));
        assert_eq!(sum.len(), 1);
        assert_eq!(nodes[0].label, "note");
        assert_eq!(nodes[0].body, "keep");
    }

    #[test]
    fn upsert_subagent_tree_and_tools() {
        let mut s = AppState::new();
        let spawn = json!({
            "subagent_id": "sa1",
            "goal": "audit src",
            "depth": 0,
            "task_index": 1,
            "model": "grok"
        });
        assert!(s.upsert_subagent(&spawn, Some("queued"), true));
        s.nudge_agents();
        assert!(s.agents_nudged);
        s.upsert_subagent(&spawn, Some("running"), true);
        s.push_agent_tool("sa1", "read_file(src/app.rs)");
        let child = json!({
            "subagent_id": "sa2",
            "goal": "write tests",
            "depth": 1,
            "parent_id": "sa1",
            "task_index": 2
        });
        s.upsert_subagent(&child, Some("running"), true);
        assert_eq!(s.running_agent_count(), 2);
        let ids = s.descendant_agent_ids("sa1");
        assert!(ids.contains(&"sa1".to_string()));
        assert!(ids.contains(&"sa2".to_string()));
        s.upsert_subagent(
            &json!({ "subagent_id": "sa1", "summary": "looks good", "status": "completed" }),
            Some("completed"),
            false,
        );
        assert_eq!(
            s.agent_rows.iter().find(|r| r.id == "sa1").unwrap().status,
            "completed"
        );
        assert_eq!(s.running_agent_count(), 1);
        s.start_turn("next".into());
        assert_eq!(s.running_agent_count(), 0);
        assert!(s.agent_rows.iter().all(|r| r.kind != "subagent"));
    }

    #[test]
    fn process_list_fields_and_output_cap() {
        let rows = parse_agent_rows(
            &json!({
                "processes": [{
                    "session_id": "proc_ab",
                    "command": "python -m http.server",
                    "status": "running",
                    "uptime_seconds": 9,
                    "pid": 4242,
                    "cwd": "/tmp/site",
                    "output_tail": "Serving HTTP on 0.0.0.0\n"
                }]
            }),
            &json!({}),
        );
        assert_eq!(rows.len(), 1);
        assert!(rows[0].is_running_process());
        assert_eq!(rows[0].pid, Some(4242));
        assert_eq!(rows[0].cwd, "/tmp/site");
        assert!(rows[0].output.contains("Serving HTTP"));
        let mut s = AppState::new();
        s.merge_agent_snapshot(
            &json!({ "processes": [{ "session_id": "proc_ab", "command": "python -m http.server", "status": "running" }] }),
            &json!({}),
        );
        s.append_process_output("proc_ab", "hello\n");
        assert!(s.agent_rows[0].output.contains("hello"));
        s.close_process("proc_ab");
        assert_eq!(s.agent_rows[0].status, "exited");
        assert_eq!(s.running_process_count(), 0);
    }
}
