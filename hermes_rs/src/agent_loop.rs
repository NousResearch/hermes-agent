//! Agent loop state machine.
//!
//! Manages the control flow for the AI agent conversation loop.
//! Rust handles state transitions, retry logic, and error classification.
//! Python does the actual I/O (API calls, tool execution, display).

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ── States ──────────────────────────────────────────────────────────────

/// Every state the agent loop can be in.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(eq, skip_from_py_object)]
pub enum LoopState {
    /// Check for user interrupt before starting iteration.
    CheckInterrupt,
    /// Build API request (messages, system prompt, tools).
    PrepareRequest,
    /// Execute API call (Python handles retries internally).
    ApiCall,
    /// Validate raw API response shape.
    ValidateResponse,
    /// Parse/normalize the response (codex vs standard).
    ParseResponse,
    /// Check for incomplete REASONING_SCRATCHPAD.
    CheckScratchpad,
    /// Run tool adapter for local models (<tool_call> XML → structured).
    AdaptToolCalls,
    /// Handle Codex incomplete response continuations.
    HandleCodexIncomplete,
    /// Validate tool call names and JSON arguments.
    ValidateToolCalls,
    /// Execute tool calls.
    ExecuteTools,
    /// Process final text response (strip think blocks, check empty).
    HandleFinalResponse,
    /// Handle errors during the loop.
    HandleError,
    /// Loop is done — build and return result.
    Complete,
}

// ── Actions ─────────────────────────────────────────────────────────────

/// What Python should do after receiving a state transition.
#[derive(Debug, Clone)]
#[pyclass]
pub enum Action {
    /// Continue to next state (no Python work needed).
    Continue,
    /// Python should execute the named phase and report back.
    Execute,
    /// Break out of loop — done or interrupted.
    Break,
    /// Retry current phase (e.g. API call retry, invalid tool retry).
    Retry,
    /// Inject a nudge message and re-enter the loop.
    Nudge,
    /// Return error result.
    Fail,
}

// ── Transition result ───────────────────────────────────────────────────

#[pyclass]
#[derive(Debug, Clone)]
pub struct Transition {
    #[pyo3(get)]
    pub state: LoopState,
    #[pyo3(get)]
    pub action: Action,
    #[pyo3(get)]
    pub message: String,
}

#[pymethods]
impl Transition {
    fn __repr__(&self) -> String {
        format!("Transition({:?}, {:?}, {:?})", self.state, self.action, self.message)
    }
}

// ── Response classification ─────────────────────────────────────────────

/// What kind of response did we get?
#[derive(Debug, Clone, PartialEq)]
#[pyclass(eq)]
pub enum ResponseKind {
    /// Normal text response, no tool calls.
    Text,
    /// Has structured tool_calls.
    ToolCalls,
    /// Response is None or has no choices.
    Invalid,
    /// finish_reason == "length" — truncated.
    Truncated,
    /// Codex incomplete status.
    CodexIncomplete,
    /// Has <tool_call> XML but parsing failed (truncated JSON).
    TruncatedToolCall,
    /// Only has <think> block with no content after.
    EmptyAfterThink,
    /// Incomplete REASONING_SCRATCHPAD.
    IncompleteScratchpad,
    /// Has tool_calls but names are invalid.
    InvalidToolNames,
    /// Has tool_calls but JSON args are invalid.
    InvalidToolJson,
}

// ── Main state machine ──────────────────────────────────────────────────

#[pyclass]
pub struct AgentLoopMachine {
    state: LoopState,
    iteration: u32,
    max_iterations: u32,
    interrupted: bool,
    needs_tool_adapter: bool,
    is_codex: bool,

    // Retry counters — exhaustive, no more hasattr
    api_retries: u32,
    max_api_retries: u32,
    invalid_tool_retries: u32,
    invalid_json_retries: u32,
    empty_content_retries: u32,
    truncated_tc_retries: u32,
    incomplete_scratchpad_retries: u32,
    codex_incomplete_retries: u32,
    codex_ack_continuations: u32,

    // Result tracking
    has_final_response: bool,
    completed: bool,
    error_message: String,
}

#[pymethods]
impl AgentLoopMachine {
    #[new]
    #[pyo3(signature = (max_iterations=60, needs_tool_adapter=false, is_codex=false, max_api_retries=6))]
    fn new(
        max_iterations: u32,
        needs_tool_adapter: bool,
        is_codex: bool,
        max_api_retries: u32,
    ) -> Self {
        Self {
            state: LoopState::CheckInterrupt,
            iteration: 0,
            max_iterations,
            interrupted: false,
            needs_tool_adapter,
            is_codex,
            api_retries: 0,
            max_api_retries,
            invalid_tool_retries: 0,
            invalid_json_retries: 0,
            empty_content_retries: 0,
            truncated_tc_retries: 0,
            incomplete_scratchpad_retries: 0,
            codex_incomplete_retries: 0,
            codex_ack_continuations: 0,
            has_final_response: false,
            completed: false,
            error_message: String::new(),
        }
    }

    /// Get the current state.
    #[getter]
    fn state(&self) -> LoopState {
        self.state.clone()
    }

    /// Current iteration count.
    #[getter]
    fn iteration(&self) -> u32 {
        self.iteration
    }

    /// Whether the loop was interrupted.
    #[getter]
    fn interrupted(&self) -> bool {
        self.interrupted
    }

    /// Whether the loop completed successfully.
    #[getter]
    fn completed(&self) -> bool {
        self.completed
    }

    /// Error message if any.
    #[getter]
    fn error_message(&self) -> String {
        self.error_message.clone()
    }

    /// Signal an interrupt.
    fn set_interrupted(&mut self) {
        self.interrupted = true;
    }

    /// Start a new iteration. Returns None if max iterations exceeded.
    fn begin_iteration(&mut self) -> Option<Transition> {
        if self.iteration >= self.max_iterations {
            self.state = LoopState::Complete;
            self.error_message = format!(
                "Max iterations ({}) exceeded",
                self.max_iterations
            );
            return Some(Transition {
                state: LoopState::Complete,
                action: Action::Break,
                message: self.error_message.clone(),
            });
        }
        self.iteration += 1;
        self.api_retries = 0;
        self.state = LoopState::CheckInterrupt;
        None
    }

    /// Drive the state machine forward based on the current state
    /// and the result from Python's execution of the previous state.
    ///
    /// `response_kind` classifies what Python got back.
    /// Returns the next Transition telling Python what to do.
    fn step(&mut self, response_kind: ResponseKind) -> Transition {
        match (&self.state, &response_kind) {
            // ── CheckInterrupt ──
            (LoopState::CheckInterrupt, _) => {
                if self.interrupted {
                    Transition {
                        state: LoopState::Complete,
                        action: Action::Break,
                        message: "Interrupted by user".into(),
                    }
                } else {
                    self.state = LoopState::PrepareRequest;
                    Transition {
                        state: LoopState::PrepareRequest,
                        action: Action::Execute,
                        message: String::new(),
                    }
                }
            }

            // ── PrepareRequest ──
            (LoopState::PrepareRequest, _) => {
                self.state = LoopState::ApiCall;
                Transition {
                    state: LoopState::ApiCall,
                    action: Action::Execute,
                    message: String::new(),
                }
            }

            // ── ApiCall ──
            (LoopState::ApiCall, ResponseKind::Invalid) => {
                self.api_retries += 1;
                if self.api_retries >= self.max_api_retries {
                    self.state = LoopState::Complete;
                    self.error_message = "Max API retries exceeded".into();
                    Transition {
                        state: LoopState::Complete,
                        action: Action::Fail,
                        message: self.error_message.clone(),
                    }
                } else {
                    Transition {
                        state: LoopState::ApiCall,
                        action: Action::Retry,
                        message: format!(
                            "API retry {}/{}",
                            self.api_retries, self.max_api_retries
                        ),
                    }
                }
            }
            (LoopState::ApiCall, _) => {
                self.api_retries = 0;
                self.state = LoopState::ValidateResponse;
                Transition {
                    state: LoopState::ValidateResponse,
                    action: Action::Continue,
                    message: String::new(),
                }
            }

            // ── ValidateResponse ──
            (LoopState::ValidateResponse, ResponseKind::Truncated) => {
                self.state = LoopState::Complete;
                self.error_message = "Response truncated (finish_reason=length)".into();
                Transition {
                    state: LoopState::Complete,
                    action: Action::Fail,
                    message: self.error_message.clone(),
                }
            }
            (LoopState::ValidateResponse, _) => {
                self.state = LoopState::ParseResponse;
                Transition {
                    state: LoopState::ParseResponse,
                    action: Action::Execute,
                    message: String::new(),
                }
            }

            // ── ParseResponse ──
            (LoopState::ParseResponse, ResponseKind::IncompleteScratchpad) => {
                self.state = LoopState::CheckScratchpad;
                Transition {
                    state: LoopState::CheckScratchpad,
                    action: Action::Continue,
                    message: String::new(),
                }
            }
            (LoopState::ParseResponse, _) => {
                if self.needs_tool_adapter {
                    self.state = LoopState::AdaptToolCalls;
                    Transition {
                        state: LoopState::AdaptToolCalls,
                        action: Action::Execute,
                        message: String::new(),
                    }
                } else if self.is_codex && response_kind == ResponseKind::CodexIncomplete {
                    self.state = LoopState::HandleCodexIncomplete;
                    Transition {
                        state: LoopState::HandleCodexIncomplete,
                        action: Action::Continue,
                        message: String::new(),
                    }
                } else {
                    self.state = LoopState::ValidateToolCalls;
                    Transition {
                        state: LoopState::ValidateToolCalls,
                        action: Action::Execute,
                        message: String::new(),
                    }
                }
            }

            // ── CheckScratchpad ──
            (LoopState::CheckScratchpad, _) => {
                self.incomplete_scratchpad_retries += 1;
                if self.incomplete_scratchpad_retries <= 2 {
                    // Retry — go back to PrepareRequest
                    self.state = LoopState::PrepareRequest;
                    Transition {
                        state: LoopState::PrepareRequest,
                        action: Action::Retry,
                        message: format!(
                            "Incomplete scratchpad retry {}/2",
                            self.incomplete_scratchpad_retries
                        ),
                    }
                } else {
                    self.incomplete_scratchpad_retries = 0;
                    self.state = LoopState::Complete;
                    self.error_message = "Incomplete REASONING_SCRATCHPAD after 2 retries".into();
                    Transition {
                        state: LoopState::Complete,
                        action: Action::Fail,
                        message: self.error_message.clone(),
                    }
                }
            }

            // ── AdaptToolCalls ──
            (LoopState::AdaptToolCalls, ResponseKind::TruncatedToolCall) => {
                self.truncated_tc_retries += 1;
                if self.truncated_tc_retries < 3 {
                    self.state = LoopState::PrepareRequest;
                    Transition {
                        state: LoopState::PrepareRequest,
                        action: Action::Nudge,
                        message: format!(
                            "Truncated tool call, nudging model ({}/3)",
                            self.truncated_tc_retries
                        ),
                    }
                } else {
                    self.truncated_tc_retries = 0;
                    // Fall through to final response with stripped content
                    self.state = LoopState::HandleFinalResponse;
                    Transition {
                        state: LoopState::HandleFinalResponse,
                        action: Action::Execute,
                        message: "Truncated tool call after 3 retries, treating as text".into(),
                    }
                }
            }
            (LoopState::AdaptToolCalls, ResponseKind::ToolCalls) => {
                self.truncated_tc_retries = 0;
                self.state = LoopState::ValidateToolCalls;
                Transition {
                    state: LoopState::ValidateToolCalls,
                    action: Action::Execute,
                    message: String::new(),
                }
            }
            (LoopState::AdaptToolCalls, _) => {
                // No tool calls found (text response)
                self.truncated_tc_retries = 0;
                if self.is_codex && response_kind == ResponseKind::CodexIncomplete {
                    self.state = LoopState::HandleCodexIncomplete;
                    Transition {
                        state: LoopState::HandleCodexIncomplete,
                        action: Action::Continue,
                        message: String::new(),
                    }
                } else {
                    self.state = LoopState::HandleFinalResponse;
                    Transition {
                        state: LoopState::HandleFinalResponse,
                        action: Action::Execute,
                        message: String::new(),
                    }
                }
            }

            // ── HandleCodexIncomplete ──
            (LoopState::HandleCodexIncomplete, _) => {
                self.codex_incomplete_retries += 1;
                if self.codex_incomplete_retries < 3 {
                    self.state = LoopState::PrepareRequest;
                    Transition {
                        state: LoopState::PrepareRequest,
                        action: Action::Retry,
                        message: format!(
                            "Codex incomplete, continuing ({}/3)",
                            self.codex_incomplete_retries
                        ),
                    }
                } else {
                    self.codex_incomplete_retries = 0;
                    self.state = LoopState::Complete;
                    self.error_message =
                        "Codex response remained incomplete after 3 attempts".into();
                    Transition {
                        state: LoopState::Complete,
                        action: Action::Fail,
                        message: self.error_message.clone(),
                    }
                }
            }

            // ── ValidateToolCalls ──
            (LoopState::ValidateToolCalls, ResponseKind::InvalidToolNames) => {
                self.invalid_tool_retries += 1;
                if self.invalid_tool_retries < 3 {
                    self.state = LoopState::PrepareRequest;
                    Transition {
                        state: LoopState::PrepareRequest,
                        action: Action::Retry,
                        message: format!(
                            "Invalid tool name, retrying ({}/3)",
                            self.invalid_tool_retries
                        ),
                    }
                } else {
                    self.invalid_tool_retries = 0;
                    self.state = LoopState::Complete;
                    self.error_message = "Max retries for invalid tool names".into();
                    Transition {
                        state: LoopState::Complete,
                        action: Action::Fail,
                        message: self.error_message.clone(),
                    }
                }
            }
            (LoopState::ValidateToolCalls, ResponseKind::InvalidToolJson) => {
                self.invalid_json_retries += 1;
                if self.invalid_json_retries < 3 {
                    self.state = LoopState::PrepareRequest;
                    Transition {
                        state: LoopState::PrepareRequest,
                        action: Action::Retry,
                        message: format!(
                            "Invalid JSON args, retrying ({}/3)",
                            self.invalid_json_retries
                        ),
                    }
                } else {
                    self.invalid_json_retries = 0;
                    self.state = LoopState::PrepareRequest;
                    Transition {
                        state: LoopState::PrepareRequest,
                        action: Action::Nudge,
                        message: "Invalid JSON after 3 retries, nudging model".into(),
                    }
                }
            }
            (LoopState::ValidateToolCalls, ResponseKind::ToolCalls) => {
                self.invalid_tool_retries = 0;
                self.invalid_json_retries = 0;
                self.state = LoopState::ExecuteTools;
                Transition {
                    state: LoopState::ExecuteTools,
                    action: Action::Execute,
                    message: String::new(),
                }
            }
            (LoopState::ValidateToolCalls, ResponseKind::Text) => {
                // No tool calls — go to final response
                self.state = LoopState::HandleFinalResponse;
                Transition {
                    state: LoopState::HandleFinalResponse,
                    action: Action::Execute,
                    message: String::new(),
                }
            }
            (LoopState::ValidateToolCalls, _) => {
                self.state = LoopState::HandleFinalResponse;
                Transition {
                    state: LoopState::HandleFinalResponse,
                    action: Action::Execute,
                    message: String::new(),
                }
            }

            // ── ExecuteTools ──
            (LoopState::ExecuteTools, _) => {
                // After tool execution, go back to next iteration
                self.state = LoopState::CheckInterrupt;
                Transition {
                    state: LoopState::CheckInterrupt,
                    action: Action::Continue,
                    message: String::new(),
                }
            }

            // ── HandleFinalResponse ──
            (LoopState::HandleFinalResponse, ResponseKind::EmptyAfterThink) => {
                self.empty_content_retries += 1;
                if self.empty_content_retries < 3 {
                    self.state = LoopState::PrepareRequest;
                    Transition {
                        state: LoopState::PrepareRequest,
                        action: Action::Nudge,
                        message: format!(
                            "Empty after think block, nudging ({}/3)",
                            self.empty_content_retries
                        ),
                    }
                } else {
                    self.empty_content_retries = 0;
                    self.state = LoopState::Complete;
                    self.error_message =
                        "Model generated only think blocks after 3 retries".into();
                    Transition {
                        state: LoopState::Complete,
                        action: Action::Fail,
                        message: self.error_message.clone(),
                    }
                }
            }
            (LoopState::HandleFinalResponse, ResponseKind::Text) => {
                self.empty_content_retries = 0;
                self.has_final_response = true;
                self.completed = true;
                self.state = LoopState::Complete;
                Transition {
                    state: LoopState::Complete,
                    action: Action::Break,
                    message: "Final response received".into(),
                }
            }
            (LoopState::HandleFinalResponse, _) => {
                // Fallback — treat as text
                self.has_final_response = true;
                self.completed = true;
                self.state = LoopState::Complete;
                Transition {
                    state: LoopState::Complete,
                    action: Action::Break,
                    message: "Final response (fallback)".into(),
                }
            }

            // ── HandleError ──
            (LoopState::HandleError, _) => {
                self.state = LoopState::Complete;
                Transition {
                    state: LoopState::Complete,
                    action: Action::Fail,
                    message: self.error_message.clone(),
                }
            }

            // ── Complete ──
            (LoopState::Complete, _) => Transition {
                state: LoopState::Complete,
                action: Action::Break,
                message: "Loop complete".into(),
            },
        }
    }

    /// Classify a response content string for the state machine.
    /// Called from Python to determine `ResponseKind` without
    /// Python needing to know the classification logic.
    #[staticmethod]
    fn classify_content(
        content: &str,
        has_tool_calls: bool,
        finish_reason: &str,
        is_codex: bool,
        has_tool_call_tag: bool,
        has_tool_call_close_tag: bool,
        has_incomplete_scratchpad: bool,
        tool_names_valid: bool,
        tool_json_valid: bool,
    ) -> ResponseKind {
        if has_incomplete_scratchpad {
            return ResponseKind::IncompleteScratchpad;
        }
        if finish_reason == "length" {
            return ResponseKind::Truncated;
        }
        if is_codex && finish_reason == "incomplete" {
            return ResponseKind::CodexIncomplete;
        }
        if has_tool_calls {
            if !tool_names_valid {
                return ResponseKind::InvalidToolNames;
            }
            if !tool_json_valid {
                return ResponseKind::InvalidToolJson;
            }
            return ResponseKind::ToolCalls;
        }
        // Truncated tool call: has opening tag but no closing (or parsing failed)
        if has_tool_call_tag && !has_tool_call_close_tag {
            return ResponseKind::TruncatedToolCall;
        }
        if has_tool_call_tag && has_tool_call_close_tag && !has_tool_calls {
            // Has complete-looking tags but parsing failed
            return ResponseKind::TruncatedToolCall;
        }
        // Check for empty-after-think
        let stripped = strip_think_blocks(content);
        if stripped.trim().is_empty() && !content.trim().is_empty() {
            return ResponseKind::EmptyAfterThink;
        }
        ResponseKind::Text
    }

    /// Reset all retry counters. Call when starting a new conversation turn.
    fn reset(&mut self) {
        self.state = LoopState::CheckInterrupt;
        self.iteration = 0;
        self.interrupted = false;
        self.api_retries = 0;
        self.invalid_tool_retries = 0;
        self.invalid_json_retries = 0;
        self.empty_content_retries = 0;
        self.truncated_tc_retries = 0;
        self.incomplete_scratchpad_retries = 0;
        self.codex_incomplete_retries = 0;
        self.codex_ack_continuations = 0;
        self.has_final_response = false;
        self.completed = false;
        self.error_message.clear();
    }

    /// Get a snapshot of all retry counters for debugging.
    fn debug_counters(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("iteration", self.iteration)?;
        d.set_item("api_retries", self.api_retries)?;
        d.set_item("invalid_tool_retries", self.invalid_tool_retries)?;
        d.set_item("invalid_json_retries", self.invalid_json_retries)?;
        d.set_item("empty_content_retries", self.empty_content_retries)?;
        d.set_item("truncated_tc_retries", self.truncated_tc_retries)?;
        d.set_item("incomplete_scratchpad_retries", self.incomplete_scratchpad_retries)?;
        d.set_item("codex_incomplete_retries", self.codex_incomplete_retries)?;
        Ok(d.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentLoopMachine(state={:?}, iter={}/{}, interrupted={})",
            self.state, self.iteration, self.max_iterations, self.interrupted
        )
    }
}

/// Strip <think>...</think> blocks from content (Rust-accelerated).
/// Handles: closed blocks, unclosed blocks, orphaned </think>.
#[pyfunction]
pub fn strip_think_blocks(content: &str) -> String {
    use once_cell::sync::Lazy;
    use regex::Regex;

    static RE_CLOSED: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?s)<think>.*?</think>\s*").unwrap());
    static RE_UNCLOSED: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?s)<think>.*").unwrap());
    static RE_ORPHANED: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?s)^.*?</think>\s*").unwrap());

    let result = RE_CLOSED.replace_all(content, "");
    let result = RE_UNCLOSED.replace_all(&result, "");
    let result = RE_ORPHANED.replace_all(&result, "");
    result.trim().to_string()
}

/// Strip <tool_call>...</tool_call> blocks and unclosed <tool_call> tags.
#[pyfunction]
pub fn strip_tool_call_blocks(content: &str) -> String {
    use once_cell::sync::Lazy;
    use regex::Regex;

    static RE_COMPLETE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?s)<tool_call>.*?</tool_call>\s*").unwrap());
    static RE_UNCLOSED: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?s)<tool_call>.*").unwrap());

    let result = RE_COMPLETE.replace_all(content, "");
    let result = RE_UNCLOSED.replace_all(&result, "");
    result.trim().to_string()
}

/// Check if content has meaningful text after stripping think blocks.
#[pyfunction]
pub fn has_content_after_think(content: &str) -> bool {
    let stripped = strip_think_blocks(content);
    !stripped.trim().is_empty()
}
