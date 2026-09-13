"""HermesCLI constructor phases: display options, model/provider routing, turn limits, toolsets, checkpoints, prompt/reasoning, runtime state, session store and UI state.

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from datetime import datetime
from hermes_cli.fallback_config import get_fallback_chain
from hermes_state_ids import new_session_id
from pathlib import Path
from rich.console import Console
from typing import Any, Dict, List, Optional
from utils import base_url_host_matches, base_url_hostname, is_truthy_value

# Log-record parity with the origin module.
logger = logging.getLogger("cli")


class CLIInitMixin:
    """HermesCLI constructor phases: display options, model/provider routing, turn limits, toolsets, checkpoints, prompt/reasoning, runtime state, session store and UI state."""

    def _init_display_options(self, verbose, compact):
        """Display-related config: compact/tool-progress/focus view, bells, streaming, previews, stream buffers."""
        from cli import CLI_CONFIG, _configure_output_history, _int_or
        self.console = Console()
        self.config = CLI_CONFIG
        display = CLI_CONFIG["display"]
        self.compact = compact if compact is not None else display.get("compact", False)
        # tool_progress: "off" | "new" | "all" | "verbose"; YAML 1.1 parses bare `off` as False.
        _raw_tp = display.get("tool_progress", "all")
        self.tool_progress_mode = "off" if _raw_tp is False else str(_raw_tp)
        # focus_view (/focus) is display-only: snaps tool_progress to "off" (stashing the
        # pre-focus mode for /focus off); never changes what is sent to the model.
        self._focus_view_enabled = bool(display.get("focus_view", False))
        self._focus_saved_tool_progress = self._focus_last_counted_tool = None
        self._focus_hidden_lines = 0
        if self._focus_view_enabled:
            from hermes_cli.focus_view import FOCUS_TOOL_PROGRESS_MODE, normalize_tool_progress_mode

            self._focus_saved_tool_progress = normalize_tool_progress_mode(self.tool_progress_mode)
            self.tool_progress_mode = FOCUS_TOOL_PROGRESS_MODE
        self.resume_display = display.get("resume_display", "full")  # "full" | "minimal"
        self.bell_on_complete = display.get("bell_on_complete", False)
        self.bell_on_prompt = display.get("bell_on_prompt", False)  # bell when a blocking modal opens
        self.show_reasoning = display.get("show_reasoning", True)
        self.reasoning_full = display.get("reasoning_full", False)
        _configure_output_history(
            enabled=display.get("persistent_output", True),
            max_lines=display.get("persistent_output_max_lines", 200),
        )
        # busy_input_mode: "interrupt" (redirect the run) | "queue" (next turn) | "steer" (inject mid-run).
        _bim = str(display.get("busy_input_mode", "interrupt")).strip().lower()
        self.busy_input_mode = _bim if _bim in ("queue", "steer") else "interrupt"

        # verbose ONLY controls global DEBUG logging; tool_progress="verbose" is independent
        # (coupling them spewed every module's DEBUG logs to the console).
        self.verbose = bool(verbose) if verbose is not None else False

        self.streaming_enabled = display.get("streaming", False)
        self.show_timestamps = display.get("timestamps", False)
        self.timestamp_format = display.get("timestamp_format", "%H:%M")
        _frm = str(display.get("final_response_markdown", "strip")).strip().lower()
        self.final_response_markdown = _frm if _frm in {"render", "strip", "raw"} else "strip"

        self._inline_diffs_enabled = display.get("inline_diffs", True)

        # Per-turn accounting: CLI-only chrome riding the tool-progress feed.
        self._turn_summary_enabled = bool(display.get("turn_summary", True))
        _ttm = str(display.get("turn_timing", "total") or "total").strip().lower()
        self._turn_timing_mode = _ttm if _ttm in {"off", "total", "split", "verbose"} else "total"
        self._spinner_token_flow_enabled = bool(display.get("spinner_token_flow", True))
        self._turn_summary_collector = None
        self._turn_summary_start = 0.0
        self._turn_token_baseline = 0
        self._interactive_turn = False  # only run()-loop turns; keeps the summary line off -Q

        _ump = display.get("user_message_preview", {})
        _ump = _ump if isinstance(_ump, dict) else {}
        self.user_message_preview_first_lines = max(1, _int_or(_ump.get("first_lines", 2), 2))
        self.user_message_preview_last_lines = max(0, _int_or(_ump.get("last_lines", 2), 2))

        # Streaming display state
        self._stream_buf = ""  # partial line buffer
        self._reasoning_preview_buf = ""  # coalesces tiny reasoning chunks
        self._stream_started = self._stream_box_opened = self._stream_box_live = False
        self._held_status_lines: list[str] = []  # agent status lines parked while a box streams
        # Possible markdown-table lines held until the block ends for wcwidth-aware re-padding.
        self._stream_table_buf: list[str] = []
        self._in_stream_table = False
        self._pending_edit_snapshots = {}
        self._last_input_mode_recovery = self._last_termios_drift_check = None  # None = never; monotonic epoch is arbitrary
        self._input_mode_recovery_notice_shown = self._termios_drift_notice_shown = False

    def _init_model_routing(self, model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget, checkpoints, pass_session_id, ignore_rules):
        """Resolve model/provider/base_url, turn limits, toolsets, checkpoints, prompt/personality, reasoning + routing config."""
        self._init_model_and_provider(model, provider, api_key, base_url)
        self._init_turn_limits(max_turns, run_budget)
        self._init_toolsets(toolsets)
        self._init_checkpoints_and_rules(checkpoints, pass_session_id, ignore_rules)
        self._init_prompt_and_reasoning(reasoning)

    def _init_model_and_provider(self, model, provider, api_key, base_url):
        """Priority: CLI args > env vars > config file."""
        from cli import CLI_CONFIG, _normalize_moa_model, _split_model_config_default
        # LLM_MODEL/OPENAI_MODEL env vars are deliberately NOT checked (multi-agent setups
        # would stomp each other through the environment).
        _model_config = CLI_CONFIG["model"]
        # A dict-valued default carries its own provider, which must feed requested_provider
        # instead of being replaced by the merged model.provider (typically "auto").
        _config_model, _nested_provider = _split_model_config_default(
            _model_config.get("default") or _model_config.get("model") or ""
        )
        # resume must not clobber an explicit -m with the session's stored model.
        self._explicit_model_override = bool(model)
        self.model = model or _config_model or ""
        _cfg_provider = _model_config.get("provider") or os.getenv("HERMES_INFERENCE_PROVIDER")
        _startup_provider_override = _startup_base_url_override = _startup_api_key_override = ""
        if self.model:
            from hermes_cli.model_switch import resolve_startup_model_route

            _startup_route = resolve_startup_model_route(
                self.model,
                explicit_provider=provider or "",
                current_provider=(provider or _nested_provider or _cfg_provider or ""),
                user_providers=CLI_CONFIG.get("providers"),
                custom_providers=CLI_CONFIG.get("custom_providers"),
            )
            if _startup_route is not None:
                self.model = _startup_route.model
                _startup_provider_override = _startup_route.provider
                _startup_base_url_override = _startup_route.base_url
                _startup_api_key_override = _startup_route.api_key
        # ``moa:<preset>`` selects the MoA virtual provider before provider resolution so the
        # real provider never sees the unknown model; the prefix wins over --provider.
        # A ``moa:<preset>`` model string selects the MoA virtual provider in one shot (parity with
        # interactive ``/moa`` and the model picker). See #56828.
        _moa_provider_override, self.model = _normalize_moa_model(self.model)

        if self.model == "":  # auto-detect from a local server
            _base_url = _model_config.get("base_url") or ""
            if base_url_hostname(_base_url) in ("localhost", "127.0.0.1"):
                from hermes_cli.runtime_provider import _auto_detect_local_model
                self.model = _auto_detect_local_model(_base_url) or self.model
        # Provider normalisation may silently override the default but must warn for an
        # explicit choice (a config model equal to the global fallback is NOT explicit).
        self._model_is_default = not model and not _config_model

        # --api-key wins; otherwise a URL-bearing startup alias carries its own credential.
        # See #28660.
        self._explicit_api_key = api_key or _startup_api_key_override or None
        self._explicit_base_url = base_url or _startup_base_url_override or None

        # Resolved lazily at use-time via _ensure_runtime_credentials().
        self.requested_provider = (
            _moa_provider_override or provider or _startup_provider_override or _nested_provider
            or _cfg_provider or "auto"
        )
        # `--provider <custom>` without `-m` uses that entry's default_model, else the global
        # default goes to the custom endpoint and the compressor gets the wrong context length.
        # Explicit `-m` still wins. See #86978.
        if not model and provider:
            try:
                from hermes_cli.runtime_provider import _get_named_custom_provider

                _named_custom = _get_named_custom_provider(provider)
            except Exception as exc:
                logger.warning(
                    "Could not resolve --provider %s default model; keeping global model.default (%s)",
                    provider, exc,
                )
                _named_custom = None
            _provider_default = str((_named_custom or {}).get("model") or "").strip()
            if _provider_default:
                self.model = _provider_default
                self._model_is_default = False
        self._provider_source: Optional[str] = None
        self.provider = self.requested_provider
        self.api_mode = "chat_completions"
        self.acp_command: Optional[str] = None
        self.acp_args: list[str] = []
        self.base_url = (
            base_url or _startup_base_url_override or _model_config.get("base_url", "")
            or os.getenv("OPENROUTER_BASE_URL", "")
        ) or None
        # Key matches the resolved base_url; re-resolved by _ensure_runtime_credentials().
        _keys = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")
        if not (self.base_url and base_url_host_matches(self.base_url, "openrouter.ai")):
            _keys = _keys[::-1]
        self.api_key = api_key or os.getenv(_keys[0]) or os.getenv(_keys[1])

    def _init_turn_limits(self, max_turns, run_budget):
        """max_turns: CLI arg > config > env var > default; run budget: CLI flag > config."""
        from cli import CLI_CONFIG
        # resolve_turn_limit() accepts "none"/"unlimited" (-> sys.maxsize) alongside ints.
        # KEEP the root-level CLI_CONFIG["max_turns"] fallback: it is never migrated on disk
        # and other config paths may bypass the load-time fold.
        from hermes_cli.config import resolve_turn_limit as _resolve_turn_limit
        self.max_turns = _resolve_turn_limit(next(
            (v for v in (max_turns, CLI_CONFIG["agent"].get("max_turns"), CLI_CONFIG.get("max_turns")) if v is not None),
            os.getenv("HERMES_MAX_ITERATIONS"),
        ))
        self.run_budget_seconds = run_budget if run_budget is not None else CLI_CONFIG["agent"].get("run_budget_seconds")

    def _init_toolsets(self, toolsets):
        from cli import CLI_CONFIG, validate_toolset
        self.enabled_toolsets = toolsets
        from agent.skill_utils import parse_config_string_list

        self.disabled_toolsets = parse_config_string_list(CLI_CONFIG["agent"].get("disabled_toolsets"))

        if toolsets and "all" not in toolsets and "*" not in toolsets:
            # MCP server names only resolve after discover_mcp_tools runs; skip them here.
            mcp_names = set((CLI_CONFIG.get("mcp_servers") or {}).keys())
            invalid = [t for t in toolsets if not validate_toolset(t) and t not in mcp_names]
            if invalid:
                self._console_print(f"[bold red]Warning: Unknown toolsets: {', '.join(invalid)}[/]")

    def _init_checkpoints_and_rules(self, checkpoints, pass_session_id, ignore_rules):
        from cli import CLI_CONFIG
        cp_cfg = CLI_CONFIG.get("checkpoints", {})
        if isinstance(cp_cfg, bool):
            cp_cfg = {"enabled": cp_cfg}
        self.checkpoints_enabled = checkpoints or cp_cfg.get("enabled", False)
        self.checkpoint_max_snapshots = cp_cfg.get("max_snapshots", 20)
        self.checkpoint_max_total_size_mb = cp_cfg.get("max_total_size_mb", 500)
        self.checkpoint_max_file_size_mb = cp_cfg.get("max_file_size_mb", 10)
        self.pass_session_id = pass_session_id
        # --ignore-rules: AIAgent skips context files (AGENTS.md/SOUL.md/...) and memory.
        self.ignore_rules = ignore_rules or is_truthy_value(os.environ.get("HERMES_IGNORE_RULES"))

    def _init_prompt_and_reasoning(self, reasoning):
        """Ephemeral system prompt/prefill, reasoning + service tier, OpenRouter routing knobs, fallback chain."""
        from cli import CLI_CONFIG, _load_prefill_messages, _parse_reasoning_config, _parse_service_tier_config, _resolve_prefill_messages_file
        # Env var wins, then hermes_cli.personality (single owner of overlay resolution).
        from hermes_cli.personality import available_personalities, resolve_ephemeral_system_prompt

        self.system_prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "") or resolve_ephemeral_system_prompt(CLI_CONFIG)
        self.personalities = available_personalities(CLI_CONFIG)

        self.prefill_messages = _load_prefill_messages(_resolve_prefill_messages_file(CLI_CONFIG))

        # Per-model override > global reasoning_effort.
        # Reasoning config (OpenRouter reasoning effort level) Per-model override > global reasoning_effort
        # — resolved through the shared chokepoint in hermes_constants (Closes #21256).
        from hermes_constants import resolve_reasoning_config
        self.reasoning_config = resolve_reasoning_config(CLI_CONFIG, self.model)
        self._explicit_reasoning_config = None
        # --reasoning wins for this run only (never persisted); unparseable -> warn and ignore.
        if reasoning is not None and str(reasoning).strip():
            _cli_reasoning = _parse_reasoning_config(reasoning)
            if _cli_reasoning is None:
                logger.warning("Unknown --reasoning '%s', keeping the configured level", reasoning)
            else:
                self.reasoning_config = _cli_reasoning
                self._explicit_reasoning_config = _cli_reasoning
        self.service_tier = _parse_service_tier_config(CLI_CONFIG["agent"].get("service_tier", ""))

        pr = CLI_CONFIG.get("provider_routing", {}) or {}
        self._provider_sort = pr.get("sort")
        self._providers_only = pr.get("only")
        self._providers_ignore = pr.get("ignore")
        self._providers_order = pr.get("order")
        self._provider_require_params = pr.get("require_parameters", False)
        self._provider_data_collection = pr.get("data_collection")

        # OpenRouter Pareto Code router coding-score floor; out-of-range = unset.
        _raw_score = (CLI_CONFIG.get("openrouter", {}) or {}).get("min_coding_score")
        self._openrouter_min_coding_score: Optional[float] = None
        if _raw_score not in {None, ""}:
            try:
                _f = float(_raw_score)
                if 0.0 <= _f <= 1.0:
                    self._openrouter_min_coding_score = _f
            except (TypeError, ValueError):
                pass

        self._fallback_model = get_fallback_chain(CLI_CONFIG)

    def _init_runtime_state(self, resume):
        """Session store + all per-run mutable state (queues, overlays, pet/voice/status-bar fields)."""
        from cli import _hermes_home
        # A signature change across turns (/model, credential rotation) rebuilds the agent.
        self._active_agent_route_signature = None
        self.agent: Optional[Any] = None  # initialized on first use
        self._tool_callbacks_installed = self._tirith_security_checked = False
        self._app = None  # prompt_toolkit Application (set in run())

        self.conversation_history: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        # Per-prompt elapsed timer shown in the status bar.
        self._prompt_start_time: Optional[float] = None
        self._prompt_duration: float = 0.0
        self._last_turn_finished_at: Optional[float] = None
        self._init_session_store()
        self._pending_title: Optional[str] = None
        self._resumed = bool(resume)
        self.session_id = resume or new_session_id(self.session_start)
        getattr(self, "_write_terminal_breadcrumb", lambda: None)()

        self._history_file = _hermes_home / ".hermes_history"
        self._last_invalidate: float | None = None  # throttles UI repaints (None = never; monotonic epoch is arbitrary)
        self._init_ui_state()

    def _init_session_store(self):
        """Open the session store early (so /title works before the first message) + opportunistic maintenance."""
        from cli import _run_checkpoint_auto_maintenance, _run_state_db_auto_maintenance
        self._session_db = None
        self._session_db_unavailable = False
        try:
            # Registry handle, not a bare SessionDB(): goals/loops/heartbeat acquire the same
            # path a moment later from the REPL thread, and a second writer repeats the full
            # open (the /proc-wide deleted-WAL scan, ~4k readlinks) while the render thread
            # holds the GIL — that repeat was the post-banner freeze before the first prompt.
            from hermes_state_registry import acquire
            self._session_db = acquire()
        except Exception as e:
            # Without a store the transcript is NOT persisted while the chat looks healthy,
            # so surface it prominently rather than only logging.
            # #41386: a failed session store means the transcript is NOT persisted to state.db — the live
            # chat looks healthy but resume later shows a truncated/empty session. A buried log line is not
            # enough; surface it prominently so the user knows persistence is off for this run and can fix
            # the store before relying on resume.
            self._session_db_unavailable = True
            logger.warning("Failed to initialize SessionDB — session will NOT be indexed for search: %s", e)
            from hermes_state_user_copy import describe_storage_failure, storage_failure_details
            failure = describe_storage_failure(e)
            def _present_store_warning():
                try:
                    Console(stderr=True).print(
                        "[bold yellow]⚠ Session store unavailable[/bold yellow] — "
                        "this conversation will [bold]NOT be saved[/bold] and cannot be resumed later. "
                        "Searching past sessions is also disabled.\n"
                        f"  Reason: {failure.gloss}.\n"
                        f"  {failure.action}\n"
                        f"  [dim]Details: {storage_failure_details(e)}[/dim]"
                    )
                except Exception:
                    print(
                        "WARNING: Session store unavailable — this conversation will NOT be "
                        f"saved and cannot be resumed later. Reason: {failure.gloss}. {failure.action}"
                    )
            # Same automatic diagnostic the gateway gates for its home channel (run_notifications).
            from gateway.warning_notifications import render_notification
            render_notification(_present_store_warning, platform="cli")
        _run_state_db_auto_maintenance(self._session_db)
        _run_checkpoint_auto_maintenance()

    def _init_ui_state(self):
        """Per-run mutable UI state; must exist before any chat() call since -q never goes through run()."""
        from cli import CLI_CONFIG, _status_bar_visible_from_display_config
        self._pending_input = queue.Queue()
        self._interrupt_queue = queue.Queue()
        self._agent_running = self._should_exit = False
        self._last_turn_interrupted = False  # /goal never auto-queues on a Ctrl+C'd turn
        self._terminal_io_broken = False  # stdout EIO: freeze UI paints instead of spinning
        self._delete_session_on_exit = False  # /exit --delete
        # /update: relaunch() runs from run() after prompt_toolkit restored terminal modes.
        # /exit --delete: when True, the current session's SQLite history and on-disk transcripts are
        # deleted during shutdown. Set by process_command() when the user runs /exit --delete or /quit
        # --delete. Ported from google-gemini/gemini-cli#19332.
        self._pending_relaunch: list[str] | None = None
        self._last_ctrl_c_time = 0
        # Blocking-prompt overlays (clarify / sudo / approval / slash-confirm / model picker).
        self._clarify_state = self._clarify_multi_base = None
        self._clarify_freetext = False
        self._clarify_prefill = ""
        self._sudo_state = self._modal_input_snapshot = self._approval_state = None
        self._slash_confirm_state = self._model_picker_state = None
        self._clarify_deadline = self._sudo_deadline = self._approval_deadline = self._slash_confirm_deadline = 0
        self._approval_lock = threading.Lock()
        try:  # composer placeholder chosen once so it stays stable on screen
            from hermes_cli.tips import get_random_composer_placeholder
            self._composer_placeholder = get_random_composer_placeholder()
        except Exception:
            self._composer_placeholder = ""
        self._command_palette_state = self._secret_state = None
        self._pending_resume_sessions = None  # armed by a bare `/resume`; the next bare number selects
        self._pending_agent_seed = None  # one-shot seed from a slash handler
        self._secret_deadline = 0
        self._tool_start_time: float = 0.0
        self._pending_tool_info: dict = {}  # function_name -> [(preview, args)] for stacked scrollback
        self._spinner_text = self._command_status = ""
        self._last_scrollback_tool: str = ""  # "new" mode dedup
        self._command_running = self._command_blocks_input = False
        # Petdex mascot (display.pet): kitty placeholders on kitty/Ghostty, half-blocks elsewhere.
        self._pet_renderer = self._pet_anim_thread = None
        self._pet_slug = self._pet_kitty_pending = ""
        self._pet_enabled = self._pet_anim_running = False
        self._pet_cols: int = 18
        self._pet_scale: float = 0.7
        self._pet_frames_cache: dict = {}
        self._pet_kitty_cache: dict = {}
        self._pet_kitty_image_id = self._pet_frame_idx = 0
        self._pet_lock = threading.Lock()
        self._pet_cfg_checked = self._pet_event_until = 0.0
        self._pet_event: str = ""
        self._pet_reasoning = self._pet_turn_error = False
        self._attached_images: list[Path] = []
        self._image_counter = 0
        # Ctrl+S prompt stash; in-memory only because drafts routinely contain secrets.
        from hermes_cli.prompt_stash import PromptStash as _PromptStash
        self._prompt_stash = _PromptStash()
        self.preloaded_skills: list[str] = []
        self._startup_skills_line_shown = False
        # skills.auto_load rendered in the preload thread; None until joined. Handed to every
        # agent this CLI builds so the prompt bytes never depend on when the agent was created.
        self._auto_load_skills_result: Optional[tuple] = None
        # Background --skills preload, joined by finalize_preloaded_skills before any agent is built.
        self._preload_skills_thread: Optional[threading.Thread] = None
        self._preload_skills_result: Optional[tuple] = None
        self._preload_skills_error: Optional[BaseException] = None
        self._preload_skills_requested: list = []
        self._preload_skills_finalized = False
        self._active_session_lease = None

        # Voice mode state (also reinitialized inside run() for interactive TUI).
        self._voice_lock = threading.Lock()
        self._voice_mode = self._voice_tts = self._voice_recording = False
        self._voice_processing = self._voice_continuous = False
        self._voice_recorder = self._voice_tts_stop = None
        self._voice_tts_done = threading.Event()
        self._voice_tts_done.set()
        self._voice_barge_capture = threading.Event()  # barge monitor is capturing the interruption
        self._voice_last_tts_text = ""  # echo guard
        self._voice_barge_phase = None  # "generation" | "playback"

        self._status_bar_visible = _status_bar_visible_from_display_config(CLI_CONFIG.get("display"))
        self._battery_visible = bool(CLI_CONFIG["display"].get("battery", False))
        # Vi/vim editing mode for the input composer (display.vim_mode, config-only).
        # Off by default: prompt_toolkit's standard emacs bindings.
        self._vim_mode = bool(CLI_CONFIG["display"].get("vim_mode", False))
        # Hide rules + status bar until the next input after a resize, so SIGWINCH cannot
        # stamp a fresh status bar over one the terminal just reflowed into scrollback.
        self._status_bar_suppressed_after_resize = self._resize_recovery_pending = False
        self._resize_recovery_lock = threading.Lock()
        self._resize_recovery_timer = self._status_bar_unsuppress_timer = None  # latter: debounced un-suppress
        self._last_resize_width = None  # width change (reflow, needs viewport clear) vs rows-only

        self._background_tasks: Dict[str, threading.Thread] = {}
        self._background_task_counter = 0

        # Cache-hit baseline, reset on model switch / compression so the bar shows the current regime.
        self._cache_hit_baseline_prompt = self._cache_hit_baseline_read = self._cache_hit_baseline_compressions = 0
        self._cache_hit_baseline_model: Optional[str] = None
