"""Writable runner initialization, separate from process ownership."""
from __future__ import annotations

class GatewayRuntimeInitMixin:
    def _init_runtime_settings(self) -> None:
        """Load ephemeral per-call config (prefill, reasoning, busy modes, timeouts, routing)."""
        from gateway.run import (Dict)
        self._prefill_messages = self._load_prefill_messages()
        self._reasoning_config = self._load_reasoning_config()
        self._service_tier = self._load_service_tier()
        self._show_reasoning = self._load_show_reasoning()
        self._busy_input_mode = self._load_busy_input_mode()
        self._busy_text_mode = self._load_busy_text_mode()
        # Secondary-profile busy modes snapshotted at multiplex startup; handlers never reread config.
        self._busy_input_modes_by_profile: Dict[str, str] = {}
        self._busy_text_modes_by_profile: Dict[str, str] = {}
        self._restart_drain_timeout = self._load_restart_drain_timeout()
        self._restart_after_turn_timeout = self._load_restart_after_turn_timeout()
        self._cron_drain_timeout = self._load_cron_drain_timeout()
        self._signal_interrupt_grace_timeout = self._load_signal_interrupt_grace_timeout()
        self._provider_routing = self._load_provider_routing()
        self._fallback_model = self._load_fallback_model()


    def _init_session_store(self) -> None:
        """Build the SessionStore (with process-registry reset guard), its async facade and the router."""
        from gateway.run import (AsyncSessionStore, DeliveryRouter, SessionStore)
        from tools.process_registry import process_registry
        self.session_store = SessionStore(
            self.config.sessions_dir, self.config,
            has_active_processes_fn=lambda key: process_registry.has_active_for_session(
                key))
        # Loop-side boundary: sync helpers use ``session_store`` directly; async handlers await this facade.
        self._async_session_store = AsyncSessionStore(self.session_store)
        self.delivery_router = DeliveryRouter(self.config)


    def _init_lifecycle_state(self) -> None:
        """Initialise run/exit/restart flags, per-session state, and completion-delivery bookkeeping."""
        from gateway.run import (Dict, List, MessageEvent, Optional, OrderedDict, Platform, SessionSource, SessionState, SessionTurnLeaseRegistry, asyncio, concurrent, threading, time)
        self._running = self._exit_cleanly = self._exit_with_failure = self._draining = False
        self._gateway_loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event = asyncio.Event()
        self._exit_reason: Optional[str] = None
        self._exit_code: Optional[int] = None
        self._profile_failed_platforms: Dict[str, Dict[Platform, asyncio.Task]] = {}
        self._systemd_watchdog = None
        # External (NAS-driven) drain, distinct from one-way ``_draining``: set while ``.drain_request.json``
        # exists — NEW turns refused, process stays up, removing the marker reverts to ``running``.
        self._external_drain_active = False
        # ``_signal_initiated_shutdown``: SIGTERM/SIGINT with no planned-stop/takeover marker (container,
        # OOM, bare kill); _stop_impl must NOT persist gateway_state=stopped or container_boot won't restart.
        self._restart_requested = self._signal_initiated_shutdown = self._restart_task_started = False
        self._restart_detached = self._restart_via_service = self._detached_restart_helper_started = False
        self._restart_command_source: Optional[SessionSource] = None
        # Construction clock: bounds the /restart redelivery guard's window (missing dedup marker = stale).
        self._startup_time: float = time.time()
        # True when booted from a chat /restart (.restart_notify.json existed). One-shot signal so the
        # marker-missing fallback suppresses a /restart only when we KNOW we just restarted.
        self._booted_from_restart: bool = False
        self._stop_task: Optional[asyncio.Task] = None
        self._restart_task: Optional[asyncio.Task] = None
        self._executor_lock = threading.Lock()
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # Set on gateway stop so the recreate-on-shutdown path can't resurrect the pool.
        self._executor_closing = False
        # ALL per-session state lives here (gateway/session_state.py); use _session_state / _peek_session_state.
        self._sessions: Dict[str, SessionState] = {}
        # Per-SESSION_ID turn lease: serializes [load history → run → flush] when two ROUTING KEYS resolve
        # to one session_id (switch_session's many-to-one mapping), which routing-key guards cannot see.
        self._turn_leases = SessionTurnLeaseRegistry()
        # Stall-notified keys clear when pending clears / activity resumes / conversation boundary.
        # Tokens for held turn leases, keyed by (routing key, run generation) so release is granted per-turn
        # and a stale unwind can never free a newer turn's lease (#28686 ownership lesson). Held turn-lease
        # tokens live on SessionState.turn.lease_token / .lease_generation (the old dict was keyed (routing
        # key, generation) so a stale unwind could never free a newer turn's lease — the generation field
        # preserves that ownership check, #28686). Runner-level queued interrupt text lives on
        # SessionState.persistent.pending_command_text (NOTE: distinct from the adapter-level
        # _pending_messages Dict[str, MessageEvent] in gateway/platforms/base.py, which shares the legacy
        # name). Last successfully-resolved (non-empty) model, keyed by session. Used as a fallback when a
        # fresh config read transiently returns an empty model (e.g. an mtime-keyed config-cache miss during
        # a post-interrupt recovery turn). Without this, the agent is built with model="" and every API call
        # fails HTTP 400 "No models provided" — the session goes silent until the user manually re-sends.
        # See #35314. The ``"*"`` session entry holds a process-wide last-known-good for sessions seen for
        # the first time. Lives on SessionState.conversation.last_resolved_model. Overflow buffer for
        # explicit /queue commands. The adapter-level _pending_messages dict is a single slot per session
        # (designed for "next-turn" follow-ups where repeated sends collapse into one event).  /queue has
        # different semantics: each invocation must produce its own full agent turn, in FIFO order, with no
        # merging. When the slot is occupied, additional /queue items land here and are promoted
        # one-at-a-time after each run's drain. Cleared on /new and /reset.  /model and other mid-session
        # operations preserve the queue. Lives on SessionState.conversation.queued_events; native image
        # paths, busy-ack debounce timestamps and the monotonic run-generation counter (#28686, NEVER reset)
        # live on SessionState too. See gateway.session_stall.
        self._session_stall_notified: Dict[str, bool] = {}
        # Startup restore gate: while restart-interrupted sessions auto-resume, real inbound messages
        # queue instead of competing with the synthetic resume turns; drained after all resume tasks end.
        self._startup_restore_in_progress = False
        self._startup_restore_queue: List[MessageEvent] = []
        self._startup_restore_tasks: List[asyncio.Task] = []
        # Set by start_gateway() only for an explicit ``--replace`` launch; scoped to each adapter's
        # cold-start connect and removed before any reconnect can run.
        self._platform_lock_takeover_on_start = False
        # Capped LRU of live SessionSources for fallback routing (shutdown notices, synthetic events) when
        # the persisted origin is missing and _parse_session_key can't recover thread_id.
        self._session_sources: "OrderedDict[str, SessionSource]" = OrderedDict()
        self._session_sources_max = 512
        # Lifecycle-scoped completion dedup: closes queue/watcher races inside one gateway without claiming
        # exactly-once across a crash; durable replay state stays owned by tools.async_delegation.
        self._completion_delivery_lock = threading.Lock()
        self._completion_deliveries_inflight: set[tuple[str, str, object]] = set()
        self._completion_deliveries_delivered: "OrderedDict[tuple[str, str, object], None]" = OrderedDict()
        self._completion_delivery_retention = 2048
        # Agent-triggered terminal completions from one conversation often land in the same scheduler
        # tick; hold them briefly so the agent gets one synthetic turn instead of one per process.
        # See #70300.
        self._completion_notification_batches: dict[tuple[str, ...], list[tuple[str, dict, asyncio.Future]]] = {}
        self._completion_notification_batch_tasks: dict[tuple[str, ...], asyncio.Task] = {}
        self._completion_notification_batch_flush_tasks: set[asyncio.Task] = set()
        self._completion_notification_batch_window = 0.1
        self._completion_notification_batches_stopping = False


    def _init_runtime_caches(self) -> None:
        """Agent cache, profile identity, Teams runtime, failed-platform tracking, slash-confirm counter."""
        from gateway.run import (Any, Dict, Optional, OrderedDict, Platform, threading)
        # AIAgent per session preserves prompt caching (fresh agent per message ~10x cost on Anthropic).
        # Value: (AIAgent, config_signature); LRU cap in _enforce_agent_cache_cap, TTL in expiry watcher.
        self._agent_cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._agent_cache_lock = threading.Lock()
        # Launch-time identity of the profile that owns ``self.adapters``; ``_authorization_adapter``
        # compares against this rather than the per-turn ``_active_profile_name()``.
        self._primary_profile_name = self._kanban_notifier_profile = self._active_profile_name()
        # Teams meeting pipeline runtime (bound later when msgraph_webhook adapter exists).
        self._teams_pipeline_runtime = None
        self._teams_pipeline_runtime_error: Optional[str] = None
        # Failed-to-connect platforms for background reconnection: Platform -> {config, attempts, next_retry}
        self._failed_platforms: Dict[Platform, Dict[str, Any]] = {}
        # Strong refs to detached fatal-error handler tasks so the loop can't GC them mid-run.
        self._fatal_handler_tasks: set = set()
        # Slash-confirm state lives in tools.slash_confirm (module-level) so adapters resolve callbacks
        # without a runner backref; local counter keeps confirm_ids compact (64-byte callback_data caps).
        import itertools
        self._slash_confirm_counter = itertools.count(1)


    def _init_startup_checks(self) -> None:
        """Ensure tirith is installed and warn when manual approvals have no automated assessor."""
        from gateway.run import (_best_effort, cfg_get, logger)
        def _ensure_tirith() -> None:
            from tools.tirith_security import ensure_installed
            ensure_installed(log_failures=False)  # downloads if needed; fail-open at scan time

        _best_effort(_ensure_tirith)

        # Manual approvals with no automated assessor (tirith off AND no auxiliary.approval) fail closed
        # on unattended gateways — surface it so operators knowingly enable one.
        try:
            from hermes_cli.config import load_config as _load_full_config
            # Startup heads-up (#30882): a gateway in manual approval mode with no automated risk assessor
            # (tirith disabled AND no auxiliary.approval model) can only gate dangerous commands /
            # execute_code scripts via live in-chat approval.
            _appr_cfg = _load_full_config()
            _appr_mode = str(
                cfg_get(_appr_cfg, "approvals", "mode", default="manual") or "manual"
            ).strip().lower()
            _tirith_on = bool(cfg_get(_appr_cfg, "security", "tirith_enabled", default=True))
            _aux_approval = cfg_get(_appr_cfg, "auxiliary", "approval", default=None)
            if _appr_mode == "manual" and not _tirith_on and not _aux_approval:
                logger.warning(
                    "Gateway approvals.mode=manual with no automated risk "
                    "assessor (security.tirith_enabled is false and "
                    "auxiliary.approval is unset): dangerous commands and "
                    "execute_code scripts will BLOCK until a human approves "
                    "them in chat. Enable security.tirith_enabled or configure "
                    "auxiliary.approval for unattended operation.")
        except Exception:
            logger.debug("approvals.mode startup check skipped", exc_info=True)


    def _init_session_db(self) -> None:
        """Open the session DB for the active scope and run opportunistic state.db / checkpoint maintenance."""
        from gateway.run import (Any, Dict, Path, _SESSION_DB_UNPINNED, logger, threading)
        # Session DB is a property caching one AsyncSessionDB per path (a handle bound here would pin the
        # root home under multiplex); priming here keeps startup diagnostics at init.
        # Initialize session database for session_search tool support. Same frozen-handle class of bug as
        # SessionStore._db (#88532): a handle bound here is pinned to the process's root home, but /resume,
        # /title, /history and session search all run inside _profile_runtime_scope on a multiplexed gateway
        # and must see that profile's own state.db.
        self._session_db_pinned: Any = _SESSION_DB_UNPINNED
        self._session_db_handles: Dict[Path, Any] = {}
        self._session_db_handles_lock = threading.Lock()
        from gateway.session_db_recovery import RecoverableHandleCache
        self._session_db_handle_cache = RecoverableHandleCache(
            handles=self._session_db_handles, lock=self._session_db_handles_lock)
        try:
            self._open_session_db_for_active_scope(raise_on_error=True)
        except Exception as e:
            # WARNING (not DEBUG) so it lands in errors.log; else an NFS HERMES_HOME silently loses /resume etc.
            logger.warning("SQLite session store not available: %s", e)
            self._session_db_init_error = str(e)  # surfaced on the home channel(s) once connected

        # Opportunistic state.db maintenance (prune + optional VACUUM), at most once per min_interval_hours.
        # A few blocking seconds per day is fine for a long-lived gateway; failures log, never raise.
        # Surface the failure to the user via their home channel(s) once the gateway connects. Without this,
        # state.db corruption or NFS/SMB lock failures silently degrade the entire gateway — messages may
        # flow but nothing is persisted, and the user has no indication until they try /resume and find
        # nothing (#88235).
        if self._session_db is not None:
            try:
                from hermes_cli.config import load_config as _load_full_config
                _sess_cfg = (_load_full_config().get("sessions") or {})
                if _sess_cfg.get("auto_archive", False):
                    self._session_db._db.maybe_auto_archive(
                        idle_days=float(_sess_cfg.get("auto_archive_days", 3)),
                        min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)))
                if _sess_cfg.get("auto_prune", False):
                    # Construction-time, before the loop serves traffic; sync DB is fine.
                    self._session_db._db.maybe_auto_prune_and_vacuum(
                        retention_days=int(_sess_cfg.get("retention_days", 90)),
                        min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)),
                        min_vacuum_interval_days=int(
                            _sess_cfg.get("min_vacuum_interval_days", 30)),
                        vacuum=bool(_sess_cfg.get("vacuum_after_prune", True)),
                        sessions_dir=self.config.sessions_dir)
            except Exception as exc:
                logger.debug("state.db auto-maintenance skipped: %s", exc)

        # Stale checkpoint repo cleanup; opt-in via checkpoints.auto_prune, idempotent via .last_prune.
        try:
            from hermes_cli.config import load_config as _load_full_config
            _ckpt_cfg = (_load_full_config().get("checkpoints") or {})
            if _ckpt_cfg.get("auto_prune", False):
                from tools.checkpoint_manager import maybe_auto_prune_checkpoints
                # delete_orphans never honoured unattended: a missing workdir is ambiguous (deleted vs.
                # unmounted share); orphan cleanup is only via explicit `hermes checkpoints prune`.
                maybe_auto_prune_checkpoints(
                    retention_days=int(_ckpt_cfg.get("retention_days", 7)),
                    min_interval_hours=int(_ckpt_cfg.get("min_interval_hours", 24)),
                    delete_orphans=False,
                    max_total_size_mb=int(_ckpt_cfg.get("max_total_size_mb", 500)))
        except Exception as exc:
            logger.debug("checkpoint auto-maintenance skipped: %s", exc)


    def _init_registries_and_clocks(self) -> None:
        """Pairing stores, hook registry, voice modes, background-task set, liveness and idle clocks."""
        from gateway.run import (Dict, List, Optional, asyncio, time)
        # ``pairing_store``: global/default store (CLI, callers without profile context); ``pairing_stores``:
        # per-profile map ``authz_mixin._is_user_authorized`` routes through (one whitelist per profile).
        from gateway.pairing import PairingStore
        from gateway.hooks import ProfileHookRegistries
        self.pairing_store = PairingStore()
        self.pairing_stores: Dict[str, "PairingStore"] = {}
        # One HookRegistry per served profile home, resolved from the active scope at emit time.
        self.hooks = ProfileHookRegistries()
        # Per-chat voice reply mode: "off" | "voice_only" | "all"
        self._voice_mode: Dict[str, str] = self._load_voice_modes()
        # Per-(guild,user) transcript dedup: the voice/STT pipeline can emit one utterance twice.
        self._recent_voice_transcripts: Dict[tuple[int, int], List[tuple[float, str]]] = {}
        # Background tasks kept referenced so they are not garbage-collected mid-execution.
        self._background_tasks: set = set()
        # Event-loop liveness heartbeat: rewritten every 30s while the loop dispatches; supervisors use
        # the file mtime / updated_at to tell "process alive" from "loop frozen".
        # See #66892.
        self._gateway_started_at: float = time.time()
        self._loop_heartbeat_task: Optional[asyncio.Task] = None
        self._loop_floor_timer_handle = self._loop_liveness_watchdog = None
        # scale-to-zero: gateway-scoped "last inbound seen" clock, stamped in _handle_message (the single
        # inbound chokepoint) and seeded to "now" so a fresh gateway isn't idle from epoch.
        self._last_inbound_at: float = time.time()
        # Re-arm cooldown after a wake so we don't go dormant again before the drained backlog updates
        # the clock; and a one-shot latch so the "platform owns the suspend" notice logs once.
        self._scale_to_zero_cooldown_until: float = 0.0
        self._scale_to_zero_no_suspend_logged: bool = False


