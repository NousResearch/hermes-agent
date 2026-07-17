"""Oneshot (-z) mode: send a prompt, get the final content block, exit.

Bypasses cli.py entirely.  No banner, no spinner, no session_id line,
no stderr chatter.  Just the agent's final text to stdout.

Toolsets = explicit --toolsets when provided, otherwise whatever the user has
configured for "cli" in `hermes tools`.
Rules / memory / AGENTS.md / preloaded skills = same as a normal chat turn.
Approvals = auto-bypassed (HERMES_YOLO_MODE=1 is set for the call).
Working directory = the user's CWD (AGENTS.md etc. resolve from there as usual).

Model / provider selection mirrors `hermes chat`:
    - Both optional. If omitted, use the user's configured default.
    - If both given, pair them exactly as given.
    - If only --model given, auto-detect the provider that serves it.
    - If only --provider given, error out (ambiguous — caller must pick a model).

Env var fallbacks (used when the corresponding arg is not passed):
    - HERMES_INFERENCE_MODEL
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable, Iterator, Optional, TextIO

from hermes_cli.fallback_config import get_fallback_chain


logger = logging.getLogger(__name__)


class _IncompleteFinalWriteError(OSError):
    """A final-response stream accepted fewer characters than requested."""

    def __init__(self) -> None:
        super().__init__("oneshot final response write was incomplete")


class _IncompleteCleanupError(RuntimeError):
    """A bounded cleanup operation reported that it did not complete."""

    def __init__(self) -> None:
        super().__init__("oneshot bounded cleanup did not complete")


class _FinalDelivery:
    """Claim, write, and flush the one-shot final response exactly once."""

    def __init__(self, real_stdout: TextIO, on_delivered: Callable[[], None]) -> None:
        self._stream = real_stdout
        self._on_delivered = on_delivered
        self._lock = threading.Lock()
        self._claimed = False
        self._delivered = False
        self._delivery_error: BaseException | None = None

    @property
    def delivered(self) -> bool:
        return self._delivered

    def deliver(self, text: str | None) -> bool:
        value = text or ""
        if not value.strip():
            return False
        with self._lock:
            if self._claimed:
                if self._delivery_error is not None:
                    raise self._delivery_error
                return False
            self._claimed = True
            try:
                written = self._stream.write(value)
                if type(written) is not int or written != len(value):
                    raise _IncompleteFinalWriteError()
                if not value.endswith("\n"):
                    newline_written = self._stream.write("\n")
                    if type(newline_written) is not int or newline_written != 1:
                        raise _IncompleteFinalWriteError()
                self._stream.flush()
            except BaseException as exc:
                self._delivery_error = exc
                raise
            self._delivered = True
        try:
            self._on_delivered()
        except Exception:
            logger.exception("oneshot final-delivery completion callback failed")
        return True


class _OneshotResources:
    """Single idempotent owner for every resource created by one-shot mode."""

    def __init__(self) -> None:
        self.agent = None
        self.session_db = None
        self.task_id: str | None = None
        self._state_lock = threading.Lock()
        self._watchdog_armed = False
        self._final_marked = False
        self._closed = False

    def _arm_watchdog_once(self, exit_code: int = 70) -> None:
        with self._state_lock:
            if self._watchdog_armed:
                return
            from hermes_cli.exit_watchdog import arm_exit_watchdog

            # Keep the lock through setup so concurrent callers cannot start
            # duplicate successful timers. A BaseException exits before the
            # latch is claimed, allowing cleanup to retry safely.
            arm_exit_watchdog(exit_code=exit_code)
            self._watchdog_armed = True

    @staticmethod
    def _log_failure(operation: str, exc: BaseException) -> None:
        logger.warning(
            "oneshot lifecycle operation failed operation=%s exception_type=%s",
            operation,
            type(exc).__name__,
        )

    def mark_final_delivered(self) -> None:
        with self._state_lock:
            if self._final_marked:
                return
            self._final_marked = True

        control_failure: BaseException | None = None
        try:
            self._arm_watchdog_once(exit_code=70)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                control_failure = exc
            else:
                self._log_failure("watchdog", exc)

        agent = self.agent
        session_db = self.session_db
        session_id = getattr(agent, "session_id", None) if agent is not None else None
        if agent is not None and session_db is not None and session_id:
            try:
                session_db.end_session(session_id, "oneshot_complete")
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    if control_failure is None:
                        control_failure = exc
                else:
                    self._log_failure("session_end", exc)
            else:
                agent._end_session_on_close = False

        logger.info(
            "oneshot final delivered session_id=%s task_id=%s",
            session_id or "",
            self.task_id or "",
        )
        if control_failure is not None:
            raise control_failure

    def close(self, primary_failure: BaseException | None = None) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True

        control_failure: BaseException | None = None
        had_failure = False

        def attempt(operation: str, callback: Callable[[], None]) -> None:
            nonlocal control_failure, had_failure
            try:
                callback()
            except BaseException as exc:
                had_failure = True
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    if control_failure is None:
                        control_failure = exc
                else:
                    self._log_failure(operation, exc)

        attempt("watchdog", lambda: self._arm_watchdog_once(exit_code=70))

        def interrupt_delegations() -> None:
            from tools.async_delegation import interrupt_all

            interrupt_all(reason="oneshot shutdown")

        attempt("async_delegation", interrupt_delegations)

        agent = self.agent

        def flush_memory() -> None:
            manager = getattr(agent, "_memory_manager", None) if agent is not None else None
            flush_pending = getattr(manager, "flush_pending", None)
            if callable(flush_pending):
                if flush_pending(timeout=10) is False:
                    raise _IncompleteCleanupError()

        attempt("memory_flush", flush_memory)

        def shutdown_memory() -> None:
            if agent is None:
                return
            shutdown = getattr(agent, "shutdown_memory_provider", None)
            if not callable(shutdown):
                return
            messages = list(getattr(agent, "_session_messages", None) or [])
            shutdown(messages)

        attempt("memory_shutdown", shutdown_memory)

        def kill_processes() -> None:
            if not self.task_id:
                return
            from tools.process_registry import process_registry

            process_registry.kill_all(task_id=self.task_id)

        attempt("process_cleanup", kill_processes)

        def cleanup_task_resources() -> None:
            if agent is not None and self.task_id:
                cleanup = getattr(agent, "_cleanup_task_resources", None)
                if callable(cleanup):
                    cleanup(self.task_id)

        attempt("task_cleanup", cleanup_task_resources)
        attempt(
            "agent_close",
            lambda: agent.close() if agent is not None else None,
        )

        def shutdown_mcp() -> None:
            from tools.mcp_tool import shutdown_mcp_servers

            shutdown_mcp_servers()

        attempt("mcp_shutdown", shutdown_mcp)

        def shutdown_auxiliary_clients() -> None:
            from agent.auxiliary_client import shutdown_cached_clients

            shutdown_cached_clients()

        attempt("auxiliary_shutdown", shutdown_auxiliary_clients)
        attempt(
            "session_db_close",
            lambda: self.session_db.close() if self.session_db is not None else None,
        )

        session_id = getattr(agent, "session_id", None) if agent is not None else None
        if not had_failure:
            logger.info(
                "oneshot cleanup completed session_id=%s task_id=%s",
                session_id or "",
                self.task_id or "",
            )
        if primary_failure is None and control_failure is not None:
            raise control_failure


class _DeliveryDispatcherState:
    def __init__(self, manager) -> None:
        self.manager = manager
        self.deliveries: dict[str, _FinalDelivery] = {}
        self.dispatcher: Callable[..., None] | None = None


_delivery_dispatch_lock = threading.RLock()
_delivery_dispatchers: dict[int, _DeliveryDispatcherState] = {}


@contextmanager
def _install_final_delivery_hook(
    delivery: _FinalDelivery,
    resources: _OneshotResources,
) -> Iterator[None]:
    """Route matching one-shot task output through one transient first hook."""
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    task_id = resources.task_id
    if not task_id:
        raise ValueError("one-shot task_id must be set before installing delivery hook")

    manager_key = id(manager)
    with _delivery_dispatch_lock:
        state = _delivery_dispatchers.get(manager_key)
        if state is None or state.manager is not manager:
            state = _DeliveryDispatcherState(manager)

            def _dispatch(
                assistant_response: str = "",
                task_id: str | None = None,
                **_kwargs,
            ) -> None:
                # Snapshot under the registry lock. Context removal after this
                # point cannot revoke the rightful in-flight delivery.
                with _delivery_dispatch_lock:
                    target = state.deliveries.get(task_id or "")
                if target is not None:
                    target.deliver(assistant_response)

            state.dispatcher = _dispatch
            callbacks = manager._hooks.setdefault("post_llm_call", [])
            callbacks.insert(0, _dispatch)
            _delivery_dispatchers[manager_key] = state
        if task_id in state.deliveries:
            raise RuntimeError("duplicate active one-shot task_id")
        state.deliveries[task_id] = delivery

    try:
        yield
    finally:
        with _delivery_dispatch_lock:
            current_state = _delivery_dispatchers.get(manager_key)
            if current_state is state:
                if state.deliveries.get(task_id) is delivery:
                    del state.deliveries[task_id]
                if not state.deliveries:
                    current = manager._hooks.get("post_llm_call", [])
                    for index in range(len(current) - 1, -1, -1):
                        if current[index] is state.dispatcher:
                            del current[index]
                    del _delivery_dispatchers[manager_key]


def _normalize_toolsets(toolsets: object = None) -> list[str] | None:
    if not toolsets:
        return None

    raw_items = [toolsets] if isinstance(toolsets, str) else toolsets
    if not isinstance(raw_items, (list, tuple)):
        raw_items = [raw_items]

    normalized: list[str] = []
    for item in raw_items:
        if isinstance(item, str):
            normalized.extend(part.strip() for part in item.split(","))
        else:
            normalized.append(str(item).strip())

    return [item for item in normalized if item] or None


def _validate_explicit_toolsets(toolsets: object = None) -> tuple[list[str] | None, str | None]:
    normalized = _normalize_toolsets(toolsets)
    if normalized is None:
        return None, None

    try:
        from toolsets import validate_toolset
    except Exception as exc:
        return None, f"hermes -z: failed to validate --toolsets: {exc}\n"

    built_in = [name for name in normalized if validate_toolset(name)]
    unresolved = [name for name in normalized if name not in built_in]

    if unresolved:
        try:
            from hermes_cli.plugins import discover_plugins

            discover_plugins()
            plugin_valid = [name for name in unresolved if validate_toolset(name)]
        except Exception:
            plugin_valid = []

        if plugin_valid:
            built_in.extend(plugin_valid)
            unresolved = [name for name in unresolved if name not in plugin_valid]

    if any(name in {"all", "*"} for name in built_in):
        ignored = [name for name in normalized if name not in {"all", "*"}]
        if ignored:
            sys.stderr.write(
                "hermes -z: --toolsets all enables every toolset; "
                f"ignoring additional entries: {', '.join(ignored)}\n"
            )
        return None, None

    mcp_names: set[str] = set()
    mcp_disabled: set[str] = set()
    if unresolved:
        try:
            from hermes_cli.config import read_raw_config
            from hermes_cli.tools_config import _parse_enabled_flag

            cfg = read_raw_config()
            mcp_servers = cfg.get("mcp_servers") if isinstance(cfg.get("mcp_servers"), dict) else {}
            for name, server_cfg in mcp_servers.items():
                if not isinstance(server_cfg, dict):
                    continue
                if _parse_enabled_flag(server_cfg.get("enabled", True), default=True):
                    mcp_names.add(str(name))
                else:
                    mcp_disabled.add(str(name))
        except Exception:
            mcp_names = set()
            mcp_disabled = set()

    mcp_valid = [name for name in unresolved if name in mcp_names]
    disabled = [name for name in unresolved if name in mcp_disabled]
    unknown = [name for name in unresolved if name not in mcp_names and name not in mcp_disabled]
    valid = built_in + mcp_valid

    if unknown:
        sys.stderr.write(f"hermes -z: ignoring unknown --toolsets entries: {', '.join(unknown)}\n")
    if disabled:
        sys.stderr.write(
            "hermes -z: ignoring disabled MCP servers (set enabled: true in config.yaml to use): "
            f"{', '.join(disabled)}\n"
        )

    if not valid:
        return None, "hermes -z: --toolsets did not contain any valid toolsets.\n"

    return valid, None


def _write_usage_file(path: Optional[str], result: dict, failure: Optional[str] = None) -> None:
    """Best-effort JSON usage report for pipelines (``-z --usage-file``).

    Written even on failure so callers can always account for spend. Never
    raises — a broken usage write must not mask the run's own outcome.
    """
    if not path:
        return
    try:
        import json

        report = {
            "estimated_cost_usd": result.get("estimated_cost_usd"),
            "cost_status": result.get("cost_status"),
            "cost_source": result.get("cost_source"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cache_read_tokens": result.get("cache_read_tokens"),
            "cache_write_tokens": result.get("cache_write_tokens"),
            "reasoning_tokens": result.get("reasoning_tokens"),
            "total_tokens": result.get("total_tokens"),
            "api_calls": result.get("api_calls"),
            "model": result.get("model"),
            "provider": result.get("provider"),
            "session_id": result.get("session_id"),
            "completed": result.get("completed"),
            "failed": bool(result.get("failed")) or failure is not None,
            # Billing-audit field: the service tier this run REQUESTED via
            # request_overrides.extra_body (e.g. OpenAI "flex"). None when
            # unset. Lets batch pipelines verify the tier they think they're
            # paying for actually went out on the wire (July 2026 incident:
            # a config-matching bug silently dropped flex -> 2.3x billing).
            "service_tier": result.get("service_tier"),
        }
        if failure is not None:
            report["failure"] = failure
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def run_oneshot(
    prompt: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    toolsets: object = None,
    usage_file: Optional[str] = None,
) -> int:
    """Execute a single prompt and print only the final content block.

    Args:
        prompt: The user message to send.
        model: Optional model override. Falls back to HERMES_INFERENCE_MODEL
            env var, then config.yaml's model.default / model.model.
        provider: Optional provider override. Falls back to config.yaml's
            model.provider, then "auto".
        toolsets: Optional comma-separated string or iterable of toolsets.
        usage_file: Optional path; when set, a JSON usage report (estimated
            cost, token counts, model, api_calls) is written there after the
            run — even when the run fails — so pipelines can account for
            spend per invocation.

    Returns the exit code.  Caller should sys.exit() with the return.
    """
    # Silence every stdlib logger for the duration.  Task 4 replaces this
    # process-global switch with handler-scoped terminal suppression while
    # preserving file logs.
    logging.disable(logging.CRITICAL)

    # --provider without --model is ambiguous: carrying the user's configured
    # model across to a different provider is usually wrong (that provider may
    # not host it), and silently picking the provider's catalog default hides
    # the mismatch.  Require the caller to be explicit.  Validate BEFORE the
    # stderr redirect so the message actually reaches the terminal.
    env_model_early = os.getenv("HERMES_INFERENCE_MODEL", "").strip()
    if provider and not ((model or "").strip() or env_model_early):
        sys.stderr.write(
            "hermes -z: --provider requires --model (or HERMES_INFERENCE_MODEL). "
            "Pass both explicitly, or neither to use your configured defaults.\n"
        )
        return 2

    explicit_toolsets, toolsets_error = _validate_explicit_toolsets(toolsets)
    if toolsets_error:
        sys.stderr.write(toolsets_error)
        return 2
    use_config_toolsets = _normalize_toolsets(toolsets) is None

    # Auto-approve any shell / tool approvals.  Non-interactive by
    # definition — a prompt would hang forever.
    os.environ["HERMES_YOLO_MODE"] = "1"
    os.environ["HERMES_ACCEPT_HOOKS"] = "1"

    # Redirect stderr AND stdout to devnull for the entire call tree.
    # We'll print the final response to the real stdout at the end.
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    devnull = open(os.devnull, "w", encoding="utf-8")

    response: Optional[str] = None
    result: dict = {}
    failure: BaseException | None = None
    resources = _OneshotResources()
    delivery = _FinalDelivery(real_stdout, resources.mark_final_delivered)
    try:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            try:
                response, result = _run_agent(
                    prompt,
                    model=model,
                    provider=provider,
                    toolsets=explicit_toolsets,
                    use_config_toolsets=use_config_toolsets,
                    delivery=delivery,
                    resources=resources,
                )
                delivery.deliver(response)
            except BaseException as exc:  # noqa: BLE001
                # Capture anything that escapes the agent (including OSError
                # from prompt_toolkit/Vt100 when stdout is a non-TTY pipe,
                # KeyboardInterrupt, SystemExit, etc.) so we can surface it on
                # the real stderr instead of crashing past the redirect with a
                # traceback that the caller never sees. A silent exit in a
                # cron / SSH / subprocess context is the worst failure mode.
                # See #30623.
                failure = exc
            try:
                resources.close(primary_failure=failure)
            except BaseException as exc:  # noqa: BLE001
                if failure is None:
                    failure = exc
    finally:
        try:
            devnull.close()
        except Exception:
            pass

    if failure is not None:
        # Re-raise control-flow exceptions so the parent handles them as usual
        # (Ctrl-C / explicit sys.exit() inside the agent).
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            _write_usage_file(usage_file, result, failure=repr(failure))
            raise failure
        _write_usage_file(usage_file, result, failure=str(failure))
        real_stderr.write(f"hermes -z: agent failed: {failure}\n")
        real_stderr.flush()
        return 1

    _write_usage_file(usage_file, result)

    if (result.get("failed") or result.get("partial")) and not delivery.delivered:
        return 2

    if not delivery.delivered:
        real_stderr.write("hermes -z: no final response was produced; treating the run as failed.\n")
        real_stderr.flush()
        return 1

    return 0


def _create_session_db_for_oneshot():
    """Best-effort SessionDB for ``hermes -z`` / oneshot mode.

    Oneshot bypasses ``HermesCLI._init_agent()``, so it must wire the SQLite
    session store itself. Without this, the ``session_search``/recall tool is
    advertised but every call returns "Session database not available.".
    """
    try:
        from hermes_state import SessionDB

        return SessionDB()
    except Exception as exc:
        logging.debug("SQLite session store not available for oneshot mode: %s", exc)
        return None


def _run_agent(
    prompt: str,
    model: Optional[str],
    provider: Optional[str],
    toolsets: object,
    use_config_toolsets: bool,
    delivery: _FinalDelivery,
    resources: _OneshotResources,
) -> tuple[str, dict]:
    """Build an AIAgent exactly like a normal CLI chat turn would, then
    run a single conversation.  Returns ``(final_response, run_result)``."""
    # Imports are local so they don't run when hermes is invoked for
    # other commands (keeps top-level CLI startup cheap).
    from hermes_cli.config import load_config
    from hermes_cli.models import detect_provider_for_model
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.tools_config import _get_platform_tools
    from run_agent import AIAgent

    cfg = load_config()

    # Resolve effective model: explicit arg → env var → config.
    model_cfg = cfg.get("model") or {}
    if isinstance(model_cfg, str):
        cfg_model = model_cfg
    else:
        cfg_model = model_cfg.get("default") or model_cfg.get("model") or ""

    env_model = os.getenv("HERMES_INFERENCE_MODEL", "").strip()
    effective_model = (model or "").strip() or env_model or cfg_model

    # Resolve effective provider: explicit arg → (auto-detect from model if
    # model was explicit) → env / config (handled inside resolve_runtime_provider).
    #
    # When --model is given without --provider, auto-detect the provider that
    # serves that model — same semantic as `/model <name>` in an interactive
    # session.  Without this, resolve_runtime_provider() would fall back to
    # the user's configured default provider, which may not host the model
    # the caller just asked for.
    effective_provider = (provider or "").strip() or None
    explicit_base_url_from_alias: Optional[str] = None
    if effective_provider is None and (model or env_model):
        # Only auto-detect when the model was explicitly requested via arg or
        # env var (not when it came from config — that's the "use my defaults"
        # path and the configured provider is already correct).
        explicit_model = (model or "").strip() or env_model
        if explicit_model:
            # First check DIRECT_ALIASES populated from config.yaml `model_aliases:`.
            # These map a user-defined alias to (model, provider, base_url) for
            # endpoints not in any catalog (local servers, custom proxies, etc.).
            try:
                from hermes_cli import model_switch as _ms
                _ms._ensure_direct_aliases()
                direct = _ms.DIRECT_ALIASES.get(explicit_model.strip().lower())
            except Exception:
                direct = None
            if direct is not None:
                effective_model = direct.model
                effective_provider = direct.provider
                if direct.base_url:
                    explicit_base_url_from_alias = direct.base_url.rstrip("/")
            else:
                cfg_provider = ""
                if isinstance(model_cfg, dict):
                    cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
                current_provider = (
                    cfg_provider
                    or os.getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower()
                    or "auto"
                )
                detected = detect_provider_for_model(explicit_model, current_provider)
                if detected:
                    effective_provider, effective_model = detected

    runtime = resolve_runtime_provider(
        requested=effective_provider,
        target_model=effective_model or None,
        explicit_base_url=explicit_base_url_from_alias,
    )

    # Pull in explicit toolsets when provided; otherwise use whatever the user
    # has enabled for "cli". sorted() gives stable ordering for config-derived
    # sets; explicit values preserve user order.
    toolsets_list = _normalize_toolsets(toolsets)
    if toolsets_list is None and use_config_toolsets:
        toolsets_list = sorted(_get_platform_tools(cfg, "cli"))

    session_db = _create_session_db_for_oneshot()
    resources.session_db = session_db
    # Read the effective fallback chain from profile config so oneshot workers
    # honour the same merge semantics as interactive CLI and gateway sessions.
    _fb = get_fallback_chain(cfg)

    agent = AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        api_mode=runtime.get("api_mode"),
        model=effective_model,
        enabled_toolsets=toolsets_list,
        quiet_mode=True,
        platform="cli",
        session_db=session_db,
        credential_pool=runtime.get("credential_pool"),
        fallback_model=_fb or None,
        # Interactive callbacks are intentionally NOT wired beyond this
        # one.  In oneshot mode there's no user sitting at a terminal:
        #   - clarify  → returns a synthetic "pick a default" instruction
        #                so the agent continues instead of stalling on
        #                the tool's built-in "not available" error
        #   - sudo password prompt → terminal_tool gates on
        #                HERMES_INTERACTIVE which we never set
        #   - shell-hook approval → auto-approved via HERMES_ACCEPT_HOOKS=1
        #                (set above); also falls back to deny on non-tty
        #   - dangerous-command approval → bypassed via HERMES_YOLO_MODE=1
        #   - skill secret capture → returns gracefully when no callback set
        clarify_callback=_oneshot_clarify_callback,
    )
    resources.agent = agent

    # Belt-and-braces: make sure AIAgent doesn't invoke any streaming
    # display callbacks that would bypass our stdout capture.
    agent.suppress_status_output = True
    agent.stream_delta_callback = None
    agent.tool_gen_callback = None

    resources.task_id = str(uuid.uuid4())
    with _install_final_delivery_hook(delivery, resources):
        result = agent.run_conversation(prompt, task_id=resources.task_id)
    return (result.get("final_response") or "", result)


def _oneshot_clarify_callback(question: str, choices=None) -> str:
    """Clarify is disabled in oneshot mode — tell the agent to pick a
    default and proceed instead of stalling or erroring."""
    if choices:
        return (
            f"[oneshot mode: no user available. Pick the best option from "
            f"{choices} using your own judgment and continue.]"
        )
    return (
        "[oneshot mode: no user available. Make the most reasonable "
        "assumption you can and continue.]"
    )
