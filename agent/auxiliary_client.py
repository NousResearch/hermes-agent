"""Shared auxiliary client router for side tasks (compression, search, vision, ...).

Text auto chain: main provider+model → OpenRouter → Nous Portal → custom endpoint →
native Anthropic → direct API-key providers → None. Vision auto chain: main
provider (if a supported vision backend) → OpenRouter → Nous → Anthropic → custom.
``auxiliary.free_only`` restricts the OpenRouter lane to ``:free`` SKUs. Codex OAuth is
in neither chain (undocumented, shifting allow-list): main provider or explicit
``auxiliary.<task>.provider`` only. HTTP 402 in call_llm() falls through the chain.
"""

import contextlib
import contextvars
import functools
import hashlib
import inspect
import json
import logging
import os
import re
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse, parse_qs, urlunparse

from agent.codex_headers import (
    CODEX_AUX_BASE_URL as _CODEX_AUX_BASE_URL,
    apply_required_codex_headers as _apply_required_codex_headers,
    codex_cloudflare_headers as _codex_cloudflare_headers,
    is_official_codex_base_url as _is_official_codex_base_url,
)

# `openai.OpenAI` is imported lazily (~240 ms cold); `OpenAI` below is a proxy
# so in-module calls, `auxiliary_client.OpenAI` reads and
# `patch("agent.auxiliary_client.OpenAI")` all keep working.
if TYPE_CHECKING:
    from openai import OpenAI  # noqa: F401 — type hints only

_OPENAI_CLS_CACHE: Optional[type] = None


def _load_openai_cls() -> type:
    """Import and cache ``openai.OpenAI``."""
    global _OPENAI_CLS_CACHE
    if _OPENAI_CLS_CACHE is None:
        from openai import OpenAI as _cls
        _OPENAI_CLS_CACHE = _cls
    return _OPENAI_CLS_CACHE


class _OpenAIProxy:
    """Lazy stand-in for ``openai.OpenAI``: forwards calls and isinstance checks, importing on first use."""
    __slots__ = ()

    def __call__(self, *args, **kwargs):
        return _load_openai_cls()(*args, **kwargs)

    def __instancecheck__(self, obj):
        return isinstance(obj, _load_openai_cls())

    def __repr__(self):
        return "<lazy openai.OpenAI proxy>"


OpenAI = _OpenAIProxy()


# Availability probe mode: check_fns only need to know whether a client is RESOLVABLE, so
# inside `aux_probe_mode()` constructors return a stub instead of importing openai + building
# httpx/SSL (~0.3s on CLI startup). Stubs are never cached (see _store_cached_client).
_aux_probe_state = threading.local()


class _AuxProbeClientStub:
    """Non-functional placeholder returned while `aux_probe_mode` is active."""
    __slots__ = ("api_key", "base_url")

    def __init__(self, api_key: str = "", base_url: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url

    def __getattr__(self, name: str) -> Any:
        # Loud failure if a probe stub ever leaks into a runtime call path.
        raise RuntimeError(
            f"_AuxProbeClientStub used as a real client (attribute {name!r}); "
            "aux_probe_mode is for availability checks only")

    def __repr__(self) -> str:
        return "<aux availability-probe client stub>"


def _aux_probe_active() -> bool:
    return bool(getattr(_aux_probe_state, "active", False))


@contextlib.contextmanager
def aux_probe_mode():
    """Resolve provider availability without constructing real SDK clients."""
    prev = getattr(_aux_probe_state, "active", False)
    _aux_probe_state.active = True
    try:
        yield
    finally:
        _aux_probe_state.active = prev


from agent.credential_pool import load_pool
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH, get_model_context_length,
    strip_codex_context_variant_suffix as _strip_codex_ctx_variant,
)
from hermes_cli.config import get_hermes_home
from hermes_constants import OPENROUTER_BASE_URL
from utils import base_url_host_matches, base_url_hostname, env_float, is_truthy_value, model_forces_max_completion_tokens, normalize_proxy_env_vars

logger = logging.getLogger(__name__)


# resolve_provider_client fall-through dedup: misconfigured-provider warnings fire on every
# retry, so only the first per process surfaces. Separate sets let tests clear each branch.
_LOGGED_UNKNOWN_PROVIDER_KEYS: set = set()
_LOGGED_UNHANDLED_AUTHTYPE_KEYS: set = set()
_LOGGED_UNSUPPORTED_EXTPROC_KEYS: set = set()
_LOGGED_UNSUPPORTED_OAUTH_KEYS: set = set()


def _resolve_aux_verify(base_url: Optional[str]) -> Any:
    """httpx ``verify`` for an aux base_url, mirroring the main client (per-provider ``ssl_ca_cert`` /
    ``ssl_verify``, ``HERMES_CA_BUNDLE`` / ``SSL_CERT_FILE``); any failure → httpx default (``True``)."""
    try:
        from agent.ssl_verify import resolve_httpx_verify
        from hermes_cli.config import get_custom_provider_tls_settings, load_config_readonly
        tls = get_custom_provider_tls_settings(str(base_url or ""), config=load_config_readonly())
        return resolve_httpx_verify(
            ca_bundle=tls.get("ssl_ca_cert"), ssl_verify=tls.get("ssl_verify"), base_url=str(base_url or ""))
    except Exception:
        return True


_WARNED_KEEPALIVE_IMPORT_SKEW = False


def _openai_http_client_kwargs(base_url: Optional[str], *, async_mode: bool = False) -> Dict[str, Any]:
    """Inject keepalive httpx client with env-only proxy (not macOS system proxy)."""
    try:
        from agent.process_bootstrap import build_keepalive_http_client
        client = build_keepalive_http_client(
            str(base_url or ""), async_mode=async_mode, verify=_resolve_aux_verify(base_url))
    except (ImportError, AttributeError):
        # Version-skewed install (Desktop runtime lagging a git tree) lacks this helper:
        # degrade to the SDK default httpx client rather than kill the job; warn once.
        global _WARNED_KEEPALIVE_IMPORT_SKEW
        if not _WARNED_KEEPALIVE_IMPORT_SKEW:
            _WARNED_KEEPALIVE_IMPORT_SKEW = True
            logger.warning(
                "agent.process_bootstrap.build_keepalive_http_client is "
                "unavailable — mixed/stale install detected (#64333). Falling "
                "back to the SDK default HTTP client. Run `hermes update` (or "
                "reinstall the Desktop app) to resync the runtime.")
        client = None
    return {"http_client": client} if client is not None else {}


def _create_openai_client(*, api_key: str, base_url: str, **kwargs: Any) -> Any:
    if _aux_probe_active():
        # Availability probe: resolved credentials/base_url are the answer.
        return _AuxProbeClientStub(api_key=api_key, base_url=base_url)
    kwargs = {**_openai_http_client_kwargs(base_url), **kwargs}
    # OpenCode Zen free tier: the keyless placeholder must never hit the wire (relay 401s any
    # unrecognized bearer) — blank the Authorization header.
    with contextlib.suppress(Exception):
        from hermes_cli.models import OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER, opencode_zen_free_headers
        if api_key == OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER:
            kwargs["default_headers"] = {**(kwargs.get("default_headers") or {}), **opencode_zen_free_headers()}
    _apply_required_codex_headers(kwargs, access_token=api_key, base_url=base_url)
    # Hermes owns aux retry/fallback policy; the SDK default (max_retries=2) would triple
    # wall time on a hung endpoint before Hermes sees one failure.
    # Hermes owns auxiliary retry + provider/model fallback policy (the same-provider transient retry in
    # call_llm plus the except-chain fallback). The OpenAI SDK's own default (max_retries=2 → up to 3
    # attempts) silently multiplies the effective wall time of every aux call by 3× on a slow/hung endpoint,
    # so a 120s timeout can stall ~360s before Hermes sees a single failure (issue #54465). Disable
    # SDK-internal retries by default and let Hermes control the budget; explicit callers can still override
    # via kwargs.
    kwargs.setdefault("max_retries", 0)
    return OpenAI(api_key=api_key, base_url=base_url, **kwargs)


# Interrupt protection for atomic aux tasks: a compression summary killed by an ordinary
# gateway interrupt degrades to a static marker, so a thread-local flag marks such calls
# protected. Explicit host cancel (Ctrl+C, /stop) still overrides it, timeouts still fire.
# ── Interrupt protection for atomic auxiliary tasks ────────────────────── Some auxiliary tasks must NOT be
# aborted mid-flight by a gateway interrupt (e.g. an incoming user message while the agent is busy). Context
# compression is the prime case: if the summary LLM call is interrupted part-way, compression falls back to
# a static "summary unavailable" marker and the real handoff is lost (#23975). A thread-local flag lets such
# a task mark its in-flight LLM call as interrupt-protected; the Codex Responses stream's cancellation check
# honors it. TIMEOUTS still fire (a hung call must die), and all OTHER aux tasks (vision, web_extract,
# title_generation, …) remain freely interruptible.
_aux_interrupt_protection = threading.local()


class AuxiliaryExplicitCancellation(BaseException):
    """Frozen signal that an auxiliary attempt was explicitly hard-cancelled. ``BaseException`` so broad
    ``except Exception`` retry/fallback code never treats a host stop as a transport failure; ``cause``
    is immutable class data so nothing re-queries a mutable host Event after the transport unwound."""
    cause = "explicit_host_cancel"

    def __init__(self) -> None:
        super().__init__("auxiliary request explicitly cancelled by host")


def _aux_interrupt_protected() -> bool:
    return bool(getattr(_aux_interrupt_protection, "active", False))


def _aux_interrupt_cancel_requested() -> bool:
    """Return whether an explicit host cancel overrides aux protection."""
    check = _capture_aux_cancel_check()
    return _captured_aux_cancel_requested(check) if check is not None else False


@contextlib.contextmanager
def aux_interrupt_protection(active: bool = True, cancel_check=None, cancel_event=None):
    """Mark this thread's aux LLM call interrupt-protected (re-entrant-safe). ``cancel_check`` /
    ``cancel_event`` keep an explicit host hard-cancel path (Event preferred); nested scopes inherit both."""
    prev = getattr(_aux_interrupt_protection, "active", False)
    prev_cancel_check = getattr(_aux_interrupt_protection, "cancel_check", None)
    prev_cancel_event = getattr(_aux_interrupt_protection, "cancel_event", None)
    _aux_interrupt_protection.active = active
    if callable(cancel_check):
        _aux_interrupt_protection.cancel_check = cancel_check
    if cancel_event is not None and callable(getattr(cancel_event, "is_set", None)):
        _aux_interrupt_protection.cancel_event = cancel_event
    try:
        yield
    finally:
        _aux_interrupt_protection.active = prev
        _aux_interrupt_protection.cancel_check = prev_cancel_check
        _aux_interrupt_protection.cancel_event = prev_cancel_event


def _capture_aux_cancel_check() -> Optional[Callable[[], Any]]:
    """Capture the current explicit-cancel source on the owning request thread."""
    is_set = getattr(getattr(_aux_interrupt_protection, "cancel_event", None), "is_set", None)
    if callable(is_set):
        return is_set
    # Return the callable itself so attempt-local decision objects keep begin_timeout_cleanup().
    check = getattr(_aux_interrupt_protection, "cancel_check", None)
    return check if callable(check) else None


def _captured_aux_cancel_requested(cancel_check: Callable[[], Any]) -> bool:
    """Read a request-thread cancellation source without leaking its failures."""
    try:
        return bool(cancel_check())
    except Exception:
        logger.debug("captured aux cancel check failed", exc_info=True)
        return False


class _AuxiliaryCancellationDecision:
    """Atomically choose explicit cancellation or provider timeout per attempt."""

    def __init__(self, source_cancel_check: Callable[[], Any]) -> None:
        self._source_cancel_check = source_cancel_check
        self._lock = threading.Lock()
        self._outcome = "active"

    def __call__(self) -> bool:
        with self._lock:
            if self._outcome == "active" and _captured_aux_cancel_requested(self._source_cancel_check):
                self._outcome = "cancelled"
            return self._outcome == "cancelled"

    def begin_timeout_cleanup(self) -> bool:
        """Return whether timeout won and destructive cleanup is permitted."""
        with self._lock:
            if self._outcome == "active":
                cancelled = _captured_aux_cancel_requested(self._source_cancel_check)
                self._outcome = "cancelled" if cancelled else "timed_out"
            return self._outcome == "timed_out"


# Forward-progress hooks for streamed aux calls: a fixed host deadline kills a SLOW model
# streaming a big summary as hard as a HUNG one, so wire consumers tick the progress hook only
# for non-empty payloads and the host extends its deadline while tokens move. Thread-local:
# the call and its stream consumption run on the installing thread.
_aux_progress = threading.local()
_aux_dispatch = threading.local()
_aux_provider_response = threading.local()
# Absolute monotonic deadline of the waiting HOST. The stream's own ceiling
# (_aux_stream_total_ceiling, >= the host's and started later) would otherwise leave an
# orphaned stream still billing after every host-ceiling timeout.
# Absolute wall-clock deadline (time.monotonic) of the HOST waiting for this auxiliary call, when it has one
# (#99692). Liveness alone is not enough: a host also stops waiting at its own total ceiling, and the
# streamed consumer below bounds itself only by _aux_stream_total_ceiling() — a budget derived from the aux
# request timeout, which is >= the host ceiling for every configured value AND starts counting later. So the
# stream that outlives its abandoned host is not an edge case; it is the guaranteed outcome of every
# total-ceiling timeout.
_aux_stream_deadline = threading.local()


def _tick_hook(local: threading.local, label: str) -> None:
    """Call the thread-local hook installed on ``local``, if any. Never raises."""
    hook = getattr(local, "hook", None)
    if hook is None:
        return
    try:
        hook()
    except Exception:
        logger.debug("aux %s hook failed", label, exc_info=True)


def _notify_aux_progress() -> None:
    """Tick the installed forward-progress hook, if any."""
    _tick_hook(_aux_progress, "progress")


def _notify_aux_dispatch() -> None:
    """Record an actual provider dispatch without claiming response progress."""
    _tick_hook(_aux_dispatch, "dispatch")


def _notify_aux_timing_response() -> None:
    """Record a content-free frame (keepalive/empty delta): counts toward
    ``time_to_first_progress_ms`` but must not reset a compression inactivity fence."""
    _tick_hook(_aux_provider_response, "provider response")


def _notify_aux_provider_response() -> None:
    """Record a provider response/chunk, then preserve the liveness signal."""
    _notify_aux_timing_response()
    _notify_aux_progress()


def _aux_progress_active() -> bool:
    return getattr(_aux_progress, "hook", None) is not None


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Field access for wire objects that may be dicts or SDK/SimpleNamespace objects."""
    val = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return default if val is None else val


def _anthropic_event_has_content(event: Any) -> bool:
    """Whether an Anthropic stream event carries a non-empty payload."""
    event_type = _field(event, "type")
    if event_type == "content_block_delta":
        delta = _field(event, "delta")
        return any(bool(_field(delta, f)) for f in ("text", "thinking", "partial_json", "signature", "citation"))
    if event_type == "content_block_start":
        block = _field(event, "content_block")
        return _field(block, "type") == "tool_use" and any(bool(_field(block, f)) for f in ("id", "name"))
    return False


def _anthropic_aux_stream_event_hook() -> Callable[[Any], None]:
    """Per-event callback for the Anthropic aux wire: progress only for substantive payloads
    (keepalives must not keep a stalled summary alive), stop at the host deadline or explicit
    cancel. The ``TimeoutError`` text must say "timed out" so ``_is_timeout_error`` classifies it."""
    host_deadline = _current_aux_stream_deadline()
    started = time.monotonic()

    def _on_event(event: Any) -> None:
        if _anthropic_event_has_content(event):
            _notify_aux_provider_response()
        else:
            _notify_aux_timing_response()
        if _aux_interrupt_cancel_requested():
            raise AuxiliaryExplicitCancellation()
        if host_deadline is not None and time.monotonic() >= host_deadline:
            raise TimeoutError(
                "Anthropic auxiliary stream timed out at the host compression "
                f"deadline after {time.monotonic() - started:.0f}s (the caller already stopped waiting)")

    return _on_event


_CODEX_PROGRESS_DELTA_TYPES = frozenset({
    "response.output_text.delta", "response.reasoning_summary_text.delta", "response.text.delta",
    "response.audio.delta", "response.function_call_arguments.delta", "response.reasoning_text.delta",
})

# A dead stream fails at the no-progress window (first token AND between tokens); a live
# stream re-arms per event, bounded by _aux_stream_total_ceiling().
_AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS = 60.0


def _codex_event_has_content(event: Any) -> bool:
    """Whether a Codex Responses event carries a non-empty payload."""
    event_type = _field(event, "type")
    if event_type in _CODEX_PROGRESS_DELTA_TYPES:
        return bool(_field(event, "delta"))
    if event_type == "response.output_item.added":
        item = _field(event, "item")
        return "function_call" in str(_field(item, "type") or "") and any(
            bool(_field(item, f)) for f in ("id", "call_id", "name", "arguments"))
    return False


@contextlib.contextmanager
def _aux_thread_local_hook(local: threading.local, hook):
    """Install one thread-local hook, restoring the prior on exit (non-callable = passthrough)."""
    previous = getattr(local, "hook", None)
    local.hook = hook if callable(hook) else previous
    try:
        yield
    finally:
        local.hook = previous


@contextlib.contextmanager
def aux_progress_hook(hook):
    """Install *hook* as the current thread's aux forward-progress callback (None = passthrough)."""
    with _aux_thread_local_hook(_aux_progress, hook):
        yield


def _current_aux_stream_deadline() -> Optional[float]:
    """The waiting host's absolute monotonic deadline, if one is installed."""
    return getattr(_aux_stream_deadline, "value", None)


@contextlib.contextmanager
def aux_stream_deadline(deadline: Optional[float]):
    """Publish the host's absolute ``time.monotonic()`` deadline to the stream consumer.

    ``None`` is a passthrough; re-entrant-safe. Host->worker return leg of the progress hook:
    without it the isolated provider daemon streams to its own ceiling after the host stopped
    waiting, billing a summary the commit fence refuses.

    ``8207862212`` releases the compression OWNER when the fence is cancelled, but the isolated provider
    daemon (:func:`_run_protected_sync_provider_call`) that holds the socket keeps streaming to its own
    ``_aux_stream_total_ceiling`` budget — >= the host's ceiling by construction — billing an abandoned
    summary the commit fence is already guaranteed to refuse, and stacking one fresh orphan per turn on a
    session that compression never managed to shrink. See #99692.
    """
    previous = getattr(_aux_stream_deadline, "value", None)
    _aux_stream_deadline.value = deadline if isinstance(deadline, (int, float)) else previous
    try:
        yield
    finally:
        _aux_stream_deadline.value = previous


def _run_protected_sync_provider_call(callback: Callable[[dict[str, Any]], Any], kwargs: dict[str, Any]) -> Any:
    """Run one protected provider callback in an attempt-isolated daemon thread.

    Aux clients are process-shared and cannot be closed to wake one request, so the callback (incl.
    stream aggregation) runs in a daemon while the owner polls cancellation; on cancel the owner
    unwinds at once and the daemon finishes under the provider timeout in ``kwargs`` (it owns no
    transcript/commit state, never holds the session lock). Unprotected / no cancel source: direct.
    """
    source_cancel_check = _capture_aux_cancel_check()
    if not _aux_interrupt_protected() or not callable(source_cancel_check):
        return callback(kwargs)
    # One linearized outcome per attempt: the host Event is reused/cleared on later turns and
    # the Codex timeout Timer may race owner polling — same lock for both.
    cancel_check = _AuxiliaryCancellationDecision(source_cancel_check)
    if cancel_check():
        raise AuxiliaryExplicitCancellation()
    # Thread-locals do not cross into the daemon: timing hooks fire from the thread running
    # the callback, and the host deadline is inert unless carried along.
    progress_hook = getattr(_aux_progress, "hook", None)
    dispatch_hook = getattr(_aux_dispatch, "hook", None)
    provider_response_hook = getattr(_aux_provider_response, "hook", None)
    host_deadline = _current_aux_stream_deadline()
    # #99692: the stream is consumed on the daemon below, and thread-locals do not cross that boundary — an
    # owner-thread-only deadline would leave the fix inert on exactly the path large-session compression
    # takes (protected call + hard-cancel source installed).
    provider_context = contextvars.copy_context()
    done = threading.Event()
    outcome: dict[str, Any] = {}

    def _provider_worker() -> None:
        try:
            with (
                aux_progress_hook(progress_hook),
                _aux_thread_local_hook(_aux_dispatch, dispatch_hook),
                _aux_thread_local_hook(_aux_provider_response, provider_response_hook),
                aux_stream_deadline(host_deadline),
                aux_interrupt_protection(cancel_check=cancel_check),
            ):
                outcome["result"] = callback(kwargs)
        except BaseException as exc:
            outcome["exception"] = exc
        finally:
            done.set()

    threading.Thread(
        target=provider_context.run, args=(_provider_worker,), name="hermes-protected-aux-provider",
        daemon=True).start()
    while True:
        # Check cancel before AND after each wait so it wins when result publication and the
        # host Event land in the same polling interval.
        if _captured_aux_cancel_requested(cancel_check):
            raise AuxiliaryExplicitCancellation()
        if not done.wait(0.02):
            continue
        if _captured_aux_cancel_requested(cancel_check):
            raise AuxiliaryExplicitCancellation()
        exception = outcome.get("exception")
        if exception is not None:
            raise exception
        return outcome.get("result")


def _client_declares(client_obj: Any, flag: str) -> bool:
    """Whether ``client_obj`` (or its class) sets ``flag`` truthy; absent → False. Capability declaration,
    not isinstance, so out-of-tree clients can opt out of wrappers unimported (cf. SUPPORTS_HERMES_TOOL_CALLS)."""
    try:
        return bool(getattr(client_obj, flag, False))
    except Exception:
        return False


def _safe_isinstance(obj: Any, maybe_type: Any) -> bool:
    """Return False instead of raising when a patched symbol is not a type."""
    try:
        return isinstance(obj, maybe_type)
    except TypeError:
        return False


def _extract_url_query_params(url: str):
    """Extract query params from URL, return (clean_url, default_query dict or None)."""
    parsed = urlparse(url)
    if parsed.query:
        return urlunparse(parsed._replace(query="")), {k: v[0] for k, v in parse_qs(parsed.query).items()}
    return url, None


# Warn only once per process about stale OPENAI_BASE_URL.
_stale_base_url_warned = False

_PROVIDER_ALIASES = {
    "google": "gemini", "google-gemini": "gemini", "google-ai-studio": "gemini",
    "x-ai": "xai", "x.ai": "xai", "grok": "xai",
    "glm": "zai", "z-ai": "zai", "z.ai": "zai", "zhipu": "zai",
    "kimi": "kimi-coding", "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn", "moonshot-cn": "kimi-coding-cn",
    "gmi-cloud": "gmi", "gmicloud": "gmi",
    "actual-computer": "actual", "actualcomputer": "actual", "aci": "actual",
    "minimax-china": "minimax-cn", "minimax_cn": "minimax-cn",
    "claude": "anthropic", "claude-code": "anthropic",
    "github": "copilot", "github-copilot": "copilot", "github-model": "copilot", "github-models": "copilot",
    "github-copilot-acp": "copilot-acp", "copilot-acp-agent": "copilot-acp",
    "tencent": "tencent-tokenhub", "tokenhub": "tencent-tokenhub", "tencent-cloud": "tencent-tokenhub",
    "tencentmaas": "tencent-tokenhub",
    "tokenplan": "tencent-tokenplan", "tencent-lkeap": "tencent-tokenplan",
}


def _normalize_aux_provider(provider: Optional[str]) -> str:
    normalized = (provider or "auto").strip().lower()
    if normalized.startswith("custom:"):
        suffix = normalized.split(":", 1)[1].strip()
        if not suffix:
            return "custom"
        normalized = suffix
    if normalized == "codex":
        return "openai-codex"
    if normalized == "main":
        # Resolve to the actual main provider so named custom providers work.
        main_prov = (_read_main_provider() or "").strip().lower()
        if not main_prov or main_prov in {"auto", "main"}:
            return "custom"
        normalized = main_prov
    return _PROVIDER_ALIASES.get(normalized, normalized)


# Sentinel from _fixed_temperature_for_model(): callers strip ``temperature`` entirely.
# Kimi/Moonshot manage it server-side — any value can conflict with gateway mode selection.
OMIT_TEMPERATURE: object = object()


def _bare_model(model: Optional[str]) -> str:
    """Lowercased model slug with any ``vendor/`` prefix stripped."""
    return (model or "").strip().lower().rsplit("/", 1)[-1]


def _is_kimi_model(model: Optional[str]) -> bool:
    """True for any Kimi / Moonshot model that manages temperature server-side."""
    bare = _bare_model(model)
    return bare.startswith("kimi-") or bare == "kimi"


def _is_arcee_trinity_thinking(model: Optional[str]) -> bool:
    """True for Arcee Trinity Large Thinking (direct or via OpenRouter)."""
    return _bare_model(model) == "trinity-large-thinking"


# Codex OAuth hard-caps gpt-5.4/5.5/5.6 at 272K (raw API/OpenRouter expose 1.05M); the default
# 50% trigger would compact at ~136K, so raise to 85% (~231K).
_CODEX_GPT54_GPT55_COMPACTION_THRESHOLD = 0.85
# gpt-5.3-codex-spark: Codex-OAuth-only, native 128K; 70% (~90K) leaves summary headroom.
_CODEX_SPARK_COMPACTION_THRESHOLD = 0.70


def _is_codex_gpt54_or_gpt55(model: Optional[str], provider: Optional[str] = None) -> bool:
    """True for gpt-5.4/5.5/5.6 (and the Daybreak Sol alias) on the Codex OAuth route only.

    Other routes expose a larger window for the same slug and keep the user's threshold.
    Prefix-matched so ``-pro`` and dated snapshots track every 272K-capped family; ``-900k``
    picker variants are excluded. Name kept for the ``compression.codex_gpt55_autoraise`` key.
    """
    bare = _codex_route_bare_model(model, provider)
    if bare is None:
        return False
    from agent.model_metadata import is_codex_context_variant
    if is_codex_context_variant(bare):
        return False
    return bare == "gpt-daybreak-blue-latest" or any(
        bare == fam or bare.startswith(fam + "-") or bare.startswith(fam + ".")
        for fam in ("gpt-5.4", "gpt-5.5", "gpt-5.6"))


def _codex_route_bare_model(model: Optional[str], provider: Optional[str]) -> Optional[str]:
    """Lowercased bare model slug when ``provider`` is the Codex OAuth route, else None."""
    return _bare_model(model) if (provider or "").strip().lower() == "openai-codex" else None


def _is_codex_spark(model: Optional[str], provider: Optional[str] = None) -> bool:
    """True for ``gpt-5.3-codex-spark`` on the Codex OAuth route (the slug exists nowhere else)."""
    return _codex_route_bare_model(model, provider) == "gpt-5.3-codex-spark"


def _fixed_temperature_for_model(
    model: Optional[str], base_url: Optional[str] = None
) -> "Optional[float] | object":
    """``OMIT_TEMPERATURE`` (drop the key; Kimi/Moonshot), a fixed ``float``, or ``None``."""
    if _is_kimi_model(model):
        logger.debug("Omitting temperature for Kimi model %r (server-managed)", model)
        return OMIT_TEMPERATURE
    return 0.5 if _is_arcee_trinity_thinking(model) else None


def _compression_threshold_for_model(
    model: Optional[str], provider: Optional[str] = None, *,
    allow_codex_gpt55_autoraise: bool = True,
) -> Optional[float]:
    """Per-model/route compression threshold override (fraction of context used), or None.

    Arcee Trinity Large Thinking → 0.75 (preserve reasoning context); Codex-route gpt-5.4/5.5/5.6
    → 0.85, gated by ``allow_codex_gpt55_autoraise``; Codex-route gpt-5.3-codex-spark → 0.70, ungated.
    """
    if _is_arcee_trinity_thinking(model):
        return 0.75
    if allow_codex_gpt55_autoraise and _is_codex_gpt54_or_gpt55(model, provider):
        return _CODEX_GPT54_GPT55_COMPACTION_THRESHOLD
    if _is_codex_spark(model, provider):
        return _CODEX_SPARK_COMPACTION_THRESHOLD
    return None


# Aux "fast tier" families, fastest first (measured p50 titling latency). Matched as substrings
# against the LIVE /v1/models catalog because pinned ids rot; rolling "-latest" aliases lead.
_FAST_MODEL_FAMILIES: tuple = (
    "gpt-mini-latest", "gpt-nano-latest", "claude-haiku-latest", "gemini-flash-latest",
    "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5-mini", "haiku-4.5", "gemini-3.6-flash", "flash-lite",
    "-nano", "-mini", "-flash", "haiku",
)

# Disqualifiers: reasoning variants think before answering; ":batch" is a queue; ":free" tiers
# are rate-limited and slowest; embedders/modality endpoints match a rung but cannot answer.
_FAST_MODEL_EXCLUDE: tuple = (
    "thinking", "reason", "-r1", "minilm", ":batch", ":free",
    "o1-", "o3-", "o4-", "codex", "audio", "-vl", "embed",
    "-tts", "-transcribe", "-realtime", "-image", "-search-preview",
)


def _model_recency_key(model_id: str) -> tuple:
    """Sort key putting a family's newest release first: digit runs compare numerically (plain
    string order picks ``gpt-3.5-mini`` over ``gpt-5.4-mini`` and breaks at 9 vs 10)."""
    # re.split with one capturing group alternates text, number, text, …
    return tuple(
        (1, float(part), "") if index % 2 else (0, 0.0, part)
        for index, part in enumerate(re.split(r"(\d+(?:\.\d+)?)", model_id.lower())) if part)


def _fast_model_from_catalog(provider_id: str) -> str:
    """Newest ``_FAST_MODEL_FAMILIES`` match from the provider's live (cached) catalog.

    "" when the catalog is unavailable or holds no small model (caller falls through to the
    curated default). Never raises; the fetch is memory+disk cached.
    """
    is_nous = provider_id.strip().lower() == "nous"
    try:
        from hermes_cli.auth import resolve_api_key_provider_credentials
        from hermes_cli.models_pricing import fetch_models_with_pricing
        from providers import get_provider_profile
        # Most /v1/models endpoints are authenticated; an anonymous 401 would read as "no small
        # model" and pin the curated default forever.
        api_key, base_url = "", ""
        try:
            creds = resolve_api_key_provider_credentials(provider_id) or {}
            api_key = str(creds.get("api_key", "")).strip()
            base_url = str(creds.get("base_url", "")).strip()
        except Exception:
            # Not an API-key provider, or nothing configured; anonymous fetch may still work.
            logger.debug("No credentials for %s catalog", provider_id, exc_info=True)
        if not api_key and is_nous:
            # Nous is OAuth (resolver raises); anonymous reads return the full catalog.
            try:
                from hermes_cli.models_pricing import _resolve_nous_pricing_credentials
                api_key, base_url = _resolve_nous_pricing_credentials()
            except Exception:
                logger.debug("No Nous credentials for catalog", exc_info=True)
        if not base_url:
            base_url = str(getattr(get_provider_profile(provider_id), "base_url", "") or "")
        base_url = base_url.rstrip("/")
        if not base_url:
            return ""
        if base_url.endswith("/v1"):  # fetch_models_with_pricing appends /v1/models
            base_url = base_url[:-3]
        # Nous-only args must match the pickers' or the seeded cache loses sale chrome and
        # policy-catalog expiry.
        _nous_kwargs = {}
        if is_nous:
            from hermes_cli.models_pricing import _NOUS_CATALOG_TTL_SECONDS
            _nous_kwargs = {"include_sale_original": True, "cache_ttl_seconds": _NOUS_CATALOG_TTL_SECONDS}
        catalog = fetch_models_with_pricing(
            api_key=api_key or None, base_url=base_url, timeout=3.0, **_nous_kwargs) or {}
    except Exception:
        logger.debug("Fast-model catalog lookup failed for %s", provider_id, exc_info=True)
        return ""
    ids = sorted((str(m) for m in catalog), key=_model_recency_key, reverse=True)
    if is_nous:
        # Narrow catalog ids by org policy, as the pickers do.
        try:
            from hermes_cli.models_pricing import nous_policy_allowed_ids, restrict_to_nous_policy
            ids = restrict_to_nous_policy(ids, nous_policy_allowed_ids())
        except Exception:
            logger.debug("Nous policy filter unavailable", exc_info=True)
    for family in _FAST_MODEL_FAMILIES:
        for model_id in ids:
            lowered = model_id.lower()
            if family in lowered and not any(x in lowered for x in _FAST_MODEL_EXCLUDE):
                return model_id
    return ""


# Default auxiliary models for direct API-key providers (cheap/fast for side tasks)
def _get_aux_model_for_provider(provider_id: str, *, prefer_fast: bool = False) -> str:
    """Cheap auxiliary model for a provider.

    Ladder: (``prefer_fast`` only) live-catalog family match, then ``ProviderProfile.resolve_aux_model``;
    then ``default_aux_model`` (curated); then the legacy dict. ``prefer_fast`` is opt-in (titling)
    so other callers keep their static behaviour and cache keys.
    """
    profile = None
    with contextlib.suppress(Exception):
        from providers import get_provider_profile
        profile = get_provider_profile(provider_id)
    picked = ""
    if prefer_fast:
        picked = _fast_model_from_catalog(provider_id)
        if not picked and profile is not None:
            try:
                picked = profile.resolve_aux_model() or ""
            except Exception:
                logger.debug("resolve_aux_model failed for %s", provider_id, exc_info=True)
    if not picked and profile is not None and profile.default_aux_model:
        picked = profile.default_aux_model
    if not picked:
        picked = _API_KEY_PROVIDER_AUX_MODELS_FALLBACK.get(provider_id, "")
    # Rungs 2-4 are policy-blind; a blocked pick is refused at request time, so drop it and
    # let the caller keep the main model.
    if picked and provider_id.strip().lower() == "nous":
        try:
            from hermes_cli.models_pricing import nous_policy_allowed_ids, restrict_to_nous_policy
            allowed = nous_policy_allowed_ids()
            if allowed and not restrict_to_nous_policy([picked], allowed):
                return ""
        except Exception:
            logger.debug("Nous policy check unavailable", exc_info=True)
    return picked


# Fallback for providers without ProviderProfile.default_aux_model (plus some pinned here).
# New providers should set default_aux_model instead.
_API_KEY_PROVIDER_AUX_MODELS_FALLBACK: Dict[str, str] = {
    "gemini": "gemini-3.6-flash", "zai": "glm-4.5-flash", "kimi-coding": "kimi-k2-turbo-preview",
    "stepfun": "step-3.5-flash", "kimi-coding-cn": "kimi-k2-turbo-preview",
    "gmi": "google/gemini-3.1-flash-lite-preview", "anthropic": "claude-haiku-4-5-20251001",
    "ai-gateway": "google/gemini-3-flash", "opencode-zen": "gemini-3-flash", "opencode-go": "glm-5",
    "kilocode": "google/gemini-3.6-flash", "ollama-cloud": "nemotron-3-nano:30b",
    "tencent-tokenhub": "hy4-preview", "tencent-tokenplan": "hy4-preview",
    # No "deepinfra": its aux model lives on the ProviderProfile (read first).
}

# Legacy alias for callers not yet using _get_aux_model_for_provider().
_API_KEY_PROVIDER_AUX_MODELS: Dict[str, str] = _API_KEY_PROVIDER_AUX_MODELS_FALLBACK

# Tasks that may opt into ``auxiliary.<task>.prefer_fast_model``.
_FAST_MODEL_TASKS: frozenset = frozenset({"title_generation"})


def _task_prefers_fast_model(task: Optional[str]) -> bool:
    """Return whether an eligible task explicitly opts into fast-model routing."""
    return task in _FAST_MODEL_TASKS and is_truthy_value(
        _get_auxiliary_task_config(task).get("prefer_fast_model"), default=False)


# Dedicated vision models for direct providers whose main chat model differs.
_PROVIDER_VISION_MODELS: Dict[str, str] = {"xiaomi": "mimo-v2.5", "zai": "glm-5v-turbo"}


def _resolve_provider_vision_default(provider: str) -> Optional[str]:
    """Provider default vision model id, or None: static ``_PROVIDER_VISION_MODELS`` (vision-only
    names absent from any catalog) win, else ``ProviderProfile.default_vision_model()``."""
    static = _PROVIDER_VISION_MODELS.get(provider)
    if static:
        return static
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(provider)
        return profile.default_vision_model() if profile is not None else None
    except Exception:
        return None


# Endpoints that reject image input: vision auto-detect skips these to the aggregator chain
# instead of returning a client that 404s (Kimi Coding Plan Anthropic wire has no image_in).
_PROVIDERS_WITHOUT_VISION: frozenset = frozenset({"kimi-coding", "kimi-coding-cn"})

# OpenRouter app attribution (always sent). `X-Title` is what the dashboard reads.
_OR_HEADERS_BASE = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "X-OpenRouter-Categories": "productivity,cli-agent",
}


def _apply_user_default_headers(headers: dict | None) -> dict | None:
    """Merge user ``model.default_headers`` onto resolved headers (user wins; ``model.extra_headers``
    alias wins over both). Mirrors ``AIAgent._apply_user_default_headers`` so a custom endpoint behind a
    WAF rejecting ``User-Agent`` / ``X-Stainless-*`` works for aux calls. SECURITY: never log values."""
    try:
        from hermes_cli.config import cfg_get, load_config
        _cfg = load_config()
        user_headers = cfg_get(_cfg, "model", "default_headers")
        alias_headers = cfg_get(_cfg, "model", "extra_headers")
        if isinstance(alias_headers, dict) and alias_headers:
            user_headers = {**(user_headers if isinstance(user_headers, dict) else {}), **alias_headers}
    except Exception:
        return headers
    if not isinstance(user_headers, dict) or not user_headers:
        return headers
    merged = dict(headers or {})
    merged.update({str(k): str(v) for k, v in user_headers.items() if v is not None})
    return merged or headers


def build_or_headers(or_config: dict | None = None) -> dict:
    """OpenRouter headers, plus response-cache headers when enabled.

    Precedence env > config > default: ``HERMES_OPENROUTER_CACHE`` overrides
    ``openrouter.response_cache``; ``HERMES_OPENROUTER_CACHE_TTL`` (1-86400 s) overrides
    ``openrouter.response_cache_ttl``. ``or_config=None`` reads from disk.
    """
    headers = dict(_OR_HEADERS_BASE)
    if or_config is None:
        try:
            from hermes_cli.config import load_config_readonly
            or_config = load_config_readonly().get("openrouter", {})
        except Exception:
            or_config = {}
    env_cache = os.environ.get("HERMES_OPENROUTER_CACHE", "").strip().lower()
    if not (env_cache in {"1", "true", "yes", "on"} if env_cache else or_config.get("response_cache", False)):
        return headers
    headers["X-OpenRouter-Cache"] = "true"
    env_ttl = os.environ.get("HERMES_OPENROUTER_CACHE_TTL", "").strip()
    if env_ttl:
        if env_ttl.isdigit() and 1 <= int(env_ttl) <= 86400:
            headers["X-OpenRouter-Cache-TTL"] = str(int(env_ttl))
    else:
        ttl = or_config.get("response_cache_ttl", 300)
        if isinstance(ttl, (int, float)) and 1 <= ttl <= 86400:
            headers["X-OpenRouter-Cache-TTL"] = str(int(ttl))
    return headers


# NVIDIA NIM cloud billing attribution; host-gated because NVIDIA_BASE_URL may be a local NIM.
_NVIDIA_NIM_CLOUD_HEADERS = {"X-BILLING-INVOKE-ORIGIN": "HermesAgent"}


def build_nvidia_nim_headers(base_url: str | None) -> dict:
    """Return NVIDIA NIM cloud attribution headers for build.nvidia.com traffic."""
    return dict(_NVIDIA_NIM_CLOUD_HEADERS) if base_url_host_matches(str(base_url or ""), "integrate.api.nvidia.com") else {}


# Vercel AI Gateway attribution (HTTP-Referer → referrerUrl, X-Title → appName).
from hermes_cli import __version__ as _HERMES_VERSION

_AI_GATEWAY_HEADERS = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
}

# Nous Portal attribution extra_body. Tags come from agent.portal_tags so the client= marker
# tracks hermes_cli.__version__ — never inline a literal here.
from agent.portal_tags import nous_portal_tags as _nous_portal_tags


def _nous_extra_body() -> dict:
    """Fresh Nous Portal ``extra_body`` (per call, so a hot-reloaded version is reflected)."""
    return {"tags": _nous_portal_tags()}


# Set at resolve time — True if the auxiliary client points to Nous Portal
auxiliary_is_nous: bool = False

# _OPENROUTER_MODEL MUST stay a :free SKU (matching the free_only warning): this lane engages
# silently, and a paid default meant spend the user never opted into. User-configured values
# are honored untouched (_warn_paid_lane_once fires).
_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
_NOUS_MODEL = "google/gemini-3.6-flash"
_NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
_AUTH_JSON_PATH = get_hermes_home() / "auth.json"

# Hosts exposing BOTH ``…/anthropic`` and a sibling OpenAI ``…/v1``. Matched on the URL *host*
# only: unconditional rewrites break Anthropic-only gateways.
_DUAL_SURFACE_ANTHROPIC_HOST_SUFFIXES = ("minimax.io", "minimax.chat", "minimaxi.com")
_DUAL_SURFACE_ANTHROPIC_HOST_PREFIXES = ("api.minimax.",)


def _is_dual_surface_anthropic_host(url: str) -> bool:
    """True when the URL's host is a known dual-surface (MiniMax-family) host."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return any(
        host == suffix or host.endswith("." + suffix) for suffix in _DUAL_SURFACE_ANTHROPIC_HOST_SUFFIXES
    ) or any(host.startswith(prefix) for prefix in _DUAL_SURFACE_ANTHROPIC_HOST_PREFIXES)


def _to_openai_base_url(base_url: str) -> str:
    """Normalize dual-surface Anthropic URLs to their OpenAI-compatible sibling.

    MiniMax-family: ``/anthropic`` → ``/v1``; ZAI Coding Plan → ``/coding/paas/v4`` (the general
    endpoint bills separately); Kimi Code ``/coding`` → ``/coding/v1`` (the OpenAI SDK path 404s
    without it). Anthropic-only gateways keep their path.
    """
    url = str(base_url or "").strip().rstrip("/")
    if url.endswith("/anthropic"):
        if base_url_host_matches(url, "open.bigmodel.cn") or base_url_host_matches(url, "api.z.ai"):
            rewritten = url[: -len("/anthropic")] + "/coding/paas/v4"
            logger.debug("Auxiliary client: rewrote ZAI base URL %s → %s", url, rewritten)
            return rewritten
        if _is_dual_surface_anthropic_host(url):
            rewritten = url[: -len("/anthropic")] + "/v1"
            logger.debug("Auxiliary client: rewrote dual-surface base URL %s → %s", url, rewritten)
            return rewritten
        logger.debug(
            "Auxiliary client: keeping Anthropic-only base URL %s (no dual-surface host match)", url)
        return url
    if base_url_host_matches(url, "api.kimi.com") and url.endswith("/coding"):
        rewritten = url + "/v1"
        logger.debug("Auxiliary client: rewrote Kimi base URL %s → %s", url, rewritten)
        return rewritten
    return url


def _load_pool_with_credentials(provider: str, note: str = "") -> Optional[Any]:
    """``load_pool(provider)`` when it has credentials, else None (never raises)."""
    try:
        pool = load_pool(provider)
    except Exception as exc:
        logger.debug("Auxiliary client: could not load pool for %s%s: %s", provider, note, exc)
        return None
    return pool if pool and pool.has_credentials() else None


def _select_pool_entry(provider: str) -> Tuple[bool, Optional[Any]]:
    """Return (pool_exists_for_provider, selected_entry)."""
    pool = _load_pool_with_credentials(provider)
    if pool is None:
        return False, None
    try:
        return True, pool.select()
    except Exception as exc:
        logger.debug("Auxiliary client: could not select pool entry for %s: %s", provider, exc)
        return True, None


def _peek_pool_entry(provider: str) -> Optional[Any]:
    """Best-effort current/next pool entry without mutating selection order."""
    pool = _load_pool_with_credentials(provider, " (peek)")
    if pool is None:
        return None
    try:
        current_fn = getattr(pool, "current", None)
        current = current_fn() if callable(current_fn) else None
        if current is not None:
            return current
        peek_fn = getattr(pool, "peek", None)
        if callable(peek_fn):
            return peek_fn()
    except Exception as exc:
        logger.debug("Auxiliary client: could not peek pool entry for %s: %s", provider, exc)
    return None


def _pool_runtime_api_key(entry: Any) -> str:
    # runtime_api_key handles provider-specific fallback (e.g. agent_key for nous); None entry → "".
    key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    return str(key or "").strip()


def _pool_runtime_base_url(entry: Any, fallback: str = "") -> str:
    if entry is None:
        return str(fallback or "").strip().rstrip("/")
    if getattr(entry, "provider", None) == "nous":
        # Canonical auth-layer reader so the env override shares one normalization path.
        from hermes_cli.auth import _nous_inference_env_override
        env_url = _nous_inference_env_override()
        if env_url:
            return env_url
    # runtime_base_url is provider-aware; fall back for non-PooledCredential entries.
    url = (getattr(entry, "runtime_base_url", None) or getattr(entry, "inference_base_url", None)
           or getattr(entry, "base_url", None) or fallback)
    return str(url or "").strip().rstrip("/")


# Hosts the aux Anthropic path may be pointed at via model.base_url; anything else falls back
# to the Anthropic default so a foreign host never leaks in.
_ANTHROPIC_COMPATIBLE_HOSTS = frozenset({"api.anthropic.com"})


def _is_anthropic_compatible_host(url: str) -> bool:
    """True for native Anthropic hosts and gateways serving Messages under a ``/anthropic`` path
    (same convention as runtime_provider / ``_wrap_if_needed``), so a configured ``model.base_url``
    whose gateway holds auth is not discarded. A bare non-Anthropic base_url is False."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        if (parsed.hostname or "").strip().lower().rstrip(".") in _ANTHROPIC_COMPATIBLE_HOSTS:
            return True
        path = (parsed.path or "").rstrip("/").lower()
        return path.endswith("/anthropic") or path.endswith("/anthropic/v1")
    except Exception:
        return False


def _nous_min_key_ttl_seconds() -> int:
    try:
        return max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800")))
    except (TypeError, ValueError):
        return 1800


def _scoped_key_env(name: str) -> str:
    """Read a provider API key env var through the profile secret scope.

    In agent turns the scope's verdict is authoritative (a scoped miss must not borrow another
    profile's key); unscoped startup/CLI paths fall back to os.environ.
    """
    if not name:
        return ""
    with contextlib.suppress(Exception):
        from agent.secret_scope import UnscopedSecretError, get_secret
        with contextlib.suppress(UnscopedSecretError):
            return (get_secret(name) or "").strip()
    return (os.getenv(name) or "").strip()


# Codex Responses → chat.completions adapter, so aux consumers need no changes.
def _parse_codex_final_response(final: Any) -> Tuple[List[str], List[Any], Any]:
    """Split a completed Responses object into (text_parts, tool_calls, usage) in chat.completions shape."""
    text_parts: List[str] = []
    tool_calls_raw: List[Any] = []
    for item in (getattr(final, "output", None) or []):
        item_type = _field(item, "type")
        if item_type == "message":
            for part in (_field(item, "content") or []):
                if _field(part, "type") in {"output_text", "text"}:
                    text_parts.append(_field(part, "text", ""))
        elif item_type == "function_call":
            tool_calls_raw.append(SimpleNamespace(
                id=_field(item, "call_id", ""), type="function",
                function=SimpleNamespace(
                    name=_field(item, "name", ""), arguments=_field(item, "arguments", "{}"))))
    usage = None
    resp_usage = getattr(final, "usage", None)
    if resp_usage:
        def _u(key: str) -> int:
            return getattr(resp_usage, key, 0) or (resp_usage.get(key, 0) if isinstance(resp_usage, dict) else 0)
        usage = SimpleNamespace(
            prompt_tokens=_u("input_tokens"), completion_tokens=_u("output_tokens"),
            total_tokens=_u("total_tokens"))
    return text_parts, tool_calls_raw, usage


def _close_quietly(target: Any, failure_note: Optional[str]) -> None:
    """Call ``target.close()`` if present; a failure is debug-logged under ``failure_note`` (silent when None)."""
    close = getattr(target, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            if failure_note:
                logger.debug("Codex auxiliary: %s", failure_note, exc_info=True)


class _CodexStreamGuard:
    """Progress-aware deadline + FD-safe timeout watchdog for one Codex aux stream attempt.

    (1) The first substantive payload must arrive within ``no_progress_timeout`` or we fail fast into
    the caller's retry/fallback chain (a dead or keepalive-only zombie must not hold the budget);
    (2) each substantive event re-arms that window (keepalive/lifecycle frames do NOT, mirroring
    commit-fence gating) so a live stream is never killed by an absolute total; (3) a hard ceiling
    from ``_aux_stream_total_ceiling`` still terminates a pathological drip.
    """

    def __init__(self, client: Any, total_timeout: Optional[float]):
        self._client = client
        self.total_timeout = total_timeout
        self._start = time.monotonic()
        self.no_progress_timeout = _AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS
        # Progress-aware stream deadlines (supersedes the old single absolute kill at ``total_timeout``).
        # Three regimes: 1. First token: the stream must produce its first substantive payload within
        # ``no_progress_timeout`` (60s default) or we fail fast and let the caller's normal retry/fallback
        # chain run — a dead (or keepalive-only zombie) Codex stream no longer holds the full 300s
        # compression budget before falling back (masoria report, Aug 2026: 3 stacked 300s waits -> 20+ min
        # stuck on "Summarizing"). 2. Streaming: every substantive event re-arms the deadline by
        # ``no_progress_timeout`` — a live stream is never killed by an absolute total, so a long reasoning
        # summary that is actually producing tokens completes instead of timing out at 300s and falling back
        # (#54915's original complaint, fixed properly). Keepalive/lifecycle frames do NOT re-arm, mirroring
        # the commit-fence progress gating (#96707). 3. Hard ceiling: an absolute backstop from
        # ``_aux_stream_total_ceiling`` (max(600s, 4x configured timeout) — the same bound the streamed
        # chat.completions path uses) so a pathological one-token-per-59s drip still terminates.
        if total_timeout is not None:
            self.no_progress_timeout = min(self.no_progress_timeout, float(total_timeout))
        self.hard_deadline = self._start + _aux_stream_total_ceiling(total_timeout)
        # The waiting host's absolute deadline clamps the ceiling so the watchdog Timer severs
        # the socket the instant the host stops waiting — a stream blocked between events
        # can't be stopped by a per-event check.
        host_deadline = _current_aux_stream_deadline()
        if isinstance(host_deadline, (int, float)) and host_deadline < self.hard_deadline:
            self.hard_deadline = float(host_deadline)
        self._deadline_lock = threading.Lock()
        self._progress_deadline = self._start + self.no_progress_timeout
        self.saw_content = threading.Event()
        self.timed_out = threading.Event()
        # Set only when the timeout WON (not when the owner hard-cancelled first): tells the
        # owner's ``finally`` the shared client's FDs still need a real close.
        self.timeout_release_pending = threading.Event()
        self.stream_finished = threading.Event()
        self._timer = None
        # The owner may return on hard cancel while this attempt is still blocked in the SDK
        # stream. Timer threads don't inherit the worker's thread-local protection state, so
        # freeze the hard-cancel source before creating the timer.
        self._protected_cancel_check = _capture_aux_cancel_check() if _aux_interrupt_protected() else None
        self._attempt_stream_lock = threading.Lock()
        self._attempt_stream: Any = None
        # The request-driving thread owns the transport FDs — see _close_client_on_timeout.
        self._owner_tid = threading.get_ident()

    def effective_deadline(self) -> float:
        with self._deadline_lock:
            return min(self.hard_deadline, self._progress_deadline)

    def cancel_requested(self) -> bool:
        """True when the frozen hard-cancel source says the owner already cancelled."""
        check = self._protected_cancel_check
        return callable(check) and _captured_aux_cancel_requested(check)

    def adopt_stream(self, stream: Any) -> None:
        with self._attempt_stream_lock:
            self._attempt_stream = stream

    def release_stream(self, stream: Any) -> None:
        """Owner-side: close the attempt stream silently and forget it."""
        _close_quietly(stream, None)
        with self._attempt_stream_lock:
            self._attempt_stream = None

    def close_attempt_stream(self, failure_note: str) -> None:
        """Closes only this attempt's stream — never the process-shared client."""
        with self._attempt_stream_lock:
            stream = self._attempt_stream
        _close_quietly(stream, failure_note)

    def record_progress(self) -> None:
        """Substantive payload re-arms the no-progress window; the hard ceiling never moves."""
        with self._deadline_lock:
            self._progress_deadline = time.monotonic() + self.no_progress_timeout

    def timeout_message(self) -> str:
        elapsed = time.monotonic() - self._start
        if time.monotonic() >= self.hard_deadline:
            return f"Codex auxiliary Responses stream exceeded {self.hard_deadline - self._start:.1f}s hard ceiling"
        if not self.saw_content.is_set():
            return (
                "Codex auxiliary Responses stream produced no output "
                f"within {float(self.no_progress_timeout):.1f}s (no-progress timeout, {elapsed:.1f}s elapsed)")
        return (
            "Codex auxiliary Responses stream stalled: no new output "
            f"for {float(self.no_progress_timeout):.1f}s ({elapsed:.1f}s elapsed)")

    def _close_client_on_timeout(self) -> None:
        begin_timeout_cleanup = getattr(self._protected_cancel_check, "begin_timeout_cleanup", None)
        if callable(begin_timeout_cleanup):
            timeout_won = bool(begin_timeout_cleanup())
        else:
            timeout_won = not self.cancel_requested()
        # Publish transport timeout only after the attempt-local decision is fixed, so owner
        # polling cannot observe completion in between.
        self.timed_out.set()
        if not timeout_won:
            # Owner already hard-cancelled. The OpenAI client is process-shared, so never
            # close/evict it here; wake only this attempt's stream if responses.create()
            # returned one, else rely on the bounded SDK timeout.
            self.close_attempt_stream("cancelled attempt stream close during timeout failed")
            return
        # FD-ownership contract: only the thread driving the request may ``close()`` this
        # client's FDs. From a stranger thread (the watchdog Timer) only ``shutdown()`` is
        # FD-safe — ``close()`` releases the raw TLS fd while the owner's OpenSSL BIO still
        # caches it, the kernel recycles it (e.g. into a SQLite handle), and the owner's TLS
        # flush corrupts that file. The owner does the real close in its ``finally``.
        # This callback has two callers — ``_check_cancelled`` on the owning thread, and the daemon watchdog
        # ``threading.Timer``, which is a stranger thread. The owning thread performs the real close in the
        # ``finally`` below, which is where the FD release belongs. See #70773.
        self.timeout_release_pending.set()
        if threading.get_ident() == self._owner_tid:
            _close_quietly(self._client, "client close during timeout failed")
        else:
            try:
                from agent.agent_runtime_helpers import force_close_tcp_sockets
                shutdown_count = force_close_tcp_sockets(self._client)
                logger.info(
                    "Codex auxiliary client aborted (timeout, tcp_force_closed=%d, "
                    "deferred_close=stranger_thread)", shutdown_count)
            except Exception:
                logger.debug("Codex auxiliary: client abort during timeout failed", exc_info=True)
            # Socket shutdown only wakes a reader on a REAL transport; the owner may be blocked
            # inside the SDK's event stream (or a socketless test double). Closing the
            # attempt-owned stream releases it without touching shared FDs.
            self.close_attempt_stream("attempt stream close during stranger-thread timeout failed")
        # The aux client cache wraps this same client; drop the entry so the next aux call
        # doesn't reuse the dead transport and fail fast.
        try:
            # After we close the httpx transport above, the cache must drop that entry — otherwise the next
            # auxiliary call (compression retry, memory flush, etc.) reuses the dead client and fails fast
            # with a connection error. See issue #23432.
            _evict_cached_client_instance(self._client)
        except Exception:
            logger.debug("Codex auxiliary: cache eviction on timeout failed", exc_info=True)

    def check_cancelled(self) -> None:
        if self.total_timeout is not None and time.monotonic() >= self.effective_deadline():
            if not self.timed_out.is_set():
                self._close_client_on_timeout()
            raise TimeoutError(self.timeout_message())
        try:
            from tools.interrupt import is_interrupted
            # Protected atomic aux tasks (compression) must not abort on a mid-flight gateway
            # interrupt (degraded fallback marker); explicit host cancel has its own exception.
            if _aux_interrupt_cancel_requested():
                raise AuxiliaryExplicitCancellation()
            # Explicit host cancellation has its own frozen exception; timeouts above still fire and other
            # aux tasks remain interruptible. See #23975.
            if is_interrupted() and not _aux_interrupt_protected():
                raise InterruptedError("Codex auxiliary Responses stream interrupted")
        except InterruptedError:
            raise
        except Exception:
            # Interrupt state is best-effort UX; never a new failure mode.
            pass

    def _watchdog_fire(self) -> None:
        # Re-armable: if progress moved the deadline forward, reschedule instead of killing a
        # live stream.
        remaining = self.effective_deadline() - time.monotonic()
        if remaining > 0:
            if not (self.timed_out.is_set() or self.stream_finished.is_set()):
                self._arm_timer(remaining)
            return
        self._close_client_on_timeout()

    def _arm_timer(self, delay: float) -> None:
        self._timer = t = threading.Timer(delay, self._watchdog_fire)
        t.daemon = True
        t.start()

    def start(self) -> None:
        """Arm the watchdog (when a total timeout exists) and run the first cancel check."""
        if self.total_timeout:
            self._arm_timer(max(self.effective_deadline() - time.monotonic(), 0.0))
        self.check_cancelled()

    def on_event(self, _event: Any) -> None:
        # TTFP telemetry records every frame, but forward progress (compression commit fence,
        # no-progress window) counts only substantive payloads — keepalives must not re-arm,
        # so a zombie stream dies at the same window as a dead connection.
        # #93650: keep bulk wire-format payload out of the SDK's GIL-holding request transform on auxiliary
        # calls too.
        if _codex_event_has_content(_event):
            self.record_progress()
            self.saw_content.set()
            _notify_aux_provider_response()
        else:
            _notify_aux_timing_response()
        self.check_cancelled()

    def finish(self) -> None:
        """Owner ``finally``: stop the watchdog and release FDs a stranger-thread timeout only shut down."""
        self.stream_finished.set()
        if self._timer is not None:
            self._timer.cancel()
        # Gated on timeout_release_pending, NOT timed_out: after a hard-cancel the shared
        # client must stay usable for other sessions.
        if self.timeout_release_pending.is_set():
            _close_quietly(self._client, "owner-thread close after timeout failed")


class _CodexCompletionsAdapter:
    """Drop-in shim routing chat.completions.create() kwargs through Codex Responses streaming."""

    def __init__(self, real_client: OpenAI, model: str):
        self._client = real_client
        self._model = model

    def _build_responses_kwargs(self, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Any]:
        """chat.completions kwargs → Responses API kwargs, ``(resp_kwargs, model, timeout)``; mirrors codex.py::build_kwargs."""
        from utils import base_url_host_matches
        # Separate system/instructions from replayable conversation messages, then route the rest through
        # the SINGLE shared chat->Responses converter used by the main agent transport
        # (agent/transports/codex.py). Maintaining a private conversion loop here let chat-style messages
        # with role="tool" leak straight into Responses input[] — which the Responses API rejects with
        # "Invalid value: 'tool'. Supported values are: 'assistant', 'system', 'developer', and 'user'."
        # (issue #5709, hit hard by flush_memories() / compression replaying real session history that
        # includes assistant tool_calls + role="tool" results). The shared converter encodes assistant tool
        # calls as `function_call` items and tool results as `function_call_output` items with a valid
        # call_id, so every Responses path normalizes tool history identically and cannot drift.
        from agent.codex_responses_adapter import _chat_messages_to_responses_input
        model = kwargs.get("model", self._model)
        host = str(getattr(self._client, "base_url", "") or "")
        is_xai = base_url_host_matches(host, "x.ai") or base_url_host_matches(host, "api.x.ai")
        is_copilot = base_url_host_matches(host, "githubcopilot.com")
        is_github = is_copilot or base_url_host_matches(host, "models.github.ai")
        # System → ``instructions``; the rest goes through the SINGLE shared chat→Responses
        # converter (a private loop here once let role="tool" leak into input[]; the shared one
        # encodes tool history as function_call/function_call_output).
        instructions = "You are a helpful assistant."
        replay_messages: List[Dict[str, Any]] = []
        for msg in kwargs.get("messages", []):
            content = msg.get("content") or ""
            if msg.get("role", "user") == "system":
                instructions = content if isinstance(content, str) else str(content)
            else:
                replay_messages.append(msg)
        # Copilot binds replayed codex_message_items ids to a backend connection that doesn't
        # survive credential rotation (401 on replay) — same guard as build_kwargs. Aux calls
        # never send ``context_management`` (main-turn feature): no compaction checkpoint.
        # Auxiliary calls (context compression, flush_memories, MoA aggregation) go through this adapter
        # instead of agent/transports/codex.py's build_kwargs, so they need the same guard applied
        # independently. See #32716.
        input_items = _chat_messages_to_responses_input(
            replay_messages, is_github_responses=is_copilot, native_compaction_eligible=False
        )
        resp_kwargs: Dict[str, Any] = {
            # Codex only knows the base slug; strip the Hermes ``-900k`` picker suffix.
            "model": _strip_codex_ctx_variant(model), "instructions": instructions,
            "input": input_items or [{"role": "user", "content": ""}], "store": False,
        }
        # Forward the chat.completions timeout; otherwise a Codex stream can sit behind a
        # dead-looking CLI until the user force-interrupts.
        timeout = kwargs.get("timeout")
        if timeout is not None:
            resp_kwargs["timeout"] = timeout
        # Per-request HTTP headers (OpenCode session affinity, Copilot x-initiator) map to real
        # headers via the SDK kwarg — forward them.
        if isinstance(kwargs.get("extra_headers"), dict) and kwargs["extra_headers"]:
            resp_kwargs["extra_headers"] = dict(kwargs["extra_headers"])
        # The Codex endpoint rejects max_output_tokens/temperature (400) — omit.
        extra_body = kwargs.get("extra_body") or {}
        if isinstance(extra_body, dict):
            # service_tier (fast mode) is a top-level Responses field; xAI's endpoint rejects it.
            service_tier = extra_body.get("service_tier")
            if isinstance(service_tier, str) and service_tier.strip() and not is_xai:
                resp_kwargs["service_tier"] = service_tier.strip()
            reasoning_cfg = extra_body.get("reasoning")
            # ``enabled: False`` leaves reasoning/include unset (Codex still thinks by default).
            if isinstance(reasoning_cfg, dict) and reasoning_cfg.get("enabled") is not False:
                # Truthy-only: Codex 400s on e.g. {"effort": null}, so falsy → default. Shared
                # per-model clamp with the main transport ("max" is gpt-5.6-only; "minimal"/"ultra" rejected).
                from agent.reasoning_effort import clamp_effort, codex_supported_efforts
                effort = clamp_effort(reasoning_cfg.get("effort") or "medium", codex_supported_efforts(model))
                resp_kwargs["reasoning"] = {"effort": effort, "summary": "auto"}
                resp_kwargs["include"] = ["reasoning.encrypted_content"]
        tools = kwargs.get("tools")
        if tools:
            # xAI Responses rejects ``pattern``/``format`` JSON Schema keywords (400); strip for
            # chat_completion_helpers.py parity. Deep-copy first — sanitizers mutate inner dicts
            # in place and would strip the caller's tool registry.
            try:
                import copy as _copy
                from tools.schema_sanitizer import strip_pattern_and_format, strip_slash_enum
                tools = _copy.deepcopy(list(tools))
                tools, _ = strip_pattern_and_format(tools)
                tools, _ = strip_slash_enum(tools)
            except Exception as exc:
                logger.warning(
                    "Auxiliary client: failed to sanitize tool schemas for "
                    "Codex/xAI Responses path: %s", exc,
                )
            converted = []
            for t in tools:
                fn = t.get("function", {}) if isinstance(t, dict) else {}
                name = fn.get("name")
                if name:
                    converted.append({
                        "type": "function", "name": name, "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    })
            if converted:
                resp_kwargs["tools"] = converted
        # Stable prompt-cache routing: key is content-addressed from the static prefix
        # (instructions + tool schemas) so it survives across turns, scoped by the owning
        # conversation (rotation-stable logical scope, else the physical session id). Skip the
        # key where the main transport does: xAI takes it in extra_body, GitHub opts out.
        try:
            # Reuse the Responses transport's single authoritative hash algorithm and session-scope
            # normalization so equivalent static prefixes route to the same cache bucket across modes,
            # without concentrating unrelated sessions into one shared bucket (see #78941).
            from agent.transports.codex import _cache_scope_from_session_id, _content_cache_key
            from agent.transports.codex import _default_prompt_cache_retention_for_request
            if not (is_xai or is_github) and "prompt_cache_key" not in resp_kwargs:
                scope = _cache_scope_from_session_id(
                    _runtime_main_value("cache_scope") or _runtime_main_value("session_id")
                )
                cache_key = _content_cache_key(resp_kwargs["instructions"], resp_kwargs.get("tools"), scope)
                if cache_key:
                    resp_kwargs["prompt_cache_key"] = cache_key
            if "prompt_cache_retention" not in resp_kwargs:
                cache_retention = _default_prompt_cache_retention_for_request(model, host)
                if cache_retention:
                    resp_kwargs["prompt_cache_retention"] = cache_retention
        except Exception:
            logger.debug("Codex auxiliary: prompt_cache_key derivation skipped", exc_info=True)
        return resp_kwargs, model, timeout

    def create(self, **kwargs) -> Any:
        # Low-level ``responses.create(stream=True)`` and assemble the final response ourselves
        # from ``response.output_item.done``: the high-level ``responses.stream()`` rebuilds from
        # ``response.completed.response.output``, which Codex returns as ``null`` (SDK crash).
        resp_kwargs, model, timeout = self._build_responses_kwargs(kwargs)
        total_timeout = timeout if isinstance(timeout, (int, float)) and timeout > 0 else None
        guard = _CodexStreamGuard(self._client, total_timeout)
        try:
            guard.start()
            from agent.codex_runtime import _bypass_sdk_request_transform, _consume_codex_event_stream
            # Keep bulk wire payload out of the SDK's GIL-holding request transform.
            stream_kwargs = _bypass_sdk_request_transform({**resp_kwargs, "stream": True})
            event_stream = self._client.responses.create(**stream_kwargs)
            guard.adopt_stream(event_stream)
            # The timer may fire while responses.create() is blocked; if the cancelled attempt
            # had no stream to close then, close it now that it is attempt-owned — never the shared client.
            if guard.timed_out.is_set() and guard.cancel_requested():
                guard.close_attempt_stream("late cancelled attempt stream close failed")
            try:
                # Some Codex-compatible hosts accept ``stream=True`` but return a completed
                # Responses object (not iterable) — don't hand it to the consumer.
                if hasattr(event_stream, "output"):
                    final = event_stream
                else:
                    final = _consume_codex_event_stream(
                        event_stream, model=str(resp_kwargs.get("model") or model), on_event=guard.on_event
                    )
            finally:
                guard.release_stream(event_stream)
            if final is None:
                raise RuntimeError("Codex auxiliary Responses stream did not return a final response")
            text_parts, tool_calls_raw, usage = _parse_codex_final_response(final)
        except Exception as exc:
            if guard.timed_out.is_set():
                raise TimeoutError(guard.timeout_message()) from exc
            logger.debug("Codex auxiliary Responses API call failed: %s", exc)
            raise
        finally:
            guard.finish()
        # Shape the result like chat.completions.
        message = SimpleNamespace(
            role="assistant", content="".join(text_parts).strip() or None,
            tool_calls=tool_calls_raw or None,
        )
        choice = SimpleNamespace(
            index=0, message=message, finish_reason="stop" if not tool_calls_raw else "tool_calls"
        )
        return SimpleNamespace(choices=[choice], model=model, usage=usage)


class _ChatShim:
    """Exposes ``client.chat.completions.create()`` over a sync or async adapter."""

    def __init__(self, adapter: Any):
        self.completions = adapter


class _AsyncCompletionsAdapter:
    """Async adapter: runs the sync adapter's ``create`` via asyncio.to_thread()."""

    def __init__(self, sync_adapter: Any):
        self._sync = sync_adapter

    async def create(self, **kwargs) -> Any:
        import asyncio
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncAuxiliaryClientBase:
    """Async-compatible wrapper matching AsyncOpenAI.chat.completions.create().

    Mirrors ``_real_client`` (when the sync wrapper has one) so cache eviction by
    leaf OpenAI client drops this async entry too instead of reusing a closed transport.
    """

    def __init__(self, sync_wrapper: Any):
        self.chat = _ChatShim(_AsyncCompletionsAdapter(sync_wrapper.chat.completions))
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url
        if hasattr(sync_wrapper, "_real_client"):
            # Mirror the sync wrapper's _real_client so cache eviction by leaf OpenAI client (e.g.
            # _close_client_on_timeout in #23482) drops this async entry too. Without this, sync and async
            # cache entries diverge on poisoning: the sync entry is evicted but the async entry keeps
            # reusing the closed transport, failing every subsequent async aux call with 'Connection error'
            # until the gateway restarts.
            self._real_client = sync_wrapper._real_client


_AsyncAnthropicCompletionsAdapter = _AsyncCompletionsAdapter  # imported by tests


class CodexAuxiliaryClient:
    """OpenAI-client-compatible wrapper routing through the Codex Responses API (.api_key/.base_url for introspection)."""

    def __init__(self, real_client: OpenAI, model: str):
        self._real_client = real_client
        self.chat = _ChatShim(_CodexCompletionsAdapter(real_client, model))
        self.api_key = real_client.api_key
        self.base_url = real_client.base_url

    def close(self):
        self._real_client.close()


class AsyncCodexAuxiliaryClient(_AsyncAuxiliaryClientBase):
    pass


def _translate_anthropic_response_format(anthropic_kwargs: Dict[str, Any], response_format: Any) -> None:
    """Merge an OpenAI response format into Anthropic ``output_config``."""
    if not isinstance(response_format, dict):
        return
    format_type = response_format.get("type")
    if format_type == "json_schema":
        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict) or "schema" not in json_schema:
            return
        schema = json_schema["schema"]
    elif format_type == "json_object":
        # Anthropic SDK has no schema-less JSON mode; only ``json_schema``.
        schema = {"type": "object"}
    else:
        return
    output_config = anthropic_kwargs.get("output_config")
    if not isinstance(output_config, dict):
        output_config = {}
        anthropic_kwargs["output_config"] = output_config
    output_config["format"] = {"type": "json_schema", "schema": schema}


class _AnthropicCompletionsAdapter:
    """OpenAI-client-compatible adapter for Anthropic Messages API."""

    def __init__(self, real_client: Any, model: str, is_oauth: bool = False, base_url: str | None = None):
        self._client = real_client
        self._model = model
        self._is_oauth = is_oauth
        # Caller URL first; fall back to the SDK client's host only for Nous Portal — a blanket
        # fallback would flip MiniMax/Zhipu aux adapters to third-party handling (strips thinking sigs).
        self._base_url = base_url or None
        if not self._base_url:
            candidate = str(getattr(real_client, "base_url", "") or "") or None
            if candidate:
                with contextlib.suppress(Exception):
                    from agent.anthropic_endpoints import _is_nous_portal_endpoint
                    if _is_nous_portal_endpoint(candidate):
                        self._base_url = candidate

    def create(self, **kwargs) -> Any:
        from agent.anthropic_adapter import build_anthropic_kwargs, create_anthropic_message
        from agent.transports import get_transport
        model = kwargs.get("model", self._model)
        # ZAI's Anthropic endpoint rejects max_tokens on vision models (code 1210);
        # callers signal this via _skip_zai_max_tokens.
        if kwargs.pop("_skip_zai_max_tokens", False):
            max_tokens = None
        else:
            max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        temperature = kwargs.get("temperature")
        # Reasoning priority: explicit per-call _reasoning_config (MoA per-slot) wins over
        # extra_body.reasoning; build_anthropic_kwargs translates to ``thinking``.
        reasoning_cfg = kwargs.get("_reasoning_config")
        if reasoning_cfg is None:
            _eb = kwargs.get("extra_body")
            _rc = _eb.get("reasoning") if isinstance(_eb, dict) else None
            if isinstance(_rc, dict):
                reasoning_cfg = _rc
        # OpenAI tool_choice (str or dict) → Anthropic-style name/mode string.
        tool_choice = kwargs.get("tool_choice")
        if isinstance(tool_choice, dict):
            choice_type = str(tool_choice.get("type", "")).lower()
            if choice_type == "function":
                tool_choice = tool_choice.get("function", {}).get("name")
            else:
                tool_choice = choice_type if choice_type in {"auto", "required", "none"} else None
        elif not isinstance(tool_choice, str):
            tool_choice = None
        anthropic_kwargs = build_anthropic_kwargs(
            model=model, messages=kwargs.get("messages", []), tools=kwargs.get("tools"),
            max_tokens=max_tokens, reasoning_config=reasoning_cfg, tool_choice=tool_choice,
            is_oauth=self._is_oauth,
            # Portal routes on ``anthropic/<slug>`` ids and replays signed thinking
            # keyed off base_url; omitting it breaks Portal model resolution.
            base_url=self._base_url,
        )
        # Opus 4.7+ rejects non-default temperature/top_p/top_k; build_anthropic_kwargs
        # also strips these as a safety net — keep both layers.
        if temperature is not None:
            from agent.anthropic_adapter import _forbids_sampling_params
            if not _forbids_sampling_params(model):
                anthropic_kwargs["temperature"] = temperature
        # Per-request HTTP headers (OpenCode session affinity) — the Anthropic SDK accepts
        # ``extra_headers`` on messages.create/stream too.
        if isinstance(kwargs.get("extra_headers"), dict) and kwargs["extra_headers"]:
            anthropic_kwargs["extra_headers"] = {
                **(anthropic_kwargs.get("extra_headers") or {}),
                **kwargs["extra_headers"],
            }
        # response_format: top-level gets the same translation as the extra_body form; when both
        # are present the extra_body form wins. Passthrough excludes ``reasoning``/``response_format``
        # (already TRANSLATED to native fields — raw would 400 on strict gateways) and ``_`` Hermes plumbing.
        # The adapter builds the Messages body from a fixed allow-list of kwargs, so before this an
        # unrecognized top-level kwarg was dropped on the floor: the request succeeded but the schema
        # contract silently became prompt compliance (#85626 review, point 2).
        top_level_response_format = kwargs.get("response_format")
        if top_level_response_format is not None:
            _translate_anthropic_response_format(anthropic_kwargs, top_level_response_format)
        caller_extra_body = kwargs.get("extra_body")
        if caller_extra_body and isinstance(caller_extra_body, dict):
            _translate_anthropic_response_format(anthropic_kwargs, caller_extra_body.get("response_format"))
            passthrough = {
                k: v for k, v in caller_extra_body.items()
                if k not in {"reasoning", "response_format"} and not str(k).startswith("_")
            }
            if passthrough:
                existing = anthropic_kwargs.get("extra_body") or {}
                if not isinstance(existing, dict):
                    existing = {}
                anthropic_kwargs["extra_body"] = {**existing, **passthrough}
        response = create_anthropic_message(
            self._client,
            anthropic_kwargs,
            # Record provider-response timing every event, but tick forward progress only for
            # substantive payloads so keepalives can't hold a stalled summary open. None keeps
            # the fast get_final_message path.
            on_stream_event=(_anthropic_aux_stream_event_hook() if _aux_progress_active() else None),
        )
        _nr = get_transport("anthropic_messages").normalize_response(response, strip_tool_prefix=self._is_oauth)
        usage = None
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "input_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "output_tokens", 0) or 0
            usage = SimpleNamespace(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                total_tokens=getattr(response.usage, "total_tokens", 0) or (prompt_tokens + completion_tokens),
            )
        # ToolCall already duck-types as OpenAI shape via properties.
        choice = SimpleNamespace(
            index=0,
            message=SimpleNamespace(content=_nr.content, tool_calls=_nr.tool_calls, reasoning=_nr.reasoning),
            finish_reason=_nr.finish_reason,
        )
        return SimpleNamespace(choices=[choice], model=model, usage=usage)


class AnthropicAuxiliaryClient:
    """OpenAI-client-compatible wrapper over a native Anthropic client."""

    def __init__(self, real_client: Any, model: str, api_key: str, base_url: str, is_oauth: bool = False):
        self._real_client = real_client
        self.chat = _ChatShim(_AnthropicCompletionsAdapter(real_client, model, is_oauth=is_oauth, base_url=base_url))
        self.api_key = api_key
        self.base_url = base_url

    def close(self):
        close_fn = getattr(self._real_client, "close", None)
        if callable(close_fn):
            close_fn()


class AsyncAnthropicAuxiliaryClient(_AsyncAuxiliaryClientBase):
    pass


class _BedrockCompletionsAdapter:
    """Translates ``chat.completions.create(**kwargs)`` into Bedrock Converse."""

    def __init__(self, region: str, model: str):
        self._region = region
        self._model = model

    def create(self, **kwargs) -> Any:
        from agent.bedrock_adapter import call_converse
        model = kwargs.get("model", self._model)
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        # OpenAI accepts ``stop`` as str or list; Converse requires a list.
        stop = kwargs.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        if kwargs.get("tool_choice") is not None:
            # Converse toolChoice isn't wired through call_converse(); surface the drop.
            logger.debug(
                "BedrockAuxiliaryClient: tool_choice=%r not supported by the "
                "Converse shim — ignored.", kwargs.get("tool_choice"),
            )
        if kwargs.get("stream"):
            # Converse streaming isn't wired here; call_llm's streaming consumer
            # detects a final object and downgrades to non-live output.
            logger.debug(
                "BedrockAuxiliaryClient: stream=True requested for %s — returning a complete response "
                "(Converse shim does not stream); caller downgrades to non-streaming.", model,
            )
        response = call_converse(
            region=self._region, model=model, messages=kwargs.get("messages", []), tools=kwargs.get("tools"),
            # Omitted/None cap → None so Bedrock uses the model max (no-cap-by-default like
            # every other aux wire). Truthiness mirrors the Anthropic shim: explicit 0 = "no cap".
            max_tokens=int(max_tokens) if max_tokens else None, temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"), stop_sequences=stop,
        )
        # Converse is complete-response here: mark provider progress only after
        # return so TTFP reflects real Bedrock latency, not dispatch/setup.
        _notify_aux_provider_response()
        return response


class BedrockAuxiliaryClient:
    """OpenAI-client-compatible wrapper over AWS Bedrock Converse API."""

    def __init__(self, region: str, model: str):
        self._region = region
        self._model = model
        self.chat = _ChatShim(_BedrockCompletionsAdapter(region, model))
        self.api_key = "aws-sdk"
        self.base_url = f"https://bedrock-runtime.{region}.amazonaws.com"

    def close(self):
        pass


class AsyncBedrockAuxiliaryClient(_AsyncAuxiliaryClientBase):
    pass


def _endpoint_speaks_anthropic_messages(base_url: str) -> bool:
    """True if ``base_url`` speaks Anthropic Messages, not OpenAI chat.completions.

    Mirrors ``hermes_cli.runtime_provider._detect_api_mode_for_url`` so aux and main agree: any
    ``/anthropic`` URL (MiniMax, Zhipu, LiteLLM), ``api.kimi.com/coding`` (chat 404s), ``api.anthropic.com``.
    """
    normalized = (base_url or "").strip().lower().rstrip("/")
    if not normalized:
        return False
    if urlparse(normalized).path.rstrip("/").endswith(("/anthropic", "/anthropic/v1")):
        return True
    hostname = base_url_hostname(normalized)
    return hostname == "api.anthropic.com" or bool(hostname == "api.kimi.com" and "/coding" in normalized)


def _maybe_wrap_anthropic(
    client_obj: Any, model: str, api_key: str, base_url: str, api_mode: Optional[str] = None
) -> Any:
    """Rewrap a plain OpenAI client in ``AnthropicAuxiliaryClient`` when the endpoint speaks Anthropic Messages.

    Single transport-correction chokepoint at the end of every ``resolve_provider_client`` branch; returns
    ``client_obj`` unchanged for probe stubs/specialized adapters, OpenAI-wire, explicit non-Anthropic
    ``api_mode``, or missing ``anthropic`` SDK.
    """
    # Anthropic/Bedrock/Codex wrappers, plus any client declaring HERMES_SKIP_TRANSPORT_WRAP
    # (native/ACP shims, in-tree or plugin), must never be re-dispatched through a wire adapter —
    # a class-attribute declaration rather than isinstance so this hot path never imports them.
    if (
        isinstance(client_obj, _AuxProbeClientStub)
        or _safe_isinstance(client_obj, (AnthropicAuxiliaryClient, BedrockAuxiliaryClient, CodexAuxiliaryClient))
        or _client_declares(client_obj, "HERMES_SKIP_TRANSPORT_WRAP")
    ):
        return client_obj
    # Explicit non-anthropic api_mode wins over URL heuristics.
    if api_mode != "anthropic_messages" and (api_mode or not _endpoint_speaks_anthropic_messages(base_url)):
        return client_obj
    try:
        from agent.anthropic_adapter import build_anthropic_client
    except ImportError:
        logger.warning(
            "Endpoint %s speaks Anthropic Messages but the anthropic SDK is "
            "not installed — falling back to OpenAI-wire (will likely 404).",
            base_url,
        )
        return client_obj
    try:
        real_client = build_anthropic_client(api_key, base_url)
    except Exception as exc:
        logger.warning(
            "Failed to build Anthropic client for %s (%s) — falling back to "
            "OpenAI-wire client.", base_url, exc,
        )
        return client_obj
    logger.debug(
        "Auxiliary transport: wrapping client in AnthropicAuxiliaryClient "
        "(model=%s, base_url=%s, api_mode=%s)",
        model, base_url[:60] if base_url else "", api_mode or "auto-detected",
    )
    return AnthropicAuxiliaryClient(real_client, model, api_key, base_url, is_oauth=False)


def _read_nous_auth() -> Optional[dict]:
    """Nous provider state dict from the credential pool or ~/.hermes/auth.json; None when not active with tokens."""
    pool_present, entry = _select_pool_entry("nous")
    if pool_present:
        if entry is None:
            return None
        return {
            "access_token": getattr(entry, "access_token", ""),
            "refresh_token": getattr(entry, "refresh_token", None),
            "agent_key": getattr(entry, "agent_key", None),
            "inference_base_url": _pool_runtime_base_url(entry, _NOUS_DEFAULT_BASE_URL),
            "portal_base_url": getattr(entry, "portal_base_url", None),
            "client_id": getattr(entry, "client_id", None),
            "scope": getattr(entry, "scope", None),
            "token_type": getattr(entry, "token_type", "Bearer"),
            "source": "pool",
        }
    try:
        if not _AUTH_JSON_PATH.is_file():
            return None
        data = json.loads(_AUTH_JSON_PATH.read_text(encoding="utf-8-sig"))
        if data.get("active_provider") != "nous":
            return None
        provider = data.get("providers", {}).get("nous", {})
        # Must have at least an access_token or agent_key.
        if not provider.get("agent_key") and not provider.get("access_token"):
            return None
        return provider
    except Exception as exc:
        logger.debug("Could not read Nous auth: %s", exc)
        return None


def _nous_api_key(provider: dict) -> str:
    """Extract a usable Nous inference JWT from stored auth state."""
    from hermes_cli.auth import _nous_invoke_jwt_is_usable
    for token_key, expiry_key in (("agent_key", "agent_key_expires_at"), ("access_token", "expires_at")):
        token = provider.get(token_key)
        if not isinstance(token, str) or not token.strip():
            continue
        if _nous_invoke_jwt_is_usable(token, scope=provider.get("scope"), expires_at=provider.get(expiry_key)):
            return token
    return ""


def _resolve_nous_pool_runtime_api(*, force_refresh: bool = False) -> Optional[tuple[str, str]]:
    """Resolve Nous auxiliary credentials from the selected pool entry."""
    try:
        from hermes_cli.auth import _agent_key_is_usable
        pool = load_pool("nous")
    except Exception as exc:
        logger.debug("Auxiliary Nous pool credential resolution failed: %s", exc)
        return None
    if not pool or not pool.has_credentials():
        return None
    try:
        entry = pool.select()
    except Exception as exc:
        logger.debug("Auxiliary Nous pool selection failed: %s", exc)
        return None
    if entry is None:
        return None

    def _entry_state(e: Any) -> Dict[str, Any]:
        return {k: getattr(e, k, None) for k in (
            "agent_key", "agent_key_expires_at", "access_token", "expires_at", "scope")}

    if force_refresh or not _agent_key_is_usable(_entry_state(entry), _nous_min_key_ttl_seconds()):
        try:
            refreshed = pool.try_refresh_current()
        except Exception as exc:
            logger.debug("Auxiliary Nous pool refresh failed: %s", exc)
            refreshed = None
        if refreshed is None:
            return None
        entry = refreshed
    api_key = _nous_api_key(_entry_state(entry))
    base_url = _pool_runtime_base_url(entry, _NOUS_DEFAULT_BASE_URL)
    if not api_key or not base_url:
        return None
    return api_key, base_url


def _resolve_nous_runtime_api(
    *, force_refresh: bool = False, stale_access_token: Optional[str] = None
) -> Optional[tuple[str, str]]:
    """Fresh Nous runtime credentials (pool first, then auth store + JWT refresh) — mirrors the main
    agent's 401 recovery. ``stale_access_token`` is the bearer that just 401'd; with ``force_refresh``
    it lets the auth store adopt a sibling process's rotation instead of re-POSTing the shared grant."""
    pooled = _resolve_nous_pool_runtime_api(force_refresh=force_refresh)
    if pooled is not None:
        return pooled
    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials
        creds = resolve_nous_runtime_credentials(
            timeout_seconds=env_float("HERMES_NOUS_TIMEOUT_SECONDS", 15),
            force_refresh=force_refresh,
            stale_access_token=stale_access_token or None,
        )
    except Exception as exc:
        logger.debug("Auxiliary Nous runtime credential resolution failed: %s", exc)
        return None
    return _creds_pair(creds)


def _creds_pair(creds: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """``(api_key, base_url)`` from a runtime-credentials dict, or None when either is missing."""
    api_key = str(creds.get("api_key") or "").strip()
    base_url = str(creds.get("base_url") or "").strip().rstrip("/")
    if not api_key or not base_url:
        return None
    return api_key, base_url


def _resolve_xai_oauth_for_aux() -> Optional[Tuple[str, str]]:
    """Fresh xAI OAuth (api_key, base_url) for aux clients, or None.

    Pool first (some xAI OAuth logins exist only as pool entries), then the singleton auth-store resolver.
    """
    try:
        from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL, _xai_validate_inference_base_url
        pool = load_pool("xai-oauth")
        if pool and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                api_key = str(
                    getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "") or ""
                ).strip()
                _url = lambda v: str(v or "").strip().rstrip("/")  # noqa: E731
                base_url = _xai_validate_inference_base_url(
                    _url(os.getenv("HERMES_XAI_BASE_URL", ""))
                    or _url(os.getenv("XAI_BASE_URL", ""))
                    or _url(getattr(entry, "runtime_base_url", None))
                    or _url(getattr(entry, "base_url", None)),
                    fallback=DEFAULT_XAI_OAUTH_BASE_URL,
                )
                if api_key and base_url:
                    return api_key, base_url
    except Exception as exc:
        logger.debug("Auxiliary xAI OAuth pool credential resolution failed: %s", exc)
    try:
        from hermes_cli.auth import resolve_xai_oauth_runtime_credentials
        creds = resolve_xai_oauth_runtime_credentials()
    except Exception as exc:
        logger.debug("Auxiliary xAI OAuth runtime credential resolution failed: %s", exc)
        return None
    return _creds_pair(creds)


def _read_codex_access_token() -> Optional[str]:
    """Valid, non-expired Codex OAuth access token; an exhausted pool falls back to the profile's auth.json token."""
    pool_present, entry = _select_pool_entry("openai-codex")
    if pool_present:
        token = _pool_runtime_api_key(entry)
        if token:
            return token
    try:
        from hermes_cli.auth import _read_codex_tokens
        access_token = _read_codex_tokens().get("tokens", {}).get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            return None
        # Expired JWTs would block the auto chain and prevent fallback to working providers.
        try:
            import base64
            payload = access_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp", 0)
            if exp and time.time() > exp:
                logger.debug("Codex access token expired (exp=%s), skipping", exp)
                return None
        except Exception:
            pass  # Non-JWT token or decode error — use as-is
        return access_token.strip()
    except Exception as exc:
        logger.debug("Could not read Codex auth for auxiliary client: %s", exc)
        return None


def _resolve_api_key_provider() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Try each API-key provider in PROVIDER_REGISTRY order; (client, model) or (None, None)."""
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials
    except ImportError:
        logger.debug("Could not import PROVIDER_REGISTRY for API-key fallback")
        return None, None
    for provider_id, pconfig in PROVIDER_REGISTRY.items():
        if pconfig.auth_type != "api_key":
            continue
        if _is_provider_unhealthy(provider_id):
            logger.debug("Auxiliary api-key chain: %s is unhealthy, skipping", provider_id)
            continue
        if provider_id == "anthropic":
            # Explicit-config gate: Claude Code credentials must not silently become aux fallback.
            with contextlib.suppress(ImportError):
                from hermes_cli.auth import is_provider_explicitly_configured
                if not is_provider_explicitly_configured("anthropic"):
                    continue
            return _try_anthropic()
        pool_present, entry = _select_pool_entry(provider_id)
        if pool_present:
            api_key = _pool_runtime_api_key(entry)
            if not api_key:
                continue
            raw_base_url = _pool_runtime_base_url(entry, pconfig.inference_base_url) or pconfig.inference_base_url
            via = " via pool"
        else:
            creds = resolve_api_key_provider_credentials(provider_id)
            api_key = str(creds.get("api_key", "")).strip()
            if not api_key:
                continue
            raw_base_url = str(creds.get("base_url", "")).strip().rstrip("/") or pconfig.inference_base_url
            via = ""
        model = _get_aux_model_for_provider(provider_id) or None
        if model is None:
            continue  # skip provider if we don't know a valid aux model
        logger.debug("Auxiliary text client: %s (%s)%s", pconfig.name, model, via)
        # Native Gemini, else OpenAI-wire + Anthropic rewrap.
        base_url = _to_openai_base_url(raw_base_url)
        if provider_id == "gemini":
            from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url
            if is_native_gemini_base_url(base_url):
                return GeminiNativeClient(api_key=api_key, base_url=base_url), model
        if base_url_host_matches(base_url, "api.kimi.com"):
            headers = {"User-Agent": "claude-code/0.1.0"}
        elif base_url_host_matches(base_url, "githubcopilot.com"):
            from hermes_cli.models import copilot_default_headers
            headers = copilot_default_headers()
        elif base_url_host_matches(base_url, "integrate.api.nvidia.com"):
            headers = build_nvidia_nim_headers(base_url)
        else:
            headers = _profile_default_headers(provider_id)
        extra = {"default_headers": headers} if headers else {}
        merged = _apply_user_default_headers(extra.get("default_headers"))
        if merged:
            extra["default_headers"] = merged
        client = _create_openai_client(api_key=api_key, base_url=base_url, **extra)
        return _maybe_wrap_anthropic(client, model, api_key, raw_base_url), model
    return None, None


def _endpoint_default_headers(
    base_url: str, provider: str, *, is_vision: bool = False, xai: bool = False,
) -> Optional[dict]:
    """Provider-specific client headers by endpoint host, merged with user ``model.default_headers``.

    Kimi Code needs the claude-code User-Agent; Copilot needs its request headers
    (``is_vision`` adds Copilot-Vision-Request); NVIDIA NIM and (optionally) xAI have
    their own fingerprints; anything else falls back to the provider profile.
    """
    if base_url_host_matches(base_url, "api.kimi.com"):
        headers: dict = {"User-Agent": "claude-code/0.1.0"}
    elif base_url_host_matches(base_url, "githubcopilot.com"):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = dict(copilot_request_headers(is_agent_turn=True, is_vision=is_vision))
    elif base_url_host_matches(base_url, "integrate.api.nvidia.com"):
        headers = dict(build_nvidia_nim_headers(base_url))
    elif xai and base_url_host_matches(base_url, "x.ai"):
        from tools.xai_http import hermes_xai_default_headers
        headers = dict(hermes_xai_default_headers())
    else:
        headers = _profile_default_headers(provider) or {}
    return _apply_user_default_headers(headers or None) or None


def _profile_default_headers(provider: str) -> Optional[dict]:
    """Client-level attribution headers from the provider profile (e.g. GMI User-Agent), or None."""
    if not provider:
        return None
    with contextlib.suppress(Exception):
        from providers import get_provider_profile
        profile = get_provider_profile(provider)
        if profile and profile.default_headers:
            return dict(profile.default_headers)
    return None


# Provider resolution helpers

_paid_lane_warned: set = set()


def _is_free_model(model: Optional[str]) -> bool:
    """True when ``model`` is a free SKU (``:free`` suffix or ``stealth/`` prefix) — naming-convention trust."""
    if not model:
        return False
    normalized = str(model).strip()
    return normalized.endswith(":free") or normalized.startswith("stealth/")


def _aux_openrouter_settings() -> Tuple[bool, str]:
    """Read (free_only, openrouter_model) from config; (False, _OPENROUTER_MODEL) on failure."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        cfg = load_config_readonly()
        free_only = bool(cfg_get(cfg, "auxiliary", "free_only", default=False))
        val = cfg_get(cfg, "auxiliary", "openrouter_model")
        model = val.strip() if isinstance(val, str) and val.strip() else _OPENROUTER_MODEL
        return free_only, model
    except Exception:
        return False, _OPENROUTER_MODEL


def _warn_paid_lane_once(model: str) -> None:
    """Log a WARNING the first time a non-free OpenRouter model is engaged."""
    if model in _paid_lane_warned:
        return
    _paid_lane_warned.add(model)
    logger.warning(
        "Auxiliary client: PAID lane engaged for auxiliary task — OpenRouter fallback model %r is not "
        "a :free SKU and may incur real spend. Set auxiliary.free_only: true to restrict auxiliary "
        "fallbacks to free models, or auxiliary.openrouter_model to a :free model.", model,
    )


def _try_openrouter(explicit_api_key: str = None, model: str = None) -> Tuple[Optional[OpenAI], Optional[str]]:
    free_only, cfg_model = _aux_openrouter_settings()
    or_model = model or cfg_model
    if free_only and not _is_free_model(or_model):
        logger.warning(
            "Auxiliary client: auxiliary.free_only is enabled but the OpenRouter fallback model %r is "
            "not a :free SKU — skipping the OpenRouter fallback. Set auxiliary.openrouter_model to a "
            ":free model (e.g. nvidia/nemotron-3-ultra-550b-a55b:free) or disable auxiliary.free_only.",
            or_model,
        )
        return None, None
    if not _is_free_model(or_model):
        _warn_paid_lane_once(or_model)
    pool_present, entry = _select_pool_entry("openrouter")
    if pool_present:
        or_key = explicit_api_key or _pool_runtime_api_key(entry)
        if or_key:
            base_url = _pool_runtime_base_url(entry, OPENROUTER_BASE_URL) or OPENROUTER_BASE_URL
            logger.debug("Auxiliary client: OpenRouter via pool")
            return _create_openai_client(
                api_key=or_key, base_url=base_url, default_headers=build_or_headers()
            ), or_model
        # Exhausted pool: fall through to OPENROUTER_API_KEY rather than fail.
        logger.debug("Auxiliary client: OpenRouter pool exhausted, trying OPENROUTER_API_KEY")
    or_key = explicit_api_key or _scoped_key_env("OPENROUTER_API_KEY")
    if not or_key:
        _mark_provider_unhealthy("openrouter", ttl=60)
        return None, None
    logger.debug("Auxiliary client: OpenRouter")
    return _create_openai_client(
        api_key=or_key, base_url=OPENROUTER_BASE_URL, default_headers=build_or_headers()
    ), or_model


def _describe_openrouter_unavailable(model: str = None) -> str:
    """Return the policy or credential reason OpenRouter was unavailable."""
    free_only, cfg_model = _aux_openrouter_settings()
    or_model = model or cfg_model
    if free_only and not _is_free_model(or_model):
        return (
            f"auxiliary.free_only rejected non-free model {or_model!r}; "
            "the request was skipped before provider availability checks"
        )
    pool_present, entry = _select_pool_entry("openrouter")
    if pool_present:
        if entry is None:
            return "OpenRouter credential pool has no usable entries (credentials may be exhausted)"
        if not _pool_runtime_api_key(entry):
            return "OpenRouter credential pool entry is missing a runtime API key"
    if not _scoped_key_env("OPENROUTER_API_KEY"):
        return "OPENROUTER_API_KEY not set"
    return "no usable OpenRouter credentials found"


def _try_nous(vision: bool = False) -> Tuple[Optional[OpenAI], Optional[str]]:
    # Cross-session rate guard: another session's 429 means skip Nous rather than pile onto the tapped RPH bucket.
    with contextlib.suppress(Exception):
        from agent.nous_rate_guard import nous_rate_limit_remaining
        _remaining = nous_rate_limit_remaining()
        if _remaining is not None and _remaining > 0:
            logger.debug("Auxiliary: skipping Nous Portal (rate-limited, resets in %.0fs)", _remaining)
            _mark_provider_unhealthy("nous", ttl=_remaining)
            return None, None
    nous = _read_nous_auth()
    runtime = _resolve_nous_runtime_api(force_refresh=False)
    if runtime is None and not nous:
        logger.warning("Auxiliary Nous client unavailable: no Nous authentication found (run: hermes auth).")
        _mark_provider_unhealthy("nous", ttl=60)
        return None, None
    if runtime is None and nous:
        logger.debug("Auxiliary Nous: runtime JWT refresh failed; checking stored auth.json token.")
    global auxiliary_is_nous
    auxiliary_is_nous = True
    logger.debug("Auxiliary client: Nous Portal")
    # Portal recommended-models is authoritative (tier-aware); _NOUS_MODEL when unreachable/null.
    # Probes skip the lookup: exact model is irrelevant and it hits the network.
    model = _NOUS_MODEL
    lane = "vision" if vision else "text"
    if not _aux_probe_active():
        try:
            from hermes_cli.models import get_nous_recommended_aux_model
            recommended = get_nous_recommended_aux_model(vision=vision)
            if recommended:
                model = recommended
                logger.debug("Auxiliary/%s: using Portal-recommended model %s", lane, model)
            else:
                logger.debug("Auxiliary/%s: no Portal recommendation, falling back to %s", lane, model)
        except Exception as exc:
            logger.debug(
                "Auxiliary/%s: recommended-models lookup failed (%s); "
                "falling back to %s",
                lane, exc, model,
            )
    if runtime is not None:
        api_key, base_url = runtime
    else:
        api_key = _nous_api_key(nous or {})
        if not api_key:
            logger.warning(
                "Auxiliary Nous client unavailable: no usable inference JWT found "
                "(run: hermes auth add nous)."
            )
            _mark_provider_unhealthy("nous", ttl=60)
            return None, None
        base_url = str(
            (nous or {}).get("inference_base_url") or os.getenv("NOUS_INFERENCE_BASE_URL", _NOUS_DEFAULT_BASE_URL)
        ).rstrip("/")
    return _create_openai_client(api_key=api_key, base_url=base_url), model


def _refresh_nous_recommended_model(*, vision: bool, stale_model: Optional[str]) -> Optional[str]:
    """Fresh Portal recommended model after a stale-model 404 (long-lived processes pin dropped models).

    Returns the fresh recommendation, else ``_NOUS_MODEL``, whichever differs from ``stale_model``; None if neither.
    """
    stale = (stale_model or "").strip().lower()
    fresh: Optional[str] = None
    try:
        from hermes_cli.models import get_nous_recommended_aux_model
        fresh = get_nous_recommended_aux_model(vision=vision, force_refresh=True)
    except Exception as exc:
        logger.debug("Nous recommended-model refresh failed (%s); using default %s", exc, _NOUS_MODEL)
    if fresh and fresh.strip().lower() != stale:
        return fresh
    return _NOUS_MODEL if _NOUS_MODEL.strip().lower() != stale else None


def _read_main_field(field: str, *, readonly: bool, lower: bool = False) -> str:
    """Main ``model.<field>``: runtime override (``set_runtime_main``) first, then config.yaml.

    The override wins so "active main model" gates see the live CLI/gateway runtime, not the persisted
    default. ``readonly`` picks ``load_config_readonly`` (model/provider) vs ``load_config`` (api_key/base_url).
    """
    override = _runtime_main_value(field)
    if isinstance(override, str) and override.strip():
        value = override.strip()
        return value.lower() if lower else value
    with contextlib.suppress(Exception):
        from hermes_cli import config as _cfg_mod
        cfg = (_cfg_mod.load_config_readonly if readonly else _cfg_mod.load_config)()
        model_cfg = cfg.get("model", {})
        if field == "model" and isinstance(model_cfg, str) and model_cfg.strip():
            return model_cfg.strip()
        if isinstance(model_cfg, dict):
            value = model_cfg.get("default" if field == "model" else field, "")
            if isinstance(value, str) and value.strip():
                value = value.strip()
                return value.lower() if lower else value
    return ""


# Module-level callables (tests patch them): model/provider (lowercased) read the readonly config;
# api_key/base_url read the full config so ``custom`` aux tasks can inherit main creds.
_read_main_model = functools.partial(_read_main_field, "model", readonly=True)
_read_main_provider = functools.partial(_read_main_field, "provider", readonly=True, lower=True)
_read_main_api_key = functools.partial(_read_main_field, "api_key", readonly=False)
_read_main_base_url = functools.partial(_read_main_field, "base_url", readonly=False)


def _resolve_moa_aggregator(preset_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """MoA preset → aggregator (provider, model); (None, None) if unresolvable. None/"" = default preset.

    "moa" is virtual — aux tasks skip the fan-out and use the aggregator slot; shared so lookup can't drift.
    """
    try:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import resolve_moa_preset
        preset = resolve_moa_preset(load_config().get("moa") or {}, preset_name or None)
        agg = preset.get("aggregator") or {}
        agg_provider = str(agg.get("provider") or "").strip()
        agg_model = str(agg.get("model") or "").strip()
        if agg_provider and agg_model and agg_provider.lower() != "moa":
            return agg_provider, agg_model
    except Exception:
        logger.debug("MoA aggregator resolution failed for preset %r", preset_name, exc_info=True)
    return None, None


def _read_main_model_for_aux() -> str:
    """Main model with MoA presets unwrapped to the aggregator's model; "" when unresolvable (a preset name would 400)."""
    model = _read_main_model()
    if (_read_main_provider() or "").strip().lower() == "moa":
        _, agg_model = _resolve_moa_aggregator(model)
        return agg_model or ""
    return model


def _read_main_api_key_if_same_host(aux_base_url: str) -> str:
    """Main api_key only when *aux_base_url* shares the main base_url's host.

    Unconditional inheritance would leak the credential to any misconfigured host; mismatch keeps ``no-key-required`` → 401.
    """
    aux_host = base_url_hostname(aux_base_url)
    if not aux_host or aux_host != base_url_hostname(_read_main_base_url()):
        return ""
    return _read_main_api_key()


# Compatibility mirrors for older readers/tests; the ContextVar below is
# authoritative (overlapping gateway sessions make a process-global unsafe).
_RUNTIME_MAIN_PROVIDER: str = ""
_RUNTIME_MAIN_MODEL: str = ""
_RUNTIME_MAIN_BASE_URL: str = ""
_RUNTIME_MAIN_API_KEY: Any = ""
_RUNTIME_MAIN_API_MODE: str = ""
_RUNTIME_MAIN_AUTH_MODE: str = ""
_RUNTIME_MAIN_CONTEXT: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("auxiliary_runtime_main", default=None)
)

_RELAY_AUX_CALL_CONTEXT: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("auxiliary_relay_call", default=None)
)


@contextlib.contextmanager
def _relay_aux_call_scope(args: tuple, kwargs: dict):
    """Bind a fresh relay call context for one auxiliary call; mark it failed on any exception."""
    task = args[0] if args else kwargs.get("task")
    token = _RELAY_AUX_CALL_CONTEXT.set({
        "task": str(task or "unknown"),
        "request_id": f"aux-{uuid.uuid4().hex}",
        "attempt_count": 0,
        "provider": "",
        "model": "",
        "response_model": None,
        "api_mode": "chat_completions",
    })
    try:
        yield
    except BaseException:
        _fail_relay_auxiliary_call()
        raise
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(token)


def _relay_auxiliary_call(callback):
    """Give every physical retry in one auxiliary call a shared Relay identity."""
    @functools.wraps(callback)
    def wrapped(*args, **kwargs):
        with _relay_aux_call_scope(args, kwargs):
            return callback(*args, **kwargs)
    return wrapped


def _relay_auxiliary_call_async(callback):
    """Async counterpart to :func:`_relay_auxiliary_call`."""
    @functools.wraps(callback)
    async def wrapped(*args, **kwargs):
        with _relay_aux_call_scope(args, kwargs):
            return await callback(*args, **kwargs)
    return wrapped


def _set_relay_auxiliary_route(provider: str | None, model: str | None, api_mode: str | None) -> None:
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return
    context["provider"] = str(provider or "auxiliary")
    context["model"] = str(model or "unknown")
    context["response_model"] = None
    context["api_mode"] = str(api_mode or "chat_completions")


def _record_route_info(
    route_info: Optional[Dict[str, str]], provider: Optional[str], model: Optional[str]
) -> None:
    """Expose the concrete route selected for one auxiliary call."""
    if route_info is not None:
        route_info["provider"] = provider or "auto"
        route_info["model"] = model or "default"


def _relay_auxiliary_metadata(
    *, provider: str | None = None, api_mode: str | None = None
) -> tuple[str, str, dict[str, Any]] | None:
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return None
    attempt_count = int(context.get("attempt_count") or 0)
    context["attempt_count"] = attempt_count + 1
    provider_name = str(provider or context.get("provider") or "auxiliary")
    model_name = str(context.get("model") or "unknown")
    return provider_name, model_name, {
        "api_mode": str(api_mode or context.get("api_mode") or "chat_completions"),
        "api_request_id": str(context["request_id"]),
        "call_role": f"auxiliary:{context['task']}",
        "retry_count": attempt_count,
        "auxiliary_task": str(context["task"]),
    }


def _relay_sync_completion(
    client: Any, kwargs: dict[str, Any], *, provider: str | None = None,
    api_mode: str | None = None, create: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    callback = create or (lambda request: client.chat.completions.create(**request))
    route = _relay_auxiliary_metadata(provider=provider, api_mode=api_mode)
    # Isolate only the provider callback so the owning thread can unwind its lease/DB
    # transaction on hard cancel without touching the shared client.
    if route is None:
        return _run_protected_sync_provider_call(callback, kwargs)
    provider_name, fallback_model, metadata = route
    from agent import relay_llm
    return relay_llm.execute_current(
        kwargs, lambda request: _run_protected_sync_provider_call(callback, request),
        name=provider_name, model_name=str(kwargs.get("model") or fallback_model),
        metadata=metadata, defer_logical_completion=True,
    )


async def _relay_async_completion(
    client: Any, kwargs: dict[str, Any], *, provider: str | None = None,
    api_mode: str | None = None, create: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    callback = create or (lambda request: client.chat.completions.create(**request))
    route = _relay_auxiliary_metadata(provider=provider, api_mode=api_mode)
    if route is None:
        return await callback(kwargs)
    provider_name, fallback_model, metadata = route
    from agent import relay_llm
    return await relay_llm.execute_current_async(
        kwargs, callback, name=provider_name, model_name=str(kwargs.get("model") or fallback_model),
        metadata=metadata, defer_logical_completion=True,
    )


def _relay_sync_stream(
    client: Any, kwargs: dict[str, Any], *, provider: str | None = None, api_mode: str | None = None
) -> Any:
    route = _relay_auxiliary_metadata(provider=provider, api_mode=api_mode)
    if route is None:
        return client.chat.completions.create(**kwargs)
    provider_name, fallback_model, metadata = route
    from agent import relay_llm
    return relay_llm.stream_current(
        kwargs, lambda request: client.chat.completions.create(**request), name=provider_name,
        model_name=str(kwargs.get("model") or fallback_model), finalizer=dict, metadata=metadata,
        completed_response_predicate=lambda value: hasattr(value, "choices"),
    )


_RUNTIME_MAIN_COMPAT_SNAPSHOT: Tuple[Any, ...] = ("", "", "", "", "", "")
_RUNTIME_MAIN_COMPAT_LOCK = threading.Lock()


def _publish_runtime_main_mirrors(values: Tuple[Any, ...]) -> None:
    """Write the legacy globals + compat snapshot (``_MAIN_RUNTIME_FIELDS`` order) under the lock."""
    global _RUNTIME_MAIN_PROVIDER, _RUNTIME_MAIN_MODEL, _RUNTIME_MAIN_BASE_URL, _RUNTIME_MAIN_API_KEY
    global _RUNTIME_MAIN_API_MODE, _RUNTIME_MAIN_AUTH_MODE, _RUNTIME_MAIN_COMPAT_SNAPSHOT
    with _RUNTIME_MAIN_COMPAT_LOCK:
        (_RUNTIME_MAIN_PROVIDER, _RUNTIME_MAIN_MODEL, _RUNTIME_MAIN_BASE_URL,
         _RUNTIME_MAIN_API_KEY, _RUNTIME_MAIN_API_MODE, _RUNTIME_MAIN_AUTH_MODE) = values
        _RUNTIME_MAIN_COMPAT_SNAPSHOT = tuple(values)


def _compat_runtime_main() -> Optional[Dict[str, Any]]:
    """Expose deliberately patched legacy globals as a main context.

    Mirrors must never become runtime inputs: a direct patch counts only when it differs from
    the mirrored snapshot and only on the main thread.
    """
    if threading.current_thread() is not threading.main_thread():
        return None
    values = (_RUNTIME_MAIN_PROVIDER, _RUNTIME_MAIN_MODEL, _RUNTIME_MAIN_BASE_URL,
              _RUNTIME_MAIN_API_KEY, _RUNTIME_MAIN_API_MODE, _RUNTIME_MAIN_AUTH_MODE)
    if values == _RUNTIME_MAIN_COMPAT_SNAPSHOT:
        return None
    return dict(zip(_MAIN_RUNTIME_FIELDS, values))


def _runtime_main_value(field: str) -> Any:
    """Read one runtime field through context-local/controlled legacy state."""
    runtime = _RUNTIME_MAIN_CONTEXT.get()
    if runtime is None:
        runtime = _compat_runtime_main()
    return (runtime.get(field) or "") if isinstance(runtime, dict) else ""


def set_runtime_main(
    provider: str, model: str, *, requested_provider: str = "", base_url: str = "",
    api_key: Any = "", api_mode: str = "", auth_mode: str = "", session_id: str = "",
    cache_scope: str = "",
) -> contextvars.Token:
    """Record the current context's live main runtime for auxiliary routing.

    Context-local so concurrent gateway sessions don't clobber each other; legacy mirrors are
    updated for old readers. ``cache_scope`` is the rotation-stable logical cache scope,
    preferred over ``session_id`` for prompt_cache_key derivation.

    ``cache_scope`` is the rotation-stable logical cache scope (compression- lineage root —
    agent/prompt_cache_scope.py) resolved once per turn by turn_context; auxiliary Responses calls prefer it
    over ``session_id`` for prompt_cache_key derivation (#79017).
    """
    runtime = {
        "provider": (provider or "").strip().lower(),
        "requested_provider": (requested_provider or "").strip().lower(),
        "model": (model or "").strip(),
        "base_url": (base_url or "").strip(),
        "api_key": api_key.strip() if isinstance(api_key, str) else api_key if callable(api_key) else "",
        "api_mode": (api_mode or "").strip(),
        "auth_mode": (auth_mode or "").strip().lower(),
        "session_id": (session_id or "").strip(),
        "cache_scope": (cache_scope or "").strip(),
    }
    # Publish authoritative context before updating the locked mirrors.
    token = _RUNTIME_MAIN_CONTEXT.set(runtime)
    _publish_runtime_main_mirrors(tuple(runtime[field] for field in _MAIN_RUNTIME_FIELDS))
    return token


def reset_runtime_main(token: contextvars.Token) -> None:
    """Restore the runtime binding that preceded one scoped turn."""
    if token is None:
        return
    try:
        _RUNTIME_MAIN_CONTEXT.reset(token)
    except (RuntimeError, ValueError):
        pass  # Tokens can't be reset from a copied Context (workers inherit values, not token ownership).


@contextlib.contextmanager
def scoped_runtime_main(main_runtime: Optional[Dict[str, Any]]):
    """Temporarily bind an explicit runtime without touching legacy mirrors."""
    runtime = _normalize_main_runtime(main_runtime)
    token = _RUNTIME_MAIN_CONTEXT.set(runtime or None)
    try:
        yield runtime
    finally:
        _RUNTIME_MAIN_CONTEXT.reset(token)


def clear_runtime_main() -> None:
    """Clear the runtime override in the current context."""
    _RUNTIME_MAIN_CONTEXT.set(None)
    _publish_runtime_main_mirrors(("", "", "", "", "", ""))


def _resolve_custom_runtime() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve the active custom/main endpoint like the main CLI (env OPENAI_BASE_URL or config-saved)."""
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(requested="custom")
    except Exception as exc:
        logger.debug("Auxiliary client: custom runtime resolution failed: %s", exc)
        runtime = None
    if not isinstance(runtime, dict):
        openai_base = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
        if not openai_base:
            return None, None, None
        runtime = {"base_url": openai_base, "api_key": _scoped_key_env("OPENAI_API_KEY")}
    custom_base = runtime.get("base_url")
    custom_key = runtime.get("api_key")
    custom_mode = runtime.get("api_mode")
    if not isinstance(custom_base, str) or not custom_base.strip():
        return None, None, None
    custom_base = custom_base.strip().rstrip("/")
    if base_url_host_matches(custom_base, "openrouter.ai"):
        return None, None, None  # requested='custom' falls back to OpenRouter when unconfigured.
    # Local servers (Ollama, vLLM, ...) ignore auth but the SDK needs a non-empty key.
    # Use a placeholder key — the OpenAI SDK requires a non-empty string but local servers ignore the
    # Authorization header. Same fix as cli.py _ensure_runtime_credentials() (PR #2556).
    if not isinstance(custom_key, str) or not custom_key.strip():
        custom_key = "no-key-required"
    if not isinstance(custom_mode, str) or not custom_mode.strip():
        custom_mode = None
    return custom_base, custom_key.strip(), custom_mode


def _current_custom_base_url() -> str:
    custom_base, _, _ = _resolve_custom_runtime()
    return custom_base or ""


def _validate_proxy_env_urls() -> None:
    """Fail fast on malformed proxy env URLs (a shell typo like ``:6153export`` otherwise surfaces as a cryptic httpx ``Invalid port``)."""
    from urllib.parse import urlparse
    normalize_proxy_env_vars()
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        value = str(os.environ.get(key) or "").strip()
        if not value:
            continue
        try:
            parsed = urlparse(value)
            if parsed.scheme:
                _ = parsed.port          # raises ValueError for e.g. '6153export'
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed proxy environment variable {key}={value!r}. "
                "Fix or unset your proxy settings and try again."
            ) from exc


def _validate_base_url(base_url: str) -> None:
    """Reject obviously broken custom endpoint URLs before they reach httpx."""
    from urllib.parse import urlparse
    candidate = str(base_url or "").strip()
    if not candidate or candidate.startswith("acp://"):
        return
    try:
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            _ = parsed.port              # raises ValueError for malformed ports
    except ValueError as exc:
        raise RuntimeError(
            f"Malformed custom endpoint URL: {candidate!r}. "
            "Run `hermes setup` or `hermes model` and enter a valid http(s) base URL."
        ) from exc


def _try_custom_endpoint() -> Tuple[Optional[Any], Optional[str]]:
    runtime = _resolve_custom_runtime()
    custom_base, custom_key, custom_mode = (*runtime, None) if len(runtime) == 2 else runtime
    if not custom_base or not custom_key:
        return None, None
    if custom_base.lower().startswith(_CODEX_AUX_BASE_URL.lower()):
        return None, None
    model = _read_main_model_for_aux() or "gpt-4o-mini"
    logger.debug("Auxiliary client: custom endpoint (%s, api_mode=%s)", model, custom_mode or "chat_completions")
    _clean_base, _dq = _extract_url_query_params(custom_base)
    _extra = {"default_query": _dq} if _dq else {}
    # User model.default_headers override SDK fingerprint headers (as on the main client) for strict gateways/WAFs.
    _custom_headers = _apply_user_default_headers(None)
    if _custom_headers:
        _extra["default_headers"] = _custom_headers
    if custom_mode == "codex_responses":
        real_client = _create_openai_client(api_key=custom_key, base_url=_clean_base, **_extra)
        return CodexAuxiliaryClient(real_client, model), model
    if custom_mode == "anthropic_messages":
        # Third-party Anthropic-compatible gateway — never OAuth (that's api.anthropic.com only).
        try:
            from agent.anthropic_adapter import build_anthropic_client
            real_client = build_anthropic_client(custom_key, custom_base)
        except ImportError:
            logger.warning(
                "Custom endpoint declares api_mode=anthropic_messages but the "
                "anthropic SDK is not installed — falling back to OpenAI-wire."
            )
            return _create_openai_client(api_key=custom_key, base_url=_clean_base, **_extra), model
        return AnthropicAuxiliaryClient(real_client, model, custom_key, custom_base, is_oauth=False), model
    # URL-based anthropic detection for custom endpoints without explicit api_mode.
    _fallback_client = _create_openai_client(api_key=custom_key, base_url=_clean_base, **_extra)
    return _maybe_wrap_anthropic(_fallback_client, model, custom_key, custom_base, custom_mode), model


def _build_xai_oauth_aux_client(model: str) -> Tuple[Optional[Any], Optional[str]]:
    """CodexAuxiliaryClient for xAI Grok OAuth (Responses API); (None, None) if not authed.

    Caller must pass an explicit model — a pinned Grok default would rot as xAI's allowlist drifts.
    """
    if not model:
        logger.warning(
            "Auxiliary client: xai-oauth requested without a model; "
            "pass model explicitly (auxiliary.<task>.model in config.yaml)."
        )
        return None, None
    resolved = _resolve_xai_oauth_for_aux()
    if resolved is None:
        return None, None
    api_key, base_url = resolved
    logger.debug("Auxiliary client: xAI OAuth (%s via Responses API)", model)
    from tools.xai_http import hermes_xai_default_headers
    real_client = _create_openai_client(
        api_key=api_key, base_url=base_url, default_headers=hermes_xai_default_headers()
    )
    return CodexAuxiliaryClient(real_client, model), model


def _build_codex_client(model: str) -> Tuple[Optional[Any], Optional[str]]:
    """CodexAuxiliaryClient for an explicit model; (None, None) without a Codex OAuth token.

    No auto-selected default: the Codex model allow-list is undocumented and drifts.
    """
    if not model:
        logger.warning(
            "Auxiliary client: openai-codex requested without a model; "
            "pass model explicitly (auxiliary.<task>.model in config.yaml)."
        )
        return None, None
    pool_present, entry = _select_pool_entry("openai-codex")
    codex_token = _pool_runtime_api_key(entry) if pool_present else None
    if codex_token:
        base_url = _pool_runtime_base_url(entry, _CODEX_AUX_BASE_URL) or _CODEX_AUX_BASE_URL
    else:
        codex_token = _read_codex_access_token()
        if not codex_token:
            return None, None
        base_url = _CODEX_AUX_BASE_URL
    logger.debug("Auxiliary client: Codex OAuth (%s via Responses API)", model)
    real_client = _create_openai_client(
        api_key=codex_token, base_url=base_url,
        default_headers=_codex_cloudflare_headers(codex_token, base_url=base_url),
    )
    return CodexAuxiliaryClient(real_client, model), model


def _try_azure_foundry(
    *, model: Optional[str] = None, explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None, api_mode: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Azure Foundry aux client via the main agent's ``_resolve_azure_foundry_runtime`` (api_key vs Entra
    callable bearer, per-model api_mode, base_url overrides). Returns ``(client, model)`` or ``(None, None)``."""
    try:
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime
        from hermes_cli.auth import AuthError
        from hermes_cli.config import load_config_readonly
    except ImportError:
        return None, None
    try:
        cfg = load_config_readonly()
        model_cfg = cfg.get("model") if isinstance(cfg, dict) else {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
    except Exception:
        model_cfg = {}
    try:
        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry", model_cfg=model_cfg,
            explicit_api_key=explicit_api_key, explicit_base_url=explicit_base_url,
            target_model=model,
        )
    except AuthError as exc:
        logger.debug("Auxiliary azure-foundry: %s", exc)
        return None, None
    except Exception as exc:
        logger.debug("Auxiliary azure-foundry runtime error: %s", exc)
        return None, None
    api_key = runtime.get("api_key")
    base_url = str(runtime.get("base_url", "") or "")
    runtime_api_mode = api_mode or runtime.get("api_mode") or "chat_completions"
    # api_key may be a callable token provider; bail only on None/"".
    if not (callable(api_key) or api_key) or not base_url:
        return None, None
    final_model = _normalize_resolved_model(model or str(model_cfg.get("default") or ""), "azure-foundry")
    if not final_model:
        # No fallback aux model for Azure (needs a deployment name): let the auto chain fall through instead of 404ing.
        logger.debug(
            "Auxiliary azure-foundry: no model resolved (model=%r, default=%r)",
            model, model_cfg.get("default"),
        )
        return None, None
    # The SDK drops api-version query params from the base URL; pass via default_query.
    _clean_base, _dq = _extract_url_query_params(base_url)
    extra: Dict[str, Any] = {"default_query": _dq} if _dq else {}
    client = _create_openai_client(api_key=api_key, base_url=_clean_base, **extra)
    if runtime_api_mode == "codex_responses":
        return CodexAuxiliaryClient(client, final_model), final_model
    if runtime_api_mode == "anthropic_messages":
        # api_key forwarded verbatim (string or Entra callable; build_anthropic_client installs the bearer hook).
        return _maybe_wrap_anthropic(client, final_model, api_key, base_url, runtime_api_mode), final_model
    return client, final_model


def _try_anthropic(explicit_api_key: str = None) -> Tuple[Optional[Any], Optional[str]]:
    try:
        from agent.anthropic_adapter import build_anthropic_client
        from agent.anthropic_credentials import resolve_anthropic_token
    except ImportError:
        return None, None
    pool_present, entry = _select_pool_entry("anthropic")
    if pool_present and entry is not None:
        token = explicit_api_key or _pool_runtime_api_key(entry)
    else:
        # Pool absent/empty: legacy resolver so a dead pool entry can't wedge aux tasks when a standalone credential exists.
        entry = None
        token = explicit_api_key or resolve_anthropic_token()
    if not token:
        return None, None
    # Honor config.yaml model.base_url only when provider is anthropic AND the URL is
    # Anthropic-compatible; a foreign host (Codex, OpenRouter) would 401 every aux call.
    base_url = _pool_runtime_base_url(entry, _ANTHROPIC_DEFAULT_BASE_URL) if pool_present else _ANTHROPIC_DEFAULT_BASE_URL
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        model_cfg = cfg.get("model")
        if isinstance(model_cfg, dict):
            cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
            if cfg_provider == "anthropic":
                cfg_base_url = (model_cfg.get("base_url") or "").strip().rstrip("/")
                if cfg_base_url and _is_anthropic_compatible_host(cfg_base_url):
                    base_url = cfg_base_url
    from agent.anthropic_credentials import _is_oauth_token
    is_oauth = _is_oauth_token(token)
    model = _get_aux_model_for_provider("anthropic") or "claude-haiku-4-5-20251001"
    if _aux_probe_active():
        # Probe: token + adapter import resolved; skip real client construction.
        return _AuxProbeClientStub(api_key="", base_url=base_url), model
    logger.debug("Auxiliary client: Anthropic native (%s) at %s (oauth=%s)", model, base_url, is_oauth)
    try:
        real_client = build_anthropic_client(token, base_url)
    except ImportError:
        return None, None  # Adapter imports fine but the anthropic SDK itself is missing.
    return AnthropicAuxiliaryClient(real_client, model, token, base_url, is_oauth=is_oauth), model


_MAIN_RUNTIME_FIELDS = ("provider", "model", "base_url", "api_key", "api_mode", "auth_mode")
_MAIN_RUNTIME_CONTEXT_FIELDS = _MAIN_RUNTIME_FIELDS + ("requested_provider",)


def _normalize_main_runtime(main_runtime: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a sanitized copy of a live main-runtime override.

    ``api_key`` may be a zero-arg callable (Entra ID token provider, accepted by the OpenAI SDK)
    — preserved as-is so aux clients share main-agent auth.
    """
    if main_runtime is None:
        # Context-local state first; compat mirrors may hold another concurrent session's endpoint/key.
        main_runtime = _RUNTIME_MAIN_CONTEXT.get()
        if main_runtime is None:
            main_runtime = _compat_runtime_main()
    if not isinstance(main_runtime, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for field in _MAIN_RUNTIME_CONTEXT_FIELDS:
        value = main_runtime.get(field)
        if field == "api_key" and callable(value) and not isinstance(value, str):
            normalized[field] = value
        elif isinstance(value, str) and value.strip():
            normalized[field] = value.strip()
    for identity_field in ("provider", "requested_provider"):
        identity = normalized.get(identity_field)
        if isinstance(identity, str):
            normalized[identity_field] = identity.lower()
    return normalized


def _get_provider_chain() -> List[tuple]:
    """Ordered provider detection chain, built at call time so ``_try_*`` patches are picked up.

    ``openai-codex`` is deliberately absent (shifting allow-list breaks guessed-model fallback).
    """
    return [
        ("openrouter", _try_openrouter), ("nous", _try_nous),
        ("local/custom", _try_custom_endpoint), ("api-key", _resolve_api_key_provider),
    ]


# "Recently 402'd" unhealthy-provider cache: a depleted provider stays so for hours, so hiding it
# for a TTL saves an RTT per aux call. In-process only (profiles may use different keys).
_AUX_UNHEALTHY_TTL_SECONDS = 600  # 10 minutes
_aux_unhealthy_until: Dict[str, float] = {}
_aux_unhealthy_logged_at: Dict[str, float] = {}
# resolved_provider / explicit-config names → chain labels.
_AUX_UNHEALTHY_LABEL_ALIASES = {
    "openrouter": "openrouter", "nous": "nous", "custom": "local/custom",
    "local/custom": "local/custom", "openai-codex": "openai-codex", "codex": "openai-codex",
}


def _normalize_chain_label(provider: str) -> str:
    """resolved_provider → chain label; unknown API-key providers fall back to the lowercased input."""
    if not provider:
        return ""
    p = str(provider).strip().lower()
    return _AUX_UNHEALTHY_LABEL_ALIASES.get(p, p)


def _mark_provider_unhealthy(provider: str, ttl: Optional[float] = None) -> None:
    """Hide ``provider`` from chain iteration until the TTL expires (after a confirmed payment error)."""
    label = _normalize_chain_label(provider)
    if not label:
        return
    ttl = _AUX_UNHEALTHY_TTL_SECONDS if ttl is None else ttl
    expires_at = time.time() + ttl
    _aux_unhealthy_until[label] = expires_at
    logger.warning(
        "Auxiliary: marking %s unhealthy for %ds (payment / credit error). "
        "Subsequent auxiliary calls will skip it until %s.",
        label, int(ttl), time.strftime("%H:%M:%S", time.localtime(expires_at)),
    )


def _is_provider_unhealthy(label: str) -> bool:
    """True iff ``label`` is unhealthy and unexpired; lazily evicts expired entries."""
    if not label:
        return False
    expires_at = _aux_unhealthy_until.get(label)
    if expires_at is None:
        return False
    if time.time() >= expires_at:
        _aux_unhealthy_until.pop(label, None)
        _aux_unhealthy_logged_at.pop(label, None)
        return False
    return True


def _log_skip_unhealthy(label: str, task: Optional[str] = None) -> None:
    """Log a skipped unhealthy provider at most once per minute per label."""
    now = time.time()
    if now - _aux_unhealthy_logged_at.get(label, 0.0) >= 60:
        _aux_unhealthy_logged_at[label] = now
        expires_at = _aux_unhealthy_until.get(label, now)
        logger.info(
            "Auxiliary %s: skipping %s (recently returned payment error, retry in %ds)",
            task or "call", label, max(0, int(expires_at - now)),
        )


def _reset_aux_unhealthy_cache() -> None:
    """Clear the unhealthy cache (tests / explicit user reset)."""
    _aux_unhealthy_until.clear()
    _aux_unhealthy_logged_at.clear()


def _contains_any(text: str, needles: Tuple[str, ...]) -> bool:
    """True when any needle is a substring of ``text``."""
    return any(kw in text for kw in needles)


# Billing-body markers (credit exhaustion wrapped in 402/403/404/429 bodies), plus daily/weekly quota
# exhaustion (functionally credit exhaustion; "resource exhausted" is the Vertex/gRPC quota phrasing).
_PAYMENT_KEYWORDS = (
    "credits", "insufficient funds", "can only afford", "billing", "payment required",
    "out of funds", "run out of funds", "balance_depleted", "no usable credits",
    "model_not_supported_on_free_tier", "not available on the free tier",
    "requires a subscription", "upgrade for access", "upgrade for higher limits",
    "reached your session usage limit", "quota exceeded", "quota_exceeded",
    "too many tokens per day", "daily limit", "tokens per day", "daily quota", "resource exhausted",
    "weekly usage limit", "weekly limit",
)


def _is_payment_error(exc: Exception) -> bool:
    """Payment/credit/quota exhaustion: HTTP 402, or a billing/quota body on 403/404/429/no-status."""
    status = getattr(exc, "status_code", None)
    return status == 402 or (
        status in {403, 404, 429, None} and _contains_any(str(exc).lower(), _PAYMENT_KEYWORDS)
    )


def _nous_portal_account_has_fresh_paid_access() -> bool:
    """Return True only when the fresh Nous account API says paid access is allowed."""
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info
        return get_nous_portal_account_info(force_fresh=True).paid_service_access is True
    except Exception as exc:
        logger.debug("Auxiliary Nous paid-entitlement refresh check failed: %s", exc)
        return False


_RATE_LIMIT_KEYWORDS = (
    "rate limit", "rate_limit", "too many requests", "try again", "retry after", "resets in"
)
_RATE_LIMIT_BILLING_KEYWORDS = (
    "credits", "insufficient funds", "billing", "payment required", "can only afford",
    "out of funds", "run out of funds", "balance_depleted", "no usable credits",
    "model_not_supported_on_free_tier", "not available on the free tier",
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """429 rate limit (not billing/quota, which _is_payment_error owns).

    OpenAI's RateLimitError may omit .status_code — matched by class name. A generic 429 without
    billing keywords counts as a rate limit.
    """
    # (PR #8023 pattern)
    if type(exc).__name__ == "RateLimitError":
        return True
    if getattr(exc, "status_code", None) != 429:
        return False
    err_lower = str(exc).lower()
    return _contains_any(err_lower, _RATE_LIMIT_KEYWORDS) or not _contains_any(err_lower, _RATE_LIMIT_BILLING_KEYWORDS)


def _is_timeout_error(exc: Exception) -> bool:
    """Full-budget request timeout, distinct from a fast connection drop.

    A timeout burns the whole ``timeout`` budget, so a same-provider retry on the compression
    path doubles wall time; fast drops stay on the retry path.
    """
    with contextlib.suppress(ImportError):
        from openai import APITimeoutError
        if isinstance(exc, APITimeoutError):
            return True
    return "Timeout" in type(exc).__name__ or "timed out" in str(exc).lower()


def _is_connection_error(exc: Exception) -> bool:
    """Connection/network errors (endpoint unreachable), as opposed to 4xx/5xx API errors."""
    with contextlib.suppress(ImportError):
        from openai import APIConnectionError, APITimeoutError
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
    if _contains_any(type(exc).__name__, ("Connection", "Timeout", "DNS", "SSL")):
        return True
    return _contains_any(str(exc).lower(), (
        "connection refused", "name or service not known", "no route to host",
        "network is unreachable", "timed out", "connection reset",
        # httpcore/httpx premature stream close — transient, retry/reroute.
        "incomplete chunked read", "peer closed connection", "response ended prematurely",
        "unexpected eof", "remoteprotocolerror", "localprotocolerror",
    ))


def _is_transient_transport_error(exc: Exception) -> bool:
    """One-off transport blip worth retrying on the SAME provider: connection/stream-close errors plus pure 5xx/408.

    Deliberately narrow: payment/auth/rate-limit errors switch provider, refresh creds, or rotate the pool.
    """
    if _is_connection_error(exc):
        return True
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return isinstance(status, int) and (status == 408 or 500 <= status < 600)


_DEFAULT_TRANSIENT_RETRIES = 2
_TRANSIENT_RETRY_BACKOFF_BASE = 1.0  # Backoff base (seconds); overridable so tests can zero it out.


def _transient_retry_count() -> int:
    """Same-provider retries for a transient blip: ``auxiliary.transient_retries``
    (default 2), clamped to [0, 6]; config-read failures fall back to default."""
    try:
        from hermes_cli.config import cfg_get, load_config
        val = cfg_get(load_config(), "auxiliary", "transient_retries")
        return _DEFAULT_TRANSIENT_RETRIES if val is None else max(0, min(int(val), 6))
    except Exception:
        return _DEFAULT_TRANSIENT_RETRIES


def _is_auth_error(exc: Exception) -> bool:
    """Auth failures that should trigger provider-specific refresh."""
    status = getattr(exc, "status_code", None)
    if status == 401:
        return True
    err_lower = str(exc).lower()
    if "error code: 401" in err_lower or "authenticationerror" in type(exc).__name__.lower():
        return True
    # xAI returns 403 "unauthenticated:bad-credentials" for expired OAuth tokens — semantically a 401.
    return "bad-credentials" in err_lower and (status == 403 or "unauthenticated" in err_lower)


def _is_unsupported_parameter_error(exc: Exception, param: str) -> bool:
    """Provider 400 for an unsupported request parameter: the parameter name plus a generic
    unsupported/unknown/unrecognized marker, so call sites can retry without the key."""
    param_lower = (param or "").lower()
    if not param_lower:
        return False
    err_lower = str(exc).lower()
    return param_lower in err_lower and _contains_any(err_lower, (
        "unsupported parameter", "unsupported_parameter", "not supported", "does not support",
        "unknown parameter", "unrecognized request argument", "unrecognized parameter", "invalid parameter",
    ))


def _is_structured_output_rejection(exc: Exception) -> bool:
    """Provider 400/422 rejecting the structured-output field, on either wire: OpenAI ``response_format``
    (incl. vLLM's ``guided_grammar``/xgrammar failures) or Anthropic ``output_config.format`` ("Extra inputs
    are not permitted"). Callers tolerate an unconstrained reply, so the reaction is one retry without it."""
    status = getattr(exc, "status_code", None)
    if status is not None and status not in {400, 422}:
        return False
    err_lower = str(exc).lower()
    # vLLM grammar-backend failures name the translated parameter, not ours.
    if _contains_any(err_lower, ("guided_grammar", "xgrammar", "compile_grammar_error")):
        return True
    if "extra inputs are not permitted" in err_lower and (
        "response_format" in err_lower or "output_config" in err_lower
    ):
        return True
    if "response_format" in err_lower and "unavailable" in err_lower:
        return True
    return _is_unsupported_parameter_error(exc, "response_format") or _is_unsupported_parameter_error(exc, "output_config")


def _without_structured_output_format(kwargs: dict) -> Optional[dict]:
    """Copy *kwargs* without ``response_format`` (top-level and ``extra_body``); None when nothing was
    removed, so call sites don't retry an unchanged request."""
    retry_kwargs = dict(kwargs)
    changed = retry_kwargs.pop("response_format", None) is not None
    extra_body = retry_kwargs.get("extra_body")
    if isinstance(extra_body, dict) and "response_format" in extra_body:
        remaining = {k: v for k, v in extra_body.items() if k != "response_format"}
        if remaining:
            retry_kwargs["extra_body"] = remaining
        else:
            retry_kwargs.pop("extra_body", None)
        changed = True
    return retry_kwargs if changed else None


def _is_model_not_found_error(exc: Exception) -> bool:
    """"Requested model doesn't exist" (404 / invalid model) — typically a long-lived process pinned a
    since-dropped model. Excludes billing keywords, which :func:`_is_payment_error` owns."""
    status = getattr(exc, "status_code", None)
    err_lower = str(exc).lower()
    if _contains_any(err_lower, (
        "credits", "insufficient funds", "billing", "out of funds", "balance_depleted",
        "no usable credits", "free tier", "free-tier", "not available on the free tier",
    )):
        return False
    if status not in {404, 400, None}:
        return False
    return _contains_any(err_lower, (
        "model does not exist", "does not exist in our configuration", "openrouter catalog",
        "is not a valid model", "no such model", "model not found",
        "the model `",            # OpenAI-style: "The model `X` does not exist"
        "model_not_found", "unknown model",
    ))


def _is_model_incompatible_error(exc: Exception) -> bool:
    """"This route cannot serve this model" 400 (capability mismatch, e.g. a Codex/ChatGPT-account
    fallback asked to run a non-OpenAI model). Auth/payment predicates don't fire, so this keeps the
    chain going instead of aborting. Excludes billing 400s and not-found 400s."""
    status = getattr(exc, "status_code", None)
    if status not in {400, None}:
        return False
    err_lower = str(exc).lower()
    if _is_model_not_found_error(exc):
        return False
    # Billing keywords checked directly: _is_payment_error is status-gated and misses 400-coded billing bodies.
    if _contains_any(err_lower, (
        "credits", "insufficient funds", "billing", "out of funds", "balance_depleted",
        "no usable credits", "payment required", "free tier", "free-tier",
        "not available on the free tier", "model_not_supported_on_free_tier", "quota",
    )):
        return False
    return _contains_any(err_lower, (
        "is not supported when using",   # codex/ChatGPT-account model gating
        "model is not supported", "not supported with this", "not supported for this account",
        "model_not_supported", "does not support this model", "unsupported model",
    ))


def _is_invalid_aux_response_error(exc: Exception) -> bool:
    """HTTP-200 empty/malformed ChatCompletions — a capability failure routed like model incompatibility."""
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return "auxiliary " in msg and "llm returned invalid response" in msg and "choices[0].message" in msg


# Tasks on a user-visible critical path (compression blocks resuming an oversized session; vision
# stalls the serialised turn queue). A same-provider retry after a full-budget timeout costs another
# whole ``timeout`` window, so they skip straight to fallback; fast blips still retry.
# Fast blips (a streaming-close or a 5xx) still retry, since those are cheap. See issue #54465 for the
# compression case.
_TIMEOUT_NO_RETRY_TASKS = frozenset({"compression", "vision"})


def _should_skip_same_provider_retry(task: Optional[str], exc: Exception) -> bool:
    """True when a transient error on a critical-path task should go straight to fallback.

    Carve-out: a fast first-token fail (dead stream within the no-progress window, zero output —
    see ``_timeout_message``) is cheap and keeps the same-provider retry; mid-stream stalls and
    hard-ceiling timeouts skip to fallback.
    """
    return task in _TIMEOUT_NO_RETRY_TASKS and _is_timeout_error(exc) and "no-progress timeout" not in str(exc)


def _evict_cached_clients(provider: str) -> None:
    """Drop cached auxiliary clients for a provider so fresh creds are used."""
    normalized = _normalize_aux_provider(provider)
    with _client_cache_lock:
        for key in [key for key in _client_cache if _normalize_aux_provider(str(key[0])) == normalized]:
            client = _client_cache.get(key, (None, None, None))[0]
            if client is not None:
                _close_cached_client(client)
            _client_cache.pop(key, None)


def _evict_cached_client_instance(target: Any) -> bool:
    """Drop cache entries whose stored client (or its ``_real_client``) is *target*; True if any evicted.

    Used when a cached client is poisoned (closed transport after a timeout). Async wrappers must
    expose the same ``_real_client`` as their sync sibling or the async entry survives.
    """
    if target is None:
        return False
    evicted = False
    with _client_cache_lock:
        for key, entry in list(_client_cache.items()):
            cached = entry[0] if entry is not None else None
            if cached is not None and (cached is target or getattr(cached, "_real_client", None) is target):
                del _client_cache[key]
                evicted = True
    return evicted


def _pool_cache_hint(provider: str, *, main_runtime: Optional[Dict[str, Any]] = None) -> str:
    """Return a stable cache discriminator for pooled providers."""
    normalized = _normalize_aux_provider(provider)
    if normalized == "auto":
        runtime = _normalize_main_runtime(main_runtime)
        normalized = _normalize_aux_provider(runtime.get("provider") or _read_main_provider())
    if normalized in {"", "auto", "custom"}:
        return ""
    entry = _peek_pool_entry(normalized)
    if entry is None:
        return ""
    entry_id = str(getattr(entry, "id", "") or "").strip()
    return f"{normalized}:{entry_id}" if entry_id else ""


# Ordered (host, provider) tables for inferring a backend from a client base URL.
_POOL_PROVIDER_BY_HOST = (
    ("chatgpt.com", "openai-codex"), ("openrouter.ai", "openrouter"),
    ("inference-api.nousresearch.com", "nous"), ("api.anthropic.com", "anthropic"),
    ("githubcopilot.com", "copilot"), ("api.kimi.com", "kimi-coding"), ("api.x.ai", "xai-oauth"),
)
_AUTH_REFRESH_PROVIDER_BY_HOST = (
    ("api.githubcopilot.com", "copilot"), ("chatgpt.com", "openai-codex"),
    ("api.anthropic.com", "anthropic"), ("inference-api.nousresearch.com", "nous"),
)


def _provider_for_host(base_url: str, table: Tuple[Tuple[str, str], ...]) -> Optional[str]:
    """First provider in ``table`` whose host matches ``base_url``, else None."""
    for host, provider in table:
        if base_url_host_matches(base_url, host):
            return provider
    return None


def _recoverable_pool_provider(
    resolved_provider: str, client: Any, main_runtime: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Infer which provider pool can recover the current auxiliary client."""
    normalized = _normalize_aux_provider(resolved_provider)
    if normalized not in {"", "auto", "custom"}:
        return normalized
    base = str(getattr(client, "base_url", "") or "")
    known = _provider_for_host(base, _POOL_PROVIDER_BY_HOST)
    if known is not None:
        return known
    # Providers outside the table (e.g. opencode-go): match base URL against registered
    # api_key providers so pool rotation works for them too.
    if main_runtime:
        rt_provider = _normalize_main_runtime(main_runtime).get("provider", "")
        if rt_provider and rt_provider not in {"", "auto", "custom"}:
            with contextlib.suppress(Exception):
                from hermes_cli.auth import PROVIDER_REGISTRY
                pconfig = PROVIDER_REGISTRY.get(rt_provider)
                if pconfig and getattr(pconfig, "auth_type", None) == "api_key":
                    rt_base = str(getattr(pconfig, "inference_base_url", "") or "").rstrip("/")
                    if rt_base and base_url_host_matches(base, base_url_hostname(rt_base)):
                        return rt_provider
    return None


def _recover_provider_pool(provider: str, exc: Exception, *, failed_api_key: str = "") -> bool:
    """Try same-provider credential-pool recovery for auxiliary calls.

    ``failed_api_key`` lets mark_exhausted_and_rotate identify the right pool entry even if
    another process already rotated (current() would be None).
    """
    normalized = _normalize_aux_provider(provider)
    try:
        pool = load_pool(normalized)
    except Exception as load_exc:
        logger.debug("Auxiliary client: could not load pool for %s recovery: %s", normalized, load_exc)
        return False
    if not pool or not pool.has_credentials():
        return False
    status_code = getattr(exc, "status_code", None)

    def _rotate(fallback_status: int) -> bool:
        error_context: Dict[str, Any] = {"message": str(exc)}
        if status_code is not None:
            error_context["status_code"] = status_code
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=status_code if status_code is not None else fallback_status,
            error_context=error_context, api_key_hint=failed_api_key or None,
        )
        if next_entry is None:
            return False
        _evict_cached_clients(normalized)
        return True

    if _is_auth_error(exc):
        if pool.try_refresh_current() is not None:
            _evict_cached_clients(normalized)
            return True
        return _rotate(401)
    if _is_payment_error(exc):
        return _rotate(402)
    if _is_rate_limit_error(exc):
        return _rotate(429)
    return False


def _prepare_same_provider_retry(
    *, task: Optional[str], resolved_provider: str, resolved_model: Optional[str],
    resolved_base_url: Optional[str], resolved_api_key: Optional[str],
    resolved_api_mode: Optional[str], main_runtime: Optional[Dict[str, Any]],
    final_model: Optional[str], messages: list, temperature: Optional[float],
    max_tokens: Optional[int], tools: Optional[list], effective_timeout: float,
    effective_extra_body: dict, reasoning_config: Optional[dict], async_mode: bool,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Rebuild (client, request kwargs) for a same-provider retry after credential recovery."""
    if task == "vision":
        effective_provider, retry_client, retry_model = resolve_vision_provider_client(
            provider=resolved_provider, model=final_model, base_url=resolved_base_url,
            api_key=resolved_api_key, async_mode=async_mode,
        )
    else:
        retry_client, retry_model = _get_cached_client(
            resolved_provider, resolved_model, async_mode=async_mode, base_url=resolved_base_url,
            api_key=resolved_api_key, api_mode=resolved_api_mode, main_runtime=main_runtime,
        )
        effective_provider = _effective_provider_for_client(retry_client, resolved_provider)
    if retry_client is None:
        raise RuntimeError(
            f"Auxiliary {task or 'call'}: provider {resolved_provider} could not be rebuilt after recovery"
        )
    retry_base = str(getattr(retry_client, "base_url", "") or "")
    retry_kwargs = _build_call_kwargs(
        effective_provider or resolved_provider, retry_model or final_model, messages,
        temperature=temperature, max_tokens=max_tokens, tools=tools, timeout=effective_timeout,
        extra_body=effective_extra_body, reasoning_config=reasoning_config,
        base_url=retry_base or resolved_base_url, task=task,
    )
    # Preserve per-request attribution headers (e.g. Copilot ``x-initiator``) so the retry keeps capability gating.
    if extra_headers:
        # Copilot's ``x-initiator: user``) across the rebuilt-client retry — dropping them here would let a
        # recovery retry silently lose capability gating (#60293).
        # Preserve per-request attribution headers across the rebuilt-client retry — see the sync variant
        # above (#60293).
        retry_kwargs["extra_headers"] = dict(extra_headers)
    if _is_anthropic_compat_endpoint(resolved_provider, retry_base):
        retry_kwargs["messages"] = _convert_openai_images_to_anthropic(retry_kwargs["messages"])
    return retry_client, retry_kwargs


def _retry_same_provider_sync(*, resolved_provider: str, resolved_api_mode: Optional[str], task: Optional[str], **prep) -> Any:
    retry_client, retry_kwargs = _prepare_same_provider_retry(
        task=task, resolved_provider=resolved_provider, resolved_api_mode=resolved_api_mode, async_mode=False, **prep,
    )
    return _validate_llm_response(
        _relay_sync_completion(retry_client, retry_kwargs, provider=resolved_provider, api_mode=resolved_api_mode), task,
    )


async def _retry_same_provider_async(*, resolved_provider: str, resolved_api_mode: Optional[str], task: Optional[str], **prep) -> Any:
    retry_client, retry_kwargs = _prepare_same_provider_retry(
        task=task, resolved_provider=resolved_provider, resolved_api_mode=resolved_api_mode, async_mode=True, **prep,
    )
    return _validate_llm_response(
        await _relay_async_completion(retry_client, retry_kwargs, provider=resolved_provider, api_mode=resolved_api_mode),
        task,
    )


def _creds_have_api_key(creds: Dict[str, Any]) -> bool:
    return bool(str(creds.get("api_key", "") or "").strip())


def _refresh_copilot_credentials() -> bool:
    from hermes_cli.copilot_auth import _jwt_cache, _token_fingerprint, exchange_copilot_token, resolve_copilot_token
    raw_token, _source = resolve_copilot_token()
    if not str(raw_token or "").strip():
        return False
    _jwt_cache.pop(_token_fingerprint(raw_token), None)
    exchange_copilot_token(raw_token)
    return True


def _refresh_codex_credentials() -> bool:
    from hermes_cli.auth import resolve_codex_runtime_credentials
    return _creds_have_api_key(resolve_codex_runtime_credentials(force_refresh=True))


def _refresh_nous_credentials() -> bool:
    from hermes_cli.auth import resolve_nous_runtime_credentials
    return _creds_have_api_key(resolve_nous_runtime_credentials(
        timeout_seconds=env_float("HERMES_NOUS_TIMEOUT_SECONDS", 15), force_refresh=True
    ))


def _refresh_anthropic_credentials() -> bool:
    from agent.anthropic_credentials import read_claude_code_credentials, _refresh_oauth_token, resolve_anthropic_token
    creds = read_claude_code_credentials()
    token = _refresh_oauth_token(creds) if isinstance(creds, dict) and creds.get("refreshToken") else None
    if not str(token or "").strip():
        token = resolve_anthropic_token()
    return bool(str(token or "").strip())


def _refresh_xai_oauth_credentials() -> bool:
    """Pool-level refresh first, then the singleton auth-store resolver."""
    pool = load_pool("xai-oauth")
    if pool and pool.has_credentials():
        pool.select()
        refreshed = pool.try_refresh_current()
        if refreshed is not None and str(getattr(refreshed, "runtime_api_key", "") or "").strip():
            return True
    from hermes_cli.auth import resolve_xai_oauth_runtime_credentials
    return _creds_have_api_key(resolve_xai_oauth_runtime_credentials(force_refresh=True))


def _refresh_vertex_credentials() -> bool:
    """Mirrors run_agent's Vertex refresh; the cache key ignores the rotating bearer, so
    without the eviction that follows, a ~1h-expired aux Vertex client 401s forever."""
    from agent.vertex_adapter import get_vertex_config
    token, base_url = get_vertex_config()
    return bool(isinstance(token, str) and token.strip() and isinstance(base_url, str) and base_url.strip())


# Each refresher returns True when a usable credential exists; the caller then evicts cached clients.
_CREDENTIAL_REFRESHERS: Dict[str, Callable[[], bool]] = {
    "copilot": _refresh_copilot_credentials, "openai-codex": _refresh_codex_credentials,
    "nous": _refresh_nous_credentials, "anthropic": _refresh_anthropic_credentials,
    "xai-oauth": _refresh_xai_oauth_credentials, "vertex": _refresh_vertex_credentials,
}


def _refresh_provider_credentials(provider: str) -> bool:
    """Refresh short-lived credentials for OAuth-backed auxiliary providers."""
    normalized = _normalize_aux_provider(provider)
    refresher = _CREDENTIAL_REFRESHERS.get(normalized)
    if refresher is None:
        return False
    try:
        if not refresher():
            return False
        _evict_cached_clients(normalized)
        return True
    except Exception as exc:
        logger.debug("Auxiliary provider credential refresh failed for %s: %s", normalized, exc)
        return False


def _auth_refresh_provider_for_route(resolved_provider: Optional[str], client_base_url: str) -> str:
    """Provider whose short-lived credentials should be refreshed; auto-routed calls keep
    ``resolved_provider == "auto"``, so infer the backend from the client's base URL."""
    normalized = _normalize_aux_provider(resolved_provider)
    if normalized and normalized != "auto":
        return normalized
    return _provider_for_host(client_base_url, _AUTH_REFRESH_PROVIDER_BY_HOST) or normalized


def _fallback_chain_entry(task: Optional[str], fb_label: str) -> Optional[Dict[str, Any]]:
    """Resolve the ``fallback_chain`` entry a ``fallback_chain[<i>](<provider>)`` label points at,
    or None when the label is not a configured-chain candidate or the index no longer resolves."""
    if not task or not fb_label:
        return None
    m = re.match(r"fallback_chain\[(\d+)\]", fb_label)
    if not m:
        return None
    try:
        chain = _get_auxiliary_task_config(task).get("fallback_chain")
        entry = chain[int(m.group(1))] if isinstance(chain, list) else None
    except Exception:
        return None
    return entry if isinstance(entry, dict) else None


def _coerce_positive_timeout(raw: Any) -> Optional[float]:
    """Coerce a config ``timeout`` to a positive float, or None (rejects bools, which are ints)."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
        return float(raw)
    return None


def _fallback_entry_timeout(task: Optional[str], fb_label: str) -> Optional[float]:
    """Per-entry ``timeout`` for a configured fallback candidate, or None (keep the task-level
    timeout). Inheriting the primary's deadline used to kill healthy-but-slower fallbacks.

    A fallback candidate previously inherited the exact timeout the primary provider was called with. When
    that deadline was tuned for the primary (or the primary simply consumed its whole budget before failing
    over), the fallback aborted on the same clock even when independently healthy — a 163k-token compression
    that needs ~90s on the fallback died at the primary's 30s deadline every turn (#62452).
    """
    entry = _fallback_chain_entry(task, fb_label)
    return _coerce_positive_timeout(entry.get("timeout") if entry else None)


def _fallback_provider_from_label(label: str) -> str:
    """Recover the provider identifier from a fallback display label."""
    match = re.match(r"(?:fallback_chain\[\d+\]|fallback_providers\[\d+\]|main-agent)\(([^)]+)\)$", label or "")
    return match.group(1).strip() if match else str(label or "").strip()


class _FallbackDestination(NamedTuple):
    provider: str
    base_url: str
    api_mode: Optional[str]
    model: Optional[str]


def _complete_fallback_destination(
    provider: str, base_url: str, api_mode: Optional[str], model: Optional[str]
) -> _FallbackDestination:
    if not api_mode:
        if _endpoint_speaks_anthropic_messages(base_url):
            api_mode = "anthropic_messages"
        else:
            with contextlib.suppress(Exception):
                from hermes_cli.runtime_provider import resolve_runtime_provider
                runtime = resolve_runtime_provider(
                    requested=provider, explicit_base_url=base_url or None, target_model=model or ""
                )
                api_mode = str(runtime.get("api_mode") or "").strip() or None
    return _FallbackDestination(provider, base_url, api_mode, model)


def _fallback_destination_from_entry(
    entry: Dict[str, Any], fb_client: Any, fb_model: Optional[str]
) -> _FallbackDestination:
    provider = str(entry.get("provider") or "").strip()
    base_url = str(entry.get("base_url") or getattr(fb_client, "base_url", "") or "").strip()
    api_mode = str(entry.get("api_mode") or entry.get("transport") or "").strip() or None
    model = fb_model or str(entry.get("model") or "").strip() or None
    return _complete_fallback_destination(provider, base_url, api_mode, model)


def _fallback_destination(
    task: Optional[str], fb_client: Any, fb_model: Optional[str], fb_label: str
) -> _FallbackDestination:
    """Route identity of a fallback request: attached destination, else configured entry, else label."""
    attached = getattr(fb_client, "_hermes_fallback_destination", None)
    if isinstance(attached, _FallbackDestination):
        return attached
    entry = _fallback_chain_entry(task, fb_label)
    if entry is not None:
        return _fallback_destination_from_entry(entry, fb_client, fb_model)
    return _complete_fallback_destination(
        _fallback_provider_from_label(fb_label), str(getattr(fb_client, "base_url", "") or ""), None, fb_model,
    )


def _replan_synchronous_cache_sections(
    messages: list, tools: Optional[list], *, destination: _FallbackDestination
) -> tuple[list, list]:
    """Strip source decoration and plan one synchronous destination locally."""
    from agent.agent_runtime_helpers import configured_cache_ttl, plan_cache_sections_for_destination
    return plan_cache_sections_for_destination(
        messages, tools, provider=destination.provider, base_url=destination.base_url,
        api_mode=destination.api_mode or "", model=destination.model or "",
        # Operator's configured TTL so fallbacks don't regress 1h → 5m default (no live agent here; read config).
        cache_ttl=configured_cache_ttl(),
    )


def _fallback_request_kwargs(
    destination: _FallbackDestination, *, task: Optional[str], messages: list,
    tools: Optional[list], temperature: Optional[float], max_tokens: Optional[int],
    effective_timeout: float, effective_extra_body: dict, reasoning_config: Optional[dict],
    fallback_entry: dict, task_config: dict, apply_fast_lane: bool,
) -> Dict[str, Any]:
    """Build request kwargs for one fallback destination (cache-section replan + fast-lane cap)."""
    fallback_max_tokens, fallback_extra_body = max_tokens, effective_extra_body
    if apply_fast_lane:
        fallback_max_tokens, fallback_extra_body = _compression_fast_lane_controls(
            task, actual_provider=destination.provider, actual_model=destination.model,
            requested_provider=fallback_entry.get("provider"),
            requested_model=fallback_entry.get("model"), route_config=fallback_entry,
            leak_guard_config=task_config, max_tokens=max_tokens, extra_body=effective_extra_body,
        )
    fallback_messages, fallback_tools = _replan_synchronous_cache_sections(messages, tools, destination=destination)
    fb_kwargs = _build_call_kwargs(
        destination.provider, destination.model, fallback_messages,
        temperature=temperature, max_tokens=fallback_max_tokens, tools=fallback_tools, timeout=effective_timeout,
        extra_body=fallback_extra_body, reasoning_config=reasoning_config, base_url=destination.base_url, task=task)
    if apply_fast_lane and fallback_max_tokens is not None and max_tokens is None:
        fb_kwargs.update(auxiliary_max_tokens_param(fallback_max_tokens, model=destination.model))
    return fb_kwargs


def _plan_fallback_candidate(
    fb_client: Any, fb_model: Optional[str], fb_label: str, *, task: Optional[str],
    effective_timeout: float, apply_fast_lane: bool, **request,
) -> Tuple[_FallbackDestination, Dict[str, Any], Callable[[str, Any, Optional[str]], Dict[str, Any]]]:
    """Resolve the destination + first-attempt kwargs for a fallback candidate.

    Returns ``(destination, kwargs, rebuild)`` where ``rebuild(provider, client, model)`` produces
    kwargs for the credential-refreshed retry destination. A configured-chain entry's own
    ``timeout`` overrides ``effective_timeout``.
    """
    fb_timeout = _fallback_entry_timeout(task, fb_label)
    if fb_timeout is not None and fb_timeout != effective_timeout:
        logger.info(
            "Auxiliary %s: %s using its configured timeout %.0fs "
            "(task-level was %.0fs)",
            task or "call", fb_label, fb_timeout, effective_timeout,
        )
        effective_timeout = fb_timeout
    destination = _fallback_destination(task, fb_client, fb_model, fb_label)
    task_config = _get_auxiliary_task_config(task) if task == "compression" else {}
    fallback_entry = _fallback_chain_entry(task, fb_label) or {}
    common = dict(
        task=task, effective_timeout=effective_timeout, fallback_entry=fallback_entry,
        task_config=task_config, apply_fast_lane=apply_fast_lane, **request,
    )

    def _rebuild(provider: str, client: Any, model: Optional[str]) -> Tuple[_FallbackDestination, Dict[str, Any]]:
        retry_destination = _FallbackDestination(
            provider, destination.base_url or str(getattr(client, "base_url", "") or ""),
            destination.api_mode, model or destination.model,
        )
        return retry_destination, _fallback_request_kwargs(retry_destination, **common)

    return destination, _fallback_request_kwargs(destination, **common), _rebuild


def _quarantine_fallback_candidate(task: Optional[str], fb_label: str, fb_provider: str, fb_err: Exception, *, tag: str = "") -> None:
    """Refresh unavailable or still 401s: token is dead. Quarantine the candidate so the caller moves on."""
    _mark_provider_unhealthy(fb_provider or fb_label)
    logger.warning("Auxiliary %s%s: fallback candidate %s has a stale/unrefreshable "
                   "credential (%s) — skipping to next fallback", task or "call", tag, fb_label, fb_err)


def _plan_fallback_auth_retry(
    destination: _FallbackDestination,
    rebuild: Callable[[str, Any, Optional[str]], Tuple[_FallbackDestination, Dict[str, Any]]], *,
    async_mode: bool,
) -> Tuple[str, Optional[Tuple[Any, Dict[str, Any], _FallbackDestination]]]:
    """After an auth error on a fallback candidate: refresh credentials and rebuild the request.
    Returns ``(refresh_provider, retry)``; ``retry`` = ``(client, kwargs, destination)`` or None."""
    fb_provider = _auth_refresh_provider_for_route(destination.provider, destination.base_url)
    if fb_provider not in {"auto", "", None} and _refresh_provider_credentials(fb_provider):
        retry_client, retry_model = _get_cached_client(
            fb_provider, destination.model, **({"async_mode": True} if async_mode else {}),
            base_url=destination.base_url or None, api_mode=destination.api_mode,
        )
        if retry_client is not None:
            retry_destination, retry_kwargs = rebuild(fb_provider, retry_client, retry_model)
            return fb_provider, (retry_client, retry_kwargs, retry_destination)
    return fb_provider, None


def _call_fallback_candidate_sync(
    fb_client: Any, fb_model: Optional[str], fb_label: str, *, task: Optional[str], messages: list,
    temperature: Optional[float], max_tokens: Optional[int], tools: Optional[list],
    effective_timeout: float, effective_extra_body: dict, reasoning_config: Optional[dict],
) -> Optional[Any]:
    """Call one fallback candidate with stale-credential recovery: on an auth error refresh its
    credentials and retry once with a rebuilt client; if that also auth-fails, quarantine the
    provider and return None so the caller moves on. Non-auth errors raise.

    ``effective_timeout`` is the task-level deadline; a configured-chain candidate with its own ``timeout``
    entry gets that instead, so a fallback tuned differently from the primary is allowed its own budget
    (#62452).
    """
    destination, fb_kwargs, rebuild = _plan_fallback_candidate(
        fb_client, fb_model, fb_label, task=task, effective_timeout=effective_timeout,
        apply_fast_lane=True, messages=messages, tools=tools, temperature=temperature,
        max_tokens=max_tokens, effective_extra_body=effective_extra_body,
        reasoning_config=reasoning_config,
    )

    def _send(client: Any, request_kwargs: Dict[str, Any], dest: _FallbackDestination) -> Any:
        return _validate_llm_response(
            _relay_sync_completion(
                client, request_kwargs, provider=dest.provider, api_mode=dest.api_mode,
                create=lambda request: _create_with_progress(
                    client, request, task,
                    force_stream=_provider_requires_stream(dest.provider, dest.base_url),
                ),
            ),
            task,
        )
    try:
        return _send(fb_client, fb_kwargs, destination)
    except Exception as fb_err:
        if not _is_auth_error(fb_err):
            raise
        fb_provider, retry = _plan_fallback_auth_retry(destination, rebuild, async_mode=False)
        if retry is not None:
            try:
                return _send(*retry)
            except Exception as retry_err:
                if not _is_auth_error(retry_err):
                    raise
        _quarantine_fallback_candidate(task, fb_label, fb_provider, fb_err)
        return None


async def _call_fallback_candidate_async(
    fb_client: Any, fb_model: Optional[str], fb_label: str, *, task: Optional[str], messages: list,
    temperature: Optional[float], max_tokens: Optional[int], tools: Optional[list],
    effective_timeout: float, effective_extra_body: dict, reasoning_config: Optional[dict],
) -> Optional[Any]:
    """Async mirror of :func:`_call_fallback_candidate_sync` (no fast-lane cap on this wire)."""
    destination, fb_kwargs, rebuild = _plan_fallback_candidate(
        fb_client, fb_model, fb_label, task=task, effective_timeout=effective_timeout,
        apply_fast_lane=False, messages=messages, tools=tools, temperature=temperature,
        max_tokens=max_tokens, effective_extra_body=effective_extra_body,
        reasoning_config=reasoning_config,
    )

    async def _send(client: Any, request_kwargs: Dict[str, Any], dest: _FallbackDestination) -> Any:
        return _validate_llm_response(
            await _relay_async_completion(client, request_kwargs, provider=dest.provider, api_mode=dest.api_mode),
            task,
        )
    try:
        return await _send(fb_client, fb_kwargs, destination)
    except Exception as fb_err:
        if not _is_auth_error(fb_err):
            raise
        fb_provider, retry = _plan_fallback_auth_retry(destination, rebuild, async_mode=True)
        if retry is not None:
            try:
                return await _send(*retry)
            except Exception as retry_err:
                if not _is_auth_error(retry_err):
                    raise
        _quarantine_fallback_candidate(task, fb_label, fb_provider, fb_err, tag=" (async)")
        return None


def _try_payment_fallback(
    failed_provider: str, task: str = None, reason: str = "payment error"
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try the auto-detection chain after a payment/credit or connection error, skipping the failed
    provider (and the main-provider path when it maps to the same backend). Returns (client, model, label) or (None, None, "")."""
    skip = failed_provider.lower().strip()
    main_provider = _read_main_provider()
    skip_labels = {skip}
    if main_provider and main_provider.lower() in skip:
        skip_labels.add(main_provider.lower())
    skip_chain_labels = {_normalize_chain_label(s) for s in skip_labels}
    tried = []
    for label, try_fn in _get_provider_chain():
        if label in skip_chain_labels:
            continue
        if _is_provider_unhealthy(label):
            _log_skip_unhealthy(label, task)
            tried.append(f"{label} (unhealthy)")
            continue
        client, model = try_fn()
        if client is not None:
            logger.info("Auxiliary %s: %s on %s — falling back to %s (%s)",
                        task or "call", reason, failed_provider, label, model or "default")
            return client, model, label
        tried.append(label)
    logger.warning("Auxiliary %s: %s on %s and no fallback available (tried: %s)",
                   task or "call", reason, failed_provider, ", ".join(tried))
    return None, None, ""


def _failed_backend_skip(failed_provider: str, failed_model: Optional[str]) -> Callable[..., bool]:
    """Predicate ``skip(provider, model, base_url="")`` → True when a candidate must be skipped for the failed
    route. Scope: ``failed_model`` → model-scoped (only that deployment; timeout/connection/rate-limit);
    None → credential-wide (whole provider; auth/payment)."""
    from agent.backend_identity import BackendIdentity, FailureScope, should_skip_candidate
    skip_model = (failed_model or "").strip().lower() or None
    failed_ident = BackendIdentity.build(provider=failed_provider, model=skip_model)
    failure_scope = FailureScope.MODEL if skip_model else FailureScope.CREDENTIAL

    def _skip(provider: str, model: Optional[str], base_url: str = "") -> bool:
        return should_skip_candidate(
            BackendIdentity.build(provider=provider, model=model, base_url=base_url), failed_ident, failure_scope,
        )
    return _skip


def _try_main_agent_model_fallback(
    failed_provider: str, task: str = None, reason: str = "error",
    failed_model: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str], str]:
    """Last-resort fallback to the main agent provider + model after the configured chain is exhausted.
    ``failed_model`` scoping per ``_failed_backend_skip``; same-URL custom endpoints serve many models,
    so a hung aux model says nothing about the main model's health. Returns (client, model, label) or (None, None, "")."""
    main_provider = (_read_main_provider() or "").strip()
    main_model = (_read_main_model() or "").strip()
    if main_provider.lower() == "moa":
        # MoA virtual provider: fall back to the preset's aggregator (the acting model).
        _agg_provider, _agg_model = _resolve_moa_aggregator(main_model)
        if not _agg_provider or not _agg_model:
            return None, None, ""
        main_provider, main_model = _agg_provider, _agg_model
    if not main_provider or not main_model or main_provider.lower() in {"auto", ""}:
        return None, None, ""
    if _failed_backend_skip(failed_provider, failed_model)(main_provider, main_model):
        return None, None, ""
    if _is_provider_unhealthy(main_provider):
        _log_skip_unhealthy(main_provider, task)
        return None, None, ""
    try:
        client, resolved_model = resolve_provider_client(provider=main_provider, model=main_model)
    except Exception:
        client, resolved_model = None, None
    if client is None:
        return None, None, ""
    label = f"main-agent({main_provider})"
    logger.info("Auxiliary %s: %s on %s — falling back to main agent model %s (%s)",
                task or "call", reason, failed_provider, label, resolved_model or main_model)
    return client, resolved_model or main_model, label


# Context-window screening for runtime fallback chains: the startup feasibility check filters
# too-small aux models; runtime chains must too, or compression stops at a reachable-but-too-small
# candidate. ``None`` (unknown) passes through.

# ── Context-window screening for runtime fallback chains (issue #52392) ── When the runtime auxiliary
# fallback chain selects a candidate that is reachable but has a context window smaller than the compression
# task requires, the call errors out instead of continuing to the next, viable candidate. The startup
# feasibility check in ``agent.conversation_compression.check_compression_model_feasibility`` already
# filters too-small auxiliary models at startup, but the runtime fallback chain
# (``_try_configured_fallback_chain`` and ``_try_main_fallback_chain``) does not apply the same filter, so
# compression can stop at the first alive door even if the room behind it is too small. The helpers below
# screen each candidate by its effective context window before it is returned. ``None`` results from
# ``get_model_context_length`` are passed through (we cannot prove a model is too small, so we do not block
# it). This preserves the existing fallback surface for unrecognised/custom models while closing the gap on
# the well-known ones.
def _task_minimum_context_length(task: Optional[str]) -> Optional[int]:
    """Minimum context length for an auxiliary task; None = no floor (only ``compression`` has one)."""
    return MINIMUM_CONTEXT_LENGTH if task == "compression" else None


def _candidate_context_window(provider: str, model: str, base_url: str = "", api_key: str = "") -> Optional[int]:
    """Best-effort context window for a fallback candidate; ``None`` = unknown (never raises; callers pass it through)."""
    if not model:
        return None
    try:
        ctx = get_model_context_length(model, base_url=base_url, api_key=api_key, provider=provider)
    except Exception as exc:
        logger.debug("Auxiliary fallback: could not resolve context window for %s/%s: %s", provider, model, exc)
        return None
    return ctx if isinstance(ctx, int) and ctx > 0 else None


def _context_too_small(
    entry: Dict[str, Any], provider: str, model: str, min_ctx: Optional[int], *,
    task: Optional[str], label: str, name_model: bool = False,
) -> Optional[str]:
    """Screen one fallback candidate by context window; returns the ``tried`` note when it is too small."""
    if min_ctx is None:
        return None
    fb_ctx = _candidate_context_window(
        provider, model, base_url=str(entry.get("base_url") or ""), api_key=_fallback_entry_api_key(entry) or "")
    if fb_ctx is None or fb_ctx >= min_ctx:
        return None
    if name_model:
        logger.info("Auxiliary %s: skipping %s (%s context=%d < min=%d), continuing chain",
                    task, label, model, fb_ctx, min_ctx)
    else:
        logger.info("Auxiliary %s: skipping %s (context=%d < min=%d), continuing chain",
                    task or "call", label, fb_ctx, min_ctx)
    return f"{label} (context too small: {fb_ctx}<{min_ctx})"


def _try_configured_fallback_chain(
    task: str, failed_provider: str, reason: str = "error", failed_model: Optional[str] = None
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try auxiliary.<task>.fallback_chain entries in order (each needs ``provider``; model/base_url/api_key optional).
    ``failed_model`` scoping per ``_failed_backend_skip`` (sibling models on the same provider still
    run after a model-scoped failure). Returns (client, model, provider_label) or (None, None, "")."""
    if not task:
        return None, None, ""
    chain = _get_auxiliary_task_config(task).get("fallback_chain")
    if not chain or not isinstance(chain, list):
        return None, None, ""
    skip = _failed_backend_skip(failed_provider, failed_model)
    tried = []
    min_ctx = _task_minimum_context_length(task)
    for i, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        fb_provider = str(entry.get("provider", "")).strip()
        if not fb_provider:
            continue
        fb_model_raw = str(entry.get("model", "")).strip()
        if skip(fb_provider, fb_model_raw, str(entry.get("base_url") or "")):
            continue
        fb_model = fb_model_raw or None
        label = f"fallback_chain[{i}]({fb_provider})"
        try:
            fb_client, resolved_model = _resolve_fallback_entry(entry)
        except Exception:
            fb_client, resolved_model = None, None
        if fb_client is not None:
            too_small = _context_too_small(
                entry, fb_provider, resolved_model, min_ctx, task=task, label=label, name_model=True,
            ) if resolved_model else None
            if too_small:
                tried.append(too_small)
                continue
            logger.info("Auxiliary %s: %s on %s — configured fallback to %s (%s)",
                        task, reason, failed_provider, label, resolved_model or fb_model or "default")
            return fb_client, resolved_model or fb_model, label
        tried.append(label)
    if tried:
        logger.debug("Auxiliary %s: configured fallback_chain exhausted (tried: %s)", task, ", ".join(tried))
    return None, None, ""


def _try_configured_fallback_for_unavailable_client(
    task: Optional[str], failed_provider: str
) -> Tuple[Optional[Any], Optional[str], str]:
    """Task fallback_chain when an explicit aux provider cannot build a client (no key/OAuth/pool creds);
    stops at the per-task chain — the main-agent model stays the runtime last resort."""
    explicit = (failed_provider or "").strip().lower()
    if not task or not explicit or explicit in {"auto"}:
        return None, None, ""
    return _try_configured_fallback_chain(task, explicit, reason="provider unavailable")


def _fallback_entry_api_key(entry: Dict[str, Any]) -> Optional[str]:
    """Resolve inline or env-backed API key via the secret-scope-aware resolver (no raw os.getenv under multiplexing)."""
    from hermes_cli.fallback_config import resolve_entry_api_key
    return resolve_entry_api_key(entry)


def _resolve_fallback_entry(entry: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    """Resolve one fallback entry through the central provider router."""
    provider = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or "").strip() or None
    if not provider or not model:
        return None, None
    client, resolved_model = resolve_provider_client(
        provider, model=model, explicit_base_url=str(entry.get("base_url") or "").strip() or None,
        explicit_api_key=_fallback_entry_api_key(entry),
        api_mode=str(entry.get("api_mode") or entry.get("transport") or "").strip() or None,
    )
    if client is not None:
        with contextlib.suppress(Exception):
            client._hermes_fallback_destination = _fallback_destination_from_entry(entry, client, resolved_model)
    return client, resolved_model


def _try_main_fallback_chain(
    task: Optional[str], failed_provider: str = "", reason: str = "error"
) -> Tuple[Optional[Any], Optional[str], str]:
    """Top-level main-agent fallback chain for a ``provider: auto`` auxiliary call: auto tasks honour the
    user's main fallback policy before the built-in discovery chain; read via ``get_fallback_chain`` so
    ``fallback_providers`` and legacy ``fallback_model`` keep the main agent's order."""
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.fallback_config import get_fallback_chain
        chain = get_fallback_chain(load_config_readonly())
    except Exception as exc:
        logger.debug("Auxiliary %s: could not load main fallback chain: %s", task or "call", exc)
        return None, None, ""
    if not chain:
        return None, None, ""
    skip = {p for p in ((failed_provider or "").strip().lower(), (_read_main_provider() or "").strip().lower(), "auto") if p}
    tried: List[str] = []
    min_ctx = _task_minimum_context_length(task)
    for i, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        fb_provider = str(entry.get("provider") or "").strip()
        fb_model = str(entry.get("model") or "").strip()
        if not fb_provider or not fb_model:
            continue
        fb_norm = fb_provider.lower()
        label = f"fallback_providers[{i}]({fb_provider})"
        if fb_norm in skip:
            tried.append(f"{label} (skipped)")
            continue
        if _is_provider_unhealthy(fb_norm):
            _log_skip_unhealthy(fb_norm, task)
            tried.append(f"{label} (unhealthy)")
            continue
        try:
            fb_client, resolved_model = _resolve_fallback_entry(entry)
        except Exception as exc:
            logger.debug("Auxiliary %s: main fallback %s failed to resolve: %s", task or "call", label, exc)
            fb_client, resolved_model = None, None
        if fb_client is not None:
            too_small = _context_too_small(
                entry, fb_provider, resolved_model or fb_model, min_ctx, task=task, label=label,
            )
            if too_small:
                tried.append(too_small)
                continue
            logger.info("Auxiliary %s: %s on %s — main fallback chain to %s (%s)",
                        task or "call", reason, failed_provider or "auto", label, resolved_model or fb_model)
            return fb_client, resolved_model or fb_model, fb_provider
        tried.append(label)
    if tried:
        logger.debug("Auxiliary %s: main fallback chain exhausted (tried: %s)", task or "call", ", ".join(tried))
    return None, None, ""


def _warn_stale_openai_base_url(runtime_provider: str) -> None:
    """Warn once when OPENAI_BASE_URL is set but config.yaml names a non-custom provider (a stale
    ~/.hermes/.env value after `hermes model` poisons routing)."""
    global _stale_base_url_warned
    if _stale_base_url_warned:
        return
    _env_base = os.getenv("OPENAI_BASE_URL", "").strip()
    _cfg_provider = runtime_provider or _read_main_provider()
    if (_env_base and _cfg_provider and _cfg_provider != "custom" and not _cfg_provider.startswith("custom:")):
        logger.warning(
            "OPENAI_BASE_URL is set (%s) but model.provider is '%s'. "
            "Auxiliary clients may route to the wrong endpoint. "
            "Run: hermes model to reconfigure, or remove "
            "OPENAI_BASE_URL from ~/.hermes/.env",
            _env_base, _cfg_provider,
        )
        _stale_base_url_warned = True


def _main_route_target(runtime: Dict[str, Any], task: Optional[str]) -> Tuple[str, str, str, Any, str]:
    """Step-1 target: (provider, model, base_url, api_key, api_mode) of the main runtime, after the
    fast-model opt-in and the MoA aggregator substitution."""
    main_provider = str(runtime.get("provider", "") or _read_main_provider() or "")
    main_model = str(runtime.get("model") or _read_main_model() or "")
    runtime_base_url = str(runtime.get("base_url") or "")
    runtime_api_key = runtime.get("api_key", "")
    runtime_api_mode = str(runtime.get("api_mode") or "")
    # Latency-critical tasks (titling only) opt in to the provider's fast model. Opt-in only:
    # every settings surface defines "auto" as the main model.
    if _task_prefers_fast_model(task) and main_provider and main_provider not in {"auto", ""}:
        fast_model = _get_aux_model_for_provider(main_provider, prefer_fast=True)
        if fast_model and fast_model != main_model:
            logger.debug("Auxiliary task %s: preferring fast model %s over main model %s",
                         task, fast_model, main_model)
            main_model = fast_model
    # MoA virtual provider: the preset name is not a wire model; run aux on the aggregator and drop
    # the facade's "moa://local" base_url / placeholder key so it uses its own credentials.
    if main_provider == "moa":
        _agg_provider, _agg_model = _resolve_moa_aggregator(main_model)
        if _agg_provider and _agg_model:
            main_provider, main_model = _agg_provider, _agg_model
            runtime_base_url = runtime_api_key = runtime_api_mode = ""
    return main_provider, main_model, runtime_base_url, runtime_api_key, runtime_api_mode


def _try_main_provider_route(
    main_provider: str, main_model: str, runtime_base_url: str, runtime_api_key: Any, runtime_api_mode: str,
) -> Optional[Tuple[Any, str, str]]:
    """Step 1: route aux onto the main provider + main model; None if unusable."""
    if not (main_provider and main_model and main_provider not in {"auto", ""}):
        return None
    resolved_provider = main_provider
    explicit_base_url = runtime_base_url or None
    explicit_api_key = None
    if runtime_base_url and main_provider == "custom":
        # Anonymous custom endpoint — pass through explicit base_url + api_key.
        explicit_api_key = runtime_api_key or None
    elif main_provider.startswith("custom:"):
        # Named custom provider (custom_providers / providers dict entry).
        _has_named_entry = False
        with contextlib.suppress(ImportError):
            from hermes_cli.runtime_provider import _get_named_custom_provider
            _has_named_entry = _get_named_custom_provider(main_provider) is not None
        if _has_named_entry:
            # KEEP the full ``custom:<name>`` so the named arm honours the entry's api_mode
            # (collapsing to "custom" strips /anthropic → 404s). base_url/api_key come from the entry.
            explicit_base_url = None
        elif runtime_base_url:
            # Config-less named custom provider (live runtime only): anonymous custom arm + runtime key.
            # See #34777.
            resolved_provider = "custom"
            explicit_api_key = runtime_api_key or None
        elif runtime_api_key:
            explicit_api_key = runtime_api_key
    elif runtime_api_key:
        # Pin aux to the main session's working key, not a re-selected (maybe exhausted) pool key.
        explicit_api_key = runtime_api_key
    # Skip if the main provider was recently 402'd (unhealthy TTL bounds the bypass).
    main_chain_label = _normalize_chain_label(resolved_provider)
    if main_chain_label and _is_provider_unhealthy(main_chain_label):
        _log_skip_unhealthy(main_chain_label)
        return None
    client, resolved = resolve_provider_client(
        resolved_provider, main_model, explicit_base_url=explicit_base_url,
        explicit_api_key=explicit_api_key, api_mode=runtime_api_mode or None,
    )
    if client is None:
        return None
    logger.info("Auxiliary auto-detect: using main provider %s (%s)", main_provider, resolved or main_model)
    return client, resolved or main_model, resolved_provider


def _try_discovery_chain() -> Tuple[Optional[OpenAI], Optional[str], str]:
    """Step 3: hardcoded aggregator/fallback chain, skipping unhealthy providers."""
    tried = []
    for label, try_fn in _get_provider_chain():
        if _is_provider_unhealthy(label):
            _log_skip_unhealthy(label)
            tried.append(f"{label} (unhealthy)")
            continue
        client, model = try_fn()
        if client is not None:
            if tried:
                logger.info("Auxiliary auto-detect: using %s (%s) — skipped: %s",
                            label, model or "default", ", ".join(tried))
            else:
                logger.info("Auxiliary auto-detect: using %s (%s)", label, model or "default")
            return client, model, label
        tried.append(label)
    logger.warning("Auxiliary auto-detect: no provider available (tried: %s). "
                   "Compression, summarization, and memory flush will not work. "
                   "Set OPENROUTER_API_KEY or configure a local model in config.yaml.", ", ".join(tried))
    return None, None, ""


def _resolve_auto_route(
    main_runtime: Optional[Dict[str, Any]] = None, task: Optional[str] = None
) -> Tuple[Optional[OpenAI], Optional[str], str]:
    """Full auto-detection chain, including the selected provider identity. Priority: (1) main provider +
    main model, regardless of provider type ("auto" means "my main model for side tasks too"; explicit
    per-task overrides still win); (2) configured fallback policy — task chain, then the main agent's
    top-level chain; (3) OpenRouter → Nous → custom → Codex → API-key providers, only with no policy
    and no working main client."""
    global auxiliary_is_nous
    auxiliary_is_nous = False  # Reset — _try_nous() will set True if it wins
    runtime = _normalize_main_runtime(main_runtime)
    _warn_stale_openai_base_url(runtime.get("provider", ""))
    main_provider, main_model, base_url, api_key, api_mode = _main_route_target(runtime, task)
    if api_mode == "claude_agent_sdk":
        return _resolve_claude_sdk_auto_route(main_model)
    routed = _try_main_provider_route(main_provider, main_model, base_url, api_key, api_mode)
    if routed is not None:
        return routed
    if task:
        fb_client, fb_model, fb_label = _try_configured_fallback_chain(
            task, main_provider or "auto", reason="main provider unavailable")
        if fb_client is not None:
            return fb_client, fb_model, _fallback_provider_from_label(fb_label)
    fb_client, fb_model, fb_label = _try_main_fallback_chain(
        task, main_provider or "auto", reason="main provider unavailable")
    if fb_client is not None:
        return fb_client, fb_model, fb_label
    return _try_discovery_chain()


def _effective_provider_for_client(client: Any, fallback: str) -> str:
    """Return the concrete provider selected for an auto-routed client."""
    effective_provider = getattr(client, "_hermes_aux_effective_provider", "")
    if isinstance(effective_provider, str) and effective_provider:
        return effective_provider
    return str(fallback or "")


def _tag_effective_provider(client: Any, provider: str) -> None:
    """Retain auto-routing identity on the client that survives cache reuse."""
    if client is None or not provider:
        return
    try:
        setattr(client, "_hermes_aux_effective_provider", provider)
    except (AttributeError, TypeError):
        logger.debug("Auxiliary client %s cannot retain effective provider %s",
                     type(client).__name__, provider)


def _resolve_claude_sdk_auto_route(main_model: str) -> Tuple[Optional[Any], Optional[str], str]:
    """Fail-closed subscription lane (claude-agent-sdk, #25267).

    When the MAIN provider is the claude-agent-sdk (subscription OAuth, never metered),
    auto-detection must NOT silently re-route auxiliary tasks (title generation, context
    compression, ...) onto metered fallback providers — that would break the provider's
    billing contract through the side door. Explicit ``auxiliary.<task>.{provider,model}``
    config is resolved before this chain and remains the operator's deliberate opt-in;
    without it, aux work stays on the same subscription-owned SDK route, and if that
    facade is unavailable aux features simply no-op.
    """
    try:
        from agent.claude_sdk_aux_client import ClaudeSdkAuxClient

        sdk_aux_model = main_model or "claude-sonnet-5"
        sdk_aux_client = ClaudeSdkAuxClient(default_model=sdk_aux_model)
        _tag_effective_provider(sdk_aux_client, "claude-agent-sdk")
        logger.debug("aux auto-detect: routing to claude-agent-sdk one-shot (subscription lane, model=%s)",
                     sdk_aux_model)
        return sdk_aux_client, sdk_aux_model, "claude-agent-sdk"
    except Exception:
        logger.warning("aux auto-detect: claude-agent-sdk one-shot client unavailable; failing closed "
                       "rather than routing auxiliary work to a metered provider.", exc_info=True)
        return None, None, ""


# Centralized Provider Router: resolve_provider_client() is the single entry point for building a configured
# client (auth, base URL, headers, API format) from (provider, model). Never read auth env vars ad-hoc.


def _to_async_client(sync_client, model: str, is_vision: bool = False):
    """Sync client → async counterpart, preserving Codex routing (``is_vision`` adds the Copilot vision header)."""
    from openai import AsyncOpenAI
    if isinstance(sync_client, _AuxProbeClientStub):
        return sync_client, model
    if isinstance(sync_client, CodexAuxiliaryClient):
        return AsyncCodexAuxiliaryClient(sync_client), model
    if isinstance(sync_client, AnthropicAuxiliaryClient):
        return AsyncAnthropicAuxiliaryClient(sync_client), model
    if isinstance(sync_client, BedrockAuxiliaryClient):
        return AsyncBedrockAuxiliaryClient(sync_client), model
    with contextlib.suppress(ImportError):
        from agent.claude_sdk_aux_client import AsyncClaudeSdkAuxClient, ClaudeSdkAuxClient

        if isinstance(sync_client, ClaudeSdkAuxClient):
            async_client = AsyncClaudeSdkAuxClient(sync_client)
            _tag_effective_provider(async_client, _effective_provider_for_client(sync_client, ""))
            return async_client, model
    with contextlib.suppress(ImportError):
        from agent.gemini_native_adapter import GeminiNativeClient, AsyncGeminiNativeClient
        if isinstance(sync_client, GeminiNativeClient):
            return AsyncGeminiNativeClient(sync_client), model
    # ACP shims (subprocess, not an HTTP pool) are already async-safe and opt out of the wrapper.
    if _client_declares(sync_client, "HERMES_SKIP_ASYNC_WRAP"):
        return sync_client, model
    sync_base_url = str(sync_client.base_url)
    async_kwargs = {"api_key": sync_client.api_key, "base_url": sync_base_url}
    if base_url_host_matches(sync_base_url, "openrouter.ai"):
        headers = _apply_user_default_headers(build_or_headers())
    elif _is_official_codex_base_url(sync_base_url):
        headers = _apply_user_default_headers(_codex_cloudflare_headers(sync_client.api_key, base_url=sync_base_url))
    else:
        # Provider for the profile-header fallback is inferred from the hostname.
        try:
            from agent.model_metadata import _infer_provider_from_url
            inferred = _infer_provider_from_url(sync_base_url) or ""
        except Exception:
            inferred = ""
        headers = _endpoint_default_headers(sync_base_url, inferred, is_vision=is_vision, xai=True)
    if headers:
        async_kwargs["default_headers"] = headers
    _apply_required_codex_headers(async_kwargs, access_token=sync_client.api_key, base_url=sync_base_url)
    async_kwargs = {**_openai_http_client_kwargs(sync_base_url, async_mode=True), **async_kwargs}
    # Hermes owns the auxiliary retry/timeout budget; disable SDK-internal retries.
    # See #54465.
    async_kwargs.setdefault("max_retries", 0)
    return AsyncOpenAI(**async_kwargs), model


def _normalize_resolved_model(model_name: Optional[str], provider: str) -> Optional[str]:
    """Normalize a resolved model for the provider that will receive it."""
    if not model_name:
        return model_name
    try:
        from hermes_cli.model_normalize import normalize_model_for_provider
        return normalize_model_for_provider(model_name, provider)
    except Exception:
        return model_name


def _named_custom_api_key(custom_entry: Dict[str, Any], provider: str, custom_base: str) -> Any:
    """Credential for a named custom provider: inline api_key → key_env → key_cmd → credential pool → placeholder.
    Aux resolves named custom providers here, not via _resolve_named_custom_runtime, so key_cmd must be
    honoured at the same precedence or every aux call 401s."""
    custom_key: Any = (custom_entry.get("api_key") or "").strip()
    custom_key_env = (custom_entry.get("key_env") or custom_entry.get("api_key_env") or "").strip()
    if not custom_key and custom_key_env:
        custom_key = _scoped_key_env(custom_key_env)
    custom_key_cmd = str(custom_entry.get("key_cmd", "") or "").strip()
    if custom_key_cmd:
        from agent.command_token_source import build_command_token_provider
        custom_key = build_command_token_provider(custom_key_cmd, custom_entry.get("name") or provider) or custom_key
    if not custom_key:
        with contextlib.suppress(Exception):
            from agent.credential_pool import custom_provider_pool_key_candidates
            pool_name = custom_entry.get("provider_key") or custom_entry.get("name") or provider
            for pool_key in custom_provider_pool_key_candidates(custom_base, pool_name):
                try:
                    pool = load_pool(pool_key)
                except Exception:
                    continue
                if not pool.has_credentials():
                    continue
                pool_entry = pool.select()
                if pool_entry is None:
                    continue
                pool_api_key = getattr(pool_entry, "runtime_api_key", None) or getattr(pool_entry, "access_token", "") or ""
                if str(pool_api_key).strip():
                    custom_key = str(pool_api_key).strip()
                    break
    return custom_key or "no-key-required"


def _build_bedrock_client(provider: str, model: Optional[str], *, raw_codex: bool) -> Tuple[Optional[Any], Optional[str]]:
    """AWS Bedrock: Claude → Anthropic Bedrock SDK (prompt caching, thinking); OpenAI models
    (GPT-5.5/5.6) → Bedrock Mantle's OpenAI Responses endpoint; everything else → Converse API."""
    try:
        from agent.bedrock_adapter import (
            has_aws_credentials, is_anthropic_bedrock_model, resolve_bedrock_runtime_region,
            is_openai_bedrock_model, bedrock_openai_base_url, resolve_bedrock_bearer_token,
            configure_bedrock_openai_client_kwargs,
        )
        from agent.anthropic_adapter import build_anthropic_bedrock_client
    except ImportError:
        logger.warning("resolve_provider_client: bedrock requested but boto3, httpx/openai, or anthropic SDK not installed")
        return None, None
    if not has_aws_credentials():
        logger.debug("resolve_provider_client: bedrock requested but no AWS credentials found")
        return None, None
    # Region must match the main runtime's resolution (bedrock.region in config first, then
    # env/profile) so aux calls never leave the primary runtime's configured region.
    # See #53880, #65076.
    region = resolve_bedrock_runtime_region()
    default_model = "anthropic.claude-haiku-4-5-20251001-v1:0"
    final_model = _normalize_resolved_model(model or default_model, provider) or default_model
    if is_openai_bedrock_model(final_model):
        # Module-level lazy ``OpenAI`` proxy on purpose so tests can patch("agent.auxiliary_client.OpenAI").
        client_kwargs: Dict[str, Any] = {
            "api_key": resolve_bedrock_bearer_token() or "aws-sdk",
            "base_url": bedrock_openai_base_url(region),
        }
        configure_bedrock_openai_client_kwargs(client_kwargs)
        client = OpenAI(**client_kwargs)
        logger.debug("resolve_provider_client: bedrock-openai (%s, %s)", final_model, region)
        return (client if raw_codex else CodexAuxiliaryClient(client, final_model)), final_model
    base_url = f"https://bedrock-runtime.{region}.amazonaws.com"
    if is_anthropic_bedrock_model(final_model):
        try:
            real_client = build_anthropic_bedrock_client(region)
        except ImportError as exc:
            logger.warning("resolve_provider_client: cannot create Bedrock client: %s", exc)
            return None, None
        client = AnthropicAuxiliaryClient(real_client, final_model, api_key="aws-sdk", base_url=base_url)
        logger.debug("resolve_provider_client: bedrock anthropic (%s, %s)", final_model, region)
    else:
        client = BedrockAuxiliaryClient(region, final_model)
        logger.debug("resolve_provider_client: bedrock converse (%s, %s)", final_model, region)
    return client, final_model


def _build_vertex_client(provider: str, model: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
    """Google Vertex AI: Gemini via the OpenAI-compatible endpoint with an OAuth2 bearer (standard OpenAI client)."""
    try:
        from agent.vertex_adapter import get_vertex_config, has_vertex_credentials
    except ImportError:
        logger.warning("resolve_provider_client: vertex requested but google-auth not installed")
        return None, None
    if not has_vertex_credentials():
        logger.debug("resolve_provider_client: vertex requested but no GCP credentials found")
        return None, None
    token, base_url = get_vertex_config()
    if not token or not base_url:
        logger.warning("resolve_provider_client: vertex requested but could not mint token / resolve project")
        return None, None
    final_model = _normalize_resolved_model(model or "google/gemini-3-flash-preview", provider)
    try:
        # Aliased import: a bare `from openai import OpenAI` would shadow the module-level lazy proxy.
        from openai import OpenAI as _VertexOpenAI
        client = _VertexOpenAI(api_key=token, base_url=base_url)
    except Exception as exc:
        logger.warning("resolve_provider_client: cannot create Vertex client: %s", exc)
        return None, None
    logger.debug("resolve_provider_client: vertex (%s)", final_model)
    return client, final_model


class _ResolveRequest(NamedTuple):
    """Normalized resolve_provider_client() arguments shared by the per-provider branch helpers."""
    provider: str
    original_provider: str
    model: Optional[str]
    async_mode: bool
    raw_codex: bool
    explicit_base_url: Optional[str]
    explicit_api_key: Optional[str]
    api_mode: Optional[str]
    main_runtime: Optional[Dict[str, Any]]
    is_vision: bool
    task: Optional[str]


_ResolveResult = Tuple[Optional[Any], Optional[str]]


def _log_once_debug(seen: set, key: Any, msg: str, *args: Any) -> None:
    """Debug-log ``msg`` the first time ``key`` is seen so per-call retries stay silent."""
    if key not in seen:
        seen.add(key)
        logger.debug(msg, *args)


def _wrap_transport(req: _ResolveRequest, client_obj: Any, final_model_str: str,
                    base_url_str: str = "", api_key_str: str = ""):
    """Wrap a plain OpenAI client in the right transport adapter; specialized wrappers pass through.
    Codex (Responses API): explicit ``api_mode=codex_responses`` (or provider ``actual``), else — with no
    explicit api_mode — api.openai.com + codex model. Anthropic (Messages): ``api_mode=anthropic_messages``,
    any ``/anthropic`` suffix, ``api.kimi.com/coding``, or ``api.anthropic.com``."""
    needs_codex = not (isinstance(client_obj, CodexAuxiliaryClient) or req.raw_codex) and (
        req.provider == "actual" or req.api_mode == "codex_responses"
        or (not req.api_mode and base_url_hostname(base_url_str) == "api.openai.com"
            and "codex" in (final_model_str or "").lower())
    )
    if needs_codex:
        logger.debug("resolve_provider_client: wrapping client in CodexAuxiliaryClient "
                     "(api_mode=%s, model=%s, base_url=%s)",
                     req.api_mode or "auto-detected", final_model_str, base_url_str[:60] if base_url_str else "")
        return CodexAuxiliaryClient(client_obj, final_model_str)
    return _maybe_wrap_anthropic(client_obj, final_model_str, api_key_str, base_url_str, req.api_mode)


def _route_client(req: _ResolveRequest, client_obj: Any, final_model_str: Optional[str]) -> _ResolveResult:
    """Return (client, model), converting to the async wrapper when ``req.async_mode``."""
    if req.async_mode:
        return _to_async_client(client_obj, final_model_str, is_vision=req.is_vision)
    return client_obj, final_model_str


def _route_or_warn(req: _ResolveRequest, client: Any, default: Optional[str], unavailable_msg: str, *args: Any) -> _ResolveResult:
    """Route ``client`` on ``req.model or default``; warn and return (None, None) when the provider produced no client."""
    if client is None:
        logger.warning(unavailable_msg, *args)
        return None, None
    return _route_client(req, client, _normalize_resolved_model(req.model or default, req.provider))


def _resolve_auto_branch(req: _ResolveRequest) -> _ResolveResult:
    """Auto: try all providers in priority order; tag the client with the effective provider (survives cache reuse)."""
    client, resolved, effective_provider = _resolve_auto_route(main_runtime=req.main_runtime, task=req.task)
    if client is None:
        return None, None
    model = req.model
    # An OpenRouter-format model override won't work on a non-OpenRouter provider (e.g. local
    # server); drop it for the provider's default.
    if model and "/" in model and resolved and "/" not in resolved:
        logger.debug("Dropping OpenRouter-format model %r for non-OpenRouter "
                     "auxiliary provider (using %r instead)", model, resolved)
        model = None
    routed_client, routed_model = _route_client(req, client, model or resolved)
    if routed_client is not None and effective_provider:
        try:
            setattr(routed_client, "_hermes_aux_effective_provider", effective_provider)
        except (AttributeError, TypeError):
            logger.debug("Auxiliary client %s cannot retain effective provider %s",
                         type(routed_client).__name__, effective_provider)
    return routed_client, routed_model


def _resolve_openrouter_branch(req: _ResolveRequest) -> _ResolveResult:
    """OpenRouter."""
    client, default = _try_openrouter(explicit_api_key=req.explicit_api_key, model=req.model)
    if client is None:
        logger.warning("resolve_provider_client: openrouter requested but %s",
                       _describe_openrouter_unavailable(model=req.model))
        return None, None
    return _route_client(req, client, _normalize_resolved_model(req.model or default, req.provider))


def _resolve_nous_branch(req: _ResolveRequest) -> _ResolveResult:
    """Nous Portal (OAuth)."""
    model = req.model
    # Vision: caller flag, _PROVIDER_VISION_MODELS override, or a known vision id.
    client, default = _try_nous(vision=(req.is_vision or model in _PROVIDER_VISION_MODELS.values()
                                        or (model or "").strip().lower() == "mimo-v2-omni"))
    if client is None:
        logger.warning("resolve_provider_client: nous requested but Nous Portal not configured (run: hermes auth)")
        return None, None
    final_model = _normalize_resolved_model(model or default, req.provider)
    # Dual-wire: anthropic/* → /v1/messages, else /chat/completions. Derive from the catalog id
    # (not a stale api_mode) so aux matches the main agent.
    from hermes_cli.providers import nous_api_mode
    client = _maybe_wrap_anthropic(
        client, final_model, str(getattr(client, "api_key", "") or ""),
        str(getattr(client, "base_url", "") or ""), nous_api_mode(final_model),
    )
    return _route_client(req, client, final_model)


def _resolve_openai_codex_branch(req: _ResolveRequest) -> _ResolveResult:
    """OpenAI Codex (OAuth → Responses API)."""
    model = req.model
    if not model:
        logger.warning("resolve_provider_client: openai-codex requested without a "
                       "model; pass model explicitly (e.g. model.model in config.yaml "
                       "or auxiliary.<task>.model for per-task aux routing).")
        return None, None
    no_token_msg = "resolve_provider_client: openai-codex requested but no Codex OAuth token found (run: hermes model)"
    if req.raw_codex:
        # Raw OpenAI client for callers needing responses.stream() (main agent loop).
        codex_token = _read_codex_access_token()
        if not codex_token:
            logger.warning(no_token_msg)
            return None, None
        raw_client = _create_openai_client(api_key=codex_token, base_url=_CODEX_AUX_BASE_URL,
                                           default_headers=_codex_cloudflare_headers(codex_token))
        return raw_client, _normalize_resolved_model(model, req.provider)
    client, default = _build_codex_client(model)
    return _route_or_warn(req, client, default, no_token_msg)


def _resolve_xai_oauth_branch(req: _ResolveRequest) -> _ResolveResult:
    """xAI Grok OAuth (device code → Responses API). Without this branch xai-oauth falls to the generic
    oauth_external arm, returns (None, None), and silently re-routes every aux task to the Step-2 fallback."""
    client, default = _build_xai_oauth_aux_client(req.model)
    return _route_or_warn(req, client, default,
                          "resolve_provider_client: xai-oauth requested but no xAI "
                          "OAuth token found (run: hermes model -> xAI Grok OAuth — SuperGrok / Premium+)")


def _resolve_custom_branch(req: _ResolveRequest) -> _ResolveResult:
    """Custom endpoint (OPENAI_BASE_URL + OPENAI_API_KEY)."""
    provider, model, main_runtime = req.provider, req.model, req.main_runtime
    # wrap_base: base for the Anthropic-wrap decision. anthropic_messages must keep the raw
    # /anthropic base while the plain OpenAI client uses the /v1-rewritten custom_base (never
    # /anthropic/chat/completions). Empty means "use custom_base".
    custom_base = custom_key = wrap_base = ""
    if req.explicit_base_url:
        custom_base = _to_openai_base_url(req.explicit_base_url).strip()
        if req.api_mode == "anthropic_messages":
            wrap_base = (req.explicit_base_url or "").strip().rstrip("/")
        custom_key = (
            (req.explicit_api_key or "").strip()
            or _scoped_key_env("OPENAI_API_KEY")
            or _read_main_api_key_if_same_host(custom_base)
            or "no-key-required"  # local servers don't need auth
        )
        if not custom_base:
            logger.warning("resolve_provider_client: explicit custom endpoint requested but base_url is empty")
            return None, None
    elif main_runtime:
        # Reuse main_runtime's concrete base_url + api_key for a named custom provider;
        # re-resolving from bare "custom" loses the name and lands on the wrong provider.
        # Re-resolution loses the provider name and falls back to OpenRouter or a wrong API-key provider —
        # the main agent already solved this, we just need to reuse its answer. (#45472)
        _main_base = str(main_runtime.get("base_url") or "").strip().rstrip("/")
        _main_key = str(main_runtime.get("api_key") or "").strip()
        if _main_base and _main_key:
            custom_base, custom_key = _main_base, _main_key
    if custom_base and custom_key:
        final_model = _normalize_resolved_model(
            model or (main_runtime.get("model") if main_runtime else None) or "gpt-4o-mini", provider,
        )
        extra = {}
        _clean_base, _dq = _extract_url_query_params(custom_base)
        if _dq:
            extra["default_query"] = _dq
        _custom_headers = _endpoint_default_headers(custom_base, provider, is_vision=req.is_vision)
        if _custom_headers:
            extra["default_headers"] = _custom_headers
        client = _create_openai_client(api_key=custom_key, base_url=_clean_base, **extra)
        client = _wrap_transport(req, client, final_model, wrap_base or custom_base, custom_key)
        return _route_client(req, client, final_model)
    # Try custom first, then API-key providers (Codex excluded here:
    # falling through to Codex with no model is a stale-constant trap).
    for try_fn in (_try_custom_endpoint, _resolve_api_key_provider):
        client, default = try_fn()
        if client is not None:
            final_model = _normalize_resolved_model(model or default, provider)
            # ``client.api_key`` may be a callable (Azure Entra bearer provider);
            # wrapping decisions only need base_url + api_mode.
            _raw_ckey = getattr(client, "api_key", "")
            _ckey = "" if (callable(_raw_ckey) and not isinstance(_raw_ckey, str)) else str(_raw_ckey or "")
            client = _wrap_transport(req, client, final_model, str(getattr(client, "base_url", "") or ""), _ckey)
            return _route_client(req, client, final_model)
    logger.warning("resolve_provider_client: custom/main requested but no endpoint credentials found")
    return None, None


def _named_custom_openai_wire_client(custom_base: str, custom_key: Any):
    """Plain OpenAI client on the /v1 equivalent of a named custom entry's base URL."""
    _clean_base, _dq = _extract_url_query_params(_to_openai_base_url(custom_base))
    _extra = {"default_query": _dq} if _dq else {}
    _headers = _apply_user_default_headers(None)
    if _headers:
        _extra["default_headers"] = _headers
    return _create_openai_client(api_key=custom_key, base_url=_clean_base, **_extra)


def _resolve_named_custom_branch(req: _ResolveRequest) -> Optional[_ResolveResult]:
    """Named custom provider (config.yaml providers dict / custom_providers list); None if no entry matches."""
    from hermes_cli.runtime_provider import _get_named_custom_provider
    provider = req.provider
    # If the raw name is an alias (``kimi`` → ``kimi-coding``) and a custom_providers entry exists
    # under it, the custom entry wins over alias rewriting. Only for aliases, so entries matching a
    # canonical name (e.g. ``nous``) still defer to the built-in.
    custom_entry = None
    if req.original_provider and req.original_provider != provider:
        custom_entry = _get_named_custom_provider(req.original_provider)
    if custom_entry is None:
        custom_entry = _get_named_custom_provider(provider)
    if not custom_entry:
        return None
    custom_base = (custom_entry.get("base_url") or "").strip()
    custom_key = _named_custom_api_key(custom_entry, provider, custom_base)
    if custom_key == "no-key-required":
        logger.warning("resolve_provider_client: named custom provider %r has no resolvable "
                       "api_key — request will be sent with placeholder no-key-required "
                       "and will 401 on auth-required endpoints", custom_entry.get("name") or provider)
    # Explicit per-task api_mode override wins over the provider entry's.
    entry_api_mode = (req.api_mode or custom_entry.get("api_mode") or "").strip()
    if not custom_base:
        logger.warning("resolve_provider_client: named custom provider %r has no base_url", provider)
        return None, None
    final_model = _normalize_resolved_model(
        req.model
        or custom_entry.get("model")
        or (req.main_runtime.get("model") if req.main_runtime else None)
        or _read_main_model_for_aux()
        or "gpt-4o-mini",
        provider,
    )
    logger.debug("resolve_provider_client: named custom provider %r (%s, api_mode=%s)",
                 provider, final_model, entry_api_mode or "chat_completions")
    # anthropic_messages: route via AnthropicAuxiliaryClient (mirrors _try_custom_endpoint);
    # the Anthropic SDK sees the original (un-rewritten) URL.
    # Mirrors the anonymous-custom branch in _try_custom_endpoint(). See #15033.
    if entry_api_mode == "anthropic_messages":
        try:
            from agent.anthropic_adapter import build_anthropic_client
            real_client = build_anthropic_client(custom_key, custom_base)
        except ImportError:
            logger.warning("Named custom provider %r declares api_mode=anthropic_messages but the anthropic SDK "
                           "is not installed — falling back to OpenAI-wire.", provider)
            return _route_client(req, _named_custom_openai_wire_client(custom_base, custom_key), final_model)
        return _route_client(
            req, AnthropicAuxiliaryClient(real_client, final_model, custom_key, custom_base, is_oauth=False), final_model)
    client = _named_custom_openai_wire_client(custom_base, custom_key)
    # codex_responses, or auto-detect via _wrap_transport (which reads the task-level api_mode).
    if entry_api_mode == "codex_responses":
        client = CodexAuxiliaryClient(client, final_model)
    else:
        client = _wrap_transport(req, client, final_model, custom_base, custom_key)
    return _route_client(req, client, final_model)


def _resolve_azure_foundry_branch(req: _ResolveRequest) -> _ResolveResult:
    """Azure Foundry via the runtime resolver: the generic PROVIDER_REGISTRY path only knows the static
    AZURE_FOUNDRY_API_KEY env var, missing ``auth_mode: entra_id`` (callable bearer) and config base_url overrides."""
    client, default_model = _try_azure_foundry(model=req.model, explicit_api_key=req.explicit_api_key,
                                               explicit_base_url=req.explicit_base_url, api_mode=req.api_mode)
    return _route_or_warn(req, client, default_model,
                          "resolve_provider_client: azure-foundry requested but "
                          "runtime resolution failed (run: hermes doctor for diagnostics)")


def _resolve_api_key_branch(req: _ResolveRequest, pconfig: Any, resolve_creds: Callable) -> _ResolveResult:
    """PROVIDER_REGISTRY ``api_key`` providers (Anthropic via its own resolver), honouring explicit overrides."""
    provider = req.provider
    if provider == "anthropic":
        client, default_model = _try_anthropic(explicit_api_key=req.explicit_api_key)
        return _route_or_warn(req, client, default_model,
                              "resolve_provider_client: anthropic requested but no Anthropic credentials found")
    creds = resolve_creds(provider)
    api_key = str(creds.get("api_key", "")).strip()
    # Explicit api_key override (fallback_model / custom_providers entry) lets callers
    # authenticate where no built-in credential is registered for this alias.
    if req.explicit_api_key:
        api_key = req.explicit_api_key.strip() or api_key
    raw_base_url = str(creds.get("base_url", "")).strip().rstrip("/") or pconfig.inference_base_url
    if req.explicit_base_url:
        raw_base_url = req.explicit_base_url.strip().rstrip("/")
    # OpenCode Zen free tier (*-free slugs) is served anonymously on the Zen relay only;
    # any bearer (even a Go subscription key) is rejected, so route keyless regardless of creds.
    try:
        from hermes_cli.models import opencode_zen_free_runtime as _oc_free_rt
        _free_rt = _oc_free_rt(provider, req.model)
    except Exception:
        _free_rt = None
    if _free_rt is not None:
        api_key = _free_rt["api_key"]
        raw_base_url = str(_free_rt["base_url"]).rstrip("/")
    if provider == "actual":
        with contextlib.suppress(Exception):
            from hermes_cli.auth import (
                ACTUAL_LOCAL_NOAUTH_PLACEHOLDER, is_actual_local_base_url, normalize_actual_base_url
            )
            raw_base_url = normalize_actual_base_url(raw_base_url)
            if not api_key and is_actual_local_base_url(raw_base_url):
                api_key = ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
    if not api_key:
        tried_sources = list(pconfig.api_key_env_vars) + (["gh auth token"] if provider == "copilot" else [])
        logger.debug("resolve_provider_client: provider %s has no API key configured (tried: %s)",
                     provider, ", ".join(tried_sources))
        return None, None
    base_url = _to_openai_base_url(raw_base_url)
    # Explicit base_url override: a fallback_model/custom_providers entry pointing a built-in name elsewhere.
    if req.explicit_base_url:
        base_url = _to_openai_base_url(req.explicit_base_url.strip().rstrip("/"))
    final_model = _normalize_resolved_model(req.model or _get_aux_model_for_provider(provider), provider)
    if provider == "gemini":
        from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url
        if is_native_gemini_base_url(base_url):
            client = GeminiNativeClient(api_key=api_key, base_url=base_url)
            logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
            return _route_client(req, client, final_model)
    headers = _endpoint_default_headers(base_url, provider, is_vision=req.is_vision, xai=True)
    client = _create_openai_client(api_key=api_key, base_url=base_url, **({"default_headers": headers} if headers else {}))
    # Copilot GPT-5+ models (except gpt-5-mini) are only reachable via the Responses API;
    # wrap so call_llm() transparently routes through responses.stream().
    if provider == "copilot" and final_model and not req.raw_codex:
        with contextlib.suppress(ImportError):
            from hermes_cli.models import _should_use_copilot_responses_api
            if _should_use_copilot_responses_api(final_model):
                logger.debug("resolve_provider_client: copilot model %s needs "
                             "Responses API — wrapping with CodexAuxiliaryClient", final_model)
                client = CodexAuxiliaryClient(client, final_model)
    # api_mode handling for any API-key provider (direct OpenAI + codex model) and Anthropic-wire
    # endpoints (api.kimi.com/coding, /anthropic gateways) without per-provider branches.
    client = _wrap_transport(req, client, final_model, raw_base_url, api_key)
    logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
    return _route_client(req, client, final_model)


def _resolve_external_process_branch(req: _ResolveRequest, creds: Dict[str, Any]) -> _ResolveResult:
    """PROVIDER_REGISTRY ``external_process`` providers, served via their registered profile."""
    provider = req.provider
    final_model = _normalize_resolved_model(
        req.model or (req.main_runtime.get("model") if req.main_runtime else None) or _read_main_model_for_aux(),
        provider,
    )
    # Keyed on the registered profile, not a provider name, so an out-of-tree ACP provider reaches
    # the auxiliary path (compression, vision, background review) exactly like the in-tree one.
    try:
        from providers import get_provider_profile as _get_provider_profile
        _extproc_profile = _get_provider_profile(provider)
    except Exception:
        _extproc_profile = None
    if _extproc_profile is not None:
        api_key = str(creds.get("api_key", "")).strip()
        base_url = str(creds.get("base_url", "")).strip()
        if not final_model:
            logger.warning("resolve_provider_client: %s requested but no model was provided or configured", provider)
            return None, None
        if not api_key or not base_url:
            logger.warning("resolve_provider_client: %s requested but external process credentials are incomplete", provider)
            return None, None
        try:
            client = _extproc_profile.create_client(
                api_key=api_key, base_url=base_url,
                command=str(creds.get("command", "")).strip() or None, args=list(creds.get("args") or []))
        except Exception:
            logger.warning("resolve_provider_client: profile %r failed to create an external-process client",
                           provider, exc_info=True)
            client = None
        if client is not None:
            logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
            return _route_client(req, client, final_model)
    _log_once_debug(_LOGGED_UNSUPPORTED_EXTPROC_KEYS, provider,
                    "resolve_provider_client: external-process provider %s not "
                    "directly supported", provider)
    return None, None


def _resolve_registry_branch(req: _ResolveRequest) -> _ResolveResult:
    """PROVIDER_REGISTRY providers, dispatched on ``auth_type``; unknown providers log once."""
    provider = req.provider
    try:
        from hermes_cli.auth import (
            PROVIDER_REGISTRY, resolve_api_key_provider_credentials,
            resolve_external_process_provider_credentials,
        )
    except ImportError:
        logger.debug("hermes_cli.auth not available for provider %s", provider)
        return None, None
    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig is None:
        _log_once_debug(_LOGGED_UNKNOWN_PROVIDER_KEYS, provider,
                        "resolve_provider_client: unknown provider %r", provider)
        return None, None
    auth_type = pconfig.auth_type
    if auth_type == "api_key":
        return _resolve_api_key_branch(req, pconfig, resolve_api_key_provider_credentials)
    if auth_type == "external_process":
        return _resolve_external_process_branch(req, resolve_external_process_provider_credentials(provider))
    if auth_type == "vertex":
        client, final_model = _build_vertex_client(provider, req.model)
    elif auth_type == "aws_sdk":
        client, final_model = _build_bedrock_client(provider, req.model, raw_codex=req.raw_codex)
    elif auth_type in {"oauth_device_code", "oauth_external"}:
        # nous / openai-codex / xai-oauth already returned from their explicit branches.
        _log_once_debug(_LOGGED_UNSUPPORTED_OAUTH_KEYS, provider,
                        "resolve_provider_client: OAuth provider %s not "
                        "directly supported, try 'auto'", provider)
        return None, None
    else:
        # The first occurrence surfaces a real schema-drift bug; per-call retries stay silent.
        _log_once_debug(_LOGGED_UNHANDLED_AUTHTYPE_KEYS, (auth_type, provider),
                        "resolve_provider_client: unhandled auth_type %s for %s",
                        auth_type, provider)
        return None, None
    return _route_client(req, client, final_model) if client is not None else (None, None)


def _resolve_claude_agent_sdk_branch(req: _ResolveRequest) -> _ResolveResult:
    """Subscription-owned Agent SDK auxiliary facade."""
    from agent.claude_sdk_aux_client import ClaudeSdkAuxClient

    final_model = _normalize_resolved_model(
        req.model or "claude-sonnet-5", req.provider
    )
    return _route_client(req, ClaudeSdkAuxClient(default_model=final_model), final_model)


# Explicit providers with a dedicated branch; anything else falls through to named custom
# providers → azure-foundry → PROVIDER_REGISTRY (order preserved from the original if-chain).
_EXPLICIT_PROVIDER_BRANCHES: Dict[str, Callable[[_ResolveRequest], _ResolveResult]] = {
    "auto": _resolve_auto_branch,
    "claude-agent-sdk": _resolve_claude_agent_sdk_branch,
    "openrouter": _resolve_openrouter_branch,
    "nous": _resolve_nous_branch,
    "openai-codex": _resolve_openai_codex_branch,
    "xai-oauth": _resolve_xai_oauth_branch,
    "custom": _resolve_custom_branch,
}


def resolve_provider_client(
    provider: str, model: str = None, async_mode: bool = False, raw_codex: bool = False,
    explicit_base_url: str = None, explicit_api_key: str = None, api_mode: str = None,
    main_runtime: Optional[Dict[str, Any]] = None, is_vision: bool = False,
    task: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Central router: return a configured client (auth, base URL, API format) for a provider + optional model.
    The client always exposes ``.chat.completions.create()``; Codex/Responses providers get an adapter.
    ``provider``: built-in name, ``custom:<name>``, "custom" (OPENAI_BASE_URL + OPENAI_API_KEY) or "auto"
    (full auto-detection chain). ``model=None`` → provider's default aux model. ``raw_codex`` → bare OpenAI
    client for ``responses.stream()`` callers. ``api_mode`` forces "codex_responses"/"chat_completions"/
    "anthropic_messages" instead of auto-detect. Returns (client, resolved_model) or (None, None)."""
    _validate_proxy_env_urls()
    # Keep the pre-alias name so a custom_providers entry named like a built-in alias
    # (e.g. "kimi" → "kimi-coding") is still reachable via the named-custom branch.
    original_provider = (provider or "").strip().lower()
    provider = _normalize_aux_provider(provider)
    # MoA chokepoint: "moa" is not an HTTP provider; resolve to the aggregator so direct callers don't
    # dead-end in unknown-provider. Unresolvable preset → leave untouched for the normal diagnostic.
    if provider == "moa":
        _agg_provider, _agg_model = _resolve_moa_aggregator(model)
        if _agg_provider and _agg_model:
            original_provider = _agg_provider.strip().lower()
            provider = _normalize_aux_provider(_agg_provider)
            model = _agg_model
            # The moa:// facade endpoint/key belong to the virtual runtime, not the aggregator.
            if explicit_base_url and str(explicit_base_url).lower().startswith("moa://"):
                explicit_base_url = None
                explicit_api_key = None
    # Model for concrete providers: caller ``model`` → catalog default (empty for OAuth-gated providers whose
    # lists drift) → configured main model (MoA → aggregator), keeping OAuth aux tasks off the Step-2 fallback.
    # Excluded: ``auto`` (a stale main slug could pair with any picked provider) and Nous + vision (the
    # Portal's tier-aware vision recommendation must win over a text-only model).
    if not model and provider != "auto" and not (provider == "nous" and is_vision):
        # ``auto`` is intentionally excluded: `_resolve_auto_route(main_runtime=...)` returns the model paired
        # with the provider it actually selected. Pre-filling an auto call from `_read_main_model()` can
        # leak a stale process-global runtime into a different provider (for example Claude model slug on
        # Codex OAuth) and override that correctly resolved model. 1. ``model`` argument (caller knew what
        # they wanted) 2. Provider's catalog default — cheap/fast model the provider registered via
        # ``ProviderProfile.default_aux_model`` or the legacy ``_API_KEY_PROVIDER_AUX_MODELS_FALLBACK``
        # dict. 3. User's main model from ``model.model`` in config.yaml. This is the load-bearing step for
        # OAuth providers: an xai-oauth user with grok-4.3 configured gets grok-4.3 for title generation
        # instead of silently dropping to whatever Step-2 fallback (#31845). When the main provider is MoA,
        # ``_read_main_model_for_aux()`` substitutes the preset's aggregator model — the preset NAME is
        # never a valid wire model id, so unset aux models default to the preset's acting model instead.
        # Each provider branch below sees a non-empty ``model`` whenever the user has *anything* configured
        # — no provider-specific empty-model guards needed. When the user has NOTHING configured (fresh
        # install, main_model also empty), the branches still hit their own missing-credentials returns and
        # ``_resolve_auto_route`` falls through to the Step-2 chain as before. Do NOT pre-fill a blank ``auto``
        # request from the config/main default here. Claude model sent to Codex after the main lane fell
        # back to gpt-5.5). Let _resolve_auto_route() return the actual current runtime model when the caller did
        # not explicitly request one. (# compression-current-model) Nous + vision is the one carve-out: the
        # branch below resolves its model from the Portal's tier-aware vision recommendation
        # (``_try_nous(vision= True)``), and ``final_model = model or default`` means anything pre-filled
        # here wins over that. The main chat model is routinely text-only (e.g. a ``:free`` chat SKU), so
        # pre-filling it sends the image to a model that cannot accept one and the Portal 404s. Leave
        # ``model`` unset and let the Portal slot through; only an explicit caller model may override it.
        model = _get_aux_model_for_provider(provider) or _read_main_model_for_aux() or model
    req = _ResolveRequest(
        provider, original_provider, model, async_mode, raw_codex,
        explicit_base_url, explicit_api_key, api_mode, main_runtime, is_vision, task,
    )
    branch = _EXPLICIT_PROVIDER_BRANCHES.get(provider)
    if branch is not None:
        return branch(req)
    # Named custom providers; an ImportError anywhere in the arm falls through to the built-ins.
    try:
        result = _resolve_named_custom_branch(req)
    except ImportError:
        result = None
    if result is not None:
        return result
    if provider == "azure-foundry":
        return _resolve_azure_foundry_branch(req)
    return _resolve_registry_branch(req)


# ── Public API ──────────────────────────────────────────────────────────────

def get_text_auxiliary_client(task: str = "", *, main_runtime: Optional[Dict[str, Any]] = None) -> Tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, default_model_slug) for text-only aux tasks; ``task`` selects auxiliary.<task> overrides."""
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(task or None)
    return resolve_provider_client(
        provider, model=model, explicit_base_url=base_url, explicit_api_key=api_key,
        api_mode=api_mode, main_runtime=main_runtime,
    )


_VISION_AUTO_PROVIDER_ORDER = ("openrouter", "nous", "deepinfra")


def _main_model_supports_vision(provider: str, model: Optional[str]) -> bool:
    """True when ``provider``/``model`` is known to accept image input; unknown capability → True (attempt the call)."""
    try:
        from agent.image_routing import _lookup_supports_vision
        from hermes_cli.config import load_config_readonly
    except ImportError:
        return True
    try:
        supports = _lookup_supports_vision(provider, model, load_config_readonly())
    except Exception:  # pragma: no cover - defensive
        return True
    return True if supports is None else bool(supports)


def _normalize_vision_provider(provider: Optional[str]) -> str:
    return _normalize_aux_provider(provider)


def _deepinfra_strict_vision_backend(model: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
    """DeepInfra vision: default model is discovered live via default_vision_model() so no hardcoded id can rot."""
    vision_model = model or _resolve_provider_vision_default("deepinfra")
    if not vision_model:
        logger.debug("Vision auto-detect: deepinfra catalog unreachable or returned no vision-tagged models — skipping")
        return None, None
    return resolve_provider_client("deepinfra", vision_model, is_vision=True)


# Strict (explicitly requested) vision backends by normalized provider name. nous MUST go
# through resolve_provider_client so anthropic/* picks wrap onto /v1/messages (a bare _try_nous
# client 404s). openai-codex has no safe default model; callers set auxiliary.<task>.model.
_STRICT_VISION_BACKENDS: Dict[str, Callable[[Optional[str]], Tuple[Optional[Any], Optional[str]]]] = {
    "copilot": lambda model: resolve_provider_client("copilot", model, is_vision=True),
    "openrouter": lambda model: _try_openrouter(model=model),
    "nous": lambda model: resolve_provider_client("nous", model, is_vision=True),
    "openai-codex": lambda model: resolve_provider_client("openai-codex", model, is_vision=True),
    "anthropic": lambda model: _try_anthropic(),
    "deepinfra": _deepinfra_strict_vision_backend,
    "custom": lambda model: _try_custom_endpoint(),
}


def _resolve_strict_vision_backend(provider: str, model: Optional[str] = None) -> Tuple[Optional[Any], Optional[str]]:
    backend = _STRICT_VISION_BACKENDS.get(_normalize_vision_provider(provider))
    return backend(model) if backend is not None else (None, None)


def get_available_vision_backends() -> List[str]:
    """Available vision backends in auto-selection order (active provider → OpenRouter → Nous → DeepInfra).

    Single source of truth for setup, tool gating, and runtime auto-routing.
    """
    available: List[str] = []
    main_provider = _read_main_provider()
    if main_provider and main_provider not in {"auto", ""}:
        if main_provider in _VISION_AUTO_PROVIDER_ORDER:
            main_ok = _resolve_strict_vision_backend(main_provider)[0] is not None
        else:
            main_ok = resolve_provider_client(main_provider, _read_main_model())[0] is not None
        if main_ok:
            available.append(main_provider)
    for p in _VISION_AUTO_PROVIDER_ORDER:  # skip if already covered by main provider
        if p not in available and _resolve_strict_vision_backend(p)[0] is not None:
            available.append(p)
    return available


def _finalize_vision_client(
    resolved_provider: str, sync_client: Any, default_model: Optional[str],
    resolved_model: Optional[str], async_mode: bool,
) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """Apply the explicit model override (and async wrapping) to a resolved vision client."""
    if sync_client is None:
        return resolved_provider, None, None
    final_model = resolved_model or default_model
    if async_mode:
        async_client, async_model = _to_async_client(sync_client, final_model, is_vision=True)
        return resolved_provider, async_client, async_model
    return resolved_provider, sync_client, final_model


def _vision_main_provider_client(
    main_provider: str, main_model: str, runtime: Dict[str, Any], resolved_model: Optional[str],
    resolved_api_mode: Optional[str],
) -> Tuple[Optional[Any], Optional[str]]:
    """Auto-detect step 1: try the main provider; (None, None) falls through to the aggregator chain."""
    # A provider vision default (static override or catalog discovery) is a *known* multimodal
    # model; the pinned chat model usually isn't, so only fall back to it when no default exists.
    provider_vision_default = _resolve_provider_vision_default(main_provider)
    vision_model = provider_vision_default or main_model
    if main_provider == "nous":
        # Nous picks its vision model from Portal tier-aware slots inside _try_nous(vision=True);
        # passing the chat model would override that and 404. Only auxiliary.vision.model may.
        sync_client, default_model = _resolve_strict_vision_backend(main_provider, resolved_model or provider_vision_default)
        if sync_client is None:
            return None, None
        logger.info("Vision auto-detect: using main provider %s (%s)", main_provider, default_model or resolved_model or main_model)
        return sync_client, default_model
    if main_provider in _PROVIDERS_WITHOUT_VISION:  # endpoint rejects image input entirely
        logger.debug("Vision auto-detect: skipping main provider %s (no vision support) — falling through to aggregator chain", main_provider)
        return None, None
    if not _main_model_supports_vision(main_provider, vision_model):
        # Known text-only model. Log only the provider name (CodeQL clear-text-logging FPs).
        logger.debug(
            "Vision auto-detect: skipping main provider %s (reports no vision capability) — falling through to aggregator chain",
            main_provider,
        )
        return None, None
    # Custom endpoints carry no built-in base_url/api_key: recover the live main endpoint from
    # set_runtime_main() or, with no live runtime recorded, the configured custom endpoint.
    rpc_base_url = rpc_api_key = None
    rpc_api_mode = resolved_api_mode
    if main_provider == "custom" or main_provider.startswith("custom:"):
        if runtime.get("base_url"):
            custom_base, custom_key, custom_mode = runtime.get("base_url"), runtime.get("api_key") or None, runtime.get("api_mode")
        else:
            custom_base, custom_key, custom_mode = _resolve_custom_runtime()
        if custom_base:
            rpc_base_url, rpc_api_key = custom_base, custom_key
            rpc_api_mode = resolved_api_mode or custom_mode or None
    rpc_client, rpc_model = resolve_provider_client(
        main_provider, vision_model, api_mode=rpc_api_mode, explicit_base_url=rpc_base_url,
        explicit_api_key=rpc_api_key, main_runtime=runtime, is_vision=True)
    if rpc_client is None:
        return None, None
    logger.info("Vision auto-detect: using main provider %s (%s)", main_provider, rpc_model or vision_model)
    return rpc_client, rpc_model or vision_model


def _vision_auto_route(
    runtime: Dict[str, Any], resolved_model: Optional[str], resolved_api_mode: Optional[str],
    async_mode: bool,
) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """Auto-detect order: 1. main provider + model, 2. OpenRouter, 3. Nous Portal, 4. DeepInfra, 5. stop."""
    main_provider = str(runtime.get("provider") or _read_main_provider())
    main_model = str(runtime.get("model") or _read_main_model())
    if main_provider.strip().lower() == "moa":
        # MoA main_model is a preset NAME, not a wire model — unwrap to the preset's aggregator
        # slot. The moa:// facade endpoint belongs to the virtual provider, not the real one.
        _agg_provider, _agg_model = _resolve_moa_aggregator(main_model)
        if _agg_provider and _agg_model:
            main_provider, main_model = _agg_provider, _agg_model
            runtime = dict(runtime, base_url="", api_key="", api_mode="")
    if main_provider and main_provider not in {"auto", "", "moa"}:
        client, default_model = _vision_main_provider_client(main_provider, main_model, runtime, resolved_model, resolved_api_mode)
        if client is not None:
            return _finalize_vision_client(main_provider, client, default_model, resolved_model, async_mode)
    # Aggregators use their dedicated vision model, not the user's main model.
    for candidate in _VISION_AUTO_PROVIDER_ORDER:
        if candidate == main_provider:
            continue  # already tried above
        sync_client, default_model = _resolve_strict_vision_backend(candidate)
        if sync_client is not None:
            return _finalize_vision_client(candidate, sync_client, default_model, resolved_model, async_mode)
    logger.debug("Auxiliary vision client: none available")
    return None, None, None


# ZAI vision must use the OpenAI-compatible endpoint: the Anthropic wire rejects max_tokens on
# multimodal calls (error 1210).
_ZAI_OPENAI_VISION_URLS = ("https://open.bigmodel.cn/api/paas/v4", "https://api.z.ai/api/paas/v4")


def resolve_vision_provider_client(
    provider: Optional[str] = None, model: Optional[str] = None, *, base_url: Optional[str] = None,
    api_key: Optional[str] = None, async_mode: bool = False,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """Resolve the client actually used for vision tasks.

    Direct endpoint overrides beat provider selection; explicit providers may force
    experimental backends; auto mode only tries backends known to work.
    """
    runtime = _normalize_main_runtime(main_runtime)
    requested, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        "vision", provider, model, base_url, api_key
    )
    requested = _normalize_vision_provider(requested)
    if resolved_base_url:
        provider_for_base_override = requested if requested and requested not in {"", "auto"} else "custom"
        client, final_model = resolve_provider_client(
            provider_for_base_override, model=resolved_model, async_mode=async_mode,
            explicit_base_url=resolved_base_url, explicit_api_key=resolved_api_key,
            api_mode=resolved_api_mode, main_runtime=runtime,
        )
        return provider_for_base_override, client, (final_model if client is not None else None)
    if requested == "auto":
        return _vision_auto_route(runtime, resolved_model, resolved_api_mode, async_mode)
    if requested in _VISION_AUTO_PROVIDER_ORDER:
        sync_client, default_model = _resolve_strict_vision_backend(requested, resolved_model)
        return _finalize_vision_client(requested, sync_client, default_model, resolved_model, async_mode)
    if requested == "zai":
        for _zai_url in _ZAI_OPENAI_VISION_URLS:
            client, final_model = _get_cached_client(
                requested, resolved_model, async_mode, base_url=_zai_url,
                api_key=resolved_api_key or None, api_mode="chat_completions", main_runtime=runtime,
                is_vision=True,
            )
            if client is not None:
                return _finalize_vision_client(requested, client, final_model, resolved_model, async_mode)
        # Fallback: try without explicit base_url (old behavior)
    client, final_model = _get_cached_client(
        requested, resolved_model, async_mode, api_mode=resolved_api_mode, main_runtime=runtime, is_vision=True,
    )
    return requested, client, (final_model if client is not None else None)


def get_auxiliary_extra_body() -> dict:
    """Return extra_body kwargs (Nous Portal product tags when Nous-backed, else {})."""
    return _nous_extra_body() if auxiliary_is_nous else {}


def auxiliary_max_tokens_param(value: int, *, model: Optional[str] = None) -> dict:
    """Max-tokens kwarg for the auxiliary provider: direct OpenAI/Copilot and newer OpenAI-family
    models (by ``model`` name, so custom endpoints fronting gpt-5.x are caught) need max_completion_tokens."""
    _custom_host = base_url_hostname(_current_custom_base_url()) or ""
    direct_openai_family = (
        not _scoped_key_env("OPENROUTER_API_KEY") and _read_nous_auth() is None
        and (_custom_host in ("api.openai.com", "api.githubcopilot.com") or _custom_host.endswith(".githubcopilot.com"))
    )
    if direct_openai_family or model_forces_max_completion_tokens(model):
        return {"max_completion_tokens": value}
    return {"max_tokens": value}


# ── Centralized LLM Call API: call_llm()/async_call_llm() own resolve → cached client → shape
# request → call → return. Every auxiliary LLM consumer should use these.

# Client cache: (provider, async_mode, base_url, api_key, api_mode, runtime_key) -> (client, default_model, loop)
# Loop identity is NOT part of the key: stale-loop entries are replaced in place on async hits,
# bounding growth to one entry per provider config (avoids fd accumulation in gateways).
# This bounds cache growth to one entry per unique provider config rather than one per (config ×
# event-loop), which previously caused unbounded fd accumulation in long-running gateway processes (#10200).
_client_cache: Dict[tuple, tuple] = {}
_client_cache_lock = threading.Lock()
_CLIENT_CACHE_MAX_SIZE = 64  # safety belt — evict oldest when exceeded


class _CallableCacheDiscriminator:
    """Hash a credential callback by identity without exposing its state."""

    __slots__ = ("_callback",)

    def __init__(self, callback: Any) -> None:
        self._callback = callback  # retained so its id cannot be reused while cached

    def __hash__(self) -> int:
        return id(self._callback)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _CallableCacheDiscriminator) and self._callback is other._callback

    def __repr__(self) -> str:
        return "<callable-api-key>"


def _runtime_cache_discriminator(field: str, value: Any) -> Any:
    """Return a hashable, secret-safe runtime cache-key component."""
    if field == "api_key" and callable(value):
        return _CallableCacheDiscriminator(value)
    if field == "api_key" and isinstance(value, str) and value:
        return ("api-key-digest", hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest())
    return value


def _client_cache_key(
    provider: str, *, async_mode: bool, base_url: Optional[str] = None,
    api_key: Optional[str] = None, api_mode: Optional[str] = None,
    main_runtime: Optional[Dict[str, Any]] = None, is_vision: bool = False,
    task: Optional[str] = None, model: Optional[str] = None,
) -> tuple:
    runtime = _normalize_main_runtime(main_runtime)
    # `auto` resolves through the main runtime and task-specific policy, so both join the key.
    runtime_key = tuple(_runtime_cache_discriminator(f, runtime.get(f, "")) for f in _MAIN_RUNTIME_FIELDS) if provider == "auto" else ()
    task_key = (task or "", _task_prefers_fast_model(task)) if provider == "auto" else ""
    pool_hint = _pool_cache_hint(provider, main_runtime=main_runtime)
    # Model MUST be in the key: concurrent calls to the same endpoint with different models would
    # share an entry, and the second builder's _store_cached_client would close the first's client.
    model_key = model or runtime.get("model", "")
    api_key_key = _runtime_cache_discriminator("api_key", api_key or "")
    return (provider, async_mode, base_url or "", api_key_key, api_mode or "", runtime_key, is_vision, task_key, pool_hint, model_key)


def _current_event_loop() -> Any:
    """``asyncio.get_event_loop()`` or None when no loop can be obtained (async cache-key binding)."""
    try:
        import asyncio as _aio
        return _aio.get_event_loop()
    except RuntimeError:
        return None


def _store_cached_client(cache_key: tuple, client: Any, default_model: Optional[str], *, bound_loop: Any = None) -> None:
    if isinstance(client, _AuxProbeClientStub):
        return  # probe stubs must never be cached — the next hit would get a dud client
    with _client_cache_lock:
        old_entry = _client_cache.get(cache_key)
        if old_entry is not None and old_entry[0] is not client:
            _close_cached_client(old_entry[0])
        _client_cache[cache_key] = (client, default_model, bound_loop)


def _refresh_nous_auxiliary_client(
    *, cache_provider: str, model: Optional[str], async_mode: bool, base_url: Optional[str] = None,
    api_key: Optional[str] = None, api_mode: Optional[str] = None,
    main_runtime: Optional[Dict[str, Any]] = None, is_vision: bool = False,
    lookup_model: Optional[str] = None, lookup_task: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Refresh Nous runtime creds, rebuild the client, and replace the cache entry.

    ``model`` is the resolved wire model stored as the entry's usable model and returned. The
    cache KEY MUST be built from ``lookup_model``/``lookup_task`` — the model and task as passed
    to ``_get_cached_client`` when the stale client was acquired — so the fresh client overwrites
    the exact entry the stale one is served from. Keying on the resolved model or an empty task
    would leave the expired client immortal and every auxiliary call 401ing forever.

    See #56889.
    For ``provider == "auto"`` the task participates in the cache key (task-specific fallback policy), so it
    MUST be carried into the key here for the same reason as ``lookup_model``; otherwise an auto-provider
    client refreshed on a 401 lands under the ``task=""`` key while the stale entry survives under the
    task-scoped key (#58894).
    """
    runtime = _resolve_nous_runtime_api(force_refresh=True, stale_access_token=api_key)
    if runtime is None:
        return None, model
    fresh_key, fresh_base_url = runtime
    sync_client = _create_openai_client(api_key=fresh_key, base_url=fresh_base_url)
    current_loop = _current_event_loop() if async_mode else None
    if async_mode:
        client, final_model = _to_async_client(sync_client, model or "", is_vision=is_vision)
    else:
        client, final_model = sync_client, model
    cache_key = _client_cache_key(
        cache_provider, async_mode=async_mode, base_url=base_url, api_key=api_key,
        api_mode=api_mode, main_runtime=main_runtime, is_vision=is_vision, task=lookup_task,
        model=lookup_model,
    )
    _store_cached_client(cache_key, client, final_model, bound_loop=current_loop)
    return client, final_model


def neuter_async_httpx_del() -> None:
    """Monkey-patch ``AsyncHttpxClientWrapper.__del__`` to be a no-op.

    The SDK's ``__del__`` schedules ``aclose()`` on the *running* loop, but the transport is
    bound to the loop the client was created on; when that loop is dead this raises "Event loop
    is closed" into prompt_toolkit's loop. Safe because cached clients are closed explicitly and
    the OS reaps the rest. Call once at CLI startup, before any ``AsyncOpenAI`` is created.
    """
    try:
        from openai._base_client import AsyncHttpxClientWrapper
        AsyncHttpxClientWrapper.__del__ = lambda self: None  # type: ignore[assignment]
    except (ImportError, AttributeError):
        pass  # Graceful degradation if the SDK changes its internals


def _force_close_async_httpx(client: Any) -> None:
    """Mark the httpx AsyncClient inside an AsyncOpenAI client as closed so ``__del__`` won't
    schedule ``aclose()`` on a dead loop. Skips the full async close — the OS drops connections."""
    with contextlib.suppress(Exception):
        from httpx._client import ClientState
        inner = getattr(client, "_client", None)
        if inner is not None and not getattr(inner, "is_closed", True):
            inner._state = ClientState.CLOSED


def _schedule_async_close(close_result: Any, client: Any) -> None:
    """Finish an async close without leaking an unawaited coroutine."""
    async def _await_close() -> None:
        try:
            await close_result
        except Exception:
            pass
        finally:
            _force_close_async_httpx(client)
    runner = _await_close()
    try:
        import asyncio as _aio
        try:
            loop = _aio.get_running_loop()
        except RuntimeError:
            _aio.run(runner)
        else:
            task = loop.create_task(runner)

            def _consume(completed_task) -> None:
                with contextlib.suppress(BaseException):
                    completed_task.exception()
            task.add_done_callback(_consume)
            runner = None
    except Exception:
        if runner is not None:
            with contextlib.suppress(Exception):
                runner.close()
        _force_close_async_httpx(client)


def _close_cached_client(client: Any, *, close_async: bool = False) -> None:
    """Close one cached client, awaiting async transports only when safe."""
    if client is None:
        return
    close_fn = getattr(client, "close", None)
    if not callable(close_fn):
        _force_close_async_httpx(client)
        return
    try:
        close_result = close_fn()
    except Exception:
        _force_close_async_httpx(client)
        return
    if inspect.isawaitable(close_result):
        if close_async:
            _schedule_async_close(close_result, client)
        else:
            # Never await a client owned by another live loop; close the coroutine (no
            # unawaited warning) and neuter the transport.
            with contextlib.suppress(Exception):
                close_result.close()
            _force_close_async_httpx(client)
        return
    _force_close_async_httpx(client)


def shutdown_cached_clients() -> None:
    """Close all cached clients; call at CLI shutdown *before* the loop closes.

    Snapshot+clear under the lock, close outside it: async teardown can block while an owner
    loop drains, and holding the lock would convoy every caller.
    """
    with _client_cache_lock:
        clients = [(entry[0], entry[2]) for entry in _client_cache.values() if entry[0] is not None]
        _client_cache.clear()
    try:
        import asyncio as _aio
        running_loop = _aio.get_running_loop()
    except RuntimeError:
        running_loop = None
    for client, owner_loop in clients:
        # A live foreign loop owns its transport — neuter only and let it finish teardown.
        # Closed loops and the current loop are safe to drain here.
        close_async = owner_loop is not None and (owner_loop.is_closed() or owner_loop is running_loop)
        _close_cached_client(client, close_async=close_async)


def cleanup_stale_async_clients() -> None:
    """Force-close cached async clients whose loop is closed; call after each agent turn
    (defense-in-depth behind ``neuter_async_httpx_del``)."""
    with _client_cache_lock:
        stale = [(key, entry[0]) for key, entry in _client_cache.items() if entry[2] is not None and entry[2].is_closed()]
        for key, _client in stale:
            del _client_cache[key]
    for _key, client in stale:
        _close_cached_client(client, close_async=True)


def _compat_model(client: Any, model: Optional[str], cached_default: Optional[str]) -> Optional[str]:
    """Keep slash-bearing model IDs only for cached clients that accept ``vendor/model`` (OpenRouter
    or a slash-bearing default). Mirrors the resolve_provider_client() guard, which cache hits skip."""
    if model and "/" in model:
        accepts_slash = any(
            obj and base_url_host_matches(str(getattr(obj, "base_url", "") or ""), "openrouter.ai")
            for obj in (client, getattr(client, "_client", None), getattr(client, "client", None))
        ) or bool(cached_default and "/" in cached_default)
        if not accepts_slash:
            return cached_default
    return model or cached_default


def _get_cached_client(
    provider: str, model: str = None, async_mode: bool = False, base_url: str = None,
    api_key: str = None, api_mode: str = None, main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False, task: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Get or create a cached client for the given provider.

    Async clients bind to the loop they were created on, so every async hit validates the cached
    loop is the current, open loop; stale entries are replaced in place (bounded, no cross-loop reuse).

    This keeps cache size bounded to one entry per unique provider config, preventing the fd-exhaustion that
    previously occurred in long-running gateways where recycled worker threads created unbounded entries
    (#10200).
    """
    current_loop = _current_event_loop() if async_mode else None
    runtime = _normalize_main_runtime(main_runtime)
    cache_key = _client_cache_key(
        provider, async_mode=async_mode, base_url=base_url, api_key=api_key, api_mode=api_mode,
        main_runtime=main_runtime, is_vision=is_vision, task=task, model=model,
    )
    with _client_cache_lock:
        if cache_key in _client_cache:
            cached_client, cached_default, cached_loop = _client_cache[cache_key]
            loop_ok = not async_mode or (
                cached_loop is not None and cached_loop is current_loop and not cached_loop.is_closed()
            )
            if loop_ok:
                return cached_client, _compat_model(cached_client, model, cached_default)
            # Stale async entry — evict. Only a closed owner loop may be awaited here; a live
            # foreign loop stays force-neutered.
            _close_cached_client(cached_client, close_async=cached_loop is not None and cached_loop.is_closed())
            del _client_cache[cache_key]
    # Build outside the lock. For pool-backed providers derive the key from the pool entry:
    # resolve_api_key_provider_credentials prefers env vars, which would bypass pool rotation
    # and retry an exhausted key.
    effective_api_key = api_key
    if not effective_api_key:
        _pe = _peek_pool_entry(_normalize_aux_provider(provider))
        if _pe is not None:
            effective_api_key = _pool_runtime_api_key(_pe) or api_key
    client, default_model = resolve_provider_client(
        provider, model, async_mode, explicit_base_url=base_url, explicit_api_key=effective_api_key,
        api_mode=api_mode, main_runtime=runtime, is_vision=is_vision, task=task,
    )
    if client is not None:
        with _client_cache_lock:
            if cache_key not in _client_cache:
                # FIFO safety-belt eviction. Do NOT close evicted clients: another caller may be
                # mid-request on one; refcount/GC handles it.
                while len(_client_cache) >= _CLIENT_CACHE_MAX_SIZE:
                    del _client_cache[next(iter(_client_cache))]
                _client_cache[cache_key] = (client, default_model, current_loop)
            else:
                built_client = client
                client, default_model, _ = _client_cache[cache_key]
                # Race loser was never exposed to a caller — safe to close now.
                _close_cached_client(built_client, close_async=async_mode)
    return client, model or default_model


# Aliases for direct REST APIs not modeled in PROVIDER_REGISTRY, so ``auxiliary.<task>.provider:
# openai`` resolves to a working ``custom`` endpoint (OPENAI_API_KEY + api.openai.com) instead of
# silently falling back to the main provider and sending OpenAI model names elsewhere.
_AUX_DIRECT_API_BASE_URLS: Dict[str, str] = {"openai": "https://api.openai.com/v1"}


# MoA virtual provider: an *explicit* `provider: moa` override (either the caller-passed `provider` arg or
# `auxiliary.<task>.provider` in config.yaml) reaches this function directly — it never goes through
# _resolve_auto_route(), which only unwraps the *implicit* "main provider is moa" case (#53827). Left as-is, "moa"
# is returned verbatim and resolve_provider_client() looks it up in PROVIDER_REGISTRY (which has no "moa"
# entry — it's not a real HTTP provider), falls to the unknown-provider dead end, and call_llm surfaces a
# nonsensical "MOA_API_KEY environment variable" error for a provider that was never meant to be reached
# over the wire. Auxiliary tasks don't need the reference fan-out — resolve to the preset's aggregator slot
# instead, exactly like the implicit path does (shared helper: _resolve_moa_aggregator).
def _unwrap_moa_provider(prov: str, mdl: Optional[str]) -> Tuple[str, Optional[str]]:
    """Resolve an *explicit* ``provider: moa`` to its preset's aggregator slot (_resolve_auto_route()
    only unwraps the implicit case; "moa" isn't in PROVIDER_REGISTRY and would dead-end)."""
    if prov.strip().lower() != "moa":
        return prov, mdl
    agg_provider, agg_model = _resolve_moa_aggregator(mdl)
    if agg_provider and agg_model:
        return agg_provider, agg_model
    return prov, mdl


def _expand_direct_api_alias(prov: Optional[str], existing_base: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """``provider: openai`` → custom + api.openai.com/v1; a user base_url is kept but the provider still becomes custom."""
    if not prov:
        return prov, existing_base
    target_base = _AUX_DIRECT_API_BASE_URLS.get(prov.strip().lower())
    if target_base is None:
        return prov, existing_base
    return "custom", existing_base or target_base


def _preserve_provider_with_base_url(prov: Optional[str]) -> bool:
    """True when a first-class provider keeps its identity alongside an explicit base_url."""
    normalized = str(prov or "").strip().lower()
    if normalized in {"", "auto", "custom"} or normalized.startswith("custom:"):
        return False
    try:
        from hermes_cli.providers import get_provider
        return get_provider(normalized) is not None
    except Exception:  # keep provider-backed routes safe when the catalog can't load
        return normalized in {
            "anthropic", "copilot", "copilot-acp", "minimax-oauth", "nous", "openai-codex", "qwen-oauth", "xai-oauth",
        }


def _resolve_task_provider_model(
    task: str = None, provider: str = None, model: str = None, base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Determine (provider, model, base_url, api_key, api_mode) for a call.

    Priority: explicit args > config auxiliary.{task}.* > "auto". A bare base_url means custom,
    but a first-class provider + base_url keeps the provider identity so its auth/transport
    shaping still applies. api_mode is "chat_completions", "codex_responses", or None (auto).
    """
    cfg_provider = cfg_model = cfg_base_url = cfg_api_key = resolved_api_mode = None
    if task:
        task_config = _get_auxiliary_task_config(task)
        cfg_provider = str(task_config.get("provider", "")).strip() or None
        cfg_model = str(task_config.get("model", "")).strip() or None
        cfg_base_url = str(task_config.get("base_url", "")).strip() or None
        cfg_api_key = str(task_config.get("api_key", "")).strip() or None
        if not cfg_api_key:  # key_env → env var when api_key is not set directly
            cfg_key_env = str(task_config.get("key_env") or task_config.get("api_key_env") or "").strip()
            if cfg_key_env:
                cfg_api_key = _scoped_key_env(cfg_key_env) or None
        resolved_api_mode = str(task_config.get("api_mode", "")).strip() or None
    # 'auto' is a sentinel ("inherit / auto-detect"), not a model id — leaking it to the wire
    # yields a 200 with an error-text body that consumers accept as output. The explicit `model`
    # kwarg needs the same normalization: MoA slots forward preset `model:` fields through it.
    if model and model.lower() == "auto":
        model = None
    if cfg_model and cfg_model.lower() == "auto":
        cfg_model = None
    resolved_model = model or cfg_model
    # Any moa:// facade endpoint belongs to the facade, not the aggregator's real provider —
    # drop it (mirrors _resolve_auto_route()).
    if provider and str(provider).strip().lower() == "moa":
        provider, resolved_model = _unwrap_moa_provider(provider, resolved_model)
        if provider and provider.lower() != "moa":
            base_url = None
            api_key = None
    elif cfg_provider and str(cfg_provider).strip().lower() == "moa":
        cfg_provider, cfg_model = _unwrap_moa_provider(cfg_provider, resolved_model)
        if cfg_provider and cfg_provider.lower() != "moa":
            resolved_model = cfg_model
            cfg_base_url = None
            cfg_api_key = None
    if provider:
        provider, base_url = _expand_direct_api_alias(provider, base_url)
    if cfg_provider:
        cfg_provider, cfg_base_url = _expand_direct_api_alias(cfg_provider, cfg_base_url)
    # An explicit provider without base_url adopts the task's configured endpoint (same or
    # unnamed provider) so the early return below carries it. Explicit "auto" is excluded — it
    # must keep flowing through auto-resolution.
    # See #58515.
    if provider and provider != "auto" and not base_url and cfg_base_url and cfg_provider in (None, provider):
        base_url = cfg_base_url
        if not api_key:
            api_key = cfg_api_key
    if base_url:
        kept = provider if _preserve_provider_with_base_url(provider) else "custom"
        return kept, resolved_model, base_url, api_key, resolved_api_mode
    if provider:
        return provider, resolved_model, base_url, api_key, resolved_api_mode
    if cfg_base_url and cfg_api_key:
        return "custom", resolved_model, cfg_base_url, cfg_api_key, resolved_api_mode
    if cfg_base_url and cfg_provider and cfg_provider != "auto":
        # base_url without api_key: keep the provider so it can resolve credentials from env
        # vars instead of locking into "custom".
        return cfg_provider, resolved_model, cfg_base_url, None, resolved_api_mode
    if cfg_provider and cfg_provider != "auto":
        return cfg_provider, resolved_model, cfg_base_url, cfg_api_key, resolved_api_mode
    return "auto", resolved_model, None, None, resolved_api_mode


_DEFAULT_AUX_TIMEOUT = 30.0

# Reasoning compression models can exceed the default 120 s config timeout, falling back to the
# deterministic marker. Bounded *floor* for config-derived compression timeouts only; never
# overrides an explicit per-call timeout.
# Compression summarises large conversation histories; a reasoning auxiliary model (e.g. Codex / GPT-5.5)
# can legitimately take longer than the default ``auxiliary.compression.timeout`` (120 s), causing the
# stream to time out and the compressor to fall back to the deterministic context marker (#54915). A floor
# is harmless for fast compression models (they finish before the deadline) and is a minimum, so a higher
# config value is kept unchanged.
_COMPRESSION_TIMEOUT_FLOOR_SECONDS = 300.0


def _get_auxiliary_task_config(task: str) -> Dict[str, Any]:
    """Config dict for auxiliary.<task>, or {} when unavailable. Plugin-registered tasks get their
    declared defaults layered under user config (user wins); built-in defaults live in DEFAULT_CONFIG."""
    if not task:
        return {}
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except ImportError:
        return {}
    aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
    task_config = aux.get(task, {}) if isinstance(aux, dict) else {}
    if not isinstance(task_config, dict):
        task_config = {}
    try:
        from hermes_cli.plugins import get_plugin_auxiliary_tasks
        for _entry in get_plugin_auxiliary_tasks():
            if _entry.get("key") == task:
                _defaults = _entry.get("defaults") or {}
                if isinstance(_defaults, dict):
                    return {**_defaults, **task_config}
                break
    except Exception:
        pass  # plugin discovery failure must not break aux task config reads
    return task_config


class CompressionFastLane(NamedTuple):
    """Explicit, non-reasoning compression route safe for a bounded summary."""

    certified_non_reasoning: bool
    max_tokens: Optional[int]
    reasoning_config: Optional[Dict[str, Any]]


def _fast_lane_config_fields(config: Dict[str, Any]) -> tuple[str, str, bool, Optional[int]]:
    """``(provider, model, non_reasoning, cap)`` from one task config.

    ``non_reasoning`` only when ``reasoning_effort`` EXPLICITLY disables thinking (unset is NOT
    non-reasoning); ``cap`` is a positive int ``max_output_tokens`` or None — booleans are config
    drift, never a cap (``int(True) == 1``).
    """
    from hermes_constants import parse_reasoning_effort
    provider = str(config.get("provider") or "").strip().lower()
    model = str(config.get("model") or "").strip()
    parsed_effort = parse_reasoning_effort(config.get("reasoning_effort"))
    non_reasoning = parsed_effort is not None and parsed_effort.get("enabled") is False
    raw_cap = config.get("max_output_tokens")
    try:
        cap = 0 if isinstance(raw_cap, bool) else int(raw_cap or 0)
    except (TypeError, ValueError):
        cap = 0
    return provider, model, non_reasoning, (cap if cap > 0 else None)


def resolve_compression_fast_lane(
    actual_provider: str, actual_model: Optional[str], *, requested_provider: Optional[str] = None,
    requested_model: Optional[str] = None, route_config: Optional[Dict[str, Any]] = None,
) -> CompressionFastLane:
    """Certify the opt-in fast lane: capped only when an explicit, operator-certified
    non-reasoning provider/model exactly matches the route actually called."""
    config = route_config if route_config is not None else _get_auxiliary_task_config("compression")
    cfg_provider, cfg_model, non_reasoning, cap = _fast_lane_config_fields(config)
    provider = str(requested_provider or "").strip().lower() or cfg_provider
    model = str(requested_model or "").strip() or cfg_model
    explicit_route = provider not in {"", "auto"} and model.lower() not in {"", "auto"}
    actual_norm = _normalize_aux_provider(_fallback_provider_from_label(str(actual_provider or "")))
    provider_matches = actual_norm == _normalize_aux_provider(provider)
    model_matches = str(actual_model or "").strip().lower() == model.lower()
    if explicit_route and provider_matches and model_matches and non_reasoning:
        return CompressionFastLane(True, cap, {"enabled": False, "effort": "none"})
    return CompressionFastLane(False, None, None)


def _compression_config_claims_fast_lane(config: Dict[str, Any]) -> bool:
    """Whether task config declares fast-only controls that cannot leak."""
    provider, model, non_reasoning, cap = _fast_lane_config_fields(config)
    return provider not in {"", "auto"} and model.lower() not in {"", "auto"} and non_reasoning and cap is not None


def _compression_fast_lane_controls(
    task: str | None, *, actual_provider: str, actual_model: str | None,
    requested_provider: str | None, requested_model: str | None, route_config: Dict[str, Any],
    leak_guard_config: Dict[str, Any], max_tokens: int | None, extra_body: Dict[str, Any],
) -> tuple[int | None, Dict[str, Any]]:
    """Apply the certified compression controls to one resolved route."""
    if task != "compression" or max_tokens is not None:
        return max_tokens, extra_body
    body = dict(extra_body)
    lane = resolve_compression_fast_lane(
        actual_provider, actual_model, requested_provider=requested_provider, requested_model=requested_model, route_config=route_config,
    )
    if lane.reasoning_config is not None:
        if "reasoning" not in body:
            body["reasoning"] = lane.reasoning_config
    elif _compression_config_claims_fast_lane(leak_guard_config):
        body.pop("reasoning", None)
    return lane.max_tokens, body


def _get_task_timeout(task: str, default: float = _DEFAULT_AUX_TIMEOUT) -> float:
    """``auxiliary.<task>.timeout`` from config, else *default*."""
    if not task:
        return default
    raw = _get_auxiliary_task_config(task).get("timeout")
    if raw is not None:
        with contextlib.suppress(ValueError, TypeError):
            return float(raw)
    return default


def _effective_aux_timeout(task: str, timeout: Optional[float]) -> float:
    """Explicit ``timeout`` wins, else config; compression gets a floor so a reasoning model
    summarising a large context isn't cut off."""
    if timeout is not None:
        return timeout
    effective = _get_task_timeout(task)
    return max(effective, _COMPRESSION_TIMEOUT_FLOOR_SECONDS) if task == "compression" else effective


def _get_task_extra_body(task: str) -> Dict[str, Any]:
    """Shallow copy of ``auxiliary.<task>.extra_body`` with ``reasoning_effort`` folded into
    ``reasoning`` unless one is configured (more specific wins). MoA tasks are excluded: their
    reasoning depth is per-slot in the preset."""
    task_config = _get_auxiliary_task_config(task)
    raw = task_config.get("extra_body")
    result = dict(raw) if isinstance(raw, dict) else {}
    if "reasoning" in result:
        return result
    effort = task_config.get("reasoning_effort")
    if effort is None or effort == "":
        return result
    if task in ("moa_reference", "moa_aggregator"):
        logger.warning(
            "auxiliary.%s.reasoning_effort is not supported — MoA reasoning depth is per-slot: set reasoning_effort "
            "on the preset's reference_models entries / aggregator instead (moa.presets.<name>...). Ignoring.",
            task,
        )
        return result
    from hermes_constants import parse_reasoning_effort
    parsed = parse_reasoning_effort(effort)
    if parsed is not None:
        result["reasoning"] = parsed
    else:
        logger.warning(
            "auxiliary.%s.reasoning_effort %r is not a valid level (none, minimal, low, medium, high, xhigh, max, ultra) — ignoring",
            task, effort,
        )
    return result


# Per-task concurrency limiting: many sessions can spawn unbounded background aux calls, each
# retrying across the fallback chain during incidents.
# During provider incidents each call also retries / fans out across the fallback chain, multiplying request
# volume on already-degraded endpoints. A per-task semaphore caps in-flight calls so retry amplification
# stays bounded. See #23324.
_aux_sync_semaphores: Dict[str, Tuple[int, threading.BoundedSemaphore]] = {}
_aux_async_semaphores: Dict[Tuple[str, int], Tuple[int, Any]] = {}
_aux_sem_lock = threading.Lock()


def _get_task_max_concurrency(task: Optional[str]) -> Optional[int]:
    """``auxiliary.<task>.max_concurrency`` as a positive int, or None. Vision uses this key for
    its encode/resize CPU pool; its LLM calls stay concurrent."""
    if not task or task == "vision":
        return None
    try:
        value = int(_get_auxiliary_task_config(task).get("max_concurrency"))
    except (TypeError, ValueError):  # missing (None) or malformed
        return None
    return value if value > 0 else None


def _cached_semaphore(store: dict, key: Any, limit: int, factory: Callable[[int], Any]) -> Any:
    """Return the cached semaphore for ``key``, rebuilding it when the limit changed."""
    with _aux_sem_lock:
        entry = store.get(key)
        if entry is None or entry[0] != limit:
            store[key] = entry = (limit, factory(limit))
        return entry[1]


def _acquire_sync_aux_semaphore(task: Optional[str]) -> Optional[threading.BoundedSemaphore]:
    """Get a per-task sync semaphore, rebuilding it after a config change."""
    limit = _get_task_max_concurrency(task)
    return None if limit is None else _cached_semaphore(_aux_sync_semaphores, task, limit, threading.BoundedSemaphore)


def _acquire_async_aux_semaphore(task: Optional[str]):
    """Get a per-task, per-event-loop async semaphore after config lookup."""
    limit = _get_task_max_concurrency(task)
    if limit is None:
        return None
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    return _cached_semaphore(_aux_async_semaphores, (task, id(loop)), limit, asyncio.Semaphore)


def _reset_aux_semaphores() -> None:
    """Drop cached semaphores (test helper)."""
    with _aux_sem_lock:
        _aux_sync_semaphores.clear()
        _aux_async_semaphores.clear()


# Anthropic-compatible endpoints reached via the OpenAI SDK wrapper; their image content blocks
# must use Anthropic format.
_ANTHROPIC_COMPAT_PROVIDERS = frozenset({"minimax", "minimax-oauth", "minimax-cn"})


def _is_anthropic_compat_endpoint(provider: str, base_url: str) -> bool:
    """True for known Anthropic-compatible providers or any ``/anthropic`` URL path."""
    return provider in _ANTHROPIC_COMPAT_PROVIDERS or "/anthropic" in (base_url or "").lower()


# OpenAI block type → (Anthropic block type, default media type for data: URLs). MiniMax's
# Anthropic-compatible endpoint wants type="video" (not "video_url"/"input_video") with the same
# ``source`` shape as "image".
_ANTHROPIC_MEDIA_BLOCKS = {"image_url": ("image", "image/png"), "video_url": ("video", "video/mp4")}


def _convert_openai_images_to_anthropic(messages: list) -> list:
    """Convert OpenAI ``image_url``/``video_url`` blocks to Anthropic ``image``/``video``;
    only list-content messages with such blocks change."""
    converted = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            converted.append(msg)
            continue
        new_content = []
        changed = False
        for block in content:
            block_type = block.get("type")
            if block_type not in _ANTHROPIC_MEDIA_BLOCKS:
                new_content.append(block)
                continue
            url = (block.get(block_type) or {}).get("url", "")
            anth_type, media_type = _ANTHROPIC_MEDIA_BLOCKS[block_type]
            if url.startswith("data:"):
                header, _, b64data = url.partition(",")
                if ":" in header and ";" in header:
                    media_type = header.split(":", 1)[1].split(";", 1)[0]
                source = {"type": "base64", "media_type": media_type, "data": b64data}
            else:
                source = {"type": "url", "url": url}
            new_content.append({"type": anth_type, "source": source})
            changed = True
        converted.append({**msg, "content": new_content} if changed else msg)
    return converted


_PROFILE_REASONING_KEYS = {
    "reasoning", "reasoning_effort", "thinking", "thinking_config", "thinkingconfig",
    "thinking_budget", "thinkingbudget", "enable_thinking", "think", "verbosity",
}


def _contains_profile_reasoning_fields(value: Any) -> bool:
    """Return whether a profile payload contains a reasoning wire control (recursive)."""
    if not isinstance(value, dict):
        return False
    return any(
        str(key).strip().lower() in _PROFILE_REASONING_KEYS or _contains_profile_reasoning_fields(nested)
        for key, nested in value.items()
    )


_NOUS_PROVIDER_NAMES = frozenset({"nous", "nous-portal", "nousresearch"})


def _nous_on_messages_wire(provider_norm: str, model: str) -> bool:
    """True when a Nous Portal route serves ``model`` over /v1/messages (dual-wire catalog)."""
    if provider_norm not in _NOUS_PROVIDER_NAMES:
        return False
    from hermes_cli.providers import nous_api_mode
    return nous_api_mode(model) == "anthropic_messages"


_NVIDIA_PROVIDER_NAMES = {"nvidia", "nvidia-nim", "nim", "build-nvidia", "nemotron"}
_GEMINI_NATIVE_PROVIDER_NAMES = {"gemini", "google", "google-gemini", "google-ai-studio"}


def _is_gemini_native_route(provider_norm: str, effective_base: str) -> bool:
    """Gemini native by provider name, else (best-effort) by base URL shape."""
    if provider_norm in _GEMINI_NATIVE_PROVIDER_NAMES:
        return True
    if not effective_base:
        return False
    try:
        from agent.gemini_native_adapter import is_native_gemini_base_url
        return is_native_gemini_base_url(effective_base)
    except Exception:
        return False


def _forwards_max_tokens(provider: str, provider_norm: str, model: str, effective_base: str, task: Optional[str]) -> bool:
    """Whether an explicit max_tokens is forwarded on this route.

    No default cap elsewhere (omitted = model max; avoids max_completion_tokens / ZAI-vision
    quirks). Forward only where mandatory or honored: Anthropic Messages wire (400 without it);
    NVIDIA NIM (empty choices[] when omitted); MoA reference slots; Gemini native (fixed 65,535
    ceiling otherwise); OpenRouter (budgets the FULL window when omitted → 402 on low credit);
    managed local llama-server (uncapped decode with no EOS burns the GPU to the context window).
    """
    return (
        _is_anthropic_compat_endpoint(provider, effective_base)
        or _nous_on_messages_wire(provider_norm, model)
        or provider_norm in _NVIDIA_PROVIDER_NAMES
        or base_url_host_matches(effective_base, "integrate.api.nvidia.com")
        or str(task) == "moa_reference"
        or _is_gemini_native_route(provider_norm, effective_base)
        or provider_norm == "openrouter"
        or base_url_host_matches(effective_base, "openrouter.ai")
        or _is_managed_local_endpoint(effective_base)
    )


def _dedupe_tool_names(tools: list, provider: str, model: str) -> list:
    """Drop duplicate tool names (Vertex/Azure/Bedrock 400 on them) with a warning."""
    seen: set = set()
    deduped: list = []
    for tool in tools:
        name = (tool.get("function") or {}).get("name", "")
        if name and name in seen:
            logger.warning("_build_call_kwargs: duplicate tool name '%s' removed (provider=%s model=%s)", name, provider, model)
            continue
        if name:
            seen.add(name)
        deduped.append(tool)
    return deduped


class _ProfileProjection(NamedTuple):
    body: Dict[str, Any]
    reasoning_extra: Dict[str, Any]
    top_level: Dict[str, Any]
    handles_reasoning: bool


def _project_provider_profile(
    provider: str, provider_norm: str, model: str, effective_base: str, reasoning_config: Optional[dict],
) -> _ProfileProjection:
    """Provider profile's extra_body / kwargs projection; partial on failure."""
    body: Dict[str, Any] = {}
    reasoning_extra: Dict[str, Any] = {}
    top_level: Dict[str, Any] = {}
    handles_reasoning = False
    try:
        from providers import get_provider_profile
        from providers.base import ProviderProfile
        profile = get_provider_profile(provider_norm)
        if profile is not None:
            body = profile.build_extra_body(model=model, base_url=effective_base, reasoning_config=reasoning_config) or {}
            reasoning_extra, top_level = profile.build_api_kwargs_extras(
                reasoning_config=reasoning_config, supports_reasoning=reasoning_config is not None,
                model=model, base_url=effective_base,
            )
            reasoning_extra = reasoning_extra or {}
            top_level = top_level or {}
            handles_reasoning = (
                type(profile).build_api_kwargs_extras is not ProviderProfile.build_api_kwargs_extras
                or _contains_profile_reasoning_fields(body)
                or _contains_profile_reasoning_fields(reasoning_extra)
                or _contains_profile_reasoning_fields(top_level)
            )
    except Exception as exc:
        logger.debug("_build_call_kwargs: provider profile projection failed for %s: %s", provider, exc)
    return _ProfileProjection(body, reasoning_extra, top_level, handles_reasoning)


def _merge_aux_extra_body(
    extra_body: Optional[dict], projection: _ProfileProjection, reasoning_config: Optional[dict], provider_norm: str,
) -> Dict[str, Any]:
    """Caller extra_body + profile body/reasoning + generic reasoning fallback + Nous tags."""
    merged_extra = dict(extra_body or {})
    merged_extra.update(projection.body)
    merged_extra.update(projection.reasoning_extra)
    if reasoning_config and isinstance(reasoning_config, dict) and not projection.handles_reasoning:
        if reasoning_config.get("enabled") is False:
            merged_extra["reasoning"] = {"enabled": False}
        else:
            merged_extra["reasoning"] = {"enabled": True, "effort": reasoning_config.get("effort") or "medium"}
    # Portal tags + sticky session_id fallback when the profile didn't supply them; session_id
    # keeps aux calls on the main turn's upstream instance (cache warmth) — tags alone are not
    # enough on /v1/messages.
    if provider_norm in _NOUS_PROVIDER_NAMES:
        if "tags" not in merged_extra:
            merged_extra["tags"] = _nous_portal_tags()
        if "session_id" not in merged_extra:
            try:
                from agent.portal_tags import get_conversation_context
                sticky_key = get_conversation_context()
            except Exception:
                sticky_key = None
            if sticky_key:
                merged_extra["session_id"] = sticky_key
    return merged_extra


def _build_call_kwargs(
    provider: str, model: str, messages: list, temperature: Optional[float] = None,
    max_tokens: Optional[int] = None, tools: Optional[list] = None, timeout: float = 30.0,
    extra_body: Optional[dict] = None, reasoning_config: Optional[dict] = None,
    base_url: Optional[str] = None, task: Optional[str] = None,
) -> dict:
    """Build kwargs for .chat.completions.create() with model/provider adjustments."""
    kwargs: Dict[str, Any] = {"model": model, "messages": messages, "timeout": timeout}
    # Per-model fixed/omitted temperature, then Opus 4.7+ sampling bans: it rejects any
    # non-default temperature/top_p/top_k, so drop silently rather than 400 when the aux model flips.
    fixed_temperature = _fixed_temperature_for_model(model, base_url)
    if fixed_temperature is OMIT_TEMPERATURE:
        temperature = None  # strip — let server choose
    elif fixed_temperature is not None:
        temperature = fixed_temperature
    if temperature is not None:
        from agent.anthropic_adapter import _forbids_sampling_params
        if not _forbids_sampling_params(model):
            kwargs["temperature"] = temperature
    effective_base = base_url or (_current_custom_base_url() if provider == "custom" else "")
    provider_norm = str(provider or "").strip().lower()
    if max_tokens is not None and _forwards_max_tokens(provider, provider_norm, model, effective_base, task):
        kwargs.update(auxiliary_max_tokens_param(max_tokens, model=model))  # picks max_completion_tokens where needed
    if tools:
        kwargs["tools"] = _dedupe_tool_names(tools, provider, model)
    # Provider profiles are the source of truth for reasoning wire shapes (top-level, nested body,
    # or extra_body.reasoning); providers without a reasoning-aware profile keep the generic
    # ``extra_body.reasoning`` fallback.
    projection = _project_provider_profile(provider, provider_norm, model, effective_base, reasoning_config)
    kwargs.update(projection.top_level)
    if merged_extra := _merge_aux_extra_body(extra_body, projection, reasoning_config, provider_norm):
        kwargs["extra_body"] = merged_extra
    # Anthropic Messages adapters take reasoning via a private kwarg that plain OpenAI SDK clients
    # would reject; Portal Claude is dual-wire, so include it only when the catalog id selects
    # /v1/messages.
    if reasoning_config and isinstance(reasoning_config, dict):
        raw_base = base_url or ""
        if (
            provider_norm == "anthropic" or _nous_on_messages_wire(provider_norm, model)
            or _endpoint_speaks_anthropic_messages(raw_base) or _is_anthropic_compat_endpoint(provider_norm, raw_base)
        ):
            kwargs["_reasoning_config"] = dict(reasoning_config)
    # OpenCode relay session affinity — same key as the main turn so compression/title/vision
    # calls stay on the conversation's warm backend.
    from agent.opencode_affinity import merge_opencode_session_headers
    return merge_opencode_session_headers(kwargs, provider, base_url, _runtime_main_value("session_id") or None)


def _validate_llm_response(
    response: Any, task: Optional[str] = None, provider: Optional[str] = None, base_url: Optional[str] = None,
) -> Any:
    """Validate the .choices[0].message shape (fail fast, not a downstream AttributeError).

    Also the single aux-usage accounting chokepoint: every successful non-streaming response
    passes here exactly once; *provider*/*base_url* are optional hints.

    See #7264.
    Recording is best-effort and never affects validation. *provider*/*base_url* are optional accounting
    hints — fallback-path calls omit them and the row keeps the model (read from the response itself) with
    an empty route. See #23270.
    """
    if response is None:
        raise RuntimeError(f"Auxiliary {task or 'call'}: LLM returned None response")
    from agent.aux_accounting import record_aux_usage
    record_aux_usage(response, task, provider=provider, base_url=base_url)
    # Adapter SimpleNamespace responses are fine — they have .choices[0].message.
    try:
        choices = response.choices
        if not choices or not hasattr(choices[0], "message"):
            raise AttributeError("missing choices[0].message")
    except (AttributeError, TypeError, IndexError) as exc:
        recovered = _recover_aux_response_message(response)
        if recovered is None:
            raise RuntimeError(
                f"Auxiliary {task or 'call'}: LLM returned invalid response (type={type(response).__name__}): "
                f"{str(response)[:120]!r}. Expected object with .choices[0].message — check provider "
                f"adapter or custom endpoint compatibility."
            ) from exc
        response = recovered
    # Retain the provider-reported model for terminal relay route attribution.
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is not None:
        model = _field(response, "model")
        if isinstance(model, str) and model.strip():
            context["response_model"] = model
    _complete_relay_auxiliary_call()
    return response


def _complete_relay_auxiliary_call(*, outcome: str = "success") -> None:
    """Close one auxiliary logical call after acceptance or terminal failure."""
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return
    from agent import relay_llm
    relay_llm.complete_logical_call(
        str(context.get("request_id") or ""), outcome=outcome,
        model_name=str(context.get("model") or "unknown"),
        provider_name=str(context.get("provider") or "auxiliary"),
        response_model_name=context.get("response_model"),
    )


def _fail_relay_auxiliary_call() -> None:
    """Close a terminally failed call without replacing its original error."""
    try:
        _complete_relay_auxiliary_call(outcome="failed")
    except Exception:
        logger.warning("Relay auxiliary failure finalization failed", exc_info=True)


def _recover_aux_response_message(response: Any) -> Optional[Any]:
    """Synthesize chat-completions shape from Responses-style text (``output_text``,
    ``output`` items) that some compatible endpoints return outside ``choices``."""
    text = _extract_aux_response_text(response)
    if not text:
        return None
    choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=getattr(response, "finish_reason", None) or "stop")
    try:
        response.choices = [choice]
        return response
    except Exception:
        return SimpleNamespace(
            id=getattr(response, "id", ""), model=getattr(response, "model", ""),
            object=getattr(response, "object", "chat.completion"), choices=[choice],
            usage=getattr(response, "usage", None),
        )


def _extract_aux_response_text(response: Any) -> str:
    """Text from Responses-style ``output_text`` or ``output[].content[].text``."""
    output_text = _field(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    output = _field(response, "output")
    if not isinstance(output, list):
        return ""
    parts: List[str] = []
    for item in output:
        item_type = _field(item, "type")
        if item_type and item_type != "message":
            continue
        for part in (_field(item, "content") or []):
            if _field(part, "type") in {"output_text", "text", None}:
                text = _field(part, "text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    return "\n".join(parts).strip()


# Streamed aggregation for progress-hooked aux calls: ``timeout`` becomes an inter-chunk idle
# timeout (httpx read timeout is per read), each chunk ticks outer watchdogs; the total ceiling
# bounds trickles.
_AUX_STREAM_CEILING_FLOOR_SECONDS = 600.0
_AUX_STREAM_CEILING_MULTIPLIER = 4.0


def _aux_stream_total_ceiling(effective_timeout: Optional[float]) -> float:
    """Absolute wall-clock bound for a streamed aux call; generous by design (the idle
    timeout is the real guard — this only stops a one-token-per-idle-window trickle)."""
    try:
        timeout = float(effective_timeout) if effective_timeout is not None else 0.0
    except (TypeError, ValueError):
        timeout = 0.0
    return max(_AUX_STREAM_CEILING_FLOOR_SECONDS, _AUX_STREAM_CEILING_MULTIPLIER * timeout)


def _client_streams_internally(client: Any) -> bool:
    """Adapters that stream inside .create() tick the hook themselves (Codex, Anthropic) or
    cannot stream (Bedrock); none accept ``stream=True`` from us."""
    return isinstance(client, (CodexAuxiliaryClient, AnthropicAuxiliaryClient, BedrockAuxiliaryClient))


_MANAGED_LOCAL_STATE_TTL_S = 15.0
_managed_local_cache: "tuple[float, str]" = (0.0, "")


def _managed_local_netloc() -> str:
    """host:port of the managed local llama-server ("" when none), read with a short TTL from
    the supervisor state file provider resolution also uses (exact match)."""
    global _managed_local_cache
    now = time.monotonic()
    ts, cached = _managed_local_cache
    if now - ts < _MANAGED_LOCAL_STATE_TTL_S:
        return cached
    try:
        from hermes_cli.local_runtime.supervisor import state_path
        raw = state_path().read_text(encoding="utf-8")
        base = str((json.loads(raw) or {}).get("base_url", ""))
        netloc = urlparse(base).netloc.lower()
    except Exception:
        netloc = ""
    _managed_local_cache = (now, netloc)
    return netloc


def _is_managed_local_endpoint(base_url: Optional[str]) -> bool:
    """True when *base_url* targets the llama-server this Hermes manages."""
    if not base_url:
        return False
    managed = _managed_local_netloc()
    if not managed:
        return False
    try:
        return urlparse(str(base_url)).netloc.lower() == managed
    except Exception:
        return False


def _provider_requires_stream(provider: str, base_url: Optional[str]) -> bool:
    """Providers that only accept streaming (non-stream = 400): Tencent Copilot, any
    ``auxiliary.stream_only_base_urls`` substring, and the managed local llama-server
    (streamed for cancellation — it only notices a dead client on socket write)."""
    _url = str(base_url or "").lower()
    if not _url:
        return False
    if base_url_host_matches(_url, "copilot.tencent.com") or _is_managed_local_endpoint(_url):
        return True
    try:
        from hermes_cli.config import load_config
        markers = (load_config() or {}).get("auxiliary", {}).get("stream_only_base_urls") or []
        if isinstance(markers, (list, tuple)):
            return any(
                isinstance(marker, str) and marker.strip() and marker.strip().lower() in _url
                for marker in markers)
    except Exception:
        pass  # Config read is best-effort; never break an aux call over it.
    return False


_AFFORDABLE_TOKENS_RE = re.compile(r"can only afford\s+([0-9][0-9,]*)", re.IGNORECASE)
# Below the floor the affordable budget can't fit a useful aux output — treat as exhaustion;
# the margin keeps provider-side token-count rounding from 402-ing the retry.
_AFFORDABLE_RETRY_FLOOR_TOKENS = 512
# See #49785.
_AFFORDABLE_RETRY_MARGIN_TOKENS = 64


def _affordable_max_tokens_from_error(exc: Exception) -> Optional[int]:
    """Affordable output budget (minus margin) from an OpenRouter credit-limited 402
    ("...but can only afford 7117": credit exists, the cap was too large); ``None``
    when no count is present or the budget is too small to be useful."""
    if not _is_payment_error(exc):
        return None
    match = _AFFORDABLE_TOKENS_RE.search(str(exc))
    if not match:
        return None
    try:
        affordable = int(match.group(1).replace(",", ""))
    except (TypeError, ValueError):
        return None
    capped = affordable - _AFFORDABLE_RETRY_MARGIN_TOKENS
    return capped if capped >= _AFFORDABLE_RETRY_FLOOR_TOKENS else None


def _create_with_progress(
    client: Any, kwargs: Dict[str, Any], task: Optional[str] = None, *, force_stream: bool = False
) -> Any:
    """Credit-aware :func:`_create_with_progress_once`: a 402 naming an affordable
    budget retries ONCE with that cap (only ever lowering); anything else re-raises."""
    try:
        return _create_with_progress_once(client, kwargs, task, force_stream=force_stream)
    except Exception as exc:
        affordable = _affordable_max_tokens_from_error(exc)
        if affordable is None:
            raise
        existing_cap = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        if isinstance(existing_cap, (int, float)) and 0 < existing_cap <= affordable:
            raise  # Already within budget — the error is something else; don't spin.
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("max_tokens", None)
        retry_kwargs.pop("max_completion_tokens", None)
        retry_kwargs.update(
            auxiliary_max_tokens_param(affordable, model=str(kwargs.get("model") or "") or None))
        logger.info("Auxiliary %s: credit-limited 402 (affordable=%d tokens); "
                    "retrying once with a clamped output cap instead of failing: %s",
                    task or "call", affordable, exc)
        return _create_with_progress_once(client, retry_kwargs, task, force_stream=force_stream)


def _stream_request_plan(kwargs: Dict[str, Any]) -> "Tuple[Dict[str, Any], str, float]":
    """(stream kwargs, model name, total ceiling) for a streamed re-aggregation."""
    stream_kwargs = dict(kwargs)
    stream_kwargs["stream"] = True
    stream_kwargs["stream_options"] = {"include_usage": True}
    return (stream_kwargs, str(kwargs.get("model") or ""),
            _aux_stream_total_ceiling(kwargs.get("timeout")))


def _create_with_progress_once(
    client: Any, kwargs: Dict[str, Any], task: Optional[str] = None, *, force_stream: bool = False
) -> Any:
    """create() that streams (and re-aggregates, ticking the hook per substantive chunk) when a
    progress hook is active or the provider is stream-only; plain ``create(**kwargs)`` otherwise
    or when the adapter streams internally. Streaming rejections fall back to a plain call —
    except under ``force_stream``.

    Behavior is byte-for-byte identical to a plain ``create(**kwargs)`` when neither trigger applies (every
    existing caller/task) or when the client's wire adapter streams internally. With a hook + a
    chunk-capable client, the request is sent with ``stream=True`` and aggregated, ticking the hook only for
    substantive chunks. The configured ``timeout`` acts per stream read (idle) rather than as a total
    budget, and outer liveness watchdogs see tokens moving. ``force_stream=True`` (stream-only providers
    such as Tencent Copilot — credit @kudi88, PR #60686) takes the same streamed path even without a hook.
    Providers that reject the streamed request fall back to the plain non-streaming call — except under
    ``force_stream``, where a stream-only provider rejects the plain call by definition, so the original
    error is surfaced to the normal recovery chains instead.
    """
    _notify_aux_dispatch()
    _notify_aux_progress()  # Preserve the watchdog's historical dispatch tick.
    if (not _aux_progress_active() and not force_stream) or _client_streams_internally(client):
        response = client.chat.completions.create(**kwargs)
        if not _client_streams_internally(client):
            _notify_aux_provider_response()
        return response
    stream_kwargs, model, total_ceiling = _stream_request_plan(kwargs)
    try:
        chunks = client.chat.completions.create(**stream_kwargs)
    except Exception as exc:
        # Genuine provider failures aren't streaming's fault — surface unchanged so the
        # recovery chains see the same error as a plain call.
        if (force_stream or _is_transient_transport_error(exc) or _is_auth_error(exc)
                or _is_payment_error(exc) or _is_rate_limit_error(exc)):
            raise
        # Possibly a streaming-specific rejection: retry non-streaming once; a genuinely bad
        # request reproduces the real error for the except-chains.
        logger.debug("Auxiliary %s: streamed request failed (%s); retrying non-streaming",
                     task or "call", exc)
        _notify_aux_dispatch()
        response = client.chat.completions.create(**kwargs)
        _notify_aux_provider_response()
        return response
    # Some shims (MoA quiet mode, defensive adapters) return a complete response despite
    # stream=True; it counts as provider response + forward progress.
    if hasattr(chunks, "choices"):
        _notify_aux_provider_response()
        return chunks
    return _aggregate_chat_stream(chunks, model=model, total_ceiling=total_ceiling)


def _close_chunk_stream(chunks: Any, *, allow_aclose: bool = False) -> Any:
    """Best-effort ``close()`` (or ``aclose()``); returns a pending awaitable or None."""
    close_fn = getattr(chunks, "close", None) or (
        getattr(chunks, "aclose", None) if allow_aclose else None)
    if not callable(close_fn):
        return None
    try:
        result = close_fn()
    except Exception:
        return None
    return result if inspect.isawaitable(result) else None


def _aggregate_chat_stream(
    chunks: Any, *, model: str = "", total_ceiling: Optional[float] = None
) -> Any:
    """Consume a chunk stream into a complete response; TimeoutError (phrased "timed out" so
    ``_is_timeout_error`` matches) when *total_ceiling* elapses."""
    acc = _ChatStreamAccumulator(
        model=model, total_ceiling=total_ceiling, host_deadline=_current_aux_stream_deadline())
    try:
        for chunk in chunks:
            acc.feed(chunk)
    finally:
        _close_chunk_stream(chunks)
    return acc.finish()


# Reasoning-detail fields whose non-empty text counts as forward progress.
_REASONING_DETAIL_TEXT_FIELDS = ("summary", "thinking", "content", "text")


class _ChatStreamAccumulator:
    """Shared per-chunk accumulation so sync and async aggregation cannot drift."""

    def __init__(self, model: str = "", total_ceiling: Optional[float] = None,
                 host_deadline: Optional[float] = None):
        self._started = time.monotonic()
        self._total_ceiling = total_ceiling
        # Absolute instant the waiting host gives up; checked alongside (not instead of) the
        # ceiling, and unaffected by pre-construction dispatch/TTFT.
        # Checked as well as (not instead of) the ceiling above: the ceiling still bounds callers with no
        # host deadline, and the host deadline is absolute, so it is unaffected by however long dispatch and
        # TTFT took before this accumulator was constructed. See #99692.
        self._host_deadline = host_deadline
        self.content_parts: List[str] = []
        self.reasoning_parts: List[str] = []
        self.reasoning_details: List[Any] = []
        self.tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        self.finish_reason = self.usage = None
        self.resp_id = ""
        self.resp_model = model or ""

    def _check_deadlines(self) -> None:
        """Raise TimeoutError past the total ceiling or the host deadline."""
        now = time.monotonic()
        if self._total_ceiling is not None and (now - self._started) >= self._total_ceiling:
            raise TimeoutError(f"Auxiliary streamed call timed out after {self._total_ceiling:.0f}s "
                               "total ceiling (stream still open but over budget)")
        if self._host_deadline is not None and now >= self._host_deadline:
            raise TimeoutError("Auxiliary streamed call timed out at the host compression "
                               f"deadline after {time.monotonic() - self._started:.0f}s "
                               "(the caller already stopped waiting; streaming on would only "
                               "pin its session lease)")

    def _feed_reasoning_details(self, delta: Any) -> bool:
        """Collect ``reasoning_details`` (OpenRouter-style thinking); True only when a detail
        carries text, so structural/signed envelopes can't keep a stall alive."""
        reasoning_details = getattr(delta, "reasoning_details", None)
        if reasoning_details is None:
            model_extra = getattr(delta, "model_extra", None)
            if isinstance(model_extra, dict):
                reasoning_details = model_extra.get("reasoning_details")
        if not isinstance(reasoning_details, list):
            return False
        made_progress = False
        for detail in reasoning_details:
            self.reasoning_details.append(detail)
            if isinstance(detail, dict) and any(
                isinstance(detail.get(f), str) and detail[f] for f in _REASONING_DETAIL_TEXT_FIELDS):
                made_progress = True
        return made_progress

    def _feed_tool_calls(self, delta: Any) -> bool:
        """Merge tool-call fragments by index; True when any fragment carried data."""
        made_progress = False
        for tc in (getattr(delta, "tool_calls", None) or []):
            idx = getattr(tc, "index", 0) or 0
            acc = self.tool_calls_acc.setdefault(idx, {"id": "", "name": "", "arguments": []})
            if getattr(tc, "id", None):
                acc["id"] = tc.id
                made_progress = True
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    acc["name"] = fn.name
                    made_progress = True
                if getattr(fn, "arguments", None):
                    acc["arguments"].append(fn.arguments)
                    made_progress = True
        return made_progress

    def feed(self, chunk: Any) -> None:
        # Every frame records transport timing (TTFP); only a substantive payload ticks the
        # forward-progress hook that keeps compression alive.
        _notify_aux_timing_response()
        self._check_deadlines()
        self.resp_id = getattr(chunk, "id", None) or self.resp_id
        self.resp_model = getattr(chunk, "model", None) or self.resp_model
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage:
            self.usage = chunk_usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return
        choice = choices[0]
        self.finish_reason = getattr(choice, "finish_reason", None) or self.finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            return
        made_progress = False
        piece = getattr(delta, "content", None)
        if piece:
            self.content_parts.append(piece)
            made_progress = True
        reasoning_piece = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        if reasoning_piece and isinstance(reasoning_piece, str):
            self.reasoning_parts.append(reasoning_piece)
            made_progress = True
        # Evaluate both unconditionally: they accumulate state, not just progress.
        made_progress |= self._feed_reasoning_details(delta)
        made_progress |= self._feed_tool_calls(delta)
        if made_progress:
            _notify_aux_progress()

    def finish(self) -> Any:
        tool_calls = None
        if self.tool_calls_acc:
            tool_calls = [
                SimpleNamespace(id=acc["id"], type="function", function=SimpleNamespace(
                    name=acc["name"], arguments="".join(acc["arguments"])))
                for _idx, acc in sorted(self.tool_calls_acc.items())]
        message = SimpleNamespace(
            role="assistant", content="".join(self.content_parts), tool_calls=tool_calls,
            reasoning="".join(self.reasoning_parts) or None,
            reasoning_details=self.reasoning_details or None,
        )
        choice = SimpleNamespace(index=0, message=message, finish_reason=self.finish_reason or "stop")
        return SimpleNamespace(id=self.resp_id, model=self.resp_model, object="chat.completion",
                               choices=[choice], usage=self.usage)


async def _aggregate_chat_stream_async(
    chunks: Any, *, model: str = "", total_ceiling: Optional[float] = None
) -> Any:
    """Async mirror of :func:`_aggregate_chat_stream` (AsyncOpenAI streams need ``async for``)."""
    acc = _ChatStreamAccumulator(
        model=model, total_ceiling=total_ceiling, host_deadline=_current_aux_stream_deadline())
    try:
        async for chunk in chunks:
            acc.feed(chunk)
    finally:
        pending = _close_chunk_stream(chunks, allow_aclose=True)
        if pending is not None:
            with contextlib.suppress(Exception):
                await pending
    return acc.finish()


async def _acreate_with_stream(client: Any, kwargs: Dict[str, Any], task: Optional[str] = None) -> Any:
    """Async create() for stream-only providers: ``stream=True`` + aggregate the async chunks."""
    stream_kwargs, model, total_ceiling = _stream_request_plan(kwargs)
    chunks = await client.chat.completions.create(**stream_kwargs)
    if hasattr(chunks, "choices"):  # shims may hand back a complete response despite stream=True
        return chunks
    return await _aggregate_chat_stream_async(chunks, model=model, total_ceiling=total_ceiling)


# Shared request head + recovery ladder for call_llm / async_call_llm: the entry points differ
# only in how a request is awaited, so route resolution and the ordered recovery ladder are
# written once. The ladder is a generator yielding ``_LadderStep`` requests and receiving the
# response (or thrown exception), so rung ORDER and accept/re-raise contracts match on both wires.
_ResolvedAuxRoute = NamedTuple("_ResolvedAuxRoute", [
    ("client", Any), ("final_model", Optional[str]), ("resolved_provider", str),
    ("effective_provider", str)])


def _resolve_call_client(
    task: Optional[str], *, provider: Optional[str], model: Optional[str], base_url: Optional[str],
    api_key: Optional[str], resolved_provider: str, resolved_model: Optional[str],
    resolved_base_url: Optional[str], resolved_api_key: Optional[str],
    resolved_api_mode: Optional[str], main_runtime: Optional[Dict[str, Any]], async_mode: bool,
) -> _ResolvedAuxRoute:
    """Resolve the client for one aux call: vision chain, or cached text client with the
    explicit-provider fallback_chain / auto-chain rescue; RuntimeError when nothing is configured."""
    effective_provider = resolved_provider
    if task == "vision":
        effective_provider, client, final_model = resolve_vision_provider_client(
            provider=resolved_provider if resolved_provider != "auto" else provider,
            model=resolved_model or model, base_url=resolved_base_url or base_url,
            api_key=resolved_api_key or api_key, async_mode=async_mode, main_runtime=main_runtime,
        )
        if client is None and resolved_provider != "auto" and not resolved_base_url:
            logger.warning("Vision provider %s unavailable, falling back to auto vision backends",
                           resolved_provider)
            effective_provider, client, final_model = resolve_vision_provider_client(
                provider="auto", model=resolved_model, async_mode=async_mode,
                main_runtime=main_runtime)
        if client is not None:
            resolved_provider = effective_provider or resolved_provider
    else:
        client, final_model = _get_cached_client(
            resolved_provider, resolved_model, async_mode=async_mode, base_url=resolved_base_url,
            api_key=resolved_api_key, api_mode=resolved_api_mode, main_runtime=main_runtime,
            task=task)
        effective_provider = _effective_provider_for_client(client, resolved_provider)
        if client is None:
            # Explicit provider with no credentials: honor the task fallback_chain before
            # raising (fallback entries may use OAuth / credential-pool auth).
            _explicit = (resolved_provider or "").strip().lower()
            if _explicit and _explicit not in {"auto", "openrouter", "custom"}:
                fb_client, fb_model, fb_label = _try_configured_fallback_for_unavailable_client(
                    task, _explicit)
                if fb_client is None:
                    raise RuntimeError(
                        f"Provider '{_explicit}' is set in config.yaml but no API key was found. "
                        f"Set the {_explicit.upper()}_API_KEY environment variable, or switch to "
                        f"a different provider with `hermes model`.")
                client, final_model = fb_client, fb_model
                if async_mode:
                    client, final_model = _to_async_client(
                        fb_client, fb_model or "", is_vision=(task == "vision"))
                resolved_provider = fb_label or resolved_provider
                effective_provider = resolved_provider
            # Auto/custom with no credentials: walk the full auto chain (not just OpenRouter).
            # model=None so each provider uses its own default.
            if client is None and not resolved_base_url:
                logger.info("Auxiliary %s: provider %s unavailable, trying auto-detection chain",
                            task or "call", resolved_provider)
                client, final_model = _get_cached_client(
                    "auto", async_mode=async_mode, main_runtime=main_runtime, task=task)
                effective_provider = _effective_provider_for_client(client, "auto")
    if client is None:
        raise RuntimeError(f"No LLM provider configured for task={task} "
                           f"provider={resolved_provider}. Run: hermes setup")
    return _ResolvedAuxRoute(client, final_model, resolved_provider, effective_provider)


_PreparedAuxRequest = NamedTuple("_PreparedAuxRequest", [
    ("client", Any), ("final_model", Optional[str]), ("kwargs", Dict[str, Any]),
    ("resolved_provider", str), ("request_provider", str), ("resolved_model", Optional[str]),
    ("resolved_base_url", Optional[str]), ("resolved_api_key", Optional[str]),
    ("resolved_api_mode", Optional[str]), ("effective_timeout", float),
    ("effective_extra_body", Dict[str, Any]), ("base_info", str)])


def _prepare_aux_request(
    task: Optional[str], *, provider: Optional[str], model: Optional[str], base_url: Optional[str],
    api_key: Optional[str], main_runtime: Dict[str, Any], messages: list,
    temperature: Optional[float], max_tokens: Optional[int], tools: Optional[list],
    timeout: Optional[float], extra_body: Optional[dict], reasoning_config: Optional[dict],
    extra_headers: Optional[Dict[str, str]], api_mode: Optional[str],
    route_info: Optional[Dict[str, str]], async_mode: bool,
) -> _PreparedAuxRequest:
    """Shared head of call_llm/async_call_llm: resolve route + client, publish it, build request kwargs.
    Sync-only: compression fast lane, per-request ``extra_headers``, and ``base_info`` falling
    back to the resolved base_url when the client exposes none."""
    resolved_provider, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        task, provider, model, base_url, api_key)
    if api_mode:
        resolved_api_mode = api_mode
    effective_extra_body = _get_task_extra_body(task)
    effective_extra_body.update(extra_body or {})
    client, final_model, resolved_provider, effective_provider = _resolve_call_client(
        task, provider=provider, model=model, base_url=base_url, api_key=api_key,
        resolved_provider=resolved_provider, resolved_model=resolved_model,
        resolved_base_url=resolved_base_url, resolved_api_key=resolved_api_key,
        resolved_api_mode=resolved_api_mode, main_runtime=main_runtime, async_mode=async_mode,
    )
    if _effective_provider_for_client(client, resolved_provider) == "claude-agent-sdk":
        # An auto-resolved SDK client belongs to the user's subscription; retain that
        # concrete identity so no recovery rung can bridge it to a metered route.
        resolved_provider = "claude-agent-sdk"
    effective_timeout = _effective_aux_timeout(task, timeout)
    request_provider = effective_provider or resolved_provider
    fast_compression_cap = None
    if not async_mode:
        compression_config = _get_auxiliary_task_config("compression") if task == "compression" else {}
        fast_compression_cap, effective_extra_body = _compression_fast_lane_controls(
            task, actual_provider=request_provider, actual_model=final_model,
            requested_provider=provider, requested_model=model, route_config=compression_config,
            leak_guard_config=compression_config, max_tokens=max_tokens,
            extra_body=effective_extra_body,
        )
    _set_relay_auxiliary_route(request_provider, final_model, resolved_api_mode)
    _record_route_info(route_info, _fallback_provider_from_label(request_provider), final_model)
    if async_mode:
        base_info = str(getattr(client, "base_url", "") or "")
    else:
        base_info = str(getattr(client, "base_url", resolved_base_url) or "")
        if task:
            logger.info("Auxiliary %s: using %s (%s)%s",
                         task, request_provider or "auto", final_model or "default",
                         f" at {base_info}" if base_info and "openrouter" not in base_info else "")
    # Client's actual base_url so endpoint-specific temperature overrides work on
    # auto-detected routes (api.moonshot.ai vs api.kimi.com/coding).
    kwargs = _build_call_kwargs(
        request_provider, final_model, messages, temperature=temperature, max_tokens=max_tokens,
        tools=tools, timeout=effective_timeout, extra_body=effective_extra_body,
        reasoning_config=reasoning_config, base_url=base_info or resolved_base_url, task=task)
    if fast_compression_cap is not None and max_tokens is None:
        # The compression route is certified non-reasoning, so a bounded summary is
        # intentional; explicit caller caps pass through untouched.
        kwargs.update(auxiliary_max_tokens_param(fast_compression_cap, model=final_model))
    if extra_headers:
        kwargs["extra_headers"] = dict(extra_headers)
    # Convert image blocks for Anthropic-compatible endpoints (e.g. MiniMax)
    client_base = str(getattr(client, "base_url", "") or "")
    if _is_anthropic_compat_endpoint(request_provider, client_base):
        kwargs["messages"] = _convert_openai_images_to_anthropic(kwargs["messages"])
    return _PreparedAuxRequest(
        client, final_model, kwargs, resolved_provider, request_provider, resolved_model,
        resolved_base_url, resolved_api_key, resolved_api_mode, effective_timeout,
        effective_extra_body, base_info)


class _LadderStep(NamedTuple):
    """A provider request the ladder asks its driver to perform. kind: "call" (client, kwargs) |
    "retry_same_provider" (provider, model) | "fallback" (fb_client, fb_model, fb_label)."""
    kind: str
    args: tuple


_RERAISE_ORIGINAL = object()

# Ordered (predicate, reason) pairs for the provider-fallback rung: first match
# wins, so a payment-flavoured 429 reads as "payment error", not "rate limit".
_FALLBACK_REASONS: Tuple[Tuple[Callable[[Exception], bool], str], ...] = (
    (_is_auth_error, "auth error"), (_is_payment_error, "payment error"),
    (_is_rate_limit_error, "rate limit"), (_is_model_incompatible_error, "model incompatible with route"),
    (_is_invalid_aux_response_error, "invalid provider response"), (_is_connection_error, "connection error"),
)


def _rung(step: "_LadderStep", accept: Callable[[Exception], bool]):
    """One ladder rung: perform ``step``; yields ``(response, None)`` on success,
    ``(None, exc)`` when ``accept(exc)`` lets the next rung handle it, else re-raises."""
    try:
        result = yield step
    except Exception as exc:
        if not accept(exc):
            raise
        return None, exc
    return result, None


def _param_rung_accepts(exc: Exception) -> bool:
    """After a parameter-strip retry: fall through to the max_tokens/payment/auth
    chains with the stripped kwargs; re-raise anything those chains won't handle."""
    return (_is_payment_error(exc) or _is_connection_error(exc) or _is_auth_error(exc)
            or "max_tokens" in str(exc) or "unsupported_parameter" in str(exc))


def _credential_rung_accepts(exc: Exception) -> bool:
    return _is_auth_error(exc) or _is_payment_error(exc) or _is_rate_limit_error(exc)


# Immutable route context shared by the recovery rungs.
_LadderRoute = NamedTuple("_LadderRoute", [
    ("client", Any), ("task", Optional[str]), ("tag", str), ("async_mode", bool), ("base_info", str),
    ("resolved_provider", str), ("resolved_model", Optional[str]), ("resolved_base_url", Optional[str]),
    ("resolved_api_key", Optional[str]), ("resolved_api_mode", Optional[str]),
    ("final_model", Optional[str]), ("main_runtime", Optional[Dict[str, Any]]),
    ("route_info", Optional[Dict[str, str]]),
])


def _ladder_parameter_rungs(
    first_err: Exception, route: _LadderRoute, kwargs: Dict[str, Any], max_tokens: Optional[int],
):
    """Rungs 1-3: retry without temperature / structured-output format / max_tokens.
    Returns ``(response, None, kwargs)`` or ``(None, narrowed_err, stripped_kwargs)``."""
    client, task, tag = route.client, route.task, route.tag
    if "temperature" in kwargs and _is_unsupported_parameter_error(first_err, "temperature"):
        retry_kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
        logger.info("Auxiliary %s%s: provider rejected temperature; retrying once without it",
                    task or "call", tag)
        resp, first_err = yield from _rung(
            _LadderStep("call", (client, retry_kwargs)), _param_rung_accepts)
        if first_err is None:
            return resp, None, retry_kwargs
        kwargs = retry_kwargs
    if _is_structured_output_rejection(first_err):
        retry_kwargs = _without_structured_output_format(kwargs)
        if retry_kwargs is not None:
            logger.info("Auxiliary %s%s: provider rejected the structured-output "
                        "format field; retrying once without it (schema "
                        "enforcement degrades to prompt compliance): %s", task or "call", tag, first_err)
            resp, first_err = yield from _rung(
                _LadderStep("call", (client, retry_kwargs)), _param_rung_accepts)
            if first_err is None:
                return resp, None, retry_kwargs
            kwargs = retry_kwargs
    err_str = str(first_err)
    # ZAI vision models reject max_tokens with code 1210 and a message that never
    # mentions "max_tokens", so detect it explicitly.
    _is_zai_param_error = "1210" in err_str and "bigmodel" in str(getattr(client, "base_url", ""))
    if max_tokens is not None and (
        "max_tokens" in err_str or "unsupported_parameter" in err_str
        or _is_unsupported_parameter_error(first_err, "max_tokens") or _is_zai_param_error
    ):
        kwargs.pop("max_tokens", None)
        kwargs.pop("max_completion_tokens", None)
        resp, first_err = yield from _rung(
            _LadderStep("call", (client, kwargs)),
            lambda exc: _is_payment_error(exc) or _is_connection_error(exc) or _is_rate_limit_error(exc),
        )
        if first_err is None:
            return resp, None, kwargs
    return None, first_err, kwargs


def _refreshed_nous_step(route: _LadderRoute, kwargs: Dict[str, Any], message: str) -> Optional[_LadderStep]:
    """Rebuild the Nous client after a credential event; None when nothing refreshed."""
    refreshed_client, refreshed_model = _refresh_nous_auxiliary_client(
        cache_provider=route.resolved_provider or "nous", model=route.final_model,
        lookup_model=route.resolved_model, lookup_task=route.task, async_mode=route.async_mode,
        base_url=route.resolved_base_url, api_key=route.resolved_api_key,
        api_mode=route.resolved_api_mode, main_runtime=route.main_runtime,
        is_vision=(route.task == "vision"),
    )
    if refreshed_client is None:
        return None
    logger.info(message, route.task or "call", route.tag)
    if refreshed_model and refreshed_model != kwargs.get("model"):
        kwargs["model"] = refreshed_model
    return _LadderStep("call", (refreshed_client, kwargs))


def _ladder_nous_rungs(
    first_err: Exception, route: _LadderRoute, kwargs: Dict[str, Any], client_is_nous: bool,
):
    """Nous-only rungs: stale-model self-heal, paid-account refresh, 401 refresh.
    Returns ``(response, None)`` or ``(None, first_err)`` to fall through."""
    client, task, tag = route.client, route.task, route.tag
    # A long-lived process can pin a Portal model since dropped from the catalog (every call
    # 404s); force a fresh Portal fetch and retry once.
    if _is_model_not_found_error(first_err) and client_is_nous:
        healed_model = _refresh_nous_recommended_model(
            vision=(task == "vision"), stale_model=kwargs.get("model"))
        if healed_model and healed_model != kwargs.get("model"):
            logger.warning("Auxiliary %s%s: model %r no longer in Nous catalog; "
                           "retrying with refreshed recommendation %r",
                           task or "call", tag, kwargs.get("model"), healed_model)
            kwargs["model"] = healed_model
            resp, first_err = yield from _rung(_LadderStep("call", (client, kwargs)), lambda exc: True)
            if first_err is None:
                return resp, None
    # Auth refresh parity with the main agent.
    if _is_payment_error(first_err) and client_is_nous and _nous_portal_account_has_fresh_paid_access():
        step = _refreshed_nous_step(
            route, kwargs,
            "Auxiliary %s%s: refreshed Nous runtime credentials after paid account check, retrying")
        if step is not None:
            resp, first_err = yield from _rung(
                step, lambda exc: _credential_rung_accepts(exc) or _is_connection_error(exc))
            if first_err is None:
                return resp, None
    if _is_auth_error(first_err) and client_is_nous:
        step = _refreshed_nous_step(
            route, kwargs, "Auxiliary %s%s: refreshed Nous runtime credentials after 401, retrying")
        if step is not None:
            return (yield step), None
    return None, first_err


def _ladder_credential_rungs(
    first_err: Exception, route: _LadderRoute, kwargs: Dict[str, Any], client_is_nous: bool,
):
    """OAuth credential refresh + same-provider retry, then credential-pool rotation.
    Returns ``(response, None)`` or ``(None, first_err)`` to fall through."""
    client, task, tag, resolved_provider = route.client, route.task, route.tag, route.resolved_provider
    auth_refresh_provider = _auth_refresh_provider_for_route(resolved_provider, route.base_info)
    if (_is_auth_error(first_err) and auth_refresh_provider not in {"auto", "", None}
            and not client_is_nous):
        if _refresh_provider_credentials(auth_refresh_provider):
            if auth_refresh_provider != _normalize_aux_provider(resolved_provider):
                # The stale client is cached under the route label (e.g. "auto"), not the
                # concrete backend we refreshed.
                _evict_cached_clients(resolved_provider)
            logger.info("Auxiliary %s%s: refreshed %s credentials after auth error, retrying",
                        task or "call", tag, auth_refresh_provider)
            return (yield _LadderStep(
                "retry_same_provider",
                (auth_refresh_provider, route.resolved_model or route.final_model))), None
    pool_provider = _recoverable_pool_provider(resolved_provider, client, main_runtime=route.main_runtime)
    # Capture the exact key used so recovery finds the right pool entry even if another
    # process rotated the pool meanwhile (current() would be None).
    _client_api_key = str(getattr(client, "api_key", "") or "")
    if pool_provider and _credential_rung_accepts(first_err):
        recovery_err = first_err
        # Skip the extra retry for clear payment/quota errors — the endpoint won't accept
        # another request with the same exhausted key.
        if _is_rate_limit_error(first_err) and not _is_payment_error(first_err):
            resp, recovery_err = yield from _rung(
                _LadderStep("call", (client, kwargs)), _credential_rung_accepts)
            if recovery_err is None:
                return resp, None
        if _recover_provider_pool(pool_provider, recovery_err, failed_api_key=_client_api_key):
            logger.info("Auxiliary %s%s: recovered %s via credential-pool rotation after %s",
                        task or "call", tag, pool_provider, type(recovery_err).__name__)
            try:
                return (yield _LadderStep(
                    "retry_same_provider", (resolved_provider, route.resolved_model))), None
            except Exception as retry2_err:
                # Rotated key also hit a wall: mark it now so concurrent processes skip it,
                # then fall through to the provider fallback.
                if (_is_payment_error(retry2_err) or _is_auth_error(retry2_err)
                        or _is_rate_limit_error(retry2_err)):
                    _recover_provider_pool(pool_provider, retry2_err)
                    first_err = retry2_err
                else:
                    raise
    return None, first_err


def _ladder_provider_fallback(first_err: Exception, route: _LadderRoute):
    """Last rung: other providers (per-task chain; then auto: main fallback chain + discovery
    chain, explicit: main-agent-model net). Returns the response or None.
    Capacity errors (payment/quota, connection, exhausted 429, model incompatible, malformed
    response) bypass the explicit-provider gate — the provider cannot serve this request
    regardless of user intent. Auth errors only fall back in auto mode."""
    task, tag, resolved_provider = route.task, route.tag, route.resolved_provider
    # Respect explicit provider choice for transient errors (auth, request validation, etc.) but allow
    # fallback when the provider clearly cannot serve the request due to capacity: payment/quota exhaustion
    # and connection failures are capacity problems, not request constraints. See #26803: daily token quota
    # (429 + "too many tokens per day") must fall back just like a 402 credit error.
    # Rate limits are included: after retries are exhausted, a 429 means the provider is at capacity. See
    # #52228. See #26803: daily token quota must fall back like a 402 credit error.
    is_auto = resolved_provider in {"auto", "", None}
    reason = next((label for predicate, label in _FALLBACK_REASONS if predicate(first_err)), None)
    is_capacity_error = any(
        predicate(first_err) for predicate, label in _FALLBACK_REASONS if label != "auth error")
    if reason is None or not (is_auto or is_capacity_error):
        return None
    if reason == "payment error":
        # Mark the concrete backend (not the "auto" label) unhealthy so later aux calls skip
        # it instead of paying another doomed RTT.
        _mark_provider_unhealthy(
            _recoverable_pool_provider(resolved_provider, route.client, main_runtime=route.main_runtime)
            or resolved_provider)
    logger.info("Auxiliary %s%s: %s on %s (%s), trying fallback",
                task or "call", tag, reason, resolved_provider, first_err)
    # Skip only the failed model for model-specific failures; 401/402 are provider-wide, so
    # keep skipping the whole provider.
    _chain_failed_model = None if reason in ("auth error", "payment error") else route.final_model
    fb_client, fb_model, fb_label = _try_configured_fallback_chain(
        task, resolved_provider or "auto", reason=reason, failed_model=_chain_failed_model)
    if fb_client is None and is_auto:
        fb_client, fb_model, fb_label = _try_main_fallback_chain(
            task, resolved_provider or "auto", reason=reason)
        if fb_client is None:
            fb_client, fb_model, fb_label = _try_payment_fallback(
                resolved_provider, task, reason=reason)
    elif fb_client is None:
        fb_client, fb_model, fb_label = _try_main_agent_model_fallback(
            resolved_provider, task, reason=reason, failed_model=_chain_failed_model)
    if fb_client is not None:
        # Second pass: the candidate credential was stale and quarantined — walk the discovery
        # chain once more (unhealthy entries are skipped).
        for _pass in range(2):
            _record_route_info(route.route_info, _fallback_provider_from_label(fb_label), fb_model)
            fb_resp = yield _LadderStep("fallback", (fb_client, fb_model, fb_label))
            if fb_resp is not None:
                return fb_resp
            if _pass == 0:
                fb_client, fb_model, fb_label = _try_payment_fallback(
                    resolved_provider, task, reason="stale fallback credential")
                if fb_client is None:
                    break
    # All fallback layers exhausted — one user-visible warning, then re-raise.
    logger.warning("Auxiliary %s%s: %s on %s and all fallbacks exhausted "
                   # All fallback layers exhausted — emit a single user-visible warning so the operator
                   # knows aux task is about to fail. (#26882) The error itself is re-raised below.
                   # (#26882)
                   "(fallback_chain + main agent model). Raising original error.",
                   task or "call", tag, reason, resolved_provider)
    return None


def _aux_recovery_ladder(
    first_err: Exception, *, client: Any, kwargs: Dict[str, Any], task: Optional[str],
    async_mode: bool, base_info: str, resolved_provider: str, resolved_model: Optional[str],
    resolved_base_url: Optional[str], resolved_api_key: Optional[str],
    resolved_api_mode: Optional[str], final_model: Optional[str], max_tokens: Optional[int],
    main_runtime: Optional[Dict[str, Any]], route_info: Optional[Dict[str, str]],
):
    """Ordered recovery rungs after the primary request failed (generator): parameter
    strips → Nous heal/refresh → credential refresh/pool rotation → provider fallback.
    Each rung returns a response, narrows ``first_err`` and falls through, or re-raises.
    Returns ``_RERAISE_ORIGINAL`` when exhausted (after evicting a connection-poisoned client)."""
    tag = " (async)" if async_mode else ""
    route = _LadderRoute(
        client, task, tag, async_mode, base_info, resolved_provider, resolved_model,
        resolved_base_url, resolved_api_key, resolved_api_mode, final_model, main_runtime, route_info)
    resp, first_err, kwargs = yield from _ladder_parameter_rungs(first_err, route, kwargs, max_tokens)
    if first_err is None:
        return resp
    client_is_nous = (resolved_provider == "nous"
                      or base_url_host_matches(base_info, "inference-api.nousresearch.com"))
    resp, first_err = yield from _ladder_nous_rungs(first_err, route, kwargs, client_is_nous)
    if first_err is None:
        return resp
    resp, first_err = yield from _ladder_credential_rungs(first_err, route, kwargs, client_is_nous)
    if first_err is None:
        return resp
    resp = yield from _ladder_provider_fallback(first_err, route)
    if resp is not None:
        return resp
    # Connection/timeout errors poison the cached client (closed transport, half-read
    # stream); evict so the next aux call rebuilds a fresh one.
    # Drop it from the cache regardless of whether we found a fallback above so the next auxiliary call
    # rebuilds a fresh client instead of reusing the dead one. See issue #23432.
    # Mirror the sync path: drop poisoned clients on connection/timeout so the next aux call rebuilds. See
    # issue #23432.
    if _is_connection_error(first_err):
        try:
            _evict_cached_client_instance(client)
        except Exception:
            logger.debug("Auxiliary%s: cache eviction after connection error failed",
                         tag, exc_info=True)
    return _RERAISE_ORIGINAL


def _drive_ladder(ladder, perform: Callable[[_LadderStep], Any]) -> Any:
    """Run a ladder generator, feeding each step's result (or exception) back in."""
    try:
        step = next(ladder)
        while True:
            try:
                result = perform(step)
            except Exception as exc:
                step = ladder.throw(exc)
            else:
                step = ladder.send(result)
    except StopIteration as stop:
        return stop.value


async def _drive_ladder_async(ladder, perform: Callable[[_LadderStep], Any]) -> Any:
    """Async twin of :func:`_drive_ladder` (``perform`` is awaited)."""
    try:
        step = next(ladder)
        while True:
            try:
                result = await perform(step)
            except Exception as exc:
                step = ladder.throw(exc)
            else:
                step = ladder.send(result)
    except StopIteration as stop:
        return stop.value


def _elapsed_ms(started_at: float, now: Optional[float] = None) -> int:
    """Whole milliseconds since ``started_at`` (clamped at 0)."""
    return max(0, int(((time.monotonic() if now is None else now) - started_at) * 1000))


def _stamp_latency_once(latency_info: Optional[Dict[str, int]], key: str, started_at: float) -> None:
    """Record ``key`` in ``latency_info`` the first time it fires."""
    if latency_info is not None and key not in latency_info:
        latency_info[key] = _elapsed_ms(started_at)


@_relay_auxiliary_call
def call_llm(
    task: str = None, *, provider: str = None, model: str = None, base_url: str = None,
    api_key: str = None, main_runtime: Optional[Dict[str, Any]] = None, messages: list,
    temperature: Optional[float] = None, max_tokens: int = None, tools: list = None,
    timeout: float = None, extra_body: dict = None, reasoning_config: Optional[dict] = None,
    extra_headers: Optional[Dict[str, str]] = None, api_mode: str = None, stream: bool = False,
    stream_options: dict = None, route_info: Optional[Dict[str, str]] = None,
    latency_info: Optional[Dict[str, int]] = None,
) -> Any:
    """Run an auxiliary LLM request, applying the configured task limit."""
    queue_started_at = time.monotonic()
    semaphore = _acquire_sync_aux_semaphore(task)
    if semaphore is not None:
        semaphore.acquire()
    request_started_at = time.monotonic()
    if latency_info is not None:
        latency_info["queue_wait_ms"] = _elapsed_ms(queue_started_at, request_started_at)
    prior_progress_hook = getattr(_aux_progress, "hook", None)
    try:
        with (
            aux_progress_hook(
                prior_progress_hook
                if callable(prior_progress_hook)
                else ((lambda: None) if latency_info is not None else None)
            ),
            _aux_thread_local_hook(_aux_dispatch, functools.partial(
                _stamp_latency_once, latency_info, "provider_dispatch_ms", request_started_at)),
            _aux_thread_local_hook(_aux_provider_response, functools.partial(
                _stamp_latency_once, latency_info, "time_to_first_progress_ms", request_started_at)),
        ):
            response = _call_llm_impl(
                task=task, provider=provider, model=model, base_url=base_url, api_key=api_key,
                main_runtime=main_runtime, messages=messages, temperature=temperature,
                max_tokens=max_tokens, tools=tools, timeout=timeout, extra_body=extra_body,
                reasoning_config=reasoning_config, extra_headers=extra_headers, api_mode=api_mode,
                stream=stream, stream_options=stream_options, route_info=route_info,
            )
        if stream and semaphore is not None:
            stream_semaphore = semaphore
            semaphore = None
            return _release_sync_semaphore_after_stream(response, stream_semaphore)
        return response
    finally:
        if latency_info is not None:
            latency_info["summary_generation_ms"] = _elapsed_ms(request_started_at)
        if semaphore is not None:
            semaphore.release()


def _release_sync_semaphore_after_stream(stream: Any, semaphore: threading.BoundedSemaphore):
    """Release a permit only after a streaming response is consumed or closed."""
    try:
        yield from stream
    finally:
        try:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        finally:
            semaphore.release()


def _plan_aux_call(
    task: Optional[str], *, async_mode: bool, provider: Optional[str], model: Optional[str],
    base_url: Optional[str], api_key: Optional[str], main_runtime: Optional[Dict[str, Any]],
    messages: list, temperature: Optional[float], max_tokens: Optional[int], tools: Optional[list],
    timeout: Optional[float], extra_body: Optional[dict], reasoning_config: Optional[dict],
    extra_headers: Optional[Dict[str, str]], api_mode: Optional[str],
    route_info: Optional[Dict[str, str]],
) -> Tuple[_PreparedAuxRequest, Dict[str, Any], Dict[str, Any]]:
    """Shared head of both call impls: prepare the request and bundle the kwargs the recovery
    drivers pass to ``_retry_same_provider_*`` / ``_call_fallback_candidate_*``. One immutable
    runtime snapshot for keying/resolution/retries/fallbacks, so a concurrent /model switch
    can't mix key and client from different runtimes."""
    main_runtime = _normalize_main_runtime(main_runtime)
    req = _prepare_aux_request(
        task, provider=provider, model=model, base_url=base_url, api_key=api_key,
        main_runtime=main_runtime, messages=messages, temperature=temperature,
        max_tokens=max_tokens, tools=tools, timeout=timeout, extra_body=extra_body,
        reasoning_config=reasoning_config, extra_headers=extra_headers,
        api_mode=api_mode, route_info=route_info, async_mode=async_mode,
    )
    candidate_kwargs = dict(
        task=task, messages=messages, temperature=temperature, max_tokens=max_tokens,
        tools=tools, effective_timeout=req.effective_timeout,
        effective_extra_body=req.effective_extra_body, reasoning_config=reasoning_config,
    )
    retry_kwargs = dict(
        candidate_kwargs, resolved_base_url=req.resolved_base_url,
        resolved_api_key=req.resolved_api_key, resolved_api_mode=req.resolved_api_mode,
        main_runtime=main_runtime, final_model=req.final_model, extra_headers=extra_headers,
    )
    return req, retry_kwargs, candidate_kwargs


def _should_retry_same_provider(task: Optional[str], exc: Exception, tag: str) -> bool:
    """True when ``exc`` is a transient transport blip worth a same-provider retry; critical-path
    tasks skip it on a full-budget timeout (``_should_skip_same_provider_retry``) and go straight
    to fallback."""
    if not _is_transient_transport_error(exc):
        return False
    if _should_skip_same_provider_retry(task, exc):
        logger.info("Auxiliary %s%s: timeout on the critical path; "
                    "skipping same-provider retry and falling back: %s", task, tag, exc)
        return False
    return True


def _ladder_step_call(
    step: _LadderStep, req: _PreparedAuxRequest, retry_kwargs: Dict[str, Any], candidate_kwargs: Dict[str, Any],
) -> Tuple[str, tuple, Dict[str, Any]]:
    """Resolve a ladder step into ``(kind, args, kwargs)`` for the sync/async performer."""
    if step.kind == "call":
        return "call", step.args, dict(provider=req.resolved_provider, api_mode=req.resolved_api_mode)
    if step.kind == "retry_same_provider":
        retry_provider, retry_model = step.args
        return "retry", (), dict(retry_kwargs, resolved_provider=retry_provider, resolved_model=retry_model)
    return "fallback", step.args, candidate_kwargs


def _start_recovery_ladder(
    first_err: Exception, req: _PreparedAuxRequest, retry_kwargs: Dict[str, Any], *,
    task: Optional[str], async_mode: bool, route_info: Optional[Dict[str, str]],
):
    """Build the recovery-ladder generator for a failed primary request."""
    return _aux_recovery_ladder(
        first_err, client=req.client, kwargs=req.kwargs, task=task, async_mode=async_mode,
        base_info=req.base_info, resolved_provider=req.resolved_provider,
        resolved_model=req.resolved_model, resolved_base_url=req.resolved_base_url,
        resolved_api_key=req.resolved_api_key, resolved_api_mode=req.resolved_api_mode,
        final_model=req.final_model, max_tokens=retry_kwargs["max_tokens"],
        main_runtime=retry_kwargs["main_runtime"], route_info=route_info)


def _call_llm_impl(
    task: str = None, *, provider: str = None, model: str = None, base_url: str = None,
    api_key: str = None, main_runtime: Optional[Dict[str, Any]] = None, messages: list,
    temperature: Optional[float] = None, max_tokens: int = None, tools: list = None,
    timeout: float = None, extra_body: dict = None, reasoning_config: Optional[dict] = None,
    extra_headers: Optional[Dict[str, str]] = None, api_mode: str = None, stream: bool = False,
    stream_options: dict = None, route_info: Optional[Dict[str, str]] = None,
) -> Any:
    """Centralized synchronous LLM call: resolve provider/model, auth, kwargs, fallbacks.
    task: aux task whose provider:model comes from config (ignored if provider set); api_mode
    overrides task config; timeout=None reads auxiliary.{task}.timeout; extra_headers override
    client defaults. stream=True returns the raw SDK stream (caller consumes/falls back)
    instead of a validated response. RuntimeError if no provider is configured."""
    req, retry_kwargs, candidate_kwargs = _plan_aux_call(
        task, async_mode=False, provider=provider, model=model, base_url=base_url,
        api_key=api_key, main_runtime=main_runtime, messages=messages,
        temperature=temperature, max_tokens=max_tokens, tools=tools, timeout=timeout,
        extra_body=extra_body, reasoning_config=reasoning_config,
        extra_headers=extra_headers, api_mode=api_mode, route_info=route_info,
    )
    client, kwargs, request_provider = req.client, req.kwargs, req.request_provider
    # Streaming path (MoA aggregator): return the raw SDK stream, skipping validation and
    # the fallback chain (they assume a complete response); the caller owns reassembly/fallback.
    if stream:
        kwargs["stream"] = True
        if stream_options:
            kwargs["stream_options"] = stream_options
        if task == "moa_aggregator" and isinstance(client, CodexAuxiliaryClient):
            # Responses-shim clients consume the stream internally and return a completed
            # object Relay's managed stream would iterate; the MoA facade wraps it as one chunk.
            return client.chat.completions.create(**kwargs)
        return _relay_sync_stream(client, kwargs, provider=request_provider, api_mode=req.resolved_api_mode)

    def _primary(**validate_kw: Any) -> Any:
        # Retry on the same provider for a transient transport blip (connection reset / streaming-close /
        # incomplete chunked read / 5xx / 408) before the except-chain below escalates to provider/model
        # fallback. A dropped connection shouldn't abandon an otherwise-healthy provider — this especially
        # matters for pinned auxiliary calls like MoA reference advisors, where "fallback to another
        # provider" is not a meaningful recovery (the advisor is a specific model), so a transient blip that
        # isn't retried simply loses that advisor for the turn (root of the run2 double-advisor "Connection
        # error" collapse — a genuine upstream blip hitting both parallel advisors at once). Attempts are
        # bounded and use exponential backoff. Count is configurable via auxiliary.transient_retries
        # (default 2 retries → 3 total attempts); a second/third failure or any non-transient error falls
        # through to ``first_err`` and the existing fallback handling unchanged. Unified home for the
        # transient retry every auxiliary task shares. (PR #16587)
        return _validate_llm_response(
            _relay_sync_completion(
                client, kwargs, provider=request_provider, api_mode=req.resolved_api_mode,
                create=lambda request: _create_with_progress(
                    client, request, task,
                    force_stream=_provider_requires_stream(
                        request_provider, req.base_info or req.resolved_base_url),
                ),
            ),
            task, **validate_kw,
        )
    try:
        # Bounded same-provider retry (exponential backoff, auxiliary.transient_retries) for
        # transient blips before escalating to fallback — a dropped connection shouldn't
        # abandon a healthy provider (matters for pinned MoA advisors).
        try:
            return _primary(provider=request_provider, base_url=req.base_info)
        except Exception as transient_err:
            if not _should_retry_same_provider(task, transient_err, ""):
                raise
            _max_transient_retries = _transient_retry_count()
            _last_transient = transient_err
            for _attempt in range(1, _max_transient_retries + 1):
                _backoff = min(_TRANSIENT_RETRY_BACKOFF_BASE * (2.0 ** (_attempt - 1)), 8.0)
                logger.info("Auxiliary %s: transient transport error (attempt %d/%d); "
                            "retrying same provider after %.1fs before fallback: %s",
                            task or "call", _attempt, _max_transient_retries, _backoff, _last_transient)
                time.sleep(_backoff)
                try:
                    return _primary()
                except Exception as retry_transient:
                    if not _is_transient_transport_error(retry_transient):
                        raise
                    _last_transient = retry_transient
            raise _last_transient
    except Exception as first_err:
        if request_provider == "claude-agent-sdk":
            # The facade is subscription-scoped; a failure is terminal for
            # this request and must not open a metered recovery route.
            raise
        def _perform(step: _LadderStep) -> Any:
            kind, args, kw = _ladder_step_call(step, req, retry_kwargs, candidate_kwargs)
            if kind == "call":
                return _validate_llm_response(_relay_sync_completion(*args, **kw), task)
            if kind == "retry":
                return _retry_same_provider_sync(**kw)
            return _call_fallback_candidate_sync(*args, **kw)
        result = _drive_ladder(
            _start_recovery_ladder(first_err, req, retry_kwargs, task=task, async_mode=False, route_info=route_info),
            _perform)
        if result is _RERAISE_ORIGINAL:
            raise
        return result


def _coerce_llm_message(response):
    """Pull a message (dict, object, or str) out of a response-or-message value: dict-shaped
    responses/bare messages (compression, proxies) and ChatCompletion objects; MagicMock
    ``reasoning_*`` attrs are deliberately not strings."""
    if response is None or isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "choices" not in response:
            return response
        choices = response.get("choices") or []
    else:
        choices = getattr(response, "choices", None)
        if not choices:
            return response
    return _message_field(choices[0], "message") if choices else None


def _message_field(msg, name):
    return msg.get(name) if isinstance(msg, dict) else getattr(msg, name, None)


def extract_content_or_reasoning(response, *, max_reasoning_chars: int | None = None) -> str:
    """Extract content from an LLM response, falling back to reasoning fields.
    Order: ``content`` (inline think blocks stripped) → ``reasoning``/``reasoning_content`` →
    ``reasoning_details`` (OpenRouter array). Accepts a response or bare message;
    ``max_reasoning_chars`` bounds a reasoning fallback so unbounded chain-of-thought can't
    become the compaction summary. Returns ``""`` if nothing found."""
    msg = _coerce_llm_message(response)
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg.strip()
    raw = _message_field(msg, "content")
    if not isinstance(raw, str):
        raw = str(raw) if raw else ""
    content = raw.strip()
    if content:
        # Mirrors _strip_think_blocks
        cleaned = re.sub(
            r"<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>"
            r".*?"
            r"</(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>",
            "", content, flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        if cleaned:
            return cleaned
    # Content is empty or reasoning-only — try structured reasoning fields
    reasoning_parts: list[str] = []
    for field in ("reasoning", "reasoning_content"):
        val = _message_field(msg, field)
        if val and isinstance(val, str) and val.strip() and val not in reasoning_parts:
            reasoning_parts.append(val.strip())
    details = _message_field(msg, "reasoning_details")
    if details and isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                summary = detail.get("summary") or detail.get("content") or detail.get("text")
                if summary and summary not in reasoning_parts:
                    reasoning_parts.append(summary.strip() if isinstance(summary, str) else str(summary))
    if not reasoning_parts:
        return ""
    text = "\n\n".join(reasoning_parts)
    if max_reasoning_chars is not None and len(text) > max_reasoning_chars:
        logger.warning("fell back to reasoning fields (%d chars); truncating to %d",
                       len(text), max_reasoning_chars)
        return text[:max_reasoning_chars]
    return text


@_relay_auxiliary_call_async
async def async_call_llm(
    task: str = None, *, provider: str = None, model: str = None, base_url: str = None,
    api_key: str = None, main_runtime: Optional[Dict[str, Any]] = None, messages: list,
    temperature: Optional[float] = None, max_tokens: int = None, tools: list = None,
    timeout: float = None, extra_body: dict = None, reasoning_config: Optional[dict] = None,
    route_info: Optional[Dict[str, str]] = None,
) -> Any:
    """Run an asynchronous auxiliary LLM request under the configured limit."""
    semaphore = _acquire_async_aux_semaphore(task)
    if semaphore is not None:
        await semaphore.acquire()
    try:
        return await _async_call_llm_impl(
            task=task, provider=provider, model=model, base_url=base_url, api_key=api_key,
            main_runtime=main_runtime, messages=messages, temperature=temperature,
            max_tokens=max_tokens, tools=tools, timeout=timeout, extra_body=extra_body,
            reasoning_config=reasoning_config, route_info=route_info,
        )
    finally:
        if semaphore is not None:
            semaphore.release()


async def _async_call_llm_impl(
    task: str = None, *, provider: str = None, model: str = None, base_url: str = None,
    api_key: str = None, main_runtime: Optional[Dict[str, Any]] = None, messages: list,
    temperature: Optional[float] = None, max_tokens: int = None, tools: list = None,
    timeout: float = None, extra_body: dict = None, reasoning_config: Optional[dict] = None,
    route_info: Optional[Dict[str, str]] = None,
) -> Any:
    """Centralized asynchronous LLM call; see call_llm() for full documentation.
    No per-request header / api_mode override on the async entry point."""
    req, retry_kwargs, candidate_kwargs = _plan_aux_call(
        task, async_mode=True, provider=provider, model=model, base_url=base_url,
        api_key=api_key, main_runtime=main_runtime, messages=messages,
        temperature=temperature, max_tokens=max_tokens, tools=tools, timeout=timeout,
        extra_body=extra_body, reasoning_config=reasoning_config,
        extra_headers=None, api_mode=None, route_info=route_info,
    )
    client, kwargs, request_provider = req.client, req.kwargs, req.request_provider
    try:
        # Retry ONCE on the same provider for a transient blip before fallback (see call_llm()).
        # (PR #16587)
        _force_stream_async = (
            _provider_requires_stream(request_provider, req.base_info or req.resolved_base_url)
            and not isinstance(client, (
                AsyncCodexAuxiliaryClient, AsyncAnthropicAuxiliaryClient, AsyncBedrockAuxiliaryClient)))

        async def _acreate(_kwargs: Dict[str, Any]) -> Any:
            if _force_stream_async:
                return await _acreate_with_stream(client, _kwargs, task)
            return await client.chat.completions.create(**_kwargs)

        async def _primary(**validate_kw: Any) -> Any:
            return _validate_llm_response(
                await _relay_async_completion(
                    client, kwargs, provider=request_provider, api_mode=req.resolved_api_mode,
                    create=_acreate),
                task, **validate_kw)
        try:
            return await _primary(provider=request_provider, base_url=req.base_info)
        except Exception as transient_err:
            # The async Codex adapter wraps the sync stream via to_thread: same TimeoutError here.
            if not _should_retry_same_provider(task, transient_err, " (async)"):
                raise
            logger.info("Auxiliary %s (async): transient transport error; retrying "
                        "once on the same provider before fallback: %s", task or "call", transient_err)
            return await _primary()
    except Exception as first_err:
        if request_provider == "claude-agent-sdk":
            # The facade is subscription-scoped; a failure is terminal for
            # this request and must not open a metered recovery route.
            raise
        async def _perform(step: _LadderStep) -> Any:
            kind, args, kw = _ladder_step_call(step, req, retry_kwargs, candidate_kwargs)
            if kind == "call":
                return _validate_llm_response(await _relay_async_completion(*args, **kw), task)
            if kind == "retry":
                return await _retry_same_provider_async(**kw)
            fb_client, fb_model, fb_label = args
            fb_client, _ = _to_async_client(fb_client, fb_model or "", is_vision=(task == "vision"))
            return await _call_fallback_candidate_async(fb_client, fb_model, fb_label, **kw)
        result = await _drive_ladder_async(
            _start_recovery_ladder(first_err, req, retry_kwargs, task=task, async_mode=True, route_info=route_info),
            _perform)
        if result is _RERAISE_ORIGINAL:
            raise
        return result


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import Path  # noqa: F401,E402
import copy  # noqa: F401,E402

NOUS_EXTRA_BODY = _nous_extra_body()

def get_async_text_auxiliary_client(task: str = "", *, main_runtime: Optional[Dict[str, Any]] = None):
    """Return (async_client, model_slug) for async consumers.

    For standard providers returns (AsyncOpenAI, model). For Codex returns
    (AsyncCodexAuxiliaryClient, model) which wraps the Responses API.
    Returns (None, None) when no provider is available.
    """
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(task or None)
    return resolve_provider_client(
        provider,
        model=model,
        async_mode=True,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
    )
# ---- END PLUGIN-COMPAT ----
