"""Opt-in strict invocation binding: pin one run to one pool credential, model and wire reasoning effort.

A controller hands the worker a nonsecret ``InvocationBinding``. Strict mode then re-verifies it at every
physical request (each retry, rotation and refresh re-enters the same gate), against the material the wire
client is actually built from, never against ``agent.reasoning_config`` or a label:

* the key the request client uses must belong to exactly the pool entry ``credential_id`` (account
  substitution on retry/refresh is refused, not repaired);
* provider, model and the effort present in the final request payload must equal the binding;
* the evidence lands on the native session row (``model_config.invocation_binding``), written once.

What this does NOT prove: no provider here exposes a supported contract that authenticates the serving
account behind a key, so the record always says ``account_attestation: unsupported`` /
``account_verified: False``. A pool id, label, key or JWT claim is a locator, not account proof.
``require_account_attestation`` therefore refuses today instead of pretending.

Stdlib-only at import time: ``hermes_state_invocation`` imports this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

BINDING_VERSION = 1
RECORD_KEY = "invocation_binding"  # sessions.model_config key holding the native evidence
SUPPORTED_API_MODES = ("anthropic_messages", "chat_completions", "codex_responses")
ACCOUNT_ATTESTATION = "unsupported"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_CONTEXT_BYTES = 64 * 1024
_CONTEXT_FIELDS = frozenset({
    "version", "invocation_id", "credential_id", "provider", "model", "reasoning_effort",
    "require_account_attestation",
})


class InvocationBindingError(RuntimeError):
    """Fail-closed refusal. ``code`` is an allowlisted slug; ``str(exc)`` is the code only, so no key,
    header, prompt or provider text can ride along into logs, results or the native record."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _clean(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvocationBindingError(code)
    return value.strip()


def _identifier(value: Any, name: str) -> None:
    """Ids are matched verbatim against pool rows and native records, so they are validated, never trimmed."""
    if not isinstance(value, str) or not value.strip():
        raise InvocationBindingError(f"{name}_missing")
    if not _ID_RE.fullmatch(value):
        raise InvocationBindingError(f"{name}_invalid")


@dataclass(frozen=True)
class InvocationBinding:
    invocation_id: str
    credential_id: str
    provider: str
    model: str
    reasoning_effort: str
    require_account_attestation: bool = False

    def __post_init__(self):
        _identifier(self.credential_id, "credential_id")
        _identifier(self.invocation_id, "invocation_id")
        object.__setattr__(self, "provider", _clean(self.provider, "binding_field_missing").lower())
        object.__setattr__(self, "model", _clean(self.model, "binding_field_missing"))
        object.__setattr__(self, "reasoning_effort", _clean(self.reasoning_effort, "binding_field_missing").lower())
        if self.require_account_attestation:
            raise InvocationBindingError("account_attestation_unsupported")

    def expected(self) -> dict[str, str]:
        return {
            "invocation_id": self.invocation_id, "credential_id": self.credential_id, "provider": self.provider,
            "model": self.model, "reasoning_effort": self.reasoning_effort,
        }

    @property
    def digest(self) -> str:
        canonical = json.dumps({"version": BINDING_VERSION, **self.expected()}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def binding_from_mapping(data: Any) -> InvocationBinding:
    if not isinstance(data, Mapping) or set(data) - _CONTEXT_FIELDS or data.get("version") != BINDING_VERSION:
        raise InvocationBindingError("binding_context_invalid")
    return InvocationBinding(
        invocation_id=data.get("invocation_id"), credential_id=data.get("credential_id"),
        provider=data.get("provider"), model=data.get("model"), reasoning_effort=data.get("reasoning_effort"),
        require_account_attestation=bool(data.get("require_account_attestation", False)),
    )


def load_binding_file(path: str) -> InvocationBinding:
    """Parse the controller's nonsecret binding file (never a credential or a grant)."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(_MAX_CONTEXT_BYTES + 1)
        if len(raw) > _MAX_CONTEXT_BYTES:
            raise ValueError("oversized")
        data = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError):
        raise InvocationBindingError("binding_context_invalid") from None
    return binding_from_mapping(data)


def load_cli_binding(path: Optional[str]) -> Optional[InvocationBinding]:
    """``--invocation-binding`` entry: None when not requested; a refused file exits 2 before any agent work."""
    if not path:
        return None
    try:
        return load_binding_file(path)
    except InvocationBindingError as exc:
        print(f"Error: invocation binding refused ({exc.code})", file=sys.stderr)
        raise SystemExit(2) from None


def build_record(binding: InvocationBinding) -> dict[str, Any]:
    """The write-once native evidence. ``expected`` is what the controller asked for; request snapshots
    carry what actually reached the wire. Account fields are constants until a provider contract exists."""
    return {
        "version": BINDING_VERSION, "invocation_id": binding.invocation_id, "digest": binding.digest,
        "expected": binding.expected(), "account_attestation": ACCOUNT_ATTESTATION, "account_verified": False,
        "bound_at": time.time(), "request_count": 0, "responded_count": 0, "failed_count": 0,
        "last_request": None, "violations": [],
    }


def wire_reasoning_effort(api_mode: str, api_kwargs: Any) -> Optional[str]:
    """Effort present in the payload about to be sent, or None when the payload does not carry one.
    ``anthropic_messages`` only counts adaptive thinking (``output_config.effort``): budget-token thinking and
    haiku-style omission prove nothing about effort."""
    if not isinstance(api_kwargs, Mapping):
        return None

    def _sub(container: Any, key: str) -> Mapping:
        value = container.get(key) if isinstance(container, Mapping) else None
        return value if isinstance(value, Mapping) else {}

    if api_mode == "anthropic_messages":
        effort = _sub(api_kwargs, "output_config").get("effort") if _sub(api_kwargs, "thinking").get("type") == "adaptive" else None
    elif api_mode == "codex_responses":
        effort = _sub(api_kwargs, "reasoning").get("effort")
    elif api_mode == "chat_completions":
        effort = api_kwargs.get("reasoning_effort") or _sub(_sub(api_kwargs, "extra_body"), "reasoning").get("effort")
    else:
        effort = None
    return effort.strip().lower() if isinstance(effort, str) and effort.strip() else None


def request_material(agent: Any) -> tuple[Any, Any]:
    """``(api_key, base_url)`` the request client is built from (not ``agent.api_key``, which may lag)."""
    if getattr(agent, "api_mode", None) == "anthropic_messages":
        return getattr(agent, "_anthropic_api_key", None), getattr(agent, "_anthropic_base_url", None)
    kwargs = getattr(agent, "_client_kwargs", None) or {}
    return kwargs.get("api_key"), kwargs.get("base_url")


def _verify_credential(agent: Any, binding: InvocationBinding, api_key: Any, base_url: Any) -> Any:
    from agent.credential_pool import credential_pool_entry_serves_endpoint, credential_pool_matches_provider

    pool = getattr(agent, "_credential_pool", None)
    if pool is None:
        raise InvocationBindingError("credential_pool_missing")
    if not credential_pool_matches_provider(pool, binding.provider, base_url=base_url):
        raise InvocationBindingError("provider_mismatch")
    entries = pool.entries()
    bound = [entry for entry in entries if entry.id == binding.credential_id]
    if len(bound) != 1:
        raise InvocationBindingError("credential_not_found")
    if not isinstance(api_key, str) or not api_key.strip():
        raise InvocationBindingError("credential_missing")
    # Exact holder set: a key shared by two entries, or held by another entry, is not provably the bound one.
    if [entry.id for entry in entries if entry.runtime_api_key == api_key] != [binding.credential_id]:
        raise InvocationBindingError("credential_mismatch")
    if not credential_pool_entry_serves_endpoint(bound[0], base_url):
        raise InvocationBindingError("endpoint_mismatch")
    return bound[0]


def _verify_route(agent: Any, binding: InvocationBinding, api_key: Any, base_url: Any) -> Any:
    if getattr(agent, "api_mode", None) not in SUPPORTED_API_MODES:
        raise InvocationBindingError("api_mode_unsupported")
    if (getattr(agent, "provider", "") or "").strip().lower() != binding.provider:
        raise InvocationBindingError("provider_mismatch")
    if getattr(agent, "_fallback_activated", False):
        raise InvocationBindingError("fallback_active")
    if getattr(agent, "model", None) != binding.model:
        raise InvocationBindingError("model_mismatch")
    return _verify_credential(agent, binding, api_key, base_url)


def verify_client_material(agent: Any, api_key: Any, base_url: Any) -> None:
    """Request-client creation gate: the key about to authenticate belongs to the bound credential."""
    binding = agent._invocation_binding
    try:
        _verify_route(agent, binding, api_key, base_url)
    except InvocationBindingError as exc:
        _note_violation(agent, binding, exc.code)
        raise


def verify_request(agent: Any, binding: InvocationBinding, api_kwargs: Any) -> dict[str, Any]:
    """Check the final payload plus client material; return the actual-side snapshot to persist."""
    api_key, base_url = request_material(agent)
    entry = _verify_route(agent, binding, api_key, base_url)
    if getattr(agent, "api_key", None) != api_key:
        raise InvocationBindingError("credential_mismatch")
    if not isinstance(api_kwargs, Mapping) or api_kwargs.get("model") != binding.model:
        raise InvocationBindingError("model_mismatch")
    wire_effort = wire_reasoning_effort(agent.api_mode, api_kwargs)
    if wire_effort is None:
        raise InvocationBindingError("reasoning_unproven")
    if wire_effort != binding.reasoning_effort:
        raise InvocationBindingError("reasoning_mismatch")
    return {
        "provider": binding.provider, "api_mode": agent.api_mode, "model": api_kwargs["model"],
        "wire_reasoning_effort": wire_effort, "credential_id": entry.id,
        "endpoint_host": urlsplit(str(base_url or "")).hostname or "", "at": time.time(),
    }


def bind_agent(agent: Any, binding: Optional[InvocationBinding]) -> None:
    """Arm strict mode on a freshly built agent; raises for anything the binding cannot cover."""
    agent._invocation_binding = binding
    agent._invocation_bound_session = None
    if binding is None:
        return
    if not isinstance(binding, InvocationBinding):
        raise InvocationBindingError("binding_context_invalid")
    agent._credential_pool_entry_id = binding.credential_id
    agent.skip_background_review = True  # the review fork is a second model client this binding does not cover
    api_key, base_url = request_material(agent)
    _verify_route(agent, binding, api_key, base_url)


def _native_write(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except InvocationBindingError:
        raise
    except Exception as exc:
        logger.warning("invocation binding native write failed: %s", type(exc).__name__)
        raise InvocationBindingError("native_record_failed") from exc


def _native_db(agent: Any) -> Any:
    if getattr(agent, "_persist_disabled", False):
        raise InvocationBindingError("native_session_unavailable")
    agent._ensure_db_session()  # lazy row creation swallows transient errors; strict mode must not
    db = getattr(agent, "_session_db", None)
    if db is None or not getattr(agent, "_session_db_created", False) or not agent.session_id:
        raise InvocationBindingError("native_session_unavailable")
    return db


def _note_violation(agent: Any, binding: InvocationBinding, code: str) -> None:
    """Best-effort: a violation is already fatal; the row only makes it visible to the controller."""
    try:
        db = getattr(agent, "_session_db", None)
        if db is not None and getattr(agent, "_invocation_bound_session", None) == agent.session_id:
            db.record_invocation_violation(agent.session_id, binding.invocation_id, code)
    except Exception:
        logger.debug("invocation binding violation note failed", exc_info=True)


def begin_request(agent: Any, api_kwargs: Any) -> int:
    """Gate one physical request: verify, bind the native row (first request), persist ``prepared``."""
    binding = agent._invocation_binding
    try:
        snapshot = verify_request(agent, binding, api_kwargs)
        db = _native_db(agent)
        if agent._invocation_bound_session != agent.session_id:
            _native_write(lambda: db.bind_invocation(agent.session_id, build_record(binding)))
            agent._invocation_bound_session = agent.session_id
        return _native_write(lambda: db.record_invocation_request(agent.session_id, binding.invocation_id, snapshot))
    except InvocationBindingError as exc:
        _note_violation(agent, binding, exc.code)
        raise


def settle_request(agent: Any, ordinal: int, state: str, error_type: Optional[str] = None) -> None:
    binding = agent._invocation_binding
    _native_write(lambda: agent._session_db.settle_invocation_request(
        agent.session_id, binding.invocation_id, ordinal, state, error_type))


def run_bound_request(agent: Any, api_kwargs: Any, send: Callable[[], Any]) -> Any:
    """Strict-mode wrapper for one physical request. A request that never settles stays ``prepared``
    in the native row and is never admissible; a failed settle after a response refuses the response."""
    ordinal = begin_request(agent, api_kwargs)
    try:
        response = send()
    except BaseException as exc:
        try:
            settle_request(agent, ordinal, "failed", type(exc).__name__)
        except InvocationBindingError:
            logger.debug("invocation binding failure settle skipped", exc_info=True)
        raise
    settle_request(agent, ordinal, "responded")
    return response


def interruptible_api_call(agent: Any, api_kwargs: Any) -> Any:
    """``AIAgent._interruptible_api_call``: the unchanged helper, wrapped by the gate in strict mode only.
    Every non-streaming and codex request (and every retry of one) passes through here."""
    from agent import chat_completion_helpers as helpers

    if getattr(agent, "_invocation_binding", None) is None:
        return helpers.interruptible_api_call(agent, api_kwargs)
    return run_bound_request(agent, api_kwargs, lambda: helpers.interruptible_api_call(agent, api_kwargs))


def interruptible_streaming_api_call(agent: Any, api_kwargs: Any, *, on_first_delta: Any = None) -> Any:
    """``AIAgent._interruptible_streaming_api_call`` with the same gate. Codex streams through
    ``_interruptible_api_call`` and is gated there, so gating it here too would double-count."""
    from agent import chat_completion_helpers as helpers

    def _send() -> Any:
        return helpers.interruptible_streaming_api_call(agent, api_kwargs, on_first_delta=on_first_delta)

    if getattr(agent, "_invocation_binding", None) is None or agent.api_mode == "codex_responses":
        return _send()
    if agent._interrupt_requested:
        raise InterruptedError("Agent interrupted before streaming API call")
    return run_bound_request(agent, api_kwargs, _send)


def result_reference(agent: Any, binding: InvocationBinding, session_id: str) -> dict[str, Any]:
    """Terminal ``result`` stub for the controller. Never ``verified``: ``bound_unattested`` means every
    request passed the credential/model/effort gate, not that the serving account was authenticated."""
    ref: dict[str, Any] = {
        "invocation_id": binding.invocation_id, "digest": binding.digest, "session_id": session_id,
        "status": "rejected", "account_attestation": ACCOUNT_ATTESTATION, "account_verified": False,
        "request_count": 0, "responded_count": 0, "failed_count": 0, "violations": [],
    }
    db = getattr(agent, "_session_db", None) if agent is not None else None
    try:
        record = db.get_invocation_binding(session_id) if db is not None and session_id else None
    except Exception:
        record = None
    if not isinstance(record, dict) or record.get("invocation_id") != binding.invocation_id:
        return ref
    for key in ("request_count", "responded_count", "failed_count"):
        ref[key] = int(record.get(key) or 0)
    ref["violations"] = list(record.get("violations") or [])
    if ref["violations"]:
        ref["status"] = "violation"
    elif ref["request_count"]:
        in_flight = ref["request_count"] - ref["responded_count"] - ref["failed_count"]
        ref["status"] = "incomplete" if in_flight > 0 or not ref["responded_count"] else "bound_unattested"
    return ref
