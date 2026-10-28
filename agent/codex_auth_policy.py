"""One-shot stdio-process Codex credential policy; not an OS sandbox.

No environment/config/RPC setter or reset: threads share the immutable launch
policy. A fork cannot inherit authority; use a new exec with an explicit opt-in.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass(frozen=True, slots=True)
class CodexAuthPolicy:
    pid: int
    read_only: bool = False


_policy: CodexAuthPolicy | None = None
_bind_lock = Lock()


def bind_codex_auth_policy(*, read_only: bool = False) -> CodexAuthPolicy:
    global _policy
    if type(read_only) is not bool:
        raise TypeError("Codex auth read_only must be a boolean")
    with _bind_lock:
        if _policy is not None:
            raise RuntimeError("Codex auth process policy is already bound")
        _policy = CodexAuthPolicy(os.getpid(), read_only)
        return _policy


def codex_auth_read_only() -> bool:
    if _policy is not None and _policy.read_only and _policy.pid != os.getpid():
        raise RuntimeError("Codex auth policy cannot cross a fork; use a fresh exec")
    return _policy is not None and _policy.read_only


def require_codex_auth_writable() -> None:
    if codex_auth_read_only():
        raise RuntimeError("Codex credential mutation is disabled by --codex-auth-read-only")


def is_codex_provider(provider: str | None) -> bool:
    # Reuse native static normalization without profile/plugin discovery.
    from agent.auxiliary_client import _normalize_chain_label
    from hermes_cli.providers import normalize_provider
    return normalize_provider(_normalize_chain_label(provider or ""), overlay_profiles=[]) == "openai-codex"


def refuse_mutable_codex_pool(provider: str) -> None:
    if codex_auth_read_only() and is_codex_provider(provider):
        require_codex_auth_writable()


def require_native_codex_runtime(api_mode: str, *, read_only: bool = False) -> None:
    if api_mode == "codex_app_server" and (read_only or codex_auth_read_only()):
        raise RuntimeError("--codex-auth-read-only does not support codex_app_server; select the native Codex runtime")


def reject_readonly_codex_fallback(error: Exception) -> None:
    if codex_auth_read_only() and is_codex_provider(getattr(error, "provider", None)):
        raise error


def readonly_codex_agent(agent: Any) -> bool:
    return codex_auth_read_only() and is_codex_provider(getattr(agent, "provider", None))


def validate_readonly_codex_primary(agent: Any) -> None:
    if readonly_codex_agent(agent):
        # The generic router may return None (e.g. without a model), allowing init fallback.
        from hermes_cli.auth import resolve_codex_runtime_credentials
        resolve_codex_runtime_credentials(read_only=True)


def refuse_codex_credential_copy(source_home: Path) -> None:
    if not codex_auth_read_only():
        return
    auth_path = source_home / "auth.json"
    refusal = "Codex credential copying is disabled by --codex-auth-read-only"
    try:
        store = json.loads(auth_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        if not auth_path.is_symlink():
            return
        raise RuntimeError(refusal) from None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise RuntimeError(refusal) from None
    if not isinstance(store, dict):
        raise RuntimeError(refusal)
    for section in ("providers", "credential_pool"):
        if section not in store:
            continue
        entries = store[section]
        if not isinstance(entries, dict) or any(is_codex_provider(provider) for provider in entries):
            raise RuntimeError(refusal)


def bind_agent_codex_auth(agent: Any, credential_pool: Any) -> Any:
    require_native_codex_runtime(agent.api_mode)
    agent.codex_auth_read_only = codex_auth_read_only()
    if agent.codex_auth_read_only and is_codex_provider(agent.provider):
        agent._credential_pool = None
        return None
    return credential_pool
