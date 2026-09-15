"""Classify secret names for the tool-credential handles apply gate.

Default remains process-environ hydration. When ``secrets.tool_credentials``
is ``"handles"`` (strip + lower), a small explicit set of tool-facing names is
withheld from ``environ`` and recorded as handles instead. Hermes provider
keys, secret-source bootstrap tokens, unknown names, and
``secrets.process_plaintext`` overrides still apply (fail-open).
"""
from __future__ import annotations

from typing import Any, FrozenSet, Mapping

# Tool-facing names withheld from environ in handles mode. Exact set only —
# never a ``*_TOKEN`` / ``*_KEY`` / ``*_SECRET`` suffix heuristic (that would
# drop OPENROUTER_API_KEY / ANTHROPIC_API_KEY and break model calls).
TOOL_FACING_ENV_VARS: FrozenSet[str] = frozenset({
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY_PATH",
    "GITHUB_APP_INSTALLATION_ID",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
})

# Still written to environ in handles mode: Hermes model providers and the
# bootstrap tokens secret sources need to fetch. Unknown names fail-open
# apply as well; this set is the explicit process-plaintext allowlist.
PROCESS_PLAINTEXT_ENV_VARS: FrozenSet[str] = frozenset({
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_TOKEN",
    "BWS_ACCESS_TOKEN",
    "OP_SERVICE_ACCOUNT_TOKEN",
})


def handles_mode(secrets_cfg: Any) -> bool:
    """True only when ``tool_credentials`` strip+lower equals ``handles``."""
    if not isinstance(secrets_cfg, Mapping):
        return False
    raw = secrets_cfg.get("tool_credentials")
    if not isinstance(raw, str):
        return False
    return raw.strip().lower() == "handles"


def _process_plaintext_overrides(secrets_cfg: Mapping[str, Any]) -> FrozenSet[str]:
    raw = secrets_cfg.get("process_plaintext")
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(n.strip() for n in raw if isinstance(n, str) and n.strip())


def should_apply_to_environ(var: str, secrets_cfg: Any) -> bool:
    """Whether ``var`` may be written to process environ for this apply pass."""
    if not handles_mode(secrets_cfg):
        return True
    cfg: Mapping[str, Any] = secrets_cfg if isinstance(secrets_cfg, Mapping) else {}
    if var in _process_plaintext_overrides(cfg) or var in PROCESS_PLAINTEXT_ENV_VARS:
        return True
    if var in TOOL_FACING_ENV_VARS:
        return False
    return True
