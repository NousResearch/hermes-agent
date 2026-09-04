"""Unified removal contract for every credential source Hermes reads from.

Readers live in ``agent.credential_pool``; what is unified here is **removal**:
``hermes auth remove <provider> <N>`` must make the entry stay gone across
``load_pool()`` calls. Each source registers a ``RemovalStep`` whose
``remove_fn`` cleans the external state the source reads from, and the
dispatcher suppresses ``(provider, source_id)`` in auth.json so the seeding
branch skips the upsert. Adding a source: wire a reader branch in
``_seed_from_*``, gate it behind ``is_source_suppressed``, register a step here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class RemovalResult:
    """Outcome of removing a credential source.

    ``cleaned``: external state actually mutated (printed to the user).
    ``hints``: diagnostics about state left intact or that the user must clean
    up. ``suppress``: call ``suppress_credential_source`` afterwards so
    ``load_pool`` skips the source; only ``manual`` entries legitimately use False.
    """

    cleaned: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    suppress: bool = True


@dataclass
class RemovalStep:
    """How to remove one credential source.

    ``provider`` ``"*"`` matches any provider. ``match_fn`` overrides literal
    ``source_id`` matching (prefix patterns like ``env:*`` / ``config:*``).
    ``remove_fn(provider, removed_entry) -> RemovalResult``.
    """

    provider: str
    source_id: str
    remove_fn: Callable[..., RemovalResult]
    match_fn: Optional[Callable[[str], bool]] = None
    description: str = ""

    def matches(self, provider: str, source: str) -> bool:
        if self.provider != "*" and self.provider != provider:
            return False
        if self.match_fn is not None:
            return self.match_fn(source)
        return source == self.source_id


def find_removal_step(provider: str, source: str) -> Optional[RemovalStep]:
    """First matching RemovalStep, or None (``manual``: nothing external to clean)."""
    return next((step for step in _REGISTRY if step.matches(provider, source)), None)


def _remove_env_source(provider: str, removed) -> RemovalResult:
    """env:<VAR> — clear from ~/.hermes/.env; hint when the shell exports it."""
    from hermes_cli.config import get_env_path, remove_env_value

    result = RemovalResult()
    env_var = removed.source[len("env:"):]
    if not env_var:
        return result

    # Detect shell vs .env BEFORE remove_env_value pops os.environ. Read the
    # .env as utf-8-sig like hermes_cli/config.py: a BOM-sensitive read would
    # misreport a Notepad-edited .env var as a shell export.
    env_in_process = bool(os.getenv(env_var))
    env_in_dotenv = False
    try:
        env_path = get_env_path()
        if env_path.exists():
            env_in_dotenv = any(
                line.strip().startswith(f"{env_var}=")
                for line in env_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
            )
    except OSError:
        pass

    if remove_env_value(env_var):
        result.cleaned.append(f"Cleared {env_var} from .env")

    if env_in_process and not env_in_dotenv:
        result.hints.extend([
            f"Note: {env_var} is still set in your shell environment "
            f"(not in ~/.hermes/.env).",
            "  Unset it there (shell profile, systemd EnvironmentFile, "
            "launchd plist, etc.) or it will keep being visible to Hermes.",
            f"  The pool entry is now suppressed — Hermes will ignore "
            f"{env_var} until you run `hermes auth add {provider}`.",
        ])
    else:
        result.hints.append(
            f"Suppressed env:{env_var} — it will not be re-seeded even "
            f"if the variable is re-exported later."
        )
    return result


def _remove_hermes_pkce(provider: str, removed) -> RemovalResult:
    """~/.hermes/.anthropic_oauth.json is ours — delete it outright."""
    from hermes_constants import get_hermes_home

    result = RemovalResult()
    oauth_file = get_hermes_home() / ".anthropic_oauth.json"
    if oauth_file.exists():
        try:
            oauth_file.unlink()
            result.cleaned.append("Cleared Hermes Anthropic OAuth credentials")
        except OSError as exc:
            result.hints.append(f"Could not delete {oauth_file}: {exc}")
    return result


def _remove_auth_store_oauth(provider: str, removed) -> RemovalResult:
    """Clear auth.json ``providers.<provider>`` (nous, minimax-oauth, xai-oauth, openai-codex).

    Suppression by the dispatcher is still required — otherwise
    ``_seed_from_singletons`` re-seeds from any path that rewrites the block.
    """
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store

    result = RemovalResult()
    with _auth_store_lock():
        auth_store = _load_auth_store()
        providers_dict = auth_store.get("providers")
        if isinstance(providers_dict, dict) and provider in providers_dict:
            del providers_dict[provider]
            _save_auth_store(auth_store)
            result.cleaned.append(f"Cleared {provider} OAuth tokens from auth store")
    return result


def _remove_xai_oauth_device_code(provider: str, removed) -> RemovalResult:
    result = _remove_auth_store_oauth(provider, removed)
    result.hints.append(
        "Run `hermes model` → xAI Grok OAuth (SuperGrok / Premium+) to re-authenticate if needed."
    )
    return result


def _remove_codex_device_code(provider: str, removed) -> RemovalResult:
    """Codex tokens also live in ~/.codex/auth.json (Codex CLI's file, kept).

    Suppress the canonical ``device_code`` key — not just ``removed.source`` —
    so a ``manual:device_code`` removal still blocks the re-seed path.
    """
    from hermes_cli.auth import suppress_credential_source

    result = _remove_auth_store_oauth(provider, removed)
    suppress_credential_source(provider, "device_code")
    result.hints.extend([
        "Suppressed openai-codex device_code source — it will not be re-seeded.",
        "Note: Codex CLI credentials still live in ~/.codex/auth.json",
        "Run `hermes auth add openai-codex` to re-enable if needed.",
    ])
    return result


def _remove_copilot_gh(provider: str, removed) -> RemovalResult:
    """The same Copilot token is seeded as gh_cli AND env:<VAR> rows, so suppress
    every variant or the duplicates resurrect the entry. gh CLI and shell state
    are left untouched."""
    from hermes_cli.auth import suppress_credential_source

    suppress_credential_source(provider, "gh_cli")
    for env_var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        suppress_credential_source(provider, f"env:{env_var}")
    return RemovalResult(hints=[
        "Suppressed all copilot token sources (gh_cli + env vars) — they will not be re-seeded.",
        "Note: Your gh CLI / shell environment is unchanged.",
        "Run `hermes auth add copilot` to re-enable if needed.",
    ])


def _suppress_only(*hints: str) -> Callable[..., RemovalResult]:
    """remove_fn for sources whose backing file/config belongs to another tool
    (Claude Code, Qwen CLI, config.yaml): never delete it, just suppress and
    explain. ``{source}`` in a hint is the removed entry's source string."""
    def remove_fn(provider: str, removed) -> RemovalResult:
        return RemovalResult(hints=[h.format(source=removed.source) for h in hints])
    return remove_fn


# ORDER MATTERS — ``find_removal_step`` returns the first match. Provider-
# specific steps precede the generic ``env:*`` step so copilot's ``env:GH_TOKEN``
# takes the copilot path (no .env edits) rather than the generic env-var removal.
_REGISTRY: List[RemovalStep] = [
    RemovalStep(
        provider="copilot", source_id="gh_cli",
        match_fn=lambda src: src == "gh_cli" or src.startswith("env:"),
        remove_fn=_remove_copilot_gh,
        description="gh auth token / COPILOT_GITHUB_TOKEN / GH_TOKEN",
    ),
    RemovalStep(
        provider="*", source_id="env:",
        match_fn=lambda src: src.startswith("env:"),
        remove_fn=_remove_env_source,
        description="Any env-seeded credential (XAI_API_KEY, DEEPSEEK_API_KEY, etc.)",
    ),
    RemovalStep(
        provider="anthropic", source_id="claude_code",
        remove_fn=_suppress_only(
            "Suppressed claude_code credential — it will not be re-seeded.",
            "Note: Claude Code credentials still live in ~/.claude/.credentials.json",
            "Run `hermes auth add anthropic` to re-enable if needed.",
        ),
        description="~/.claude/.credentials.json",
    ),
    RemovalStep(
        provider="anthropic", source_id="hermes_pkce",
        remove_fn=_remove_hermes_pkce,
        description="~/.hermes/.anthropic_oauth.json",
    ),
    RemovalStep(
        provider="nous", source_id="device_code",
        remove_fn=_remove_auth_store_oauth,
        description="auth.json providers.nous",
    ),
    RemovalStep(
        provider="openai-codex", source_id="device_code",
        match_fn=lambda src: src == "device_code" or src.endswith(":device_code"),
        remove_fn=_remove_codex_device_code,
        description="auth.json providers.openai-codex + ~/.codex/auth.json",
    ),
    RemovalStep(
        provider="xai-oauth", source_id="device_code",
        remove_fn=_remove_xai_oauth_device_code,
        description="auth.json providers.xai-oauth",
    ),
    RemovalStep(
        provider="qwen-oauth", source_id="qwen-cli",
        remove_fn=_suppress_only(
            "Suppressed qwen-cli credential — it will not be re-seeded.",
            "Note: Qwen CLI credentials still live in ~/.qwen/oauth_creds.json",
            "Run `hermes auth add qwen-oauth` to re-enable if needed.",
        ),
        description="~/.qwen/oauth_creds.json",
    ),
    RemovalStep(
        provider="minimax-oauth", source_id="oauth",
        remove_fn=_remove_auth_store_oauth,
        description="auth.json providers.minimax-oauth",
    ),
    RemovalStep(
        provider="*", source_id="config:",
        match_fn=lambda src: src.startswith("config:") or src == "model_config",
        remove_fn=_suppress_only(
            "Suppressed {source} — it will not be re-seeded.",
            "Note: The underlying value in config.yaml is unchanged.  Edit it "
            "directly if you want to remove the credential from disk.",
        ),
        description="Custom provider config.yaml api_key field",
    ),
]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def register(step: RemovalStep) -> RemovalStep:
    _REGISTRY.append(step)
    return step
# ---- END PLUGIN-COMPAT ----
