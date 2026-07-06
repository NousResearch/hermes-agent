"""Applying fetched Proton Pass secrets to the environment.

This module owns:

* :class:`FetchResult` — the outcome of one pull (shared shape with the
  Bitwarden source so env_loader records both identically).
* :func:`plan_application` — the SINGLE application planner.  It decides, for
  each fetched secret, whether it is applied or skipped and WHY.  Both
  :func:`apply_protonpass_secrets` (which applies the plan) and the CLI ``sync``
  / dry-run (which render the plan) use it, so the skip rules
  (bootstrap-token, already-set) live in exactly ONE place.
* :func:`apply_protonpass_secrets` — the legacy fetch-and-apply compatibility
  entry point used by direct callers and the Proton Pass CLI.  It NEVER raises:
  any failure returns a :class:`FetchResult` with ``error`` set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from agent.secret_sources.base import FetchResult

from .config import ProtonPassConfig, strip_bootstrap_ref
from .fetch import fetch_protonpass_secrets
from .install import find_pass_cli
from .session import _redact_token

# Skip-reason constants used by the planner so callers can branch / render
# without re-deriving the rules.
SKIP_BOOTSTRAP_TOKEN = "bootstrap-token"
SKIP_ALREADY_SET = "already-set"


@dataclass
class PlanItem:
    """One entry in an application plan.

    ``applied`` is True when the value should be set into the environment;
    False entries carry a ``reason`` (one of the ``SKIP_*`` constants).
    ``overrides`` records whether applying replaces an existing value (for
    dry-run rendering).
    """

    name: str
    value: str
    applied: bool
    reason: Optional[str] = None
    overrides: bool = False


def plan_application(
    secrets: Mapping[str, str],
    environ: Mapping[str, str],
    *,
    override_existing: bool,
    token_env: str,
) -> List[PlanItem]:
    """Decide, per fetched secret, whether it is applied or skipped and why.

    This is the SINGLE home for the application invariants:

    * The bootstrap service-token env var is NEVER overwritten — applying it
      would clobber the very token we authenticated with.  (Skip reason
      :data:`SKIP_BOOTSTRAP_TOKEN`.)
    * An already-set var is kept unless ``override_existing`` is True.  (Skip
      reason :data:`SKIP_ALREADY_SET`.)

    Returns a deterministic, name-sorted list of :class:`PlanItem`.  Callers
    apply it (set env) or render it (CLI dry-run) without re-deriving any rule.
    """
    plan: List[PlanItem] = []
    for name in sorted(secrets):
        value = secrets[name]
        if name == token_env:
            plan.append(
                PlanItem(name=name, value=value, applied=False,
                         reason=SKIP_BOOTSTRAP_TOKEN)
            )
            continue
        already = bool(environ.get(name))
        if already and not override_existing:
            plan.append(
                PlanItem(name=name, value=value, applied=False,
                         reason=SKIP_ALREADY_SET)
            )
            continue
        plan.append(
            PlanItem(name=name, value=value, applied=True, overrides=already)
        )
    return plan


def apply_protonpass_secrets(
    *,
    enabled: bool,
    config: Optional[ProtonPassConfig] = None,
    service_token_env: str = "PROTON_PASS_PERSONAL_ACCESS_TOKEN",
    vault: str = "",
    env_refs: Optional[Dict[str, str]] = None,
    override_existing: bool = False,
    cache_ttl_seconds: float = 300,
    auto_install: bool = True,
    home_path: Optional[Path] = None,
) -> FetchResult:
    """Pull secrets from Proton Pass and set them on ``os.environ``.

    This is the function ``load_hermes_dotenv()`` calls after the .env files
    have loaded.  It is intentionally defensive — any failure returns a
    :class:`FetchResult` with ``error`` set; it NEVER raises.

    Pass a parsed :class:`ProtonPassConfig` as ``config`` (preferred — the
    env_loader path does this so it no longer re-splats the config's seven
    fields) and it supplies ``service_token_env`` / ``vault`` / ``env_refs`` /
    ``override_existing`` / ``cache_ttl_seconds`` / ``auto_install``.  When
    ``config`` is omitted the individual keyword args are used instead (mirror
    the ``secrets.protonpass.*`` keys), so existing public callers keep working.
    ``enabled`` is always taken from the explicit argument: the env_loader
    registry has already confirmed the source is enabled by the time it calls
    here, and other callers gate enablement themselves.
    """
    result = FetchResult()

    if not enabled:
        return result

    # Resolve the effective config once: an explicit ``config`` wins; otherwise
    # build one from the individual kwargs.  ``has_fetch_target()`` is then the
    # single home for the no-vault-no-refs invariant (shared with fetch.py and
    # the CLI) instead of re-deriving ``not vault and not env_refs`` here.
    if config is None:
        config = ProtonPassConfig(
            enabled=enabled,
            service_token_env=service_token_env,
            vault=vault,
            env_refs=dict(env_refs or {}),
            cache_ttl_seconds=cache_ttl_seconds,
            override_existing=override_existing,
            auto_install=auto_install,
        )
    service_token_env = config.service_token_env
    vault = config.vault
    env_refs = dict(config.env_refs)
    override_existing = config.override_existing
    cache_ttl_seconds = config.cache_ttl_seconds
    auto_install = config.auto_install

    service_token = os.environ.get(service_token_env, "").strip()
    if not service_token:
        result.error = (
            f"secrets.protonpass.enabled is true but {service_token_env} is "
            "not set.  Run `hermes secrets protonpass setup`."
        )
        return result

    # Strip — before the has-fetch-target pre-check — a MODE B ref that targets
    # the bootstrap token's own env var, via the single centralized helper.
    # Fetching it would be wasted work and we'd refuse to apply it anyway (it
    # must never clobber the token we authenticated with).  We still record the
    # skipped name + warning into the FetchResult so it is reported exactly as
    # before; passing the already-filtered refs into ``fetch`` (alongside
    # ``bootstrap_env``) is a no-op there, so there is no duplicate warning.
    env_refs, bootstrap_skipped, bootstrap_warnings = strip_bootstrap_ref(
        env_refs, service_token_env
    )
    result.skipped.extend(bootstrap_skipped)
    result.warnings.extend(bootstrap_warnings)

    # Re-check the invariant through ``has_fetch_target()`` AFTER stripping the
    # bootstrap ref above (which can empty an env-refs-only config), so the
    # single source of truth still decides "nothing to fetch".
    if not ProtonPassConfig(vault=vault, env_refs=env_refs).has_fetch_target():
        result.error = (
            "secrets.protonpass has neither a vault (MODE A) nor env refs "
            "(MODE B).  Run `hermes secrets protonpass setup`."
        )
        return result

    binary = find_pass_cli(install_if_missing=auto_install)
    result.binary_path = binary
    if binary is None:
        result.error = (
            "pass-cli binary not available. Run `hermes secrets protonpass "
            "install` to download the verified pinned version, or leave "
            "`auto_install: true` in the secrets.protonpass config so Hermes "
            "downloads it automatically."
        )
        return result

    try:
        secrets, warnings = fetch_protonpass_secrets(
            service_token=service_token,
            vault=vault,
            env_refs=env_refs,
            binary=binary,
            cache_ttl_seconds=cache_ttl_seconds,
            auto_install=auto_install,
            home_path=home_path,
            bootstrap_env=service_token_env,
        )
    except Exception as exc:  # noqa: BLE001 — apply_* must NEVER raise
        # Honor the documented "never raises" contract at the API boundary: a
        # non-RuntimeError (e.g. an unexpected bug in fetch/session/install)
        # must still degrade to a FetchResult with ``error`` set rather than
        # crashing startup.  _redact_token already scrubs subprocess output;
        # scrub again here in case a message ever interpolates the token.
        result.error = _redact_token(str(exc), service_token)
        return result

    result.secrets = secrets
    result.warnings.extend(warnings)

    # Single planner decides applied vs skipped (+ reason) for every secret.
    plan = plan_application(
        secrets,
        os.environ,
        override_existing=override_existing,
        token_env=service_token_env,
    )
    for item in plan:
        if item.applied:
            os.environ[item.name] = item.value
            result.applied.append(item.name)
        else:
            result.skipped.append(item.name)

    return result
