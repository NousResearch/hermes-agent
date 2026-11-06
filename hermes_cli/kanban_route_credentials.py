"""Q3 of the dispatch preflight: can the assignee profile serve the provider this card pins?

A card can carry the router's pick as ``model_override``/``provider_override``. Dispatch admits
it, ``gateway.session_kanban.build_kanban_policy`` turns the pin into the worker's ``model`` and
``provider``, and the worker then resolves credentials from **its own profile home**. When that
home holds no credential for the pinned provider the worker does not refuse: it dies inside
``agent/agent_init.py::_routed_client_kwargs`` with ``ProviderNotConfiguredError: No LLM provider
configured``, before a single provider call. Card t_59149063 burned three admissions that way
(runs 721/722/723, zero ``session_model_usage`` rows) and blocked.

So the pin is judged before the claim, in the environment the worker will actually get: a
short-lived child process with the assignee's ``HERMES_HOME`` and the same scrubbed env
``_default_spawn`` builds. The child answers only what a spawn would reveal — whether the home
holds credential material for that provider — and never constructs a client: no network, no OAuth
refresh, no single-use grant rotation, no pool write. A gate must not spend what it is measuring.

**Decidable** providers are exactly two shapes, each mirrored from the resolver its worker really
runs:

* ``openrouter``: the routed client path (``agent/agent_init.py::_routed_client_kwargs`` →
  ``resolve_provider_client(raw_codex=True)`` → ``auxiliary_client._try_openrouter``) reads a
  credential-pool entry with a runtime key, then ``_scoped_key_env("OPENROUTER_API_KEY")`` — and
  nothing else. No ``model.key_env`` pointer, no ``OPENAI_API_KEY`` home, no prefix validation:
  any non-empty value is handed to the client.
* every other registry provider declared ``auth_type: api_key``: the ladder
  ``auth._resolve_api_key_provider_secret`` reads — ``model.key_env`` (Desktop-saved pointer) → the
  registry's ``api_key_env_vars`` (rotation siblings included) → the credential pool — with the
  ``has_usable_secret`` and key-prefix rules runtime applies, so a malformed key in ``.env`` is not
  mistaken for a credential. ``model.api_key`` in ``config.yaml`` is **not** a source: the registry
  ``api_key`` runtime never reads it (only an Azure-base-url anthropic route does).
**Everything else is undecidable here and admitted**: OAuth and external-process providers resolve
a grant from the global-root pool or the subprocess's own login store, anthropic also reads the
Claude Code / Hermes OAuth store, the local no-auth providers (``lmstudio``, ``actual``) are served
a placeholder key by design, and local/custom endpoints may need no key at all. A refusal has to be
provable, so an undecidable shape is never parked.

A pool that HAS credentials but cannot serve right now (every entry cooling, quota exhausted) is a
provider-health question rather than a credential gap, and is deliberately admitted: that failure
already names itself as ``ProviderCredentialsExhaustedError``, and t_376d9390 owns the policy for
it. This gate refuses only the state that crashes a worker with no diagnosis.

The probe failing (missing interpreter, timeout, unreadable profile) is NOT evidence that the
profile cannot serve: it fails open and logs, exactly like Q2's unreadable rule. Nothing here
changes which provider a card runs on — a pin that cannot be served is parked with a reason, never
silently re-pointed at the profile default.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

_log = logging.getLogger(__name__)

ROUTE_REFUSAL_MARKER = "route credential refusal"

#: Q3's label in a ``routing_refused`` event, beside the fleet rule's Q1/Q2.
QUESTION = "Q3"

#: Cold import of ``agent.credential_pool`` on this host is ~2 s; a wedged child is not a verdict.
DEFAULT_PROBE_TIMEOUT_SECONDS = 45

_OPENROUTER_ENV_VARS = ("OPENROUTER_API_KEY",)
_OPENROUTER_ENV_URL_VARS: tuple[str, ...] = ()

#: Local servers the runtime serves with a placeholder key and no credential at all
#: (``resolve_api_key_provider_credentials``): ``lmstudio`` always, ``actual`` when its configured
#: base URL is loopback. Nothing to refuse, and ``actual``'s loopback test needs a URL this probe
#: does not resolve, so both are non-decisive.
_NO_AUTH_LOCAL_PROVIDERS = frozenset({"lmstudio", "actual"})

#: Providers whose token also resolves from a login store outside the env/pool ladder:
#: ``agent.anthropic_credentials.resolve_anthropic_token`` also reads the Claude Code keychain/file
#: and the Hermes OAuth file, so "nothing in ``.env`` or ``auth.json``" is not proof of a crash.
_EXTERNAL_TOKEN_STORE_PROVIDERS = frozenset({"anthropic"})


@dataclass(frozen=True)
class RouteProbe:
    """What the assignee's own home holds for one pinned provider.

    ``present is None`` means the question could not be decided for this provider shape (or the
    probe itself failed, in ``detail``): dispatch admits, because an unmeasurable environment is
    not evidence of a missing credential.
    """

    profile: str
    provider: str
    kind: str
    present: Optional[bool]
    env_vars: tuple[str, ...] = ()
    detail: str = ""

    @property
    def decisive(self) -> bool:
        return self.kind == "api_key" and self.present is not None


@dataclass(frozen=True)
class RouteCredentialRefusal:
    """A pinned route the assignee profile cannot serve, with the provider named."""

    profile: str
    provider: str
    model: Optional[str]
    env_vars: tuple[str, ...] = ()
    question: str = QUESTION

    def as_text(self) -> str:
        target = f"provider '{self.provider}'"
        if self.model:
            target += f" (model '{self.model}')"
        remedy = (
            f"Set {self.env_vars[0]} in that profile's .env"
            if self.env_vars else "Add a credential for it on that profile"
        )
        return (
            f"{ROUTE_REFUSAL_MARKER}: profile '{self.profile}' is pinned to {target} but its home "
            f"holds no {self.provider} credential, so the worker would die while its agent is "
            f"constructed (ProviderNotConfiguredError) before the first provider call. {remedy}, "
            f"reassign the card to a profile that serves {self.provider}, or clear the pin."
        )

    def as_event(self) -> dict:
        return {
            "question": self.question,
            "gate": "route_credential",
            "profile": self.profile,
            "provider": self.provider,
            "model": self.model,
            "env_vars": list(self.env_vars),
            "reason": "no-credential-in-profile-home",
        }

    def park(self, conn, task_id: str) -> None:
        """Record the refusal and park the card ``blocked``/``capability``.

        A credential gap is a capability wall — install the key or move the card — so the reason
        must survive on the board, not just in this tick's return value.
        """
        from hermes_cli import kanban_db as kb

        try:
            with kb.write_txn(conn):
                kb._append_event(conn, task_id, "routing_refused", self.as_event())
        except Exception:  # the block below is the part that must not be skipped
            _log.debug("kanban dispatch: routing_refused event failed for %s", task_id, exc_info=True)
        try:
            kb.block_task(conn, task_id, reason=f"{self.as_text()} [card {task_id}]",
                          kind="capability")
        except Exception:
            _log.exception("kanban dispatch: could not park route-refused card %s", task_id)


def provider_shape(provider: Optional[str]) -> tuple[str, tuple[str, ...], str]:
    """``(kind, api_key_env_vars, why)`` for a pinned provider id.

    ``kind == "api_key"`` only where "no usable credential material" is a *complete* explanation
    for the worker's death: OpenRouter (whose routed client reads a pool entry and its own env var,
    nothing else) and registry providers declared ``auth_type: api_key`` whose ladder is
    ``model.key_env`` → registry env vars → pool. A registry row is not enough on
    its own — ``anthropic`` is declared ``api_key`` but also resolves the Claude Code / Hermes OAuth
    token, and ``lmstudio``/``actual`` are served a no-auth placeholder. Those, provider-owned
    secret resolution, OAuth, external-process and custom routes all answer ``"undecidable"`` with
    the reason, so a refusal can never be invented from a provider whose credentials live somewhere
    this probe cannot see.
    """
    key = (provider or "").strip().lower()
    if not key or key == "auto":
        return "undecidable", (), "no provider is pinned, so the profile's own route applies"
    if key == "openrouter":
        return "api_key", _OPENROUTER_ENV_VARS, ""
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY.get(key)
    except Exception as exc:  # a registry that cannot load is not a verdict
        return "undecidable", (), f"provider registry unreadable: {type(exc).__name__}"
    if pconfig is None:
        return "undecidable", (), "not a registered provider (custom or plugin route)"
    auth_type = str(getattr(pconfig, "auth_type", "") or "")
    if auth_type != "api_key":
        return "undecidable", (), f"{key} is auth_type={auth_type or 'unknown'}"
    if key in _NO_AUTH_LOCAL_PROVIDERS:
        return "undecidable", (), f"{key} is a local no-auth provider (runtime supplies a placeholder key)"
    if key in _EXTERNAL_TOKEN_STORE_PROVIDERS:
        return "undecidable", (), (f"{key} also resolves a token from a login store outside the "
                                   "env/pool ladder, which this probe does not read")
    try:
        from providers import get_provider_profile
        from providers.base import ProviderProfile
        profile = get_provider_profile(key)
        owned_resolution = (profile is not None
                            and type(profile).resolve_api_key_secret is not ProviderProfile.resolve_api_key_secret)
    except Exception:  # a profile package that cannot load is not a verdict
        owned_resolution = False
    if owned_resolution:
        return "undecidable", (), f"{key} resolves its own secret outside the env/pool ladder"
    return "api_key", tuple(getattr(pconfig, "api_key_env_vars", ()) or ()), ""


def _local_endpoint_bypass() -> str:
    """Why the runtime may skip the credential ladder for this home, or ``""``.

    ``runtime_provider._local_endpoint_bypass`` short-circuits the ladder for an auto/unset
    ``model.provider`` with a non-cloud ``model.base_url``, so a home in that shape can serve a pin
    with no provider credential at all. Not producible from the env/pool files, so non-decisive.
    """
    try:
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider import _LOCAL_BYPASS_CLOUD_HOSTS
        from utils import base_url_host_matches

        model_cfg = (load_config() or {}).get("model")
        if not isinstance(model_cfg, dict):
            return ""
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        base_url = str(model_cfg.get("base_url") or "").strip()
        if (base_url and cfg_provider in ("", "auto")
                and not any(base_url_host_matches(base_url, host) for host in _LOCAL_BYPASS_CLOUD_HOSTS)):
            return ("model.provider is auto with a local base_url, so the runtime may resolve "
                    "without a provider credential")
    except Exception:
        return ""
    return ""


def _openrouter_sources() -> list[str]:
    """Mirror ``auxiliary_client._try_openrouter`` read-only — the resolver the routed worker runs.

    ``_routed_client_kwargs`` calls ``resolve_provider_client(provider, model=…, raw_codex=True)``
    with no explicit key, and for ``openrouter`` that is ``_try_openrouter``, whose entire
    credential list is: a credential-pool entry with a runtime API key, then
    ``_scoped_key_env("OPENROUTER_API_KEY")``. It does **not** read the ``model.key_env`` pointer,
    the legacy ``OPENAI_API_KEY`` home, or the registry prefix rules — any non-empty value is
    handed to the OpenRouter client. Counting those extra rungs here admitted homes whose worker
    then died at init (or silently took the init fallback chain, breaking the pin), so this rung
    mirrors the worker exactly: present means the worker gets a key, not that some other resolver
    would.

    Read-only: ``read_credential_pool`` never seeds, heals or persists (``load_pool`` would ingest
    an env key into the pool as a side effect of merely asking). An ``env:<VAR>`` row is persisted
    WITHOUT its secret and re-hydrated from that var on every load, so the row alone is not a
    credential. A pool whose entries are all cooling still counts as present here — that is
    provider health (t_376d9390), not a credential gap.
    """
    from hermes_cli.auth import read_credential_pool
    from hermes_cli.config import get_env_value_prefer_dotenv

    sources: list[str] = []
    if (get_env_value_prefer_dotenv("OPENROUTER_API_KEY") or "").strip():
        sources.append("env:OPENROUTER_API_KEY")
    for row in read_credential_pool("openrouter") or []:
        if not isinstance(row, Mapping):
            continue
        value = str(row.get("access_token") or row.get("runtime_api_key") or "").strip()
        if value:
            sources.append("auth.json")
            break
        source = str(row.get("source") or "")
        if not source.startswith("env:"):
            continue
        env_name = source.split(":", 1)[1].strip()
        if env_name and (get_env_value_prefer_dotenv(env_name) or "").strip():
            sources.append(f"auth.json:{env_name}")
            break
    return sources


def _usable_sources(provider: str, env_vars: Sequence[str]) -> list[str]:
    """Mirror the worker's own credential resolution read-only for a registry ``api_key`` provider.

    ``auth._resolve_api_key_provider_secret``: ``model.key_env`` → registry env vars → pool, with
    the usable-secret / declared-prefix rules. ``openrouter`` does NOT use this ladder — its routed
    client has its own resolver — so it takes :func:`_openrouter_sources`.
    """
    if provider == "openrouter":
        return _openrouter_sources()

    from hermes_cli.auth import _model_level_key_env, _usable_declared_secret, read_credential_pool
    from hermes_cli.config import get_env_value_prefer_dotenv

    sources: list[str] = []
    # Desktop-saved pointer (#106336): consulted only when config.yaml's model targets this
    # provider, exactly as ``_model_level_key_env`` decides it for the runtime.
    key_env = _model_level_key_env(provider)
    if key_env and _usable_declared_secret(provider, get_env_value_prefer_dotenv(key_env), key_env):
        sources.append(f"env:{key_env}")
    # ``_seed_from_env`` also backs a rotation pool with ``<VAR>_2``, ``<VAR>_3``, ... — the run
    # stops at the first sibling the home does not define.
    candidates = list(env_vars)
    for base in env_vars:
        n = 2
        while True:
            name = f"{base}_{n}"
            if not (get_env_value_prefer_dotenv(name) or "").strip():
                break
            candidates.append(name)
            n += 1
    for env_var in candidates:
        if _usable_declared_secret(provider, get_env_value_prefer_dotenv(env_var), env_var):
            sources.append(f"env:{env_var}")
    # ``read_credential_pool`` is a file read; ``load_pool`` would heal/ingest/persist as a side
    # effect of merely asking, which a gate must not do. An ``env:<VAR>`` row is persisted WITHOUT
    # its secret and re-hydrated from that env var on every load (``_seed_from_env``), so the row's
    # presence alone is not a credential.
    for row in read_credential_pool(provider) or []:
        if not isinstance(row, Mapping):
            continue
        value = row.get("access_token") or row.get("runtime_api_key") or ""
        if _usable_declared_secret(provider, value, "auth.json"):
            sources.append("auth.json")
            break
        source = str(row.get("source") or "")
        if not source.startswith("env:"):
            continue
        env_name = source.split(":", 1)[1].strip()
        if env_name and _usable_declared_secret(
                provider, get_env_value_prefer_dotenv(env_name), source):
            sources.append(f"auth.json:{env_name}")
            break
    return sources


def probe_payload(provider: str) -> dict:
    """Child side: what this process's home holds for *provider*. Prints no credential.

    Read-only on purpose (``read_credential_pool`` never seeds or persists, ``load_pool`` would
    ingest an env key into a paid pool as a side effect of merely asking). ``present is None`` is a
    non-decisive answer — the caller admits the card rather than inventing a refusal.
    """
    key = (provider or "").strip().lower()
    kind, env_vars, why = provider_shape(key)
    payload = {"provider": key, "kind": kind, "present": None, "env_vars": list(env_vars), "why": why,
               "sources": []}
    if kind != "api_key":
        return payload
    why = _local_endpoint_bypass()
    if why:
        payload["why"] = why
        return payload
    try:
        sources = _usable_sources(key, env_vars)
    except Exception as exc:
        payload["why"] = f"credential ladder unreadable: {type(exc).__name__}: {exc}"
        return payload
    payload["present"] = bool(sources)
    payload["sources"] = sources
    return payload


def _install_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _probe_env(profile_home: str) -> dict[str, str]:
    """The env a worker for *profile_home* would get, as far as credentials are concerned.

    The dispatcher's own process env is the wrong answer — a launched gateway holds the profile it
    was started for, so an unscrubbed read answers for the wrong profile (the reason
    ``_default_spawn`` scrubs the child it spawns). Handing the child the parent env with the
    *launch* profile's ``.env`` keys stripped, plus the assignee's ``HERMES_HOME``, leaves it
    reading that home's files: keys this profile owns survive, another profile's never cross.
    Session routing and board bindings are dropped outright — a probe is not a turn.
    """
    from gateway.session_context import _VAR_MAP
    from tools.environments.local import strip_launch_profile_env

    env = dict(os.environ)
    for key in (*_VAR_MAP, *(name for name in env if name.startswith("HERMES_KANBAN_"))):
        env.pop(key, None)
    env["HERMES_HOME"] = profile_home
    strip_launch_profile_env(env, profile_home)
    # The probe imports hermes directly, never through ``hermes``'s launcher: skip its
    # dependency-consistency relaunch so a probe cannot restart a source update.
    env["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    root = str(_install_root())
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH", "")) if p)
    return env


def _run_child(provider: str, profile_home: str, timeout: int) -> dict:
    command = [sys.executable, "-m", "hermes_cli.kanban_route_credentials", "--probe", provider]
    completed = subprocess.run(
        command, cwd=str(_install_root()), env=_probe_env(profile_home),
        capture_output=True, text=True, timeout=timeout, check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"probe exited {completed.returncode}: {(completed.stderr or '').strip()[-300:]}")
    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("probe printed nothing")
    return json.loads(lines[-1])


def probe(
    profile: str,
    provider: str,
    *,
    timeout: int = DEFAULT_PROBE_TIMEOUT_SECONDS,
    runner: Optional[Callable[[str, str, int], Mapping]] = None,
) -> RouteProbe:
    """What *profile*'s own home holds for *provider*: a verdict, or an undecidable one."""
    key = (provider or "").strip().lower()
    kind, env_vars, why = provider_shape(key)
    if kind != "api_key":
        return RouteProbe(profile, key, kind, None, env_vars, why)
    from hermes_cli.profiles import resolve_profile_env

    try:
        profile_home = resolve_profile_env(profile)
    except Exception as exc:
        return RouteProbe(profile, key, "error", None, env_vars,
                          f"profile home unresolvable: {type(exc).__name__}: {exc}")
    run = runner or (lambda name, home, limit: _run_child(name, home, limit))
    try:
        payload = run(key, profile_home, timeout)
    except Exception as exc:
        return RouteProbe(profile, key, "error", None, env_vars,
                          f"probe failed: {type(exc).__name__}: {exc}")
    if not isinstance(payload, Mapping) or "present" not in payload:
        return RouteProbe(profile, key, "error", None, env_vars, "probe returned no verdict")
    present = payload.get("present")
    if present is not None and not isinstance(present, bool):
        return RouteProbe(profile, key, "error", None, env_vars, "probe verdict was not boolean")
    return RouteProbe(
        profile, key, str(payload.get("kind") or kind), present,
        tuple(payload.get("env_vars") or env_vars), str(payload.get("why") or ""),
    )


def route_credential_refusal(
    profile: str,
    provider: Optional[str],
    model: Optional[str] = None,
    *,
    timeout: int = DEFAULT_PROBE_TIMEOUT_SECONDS,
    probe_fn: Optional[Callable[..., RouteProbe]] = None,
) -> Optional[RouteCredentialRefusal]:
    """Q3 for one dispatch candidate: the refusal, or ``None``.

    Refuses only a *decisive* verdict that the home holds nothing. An undecidable provider shape
    and a failed probe both return ``None`` — the card is admitted and the worker's own guard
    remains the backstop, because a gate that parks cards when its own check breaks is worse than
    the bug it closes.
    """
    pinned = (provider or "").strip()
    if not pinned:
        return None
    try:
        verdict = (probe_fn or probe)(profile, pinned, timeout=timeout)
    except Exception:
        # A checker that breaks must not park a card (Q2's unreadable-rule rule): the worker's
        # own guard is still there, and a refusal invented from a failed probe is worse than the
        # bug this closes.
        _log.warning(
            "kanban dispatch: route credential probe failed for %r on %r; admitting the card",
            provider, profile, exc_info=True,
        )
        return None
    if verdict.kind != "api_key":
        _log.debug(
            "kanban dispatch: route credential check undecided for %s on %r (%s)",
            provider, profile, verdict.detail or "undecidable provider shape",
        )
        return None
    if verdict.present is not False:
        return None
    return RouteCredentialRefusal(
        profile=profile, provider=verdict.provider or (provider or "").strip().lower(),
        model=(model or None), env_vars=verdict.env_vars,
    )


def pinned_route(row) -> tuple[Optional[str], Optional[str]]:
    """``(provider, model)`` pinned on a dispatch row, or ``(None, None)``.

    Missing columns read as unpinned: a lane query that does not select the pin cannot refuse a
    card for it.
    """

    def _get(name: str):
        try:
            return row[name]
        except (IndexError, KeyError, TypeError):
            return None

    provider = _get("provider_override")
    provider = str(provider).strip() if provider else None
    model = _get("model_override")
    model = str(model).strip() if model else None
    return (provider or None), (model or None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Child entry point: ``--probe <provider>``, one JSON verdict on stdout."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m hermes_cli.kanban_route_credentials",
        description="Report what this Hermes home holds for a pinned provider.",
    )
    parser.add_argument("--probe", required=True, metavar="PROVIDER")
    args = parser.parse_args(argv)
    print(json.dumps(probe_payload(args.probe)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
