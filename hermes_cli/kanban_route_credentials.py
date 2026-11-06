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

**Decidable** providers are the ones whose failure mode is exactly this crash: ``openrouter`` and
every registry provider with ``auth_type == "api_key"``. **Everything else is undecidable here and
admitted**: OAuth and external-process providers can resolve a grant from the global-root pool or
from the subprocess's own login store, and local/custom endpoints may need no key at all. A
refusal has to be provable, so an undecidable shape is never parked.

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

    ``kind == "api_key"`` only where "no credential material" is a *complete* explanation for the
    worker's death: OpenRouter (whose only credential is its pool) and registry providers declared
    ``auth_type: api_key``. Every other shape answers ``"undecidable"`` with the reason, so the
    refusal can never be invented from a provider whose credentials live somewhere this probe
    cannot see.
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
    return "api_key", tuple(getattr(pconfig, "api_key_env_vars", ()) or ()), ""


def probe_payload(provider: str) -> dict:
    """Child side: what this process's home holds for *provider*. Prints no credential.

    Read-only on purpose (``read_credential_pool`` never seeds or persists; ``load_pool`` would
    ingest an env key into a paid pool as a side effect of merely asking).
    """
    key = (provider or "").strip().lower()
    kind, env_vars, why = provider_shape(key)
    payload = {"provider": key, "kind": kind, "present": None, "env_vars": list(env_vars), "why": why,
               "sources": []}
    if kind != "api_key":
        return payload
    sources: list[str] = []
    from agent.credential_pool import get_env_prefer_dotenv
    for env_var in env_vars:
        if (get_env_prefer_dotenv(env_var) or "").strip():
            sources.append(f"env:{env_var}")
    try:
        from hermes_cli.auth import read_credential_pool
        if read_credential_pool(key):
            sources.append("auth.json")
    except Exception as exc:
        payload["why"] = f"credential pool unreadable: {type(exc).__name__}: {exc}"
        return payload
    try:
        from hermes_cli.config import load_config
        model_cfg = (load_config() or {}).get("model")
        if isinstance(model_cfg, dict) and str(model_cfg.get("provider") or "").strip().lower() == key:
            if str(model_cfg.get("api_key") or "").strip():
                sources.append("config.yaml")
    except Exception:  # config is one of three sources; its absence is not a verdict
        _log.debug("route credential probe: config unreadable", exc_info=True)
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
