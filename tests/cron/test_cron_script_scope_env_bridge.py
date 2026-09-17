"""Profile-scoped env bridge for cron script children (no_agent + pre-run scripts).

Gateway-dispatched restart-safe cron workers install the firing profile's
secret scope as a ContextVar (``cron.scheduler._run_external_worker_payload``)
but never in ``os.environ``: under multiplex + home override
``load_hermes_dotenv`` deliberately skips the process-global dotenv install
(anti-cross-profile-leak guard), so ``build_subprocess_env(base=None)`` — an
``os.environ`` snapshot — passed the profile's credentials nowhere and every
env-reading no_agent script failed deterministically whenever the gateway
claimed the fire (the standalone ticker's process env carried the keys, so
ticker fires kept succeeding — the job flapped between runners).

Contracts proven here (real child processes, temp HERMES_HOME):

1. Behavior: the external-worker shape (multiplex + home override + scope) →
   the script child sees the scope credential.
2. Isolation invariant: a SIBLING profile's ambient ``os.environ`` value for a
   scope-defined credential name never reaches the child while the installed
   scope's value does (``get_secret`` precedence: scope beats ambient env).
3. Precedence: names the scope lacks flow from the ambient env as before;
   global env names are never bridged from a scope entry.
4. Single-profile parity: scopeless runs see the ambient env exactly as the
   plain sanitizer produced; a scope fills gaps without new failures.
5. The pre-run script path (agent-backed jobs) shares the bridge.

Probe names avoid the conftest credential-shape redactor and the child-env
scrub lists on purpose: these tests pin the BRIDGE plumbing, not the scrub
policy (which has its own tests, e.g. test_cron_script.py
``test_script_subprocess_env_sanitized``).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

# Deliberately NOT credential-shaped (no *_KEY/*_TOKEN/*_SECRET suffix): the
# sanitizer must keep forwarding these names, and output redaction must leave
# the probe lines readable.
_SCOPE_NAME = "SCRIPT_BRIDGE_PROBE_VAR"
_SIBLING_NAME = "SCRIPT_BRIDGE_SIBLING_VAR"


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with the scripts/cron layout; returns (home, scripts_dir)."""
    home = tmp_path / "profile"
    (home / "scripts").mkdir(parents=True)
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv(_SCOPE_NAME, raising=False)
    monkeypatch.delenv(_SIBLING_NAME, raising=False)
    return home, home / "scripts"


@pytest.fixture(autouse=True)
def _reset_home_override():
    """``set_hermes_home_override`` is a ContextVar with no fixture support;
    leaking it across tests would route unrelated code to a dead tmp home."""
    from hermes_constants import get_hermes_home_override, set_hermes_home_override

    before = get_hermes_home_override()
    yield
    if get_hermes_home_override() is not before:
        set_hermes_home_override(before)


def _external_worker_shape(monkeypatch, home: Path, dotenv: str):
    """Install the exact per-fire context ``_run_external_worker_payload`` builds:
    home override + multiplex flag + the profile's secret scope."""
    from agent.secret_scope import (
        build_profile_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )
    from hermes_constants import set_hermes_home_override

    (home / ".env").write_text(dotenv, encoding="utf-8")
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    set_hermes_home_override(str(home))
    return set_secret_scope(build_profile_secret_scope(home))


def _write_probe(scripts_dir: Path, name: str) -> None:
    (scripts_dir / name).write_text(
        textwrap.dedent(
            f"""\
            import os
            for key in ({_SCOPE_NAME!r}, {_SIBLING_NAME!r}):
                print(f"{{key}}={{os.environ.get(key) or 'ABSENT'}}")
            """
        ),
        encoding="utf-8",
    )


def _read_probe(output: str) -> dict[str, str]:
    values = {}
    for line in output.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


# ---------------------------------------------------------------------------
# 1. Behavior: the reported failure now passes
# ---------------------------------------------------------------------------


def test_external_worker_no_agent_script_reads_scope_credential(cron_env, monkeypatch):
    """The reported failure, end to end: multiplex external worker + no_agent
    script reading a profile-scope credential → the REAL child process sees it."""
    from agent.secret_scope import reset_secret_scope
    from cron.scheduler_script import _run_job_script

    home, scripts_dir = cron_env
    token = _external_worker_shape(
        monkeypatch, home, dotenv=f"{_SCOPE_NAME}=scope-value\n")

    try:
        _write_probe(scripts_dir, "scope_reader.py")
        ok, output = _run_job_script("scope_reader.py")
    finally:
        reset_secret_scope(token)

    assert ok is True, output
    assert _read_probe(output)[_SCOPE_NAME] == "scope-value"


# ---------------------------------------------------------------------------
# 2. Isolation invariant: sibling ambient value never wins a scoped name
# ---------------------------------------------------------------------------


def test_scope_value_reaches_child_while_sibling_ambient_value_does_not(
    cron_env, monkeypatch,
):
    """The multiplex leak-guard invariant: the installed scope's value reaches
    the child, and a sibling profile's ambient os.environ value for the SAME
    name cannot — get_secret's precedence (scope beats ambient env), which is
    what stops a shared launch env from leaking into a routed profile's child."""
    from agent.secret_scope import reset_secret_scope
    from cron.scheduler_script import _run_job_script

    home, scripts_dir = cron_env
    # Sibling profile's ambient process env (what a shared launch env carries).
    monkeypatch.setenv(_SCOPE_NAME, "sibling-ambient-value")
    token = _external_worker_shape(
        monkeypatch, home, dotenv=f"{_SCOPE_NAME}=scope-value\n")

    try:
        _write_probe(scripts_dir, "isolation_probe.py")
        ok, output = _run_job_script("isolation_probe.py")
    finally:
        reset_secret_scope(token)

    assert ok is True, output
    assert _read_probe(output)[_SCOPE_NAME] == "scope-value"


def test_scopeless_multiplex_run_keeps_ambient_flow(cron_env, monkeypatch):
    """Names the scope lacks flow from the ambient env exactly as before — the
    bridge only ever overlays what the firing profile's scope defines."""
    from agent.secret_scope import reset_secret_scope
    from cron.scheduler_script import _run_job_script

    _, scripts_dir = cron_env
    monkeypatch.setenv(_SIBLING_NAME, "ambient-value")
    # Scope installs WITHOUT the sibling name.
    token = _external_worker_shape(monkeypatch, scripts_dir.parent, dotenv="")

    try:
        _write_probe(scripts_dir, "ambient_probe.py")
        ok, output = _run_job_script("ambient_probe.py")
    finally:
        reset_secret_scope(token)

    assert ok is True, output
    assert _read_probe(output)[_SIBLING_NAME] == "ambient-value"


def test_global_env_names_never_bridge_from_scope(cron_env, monkeypatch):
    """Process-global settings (HERMES_HOME etc.) are not profile secrets: the
    bridge must never overlay them from a scope entry."""
    from agent.secret_scope import reset_secret_scope
    from cron.scheduler_script import _run_job_script

    home, scripts_dir = cron_env
    (scripts_dir / "global_probe.py").write_text(
        "import os\n"
        "print(f\"HERMES_HOME={os.environ.get('HERMES_HOME', '')}\")\n",
        encoding="utf-8",
    )
    token = _external_worker_shape(
        monkeypatch, home,
        dotenv=f"{_SCOPE_NAME}=scope-value\nHERMES_HOME=/evil/scope-home\n")

    try:
        ok, output = _run_job_script("global_probe.py")
    finally:
        reset_secret_scope(token)

    assert ok is True, output
    assert _read_probe(output)["HERMES_HOME"] == str(home), (
        "a scope entry for a global env name must never mask the process value")


# ---------------------------------------------------------------------------
# 3. Single-profile (ticker) path parity
# ---------------------------------------------------------------------------


def test_scopeless_run_env_has_no_bridge(cron_env):
    """No scope installed → the bridge contributes nothing; the child env is
    exactly what the plain sanitizer produced before the bridge existed."""
    from cron.scheduler_script import _profile_scope_env

    assert _profile_scope_env() == {}


# ---------------------------------------------------------------------------
# 3. Single-profile (ticker) path parity
# ---------------------------------------------------------------------------


def test_non_multiplex_scope_does_not_change_child_env(cron_env, monkeypatch):
    """Standalone ticker shape: process env IS the profile's env, so a scope
    present on the run must NOT alter the child (process env > profile .env) —
    the same branch ``cron_env_setting`` takes outside multiplex."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from cron.scheduler_script import _profile_scope_env, _run_job_script

    home, scripts_dir = cron_env
    (home / ".env").write_text(f"{_SCOPE_NAME}=scope-value\n", encoding="utf-8")
    token = set_secret_scope(build_profile_secret_scope(home))

    assert _profile_scope_env() == {}

    (scripts_dir / "unbridged_probe.py").write_text(
        f"import os\nprint(os.environ.get({_SCOPE_NAME!r}) or 'ABSENT')\n",
        encoding="utf-8",
    )
    try:
        ok, output = _run_job_script("unbridged_probe.py")
    finally:
        reset_secret_scope(token)

    assert ok is True, output
    assert output.strip() == "ABSENT"


def test_multiplex_without_scope_falls_back_to_profile_dotenv(cron_env, monkeypatch):
    """The tick-loop shape (home override, no scope installed yet) resolves the
    overlay from the active profile's .env — the same fallback
    ``cron_env_setting`` uses, so a fire can never fall back to a launch
    profile's ambient value."""
    from cron.scheduler_script import _profile_scope_env, _run_job_script
    from hermes_constants import set_hermes_home_override

    home, scripts_dir = cron_env
    (home / ".env").write_text(f"{_SCOPE_NAME}=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    set_hermes_home_override(str(home))

    assert _profile_scope_env() == {_SCOPE_NAME: "from-dotenv"}

    (scripts_dir / "dotenv_probe.py").write_text(
        f"import os\nprint(os.environ.get({_SCOPE_NAME!r}) or 'ABSENT')\n",
        encoding="utf-8",
    )
    ok, output = _run_job_script("dotenv_probe.py")

    assert ok is True, output
    assert output.strip() == "from-dotenv"


def test_bridge_only_carries_usable_profile_names(cron_env, monkeypatch):
    """The overlay shape under multiplex: non-str pairs are skipped instead of
    crashing a fire, and global env names are never bridged from a scope entry."""
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from cron.scheduler_script import _profile_scope_env

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    assert _profile_scope_env() == {}  # no scope, empty .env → nothing to bridge

    token = set_secret_scope({
        _SCOPE_NAME: "ok",
        "HERMES_HOME": "/evil/scope-home",
        "PATH": "/evil/bin",
        7: "bad",
        "NULLVAL": None,
        "": "empty-name",
    })  # type: ignore[dict-item]
    try:
        bridge = _profile_scope_env()
    finally:
        reset_secret_scope(token)

    assert bridge == {_SCOPE_NAME: "ok"}

    empty_token = set_secret_scope({})
    try:
        assert _profile_scope_env() == {}
    finally:
        reset_secret_scope(empty_token)


# ---------------------------------------------------------------------------
# 4. Pre-run script path shares the bridge (agent-backed script jobs)
# ---------------------------------------------------------------------------


def test_pre_run_script_path_reads_scope_credential(cron_env, monkeypatch):
    """``_build_job_prompt``'s pre-run script path goes through the same
    ``_run_job_script``, so agent-backed script jobs get the bridge too."""
    from cron.scheduler_prompt import _build_job_prompt

    home, scripts_dir = cron_env
    token = _external_worker_shape(
        monkeypatch, home, dotenv=f"{_SCOPE_NAME}=scope-value\n")

    (scripts_dir / "keyed.py").write_text(
        "import os\n"
        f"key = os.environ.get({_SCOPE_NAME!r}) or 'MISSING'\n"
        "print('probe=' + key)\n",
        encoding="utf-8",
    )
    try:
        prompt = _build_job_prompt({"prompt": "Check.", "script": "keyed.py"})
    finally:
        from agent.secret_scope import reset_secret_scope
        reset_secret_scope(token)

    assert "## Script Output" in prompt
    assert "probe=scope-value" in prompt
