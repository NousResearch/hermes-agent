"""Executable class map for the profile-boundary seam (#82936 map).

One xfail per open member of the class: profile-scoped state (secrets, config,
session identity) resolved from **ambient process state at use time** instead of
being **bound to the owning profile/session at creation time**.

Each test asserts the *correct* profile-bound behavior and currently fails on
main, so it is marked ``xfail(strict=False)``. When a member's fix lands, its
test flips to XPASS and the seam has a live scoreboard — same format as the
delegation-seam map in ``tests/tools/test_delegation_fallback_class.py``.

Members and in-flight fixes are tracked on the class table in #82936. The
junction member (default-profile secrets reaching child processes) is
cross-listed in the credential-inheritance EPIC #83565 — the two classes
compose: that one is parent→child env flow, this one is ambient-vs-bound
profile resolution.

This file fixes nothing by itself and competes with no open PR.
"""

import os
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


# ---------------------------------------------------------------------------
# Member 1 — #82936: multiplexed terminal env includes another profile's
# undeclared secrets (junction member, cross-listed in EPIC #83565).
#
# Root cause under test: tools/environments/local.py::_make_run_env builds
# ``merged = dict(os.environ | env)`` and copies any key that is neither a
# Hermes-internal secret nor blocklisted. An *undeclared* variable — one the
# active profile neither defines in its own .env nor lists under
# terminal.env_passthrough — is copied verbatim from os.environ, which in a
# multiplexed gateway holds the default profile's .env loaded at startup.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="#82936: _make_run_env copies undeclared os.environ vars into "
    "secondary-profile subprocess env under multiplexing",
)
def test_multiplex_terminal_env_excludes_other_profiles_undeclared_secret(
    monkeypatch,
):
    """A secondary profile's run env must not include a secret that only the
    default profile defines, when the secondary neither defines nor declares
    it for passthrough."""
    from agent.secret_scope import (
        set_multiplex_active,
        set_secret_scope,
        reset_secret_scope,
    )
    from tools.environments.local import _make_run_env

    # Default profile's .env was loaded into the process environment at
    # gateway startup (the reported deployment shape).
    monkeypatch.setenv("SERVICE_PASSWORD", "default-profile-secret")

    set_multiplex_active(True)
    # Secondary profile's secret scope: does NOT define SERVICE_PASSWORD.
    token = set_secret_scope({"SECONDARY_ONLY_KEY": "b-value"})
    try:
        run_env = _make_run_env({})
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)

    assert "SERVICE_PASSWORD" not in run_env, (
        "secondary profile's terminal subprocess sees the default profile's "
        "undeclared secret straight from os.environ"
    )


# ---------------------------------------------------------------------------
# Member 2 — #81952: corrupt profile config.yaml silently falls back to
# DEFAULT_CONFIG (and from there to a paid default model) instead of failing
# loudly. The silent fallback is documented behavior today —
# hermes_cli/config.py::_backup_corrupt_config's own docstring: "When the YAML
# can't be parsed, load_config() silently falls back to DEFAULT_CONFIG."
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="#81952: load_config() silently returns DEFAULT_CONFIG when the "
    "profile's config.yaml is corrupt",
)
def test_corrupt_profile_config_fails_loudly(tmp_path):
    """Loading a syntactically corrupt config.yaml must surface an error to
    the caller (raise, or an explicit error channel) — not silently hand back
    defaults that reroute the profile onto a paid default model."""
    import hermes_cli.config as config_mod

    profile_home = tmp_path / "profiles" / "b"
    profile_home.mkdir(parents=True)
    corrupt = profile_home / "config.yaml"
    # Unclosed flow mapping + tab indentation: reliably unparseable YAML.
    corrupt.write_text("model: {default: [unclosed\n\tbad: :::\n", encoding="utf-8")

    token = set_hermes_home_override(str(profile_home))
    try:
        with pytest.raises(Exception):
            config_mod.load_config()
    finally:
        reset_hermes_home_override(token)


# ---------------------------------------------------------------------------
# Member 3 — #83346: gateway session-key fallback resolves the profile from
# ambient get_active_profile_name() at call time when the event source does
# not carry one (gateway/run.py fallback around build_session_key). The
# pending-clarify entry for a named profile is stored under agent:<profile>,
# so an ambient resolution to a *different* active profile misroutes clarify
# replies through the legacy/busy-guard key.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="#83346: session-key fallback resolves profile from ambient "
    "active-profile state instead of the session's owning profile",
)
def test_session_key_profile_binding_not_ambient(monkeypatch, tmp_path):
    """Two key computations for the SAME sourced event must agree regardless
    of which profile happens to be ambiently active at call time."""
    from gateway.session import SessionSource, build_session_key
    from gateway.platforms.base import Platform

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="u1",
        profile=None,  # event does not carry a profile — the fallback case
    )

    import hermes_cli.profiles as profiles_mod

    monkeypatch.setattr(
        profiles_mod, "get_active_profile_name", lambda: "profile-a"
    )
    key_under_a = build_session_key(source, profile=_ambient_profile(profiles_mod))

    monkeypatch.setattr(
        profiles_mod, "get_active_profile_name", lambda: "profile-b"
    )
    key_under_b = build_session_key(source, profile=_ambient_profile(profiles_mod))

    assert key_under_a == key_under_b, (
        "the same event resolves to different session keys depending on the "
        "ambient active profile at call time"
    )


def _ambient_profile(profiles_mod):
    """Mirror the gateway fallback's ambient resolution (gateway/run.py)."""
    try:
        return profiles_mod.get_active_profile_name() or "default"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Member 4 — #80318: kanban workers run under a profile-scoped HERMES_HOME,
# which hides root-config MoA presets. resolve_moa_preset against the profile
# config either silently collapses to the code-default preset (older builds)
# or raises MoAPresetNotFoundError (current main) — either way the worker
# cannot run the preset the user configured in the ROOT config.yaml, while
# the same preset resolves fine in non-worker sessions.
#
# Note: #82117 (serve .env leak) is NOT in this file: the functions its
# report cites (_profile_env_value / _profile_scope) live in the desktop
# app's repo, not here — tracked on the #82936 table only.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="#80318: profile-scoped HERMES_HOME hides root-config MoA presets "
    "from kanban workers",
)
def test_profile_scope_resolves_root_moa_preset(tmp_path):
    """A preset authored in the root config must resolve inside a
    profile-scoped load (the kanban-worker shape: HERMES_HOME points at
    profiles/<name>, whose config.yaml has no ``moa:`` block)."""
    import yaml

    import hermes_cli.config as config_mod
    from hermes_cli.moa_config import resolve_moa_preset

    root_home = tmp_path
    preset = {
        "reference_models": [
            {"provider": "opencode-go", "model": "deepseek-v4-flash"},
            {"provider": "opencode-go", "model": "mimo-v2.5"},
        ],
        "aggregator": {"provider": "opencode-go", "model": "deepseek-v4-flash"},
    }
    (root_home / "config.yaml").write_text(
        yaml.safe_dump({"moa": {"presets": {"Speed Quality": preset}}}),
        encoding="utf-8",
    )
    profile_home = root_home / "profiles" / "reviewer"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "Speed Quality", "provider": "moa"}}),
        encoding="utf-8",
    )

    # The kanban dispatcher injects HERMES_HOME=<root>/profiles/<name> into
    # every worker subprocess; load_config() then reads the profile config.
    token = set_hermes_home_override(str(profile_home))
    try:
        moa_raw = config_mod.load_config().get("moa") or {}
        resolved = resolve_moa_preset(moa_raw, "Speed Quality")
    finally:
        reset_hermes_home_override(token)

    agg = (resolved.get("aggregator") or {}).get("model")
    assert agg == "deepseek-v4-flash", (
        "worker resolved a different preset than the one configured in the "
        f"root config (aggregator={agg!r})"
    )


# ---------------------------------------------------------------------------
# Member 5 — #83197 / #83557 (duplicate pair, @mjshorty first): cron
# run_one_job installs the job profile's secret scope but resets it in the
# run-job ``finally`` BEFORE ``_deliver_result`` runs, so delivery resolves
# platform tokens without the owning profile's scope (and under multiplexing
# falls back to the shared/default environment).
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="#83197/#83557: cron secret scope reset before _deliver_result; "
    "delivery runs unscoped",
)
def test_cron_delivery_runs_inside_job_profile_secret_scope(
    monkeypatch, tmp_path
):
    """At ``_deliver_result`` time the job profile's secret scope must still
    be installed — delivery is part of the job, not an afterthought."""
    import cron.scheduler as sched
    from agent.secret_scope import current_secret_scope

    # Job profile home with a delivery credential in its .env.
    profile_home = tmp_path / "profiles" / "reviewer"
    profile_home.mkdir(parents=True)
    (profile_home / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=reviewer-token\n", encoding="utf-8"
    )
    monkeypatch.setattr(sched, "_get_hermes_home", lambda: profile_home)

    captured: dict = {}

    def fake_deliver(job, content, adapters=None, loop=None):
        captured["scope"] = current_secret_scope()
        return None

    monkeypatch.setattr(sched, "_deliver_result", fake_deliver)
    monkeypatch.setattr(sched, "claim_dispatch", lambda _id: True)
    monkeypatch.setattr(
        sched,
        "run_job",
        lambda job, defer_agent_teardown=None, extra_prompt=None: (
            True,
            "ran",
            "job report text",
            None,
        ),
    )
    monkeypatch.setattr(sched, "_is_interrupted", lambda _id: False)
    monkeypatch.setattr(sched, "save_job_output", lambda *a, **k: None)
    monkeypatch.setattr(sched, "mark_job_run", lambda *a, **k: None)

    job = {
        "id": "job-1",
        "execution_id": "exec-1",
        "name": "t",
        "deliver": "local",
    }
    assert sched.run_one_job(job) is True
    assert "scope" in captured, "_deliver_result was never invoked"
    scope = captured["scope"]
    assert scope is not None and scope.get("TELEGRAM_BOT_TOKEN") == "reviewer-token", (
        "delivery ran without the job profile's secret scope installed "
        f"(scope={scope!r})"
    )
