"""End-to-end credential isolation proof for multiplex mode (Workstream A).

These exercise the REAL resolution path (runtime_provider, secret scope, MCP
interpolation) rather than mocking it, proving the property that matters: two
profiles with different keys never see each other's, and an unscoped read in
multiplex mode fails closed instead of leaking.
"""
import pytest

from pathlib import Path

from agent import secret_scope as ss


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class TestRuntimeProviderUsesScope:
    """runtime_provider's credential reads (agent.secret_scope.get_secret_str) resolve through the scope."""

    def test_getenv_two_profiles_isolated(self, monkeypatch):
        from agent.secret_scope import get_secret_str as _getenv
        ss.set_multiplex_active(True)

        tok_a = ss.set_secret_scope({"OPENAI_API_KEY": "sk-A"})
        try:
            assert _getenv("OPENAI_API_KEY") == "sk-A"
        finally:
            ss.reset_secret_scope(tok_a)

        tok_b = ss.set_secret_scope({"OPENAI_API_KEY": "sk-B"})
        try:
            assert _getenv("OPENAI_API_KEY") == "sk-B"
        finally:
            ss.reset_secret_scope(tok_b)


class TestMcpInterpolationUsesScope:
    """MCP config ${VAR} interpolation resolves through the secret scope."""

    def test_interpolation_reads_scope(self, monkeypatch):
        from tools.mcp_tool_config import _interpolate_env_vars
        monkeypatch.setenv("MY_MCP_TOKEN", "global-token")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"MY_MCP_TOKEN": "profile-token"})
        try:
            cfg = {"env": {"TOKEN": "${MY_MCP_TOKEN}"}}
            assert _interpolate_env_vars(cfg) == {"env": {"TOKEN": "profile-token"}}
        finally:
            ss.reset_secret_scope(tok)


class TestProfilePathResolutionUnderMultiplexScope:
    """Profile-scoped paths must follow the per-turn _profile_runtime_scope.

    The multiplexed gateway (gateway.multiplex_profiles) serves every profile
    from ONE process, scoping each inbound turn with _profile_runtime_scope —
    the same in-process-many-profiles topology as the desktop tui_gateway. The
    profile-isolation fixes (per-call path resolution + thread context
    propagation) must therefore hold under THIS scope too, not just desktop.
    This is the regression guard proving reachability is not desktop-only.
    """

    def _profiles(self, tmp_path):
        prof_a = tmp_path / "profA"
        prof_b = tmp_path / "profB"
        for p in (prof_a, prof_b):
            (p / "skills").mkdir(parents=True, exist_ok=True)
            (p / "state").mkdir(parents=True, exist_ok=True)
        return prof_a, prof_b

    def test_skills_dir_follows_multiplex_scope(self, tmp_path):
        from gateway.run import _profile_runtime_scope
        import tools.skills_hub as sh

        prof_a, prof_b = self._profiles(tmp_path)
        with _profile_runtime_scope(prof_a):
            a_seen = Path(sh.SKILLS_DIR)
        with _profile_runtime_scope(prof_b):
            b_seen = Path(sh.SKILLS_DIR)

        assert a_seen == prof_a / "skills"
        assert b_seen == prof_b / "skills"


def test_turn_scoped_dotenv_reload_does_not_pollute_process_env(tmp_path, monkeypatch):
    """A routed profile reload must stay inside its context-local scope.

    ``load_hermes_dotenv`` has several lazy-import and cron call sites beyond
    the gateway's guarded reload helper.  Any one of them can run during a
    multiplexed turn, so the loader itself must not copy the active profile's
    ``.env`` into the shared process environment.
    """
    import os

    from agent.secret_scope import get_secret
    from gateway.run import _profile_runtime_scope
    from hermes_cli.env_loader import load_hermes_dotenv
    from hermes_constants import get_hermes_home

    profile_a = tmp_path / "profiles" / "a"
    profile_b = tmp_path / "profiles" / "b"
    profile_a.mkdir(parents=True)
    profile_b.mkdir(parents=True)
    (profile_a / ".env").write_text(
        "PROFILE_SCOPED_API_KEY=secret-a\n"
        "DISCORD_ALLOWED_CHANNELS=profile-a-only\n",
        encoding="utf-8",
    )
    (profile_b / ".env").write_text(
        "PROFILE_SCOPED_API_KEY=secret-b\n"
        "DISCORD_ALLOWED_CHANNELS=profile-b-only\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PROFILE_SCOPED_API_KEY", raising=False)
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "all-channels")

    ss.set_multiplex_active(True)
    with _profile_runtime_scope(profile_a):
        assert get_secret("PROFILE_SCOPED_API_KEY") == "secret-a"
        assert get_secret("DISCORD_ALLOWED_CHANNELS") == "profile-a-only"
        assert load_hermes_dotenv(hermes_home=get_hermes_home()) == []
        assert "PROFILE_SCOPED_API_KEY" not in os.environ
        assert os.environ["DISCORD_ALLOWED_CHANNELS"] == "all-channels"

    with _profile_runtime_scope(profile_b):
        assert get_secret("PROFILE_SCOPED_API_KEY") == "secret-b"
        assert get_secret("DISCORD_ALLOWED_CHANNELS") == "profile-b-only"
        assert load_hermes_dotenv(hermes_home=get_hermes_home()) == []
        assert "PROFILE_SCOPED_API_KEY" not in os.environ
        assert os.environ["DISCORD_ALLOWED_CHANNELS"] == "all-channels"


def test_launch_home_dotenv_still_loads_under_multiplex(tmp_path, monkeypatch):
    """#125530: the multiplex guard skips only FOREIGN (routed) home loads.

    The launch profile's own scoped bodies bind an override naming the launch home, so a
    launch-home load legitimately arrives with an override set; skipping it hid the launch
    home's ``.env`` (a fallback_providers key) from the process env. A routed profile's
    ``.env`` is still never copied into ``os.environ``.
    """
    import os

    from hermes_cli.env_loader import load_hermes_dotenv
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch = tmp_path / "launch"
    routed = tmp_path / "profiles" / "routed"
    launch.mkdir()
    routed.mkdir(parents=True)
    (launch / ".env").write_text("LAUNCH_ONLY_FALLBACK_KEY=launch-key\n", encoding="utf-8")
    (routed / ".env").write_text("LAUNCH_ONLY_FALLBACK_KEY=routed-secret\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("LAUNCH_ONLY_FALLBACK_KEY", raising=False)
    ss.set_multiplex_active(True)

    def _load(home):
        token = set_hermes_home_override(str(home))
        try:
            return load_hermes_dotenv(hermes_home=home)
        finally:
            reset_hermes_home_override(token)

    assert _load(launch) == [launch / ".env"]
    assert os.environ["LAUNCH_ONLY_FALLBACK_KEY"] == "launch-key"
    assert _load(routed) == []
    assert os.environ["LAUNCH_ONLY_FALLBACK_KEY"] == "launch-key"
    assert _load(launch) == [launch / ".env"]
    assert os.environ["LAUNCH_ONLY_FALLBACK_KEY"] == "launch-key"


def test_launch_home_load_inside_foreign_turn_keeps_routed_cwd_out_of_process_env(tmp_path, monkeypatch):
    """A launch-targeted load while a FOREIGN profile is routed is still skipped: its terminal
    bridge reads config through the override, so it would copy the routed profile's ``terminal.cwd``
    into the shared ``TERMINAL_CWD`` that the launch profile's own turns and cron jobs read."""
    import os

    from hermes_cli.env_loader import load_hermes_dotenv
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch = tmp_path / "launch"
    routed = tmp_path / "profiles" / "routed"
    launch_work = tmp_path / "work-launch"
    routed_work = tmp_path / "work-routed"
    for d in (launch, routed, launch_work, routed_work):
        d.mkdir(parents=True)
    (launch / ".env").write_text("LAUNCH_ONLY_FALLBACK_KEY=launch-key\n", encoding="utf-8")
    (launch / "config.yaml").write_text(f"terminal:\n  backend: local\n  cwd: {launch_work}\n", encoding="utf-8")
    (routed / "config.yaml").write_text(f"terminal:\n  backend: local\n  cwd: {routed_work}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("TERMINAL_CWD", str(launch_work))
    ss.set_multiplex_active(True)

    token = set_hermes_home_override(str(routed))
    try:
        assert load_hermes_dotenv(hermes_home=launch) == []
    finally:
        reset_hermes_home_override(token)
    assert os.environ["TERMINAL_CWD"] == str(launch_work)


def test_cold_profile_hydrates_external_source_without_global_env(
    tmp_path, monkeypatch
):
    """The first routed secondary turn must resolve its own source locally."""
    import os

    from agent.secret_sources.base import FetchResult
    from agent.secret_sources.registry import AppliedVar, ApplyReport, SourceReport
    from agent.secret_sources import registry
    from agent.secret_scope import get_secret
    from hermes_cli import env_loader
    from gateway.run import _profile_runtime_scope

    profile = tmp_path / "profiles" / "secondary"
    sibling = tmp_path / "profiles" / "sibling"
    profile.mkdir(parents=True)
    sibling.mkdir(parents=True)
    (profile / ".env").write_text(
        "EXPLICIT_API_KEY=dotenv-wins\n", encoding="utf-8"
    )
    monkeypatch.delenv("TEST_PROVIDER_API_KEY", raising=False)
    monkeypatch.delenv("EXPLICIT_API_KEY", raising=False)
    monkeypatch.setattr(
        env_loader,
        "_load_secrets_config",
        lambda home: (
            {"fake-source": {"enabled": True}}
            if Path(home).resolve() == profile.resolve()
            else {}
        ),
    )

    calls = {"count": 0}

    def _fake_apply_all(_cfg, _home, *, environ=None):
        calls["count"] += 1
        assert environ is not os.environ
        assert environ is not None
        assert environ["EXPLICIT_API_KEY"] == "dotenv-wins"
        environ["TEST_PROVIDER_API_KEY"] = "profile-only"
        return ApplyReport(
            sources=[
                SourceReport(
                    name="fake-source",
                    label="Fake Source",
                    result=FetchResult(),
                    applied=["TEST_PROVIDER_API_KEY"],
                )
            ],
            provenance={
                "TEST_PROVIDER_API_KEY": AppliedVar(
                    name="TEST_PROVIDER_API_KEY",
                    source="fake-source",
                    shape="mapped",
                    overrode_env=False,
                )
            },
        )

    monkeypatch.setattr(registry, "apply_all", _fake_apply_all)
    env_loader.reset_secret_source_cache()

    with _profile_runtime_scope(profile):
        assert get_secret("TEST_PROVIDER_API_KEY") == "profile-only"
        assert get_secret("EXPLICIT_API_KEY") == "dotenv-wins"
        assert env_loader.get_secret_source_values(profile) == {
            "TEST_PROVIDER_API_KEY": "profile-only"
        }
    with _profile_runtime_scope(profile):
        assert get_secret("TEST_PROVIDER_API_KEY") == "profile-only"
    with _profile_runtime_scope(sibling):
        assert get_secret("TEST_PROVIDER_API_KEY") is None

    assert calls["count"] == 1
    assert "TEST_PROVIDER_API_KEY" not in os.environ
    assert "EXPLICIT_API_KEY" not in os.environ
