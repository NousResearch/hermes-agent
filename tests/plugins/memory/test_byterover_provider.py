"""Tests for the ByteRover memory provider config gates, ``memory.byterover.workdir``
and ``memory.byterover.curate_timeout``."""

import pytest

from plugins.memory.byterover import ByteRoverMemoryProvider, _get_brv_cwd


def test_auto_extract_false_skips_sync_turn(monkeypatch):
    calls = []
    provider = ByteRoverMemoryProvider({"auto_extract": False})
    provider.initialize("session-1")

    monkeypatch.setattr("plugins.memory.byterover._run_brv", lambda *args, **kwargs: calls.append((args, kwargs)))

    provider.sync_turn("please remember this detail", "acknowledged")

    assert calls == []
    assert provider._sync_thread is None


# ---------------------------------------------------------------------------
# memory.byterover.workdir — shared-tree override
# ---------------------------------------------------------------------------

def _home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_workdir_unset_falls_back_to_profile_home(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    assert _get_brv_cwd({}) == home / "byterover"
    assert _get_brv_cwd(None) == home / "byterover"


def test_workdir_absolute_path_used_verbatim(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)  # set, but the override must win
    shared = tmp_path / "shared-tree"
    assert _get_brv_cwd({"workdir": str(shared)}) == shared.resolve()


def test_workdir_tilde_expands(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # expanduser() reads $HOME, not Path.home()
    resolved = _get_brv_cwd({"workdir": "~/fleet/byterover"})
    assert resolved == (tmp_path / "fleet" / "byterover").resolve()


def test_workdir_empty_string_is_ignored(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    assert _get_brv_cwd({"workdir": "  "}) == home / "byterover"


def test_two_profiles_same_workdir_see_identical_tree(tmp_path, monkeypatch):
    """The sharing contract: two HERMES_HOME-isolated workers initialize to one cwd."""
    shared = tmp_path / "shared-tree"
    seen = []
    for profile in ("opnory-builder", "opnory-verifier"):
        home = tmp_path / profile
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        provider = ByteRoverMemoryProvider({"workdir": str(shared)})
        provider.initialize(f"session-{profile}")
        seen.append(provider._cwd)
    assert seen[0] == seen[1] == str(shared.resolve())


def test_provider_tools_all_receive_provider_cwd(tmp_path, monkeypatch):
    """brv_query / brv_curate / brv_status must run in the configured workdir, never the
    profile tree. Regression guard for the _cwd handoff in initialize()."""
    shared = tmp_path / "shared-tree"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))  # a DIFFERENT home
    seen = {}

    def fake_run_brv(args, timeout=0, cwd=None):
        seen[tuple(args)[:2]] = cwd
        return {"success": True, "output": '{"result": "ok"}'}

    monkeypatch.setattr("plugins.memory.byterover._run_brv", fake_run_brv)

    provider = ByteRoverMemoryProvider({"workdir": str(shared)})
    provider.initialize("session-x")
    for tool, args in (("brv_query", {"query": "RBAC enforcement design"}),
                       ("brv_curate", {"content": "issuer mismatch confirmed"}),
                       ("brv_status", {})):
        provider.handle_tool_call(tool, args)
    assert provider._cwd == str(shared.resolve())
    assert seen and set(seen.values()) == {provider._cwd}, seen


def test_curate_timeout_is_a_structured_failure_not_an_exception(tmp_path, monkeypatch):
    """A brv curate timeout must surface as {success: False, error: "timed out..."} —
    never raise into the agent, and never be misread as "memory says no". Exercises the
    REAL _run_brv try/except by making its subprocess.run raise TimeoutExpired."""
    import subprocess as _sp
    import plugins.memory.byterover as byterover_mod

    def boom(*a, **k):
        raise _sp.TimeoutExpired(cmd=["brv"], timeout=byterover_mod._CURATE_TIMEOUT)

    monkeypatch.setattr(byterover_mod.subprocess, "run", boom)
    provider = ByteRoverMemoryProvider({})
    provider.initialize("session-timeout")
    result = provider._curate("some fact")
    assert result["success"] is False
    assert "timed out" in result["error"]


# ---------------------------------------------------------------------------
# memory.byterover.curate_timeout — configurable curate ceiling
# ---------------------------------------------------------------------------

def test_curate_timeout_defaults_to_120_for_backward_compat():
    """Existing deployments (no curate_timeout key) must keep the exact old ceiling."""
    assert ByteRoverMemoryProvider({})._curate_timeout == 120


def test_curate_timeout_can_be_overridden():
    assert ByteRoverMemoryProvider({"curate_timeout": 360})._curate_timeout == 360


def test_curate_timeout_invalid_value_falls_back_to_default():
    provider = ByteRoverMemoryProvider({"curate_timeout": "whenever"})
    assert provider._curate_timeout == 120


@pytest.mark.parametrize("bad,clamped", [(0, 1), (-30, 1), (99999, 3600)])
def test_curate_timeout_is_clamped_to_safe_bounds(bad, clamped):
    """A typo can never wedge a turn open-endedly or zero out the subprocess timeout."""
    assert ByteRoverMemoryProvider({"curate_timeout": bad})._curate_timeout == clamped


def test_curate_passes_configured_timeout_to_run_brv(tmp_path, monkeypatch):
    """The configured ceiling must reach _run_brv — not the module constant."""
    seen = {}

    def fake_run_brv(args, timeout=0, cwd=None):
        seen["timeout"] = timeout
        return {"success": True, "output": "ok"}

    monkeypatch.setattr("plugins.memory.byterover._run_brv", fake_run_brv)
    provider = ByteRoverMemoryProvider({"curate_timeout": 360})
    provider.initialize("session-ceiling")
    provider._curate("remember this")
    assert seen["timeout"] == 360


def test_configured_curate_timeout_names_itself_in_the_timeout_error(tmp_path, monkeypatch):
    """The structured failure must report the ceiling actually in effect, so a fleet
    operator reading the log can tell a configured 360s kill from the old 120s one."""
    import subprocess as _sp
    import plugins.memory.byterover as byterover_mod

    def boom(*a, **k):
        raise _sp.TimeoutExpired(cmd=["brv"], timeout=360)

    monkeypatch.setattr(byterover_mod.subprocess, "run", boom)
    provider = ByteRoverMemoryProvider({"curate_timeout": 360})
    provider.initialize("session-timeout-360")
    result = provider._curate("some fact")
    assert result == {"success": False, "error": "brv timed out after 360s"}


def test_config_schema_declares_workdir_and_curate_timeout():
    """Both fleet settings must be declared: the dashboard save route only
    writes schema fields, so an undeclared key silently vanishes on save."""
    schema = ByteRoverMemoryProvider().get_config_schema()
    keys = {field["key"] for field in schema}
    assert "workdir" in keys, "workdir undeclared: dashboard saves would drop it"
    assert "curate_timeout" in keys, "curate_timeout undeclared: dashboard saves would drop it"
    curate = next(f for f in schema if f["key"] == "curate_timeout")
    assert curate.get("kind") == "integer", "curate_timeout must coerce as integer"
    assert curate.get("minimum") == 1 and curate.get("maximum") == 3600, "bounds must match the plugin's clamping"
