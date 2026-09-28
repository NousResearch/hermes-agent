"""Tests for the ByteRover memory provider config gates and ``memory.byterover.workdir``."""

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
