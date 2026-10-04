"""A service definition must never name a code root that cannot resolve its dependencies.

Regression for the 2026-09-26 outage: the launchd plist was regenerated from a pinned environment copy
(`<installs>/<key>/environments/<env>/workspace`), whose install key owns no state under
`~/.hermes/installs`, so every launch died with "no dependency environment is committed for this
install" (exit 1) and launchd respawned it forever.

The definitions these tests read are minted by ``generate_launchd_plist()`` itself -- the real
``shlex.join`` + XML escaping -- rather than hand-written, because a fixture of the shape the
generator does NOT emit is exactly how a parser that cannot read the real one slips through review.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import gateway as gateway_cli
from hermes_cli import gateway_launchd


def _patch_resolvers(monkeypatch, *, store_python, committed, payload=None):
    # Inline imports inside the predicate read the DEFINING modules at call time, so patch there.
    import hermes_cli._launchers as launchers
    import pm.environments as environments

    monkeypatch.setattr(launchers, "resolve_store_python", lambda root: store_python)
    monkeypatch.setattr(environments, "committed_venv", lambda root: committed)
    monkeypatch.setattr(environments, "payload_venv", lambda root: payload)


def _launchctl_run(cmd, **kwargs):
    """The gateway service is registered but idle: `launchctl list` succeeds without a PID."""
    if isinstance(cmd, list) and cmd[:2] == ["launchctl", "list"]:
        return SimpleNamespace(returncode=0, stdout='{\n    "Label" = "ai.hermes.gateway";\n}', stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _service_env(tmp_path, monkeypatch, *, project_root, store_python=None, committed=None, payload=None):
    """Everything the generator and ``launchd_status`` read, pointed at throwaway paths.

    Returns ``(plist_path, home)``. A non-None ``store_python`` selects the store-Python install shape
    (a persisted ``.hermes/bin/hermes`` launcher); None selects an externally owned runtime (Nix, a
    developer venv), which has no launcher path and is launched through its own interpreter.
    """
    _patch_resolvers(monkeypatch, store_python=store_python, committed=committed, payload=payload)
    home = tmp_path / "home"
    (home / "logs").mkdir(parents=True, exist_ok=True)
    plist_path = home / "Library" / "LaunchAgents" / "ai.hermes.gateway.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: home)
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gateway_cli, "_profile_arg", lambda *args, **kwargs: "")
    monkeypatch.setattr(gateway_cli.subprocess, "run", _launchctl_run)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda cleanup_stale=False: None)
    # Config-derived and irrelevant here: keep the generated definition deterministic.
    import hermes_cli.resource_limits as resource_limits

    monkeypatch.setattr(resource_limits, "configured_nofile_soft_limit", lambda *args, **kwargs: None)
    return plist_path, home


def _install_root(tmp_path, name="hermes-agent"):
    root = tmp_path / name
    root.mkdir(parents=True)
    return root


# --------------------------------------------------------------------------------------------------
# Viability matrix
# --------------------------------------------------------------------------------------------------

def test_install_root_with_committed_generation_is_viable(tmp_path, monkeypatch):
    _patch_resolvers(monkeypatch, store_python=tmp_path / "bin/python3", committed=tmp_path / "venv")
    assert gateway_launchd.launcher_root_is_viable(tmp_path) is True


def test_pinned_environment_workspace_is_not_viable(tmp_path, monkeypatch):
    """The exact outage shape: store Python, no committed generation, no payload venv."""
    _patch_resolvers(monkeypatch, store_python=tmp_path / "bin/python3", committed=None, payload=None)
    assert gateway_launchd.launcher_root_is_viable(tmp_path) is False


def test_externally_owned_runtime_is_viable(tmp_path, monkeypatch):
    """Nix / developer venv: no store Python, so bootstrap keeps the interpreter's own packages."""
    _patch_resolvers(monkeypatch, store_python=None, committed=None, payload=None)
    assert gateway_launchd.launcher_root_is_viable(tmp_path) is True


def test_payload_venv_is_viable(tmp_path, monkeypatch):
    """Sealed payload: payload_venv is enough even with no committed generation."""
    _patch_resolvers(monkeypatch, store_python=tmp_path / "bin/python3", committed=None,
                     payload=tmp_path / "payload-venv")
    assert gateway_launchd.launcher_root_is_viable(tmp_path) is True


def test_assert_refuses_an_unviable_root(tmp_path, monkeypatch):
    _patch_resolvers(monkeypatch, store_python=tmp_path / "bin/python3", committed=None, payload=None)
    with pytest.raises(RuntimeError, match="cannot resolve its dependencies"):
        gateway_launchd.assert_launcher_root_is_viable(tmp_path)


# --------------------------------------------------------------------------------------------------
# Reading a code root back out of a definition
# --------------------------------------------------------------------------------------------------

def test_a_root_with_a_space_names_itself(tmp_path, monkeypatch):
    """``shlex.join`` quotes the launcher, so the definition holds ``exec '<path with a space>'``."""
    root = _install_root(tmp_path, "John Appleseed/hermes-agent")
    _service_env(tmp_path, monkeypatch, project_root=root, store_python=tmp_path / "bin/python3",
                 committed=tmp_path / "venv")

    definition = gateway_cli.generate_launchd_plist()

    assert "&#x27;" in definition, "the generator must have had to quote this launcher"
    assert gateway_launchd.installed_service_launcher_root(definition) == root


def test_the_pinned_environment_workspace_shape_names_itself(tmp_path, monkeypatch):
    """Unquoted launcher: the ordinary store-Python install must keep working."""
    root = tmp_path / "installs" / "abc" / "environments" / "def" / "workspace"
    root.mkdir(parents=True)
    _service_env(tmp_path, monkeypatch, project_root=root, store_python=tmp_path / "bin/python3",
                 committed=tmp_path / "venv")

    assert gateway_launchd.installed_service_launcher_root(
        gateway_cli.generate_launchd_plist()
    ) == root


def test_an_external_runtime_names_its_root_through_its_bootstrap(tmp_path, monkeypatch):
    """Nix / developer venv: no launcher path at all, so its ``sys.path`` entry is the only signal."""
    root = _install_root(tmp_path, "nix-hermes")
    _service_env(tmp_path, monkeypatch, project_root=root, store_python=None)

    definition = gateway_cli.generate_launchd_plist()

    assert "/.hermes/bin/hermes" not in definition, "an external runtime must not mint a launcher"
    assert gateway_launchd.installed_service_launcher_root(definition) == root


def test_a_definition_that_names_no_root_names_none():
    assert gateway_launchd.installed_service_launcher_root("<plist/>") is None
    # A wrapped command that is neither this install's launcher nor a bootstrap that prepends a path.
    assert gateway_launchd.installed_service_launcher_root(
        '<string>$.system(&quot;exec /usr/bin/true&quot;)</string>'
    ) is None


# --------------------------------------------------------------------------------------------------
# The promises the status page makes
# --------------------------------------------------------------------------------------------------

def test_the_definition_this_install_writes_reads_as_current(tmp_path, monkeypatch):
    root = _install_root(tmp_path)
    plist_path, _ = _service_env(tmp_path, monkeypatch, project_root=root,
                                 store_python=tmp_path / "bin/python3", committed=tmp_path / "venv")
    plist_path.write_text(gateway_cli.generate_launchd_plist(), encoding="utf-8")

    assert gateway_cli.launchd_plist_is_current() is True
    assert gateway_cli.launchd_status() is True


def test_a_stale_definition_reports_not_ok_and_says_how_to_fix_it(tmp_path, monkeypatch, capsys):
    root = _install_root(tmp_path)
    plist_path, _ = _service_env(tmp_path, monkeypatch, project_root=root,
                                 store_python=tmp_path / "bin/python3", committed=tmp_path / "venv")
    plist_path.write_text(
        gateway_cli.generate_launchd_plist().replace("<integer>30</integer>", "<integer>31</integer>"),
        encoding="utf-8",
    )

    assert gateway_cli.launchd_status() is False
    out = capsys.readouterr().out
    assert "stale" in out
    assert "Run: hermes gateway start" in out


def test_a_definition_running_another_code_root_is_named(tmp_path, monkeypatch, capsys):
    """The definition was written from a different checkout: name it, and never say "matches"."""
    written_from = _install_root(tmp_path, "John Appleseed/other-checkout")
    plist_path, _ = _service_env(tmp_path, monkeypatch, project_root=written_from,
                                 store_python=tmp_path / "bin/python3", committed=tmp_path / "venv")
    plist_path.write_text(gateway_cli.generate_launchd_plist(), encoding="utf-8")

    current = _install_root(tmp_path, "current-checkout")
    _service_env(tmp_path, monkeypatch, project_root=current, store_python=tmp_path / "bin/python3",
                 committed=tmp_path / "venv")

    assert gateway_cli.launchd_status() is False
    out = capsys.readouterr().out
    assert f"DIFFERENT code root: {written_from}" in out
    assert f"Current code root: {current}" in out
    assert "matches the current Hermes install" not in out


def test_an_unviable_code_root_reports_not_ok_without_raising(tmp_path, monkeypatch, capsys):
    root = _install_root(tmp_path)
    plist_path, _ = _service_env(tmp_path, monkeypatch, project_root=root,
                                 store_python=tmp_path / "bin/python3", committed=tmp_path / "venv")
    plist_path.write_text(gateway_cli.generate_launchd_plist(), encoding="utf-8")
    # ...and now the root this install runs from cannot resolve its own dependencies.
    _patch_resolvers(monkeypatch, store_python=tmp_path / "bin/python3", committed=None, payload=None)

    assert gateway_cli.launchd_plist_is_current() is False  # a read, not a traceback
    assert gateway_cli.launchd_status() is False
    assert "cannot resolve its dependencies" in capsys.readouterr().out


def test_a_refused_regeneration_leaves_the_previous_definition_untouched(tmp_path, monkeypatch):
    """"Fails closed before the write" is only true if the file actually survives."""
    root = _install_root(tmp_path)
    plist_path, _ = _service_env(tmp_path, monkeypatch, project_root=root,
                                 store_python=tmp_path / "bin/python3", committed=tmp_path / "venv")
    previous = gateway_cli.generate_launchd_plist()
    plist_path.write_text(previous, encoding="utf-8")

    _patch_resolvers(monkeypatch, store_python=tmp_path / "bin/python3", committed=None, payload=None)

    with pytest.raises(gateway_launchd.UnviableLauncherRootError, match="cannot resolve its dependencies"):
        gateway_cli.launchd_install(force=True)
    assert plist_path.read_text(encoding="utf-8") == previous


def test_a_refused_definition_surfaces_as_guidance_not_a_traceback():
    from hermes_cli.gateway_command_errors import explain_service_failure

    exc = gateway_launchd.UnviableLauncherRootError(
        "refusing to write a service definition whose code root cannot resolve its dependencies: /x"
    )

    lines = explain_service_failure(exc)

    assert lines is not None
    assert any("cannot resolve its dependencies" in line for line in lines)
    assert any("left unchanged" in line for line in lines)
