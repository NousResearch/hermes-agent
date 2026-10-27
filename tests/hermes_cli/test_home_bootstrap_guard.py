"""The home bootstrap must refuse targets that are obviously not a Hermes home.

``ensure_hermes_home()`` seeds ``_HERMES_HOME_SUBDIRS`` plus ``DEFAULT_SOUL_MD`` into
whatever ``HERMES_HOME`` names. With a wrong value it invented a home in five
dot-directories under ``~/.hermes/profiles`` (2026-10-21, one invocation per second in
alphabetical order), in ``profiles/devops/cache`` (2026-10-24, gateway stdio preserved
there) and in an archived profile (2026-10-25). Each stub was a 667-byte ``SOUL.md``
that ``.scripts/soul_integrity.py`` then reported as a real, failing identity.

The guard judges the *final* path component only — the default home is ``.hermes`` and
must keep bootstrapping — and refuses a ``cache`` component anywhere, because a cache
directory is purgeable by design.
"""

import logging
from pathlib import Path

import pytest

import hermes_constants
from hermes_cli import config
from hermes_cli.config import DEFAULT_SOUL_MD
from hermes_cli.config_home import initialize_home

_SUBDIRS = ("cron", "sessions", "logs", "memories")


def _set_home(monkeypatch, path: Path) -> Path:
    """Point HERMES_HOME at *path* and forget any memo for it (raw and resolved)."""
    monkeypatch.setenv("HERMES_HOME", str(path))
    absolute = path if path.is_absolute() else Path.cwd() / path
    config._HERMES_HOME_ENSURED.discard(str(path))
    config._HERMES_HOME_ENSURED.discard(str(absolute))
    config._HERMES_HOME_ENSURED.discard(str(absolute.resolve()))
    return path


# ── refused: an obviously-wrong HERMES_HOME creates nothing ──────────────────


@pytest.mark.parametrize("name", [".git", ".githooks", ".hidden", ".worktrees"])
def test_hidden_final_component_creates_nothing(tmp_path, monkeypatch, caplog, name):
    home = _set_home(monkeypatch, tmp_path / name)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.config_home"):
        config.ensure_hermes_home()

    assert not home.exists()
    assert not (home / "SOUL.md").exists()
    messages = [r.getMessage() for r in caplog.records if r.name == "hermes_cli.config_home"]
    assert len(messages) == 1, messages
    assert str(home) in messages[0]


@pytest.mark.parametrize("parts", [("cache",), ("profile", "cache"), ("cache", "hermes")])
def test_cache_component_creates_nothing(tmp_path, monkeypatch, caplog, parts):
    home = _set_home(monkeypatch, tmp_path.joinpath(*parts))

    with caplog.at_level(logging.WARNING, logger="hermes_cli.config_home"):
        config.ensure_hermes_home()

    assert not home.exists()
    assert not (tmp_path / parts[0]).exists()
    messages = [r.getMessage() for r in caplog.records if r.name == "hermes_cli.config_home"]
    assert len(messages) == 1, messages
    assert str(home) in messages[0]


def test_refusal_warns_once_per_home_path(tmp_path, monkeypatch, caplog):
    """A refused home is re-checked on the next load_config but must not re-warn."""
    _set_home(monkeypatch, tmp_path / ".hidden")

    with caplog.at_level(logging.WARNING, logger="hermes_cli.config_home"):
        config.ensure_hermes_home()
        config.ensure_hermes_home()

    messages = [r.getMessage() for r in caplog.records if r.name == "hermes_cli.config_home"]
    assert len(messages) == 1, messages


def test_initialize_home_refuses_directly(tmp_path):
    """The mutation choke point refuses even when a caller bypasses get_hermes_home()."""
    home = tmp_path / ".githooks"

    initialize_home(home, _SUBDIRS, set())

    assert not home.exists()


def test_relative_hidden_home_creates_nothing_in_cwd(tmp_path, monkeypatch):
    """HERMES_HOME='.git' with a cwd inside the profiles root is the 2026-10-21 shape."""
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    _set_home(monkeypatch, Path(".git"))

    config.ensure_hermes_home()

    assert not (work / ".git").exists()
    assert sorted(p.name for p in work.iterdir()) == []


# ── still seeded: real homes and near-miss names ────────────────────────────


def test_real_default_home_is_still_seeded(tmp_path, monkeypatch):
    """The default home itself starts with a dot; the guard must not block it."""
    home = _set_home(monkeypatch, tmp_path / ".hermes")

    config.ensure_hermes_home()

    assert home.is_dir()
    assert all((home / name).is_dir() for name in config._HERMES_HOME_SUBDIRS)
    assert (home / "SOUL.md").read_text(encoding="utf-8") == DEFAULT_SOUL_MD


def test_named_profile_home_is_still_seeded(tmp_path, monkeypatch):
    home = tmp_path / ".hermes" / "profiles" / "coder"
    home.mkdir(parents=True)  # profile creation is explicit, then the boot fills it
    _set_home(monkeypatch, home)

    config.ensure_hermes_home()

    assert (home / "sessions").is_dir()
    assert (home / "SOUL.md").read_text(encoding="utf-8") == DEFAULT_SOUL_MD


def test_hidden_parent_component_is_not_a_rejection(tmp_path, monkeypatch):
    """Only the final component is judged hidden (``~/.config/hermes`` is a real home)."""
    home = _set_home(monkeypatch, tmp_path / ".config" / "hermes")

    config.ensure_hermes_home()

    assert (home / "sessions").is_dir()
    assert (home / "SOUL.md").is_file()


def test_cache_lookalike_component_is_not_a_rejection(tmp_path, monkeypatch):
    """A component merely containing 'cache' (image_cache) is not a cache directory."""
    home = _set_home(monkeypatch, tmp_path / "image_cache" / "hermes")

    config.ensure_hermes_home()

    assert (home / "sessions").is_dir()
    assert (home / "SOUL.md").is_file()


# ── relative HERMES_HOME resolves before any rule is applied ────────────────


def test_relative_hermes_home_resolves_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", "relhome")

    assert hermes_constants._expand_hermes_home("relhome") == tmp_path / "relhome"
    home = hermes_constants.get_hermes_home()
    assert home.is_absolute(), home
    assert home == tmp_path / "relhome"


# ── the other writers that materialized a "home" ────────────────────────────


def test_logging_creates_no_log_dir_in_a_rejected_home(tmp_path):
    """`hermes gateway start` once captured its stdout into profiles/devops/cache/logs."""
    from hermes_logging import setup_logging

    home = tmp_path / "cache" / "hermes"

    returned = setup_logging(hermes_home=home, force=True)

    assert returned == home / "logs"
    assert not home.exists()


def test_scratch_export_skips_a_rejected_home(tmp_path):
    """TMPDIR must not be repointed at cache/scratch inside a non-home."""
    home = tmp_path / ".hidden"
    env = {"HERMES_HOME": str(home)}

    assert hermes_constants.apply_scratch_tmp_env(env) is False
    assert "TMPDIR" not in env and hermes_constants.SCRATCH_DIR_MARKER_ENV not in env
    assert not home.exists()


def test_boot_bootstrap_writes_nothing_in_a_rejected_home(tmp_path, monkeypatch):
    """``maybe_run_boot_bootstrap`` must not create ``installs/<key>/bootstrap/*.json``.

    The writer does ``mkdir(parents=True)``; on the first-pass fix a rejected
    ``HERMES_HOME`` still got ``<target>/installs/<sha>/bootstrap/default.json``
    (QA probe 2026-10-27).
    """
    from hermes_cli.boot_bootstrap import maybe_run_boot_bootstrap

    home = tmp_path / ".hidden"
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Forget any prior refusal memo for this path.
    from hermes_cli import config_home as _ch
    _ch._REJECTED_HOMES_WARNED.discard(str(home))
    project_root = tmp_path / "code"
    project_root.mkdir()
    # Give the project_root an identity so ``needs_bootstrap`` returns one and
    # the writer would actually mkdir+write on pristine HEAD.
    monkeypatch.setattr(
        "hermes_cli.boot_bootstrap.current_install_identity",
        lambda _root: "a" * 40,
    )
    # Stub the post-update steps so the test does not run real migrations.
    import hermes_cli.post_update as _pu
    monkeypatch.setattr(_pu, "run_steps", lambda _steps: {"step": "ok"})
    monkeypatch.setattr(_pu, "BOOT_HOME_STEPS", [])

    maybe_run_boot_bootstrap(project_root)

    assert not home.exists()
    assert not (tmp_path / "installs").exists()


def test_gateway_spawn_writes_no_logs_in_a_rejected_home(tmp_path, monkeypatch):
    """``spawn_unmanaged_gateway`` must not create ``logs/gateway-stdio.log``.

    Incident #2 (2026-10-24): ``hermes gateway start`` captured stdout into
    ``profiles/devops/cache/logs/gateway-stdio.log``.
    """
    import subprocess as _sp
    from hermes_cli.gateway_runtime_start import spawn_unmanaged_gateway

    home = tmp_path / "cache" / "hermes"
    # Stub the heavy imports the spawn pulls in; we only need the mkdir gate.
    monkeypatch.setattr(
        "hermes_cli.gateway_runtime_start.subprocess.Popen",
        lambda *a, **kw: _sp.CompletedProcess(a, 0),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway_runtime_start.remaining",
        lambda _deadline: None,
    )

    spawn_unmanaged_gateway(home, deadline=1e9, idle_exit=False)

    assert not home.exists()
    assert not (home / "logs").exists()
