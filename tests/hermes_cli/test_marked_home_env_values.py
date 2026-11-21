"""A profile a plugin runs elsewhere: this host takes no env value from its home on demand.

``load_hermes_dotenv`` already loads nothing from a marked home. The readers that open the home's
``.env`` later, one value at a time, all parse it with ``agent.secret_scope.load_env_file``:
``hermes_cli.config.load_env`` / ``get_env_value`` / ``get_env_value_prefer_dotenv`` /
``reload_env``, the credential pool's seeding from ``.env`` (which writes ``auth.json``), and the
secret scope a multi-profile host builds for a profile. The backend the plugin runs writes that
file, so for a marked home the tokenizer answers with nothing and does not open it. Writes of the
user's own keys (``save_env_value`` / ``remove_env_value``) still land in the file, and a file in
any other folder reads as before.

``home_runs_elsewhere`` is the check they share: ``run_elsewhere_for_home`` kept per process and
home, asked again whenever a stat of the home, the marker file or its folder changes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent import secret_scope
from hermes_cli import config as cfg
from hermes_cli import profiles_run_elsewhere as pre

PLUGIN = "sandbox-plugin"
_KEYS = ("MH2_KEY", "MH2_OTHER", "MH2_SAVED", "OPENROUTER_API_KEY", "HERMES_REDACT_SECRETS")


@pytest.fixture(autouse=True)
def _clean_process_env(monkeypatch):
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)
    yield
    for key in _KEYS:
        os.environ.pop(key, None)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A Hermes root whose named profile ``ops`` is this process's home; ``.env`` files in both."""
    home = tmp_path / ".hermes"
    ops = home / "profiles" / "ops"
    ops.mkdir(parents=True)
    for folder in (home, ops):
        (folder / "config.yaml").write_text("{}\n", encoding="utf-8")
    (home / ".env").write_text("MH2_KEY=from-the-default-home\n", encoding="utf-8")
    (ops / ".env").write_text("MH2_KEY=from-the-ops-home\nMH2_OTHER=second\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(ops))
    return home


def _ops(root: Path) -> Path:
    return root / "profiles" / "ops"


def _mark(root: Path, profile: str = "ops") -> None:
    pre.mark_profile_run_elsewhere(profile, PLUGIN, root=root)


# -- the single-value readers ----------------------------------------------------------------------

def test_a_home_that_runs_here_gives_its_env_values(root):
    assert cfg.load_env() == {"MH2_KEY": "from-the-ops-home", "MH2_OTHER": "second"}
    assert cfg.get_env_value("MH2_KEY") == "from-the-ops-home"
    assert cfg.get_env_value_prefer_dotenv("MH2_OTHER") == "second"


def test_a_marked_home_gives_no_env_value(root):
    _mark(root)
    assert cfg.load_env() == {}
    assert cfg.get_env_value("MH2_KEY") is None
    assert cfg.get_env_value_prefer_dotenv("MH2_OTHER") is None
    assert secret_scope.load_env_file(_ops(root) / ".env") == {}
    assert secret_scope.load_env_file(_ops(root) / ".op.env") == {}


def test_a_marked_homes_env_file_is_not_opened(root, monkeypatch):
    """The backend can put anything under that name (a link, a pipe): the host does not open it."""
    _mark(root)
    opened = []
    monkeypatch.setattr(secret_scope, "open", lambda path, *a, **k: opened.append(path) or pytest.fail("opened"),
                        raising=False)
    assert cfg.load_env() == {} and cfg.get_env_value("MH2_KEY") is None
    assert opened == []


def test_the_process_environment_still_answers_for_a_marked_home(root, monkeypatch):
    """What the marking plugin (or the user's shell) put into this process is the host's own."""
    _mark(root)
    monkeypatch.setenv("MH2_KEY", "from-the-process")
    assert cfg.get_env_value("MH2_KEY") == "from-the-process"
    assert cfg.get_env_value_prefer_dotenv("MH2_KEY") == "from-the-process"


def test_only_the_marked_profiles_home_is_withheld(root):
    _mark(root)
    assert secret_scope.load_env_file(root / ".env") == {"MH2_KEY": "from-the-default-home"}


def test_a_marked_default_profile_leaves_named_profiles_alone(root):
    _mark(root, "default")
    assert secret_scope.load_env_file(root / ".env") == {}
    assert cfg.load_env()["MH2_KEY"] == "from-the-ops-home"


def test_a_copy_outside_the_home_reads_as_before(root):
    """A host-only copy the marking plugin keeps (any folder that is no profile home) is not the
    marked home: the tokenizer reads it."""
    _mark(root)
    pin = root / "plugin-state" / "ops" / "pins"
    pin.mkdir(parents=True)
    (pin / ".env").write_text("MH2_KEY=pinned\n", encoding="utf-8")
    assert secret_scope.load_env_file(pin / ".env") == {"MH2_KEY": "pinned"}


def test_another_path_to_the_marked_home_is_withheld_too(root):
    _mark(root)
    other = root / "profiles" / "dev" / ".." / "OPS"
    (root / "profiles" / "dev").mkdir()
    assert secret_scope.load_env_file(other / ".env") in ({}, )  # as written or resolved: the same profile
    assert secret_scope.load_env_file(Path(str(_ops(root)) + os.sep) / ".env") == {}


def test_the_backend_the_plugin_runs_reads_its_own_env_file(root, monkeypatch):
    """Inside that backend the marker folder is denied: the home reads as not marked."""
    _mark(root)
    real_listdir = os.listdir

    def listdir(path):
        if Path(path).name == pre.RUN_ELSEWHERE_DIRNAME:
            raise PermissionError(13, "Permission denied", str(path))
        return real_listdir(path)

    monkeypatch.setattr(pre.os, "listdir", listdir)
    assert cfg.get_env_value("MH2_KEY") == "from-the-ops-home"


def test_an_unreadable_marker_folder_withholds_every_homes_env_values(root):
    """The rule's own fail-closed case: a marker folder that cannot be read marks every profile."""
    pre.run_elsewhere_dir(root).write_text("not a folder", encoding="utf-8")
    assert cfg.load_env() == {}
    assert secret_scope.load_env_file(root / ".env") == {}
    pre.run_elsewhere_dir(root).unlink()  # readable again: asked again, not remembered
    assert cfg.get_env_value("MH2_KEY") == "from-the-ops-home"
    assert secret_scope.load_env_file(root / ".env") == {"MH2_KEY": "from-the-default-home"}


def test_a_marker_folder_read_error_is_not_remembered(root, monkeypatch):
    """A read error that leaves the folder as it was (nothing a stat would show) marks every home
    only for as long as it lasts."""
    pre.run_elsewhere_dir(root).mkdir()
    failing = [True]
    real_listdir = os.listdir

    def listdir(path):
        if failing[0] and Path(path).name == pre.RUN_ELSEWHERE_DIRNAME:
            raise OSError(5, "Input/output error", str(path))
        return real_listdir(path)

    monkeypatch.setattr(pre.os, "listdir", listdir)
    assert cfg.load_env() == {}
    failing[0] = False
    assert cfg.get_env_value("MH2_KEY") == "from-the-ops-home"


# -- the check is kept per process and home, and never stale ---------------------------------------

def test_marking_and_clearing_are_seen_at_the_next_read(root):
    assert cfg.get_env_value("MH2_KEY") == "from-the-ops-home"
    _mark(root)
    assert cfg.get_env_value("MH2_KEY") is None
    assert pre.clear_profile_run_elsewhere("ops", PLUGIN, root=root)
    assert cfg.get_env_value("MH2_KEY") == "from-the-ops-home"
    _mark(root)
    assert cfg.get_env_value("MH2_KEY") is None


def test_a_marker_written_by_another_process_is_seen_at_the_next_read(root):
    """No call into this process: the marker file just appears, then goes."""
    assert cfg.get_env_value("MH2_KEY") == "from-the-ops-home"
    folder = pre.run_elsewhere_dir(root)
    folder.mkdir()
    assert cfg.get_env_value("MH2_KEY") == "from-the-ops-home"  # a folder with no marker for ops
    (folder / "dev.json").write_text("{}", encoding="utf-8")
    assert cfg.get_env_value("MH2_KEY") == "from-the-ops-home"  # another profile's marker
    marker = folder / "ops.json"
    marker.write_text(json.dumps({"plugin": PLUGIN}), encoding="utf-8")
    assert cfg.get_env_value("MH2_KEY") is None
    marker.unlink()
    assert cfg.get_env_value("MH2_KEY") == "from-the-ops-home"


def test_a_marker_is_seen_when_its_folders_time_does_not_move(root):
    """Not every filesystem touches a folder's time when a file is added to it, and two changes can
    share one tick of its clock: the marker file's own stat is part of what is compared."""
    folder = pre.run_elsewhere_dir(root)
    folder.mkdir()
    (folder / "dev.json").write_text("{}", encoding="utf-8")
    assert cfg.get_env_value("MH2_KEY") == "from-the-ops-home"
    before = os.stat(folder)
    marker = folder / "ops.json"
    for change, expected in ((lambda: marker.write_text("{}", encoding="utf-8"), None),
                             (marker.unlink, "from-the-ops-home")):
        change()
        os.utime(folder, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = os.stat(folder)
        assert (after.st_mode, after.st_size, after.st_mtime_ns) == (before.st_mode, before.st_size, before.st_mtime_ns)
        assert cfg.get_env_value("MH2_KEY") == expected


def _point(link: Path, target: Path) -> None:
    """Make *link* lead to the folder *target*, wherever it led before: a symlink, or on Windows a
    junction when this account may not create symlinks. Skips where the OS offers neither."""
    if os.path.lexists(link):
        os.rmdir(link) if os.name == "nt" else os.unlink(link)  # the link alone, not what it leads to
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        if os.name != "nt":
            pytest.skip(f"cannot create a directory symlink here: {exc}")
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


@pytest.mark.parametrize("among_the_profiles", [False, True])
def test_a_home_link_pointed_at_another_profile_is_seen_at_the_next_read(root, tmp_path, among_the_profiles):
    """The home may be a symlink or junction, and a link can be pointed at another profile while
    the process runs. No marker is written or removed then: the kept answer is for the directory
    the path led to, and the home's own stat is part of what is compared."""
    dev = root / "profiles" / "dev"
    dev.mkdir()
    (dev / ".env").write_text("MH2_KEY=from-the-dev-home\n", encoding="utf-8")
    _mark(root)  # ops, before the first read
    link = root / "profiles" / "linked" if among_the_profiles else tmp_path / "home-link"
    _point(link, dev)
    assert not pre.home_runs_elsewhere(link)
    assert secret_scope.load_env_file(link / ".env") == {"MH2_KEY": "from-the-dev-home"}

    _point(link, _ops(root))
    assert pre.run_elsewhere_for_home(link).profile == "ops"
    assert pre.home_runs_elsewhere(link)
    assert secret_scope.load_env_file(link / ".env") == {}

    _point(link, dev)
    assert pre.run_elsewhere_for_home(link) is None
    assert not pre.home_runs_elsewhere(link)
    assert secret_scope.load_env_file(link / ".env") == {"MH2_KEY": "from-the-dev-home"}


def test_the_marker_folder_is_listed_once_while_nothing_changes(root, monkeypatch):
    """Value reads come by the hundred: between changes each costs stats, not a resolve and a listing."""
    listings, resolves = [], []
    real_listdir, real_readings = os.listdir, pre._home_readings

    def listdir(path):
        if Path(path).name == pre.RUN_ELSEWHERE_DIRNAME:
            listings.append(path)
        return real_listdir(path)

    monkeypatch.setattr(pre.os, "listdir", listdir)
    monkeypatch.setattr(pre, "_home_readings", lambda home: resolves.append(home) or real_readings(home))
    for _ in range(200):
        assert cfg.get_env_value("MH2_KEY") == "from-the-ops-home"
    assert len(resolves) == 1 and len(listings) <= 2, (len(resolves), len(listings))
    _mark(root)
    for _ in range(200):
        assert cfg.get_env_value("MH2_KEY") is None
    assert len(resolves) == 2, len(resolves)


def test_home_runs_elsewhere_agrees_with_the_uncached_check(root):
    homes = (root, _ops(root), root / "profiles" / "dev")
    for step in (lambda: None, lambda: _mark(root), lambda: _mark(root, "default"),
                 lambda: pre.clear_profile_run_elsewhere("ops", PLUGIN, root=root)):
        step()
        for home in homes:
            assert pre.home_runs_elsewhere(home) is (pre.run_elsewhere_for_home(home) is not None)


# -- what the values used to reach -----------------------------------------------------------------

def test_reload_env_publishes_nothing_from_a_marked_home(root):
    _mark(root)
    cfg.reload_env()
    assert "MH2_KEY" not in os.environ and "MH2_OTHER" not in os.environ


def test_a_marked_homes_env_file_does_not_set_the_redaction_policy(root):
    from agent import redact
    from hermes_constants import hermes_home_key

    (_ops(root) / ".env").write_text("HERMES_REDACT_SECRETS=false\n", encoding="utf-8")
    cfg.load_env()
    assert redact._ENV_POLICY_BY_HOME.get(hermes_home_key()) == "false"  # a home that runs here: recorded
    _mark(root)
    cfg.load_env()
    assert redact._ENV_POLICY_BY_HOME.get(hermes_home_key()) is None


def test_the_credential_pool_seeds_nothing_from_a_marked_homes_env_file(root):
    """``hermes dump`` / ``auth list`` load the pool, which copies ``.env`` keys into auth.json."""
    from agent.credential_pool import load_pool

    (_ops(root) / ".env").write_text("OPENROUTER_API_KEY=sk-or-planted-by-the-backend\n", encoding="utf-8")
    _mark(root)
    assert not load_pool("openrouter").has_credentials()
    auth = _ops(root) / "auth.json"
    assert not auth.exists() or "OPENROUTER_API_KEY" not in auth.read_text(encoding="utf-8")


def test_the_credential_pool_seeds_from_the_env_file_of_a_home_that_runs_here(root):
    from agent.credential_pool import load_pool

    (_ops(root) / ".env").write_text("OPENROUTER_API_KEY=sk-or-the-users-own\n", encoding="utf-8")
    assert load_pool("openrouter").has_credentials()
    assert "env:OPENROUTER_API_KEY" in (_ops(root) / "auth.json").read_text(encoding="utf-8")


def test_a_served_profiles_secret_scope_holds_nothing_from_a_marked_home(root):
    """The mapping a multi-profile host builds for a profile it serves."""
    assert secret_scope.build_profile_secret_scope(_ops(root)).get("MH2_KEY") == "from-the-ops-home"
    _mark(root)
    scope = secret_scope.build_profile_secret_scope(_ops(root))
    assert "MH2_KEY" not in scope and "MH2_OTHER" not in scope


def _lmstudio_probes(root: Path, monkeypatch) -> list:
    """Render the status section that probes LM Studio when it is the profile's provider; returns
    the addresses probed."""
    from types import SimpleNamespace

    from hermes_cli import models_local, status_auth

    (_ops(root) / "config.yaml").write_text(json.dumps(
        {"model": {"provider": "lmstudio", "default": "m", "base_url": "http://127.0.0.1:9/v1"}}), encoding="utf-8")
    probed: list = []
    monkeypatch.setattr(models_local, "probe_lmstudio_models",
                        lambda api_key="", base_url="", timeout=0: probed.append(base_url) or [])
    status_auth._render_apikey_providers(SimpleNamespace(config=cfg.load_config()))
    return probed


def test_status_probes_the_model_address_of_a_profile_that_runs_here(root, monkeypatch, capsys):
    assert _lmstudio_probes(root, monkeypatch) == ["http://127.0.0.1:9/v1"]


def test_status_sends_no_probe_to_the_address_a_marked_profiles_config_names(root, monkeypatch, capsys):
    """``model.base_url`` is in config.yaml, which that backend writes too: with the ``.env`` values
    out of reach the probe would still go to an address of its choosing."""
    _mark(root)
    assert _lmstudio_probes(root, monkeypatch) == []


# -- writes of the user's own keys still land ------------------------------------------------------

def test_a_key_saved_for_a_marked_home_still_lands_in_its_env_file(root):
    """The marking plugin's host process saves the user's keys through these while it is marked."""
    _mark(root)
    cfg.save_env_value("MH2_SAVED", "the-users-new-key")
    text = (_ops(root) / ".env").read_text(encoding="utf-8")
    assert "MH2_SAVED=the-users-new-key" in text and "MH2_KEY=from-the-ops-home" in text
    assert os.environ.get("MH2_SAVED") == "the-users-new-key"  # published to this process, as before
    assert cfg.load_env() == {}  # and still not read back from the file
    assert cfg.remove_env_value("MH2_SAVED") is True
    assert "MH2_SAVED" not in (_ops(root) / ".env").read_text(encoding="utf-8")
    assert "MH2_SAVED" not in os.environ


# -- a new profile under a name that is already marked ---------------------------------------------

def _marked_new_profile(root: Path, env_text: str) -> Path:
    """``profiles/fresh``, just created with *env_text* as its ``.env``, under a name whose marker is
    already there (its predecessor was deleted and the marker stayed)."""
    _mark(root, "fresh")
    fresh = root / "profiles" / "fresh"
    fresh.mkdir()
    (fresh / "config.yaml").write_text("{}\n", encoding="utf-8")
    (fresh / ".env").write_text(env_text, encoding="utf-8")
    assert secret_scope.load_env_file(fresh / ".env") == {}  # marked from its first moment: not read here
    return fresh


def test_the_launch_keys_are_not_copied_over_the_keys_of_a_clone_made_under_a_marked_name(root):
    """Creating a profile mirrors the launch profile's ``.env`` into it, but only over the seeded stub,
    never over the keys a clone brought. That check reads the new file itself: for a marked home the
    tokenizer answers with nothing, which is not the same as a file that holds nothing."""
    from hermes_cli import profile_provision

    fresh = _marked_new_profile(root, "MH2_KEY=the-clones-own-key\n")
    mirrored: dict = {}
    profile_provision.mirror_launch_credentials(fresh, mirrored)
    assert mirrored["env"] is False
    assert (fresh / ".env").read_text(encoding="utf-8") == "MH2_KEY=the-clones-own-key\n"


def test_the_launch_keys_are_still_copied_over_the_stub_of_a_new_profile_under_a_marked_name(root):
    """The stub is what every new profile starts with; a marker on the name changes nothing about it."""
    from hermes_cli import profile_provision
    from hermes_cli.profiles import _PLACEHOLDER_ENV

    fresh = _marked_new_profile(root, _PLACEHOLDER_ENV)
    mirrored: dict = {}
    profile_provision.mirror_launch_credentials(fresh, mirrored)
    assert mirrored["env"] is True
    assert "MH2_KEY=from-the-ops-home" in (fresh / ".env").read_text(encoding="utf-8")


# -- what the user is told -------------------------------------------------------------------------

def _status(capsys, **flags) -> str:
    from types import SimpleNamespace

    from hermes_cli import status

    status.show_status(SimpleNamespace(full=False, deep=False, **flags))
    return capsys.readouterr().out


def test_status_says_once_that_a_marked_profiles_keys_are_not_read_here(root, capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert "runs elsewhere" not in _status(capsys)
    _mark(root)
    out = _status(capsys)
    assert out.count("runs elsewhere") == 1
    line = next(line for line in out.splitlines() if "runs elsewhere" in line)
    assert "'ops'" in line and PLUGIN in line and "keys" in line


def test_config_show_says_once_that_a_marked_profiles_keys_are_not_read_here(root, capsys, monkeypatch):
    from hermes_cli import config_show

    monkeypatch.setenv("NO_COLOR", "1")
    config_show.show_config()
    assert "runs elsewhere" not in capsys.readouterr().out
    _mark(root)
    config_show.show_config()
    out = capsys.readouterr().out
    assert out.count("runs elsewhere") == 1 and PLUGIN in out
