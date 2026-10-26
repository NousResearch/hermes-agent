"""config.yaml write integrity: a failed read never reaches disk, and a save keeps what the user wrote."""

import errno
import os
import shutil

import pytest
import hermes_yaml as yaml

import hermes_cli.config as config_mod
from hermes_cli.config import DEFAULT_CONFIG, load_config, migrate_config, read_raw_config, save_config

_CONFIG = """# hand-tuned
model:
  default: my-org/custom-model
display:
  skin: mono
terminal:
  backend: docker
approvals:
  deny:
  - rm -rf /
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    from tui_gateway import server
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    (tmp_path / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    yield tmp_path
    server._cfg_cache = server._cfg_sig = server._cfg_path = None


class _ReadFaults:
    """Counts parses of config.yaml and fails the chosen one with a transient EMFILE (intact file)."""

    def __init__(self, monkeypatch, path):
        import utils
        self.path, self.count, self.fail_at, real = str(path), 0, 0, utils.fast_safe_load

        def flaky(stream):
            if getattr(stream, "name", None) == self.path:
                self.count += 1
                if self.count == self.fail_at:
                    raise OSError(errno.EMFILE, "Too many open files")
            return real(stream)
        # Every config read parses through the file config backend, which calls utils.fast_safe_load.
        monkeypatch.setattr(utils, "fast_safe_load", flaky)

    def arm(self, fail_at=0):
        self.count, self.fail_at = 0, fail_at


def _fresh_process(home, text, *, keep_last_known_good=False):
    """*text* on a new inode with cold caches (and no last-known-good copy unless asked)."""
    from tui_gateway import server
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    if not keep_last_known_good:
        config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
        shutil.rmtree(home / "backups", ignore_errors=True)
    tmp = home / ".config.yaml.new"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, home / "config.yaml")


def _save_loaded():
    cfg = load_config()
    cfg["display"]["skin"] = "ares"
    save_config(cfg)


def _save_raw():  # the shape every migration step and ``write_platform_config_field(raw=True)`` use
    cfg = read_raw_config()
    cfg.setdefault("display", {})["skin"] = "ares"
    save_config(cfg)


def _tui_config_set():
    from tui_gateway import server
    reply = server._methods["config.set"](1, {"key": "skin", "value": "ares"})
    if "error" in reply:
        raise RuntimeError(reply["error"]["message"])


def _dashboard_put():
    from starlette.testclient import TestClient
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    client = TestClient(app, raise_server_exceptions=False)
    r = client.put("/api/config", json={"config": {"display": {"skin": "ares"}}},
                   headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
    if r.status_code != 200:
        raise RuntimeError(r.text)


@pytest.mark.parametrize("last_known_good", ["none", "stale"])
@pytest.mark.parametrize("write", [_save_loaded, _save_raw, _tui_config_set, _dashboard_put],
                         ids=["load_config", "read_raw_config", "tui-config.set", "dashboard-put"])
def test_no_single_transient_read_error_reaches_the_file(write, last_known_good, home, monkeypatch):
    """Fail each config read the write performs, one at a time: the write either lands exactly as
    it would have, or is refused with the file byte-identical, and the retry is not poisoned."""
    path = home / "config.yaml"
    before = _CONFIG + "timezone: Europe/Paris\n"
    faults = _ReadFaults(monkeypatch, path)

    def run(k):
        _fresh_process(home, before)
        if last_known_good == "stale":  # loaded earlier in this process, edited elsewhere since
            _fresh_process(home, _CONFIG)
            load_config()
            _fresh_process(home, before, keep_last_known_good=True)
        faults.arm(k)
        try:
            write()
        except RuntimeError:
            return "refused"
        return "ok"

    assert run(0) == "ok"
    good_after, reads = path.read_bytes(), faults.count
    assert b"skin: ares" in good_after and reads >= 1
    for k in range(1, reads + 1):
        if run(k) == "refused":
            assert path.read_text(encoding="utf-8") == before, f"read {k}/{reads} failed, file changed"
            faults.arm()
            write()  # the retry is not poisoned by a cached fallback
        assert path.read_bytes() == good_after, f"read {k}/{reads} failed, the saved file differs"


@pytest.mark.parametrize("read", [load_config, read_raw_config], ids=["load_config", "read_raw_config"])
def test_transient_read_error_is_not_recorded_as_a_corrupt_config(read, home, monkeypatch, capsys):
    """One EMFILE on an intact file must not leave the process treating config.yaml as corrupt: the
    provider auto-resolution refusal (`corrupt_config`) keyed on the file signature would otherwise
    fire until the file is next edited, and the good file would be copied away as `.corrupt`."""
    from hermes_cli.auth import _refuse_env_adoption_if_config_corrupt
    from hermes_cli.config_read_errors import _CONFIG_PARSE_WARNED, get_active_config_parse_failure
    path = home / "config.yaml"
    _fresh_process(home, _CONFIG)
    _CONFIG_PARSE_WARNED.clear()
    faults = _ReadFaults(monkeypatch, path)
    faults.arm(1)

    read()
    assert get_active_config_parse_failure() is not None  # unreadable right now: the refusal holds
    assert read()["display"]["skin"] == "mono"

    assert get_active_config_parse_failure() is None
    _refuse_env_adoption_if_config_corrupt()
    assert not list((home / "backups").glob("**/*.corrupt.*"))
    assert "could not be read" in capsys.readouterr().err


def test_unreadable_config_serves_one_cached_fallback_until_it_reads(home, monkeypatch):
    """While config.yaml stays unreadable, loads serve one cached fallback (the backup is parsed
    once, not per call: ~250x slower loads before), and the first load once the file opens again
    reads the real file even though its signature never changed."""
    import builtins
    from hermes_cli import config_backend, config_backups
    from hermes_cli.config_read_errors import FailedConfigRead
    path = home / "config.yaml"
    _fresh_process(home, _CONFIG)
    load_config()  # leaves the `good` backup a fresh process falls back to
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    blocked, rebuilds, real_backup = [True], [], config_backups.load_newest_good_backup

    def guarded_open(file, *args, **kwargs):
        if blocked[0] and str(file) == str(path):
            raise OSError(errno.EMFILE, "Too many open files")
        return builtins.open(file, *args, **kwargs)
    monkeypatch.setattr(config_backend, "open", guarded_open, raising=False)
    monkeypatch.setattr(config_backups, "load_newest_good_backup", lambda p: rebuilds.append(p) or real_backup(p))

    for _ in range(5):
        cfg = load_config()
        assert isinstance(cfg, FailedConfigRead) and cfg["display"]["skin"] == "mono"
    assert len(rebuilds) == 1
    blocked[0] = False
    assert type(load_config()) is dict


def test_save_refusal_for_bad_yaml_asks_for_an_edit_not_a_retry(home):
    _fresh_process(home, "model: [unclosed\n")
    cfg = load_config()
    cfg["display"]["skin"] = "ares"
    with pytest.raises(RuntimeError, match="has a formatting error") as refusal:
        save_config(cfg)
    assert "hermes config edit" in str(refusal.value) and "Try again" not in str(refusal.value)
    assert (home / "config.yaml").read_text(encoding="utf-8") == "model: [unclosed\n"


@pytest.mark.parametrize("operation", ["save", "partial_save", "migrate"])
def test_authored_nulls_survive_config_writes(tmp_path, monkeypatch, operation):
    from hermes_cli.resource_limits import configured_nofile_soft_limit
    from agent.agent_runtime_helpers import prompt_caching_disabled_from_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    seed = {
        "_config_version": DEFAULT_CONFIG["_config_version"] - (operation == "migrate"),
        "runtime": {"nofile_soft_limit": None},
        "prompt_caching": {"cache_ttl": None},
        "x_null_preservation": {"nested": {"optional": None}, "keep": "value"},
    }
    config_path.write_text(yaml.safe_dump(seed), encoding="utf-8")

    if operation == "migrate":
        migrate_config(interactive=False, quiet=True)
    elif operation == "partial_save":
        save_config({"display": {"skin": "mono"}}, merge_existing=True)
    else:
        config = load_config()
        config["display"]["skin"] = "mono"
        save_config(config)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["runtime"]["nofile_soft_limit"] is None
    assert raw["prompt_caching"]["cache_ttl"] is None
    assert raw["x_null_preservation"] == seed["x_null_preservation"]
    assert "terminal" not in raw
    assert "agent" not in raw  # no section the user never wrote, not even an empty one
    assert configured_nofile_soft_limit() is None
    assert prompt_caching_disabled_from_config() is True
    if operation == "migrate":
        assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
    else:
        assert raw["display"]["skin"] == "mono"
