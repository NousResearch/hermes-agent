"""A plugin selection published while another process writes the same config.yaml.

The other writer is a real second process using the real config writer; the publication guard is the
real one. An edit to a key outside the plugin selection (a dashboard prompt's answer) must not refuse
the publication nor be overwritten by it; an edit to the selection itself still refuses it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _write_key_from_another_process(home: Path, key: str, value: str) -> None:
    code = ("import sys, json\n"
            "from hermes_cli.config_backend import write_config_key\n"
            "write_config_key(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))\n")
    subprocess.run([sys.executable, "-c", code, str(home / "config.yaml"), key, value],
                   cwd=_REPO, env={"HERMES_HOME": str(home), "HOME": str(home.parent), "PATH": "/usr/bin:/bin",
                                   "PYTHONPATH": str(_REPO)}, check=True, timeout=120)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from pm import paths

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    (home / "config.yaml").write_text("plugins:\n  enabled: [feishu]\n  disabled: []\nmodel: m\n", encoding="utf-8")
    return home, repo


def test_an_unrelated_key_written_meanwhile_neither_refuses_nor_is_lost(prepared):
    """The default install of a platform plugin enabled it while the user answered the dashboard's metrics
    prompt: the whole-file compare refused the enable, leaving the plugin installed but disabled."""
    from hermes_cli.runtime_state import finish_publication, runtime_lock
    from hermes_yaml import safe_load
    from pm.publication import PluginSelection

    home, repo = prepared
    with runtime_lock(repo):
        change = PluginSelection({"home": str(home), "enabled": ["feishu", "qqbot"], "disabled": []})
        _write_key_from_another_process(home, "telemetry.shared_metrics.prompted", "true")
        change.publish(repo)
        finish_publication(repo)

    config = safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert config["plugins"]["enabled"] == ["feishu", "qqbot"]
    assert config["telemetry"]["shared_metrics"]["prompted"] is True
    assert config["model"] == "m"


def test_a_plugin_selection_written_meanwhile_still_refuses_and_survives(prepared):
    from hermes_cli.runtime_state import runtime_lock
    from pm.publication import PluginSelection, SelectionChanged

    home, repo = prepared
    with runtime_lock(repo):
        change = PluginSelection({"home": str(home), "enabled": ["feishu", "qqbot"], "disabled": []})
        _write_key_from_another_process(home, "plugins.enabled", '["feishu", "other"]')
        theirs = (home / "config.yaml").read_bytes()
        with pytest.raises(SelectionChanged, match="changed while preparing publication"):
            change.publish(repo)

    assert (home / "config.yaml").read_bytes() == theirs
