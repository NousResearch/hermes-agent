import subprocess

from hermes_cli import main as hm


def test_ensure_tui_node_passes_helper_path_as_bash_argument(monkeypatch, tmp_path):
    """Do not interpolate the helper path into shell code."""
    project_root = tmp_path / 'hermes-$(touch pwned)"quoted`tick`'
    helper = project_root / "scripts" / "lib" / "node-bootstrap.sh"
    helper.parent.mkdir(parents=True)
    helper.write_text("ensure_node() { :; }\n", encoding="utf-8")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hm, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(hm.shutil, "which", lambda _name: None)
    monkeypatch.setattr(hm.subprocess, "run", fake_run)
    monkeypatch.delenv("HERMES_SKIP_NODE_BOOTSTRAP", raising=False)

    hm._ensure_tui_node()

    cmd, kwargs = calls[0]
    assert cmd == [
        "bash",
        "-c",
        'source -- "$1" >&2 && ensure_node >&2 && command -v node',
        "bash",
        str(helper),
    ]
    assert str(helper) not in cmd[2]
    assert kwargs["env"]["HERMES_HOME"]
