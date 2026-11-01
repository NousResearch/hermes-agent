"""POSIX publication -> generated command -> real Python bootstrap E2E (#129973).

The installed application's entry point is a deterministic probe; publication,
interpreter execution, dependency selection and argument forwarding are real.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from hermes_cli import _launchers
from hermes_cli.venv_sync import publish_launchers
from tests.hermes_cli.test_source_launcher_publication import fixture_tree, select_generation


def _install_probe(repo):
    body = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "def main():\n"
        "    import selected_probe\n"
        "    print(json.dumps({'root': str(Path(__file__).resolve().parents[1]),\n"
        "        'value': selected_probe.VALUE, 'argv': sys.argv[1:],\n"
        "        'original_argv': sys.orig_argv, 'isolated': sys.flags.isolated}))\n"
        "    return 7\n"
        "if __name__ == '__main__':\n    sys.exit(main())\n"
    )
    for relative in ("hermes_cli/main.py", "acp_adapter/entry.py"):
        (repo / relative).write_text(body, encoding="utf-8")


def _run_command(entry, *args):
    env = {
        "HOME": os.environ["HOME"], "HERMES_HOME": os.environ["HERMES_HOME"],
        "PATH": os.defpath, "LANG": "C.UTF-8", "TMPDIR": os.environ["TMPDIR"],
        "HERMES_SKIP_UPDATE_CHECK": "1", "HERMES_DISABLE_LAZY_INSTALLS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run([str(entry), *args], env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=30, check=False)
    assert result.returncode == 7, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.platforms("posix")
def test_named_commands_survive_republication_and_deleted_python(tmp_path, monkeypatch):
    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    _install_probe(repo)
    select_generation(repo, "selected", "selected-generation")
    publish_launchers(repo, create=False)
    args = ["", "spaces and café", "apostrophe's", "$HOME; echo no"]
    local = repo / ".hermes/bin"
    for name in _launchers.ENTRY_POINTS:
        receipt = _run_command(local / name, *args)
        assert receipt["isolated"] == 1
        assert receipt["argv"] == args
        assert receipt["value"] == "selected-generation"
        assert "-c" not in receipt["original_argv"]
        assert Path(receipt["original_argv"][2]).name == name

    targets = [local / name for name in _launchers.ENTRY_POINTS]
    targets += [repo / ".hermes/.launcher" / name for name in _launchers.ENTRY_POINTS]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in targets}
    publish_launchers(repo, create=False)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in targets}

    old_python = home / "tools/obsolete/bin/python3"
    old_python.parent.mkdir(parents=True)
    old_python.symlink_to(interpreter)
    convenience = tmp_path / ".local/bin"
    convenience.mkdir(parents=True)
    direct = _launchers.mint_launcher("hermes", repo, convenience, old_python, None)
    assert direct is not None
    shutil.rmtree(old_python.parents[1])
    assert not old_python.exists()
    publish_launchers(repo, create=False)
    for entry in (direct, local / "hermes"):
        receipt = _run_command(entry, *args)
        assert receipt["root"] == str(repo)
        assert receipt["isolated"] == 1
        assert receipt["argv"] == args
        assert "-c" not in receipt["original_argv"]


@pytest.mark.platforms("posix")
def test_publication_preserves_foreign_named_command_with_inert_owner_text(tmp_path, monkeypatch):
    root_a, _, _ = fixture_tree(tmp_path / "installation-a", monkeypatch)
    root_b, _, interpreter = fixture_tree(tmp_path / "installation-b", monkeypatch)
    for root, label in ((root_a, "installation-a"), (root_b, "installation-b")):
        _install_probe(root)
        select_generation(root, "selected", label)
    convenience = Path.home() / ".local/bin"
    convenience.mkdir(parents=True)
    wrapper = convenience / "hermes"
    private = convenience.parent / ".launcher/hermes"
    private.parent.mkdir()
    # This is a valid existing named launcher for B, even when A's publisher
    # predates named launchers. Use the real bootstrap generator, not a mock.
    body = _launchers._launcher_script("hermes", root_b, None)
    marker = f"sys.path.insert(0, {str(root_a)!r})"
    inert_mentions = (
        f"\n# Historical note: {marker}\n",
        f"\n_HISTORICAL_ROOT = {marker!r}\n",
        f"\ndef _unused_historical_root():\n    {marker}\n",
    )
    for inert in inert_mentions:
        private.write_text(f"#!{interpreter} -I\n{body}{inert}", encoding="utf-8")
        private.chmod(0o755)
        assert _launchers._write_shell(wrapper, [str(interpreter), "-I", str(private)])
        before = {p: p.read_bytes() for p in (wrapper, private)}
        assert _run_command(wrapper)["value"] == "installation-b"
        publish_launchers(root_a, create=False)
        assert before == {p: p.read_bytes() for p in (wrapper, private)}
        receipt = _run_command(wrapper)
        assert receipt["root"] == str(root_b)
        assert receipt["value"] == "installation-b"
