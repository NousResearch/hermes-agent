"""Passive opt-out keeps explicit update checks available."""
import subprocess

from hermes_constants import get_hermes_home


def test_explicit_check_fetches_local_origin_despite_passive_opt_out(tmp_path, monkeypatch, capsys):
    from hermes_cli import main
    from hermes_cli.update_cmd import _cmd_update_check

    remote = tmp_path / "remote"
    local = tmp_path / "checkout"
    def git(*args):
        return subprocess.run(["git", *map(str, args)], check=True, capture_output=True, text=True)
    git("init", "-b", "main", remote)
    git("-C", remote, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--allow-empty", "-m", "initial")
    git("clone", remote, local)
    git("-C", remote, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--allow-empty", "-m", "next")
    monkeypatch.setattr(main, "PROJECT_ROOT", local)
    (get_hermes_home() / "config.yaml").write_text("updates:\n  check: false\n", encoding="utf-8")
    _cmd_update_check()
    output = capsys.readouterr().out
    assert "Fetching from origin" in output
    assert "1 commit" in output
