from pathlib import Path
import shutil
import subprocess

from agent import coding_context as cc


def test_project_facts_ignore_dotfiles_repo_at_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "main.py").write_text("print('hi')\n")
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(home),
    }

    for args in (
        ["init", "-q", "-b", "main"],
        ["add", "-A"],
        ["commit", "-q", "-m", "init"],
    ):
        subprocess.run([shutil.which("git"), "-C", str(home), *args], check=True, env=env)

    monkeypatch.setattr(Path, "home", lambda: home)
    workspace = home / "scratch"
    workspace.mkdir()

    assert cc.project_facts_for(workspace) is None
