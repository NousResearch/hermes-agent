"""GitHub CLI path resolution contracts."""
from __future__ import annotations

import os
from pathlib import Path


def test_explicit_gh_binary_is_one_literal_argv_element(tmp_path, monkeypatch):
    from hermes_cli.github_cli import gh_argv, resolve_gh_binary

    binary = tmp_path / "bin with spaces ; $(meta)" / "gh wrapper"
    monkeypatch.setenv("HERMES_GH_BIN", str(binary))
    monkeypatch.setenv("PATH", "")

    assert resolve_gh_binary() == str(binary)
    assert gh_argv("api", "user") == [str(binary), "api", "user"]


def test_gh_binary_falls_back_to_path_and_missing_is_none(tmp_path, monkeypatch):
    from hermes_cli.github_cli import gh_argv, resolve_gh_binary

    monkeypatch.delenv("HERMES_GH_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve_gh_binary() is None
    assert gh_argv("auth", "status") is None

    binary = tmp_path / ("gh.exe" if os.name == "nt" else "gh")
    binary.write_text("stub")
    binary.chmod(0o755)
    assert resolve_gh_binary() == str(binary)
    assert gh_argv("auth", "status") == [str(binary), "auth", "status"]


def test_pr_acceptance_uses_configured_wrapper(tmp_path, monkeypatch):
    from hermes_cli.kanban_pr_acceptance import _api

    log = tmp_path / "argv.json"
    wrapper = tmp_path / "wrapper dir ; safe" / "gh"
    wrapper.parent.mkdir()
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import json,sys\n"
        f"open({str(log)!r},'w').write(json.dumps(sys.argv))\n"
        "print('{}')\n"
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("HERMES_GH_BIN", str(wrapper))
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    assert _api("repos/acme/repo") == {}
    import json
    assert json.loads(log.read_text()) == [str(wrapper), "api", "repos/acme/repo", "--hostname", "github.com"]
