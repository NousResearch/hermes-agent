"""`hermes debug` must not report a shell-only API key as plainly "set".

The dump reads ``os.getenv`` — the invoking terminal's environment — but the
managed backends (launchd / systemd / the desktop-spawned ``serve`` process)
load credentials from ``~/.hermes/.env``, not the login shell. A key exported
in the shell but absent from ``.env`` is invisible to the backend, yet the dump
used to print a bare "set", sending support down a phantom "the key is
configured" path (the real cause behind gated tools like ``web_search`` going
missing on Desktop). The dump now flags that mismatch.
"""

from pathlib import Path
from types import SimpleNamespace


def _api_key_line(out: str, label: str) -> str:
    for line in out.splitlines():
        if line.strip().startswith(f"{label} "):
            return line
    raise AssertionError(f"no '{label}' api_keys line in dump output:\n{out}")


def test_dump_flags_shell_only_key_not_in_dotenv(monkeypatch, capsys, tmp_path):
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    # .env has some OTHER key but NOT firecrawl.
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-or-xxxx\n")
    # firecrawl is exported in the (test) shell only.
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-shell-only")

    dump.run_dump(SimpleNamespace(show_keys=False))

    line = _api_key_line(capsys.readouterr().out, "firecrawl")
    assert "set" in line
    assert "shell only" in line
    assert ".env" in line


def test_dump_does_not_flag_key_present_in_dotenv(monkeypatch, capsys, tmp_path):
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text("FIRECRAWL_API_KEY=fc-in-dotenv\n")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-in-dotenv")

    dump.run_dump(SimpleNamespace(show_keys=False))

    line = _api_key_line(capsys.readouterr().out, "firecrawl")
    assert "set" in line
    assert "shell only" not in line


def test_dump_leaves_unset_key_untouched(monkeypatch, capsys, tmp_path):
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-or-xxxx\n")

    dump.run_dump(SimpleNamespace(show_keys=False))

    line = _api_key_line(capsys.readouterr().out, "tavily")
    assert "not set" in line
    assert "shell only" not in line
