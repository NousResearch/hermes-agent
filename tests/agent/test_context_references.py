from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Hermes Tests")
    _git(repo, "config", "user.email", "tests@example.com")

    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text(
        "def alpha():\n"
        "    return 'a'\n\n"
        "def beta():\n"
        "    return 'b'\n",
        encoding="utf-8",
    )
    (repo / "src" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02binary")

    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")

    (repo / "src" / "main.py").write_text(
        "def alpha():\n"
        "    return 'changed'\n\n"
        "def beta():\n"
        "    return 'b'\n",
        encoding="utf-8",
    )
    (repo / "src" / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "src/helper.py")
    return repo


def test_parse_typed_references_ignores_emails_and_handles():
    from agent.context_references import parse_context_references

    message = (
        "email me at user@example.com and ping @teammate "
        "but include @file:src/main.py:1-2 plus @diff and @git:2 "
        "and @url:https://example.com/docs"
    )

    refs = parse_context_references(message)

    assert [ref.kind for ref in refs] == ["file", "diff", "git", "url"]
    assert refs[0].target == "src/main.py"
    assert refs[0].line_start == 1
    assert refs[0].line_end == 2
    assert refs[2].target == "2"








def test_folder_listing_falls_back_when_rg_is_blocked(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    real_run = subprocess.run

    def blocked_rg(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args")
        if isinstance(cmd, list) and cmd and cmd[0] == "rg":
            raise PermissionError("rg blocked by policy")
        return real_run(*args, **kwargs)

    with patch("agent.context_references.subprocess.run", side_effect=blocked_rg):
        result = preprocess_context_references(
            "Review @folder:src/",
            cwd=sample_repo,
            context_length=100_000,
        )

    assert result.expanded
    assert "src/" in result.message
    assert "main.py" in result.message
    assert "helper.py" in result.message
    assert not result.warnings






def test_missing_file_becomes_warning(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    result = preprocess_context_references(
        "Check @file:nope.txt",
        cwd=sample_repo,
        context_length=100_000,
    )

    assert result.expanded
    assert len(result.warnings) == 1
    assert "not found" in result.message.lower()
















@pytest.mark.asyncio
async def test_blocks_canonical_read_denylist_credential_stores(tmp_path: Path, monkeypatch):
    """@file expansion must honour the canonical read deny-list.

    The narrow in-module list historically missed the real credential stores
    (provider keys, OAuth tokens, MCP tokens, project-local .env). Because the
    gateway routes untrusted remote message text through reference expansion,
    a chat peer could otherwise attach `@file:~/.hermes/auth.json` and read the
    operator's keys into context. These must all be refused, with their secret
    bodies kept out of the expanded message.
    """
    from agent.context_references import preprocess_context_references_async

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    hermes_home = tmp_path / ".hermes"
    (hermes_home).mkdir(parents=True)

    auth_json = hermes_home / "auth.json"
    auth_json.write_text('{"openai": "sk-AUTHJSON-SECRET"}\n', encoding="utf-8")

    oauth = hermes_home / ".anthropic_oauth.json"
    oauth.write_text('{"access_token": "OAUTH-SECRET"}\n', encoding="utf-8")

    mcp_token = hermes_home / "mcp-tokens" / "github.json"
    mcp_token.parent.mkdir(parents=True)
    mcp_token.write_text('{"token": "MCP-TOKEN-SECRET"}\n', encoding="utf-8")

    project_env = tmp_path / "project" / ".env"
    project_env.parent.mkdir(parents=True)
    project_env.write_text("DB_PASSWORD=ENV-SECRET\n", encoding="utf-8")

    result = await preprocess_context_references_async(
        "inspect @file:.hermes/auth.json and @file:.hermes/.anthropic_oauth.json "
        "and @file:.hermes/mcp-tokens/github.json and @file:project/.env",
        cwd=tmp_path,
        allowed_root=tmp_path,
        context_length=100_000,
    )

    assert result.expanded
    for secret in (
        "sk-AUTHJSON-SECRET",
        "OAUTH-SECRET",
        "MCP-TOKEN-SECRET",
        "ENV-SECRET",
    ):
        assert secret not in result.message
    assert sum("sensitive credential" in warning for warning in result.warnings) == 4


@pytest.mark.asyncio
async def test_canonical_guard_fails_closed_when_lookup_raises(tmp_path: Path, monkeypatch):
    """If the canonical read guard raises, the reference must fail CLOSED.

    The guard exists specifically to cover credential stores the narrow local
    list misses (auth.json, ...). If get_read_block_error ever raised, silently
    falling through to the local list would re-open that exact hole — and the
    gateway feeds untrusted remote text here, so a chat peer could then attach
    auth.json. The reference must be refused and the secret kept out of the
    expanded message.
    """
    from agent.context_references import preprocess_context_references_async

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True)
    auth_json = hermes_home / "auth.json"
    auth_json.write_text('{"openai": "sk-AUTHJSON-SECRET"}\n', encoding="utf-8")

    def _boom(_path):
        raise RuntimeError("guard resolution failed")

    monkeypatch.setattr("agent.file_safety.get_read_block_error", _boom)

    result = await preprocess_context_references_async(
        "inspect @file:.hermes/auth.json",
        cwd=tmp_path,
        allowed_root=tmp_path,
        context_length=100_000,
    )

    assert "sk-AUTHJSON-SECRET" not in result.message
    assert any(
        "credential deny-list" in warning or "sensitive credential" in warning
        for warning in result.warnings
    )


@pytest.mark.parametrize(
    "value",
    [
        "/tmp/plain.png",
        "/Users/me/Library/Application Support/Hermes/composer-images/a.png",
        r"C:\Users\John Doe\Pictures\cat.png",
        "/tmp/report (final).pdf",
        "/tmp/it's here.png",
        '/tmp/say "hi".png',
    ],
)
def test_format_reference_value_round_trips_through_the_parser(value):
    """Whatever the path contains, the formatted ref must parse back whole —
    an unquoted value stops at the first space and strands the tail as text."""
    from agent.context_references import REFERENCE_PATTERN, format_reference_value

    match = REFERENCE_PATTERN.search(f"@file:{format_reference_value(value)}")

    assert match is not None
    assert match.group("value").strip("`\"'") == value
