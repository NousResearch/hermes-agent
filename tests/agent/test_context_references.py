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


def test_parse_references_strips_trailing_punctuation():
    from agent.context_references import parse_context_references

    refs = parse_context_references(
        "review @file:README.md, then see (@url:https://example.com/docs)."
    )

    assert [ref.kind for ref in refs] == ["file", "url"]
    assert refs[0].target == "README.md"
    assert refs[1].target == "https://example.com/docs"


def test_parse_quoted_references_with_spaces_and_preserve_unquoted_ranges():
    from agent.context_references import parse_context_references

    refs = parse_context_references(
        'review @file:"C:\\Users\\Simba\\My Project\\main.py":7-9 '
        'and @folder:"docs and specs" plus @file:src/main.py:1-2'
    )

    assert [ref.kind for ref in refs] == ["file", "folder", "file"]
    assert refs[0].target == r"C:\Users\Simba\My Project\main.py"
    assert refs[0].line_start == 7
    assert refs[0].line_end == 9
    assert refs[1].target == "docs and specs"
    assert refs[2].target == "src/main.py"
    assert refs[2].line_start == 1
    assert refs[2].line_end == 2


def test_expand_file_range_and_folder_listing(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    result = preprocess_context_references(
        "Review @file:src/main.py:1-2 and @folder:src/",
        cwd=sample_repo,
        context_length=100_000,
    )

    assert result.expanded
    assert "Review and" in result.message
    assert "Review @file:src/main.py:1-2" not in result.message
    assert "--- Attached Context ---" in result.message
    assert "def alpha():" in result.message
    assert "return 'changed'" in result.message
    assert "def beta():" not in result.message
    assert "src/" in result.message
    assert "main.py" in result.message
    assert "helper.py" in result.message
    assert result.injected_tokens > 0
    assert not result.warnings


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


def test_expand_quoted_file_reference_with_spaces(tmp_path: Path):
    from agent.context_references import preprocess_context_references

    workspace = tmp_path / "repo"
    folder = workspace / "docs and specs"
    folder.mkdir(parents=True)
    file_path = folder / "release notes.txt"
    file_path.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    result = preprocess_context_references(
        'Review @file:"docs and specs/release notes.txt":2-3',
        cwd=workspace,
        context_length=100_000,
    )

    assert result.expanded
    assert result.message.startswith("Review")
    assert "line 1" not in result.message
    assert "line 2" in result.message
    assert "line 3" in result.message
    assert "release notes.txt" in result.message
    assert not result.warnings


def test_expand_git_diff_staged_and_log(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    result = preprocess_context_references(
        "Inspect @diff and @staged and @git:1",
        cwd=sample_repo,
        context_length=100_000,
    )

    assert result.expanded
    assert "git diff" in result.message
    assert "git diff --staged" in result.message
    assert "git log -1 -p" in result.message
    assert "initial" in result.message
    assert "return 'changed'" in result.message
    assert "VALUE = 2" in result.message


def test_binary_and_missing_files_become_warnings(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    result = preprocess_context_references(
        "Check @file:blob.bin and @file:nope.txt",
        cwd=sample_repo,
        context_length=100_000,
    )

    assert result.expanded
    assert len(result.warnings) == 2
    assert "binary" in result.message.lower()
    assert "not found" in result.message.lower()


def test_soft_budget_warns_and_hard_budget_refuses(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    soft = preprocess_context_references(
        "Check @file:src/main.py",
        cwd=sample_repo,
        context_length=100,
    )
    assert soft.expanded
    assert any("25%" in warning for warning in soft.warnings)

    hard = preprocess_context_references(
        "Check @file:src/main.py and @file:README.md",
        cwd=sample_repo,
        context_length=20,
    )
    assert not hard.expanded
    assert hard.blocked
    assert "@file:src/main.py" in hard.message
    assert any("50%" in warning for warning in hard.warnings)


@pytest.mark.asyncio
async def test_async_url_expansion_uses_fetcher(sample_repo: Path):
    from agent.context_references import preprocess_context_references_async

    async def fake_fetch(url: str) -> str:
        assert url == "https://example.com/spec"
        return "# Spec\n\nImportant details."

    result = await preprocess_context_references_async(
        "Use @url:https://example.com/spec",
        cwd=sample_repo,
        context_length=100_000,
        url_fetcher=fake_fetch,
    )

    assert result.expanded
    assert "Important details." in result.message
    assert result.injected_tokens > 0


def test_sync_url_expansion_uses_async_fetcher(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    async def fake_fetch(url: str) -> str:
        await asyncio.sleep(0)
        return f"Content for {url}"

    result = preprocess_context_references(
        "Use @url:https://example.com/spec",
        cwd=sample_repo,
        context_length=100_000,
        url_fetcher=fake_fetch,
    )

    assert result.expanded
    assert "Content for https://example.com/spec" in result.message


def test_restricts_paths_to_allowed_root(tmp_path: Path):
    from agent.context_references import preprocess_context_references

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("inside\n", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("outside\n", encoding="utf-8")

    result = preprocess_context_references(
        "read @file:../secret.txt and @file:notes.txt",
        cwd=workspace,
        context_length=100_000,
        allowed_root=workspace,
    )

    assert result.expanded
    assert "```\noutside\n```" not in result.message
    assert "inside" in result.message
    assert any("outside the allowed workspace" in warning for warning in result.warnings)


def test_defaults_allowed_root_to_cwd(tmp_path: Path):
    from agent.context_references import preprocess_context_references

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("outside\n", encoding="utf-8")

    result = preprocess_context_references(
        f"read @file:{secret}",
        cwd=workspace,
        context_length=100_000,
    )

    assert result.expanded
    assert "```\noutside\n```" not in result.message
    assert any("outside the allowed workspace" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_blocks_sensitive_home_and_hermes_paths(tmp_path: Path, monkeypatch):
    from agent.context_references import preprocess_context_references_async

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    hermes_env = tmp_path / ".hermes" / ".env"
    hermes_env.parent.mkdir(parents=True)
    hermes_env.write_text("API_KEY=super-secret\n", encoding="utf-8")

    ssh_key = tmp_path / ".ssh" / "id_rsa"
    ssh_key.parent.mkdir(parents=True)
    ssh_key.write_text("PRIVATE-KEY\n", encoding="utf-8")

    result = await preprocess_context_references_async(
        "read @file:.hermes/.env and @file:.ssh/id_rsa",
        cwd=tmp_path,
        allowed_root=tmp_path,
        context_length=100_000,
    )

    assert result.expanded
    assert "API_KEY=super-secret" not in result.message
    assert "PRIVATE-KEY" not in result.message
    assert any("sensitive credential" in warning for warning in result.warnings)
