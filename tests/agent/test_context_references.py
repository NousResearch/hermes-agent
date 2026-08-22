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
async def test_permissions_deny_blocks_file_reference_before_content_read(
    tmp_path: Path,
):
    from agent.context_references import preprocess_context_references_async

    secret = tmp_path / "secret.txt"
    secret.write_text("REFERENCE SECRET", encoding="utf-8")
    original_read_text = Path.read_text

    def guarded_read_text(path_obj, *args, **kwargs):
        if path_obj == secret:
            raise AssertionError("denied reference content was read")
        return original_read_text(path_obj, *args, **kwargs)

    with (
        patch(
            "agent.deny_policy.permissions_deny_paths",
            return_value=[str(secret)],
        ),
        patch.object(Path, "read_text", guarded_read_text),
    ):
        result = await preprocess_context_references_async(
            "inspect @file:secret.txt",
            cwd=tmp_path,
            context_length=100_000,
        )

    assert "REFERENCE SECRET" not in result.message
    assert any("permissions.deny.paths" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_permissions_deny_relative_rule_blocks_file_reference(
    tmp_path: Path,
):
    from agent.context_references import preprocess_context_references_async

    secret = tmp_path / "secret.txt"
    secret.write_text("RELATIVE RULE SECRET", encoding="utf-8")

    with patch(
        "agent.deny_policy.permissions_deny_paths",
        return_value=["secret.txt"],
    ):
        result = await preprocess_context_references_async(
            "inspect @file:secret.txt",
            cwd=tmp_path,
            context_length=100_000,
        )

    assert "RELATIVE RULE SECRET" not in result.message
    assert any("permissions.deny.paths" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_permissions_deny_globbed_directory_blocks_descendant_reference(
    tmp_path: Path,
):
    from agent.context_references import preprocess_context_references_async

    private = tmp_path / "private1"
    private.mkdir()
    secret = private / "notes.txt"
    secret.write_text("GLOBBED DIRECTORY SECRET", encoding="utf-8")

    with patch(
        "agent.deny_policy.permissions_deny_paths",
        return_value=[str(tmp_path / "private?")],
    ):
        result = await preprocess_context_references_async(
            "inspect @file:private1/notes.txt",
            cwd=tmp_path,
            context_length=100_000,
        )

    assert "GLOBBED DIRECTORY SECRET" not in result.message
    assert any("permissions.deny.paths" in warning for warning in result.warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reference", "patterns"),
    [
        ("@file:secret.txt", ["secret.txt"]),
        ("@folder:private", ["private/**"]),
    ],
)
async def test_permissions_deny_blocks_reference_before_path_resolution(
    tmp_path: Path,
    reference: str,
    patterns: list[str],
):
    from agent.context_references import preprocess_context_references_async

    with (
        patch(
            "agent.deny_policy.permissions_deny_paths",
            return_value=patterns,
        ),
        patch(
            "agent.context_references._resolve_path",
            side_effect=AssertionError("reference resolved before lexical deny"),
        ) as mock_resolve,
    ):
        result = await preprocess_context_references_async(
            f"inspect {reference}",
            cwd=tmp_path,
            context_length=100_000,
        )

    mock_resolve.assert_not_called()
    assert any("permissions.deny.paths" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_permissions_deny_preflights_cwd_ancestors_before_resolve(
    tmp_path: Path,
):
    from agent.context_references import preprocess_context_references_async

    denied_ancestor = tmp_path / "private1"
    cwd = denied_ancestor / "src"
    cwd.mkdir(parents=True)
    original_resolve = Path.resolve

    def guarded_resolve(path_obj, *args, **kwargs):
        if path_obj == denied_ancestor or denied_ancestor in path_obj.parents:
            raise AssertionError("denied cwd ancestry resolved before lexical preflight")
        return original_resolve(path_obj, *args, **kwargs)

    with (
        patch(
            "agent.deny_policy.permissions_deny_paths",
            return_value=[str(tmp_path / "private?")],
        ),
        patch.object(Path, "resolve", autospec=True, side_effect=guarded_resolve),
    ):
        result = await preprocess_context_references_async(
            "inspect @file:notes.txt",
            cwd=cwd,
            context_length=100_000,
        )

    assert result.blocked
    assert any("permissions.deny.paths" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_file_reference_rechecks_resolved_sensitive_home_alias(tmp_path: Path):
    import os

    from agent.context_references import preprocess_context_references_async

    home = tmp_path / "home"
    protected = home / ".aws"
    protected.mkdir(parents=True)
    secret = protected / "credentials"
    secret.write_text("AWS ALIAS CREDENTIAL SECRET", encoding="utf-8")
    alias = tmp_path / "notes.txt"
    alias.symlink_to(secret)
    original_expanduser = os.path.expanduser

    def fake_expanduser(value):
        return str(home) if value == "~" else original_expanduser(value)

    with (
        patch("agent.context_references.os.path.expanduser", fake_expanduser),
        patch("agent.deny_policy.permissions_deny_paths", return_value=[]),
    ):
        result = await preprocess_context_references_async(
            "inspect @file:notes.txt",
            cwd=tmp_path,
            allowed_root=tmp_path,
            context_length=100_000,
        )

    assert "AWS ALIAS CREDENTIAL SECRET" not in result.message
    assert any("sensitive" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_folder_reference_rechecks_resolved_sensitive_home_alias(tmp_path: Path):
    import os

    from agent.context_references import preprocess_context_references_async

    home = tmp_path / "home"
    protected = home / ".aws"
    protected.mkdir(parents=True)
    (protected / "credentials").write_text("FOLDER ALIAS SECRET")
    alias = tmp_path / "docs"
    alias.symlink_to(protected, target_is_directory=True)
    original_expanduser = os.path.expanduser

    def fake_expanduser(value):
        return str(home) if value == "~" else original_expanduser(value)

    with (
        patch("agent.context_references.os.path.expanduser", fake_expanduser),
        patch("agent.deny_policy.permissions_deny_paths", return_value=[]),
    ):
        result = await preprocess_context_references_async(
            "inspect @folder:docs",
            cwd=tmp_path,
            allowed_root=tmp_path,
            context_length=100_000,
        )

    assert "credentials" not in result.message
    assert any("sensitive" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_permissions_deny_blocks_git_reference_before_subprocess(tmp_path: Path):
    from agent.context_references import preprocess_context_references_async

    root = tmp_path / "repo"
    cwd = root / "src"
    denied = root / "private"
    cwd.mkdir(parents=True)
    denied.mkdir()
    git_marker = root / ".git"
    original_exists = Path.exists

    def guarded_exists(path_obj):
        if path_obj == git_marker:
            raise AssertionError("denied-overlap ancestor .git was probed")
        return original_exists(path_obj)

    denied_rule = str(denied / "**")
    with (
        patch(
            "agent.deny_policy.permissions_deny_paths",
            return_value=[denied_rule],
        ),
        patch.object(Path, "exists", guarded_exists),
        patch("agent.context_references.subprocess.run") as mock_run,
    ):
        result = await preprocess_context_references_async(
            "review @diff",
            cwd=cwd,
            context_length=100_000,
        )

    mock_run.assert_not_called()
    assert any("permissions.deny.paths" in warning for warning in result.warnings)
    assert all(denied_rule not in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_folder_deny_warning_does_not_disclose_descendant_rule(tmp_path: Path):
    from agent.context_references import preprocess_context_references_async

    root = tmp_path / "workspace"
    denied = root / "private" / "vault"
    denied.mkdir(parents=True)

    denied_rule = str(denied / "**")
    with patch(
        "agent.deny_policy.permissions_deny_paths",
        return_value=[denied_rule],
    ):
        result = await preprocess_context_references_async(
            f"inspect @folder:{root}",
            cwd=tmp_path,
            context_length=100_000,
        )

    assert any("permissions.deny.paths" in warning for warning in result.warnings)
    assert all(denied_rule not in warning for warning in result.warnings)


def test_binary_reference_block_maps_host_attachment_to_container_path(tmp_path: Path, monkeypatch):
    """Docker backend: a staged binary attachment's host path is rendered as the
    bind-mounted in-container path so the agent's tools can read it.

    Regression test for #76577 — the container has its own filesystem, so the
    gateway host path would dangle inside the sandbox.
    """
    from agent.context_references import preprocess_context_references

    hermes_home = tmp_path / ".hermes"
    attachments = hermes_home / "attachments"
    attachments.mkdir(parents=True)
    payload = attachments / "archive.zip"
    payload.write_bytes(b"PK\x03\x04binary-zip-bytes")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    result = preprocess_context_references(
        f"Read the attachment @file:{payload}",
        cwd=tmp_path,
        context_length=100_000,
    )

    assert result.expanded
    # Default container base for the docker backend is /root/.hermes.
    assert "/root/.hermes/attachments/archive.zip" in result.message
    assert "binary file, not inlined" in result.message


def test_binary_reference_block_keeps_host_path_on_local_backend(tmp_path: Path, monkeypatch):
    """Local backend: no translation — the agent's tools run on the host."""
    from agent.context_references import preprocess_context_references

    hermes_home = tmp_path / ".hermes"
    attachments = hermes_home / "attachments"
    attachments.mkdir(parents=True)
    payload = attachments / "archive.zip"
    payload.write_bytes(b"PK\x03\x04binary-zip-bytes")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")

    result = preprocess_context_references(
        f"Read the attachment @file:{payload}",
        cwd=tmp_path,
        context_length=100_000,
    )

    assert result.expanded
    assert str(payload) in result.message
    assert "/root/.hermes/attachments/" not in result.message


@pytest.mark.asyncio
async def test_permissions_deny_blocks_folder_with_denied_descendant_before_listing(
    tmp_path: Path,
):
    from agent.context_references import preprocess_context_references_async

    folder = tmp_path / "project"
    secret = folder / "private" / "secret.txt"
    secret.parent.mkdir(parents=True)
    secret.write_text("FOLDER SECRET", encoding="utf-8")

    with (
        patch(
            "agent.deny_policy.permissions_deny_paths",
            return_value=[str(secret)],
        ),
        patch(
            "agent.context_references._build_folder_listing",
            side_effect=AssertionError("denied folder was enumerated"),
        ),
    ):
        result = await preprocess_context_references_async(
            "inspect @folder:project",
            cwd=tmp_path,
            context_length=100_000,
        )

    assert "FOLDER SECRET" not in result.message
    assert any("permissions.deny.paths" in warning for warning in result.warnings)


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
