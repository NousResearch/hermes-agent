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


def test_non_utf8_text_file_is_not_dropped_from_context(tmp_path: Path, monkeypatch):
    """A locale-encoded text file must reach the model as something actionable.

    Regression test for #84206. `_is_binary_file` classifies a GB18030 CSV as
    text — correctly: `text/csv` mime, no NUL bytes — so it fell through to a
    strict UTF-8 `read_text`, and the resulting UnicodeDecodeError surfaced as a
    bare warning. The file was on disk and readable, but the model never learned
    it existed.

    GB18030 is only the reproducible case; the same holds for Shift_JIS, Big5,
    or any cp125x export from banking/accounting tooling.

    Uses `.txt` rather than the issue's `.csv` on purpose: `mimetypes` consults
    the Windows registry, where `.csv` resolves to `application/vnd.ms-excel`,
    so `_is_binary_file` short-circuits to the binary block and the decode is
    never reached. `.txt` is `text/plain` on every platform, so this exercises
    the real path everywhere instead of passing vacuously on Windows.
    """
    from agent.context_references import preprocess_context_references

    monkeypatch.setenv("TERMINAL_ENV", "local")
    sample = tmp_path / "gb18030-sample.txt"
    sample.write_bytes("交易时间,商户,金额\n2026-08-11,测试商户,19.80\n".encode("gb18030"))

    result = preprocess_context_references(
        f"Please summarize this CSV @file:{sample}",
        cwd=tmp_path,
        context_length=100_000,
    )

    assert result.expanded
    # It must not degrade into a raw codec error the model can do nothing with.
    assert not any("codec can't decode" in warning for warning in result.warnings), (
        result.warnings
    )
    # It must be told the file is text-but-not-decodable, and where to find it.
    assert "does not decode as utf-8" in result.message.lower()
    assert str(sample) in result.message
    # The offending byte and its offset make the retry one-shot rather than a guess.
    assert "0xbd" in result.message and "offset 0" in result.message
    assert "no byte-order mark" in result.message.lower()
    # Hints must stay hints, and the two lossy escapes must be named and refused.
    assert "gb18030" in result.message
    assert "not a detection result" in result.message
    assert 'errors="replace"' in result.message


def _bom_sample(encoding: str, text: str) -> bytes:
    """Encode ``text`` with an explicit BOM for ``encoding``.

    ``str.encode("utf-16")`` emits the *platform-native* BOM, so the big-endian
    variants have to be assembled by hand to be tested at all.
    """
    import codecs

    explicit = {
        "utf-8-sig": codecs.BOM_UTF8 + text.encode("utf-8"),
        "utf-16-le": codecs.BOM_UTF16_LE + text.encode("utf-16-le"),
        "utf-16-be": codecs.BOM_UTF16_BE + text.encode("utf-16-be"),
        "utf-32-le": codecs.BOM_UTF32_LE + text.encode("utf-32-le"),
        "utf-32-be": codecs.BOM_UTF32_BE + text.encode("utf-32-be"),
    }
    return explicit[encoding]


@pytest.mark.parametrize(
    "encoding",
    ["utf-8-sig", "utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"],
)
def test_bom_marked_unicode_files_are_inlined(tmp_path: Path, monkeypatch, encoding: str):
    """A byte-order mark names the encoding, so these files inline normally.

    Follow-up to #84206. Two separate bugs met here. `_is_binary_file` sniffs for
    NUL bytes, and UTF-16/32 pad ASCII with NULs — so BOM-marked Unicode text was
    classified binary and diverted to the binary block before any decoder saw it.
    Anything that survived that then hit a hardcoded `encoding="utf-8"` read.

    Neither needs guessing to fix: a BOM is a deterministic declaration, unlike
    the locale encodings that legitimately fall through to the actionable block.
    """
    from agent.context_references import preprocess_context_references

    monkeypatch.setenv("TERMINAL_ENV", "local")
    sample = tmp_path / f"{encoding}-sample.txt"
    sample.write_bytes(_bom_sample(encoding, "ledger total\n交易时间,19.80\n"))

    result = preprocess_context_references(
        f"Summarize @file:{sample}",
        cwd=tmp_path,
        context_length=100_000,
    )

    assert result.expanded
    assert not result.warnings, result.warnings
    # Inlined as text — not diverted to the binary or undecodable block.
    assert "ledger total" in result.message
    assert "交易时间,19.80" in result.message
    assert "not inlined" not in result.message
    # The BOM is metadata, not content: the codec must consume it.
    assert "﻿" not in result.message


def test_utf32_le_bom_is_not_claimed_by_the_utf16_probe(tmp_path: Path, monkeypatch):
    """UTF-32 must be probed before UTF-16, because the LE prefixes overlap.

    `BOM_UTF32_LE` is b"\\xff\\xfe\\x00\\x00" and opens with `BOM_UTF16_LE`
    (b"\\xff\\xfe"). Probing UTF-16 first therefore matches every UTF-32 LE file
    and decodes it as UTF-16 — which does not raise, it just yields NUL-padded
    mojibake. That is the silent-corruption failure this fix exists to avoid, so
    it gets a test of its own rather than riding on the parametrized case.
    """
    from agent.context_references import _detect_bom_encoding, preprocess_context_references

    monkeypatch.setenv("TERMINAL_ENV", "local")
    sample = tmp_path / "utf32-sample.txt"
    sample.write_bytes(_bom_sample("utf-32-le", "alpha\n"))

    assert _detect_bom_encoding(sample) == "utf-32"

    result = preprocess_context_references(
        f"Summarize @file:{sample}",
        cwd=tmp_path,
        context_length=100_000,
    )

    assert "alpha" in result.message
    assert "\x00" not in result.message


def test_bom_marked_file_honours_a_line_range(tmp_path: Path, monkeypatch):
    """`@file:...:2-3` must slice BOM-marked text like any other text file."""
    from agent.context_references import preprocess_context_references

    monkeypatch.setenv("TERMINAL_ENV", "local")
    sample = tmp_path / "ranged.txt"
    sample.write_bytes(_bom_sample("utf-16-le", "one\ntwo\nthree\nfour\n"))

    result = preprocess_context_references(
        f"Read @file:`{sample}`:2-3",
        cwd=tmp_path,
        context_length=100_000,
    )

    assert result.expanded
    assert "two" in result.message and "three" in result.message
    assert "one" not in result.message and "four" not in result.message


def test_undecodable_block_reports_the_requested_line_range(tmp_path: Path, monkeypatch):
    """The block must carry the line range forward so the retry is one-shot.

    Without it the agent re-reads the whole file with an explicit encoding and
    has to rediscover which lines were asked for — the range was in the original
    reference and is lost the moment expansion fails.
    """
    from agent.context_references import preprocess_context_references

    monkeypatch.setenv("TERMINAL_ENV", "local")
    sample = tmp_path / "gb18030-ranged.txt"
    sample.write_bytes("交易时间\n测试商户\n金额\n".encode("gb18030"))

    result = preprocess_context_references(
        f"Read @file:`{sample}`:2-3",
        cwd=tmp_path,
        context_length=100_000,
    )

    assert result.expanded
    assert "lines 2-3 requested" in result.message
    assert "does not decode as utf-8" in result.message.lower()
