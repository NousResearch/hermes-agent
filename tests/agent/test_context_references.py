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


def test_folder_listing_outside_cwd_inside_widened_allowed_root(tmp_path: Path):
    """allowed_root may be widened beyond cwd; @folder: targets there must
    still produce a listing rather than a ValueError-as-warning."""
    from agent.context_references import preprocess_context_references

    cwd = tmp_path / "proj"
    cwd.mkdir()
    shared = tmp_path / "shared"
    (shared / "sub").mkdir(parents=True)
    (shared / "a.txt").write_text("x\n", encoding="utf-8")
    (shared / "sub" / "b.txt").write_text("y\n", encoding="utf-8")

    result = preprocess_context_references(
        "Review @folder:../shared",
        cwd=cwd,
        allowed_root=tmp_path,
        context_length=100_000,
    )

    assert result.expanded
    assert not result.warnings
    assert "shared/" in result.message
    assert "a.txt" in result.message
    assert "b.txt" in result.message


def test_folder_listing_outside_cwd_rg_blocked_fallback(tmp_path: Path):
    """Same widened-root target with rg unavailable: the os.walk fallback
    never assumed cwd either."""
    from agent.context_references import preprocess_context_references

    cwd = tmp_path / "proj"
    cwd.mkdir()
    shared = tmp_path / "shared"
    (shared / "sub").mkdir(parents=True)
    (shared / "a.txt").write_text("x\n", encoding="utf-8")
    (shared / "sub" / "b.txt").write_text("y\n", encoding="utf-8")

    real_run = subprocess.run

    def blocked_rg(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args")
        if isinstance(cmd, list) and cmd and cmd[0] == "rg":
            raise PermissionError("rg blocked by policy")
        return real_run(*args, **kwargs)

    with patch("agent.context_references.subprocess.run", side_effect=blocked_rg):
        result = preprocess_context_references(
            "Review @folder:../shared",
            cwd=cwd,
            allowed_root=tmp_path,
            context_length=100_000,
        )

    assert result.expanded
    assert not result.warnings
    assert "shared/" in result.message
    assert "b.txt" in result.message


def test_file_reference_outside_cwd_inside_widened_allowed_root(tmp_path: Path):
    """Control: @file: never assumed cwd, so it already worked under a
    widened allowed_root; pin that parity with @folder:."""
    from agent.context_references import preprocess_context_references

    cwd = tmp_path / "proj"
    cwd.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "a.txt").write_text("SHARED-CONTENT\n", encoding="utf-8")

    result = preprocess_context_references(
        "Read @file:../shared/a.txt",
        cwd=cwd,
        allowed_root=tmp_path,
        context_length=100_000,
    )

    assert result.expanded
    assert not result.warnings
    assert "SHARED-CONTENT" in result.message


def test_folder_listing_inside_cwd_unchanged(sample_repo: Path):
    """Control: in-cwd listings keep their cwd-relative display shape."""
    from agent.context_references import preprocess_context_references

    result = preprocess_context_references(
        "Review @folder:src/",
        cwd=sample_repo,
        context_length=100_000,
    )

    assert result.expanded
    assert not result.warnings
    assert "src/" in result.message
    assert "main.py" in result.message






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


def test_oversized_text_file_falls_back_to_tool_readable_path(tmp_path: Path):
    from agent.context_references import preprocess_context_references

    payload = tmp_path / "large.txt"
    payload.write_text("FULL-CONTENT-MARKER\n" + ("x" * 8_000), encoding="utf-8")

    result = preprocess_context_references(
        f"Inspect @file:{payload.name}",
        cwd=tmp_path,
        context_length=1_000,
    )

    assert result.expanded
    assert not result.blocked
    assert str(payload) in result.message
    assert "too large to inline safely" in result.message
    assert "read_file" in result.message
    assert "FULL-CONTENT-MARKER" not in result.message
    # The fallback block alone carries the message — a companion warning would
    # repeat "too large to inline safely" under --- Context Warnings ---.
    assert not result.warnings


def test_file_line_range_is_applied_before_oversized_fallback(tmp_path: Path):
    from agent.context_references import preprocess_context_references

    payload = tmp_path / "large.txt"
    payload.write_text(
        "first line\nsecond line\n" + "\n".join("x" * 200 for _ in range(100)),
        encoding="utf-8",
    )

    result = preprocess_context_references(
        f"Inspect @file:{payload.name}:1-2",
        cwd=tmp_path,
        context_length=1_000,
    )

    assert result.expanded
    assert not result.blocked
    assert "first line\nsecond line" in result.message
    assert "too large to inline safely" not in result.message


def test_oversized_file_refused_without_full_read(tmp_path: Path, monkeypatch):
    from agent import context_references
    from agent.context_references import preprocess_context_references

    payload = tmp_path / "huge.txt"
    payload.write_text("x" * 100_000, encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError("oversized file was read in full")

    monkeypatch.setattr(Path, "read_text", _boom)
    result = preprocess_context_references(
        f"Inspect @file:{payload.name}", cwd=tmp_path, context_length=1_000,
    )
    assert "too large to inline safely" in result.message
    assert str(payload) in result.message


def test_line_range_ref_streams_only_the_window(tmp_path: Path, monkeypatch):
    from agent.context_references import preprocess_context_references

    def _boom(*args, **kwargs):
        raise AssertionError("ranged ref read the whole file")

    monkeypatch.setattr(Path, "read_text", _boom)
    small = tmp_path / "small.txt"
    small.write_text("first\nsecond\nthird\nfourth\n", encoding="utf-8")
    result = preprocess_context_references(
        f"Inspect @file:{small.name}:2-3", cwd=tmp_path, context_length=1_000,
    )
    assert "second\nthird" in result.message
    assert "first" not in result.message
    assert "fourth" not in result.message


def test_binary_sniff_reads_prefix_only(tmp_path: Path):
    from agent.context_references import _is_binary_file

    # .txt keeps the mime guess on the text side so the byte sniff runs.
    payload = tmp_path / "data.txt"
    payload.write_bytes(b"\x00" * 10 + b"x" * 1_000_000)
    read_calls = []
    orig_open = Path.open

    def _counting_open(self, *args, **kwargs):
        fh = orig_open(self, *args, **kwargs)
        if "b" in (args[0] if args else kwargs.get("mode", "r")):
            orig_read = fh.read
            def _read(*a, **k):
                data = orig_read(*a, **k)
                read_calls.append(len(data))
                return data
            fh.read = _read
        return fh

    with patch.object(Path, "open", _counting_open):
        assert _is_binary_file(payload)
    assert read_calls and max(read_calls) <= 4096


def test_folder_listing_caps_line_count_io(tmp_path: Path, monkeypatch):
    from agent import context_references
    from agent.context_references import preprocess_context_references

    assets = tmp_path / "assets"
    assets.mkdir()
    big = assets / "big.txt"
    big.write_bytes(b"line\n" * (context_references._LINE_COUNT_MAX_BYTES // 5 + 2))
    (assets / "small.txt").write_text("a\nb\n", encoding="utf-8")

    result = preprocess_context_references("list @folder:assets", cwd=tmp_path, context_length=10_000_000)
    assert "big.txt" in result.message and "bytes" in result.message
    assert "3 lines" in result.message  # small file still reports a line count


def test_line_range_bounds_reads_on_single_line_file(tmp_path: Path, monkeypatch):
    from agent.context_references import preprocess_context_references

    payload = tmp_path / "oneline.txt"
    payload.write_text("x" * 100_000, encoding="utf-8")  # one giant line, no newline

    read_sizes, returned = [], []
    orig_open = Path.open

    def _counting_open(self, *args, **kwargs):
        fh = orig_open(self, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if "b" not in mode:
            orig_readline = fh.readline

            def _readline(*a, **k):
                read_sizes.append(a[0] if a else -1)
                piece = orig_readline(*a, **k)
                returned.append(len(piece))
                return piece

            fh.readline = _readline
        return fh

    monkeypatch.setattr(Path, "open", _counting_open)
    result = preprocess_context_references(
        f"Inspect @file:{payload.name}:1-1", cwd=tmp_path, context_length=1_000,
    )
    assert "too large to inline safely" in result.message
    # hard_limit = 500 tokens -> char budget 2000 -> each readline bounded at 2001
    assert read_sizes and max(read_sizes) <= 2001
    # ... and the giant line is never materialized: reading stops once the budget is exceeded
    # instead of collecting every 2001-char piece up to the newline (review follow-up).
    assert sum(returned) <= 2 * 2001


def test_run_quiet_caps_child_output(tmp_path: Path):
    import sys
    from agent.context_references import _MAX_QUIET_OUTPUT_BYTES, _run_quiet

    huge = _run_quiet(
        [sys.executable, "-c", f"import sys; sys.stdout.write('x' * {_MAX_QUIET_OUTPUT_BYTES + 1000})"],
        tmp_path, 30,
    )
    assert huge.returncode != 0
    assert len(huge.stdout) <= _MAX_QUIET_OUTPUT_BYTES

    ok = _run_quiet([sys.executable, "-c", "print('hello')"], tmp_path, 30)
    assert ok.returncode == 0 and "hello" in ok.stdout


def test_reference_count_is_capped(tmp_path: Path):
    from agent.context_references import _MAX_EXPANDED_REFERENCES, preprocess_context_references

    for i in range(_MAX_EXPANDED_REFERENCES + 4):
        (tmp_path / f"f{i}.txt").write_text("data", encoding="utf-8")
    result = preprocess_context_references(
        " ".join(f"@file:f{i}.txt" for i in range(_MAX_EXPANDED_REFERENCES + 4)),
        cwd=tmp_path, context_length=10_000_000,
    )
    skipped = [w for w in result.warnings if "not expanded" in w]
    assert len(skipped) == 4


def test_multiple_individually_safe_files_still_obey_aggregate_limit(tmp_path: Path):
    from agent.context_references import preprocess_context_references

    for name in ("first.txt", "second.txt"):
        (tmp_path / name).write_text("x" * 1_200, encoding="utf-8")

    result = preprocess_context_references(
        "Inspect @file:first.txt and @file:second.txt",
        cwd=tmp_path,
        context_length=1_000,
    )

    assert result.blocked
    assert not result.expanded
    assert "context injection refused" in "\n".join(result.warnings)


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


def test_oversized_text_reference_maps_host_attachment_to_container_path(
    tmp_path: Path, monkeypatch
):
    from agent.context_references import preprocess_context_references

    hermes_home = tmp_path / ".hermes"
    attachments = hermes_home / "attachments"
    attachments.mkdir(parents=True)
    payload = attachments / "large.txt"
    payload.write_text("x" * 8_000, encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    result = preprocess_context_references(
        f"Read the attachment @file:{payload}",
        cwd=tmp_path,
        context_length=1_000,
    )

    assert result.expanded
    assert not result.blocked
    attached_context = result.message.split("--- Attached Context ---", 1)[1]
    assert "/root/.hermes/attachments/large.txt" in attached_context
    assert "too large to inline safely" in result.message


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


@pytest.mark.asyncio
async def test_side_thread_expansion_guards_the_served_profile_home(tmp_path: Path, monkeypatch):
    """Inside a running loop (the gateway / TUI turn) the sync wrapper hops to a side thread; that
    thread must inherit the caller's profile scope so the credential guard checks the SERVED
    profile's home, not the launch profile's (a served profile's skill-hub cache was attachable)."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from agent.context_references import preprocess_context_references

    launch_home = tmp_path / "launch"
    served_home = launch_home / "profiles" / "b"
    hub_file = served_home / "skills" / ".hub" / "injected.md"
    hub_file.parent.mkdir(parents=True)
    hub_file.write_text("HUB-CACHE-BODY\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    token = set_hermes_home_override(served_home)
    try:
        result = preprocess_context_references(
            "read @file:profiles/b/skills/.hub/injected.md", cwd=launch_home, allowed_root=launch_home,
            context_length=100_000)
    finally:
        reset_hermes_home_override(token)

    assert "HUB-CACHE-BODY" not in result.message
    assert any("internal Hermes path" in w for w in result.warnings)
