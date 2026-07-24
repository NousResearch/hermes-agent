"""Tests for HERMES_HOME credential-file read blocking in file_safety.

Regression for https://github.com/NousResearch/hermes-agent/issues/17656 —
``read_file`` was previously only sandboxed against ``HERMES_HOME`` itself,
which left ``auth.json`` and ``.anthropic_oauth.json`` (plaintext provider
keys + OAuth tokens) readable by the agent. A prompt-injection reaching
``read_file`` could exfiltrate active credentials.

These tests verify that ``get_read_block_error`` returns a denial message
for the credential stores while leaving arbitrary ``HERMES_HOME`` files
readable, and that the existing ``skills/.hub`` deny still applies.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """Point ``_hermes_home_path()`` at a tmp dir for isolated checks."""
    import agent.file_safety as fs

    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: home)
    return home


def _create(home: Path, rel: str | Path) -> Path:
    """Create the file (with parents) so realpath() resolves it."""
    p = home / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("dummy", encoding="utf-8")
    return p


def test_auth_json_blocked(fake_home):
    from agent.file_safety import get_read_block_error

    auth = _create(fake_home, "auth.json")
    err = get_read_block_error(str(auth))
    assert err is not None
    assert "credential store" in err
    assert "auth.json" in err


def test_auth_lock_blocked(fake_home):
    from agent.file_safety import get_read_block_error

    lock = _create(fake_home, "auth.lock")
    err = get_read_block_error(str(lock))
    assert err is not None
    assert "credential store" in err


def test_anthropic_oauth_json_blocked(fake_home):
    from agent.file_safety import get_read_block_error

    oauth = _create(fake_home, ".anthropic_oauth.json")
    err = get_read_block_error(str(oauth))
    assert err is not None
    assert "credential store" in err


def test_google_oauth_json_blocked(fake_home):
    """Gemini OAuth tokens live under auth/google_oauth.json — blocked."""
    from agent.file_safety import get_read_block_error

    oauth = _create(fake_home, Path("auth") / "google_oauth.json")
    err = get_read_block_error(str(oauth))
    assert err is not None
    assert "credential store" in err


def test_arbitrary_hermes_home_file_not_blocked(fake_home):
    """Non-credential files inside HERMES_HOME stay readable."""
    from agent.file_safety import get_read_block_error

    safe = _create(fake_home, "session_log.txt")
    assert get_read_block_error(str(safe)) is None


def test_subdirectory_named_auth_json_not_blocked(fake_home):
    """Only the top-level auth.json is the credential store; a file with the
    same name in a subdirectory (e.g., a skill mock) must remain readable."""
    from agent.file_safety import get_read_block_error

    nested = _create(fake_home, Path("skills") / "my-skill" / "auth.json")
    assert get_read_block_error(str(nested)) is None


def test_skills_hub_block_still_applies(fake_home):
    """Regression guard: the original skills/.hub deny must keep working."""
    from agent.file_safety import get_read_block_error

    hub_file = _create(fake_home, "skills/.hub/manifest.json")
    err = get_read_block_error(str(hub_file))
    assert err is not None
    assert "internal Hermes cache file" in err


def test_path_traversal_resolves_to_blocked(fake_home, tmp_path):
    """A path that traverses through a sibling dir back into HERMES_HOME's
    auth.json must still be caught — the check resolves through realpath."""
    from agent.file_safety import get_read_block_error

    _create(fake_home, "auth.json")
    sibling = tmp_path / "elsewhere"
    sibling.mkdir()
    traversal = sibling / ".." / "hermes_home" / "auth.json"
    err = get_read_block_error(str(traversal))
    assert err is not None
    assert "credential store" in err


def test_symlink_to_auth_json_blocked(fake_home, tmp_path):
    """A symlink pointing at HERMES_HOME/auth.json from outside the home
    must be blocked — readlink-resolution catches the indirection."""
    from agent.file_safety import get_read_block_error

    target = _create(fake_home, "auth.json")
    link = tmp_path / "shim.json"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform/filesystem")
    err = get_read_block_error(str(link))
    assert err is not None
    assert "credential store" in err


def test_read_file_tool_blocks_relative_path_under_terminal_cwd(
    fake_home, tmp_path, monkeypatch
):
    """Bypass guard: a relative path like ``"auth.json"`` resolved by
    ``read_file_tool`` against ``TERMINAL_CWD == HERMES_HOME`` must still
    be blocked, even though ``get_read_block_error``'s own ``resolve()``
    is anchored at the (different) Python process cwd.
    """
    import json

    import tools.file_tools as ft
    import tools.terminal_tool as terminal_tool

    _create(fake_home, "auth.json")
    # Force the file_tools resolver to anchor relative paths at HERMES_HOME
    # while the Python process cwd remains tmp_path (a different directory).
    monkeypatch.setenv("TERMINAL_CWD", str(fake_home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        terminal_tool, "_session_cwd", {}
    )

    out = json.loads(ft.read_file_tool("auth.json"))
    assert "error" in out
    assert "credential store" in out["error"]


def test_read_file_tool_blocks_nested_google_oauth_path(
    fake_home, tmp_path, monkeypatch
):
    """The real read_file tool must not return Gemini OAuth token material."""
    import json

    import tools.file_tools as ft
    import tools.terminal_tool as terminal_tool

    oauth = _create(fake_home, Path("auth") / "google_oauth.json")
    oauth.write_text(
        json.dumps(
            {
                "refresh": "REFRESH_TOKEN_MARKER",
                "access": "ACCESS_TOKEN_MARKER",
                "email": "user@example.com",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        terminal_tool, "_session_cwd", {}
    )

    out = json.loads(ft.read_file_tool(str(oauth), task_id="google-oauth-test"))
    assert "error" in out
    assert "credential store" in out["error"]
    assert "REFRESH_TOKEN_MARKER" not in json.dumps(out)
    assert "ACCESS_TOKEN_MARKER" not in json.dumps(out)


def test_search_tool_blocks_direct_auth_json_path(fake_home, monkeypatch):
    """Searching a credential file directly must not invoke the search backend."""
    import json

    import tools.file_tools as ft
    import tools.terminal_tool as terminal_tool

    auth = _create(fake_home, "auth.json")
    auth.write_text("SEARCH_DIRECT_AUTH_SECRET", encoding="utf-8")

    def fail_if_called(task_id="default"):
        raise AssertionError("search backend should not run for blocked path")

    monkeypatch.setattr(ft, "_get_file_ops", fail_if_called)

    out = json.loads(
        ft.search_tool(
            pattern="SEARCH_DIRECT_AUTH_SECRET",
            path=str(auth),
            task_id="search-direct-auth-json",
        )
    )
    raw = json.dumps(out)
    assert "error" in out
    assert "credential store" in out["error"]
    assert "SEARCH_DIRECT_AUTH_SECRET" not in raw


def test_search_tool_filters_credential_results(fake_home, tmp_path, monkeypatch):
    """Directory searches omit credential and MCP-token result entries."""
    import json

    from tools.file_operations import SearchMatch, SearchResult
    import tools.file_tools as ft
    import tools.terminal_tool as terminal_tool

    auth = _create(fake_home, "auth.json")
    token = _create(fake_home, Path("mcp-tokens") / "provider.json")
    safe = _create(fake_home, "notes.txt")

    class FakeFileOps:
        def search(self, **kwargs):
            return SearchResult(
                matches=[
                    SearchMatch(
                        path=str(auth),
                        line_number=1,
                        content="SEARCH_AUTH_SECRET",
                    ),
                    SearchMatch(
                        path=str(token),
                        line_number=1,
                        content="SEARCH_MCP_SECRET",
                    ),
                    SearchMatch(
                        path=str(safe),
                        line_number=1,
                        content="public note",
                    ),
                ],
                files=[str(auth), str(token), str(safe)],
                total_count=5,
                truncated=True,
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": FakeFileOps())
    monkeypatch.setattr(
        terminal_tool, "_session_cwd", {}
    )

    search_response = ft.search_tool(
        pattern="SEARCH",
        path=str(fake_home),
        task_id="search-filter-credentials",
    )
    out = json.loads(search_response.split("\n\n[Hint:", 1)[0])
    raw = json.dumps(out)
    returned_paths = {
        match["path"] for match in out.get("matches", [])
    } | set(out.get("files", []))

    assert "SEARCH_AUTH_SECRET" not in raw
    assert "SEARCH_MCP_SECRET" not in raw
    assert str(auth) not in returned_paths
    assert str(token) not in returned_paths
    assert "public note" in raw
    assert str(safe) in returned_paths
    assert out["_omitted"].startswith("4 result(s) omitted")
    assert out["total_count"] == 5
    assert out["truncated"] is True
    assert "[Hint: Results truncated." in search_response


# ---------------------------------------------------------------------------
# Widening: .env, webhook_subscriptions.json, mcp-tokens/
# ---------------------------------------------------------------------------


def test_dotenv_blocked(fake_home):
    """.env in HERMES_HOME holds API keys — blocked."""
    from agent.file_safety import get_read_block_error

    env = _create(fake_home, ".env")
    err = get_read_block_error(str(env))
    assert err is not None
    assert "credential store" in err


def test_webhook_subscriptions_blocked(fake_home):
    """webhook_subscriptions.json holds per-route HMAC secrets — blocked."""
    from agent.file_safety import get_read_block_error

    subs = _create(fake_home, "webhook_subscriptions.json")
    err = get_read_block_error(str(subs))
    assert err is not None
    assert "credential store" in err


def test_mcp_tokens_file_blocked(fake_home):
    """Files under mcp-tokens/ hold OAuth tokens — blocked."""
    from agent.file_safety import get_read_block_error

    tok = _create(fake_home, Path("mcp-tokens") / "github.json")
    err = get_read_block_error(str(tok))
    assert err is not None
    assert "MCP token" in err


def test_mcp_tokens_nested_blocked(fake_home):
    """Nested files inside mcp-tokens/ are also blocked."""
    from agent.file_safety import get_read_block_error

    tok = _create(fake_home, Path("mcp-tokens") / "providers" / "azure.json")
    err = get_read_block_error(str(tok))
    assert err is not None
    assert "MCP token" in err


def test_mcp_tokens_dir_itself_blocked(fake_home):
    """The mcp-tokens directory itself is blocked (listing is exfiltrating)."""
    from agent.file_safety import get_read_block_error

    tokens_dir = fake_home / "mcp-tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    err = get_read_block_error(str(tokens_dir))
    assert err is not None
    assert "MCP token" in err


def test_identically_named_hermes_files_outside_home_not_blocked(
    fake_home, tmp_path
):
    """Hermes-specific filenames (``auth.json``, ``mcp-tokens/``, ``google_oauth.json``)
    outside HERMES_HOME must remain readable — the gate is per-location for
    those, not per-filename. ``.env`` is the exception: it's blocked anywhere
    on disk (see test_project_local_env_blocked) because the basename always
    means \"secret-bearing environment file\" regardless of directory."""
    from agent.file_safety import get_read_block_error

    project = tmp_path / "myproject"
    project.mkdir()
    # auth.json outside HERMES_HOME — readable (per-location gate).
    p = project / "auth.json"
    p.write_text("not secret here", encoding="utf-8")
    assert get_read_block_error(str(p)) is None, (
        "auth.json outside HERMES_HOME should NOT be blocked"
    )

    google_oauth = project / "auth" / "google_oauth.json"
    google_oauth.parent.mkdir()
    google_oauth.write_text("not really a token", encoding="utf-8")
    assert get_read_block_error(str(google_oauth)) is None

    tokens = project / "mcp-tokens"
    tokens.mkdir()
    tok_file = tokens / "token.json"
    tok_file.write_text("not really a token", encoding="utf-8")
    assert get_read_block_error(str(tok_file)) is None


def test_non_secret_auth_subtree_file_not_blocked(fake_home):
    """Only the known Google OAuth token path is blocked, not all auth/*."""
    from agent.file_safety import get_read_block_error

    note = _create(fake_home, Path("auth") / "notes.json")
    assert get_read_block_error(str(note)) is None


def test_config_yaml_not_blocked(fake_home):
    """config.yaml is NOT a credential file — agent should still be
    able to read it for debugging.  (Writes are denied separately by
    is_write_denied; reads stay allowed.)"""
    from agent.file_safety import get_read_block_error

    cfg = _create(fake_home, "config.yaml")
    assert get_read_block_error(str(cfg)) is None


def test_profile_mode_blocks_root_credentials(tmp_path, monkeypatch):
    """Under a profile, HERMES_HOME = <root>/profiles/<name>, but
    <root>/auth.json must ALSO be blocked — credentials at root are
    inherited by every profile."""
    import agent.file_safety as fs

    root = tmp_path / "hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: profile)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)

    from agent.file_safety import get_read_block_error

    # Profile-local credential store: blocked
    profile_auth = profile / "auth.json"
    profile_auth.write_text("x")
    assert "credential store" in (get_read_block_error(str(profile_auth)) or "")

    # Root-level credential store: ALSO blocked (this is the widening)
    root_auth = root / "auth.json"
    root_auth.write_text("x")
    assert "credential store" in (get_read_block_error(str(root_auth)) or "")

    # Root-level .env: blocked too
    root_env = root / ".env"
    root_env.write_text("x")
    assert "credential store" in (get_read_block_error(str(root_env)) or "")

    # Root-level Google OAuth token store: blocked too
    root_google_oauth = root / "auth" / "google_oauth.json"
    root_google_oauth.parent.mkdir(parents=True, exist_ok=True)
    root_google_oauth.write_text("x")
    assert "credential store" in (
        get_read_block_error(str(root_google_oauth)) or ""
    )

    # Root-level mcp-tokens: blocked
    root_tok = root / "mcp-tokens" / "gh.json"
    root_tok.parent.mkdir(parents=True, exist_ok=True)
    root_tok.write_text("x")
    assert "MCP token" in (get_read_block_error(str(root_tok)) or "")


# ---------------------------------------------------------------------------
# Broadened read-deny: secret-bearing credential material anywhere on disk
# ---------------------------------------------------------------------------


class TestBroadenedSecretReadDeny:
    """Credential material beyond .env is denied to read_file.

    Cloud service-account JSON keys, private keys, and cloud-CLI credential
    directories (``~/.aws``, ``~/.ssh``, ``~/.config/gcloud`` …) live outside
    HERMES_HOME, so the existing per-location gates never saw them. A prompt
    injection reaching ``read_file`` could exfiltrate them verbatim. These
    assert the broadened ``_looks_like_secret_read`` denial — defense-in-depth,
    not a boundary (the terminal tool can still read these paths).
    """

    def test_gcloud_service_account_json_denied(self):
        from agent.file_safety import get_read_block_error
        err = get_read_block_error("~/.config/gcloud/my-app-service-account.json")
        assert err is not None
        assert "secret" in err.lower()

    def test_aws_credentials_denied(self):
        from agent.file_safety import get_read_block_error
        assert get_read_block_error("~/.aws/credentials") is not None

    def test_ssh_private_key_denied(self):
        from agent.file_safety import get_read_block_error
        assert get_read_block_error("~/.ssh/id_rsa") is not None

    def test_private_key_pem_denied(self):
        """A private-key-named PEM (privkey.pem) stays denied after the
        blanket ``.pem`` deny is replaced by the private-key name heuristic."""
        from agent.file_safety import get_read_block_error
        assert get_read_block_error("/tmp/some/privkey.pem") is not None

    def test_mounted_service_account_denied(self):
        from agent.file_safety import get_read_block_error
        assert get_read_block_error("/secrets/prod-service-account.json") is not None

    def test_ordinary_json_allowed(self):
        from agent.file_safety import get_read_block_error
        assert get_read_block_error("/tmp/project/data.json") is None

    def test_env_example_still_allowed(self):
        from agent.file_safety import get_read_block_error
        assert get_read_block_error("/tmp/project/.env.example") is None

    def test_underscore_service_account_denied(self):
        from agent.file_safety import get_read_block_error
        assert get_read_block_error("/secrets/foo_service_account.json") is not None


class TestSecretReadWindowsPaths:
    """FIX 1 — the dir-segment check must use path *components*, not POSIX
    substring matching, so it survives Windows backslash separators.

    The helper receives an already-resolved ``Path``. On Windows that is a
    ``WindowsPath`` whose ``.parts`` split on ``\\``; the old substring check
    (``"/.aws/" in str(resolved).lower()``) never matched a backslash path
    even on Windows. We drive ``_looks_like_secret_read`` with a
    ``PureWindowsPath`` to reproduce the Windows ``.parts`` shape on any host.
    """

    def test_windows_aws_config_denied(self):
        from pathlib import PureWindowsPath

        from agent.file_safety import _looks_like_secret_read

        # config is NOT a denied basename, so a hit here is purely the
        # ``.aws`` directory-component rule (not the basename rule).
        p = PureWindowsPath(r"C:\Users\bob\.aws\config")
        assert _looks_like_secret_read(p) is True

    def test_windows_ssh_dir_denied(self):
        from pathlib import PureWindowsPath

        from agent.file_safety import _looks_like_secret_read

        p = PureWindowsPath(r"C:\Users\bob\.ssh\known_hosts")
        assert _looks_like_secret_read(p) is True

    def test_ssh_config_still_denied(self):
        """Regression anchor for the parts refactor: a POSIX ``~/.ssh/config``
        (non-basename file under ``.ssh``) must stay denied."""
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("~/.ssh/config") is not None

    def test_gcloud_tokens_denied(self):
        """gcloud dir denies only on ``.config`` + ``gcloud`` adjacency."""
        from agent.file_safety import get_read_block_error

        assert (
            get_read_block_error("~/.config/gcloud/access_tokens.db") is not None
        )

    def test_config_without_gcloud_not_blocked(self):
        """A plain ``.config`` file (no adjacent ``gcloud``) is not a
        credential dir — the adjacency rule must stay precise."""
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/home/bob/.config/app/settings.toml") is None


class TestSecretReadPemScoping:
    """FIX 2 — the blanket ``.pem`` deny is replaced by a name heuristic:
    private-key material is denied, public certs are allowed.
    """

    # ---- allowed public certs ------------------------------------------
    def test_public_fullchain_pem_allowed(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/etc/letsencrypt/certs/fullchain.pem") is None

    def test_public_chain_pem_allowed(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/etc/ssl/chain.pem") is None

    def test_public_cert_pem_allowed(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/etc/ssl/cert.pem") is None

    def test_ca_bundle_pem_allowed(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/etc/ssl/ca-bundle.pem") is None

    def test_crt_allowed(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/etc/ssl/server.crt") is None

    def test_cer_allowed(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/etc/ssl/server.cer") is None

    def test_cert_ext_allowed(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/etc/ssl/server.cert") is None

    # ---- denied private keys / keystores -------------------------------
    def test_private_key_pem_still_denied(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/etc/ssl/private/privkey.pem") is not None

    def test_key_pem_denied(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/etc/ssl/key.pem") is not None

    def test_hyphen_key_pem_denied(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/etc/ssl/server-key.pem") is not None

    def test_underscore_key_pem_denied(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/etc/ssl/server_key.pem") is not None

    def test_private_named_pem_denied(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/etc/ssl/my-private-thing.pem") is not None

    def test_dot_key_denied(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/etc/ssl/server.key") is not None

    def test_pfx_denied(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/certs/bundle.pfx") is not None

    def test_p12_denied(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/certs/bundle.p12") is not None

    def test_keystore_denied(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/app/config/app.keystore") is not None

    def test_jks_denied(self):
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/app/config/app.jks") is not None

    def test_existing_basename_keys_still_denied(self):
        """id_rsa etc. basename denials survive the PEM refactor."""
        from agent.file_safety import get_read_block_error

        assert get_read_block_error("/home/bob/keys/id_rsa") is not None

    def test_service_account_json_still_denied(self):
        """Service-account JSON check survives the PEM refactor."""
        from agent.file_safety import get_read_block_error

        assert (
            get_read_block_error("/secrets/prod-service-account.json") is not None
        )
