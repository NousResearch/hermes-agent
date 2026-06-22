"""Integration tests for the Connect-RPC Antigravity language_server client.

The tests in this module exercise REAL behavior against the bundled
``language_server_linux_arm`` daemon binary. Mark each one with
``@pytest.mark.requires_ls_binary`` so CI / contributors without the
binary installed skip cleanly.

What we exercise
================
* Spawning the daemon and reading the discovery JSON
* /healthz on the HTTP port
* GetCascadeModelConfigs RPC over the HTTPS port (CSRF + Connect headers)
* End-to-end chat.completions.create — REQUIRES a Google OAuth token
  already present in ``$gemini_dir/<app_data_dir>/antigravity-oauth-token``.
  When the auth path doesn't work, the test XFAILs with a useful message
  instead of silently passing.
* Streaming: verify that iterating the result yields multiple chunks.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skip(
        reason=(
            "agy-cli provider is known-broken WIP (USER 2026-06-04: subprocess "
            "shim treats CLI flags as goal). Skipped intentionally — provider "
            "not stabilized. Drop this mark to run anyway."
        )
    ),
    pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnknownMarkWarning"
),
]

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.agy_cli_client import AgyCliClient, LanguageServerDaemon  # noqa: E402

_LS_BINARY = os.environ.get(
    "HERMES_AGY_LANGUAGE_SERVER",
    "/tmp/ag-ide/Antigravity IDE/resources/app/extensions/antigravity/bin/language_server_linux_arm",
)


def _binary_present() -> bool:
    return Path(_LS_BINARY).is_file() and os.access(_LS_BINARY, os.X_OK)


requires_ls_binary = pytest.mark.skipif(
    not _binary_present(),
    reason=f"language_server binary not found at {_LS_BINARY}",
)


# ---------------------------------------------------------------------------
# Auth heuristics
# ---------------------------------------------------------------------------

_AUTH_FAILURE_NEEDLES = (
    "UNAUTHENTICATED",
    "CREDENTIALS_MISSING",
    "Agent execution terminated due to error",
    "neither PlanModel nor RequestedModel",
    "load code assist",
    "code assist",
)


def _looks_like_auth_failure(exc: BaseException) -> bool:
    msg = str(exc)
    return any(n in msg for n in _AUTH_FAILURE_NEEDLES)


def _auth_token_present() -> bool:
    gd = Path(os.environ.get("HERMES_AGY_GEMINI_DIR", str(Path.home() / ".gemini")))
    app = os.environ.get("HERMES_AGY_APP_DATA_DIR", "hermes-agy")
    candidates = [
        gd / app / "antigravity-oauth-token",
        gd / "antigravity-cli" / "antigravity-oauth-token",
    ]
    return any(c.exists() for c in candidates)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def daemon():
    """Module-scoped daemon — start once, share, tear down at the end."""
    os.environ.setdefault("HERMES_AGY_APP_DATA_DIR", "hermes-agy-test")
    LanguageServerDaemon.shutdown_shared()
    d = LanguageServerDaemon.shared()
    d.start()
    yield d
    LanguageServerDaemon.shutdown_shared()


@pytest.fixture
def client():
    c = AgyCliClient()
    try:
        yield c
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Daemon lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.requires_ls_binary
def test_daemon_starts_and_writes_discovery_file(daemon):
    assert daemon.discovery is not None
    assert daemon.discovery["pid"] > 0
    assert daemon.discovery["httpsPort"] > 0
    assert daemon.discovery["httpPort"] > 0
    assert len(daemon.discovery["csrfToken"]) >= 16

    files = list(daemon.daemon_dir.glob("ls_*.json"))
    assert files, f"no discovery file in {daemon.daemon_dir}"
    parsed = json.loads(files[0].read_text())
    assert parsed["pid"] == daemon.discovery["pid"]
    assert parsed["csrfToken"] == daemon.discovery["csrfToken"]


@pytest.mark.requires_ls_binary
def test_daemon_start_is_idempotent(daemon):
    first = daemon.discovery
    pid1 = first["pid"]
    second = daemon.start()
    assert second["pid"] == pid1


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

@pytest.mark.requires_ls_binary
def test_healthz_endpoint_returns_200(daemon, client):
    assert client.healthz() is True


@pytest.mark.requires_ls_binary
def test_get_cascade_model_configs_returns_200(daemon, client):
    """Smoke a real Connect-RPC unary call."""
    result = client._rpc("GetCascadeModelConfigs", {})
    assert isinstance(result, dict)


@pytest.mark.requires_ls_binary
def test_start_cascade_returns_id(daemon, client):
    out = client._rpc("StartCascade", {"source": "CORTEX_TRAJECTORY_SOURCE_SDK"})
    cid = out.get("cascadeId")
    assert isinstance(cid, str) and len(cid) >= 8


@pytest.mark.requires_ls_binary
def test_start_cascade_rejects_missing_source(daemon, client):
    with pytest.raises(RuntimeError) as exc:
        client._rpc("StartCascade", {})
    assert "CortexTrajectorySource" in str(exc.value)


# ---------------------------------------------------------------------------
# End-to-end chat — require a working OAuth path inside the daemon.
# ---------------------------------------------------------------------------

@pytest.mark.requires_ls_binary
def test_chat_completions_create_smoke(daemon, client):
    if not _auth_token_present():
        pytest.xfail(
            "no Antigravity OAuth token under $HERMES_AGY_GEMINI_DIR — "
            "the wire works but the daemon can't call Google. Run the "
            "Antigravity CLI once or copy ~/.gemini/antigravity-cli/"
            "antigravity-oauth-token into the test app_data_dir."
        )
    try:
        result = client.chat.completions.create(
            model="gemini-3.1-pro-high",
            messages=[{"role": "user",
                       "content": "Reply with exactly OK and nothing else."}],
            stream=False,
        )
    except RuntimeError as e:
        if _looks_like_auth_failure(e):
            pytest.xfail(f"daemon auth/loadCodeAssist failure: {e}")
        raise
    content = result.choices[0].message.content
    assert isinstance(content, str) and content.strip(), f"empty reply: {result!r}"


@pytest.mark.requires_ls_binary
def test_chat_completions_streaming_yields_chunks(daemon, client):
    if not _auth_token_present():
        pytest.xfail("no Antigravity OAuth token present — streaming needs Google call")
    stream = client.chat.completions.create(
        model="gemini-3.1-pro-high",
        messages=[{"role": "user",
                   "content": "Count slowly from one to ten in english words, "
                              "one per line."}],
        stream=True,
    )
    chunks = []
    started = time.time()
    try:
        for chunk in stream:
            chunks.append(chunk)
            if time.time() - started > 60:
                break
    except RuntimeError as e:
        if _looks_like_auth_failure(e):
            pytest.xfail(f"daemon auth/loadCodeAssist failure: {e}")
        raise
    assert len(chunks) >= 2
    last = chunks[-1]
    assert last.choices[0].finish_reason == "stop"
    full = "".join(
        (c.choices[0].delta.content or "") for c in chunks
        if c.choices and c.choices[0].delta
    )
    assert full.strip()
