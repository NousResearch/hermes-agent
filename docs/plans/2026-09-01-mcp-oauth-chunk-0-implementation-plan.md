# MCP OAuth Chunk 0 Baseline Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic, executable evidence that failed MCP OAuth reauthorization mutates or deletes active credentials on current `NousResearch/main`, while proving that runtime parking and reconnect preserve durable credentials.

**Architecture:** Add a test-only OAuth peer at the existing `_probe_single_server()` provider boundary. The peer writes partial state through the real `HermesTokenStorage`, injects typed failures at eight lifecycle points, and records a safe event ledger; surface tests drive the real CLI, dashboard, and Desktop/TUI workers around that boundary. The pre-token injection points and token exchange also carry an optional failure *kind* (`definitive` / `indeterminate`), and the probe point yields a classifiable outcome (`authenticated` / `rejected` / `indeterminate`) — Chunk 0 only *exposes* these; Chunk 2 classifies them and Chunk 3 builds the retry / `probe=deferred` behavior. Known defects are narrowly typed strict expected failures, while already-correct preservation behavior remains an ordinary passing invariant.

**Tech Stack:** Python 3, pytest, MCP Python SDK-compatible records, `HermesTokenStorage`, `DashboardOAuthFlow`, and Hermes' `scripts/run_tests.sh` wrapper.

**Spec:** [`../design/mcp-oauth-00-baseline-reproduction.md`](../design/mcp-oauth-00-baseline-reproduction.md)

## Global Constraints

- Implement from repository commit `0f428209e600727c7d1d2bc5731c92eb21081d3f` or revalidate every referenced symbol against a newer base before editing.
- Change tests only. Do not modify production behavior, configuration, or credential formats in Chunk 0.
- Execute real surface entry points and real `HermesTokenStorage`; patch the provider/probe boundary, not persistence methods.
- Set `HERMES_HOME` to `str(tmp_path.resolve())` — a canonical spelling, so it already matches the identity digest Chunk 1 derives. Never read or write the user's `~/.hermes`, Keychain, or live credentials.
- Do not add F-2 taxonomy assertions (retry / abort / `probe=deferred` outcomes) — Chunk 0 only exposes the kind and probe-outcome capability; Chunk 2 and Chunk 3 assert the behavior.
- Do not add the Chunk 4 token time-model fields (`accepted_at_utc`, `expires_at`, `original_expires_in`) or an injectable clock. The OLD/NEW token fixtures use the legacy on-disk shape (`access_token`, `refresh_token`, `token_type`, `expires_in`).
- Use unmistakably fake sentinel values and never print token, refresh-token, client-secret, authorization-code, callback-state, or absolute temporary-path values.
- Perform no live network requests, browser launches, callback listeners, or fixed sleeps.
- Do not inspect production source text or assert implementation line contents.
- Use `scripts/run_tests.sh`; never invoke `pytest` directly.
- Mark only exact, documented corruption signatures as strict expected failures tied to GH #76590. Unexpected exceptions and state shapes must fail normally.
- Preserve the positive invariant that runtime parking, reconnect, and shutdown do not delete durable OAuth state.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `tests/fakes/mcp_oauth_peer.py` | Create | Artifact oracle, fake sentinels, failure enum, typed failures, safe diagnostics, and deterministic probe-boundary peer. |
| `tests/tools/test_mcp_oauth_reauth_regression.py` | Create | Harness contract tests and the eight-point dashboard worker matrix. |
| `tests/hermes_cli/test_mcp_reauth_lifecycle.py` | Create | Direct CLI `_reauth_oauth_server()` baseline scenarios. |
| `tests/hermes_cli/test_mcp_dashboard_oauth.py` | Inspect | Existing route-to-worker contract; modify only if that seam is no longer covered. |
| `tests/tui_gateway/test_mcp_oauth_reauth.py` | Create | Direct Desktop/TUI `_worker()` parity scenarios. |
| `tests/tools/test_mcp_initial_connect_shutdown.py` | Modify | Positive OAuth HTTP parking/reconnect/shutdown credential-retention test. |

Production seams used without modification:

- `tools/mcp_oauth.py:457-675` — `HermesTokenStorage` persistence, removal, snapshot, and restore.
- `tools/mcp_oauth_manager.py:770-819` — durable `remove()`, provider `restore_entry()`, and memory-only `evict()`.
- `hermes_cli/mcp_config.py:810-909` — CLI `_reauth_oauth_server()`.
- `hermes_cli/web_server.py:13776-13855` — dashboard transaction lock and `_run_dashboard_mcp_oauth()`.
- `tui_gateway/mcp_oauth_sessions.py:163-247` — Desktop/TUI `_worker()`.
- `tools/mcp_tool.py:4020-4380` — runtime retry and parking lifecycle.

---

### Task 1: Artifact Oracle and Typed Failure Vocabulary

**Files:**
- Create: `tests/fakes/mcp_oauth_peer.py`
- Create: `tests/tools/test_mcp_oauth_reauth_regression.py`

**Interfaces:**
- Produces: `OAuthFailurePoint`, `InjectedOAuthFailure`, `KnownCredentialLoss`, `OAuthArtifactState`, `seed_old_oauth_state()`, `capture_oauth_state()`, and `raise_known_mutation()`.
- Consumes: `tools.mcp_oauth.HermesTokenStorage` path resolution.

- [ ] **Step 1: Write oracle tests before the helper exists**

Create `tests/tools/test_mcp_oauth_reauth_regression.py`:

```python
import pytest

from tests.fakes.mcp_oauth_peer import (
    KnownCredentialLoss,
    OAuthArtifactState,
    OAuthFailurePoint,
    capture_oauth_state,
    raise_known_mutation,
    seed_old_oauth_state,
)


def test_seed_and_capture_old_oauth_state_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    seeded = seed_old_oauth_state(tmp_path, "reports")
    assert capture_oauth_state(tmp_path, "reports") == seeded
    assert seeded.labels() == ("OLD", "OLD", "OLD")
    assert seeded.safe_summary() == "token=OLD client=OLD metadata=OLD"


def test_safe_summary_never_contains_fake_secret_payloads():
    state = OAuthArtifactState(
        token=b'{"access_token":"OLD_ACCESS_TOKEN_FOR_TEST_ONLY"}',
        client=b'{"client_secret":"OLD_CLIENT_SECRET_FOR_TEST_ONLY"}',
        metadata=b'{"token_endpoint":"https://old-auth.invalid/token"}',
    )
    summary = state.safe_summary()
    assert summary == "token=OLD client=OLD metadata=OLD"
    assert "ACCESS_TOKEN" not in summary
    assert "CLIENT_SECRET" not in summary
    assert "old-auth.invalid" not in summary


def test_raise_known_mutation_rejects_unknown_corruption_shape():
    before = OAuthArtifactState(b"OLD_TOKEN", b"OLD_CLIENT", b"OLD_META")
    unexpected = OAuthArtifactState(None, None, b"PARTIAL_META")
    with pytest.raises(AssertionError, match="unexpected OAuth artifact state"):
        raise_known_mutation(
            before=before,
            after=unexpected,
            expected_labels=("MISSING", "PARTIAL", "PARTIAL"),
            surface="dashboard",
            failure_point=OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION,
        )


def test_raise_known_mutation_types_the_exact_known_bug():
    before = OAuthArtifactState(b"OLD_TOKEN", b"OLD_CLIENT", b"OLD_META")
    exact = OAuthArtifactState(None, b"PARTIAL_CLIENT", b"PARTIAL_META")
    with pytest.raises(KnownCredentialLoss, match="surface=dashboard"):
        raise_known_mutation(
            before=before,
            after=exact,
            expected_labels=("MISSING", "PARTIAL", "PARTIAL"),
            surface="dashboard",
            failure_point=OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION,
        )
```

- [ ] **Step 2: Run the oracle tests and verify collection fails**

Run: `scripts/run_tests.sh tests/tools/test_mcp_oauth_reauth_regression.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'tests.fakes.mcp_oauth_peer'`.

- [ ] **Step 3: Implement the artifact oracle**

Create `tests/fakes/mcp_oauth_peer.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tools.mcp_oauth import HermesTokenStorage


class OAuthFailurePoint(str, Enum):
    PROTECTED_RESOURCE_DISCOVERY = "protected_resource_discovery"
    AUTHORIZATION_SERVER_DISCOVERY = "authorization_server_discovery"
    DYNAMIC_CLIENT_REGISTRATION = "dynamic_client_registration"
    AUTHORIZATION_URL_PUBLICATION = "authorization_url_publication"
    CALLBACK_RECEIPT = "callback_receipt"
    TOKEN_EXCHANGE = "token_exchange"
    TOKEN_PERSISTENCE = "token_persistence"
    MCP_INITIALIZATION = "mcp_initialization"


class InjectedOAuthFailure(RuntimeError):
    def __init__(self, point: OAuthFailurePoint, events: tuple[str, ...]):
        self.point = point
        self.events = events
        super().__init__(f"injected OAuth failure after {point.value}")


class KnownCredentialLoss(AssertionError):
    """Raised only after a test recognizes GH #76590's exact state shape."""


@dataclass(frozen=True)
class OAuthArtifactState:
    token: bytes | None
    client: bytes | None
    metadata: bytes | None

    @staticmethod
    def _label(value: bytes | None) -> str:
        if value is None:
            return "MISSING"
        if b"OLD_" in value:
            return "OLD"
        if b"PARTIAL_" in value:
            return "PARTIAL"
        if b"NEW_" in value:
            return "NEW"
        return "OTHER"

    def labels(self) -> tuple[str, str, str]:
        return tuple(self._label(value) for value in (self.token, self.client, self.metadata))

    def safe_summary(self) -> str:
        token, client, metadata = self.labels()
        return f"token={token} client={client} metadata={metadata}"


_OLD_DOCUMENTS = {
    "token": {"access_token": "OLD_ACCESS_TOKEN_FOR_TEST_ONLY", "refresh_token": "OLD_REFRESH_TOKEN_FOR_TEST_ONLY", "token_type": "Bearer", "expires_in": 3600},
    "client": {"client_id": "OLD_CLIENT_ID_FOR_TEST_ONLY", "client_secret": "OLD_CLIENT_SECRET_FOR_TEST_ONLY", "redirect_uris": ["http://127.0.0.1:43111/callback"], "token_endpoint_auth_method": "client_secret_post"},
    "metadata": {"issuer": "https://old-auth.invalid", "authorization_endpoint": "https://old-auth.invalid/authorize", "token_endpoint": "https://old-auth.invalid/token"},
}


def _stable_json(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def seed_old_oauth_state(home: Path, server_name: str) -> OAuthArtifactState:
    storage = HermesTokenStorage(server_name, hermes_home=Path(home).resolve())
    paths = (storage._tokens_path(), storage._client_info_path(), storage._meta_path())
    payloads = tuple(_stable_json(_OLD_DOCUMENTS[key]) for key in ("token", "client", "metadata"))
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    paths[0].parent.chmod(0o700)
    for path, payload in zip(paths, payloads, strict=True):
        path.write_bytes(payload)
        path.chmod(0o600)
    return OAuthArtifactState(*payloads)


def capture_oauth_state(home: Path, server_name: str) -> OAuthArtifactState:
    storage = HermesTokenStorage(server_name, hermes_home=Path(home).resolve())

    def read(path: Path) -> bytes | None:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    return OAuthArtifactState(read(storage._tokens_path()), read(storage._client_info_path()), read(storage._meta_path()))


def raise_known_mutation(
    *,
    before: OAuthArtifactState,
    after: OAuthArtifactState,
    expected_labels: tuple[str, str, str],
    surface: str,
    failure_point: OAuthFailurePoint,
) -> None:
    if after == before:
        return
    if after.labels() != expected_labels:
        raise AssertionError(
            "unexpected OAuth artifact state: "
            f"surface={surface} failure={failure_point.value} after={after.safe_summary()}"
        )
    raise KnownCredentialLoss(
        f"surface={surface} before={before.safe_summary()} "
        f"failure={failure_point.value} after={after.safe_summary()}"
    )
```

- [ ] **Step 4: Run the oracle tests**

Run: `scripts/run_tests.sh tests/tools/test_mcp_oauth_reauth_regression.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/fakes/mcp_oauth_peer.py tests/tools/test_mcp_oauth_reauth_regression.py
git commit -m "test(mcp): add OAuth artifact oracle"
```

---

### Task 2: Deterministic OAuth/MCP Failure Peer with Failure-Kind and Probe-Outcome Capability

**Files:**
- Modify: `tests/fakes/mcp_oauth_peer.py`
- Modify: `tests/tools/test_mcp_oauth_reauth_regression.py`

**Interfaces:**
- Consumes: Task 1's vocabulary and artifact oracle.
- Produces: `FakeOAuthMCPPeer(failure_point, *, kind=OAuthFailureKind.DEFINITIVE, probe_outcome=ProbeOutcome.REJECTED)`, `.probe(server_name, config, connect_timeout=None, *, details=None) -> list[tuple[str, str]]`, `.completed_events`, `.connect_timeouts`.
- Produces: `OAuthFailureKind` (`DEFINITIVE`, `INDETERMINATE`), `ProbeOutcome` (`AUTHENTICATED`, `REJECTED`, `INDETERMINATE`).
- Produces: `InjectedOAuthFailure.kind`, `.retry_after` (set only for an `INDETERMINATE` `HTTP 429`), and `.probe_outcome` (set only when `failure_point is MCP_INITIALIZATION`).

Steps 1–5 build the position-only peer and the baseline matrices depend on it with the defaults. Steps 6–10 add the kind and probe-outcome attributes that Chunk 2 (classifier) and Chunk 3 (retry, `probe=deferred`) consume. Chunk 0 asserts only that the attributes are exposed correctly — never a downstream retry, abort, or commit decision.

- [ ] **Step 1: Add failure-point contract tests**

Append to `tests/tools/test_mcp_oauth_reauth_regression.py`:

```python
from tests.fakes.mcp_oauth_peer import FakeOAuthMCPPeer, InjectedOAuthFailure


@pytest.mark.parametrize("failure_point", list(OAuthFailurePoint))
def test_fake_peer_fails_after_exact_requested_event(tmp_path, monkeypatch, failure_point):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    peer = FakeOAuthMCPPeer(failure_point)
    with pytest.raises(InjectedOAuthFailure) as caught:
        peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"}, connect_timeout=315)
    assert caught.value.point is failure_point
    assert peer.completed_events[-1] == failure_point.value
    assert peer.connect_timeouts == [315]


@pytest.mark.parametrize(
    ("failure_point", "expected_labels"),
    [
        (OAuthFailurePoint.PROTECTED_RESOURCE_DISCOVERY, ("MISSING", "MISSING", "MISSING")),
        (OAuthFailurePoint.AUTHORIZATION_SERVER_DISCOVERY, ("MISSING", "MISSING", "PARTIAL")),
        (OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION, ("MISSING", "PARTIAL", "PARTIAL")),
        (OAuthFailurePoint.AUTHORIZATION_URL_PUBLICATION, ("MISSING", "PARTIAL", "PARTIAL")),
        (OAuthFailurePoint.CALLBACK_RECEIPT, ("MISSING", "PARTIAL", "PARTIAL")),
        (OAuthFailurePoint.TOKEN_EXCHANGE, ("MISSING", "PARTIAL", "PARTIAL")),
        (OAuthFailurePoint.TOKEN_PERSISTENCE, ("NEW", "PARTIAL", "PARTIAL")),
        (OAuthFailurePoint.MCP_INITIALIZATION, ("NEW", "PARTIAL", "PARTIAL")),
    ],
)
def test_fake_peer_persists_only_completed_stage_effects(tmp_path, monkeypatch, failure_point, expected_labels):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    peer = FakeOAuthMCPPeer(failure_point)
    with pytest.raises(InjectedOAuthFailure):
        peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"})
    assert capture_oauth_state(tmp_path, "reports").labels() == expected_labels
```

- [ ] **Step 2: Verify the tests fail because the peer is missing**

Run: `scripts/run_tests.sh tests/tools/test_mcp_oauth_reauth_regression.py -q`

Expected: FAIL during collection with `ImportError: cannot import name 'FakeOAuthMCPPeer'`.

- [ ] **Step 3: Implement the peer with real storage writes**

Append to `tests/fakes/mcp_oauth_peer.py`:

```python
import asyncio


@dataclass(frozen=True)
class _DumpableOAuthRecord:
    payload: dict

    def model_dump(self, **_kwargs) -> dict:
        return dict(self.payload)


class FakeOAuthMCPPeer:
    def __init__(self, failure_point: OAuthFailurePoint | None):
        self.failure_point = failure_point
        self.completed_events: list[str] = []
        self.connect_timeouts: list[float | None] = []
        self._new_token = _DumpableOAuthRecord({"access_token": "NEW_ACCESS_TOKEN_FOR_TEST_ONLY", "refresh_token": "NEW_REFRESH_TOKEN_FOR_TEST_ONLY", "token_type": "Bearer", "expires_in": 3600})

    def _complete(self, point: OAuthFailurePoint, storage: HermesTokenStorage) -> None:
        if point is OAuthFailurePoint.AUTHORIZATION_SERVER_DISCOVERY:
            storage.save_oauth_metadata(_DumpableOAuthRecord({"issuer": "https://partial-auth.invalid", "authorization_endpoint": "https://partial-auth.invalid/authorize", "token_endpoint": "https://partial-auth.invalid/token", "marker": "PARTIAL_METADATA_FOR_TEST_ONLY"}))
        elif point is OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION:
            asyncio.run(storage.set_client_info(_DumpableOAuthRecord({"client_id": "PARTIAL_CLIENT_ID_FOR_TEST_ONLY", "client_secret": "PARTIAL_CLIENT_SECRET_FOR_TEST_ONLY", "redirect_uris": ["http://127.0.0.1:43112/callback"], "token_endpoint_auth_method": "client_secret_post"})))
        elif point is OAuthFailurePoint.TOKEN_PERSISTENCE:
            asyncio.run(storage.set_tokens(self._new_token))
        self.completed_events.append(point.value)

    def probe(
        self,
        server_name: str,
        config: dict,
        connect_timeout: float | None = None,
        *,
        details: dict | None = None,
    ) -> list[tuple[str, str]]:
        del config, details
        self.connect_timeouts.append(connect_timeout)
        storage = HermesTokenStorage(server_name)
        for point in OAuthFailurePoint:
            self._complete(point, storage)
            if point is self.failure_point:
                raise InjectedOAuthFailure(point, tuple(self.completed_events))
        return [("fake_tool", "Deterministic fake MCP tool")]
```

- [ ] **Step 4: Run the peer contract**

Run: `scripts/run_tests.sh tests/tools/test_mcp_oauth_reauth_regression.py -q`

Expected: `20 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/fakes/mcp_oauth_peer.py tests/tools/test_mcp_oauth_reauth_regression.py
git commit -m "test(mcp): add deterministic OAuth failure peer"
```

- [ ] **Step 6: Add failure-kind and probe-outcome contract tests**

Append to `tests/tools/test_mcp_oauth_reauth_regression.py`:

```python
from tests.fakes.mcp_oauth_peer import OAuthFailureKind, ProbeOutcome

_PRE_TOKEN_KINDED = [
    OAuthFailurePoint.PROTECTED_RESOURCE_DISCOVERY,
    OAuthFailurePoint.AUTHORIZATION_SERVER_DISCOVERY,
    OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION,
    OAuthFailurePoint.TOKEN_EXCHANGE,
]


@pytest.mark.parametrize("failure_point", _PRE_TOKEN_KINDED)
def test_pre_token_failure_carries_default_definitive_kind(tmp_path, monkeypatch, failure_point):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    peer = FakeOAuthMCPPeer(failure_point)
    with pytest.raises(InjectedOAuthFailure) as caught:
        peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"})
    assert caught.value.kind is OAuthFailureKind.DEFINITIVE
    assert caught.value.retry_after is None


@pytest.mark.parametrize("failure_point", _PRE_TOKEN_KINDED)
def test_pre_token_failure_reports_indeterminate_kind_and_retry_after(tmp_path, monkeypatch, failure_point):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    peer = FakeOAuthMCPPeer(failure_point, kind=OAuthFailureKind.INDETERMINATE)
    with pytest.raises(InjectedOAuthFailure) as caught:
        peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"})
    assert caught.value.kind is OAuthFailureKind.INDETERMINATE
    assert isinstance(caught.value.retry_after, (int, float))


def test_authenticated_probe_returns_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    peer = FakeOAuthMCPPeer(None)
    assert peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"}) == [("fake_tool", "Deterministic fake MCP tool")]


@pytest.mark.parametrize("outcome", [ProbeOutcome.REJECTED, ProbeOutcome.INDETERMINATE])
def test_probe_point_reports_requested_failing_outcome(tmp_path, monkeypatch, outcome):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    peer = FakeOAuthMCPPeer(OAuthFailurePoint.MCP_INITIALIZATION, probe_outcome=outcome)
    with pytest.raises(InjectedOAuthFailure) as caught:
        peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"})
    assert caught.value.probe_outcome is outcome


def test_publication_and_callback_points_take_no_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    for point in (OAuthFailurePoint.AUTHORIZATION_URL_PUBLICATION, OAuthFailurePoint.CALLBACK_RECEIPT):
        peer = FakeOAuthMCPPeer(point)
        with pytest.raises(InjectedOAuthFailure) as caught:
            peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"})
        assert caught.value.kind is None
```

- [ ] **Step 7: Verify the new tests fail**

Run: `scripts/run_tests.sh tests/tools/test_mcp_oauth_reauth_regression.py -q`

Expected: FAIL during collection with `ImportError: cannot import name 'OAuthFailureKind'`.

- [ ] **Step 8: Add failure-kind and probe-outcome to the peer**

In `tests/fakes/mcp_oauth_peer.py`:

- Add `class OAuthFailureKind(str, Enum): DEFINITIVE = "definitive"; INDETERMINATE = "indeterminate"`.
- Add `class ProbeOutcome(str, Enum): AUTHENTICATED = "authenticated"; REJECTED = "rejected"; INDETERMINATE = "indeterminate"`.
- Extend `InjectedOAuthFailure.__init__` with keyword-only `kind: OAuthFailureKind | None = None`, `retry_after: float | None = None`, `probe_outcome: ProbeOutcome | None = None`; store all three.
- `FakeOAuthMCPPeer.__init__(self, failure_point, *, kind=OAuthFailureKind.DEFINITIVE, probe_outcome=ProbeOutcome.REJECTED)`; store both. `probe_outcome` is constrained to `REJECTED` / `INDETERMINATE` (the `authenticated` probe is simply `failure_point=None`).
- `_KINDED_POINTS = {PROTECTED_RESOURCE_DISCOVERY, AUTHORIZATION_SERVER_DISCOVERY, DYNAMIC_CLIENT_REGISTRATION, TOKEN_EXCHANGE}`.
- When raising at `self.failure_point`:
  - if in `_KINDED_POINTS`: pass `kind=self.kind`, and `retry_after=1.0` when `self.kind is INDETERMINATE`.
  - if `MCP_INITIALIZATION`: raise with `probe_outcome=self.probe_outcome` (and `kind=None`).
  - otherwise (`AUTHORIZATION_URL_PUBLICATION`, `CALLBACK_RECEIPT`, `TOKEN_PERSISTENCE`): raise with no `kind`.
- `failure_point is None` runs the whole flow and returns the tool list (the `authenticated` probe path).
- The `definitive` kind maps to HTTP 400 / `invalid_grant` / `invalid_client` / unsupported registration; `indeterminate` to HTTP 5xx / connection error / timeout / 429. The peer models the classification inputs, not the codes themselves.

- [ ] **Step 9: Run the extended peer contract**

Run: `scripts/run_tests.sh tests/tools/test_mcp_oauth_reauth_regression.py -q`

Expected: all Task 1, Task 2, and the new kind/probe-outcome tests pass; no `XPASS`.

- [ ] **Step 10: Commit**

```bash
git add tests/fakes/mcp_oauth_peer.py tests/tools/test_mcp_oauth_reauth_regression.py
git commit -m "test(mcp): add failure-kind and probe-outcome capability to the OAuth peer"
```

---

### Task 3: Dashboard Eight-Point Baseline Matrix

**Files:**
- Modify: `tests/tools/test_mcp_oauth_reauth_regression.py`
- Inspect: `tests/hermes_cli/test_mcp_dashboard_oauth.py:27-58`

**Interfaces:**
- Consumes: Tasks 1-2.
- Exercises: `hermes_cli.web_server._run_dashboard_mcp_oauth(flow, cfg) -> None` and `DashboardOAuthFlow.worker_done: bool`.
- Produces: One passing pre-write preservation row and seven strict expected failures.

The baseline matrix uses `FakeOAuthMCPPeer(failure_point)` with the defaults — `kind=DEFINITIVE`, and a failing probe at `MCP_INITIALIZATION`. The current bug loses the token regardless of kind, so Tasks 3–5 do not parametrize over kind; the `indeterminate` and probe-outcome variants are Chunk 3's to assert.

- [ ] **Step 1: Add the strict marker and current-result table**

Append:

```python
from tools.mcp_dashboard_oauth import DashboardOAuthFlow

_GH_76590 = pytest.mark.xfail(strict=True, raises=KnownCredentialLoss, reason="GH #76590: failed MCP OAuth reauthorization mutates active credentials")

_DASHBOARD_FAILURES = [
    pytest.param(OAuthFailurePoint.PROTECTED_RESOURCE_DISCOVERY, None, id="before-write-preserves-old"),
    pytest.param(OAuthFailurePoint.AUTHORIZATION_SERVER_DISCOVERY, ("MISSING", "MISSING", "PARTIAL"), marks=_GH_76590),
    pytest.param(OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION, ("MISSING", "PARTIAL", "PARTIAL"), marks=_GH_76590),
    pytest.param(OAuthFailurePoint.AUTHORIZATION_URL_PUBLICATION, ("MISSING", "PARTIAL", "PARTIAL"), marks=_GH_76590),
    pytest.param(OAuthFailurePoint.CALLBACK_RECEIPT, ("MISSING", "PARTIAL", "PARTIAL"), marks=_GH_76590),
    pytest.param(OAuthFailurePoint.TOKEN_EXCHANGE, ("MISSING", "PARTIAL", "PARTIAL"), marks=_GH_76590),
    pytest.param(OAuthFailurePoint.TOKEN_PERSISTENCE, ("NEW", "PARTIAL", "PARTIAL"), marks=_GH_76590),
    pytest.param(OAuthFailurePoint.MCP_INITIALIZATION, ("NEW", "PARTIAL", "PARTIAL"), marks=_GH_76590),
]
```

- [ ] **Step 2: Add the direct production-worker matrix**

```python
@pytest.mark.parametrize(("failure_point", "broken_labels"), _DASHBOARD_FAILURES)
def test_dashboard_failed_reauth_preserves_active_state(tmp_path, monkeypatch, failure_point, broken_labels):
    from hermes_cli import mcp_config, web_server
    from tools.mcp_oauth_manager import reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    reset_manager_for_tests()
    before = seed_old_oauth_state(tmp_path, "reports")
    peer = FakeOAuthMCPPeer(failure_point)
    flow = DashboardOAuthFlow(flow_id=f"dashboard-{failure_point.value}", server_name="reports", profile=None, hermes_home=str(tmp_path), redirect_uri="https://dashboard.invalid/api/mcp/oauth/callback/reports")
    monkeypatch.setattr(mcp_config, "_probe_single_server", peer.probe)
    monkeypatch.setattr(mcp_config, "_save_mcp_server", lambda *_args: True)

    web_server._run_dashboard_mcp_oauth(flow, {"url": "https://mcp.invalid/mcp", "auth": "oauth"})

    assert flow.status == "error"
    assert flow.worker_done is True
    assert failure_point.value in (flow.error or "")
    assert "ACCESS_TOKEN_FOR_TEST_ONLY" not in (flow.error or "")
    after = capture_oauth_state(tmp_path, "reports")
    if broken_labels is None:
        assert after == before
    else:
        raise_known_mutation(before=before, after=after, expected_labels=broken_labels, surface="dashboard", failure_point=failure_point)
```

- [ ] **Step 3: Run the matrix and verify its baseline shape**

Run: `scripts/run_tests.sh tests/tools/test_mcp_oauth_reauth_regression.py -q -rxX`

Expected: harness tests pass, one dashboard preservation row passes, and exactly seven dashboard rows report `XFAIL`. The command exits zero.

- [ ] **Step 4: Verify route coverage instead of duplicating it**

Read `tests/hermes_cli/test_mcp_dashboard_oauth.py:27-58`. If `test_hosted_auth_start_returns_public_authorization_url` still patches `_run_dashboard_mcp_oauth` and proves dispatch, make no change to this file.

- [ ] **Step 5: Run existing dashboard contracts**

Run: `scripts/run_tests.sh tests/hermes_cli/test_mcp_dashboard_oauth.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/tools/test_mcp_oauth_reauth_regression.py
git commit -m "test(mcp): reproduce dashboard OAuth state loss"
```

---

### Task 4: CLI Reauthorization Baseline

**Files:**
- Create: `tests/hermes_cli/test_mcp_reauth_lifecycle.py`

**Interfaces:**
- Consumes: Tasks 1-2.
- Exercises: `hermes_cli.mcp_config._reauth_oauth_server(name: str, server_config: dict) -> bool`.
- Produces: Three strict expected failures covering pre-write deletion, partial state, and post-token failure.

- [ ] **Step 1: Add the three CLI scenarios**

Create `tests/hermes_cli/test_mcp_reauth_lifecycle.py`:

```python
import pytest

from tests.fakes.mcp_oauth_peer import FakeOAuthMCPPeer, KnownCredentialLoss, OAuthFailurePoint, capture_oauth_state, raise_known_mutation, seed_old_oauth_state

_GH_76590 = pytest.mark.xfail(strict=True, raises=KnownCredentialLoss, reason="GH #76590: CLI reauthorization deletes active credentials before success")

@pytest.mark.parametrize(
    ("failure_point", "broken_labels"),
    [
        pytest.param(OAuthFailurePoint.PROTECTED_RESOURCE_DISCOVERY, ("MISSING", "MISSING", "MISSING"), marks=_GH_76590),
        pytest.param(OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION, ("MISSING", "PARTIAL", "PARTIAL"), marks=_GH_76590),
        pytest.param(OAuthFailurePoint.MCP_INITIALIZATION, ("NEW", "PARTIAL", "PARTIAL"), marks=_GH_76590),
    ],
)
def test_cli_failed_reauth_preserves_active_state(tmp_path, monkeypatch, capsys, failure_point, broken_labels):
    from hermes_cli import mcp_config
    from tools.mcp_oauth_manager import reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    reset_manager_for_tests()
    before = seed_old_oauth_state(tmp_path, "reports")
    peer = FakeOAuthMCPPeer(failure_point)
    monkeypatch.setattr(mcp_config, "_probe_single_server", peer.probe)

    result = mcp_config._reauth_oauth_server("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"})

    output = capsys.readouterr().out
    assert result is False
    assert "Authentication failed" in output
    assert "ACCESS_TOKEN_FOR_TEST_ONLY" not in output
    assert peer.connect_timeouts == [315.0]
    raise_known_mutation(before=before, after=capture_oauth_state(tmp_path, "reports"), expected_labels=broken_labels, surface="cli", failure_point=failure_point)
```

- [ ] **Step 2: Run and verify the three typed expected failures**

Run: `scripts/run_tests.sh tests/hermes_cli/test_mcp_reauth_lifecycle.py -q -rxX`

Expected: exactly `3 xfailed`; command exits zero.

- [ ] **Step 3: Run existing CLI dispatch coverage**

Run: `scripts/run_tests.sh tests/hermes_cli/test_mcp_config.py -k 'McpLogin or McpReauth' -q`

Expected: PASS. Do not duplicate the existing command-to-helper delegation tests.

- [ ] **Step 4: Commit**

```bash
git add tests/hermes_cli/test_mcp_reauth_lifecycle.py
git commit -m "test(mcp): reproduce CLI OAuth state loss"
```

---

### Task 5: Desktop/TUI RPC Reauthorization Baseline

**Files:**
- Create: `tests/tui_gateway/test_mcp_oauth_reauth.py`

**Interfaces:**
- Consumes: Tasks 1-2.
- Exercises: `tui_gateway.mcp_oauth_sessions._worker(session_id, hermes_home, server_name, cfg, reconnect_live) -> None`.
- Produces: One passing pre-write restoration row and two strict expected failures.

- [ ] **Step 1: Add an isolated session fixture and parity scenarios**

Create `tests/tui_gateway/test_mcp_oauth_reauth.py`:

```python
import time
import pytest

from tests.fakes.mcp_oauth_peer import FakeOAuthMCPPeer, KnownCredentialLoss, OAuthFailurePoint, capture_oauth_state, raise_known_mutation, seed_old_oauth_state
from tools.mcp_dashboard_oauth import DashboardOAuthFlow
from tui_gateway import mcp_oauth_sessions

_GH_76590 = pytest.mark.xfail(strict=True, raises=KnownCredentialLoss, reason="GH #76590: TUI reauthorization partial state suppresses rollback")

@pytest.fixture(autouse=True)
def _clear_oauth_sessions():
    with mcp_oauth_sessions._sessions_lock:
        mcp_oauth_sessions._sessions.clear()
    yield
    with mcp_oauth_sessions._sessions_lock:
        mcp_oauth_sessions._sessions.clear()

@pytest.mark.parametrize(
    ("failure_point", "broken_labels"),
    [
        pytest.param(OAuthFailurePoint.PROTECTED_RESOURCE_DISCOVERY, None),
        pytest.param(OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION, ("MISSING", "PARTIAL", "PARTIAL"), marks=_GH_76590),
        pytest.param(OAuthFailurePoint.MCP_INITIALIZATION, ("NEW", "PARTIAL", "PARTIAL"), marks=_GH_76590),
    ],
)
def test_tui_failed_reauth_preserves_active_state(tmp_path, monkeypatch, failure_point, broken_labels):
    from hermes_cli import mcp_config
    from tools.mcp_oauth_manager import reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    reset_manager_for_tests()
    before = seed_old_oauth_state(tmp_path, "reports")
    peer = FakeOAuthMCPPeer(failure_point)
    monkeypatch.setattr(mcp_config, "_probe_single_server", peer.probe)
    monkeypatch.setattr(mcp_config, "_save_mcp_server", lambda *_args: True)
    session_id = f"tui-{failure_point.value}"
    flow = DashboardOAuthFlow(flow_id=session_id, server_name="reports", profile=None, hermes_home=str(tmp_path), redirect_uri="http://127.0.0.1:43113/callback")
    with mcp_oauth_sessions._sessions_lock:
        mcp_oauth_sessions._sessions[session_id] = {"session_id": session_id, "server_name": "reports", "hermes_home": str(tmp_path), "flow": flow, "httpd": None, "created_at": time.time()}

    mcp_oauth_sessions._worker(session_id, str(tmp_path), "reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"}, False)

    assert flow.status == "error"
    assert flow.worker_done is True
    assert failure_point.value in (flow.error or "")
    assert mcp_oauth_sessions._sessions[session_id]["httpd"] is None
    after = capture_oauth_state(tmp_path, "reports")
    if broken_labels is None:
        assert after == before
    else:
        raise_known_mutation(before=before, after=after, expected_labels=broken_labels, surface="tui", failure_point=failure_point)
```

- [ ] **Step 2: Run and verify one pass plus two expected failures**

Run: `scripts/run_tests.sh tests/tui_gateway/test_mcp_oauth_reauth.py -q -rxX`

Expected: `1 passed, 2 xfailed`; command exits zero.

- [ ] **Step 3: Run existing callback-relay contracts**

Run: `scripts/run_tests.sh tests/tui_gateway/test_mcp_oauth_client_callback.py -q`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/tui_gateway/test_mcp_oauth_reauth.py
git commit -m "test(mcp): reproduce TUI OAuth state loss"
```

---

### Task 6: Runtime Parking and Reconnect Preservation Control

**Files:**
- Modify: `tests/tools/test_mcp_initial_connect_shutdown.py`

**Interfaces:**
- Consumes: Task 1's seed and capture helpers.
- Exercises: `register_mcp_servers()`, the real HTTP retry/park loop, `reconnect_mcp_server()`, and `shutdown_mcp_servers()`.
- Produces: A normal passing invariant with no expected-failure marker.

- [ ] **Step 1: Add imports and the preservation test adjacent to `test_initial_auth_failure_is_retained_and_reaped`**

Add:

```python
from tests.fakes.mcp_oauth_peer import capture_oauth_state, seed_old_oauth_state
```

Add the test:

```python
def test_oauth_http_parking_reconnect_and_shutdown_preserve_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    from tools import mcp_tool

    _reset_mcp_state(mcp_tool)
    before = seed_old_oauth_state(tmp_path, "oauth-parking")
    created = []
    second_attempt = threading.Event()

    class _OAuthFailingServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            self.attempts = 0
            created.append(self)

        async def _run_http(self, config):
            del config
            self.attempts += 1
            if self.attempts == 1:
                raise PermissionError("deterministic OAuth authentication failure")
            second_attempt.set()
            raise ConnectionError("deterministic transient reconnect failure")

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _OAuthFailingServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 3600)
    monkeypatch.setattr(mcp_tool, "_is_auth_error", lambda exc: isinstance(exc, PermissionError))

    try:
        assert mcp_tool.register_mcp_servers({"oauth-parking": {"url": "https://mcp.invalid/mcp", "auth": "oauth", "connect_timeout": 5}}) == []
        server = created[0]
        assert server._task is not None and not server._task.done()
        assert capture_oauth_state(tmp_path, "oauth-parking") == before

        assert mcp_tool.reconnect_mcp_server("oauth-parking") is True
        assert second_attempt.wait(timeout=5), "parked OAuth server did not retry"
        assert capture_oauth_state(tmp_path, "oauth-parking") == before

        mcp_tool.shutdown_mcp_servers()
        assert server._task.done()
        assert capture_oauth_state(tmp_path, "oauth-parking") == before
    finally:
        _cleanup_mcp_state(mcp_tool, created)
```

- [ ] **Step 2: Prove the test is sensitive to mutation**

Temporarily invert the assertion immediately after `second_attempt.wait(...)` to `!= before`.

Run: `scripts/run_tests.sh tests/tools/test_mcp_initial_connect_shutdown.py -k oauth_http_parking -q`

Expected: FAIL at the inverted assertion because current runtime behavior preserves the artifacts.

- [ ] **Step 3: Restore the preservation assertion and run the focused test**

Run: `scripts/run_tests.sh tests/tools/test_mcp_initial_connect_shutdown.py -k oauth_http_parking -q`

Expected: `1 passed`.

- [ ] **Step 4: Run the entire lifecycle file**

Run: `scripts/run_tests.sh tests/tools/test_mcp_initial_connect_shutdown.py -q`

Expected: PASS with no pending MCP-loop task or cleanup warning.

- [ ] **Step 5: Commit**

```bash
git add tests/tools/test_mcp_initial_connect_shutdown.py
git commit -m "test(mcp): preserve OAuth tokens while runtime parks"
```

---

### Task 7: Full Verification and Baseline Evidence

**Files:**
- Verify: all paths in the File Structure table.

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: Reproducible PR evidence and a clean tests-only implementation diff relative to commit `0f428209e600727c7d1d2bc5731c92eb21081d3f`.

- [ ] **Step 1: Run the complete Chunk 0 set with expected-failure details**

```bash
scripts/run_tests.sh \
  tests/tools/test_mcp_oauth_reauth_regression.py \
  tests/hermes_cli/test_mcp_reauth_lifecycle.py \
  tests/hermes_cli/test_mcp_dashboard_oauth.py \
  tests/tui_gateway/test_mcp_oauth_reauth.py \
  tests/tools/test_mcp_initial_connect_shutdown.py \
  -q -rxX
```

Expected: dashboard has one preservation pass and seven typed expected failures; CLI has three typed expected failures; TUI has one preservation pass and two typed expected failures; runtime preservation passes. No `XPASS`, ordinary failure, error, hang, or flaky retry appears.

- [ ] **Step 2: Run neighboring OAuth regressions**

```bash
scripts/run_tests.sh \
  tests/tools/test_mcp_oauth.py \
  tests/tools/test_mcp_oauth_manager.py \
  tests/tools/test_mcp_dashboard_oauth.py \
  tests/tui_gateway/test_mcp_oauth_client_callback.py \
  tests/hermes_cli/test_mcp_config.py \
  -q
```

Expected: PASS without a flaky retry summary.

- [ ] **Step 3: Verify formatting and implementation scope**

```bash
git diff --check
git diff --name-only 0f428209e600727c7d1d2bc5731c92eb21081d3f...HEAD
```

Expected: the first command prints nothing. The second lists only the five created/modified test files from Tasks 1-6; `tests/hermes_cli/test_mcp_dashboard_oauth.py` remains absent unless its existing route seam had disappeared and required restoration.

- [ ] **Step 4: Audit diagnostics for leakage**

```bash
rg -n "safe_summary|KnownCredentialLoss|ACCESS_TOKEN_FOR_TEST_ONLY|REFRESH_TOKEN_FOR_TEST_ONLY|CLIENT_SECRET_FOR_TEST_ONLY|/private/|/Users/" \
  tests/fakes/mcp_oauth_peer.py \
  tests/tools/test_mcp_oauth_reauth_regression.py \
  tests/hermes_cli/test_mcp_reauth_lifecycle.py \
  tests/tui_gateway/test_mcp_oauth_reauth.py
```

Expected: sentinel definitions and negative assertions may match. No formatted exception, expected-failure reason, or safe summary exposes a sentinel value or absolute user path.

- [ ] **Step 5: Prepare the PR evidence without closing the issue**

Use this sanitized result shape:

```text
PASS   dashboard failure before a replacement write restores OLD
XFAIL  dashboard partial metadata/client suppresses OLD restoration (#76590)
XFAIL  CLI failure deletes OLD without rollback (#76590)
XFAIL  TUI partial metadata/client suppresses OLD restoration (#76590)
PASS   runtime auth failure parks, reconnects, and shuts down without changing OAuth artifacts
```

State that no production behavior changed and no live OAuth provider was contacted. Use `Relates to #76590`, not `Fixes #76590`; Chunk 3 closes the issue after the expected failures become passing preservation tests.

---

## Chunk 3 Reuse Contract

Chunk 3 must reuse this harness without rewriting its lifecycle vocabulary or artifact oracle:

1. Route staged reauthorization through the same `FakeOAuthMCPPeer.probe()` boundary.
2. Require `after == before` for every pre-commit failure, and for every `rejected` probe.
3. Drive the F-2 taxonomy through the Chunk 0 `kind=` / `probe_outcome=` parameters — an `indeterminate` pre-token failure through one retry then `authorization_endpoint_unavailable`; an `indeterminate` probe through one retry then a commit flagged `probe=deferred`. Do not add new injection points or a new peer.
4. Remove each `_GH_76590` marker when its row passes.
5. Treat `XPASS(strict)` as incomplete marker cleanup and any remaining `XFAIL` as an incomplete transactional fix.
6. Retain the runtime parking/reconnect preservation test unchanged.

Chunk 2 consumes the same `kind` / `probe_outcome` attributes to test its `classify_outcome` classifier and the `authorization_endpoint_unavailable` code, without the retry or commit behavior.

Chunk 0 does not decide the store protocol, staged adapter implementation, bundle revision format, Keychain identity, expiration policy, or migration command. Those remain owned by Chunks 1 through 7.
