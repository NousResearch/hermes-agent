# LLM Egress Firewall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce a fail-closed, source-bound privacy boundary on every centralized Hermes LLM request and record content-free decisions.

**Architecture:** A new core firewall classifies destinations, validates Hermes-issued source-slice grants, scans the final serialized request, and appends a content-free receipt before any remote callback. Main, Relay-rewritten, and auxiliary requests share this owner; trusted read surfaces create immutable provenance grants.

**Tech Stack:** Python 3.11, dataclasses, hashlib, JSON, `fcntl`, existing Hermes file-safety/redaction/provider plumbing, pytest.

**Spec:** `docs/superpowers/specs/2026-08-25-llm-egress-and-model-routing-design.md`

## Global Constraints

- Unknown, LAN, Tailscale, container-DNS, ACP, and missing destinations use remote policy and fail closed.
- Remote source bytes require an exact canonical path, line interval, SHA-256, session, turn, request, and policy binding created by Hermes.
- Secret detection rejects the request; it never silently rewrites source.
- Receipts contain no prompt, source, tool output, credential, or absolute home path.
- Routing fallback may not weaken privacy, approval, CI, identity, or merge gates.
- Phase one covers main, Relay, and centralized auxiliary paths; standalone SDK utilities remain explicitly out of scope until migrated.

---

### Task 1: Firewall policy, destination classification, and receipts

**Files:**
- Create: `agent/llm_egress_firewall.py`
- Create: `tests/agent/test_llm_egress_firewall.py`

**Interfaces:**
- Produces: `DestinationClass`, `SourceGrant`, `EgressDecision`, `EgressBlocked`, `classify_destination(provider: str, base_url: str | None, api_mode: str | None) -> DestinationClass`, and `LLMEgressFirewall.preflight(...) -> EgressDecision`.
- Consumes: `agent.file_safety.get_read_block_error` and `agent.redact.redact_sensitive_text`.

- [ ] **Step 1: Write failing destination-classification and receipt-content tests**

```python
def test_lan_and_unknown_are_remote_while_numeric_loopback_is_loopback():
    assert classify_destination("ollama", "http://127.0.0.1:11434", None).value == "loopback"
    assert classify_destination("custom", "http://192.168.1.9:8000", None).value == "remote"
    assert classify_destination("custom", None, None).value == "unknown"

def test_allow_receipt_contains_hashes_and_counts_but_no_payload(tmp_path):
    decision = firewall(tmp_path).preflight(request, remote_route, grants=(grant,))
    receipt = json.loads((tmp_path / "llm-egress-receipts.jsonl").read_text().splitlines()[0])
    assert receipt["payload_sha256"] == decision.payload_sha256
    assert "private source" not in json.dumps(receipt)
```

- [ ] **Step 2: Run the focused tests and verify missing symbols fail**

Run: `pytest -q tests/agent/test_llm_egress_firewall.py`

Expected: collection or import failure for `agent.llm_egress_firewall`.

- [ ] **Step 3: Implement strict classification, immutable dataclasses, byte/token caps, forced secret comparison, and locked `0600` append**

```python
class DestinationClass(StrEnum):
    LOCAL_PROCESS = "local_process"
    LOOPBACK = "loopback"
    REMOTE = "remote"
    UNKNOWN = "unknown"

@dataclass(frozen=True, slots=True)
class SourceGrant:
    canonical_path: Path
    display_path: str
    line_start: int
    line_end: int
    content_sha256: str
    byte_count: int
    session_id: str
    turn_id: str
    request_id: str
    policy_digest: str
```

Open the receipt ledger with `os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW`, mode `0o600`, and `fcntl.flock(fd, fcntl.LOCK_EX)` around one newline-delimited JSON write.

- [ ] **Step 4: Run boundary, scanner-error, permission, and concurrent-append tests**

Run: `pytest -q tests/agent/test_llm_egress_firewall.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the core**

```bash
git add agent/llm_egress_firewall.py tests/agent/test_llm_egress_firewall.py
git commit -m "feat(security): add source-bound LLM egress firewall"
```

### Task 2: Trusted exact-slice provenance

**Files:**
- Modify: `agent/context_references.py`
- Modify: `tools/file_tools.py`
- Modify: `agent/tool_executor.py`
- Create: `agent/source_provenance.py`
- Modify: `tests/agent/test_context_references.py`
- Modify: `tests/tools/test_file_tools.py`
- Create: `tests/agent/test_source_provenance.py`

**Interfaces:**
- Consumes: `SourceGrant` from Task 1.
- Produces: `SourceProvenanceRegistry.issue_file_slice(...) -> SourceGrant`, `grants_for_request(request_id: str) -> tuple[SourceGrant, ...]`, and `clear_request(request_id: str) -> None`.

- [ ] **Step 1: Write failing exact-range, adjacent-line, mutation, symlink, and forged-result tests**

```python
def test_file_reference_issues_only_the_resolved_exact_slice(tmp_path):
    result = expand("@file:src/app.py:4-7", request_id="req-1")
    grant = registry.grants_for_request("req-1")[0]
    assert (grant.line_start, grant.line_end) == (4, 7)
    assert grant.content_sha256 == sha256(exact_lines(result, 4, 7)).hexdigest()

def test_terminal_cat_output_cannot_issue_source_provenance():
    executor.attach_tool_result("terminal", "private source", request_id="req-1")
    assert registry.grants_for_request("req-1") == ()
```

- [ ] **Step 2: Run focused tests and verify they fail before provenance exists**

Run: `pytest -q tests/agent/test_source_provenance.py tests/agent/test_context_references.py tests/tools/test_file_tools.py`

- [ ] **Step 3: Implement a request-scoped registry and issue grants only after canonical read and redaction success**

```python
def issue_file_slice(self, *, path: Path, line_start: int, line_end: int,
                     content: bytes, session_id: str, turn_id: str,
                     request_id: str, policy_digest: str) -> SourceGrant:
    canonical = path.resolve(strict=True)
    if get_read_block_error(canonical) is not None:
        raise SourceProvenanceError("sensitive_path")
    return SourceGrant(canonical, safe_display_path(canonical), line_start, line_end,
                       sha256(content).hexdigest(), len(content), session_id,
                       turn_id, request_id, policy_digest)
```

Only `@file` exact slices and `read_file_tool` successful bounded reads call this method. `tool_executor` attaches the opaque grant identity to trusted conversation metadata without putting it in provider messages.

- [ ] **Step 4: Run focused provenance tests**

Run: `pytest -q tests/agent/test_source_provenance.py tests/agent/test_context_references.py tests/tools/test_file_tools.py`

Expected: all pass.

- [ ] **Step 5: Commit provenance producers**

```bash
git add agent/source_provenance.py agent/context_references.py tools/file_tools.py agent/tool_executor.py tests/agent/test_source_provenance.py tests/agent/test_context_references.py tests/tools/test_file_tools.py
git commit -m "feat(security): bind exact source slices to requests"
```

### Task 3: Main request and Relay enforcement

**Files:**
- Modify: `agent/conversation_loop.py`
- Modify: `agent/relay_llm.py`
- Modify: `tests/run_agent/test_run_agent.py`
- Modify: `tests/run_agent/test_run_agent_codex_responses.py`
- Modify: `tests/agent/test_relay_llm.py`
- Modify: `tests/hermes_cli/test_plugins.py`

**Interfaces:**
- Consumes: `LLMEgressFirewall.preflight` and `SourceProvenanceRegistry.grants_for_request`.
- Produces: one mandatory final-body preflight before each provider callback.

- [ ] **Step 1: Write failing tests proving middleware and Relay cannot inject forbidden bytes**

```python
def test_execution_middleware_secret_injection_makes_zero_provider_calls(fake_provider):
    with pytest.raises(EgressBlocked):
        run_agent(execution_middleware=inject_canary, provider=fake_provider)
    assert fake_provider.calls == []

def test_relay_rewrite_is_preflighted_after_interceptor(fake_provider):
    relay = relay_with_interceptor(lambda body: add_canary(body))
    with pytest.raises(EgressBlocked):
        relay.complete(request)
    assert fake_provider.calls == []
```

- [ ] **Step 2: Run the named main/Responses/Relay tests and confirm provider callbacks currently fire**

Run: `pytest -q tests/run_agent/test_run_agent.py tests/run_agent/test_run_agent_codex_responses.py tests/agent/test_relay_llm.py tests/hermes_cli/test_plugins.py -k egress`

- [ ] **Step 3: Preflight `_perform_api_call`'s final logical body and each Relay `final_request` immediately before callbacks**

```python
decision = agent.llm_egress_firewall.preflight(
    request=api_kwargs,
    route=resolved_egress_route,
    grants=agent.source_provenance.grants_for_request(request_id),
)
```

Do not catch `EgressBlocked` in middleware exception-swallowing paths. Translate it once at the conversation boundary into a typed local-fallback or blocked turn result.

- [ ] **Step 4: Run the main and Relay suites**

Run: `pytest -q tests/run_agent/test_run_agent.py tests/run_agent/test_run_agent_codex_responses.py tests/agent/test_relay_llm.py tests/hermes_cli/test_plugins.py`

- [ ] **Step 5: Commit centralized main-path enforcement**

```bash
git add agent/conversation_loop.py agent/relay_llm.py tests/run_agent/test_run_agent.py tests/run_agent/test_run_agent_codex_responses.py tests/agent/test_relay_llm.py tests/hermes_cli/test_plugins.py
git commit -m "feat(security): enforce final LLM request preflight"
```

### Task 4: Auxiliary enforcement and privacy-compatible fallback

**Files:**
- Modify: `agent/auxiliary_client.py`
- Modify: `tests/agent/test_auxiliary_client.py`
- Modify: `tests/agent/test_auxiliary_relay.py`
- Modify: `tests/agent/test_auxiliary_transient_retry.py`

**Interfaces:**
- Consumes: firewall and provenance interfaces from Tasks 1-2.
- Produces: `resolve_privacy_fallback(task_kind: str, blocked_route: Route) -> Route | None` and enforcement for sync, async, and streaming auxiliary calls.

- [ ] **Step 1: Write failing compression, title, vision, review, curator, reference-aggregator, MCP, approval, and Skills Hub tests**

```python
@pytest.mark.parametrize("task_kind", [
    "compression", "title", "vision", "review", "curator",
    "reference_aggregator", "mcp", "approval", "skills_hub",
])
def test_auxiliary_remote_canary_uses_local_fallback_without_remote_call(task_kind):
    result = client.call(task_kind, payload_with_canary)
    assert result.route.destination_class in {"local_process", "loopback"}
    assert remote.calls == []
```

- [ ] **Step 2: Run auxiliary tests and observe unguarded direct callbacks**

Run: `pytest -q tests/agent/test_auxiliary_client.py tests/agent/test_auxiliary_relay.py tests/agent/test_auxiliary_transient_retry.py -k egress`

- [ ] **Step 3: Enforce before direct `_relay_sync_completion`, `_relay_async_completion`, and `_relay_sync_stream` callbacks**

```python
try:
    firewall.preflight(request=kwargs, route=route, grants=grants)
except EgressBlocked:
    fallback = resolve_privacy_fallback(task_kind, route)
    if fallback is None:
        raise
    return invoke_once(fallback, kwargs)
```

Privacy fallback does not consume the transient provider retry budget and cannot return to the blocked remote route.

- [ ] **Step 4: Run complete auxiliary tests**

Run: `pytest -q tests/agent/test_auxiliary_client.py tests/agent/test_auxiliary_relay.py tests/agent/test_auxiliary_transient_retry.py tests/agent/test_auxiliary_concurrency.py`

- [ ] **Step 5: Commit auxiliary enforcement**

```bash
git add agent/auxiliary_client.py tests/agent/test_auxiliary_client.py tests/agent/test_auxiliary_relay.py tests/agent/test_auxiliary_transient_retry.py
git commit -m "feat(security): guard auxiliary model egress"
```

### Task 5: Configuration, observability, and end-to-end proof

**Files:**
- Modify: `cli-config.yaml.example`
- Modify: `docs/security/network-egress-isolation.md`
- Create: `tests/e2e/test_llm_egress_firewall.py`
- Modify: `tests/monitoring/test_export_redaction.py`

**Interfaces:**
- Consumes: all prior task interfaces.
- Produces: validated `llm_egress` configuration and end-to-end acceptance evidence.

- [ ] **Step 1: Write failing config and fake-provider E2E tests**

```python
def test_remote_exact_slice_allowed_but_neighbor_and_terminal_copy_denied(fake_http):
    assert run_remote("@file:src/app.py:4-7").status == "ok"
    assert run_remote("@file:src/app.py:3-8").reason == "slice_not_granted"
    assert run_remote(terminal_cat("src/app.py")).reason == "untrusted_provenance"
    assert fake_http.request_count == 1
```

- [ ] **Step 2: Run E2E and monitoring tests to verify the configuration is absent**

Run: `pytest -q tests/e2e/test_llm_egress_firewall.py tests/monitoring/test_export_redaction.py`

- [ ] **Step 3: Add strict defaults, schema validation, receipt documentation, and the explicit phase-one limitation**

```yaml
llm_egress:
  mode: enforce
  remote_source_slices: false
  max_serialized_bytes: 262144
  conservative_chars_per_token: 3
  receipt_required: true
```

- [ ] **Step 4: Run security and regression verification**

Run: `pytest -q tests/agent/test_llm_egress_firewall.py tests/agent/test_source_provenance.py tests/agent/test_context_references.py tests/agent/test_file_safety_credentials.py tests/agent/test_relay_llm.py tests/agent/test_auxiliary_client.py tests/e2e/test_llm_egress_firewall.py tests/monitoring/test_export_redaction.py`

Run: `ruff check agent/llm_egress_firewall.py agent/source_provenance.py tests/agent/test_llm_egress_firewall.py tests/agent/test_source_provenance.py`

- [ ] **Step 5: Commit documentation and acceptance proof**

```bash
git add cli-config.yaml.example docs/security/network-egress-isolation.md tests/e2e/test_llm_egress_firewall.py tests/monitoring/test_export_redaction.py
git commit -m "test(security): certify guarded LLM egress"
```
