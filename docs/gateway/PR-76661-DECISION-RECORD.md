---
tags: [federation, decision-record, security, proxy-mode]
---

# PR #76661 Decision Record — Federation Cluster

> **Audit Trail**: This document captures all research, alternatives
> considered, and the rationale for the implemented approach. To be
> reviewed by maintainers alongside the SECURITY-AUDIT-PR76661.md and
> SECURITY-BASELINE.md.

## 1. Problem Statement

We need **multi-device task relay** for Hermes: when a device in a Hermes
federation goes offline, the in-flight task should be automatically
picked up by another device, with the user able to control the approval
flow.

### 1.1 User Scenarios

| Scenario | Description |
|---|---|
| **A** | Mac A runs long task, Mac A dies, Mac B picks up |
| **B** | User on iPhone (remote) sends message, runs on home Mac |
| **C** | Client (any HTTP) connects to remote Hermes API server |
| **D** | iPhone + Mac + Mac all in same Apple ID, automatic sync |
| **E** | Cross-platform: phone, laptop, cloud VM |

### 1.2 Success Criteria

1. **Task continues** when a node dies mid-flight (5s detection)
2. **User decides** whether to allow relay (ask/auto/review modes)
3. **Multi-transport** — any connection method works
4. **Capability matching** — task routes to best-fit node
5. **No single point of failure** — multi-node failover
6. **Defense in depth** — survives 0day in any single primitive

---

## 2. Research: Existing Solutions

### 2.1 Survey

We evaluated 5 candidate approaches:

| # | Approach | Stack | LOC | Dependencies |
|---|---|---|---|---|
| 1 | iCloud Drive SQLite sync (PR #76661 v1) | SQLite + iCloud | 9723 | None |
| 2 | WebSocket + mDNS (PR #76661 v2) | WS + Bonjour | 9723 | Apple devices |
| 3 | Proxy Mode + active probing (this PR) | HTTP + SSE | +1000 | None |
| 4 | libp2p / IPFS | libp2p | +3000 | libp2p runtime |
| 5 | Custom ad-hoc polling | HTTP long-poll | +500 | None |

### 2.2 Proxy Mode — already implemented

Hermes has `gateway/run.py:23530: _run_agent_via_proxy()`:

```python
async def _run_agent_via_proxy(self, ...):
    """Forward to remote Hermes API server."""
    async with session.post(
        f"{proxy_url}/v1/chat/completions",
        json=body, headers=headers
    ) as resp:
        # Parse SSE stream
```

**Capabilities**:
- HTTP + SSE (OpenAI Chat Completions protocol)
- Session continuity via `X-Hermes-Session-Id`
- Bearer Token auth

**Limitations**:
- 1:1 topology (one client → one server)
- 30min timeout on dead server
- No in-flight task relay
- No capability matching
- No cluster visibility

### 2.3 PR #76661 Original — over-engineered

Original PR introduced 12 phases, 9723 LOC, 159 tests:

```
Phase 1-2: Heartbeat + protocol
Phase 3: Raft-lite consensus
Phase 5: mDNS discovery
Phase 6: Security hardening
Phase 7-9: Memory sync / cron / skill sync
Phase 10-12: Cluster, API, Desktop Bridge
```

**Issues identified**:
- ⚠️ 90s relay latency (60s heartbeat + 30s threshold)
- ⚠️ iCloud sync dependency (5-30s lag)
- ⚠️ Apple's only (iCloud limits)
- ⚠️ Doesn't leverage Proxy Mode (reinvents)
- ⚠️ Sweeper verdict: `salvageability: low`

### 2.4 Comparison

| Capability | Proxy Mode | PR #76661 v1 | libp2p | Active Probe |
|---|---|---|---|---|
| 1:1 reroute | ✅ | ❌ | ✅ | ✅ |
| N:N cluster | ❌ | ✅ | ✅ | ✅ |
| Sub-second health | ❌ | ❌ | ✅ | ✅ |
| In-flight relay | ❌ | ✅ | ⚠️ | ✅ |
| Apple-optimized | ❌ | ✅ | ❌ | ✅ |
| Cross-platform | ✅ | ❌ | ✅ | ✅ |
| Transport choice | ❌ | ❌ | ✅ | ✅ |
| Total LOC | 0 (existing) | 9723 | 3000+ | +1000 |
| 0day resilience | None | HMAC only | libp2p | Pluggable |

---

## 3. Decision

### 3.1 Chosen Approach: Cluster Federation on top of Proxy Mode

**Rationale:**

1. **Reuse over rebuild** — Proxy Mode already handles HTTP+SSE+session.
   Adding 1000 LOC of cluster coordination on top is **10x more efficient**
   than rewriting 9723 LOC.

2. **Transport-agnostic** — same code works for HTTP, WebSocket, iCloud,
   mDNS, future transports. Federation = "any connected node can cooperate".

3. **Sub-second detection** — active HTTP probing (1s) vs SQLite polling
   (60s+). 50x faster relay.

4. **Smaller attack surface** — 1000 LOC is easier to audit than 9723 LOC.

5. **AI-driven decision** — confidence-based approval (ask/auto/review)
   instead of blanket relay.

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: AI Decision (NEW)                                  │
│  - confidence-based approval                                 │
│  - ask/auto/review modes                                     │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Coordination (NEW)                                 │
│  - node selection, task routing, relay coordination          │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Transport (NEW, abstract)                          │
│  - HTTP, WebSocket, iCloud, mDNS — pluggable                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Proxy Mode (REUSE, 0 LOC changed)                  │
│  - HTTP+SSE, OpenAI protocol, session continuity             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 What we keep from PR #76661

- ✅ Federation protocol (FedMessage, MessageType, PeerInfo)
- ✅ HMAC signature scheme
- ✅ Heartbeat state machine
- ✅ Test infrastructure (159 tests)
- ✅ CLI (`hermes fed` subcommand)
- ✅ Desktop overlay UI

### 3.4 What we replace

| Old | New | Reason |
|---|---|---|
| iCloud SQLite (60s polling) | HTTP active probe (1s) | 60x faster |
| Whisper-only relay | ask/auto/review modes | User control |
| Discovery via mDNS | Discovery via any transport | Apple+ cross-platform |
| Static node list | Auto-discovery + manual | Flexible |

### 3.5 What we drop

- ❌ Cron relay (out of scope, doesn't help with task relay)
- ❌ Memory sync (privacy concerns, separate PR)
- ❌ Skill sync (out of scope)
- ❌ Leader election (premature optimization)
- ❌ Custom SQLite consensus (proxy mode already has reliable transport)

### 3.6 Security First

Because the chosen approach involves **multi-node trust** and **task
content sharing**, we deferred implementation until security assessment
complete. See SECURITY-BASELINE.md and SECURITY-AUDIT-PR76661.md.

**8 pillars** of security audited:
1. Identity & Authentication
2. Transport Security
3. Data Integrity
4. Authorization
5. Resilience
6. Audit & Logging
7. Privacy
8. Operational Security

**4 critical findings** must be fixed before merge:
- TLS mandatory
- Trust 评级
- 3-failure death rule
- Audit trail

---

## 4. Alternatives Considered (and Why Not Chosen)

### 4.1 "Just use iCloud SQLite" (PR #76661 v1)

**Pros**: Apple-optimized, no infrastructure.
**Cons**: 60s relay latency, Apple-only, slow.
**Verdict**: Rejected — too slow for user's "5s relay" requirement.

### 4.2 "Just use libp2p"

**Pros**: Mature P2P framework, transport-agnostic.
**Cons**: 3000+ LOC dependency, complex.
**Verdict**: Rejected — overkill for our use case.

### 4.3 "Just use Tailscale + existing tools"

**Pros**: Easy networking.
**Cons**: Doesn't solve task relay, only connectivity.
**Verdict**: Adopted as one transport option, not the whole solution.

### 4.4 "Close PR #76661, don't add federation"

**Pros**: Less code, less attack surface.
**Cons**: User's explicit need for multi-device relay unmet.
**Verdict**: Rejected — user requirement is non-negotiable.

### 4.5 "Use Proxy Mode + minimal hub"

**Pros**: Minimal code, leverages existing.
**Cons**: 1:N only (no N:N).
**Verdict**: Adopted as Layer 1.

---

## 5. Implementation Plan

### Phase 16: Transport Abstraction (Days 1-2)

- [ ] `gateway/federation/transport.py` — abstract Transport interface
- [ ] 4 concrete transports: HTTP, WebSocket, iCloud, mDNS
- [ ] Transport negotiation (pick best available)

### Phase 17: Coordination Layer (Days 3-5)

- [ ] `gateway/federation/cluster.py` — ClusterCoordinator
- [ ] Node registry, capability tracking
- [ ] Active probing (1s interval)
- [ ] 3-failure death rule

### Phase 18: AI Decision Layer (Days 6-8)

- [ ] `gateway/federation/evaluator.py` — ConfidenceEvaluator
- [ ] `gateway/federation/policy.py` — ApprovalPolicy
- [ ] ask/auto/review modes
- [ ] User notification + Telegram integration

### Phase 19: Security Hardening (Days 9-10)

- [ ] Ed25519 node identity
- [ ] Trust 评级
- [ ] Mandatory TLS enforcement
- [ ] Audit log
- [ ] Pen tests (PEN-1 to PEN-8)
- [ ] Fuzz tests

### Phase 20: Documentation (Day 11)

- [ ] `docs/gateway/federation.md` — user guide
- [ ] `docs/gateway/federation-transport.md` — transport layer
- [ ] `docs/gateway/PR-76661-DECISION-RECORD.md` — this file
- [ ] `docs/gateway/SECURITY-AUDIT-PR76661.md` — audit report
- [ ] Update PR description

### Phase 21: Real Device Test (Day 12)

- [ ] 2-3 Mac mini M5 Max + MacBook
- [ ] Multi-transport test
- [ ] Failure scenario dry-run
- [ ] CI + push

---

## 6. Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Transport incompat | High | Layer 2 abstract + fallback |
| TLS downgrade attack | Critical | Default `require_tls: true` |
| Compromise node | Critical | Trust 评级 + sensitive task filter |
| False death | Medium | 3-failure rule |
| User decision delay | Medium | 10s timeout = auto-accept |
| iCloud sync conflict | Medium | Last-write-wins + audit log |
| 0day crypto | High | Crypto-agile + multi-layer defense |
| Token leak | Critical | Keychain + never log |

---

## 7. Success Metrics

| Metric | Target | Measurement |
|---|---|---|
| Detection latency | ≤ 5s | Heartbeat probe interval × 3 |
| Relay latency | ≤ 15s | Detection + decision + claim |
| False death rate | < 1% | Integration tests |
| Test coverage | ≥ 90% | pytest-cov |
| Security audit | 100% baseline | SECURITY-AUDIT-PR76661 |
| User approval | < 10s | avg time-to-decision |
| Transport diversity | ≥ 3 | Active transports |

---

## 8. Compliance

| Policy | Documentation |
|---|---|
| OWASP API Security Top 10 | Pillar 2, 4, 5 |
| NIST SP 800-57 (Key Mgmt) | Pillar 1, 8 |
| CIS Controls | Pillar 5, 6 |
| RFC 8446 (TLS 1.3) | Pillar 2 |
| RFC 8032 (Ed25519) | Pillar 1 |

---

## 9. Open Questions

1. Should we bundle Pen Test Suite as a separate workflow?
2. Will iCloud sync conflict resolution use CRDT or last-write-wins?
3. Should ASK mode use Telegram inline keyboard or button URL?
4. Should we offer a "federation-only" build for users who don't want the cluster?

---

## 10. References

- [SECURITY-BASELINE.md](SECURITY-BASELINE.md) — 8-pillar baseline
- [SECURITY-AUDIT-PR76661.md](SECURITY-AUDIT-PR76661.md) — audit report
- [Hermes Proxy Mode reference](https://github.com/NousResearch/hermes-agent/blob/main/gateway/run.py#L23530)
- [PR #76661](https://github.com/NousResearch/hermes-agent/pull/76661)

---

## 11. Change Log

| Date | Author | Change |
|---|---|---|
| 2026-08-05 | 鲸 (ENTJ) | Initial decision record |
| 2026-08-05 | 鲸 | Security baseline + audit |
| TBD | 鲸 | Implementation |
| TBD | Maintainer | Review |
