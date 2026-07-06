# GitHub Ask-Hermes Topic Context Ordering — Implementation Plan (v3, post-review-2)

**Goal:** GitHub "Ask Hermes" questions on a PR behave EXACTLY like Alfredo typing in that Telegram topic: ONE shared, ORDERED conversation. "this too" sees prior Q&A; answers never post out of order.

**Status:** v3. Folds the 5 CONVERGED blockers from round-2 dual-review (both reviewers, same findings). v2's serializer-in-run.py was fatally late.

## Root cause (verified, PR 7735)
- `webhook.py:668` `session_chat_id = f"webhook:{route_name}:{delivery_id}"` → unique session per delivery.
- `webhook.py:712` `asyncio.create_task(self.handle_message(event))` → concurrent, no order.
- Result: Q2 (398s) posted before Q1 (1008s); Q2 never saw Q1.

## The v2→v3 correction (why the 5 blockers killed v2)

| # | v2 was wrong because | v3 fix |
|---|---|---|
| 1 FATAL | serializer in run.py learns the key AFTER tasks already started (base.py:3972 computes key, 4144 sets guard, BEFORE pre_gateway_dispatch at run.py:7451) | resolve binding at **webhook intake** before create_task → the EXISTING `_active_sessions` guard (base.py:3987/4144) serializes FIFO. No new serializer. |
| 2 | bridge's final response is "Posted to topic X", not the answer body; answer goes out via Bot API (bypasses session). Shared key alone gives Q2 Q1's *question* but not Q1's *answer*. | bridge skill: final assistant message IS the answer (deliver=log discards for delivery, but it lands in the shared transcript). Seed demoted to legacy/fallback to avoid double-context. |
| 3 | "≤1 rewrite" merge policy undefined | exact deterministic merge contract (below) + permutation tests |
| 4 | raw `session_key_suffix` string = collision footgun | constrained `SessionBinding(namespace, key)`; core builds `agent:<profile>:<namespace>:<key>` |
| 5 | reverse-scan "pick OR refuse" contradictory | exact-unique match: 0=no bind, 1=bind, >1=REFUSE+log |

## v3 Architecture

### Binding resolved at INTAKE (the load-bearing change)
A new generic, synchronous resolution step in `webhook.py._handle_webhook` BEFORE `create_task`:
- compute `arrival_seq` (monotonic) and the `SessionBinding` for the event.
- the binding is computed by a synchronous resolver (a new `pre_session_binding` hook OR a route-config field — see Decision 1) so it is known before any task scheduling.
- stamp both onto the event/source.

Because `handle_message` (base.py:3954) computes the session key (3972) and sets the `_active_sessions` guard (4144) synchronously on first entry, and queues a same-key second arrival as pending (4130-4135), two deliveries with the SAME binding serialize FIFO through the existing machinery. `arrival_seq` orders the pending-drain as a robustness tiebreaker.

### `SessionBinding` primitive (core)

```data-model
{"entities":[{"name":"SessionBinding (core, new)","change":"added","fields":[{"name":"namespace","type":"str","note":"reserved segment, e.g. 'gh-review'. NOT a platform name. validated charset."},{"name":"key","type":"str","note":"opaque id, e.g. topic_key. validated: no colon-impersonation, bounded len."}]},{"name":"SessionSource (core)","change":"modified","fields":[{"name":"session_binding","type":"SessionBinding?","change":"added","note":"build_session_key emits agent:<profile>:<namespace>:<key> -> profile-safe, collision-safe"}]},{"name":"MessageEvent (core)","change":"modified","fields":[{"name":"arrival_seq","type":"int?","change":"added","note":"monotonic, stamped at webhook intake -> receipt order for pending-drain"}]}]}
```

### Hook merge contract (run.py, replaces short-circuit at 7483-7490)
1. invoke ALL pre_gateway_dispatch hooks in registration order.
2. any `skip` → terminal for dispatch (logged); still note other directives' side effects already ran.
3. collect `session_binding` from any result; **>1 DISTINCT binding → refuse all + log** (fail-closed).
4. first `rewrite` in registration order wins; later rewrites ignored with a warning.
5. `allow` is a no-op; never suppresses binding/rewrite collection.

(Binding via this hook path is the FALLBACK; the primary binding is resolved at intake per Decision 1. Both funnel through the same merge contract.)

### Telegram reverse-bind (exact-unique)

```annotated-code
{"filename":"gh-review-opener telegram hook","language":"python","code":"hits = [k for k,v in topics.items()\n        if str(v.get('thread_id'))==tid and str(v.get('chat_id'))==chat]\nif len(hits)==1: bind(hits[0])\nelif len(hits)==0: return None        # no binding\nelse: log.warning('ambiguous',hits); return None  # REFUSE","annotations":[{"lines":"1-2","label":"durable source","note":"gh-review-topics.json maps topic_key->{chat_id,thread_id}; reverse-scan it, NOT the one-shot seed sidecar"},{"lines":"5","label":"fail-closed","note":">1 stale key for one thread -> refuse, never leak a reply into the wrong PR session"}]}
```

## Files

```file-tree
{"title":"v3 touched","entries":[{"path":"gateway/session.py","change":"modified","note":"SessionBinding type + session_binding field + namespaced build + roundtrip"},{"path":"gateway/platforms/base.py","change":"modified","note":"MessageEvent.arrival_seq"},{"path":"gateway/platforms/webhook.py","change":"modified","note":"resolve binding + stamp arrival_seq at intake BEFORE create_task"},{"path":"gateway/run.py","change":"modified","note":"hook-result MERGE contract"},{"path":"plugins/gh-review-opener/__init__.py","change":"modified","note":"GitHub side binds via topic_key; Telegram hook reverse-scans topics.json exact-unique"},{"path":"skills/.../github-review-bridge/SKILL.md","change":"modified","note":"final assistant message = the answer body; seed demoted to fallback"}]}
```

## Tasks (TDD)

| # | Task | Proves |
|---|---|---|
| 1 | SessionBinding type + validation (namespace MUST reject any Platform enum value, e.g. `webhook`/`telegram` — not just charset) + namespaced build_session_key + roundtrip | multiplex -> agent:coder:gh-review:k; invalid bindings rejected; namespace==platform refused; no collision w/ platform keys |
| 2 | MessageEvent.arrival_seq | field present, serialized |
| 3 | webhook intake resolves binding + stamps arrival_seq BEFORE create_task | binding known pre-dispatch; the FIFO-critical step |
| 4 | existing _active_sessions guard serializes same-binding deliveries via a TRUE FIFO queue (NOT merge_text=True, which coalesces TEXT events) + HARD-ASSERT webhook synthetic turns take the queue branch (never steer/interrupt) | Q2 delayed-before-bind STILL waits for Q1 (regression w/ artificial seq1 delay) under steer AND interrupt |
| 5 | hook-result MERGE contract | gh-review binding + topic-rename seed both apply; 2 rewrites deterministic; >1 binding refused; allow no-op |
| 6 | opener GitHub + Telegram(exact-unique reverse-scan) binding | both surfaces -> one key; ambiguous thread refused |
| 7 | bridge final response = answer body; transcript holds it | Q2 loads Q1 question AND answer w/o consuming a seed |
| 8 | webhook delivery routing unchanged | _delivery_info keyed per delivery_id, independent of binding |
| 9 | edge matrix + E2E smoke + cache-evict + docs + restart + live PR | new/reused/force-new/429/recreated all green |

## Risks / decisions

> [!RISK] Human steer/interrupt semantics. Telegram in-topic replies bind to the shared key too. Bypassing busy-handling for them would break Alfredo steering a running answer. Mitigation: ONLY synthetic webhook turns (chat_type==webhook + arrival_seq) take the ordered-queue path; human Telegram replies keep normal steer/interrupt but use the shared key. Tested both modes (Task 4).

> [!RISK] Binding resolved at intake needs the resolver to be synchronous + cheap (no network). gh-review topic_key derivation is pure (`derive_key_name`). Telegram reverse-scan is a small JSON read. Both safe in the request handler.

> [!DECISION] 1. Intake resolver mechanism: (a) new generic `pre_session_binding` hook fired synchronously in webhook intake, or (b) a route-config `session_binding_from` field. Rec: **(a) hook** — keeps gh-review logic in the plugin, reuses the plugin system, no new config schema. Slightly more core surface than (b).

> [!DECISION] 2. force_new_topic = own nonce'd key = separate conversation. Rec: KEEP SEPARATE (force-new = fresh topic by intent).

> [!DECISION] 3. Telegram in-topic history migration: post-ship, replies key to the binding instead of the old telegram thread key; pre-existing history not carried (no transcript merge). Rec: ACCEPT (short-lived Q&A topics).

> [!DECISION] 4. Answer-in-transcript: make the bridge's FINAL assistant message the answer body (deliver=log discards it for delivery; it lands in the shared transcript so Q2 sees it). Seed demoted to fallback. Rec: ACCEPT — cleanest, no mirroring, no double-context.

---

## Build Notes (deviations from plan)

Folded during implementation + pre-merge review (all verified):
- **FIFO wiring (pre-merge blocker 1).** v3 said "the existing _active_sessions guard serializes." That collapses concurrency but does NOT stop `busy_input_mode: steer` from splicing Q2 into Q1, nor `interrupt` from aborting Q1, nor `merge_text` coalescing. Added an explicit early branch in `run.py _handle_active_session_busy_message`: a synthetic webhook turn (`chat_type=="webhook"` + `arrival_seq` set) routes through `_queue_or_replace_pending_event` (per-turn FIFO, no coalescing) and returns before steer/interrupt. This is what makes `arrival_seq` load-bearing. Tested under steer/interrupt/queue.
- **Multiplex guard key (pre-merge blocker 2).** `base.py handle_message` computed the active-session guard key via bare `build_session_key` (no profile), so two same-binding deliveries under different `/p/<profile>` routes collided. Routed the guard through `self._session_store._generate_session_key(source)` (profile + binding aware), fallback to `build_session_key` when no store. No-op for default-profile no-binding traffic.
- **Blocker 3 (plugin/skill not in repo diff) = N/A.** The `gh-review-opener` plugin and `github-review-bridge` skill live under `~/.hermes/` BY DESIGN (survive `hermes update`, which git-resets the fork clone). Edited the live files directly; verified in place. Repo diff is core-only on purpose.

## Verification (final)
- 150 tests pass (test_session_binding, test_pre_dispatch_merge, test_webhook_intake_binding, test_webhook_fifo_ordering, plugin test_opener_binding, + dispatch/session/topic regression).
- E2E smoke: PR 7735 Q1+Q2+Telegram reply all resolve to `agent:main:gh-review:gh-pr-tdr-autosync-atx-activix-crm-7735`; delivery routing per-delivery intact.
- Full-suite mass failures are PRE-EXISTING harness pollution (proven: test_busy_session_ack::test_telegram_omits_status_detail_by_default fails identically on clean main with the same neighbor batch; passes isolated).
