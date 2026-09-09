# Layer 7 preservation: retained source validation

This archive preserves the public descriptions and revision-specific review/test evidence from the two preservation sources before consolidation. **Historical status statements below are not current-head readiness claims.** The active preservation carrier is #104601; guarded recovery and participant emergency Stop belong to draft #105079. Original PR numbers, authored source links, failed checks and overlapping test scopes are retained verbatim. Do not sum those test selections.

The retained runtime source is `db832ed1a6de377f5fc0b31ff8aa5eb0d138cc2d`, which already includes `97d4f4dbb82efb0309a389de3b3e8348702517bf`. Consolidation adds documentation only above that source; it does not enable recovery. The active PR body and GitHub checks report current readiness.

## Historical #104601

Source: https://github.com/NousResearch/hermes-agent/pull/104601, head `97d4f4dbb82efb0309a389de3b3e8348702517bf`.

<details>
<summary>Retained public description and evidence</summary>

## Current Passive-Lineage Source

Current head: [`97d4f4dbb82efb0309a389de3b3e8348702517bf`](https://github.com/dokterdok/hermes-agent/commit/97d4f4dbb82efb0309a389de3b3e8348702517bf). [Increment above the prior public source](https://github.com/dokterdok/hermes-agent/compare/779113231b8df89c003ab1334b8ce949cf57ffcf...97d4f4dbb82efb0309a389de3b3e8348702517bf). The [required exact-head CI aggregate passed](https://github.com/NousResearch/hermes-agent/actions/runs/34229587480) on this source.

This extends the existing publisher/retirement owner with version-2 passive history and retirement while retaining unversioned legacy behavior. RoomLink protocol 2 and the catalog digest remain unchanged; capability negotiation is an optional response sibling. It adds no second publisher, model tool or execution authority.

- Owner-pinned, bounded authority descriptors are checked against actual contiguous canonical claims. The verified prefix and current sender remain distinct; an unreceived transition is pending, not verified.
- Fresh explicit `passive_only: true` invitations with `replication: true` issue only status/replication permission. Owner enrollment, exact grant scope and write-transaction revocation checks remain separate gates.
- Publisher restart/lost-ACK replay preserves exact pending coverage and rejects unsupported lineage rather than downgrading. Late old grants/notices cannot change the new copy; retirement/replacement retains one durable winner.
- Quarantined evidence no longer exposes positive lineage-verification fields. The corrupt bytes and quarantine remain retained; the correction does not repair history or make the copy merely pending.
- Descriptor bounds are 8 KiB each and 4 MiB/512 retained rows across the two descriptor stores; the existing 16 KiB retirement request limit remains. Work-record v2 is not implemented or advertised here; #104759 owns that next layer.

The initial 16-file selection passed 259 tests. Independent source review found one false-verification P2; its correction passed 112 affected tests and both preserved reviewer probes, and a focused re-review approved `0e083b8`. The latest CI follow-up restores the legacy replication-validation message and maps the existing contributor email using the repository mechanism; 56 invitation/RPC/HTTP tests and 11 contributor-map tests passed. Overlapping runs are not added together. No live backend, installed client or actual takeover was exercised.

**Partial-copy reclamation is corrected in this head.** V2 retirement now binds payload reclamation to the same non-NULL versioned lineage rather than requiring the earlier verified-prefix authority to equal the retiring sender. Legacy/mixed schemas, unsupported formats, mismatched lineage and quarantine remain fail-closed; the retired identity and exact notice replay remain intact. The original age/room/byte-pressure failures were preserved. A 154-test affected selection passed, and a separate focused safety review approved the correction with real ingress-triggered reclamation and mixed-schema controls. These selections overlap and are not summed. Retirement is not immediate physical erasure: the normal 90-day age threshold and earlier eligible capacity reclamation are opportunistic, while the retired identity remains reserved.

**No takeover is enabled.** Original-local rejoin, accepted-work/Stop/approval reconciliation, accepted-tail policy, independently established authority, guarded activation and physical-host acceptance remain in #105197/later gates. A descriptor, local epoch or operator click is not proof of one globally acting authority. The staged Desktop review timeline is unchanged.

<details>
<summary>Earlier implementation, pinned validation and field evidence (through 7791132)</summary>

## Preserve Group Chat History Beyond Its Home Gateway

When a group's home gateway goes offline, its participants should not lose the conversation. This PR adds authenticated delivery into the existing passive history store, the first step toward host-offline continuity in #97681.

**This does not enable takeover.** A stored copy is not permission to become a second coordinator. `groups.promote` and `groups.demote` remain disabled, and this does not claim zero loss of a source's unreplicated tail.

## What Is Implemented

- An explicit replication option on room-member invitations. Existing grants gain no new permission.
- RoomLink delivery bound to the authorized group, home, epoch, member, installation and profile. Broad API credentials cannot be substituted at the replica endpoint.
- Signature, expiry, reservation and revocation checks, including revalidation in the same transaction that persists history.
- Existing contiguous replay, duplicate-content checks, storage budgets, fixed membership and final disband protection. Rename events update the copied name without rewriting history.
- Bounded UTF-8 HTTP transport that also handles near-limit non-ASCII history.
- Gateway-owned background delivery, independent of Bot turns and Desktop. Shared target checkpoints and OS locks coordinate profiles and processes using the same participant gateway.
- Durable pending-page recovery after lost acknowledgments or restart. Unavailable targets back off; failed optional worker startup does not prevent normal Group Chat operation.
- Already-profiled endpoints stay on the selected profile, and temporary storage limits recover without issuing a new grant.
- Owner-authorized cleanup of participant copies after Disband, even after ordinary Bot grants expire or are revoked and the home removes its live routes. A lost cleanup acknowledgment is safe to retry.

The receiver, background publisher and copy-retirement lifecycle are implemented. This increment is API-first: owners enroll cleanup permission on both sides, then request `replication: true` through the existing invitation flow. Automatic opt-in from Desktop creation is not wired here. No new model tool, OAuth copying, whole-workspace sync or universal file mirroring is added.

Disband records that a participant's copy is retired without pretending it received missing history. The separate cleanup permission cannot run a Bot, read files or change authority. Its delivery survives removal of ordinary routes; the target denies late pages and retains the retired identity after reclaiming payloads.

## Review Boundary

<details>
<summary>Pinned review boundary</summary>

Earlier source: [`779113231b8df89c003ab1334b8ce949cf57ffcf`](https://github.com/dokterdok/hermes-agent/commit/779113231b8df89c003ab1334b8ce949cf57ffcf). Verified GitHub merge-base with main: `006b1beb00d9d25230571d14277aca3d70e5e11f` ([submitted diff](https://github.com/dokterdok/hermes-agent/compare/006b1beb00d9d25230571d14277aca3d70e5e11f...779113231b8df89c003ab1334b8ce949cf57ffcf)). This is the complete submitted comparison, not automatically a prerequisite-free increment.

The only change above reviewed `d91bb120884` creates the reviewer profile used by the invitation test in `tests/tui_gateway/test_groups_methods.py`. Runtime source is unchanged; the fixture retains its original cherry-pick provenance.

Includes exact #99107 `a4d79d874d768a6fa04c5f7e3b5310ded554765e` and #100016 `7fea9afa96a5614904a58a74d78184007c5495f0` histories. The earlier equivalent guard commits are historical provenance, not two runtime implementations.

</details>


Depends on the separately reviewable #99107 safety source and #100016's route/Disband source. Both authored histories are preserved, including the early-fence repair ported from the existing field implementation. Teknium's landed #99047 passive store remains the foundation; storage and RoomLink are extended rather than replaced. The new copy layer is not all of the main-based diff: prerequisite commits belong to their owning PRs.

This is separate from responder selection in #95163 and member editing in #102253. Fixed membership remains intentional in this increment. The staged Desktop review timeline is unchanged.

## Validation

Earlier `779113231b8` passed **42 tests in `tests/tui_gateway/test_groups_methods.py`** after the one-line fixture correction. This is the exact-head focused receipt, not a new run of the earlier 95-check retirement selection. The historical hosted CI receipt is recorded below.

Real temporary SQLite stores and loopback HTTP cover invitation, scoped transfer, durable passive state, old-grant refusal, cross-room/target rebinding, revocation during validation, conflicting replay, rename, terminal disband and the near-limit Unicode case. No provider calls or production credentials are used.

Independent adversarial review covered the receiver, publisher, copy-retirement permissions and early Disband ordering. Confirmed blockers were fixed and rechecked; no confirmed P1/P2 remains in those reviewed source increments. Current-head CI and independently hosted process acceptance are reported separately.

<details>
<summary>Reproduce the focused checks and understand their limits</summary>

Use the repository's installed test environment and canonical runner:

```sh
scripts/run_tests.sh -j 2 \
  tests/gateway/test_hosted_room_replica_ingress.py \
  tests/gateway/test_api_server_room_replicas.py \
  tests/gateway/test_hosted_room_replicas.py \
  tests/gateway/test_api_server_room_grants.py \
  tests/tui_gateway/test_hosted_room_replication.py \
  tests/tui_gateway/test_hosted_room_replication_http.py \
  tests/tui_gateway/test_hosted_room_same_thread_recovery.py \
  tests/tui_gateway/test_hosted_room_peer_http.py \
  tests/tui_gateway/test_groups_replication_methods.py
```

On macOS, **300 focused tests across 14 files passed** at `8f61eaf2988`, including the owners above and affected existing room, service, grant and RPC tests. The real HTTP test drops a response after durable receipt, restarts the publisher, appends a follow-up and rename, and verifies exactly one copied history. Separate processes exercise the shared-target lock. These tests use no provider calls.

Migration coverage uses a stopped-worker, old-format database, then a fresh service to verify real backfill, unchanged canonical history and the original follow-up content. No production migration behavior was changed for that fixture.

This is a focused selection, not a full-suite or deployed-fleet claim. No native-client or real multi-host UAT of this new increment is claimed. CI is reported separately for the current head.

Additional bounded checks cover current profile routing and capacity recovery. At `bb02ee797d7`, all three real-HTTP publisher scenarios pass, including abrupt publisher-process death after remote persistence and before acknowledgment. This is process restart on the retained source database, not authority-host takeover.

On native Windows, 62 selected publisher, HTTP and ingress tests passed at `8f61eaf2988`, including actual `msvcrt` cross-process locking. Test dependencies were isolated from the existing runtime and removed afterward. Later profile/capacity fixes and the added hard-crash case retain their separate macOS/CI evidence; they are not relabeled as Windows passes.

The retirement implementation and early-fence composition at `d038f9ac665` passed **161 checks across eight files**, including the registered `groups.disband` path through real HTTP revocation, route removal and copy retirement. The dedicated owners are `test_hosted_room_replica_retirement.py`, `test_api_replica_retirement.py`, `test_replica_retirement_disband_rpc.py`, `test_hosted_room_disband_fence.py` and `test_groups_disband_fence.py`. Separate independent review exercised 131 retirement/copy checks and 25 lower-fence checks; these overlap and are not an aggregate.

The same production source passed a bounded check on **two separate physical hosts**: independently generated identities/secrets, real API/service workers and SQLite stores, Unicode history, registered Disband, ordinary-grant revocation, partial-copy retirement and a dropped retirement acknowledgment. The scenario completed in 10.36 seconds; complete staging and cleanup took 53.41 seconds. It used isolated processes and loopback HTTP carried through verified SSH forwarding, not a new direct-HTTPS or production-deployment UAT. No Bot/model task was invoked, and all temporary resources were removed.

Earlier head `74db007a544` corrected two Windows test assertions. Both persistence tests remain active; only POSIX mode-bit assertions are conditional. Twenty local tests pass. That change does not certify NTFS ACLs or alter runtime permission policy.

Reviewed runtime head `d91bb120884` addresses the subsequent retirement review: both UPDATE guards protect the source and destination identity, and versioned guards migrate already-initialized databases when the retirement owner initializes. Eight previously failing destination cases passed, including retirement before any page and after reclamation; legal same-identity bookkeeping remains allowed. At that head, **95 focused checks across five files passed**, with a separate overlapping 21-test independent check and no confirmed P1/P2 in this correction. These are SQLite/HTTP regressions, not a new fleet rollout, takeover test or current-head rerun.

</details>

## Readiness And Recovery Limits

**Historical CI at `779113231b8`: all 25 applicable checks passed**, verified September 7, including [Python and OS-specific checks](https://github.com/NousResearch/hermes-agent/actions/runs/34144470327), [Nix](https://github.com/NousResearch/hermes-agent/actions/runs/34144469579) and [both Docker architectures](https://github.com/NousResearch/hermes-agent/actions/runs/34144469562). Eleven checks are skipped and the separate scanner result is neutral. The authored retirement correction is composed into #104759 and #98307; their descriptions track the exact source and integration checks. The standalone source's earlier two-host receipt remains separately scoped, not a production deployment of the correction.

Copies can still have an unreceived history tail: explicit copy retirement permits safe cleanup without fabricating that tail. A permanently unreachable target or revoked cleanup permission remains an explicit unresolved condition.

Safe succession additionally requires an exclusive authority decision and complete recovery state for accepted work, Stop and revocations. History copying alone does not satisfy those later gates.

#104759 separately adds passive task/receipt/Stop records. Its continuous-history scheduling correction remains independently owned there; neither PR enables takeover.


</details>


</details>

## Historical #104759

Source: https://github.com/NousResearch/hermes-agent/pull/104759, head `db832ed1a6de377f5fc0b31ff8aa5eb0d138cc2d`.

<details>
<summary>Retained public description and evidence</summary>

## Remember Unfinished Group Chat Work

A copied conversation is not enough to recover a Group Chat. A Bot may have accepted work without replying yet, or a Stop request may still be waiting for confirmation. Losing those facts can lead to duplicate work or an unsafe retry.

This next step retains bounded, read-only task and receipt records on explicitly authorized participant gateways. A surviving gateway can report the work it last observed, separately from the conversation it copied. It cannot use those records to start tasks or take control of the group.

**This is recovery evidence, not automatic failover.** An unreplicated final update can still be lost, and an unknown outcome must not become an automatic retry.

## Scope

- Preserve initial-epoch v1 record bytes and validation while adding negotiated v2 producer scope: current gateway, authority epoch and lineage digest, bound to verified retained history. Original run receipts keep their original authority; execution generation is not an authority epoch.
- Keep producer-scoped snapshots/revisions and immutable pending deliveries. Superseded pending records retain their exact payload and prior outcome; a successor does not retag, transmit or acknowledge them with a new grant.
- Migrate legacy evidence atomically and retain malformed/orphan records as explicitly invalid local evidence. Aggregate byte/row limits charge all retained scopes and metadata; existing overage is not evicted to unblock capture.
- Validate the record actually being consumed instead of mutating every room on each initialization. Requested replica inspection can mark an invalid current record unusable without changing its bytes, clearing quarantine or weakening retirement guards. Shared capacity and the lower history auditor remain separate database-wide concerns.
- Explicit invitation opt-in, separate from history-only replication. Existing grants gain no access.
- Consistent, versioned records of task identities/phases, exact peer receipts and Stop/closing facts. Task-state changes are tracked even when the conversation has not changed.
- No prompts, result bodies, tool commands, private paths, broad credentials, OAuth data or file contents in this payload.
- The existing bounded publisher, target coordination and retry path deliver the records. Acknowledgment loss must not replace an unresolved record with a newer one.
- A busy conversation does not need to pause for its task records to be copied. Frozen snapshots wait for their matching history, then use an eligible route; blocked record delivery does not stop healthy history copying.
- Source-capture failures are retained on the affected route and remain visible after restart. Successful unchanged recapture clears a transient error only when that target already acknowledged the record; pending or refused work and sibling-route failures are not hidden, and no duplicate delivery is needed.
- The participant retains passive state and exposes a summary through `groups.replica_state`. Quarantined, retired or conflicting records must not be treated as usable recovery state.

This is API-first work. It does not add a new Desktop or messaging recovery screen, import a scheduler, resolve approvals or change the group's authority. Core records do not certify the integration build's richer approval/retry journals or reproduce a Bot's live process.

## Review Boundary

<details>
<summary>Pinned review boundary</summary>

Current source: [`db832ed1a6de377f5fc0b31ff8aa5eb0d138cc2d`](https://github.com/dokterdok/hermes-agent/commit/db832ed1a6de377f5fc0b31ff8aa5eb0d138cc2d), a normal merge of reviewed correction `7e50be215370937db765b174e8b582af49a575dd` and pinned upstream main `22488b8c62d3c92f25149053ae8df68fb0afcb35`. It retains upstream alignment `5f04a129dad08594d88d42a2581f0971d4e53aa8`, an authored merge of reviewed work-v2 `da47953427e1561478cfd73c99633a5e7ca1ddf0` with pinned upstream main `b2aa855b626ff8688eb34b95c60ee8b6a4af3679`. The [submitted main-based comparison](https://github.com/NousResearch/hermes-agent/pull/104759/files) includes inherited prerequisites and is not the work-evidence-only review scope.

The authored compatibility merge `c1e7cfb4e32ac917f159eea49c7ce035c9ca820a` preserves accepted #104601 `97d4f4dbb82efb0309a389de3b3e8348702517bf` and the earlier publisher correction. The reviewed v2 checkpoint adds producer-scoped work records and the migration, stored-binding, retry/fairness and fault-isolation corrections. The subsequent upstream alignment preserves all 93 lower Python source/test files byte-for-byte. Only `compat_manifest.json` and `COMPAT_MANIFEST.md` combine both parents; the parent checked their exact semantic union and adjacent runtime imports. Workflow changes in the alignment are inherited upstream bytes, not a CI workaround. The full source history is retained. This is passive evidence, not recovery activation.

The fixture change at `c2618f74336` creates the reviewer profile used by the invitation test in `tests/tui_gateway/test_groups_methods.py` and retains its original cherry-pick provenance. The preceding [two-file correction above that head](https://github.com/dokterdok/hermes-agent/compare/c2618f74336a2b7f2b91cffdfcb84748639a6274...68cd7a65deb2132ca38cdfabb90d368b835522d9) persists source-capture failures per route and clears recovered unchanged captures without hiding unresolved delivery. One independent review and a focused correction re-review closed its confirmed P2.

Builds on exact #104601 `97d4f4dbb82efb0309a389de3b3e8348702517bf`, including accepted lineage, quarantine-label and partial-copy reclamation corrections, plus the #99107/#100016 prerequisites. The incremental comparison below is the review scope, not all inherited changes against main.

</details>


Builds on #104601's history delivery and copy-retirement source, with its #99107/#100016 prerequisites and original contributor history preserved. Review the [reviewed work-evidence increment above that accepted source](https://github.com/dokterdok/hermes-agent/compare/97d4f4dbb82efb0309a389de3b3e8348702517bf...da47953427e1561478cfd73c99633a5e7ca1ddf0): the work-evidence implementation, versioned lineage lifecycle and reviewed corrections, not all inherited changes in the main-based diff. The source histories are retained through a normal merge, not copied into a second implementation.

The main owners are `gateway/hosted_room_work_records.py`, its small HTTP handler, and the existing publisher in `tui_gateway/hosted_room_replication.py`. No second delivery engine is added.

#97681 tracks the staged host-offline goal. mystickcal's #94048 is adjacent single-session side-effect reconciliation, not a dependency or duplicated execution engine here. wanxun123's #25575 describes broader cross-server state snapshots; this layer deliberately stays scoped to Group Chat evidence.

## Validation

**Current upstream refresh `db832ed1` passed the eight-case canonical SQLite/unit-render selection.** All four reviewed CI-correction files are byte/mode-identical to `7e50be21`; every resulting file matches one of the merge parents. Upstream independently fixed the same no-root unit fixture in `9b0d75ce`, so the one conflict was resolved by retaining the already-reviewed temporary-home test in its extracted sibling without duplicate classes. The prior head had no CI because GitHub reported a merge conflict after main advanced. This refresh is mergeable and [CI passed](https://github.com/NousResearch/hermes-agent/actions/runs/34279553484), including the required aggregate. Docker and Nix also passed on the exact head. This does not establish native or recovery-activation acceptance.

**Reviewed correction `7e50be21` has bounded independent approval.** Standalone work-record initialization now reserves the SQLite writer before schema reads, avoiding a deferred read-to-write upgrade race between publisher connections. Existing caller-owned transactions retain their savepoint boundaries. The test retains the original timing/concurrency assertions, covers actual WAL/DELETE modes without disabling the WAL safety fallback, and rejects unintended peer requests. The inherited service-unit ordering fixture now uses a distinct temporary caller home instead of probing `/root`; its test module was split mechanically to keep both files below 2,000 lines, preserving all 106 methods, decorators and assertion ASTs. No service-runtime implementation changed.

The worker's publisher/storage selections passed **153 distinct tests across 11 files**; the parent separately passed 39 database controls and the exact combined eight-case gate. The independent reviewer passed eight SQLite controls, the single unit-render regression, and five supplemental failure-atomicity scenarios (body failure, busy BEGIN, and busy DELETE commit rollback). These selections overlap and are not summed. Broader local service-control testing was not green and is not represented as full CLI/native acceptance; full isolated/hosted verification remains required. **Exact-head CI for `db832ed1` passed**, including the required aggregate, Docker and Nix.

**Previous upstream-aligned source `5f04a129` passed 323 canonical tests across 23 selected files and 15 external controls across five files**, with zero retries. The parent separately reran two settled-retry tests and four compatibility-import controls. These selections overlap and are not summed. This is a bounded merge/compatibility acceptance over the previously reviewed v2 source, not a new full-repository audit. **CI on that prior head failed.** [Run 34266507459](https://github.com/NousResearch/hermes-agent/actions/runs/34266507459) executed the Linux suite and reported two failures: the inherited service-unit ordering fixture raised PermissionError while probing `/root/.local/bin`, and the work-record publisher unavailable-target test did not reach its callback within its wait budget. The required aggregate failed; Docker and Nix passed. Both failures have corrections in the new head, as described above. This failed run remains negative evidence, distinct from the earlier zero-job startup failures below.

**Reviewed v2 checkpoint `da47953427` passed 321 tests across 22 selected files**, with zero retries, scoped Ruff and diff checks. All 11 unchanged original/delta reviewer cases passed; the parent separately reran the five delta cases and 19 quarantine/retirement controls. The final independent design review approved the exact source, reran all 11 external cases and passed three additional requested-room/type-preservation controls. These selections overlap and are not summed.

The local review process exposed migration data loss, metadata undercharging, invalid-record retries and capability-outage starvation. Their corrections were retained and verified. A remaining cross-room classification failure led to replacing the global mutation scan with record/room-owned validation—not clearing quarantine or suppressing audit errors. The final review closed that defect and confirmed the earlier fixes. Prior failed candidates and test evidence are preserved; earlier passing selections were not treated as approval.

History, retirement and work response capabilities now advertise their implemented `[1,2]` formats, outside the unchanged RoomLink/catalog shape. Legacy work remains v1; unsupported later-work routes fail visibly without downgrade or false acknowledgment. V2 observations explicitly retain unknown prior-authority work and do not establish complete recovery evidence.

**CI on the prior v2 head `da47953427` was not cleared.** Main CI runs [34250779047](https://github.com/NousResearch/hermes-agent/actions/runs/34250779047) and [34251078047](https://github.com/NousResearch/hermes-agent/actions/runs/34251078047) failed before creating any jobs; the latter reports GitHub’s generic unexpected-error annotation. No required aggregate was produced. This is not a passing test result, and no workflow/source change is justified from that annotation alone. Local review approval does not replace hosted CI. A maintainer rerun was requested for that head, without requesting a change to any deferred review timeline. Those failures remain recorded; they are not results for the newly aligned head.

No current-v2 physical-host, installed-client, commute, deployment or recovery-activation acceptance is claimed.

**Historical compatibility baseline `c1e7cfb4e` passed 483 tests across 34 selected files**, with zero retries, scoped Ruff and diff checks. One independent integration review approved the actual merge/compatibility changes and passed 47 distinct tests across eight files, including actual HTTP old-grant refusals, successor history progress, untouched historical work bytes, retirement cleanup, and passive-only invitation consumers. Those tests overlap the parent selection and are not summed. The [main CI run on that baseline](https://github.com/NousResearch/hermes-agent/actions/runs/34232227967) failed before creating any jobs, with GitHub’s “An unexpected error has occurred” annotation. Docker and Nix passed, but no required aggregate was produced, so hosted CI is not cleared. The CI workflow is byte-identical to the accepted lower head, and no workflow change is proposed from this annotation alone. The earlier failed Windows receipt remains below.

That baseline advertised history/retirement `[1,2]` and work `[1]`; it refused later work without blocking valid history or rewriting old pending records. The current v2 source above supersedes that implementation limitation, not the older evidence boundary.

**Previous correction `68cd7a65deb` passed 32 selected canonical tests across four files and both unchanged independent publisher probes**, with zero retries. The quiet unchanged-capture case failed before correction; pending-history and refused-delivery controls retained their failure state. The focused re-review independently passed those three cases and approved the correction. These selections overlap and are not summed.

**Preserved failed CI on previous head `68cd7a65deb`:** the [Windows job on that head](https://github.com/NousResearch/hermes-agent/actions/runs/34205190916) failed `test_progress_advances_while_the_orchestrator_blocks` because its desktop-updater self-test did not expose a progress-server URL within the startup budget. That test and its PowerShell script are unchanged by this publisher correction. That head’s required aggregate was red; this is not relabeled a passing publisher test or a confirmed product regression. A failed-job rerun was denied because upstream repository administration is required. The exact same self-test subsequently passed once on a separate native Windows host at `68cd7a65deb`, using Python 3.11.15, the canonical runner, one worker and zero retries (1 passed, 0 failed). It ran `-SelfTestUi -NoUi`, without an actual update or app restart. This single control narrows the diagnosis but does not clear hosted CI or establish the hosted startup failure's cause.

Earlier `c2618f74336` passed **42 tests in `tests/tui_gateway/test_groups_methods.py`** after the one-line fixture correction. That was its exact-head fixture receipt, not a new run of the earlier 170-test runtime selection.

Reviewed runtime head `2156240c0ea` passed **170 focused tests across eleven files**, with no retries. The continuous-history review finding is fixed, including alternate-route recovery and preserving history progress when work-record routes are blocked. Independent review closed all confirmed P1/P2 in that correction; its final 12-case check retains its separate, overlapping scope. The exact authored retirement correction from #104601 is included and unchanged by the fixture.

**Historical CI receipt: all 25 applicable checks passed on `c2618f74336`**, verified September 7, including [Python and OS-specific checks](https://github.com/NousResearch/hermes-agent/actions/runs/34144826392), [Nix](https://github.com/NousResearch/hermes-agent/actions/runs/34144825609) and [both Docker architectures](https://github.com/NousResearch/hermes-agent/actions/runs/34144825578). Eleven checks are skipped and the separate scanner result is neutral. The runtime fixes were composed in #98307 with a separate 89-test targeted integration receipt. A separately scoped two-host follow-up delivered queued, running and cancelled records while history kept arriving, including identical replay after a lost acknowledgment. No live gateways were updated by this source repair.

<details>
<summary>Focused checks, reproduction and limits</summary>

```sh
scripts/run_tests.sh -j 2 \
  tests/gateway/test_hosted_room_work_records.py \
  tests/gateway/test_api_server_room_work_records.py \
  tests/gateway/test_work_record_review_regressions.py \
  tests/tui_gateway/test_work_record_delivery_fairness.py \
  tests/tui_gateway/test_work_record_publisher.py \
  tests/tui_gateway/test_work_record_failure_visibility.py \
  tests/gateway/test_hosted_room_replica_ingress.py \
  tests/gateway/test_api_server_room_replicas.py \
  tests/gateway/test_api_replica_retirement.py \
  tests/tui_gateway/test_hosted_room_replication.py \
  tests/tui_gateway/test_hosted_room_replication_http.py \
  tests/tui_gateway/test_groups_methods.py
```

Real SQLite and HTTP checks cover scope/revocation races, task-only revisions, a consistent source view, private-data exclusion, bounded/unavailable records, exact receipts and Stop, dropped/partial acknowledgments, publisher restart, alternate profiles and retirement. Review reproductions are retained: a refused primary stranded valid alternates, and a damaged prefix was accepted before its audit. The fixes reuse per-route state, the existing queue and the existing replica auditor. A separate transient-outage case checks recovery when only one of two previously unavailable profiles returns.

The new regression selection adds 5/20/100-turn continuous-history cases, a real HTTP queued-to-running transition with a lost record acknowledgment, restart with an immutable anchor, empty-group ordering, and recovered/blocked alternate routes. The corrected ranking gives deliverable work a turn without relaxing history-prefix or durable refusal checks. Earlier source `116d5a72a99` passed 175 checks across ten files; that selection overlaps the 170-test receipt at `2156240c0ea` and they are not added together.

The main source suite is a focused macOS selection, not a full-repository run. A separate native **Windows run passed 70 tests across six files** on exact `6eca08e21fa`, including the new record paths, review regressions, publisher isolation and the two earlier portability fixes. It used the existing native Python and a temporary test dependency overlay, removed afterward; the user's runtime and checkout were unchanged. HTTP tests use real handlers and stores; the threaded unavailable-target case uses a contract-mocked transport. Inherited history-publisher process-crash coverage is not relabeled as a new work-record crash test. No new model or provider calls were made.

The separately scoped physical-host check on field `04979b174993` completed its scenario in 40.32 seconds, with 59.90 seconds total staging/cleanup. It used fresh identities and temporary stores in existing pinned runtime images, with loopback HTTP carried through verified SSH forwarding. This is not direct peer-HTTPS acceptance, a deployed-fleet claim, or proof of actual Bot execution or a completed takeover. Synthetic driver/receipt transitions used real store APIs; both metadata and retirement ACK loss recovered through the existing publisher.

The new busy-history follow-up ran once on `d66512bfe984`, whose relevant owners exactly match this source. Over 273 additional history appends, queued, running and cancelled metadata arrived without waiting for quiet; all target prefixes covered their recorded anchors. The deliberately lost ACK replayed the same serialized body. Scenario time was 26.77 seconds, or 44.27 seconds including staging/cleanup. It reused existing images with isolated identities/stores and verified SSH forwarding, used no model calls, and left no test containers or staging behind. These are observations from this controlled case, not a latency guarantee or authority-loss test.

</details>

## Contribution Checks

- [x] Contribution guide and existing open/merged work checked; related work is identified above.
- [x] Conventional authored commits and dependency history preserved.
- [x] Changed code/test owners remain below 2,000 physical lines; setup and limitation docs updated.
- [x] Focused canonical tests pass; no runtime dependencies, model tools or configuration defaults added.
- [x] Independent source review and narrow fix recheck completed.
- [x] Exact-head CI for `db832ed1` passed: required aggregate, Docker and Nix verified on that SHA. Prior `5f04a129` test failures, `da47953427`/c1 startup failures and `68cd7a65deb` Windows failure remain preserved above.
- [ ] Current-v2 two-physical-host acceptance remains. Historical v1 two-host and busy-history receipts above retain their exact scope.
- [x] Native-Windows record-path checks at `6eca08e21fa`, with isolated dependencies and verified cleanup; not a new Windows run of the correction.

Model-driven recovery and automatic successor execution are outside this PR; the synthetic-state tests do not claim either.

## What Remains

Producer-scoped work-record v2, byte-preserving migration, independent successor capture and negotiated delivery are now implemented and source-reviewed. They still carry explicit incompleteness: a successor observation does not certify an unseen accepted tail or recreate missing original work. #105197 must integrate the multi-scope evidence, updated shared-budget interface and reconciliation/identity/fencing requirements before enabling manual recovery. This PR does not enable that recovery action.

Safe continuation still needs a globally exclusive authority decision, an explicit accepted-tail durability policy and authorized access to already-running work. These records alone do not provide any of those. File availability remains separate from whether healthy participants can continue.


</details>
