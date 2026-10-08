# Independent AR: combined upstream scope17 branch

Outcome: **NO FINDINGS**. Independent read-only reviewer: `/root/sdk_upstream_stack_ar`.

Reviewed HEAD `076c4c904650f829a98bd04f83e8737495013435` against upstream base `8a33891bdd58c3e0795ebfb277bcdde1003e91ea`. The worktree was clean; `git diff --check` passed; nine paths changed; `agent/auxiliary_client.py` remained at the base's 8,400-line cap.

The reviewer traced combined timeout/cancel classification, uncached Codex client and worker-owned finalization, retry client reacquisition, narrowed recovery kwargs, fallback task budget and forced stream, Nous wire-mode refresh, router timeout-shim fallback, and the pinned credential-pool guard. The corrected attempt, recovery, and routing slice reviews each reached literal NO FINDINGS. No source, test, provider, or network changes were made by the integration reviewer.

The existing HTTP auth end-to-end test's HomeIOGuard failure occurs on both the parent and patched code before the changed paths. It remains a validation limit, not evidence of a branch regression. This review does not establish upstream merge, installed runtime participation, a real provider response, or Gordon activation.
