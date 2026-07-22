# Tasks

- [x] Explore scheduler, job claims, execution ledger, provider fire path, and existing E2E tests.
- [x] Specify ownership, deadline, and reaping invariants.

- [x] Keep child ownership through termination and confirmed reaping.
- [x] Bound result polling by the remaining wall deadline.
- [x] Make external-fire claims durable per execution and protect live owners.
- [x] Add regression coverage for live ownership and process descendants/non-cooperative workers.
- [x] Complete Linux cgroup-v2 containment capability discovery: prove delegated child removal and writable cgroup-wide termination before reporting hard containment available.
- [x] Align `contained` status and user-facing cron documentation with the proven capability boundary; until the preceding proof exists, do not describe `contained` as the complete hard-containment contract.
- [x] Remove process-global subreaper, PDEATHSIG, `/proc` descendant harvesting, and destructive PID cleanup.
- [x] Add capability/fallback semantics and detached-descendant/unrelated-process regressions.
- [x] Complete atomic create→park→assign+verify→release, opaque-handle revalidation, and cleanup-failure retention semantics in the production supervisor path.
- [x] Add a capability-gated real Linux cgroup-v2 supervisor integration regression through `_run_isolated_cron_job`, covering parent registration, parked-child assignment and membership/identity/parent verification, release only after verification, owned-boundary teardown, immediate leader exit with `start_new_session=True`, unrelated-process preservation, and explicit unavailable/unsupported status when discovery fails before allocation.
- [x] Add injected production-path regressions proving assignment/identity-parent validation, allocation cleanup, and cgroup termination failures retain ownership as `cleanup_failed` and never release a parked workload after failed verification.
