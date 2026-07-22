# Cron process-isolation repair specification

## ADDED Requirements

### Requirement: authoritative Linux containment
LLM-backed cron workload MUST be assigned to a dedicated, verifiably owned Linux cgroup-v2 boundary before agent/tool execution starts when hard detached-session containment is claimed. A separately implemented systemd transient unit/scope with control-group termination MAY satisfy the same contract only after it proves equivalent ownership and termination semantics. Final termination MUST target that owned boundary, not a `/proc` descendant list. If capability or ownership verification is unavailable, fail closed for hard containment and expose process-group cleanup only as best effort. macOS and Windows MUST NOT claim equivalent detached-session containment.

Capability discovery MUST prove the delegated parent/control group, empty dedicated boundary, stable owned identity and parent, PID membership readback, and cgroup-targeted termination primitive before reporting containment as available. A permission, delegation, systemd, or kill-capability failure discovered before boundary allocation MUST report Linux containment as unavailable and must not represent any later best-effort process-group cleanup as equivalent containment. An identity, assignment, membership, or parent-validation failure after the child is parked MUST prevent release to agent/tool execution and tear down only the still-owned boundary.

The boundary lifecycle MUST create an empty boundary, park the child, assign its PID, verify actual membership and owned identity/parent, and release the child only after all checks pass. The scheduler MUST retain the opaque boundary handle and ownership registry until the boundary is empty and the child is reaped; a failed revalidation or teardown MUST be reported as cleanup failure and MUST NOT silently release ownership.

#### Scenario: detached descendant remains after leader exit
- GIVEN a Linux hard-containment boundary was created and verified before workload start
- WHEN the leader exits immediately after starting a `start_new_session=True` descendant
- THEN teardown kills the owned cgroup/unit and the detached descendant cannot survive

#### Scenario: unrelated process is outside the boundary
- GIVEN an unrelated process is running before the cron boundary is created
- WHEN cron teardown runs
- THEN the unrelated process is not signalled or killed

#### Scenario: capability is unavailable
- GIVEN cgroup delegation or ownership verification cannot be established
- WHEN an isolated agent cron run is requested
- THEN the scheduler reports hard containment unavailable and does not claim equivalent detached-session cleanup

#### Scenario: assignment or ownership verification fails
- GIVEN boundary allocation succeeds but assignment, membership verification, identity revalidation, or parent validation fails
- WHEN the child is still parked
- THEN the child is never released to agent/tool execution
- AND the owned boundary is torn down or retained as an explicit cleanup failure

#### Scenario: teardown cannot prove emptiness
- GIVEN the owned boundary kill operation returns but membership or child reaping cannot be confirmed
- WHEN teardown completes its bounded wait
- THEN ownership remains registered and the run reports cleanup failure

#### Scenario: unsupported platform status
- GIVEN the scheduler runs on macOS or Windows
- WHEN hard-containment capability is queried
- THEN it reports unsupported and never claims detached-session containment

### Requirement: retain child ownership through reaping
The parent supervisor MUST keep a running child registered until it has force-signalled the child process group as needed and `join` confirms the child is reaped.

#### Scenario: shutdown races child cleanup
- GIVEN a child is live or has exited while descendants remain
- WHEN shutdown or supervisor cleanup runs
- THEN the registry still exposes the child until cleanup and reaping finish
- AND descendants in the child process group are terminated where the OS permits

### Requirement: bound result polling
The supervisor MUST bound each result-pipe poll by the remaining wall-clock deadline.

#### Scenario: wall deadline expires during a poll interval
- GIVEN a finite wall limit remains less than the normal poll interval
- WHEN no result is available
- THEN the supervisor wakes by the deadline, terminates and reaps the child, and returns a timeout failure

### Requirement: durable external-fire ownership
An external fire claim MUST record its execution ID. An expired claim MUST NOT be replaced while its durable execution owner is provably live; liveness uncertainty MUST fail closed.

#### Scenario: advisory claim expires while execution is live
- GIVEN a fire claim timestamp is older than its TTL
- AND the claim references a claimed or running execution whose PID/start owner is live
- WHEN another fire arrives
- THEN the new fire loses the claim and no concurrent child is started

#### Scenario: claim owner is dead
- GIVEN the claim references an execution whose owner is no longer live
- WHEN another fire arrives
- THEN the stale claim may be replaced and the new execution ID becomes durable ownership
