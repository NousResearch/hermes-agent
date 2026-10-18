# Hermes Phase 1 — Controlled Execution Supervisor PoC

## Identity and scope

- Branch: `codex/mission-ownership-phase1`.
- Base candidate before this experiment: `e24614fb035c57958d8e859e98ccdf4d8132af54`.
- PoC implementation and test commit: `d6dc199f2b4f32c3bf7e1a1aab694302bd83cef8` (`Add isolated mission execution supervisor proof of concept`).
- The final report evidence is committed in a documentation-only follow-up on the same branch; the implementation revision above is the tested PoC.
- This is a disposable Linux experiment, not Hermes runtime integration. No live Hermes installation, profile, gateway, board, production credential, or canonical X checkout was changed.
- Runtime used: host systemd user manager (systemd 261, cgroup v2), Bubblewrap, isolated Python virtual environment, disposable profile/workspace/SQLite board below `/home/ftr3d/.hermes-phase1-isolated-20261018`.

## Working demonstration

The PoC starts an actual Python worker under a transient `systemd-run --user --scope` scope with `KillMode=control-group`, then runs it through Bubblewrap. The worker starts a child with `setsid`, records its PID/session, and exits. The supervisor observes the child in the scope's `cgroup.procs` and `/proc/<pid>/cgroup`, while the parent has exited. It refuses handoff while `cgroup.events` reports `populated 1`. After the child finishes, the supervisor observes an empty scope and atomically removes its durable execution lease, records quiescence, and hands ownership to the next generation in the same SQLite transaction.

Observed demonstration:

- Detached child PID `115355`, session ID `115355`, remained in `/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-mission-poc-a0e7dcfbccc1.scope` after parent PID `115352` exited.
- Handoff while populated returned `transferred: false`; reason identified a non-quiescent mission and active `systemd-scope` lease.
- Once empty, handoff returned `transferred: true`, generation `2`, `populated: false`.
- The experiment also started two different missions simultaneously in separate scopes.

## Process and filesystem boundary

The worker has a fresh Bubblewrap root. `/usr` and the minimum needed `/etc` files are read-only; the disposable approved input and protected mission sample are mounted read-only; only `/mission/output` is writable. `/home`, `/root`, `/run`, `/sys`, and `/proc` are unavailable to the worker, and the network namespace is unshared. The supervisor's systemd manager access is outside the worker namespace and limited to its scope lifecycle calls. The worker environment is cleared; no production credentials or remote mutating API credentials are supplied.

The worker successfully read the approved input and wrote the authorized output. Attempts to write the input, protected mission file, an unapproved output path, a host absolute path, or cgroup controls were denied. The protected mission file remained byte-identical. This is an observed boundary for the tested Bubblewrap invocation and host, not a general sandbox certification.

The test harness refuses a database outside its declared isolation root and refuses to initialize a worker root in the canonical X path. Those refusal paths ran as subprocess tests. All generated board/workspace/profile/temp/XDG paths were under the disposable root. Live-path refusal tests did not open or modify the live target.

## Restart, cancellation, failure, and race evidence

- A new supervisor process recovered a durable active operation after the launcher's worker parent exited. It observed the detached child still populated and refused handoff.
- Cancellation issued `SIGKILL` through the scope, waited for cgroup emptiness, and only then allowed handoff. The recorded child PID disappeared and the protected file hash remained unchanged. A stale old owner was rejected after generation transfer.
- A simulated abrupt parent exit with a surviving detached child was tested; recovery and cancellation occurred from a separate supervisor process. This does not prove recovery from every possible crash point.
- Deterministic barriers tested handoff during `starting` and `cancelling`; both failed closed. Two concurrent handoff attempts produced exactly one successful ownership transfer. Two separate missions launched concurrently in distinct scopes. Cancellation/handoff and termination/handoff races were exercised.
- An unresolved `starting` row deliberately remains fail-closed. If the supervisor dies before it can persist/query a scope identity, this PoC cannot safely infer that no launch is still in flight. It requires operator reconciliation; it does not reclaim by age.
- Scope termination and `cgroup.events populated=0` are the quiescence evidence. A stuck kernel task or failed systemd manager call times out and retains the lease. This experiment did not force a D-state worker or user-manager outage.

## Escape probes and exact limits

Inside the worker, attempts to start a systemd scope or timer failed because the manager socket and runtime environment were absent. `systemctl --user start` printed that it was running in a chroot and ignored the start request; notably, it returned exit code 0. Host manager queries confirmed the requested service/scope/timer units were `not-found`. The Docker CLI was present but the Docker socket was not mounted and no daemon connection succeeded. `at` and `crontab` were absent on this host, so those schedulers were not actually invoked. These probes establish only the tested worker image's inability to reach those interfaces here; they do not prove those daemons are universally unavailable on Omarchy.

Required Hermes-side restrictions for this boundary:

- Do not let a worker call unrestricted `systemctl`, `systemd-run`, Docker/Podman, `at`, cron, `nohup`, terminal backgrounding, or other host service/scheduler interfaces. A success-shaped no-op such as this `systemctl` chroot response must not be treated as accepted work.
- Do not pass manager sockets, container sockets, host `/run`, writable host bind mounts, host process/cgroup controls, or network access into the worker.
- Local child processes created inside the scope are accounted for, including `setsid` children. Processes started through an accessible host manager, containers, another user's daemon, or a remote API would be outside that scope. Remote side effects cannot be rolled back or quiesced by local cgroups.
- A user process with another accessible execution API or credentials could still create side effects beyond this local boundary. This design therefore requires removing those capabilities from substantial workers or adapting them to report through the supervisor.

## Runtime and maintenance cost

Four measured scope-plus-Bubblewrap launch samples were `179.19`, `179.20`, `184.48`, and `186.43` ms; median `184.48` ms. This includes scope setup, Bubblewrap startup, worker readiness, and polling at 20 ms intervals. For the sample Python worker, cgroup `memory.current` was about 6.4 MB and `memory.peak` about 13.0 MB. These are one-host PoC measurements, not application-wide benchmarks. Expected cost is one transient scope and one Bubblewrap process per substantial local operation, plus a small persistent lease/operation record. Existing agents would see roughly 0.18 seconds of added launch time in this minimal path; tool/runtime startup may dominate for longer tasks.

Maintenance is moderate: a small supervisor adapter must keep scope creation, startup uncertainty, cancellation, cgroup inspection, and durable lease transitions correct across systemd/Bubblewrap changes. The PoC is not production hardened: it lacks bounded resource quotas, a mature restart reconciliation protocol for every crash window, log/artifact streaming, and policy for non-local effects.

## Hermes integration delta (not implemented)

The smallest integration is a single controlled-execution adapter at the existing shared mutating dispatcher boundary. It would acquire a durable operation lease before launch, launch the worker in the scope/sandbox, retain the lease for the whole scope lifetime, and release/transfer only after systemd confirms the scope empty and a result record is durably stored. The current application-level mission owner remains the admission and handoff authority.

Fred's genuinely small reversible actions can remain on current tools under their existing authority. Substantial filesystem/code mutation should go through this adapter. Terminal/code execution, inline tools, registry tools, delegated calls, nested/proxied tools, plugins, browser/connector tools, and background operations need either a proven adapter into the same scope or a policy restriction. Remote API writes cannot be made quiescent by this local supervisor and must remain prohibited for substantial execution unless separately controlled. Full integration would need an inventory of actual Hermes dispatch paths and could require executor changes; this PoC has not established that every current mutator can be intercepted centrally.

Accepted artifacts can later be validated by a trusted checker against the mission contract and recorded hashes, then presented for explicit integration into the owning repository (for example, reviewable patch application). The worker should not receive write access to the canonical repository. That validation/integration workflow is not built here.

## Completion-test correction

The four clean-checkout failures were fixture errors: each passed an already committed, unchanged test file to the existing Python-code validator, whose unchanged-from-HEAD check correctly rejects it. I corrected those tests to construct a temporary Git repository with a baseline commit and a genuinely modified Python artifact. The stale-hash case mutates that artifact after receipt creation. A new test explicitly asserts that an unchanged tracked artifact is rejected. The validator and acceptance conditions were not changed. These tests cover code-change missions only; documentation, verification-only, and unchanged-artifact outcomes remain unsupported by the current narrow Phase 1 validator and require an explicit future acceptance policy.

## Test results and coverage

Commands executed with the project's canonical runner and isolated virtualenv:

```text
HERMES_PYTHON=/home/ftr3d/.hermes-phase1-isolated-20261018/venv/bin/python \
  scripts/run_tests.sh tests/scripts/test_mission_supervisor_poc.py -q -m integration
4 passed, 0 failed (7.1s; runner 7.2s)

HERMES_PYTHON=/home/ftr3d/.hermes-phase1-isolated-20261018/venv/bin/python \
  scripts/run_tests.sh mission_ownership_tests/test_mission_ownership.py -q
31 passed, 0 failed (9.5s; runner 9.6s)

ruff check scripts/mission_supervisor_poc.py tests/scripts/test_mission_supervisor_poc.py \
  mission_ownership_tests/test_mission_ownership.py
All checks passed.

git diff --check
passed
```

The host integration tests cover actual systemd scope creation, Bubblewrap worker execution, detached `setsid` child tracking, handoff refusal and successful quiescent transfer, write denials, manager/container/scheduler escape probes, process crash/restart/cancel recovery, stale-owner refusal, startup/cancel/parallel transfer races, concurrent independent missions, and isolation refusal. The completion tests run the existing trusted validator against real temporary Git state.

Not tested: Telegram polling, interactive CLI, desktop UI, dashboard/web routes, real model/tool provider calls, every native Hermes tool, specialist runtime, plugin-created descendants, remote APIs, container/scheduler daemon escape on a host where those sockets/services are exposed, systemd manager outage, power loss, forced scope-creation timeout, kernel D-state cleanup, resource exhaustion, or live profile coexistence. This is a host integration PoC, not Hermes end-to-end validation.

## Recommendation

**Feasible minimal supervisor, with further engineering before staging.** The measured local boundary correctly accounts for the tested detached child, denies tested writes outside an explicit output directory, and fails closed during observed launch/cancel/handoff races. It is proportionate as a deterministic local execution adapter if Hermes can route all substantial local mutators through it and remove manager, container, scheduler, network, and remote-write bypasses. The PoC is not ready to integrate or deploy: first prove a central dispatcher interception point, handle all launch/restart failure windows, define resource limits and artifact handoff, and inventory/restrict every mutating native tool. No live integration or deployment was performed.
