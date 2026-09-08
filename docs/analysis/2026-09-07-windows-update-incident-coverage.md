# Windows updater incident coverage, 2026-09-07

This note compares the Windows recovery on DESKTOP-O91444G with updater PR
[#104687](https://github.com/NousResearch/hermes-agent/pull/104687), reviewed at
`1868d36422a36491174b225c5af7c19466a5c8ba`. It records what the incident proves,
which mechanisms already cover it, and what remains unverified.

## Observed recovery

The managed installation started at `91c2fea09141e6f8a6fcff036db3491f02518236`.
The recovery installed fork commit `d45744d75f76b685382df04e66707946d1db6e0d`
through the real CLI update command. It did not install or exercise this PR's
Desktop update action.

The operator stopped the Desktop and its local backend, messaging gateway,
operator serve process, and usage-tracker service host and children. Remaining
native-module holders included MCP bridge processes owned by two Codex parent
processes. The Codex parents were left running.

A helper running outside the managed venv held the installed version's shared
update marker and the target fork's bridge-quiesce lease together. Two process
snapshots ten seconds apart found no managed venv/runtime holders. The helper
then launched the real updater as its child; both update and Desktop build
commands exited zero. The resulting checkout was clean at the target commit,
and its CLI receipt recorded a successful update. Desktop, gateway, operator
serve and usage service were observed running after restoration.

This is evidence for the recovery, not an end-to-end success receipt for PR
#104687. The receipt separately reported its backup step as disabled or failed;
no new backup success was established.

## Coverage at the reviewed PR head

| Requirement | Existing implementation | Limit of the evidence |
| --- | --- | --- |
| Claim before draining holders | `applyWindowsUpdate` in `apps/desktop/electron/windows-update-apply.ts` acquires the update marker before preflight. `update-preflight.ts` checks the original claim through scanning and termination, and before authorizing mutation. | Source trace; no new Desktop cutover run. |
| Stop existing MCP bridges without killing their host applications | `agent/transports/hermes_tools_mcp_server.py` polls the shared marker every 0.5 seconds and exits the bridge. `hermes_mcp_update_gate.py` accepts the observed exact `-P -m` invocation. | Covers shared-marker readers; see the version transition below. |
| Keep respawned bridges from loading native modules | `agent/__init__.py` checks the marker before `jiter_preload`. | Applies to the exact MCP module launch, not arbitrary Python processes. |
| Attribute the actual native-module holder | `install-lock-probe.ts` and `windows-restart-manager.ts` attribute locks to the target venv. A shared-runtime Python process can supply the venv `.pyd` it maps as its resource proof. | Excluding shared-runtime DLLs from the mutation set does not exclude processes that map target-venv modules. |
| Preserve exact termination checks | `windows-process-terminate-scripts.ts` rechecks process generation and requires a mutation-resource proof for shared-runtime images. The force-release loop uses one absolute budget. | A serial loop alone does not prove exhaustion. Count only holders remaining after tracked-tree stop, gateway stop, and cooperative MCP exit. |
| Carry ownership through detached handoff | `windows.ps1` adopts the Desktop's exact marker body and acknowledges with a nonce and its process generation. The Desktop verifies that acknowledgement; the CLI child can adopt its updater ancestor's claim through `hermes_cli/update_lock.py`. | The recovery helper was also its CLI child's ancestor. Its explicit handoff PID was not evidence of broken ancestry adoption. |
| Restore plugin services stopped by the updater | `desktop-plugin-host-restore.ts` persists stopped supervisors and compensates if recording fails. `main.ts` replays the ledger on startup, update completion and abort. | The manual recovery stopped the usage host outside this ledger, so its manual restart does not demonstrate a missing Desktop restoration path. |

The reviewed Python gate already includes the POSIX `OverflowError` handling
and Windows DWORD guards in its PID helpers. Those fixes do not need to be
ported again from the recovery branch.

## Version transition and host-specific behavior

The recovery crossed from shared-marker readers at `91c2fea091` to the fork's
`.hermes-venv-quiesce` readers. Holding both artifacts kept old running readers
and newly launched readers paused across source replacement. The upstream PR
instead uses the shared `.hermes-update-in-progress` marker consistently.
Copying the fork's separate lease protocol into the PR is not justified by this
recovery. A mixed Desktop/source deployment or a transition to a lease-only
fork needs its own compatibility assessment; no such cutover is proved here.

The actual `Hermes_Gateway_Watchdog` and `Hermes_Serve_Watchdog` task actions
point to host-local VBS scripts. Inspection of those exact scripts found that
both check the shared update marker, the lease and their CAS artifacts, then
exit before starting managed Python while a gate exists or cannot be read.
Their absence from the repository is not evidence that this host's watchdogs
ignore the marker. These custom scripts are not a guarantee about every
installation's launchers.

The primary `Hermes_Gateway` task rejected disabling with OS Access Denied. It
had no repeating trigger and remained enabled throughout the successful
recovery. Task permission denial does not establish that terminating a holder
would also require elevation.

The recovery's manual stops and task changes were operational steps. They do
not establish a need for a new CLI drain mode or for bypassing exact process
identity/resource checks. CLI refusal to update while unknown holders remain
is not itself a demonstrated defect.

## Follow-up and verification boundary

One concrete UI defect was found at the reviewed head: the access-denied
holder message tells the user to choose **Force update (Administrator)**,
but this branch has no such control. The focused correction is to direct the
user to close the listed processes, using administrator permission if needed,
and retry. The refusal must continue to prevent mutation while holders remain.

An adjacent observation is the Desktop gateway-stop wrapper's 20-second
timeout versus a CLI drain that may take longer. The recovery had no active
agents and completed its stop cleanly, so this was not established as its cause.

No normal Desktop update run was performed during this follow-up. The draft's
two isolated end-to-end runs and planned MCP-gate PR separation remain open.
The running managed installation was left unchanged during the branch review.
