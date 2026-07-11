/**
 * first-run-flow.ts
 *
 * Pure, electron-free orchestration for the desktop's first-run install-vs-connect
 * boot flow: the glue that sits between the electron-free first-run GATE
 * (first-run-gate.ts, the one-shot latch) and main.ts's electron-coupled boot
 * path. The gate is only a latch; these helpers encode the actual PRODUCTION
 * decisions main.ts makes around it — when a bootstrap-needed primary boot parks,
 * how an abort is tagged and recognized, and what the 'install' choice persists
 * and rewrites. Extracting them here (no `import 'electron'`) makes that logic
 * behaviorally testable with `node --test` — same pattern as connection-config.ts
 * and first-run-gate.ts — instead of only pinnable via source-assertion regex.
 *
 * main.ts wires these three call sites (dependency-injected so this stays
 * electron-free and each collaborator — gate, fs-backed config, marker writer —
 * is a real production function at the call site, a fake in tests):
 *   - ensureRuntime()'s 'bootstrap-needed' branch calls guardFirstRunBootstrap()
 *     BEFORE handOffWindowsBootstrapRecovery(): a fresh machine (any OS) must pick
 *     install-vs-connect first; the recovery handoff only heals an already-chosen
 *     interrupted install, so a user picking "connect to a server" must never
 *     trigger it. Only the primary boot enables the gate; pooled/secondary boots
 *     bootstrap directly (they render no overlay).
 *   - startHermes()'s .catch() uses isFirstRunAborted() to recognize the
 *     non-latching abort and re-run itself against the freshly-saved remote.
 *   - the hermes:first-run:choose IPC handler is applyFirstRunChoice().
 *
 * The abort is deliberately NON-latching: on 'abort' guardFirstRunBootstrap()
 * throws a firstRunAborted-tagged Error rather than failing boot. main.ts's
 * startHermes catch nulls its connection cache and re-runs; because the remote
 * config was already written to disk by connection-config:apply before the
 * gate was aborted, the re-run resolves the remote backend and never re-reaches
 * bootstrap-needed — bounded to one effective retry. If the applied config is
 * NOT remote, the re-run legitimately re-parks the gate (the overlay
 * re-appears): correct, the user must still choose.
 */

// The abort error guardFirstRunBootstrap throws and isFirstRunAborted matches.
// The `firstRunAborted` tag is the whole contract: it tells startHermes's catch
// "this is the go-remote handoff, re-run me" instead of "boot failed, latch it".
export type FirstRunAbortedError = Error & { firstRunAborted?: boolean }

// A boolean, or a lazy predicate evaluated only when the gate is enabled. main.ts
// passes the predicate form (firstRunInstallChosen, an uncached file read) so it
// runs at most once per gated primary boot and never on a pooled boot — preserving
// the `enabled && !installChosen()` short-circuit exactly.
type InstallChosen = boolean | (() => boolean)

interface FirstRunGateLike {
  waitForDecision: () => Promise<'install' | 'abort'>
  chooseInstall: () => void
  state: () => { required: boolean }
}

/**
 * Park a bootstrap-needed PRIMARY boot on the first-run gate until the user
 * decides, then either return (proceed with the local install) or throw the
 * firstRunAborted error (go remote). No-op — returns immediately — when the gate
 * is disabled (pooled/secondary boot) or the user already chose install in a
 * prior run (persisted marker), so an interrupted install / unattended relaunch
 * resumes bootstrap directly. This is exactly what ensureRuntime's
 * 'bootstrap-needed' branch runs, BEFORE the Windows recovery handoff.
 *
 * @param deps.gate the shared first-run gate (createFirstRunGate).
 * @param deps.enabled true only for the primary boot (ensureRuntime's option).
 * @param deps.installChosen boolean, or a predicate evaluated only when enabled.
 */
async function guardFirstRunBootstrap({
  gate,
  enabled,
  installChosen
}: {
  gate: FirstRunGateLike
  enabled: boolean
  installChosen: InstallChosen
}): Promise<void> {
  if (!enabled) {
    return
  }

  // Evaluate installChosen lazily so the file read only happens on a gated boot.
  if (typeof installChosen === 'function' ? installChosen() : installChosen) {
    return
  }

  const decision = await gate.waitForDecision()

  if (decision === 'abort') {
    // Non-latching: signal startHermes to re-dial the freshly-saved remote
    // instead of failing boot. The tag is what its catch keys off of.
    const abortedError: FirstRunAbortedError = new Error(
      'First-run bootstrap aborted: connecting to a remote Hermes backend instead.'
    )

    abortedError.firstRunAborted = true
    throw abortedError
  }
}

/**
 * The predicate startHermes's .catch() uses to recognize a first-run abort (vs a
 * genuine boot failure, which takes the latching path). Kept here next to the
 * thrower so the tag they agree on lives in one place.
 */
function isFirstRunAborted(error: unknown): boolean {
  return Boolean((error as FirstRunAbortedError | null | undefined)?.firstRunAborted)
}

/**
 * The full body of the hermes:first-run:choose IPC handler. On 'install':
 *   1. Persist the install marker so an interrupted install / unattended
 *      relaunch resumes bootstrap directly instead of re-asking (see
 *      readFirstRunInstallChoice / guardFirstRunBootstrap's short-circuit).
 *   2. If the saved connection config carries a remote-like mode — the remote
 *      form may have persisted mode:'remote' via an OAuth sign-in that saves
 *      before the user clicks Connect, while they were still deciding — rewrite
 *      it back to 'local' so the next launch spawns the fresh local install
 *      instead of dialing a half-configured remote. Preserve the remote
 *      block/profiles so a saved URL/token can be reused later from Settings.
 *   3. Release the gate (chooseInstall) so the parked boot proceeds.
 * Non-'install' choices are ignored. Always returns the gate's post-choice state.
 *
 * Dependency-injected so this stays electron-free: main.ts passes the real
 * fs-backed readDesktopConnectionConfig / writeDesktopConnectionConfig /
 * writeFirstRunInstallChoice and connection-config.ts's modeIsRemoteLike.
 */
function applyFirstRunChoice({
  choice,
  gate,
  readConnectionConfig,
  writeConnectionConfig,
  writeInstallChoice,
  modeIsRemoteLike
}: {
  choice: unknown
  gate: FirstRunGateLike
  readConnectionConfig: () => any
  writeConnectionConfig: (config: any) => void
  writeInstallChoice: () => void
  modeIsRemoteLike: (mode: unknown) => boolean
}): { required: boolean } {
  if (choice === 'install') {
    writeInstallChoice()

    const config = readConnectionConfig()

    if (modeIsRemoteLike(config.mode)) {
      writeConnectionConfig({ ...config, mode: 'local' })
    }

    gate.chooseInstall()
  }

  return gate.state()
}

/**
 * Tolerant parse of the first-run marker (first-run.json): true only when it
 * records {choice:'install'}. Missing or malformed → false (fall through to
 * asking). `source` is either the raw file contents or a reader that returns
 * them (main.ts injects a fs.readFileSync closure); a throwing reader is treated
 * as "no recorded choice".
 */
function readFirstRunInstallChoice(source: string | (() => string)): boolean {
  try {
    const raw = typeof source === 'function' ? source() : source
    const parsed = JSON.parse(raw)

    return Boolean(parsed && typeof parsed === 'object' && parsed.choice === 'install')
  } catch {
    return false
  }
}

export { applyFirstRunChoice, guardFirstRunBootstrap, isFirstRunAborted, readFirstRunInstallChoice }
