import path from 'node:path'

import { buildDesktopBackendEnv } from './backend-env'
import { canImportHermesCli, shouldTrustHermesOverride, verifyHermesCli } from './backend-probes'
import { describeBootstrapFailure, missingInstallPartMessage } from './bootstrap-failure-copy'
import { runBootstrap } from './bootstrap-runner'
import { getVenvSitePackagesEntries } from './windows-hermes-path'

export interface DesktopLocalRuntimeState {
  bootstrapFailure: Error | null
  bootstrapAbortController: AbortController | null
  bootstrapRepairRequested: boolean
  bootstrapRepairAttempt: number
}

export interface DesktopLocalRuntimeDeps {
  hermesHome: string
  activeRoot: string
  venvRoot: string
  sourceRepoRoot: string
  installStamp: any
  isWindows: boolean
  isPackaged: boolean
  isWsl: boolean
  fileExists: (filePath: string) => boolean
  findPythonForRoot: (root: string) => Promise<string | null>
  venvRootForPython: (python: string, root: string) => string | null
  getVenvPython: (venvRoot: string) => string
  findSystemPython: () => Promise<string | null>
  isHermesSourceRoot: (root: string) => boolean
  activeRuntimeState: () => Promise<any>
  findOnPath: (command: string) => string | null
  isWindowsBinaryPathInWsl: (command: string, options: { isWsl: boolean }) => boolean
  looksLikeDesktopAppBinary: (command: string) => boolean
  unwrapWindowsVenvHermesCommand: (command: string, backendArgs: string[]) => Promise<any>
  isCommandScript: (command: string) => boolean
  rememberLog: (message: string) => void
  localBackendLifecycle: any
  firstRunBoot: any
  handOffWindowsBootstrapRecovery: (reason: string) => Promise<boolean>
  writeBootstrapMarker: (payload: any) => void
  resolveGitBinary: () => string
  findGitBash: () => string | null
  state: DesktopLocalRuntimeState
}

export function createDesktopLocalRuntime(deps: DesktopLocalRuntimeDeps) {
  const HERMES_HOME = deps.hermesHome
  const ACTIVE_HERMES_ROOT = deps.activeRoot
  const VENV_ROOT = deps.venvRoot
  const SOURCE_REPO_ROOT = deps.sourceRepoRoot
  const INSTALL_STAMP = deps.installStamp
  const IS_WINDOWS = deps.isWindows
  const IS_PACKAGED = deps.isPackaged
  const IS_WSL = deps.isWsl

  const {
    fileExists,
    findPythonForRoot,
    venvRootForPython,
    getVenvPython,
    findSystemPython,
    isHermesSourceRoot,
    activeRuntimeState,
    findOnPath,
    isWindowsBinaryPathInWsl,
    looksLikeDesktopAppBinary,
    unwrapWindowsVenvHermesCommand,
    isCommandScript,
    rememberLog,
    localBackendLifecycle,
    firstRunBoot,
    handOffWindowsBootstrapRecovery,
    writeBootstrapMarker,
    resolveGitBinary,
    findGitBash,
    state
  } = deps

  async function createPythonBackend(root, label, backendArgs, options: any = {}) {
    const python = await findPythonForRoot(root)

    if (!python) {
      return null
    }

    // The venv whose interpreter we selected is the venv whose site-packages
    // belong on PYTHONPATH — findPythonForRoot may have picked `.venv` over
    // `venv`, and mixing the two crashes the backend on its first native
    // import (see venvRootForPython). Fall back to root/venv only for a
    // system python, where the historical layout is the best guess.
    const venvRoot = venvRootForPython(python, root) ?? path.join(root, 'venv')
    const venvPython = getVenvPython(venvRoot)
    const command = IS_WINDOWS && fileExists(venvPython) ? venvPython : python

    return {
      kind: 'python',
      label,
      command,
      args: ['-m', 'hermes_cli.main', ...backendArgs],
      env: buildDesktopBackendEnv({
        hermesHome: HERMES_HOME,
        pythonPathEntries: [root, ...getVenvSitePackagesEntries(venvRoot)],
        venvRoot
      }),
      root,
      bootstrap: Boolean(options.bootstrap),
      shell: false
    }
  }

  // createActiveBackend — build a backend pointing at ACTIVE_HERMES_ROOT, the
  // canonical install location shared with the CLI installer. The venv at
  // VENV_ROOT may not exist yet on first run; bootstrap=true tells
  // ensureRuntime() to create / refresh it before launch.
  async function createActiveBackend(backendArgs) {
    const venvPython = getVenvPython(VENV_ROOT)
    const command = fileExists(venvPython) ? venvPython : await findSystemPython()

    return {
      kind: 'python',
      label: `Hermes at ${ACTIVE_HERMES_ROOT}`,
      command,
      args: ['-m', 'hermes_cli.main', ...backendArgs],
      env: buildDesktopBackendEnv({
        hermesHome: HERMES_HOME,
        pythonPathEntries: [ACTIVE_HERMES_ROOT, ...getVenvSitePackagesEntries(VENV_ROOT)],
        venvRoot: VENV_ROOT
      }),
      root: ACTIVE_HERMES_ROOT,
      bootstrap: true,
      shell: false
    }
  }

  async function resolveHermesBackend(backendArgs) {
    // 1. Explicit override -- HERMES_DESKTOP_HERMES_ROOT points at a developer
    //    checkout. Honour it as-is (no bootstrap; the user is driving).
    const overrideRoot = process.env.HERMES_DESKTOP_HERMES_ROOT && path.resolve(process.env.HERMES_DESKTOP_HERMES_ROOT)

    if (overrideRoot && isHermesSourceRoot(overrideRoot)) {
      const backend = await createPythonBackend(overrideRoot, `Hermes source at ${overrideRoot}`, backendArgs)

      if (backend) {
        return backend
      }
    }

    // 2. Development source -- when running `npm run dev` from a checkout, the
    //    cloned repo at SOURCE_REPO_ROOT takes precedence over ACTIVE and any
    //    installed `hermes` on PATH so local Python edits are actually exercised.
    //    (In dev with no checkout, SOURCE_REPO_ROOT won't pass isHermesSourceRoot.)
    if (!IS_PACKAGED && isHermesSourceRoot(SOURCE_REPO_ROOT)) {
      const backend = await createPythonBackend(SOURCE_REPO_ROOT, `Hermes source at ${SOURCE_REPO_ROOT}`, backendArgs)

      if (backend) {
        return backend
      }
    }

    // 3. ACTIVE_HERMES_ROOT — the canonical install at
    //    %LOCALAPPDATA%\\hermes\\hermes-agent (Windows) or ~/.hermes/hermes-agent.
    //    A valid bootstrap marker proves Desktop finished the first-run install
    //    flow, but marker provenance is NOT the same thing as runtime usability:
    //    the CLI can create the exact same repo+venv layout, and older desktop
    //    builds could leave a healthy install behind without the marker. If the
    //    active runtime is usable, launch it directly; only fall through to
    //    bootstrap when the runtime itself is unusable.
    const activeRuntime = await activeRuntimeState()

    if (activeRuntime.shouldUseActiveRuntime && !state.bootstrapRepairRequested) {
      if (!activeRuntime.hasValidMarker) {
        rememberLog(
          `[bootstrap] Active Hermes runtime at ${ACTIVE_HERMES_ROOT} is usable but the bootstrap marker is missing or stale; skipping first-run bootstrap.`
        )
      }

      return createActiveBackend(backendArgs)
    }

    if (state.bootstrapRepairRequested) {
      rememberLog('[bootstrap] repair requested; bypassing the usable active runtime to re-run the installer')
    }

    // 4. Existing `hermes` on PATH -- installed via install.ps1 / install.sh from
    //    a previous tool-only setup, or pip-installed system-wide. Use it but
    //    do NOT write a bootstrap marker; the user did this themselves and we
    //    don't want to take ownership of an install we didn't perform.
    //    HERMES_DESKTOP_IGNORE_EXISTING=1 forces the bootstrap path for testing.
    if (process.env.HERMES_DESKTOP_IGNORE_EXISTING !== '1') {
      let hermesCommand = null
      const hermesOverride = process.env.HERMES_DESKTOP_HERMES

      if (hermesOverride) {
        const resolvedOverride = findOnPath(hermesOverride)

        if (resolvedOverride) {
          hermesCommand = resolvedOverride
        } else if (!isWindowsBinaryPathInWsl(hermesOverride, { isWsl: IS_WSL })) {
          hermesCommand = hermesOverride
        } else {
          rememberLog(`Ignoring Windows Hermes override under WSL: ${hermesOverride}`)
        }
      } else {
        hermesCommand = findOnPath('hermes')
      }

      if (hermesCommand) {
        if (looksLikeDesktopAppBinary(hermesCommand)) {
          rememberLog(`Ignoring desktop app executable on PATH while resolving Hermes CLI: ${hermesCommand}`)
          hermesCommand = null
        }
      }

      if (hermesCommand) {
        const unwrapped = await unwrapWindowsVenvHermesCommand(hermesCommand, backendArgs)

        if (unwrapped) {
          return unwrapped
        }

        // Smoke-test the candidate before trusting it. A `hermes` shim
        // left behind by a half-uninstalled pip install (or a venv
        // entry-point pointing at a deleted interpreter) still resolves
        // via findOnPath but explodes on spawn -- the user then sees a
        // dead backend instead of the first-launch installer. The cheap
        // `--version` probe (see backend-probes.ts) catches that case
        // and lets the resolver fall through to step 6 / bootstrap.
        const shellForProbe = isCommandScript(hermesCommand)

        // HERMES_DESKTOP_HERMES is an explicit deployment override (used by
        // the Nix wrapper), not a discovered PATH candidate. It must not fall
        // through to the install-script bootstrap if the optional probe times
        // out under load; the pinned backend is the only valid runtime there.
        if (
          shouldTrustHermesOverride(hermesOverride) ||
          (await verifyHermesCli(hermesCommand, { shell: shellForProbe }))
        ) {
          // `unwrapped` above already answered "is this a Windows venv shim?" —
          // it was null (not a shim, or its import probe failed). Do NOT re-run
          // unwrapWindowsVenvHermesCommand here: the second call repeats the
          // same un-memoized import probe, costing up to another full probe
          // timeout on the boot path for an answer we already have.
          return {
            label: `existing Hermes CLI at ${hermesCommand}`,
            command: hermesCommand,
            args: backendArgs,
            bootstrap: false,
            env: {},
            kind: 'command',
            shell: shellForProbe
          }
        }

        rememberLog(
          `Ignoring existing Hermes CLI at ${hermesCommand}: --version probe failed; falling through to bootstrap.`
        )
      }
    }

    // 5. Last-ditch: pip-installed hermes_cli module via system Python.
    //    Same rationale as #4 -- the user installed this; we use it but don't
    //    take ownership.
    const python = await findSystemPython()

    if (python) {
      // Same smoke-test rationale as step 4: a system Python in the
      // SUPPORTED_VERSIONS range can be registered (PEP 514) without
      // having hermes_cli installed -- common on dev boxes that have
      // a python.org install from prior unrelated work. Returning that
      // backend hands the spawn step a guaranteed ModuleNotFoundError.
      // Verify the import works before trusting the candidate; on
      // failure, fall through to step 6 so the bootstrap runner pulls
      // a uv-managed 3.11 into %LOCALAPPDATA%\hermes\hermes-agent\venv.
      if (await canImportHermesCli(python)) {
        return {
          kind: 'python',
          label: `installed hermes_cli module via ${python}`,
          command: python,
          args: ['-m', 'hermes_cli.main', ...backendArgs],
          bootstrap: false,
          env: {},
          shell: false
        }
      }

      rememberLog(`Ignoring system Python ${python}: hermes_cli is not importable; falling through to bootstrap.`)
    }

    // 6. Nothing usable yet -- signal the bootstrap runner that we need to
    //    clone+install. Phase 1D's bootstrap-runner consumes this sentinel
    //    and drives install.ps1 stages with a progress UI. Until 1D lands,
    //    callers see the sentinel and surface it as a user-facing error
    //    explaining what's missing.
    //
    //    We deliberately do NOT throw here -- throwing inside
    //    resolveHermesBackend was the old "no payload" path and forced the
    //    user into a dead end. With the bootstrap protocol, "no install yet"
    //    is a recoverable state the GUI can drive through.
    return {
      kind: 'bootstrap-needed',
      label: 'Hermes Agent not installed yet; bootstrap required',
      command: null,
      args: backendArgs,
      bootstrap: true,
      env: {},
      shell: false,
      // Hints for the bootstrap runner / UI layer:
      activeRoot: ACTIVE_HERMES_ROOT,
      installStamp: INSTALL_STAMP, // may be null in dev
      isPackaged: IS_PACKAGED,
      platform: process.platform
    }
  }

  function ensureRuntime(backend: any, assertStillOwned: () => void): Promise<any> {
    return localBackendLifecycle.start(() => runEnsureRuntime(backend, assertStillOwned))
  }

  async function runEnsureRuntime(backend: any, assertStillOwned: () => void): Promise<any> {
    localBackendLifecycle.assertCanStart()
    assertStillOwned()

    if (!backend.bootstrap) {
      await firstRunBoot.advanceBootProgress('runtime.external', `Using ${backend.label}`, 32)

      return backend
    }

    // backend.kind === 'bootstrap-needed' means resolveHermesBackend couldn't
    // find anything to spawn. Hand off to the bootstrap runner which drives the
    // platform installer, writes the bootstrap-complete marker on success, then
    // we re-resolve to get the now-installed backend.
    //
    // Phase 1D status: bootstrap runs but events go to desktop.log only
    // (renderer window isn't created until later in startBackend). Phase 1E
    // will rewire startup to spawn the window first and route bootstrap events
    // to a renderer-side install overlay.
    if (backend.kind === 'bootstrap-needed') {
      rememberLog('[bootstrap] no Hermes install found; starting first-launch bootstrap')

      if (await handOffWindowsBootstrapRecovery('bootstrap-needed')) {
        const handoffError: Error & { isBootstrapFailure?: boolean; bootstrapHandedOff?: boolean } = new Error(
          'Hermes recovery was handed off to Hermes Setup. The desktop will restart when recovery completes.'
        )

        handoffError.isBootstrapFailure = true
        handoffError.bootstrapHandedOff = true
        state.bootstrapFailure = handoffError
        throw handoffError
      }

      // Eagerly flip the bootstrap UI state to 'active' so the renderer
      // shows the install overlay BEFORE the runner finishes fetching the
      // manifest (which on slow networks can take tens of seconds and would
      // otherwise leave the user staring at the generic 'Preparing' splash).
      // We emit a synthetic manifest with an empty stages list -- the real
      // manifest event will overwrite it once install.ps1 -Manifest returns.
      try {
        firstRunBoot.broadcastBootstrapEvent({
          type: 'manifest',
          stages: [],
          protocolVersion: null
        })
      } catch {
        void 0
      }

      localBackendLifecycle.assertCanStart()
      state.bootstrapAbortController = new AbortController()

      // The repair request has been honoured by reaching the installer; clear it
      // so a later boot isn't forced through bootstrap again.
      state.bootstrapRepairRequested = false
      state.bootstrapRepairAttempt = 0

      const bootstrapResult = await runBootstrap({
        installStamp: backend.installStamp,
        activeRoot: backend.activeRoot,
        sourceRepoRoot: SOURCE_REPO_ROOT,
        hermesHome: HERMES_HOME,
        logRoot: path.join(HERMES_HOME, 'logs'),
        abortSignal: state.bootstrapAbortController.signal,
        onEvent: ev => {
          // Tee every bootstrap event to (a) the desktop log for forensics
          // and (b) the renderer for live progress UI. Either may be absent;
          // tolerate both gracefully so a renderer crash doesn't stall the
          // bootstrap and a log-write failure doesn't suppress the UI signal.
          try {
            rememberLog(`[bootstrap] ${JSON.stringify(ev)}`)
          } catch {
            void 0
          }

          try {
            firstRunBoot.broadcastBootstrapEvent(ev)
          } catch {
            void 0
          }
        },
        writeMarker: writeBootstrapMarker,
        gitBinary: resolveGitBinary()
      })

      state.bootstrapAbortController = null

      if (bootstrapResult.cancelled) {
        const cancelledError = new Error('Hermes install was cancelled.') as any
        cancelledError.isBootstrapFailure = true
        cancelledError.bootstrapCancelled = true
        state.bootstrapFailure = cancelledError
        throw cancelledError
      }

      if (!bootstrapResult.ok) {
        // Plain lead sentence + trailing "Details:" line; the install overlay
        // shows this verbatim and offers Reload and retry / Open logs itself.
        const bootstrapError = new Error(
          describeBootstrapFailure(bootstrapResult.failedStage, bootstrapResult.error)
        ) as any

        bootstrapError.isBootstrapFailure = true
        bootstrapError.failedStage = bootstrapResult.failedStage || null
        // Latch the failure so subsequent startHermes() calls return this
        // same error without re-running install.ps1.  Cleared by the
        // hermes:bootstrap:reset IPC (renderer's "Reload and retry").
        state.bootstrapFailure = bootstrapError
        throw bootstrapError
      }

      rememberLog('[bootstrap] bootstrap complete; marker written. Re-resolving backend.')

      // Re-resolve now that the install exists. The new resolution lands in
      // step 3 (bootstrap-complete marker) and we recurse to wire venvPython.
      return ensureRuntime(await resolveHermesBackend(backend.args), assertStillOwned)
    }

    // bootstrap=true with a real backend (createActiveBackend path) means we
    // have a checkout and need to ensure the venv-derived Python command is
    // wired into the backend before launch. Same code path the old factory
    // sync flow exited through, minus all the factory/pip/marker machinery
    // (install.ps1 owns those concerns now and the bootstrap-complete marker
    // attests they ran successfully).
    if (!isHermesSourceRoot(ACTIVE_HERMES_ROOT)) {
      throw new Error(
        missingInstallPartMessage(`Hermes source files are missing or incomplete at ${ACTIVE_HERMES_ROOT}`)
      )
    }

    // On Windows, preflight Git Bash. Hermes' terminal tool calls bash.exe
    // directly (tools/environments/local.py); without it the agent can't run
    // terminal commands. install.ps1's Stage-Git puts PortableGit at
    // %LOCALAPPDATA%\hermes\git\, which findGitBash() picks up, so for any
    // user who completed the bootstrap this is a no-op. For users who got
    // here via an external `hermes` on PATH, this check still helps.
    if (IS_WINDOWS && !findGitBash()) {
      throw new Error(
        "Hermes needs a helper called Git for Windows, which isn't installed. " +
          'Choose Repair install to add it automatically, or install it yourself from git-scm.com and reopen Hermes.'
      )
    }

    const venvPython = getVenvPython(VENV_ROOT)

    if (!fileExists(venvPython)) {
      // No venv at the expected location AND no bootstrap-needed sentinel
      // means we have a half-installed checkout: .git exists, source files
      // exist, but venv is missing or broken. This shouldn't happen in
      // normal flow because activeRuntimeState() requires isHermesSourceRoot()
      // plus an importable hermes_cli before it hands back the active runtime.
      // If we hit this, the user (or a deleted venv) broke the invariant; tell
      // them to re-run the install.
      throw new Error(missingInstallPartMessage(`Python environment missing at ${VENV_ROOT}`))
    }

    backend.command = getVenvPython(VENV_ROOT)
    backend.label = `Hermes at ${ACTIVE_HERMES_ROOT} (venv: ${VENV_ROOT})`
    firstRunBoot.updateBootProgress({
      phase: 'runtime.ready',
      message: 'Hermes runtime is ready',
      progress: 82,
      running: true,
      error: null
    })

    return backend
  }

  return { resolveHermesBackend, ensureRuntime }
}
