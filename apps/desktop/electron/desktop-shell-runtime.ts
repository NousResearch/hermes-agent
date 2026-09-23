import type * as childProcess from 'node:child_process'
import type * as nodeFs from 'node:fs'
import type * as nodeOs from 'node:os'
import type * as nodePath from 'node:path'

import { type BundleSkewStamp, detectBundleSkew, type RunGit } from './bundle-skew'
import { type BundleSwapStamp, detectBundleSwap } from './bundle-swap'
import {
  buildPosixCleanupScript,
  buildWindowsCleanupScript,
  modeRemovesAgent,
  modeRemovesUserData,
  resolveRemovableAppPath,
  shouldRemoveAppBundle,
  uninstallArgsForMode
} from './desktop-uninstall'

// Main keeps the module-scope timing and IPC registrations. Dependencies are
// injected once before the initial About seed; callback services and the
// handoff setter stay bound to main's live state.
export interface DesktopShellRuntimeDependencies {
  ACTIVE_HERMES_ROOT: string
  APP_NAME: string
  HERMES_HOME: string
  INSTALL_STAMP: (BundleSkewStamp & BundleSwapStamp) | null
  IS_PACKAGED: boolean
  IS_WINDOWS: boolean
  VENV_ROOT: string
  app: Electron.App
  buildNoSandboxRelaunchArgs: (args: string[]) => string[]
  exitAfterBackendShutdown: (code: number) => Promise<unknown>
  fileExists: (filename: string) => boolean
  findSystemPython: () => Promise<string | null>
  fs: typeof nodeFs
  getVenvPython: (root: string) => string
  hiddenWindowsChildOptions: (options: any) => any
  isHermesSourceRoot: (root: string) => boolean
  loadInstallStamp: () => BundleSwapStamp | null
  os: typeof nodeOs
  path: typeof nodePath
  process: NodeJS.Process
  releaseBackendLock: (root: string, tag: string) => Promise<unknown>
  rememberLog: (message: string) => void
  resolveUpdateRoot: () => string
  runGit: RunGit
  setQuittingForHandoff: () => void
  spawn: typeof childProcess.spawn
}

export function createDesktopShellRuntime(deps: DesktopShellRuntimeDependencies) {
  const {
    ACTIVE_HERMES_ROOT,
    APP_NAME,
    HERMES_HOME,
    INSTALL_STAMP,
    IS_PACKAGED,
    IS_WINDOWS,
    VENV_ROOT,
    app,
    buildNoSandboxRelaunchArgs,
    exitAfterBackendShutdown,
    fileExists,
    findSystemPython,
    fs,
    getVenvPython,
    hiddenWindowsChildOptions,
    isHermesSourceRoot,
    loadInstallStamp,
    os,
    path,
    process,
    releaseBackendLock,
    rememberLog,
    resolveUpdateRoot,
    runGit,
    setQuittingForHandoff,
    spawn
  } = deps

  // Resolve the canonical Hermes version (the one `release.py` bumps in
  // hermes_cli/__init__.py + pyproject.toml) so the desktop About panel shows the
  // real Hermes version instead of the Electron app's own package.json version,
  // which historically drifted (stuck at 0.0.2). Falls back to app.getVersion()
  // when the source tree can't be read (e.g. a packaged build without the repo).
  function resolveHermesVersion() {
    try {
      const root = resolveUpdateRoot()
      const initPath = path.join(root, 'hermes_cli', '__init__.py')

      if (fileExists(initPath)) {
        const raw = fs.readFileSync(initPath, 'utf8')
        const match = raw.match(/__version__\s*=\s*["']([^"']+)["']/)

        if (match) {
          return match[1]
        }
      }
    } catch {
      // Fall through to the Electron app version below.
    }

    return app.getVersion()
  }

  // Renderer-bundle skew: `hermes update` moves the SOURCE TREE, but the UI
  // (including bundled plugins like Bot Mode) is compiled into this binary at
  // build time. A terminal-side update — or an in-app update whose bundle-swap
  // leg failed — leaves the new runtime running under an old renderer, so About
  // shows the new version while the sidebar is missing that version's desktop
  // features. Compare the build stamp's commit against the tree, scoped to
  // apps/desktop/, and warn when the running renderer is provably behind.
  // Fail-quiet: dev runs (no stamp), non-git builds, and shallow-clone gaps all
  // report in-sync rather than risk a false "your install is torn" warning.
  async function detectRendererSkew() {
    return detectBundleSkew(INSTALL_STAMP, runGit, resolveUpdateRoot())
  }

  // Re-resolve the live Hermes version and push it into the native About panel
  // just before showing it, so an in-place `hermes update` is reflected without
  // an app restart. macOS only — `showAboutPanel()` is a no-op elsewhere, and the
  // other platforms don't use this menu item.
  function showAboutPanelFresh() {
    void detectRendererSkew().then(skew => {
      app.setAboutPanelOptions({
        applicationName: APP_NAME,
        applicationVersion: skew.outOfSync
          ? `${resolveHermesVersion()} — app build out of date, update the desktop app`
          : resolveHermesVersion(),
        copyright: 'Copyright © 2026 Nous Research'
      })
      app.showAboutPanel()
    })
  }

  async function getVersionInfo() {
    const skew = await detectRendererSkew()

    return {
      appVersion: resolveHermesVersion(),
      electronVersion: process.versions.electron,
      nodeVersion: process.versions.node,
      platform: process.platform,
      hermesRoot: resolveUpdateRoot(),
      bundleOutOfSync: skew.outOfSync,
      bundleCommitsBehind: skew.desktopCommitsBehind,
      // True when the bundle on disk is not the one this process loaded — a
      // plain app restart (no rebuild, no installer) clears the skew above.
      // Packaged only: a dev `--build-only` rewrites build/install-stamp.json
      // under a running `npm start`, which is a rebuild the developer asked for,
      // not a torn install to offer a restart for.
      bundleSwapPending: IS_PACKAGED && detectBundleSwap(INSTALL_STAMP, loadInstallStamp())
    }
  }

  // The About page's "Restart Hermes" button (shown when bundleSwapPending):
  // load the already-swapped bundle without asking the user to quit manually.
  // app.relaunch() re-executes by path, so the fresh process picks up whatever
  // bundle now lives there.

  async function relaunchAfterBundleSwap() {
    rememberLog('[updates] renderer requested an app relaunch (swapped bundle pending)')
    app.relaunch({ args: buildNoSandboxRelaunchArgs(process.argv.slice(1)) })
    void exitAfterBackendShutdown(0)
  }

  // Host facts the guided first run asks for once, to decide whether "set this
  // machine up" is the likeliest first task or just one option among several.
  // Age is the birthtime of the user's home directory — when the OS created this
  // account, the closest thing to "when did this machine become theirs" that
  // costs a single stat. Filesystems that keep no birthtime report null, and the
  // flow reads unknown as not-new.

  async function getMachineProfile() {
    let ageDays: null | number = null

    try {
      const { birthtimeMs } = fs.statSync(os.homedir())

      if (birthtimeMs > 0) {
        ageDays = Math.max(0, Math.floor((Date.now() - birthtimeMs) / 86_400_000))
      }
    } catch {
      // Unknown age — the option still shows, it just doesn't lead.
    }

    // The OS login name powers a first-name SUGGESTION in the guided chat ("or
    // I can just call you akp"). Best-effort: an unidentifiable user just gets
    // no suggestion.
    let username = ''

    try {
      username = os.userInfo().username
    } catch {
      // No account name to suggest — the guide simply asks.
    }

    return {
      ageDays,
      arch: process.arch,
      // What the OS is set to, so a first run can open in the user's own
      // language instead of asking them to go and find the setting. Chromium
      // resolves this from the real OS preference (not the app's own bundle),
      // so it is the honest answer even though every UI string is English
      // until a translation exists.
      locale: app.getLocale() || '',
      model: readHardwareModel(),
      nvidia: await hasNvidiaGpu(),
      platform: process.platform,
      release: os.release(),
      username
    }
  }

  /** The board's own name for itself. Firmware writes it to the device tree on
   *  ARM systems (`NVIDIA_DGX_Spark`), which is how the first run can greet a
   *  DGX Spark as a Spark instead of "a Linux box". Empty everywhere else,
   *  Windows included — the RTX Spark is identified from the GPU instead. */
  function readHardwareModel(): string {
    try {
      return fs.readFileSync('/proc/device-tree/model', 'utf8').replace(/\0/g, '').trim()
    } catch {
      return ''
    }
  }

  const NVIDIA_PCI_VENDOR_ID = 0x10de

  /** Chromium already enumerated the GPUs to decide how to composite, so this is
   *  a lookup rather than a probe — no subprocess, no vendor tooling that a
   *  just-unboxed machine may not have yet. Paired with Windows-on-Arm it is what
   *  names an RTX Spark. */
  async function hasNvidiaGpu(): Promise<boolean> {
    try {
      // SAFETY: Electron's basic GPU info is Chromium's GPU record; each gpuDevice has a numeric PCI vendorId.
      const info = (await app.getGPUInfo('basic')) as { gpuDevice?: { vendorId?: number }[] }

      return (info.gpuDevice ?? []).some(device => device.vendorId === NVIDIA_PCI_VENDOR_ID)
    } catch {
      return false
    }
  }

  // ===========================================================================
  // Uninstall — remove the Chat GUI (and optionally the agent / user data).
  // ===========================================================================
  //
  // The renderer's About → Danger Zone surfaces three options that mirror the
  // CLI exactly: GUI only, Lite (keep user data), Full. We ask the agent to do
  // the actual removal via `hermes uninstall …` so the cross-platform PATH /
  // registry / service / node-symlink cleanup all lives in one place
  // (hermes_cli/uninstall.py + hermes_cli/gui_uninstall.py).
  //
  // getUninstallSummary() shells out to `--gui-summary` (a fast, no-side-effect
  // JSON probe) so the UI can gate options on what's actually installed — and
  // detect a missing agent (a future "lite client" that ships without the
  // bundled agent), hiding the agent/full options when there's nothing to remove.

  function uninstallVenvPython() {
    return getVenvPython(VENV_ROOT)
  }

  async function getUninstallSummary() {
    const py = uninstallVenvPython()
    const agentRoot = ACTIVE_HERMES_ROOT

    // Fast JS-side fallback used when the agent venv is gone (lite client) or the
    // probe fails — the renderer still needs *something* to render options from.
    const fallback = () => ({
      hermes_home: HERMES_HOME,
      agent_installed: isHermesSourceRoot(agentRoot) && fileExists(py),
      gui_installed: true,
      source_built_artifacts: [],
      packaged_app_paths: [],
      userdata_dir: app.getPath('userData'),
      userdata_exists: true,
      platform: process.platform,
      probe: 'fallback'
    })

    if (!fileExists(py)) {
      return fallback()
    }

    return new Promise(resolve => {
      let stdout = ''
      let settled = false

      const done = value => {
        if (settled) {
          return
        }

        settled = true
        resolve(value)
      }

      try {
        const child = spawn(
          py,
          ['-m', 'hermes_cli.main', 'uninstall', '--gui-summary'],
          hiddenWindowsChildOptions({
            cwd: agentRoot,
            env: { ...process.env, HERMES_HOME, NO_COLOR: '1' },
            stdio: ['ignore', 'pipe', 'ignore']
          })
        )

        child.stdout.on('data', chunk => {
          stdout += chunk.toString()
        })
        child.on('error', () => done(fallback()))
        child.on('exit', code => {
          if (code !== 0) {
            return done(fallback())
          }

          try {
            const line = stdout.trim().split('\n').filter(Boolean).pop() || '{}'
            const parsed = JSON.parse(line)
            // The app bundle the renderer would be removing on *this* machine,
            // resolved from the running exe (the Python probe only knows the
            // standard locations, not where THIS build actually runs from).
            parsed.running_app_path = resolveRemovableAppPath(process.execPath, process.platform, process.env)
            done(parsed)
          } catch {
            done(fallback())
          }
        })
        setTimeout(() => done(fallback()), 8000)
      } catch {
        done(fallback())
      }
    })
  }

  async function runDesktopUninstall(mode) {
    let uninstallArgs

    try {
      uninstallArgs = uninstallArgsForMode(mode)
    } catch (error) {
      return { ok: false, error: 'invalid-mode', message: error.message }
    }

    const venvPy = uninstallVenvPython()

    if (!fileExists(venvPy)) {
      return {
        ok: false,
        error: 'agent-missing',
        message: `Can't run the uninstaller: no Hermes agent venv at ${VENV_ROOT}.`
      }
    }

    // Interpreter choice (Finding 3): lite/full rmtree the venv that holds the
    // running python.exe. On Windows a running .exe is mandatory-locked, so the
    // rmtree must NOT be driven by the venv's own interpreter — use a system
    // Python with PYTHONPATH=<agentRoot> so `import hermes_cli` resolves from
    // source while the venv is torn down. gui-only doesn't touch the venv, so the
    // venv python is fine there. If no system Python exists (the Windows edge
    // case), fall back to the venv python — gui-only is unaffected; lite/full may
    // leave venv remnants the user can delete, which we log.
    let py = venvPy
    let pythonPath = null

    if (modeRemovesAgent(mode)) {
      const sysPy = await findSystemPython()

      if (sysPy) {
        py = sysPy
        pythonPath = ACTIVE_HERMES_ROOT
      } else if (IS_WINDOWS) {
        rememberLog(
          '[uninstall] no system Python found for lite/full on Windows; falling back ' +
            'to the venv python — venv files locked by the running interpreter may ' +
            'remain and need manual deletion.'
        )
      }
    }

    const appPath = resolveRemovableAppPath(process.execPath, process.platform, process.env)
    const removeBundle = shouldRemoveAppBundle(IS_PACKAGED, appPath) ? appPath : null

    // CRITICAL (Windows): tear down every backend the desktop owns and wait for
    // the venv shim to unlock BEFORE the cleanup script runs. lite/full delete
    // the venv, and even gui-only removes the install tree's GUI artifacts — a
    // live backend grandchild (gateway / pty / REPL) holding a mandatory file
    // lock would make the script's rmdir half-fail (#37532 for the update path).
    // Reuses the incident-hardened update teardown; no-op on macOS/Linux.
    try {
      await releaseBackendLock(ACTIVE_HERMES_ROOT, 'uninstall')
    } catch (error) {
      rememberLog(`[uninstall] backend teardown errored (continuing): ${error.message}`)
    }

    const scriptArgs = {
      desktopPid: process.pid,
      pythonExe: py,
      pythonPath,
      agentRoot: ACTIVE_HERMES_ROOT,
      uninstallArgs,
      appPath: removeBundle,
      hermesHome: HERMES_HOME
    }

    let scriptPath
    let runner
    let runnerArgs

    try {
      if (IS_WINDOWS) {
        scriptPath = path.join(app.getPath('temp'), `hermes-uninstall-${Date.now()}.cmd`)
        fs.writeFileSync(scriptPath, buildWindowsCleanupScript(scriptArgs))
        runner = process.env.ComSpec || 'cmd.exe'
        runnerArgs = ['/c', scriptPath]
      } else {
        scriptPath = path.join(app.getPath('temp'), `hermes-uninstall-${Date.now()}.sh`)
        fs.writeFileSync(scriptPath, buildPosixCleanupScript(scriptArgs), { mode: 0o755 })
        runner = '/bin/bash'
        runnerArgs = [scriptPath]
      }
    } catch (error) {
      return { ok: false, error: 'script-write-failed', message: error.message }
    }

    try {
      const child = spawn(runner, runnerArgs, {
        detached: true,
        stdio: 'ignore',
        windowsHide: true
      })

      child.unref()
    } catch (error) {
      return { ok: false, error: 'spawn-failed', message: error.message }
    }

    rememberLog(
      `[uninstall] launched detached cleanup (${mode}): ${scriptPath} ` +
        `(removesAgent=${modeRemovesAgent(mode)} removesUserData=${modeRemovesUserData(mode)} bundle=${removeBundle || 'none'})`
    )

    // Give the renderer a beat to show its "uninstalling…" state, then quit so
    // the venv python shim + app bundle unlock and the cleanup script can run.
    setQuittingForHandoff()
    setTimeout(() => app.quit(), 800)

    return { ok: true, mode, willRemoveAppBundle: Boolean(removeBundle), scriptPath }
  }

  return {
    resolveHermesVersion,
    showAboutPanelFresh,
    getVersionInfo,
    relaunchAfterBundleSwap,
    getMachineProfile,
    getUninstallSummary,
    runDesktopUninstall
  }
}
