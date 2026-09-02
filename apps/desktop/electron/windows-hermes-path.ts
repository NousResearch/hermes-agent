/**
 * windows-hermes-path.ts
 *
 * Pure, dependency-injected pieces of Windows `hermes` resolution pulled out
 * of main.ts's findOnPath(), handOffWindowsBootstrapRecovery(), and
 * unwrapWindowsVenvHermesCommand(). Each of the three functions here pins one
 * of the Windows resolution bugs that caused desktop reinstall loops:
 *
 *   1. buildPathExtCandidates() — findOnPath() tried the empty extension
 *      FIRST, so an extensionless Git-Bash `hermes` shim shadowed the real
 *      hermes.cmd/hermes.exe; the shim then failed the --version probe and
 *      the desktop fell through to a spurious bootstrap/repair. The fix:
 *      PATHEXT extensions first, empty extension LAST.
 *   2. chooseUpdaterArgs() — handOffWindowsBootstrapRecovery() must separate
 *      install provenance from updater viability. A bootstrap-complete marker
 *      can outlive a deleted venv, while the updater needs BOTH the venv Python
 *      and Hermes launcher. Marker-only or partial runtimes must use --repair;
 *      only a runnable pair can use --update.
 *   3. resolveVenvHermesCommand() — unwrapWindowsVenvHermesCommand() returned
 *      the venv python with NO runtime probe (bypassing the caller's
 *      --version check too), so a venv broken mid-update (e.g. missing
 *      python-dotenv) was re-selected forever: Retry / "Repair install"
 *      resolved the same dead interpreter instead of falling through to the
 *      bootstrap installer. The fix: probe-before-trust.
 *
 * Kept in a standalone ts module (no Electron imports, dependencies passed
 * as parameters) so it can be unit-tested with `node --test` without
 * mocking Electron or the filesystem, same pattern as backend-probes.ts and
 * backend-command.ts.
 */

import fs from 'node:fs'
import path from 'node:path'

/**
 * Build the ordered list of extensions findOnPath() should try when
 * resolving a bare command name off PATH.
 *
 * On Windows this MUST try PATHEXT extensions (.COM;.EXE;.BAT;.CMD by
 * default) BEFORE the bare/empty-extension name: a real command resolves via
 * its .exe/.cmd per Windows command-resolution semantics, and an
 * extensionless file (e.g. a Git-Bash shell-script shim named `hermes`) must
 * not shadow `hermes.cmd`/`hermes.exe`. The empty entry is kept LAST so
 * callers that already include the extension (py.exe, pwsh.exe,
 * powershell.exe) still resolve.
 *
 * On non-Windows platforms there is no PATHEXT concept: only the bare name
 * is tried.
 *
 * @param {string | undefined} pathext - process.env.PATHEXT (or undefined).
 * @param {boolean} isWindows
 * @returns {string[]} extensions to try, in order, always ending in ''.
 */
export function buildPathExtCandidates(pathext: string | undefined, isWindows: boolean): string[] {
  if (!isWindows) {
    return ['']
  }

  return [...(pathext || '.COM;.EXE;.BAT;.CMD').split(';').filter(Boolean), '']
}

/**
 * Choose the Windows bootstrap-recovery invocation. The gentle in-place
 * updater can only start when both pieces of its runtime contract exist: the
 * venv Python interpreter and the Hermes launcher that drives `hermes update`.
 * A bootstrap-complete marker proves install provenance, not current runtime
 * usability, and may remain after the venv is removed or quarantined.
 *
 * @param {BootstrapRecoverySignals} signals
 * @param {string} branch
 * @returns {string[]} updater argv, e.g. ['--update', '--branch', 'main'].
 */
export interface BootstrapRecoverySignals {
  hasBootstrapMarker: boolean
  hasVenvHermes: boolean
  hasVenvPython: boolean
}

export function chooseUpdaterArgs(signals: BootstrapRecoverySignals, branch: string): string[] {
  const canRunUpdater = signals.hasVenvHermes && signals.hasVenvPython

  return canRunUpdater ? ['--update', '--branch', branch] : ['--repair', '--branch', branch]
}

/**
 * Resolve the site-packages directory entries for a Python venv.
 *
 * On Windows, venv layout is `<venvRoot>/Lib/site-packages`.
 * On POSIX, it's `<venvRoot>/lib/python<version>/site-packages` where
 * `<version>` (e.g. `3.12`) is read from the venv's `pyvenv.cfg`
 * `version_info` field.
 *
 * Returns only directories that actually exist on disk. Returns an empty
 * array when `venvRoot` is falsy or no matching site-packages dir is found.
 *
 * Extracted from main.ts so the platform branching can be tested without
 * reading source text. `isWindows` and `directoryExists` are injectable;
 * `readFile` defaults to `fs.readFileSync` but can be overridden for tests.
 */
export function getVenvSitePackagesEntries(
  venvRoot: string | undefined | null,
  opts: {
    isWindows?: boolean
    directoryExists?: (p: string) => boolean
    readFile?: (p: string) => string | undefined
  } = {}
): string[] {
  const entries: string[] = []

  if (!venvRoot) {
    return entries
  }

  const isWindows = opts.isWindows ?? process.platform === 'win32'

  const directoryExists =
    opts.directoryExists ??
    ((p: string) => {
      try {
        return fs.statSync(p).isDirectory()
      } catch {
        return false
      }
    })

  const readFile =
    opts.readFile ??
    ((p: string) => {
      try {
        return fs.readFileSync(p, 'utf8')
      } catch {
        return undefined
      }
    })

  if (isWindows) {
    const sitePackages = path.join(venvRoot, 'Lib', 'site-packages')

    if (directoryExists(sitePackages)) {
      entries.push(sitePackages)
    }

    return entries
  }

  const cfg = readFile(path.join(venvRoot, 'pyvenv.cfg'))

  const version = (() => {
    if (!cfg) {
      return null
    }

    const match = cfg.match(/^version_info\s*=\s*(\d+\.\d+)/im)

    return match ? match[1].trim() : null
  })()

  if (version) {
    const sitePackages = path.join(venvRoot, 'lib', `python${version}`, 'site-packages')

    if (directoryExists(sitePackages)) {
      entries.push(sitePackages)
    }
  }

  return entries
}

export interface ResolveVenvHermesCommandDeps {
  isWindows: boolean
  isCommandScript: (command: string) => boolean
  fileExists: (filePath: string) => boolean
  directoryExists: (filePath: string) => boolean
  canImportHermesCli: (python: string, opts?: { env?: Record<string, string> }) => boolean
  getVenvPython: (venvRoot: string) => string
  getVenvSitePackagesEntries: (venvRoot: string) => string[]
  buildDesktopBackendEnv: (opts: {
    hermesHome: string
    pythonPathEntries: string[]
    venvRoot: string
  }) => Record<string, string>
  hermesHome: string
  resolvePath: (...segments: string[]) => string
  dirname: (p: string) => string
  basename: (p: string) => string
  rememberLog?: (message: string) => void
}

/**
 * If `command` is a Windows venv `hermes`/`hermes.exe` console-script shim
 * (i.e. `<venvRoot>/Scripts/hermes(.exe)`), resolve it to the underlying
 * venv python invoked as `python -m hermes_cli.main <backendArgs>` — but
 * ONLY after smoke-testing that interpreter with canImportHermesCli(). A
 * venv whose update died mid-`pip install` still has python.exe + hermes.exe
 * on disk, but the backend dies on its first import (e.g.
 * ModuleNotFoundError: dotenv) before the gateway ever binds. Returning it
 * unprobed also bypasses the caller's `--version` probe, so Retry/"Repair
 * install" re-resolves the same broken venv forever instead of falling
 * through to the bootstrap installer.
 *
 * Mirrors isActiveRuntimeUsable(): probes with the checkout on PYTHONPATH so
 * a healthy source-tree venv passes.
 *
 * Returns null when `command` is not a venv hermes shim, the underlying
 * python doesn't exist, or the import probe fails. Otherwise returns the
 * resolved backend descriptor.
 */
export function resolveVenvHermesCommand(
  command: string,
  backendArgs: string[],
  deps: ResolveVenvHermesCommandDeps
): {
  label: string
  command: string
  args: string[]
  bootstrap: false
  env: Record<string, string>
  kind: 'python'
  root: string
  shell: false
} | null {
  const {
    isWindows,
    isCommandScript,
    fileExists,
    directoryExists,
    canImportHermesCli,
    getVenvPython,
    getVenvSitePackagesEntries,
    buildDesktopBackendEnv,
    hermesHome,
    resolvePath,
    dirname,
    basename,
    rememberLog
  } = deps

  if (!isWindows || !command || isCommandScript(command)) {
    return null
  }

  const resolved = resolvePath(String(command))

  if (!/^hermes(?:\.exe)?$/i.test(basename(resolved))) {
    return null
  }

  const scriptsDir = dirname(resolved)

  if (basename(scriptsDir).toLowerCase() !== 'scripts') {
    return null
  }

  const venvRoot = dirname(scriptsDir)
  const python = getVenvPython(venvRoot)

  if (!fileExists(python)) {
    return null
  }

  const root = dirname(venvRoot)

  if (
    !canImportHermesCli(python, {
      env: {
        PYTHONPATH: [...(directoryExists(root) ? [root] : []), process.env.PYTHONPATH]
          .filter((entry): entry is string => Boolean(entry))
          .join(path.delimiter)
      }
    })
  ) {
    rememberLog?.(
      `Ignoring venv Hermes at ${python}: runtime import probe failed (broken/partial venv); falling through to bootstrap.`
    )

    return null
  }

  return {
    label: `existing Hermes Python at ${python}`,
    command: python,
    args: ['-m', 'hermes_cli.main', ...backendArgs],
    bootstrap: false,
    env: buildDesktopBackendEnv({
      hermesHome,
      pythonPathEntries: [...(directoryExists(root) ? [root] : []), ...getVenvSitePackagesEntries(venvRoot)],
      venvRoot
    }),
    kind: 'python',
    root,
    shell: false
  }
}
