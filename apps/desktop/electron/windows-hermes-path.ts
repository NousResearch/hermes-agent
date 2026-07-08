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
 *   2. chooseUpdaterArgs() — handOffWindowsBootstrapRecovery() chose
 *      --update vs the destructive --repair by checking ONLY
 *      venv\Scripts\hermes.exe (the console-script shim, written at the END
 *      of venv setup and absent in interrupted states), so it escalated to a
 *      full venv recreate even on healthy installs. The fix: gate on ANY
 *      real-install signal, not just the shim.
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
 * Choose the Windows bootstrap-recovery updater invocation: the gentle
 * in-place --update when ANY real-install signal is present, the
 * destructive --repair (full venv recreate) otherwise.
 *
 * haveRealInstall must be computed by the caller from ALL real-install
 * signals (venv python interpreter, venv hermes shim, bootstrap-complete
 * marker) — gating on just the hermes.exe console-script shim alone is the
 * regression this function's callers must avoid: that shim is written at
 * the END of venv setup and is absent in exactly the interrupted/quarantined
 * states this recovery exists to heal.
 *
 * @param {boolean} haveRealInstall
 * @param {string} branch
 * @returns {string[]} updater argv, e.g. ['--update', '--branch', 'main'].
 */
export function chooseUpdaterArgs(haveRealInstall: boolean, branch: string): string[] {
  return haveRealInstall ? ['--update', '--branch', branch] : ['--repair', '--branch', branch]
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
