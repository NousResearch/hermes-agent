import fs from 'node:fs'
import path from 'node:path'

import { execText } from './backend-claim'
import { dashboardFallbackArgs } from './backend-command'
import { buildDesktopBackendEnv } from './backend-env'
import { canImportHermesCli, PROBE_TIMEOUT_MS } from './backend-probes'
import { createBackendServeSupportResolver } from './backend-serve-support'
import { isWindowsBinaryPathInWsl } from './bootstrap-platform'
import { findGitBash as _findGitBash } from './find-git-bash'
import { buildPathExtCandidates, getVenvSitePackagesEntries, resolveVenvHermesCommand } from './windows-hermes-path'

interface DesktopRuntimeDiscoveryDeps {
  hermesHome: string
  isWindows: boolean
  isWsl: boolean
  fileExists: (filePath: string) => boolean
  directoryExists: (directoryPath: string) => boolean
  rememberLog: (message: string) => void
}

export function createDesktopRuntimeDiscovery(deps: DesktopRuntimeDiscoveryDeps) {
  const {
    hermesHome: HERMES_HOME,
    isWindows: IS_WINDOWS,
    isWsl: IS_WSL,
    fileExists,
    directoryExists,
    rememberLog
  } = deps

  function findOnPath(command) {
    if (!command) {
      return null
    }

    if (path.isAbsolute(command) || command.includes(path.sep) || (IS_WINDOWS && command.includes('/'))) {
      if (!fileExists(command)) {
        return null
      }

      if (isWindowsBinaryPathInWsl(command, { isWsl: IS_WSL })) {
        return null
      }

      return command
    }

    const pathEntries = String(process.env.PATH || '')
      .split(path.delimiter)
      .filter(Boolean)

    // On Windows, try PATHEXT extensions BEFORE the bare (empty-extension) name.
    // A real command must resolve via its .exe/.cmd (Windows command-resolution
    // semantics consult PATHEXT); an extensionless file — e.g. a Git-Bash
    // shell-script shim named `hermes` — must not shadow `hermes.cmd`/`hermes.exe`.
    // The empty entry is kept LAST so callers that already include the extension
    // (py.exe, pwsh.exe, powershell.exe) still resolve.
    const extensions = buildPathExtCandidates(process.env.PATHEXT, IS_WINDOWS)

    for (const entry of pathEntries) {
      for (const extension of extensions) {
        const candidate = path.join(entry, `${command}${extension}`)

        if (fileExists(candidate)) {
          return candidate
        }
      }
    }

    return null
  }

  function isCommandScript(command) {
    return IS_WINDOWS && /\.(cmd|bat)$/i.test(command || '')
  }

  async function unwrapWindowsVenvHermesCommand(command, backendArgs) {
    return resolveVenvHermesCommand(command, backendArgs, {
      isWindows: IS_WINDOWS,
      isCommandScript,
      fileExists,
      directoryExists,
      canImportHermesCli,
      getVenvPython,
      getVenvSitePackagesEntries,
      buildDesktopBackendEnv,
      hermesHome: HERMES_HOME,
      resolvePath: (...segments) => path.resolve(...segments),
      dirname: p => path.dirname(p),
      basename: p => path.basename(p),
      rememberLog
    })
  }

  // Does the resolved runtime understand the `serve` subcommand? The desktop
  // spawns `hermes serve`; runtimes older than serve only have `dashboard`. We
  // detect support so getBackendArgsForRuntime() can route old runtimes through
  // the legacy `dashboard --no-open` form instead of crashing on an unknown
  // subcommand (would brick every user mid-upgrade — #54568 follow-up).
  // Fast-path / probe / cache strategy: see backend-serve-support.ts header.
  const backendSupportsServe = createBackendServeSupportResolver(HERMES_HOME, rememberLog)

  // Given a resolved backend whose args target `serve`, return the args the
  // runtime actually understands: unchanged when `serve` is supported, or
  // rewritten to `dashboard --no-open` for older runtimes.
  async function getBackendArgsForRuntime(backend) {
    return (await backendSupportsServe(backend)) ? backend.args : dashboardFallbackArgs(backend.args)
  }

  function normalizeExecutablePathForCompare(commandPath) {
    if (!commandPath) {
      return null
    }

    let resolved = path.resolve(String(commandPath))

    try {
      resolved = fs.realpathSync.native ? fs.realpathSync.native(resolved) : fs.realpathSync(resolved)
    } catch {
      // Fallback to path.resolve() above.
    }

    return IS_WINDOWS ? resolved.toLowerCase() : resolved
  }

  function looksLikeDesktopAppBinary(commandPath) {
    if (!IS_WINDOWS || !commandPath) {
      return false
    }

    const normalizedCandidate = normalizeExecutablePathForCompare(commandPath)
    const normalizedCurrentExec = normalizeExecutablePathForCompare(process.execPath)

    if (normalizedCandidate && normalizedCurrentExec && normalizedCandidate === normalizedCurrentExec) {
      return true
    }

    let resolved = path.resolve(String(commandPath))

    try {
      resolved = fs.realpathSync.native ? fs.realpathSync.native(resolved) : fs.realpathSync(resolved)
    } catch {
      // Keep resolved path fallback.
    }

    const resourcesDir = path.join(path.dirname(resolved), 'resources')

    return (
      fileExists(path.join(resourcesDir, 'app.asar')) || directoryExists(path.join(resourcesDir, 'app.asar.unpacked'))
    )
  }

  function isHermesSourceRoot(root) {
    return directoryExists(root) && fileExists(path.join(root, 'hermes_cli', 'main.py'))
  }

  async function findPythonForRoot(root) {
    const override = process.env.HERMES_DESKTOP_PYTHON

    if (override && fileExists(override)) {
      return override
    }

    const relativePaths = IS_WINDOWS
      ? [path.join('.venv', 'Scripts', 'python.exe'), path.join('venv', 'Scripts', 'python.exe')]
      : [path.join('.venv', 'bin', 'python'), path.join('venv', 'bin', 'python')]

    for (const relativePath of relativePaths) {
      const candidate = path.join(root, relativePath)

      if (fileExists(candidate)) {
        return candidate
      }
    }

    return findSystemPython()
  }

  async function findSystemPython() {
    if (!IS_WINDOWS) {
      // POSIX systems: PATH lookup is safe.
      for (const command of ['python3', 'python']) {
        const candidate = findOnPath(command)

        if (candidate) {
          return candidate
        }
      }

      return null
    }

    // Windows: PATH-based detection has TWO landmines we have to dodge.
    //
    //  (1) The Microsoft Store "Python stub" lives at
    //      %LOCALAPPDATA%\Microsoft\WindowsApps\python.exe and is on PATH
    //      by default on modern Windows. It's a redirector that opens the
    //      Store window if no Store Python is installed. Running it for
    //      `-m venv` would either succeed (real Store install — fine) or
    //      pop the Store dialog (bad UX during boot).
    //  (2) `py.exe` (Python launcher) is missing from per-user installs
    //      that didn't check the launcher option, so PATH-only checks
    //      miss real Python 3.13 installs (user-reported case).
    //
    // We also restrict ourselves to Python 3.11–3.13. 3.14 is the latest
    // CPython but several Hermes deps (notably pywinpty's Rust-built
    // windows_x86_64_msvc crate) don't yet publish 3.14 wheels, and
    // `pip install -e .` falls back to source-build, which fails without
    // a Rust toolchain. install.ps1 sidesteps this by pinning to 3.11
    // via uv; until we add the same uv-managed Python pathway here, the
    // simplest fix is to refuse 3.14 detection and let the NSIS prereq
    // page offer to install 3.11 alongside.
    //
    // Strategy: probe in three passes, in order from most-precise to
    // least-precise, and ONLY use PATH lookup as a last resort after
    // confirming the candidate isn't the WindowsApps redirector.
    //
    //  Pass 1: PEP 514 registry — every standards-compliant Python
    //          installer registers itself at SOFTWARE\Python\PythonCore.
    //          The MS Store stub does NOT register here, so a hit means
    //          a real Python install. Versions are explicit so we
    //          inherently filter 3.14 out.
    //  Pass 2: Filesystem probe of standard install locations
    //          (Program Files, LocalAppData\Programs\Python). Same
    //          version filtering by directory name.
    //  Pass 3: PATH lookup of `py.exe` (the launcher itself never
    //          triggers the Store) — but call it with a version flag so
    //          we resolve to a SPECIFIC supported version, not whatever
    //          py.exe's default is (which on a 3.14-only box would be
    //          3.14).

    const SUPPORTED_VERSIONS = ['3.11', '3.12', '3.13']
    const SUPPORTED_VERSIONS_NO_DOT = ['311', '312', '313']

    // Pass 1: registry. Use `reg query` since main process doesn't have
    // a reliable in-process registry API across all electron versions.
    for (const hive of ['HKLM', 'HKCU']) {
      for (const version of SUPPORTED_VERSIONS) {
        try {
          const out = await execText(
            'reg',
            ['query', `${hive}\\SOFTWARE\\Python\\PythonCore\\${version}\\InstallPath`, '/ve', '/reg:64'],
            { timeout: 5_000 }
          )

          // Output format: "    (Default)    REG_SZ    C:\Path\To\Python\"
          const match = out.match(/REG_SZ\s+(.+?)\s*$/m)

          if (match) {
            const installPath = match[1].trim()
            const pythonExe = path.join(installPath, 'python.exe')

            if (fileExists(pythonExe)) {
              return pythonExe
            }
          }
        } catch {
          // Key not present — try next.
        }
      }
    }

    // Pass 2: filesystem probe of standard locations.
    const programFiles = process.env['ProgramFiles'] || 'C:\\Program Files'
    const localAppData = process.env.LOCALAPPDATA || ''

    for (const versionDir of SUPPORTED_VERSIONS_NO_DOT) {
      const systemWide = path.join(programFiles, `Python${versionDir}`, 'python.exe')

      if (fileExists(systemWide)) {
        return systemWide
      }

      if (localAppData) {
        const perUser = path.join(localAppData, 'Programs', 'Python', `Python${versionDir}`, 'python.exe')

        if (fileExists(perUser)) {
          return perUser
        }
      }
    }

    // Pass 3: py.exe with explicit version flag. The launcher itself is
    // safe to invoke (no Store popup) and `py -3.13 -c "import sys;
    // print(sys.executable)"` resolves to the actual python.exe path of
    // the requested version. We try in version-priority order so the
    // first hit wins.
    const pyExe = findOnPath('py.exe')

    if (pyExe) {
      for (const version of SUPPORTED_VERSIONS) {
        try {
          const out = await execText(pyExe, [`-${version}`, '-c', 'import sys; print(sys.executable)'], {
            timeout: PROBE_TIMEOUT_MS
          })

          const candidate = out.trim()

          if (candidate && fileExists(candidate)) {
            return candidate
          }
        } catch {
          // py couldn't find that version — try next.
        }
      }
    }

    // We deliberately do NOT fall back to plain `python.exe` on PATH.
    // Without a way to verify the version safely (running `python -V`
    // risks the Microsoft Store popup), accepting whatever's there
    // could land us on 3.14 and trigger the Rust-build-from-source
    // failure. Better to return null and let the NSIS prereq page
    // offer to install a known-good 3.11 via winget.
    return null
  }

  // findGitBash — locate bash.exe on Windows. Resolves HERMES_GIT_BASH_PATH
  // first (mirrors tools/environments/local.py:_find_bash), then PortableGit,
  // standard install locations, and finally PATH.
  function findGitBash() {
    return _findGitBash({
      isWindows: IS_WINDOWS,
      env: process.env,
      fileExists,
      findOnPath
    })
  }

  function getVenvPython(venvRoot) {
    return path.join(venvRoot, IS_WINDOWS ? path.join('Scripts', 'python.exe') : path.join('bin', 'python'))
  }

  // Map a selected interpreter back to the venv that OWNS it (the directory
  // above bin/ or Scripts/), but only when that venv lives inside `root`.
  // Returns null for system pythons — they own no site-packages we should mount.
  //
  // This exists because findPythonForRoot() probes `.venv` before `venv`, and a
  // checkout can legitimately have BOTH (dev tooling venv + the CLI install
  // venv, possibly on different Python versions). The interpreter and the
  // site-packages placed on PYTHONPATH must come from the SAME venv: pairing a
  // .venv 3.12 python with venv/lib/python3.11/site-packages makes the backend
  // die on its first native import (pydantic_core) before the gateway binds —
  // the renderer then reports "Gateway offline" on every profile.
  function venvRootForPython(python: string, root: string) {
    const parent = path.dirname(python)
    const binName = path.basename(parent).toLowerCase()

    if (binName !== 'bin' && binName !== 'scripts') {
      return null
    }

    const candidate = path.dirname(parent)
    const relative = path.relative(root, candidate)

    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
      return null
    }

    return candidate
  }

  return {
    findOnPath,
    isCommandScript,
    unwrapWindowsVenvHermesCommand,
    getBackendArgsForRuntime,
    looksLikeDesktopAppBinary,
    isHermesSourceRoot,
    findPythonForRoot,
    findSystemPython,
    findGitBash,
    getVenvPython,
    venvRootForPython
  }
}
