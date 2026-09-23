import path from 'node:path'

import { selectRunnableBinary } from './select-runnable-binary'

export interface ExecutableDiscoveryDeps {
  isWindows: boolean
  fileExists: (filePath: string) => boolean
  findOnPath: (command: string) => string | null
  getHomePath: () => string
  execFileSync: (
    candidate: string,
    args: string[],
    options: { stdio: 'ignore'; timeout: number; windowsHide: boolean }
  ) => unknown
}

export function createExecutableDiscoveryRuntime(deps: ExecutableDiscoveryDeps) {
  const IS_WINDOWS = deps.isWindows
  const { fileExists, findOnPath, getHomePath, execFileSync } = deps

  // resolveGitBinary — locate git.exe on Windows. A fresh installer-driven
  // install only has PortableGit under %LOCALAPPDATA%\hermes\git (never on
  // PATH), so a bare spawn('git') ENOENTs and self-update checks fail with
  // "Couldn't check for updates". Mirror findGitBash: PortableGit first, then
  // standard Git-for-Windows locations, then PATH. Cached after first probe.
  let _gitBinaryCache = null

  // A binary can exist on disk and still be unlaunchable — on macOS an
  // Intel-only build ahead on PATH (e.g. a pre-Rosetta-removal Homebrew)
  // fails at spawn time with errno -86 (EBADARCH), which callers then report
  // as an update-server/network problem. Probing `git --version` before
  // committing to a candidate skips such entries; the existence-only
  // fallback keeps behaviour unchanged where the probe itself cannot run.
  function binaryRuns(candidate) {
    try {
      execFileSync(candidate, ['--version'], {
        stdio: 'ignore',
        timeout: 5000,
        windowsHide: true
      })

      return true
    } catch {
      return false
    }
  }

  function findPathCandidates(command) {
    const pathEntries = String(process.env.PATH || '')
      .split(path.delimiter)
      .filter(Boolean)

    const candidates = []

    for (const entry of pathEntries) {
      const candidate = path.join(entry, command)

      if (fileExists(candidate)) {
        candidates.push(candidate)
      }
    }

    return candidates
  }

  function resolveGitBinary() {
    if (_gitBinaryCache) {
      return _gitBinaryCache
    }

    if (!IS_WINDOWS) {
      // Every PATH hit, probed — the first entry that merely exists can be
      // unlaunchable while a working system git sits later on the same PATH.
      const selected = selectRunnableBinary({
        candidates: findPathCandidates('git'),
        fileExists,
        binaryRuns
      })

      _gitBinaryCache = selected || 'git'

      return _gitBinaryCache
    }

    const localAppData = process.env.LOCALAPPDATA || ''
    const candidates = []

    if (localAppData) {
      candidates.push(path.join(localAppData, 'hermes', 'git', 'cmd', 'git.exe'))
      candidates.push(path.join(localAppData, 'hermes', 'git', 'bin', 'git.exe'))
    }

    candidates.push(path.join(process.env['ProgramFiles'] || 'C:\\Program Files', 'Git', 'cmd', 'git.exe'))
    candidates.push(path.join(process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)', 'Git', 'cmd', 'git.exe'))

    if (localAppData) {
      candidates.push(path.join(localAppData, 'Programs', 'Git', 'cmd', 'git.exe'))
    }

    _gitBinaryCache = candidates.find(fileExists) || findOnPath('git') || 'git'

    return _gitBinaryCache
  }

  // resolveGhBinary — locate the GitHub CLI. GUI-launched apps get a minimal PATH
  // that omits Homebrew (/opt/homebrew/bin, /usr/local/bin) where `gh` usually
  // lives, so a bare spawn('gh') ENOENTs even though `gh` works in the user's
  // terminal. Check the common install locations first, then PATH. Cached.
  let _ghBinaryCache = null

  function resolveGhBinary() {
    if (_ghBinaryCache) {
      return _ghBinaryCache
    }

    const candidates = []

    if (IS_WINDOWS) {
      candidates.push(path.join(process.env['ProgramFiles'] || 'C:\\Program Files', 'GitHub CLI', 'gh.exe'))

      if (process.env.LOCALAPPDATA) {
        candidates.push(path.join(process.env.LOCALAPPDATA, 'Microsoft', 'WinGet', 'Links', 'gh.exe'))
      }
    } else {
      const home = getHomePath()
      // Preserve the original resolver body through extraction.
      // prettier-ignore
      candidates.push('/opt/homebrew/bin/gh', '/usr/local/bin/gh', '/usr/bin/gh', path.join(home, '.local', 'bin', 'gh'))
      // PATH hits go through the same probe: a bare findOnPath fallback would
      // re-select an unlaunchable first hit when none of the fixed locations exist.
      candidates.push(...findPathCandidates('gh'))
    }

    // Same selection rule as git: an existing-but-unlaunchable candidate (e.g.
    // an Intel-only build from a stale Homebrew) must not shadow a working one
    // further down the list, and PATH is only consulted when none of the
    // explicit candidates is usable.
    const selected = selectRunnableBinary({
      candidates,
      fileExists,
      binaryRuns
    })

    _ghBinaryCache = selected || findOnPath('gh') || 'gh'

    return _ghBinaryCache
  }

  return { resolveGitBinary, resolveGhBinary }
}
