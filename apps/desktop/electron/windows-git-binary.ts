/**
 * windows-git-binary.ts
 *
 * Resolve the git binary used by the Desktop review pane, with a Windows-specific
 * short-path conversion so simple-git's customBinaryPlugin accepts the path.
 */

import path from 'node:path'

import { toShortPath as defaultToShortPath } from './windows-short-path'

export interface ResolveGitBinaryDeps {
  isWindows: boolean
  fileExists: (filePath: string) => boolean
  findOnPath: (command: string) => string | null
  toShortPath?: (filePath: string) => string
  env?: {
    LOCALAPPDATA?: string | undefined
    ProgramFiles?: string | undefined
    'ProgramFiles(x86)'?: string | undefined
  }
}

let _gitBinaryCache: string | null = null

export function resetGitBinaryCache(): void {
  _gitBinaryCache = null
}

/**
 * Locate git.exe. On Windows, convert an absolute resolved path to its 8.3
 * short form before caching so simple-git's argument validation accepts it.
 */
export function resolveGitBinary(deps: ResolveGitBinaryDeps): string {
  if (_gitBinaryCache !== null) {
    return _gitBinaryCache
  }

  const {
    isWindows,
    fileExists,
    findOnPath,
    toShortPath = defaultToShortPath,
    env = process.env
  } = deps

  if (!isWindows) {
    _gitBinaryCache = findOnPath('git') || 'git'

    return _gitBinaryCache
  }

  const localAppData = env.LOCALAPPDATA || ''
  const candidates: string[] = []

  if (localAppData) {
    candidates.push(path.win32.join(localAppData, 'hermes', 'git', 'cmd', 'git.exe'))
    candidates.push(path.win32.join(localAppData, 'hermes', 'git', 'bin', 'git.exe'))
  }

  candidates.push(
    path.win32.join(env.ProgramFiles || 'C:\\Program Files', 'Git', 'cmd', 'git.exe'),
    path.win32.join(env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)', 'Git', 'cmd', 'git.exe')
  )

  if (localAppData) {
    candidates.push(path.win32.join(localAppData, 'Programs', 'Git', 'cmd', 'git.exe'))
  }

  _gitBinaryCache = candidates.find(fileExists) || findOnPath('git') || 'git'

  if (_gitBinaryCache !== 'git' && path.win32.isAbsolute(_gitBinaryCache)) {
    _gitBinaryCache = toShortPath(_gitBinaryCache)
  }

  return _gitBinaryCache
}
