import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

export const APPLE_GIT_STUB = '/usr/bin/git'
export const XCODE_SELECT_PATH = '/usr/bin/xcode-select'
export const XCODE_SELECT_TIMEOUT_MS = 2_000
export const NEGATIVE_CACHE_TTL_MS = 5_000

export interface CltGitCache {
  check: (probe: () => boolean, opts?: { nowMs?: () => number }) => boolean
}

export interface MacGitCheckOptions {
  platform?: NodeJS.Platform
  env?: NodeJS.ProcessEnv
  appleStubPath?: string
  developerDirProbe?: (env: NodeJS.ProcessEnv) => string
  cache?: CltGitCache
  nowMs?: () => number
  realpath?: (filePath: string) => string
  isExecutableFile?: (filePath: string) => boolean
}

export interface PosixGitResolveOptions extends MacGitCheckOptions {
  candidates: string[]
  // Tried after `candidates`. A Finder/Dock launch whose login-shell PATH merge failed sees only
  // /usr/bin:/bin:/usr/sbin:/sbin, while gh's env and the backend's PATH still reach Homebrew git.
  fallbackDirs?: readonly string[]
  fileExists: (filePath: string) => boolean
  binaryRuns: (filePath: string) => boolean
}

export function createCltGitCache(negativeTtlMs: number = NEGATIVE_CACHE_TTL_MS): CltGitCache {
  let usable: boolean | null = null
  let expiresAt: number | null = null

  const readCached = (now: number): boolean | null => {
    if (usable === null) {
      return null
    }

    if (expiresAt === null || now < expiresAt) {
      return usable
    }

    usable = null
    expiresAt = null

    return null
  }

  return {
    check(probe, opts = {}) {
      const nowMs = opts.nowMs ?? Date.now
      const hit = readCached(nowMs())

      if (hit !== null) {
        return hit
      }

      const result = Boolean(probe())

      usable = result
      expiresAt = result ? null : nowMs() + negativeTtlMs

      return result
    }
  }
}

const defaultCltCache = createCltGitCache()

function defaultRealpath(filePath: string): string {
  try {
    return fs.realpathSync(filePath)
  } catch {
    return path.resolve(filePath)
  }
}

function defaultIsExecutableFile(filePath: string): boolean {
  try {
    if (!fs.statSync(filePath).isFile()) {
      return false
    }

    fs.accessSync(filePath, fs.constants.X_OK)

    return true
  } catch {
    return false
  }
}

export function hasDeveloperGit(
  devDir: string,
  isExecutableFile: (filePath: string) => boolean = defaultIsExecutableFile
): boolean {
  const trimmed = devDir.trim()

  if (!trimmed) {
    return false
  }

  // xcode-select(1): DEVELOPER_DIR may name either the Developer dir or Xcode.app.
  for (const root of [trimmed, path.join(trimmed, 'Contents', 'Developer')]) {
    if (isExecutableFile(path.join(root, 'usr', 'bin', 'git'))) {
      return true
    }
  }

  return false
}

export function probeXcodeSelectDeveloperDir(
  env: NodeJS.ProcessEnv = process.env,
  opts: { xcodeSelect?: string; timeoutMs?: number } = {}
): string {
  try {
    const out = execFileSync(opts.xcodeSelect ?? XCODE_SELECT_PATH, ['-p'], {
      encoding: 'utf8',
      env,
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout: opts.timeoutMs ?? XCODE_SELECT_TIMEOUT_MS
    })

    return String(out || '').trim()
  } catch {
    return ''
  }
}

export function isUsableMacGitCandidate(candidate: string, opts: MacGitCheckOptions = {}): boolean {
  if (!candidate) {
    return false
  }

  const platform = opts.platform ?? process.platform

  if (platform !== 'darwin') {
    return true
  }

  const realpath = opts.realpath ?? defaultRealpath
  const appleStubPath = opts.appleStubPath ?? APPLE_GIT_STUB

  if (realpath(candidate) !== realpath(appleStubPath)) {
    return true
  }

  const env = opts.env ?? process.env
  const isExecutableFile = opts.isExecutableFile ?? defaultIsExecutableFile
  const devDir = String(env.DEVELOPER_DIR || '').trim()

  if (devDir) {
    return hasDeveloperGit(devDir, isExecutableFile)
  }

  const probe = opts.developerDirProbe ?? (probeEnv => probeXcodeSelectDeveloperDir(probeEnv))
  const cache = opts.cache ?? defaultCltCache

  return cache.check(() => hasDeveloperGit(probe(env), isExecutableFile), { nowMs: opts.nowMs })
}

export function resolvePosixGitBinary(opts: PosixGitResolveOptions): string | null {
  const fallbacks = (opts.fallbackDirs ?? []).map(dir => path.join(dir, 'git'))
  const existing = [...new Set([...opts.candidates, ...fallbacks])].filter(opts.fileExists)
  let firstNonStub: string | null = null
  let blockedCltStub = false

  for (const candidate of existing) {
    if (!isUsableMacGitCandidate(candidate, opts)) {
      blockedCltStub = true

      continue
    }

    firstNonStub ??= candidate

    if (opts.binaryRuns(candidate)) {
      return candidate
    }
  }

  if (firstNonStub) {
    return firstNonStub
  }

  if (blockedCltStub) {
    return null
  }

  const platform = opts.platform ?? process.platform
  const env = opts.env ?? process.env
  const appleStubPath = opts.appleStubPath ?? APPLE_GIT_STUB

  // POSIX execvp falls back to /usr/bin:/bin when PATH is unset, which reaches the stub.
  if (
    platform === 'darwin' &&
    env.PATH === undefined &&
    opts.fileExists(appleStubPath) &&
    !isUsableMacGitCandidate(appleStubPath, opts)
  ) {
    return null
  }

  return 'git'
}

export function runOptionalGit<TResult extends { code: number | null; stdout: string; stderr: string }, TOptions>(
  gitBin: string | null,
  args: string[],
  options: TOptions,
  runGit: (git: string, args: string[], options: TOptions) => Promise<TResult>
): Promise<TResult | { code: number; stdout: string; stderr: string }> {
  return gitBin
    ? runGit(gitBin, args, options)
    : Promise.resolve({ code: 1, stdout: '', stderr: 'git is not available' })
}
