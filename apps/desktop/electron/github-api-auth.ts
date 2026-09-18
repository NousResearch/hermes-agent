import { execFile as nodeExecFile } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { delimiterForPlatform, pathEnvKey, POSIX_SANE_PATH_ENTRIES } from './backend-env'

// Credentials for the self-update check's api.github.com calls.
//
// The passive check asks the REST API for the branch tip SHA. Unauthenticated
// that budget is 60 requests/hour keyed on the *client IP*, so a shared exit —
// office NAT, VPN, a proxy node many users sit behind — exhausts the bucket for
// everyone on it and the check reports a rate limit that reads as "Hermes can't
// reach the update server". Authenticating moves the caller onto the token's
// 5,000/hour budget.
//
// Credential ladder, mirroring the Python client (tools/skills_hub_github.py::GitHubAuth):
//   1. GITHUB_TOKEN, then GH_TOKEN, read from the process env per request and never stored.
//   2. `gh auth token` — the gh CLI's own login. A GUI-launched app inherits a
//      minimal environment, so this is the only rung that helps most desktop
//      users; the answer (token or "none") is cached for the process lifetime
//      so the check never spawns gh more than once.
//   3. Anonymous.
// A token GitHub rejects (401) drops the caller back to rung 3 for that request;
// the token itself never reaches a log line or an error string.

/** Env vars consulted, in precedence order. */
export const GITHUB_TOKEN_ENV_VARS = ['GITHUB_TOKEN', 'GH_TOKEN'] as const

/** `gh auth token` must answer within this; a wedged keyring prompt must not stall the check. */
export const GH_CLI_TIMEOUT_MS = 3000

export type GitHubTokenSource = 'env' | 'gh-cli'

export interface GitHubCredential {
  token: string
  source: GitHubTokenSource
}

type Env = Record<string, string | undefined>

interface GhCliOptions {
  env?: Env
  platform?: NodeJS.Platform
  execFileFn?: typeof nodeExecFile
  exists?: (filePath: string) => boolean
  timeoutMs?: number
}

/** First non-blank env token, trimmed. A blank value falls through to the next. */
export function githubTokenFromEnv(env: Env = {}): string | null {
  for (const name of GITHUB_TOKEN_ENV_VARS) {
    const value = env[name]

    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }

  return null
}

/**
 * Directories searched for the gh CLI: PATH first, then the install locations
 * a GUI launch's minimal PATH omits — the same sane POSIX entries backend-env
 * appends for the backend, plus the GitHub CLI installer's Windows targets.
 */
export function ghCliSearchDirs(env: Env = process.env, platform: NodeJS.Platform = process.platform): string[] {
  const dirs = String(env[pathEnvKey(env, platform)] || '')
    .split(delimiterForPlatform(platform))
    .filter(Boolean)

  if (platform === 'win32') {
    const programFiles = env.ProgramFiles || 'C:\\Program Files'
    const localAppData = env.LOCALAPPDATA

    dirs.push(path.win32.join(programFiles, 'GitHub CLI'))

    if (localAppData) {
      dirs.push(path.win32.join(localAppData, 'Programs', 'GitHub CLI'))
    }
  } else {
    dirs.push(...POSIX_SANE_PATH_ENTRIES, '/home/linuxbrew/.linuxbrew/bin')

    if (env.HOME) {
      dirs.push(path.posix.join(env.HOME, '.local', 'bin'), path.posix.join(env.HOME, '.nix-profile', 'bin'))
    }
  }

  return [...new Set(dirs)]
}

/** Absolute path of the gh CLI, or null when no candidate directory has one. */
export function findGhCli(
  env: Env = process.env,
  platform: NodeJS.Platform = process.platform,
  exists: (filePath: string) => boolean = fs.existsSync
): string | null {
  const pathModule = platform === 'win32' ? path.win32 : path.posix
  const executable = platform === 'win32' ? 'gh.exe' : 'gh'

  for (const dir of ghCliSearchDirs(env, platform)) {
    const candidate = pathModule.join(dir, executable)

    if (exists(candidate)) {
      return candidate
    }
  }

  return null
}

/**
 * Run `gh auth token` once. argv only (no shell), stdin closed immediately,
 * bounded by GH_CLI_TIMEOUT_MS. Missing gh, a logged-out gh (exit 1), a
 * timeout, or a spawn failure all resolve to null — the caller goes anonymous.
 */
export function readGhCliToken({
  env = process.env,
  platform = process.platform,
  execFileFn = nodeExecFile,
  exists = fs.existsSync,
  timeoutMs = GH_CLI_TIMEOUT_MS
}: GhCliOptions = {}): Promise<string | null> {
  const gh = findGhCli(env, platform, exists)

  if (!gh) {
    return Promise.resolve(null)
  }

  return new Promise(resolve => {
    try {
      const child = execFileFn(
        gh,
        ['auth', 'token'],
        { encoding: 'utf8', timeout: timeoutMs, windowsHide: true, env: env as NodeJS.ProcessEnv },
        (error, stdout) => {
          const token = typeof stdout === 'string' ? stdout.trim() : ''

          resolve(!error && token ? token : null)
        }
      )

      // gh may prompt on a TTY-less stdin when its keyring is locked; never wait on it.
      child?.stdin?.end?.()
    } catch {
      resolve(null)
    }
  })
}

let ghCliTokenPromise: Promise<string | null> | null = null

/**
 * `gh auth token`, cached for the process once it answers with a token. A "none" answer (gh missing,
 * logged out, hung) is not cached: a `gh auth login` after launch is honoured by the next check, and
 * the check is passive and hourly, so re-asking costs one bounded spawn per check at most.
 */
export function githubTokenFromGhCli(options: GhCliOptions = {}): Promise<string | null> {
  ghCliTokenPromise ??= readGhCliToken(options).then(token => {
    if (token === null) {
      ghCliTokenPromise = null
    }

    return token
  })

  return ghCliTokenPromise
}

/** Drop the cached gh answer so the next request asks gh again (rejected token, tests). */
export function forgetGhCliToken(): void {
  ghCliTokenPromise = null
}

/** Walk the ladder: env token, then the cached gh CLI login, else null (anonymous). */
export async function resolveGitHubCredential(options: GhCliOptions = {}): Promise<GitHubCredential | null> {
  const env = options.env ?? process.env
  const fromEnv = githubTokenFromEnv(env)

  if (fromEnv) {
    return { token: fromEnv, source: 'env' }
  }

  const fromGh = await githubTokenFromGhCli({ ...options, env })

  return fromGh ? { token: fromGh, source: 'gh-cli' } : null
}

/**
 * Headers for api.github.com. The `token` scheme (not `Bearer`) matches the
 * Python client. Anonymous callers get the base headers untouched: an empty
 * `Authorization` header is a 401, so it must be absent, not blank.
 */
export function githubApiHeaders(base: Record<string, string>, token?: string | null): Record<string, string> {
  const headers = { ...base }

  if (token) {
    headers.Authorization = `token ${token}`
  }

  return headers
}

/**
 * True when api.github.com rejected the credential itself (HTTP 401 on an
 * authenticated call). A stale or revoked token must not fail the update
 * check closed — the caller retries anonymously, which is exactly what worked
 * before credentials were wired in. Anonymous 401s and every other status are
 * not the token's fault and are surfaced as-is.
 */
export function githubTokenRejected(error: { statusCode?: number; authenticated?: boolean } | null | undefined): boolean {
  return error?.statusCode === 401 && error?.authenticated === true
}

/** Log-safe name of a rejected credential's source; never includes the token. */
export function describeGitHubCredentialSource(source: GitHubTokenSource): string {
  return source === 'env' ? 'the GITHUB_TOKEN / GH_TOKEN from the environment' : 'the gh CLI login (`gh auth token`)'
}
