// Picks a working ssh executable on Windows. The Win32 OpenSSH client under
// System32 is the usual choice, but on some machines that binary is broken —
// it dies instantly (exit 255, no output, no console) even though the file
// exists and the "OpenSSH Client" capability reports Installed (#103288).
// Rather than trusting the path, each candidate is health-checked with
// `ssh -V` and the first one that actually runs wins. Non-Windows platforms
// keep the historical bare `ssh` (PATH resolution) untouched.
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { findGitBash } from './find-git-bash'

// Per-probe ceiling, and a total budget for the whole resolution so a chain
// of hanging binaries cannot stall boot indefinitely. The final candidate
// (the Git for Windows fallback when present) always gets probed — earlier
// candidates are skipped once the budget can no longer cover them AND the
// reserved final probe.
const PROBE_TIMEOUT_MS = 4_000
const TOTAL_PROBE_BUDGET_MS = 12_000
const PROBE_KILL_GRACE_CAP_MS = 1_000

export interface SshBinaryDeps {
  platform?: NodeJS.Platform | string
  env?: Record<string, string | undefined>
  spawnFn?: any
  fileExists?: (filePath: string) => boolean
  log?: (line: string) => void
  probeTimeoutMs?: number
  budgetMs?: number
}

// Candidate paths are Windows paths regardless of host platform (unit tests
// exercise this on POSIX CI hosts too), so always join with win32 semantics —
// plain path.join would produce mixed forward-slash paths off Windows.
const joinWin = path.win32.join

// Resolve `ssh` against PATH so a bare-ssh candidate that points at the same
// binary as an earlier candidate can be deduped instead of probed twice.
function resolveSshOnPath(env: Record<string, string | undefined>, fileExists: (p: string) => boolean): string | null {
  for (const dir of String(env.Path || env.PATH || '').split(';')) {
    if (!dir.trim()) {
      continue
    }

    const candidate = joinWin(dir, 'ssh.exe')

    if (fileExists(candidate)) {
      return candidate
    }
  }

  return null
}

// Git for Windows bundles a full MSYS OpenSSH. Reuse the repo's existing Git
// discovery (HERMES_GIT_BASH_PATH, PortableGit under %LOCALAPPDATA%\hermes\git,
// ProgramFiles / ProgramFiles(x86), user-scoped install) and take the sibling
// usr/bin/ssh.exe from the same install root.
function gitSshCandidates(env: Record<string, string | undefined>, fileExists: (p: string) => boolean): string[] {
  const bash = findGitBash({ isWindows: true, env, fileExists })

  if (!bash) {
    return []
  }

  const binDir = path.win32.dirname(bash)
  const parent = path.win32.dirname(binDir)
  const root =
    path.win32.basename(binDir).toLowerCase() === 'bin'
      ? path.win32.basename(parent).toLowerCase() === 'usr'
        ? path.win32.dirname(parent) // <root>\usr\bin\bash.exe -> <root>
        : parent // <root>\bin\bash.exe -> <root>
      : binDir // custom layout: best effort, the probe decides

  return [joinWin(root, 'usr', 'bin', 'ssh.exe')]
}

// Candidate order: explicit override, the Windows inbox OpenSSH client,
// whatever PATH resolves, then the Git for Windows bundled MSYS OpenSSH.
// Dedupe is case/separator-insensitive so e.g. a PATH-resolved ssh that IS
// the System32 binary is not probed twice.
export function sshBinaryCandidates(
  env: Record<string, string | undefined>,
  fileExists: (p: string) => boolean = fs.existsSync
): string[] {
  const candidates = [
    env.HERMES_SSH_PATH,
    joinWin(env.SystemRoot || 'C:\\Windows', 'System32', 'OpenSSH', 'ssh.exe'),
    resolveSshOnPath(env, fileExists) || 'ssh',
    ...gitSshCandidates(env, fileExists)
  ]

  const seen = new Set<string>()

  return candidates.filter((c): c is string => {
    if (!c || !c.trim()) {
      return false
    }

    const key = c.replace(/\//g, '\\').toLowerCase()

    if (seen.has(key)) {
      return false
    }

    seen.add(key)

    return true
  })
}

// A candidate is healthy when `candidate -V` exits 0 and reports an OpenSSH
// version string. A binary that is missing (spawn error), hangs, or dies
// instantly with no output — the #103288 failure mode — is unhealthy. On
// timeout the child is SIGKILLed and the probe waits for the close event (or
// a short grace period) before settling, so a failed kill cannot leave a
// zombie probe running in parallel with the next candidate.
export function probeSshBinary(candidate: string, spawnFn: any, timeoutMs: number = PROBE_TIMEOUT_MS): Promise<boolean> {
  return new Promise(resolve => {
    let child

    try {
      child = spawnFn(candidate, ['-V'], { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true })
    } catch {
      resolve(false)

      return
    }

    let output = ''
    let settled = false
    let timedOut = false
    let graceTimer: any = null

    const finish = (healthy: boolean) => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)

      if (graceTimer) {
        clearTimeout(graceTimer)
      }

      resolve(healthy)
    }

    const timer = setTimeout(() => {
      timedOut = true

      try {
        child.kill('SIGKILL')
      } catch {
        // already gone
      }

      // A successful kill reaps the child and `close` settles the probe; this
      // grace timer only covers the kill having silently failed. The grace
      // never exceeds the probe timeout itself, keeping short-timeout (test)
      // probes fast.
      graceTimer = setTimeout(() => finish(false), Math.min(PROBE_KILL_GRACE_CAP_MS, timeoutMs))
      graceTimer.unref?.()
    }, timeoutMs)

    timer.unref?.()

    // `ssh -V` prints to stdout on newer OpenSSH and stderr on older builds.
    child.stdout?.on('data', d => {
      output += d.toString()
    })
    child.stderr?.on('data', d => {
      output += d.toString()
    })
    child.on('error', () => finish(false))
    child.on('close', code => finish(!timedOut && code === 0 && /openssh/i.test(output)))
  })
}

export async function resolveSshBinary({
  platform = process.platform,
  env = process.env,
  spawnFn = spawn,
  fileExists = fs.existsSync,
  log = () => {},
  probeTimeoutMs = PROBE_TIMEOUT_MS,
  budgetMs = TOTAL_PROBE_BUDGET_MS
}: SshBinaryDeps = {}): Promise<string> {
  if (platform !== 'win32') {
    return 'ssh'
  }

  const deadline = Date.now() + budgetMs
  // The final candidate (the Git for Windows fallback when one was
  // discovered) is guaranteed a full probe: earlier candidates only get
  // budget that still leaves a complete final probe in reserve. A chain of
  // hanging binaries therefore degrades to "skip the middle, still try the
  // bundled fallback" instead of never reaching it.
  const reserveMs = probeTimeoutMs + Math.min(PROBE_KILL_GRACE_CAP_MS, probeTimeoutMs)
  const candidates = sshBinaryCandidates(env, fileExists)

  for (let i = 0; i < candidates.length; i++) {
    const candidate = candidates[i]
    const isFinal = i === candidates.length - 1
    const remaining = deadline - reserveMs - Date.now()

    if (!isFinal && remaining <= 0) {
      log(`[ssh] probe budget exhausted; skipping ${candidate} to preserve the final fallback probe`)

      continue
    }

    const timeoutMs = isFinal ? probeTimeoutMs : Math.min(probeTimeoutMs, remaining)

    // An explicit HERMES_SSH_PATH override that fails the health check is
    // warned about but not fatal — the chain below may still find a working
    // binary, which is strictly better than refusing to connect.
    if (await probeSshBinary(candidate, spawnFn, timeoutMs)) {
      log(`[ssh] using ssh binary: ${candidate}`)

      return candidate
    }

    log(`[ssh] ssh candidate failed its health check: ${candidate}`)
  }

  // Nothing probed clean. Return the bare command so downstream errors read
  // exactly like they did before this fallback existed.
  log('[ssh] no working ssh binary found; falling back to bare "ssh"')

  return 'ssh'
}

// The resolved binary cannot change while the app is running, so probe once
// per process. The first caller's logger wins; later callers share the
// cached result.
let cachedSshBinary: Promise<string> | null = null

export function getSshBinary(log?: (line: string) => void): Promise<string> {
  if (!cachedSshBinary) {
    cachedSshBinary = resolveSshBinary({ log })
  }

  return cachedSshBinary
}
