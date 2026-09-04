/**
 * run-git.ts
 *
 * Dependency-free wrapper around spawning `git` for the desktop update flow,
 * extracted from main.ts so it can be unit-tested in isolation.
 *
 * The bound (hard timeout) is the whole point: `runGit` used to resolve only
 * on the child's `exit` event with no timeout, so a blackholed git call —
 * TCP connect and TLS complete, but HTTP never answers — left the fetch
 * never exiting, the promise never settling, and the update check pinned on
 * "Looking for updates…" forever.
 *
 * On timeout the child is SIGTERM'd then SIGKILL'd after a short grace, the
 * timer is cleared, and the promise RESOLVES (never rejects) with a
 * `timedOut` result so the caller's existing fetch-failed path surfaces a
 * clear error instead of spinning forever.
 */

import { spawn } from 'node:child_process'

import { hiddenWindowsChildOptions } from './windows-child-options'

export interface RunGitOptions {
  cwd?: string
  env?: Record<string, string>
  onLine?: (stream: 'stdout' | 'stderr', text: string) => void
  /** Hard wall-clock bound for the whole git run, in ms. Default 15_000. */
  timeoutMs?: number
}

export interface RunGitResult {
  /** Process exit code. `null` when the run did not finish (timed out). */
  code: number | null
  stdout: string
  stderr: string
  /** True when the hard timeout fired and the child was terminated. */
  timedOut?: boolean
}

const DEFAULT_TIMEOUT_MS = 15_000
/** Grace between SIGTERM and SIGKILL so a child that ignores TERM still dies. */
const SIGKILL_GRACE_MS = 500

/**
 * Build a `runGit(args, options)` bound to a git-binary resolver. The
 * resolver is injected so the update flow keeps its Windows-specific git
 * discovery while tests can substitute a scriptable fake binary.
 */
export function createRunGit(resolveGitBinary: () => string) {
  return function runGit(
    args: string[],
    options: RunGitOptions = {}
  ): Promise<RunGitResult> {
    const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS

    return new Promise((resolve, reject) => {
      const child = spawn(
        resolveGitBinary(),
        process.platform === 'win32' ? ['-c', 'windows.appendAtomically=false', ...args] : args,
        hiddenWindowsChildOptions({
          cwd: options.cwd,
          env: { ...process.env, ...(options.env || {}), GIT_TERMINAL_PROMPT: '0' },
          stdio: ['ignore', 'pipe', 'pipe']
        })
      )

      let stdout = ''
      let stderr = ''
      let settled = false

      child.stdout.on('data', chunk => {
        const text = chunk.toString()
        stdout += text
        options.onLine?.('stdout', text)
      })
      child.stderr.on('data', chunk => {
        const text = chunk.toString()
        stderr += text
        options.onLine?.('stderr', text)
      })

      const timer = setTimeout(() => {
        if (settled) {return}
        settled = true

        // Terminate the wedged child: SIGTERM first, then SIGKILL after a
        // short grace for anything that ignores TERM.
        try {
          child.kill('SIGTERM')
        } catch {
          /* already gone */
        }

        setTimeout(() => {
          try {
            if (child.exitCode === null) {child.kill('SIGKILL')}
          } catch {
            /* already gone */
          }
        }, SIGKILL_GRACE_MS).unref()
        resolve({
          code: null,
          stdout,
          stderr: (stderr ? stderr + '\n' : '') + `git operation timed out after ${timeoutMs}ms`,
          timedOut: true
        })
      }, timeoutMs)

      // Never let the pending timer hold the process open after the child exits.
      timer.unref()

      child.once('error', err => {
        if (settled) {return}
        settled = true
        clearTimeout(timer)
        reject(err)
      })
      // Resolve on 'close' (not 'exit'): 'exit' can fire before the final
      // stdout/stderr 'data' events are delivered, losing the tail of the
      // output. 'close' fires only once the child has exited AND its stdio
      // streams have fully drained, so captured output is complete.
      child.once('close', code => {
        if (settled) {return}
        settled = true
        clearTimeout(timer)
        resolve({ code, stdout, stderr, timedOut: false })
      })
    })
  }
}
