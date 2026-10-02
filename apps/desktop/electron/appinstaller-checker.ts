/**
 * appinstaller-checker.ts
 *
 * Bounded child run for the App Installer update checker
 * (scripts/check-appinstaller-update.py). The checker runs on the
 * update-check path, so a wedged child must never hang the check: the
 * deadline resolves the promise AT the deadline with an honest unknown,
 * while execFile's own timeout performs the bounded kill. Two independent
 * bounds on purpose — a kill that the child never answers (no 'close' ever)
 * still cannot hold the check hostage.
 */

import { execFile } from 'node:child_process'

export const APPINSTALLER_CHECK_TIMEOUT_MS = 20_000

export interface AppInstallerCheckerDeps {
  env?: NodeJS.ProcessEnv
  timeoutMs?: number
  args?: readonly string[]
  /** Mutating helpers must exit before their caller can restore the backend. */
  waitForExit?: boolean
  /** Diagnostic sink for checker stderr; never breaks the result. */
  onStderr?: (text: string) => void
  execFileImpl?: ExecFileImpl
}

export interface AppInstallerCheckResult {
  code: number
  stdout: string
}

type CheckerError = Error & { code?: string | number; killed?: boolean }

/** Shape of the injected exec primitive (node's execFile in production). */
export type ExecFileImpl = (
  file: string,
  args: readonly string[],
  options: { encoding: 'utf8'; timeout: number; windowsHide: boolean; env?: NodeJS.ProcessEnv },
  callback: (error: CheckerError | null, stdout: string, stderr: string) => void
) => unknown

/**
 * Run `python <script>` with a hard deadline. Resolves (never rejects):
 * exit code and stdout map through unchanged so the checker's own JSON
 * contract survives; a deadline hit or an unspawnable interpreter resolves
 * the caller's "unknown" shape instead of a fake "no update".
 */
export function runAppInstallerChecker(
  python: string,
  script: string,
  deps: AppInstallerCheckerDeps = {}
): Promise<AppInstallerCheckResult> {
  const timeoutMs = deps.timeoutMs ?? APPINSTALLER_CHECK_TIMEOUT_MS
  const exec = deps.execFileImpl ?? execFile

  const deadlineResult = (): AppInstallerCheckResult => ({
    code: 1,
    stdout: JSON.stringify({ available: null, error: `checker timed out after ${timeoutMs}ms` })
  })

  return new Promise(resolve => {
    let settled = false

    const finish = (result: AppInstallerCheckResult): void => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(deadline)
      resolve(result)
    }

    // Reads can return unknown at the deadline. Store mutations must wait for
    // execFile's close callback after its timeout kill before recovery starts.
    const deadline = deps.waitForExit ? undefined : setTimeout(() => finish(deadlineResult()), timeoutMs)

    try {
      exec(
        python,
        [script, ...(deps.args ?? [])],
        { encoding: 'utf8', timeout: timeoutMs, windowsHide: true, env: deps.env },
        (error, stdout, stderr) => {
          if (stderr) {
            // A throwing diagnostic sink must never break the checker result.
            try {
              deps.onStderr?.(String(stderr))
            } catch {
              /* ignore */
            }
          }

          if (error?.killed) {
            finish(deadlineResult())

            return
          }

          if (error) {
            if (typeof error.code === 'number') {
              finish({ code: error.code, stdout: String(stdout || '') })
            } else {
              finish({ code: 1, stdout: JSON.stringify({ available: null, error: error.message }) })
            }

            return
          }

          finish({ code: 0, stdout: String(stdout || '') })
        }
      )
    } catch (err) {
      // execFile can throw synchronously (bad arguments); the deadline timer
      // must not be left pending — resolve the unknown shape right here.
      finish({ code: 1, stdout: JSON.stringify({ available: null, error: (err as Error).message }) })
    }
  })
}
