/**
 * Consume the detached update hand-off's result file (#82328 follow-up).
 *
 * scripts/desktop-update/windows.ps1 runs hidden/detached — the user never sees its
 * console. It writes HERMES_HOME/.hermes-update-result.json on every exit
 * path; the relaunched Desktop reads it exactly once on boot and surfaces
 * failures (a silent failed update looks identical to "nothing happened",
 * which is how the 2026-08-09 'closed the app then nothing' report was
 * born). Read-and-delete so a result is reported at most once; results
 * older than the freshness window are discarded unread (a stale file from a
 * crashed relaunch chain must not resurface days later).
 */

import fs from 'fs'
import path from 'path'

export const HANDOFF_RESULT_MAX_AGE_MS = 30 * 60 * 1000

export interface HandoffResult {
  ok: boolean
  exitCode: number
  /** Update succeeded but the user must act (reopen the app, reinstall the
   * GUI package, fix the sandbox helper). The consumer must SURFACE these —
   * an ok:true manual result that only gets logged never reaches the user
   * on exactly the machines where no shim/notifier could show it live. */
  manual: boolean
  message: string
  branch: string
}

export function handoffResultPath(hermesHome: string): string {
  return path.join(hermesHome, '.hermes-update-result.json')
}

export function readAndConsumeHandoffResult(
  hermesHome: string,
  { now = Date.now, maxAgeMs = HANDOFF_RESULT_MAX_AGE_MS }: { now?: () => number; maxAgeMs?: number } = {}
): HandoffResult | null {
  const file = handoffResultPath(hermesHome)
  let raw: string

  try {
    raw = fs.readFileSync(file, 'utf8')
  } catch {
    return null
  }

  // Consume unconditionally — even a malformed/stale file must not be
  // re-reported on every subsequent boot.
  try {
    fs.unlinkSync(file)
  } catch {
    // Best-effort; a locked file just gets consumed on the next boot.
  }

  let parsed: any

  try {
    parsed = JSON.parse(raw)
  } catch {
    return null
  }

  const finishedAt = Number(parsed?.finished_at)

  if (!Number.isFinite(finishedAt) || now() - finishedAt * 1000 > maxAgeMs) {
    return null
  }

  return {
    ok: Boolean(parsed?.ok),
    exitCode: Number.isFinite(Number(parsed?.exit_code)) ? Number(parsed.exit_code) : 1,
    manual: Boolean(parsed?.manual),
    message: typeof parsed?.message === 'string' ? parsed.message : '',
    branch: typeof parsed?.branch === 'string' ? parsed.branch : ''
  }
}
