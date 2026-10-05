/**
 * Consume the detached update hand-off's result file (#82328 follow-up).
 *
 * scripts/desktop-update/windows.ps1 runs hidden/detached — the user never sees its
 * console. It writes HERMES_HOME/.hermes-update-result.json on every exit
 * path; the relaunched Desktop reads it exactly once on boot and surfaces
 * failures (a silent failed update looks identical to "nothing happened",
 * which is how the 2026-08-09 'closed the app then nothing' report was
 * born). Read-and-delete so a result is reported at most once; ordinary
 * results older than the freshness window are discarded unread (a stale
 * file from a crashed relaunch chain must not resurface days later).
 *
 * manual:true results are exempt from the freshness window. They are the
 * durable action-required channel — on a browserless Linux box with no
 * working notifier, the boot dialog is the FIRST and ONLY place the message
 * ever surfaces, and the user may not reopen Hermes within 30 minutes.
 * Dropping it as stale strands exactly the machine it exists to serve. It is
 * still consumed once (the file is unlinked before any age check), so it
 * cannot resurface on a later boot.
 *
 * Vocabulary (C2): `ok:false` ONLY when the install is still on the previous
 * version; `ok:true` + `warnings` when the update committed but follow-up work
 * failed.
 */

import fs from 'fs'
import path from 'path'

import { RUN_ID_RE } from './update-marker-judge'

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
  /** C2: `ok:true` with follow-up work that failed after the commit point. */
  warnings: string[]
}

export function handoffResultPath(hermesHome: string): string {
  return path.join(hermesHome, '.hermes-update-result.json')
}

/** Read, parse, then consume the result file; null when absent or unparseable (a JSON `null` body is
 * still a parsed result, hence the wrapper). */
function consumeHandoffResultFile(file: string, log: (line: string) => void): { parsed: any } | null {
  let raw: string

  try {
    raw = fs.readFileSync(file, 'utf8')
  } catch {
    return null
  }

  let parsed: any

  try {
    parsed = JSON.parse(raw)
  } catch (error) {
    log(`[updates] hand-off result is not valid JSON (${(error as Error).message}); kept as ${path.basename(file)}.corrupt`)

    try {
      fs.renameSync(file, `${file}.corrupt`)
    } catch {
      void 0
    }

    return null
  }

  // Consumed once parsed — a stale or foreign result must not be re-reported
  // on every later boot.
  try {
    fs.unlinkSync(file)
  } catch {
    // Best-effort; a locked file just gets consumed on the next boot.
  }

  return { parsed }
}

/** Why a parsed result belongs to another run (or is malformed), or null when it is this run's. */
function handoffRunMismatch(
  parsed: any,
  expectedRunId: string | null,
  expectedStartedAt: number | null
): string | null {
  const runId = parsed?.run_id
  const startedAt = Number(parsed?.started_at)

  // Missing means legacy. A present but malformed ID must not downgrade to
  // weaker timestamp matching (nor be normalized into another run).
  if (runId !== undefined && (typeof runId !== 'string' || RUN_ID_RE.exec(runId)?.[0] !== runId)) {
    return '[updates] hand-off result has an invalid run_id; discarded'
  }

  if (expectedRunId !== null && runId !== undefined) {
    return runId !== expectedRunId ? `[updates] hand-off result is for run ${runId}, not ${expectedRunId}; discarded` : null
  }

  if (expectedStartedAt !== null && Number.isFinite(startedAt) && startedAt !== expectedStartedAt) {
    return `[updates] hand-off result is for the run started at ${startedAt}, not ${expectedStartedAt}; discarded`
  }

  return null
}

function toHandoffResult(parsed: any, manual: boolean): HandoffResult {
  return {
    ok: Boolean(parsed?.ok),
    exitCode: Number.isFinite(Number(parsed?.exit_code)) ? Number(parsed.exit_code) : 1,
    manual,
    message: typeof parsed?.message === 'string' ? parsed.message : '',
    branch: typeof parsed?.branch === 'string' ? parsed.branch : '',
    warnings: Array.isArray(parsed?.warnings) ? parsed.warnings.map(String).filter(Boolean) : []
  }
}

/**
 * Parse first, then consume (desktop V19): an unparseable file is renamed to
 * `.corrupt` and logged — never silently dropped — so a torn result stays
 * inspectable. Match the stable marker run ID, not line 2 (a heartbeat that
 * can change before this Desktop even opens). Only older producers without
 * run_id, or boots without an identified marker, use started_at correlation.
 */
export function readAndConsumeHandoffResult(
  hermesHome: string,
  {
    now = Date.now,
    maxAgeMs = HANDOFF_RESULT_MAX_AGE_MS,
    expectedStartedAt = null,
    expectedRunId = null,
    log = () => {}
  }: {
    now?: () => number
    maxAgeMs?: number
    expectedStartedAt?: number | null
    expectedRunId?: string | null
    log?: (line: string) => void
  } = {}
): HandoffResult | null {
  const consumed = consumeHandoffResultFile(handoffResultPath(hermesHome), log)

  if (consumed === null) {
    return null
  }

  const { parsed } = consumed

  const manual = Boolean(parsed?.manual)
  const finishedAt = Number(parsed?.finished_at)

  if (!Number.isFinite(finishedAt)) {
    log('[updates] hand-off result has no finished_at; discarded')

    return null
  }

  const mismatch = handoffRunMismatch(parsed, expectedRunId, expectedStartedAt)

  if (mismatch !== null) {
    log(mismatch)

    return null
  }

  // Ordinary results expire; a manual (action-required) result never does —
  // it's the last-resort surface for machines with no live channel, so the
  // user must see it whenever they next reopen, not only within the window.
  if (!manual && now() - finishedAt * 1000 > maxAgeMs) {
    return null
  }

  return toHandoffResult(parsed, manual)
}
