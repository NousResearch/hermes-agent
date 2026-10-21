// Shared-metrics report for PACKAGED self-updates (electron-updater, App
// Installer, Store). Checkout hand-offs run `hermes update`, whose receipt the
// backend already counts, so they never reach this module.
//
// A packaged apply usually ends with the installer quitting the app, so the
// run is persisted as a pending record BEFORE apply starts and reported on the
// next backend attach; the file is deleted only once the RPC has resolved.

import fs from 'node:fs'
import path from 'node:path'

import type { UpdaterApplyResultWire } from './index'

export type UpdateRunOutcome = 'success' | 'failed' | 'noop' | 'refused'
export type UpdateFailedStage = 'download' | 'verify' | 'apply' | 'restart' | 'other'

export const PENDING_UPDATE_RUN_FILE = 'update-metric-pending.json'

const PENDING_TTL_MS = 30 * 24 * 60 * 60 * 1000

// The strategies throw plain Errors with human text; the last progress stage
// they announced is the only message-free signal of where a run died.
const FAILED_STAGE_BY_PROGRESS: Readonly<Record<string, UpdateFailedStage>> = {
  fetch: 'download',
  prepare: 'verify',
  restart: 'apply'
}

/** A run that has no `outcome` yet was handed to the installer; the next launch decides it. */
export interface PendingUpdateRun {
  v: 1
  mechanism: string
  startedAt: number
  fromVersion: string
  fromCommitDate?: number
  outcome?: UpdateRunOutcome
  failedStage?: UpdateFailedStage
  durationMs?: number
}

/** `shared_metrics.update_run` params (SharedMetricsUpdateRunParams minus the profile the renderer routes by). */
export interface UpdateRunReport {
  outcome: UpdateRunOutcome
  mechanism: string
  duration_ms: number
  failed_stage?: UpdateFailedStage
  from_commit_date?: number
}

export interface UpdateRunRecorderDeps {
  dir: () => string
  appVersion: () => string
  now?: () => number
  /** A settled record is ready to send while the backend may still be attached. */
  onRecorded?: () => void
}

export function failedStageForProgress(stage: string | null | undefined): UpdateFailedStage {
  return stage && Object.hasOwn(FAILED_STAGE_BY_PROGRESS, stage) ? FAILED_STAGE_BY_PROGRESS[stage] : 'other'
}

function settledOutcome(result: UpdaterApplyResultWire): UpdateRunOutcome | null {
  if (result.handedOff === true) {
    return null
  }

  // ok:false is the discontinued-build refusal and manual means "install it
  // yourself": neither attempted an install, so neither is a failure.
  return !result.ok || result.manual === true ? 'refused' : 'noop'
}

function isPendingRun(value: unknown): value is PendingUpdateRun {
  const run = value as PendingUpdateRun | null

  return (
    typeof run === 'object' &&
    run !== null &&
    run.v === 1 &&
    typeof run.mechanism === 'string' &&
    typeof run.startedAt === 'number' &&
    typeof run.fromVersion === 'string'
  )
}

export class UpdateRunRecorder {
  private active: { run: PendingUpdateRun; stage: string | null } | null = null
  private handedOff = false
  private inFlight = false

  constructor(private readonly deps: UpdateRunRecorderDeps) {}

  private get file(): string {
    return path.join(this.deps.dir(), PENDING_UPDATE_RUN_FILE)
  }

  private now(): number {
    return (this.deps.now ?? Date.now)()
  }

  /** Wrap a packaged apply; `mechanism` undefined (checkout) runs it unrecorded. */
  async track(
    mechanism: string | undefined,
    fromCommitDate: number | null | undefined,
    apply: () => Promise<UpdaterApplyResultWire>
  ): Promise<UpdaterApplyResultWire> {
    if (!mechanism) {
      return apply()
    }

    this.begin(mechanism, fromCommitDate)

    try {
      const result = await apply()
      this.finish(settledOutcome(result))

      return result
    } catch (error) {
      this.finish('failed')
      throw error
    }
  }

  noteProgress(stage: string): void {
    if (this.active && Object.hasOwn(FAILED_STAGE_BY_PROGRESS, stage)) {
      this.active.stage = stage
    }
  }

  private begin(mechanism: string, fromCommitDate: number | null | undefined): void {
    const run: PendingUpdateRun = {
      v: 1,
      mechanism,
      startedAt: this.now(),
      fromVersion: this.deps.appVersion(),
      ...(typeof fromCommitDate === 'number' ? { fromCommitDate } : {})
    }

    this.active = { run, stage: null }
    // Written up front: the installer may quit the app before apply() returns.
    this.write(run)
  }

  private finish(outcome: UpdateRunOutcome | null): void {
    const active = this.active
    this.active = null

    if (!active) {
      return
    }

    if (outcome === null) {
      this.handedOff = true

      return
    }

    this.write({
      ...active.run,
      outcome,
      durationMs: this.now() - active.run.startedAt,
      ...(outcome === 'failed' ? { failedStage: failedStageForProgress(active.stage) } : {})
    })
    this.deps.onRecorded?.()
  }

  private write(run: PendingUpdateRun): void {
    try {
      const tmp = `${this.file}.tmp`
      fs.mkdirSync(path.dirname(tmp), { recursive: true })
      fs.writeFileSync(tmp, JSON.stringify(run))
      fs.renameSync(tmp, this.file)
    } catch {
      // Telemetry never blocks an update.
    }
  }

  private read(): PendingUpdateRun | null {
    try {
      const parsed: unknown = JSON.parse(fs.readFileSync(this.file, 'utf8'))

      if (isPendingRun(parsed) && this.now() - parsed.startedAt <= PENDING_TTL_MS) {
        return parsed
      }

      this.remove()
    } catch {
      // Missing or unreadable: nothing to report.
    }

    return null
  }

  private remove(): void {
    try {
      fs.rmSync(this.file, { force: true })
    } catch {
      // Retried on the next take().
    }
  }

  /**
   * Claim the pending run for one send. Returns null while an apply is running
   * or has handed off in this process, and while a previous claim is unacked.
   */
  take(): UpdateRunReport | null {
    if (this.active || this.handedOff || this.inFlight) {
      return null
    }

    const run = this.read()

    if (!run) {
      return null
    }

    this.inFlight = true

    return this.report(run)
  }

  /** Delete the record only after the RPC resolved; a failed send keeps it for the next attach. */
  ack(sent: boolean): void {
    if (!this.inFlight) {
      return
    }

    if (sent) {
      this.remove()
    }

    this.inFlight = false
  }

  private report(run: PendingUpdateRun): UpdateRunReport {
    // A hand-off record is decided here: a new version means the installer finished.
    const outcome = run.outcome ?? (this.deps.appVersion() !== run.fromVersion ? 'success' : 'failed')
    const failedStage = run.outcome ? run.failedStage : 'apply'

    return {
      outcome,
      mechanism: run.mechanism,
      duration_ms: run.durationMs ?? Math.max(0, this.now() - run.startedAt),
      ...(outcome === 'failed' ? { failed_stage: failedStage ?? 'other' } : {}),
      ...(run.fromCommitDate !== undefined ? { from_commit_date: run.fromCommitDate } : {})
    }
  }
}

interface IpcHandleTarget {
  handle(channel: string, listener: (event: unknown, ...args: unknown[]) => unknown): void
}

export function registerUpdateMetricsIpc(ipc: IpcHandleTarget, recorder: UpdateRunRecorder): void {
  ipc.handle('hermes:updates:metric:take', (): UpdateRunReport | null => recorder.take())
  ipc.handle('hermes:updates:metric:ack', (_event: unknown, sent: unknown): void => recorder.ack(sent === true))
}
