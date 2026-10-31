// May an in-app update hand off right now without killing agent work?
//
// The hand-off quits the app and runs `hermes update`, which restarts the
// local backend with `agent.restart_drain_timeout: 0` — every running chat
// turn, cron run and background delegation in that process is interrupted.
// So before the hand-off the main process merges two views of "busy":
//
// - the renderers' active-work reports (titled chats this app is streaming);
// - each local backend's `/api/health/idle` proof (session table, cron ledger,
//   delegations, prompts waiting on a human). Unreadable = indeterminate =
//   busy: a backend we cannot read is never assumed idle.
//
// Busy without consent refuses the hand-off and names what is running. The
// user then picks "Update now" (explicit consent to interrupt) or "Update when
// idle" (UpdateWhenIdle below). Pure and Electron-free so it is testable.

import type { ActiveWork } from './quit-guard'

/** One named thing an update would interrupt. */
export type UpdateBusyReason =
  | { kind: 'chats'; count: number; titles: string[] }
  | { kind: 'cron'; jobs: string[] }
  | { kind: 'delegation' }
  | { kind: 'input' }
  | { kind: 'unknown' }

export interface UpdateReadiness {
  ready: boolean
  reasons: UpdateBusyReason[]
}

/** `/api/health/idle` body. `idle: null` means the backend could not prove idle. */
export interface IdleProofWire {
  idle?: boolean | null
  reason?: string | null
  detail?: string | null
}

/** A local backend's answer: its idle proof, or `null` when the probe itself failed. */
export type BackendIdleAnswer = IdleProofWire | null

function idsAfter(prefix: string, detail: string): string[] {
  return detail
    .slice(prefix.length)
    .split(',')
    .map(id => id.trim())
    .filter(Boolean)
}

interface BackendBusy {
  sessions: string[]
  reason: UpdateBusyReason | null
}

// `detail` is busy_ledger()'s name for the first ledger holding work.
const LEDGER_REASONS: ReadonlyArray<[string, (detail: string) => BackendBusy]> = [
  ['session:', detail => ({ sessions: idsAfter('session:', detail), reason: null })],
  ['cron:', detail => ({ sessions: [], reason: { kind: 'cron', jobs: idsAfter('cron:', detail) } })],
  ['delegation', () => ({ sessions: [], reason: { kind: 'delegation' } })]
]

function backendBusy(answer: BackendIdleAnswer): BackendBusy | null {
  if (answer?.idle === true) {
    return null
  }

  if (answer?.idle === false && answer.reason === 'awaiting_human_input') {
    return { sessions: [], reason: { kind: 'input' } }
  }

  const detail = answer?.idle === false && typeof answer.detail === 'string' ? answer.detail : ''
  const match = LEDGER_REASONS.find(([prefix]) => detail.startsWith(prefix))

  // A turn with no readable ledger name (older backend, retirement admission)
  // and every indeterminate answer: something may be running, we cannot say what.
  return match ? match[1](detail) : { sessions: [], reason: { kind: 'unknown' } }
}

function reasonKey(reason: UpdateBusyReason): string {
  return reason.kind === 'cron' ? `cron:${reason.jobs.join(',')}` : reason.kind
}

/**
 * Merge the renderers' report with every local backend's idle proof.
 *
 * Chats are counted once: a turn this app streams shows up both as a renderer
 * title and as a backend `session:` id, so the count is the larger of the two
 * (another client's turn on the same backend still counts).
 */
export function mergeUpdateReadiness(work: ActiveWork, answers: readonly BackendIdleAnswer[]): UpdateReadiness {
  const sessions = new Set<string>()
  const reasons = new Map<string, UpdateBusyReason>()

  for (const answer of answers) {
    const busy = backendBusy(answer)

    busy?.sessions.forEach(id => sessions.add(id))

    if (busy?.reason) {
      reasons.set(reasonKey(busy.reason), busy.reason)
    }
  }

  const chatCount = Math.max(work.count, sessions.size)
  const ordered: UpdateBusyReason[] = []

  if (chatCount > 0) {
    ordered.push({ kind: 'chats', count: chatCount, titles: work.titles.slice(0, chatCount) })
  }

  ordered.push(...reasons.values())

  return { ready: ordered.length === 0, reasons: ordered }
}

/** One log line naming what holds the update. */
export function describeUpdateBusy(reasons: readonly UpdateBusyReason[]): string {
  return reasons
    .map(reason => {
      switch (reason.kind) {
        case 'chats':
          return `${reason.count} chat(s) working`

        case 'cron':
          return `cron ${reason.jobs.join(',') || '?'}`

        case 'delegation':
          return 'background delegation'

        case 'input':
          return 'prompt awaiting the user'

        default:
          return 'backend could not prove idle'
      }
    })
    .join('; ')
}

export type WhenIdleOutcome<T> =
  { kind: 'fired'; result: T } | { kind: 'cancelled' } | { kind: 'timed-out'; reasons: UpdateBusyReason[] }

export interface WhenIdleDeps<T> {
  read: () => Promise<UpdateReadiness>
  /**
   * Run the hand-off. It re-checks readiness itself right before tearing
   * anything down and answers `busy` (with reasons) when work started in the
   * gap; the wait then simply continues. `stillWanted` turns false once the
   * wait is cancelled: the hand-off checks it after its own backend requests,
   * before anything is torn down.
   */
  fire: (stillWanted: () => boolean) => Promise<{ busy: UpdateBusyReason[] } | { result: T }>
  publish: (reasons: UpdateBusyReason[], deadlineAt: number) => void
  sleep: (ms: number) => Promise<void>
  now: () => number
  pollMs: number
  maxWaitMs: number
  /** Consecutive idle reads required before firing (a 1 s gap between two turns is not idle). */
  settleReads?: number
}

/**
 * "Update when idle": poll readiness until it holds for `settleReads`
 * consecutive reads, then fire. Bounded by `maxWaitMs`; cancellable at any
 * await. Cron runs are deliberately NOT held while waiting — a job that fires
 * in the gap becomes a named reason on the next read, and the final freeze
 * happens inside `fire` (the backend's admission fence), where it is atomic.
 */
export class UpdateWhenIdle<T> {
  private generation = 0
  private running = false

  get waiting(): boolean {
    return this.running
  }

  /** Stop the current wait. True when one was running. */
  cancel(): boolean {
    const was = this.running
    this.generation += 1
    this.running = false

    return was
  }

  async run(deps: WhenIdleDeps<T>): Promise<WhenIdleOutcome<T>> {
    if (this.running) {
      throw new Error('An update is already waiting for Hermes to be idle.')
    }

    this.running = true
    const generation = ++this.generation
    const live = () => this.generation === generation
    const deadlineAt = deps.now() + deps.maxWaitMs
    const settleReads = Math.max(1, deps.settleReads ?? 2)
    let idleReads = 0
    let lastReasons: UpdateBusyReason[] = []

    try {
      while (live()) {
        const readiness = await deps.read()

        if (!live()) {
          return { kind: 'cancelled' }
        }

        idleReads = readiness.ready ? idleReads + 1 : 0
        lastReasons = readiness.reasons.length ? readiness.reasons : lastReasons

        // A read that lands past the bound (a laptop that slept mid-wait) is too late to act on.
        if (deps.now() >= deadlineAt) {
          return { kind: 'timed-out', reasons: lastReasons }
        }

        if (idleReads >= settleReads) {
          const fired = await deps.fire(live)

          if ('result' in fired) {
            return { kind: 'fired', result: fired.result }
          }

          // Work started between the last read and the freeze: keep waiting.
          idleReads = 0
          lastReasons = fired.busy
        }

        if (!live()) {
          return { kind: 'cancelled' }
        }

        if (deps.now() >= deadlineAt) {
          return { kind: 'timed-out', reasons: lastReasons }
        }

        deps.publish(readiness.ready ? [] : readiness.reasons, deadlineAt)
        await deps.sleep(Math.min(deps.pollMs, Math.max(0, deadlineAt - deps.now())))
      }

      return { kind: 'cancelled' }
    } finally {
      if (live()) {
        this.running = false
      }
    }
  }
}
