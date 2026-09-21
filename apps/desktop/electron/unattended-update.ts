/**
 * Unattended (scheduled) Windows Desktop update — pure, dependency-free core.
 *
 * A LOCAL-OPT-IN schedule: the user enables it in the Desktop's own Settings,
 * picks a local wall-clock time, and Hermes may run its OWN updater at that
 * time unattended. It is OFF by default and never turns itself on.
 *
 * THE CONTRACT (what this module guarantees and what it deliberately does NOT):
 *
 *   * Default OFF  — `parseUnattendedSchedule` fails CLOSED: anything that
 *     isn't an unambiguous `{ enabled: true, hour: 0-23, minute: 0-59 }`
 *     parses to disabled (including garbage input and missing config).
 *   * Local only   — the schedule is stored in the Desktop's own userData
 *     JSON (`updates.json`), read/written by the Electron main process. It is
 *     never stored on a remote backend or in any config a remote agent could
 *     reach, and it is only configurable through the local Desktop Settings
 *     IPC bridge (`hermes:updates:schedule:*`). Nothing here reads git.
 *   * Own updater only — this module only ever decides WHEN to attempt an
 *     update; it never executes anything. The main-process driver feeds the
 *     decision into Hermes' existing `applyUpdates()` hand-off, which spawns
 *     only the repo-owned `scripts/desktop-update/windows.ps1` (PowerShell)
 *     or the staged signed `hermes-setup.exe`. There is no raw `git` call and
 *     no OS shell beyond that narrow updater path.
 *   * Refuses when unsafe — `decideScheduledAttempt` returns `not-due` /
 *     `disabled` / `not-windows` / `recently-ran` without ever suggesting a
 *     run; the driver separately refuses when a live update lock owns the
 *     tree (`updateHandoffConflict`). An unattended update never mutates a
 *     checkout another updater already owns.
 *   * Relaunch — the hand-off script rebuilds and relaunches the Desktop and
 *     writes `.hermes-update-result.json`; the Desktop itself does no
 *     relaunch work and this module none at all.
 *   * Least privilege — everything runs as the same unprivileged user that
 *     launched the Desktop; there is no elevation, no Task Scheduler, and no
 *     service registration.
 *
 * Scheduling model (deliberately minimal and safe): the timer is armed for the
 * NEXT occurrence of the chosen local time. If the app is running then, it
 * gates a single attempt and then re-arms for the next day. If the app is
 * closed at that instant, that night's update is simply skipped — an
 * unattended updater must never surprise a user who quit the app. A bounded
 * GRACE WINDOW after the target time lets a Desktop that launched just after
 * the minute still catch its slot (see `isWithinScheduledWindow`), and the
 * `cooldownMs` guard in `decideScheduledAttempt` prevents more than one attempt
 * ever landing inside a single window.
 */

export type ScheduledAttemptReason =
  | 'enabled-and-due'
  | 'disabled'
  | 'not-due'
  | 'not-windows'
  | 'recently-ran'

export interface ScheduledAttemptDecision {
  shouldRun: boolean
  reason: ScheduledAttemptReason
}

/** The persisted, locally-stored schedule. `enabled:false` is the default. */
export interface UnattendedSchedule {
  enabled: boolean
  /** Local wall-clock hour, 0..23. */
  hour: number
  /** Local wall-clock minute, 0..59. */
  minute: number
}

export interface LocalTime {
  hour: number
  minute: number
}

export const DEFAULT_UNATTENDED_SCHEDULE: UnattendedSchedule = {
  enabled: false,
  hour: 3,
  minute: 0
}

/** Clamp a time to the valid integer range, or fall back to the default. */
function clampInt(raw: unknown, lower: number, upper: number, fallback: number): number {
  const n = typeof raw === 'number' ? raw : Number.parseFloat(String(raw))

  if (!Number.isFinite(n)) {
    return fallback
  }

  const clamped = Math.floor(n)

  return clamped >= lower && clamped <= upper ? clamped : fallback
}

/**
 * Parse untrusted persisted/remote-presented schedule input into a valid,
 * FAIL-CLOSED schedule. Any malformed, partial, or non-`{enabled:true}` value
 * decodes to disabled so a corrupt config can never arm an update.
 */
export function parseUnattendedSchedule(raw: unknown): UnattendedSchedule {
  if (!raw || typeof raw !== 'object') {
    return { ...DEFAULT_UNATTENDED_SCHEDULE }
  }

  const record = raw as Record<string, unknown>
  const enabled = record.enabled === true

  // Only `=== true` enables; `1`, `"true"`, `"yes"` and truthy garbage do NOT.
  if (!enabled) {
    return { ...DEFAULT_UNATTENDED_SCHEDULE }
  }

  return {
    enabled: true,
    hour: clampInt(record.hour, 0, 23, DEFAULT_UNATTENDED_SCHEDULE.hour),
    minute: clampInt(record.minute, 0, 59, DEFAULT_UNATTENDED_SCHEDULE.minute)
  }
}

/** Hour + minute of `date` in the LOCAL timezone (matches `setHours`/`now`). */
export function localTimeOf(date: Date): LocalTime {
  return { hour: date.getHours(), minute: date.getMinutes() }
}

/** Whole minutes elapsed since local midnight. */
function minutesSinceMidnight(date: Date): number {
  return date.getHours() * 60 + date.getMinutes()
}

/**
 * Milliseconds until the next occurrence of `hour:minute` in LOCAL time.
 * Positive-only: if today's slot has passed, rolls forward to tomorrow.
 * Returns 0 when `now` is exactly at the target minute with zero seconds.
 */
export function msUntilNext(date: Date, schedule: UnattendedSchedule): number {
  const target = schedule.hour * 60 + schedule.minute
  let deltaMin = target - minutesSinceMidnight(date)

  // Subtract the current second+ms so the timer lands as close to the minute
  // boundary as possible rather than a full minute late.
  const subSecMs = date.getSeconds() * 1000 + date.getMilliseconds()

  // Roll forward to tomorrow only once today's target has genuinely passed:
  // strictly before => a positive delay to today; exactly at (0s,0ms) => 0
  // (it is due right now); a second past (or later) today => roll to tomorrow.
  if (deltaMin < 0 || (deltaMin === 0 && subSecMs !== 0)) {
    deltaMin += 24 * 60
  }

  return deltaMin * 60_000 - subSecMs
}

/** Milliseconds since the local midnight boundary. */
function msSinceMidnight(date: Date): number {
  return (
    date.getHours() * 3_600_000 +
    date.getMinutes() * 60_000 +
    date.getSeconds() * 1000 +
    date.getMilliseconds()
  )
}

/**
 * True when `date` is inside a GRACE WINDOW around the scheduled local time.
 *
 * The window is centered on the target minute so a scheduled moment of 03:00
 * with `graceMs = 300_000` accepts anything from 02:57:30 to 03:02:30. This
 * catches a Desktop that launched marginally late while keeping the net tight
 * enough that an unattended update can't drift far from the user's chosen time.
 */
export function isWithinScheduledWindow(
  date: Date,
  schedule: UnattendedSchedule,
  graceMs = 300_000
): boolean {
  if (!schedule.enabled) {
    return false
  }

  const target = schedule.hour * 3_600_000 + schedule.minute * 60_000
  const now = msSinceMidnight(date)
  let diff = now - target
  diff = ((diff % 86_400_000) + 86_400_000) % 86_400_000

  return diff <= graceMs || 86_400_000 - diff <= graceMs
}

export interface DecideScheduledAttemptOptions {
  schedule: UnattendedSchedule
  now: Date
  /** Unix ms of the last successful unattended attempt, or null if never. */
  lastRunAtMs: number | null
  /** Skip if a run landed within this many ms ago. */
  cooldownMs: number
  /** Passing false skips (the feature is Windows-only by policy). */
  isWindows: boolean
  /** Grace window around the target local time. */
  graceMs?: number
}

/**
 * Pure decision: should the unattended updater attempt a run RIGHT NOW?
 *
 * Every refusal is non-destructive and re-arms the timer for the next day;
 * nothing here spawns or mutates anything. `enabled-and-due` is the ONLY
 * shouldRun value, so a caller that mis-wires the refusal cases still can't
 * trigger an update.
 */
export function decideScheduledAttempt(options: DecideScheduledAttemptOptions): ScheduledAttemptDecision {
  const { schedule, now, lastRunAtMs, cooldownMs, isWindows, graceMs = 300_000 } = options

  if (!schedule.enabled) {
    return { shouldRun: false, reason: 'disabled' }
  }

  if (!isWindows) {
    return { shouldRun: false, reason: 'not-windows' }
  }

  if (!isWithinScheduledWindow(now, schedule, graceMs)) {
    return { shouldRun: false, reason: 'not-due' }
  }

  if (lastRunAtMs !== null && Number.isFinite(lastRunAtMs) && now.getTime() - lastRunAtMs < cooldownMs) {
    return { shouldRun: false, reason: 'recently-ran' }
  }

  return { shouldRun: true, reason: 'enabled-and-due' }
}

export interface ArmScheduledUpdateOptions {
  schedule: UnattendedSchedule
  /** Injectable clock for tests. */
  now?: () => Date
  setTimeoutFn?: (callback: () => void, ms: number) => unknown
  clearTimeoutFn?: (timer: unknown) => void
  /** Fired (once) when the timer lands on a scheduled slot. Re-arms after. */
  onDue: () => void
}

export interface ArmScheduledUpdateHandle {
  /** Milliseconds until the next scheduled fire (informational). */
  delayMs: number
  /** Drop the timer; safe to call more than once. */
  cancel: () => void
}

/**
 * Drive a UI scheduler timer: arm for the NEXT local occurrence, fire
 * `onDue`, then re-arm for the day after. If `schedule.enabled` is false this
 * is a no-op returning an already-cancelled handle. The caller's `onDue` is
 * responsible for applying `decideScheduledAttempt` + safety gates — the
 * re-arm happens regardless so a refused (conflict/skip) fire still tries
 * again tomorrow.
 */
export function armScheduledUpdate(options: ArmScheduledUpdateOptions): ArmScheduledUpdateHandle {
  const schedule = parseUnattendedSchedule(options.schedule)
  const now = options.now ?? (() => new Date())
  const setTimeoutFn = options.setTimeoutFn ?? setTimeout
  const clearTimeoutFn = options.clearTimeoutFn ?? ((_timer: unknown) => void 0)
  let timer: unknown = null
  let disposed = false

  const arm = () => {
    if (disposed || !schedule.enabled) {
      return
    }

    const delay = msUntilNext(now(), schedule)
    timer = setTimeoutFn(() => {
      options.onDue()
      arm()
    }, delay)
  }

  arm()

  return {
    get delayMs(): number {
      return schedule.enabled ? msUntilNext(now(), schedule) : 0
    },
    cancel() {
      disposed = true

      if (timer !== null) {
        clearTimeoutFn(timer)
        timer = null
      }
    }
  }
}