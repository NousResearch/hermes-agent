import type { IpcMain } from 'electron'

/** What main knows about the user being at this computer, pushed on `hermes:presence`. */
export interface PresenceSnapshot {
  /** The OS idle clock answered for this sample; without it the renderer falls back to its own page input. */
  system: boolean
  /** Seconds since the last input in any app; null without a system clock. */
  idleSeconds: null | number
  locked: boolean
  suspended: boolean
  /** Grows with every sample, so the renderer can drop one that arrives after a newer one. */
  seq: number
  /** Wall time (ms) of the sample, so a late delivery does not read as fresher than it is. */
  sampledAt: number
}

// Renderer timers are throttled in hidden windows, so main drives the cadence: the poll notices the
// user coming back, and a push (an IPC event still wakes a throttled renderer) lets it report.
export const PRESENCE_POLL_MS = 15_000
export const PRESENCE_ACTIVE_SECONDS = 60
export const PRESENCE_PUSH_MS = 45_000

// Electron answers `'locked'` from `getSystemIdleState`, and emits `lock-screen`/`unlock-screen`, only
// on macOS and Windows. A lock is not reported on Linux: no input reaches the window under a screen
// locker, so the page-input fallback reads as away a minute after the last input.
const LOCK_STATE_PLATFORMS = new Set<NodeJS.Platform>(['darwin', 'win32'])

/**
 * Whether the OS idle clock counts only the user's input in every app. True on macOS and Windows.
 * Not on Linux: Wayland hides it, and under X11 Hermes itself resets it after every reply (releasing
 * its keep-awake blocker, and when the completion sound ends), which would read as the user coming back.
 */
export function presenceIdleClockSupported(platform: NodeJS.Platform = process.platform): boolean {
  return platform === 'darwin' || platform === 'win32'
}

export interface PresenceWebContents {
  isDestroyed(): boolean
  send(channel: string, ...args: unknown[]): void
}

// Method syntax (bivariant) so Electron's overloaded `on`/`once` satisfy these seams and tests can
// pass tiny fakes.
export interface PresenceWindow {
  webContents: PresenceWebContents
  once(event: 'closed', listener: () => void): unknown
}

export interface PresencePowerMonitor {
  getSystemIdleTime(): number
  getSystemIdleState(idleThreshold: number): string
  on(event: 'lock-screen' | 'resume' | 'suspend' | 'unlock-screen', listener: () => void): unknown
}

export interface AttachPresenceSourceOptions {
  app: { on(event: 'browser-window-created', listener: (event: unknown, window: PresenceWindow) => void): unknown }
  ipc: Pick<IpcMain, 'handle'>
  monitor: PresencePowerMonitor
  /** Windows that already exist when this attaches. */
  windows: PresenceWindow[]
  /** Whether to sample the OS idle clock; defaults to `presenceIdleClockSupported(platform)`. */
  supported?: boolean
  platform?: NodeJS.Platform
  now?: () => number
}

/** Send one IPC event to every live window. */
export function sendToEveryWindow(
  windows: Iterable<{ webContents?: PresenceWebContents }>,
  channel: string,
  payload: unknown
) {
  for (const { webContents } of windows) {
    if (webContents && !webContents.isDestroyed()) {
      webContents.send(channel, payload)
    }
  }
}

/** The OS idle time, or null when the clock throws or answers anything but a finite time >= 0. */
function readIdleSeconds(monitor: PresencePowerMonitor): null | number {
  try {
    const seconds = monitor.getSystemIdleTime()

    return Number.isFinite(seconds) && seconds >= 0 ? seconds : null
  } catch {
    return null
  }
}

/** Whether the OS says the screen is locked; null where it cannot say. */
function readLocked(monitor: PresencePowerMonitor, platform: NodeJS.Platform): boolean | null {
  if (!LOCK_STATE_PLATFORMS.has(platform)) {
    return null
  }

  try {
    const state = monitor.getSystemIdleState(PRESENCE_ACTIVE_SECONDS)

    return state === 'locked' ? true : state === 'active' || state === 'idle' ? false : null
  } catch {
    return null
  }
}

/**
 * The main-process half of presence: answers `hermes:presence:get` and pushes `hermes:presence` on
 * lock/unlock/suspend/resume (and on a poll that sees the lock or suspend state change) whatever the
 * idle state, and whether or not an idle clock exists; where the OS idle clock is trustworthy, also
 * when the user comes back (idle drops under a minute) and every 45 seconds while they stay. The
 * idle clock alone pushes nothing while the user is idle, and nothing polls while no window exists.
 */
export function attachPresenceSource({
  app,
  ipc,
  monitor,
  windows: initialWindows,
  platform = process.platform,
  supported = presenceIdleClockSupported(platform),
  now = Date.now
}: AttachPresenceSourceOptions): void {
  const windows = new Set<PresenceWindow>()
  let locked = false
  let lockedAt = 0
  let suspended = false
  let suspendedAt = 0
  let seq = 0
  let wasActive = false
  let lastPushAt = Number.NEGATIVE_INFINITY
  let poller: ReturnType<typeof setInterval> | null = null

  // The OS may still report the old state for a moment after an event, so a sample can end a lock
  // only once a full poll interval has passed since it began; it can always start one.
  const reconcileLock = () => {
    const osLocked = readLocked(monitor, platform)

    if (osLocked && !locked) {
      locked = true
      lockedAt = now()
    } else if (osLocked === false && locked && now() - lockedAt >= PRESENCE_POLL_MS) {
      locked = false
    }
  }

  // Every read of the clock goes through here: a failing clock is "no system source" for this sample.
  const sample = (): PresenceSnapshot => {
    reconcileLock()
    const idleSeconds = supported ? readIdleSeconds(monitor) : null
    seq += 1

    return { system: idleSeconds !== null, idleSeconds, locked, suspended, seq, sampledAt: now() }
  }

  const isActive = (state: PresenceSnapshot) =>
    !state.locked && !state.suspended && (state.idleSeconds ?? Infinity) < PRESENCE_ACTIVE_SECONDS

  const push = (state = sample()) => {
    wasActive = isActive(state)
    lastPushAt = now()
    sendToEveryWindow(windows, 'hermes:presence', state)
  }

  const poll = () => {
    const before = { locked, suspended }

    // Timers do not run while the machine sleeps: a tick a full interval after `suspend` proves it
    // woke, even when `resume` never arrived.
    if (suspended && now() - suspendedAt >= PRESENCE_POLL_MS) {
      suspended = false
    }

    const state = sample()
    const active = isActive(state)
    const changed = state.locked !== before.locked || state.suspended !== before.suspended

    if (changed || (active && (!wasActive || now() - lastPushAt >= PRESENCE_PUSH_MS))) {
      push(state)
    } else {
      wasActive = active
    }
  }

  const track = (window: PresenceWindow) => {
    windows.add(window)
    window.once('closed', () => {
      windows.delete(window)

      if (windows.size === 0 && poller) {
        clearInterval(poller)
        poller = null
      }
    })
    poller ??= setInterval(poll, PRESENCE_POLL_MS)
  }

  const flag = (apply: () => void) => () => {
    apply()
    push()
  }

  initialWindows.forEach(track)
  app.on('browser-window-created', (_event, window) => track(window))
  monitor.on(
    'lock-screen',
    flag(() => {
      locked = true
      lockedAt = now()
    })
  )
  monitor.on(
    'unlock-screen',
    flag(() => (locked = false))
  )
  monitor.on(
    'suspend',
    flag(() => {
      suspended = true
      suspendedAt = now()
    })
  )
  monitor.on(
    'resume',
    flag(() => (suspended = false))
  )
  ipc.handle('hermes:presence:get', sample)
}
