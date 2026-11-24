import { EventEmitter } from 'node:events'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { attachPresenceSource, PRESENCE_POLL_MS, type PresenceSnapshot } from './presence'

class FakeWindow extends EventEmitter {
  readonly sent: PresenceSnapshot[] = []
  readonly webContents = {
    isDestroyed: () => false,
    send: (channel: string, payload: PresenceSnapshot) => {
      if (channel === 'hermes:presence') {
        this.sent.push(payload)
      }
    }
  }
}

interface HarnessOptions {
  /** `'platform'` leaves the clock choice to the source's own default for `platform`. */
  supported?: boolean | 'platform'
  windows?: FakeWindow[]
  platform?: NodeJS.Platform
  /** Replaces the idle clock (a throwing or nonsense clock). */
  clock?: () => number
}

function harness({ supported = true, windows = [new FakeWindow()], platform = 'linux', clock }: HarnessOptions = {}) {
  const power = new EventEmitter()
  const app = new EventEmitter()
  const handlers = new Map<string, () => unknown>()
  let idleSeconds = 0
  let idleState = 'active'
  const getSystemIdleTime = vi.fn(clock ?? (() => idleSeconds))

  attachPresenceSource({
    app,
    ipc: { handle: (channel: string, handler: () => unknown) => void handlers.set(channel, handler) } as never,
    monitor: { getSystemIdleState: () => idleState, getSystemIdleTime, on: (event, listener) => power.on(event, listener) },
    platform,
    supported: supported === 'platform' ? undefined : supported,
    windows
  })

  return {
    app,
    getSystemIdleTime,
    power,
    get: () => handlers.get('hermes:presence:get')?.() as PresenceSnapshot,
    // Idle time follows the clock until the user touches the machine.
    idleFor: (seconds: number) => void (idleSeconds = seconds),
    osState: (state: string) => void (idleState = state),
    tick: (seconds: number, { input = false } = {}) => {
      for (let elapsed = 0; elapsed < seconds; elapsed += PRESENCE_POLL_MS / 1000) {
        idleSeconds = input ? 0 : idleSeconds + PRESENCE_POLL_MS / 1000
        vi.advanceTimersByTime(PRESENCE_POLL_MS)
      }
    }
  }
}

const flags = (sent: PresenceSnapshot[]) => sent.map(({ locked, suspended, system }) => ({ locked, suspended, system }))

beforeEach(() => void vi.useFakeTimers())
afterEach(() => void vi.useRealTimers())

describe('attachPresenceSource', () => {
  it('stays silent while idle time climbs', () => {
    const window = new FakeWindow()
    const presence = harness({ windows: [window] })

    presence.idleFor(120)
    presence.tick(300)

    expect(window.sent).toEqual([])
    expect(presence.get()).toMatchObject({ system: true, idleSeconds: 420, locked: false, suspended: false })
  })

  it('pushes once when the user comes back, then once per 45 seconds while they stay', () => {
    const window = new FakeWindow()
    const presence = harness({ windows: [window] })

    presence.idleFor(120)
    presence.tick(30)
    presence.tick(15, { input: true })
    expect(window.sent).toEqual([expect.objectContaining({ system: true, idleSeconds: 0, locked: false, suspended: false })])

    presence.tick(30, { input: true })
    expect(window.sent).toHaveLength(1)
    presence.tick(15, { input: true })
    expect(window.sent).toHaveLength(2)
    presence.tick(90, { input: true })
    expect(window.sent).toHaveLength(4)

    // Walking away stops the pushes once the last input is a minute old.
    presence.tick(300)
    expect(window.sent).toHaveLength(5)
  })

  it('pushes lock and suspend at once with their flags, and clears them on unlock and resume', () => {
    const window = new FakeWindow()
    const presence = harness({ windows: [window] })

    presence.idleFor(5)
    presence.power.emit('lock-screen')
    presence.power.emit('suspend')
    presence.power.emit('resume')
    presence.power.emit('unlock-screen')

    expect(window.sent.map(({ locked, suspended }) => ({ locked, suspended }))).toEqual([
      { locked: true, suspended: false },
      { locked: true, suspended: true },
      { locked: true, suspended: false },
      { locked: false, suspended: false }
    ])

    // Locked time is not presence: nothing is pushed until it is unlocked.
    presence.power.emit('lock-screen')
    presence.tick(120, { input: true })
    expect(window.sent).toHaveLength(5)
    expect(presence.get().locked).toBe(true)
  })

  it('stamps every sample with a growing sequence number and the time it was taken', () => {
    const window = new FakeWindow()
    const presence = harness({ windows: [window] })

    const first = presence.get()
    vi.advanceTimersByTime(1000)
    presence.power.emit('lock-screen')
    const after = presence.get()

    expect(window.sent[0]!.seq).toBeGreaterThan(first.seq)
    expect(after.seq).toBeGreaterThan(window.sent[0]!.seq)
    expect(window.sent[0]!.sampledAt).toBe(Date.now())
    expect(first.sampledAt).toBe(Date.now() - 1000)
  })

  it('still pushes lock and suspend without a trustworthy idle clock, and never reads it', () => {
    const window = new FakeWindow()
    const presence = harness({ supported: false, windows: [window] })

    presence.power.emit('lock-screen')
    presence.power.emit('suspend')
    presence.tick(120, { input: true })

    expect(flags(window.sent).slice(0, 2)).toEqual([
      { locked: true, suspended: false, system: false },
      { locked: true, suspended: true, system: false }
    ])
    expect(presence.get()).toMatchObject({ system: false, idleSeconds: null, locked: true })
    expect(presence.getSystemIdleTime).not.toHaveBeenCalled()
  })

  it('keeps a throwing idle clock inside main, as no system source, while lock still works', () => {
    const window = new FakeWindow()

    const presence = harness({
      clock: () => {
        throw new Error('clock unavailable')
      },
      windows: [window]
    })

    expect(() => presence.tick(15)).not.toThrow()
    expect(presence.get()).toMatchObject({ system: false, idleSeconds: null })

    presence.power.emit('lock-screen')
    expect(flags(window.sent)).toEqual([{ locked: true, suspended: false, system: false }])
  })

  it.each([-1, Number.NaN, Number.POSITIVE_INFINITY])('treats an idle time of %s as no system source', seconds => {
    const window = new FakeWindow()
    const presence = harness({ clock: () => seconds, windows: [window] })

    presence.tick(60)

    expect(window.sent).toEqual([])
    expect(presence.get()).toMatchObject({ system: false, idleSeconds: null })
  })

  it('ends a suspend whose resume never arrived at the next tick a full interval later', () => {
    const window = new FakeWindow()
    const presence = harness({ supported: false, windows: [window] })

    presence.power.emit('suspend')
    presence.tick(15)

    expect(flags(window.sent)).toEqual([
      { locked: false, suspended: true, system: false },
      { locked: false, suspended: false, system: false }
    ])
  })

  it('reconciles the lock with the OS on every sample where the OS reports it', () => {
    const window = new FakeWindow()
    const presence = harness({ platform: 'darwin', windows: [window] })

    presence.osState('locked')
    presence.tick(15)
    expect(flags(window.sent)).toEqual([{ locked: true, suspended: false, system: true }])

    // A missed unlock does not latch: the OS says it is open again.
    presence.osState('active')
    presence.tick(15, { input: true })
    expect(window.sent.at(-1)).toMatchObject({ locked: false })
    expect(presence.get().locked).toBe(false)
  })

  it('keeps the lock event where the OS cannot report a lock screen', () => {
    const presence = harness({ platform: 'linux' })

    presence.power.emit('lock-screen')
    presence.osState('idle')
    presence.tick(30, { input: true })

    expect(presence.get().locked).toBe(true)
  })

  // Under X11 the clock is reset by Hermes itself after a reply (no user input), so Linux never
  // samples it, X11 or Wayland: presence falls back to input inside the window.
  it.each([
    ['X11', { DISPLAY: ':0', XDG_SESSION_TYPE: 'x11', WAYLAND_DISPLAY: undefined }],
    ['Wayland', { DISPLAY: ':0', XDG_SESSION_TYPE: 'wayland', WAYLAND_DISPLAY: 'wayland-0' }]
  ])('on Linux under %s reports no system clock, and still pushes lock, suspend and resume', (_session, env) => {
    for (const [name, value] of Object.entries(env)) {
      vi.stubEnv(name, value)
    }

    try {
      const window = new FakeWindow()
      const presence = harness({ platform: 'linux', supported: 'platform', windows: [window] })

      presence.tick(60, { input: true })
      expect(window.sent).toEqual([])
      expect(presence.get()).toMatchObject({ system: false, idleSeconds: null })
      expect(presence.getSystemIdleTime).not.toHaveBeenCalled()

      presence.power.emit('lock-screen')
      presence.power.emit('suspend')
      presence.power.emit('resume')
      expect(flags(window.sent)).toEqual([
        { locked: true, suspended: false, system: false },
        { locked: true, suspended: true, system: false },
        { locked: true, suspended: false, system: false }
      ])
    } finally {
      vi.unstubAllEnvs()
    }
  })

  it.each(['darwin', 'win32'] as const)('on %s samples the OS idle clock by default', platform => {
    const window = new FakeWindow()
    const presence = harness({ platform, supported: 'platform', windows: [window] })

    presence.idleFor(120)
    presence.tick(15, { input: true })

    expect(window.sent).toEqual([expect.objectContaining({ system: true, idleSeconds: 0 })])
    expect(presence.get()).toMatchObject({ system: true, idleSeconds: 0 })
  })

  it('polls only while at least one window exists', () => {
    const presence = harness({ windows: [] })

    presence.tick(60)
    expect(presence.getSystemIdleTime).not.toHaveBeenCalled()

    const window = new FakeWindow()
    presence.app.emit('browser-window-created', {}, window)
    presence.tick(15, { input: true })
    expect(window.sent).toHaveLength(1)

    window.emit('closed')
    presence.getSystemIdleTime.mockClear()
    presence.tick(60)
    expect(presence.getSystemIdleTime).not.toHaveBeenCalled()
  })
})
