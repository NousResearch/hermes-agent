import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HudPrefsStatus } from '@/lib/hud-prefs'
import { $activeGatewayProfile } from '@/store/profile'

import {
  $hudAsk,
  $hudLaunchOptions,
  $hudPrefs,
  dismissHudAsk,
  hudCycleAgent,
  loadHudPrefs,
  openHudRoom,
  refreshHudLaunchOptions,
  setHudPrefs,
  toggleHudFollow,
  watchHudAsk,
  watchHudPrefs
} from './hud'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const status = (overrides: Partial<HudPrefsStatus> = {}): HudPrefsStatus => ({
  follow: false,
  askShortcut: 'CommandOrControl+Alt+H',
  askOnRightClick: false,
  pets: true,
  petByAgent: {},
  askError: null,
  askHookAvailable: false,
  askHookReason: 'not installed',
  askRegistered: true,
  followSupported: true,
  ...overrides
})

const open = vi.fn().mockResolvedValue({ ok: true })
const getPrefs = vi.fn()
const setPrefs = vi.fn()
const launchOptions = vi.fn()
const openRoom = vi.fn()
const takePendingAsk = vi.fn()
let prefsListener: ((s: HudPrefsStatus) => void) | null = null
let askListener: ((p: unknown) => void) | null = null

function installBridge() {
  desktopWindow.hermesDesktop = {
    hud: {
      open,
      getPrefs,
      setPrefs,
      launchOptions,
      openRoom,
      takePendingAsk,
      onPrefs: (cb: (s: HudPrefsStatus) => void) => {
        prefsListener = cb

        return () => {
          prefsListener = null
        }
      },
      onAsk: (cb: (p: unknown) => void) => {
        askListener = cb

        return () => {
          askListener = null
        }
      }
    }
  } as unknown as Window['hermesDesktop']
}

beforeEach(() => {
  vi.clearAllMocks()
  getPrefs.mockResolvedValue(status())
  setPrefs.mockImplementation(async (patch: Partial<HudPrefsStatus>) => status({ ...patch }))
  launchOptions.mockResolvedValue({ agents: [], groups: [] })
  openRoom.mockResolvedValue({ ok: true })
  takePendingAsk.mockResolvedValue(null)
  installBridge()
  $hudPrefs.set(null)
  $hudLaunchOptions.set({ agents: [], groups: [] })
  $hudAsk.set(null)
  $activeGatewayProfile.set('default')
})

afterEach(() => {
  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('prefs', () => {
  it('loads the status from main and mirrors broadcasts', async () => {
    expect(await loadHudPrefs()).toEqual(status())
    expect($hudPrefs.get()?.askRegistered).toBe(true)

    const off = watchHudPrefs()
    prefsListener?.(status({ follow: true }))
    expect($hudPrefs.get()?.follow).toBe(true)
    off()
    expect(prefsListener).toBeNull()
  })

  it('toggleHudFollow flips the current value through main', async () => {
    $hudPrefs.set(status({ follow: false }))
    toggleHudFollow()
    await Promise.resolve()

    expect(setPrefs).toHaveBeenCalledWith({ follow: true })
  })

  it('setHudPrefs adopts the authoritative reply, including a rejected chord', async () => {
    setPrefs.mockResolvedValueOnce(status({ askError: 'invalid' }))
    const result = await setHudPrefs({ askShortcut: 'H' })

    expect(result?.askError).toBe('invalid')
    expect($hudPrefs.get()?.askError).toBe('invalid')
  })

  it('is inert without the bridge', async () => {
    delete desktopWindow.hermesDesktop
    expect(await loadHudPrefs()).toBeNull()
    expect(await setHudPrefs({ follow: true })).toBeNull()
    expect(watchHudPrefs()).toBeTypeOf('function')
  })
})

describe('launch options + rooms', () => {
  it('refreshes from main and tolerates a malformed reply', async () => {
    launchOptions.mockResolvedValueOnce({ agents: [{ profile: 'a', displayName: 'A', reachable: true }] })
    expect(await refreshHudLaunchOptions()).toEqual({
      agents: [{ profile: 'a', displayName: 'A', reachable: true }],
      groups: []
    })
  })

  it('openHudRoom trims and forwards, and reports main\u2019s answer', async () => {
    expect(await openHudRoom('  Design ')).toBe(true)
    expect(openRoom).toHaveBeenCalledWith('Design')
    expect(await openHudRoom('   ')).toBe(false)
    openRoom.mockResolvedValueOnce({ ok: false })
    expect(await openHudRoom('Gone')).toBe(false)
  })
})

describe('hudCycleAgent', () => {
  const agents = [
    { profile: 'default', displayName: 'Hermes', reachable: true },
    { profile: 'linus', displayName: 'Linus', reachable: true },
    { profile: 'ghost', displayName: 'Ghost', reachable: false },
    { profile: 'ada', displayName: 'Ada', reachable: true }
  ]

  it('steps to the next reachable agent, wrapping, through openHudForProfile', () => {
    $hudLaunchOptions.set({ agents, groups: [] })
    $activeGatewayProfile.set('linus')

    expect(hudCycleAgent(1)).toBe(true)
    expect(open).toHaveBeenLastCalledWith({ sessionId: null, profile: 'ada' })

    $activeGatewayProfile.set('ada')
    expect(hudCycleAgent(1)).toBe(true)
    expect(open).toHaveBeenLastCalledWith({ sessionId: null, profile: 'default' })

    $activeGatewayProfile.set('default')
    expect(hudCycleAgent(-1)).toBe(true)
    expect(open).toHaveBeenLastCalledWith({ sessionId: null, profile: 'ada' })
  })

  it('starts from the first agent when the active profile is not listed', () => {
    $hudLaunchOptions.set({ agents, groups: [] })
    $activeGatewayProfile.set('unknown')

    expect(hudCycleAgent(1)).toBe(true)
    expect(open).toHaveBeenLastCalledWith({ sessionId: null, profile: 'default' })
  })

  it('has nowhere to go with fewer than two reachable agents', () => {
    $hudLaunchOptions.set({ agents: agents.slice(0, 1), groups: [] })
    expect(hudCycleAgent(1)).toBe(false)
    expect(open).not.toHaveBeenCalled()
  })
})

describe('ask payloads', () => {
  const payload = {
    app: 'Figma',
    title: 'Board',
    cursor: { x: 1, y: 2 },
    imagePath: 'C:/x.png',
    thumbnail: '',
    via: 'shortcut' as const
  }

  it('collects the parked payload and then live pushes; dismiss clears', async () => {
    takePendingAsk.mockResolvedValueOnce(payload)
    const off = watchHudAsk()
    await Promise.resolve()
    await Promise.resolve()

    expect($hudAsk.get()).toEqual(payload)
    dismissHudAsk()
    expect($hudAsk.get()).toBeNull()

    askListener?.({ ...payload, via: 'right-click' })
    expect($hudAsk.get()?.via).toBe('right-click')
    off()
    expect(askListener).toBeNull()
  })
})
