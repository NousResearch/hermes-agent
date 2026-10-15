import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Keep Electron's permissively typed dependency graph out of the strict
// renderer compilation while exercising its real writer through the IPC seam.
const { createDesktopProfilePreferences } = await vi.importActual<{
  createDesktopProfilePreferences: (configPath: string) => {
    readActive: () => null | string
    remember: (name: unknown) => null | string
  }
}>('../../electron/desktop-profile')

// Picking a profile must stay on the source the user is LOOKING at. $profiles
// is the active gateway's list, so a pick made while a registry source is live
// names one of THAT source's profiles. Routing it through the profile-only
// path resolved the descriptor with a bare name, which the main process
// answers against the primary — the gateway snapped back home and the pick
// looked like it never took. Default on the explicit `local` source is the
// exception: that name is also the window primary's profile key, so the
// profile-only door would activate a remote-primary VPS.

const ensureGatewayForProfile = vi.fn(async (_profile: string) => undefined)
const ensureGatewayForAgent = vi.fn(async (_connectionId: null | string, _profile: string) => true)
const openGatewayForProfile = vi.fn(async (_profile: string) => undefined)
const activeGatewayConnectionId = vi.fn<() => null | string>(() => null)
const $gateway = atom<unknown>({ id: 'live-socket' })
const resetStarmapGraph = vi.fn()

vi.mock('@/store/gateway', () => ({
  $gateway,
  activeGatewayConnectionId,
  // Activation now verifies the socket's route before publishing the profile.
  activeGatewayProfileKey: () => ensureGatewayForProfile.mock.lastCall?.[0] ?? $activeGatewayProfile.get(),
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  openGatewayForProfile
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph }))
// Notification presentation is external to selection and persistence.
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))

const { $activeGatewayProfile, newSessionInProfile, selectProfile } = await import('./profile')

beforeEach(() => {
  ensureGatewayForProfile.mockClear()
  ensureGatewayForAgent.mockClear()
  activeGatewayConnectionId.mockReset()
  activeGatewayConnectionId.mockReturnValue(null)
  $gateway.set({ id: 'live-socket' })
  $activeGatewayProfile.set('default')
  // resolveConnectionForAgent is best-effort; without a bridge it resolves
  // null and the previous descriptor stays, which is fine here.
  ;(globalThis as { window?: unknown }).window = {}
})

describe('selectProfile', () => {
  it('activates the pick on the live registry source, not the primary', async () => {
    activeGatewayConnectionId.mockReturnValue('mini')

    selectProfile('researcher')

    await vi.waitFor(() => expect(ensureGatewayForAgent).toHaveBeenCalledWith('mini', 'researcher'))
    expect(ensureGatewayForProfile).not.toHaveBeenCalled()
  })

  it('keeps the legacy profile-only path when the primary is live', async () => {
    activeGatewayConnectionId.mockReturnValue(null)

    selectProfile('ops')

    await vi.waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledWith('ops'))
    expect(ensureGatewayForAgent).not.toHaveBeenCalled()
  })

  it('keeps the legacy profile-only path when the explicit local source is live', async () => {
    activeGatewayConnectionId.mockReturnValue('local')

    selectProfile('override-profile')

    await vi.waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledWith('override-profile'))
    expect(ensureGatewayForAgent).not.toHaveBeenCalled()
  })

  it('keeps Default on the explicit local source instead of the window primary', async () => {
    activeGatewayConnectionId.mockReturnValue('local')

    selectProfile('default')

    await vi.waitFor(() => expect(ensureGatewayForAgent).toHaveBeenCalledWith('local', 'default'))
    expect(ensureGatewayForProfile).not.toHaveBeenCalled()
  })
})

describe('newSessionInProfile', () => {
  it('opens the new chat on the live registry source', async () => {
    activeGatewayConnectionId.mockReturnValue('mini')

    newSessionInProfile('designer')

    await vi.waitFor(() => expect(ensureGatewayForAgent).toHaveBeenCalledWith('mini', 'designer'))
    expect(ensureGatewayForProfile).not.toHaveBeenCalled()
  })

  it('keeps the legacy profile-only path for a new chat on the explicit local source', async () => {
    activeGatewayConnectionId.mockReturnValue('local')

    newSessionInProfile('override-profile')

    await vi.waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledWith('override-profile'))
    expect(ensureGatewayForAgent).not.toHaveBeenCalled()
  })

  it('opens a Default new chat on the explicit local source, not the window primary', async () => {
    activeGatewayConnectionId.mockReturnValue('local')

    newSessionInProfile('default')

    await vi.waitFor(() => expect(ensureGatewayForAgent).toHaveBeenCalledWith('local', 'default'))
    expect(ensureGatewayForProfile).not.toHaveBeenCalled()
  })
})

describe('selectProfile startup preference (#79886)', () => {
  let root: string
  let preferences: ReturnType<typeof createDesktopProfilePreferences>
  const rememberProfile = vi.fn(async (name: null | string) => ({ profile: preferences.remember(name) }))
  const getConnectionConfig = vi.fn(async () => ({ mode: 'local' }))

  beforeEach(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), 'profile-selection-'))
    preferences = createDesktopProfilePreferences(path.join(root, 'active-profile.json'))
    preferences.remember('default')
    rememberProfile.mockClear()
    getConnectionConfig.mockReset()
    getConnectionConfig.mockResolvedValue({ mode: 'local' })

    const getConnection = vi.fn(async () => ({ mode: 'local' }))

    ;(globalThis as { window?: unknown }).window = {
      hermesDesktop: {
        getConnection,
        getConnectionConfig,
        profile: { remember: rememberProfile }
      }
    }
  })

  afterEach(() => fs.rmSync(root, { recursive: true, force: true }))

  it.each(['beta', 'alpha'])('ignores late eligibility after a newer %s pick', async latest => {
    let resolveOlder!: (config: { mode: 'local' }) => void

    const olderConfig = new Promise<{ mode: 'local' }>(resolve => {
      resolveOlder = resolve
    })

    getConnectionConfig.mockImplementationOnce(() => olderConfig)
    selectProfile('alpha')
    await vi.waitFor(() => expect($activeGatewayProfile.get()).toBe('alpha'))
    expect(rememberProfile).not.toHaveBeenCalled()

    selectProfile(latest)
    await vi.waitFor(() => expect(rememberProfile).toHaveBeenCalledOnce())
    expect(preferences.readActive()).toBe(latest)

    resolveOlder({ mode: 'local' })
    await new Promise(resolve => setTimeout(resolve, 0))

    expect($activeGatewayProfile.get()).toBe(latest)
    expect(preferences.readActive()).toBe(latest)
    expect(rememberProfile).toHaveBeenCalledOnce()
  })

  it.each(['registry', 'ssh', 'failed'])(
    'preserves the last successful local selection after a newer %s pick',
    async kind => {
      let resolveOlder!: (config: { mode: string }) => void

      const olderConfig = new Promise<{ mode: string }>(resolve => {
        resolveOlder = resolve
      })

      getConnectionConfig.mockImplementationOnce(() => olderConfig)
      selectProfile('alpha')
      await vi.waitFor(() => expect($activeGatewayProfile.get()).toBe('alpha'))

      if (kind === 'registry') {
        // A registry pick must not discard the last successful local choice.
        activeGatewayConnectionId.mockReturnValue('mini')
      } else if (kind === 'ssh') {
        getConnectionConfig.mockResolvedValue({ mode: 'ssh' })
      } else {
        ensureGatewayForProfile.mockRejectedValueOnce(new Error('backend unavailable'))
      }

      const latest = kind === 'registry' ? 'alpha' : 'beta'
      selectProfile(latest)

      if (kind === 'registry') {
        await vi.waitFor(() => expect(ensureGatewayForAgent).toHaveBeenCalledWith('mini', latest))
      } else {
        await vi.waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledWith(latest))
      }

      await new Promise(resolve => setTimeout(resolve, 0))
      expect(rememberProfile).not.toHaveBeenCalled()

      resolveOlder({ mode: 'local' })
      await new Promise(resolve => setTimeout(resolve, 0))

      expect($activeGatewayProfile.get()).toBe(kind === 'failed' ? 'alpha' : latest)
      expect(preferences.readActive()).toBe('alpha')
      expect(rememberProfile).toHaveBeenCalledOnce()
      expect(rememberProfile).toHaveBeenCalledWith('alpha')
    }
  )

  it('does not persist a pick whose gateway activation fails', async () => {
    ensureGatewayForProfile.mockRejectedValueOnce(new Error('backend unavailable'))
    selectProfile('unreachable')
    await vi.waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledWith('unreachable'))
    await new Promise(resolve => setTimeout(resolve, 0))
    expect($activeGatewayProfile.get()).toBe('default')
    expect(preferences.readActive()).toBe('default')
    expect(rememberProfile).not.toHaveBeenCalled()
  })

  it('remembers the selected workspace for the next Desktop launch', async () => {
    activeGatewayConnectionId.mockReturnValue(null)

    selectProfile('tilly')

    await vi.waitFor(() => expect(rememberProfile).toHaveBeenCalledWith('tilly'))
    expect(ensureGatewayForProfile).toHaveBeenCalledWith('tilly')
  })

  it('waits for gateway activation before replacing the startup preference', async () => {
    let resolveGateway!: () => void

    activeGatewayConnectionId.mockReturnValue(null)
    ensureGatewayForProfile.mockImplementationOnce(
      () =>
        new Promise<undefined>(resolve => {
          resolveGateway = () => resolve(undefined)
        })
    )

    selectProfile('tilly')
    await vi.waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledWith('tilly'))
    expect(rememberProfile).not.toHaveBeenCalled()

    resolveGateway()

    await vi.waitFor(() => expect(rememberProfile).toHaveBeenCalledWith('tilly'))
  })

  it('does not replace the startup preference for a registry-source pick', async () => {
    activeGatewayConnectionId.mockReturnValue('mini')

    selectProfile('researcher')

    await vi.waitFor(() => expect(ensureGatewayForAgent).toHaveBeenCalledWith('mini', 'researcher'))
    expect(rememberProfile).not.toHaveBeenCalled()
  })

  it('remembers an already-active local profile after returning from All Profiles', async () => {
    activeGatewayConnectionId.mockReturnValue(null)
    $activeGatewayProfile.set('tilly')

    selectProfile('tilly')

    await vi.waitFor(() => expect(rememberProfile).toHaveBeenCalledWith('tilly'))
  })

  it('keeps local startup persistence when the backend descriptor lookup fails', async () => {
    activeGatewayConnectionId.mockReturnValue(null)

    const getConnection = vi.fn(async () => {
      throw new Error('descriptor unavailable')
    })

    const getConnectionConfig = vi.fn(async () => ({ mode: 'local' }))

    ;(globalThis as { window?: unknown }).window = {
      hermesDesktop: {
        getConnection,
        getConnectionConfig,
        profile: { remember: rememberProfile }
      }
    }

    selectProfile('tilly')

    await vi.waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledWith('tilly'))
    await vi.waitFor(() => expect(rememberProfile).toHaveBeenCalledWith('tilly'))
  })

  it('does not replace the local startup preference for a profile SSH override', async () => {
    activeGatewayConnectionId.mockReturnValue(null)

    const getConnection = vi.fn(async () => ({ mode: 'remote', remoteKind: 'ssh' }))

    const getConnectionConfig = vi.fn(async () => ({ mode: 'ssh' }))

    ;(globalThis as { window?: unknown }).window = {
      hermesDesktop: {
        getConnection,
        getConnectionConfig,
        profile: { remember: rememberProfile }
      }
    }

    selectProfile('macmini-hermes')

    await vi.waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledWith('macmini-hermes'))
    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledWith('macmini-hermes'))
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(rememberProfile).not.toHaveBeenCalled()
  })
})
