import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $notifications } from './notifications'
import {
  $profileRemoteOverrides,
  $remoteOverrideDialogProfile,
  closeRemoteOverrideDialog,
  notifyRemoteOverrideAuthFailure,
  openRemoteOverrideDialog,
  refreshProfileRemoteOverrides,
  remoteHostLabel
} from './profile-remote-override'

const getConnectionConfig = vi.fn()

beforeEach(() => {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { getConnectionConfig }
  })
  $profileRemoteOverrides.set({})
  $remoteOverrideDialogProfile.set(null)
  $notifications.set([])
})

afterEach(() => {
  vi.clearAllMocks()
})

describe('remoteHostLabel', () => {
  it('keeps a non-default port and drops default ones', () => {
    expect(remoteHostLabel('https://hermes.example.com:8443/x')).toBe('hermes.example.com:8443')
    expect(remoteHostLabel('https://hermes.example.com:443')).toBe('hermes.example.com')
    expect(remoteHostLabel('http://hermes.example.com:80')).toBe('hermes.example.com')
  })

  it('returns empty for unparseable input', () => {
    expect(remoteHostLabel('not a url')).toBe('')
    expect(remoteHostLabel('')).toBe('')
  })
})

describe('refreshProfileRemoteOverrides', () => {
  it('publishes only profiles whose scope resolves to a remote override', async () => {
    getConnectionConfig.mockImplementation(async (profile: string) => {
      if (profile === 'work') {
        return { mode: 'remote', remoteUrl: 'https://work.example.com' }
      }

      if (profile === 'cloudy') {
        return { mode: 'cloud', remoteUrl: 'https://cloud.example.com:8443' }
      }

      return { mode: 'local', remoteUrl: '' }
    })

    await refreshProfileRemoteOverrides(['work', 'cloudy', 'home'])

    expect($profileRemoteOverrides.get()).toEqual({
      work: { host: 'work.example.com', url: 'https://work.example.com' },
      cloudy: { host: 'cloud.example.com:8443', url: 'https://cloud.example.com:8443' }
    })
  })

  it('drops a stale override entry when the profile went back to local', async () => {
    $profileRemoteOverrides.set({ work: { host: 'old.example.com', url: 'https://old.example.com' } })
    getConnectionConfig.mockResolvedValue({ mode: 'local', remoteUrl: '' })

    await refreshProfileRemoteOverrides(['work'])

    expect($profileRemoteOverrides.get()).toEqual({})
  })

  it('leaves a profile unbadged when its scope read fails, without failing the rest', async () => {
    getConnectionConfig.mockImplementation(async (profile: string) => {
      if (profile === 'broken') {
        throw new Error('bridge unavailable')
      }

      return { mode: 'remote', remoteUrl: 'https://ok.example.com' }
    })

    await refreshProfileRemoteOverrides(['broken', 'work'])

    expect(Object.keys($profileRemoteOverrides.get())).toEqual(['work'])
  })
})

describe('notifyRemoteOverrideAuthFailure', () => {
  beforeEach(() => {
    $profileRemoteOverrides.set({ work: { host: 'work.example.com', url: 'https://work.example.com' } })
  })

  it('surfaces a re-enter-token toast whose action reopens the dialog on a 401', () => {
    const handled = notifyRemoteOverrideAuthFailure('work', new Error('WebSocket handshake failed: 401 Unauthorized'))

    expect(handled).toBe(true)
    const toast = $notifications.get()[0]
    expect(toast.kind).toBe('error')
    expect(toast.action).toBeTruthy()

    toast.action?.onClick()
    expect($remoteOverrideDialogProfile.get()).toBe('work')
  })

  it('ignores connectivity failures — a down host is not a rotated token', () => {
    expect(notifyRemoteOverrideAuthFailure('work', new Error('connect ECONNREFUSED 10.0.0.5:8000'))).toBe(false)
    expect(notifyRemoteOverrideAuthFailure('work', new Error('timeout waiting for backend'))).toBe(false)
    expect($notifications.get()).toEqual([])
  })

  it('ignores profiles without an override entirely', () => {
    expect(notifyRemoteOverrideAuthFailure('home', new Error('401 Unauthorized'))).toBe(false)
    expect($notifications.get()).toEqual([])
  })
})

describe('dialog open/close atoms', () => {
  it('opens for a profile and closes back to null', () => {
    openRemoteOverrideDialog('work')
    expect($remoteOverrideDialogProfile.get()).toBe('work')
    closeRemoteOverrideDialog()
    expect($remoteOverrideDialogProfile.get()).toBeNull()
  })
})
