import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopManagedStatus } from '@/global'

import {
  $authState,
  handleAuthGate,
  markManagedUnavailable,
  markSignedIn,
  refreshAuthStatus,
  signOutAccount
} from './auth'

function status(overrides: Partial<DesktopManagedStatus> = {}): DesktopManagedStatus {
  return {
    baseUrl: 'https://apex-nodes.com/relay/v1',
    email: '',
    enabled: true,
    model: 'deepseek-v4-pro',
    modelDisplay: 'deepseek-v4-pro-APEX',
    name: '',
    plan: '',
    provider: 'custom',
    signedIn: false,
    ...overrides
  }
}

function installManagedMock(managed: Record<string, unknown>) {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { managed }
  })
}

// Reset the atom + storage before each test so cases don't bleed. The atom is a
// module singleton, so we set it back to the pristine (never-checked) shape.
function resetAuth() {
  window.localStorage.clear()
  $authState.set({ account: { email: '', name: '', plan: '' }, enabled: null, gateReason: null, status: 'checking' })
}

beforeEach(() => {
  resetAuth()
})

afterEach(() => {
  vi.restoreAllMocks()
  // @ts-expect-error — tearing down the injected global between tests.
  delete window.hermesDesktop
})

describe('refreshAuthStatus', () => {
  it('signs the user in and surfaces the account when the relay key is on disk', async () => {
    installManagedMock({ status: vi.fn().mockResolvedValue(status({ signedIn: true, email: 'jane@apex-nodes.com', plan: 'pro' })) })

    await refreshAuthStatus()

    const state = $authState.get()
    expect(state.status).toBe('signed-in')
    expect(state.enabled).toBe(true)
    expect(state.account).toEqual({ email: 'jane@apex-nodes.com', name: '', plan: 'pro' })
  })

  it('gates to signed-out when managed is enabled but no key is present', async () => {
    installManagedMock({ status: vi.fn().mockResolvedValue(status({ signedIn: false })) })

    await refreshAuthStatus()

    expect($authState.get().status).toBe('signed-out')
  })

  it('does not gate on a managed-disabled build (chat flows through)', async () => {
    installManagedMock({ status: vi.fn().mockResolvedValue(status({ enabled: false })) })

    await refreshAuthStatus()

    const state = $authState.get()
    expect(state.enabled).toBe(false)
    expect(state.status).toBe('signed-in')
  })

  it('treats a missing desktop bridge as managed-disabled (dev preview)', async () => {
    // No window.hermesDesktop at all.
    await refreshAuthStatus()

    expect($authState.get().enabled).toBe(false)
  })

  it('keeps a cached signed-in user through a transient status() failure', async () => {
    window.localStorage.setItem('apexnodes-desktop-signed-in-v1', '1')
    installManagedMock({ status: vi.fn().mockRejectedValue(new Error('ipc down')) })

    await refreshAuthStatus()

    expect($authState.get().status).toBe('signed-in')
  })

  it('does not downgrade a disabled account to signed-out on re-check', async () => {
    // Account was disabled mid-session…
    handleAuthGate({ reason: 'account_disabled', statusCode: 403 })
    expect($authState.get().status).toBe('disabled')

    // …a later status() (still no key) must preserve the disabled message.
    installManagedMock({ status: vi.fn().mockResolvedValue(status({ signedIn: false })) })
    await refreshAuthStatus()

    expect($authState.get().status).toBe('disabled')
  })
})

describe('handleAuthGate (continuous gate)', () => {
  it('flips to signed-out on a 401 (login lost)', () => {
    $authState.set({ ...$authState.get(), enabled: true, status: 'signed-in' })

    handleAuthGate({ reason: 'unauthorized', statusCode: 401 })

    const state = $authState.get()
    expect(state.status).toBe('signed-out')
    expect(state.gateReason).toBe('unauthorized')
    expect(window.localStorage.getItem('apexnodes-desktop-signed-in-v1')).toBeNull()
  })

  it('flips to disabled on a 403 account_disabled', () => {
    $authState.set({ ...$authState.get(), enabled: true, status: 'signed-in' })

    handleAuthGate({ reason: 'account_disabled', statusCode: 403 })

    expect($authState.get().status).toBe('disabled')
  })

  it('is a no-op on a managed-disabled build', () => {
    $authState.set({ ...$authState.get(), enabled: false, status: 'signed-in' })

    handleAuthGate({ reason: 'unauthorized', statusCode: 401 })

    expect($authState.get().status).toBe('signed-in')
  })
})

describe('markSignedIn / markManagedUnavailable / signOutAccount', () => {
  it('markSignedIn unblocks chat and caches the session', () => {
    installManagedMock({ status: vi.fn().mockResolvedValue(status({ signedIn: true })) })

    markSignedIn({ email: 'j@apex-nodes.com' })

    expect($authState.get().status).toBe('signed-in')
    expect(window.localStorage.getItem('apexnodes-desktop-signed-in-v1')).toBe('1')
  })

  it('markManagedUnavailable drops the account gate (BYOK fallback)', () => {
    $authState.set({ ...$authState.get(), enabled: true, status: 'signed-out' })

    markManagedUnavailable()

    const state = $authState.get()
    expect(state.enabled).toBe(false)
    expect(state.status).toBe('signed-in')
  })

  it('signOutAccount clears the relay key and returns to signed-out', async () => {
    const signOut = vi.fn().mockResolvedValue({ ok: true })
    installManagedMock({ signOut })
    $authState.set({ ...$authState.get(), enabled: true, status: 'signed-in' })

    await signOutAccount()

    expect(signOut).toHaveBeenCalledOnce()
    expect($authState.get().status).toBe('signed-out')
    expect($authState.get().gateReason).toBeNull()
  })
})
