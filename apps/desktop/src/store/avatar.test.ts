import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $activeProfileName,
  $avatarDataUrl,
  $avatarLoading,
  $profileAvatarDataUrl,
  loadAvatar,
  resetAvatar,
  setAvatar
} from './avatar'

// ── Mocks ──────────────────────────────────────────────────────────────────

let mockIpc = {
  get: vi.fn<[], Promise<string | null>>(),
  set: vi.fn<[string], Promise<string | null>>(),
  reset: vi.fn<[], Promise<void>>()
}

let store: Record<string, string> = {}

beforeEach(() => {
  // Reset nanostores to defaults
  $avatarDataUrl.set(null)
  $avatarLoading.set(false)
  $activeProfileName.set('default')

  // Reset IPC mocks
  mockIpc = {
    get: vi.fn<[], Promise<string | null>>(),
    set: vi.fn<[string], Promise<string | null>>(),
    reset: vi.fn<[], Promise<void>>()
  }

  // Stub window.hermesDesktop
  ;(globalThis as any).window = {
    hermesDesktop: { avatar: mockIpc }
  }

  // Stub localStorage with a real-ish in-memory store
  store = {}
  const storageMock: Storage = {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value }),
    removeItem: vi.fn((key: string) => { delete store[key] }),
    clear: vi.fn(() => { store = {} }),
    key: vi.fn((index: number) => Object.keys(store)[index] ?? null),
    get length() { return Object.keys(store).length }
  }
  Object.defineProperty(globalThis, 'localStorage', {
    value: storageMock,
    writable: true
  })
})

afterEach(() => {
  vi.restoreAllMocks()
})

// ── Helpers ────────────────────────────────────────────────────────────────

const PNG_URL = 'data:image/png;base64,iVBORw0KGgo=='

// ── Tests ──────────────────────────────────────────────────────────────────

describe('loadAvatar', () => {
  it('sets $avatarDataUrl from IPC get() result', async () => {
    mockIpc.get.mockResolvedValue(PNG_URL)

    const result = await loadAvatar()

    expect(result).toBe(PNG_URL)
    expect($avatarDataUrl.get()).toBe(PNG_URL)
    expect($avatarLoading.get()).toBe(false)
  })

  it('sets null when IPC get() returns null', async () => {
    mockIpc.get.mockResolvedValue(null)

    const result = await loadAvatar()

    expect(result).toBeNull()
    expect($avatarDataUrl.get()).toBeNull()
  })

  it('sets null and logs when IPC get() rejects', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    mockIpc.get.mockRejectedValue(new Error('IPC error'))

    const result = await loadAvatar()

    expect(result).toBeNull()
    expect($avatarDataUrl.get()).toBeNull()
    expect(consoleSpy).toHaveBeenCalledWith(
      '[avatar] Failed to load avatar:',
      expect.any(Error)
    )
    consoleSpy.mockRestore()
  })

  it('sets $avatarLoading to true then false during load', async () => {
    expect($avatarLoading.get()).toBe(false)
    mockIpc.get.mockImplementation(async () => {
      expect($avatarLoading.get()).toBe(true) // loading while fetching
      return PNG_URL
    })

    await loadAvatar()

    expect($avatarLoading.get()).toBe(false)
  })

  it('resets $avatarLoading on error', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    mockIpc.get.mockRejectedValue(new Error('boom'))

    await loadAvatar()

    expect($avatarLoading.get()).toBe(false)
  })
})

describe('setAvatar', () => {
  it('calls IPC set and updates $avatarDataUrl', async () => {
    mockIpc.set.mockResolvedValue(PNG_URL)

    const result = await setAvatar(PNG_URL)

    expect(result).toBe(PNG_URL)
    expect(mockIpc.set).toHaveBeenCalledWith(PNG_URL)
    expect($avatarDataUrl.get()).toBe(PNG_URL)
  })

  it('persists to localStorage for the active profile', async () => {
    mockIpc.set.mockResolvedValue(PNG_URL)
    $activeProfileName.set('haena')

    await setAvatar(PNG_URL)

    expect(localStorage.setItem).toHaveBeenCalledWith('hermes.avatar.haena', PNG_URL)
  })

  it('persists to localStorage for an explicit profile override', async () => {
    mockIpc.set.mockResolvedValue(PNG_URL)

    await setAvatar(PNG_URL, 'mimo')

    expect(localStorage.setItem).toHaveBeenCalledWith('hermes.avatar.mimo', PNG_URL)
  })

  it('falls back to "default" profile when no active profile is set', async () => {
    mockIpc.set.mockResolvedValue(PNG_URL)
    $activeProfileName.set(null as any)

    await setAvatar(PNG_URL)

    expect(localStorage.setItem).toHaveBeenCalledWith('hermes.avatar.default', PNG_URL)
  })

  it('throws when IPC set() rejects', async () => {
    mockIpc.set.mockRejectedValue(new Error('disk full'))

    await expect(setAvatar(PNG_URL)).rejects.toThrow('disk full')
  })
})

describe('resetAvatar', () => {
  it('clears only a single per-profile localStorage override when profile is provided', async () => {
    // Seed two profiles + some other key
    store['hermes.avatar.haena'] = PNG_URL
    store['hermes.avatar.mimo'] = PNG_URL
    store['other.key'] = 'keep-me'

    await resetAvatar('haena')

    // haena's key should be gone
    expect(localStorage.removeItem).toHaveBeenCalledWith('hermes.avatar.haena')
    // mimo's should still be there
    expect(store['hermes.avatar.mimo']).toBe(PNG_URL)
    // non-avatar key should be untouched
    expect(store['other.key']).toBe('keep-me')
    // IPC should NOT be called
    expect(mockIpc.reset).not.toHaveBeenCalled()
  })

  it('clears IPC default + all localStorage overrides when no profile is provided', async () => {
    // Seed several profiles
    store['hermes.avatar.haena'] = PNG_URL
    store['hermes.avatar.mimo'] = PNG_URL
    store['hermes.avatar.default'] = PNG_URL
    store['other.key'] = 'keep-me'

    await resetAvatar()

    // IPC reset should be called
    expect(mockIpc.reset).toHaveBeenCalledOnce()
    // $avatarDataUrl should be nulled
    expect($avatarDataUrl.get()).toBeNull()
    // All avatar localStorage keys should be removed
    expect(store['hermes.avatar.haena']).toBeUndefined()
    expect(store['hermes.avatar.mimo']).toBeUndefined()
    expect(store['hermes.avatar.default']).toBeUndefined()
    // Non-avatar key should be untouched
    expect(store['other.key']).toBe('keep-me')
  })

  it('handles IPC reset rejection gracefully', async () => {
    mockIpc.reset.mockRejectedValue(new Error('file locked'))

    await expect(resetAvatar()).rejects.toThrow('file locked')
  })

  it('handles missing localStorage keys during reset without profile', async () => {
    // No avatar keys set

    await resetAvatar()

    expect(mockIpc.reset).toHaveBeenCalledOnce()
    expect($avatarDataUrl.get()).toBeNull()
  })
})

describe('$profileAvatarDataUrl', () => {
  it('returns default IPC avatar when no per-profile override exists', () => {
    $avatarDataUrl.set(PNG_URL)
    $activeProfileName.set('default')

    expect($profileAvatarDataUrl.get()).toBe(PNG_URL)
  })

  it('returns per-profile override when set in localStorage', () => {
    $avatarDataUrl.set(PNG_URL)
    store['hermes.avatar.haena'] = 'data:image/png;base64,HAENA=='

    $activeProfileName.set('haena')

    expect($profileAvatarDataUrl.get()).toBe('data:image/png;base64,HAENA==')
  })

  it('falls back to default when localStorage has no matching key', () => {
    $avatarDataUrl.set(PNG_URL)
    // localStorage is empty
    $activeProfileName.set('unknown-profile')

    expect($profileAvatarDataUrl.get()).toBe(PNG_URL)
  })

  it('returns null when default is null and no localStorage override', () => {
    $avatarDataUrl.set(null)
    $activeProfileName.set('default')

    expect($profileAvatarDataUrl.get()).toBeNull()
  })

  it('reacts to profile switching', () => {
    $avatarDataUrl.set(PNG_URL)
    store['hermes.avatar.mimo'] = 'data:image/png;base64,MIMO=='

    $activeProfileName.set('default')
    expect($profileAvatarDataUrl.get()).toBe(PNG_URL)

    $activeProfileName.set('mimo')
    expect($profileAvatarDataUrl.get()).toBe('data:image/png;base64,MIMO==')
  })
})
