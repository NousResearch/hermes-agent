import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Keep the side-effecting import chains of profile.ts (gateway sockets, REST
// query client) and projects.ts (desktop fs/git, gateway) inert: this suite
// exercises only the persisted workspace↔profile binding atoms.
vi.mock('@/store/gateway', () => ({
  $gateway: { subscribe: vi.fn() },
  activeGateway: vi.fn(),
  ensureActiveGatewayOpen: vi.fn(),
  ensureGatewayForAgent: vi.fn(),
  ensureGatewayForProfile: vi.fn(),
  openGatewayForProfile: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getHermesConfig: vi.fn(),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  hermesApi: vi.fn(),
  setApiRequestProfile: vi.fn(),
  STARTUP_REQUEST_TIMEOUT_MS: 1000
}))

vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))
vi.mock('@/i18n', () => ({ translateNow: (key: string) => key }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn() }))

vi.mock('@/lib/desktop-fs', () => ({
  desktopDefaultCwd: vi.fn(),
  isDesktopFsRemoteMode: vi.fn(),
  selectDesktopPaths: vi.fn(),
  writeDesktopFileText: vi.fn()
}))

vi.mock('@/lib/desktop-git', () => ({
  desktopGit: vi.fn()
}))

const { $workspaceProfileBindings, bindWorkspaceProfile, unbindWorkspaceProfile } = await import('./projects')

const {
  sanitizeWorkspaceProfileBindings,
  WORKSPACE_PROFILE_BINDINGS_KEY,
  workspaceBoundProfiles,
  workspaceProfileBindingsCodec
} = await import('./workspace-profiles')

beforeEach(() => {
  window.localStorage.clear()
  $workspaceProfileBindings.set({})
})

afterEach(() => {
  window.localStorage.clear()
})

describe('sanitizeWorkspaceProfileBindings (stale entries inert at read time)', () => {
  it('returns an empty record for non-object payloads', () => {
    expect(sanitizeWorkspaceProfileBindings(null)).toEqual({})
    expect(sanitizeWorkspaceProfileBindings('{"p1":["a"]}')).toEqual({})
    expect(sanitizeWorkspaceProfileBindings(['p1'])).toEqual({})
    expect(sanitizeWorkspaceProfileBindings(undefined)).toEqual({})
  })

  it('drops unusable entries and canonicalizes the rest (trim-only keys, case kept)', () => {
    const clean = sanitizeWorkspaceProfileBindings({
      p_bad_string: 'coder',
      p_empty: [],
      ' p_ok ': [' Coder ', 'Coder', '', 42, null, ' default ']
    })

    // Trimmed id; canonicalized names via normalizeProfileKey (trim-only, so
    // " Coder " and "Coder" are ONE binding); non-string members dropped; the
    // emptied entry vanishes entirely so "has bindings" stays falsy.
    expect(clean).toEqual({ p_ok: ['Coder', 'default'] })
  })

  it('keeps first-seen order while deduping repeats', () => {
    expect(sanitizeWorkspaceProfileBindings({ p1: [' b ', 'a', 'b'] })).toEqual({ p1: ['b', 'a'] })
  })
})

describe('workspaceBoundProfiles (the one rail resolver)', () => {
  it('answers the stored list for a bound workspace', () => {
    const bindings = sanitizeWorkspaceProfileBindings({ p1: ['coder'], ' p2 ': ['x'] })

    expect(workspaceBoundProfiles(bindings, 'p1')).toEqual(['coder'])
  })

  it('answers null when filtering is off', () => {
    const bindings = { p1: ['coder'] }

    expect(workspaceBoundProfiles(bindings, '')).toBeNull()
    expect(workspaceBoundProfiles(bindings, null)).toBeNull()
    expect(workspaceBoundProfiles(bindings, undefined)).toBeNull()
    expect(workspaceBoundProfiles(bindings, '__all_projects__')).toBeNull()
    expect(workspaceBoundProfiles({ p_empty: [], p1: [] }, 'p1')).toBeNull()
  })
})

describe('$workspaceProfileBindings round-trip (#64221)', () => {
  it('bind persists immediately to localStorage under the scoped key', () => {
    bindWorkspaceProfile('p_abc', ' Coder ')

    expect($workspaceProfileBindings.get()).toEqual({ p_abc: ['Coder'] })
    expect(window.localStorage.getItem(WORKSPACE_PROFILE_BINDINGS_KEY)).toBe(JSON.stringify({ p_abc: ['Coder'] }))
  })

  it('bind canonicalizes the profile key and dedupes repeats without rewriting', () => {
    bindWorkspaceProfile('p_abc', ' coder ')
    const before = $workspaceProfileBindings.get()

    bindWorkspaceProfile('p_abc', 'coder')

    expect($workspaceProfileBindings.get()).toEqual({ p_abc: ['coder'] })
    expect($workspaceProfileBindings.get()).toBe(before)
  })

  it('unbind drops the entry once the last profile leaves, removing the storage key', () => {
    bindWorkspaceProfile('p_abc', 'coder')
    bindWorkspaceProfile('p_abc', 'writer')

    unbindWorkspaceProfile('p_abc', 'writer')
    expect(window.localStorage.getItem(WORKSPACE_PROFILE_BINDINGS_KEY)).toBe(JSON.stringify({ p_abc: ['coder'] }))

    unbindWorkspaceProfile('p_abc', ' coder ')
    expect($workspaceProfileBindings.get()).toEqual({})
    expect(window.localStorage.getItem(WORKSPACE_PROFILE_BINDINGS_KEY)).toBeNull()
  })

  it('unbind of an unknown profile or synthetic scope is a no-op', () => {
    bindWorkspaceProfile('p_abc', 'coder')

    unbindWorkspaceProfile('p_abc', 'ghost')
    unbindWorkspaceProfile('__all_projects__', 'coder')

    expect($workspaceProfileBindings.get()).toEqual({ p_abc: ['coder'] })
  })

  it('a fresh read through the production codec sanitizes stale stored shapes', async () => {
    // A reload re-runs exactly this chain: persistentAtom(KEY, {}, codec)
    // reading the stored payload. Seed the raw shapes an old/broken build
    // might have left behind and boot a new atom from it.
    window.localStorage.setItem(
      WORKSPACE_PROFILE_BINDINGS_KEY,
      JSON.stringify({
        p_live: [' coder ', 'coder'],
        p_dead: [],
        p_junk: 'not-an-array'
      })
    )

    const { persistentAtom } = await import('@/lib/persisted')
    const $reloaded = persistentAtom(WORKSPACE_PROFILE_BINDINGS_KEY, {}, workspaceProfileBindingsCodec)

    expect($reloaded.get()).toEqual({ p_live: ['coder'] })
  })
})
