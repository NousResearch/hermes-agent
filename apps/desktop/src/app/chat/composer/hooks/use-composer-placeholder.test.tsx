import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileInfo } from '@/types/hermes'

import {
  $activeGatewayProfile,
  $profileColors,
  $profiles,
  $showAllProfiles
} from '@/store/profile'

import { useComposerPlaceholder } from './use-composer-placeholder'

// `set*` setters are the real store setters; the placeholder hook reads the
// atoms, so flipping them in tests is the same code path the renderer uses
// when the user switches profiles or toggles "Show all profiles".

const NAMED_A: ProfileInfo = {
  has_env: false,
  is_default: false,
  model: null,
  name: 'the-best-programmer',
  path: '/home/user/.hermes/profiles/the-best-programmer',
  provider: null,
  skill_count: 0
}
const NAMED_B: ProfileInfo = { ...NAMED_A, name: 'the-best-english-teacher' }
const DEFAULT_PROFILE: ProfileInfo = { ...NAMED_A, is_default: true, name: 'default' }

beforeEach(() => {
  // Pin the resting placeholder seed so snapshot-style assertions don't
  // depend on Math.random — pickPlaceholder selects by `Math.floor(seed % N)`.
  vi.spyOn(Math, 'random').mockReturnValue(0)
})

afterEach(() => {
  cleanup()
  // Reset atoms so the next test starts from a known baseline.
  $profiles.set([])
  $activeGatewayProfile.set('default')
  $profileColors.set({})
  $showAllProfiles.set(false)
})

describe('useComposerPlaceholder — single-profile scope (#77871 empty-state slice)', () => {
  it('returns the original placeholder when only one profile exists, even with show-all on', () => {
    $profiles.set([DEFAULT_PROFILE])
    $activeGatewayProfile.set('default')
    $showAllProfiles.set(true)

    const { result } = renderHook(() =>
      useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })
    )

    // Gate stays closed: profiles.length === 1 short-circuits to baseText.
    expect(result.current.profileColor).toBeNull()
    expect(result.current.text).not.toMatch(/^[^\s·]+\s·\s/)
  })

  it('returns the original placeholder when show-all is off and profiles.length > 1', () => {
    $profiles.set([DEFAULT_PROFILE, NAMED_A])
    $activeGatewayProfile.set('the-best-programmer')
    $showAllProfiles.set(false)

    const { result } = renderHook(() =>
      useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })
    )

    expect(result.current.profileColor).toBeNull()
    expect(result.current.text).not.toMatch(/^[^\s·]+\s·\s/)
  })
})

describe('useComposerPlaceholder — multi-profile + show-all gate (#77871)', () => {
  it('prepends the active profile name when both conditions hold', () => {
    $profiles.set([DEFAULT_PROFILE, NAMED_A])
    $activeGatewayProfile.set('the-best-programmer')
    $showAllProfiles.set(true)

    const { result } = renderHook(() =>
      useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })
    )

    expect(result.current.text.startsWith('the-best-programmer · ')).toBe(true)
    expect(result.current.profileColor).not.toBeNull()
  })

  it('uses the resolved profile color from $profileColors when set', () => {
    $profiles.set([DEFAULT_PROFILE, NAMED_A])
    $activeGatewayProfile.set('the-best-programmer')
    $profileColors.set({ 'the-best-programmer': 'hsl(3 68% 58%)' })
    $showAllProfiles.set(true)

    const { result } = renderHook(() =>
      useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })
    )

    expect(result.current.profileColor).toBe('hsl(3 68% 58%)')
  })

  it('falls back to the deterministic palette when no override is set', () => {
    $profiles.set([DEFAULT_PROFILE, NAMED_A])
    $activeGatewayProfile.set('the-best-programmer')
    $showAllProfiles.set(true)
    // profileColors left empty by afterEach.

    const { result } = renderHook(() =>
      useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })
    )

    // profileColor() returns an hsl(...) string for named profiles; we only
    // care that it's non-null and shaped like the palette the picker produces.
    expect(result.current.profileColor).toMatch(/^hsl\(\d+\s\d+%\s\d+%\)$/)
  })

  it('returns no profile color when the active profile resolves to "default"', () => {
    // resolveProfileColor() deliberately returns null for 'default', so the
    // CSS gradient won't paint — but the gate still trips and the prefix is
    // visible (just monochrome). This is the live behavior we observed in
    // the preview pass when the renderer was on the synthetic default profile.
    $profiles.set([DEFAULT_PROFILE, NAMED_A])
    $activeGatewayProfile.set('default')
    $showAllProfiles.set(true)

    const { result } = renderHook(() =>
      useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })
    )

    expect(result.current.profileColor).toBeNull()
    expect(result.current.text.startsWith('default · ')).toBe(true)
  })

  it('falls back to the activeGatewayProfile key when the profile list omits it', () => {
    // An out-of-band profile name (e.g. just spawned but not yet in $profiles)
    // must still render something coherent rather than throwing.
    $profiles.set([DEFAULT_PROFILE, NAMED_A])
    $activeGatewayProfile.set('a-profile-we-havent-listed-yet')
    $showAllProfiles.set(true)

    const { result } = renderHook(() =>
      useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })
    )

    expect(result.current.text.startsWith('a-profile-we-havent-listed-yet · ')).toBe(true)
  })

  it('handles a second named profile equally — the prefix follows the active one', () => {
    $profiles.set([DEFAULT_PROFILE, NAMED_A, NAMED_B])
    $activeGatewayProfile.set('the-best-english-teacher')
    $showAllProfiles.set(true)

    const { result } = renderHook(() =>
      useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })
    )

    expect(result.current.text.startsWith('the-best-english-teacher · ')).toBe(true)
  })
})

describe('useComposerPlaceholder — disabled states bypass the prefix', () => {
  it.each([
    { reconnecting: true, expected: 'Reconnecting to Hermes…' },
    { reconnecting: false, expected: 'Starting Hermes...' }
  ])('returns the $expected placeholder when disabled=true, reconnecting=$reconnecting', ({ reconnecting, expected }) => {
    $profiles.set([DEFAULT_PROFILE, NAMED_A])
    $activeGatewayProfile.set('the-best-programmer')
    $showAllProfiles.set(true)

    const { result } = renderHook(() =>
      useComposerPlaceholder({ disabled: true, reconnecting, sessionId: null })
    )

    expect(result.current.text).toBe(expected)
    expect(result.current.profileColor).toBeNull()
    expect(result.current.text).not.toMatch(/^[^\s·]+\s·\s/)
  })
})

describe('useComposerPlaceholder — reactivity (#77871)', () => {
  it('re-evaluates the prefix when showAllProfiles flips from false → true', () => {
    $profiles.set([DEFAULT_PROFILE, NAMED_A])
    $activeGatewayProfile.set('the-best-programmer')
    $showAllProfiles.set(false)

    const { result } = renderHook(() =>
      useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })
    )

    expect(result.current.text).not.toMatch(/^[^\s·]+\s·\s/)

    act(() => {
      $showAllProfiles.set(true)
    })

    expect(result.current.text.startsWith('the-best-programmer · ')).toBe(true)
  })

  it('re-evaluates the prefix when the active profile switches between named profiles', () => {
    $profiles.set([DEFAULT_PROFILE, NAMED_A, NAMED_B])
    $activeGatewayProfile.set('the-best-programmer')
    $showAllProfiles.set(true)

    const { result } = renderHook(() =>
      useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })
    )

    expect(result.current.text.startsWith('the-best-programmer · ')).toBe(true)

    act(() => {
      $activeGatewayProfile.set('the-best-english-teacher')
    })

    expect(result.current.text.startsWith('the-best-english-teacher · ')).toBe(true)
  })
})
