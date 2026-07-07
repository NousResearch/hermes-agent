import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { profileRestoreSuperseded, sameCronSignature } from './desktop-controller-utils'

const session = (id: string, title: string | null): SessionInfo => ({ id, title }) as SessionInfo

describe('sameCronSignature', () => {
  it('is false when the lengths differ', () => {
    expect(sameCronSignature([session('a', 't')], [])).toBe(false)
  })

  it('is true when ids and titles match in order', () => {
    const a = [session('a', 'one'), session('b', 'two')]
    const b = [session('a', 'one'), session('b', 'two')]
    expect(sameCronSignature(a, b)).toBe(true)
  })

  it('is false when a title changed', () => {
    const a = [session('a', 'one')]
    const b = [session('a', 'renamed')]
    expect(sameCronSignature(a, b)).toBe(false)
  })

  it('is false when order differs', () => {
    const a = [session('a', 't'), session('b', 't')]
    const b = [session('b', 't'), session('a', 't')]
    expect(sameCronSignature(a, b)).toBe(false)
  })
})

describe('profileRestoreSuperseded', () => {
  it('does NOT abort for the pre-switch leftover selection (switch away from an open session)', () => {
    // Switching A → B while viewing one of A's sessions: A's id still sits in
    // the selection when the probe starts AND when it resolves. The probe must
    // proceed so B's unlisted/deleted memo gets restored or forgotten.
    expect(
      profileRestoreSuperseded({
        currentToken: 3,
        requestToken: 3,
        selectedAtRequest: 'a-open-session',
        selectedNow: 'a-open-session'
      })
    ).toBe(false)
  })

  it('proceeds when nothing was selected before or during the probe', () => {
    expect(
      profileRestoreSuperseded({ currentToken: 1, requestToken: 1, selectedAtRequest: null, selectedNow: null })
    ).toBe(false)
  })

  it('aborts when the user opened a chat while probing', () => {
    expect(
      profileRestoreSuperseded({
        currentToken: 3,
        requestToken: 3,
        selectedAtRequest: null,
        selectedNow: 'user-opened-this'
      })
    ).toBe(true)
  })

  it('aborts when the selection changed to a different session while probing', () => {
    expect(
      profileRestoreSuperseded({
        currentToken: 3,
        requestToken: 3,
        selectedAtRequest: 'a-open-session',
        selectedNow: 'newer-session'
      })
    ).toBe(true)
  })

  it('aborts when the selection was cleared (fresh draft) while probing', () => {
    expect(
      profileRestoreSuperseded({
        currentToken: 3,
        requestToken: 3,
        selectedAtRequest: 'a-open-session',
        selectedNow: null
      })
    ).toBe(true)
  })

  it('aborts when a newer switch bumped the token', () => {
    expect(
      profileRestoreSuperseded({
        currentToken: 4,
        requestToken: 3,
        selectedAtRequest: 'a-open-session',
        selectedNow: 'a-open-session'
      })
    ).toBe(true)
  })
})
