import { beforeEach, describe, expect, it } from 'vitest'

import { setActiveProfile } from '@/store/profile'

import {
  clearThreadScrollPosition,
  getThreadScrollPosition,
  planThreadScrollRestore,
  saveThreadScrollPosition,
  THREAD_SCROLL_BOTTOM,
  THREAD_SCROLL_MEMORY_LIMIT,
  threadScrollDistanceFromBottom,
  threadScrollStateFromMetrics,
  threadScrollStorageKey,
  threadScrollTargetTop
} from './thread-scroll'

beforeEach(() => {
  window.localStorage.clear()
  setActiveProfile('default')
})

describe('threadScrollStateFromMetrics', () => {
  it('classifies within the sticky threshold as bottom', () => {
    expect(threadScrollStateFromMetrics({ clientHeight: 800, scrollHeight: 2000, scrollTop: 1195 })).toEqual(
      THREAD_SCROLL_BOTTOM
    )
  })

  it('classifies a real reading offset as an exact distance-from-bottom', () => {
    expect(threadScrollStateFromMetrics({ clientHeight: 800, scrollHeight: 5000, scrollTop: 1000 })).toEqual({
      fromBottom: 3200,
      kind: 'offset'
    })
  })

  it('clamps overscroll bounce to zero distance', () => {
    expect(threadScrollDistanceFromBottom({ clientHeight: 800, scrollHeight: 2000, scrollTop: 2000 })).toBe(0)
    expect(threadScrollDistanceFromBottom({ clientHeight: 800, scrollHeight: 2000, scrollTop: 2200 })).toBe(0)
  })
})

describe('threadScrollTargetTop', () => {
  it('maps bottom to the maximum scroll offset', () => {
    expect(threadScrollTargetTop(THREAD_SCROLL_BOTTOM, { clientHeight: 800, scrollHeight: 5000 })).toBe(4200)
  })

  it('maps an offset to bottom-anchored scrollTop', () => {
    expect(threadScrollTargetTop({ fromBottom: 1200, kind: 'offset' }, { clientHeight: 800, scrollHeight: 5000 })).toBe(
      3000
    )
  })

  it('clamps an offset deeper than the current content to zero', () => {
    expect(threadScrollTargetTop({ fromBottom: 9000, kind: 'offset' }, { clientHeight: 800, scrollHeight: 5000 })).toBe(0)
  })
})

describe('threadScrollStorageKey', () => {
  it('scopes the key per profile with the encodeURIComponent suffix', () => {
    expect(threadScrollStorageKey('work')).toBe('hermes.desktop.threadScroll.v1.profile.work')
    expect(threadScrollStorageKey('a/b')).toBe('hermes.desktop.threadScroll.v1.profile.a%2Fb')
  })

  it('normalizes empty profiles to default', () => {
    expect(threadScrollStorageKey('  ')).toBe('hermes.desktop.threadScroll.v1.profile.default')
  })
})

describe('per-session scroll persistence', () => {
  it('round-trips a saved position under the active profile', () => {
    setActiveProfile('work')
    saveThreadScrollPosition('session-a', { fromBottom: 240, kind: 'offset' })

    expect(getThreadScrollPosition('session-a')).toEqual({ fromBottom: 240, kind: 'offset' })
  })

  it('isolates positions between profiles (#67709 pattern)', () => {
    setActiveProfile('work')
    saveThreadScrollPosition('session-a', { fromBottom: 240, kind: 'offset' })

    setActiveProfile('personal')
    expect(getThreadScrollPosition('session-a')).toBeUndefined()
  })

  it('does not leak a saved position into another session key', () => {
    saveThreadScrollPosition('session-a', { fromBottom: 240, kind: 'offset' })

    expect(getThreadScrollPosition('session-b')).toBeUndefined()
  })

  it('clears a saved position', () => {
    saveThreadScrollPosition('session-a', THREAD_SCROLL_BOTTOM)

    clearThreadScrollPosition('session-a')

    expect(getThreadScrollPosition('session-a')).toBeUndefined()
  })

  it('is a no-op when clearing an absent position', () => {
    clearThreadScrollPosition('never-saved')
    expect(getThreadScrollPosition('never-saved')).toBeUndefined()
  })

  it('evicts the least-recently-saved session past the limit', () => {
    for (let i = 0; i < THREAD_SCROLL_MEMORY_LIMIT; i++) {
      saveThreadScrollPosition(`session-${i}`, THREAD_SCROLL_BOTTOM)
    }

    // Re-touch session-0 so it is the most recent, then add one more.
    saveThreadScrollPosition('session-0', THREAD_SCROLL_BOTTOM)
    saveThreadScrollPosition('overflow', THREAD_SCROLL_BOTTOM)

    // session-1 was never re-touched → it is now the oldest and got evicted.
    expect(getThreadScrollPosition('session-1')).toBeUndefined()
    expect(getThreadScrollPosition('overflow')).toEqual(THREAD_SCROLL_BOTTOM)
  })

  it('drops corrupt JSON on load', () => {
    window.localStorage.setItem(threadScrollStorageKey('default'), '{not-json')

    expect(getThreadScrollPosition('session-a')).toBeUndefined()
  })

  it('drops entries with invalid shapes, keeping valid ones', () => {
    window.localStorage.setItem(
      threadScrollStorageKey('default'),
      JSON.stringify({
        good: { fromBottom: 240, kind: 'offset' },
        badKind: { fromBottom: 240, kind: 'sideways' },
        badFromBottom: { fromBottom: '240', kind: 'offset' },
        nullEntry: null,
        arrayEntry: [1, 2, 3]
      })
    )

    expect(getThreadScrollPosition('good')).toEqual({ fromBottom: 240, kind: 'offset' })
    expect(getThreadScrollPosition('badKind')).toBeUndefined()
    expect(getThreadScrollPosition('badFromBottom')).toBeUndefined()
    expect(getThreadScrollPosition('nullEntry')).toBeUndefined()
    expect(getThreadScrollPosition('arrayEntry')).toBeUndefined()
  })

  it('persists bottom state as a first-class entry', () => {
    saveThreadScrollPosition('session-a', THREAD_SCROLL_BOTTOM)

    expect(getThreadScrollPosition('session-a')).toEqual(THREAD_SCROLL_BOTTOM)
  })
})

describe('planThreadScrollRestore (warm/cold switch lifecycle)', () => {
  it('cold switch: no transcript yet → forget the gate and do not restore', () => {
    const plan = planThreadScrollRestore('session-a', 'session-b', false, true)

    expect(plan).toEqual({ cold: true, gate: null, restore: false })
  })

  it('first content for a key → restore', () => {
    expect(planThreadScrollRestore(undefined, 'session-a', true, false)).toEqual({
      cold: false,
      gate: 'session-a',
      restore: true
    })
  })

  it('warm switch to a different key → restore the new session', () => {
    expect(planThreadScrollRestore('session-a', 'session-b', true, true)).toEqual({
      cold: false,
      gate: 'session-b',
      restore: true
    })
  })

  it('same key, already settled → record only, no re-restore', () => {
    expect(planThreadScrollRestore('session-a', 'session-a', true, true)).toEqual({
      cold: false,
      gate: 'session-a',
      restore: false
    })
  })

  it('same key, still settling → re-arm the restore instead of stranding the viewport', () => {
    expect(planThreadScrollRestore('session-a', 'session-a', true, false)).toEqual({
      cold: false,
      gate: 'session-a',
      restore: true
    })
  })
})
