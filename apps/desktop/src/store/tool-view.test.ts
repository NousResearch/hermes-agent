import { describe, expect, it } from 'vitest'

import { $anyToolDisclosureOpen, $toolDisclosureOpen } from './tool-view'

// Mount and unmount exactly as a rendered tool row does: subscribe (useStore)
// then unsubscribe (unmount). Everything else — an un-subscribed call — never
// attaches a listener, so eviction could only be observed through this shape.
const mountAndUnmount = (atom: { subscribe: (listener: () => void) => () => void }) => {
  const unsubscribe = atom.subscribe(() => undefined)
  unsubscribe()
}

describe('$toolDisclosureOpen cache', () => {
  it('keeps a stable identity while the row is subscribed', () => {
    const unsubscribe = $toolDisclosureOpen('tool-entry:stable:0').subscribe(() => undefined)

    try {
      expect($toolDisclosureOpen('tool-entry:stable:0')).toBe($toolDisclosureOpen('tool-entry:stable:0'))
    } finally {
      unsubscribe()
    }
  })

  it('releases the cache entry once its row unmounts', () => {
    const row = $toolDisclosureOpen('tool-entry:evicted:0')

    mountAndUnmount(row)

    // A remount gets a fresh atom: the unmounted row's entry was released
    // instead of being retained for the window's lifetime (#131121).
    expect($toolDisclosureOpen('tool-entry:evicted:0')).not.toBe(row)
  })
})

describe('$anyToolDisclosureOpen cache', () => {
  it('keeps a stable identity while the run is subscribed', () => {
    const ids = ['tool-entry:stable-run:0', 'tool-entry:stable-run:1'] as const
    const unsubscribe = $anyToolDisclosureOpen(ids).subscribe(() => undefined)

    try {
      expect($anyToolDisclosureOpen(ids)).toBe($anyToolDisclosureOpen(ids))
    } finally {
      unsubscribe()
    }
  })

  it('releases the cache entry once its run unmounts', () => {
    const ids = ['tool-entry:evicted-run:0', 'tool-entry:evicted-run:1'] as const
    const run = $anyToolDisclosureOpen(ids)

    mountAndUnmount(run)

    expect($anyToolDisclosureOpen(ids)).not.toBe(run)
  })
})
