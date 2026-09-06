import { describe, expect, it } from 'vitest'

import type { Contribution } from '@/contrib/types'

import { group } from '../model'

import { fixedTrackSize, type TrackContext } from './track-model'

function ctx(overrides: TrackContext['overrides'], panes: Record<string, Contribution> = {}): TrackContext {
  return {
    paneFor: id => panes[id],
    paneGone: () => false,
    overrides
  }
}

describe('fixedTrackSize lock precedence', () => {
  it('lockedWidth (420) wins over widthOverride (300) when locked', () => {
    // Scenario: sidebar was sash-dragged to 300px, then user locks at 420px
    // (measured DOM size). The track model MUST use 420, not 300.
    const node = group(['sidebar'], { id: 'sidebar-zone' })

    const c = ctx({
      sidebar: {
        lockWidth: true,
        lockedWidth: 420,
        widthOverride: 300
      }
    })

    expect(fixedTrackSize(node, 'row', c)).toBe('420px')
  })

  it('lockedHeight (500) wins over heightOverride (350) when locked', () => {
    const node = group(['terminal'], { id: 'terminal-zone' })

    const c = ctx({
      terminal: {
        lockHeight: true,
        lockedHeight: 500,
        heightOverride: 350
      }
    })

    expect(fixedTrackSize(node, 'column', c)).toBe('500px')
  })

  it('falls back to override when lockedPx is undefined', () => {
    const node = group(['sidebar'], { id: 'sidebar-zone' })

    const c = ctx({
      sidebar: {
        lockWidth: true,
        widthOverride: 300
      }
    })

    // lockedWidth absent — override is the fallback
    expect(fixedTrackSize(node, 'row', c)).toBe('300px')
  })

  it('falls back to declared CSS when both lockedPx and override are absent', () => {
    const node = group(['sidebar'], { id: 'sidebar-zone' })

    const c = ctx(
      { sidebar: { lockWidth: true } },
      { sidebar: { data: { width: '237px' }, id: 'sidebar', render: () => null, title: 'S' } as Contribution }
    )

    expect(fixedTrackSize(node, 'row', c)).toBe('237px')
  })

  it('locked pane with no lockedPx and no override falls through to declared', () => {
    const node = group(['terminal'], { id: 'terminal-zone' })

    const c = ctx(
      { terminal: { lockHeight: true } },
      { terminal: { data: { height: '38vh' }, id: 'terminal', render: () => null, title: 'T' } as Contribution }
    )

    expect(fixedTrackSize(node, 'column', c)).toBe('38vh')
  })
})
