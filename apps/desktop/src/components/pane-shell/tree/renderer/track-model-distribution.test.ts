import { describe, expect, test } from 'vitest'

import type { Contribution } from '@/contrib/types'

import { group } from '../model'

import { fixedTrackSize, MINIMIZED_TRACK, type TrackContext } from './track-model'

const contribution = (id: string, data: Record<string, unknown>): Contribution => ({
  area: 'panes',
  data,
  id,
  render: () => null,
  title: id
})

const context = (data: Record<string, unknown>, ignoreFixedSizing = false): TrackContext => ({
  ignoreFixedSizing,
  overrides: {},
  paneFor: id => contribution(id, data),
  paneGone: () => false
})

describe('fixed track sizing and pane distribution', () => {
  test('equal-all ignores a declared width while equal-flex preserves it', () => {
    const node = group(['files'], { id: 'files-zone' })

    expect(fixedTrackSize(node, 'row', context({ width: '240px' }))).toBe('240px')
    expect(fixedTrackSize(node, 'row', context({ width: '240px' }, true))).toBeNull()
  })

  test('minimized groups remain rails in equal-all mode', () => {
    const node = group(['terminal'], { id: 'terminal-zone', minimized: true })

    expect(fixedTrackSize(node, 'row', context({}, true))).toBe(MINIMIZED_TRACK)
  })
})
