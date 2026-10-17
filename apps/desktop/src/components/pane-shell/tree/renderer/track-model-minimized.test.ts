import { describe, expect, it } from 'vitest'

import type { LayoutNode } from '../model'

import { fixedTrackSize, MINIMIZED_TRACK, type TrackContext } from './track-model'

const panes: Record<string, { placement?: string; width?: string; height?: string }> = {
  workspace: { placement: 'main' },
  terminal: { placement: 'bottom', height: '38vh' },
  files: { placement: 'right', width: '237px' }
}

const ctx: TrackContext = {
  paneFor: id => (panes[id] ? ({ id, area: 'panes', data: panes[id] } as never) : undefined),
  paneGone: () => false,
  overrides: {}
}

// Basic layout: chat over a minimized terminal rail.
const mainColumn: LayoutNode = {
  type: 'split',
  id: 's-main',
  orientation: 'column',
  children: [
    { type: 'group', id: 'g-main', panes: ['workspace'], active: 'workspace' },
    { type: 'group', id: 'g-term', panes: ['terminal'], active: 'terminal', minimized: true }
  ],
  weights: [3, 1]
} as LayoutNode

describe('fixedTrackSize with a minimized zone', () => {
  it('keeps the minimized strip along its own split axis', () => {
    expect(fixedTrackSize(mainColumn, 'column', ctx)).toBeNull()
    expect(fixedTrackSize((mainColumn as { children: LayoutNode[] }).children[1], 'column', ctx)).toBe(
      MINIMIZED_TRACK
    )
  })

  it('does not pin the main column to the strip width across the axis', () => {
    // A bottom rail is 1.75rem TALL, not wide. Reporting it across the axis
    // made the main column a fixed 1.75rem track, so an uncapped plugin pane
    // in the right rail became the absorber and the chat collapsed to 28px.
    expect(fixedTrackSize(mainColumn, 'row', ctx)).toBeNull()
  })

  it('still sizes a minimized side rail along the row axis', () => {
    const files: LayoutNode = { type: 'group', id: 'g-files', panes: ['files'], active: 'files', minimized: true } as LayoutNode

    expect(fixedTrackSize(files, 'row', ctx)).toBe(MINIMIZED_TRACK)
  })
})
