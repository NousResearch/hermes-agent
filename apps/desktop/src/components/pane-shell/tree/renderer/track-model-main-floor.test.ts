import { describe, expect, it } from 'vitest'

import { group, split } from '../model'

import { fixedTrackSize, type TrackContext } from './track-model'

// The shipped regression: newswire ticker docked bottom of the workspace, then
// a minimized Cronjobs rail enforced right of the same workspace (Bot Mode).
// The resulting center column is [ row(main | routines-minimized), ticker ].
// Before the main-floor rule, the minimized rail's 28px strip made the whole
// row "fixed", the chat collapsed to the rail, and the ticker — the column's
// uncapped absorber — swallowed the window.
const bugTree = split(
  'column',
  [
    split('row', [
      group(['workspace'], { id: 'grp-main' }),
      group(['hermes-bots:routines'], { id: 'g-routines', minimized: true })
    ]),
    group(['hermes-newswire:ticker'], { id: 'g-ticker' })
  ],
  [1, 1],
  's-bug'
)

const panes: Record<string, { placement?: string; width?: string; height?: string }> = {
  workspace: { placement: 'main' },
  'hermes-bots:routines': { placement: 'main', width: '250px' },
  'hermes-newswire:ticker': { placement: 'main', height: '20px' }
}

const ctx: TrackContext = {
  paneFor: id => ({ id, area: 'panes', title: id, data: panes[id] ?? {}, render: () => null }),
  paneGone: () => false,
  overrides: {}
}

describe('fixedTrackSize — main floor across a split axis', () => {
  it('keeps a main-bearing column flex when a minimized rail shares the track (newswire takeover)', () => {
    // Along the column's own axis: [row(main|rail), ticker] must stay FLEX so
    // the chat takes the leftover and the ticker keeps its declared 20px.
    expect(fixedTrackSize(bugTree, 'column', ctx)).toBeNull()
  })

  it('keeps the same column flex along the ROOT row axis — not a fixed 28px sliver', () => {
    // The root row measures the column across the row axis; before the fix the
    // minimized rail's strip made the whole column "fixed" there too.
    expect(fixedTrackSize(bugTree, 'row', ctx)).toBeNull()
  })

  it('still collapses a NO-main split to its rail sizes when all children are fixed', () => {
    // Two fixed tracks with no main pane (the all-fixed rail case) keep
    // sizing from their content — the rule must not loosen this branch.
    const railTree = split('row', [
      group(['a'], { id: 'ga' }),
      group(['b'], { id: 'gb' })
    ])

    const railCtx: TrackContext = {
      ...ctx,
      paneFor: id => ({
        id,
        area: 'panes',
        title: id,
        data: { height: id === 'a' ? '20px' : '40px' },
        render: () => null
      })
    }

    // Across the row's axis (column): both children fixed → max(20px, 40px).
    expect(fixedTrackSize(railTree, 'column', railCtx)).toBe('max(20px, 40px)')
  })

  it('leaves the default right rail unchanged (review | files above the terminal)', () => {
    // No main pane in the right column: the terminal's flex sibling must not
    // demote the rail — cross-axis cssMax stands.
    const rightColumn = split(
      'column',
      [
        split('row', [
          group(['review'], { id: 'grp-review' }),
          group(['files'], { id: 'grp-files' })
        ]),
        group(['terminal'], { id: 'grp-terminal' })
      ],
      [1.6, 1],
      'spl-right'
    )

    const defaultCtx: TrackContext = {
      ...ctx,
      paneFor: id => ({
        id,
        area: 'panes',
        title: id,
        data:
          id === 'review' || id === 'files'
            ? { placement: 'right', width: '237px' }
            : { placement: 'bottom', height: '20vh' },
        render: () => null
      })
    }

    // Row axis (cross for the column): rail children fixed, terminal flexes.
    expect(fixedTrackSize(rightColumn, 'row', defaultCtx)).toBe('calc(237px + 237px)')
  })
})
