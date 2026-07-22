import { describe, expect, it } from 'vitest'

import { resolveTitlebarEdgeTargets } from './titlebar-controls'

describe('resolveTitlebarEdgeTargets', () => {
  it('keeps the original edge bindings in left-to-right locales', () => {
    expect(resolveTitlebarEdgeTargets(false)).toEqual({
      left: 'sessions',
      right: 'files'
    })
  })

  it('mirrors the edge bindings when the pane row is right-to-left', () => {
    expect(resolveTitlebarEdgeTargets(true)).toEqual({
      left: 'files',
      right: 'sessions'
    })
  })
})
