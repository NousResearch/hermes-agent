import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/components/pane-shell/tree/store', () => ({
  revealTreePane: vi.fn()
}))

import { $canvasTabs, closeCanvasTile, openCanvasTile } from './canvas-tile'

afterEach(() => {
  closeCanvasTile('pen')
})

describe('openCanvasTile', () => {
  it('keeps one pane per provider when the live .pen changes', () => {
    openCanvasTile({ docId: 'a', provider: 'pen', title: 'A', url: 'https://app.pen.dev/new?embed' })
    openCanvasTile({ docId: 'b', provider: 'pen', title: 'B', url: 'https://app.pen.dev/new?embed' })

    expect($canvasTabs.get()).toEqual([
      { docId: 'b', provider: 'pen', title: 'B', url: 'https://app.pen.dev/new?embed' }
    ])
  })
})
