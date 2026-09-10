import { beforeEach, describe, expect, it, vi } from 'vitest'

const togglePane = vi.fn()
const revealTreePane = vi.fn()

vi.mock('./panes', async () => ({
  ...(await vi.importActual('./panes')),
  togglePane
}))

vi.mock('@/components/pane-shell/tree/store', async () => ({
  ...(await vi.importActual('@/components/pane-shell/tree/store')),
  revealTreePane: vi.fn(revealTreePane)
}))

describe('toggleFileBrowserOpen', () => {
  beforeEach(() => {
    togglePane.mockClear()
    revealTreePane.mockClear()
  })

  it('closes the pane even when the tree tab is not the visible active tab', async () => {
    const { toggleFileBrowserOpen } = await import('./layout')

    toggleFileBrowserOpen()

    expect(togglePane).toHaveBeenCalledWith('file-browser')
    expect(revealTreePane).not.toHaveBeenCalled()
  })
})
