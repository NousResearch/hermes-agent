import { beforeEach, describe, expect, it, vi } from 'vitest'

import { revealDesktopPane } from './pane-focus'

const { openReview, revealTreePane, setFileBrowserOpen, setSidebarOpen, setTerminalTakeover } = vi.hoisted(() => ({
  openReview: vi.fn(),
  revealTreePane: vi.fn(),
  setFileBrowserOpen: vi.fn(),
  setSidebarOpen: vi.fn(),
  setTerminalTakeover: vi.fn()
}))

vi.mock('@/app/right-sidebar/store', () => ({ setTerminalTakeover }))
vi.mock('@/components/pane-shell/tree/store', () => ({ revealTreePane }))
vi.mock('./layout', () => ({ setFileBrowserOpen, setSidebarOpen }))
vi.mock('./review', () => ({ openReview }))

describe('revealDesktopPane', () => {
  beforeEach(() => vi.clearAllMocks())

  it("drives each pane's own reveal path", () => {
    revealDesktopPane('chat')
    expect(revealTreePane).toHaveBeenCalledWith('workspace')
    revealDesktopPane('files')
    expect(setFileBrowserOpen).toHaveBeenCalledWith(true)
    revealDesktopPane('review')
    expect(openReview).toHaveBeenCalledOnce()
    revealDesktopPane('sessions')
    expect(setSidebarOpen).toHaveBeenCalledWith(true)
    revealDesktopPane('terminal')
    expect(setTerminalTakeover).toHaveBeenCalledWith(true)
  })

  it('returns false for an unknown pane and touches nothing', () => {
    expect(revealDesktopPane('nope')).toBe(false)
    expect(revealTreePane).not.toHaveBeenCalled()
  })

  it('returns true for a known pane', () => {
    expect(revealDesktopPane('terminal')).toBe(true)
  })
})
