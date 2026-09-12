import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { NodeContextMenu, type NodeMenuTarget } from './node-context-menu'

vi.mock('@/app/learning/archive-skill-confirm-dialog', () => ({
  ArchiveSkillConfirmDialog: () => null,
  fireOptimistic: vi.fn()
}))
vi.mock('@/components/chat/code-editor', () => ({ CodeEditor: () => null }))
vi.mock('@/components/ui/button', () => ({ Button: () => null }))
vi.mock('@/components/ui/confirm-dialog', () => ({ ConfirmDialog: () => null }))
vi.mock('@/components/ui/dialog', () => ({
  Dialog: ({ children }: { children: React.ReactNode }) => children,
  DialogContent: ({ children }: { children: React.ReactNode }) => children,
  DialogFooter: ({ children }: { children: React.ReactNode }) => children,
  DialogHeader: ({ children }: { children: React.ReactNode }) => children,
  DialogTitle: ({ children }: { children: React.ReactNode }) => children
}))
vi.mock('@/hermes', () => ({
  deleteLearningNode: vi.fn(),
  editLearningNode: vi.fn(),
  getLearningNode: vi.fn()
}))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
vi.mock('@/store/starmap', () => ({ evictStarmapNode: vi.fn(), loadStarmapGraph: vi.fn() }))
vi.mock('../hooks/use-on-profile-switch', () => ({ useOnProfileSwitch: vi.fn() }))

const menuWidth = 200
const menuHeight = 160

function setViewport(width: number, height: number) {
  Object.defineProperties(window, {
    innerHeight: { configurable: true, value: height },
    innerWidth: { configurable: true, value: width }
  })
}

function target(x: number, y: number): NodeMenuTarget {
  return { id: 'memory-1', kind: 'memory', label: 'Test memory', x, y }
}

function menu() {
  return screen.getByText('Test memory').parentElement as HTMLDivElement
}

beforeEach(() => {
  setViewport(1024, 768)
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
    bottom: menuHeight,
    height: menuHeight,
    left: 0,
    right: menuWidth,
    toJSON: () => ({}),
    top: 0,
    width: menuWidth,
    x: 0,
    y: 0
  })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('NodeContextMenu viewport positioning', () => {
  it('preserves the exact anchor when the measured menu fits', async () => {
    render(<NodeContextMenu onClose={vi.fn()} onNodeRemoved={vi.fn()} target={target(120, 80)} />)

    await waitFor(() => expect(menu().style.left).toBe('120px'))
    expect(menu().style.top).toBe('80px')
  })

  it('clamps a bottom-right anchor using the measured menu size without going negative', async () => {
    const { rerender } = render(
      <NodeContextMenu onClose={vi.fn()} onNodeRemoved={vi.fn()} target={target(1000, 750)} />
    )

    await waitFor(() => expect(menu().style.left).toBe('824px'))
    expect(menu().style.top).toBe('608px')

    setViewport(120, 100)
    rerender(<NodeContextMenu onClose={vi.fn()} onNodeRemoved={vi.fn()} target={target(110, 90)} />)

    await waitFor(() => expect(menu().style.left).toBe('0px'))
    expect(menu().style.top).toBe('0px')
  })
})
