import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { triggerHaptic } from '@/lib/haptics'

import { ThreadTimeline } from './timeline'

const mockState = vi.hoisted(() => ({
  messages: [] as Array<{ content: unknown; id: string; role: string }>
}))

vi.mock('@assistant-ui/react', () => ({
  useAuiState: (selector: (state: { thread: { messages: typeof mockState.messages } }) => unknown) =>
    selector({ thread: { messages: mockState.messages } })
}))

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: vi.fn()
}))

const rect = (top: number): DOMRect =>
  ({
    bottom: top,
    height: 0,
    left: 0,
    right: 0,
    toJSON: () => ({}),
    top,
    width: 0,
    x: 0,
    y: top
  }) as DOMRect

describe('ThreadTimeline', () => {
  beforeEach(() => {
    vi.stubGlobal('CSS', undefined)
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
      window.setTimeout(() => callback(performance.now()), 0)
    )
    vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))

    mockState.messages = [
      { id: 'prompt-1', role: 'user', content: [{ type: 'text', text: 'first prompt' }] },
      { id: 'prompt-"quoted"', role: 'user', content: [{ type: 'text', text: 'quoted prompt' }] },
      { id: 'prompt-3', role: 'user', content: [{ type: 'text', text: 'third prompt' }] },
      { id: 'prompt-4', role: 'user', content: [{ type: 'text', text: 'fourth prompt' }] }
    ]
  })

  afterEach(() => {
    cleanup()
    document.body.replaceChildren()
    vi.clearAllMocks()
    vi.unstubAllGlobals()
  })

  it('jumps to a prompt without native CSS.escape support', () => {
    const viewport = document.createElement('div')
    viewport.setAttribute('data-slot', 'aui_thread-viewport')
    viewport.getBoundingClientRect = () => rect(0)

    const target = document.createElement('div')
    target.setAttribute('data-message-id', 'prompt-"quoted"')
    target.getBoundingClientRect = () => rect(80)
    viewport.appendChild(target)
    document.body.appendChild(viewport)

    render(<ThreadTimeline />)

    const [tick] = screen.getAllByRole('button', { name: 'quoted prompt' })

    expect(() => fireEvent.click(tick)).not.toThrow()
    expect(triggerHaptic).toHaveBeenCalledWith('selection')
  })
})
