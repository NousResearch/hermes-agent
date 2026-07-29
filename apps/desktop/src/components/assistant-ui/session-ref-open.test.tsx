import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { __resetSessionLinkTitleCache } from '@/lib/session-link-title'

import { DirectiveContent } from './directive-text'
import { MarkdownTextContent } from './markdown-text'

const openSession = vi.fn()

vi.mock('@/app/open-session', () => ({
  openSession: (...args: unknown[]) => openSession(...args)
}))

afterEach(() => {
  cleanup()
  openSession.mockClear()
  __resetSessionLinkTitleCache()
})

// Both surfaces render a session ref differently — an inline link in agent
// prose, a chip in the user's own message — but either one opens the session
// it names via the shared door (focus if on screen, else a stacked tab).
describe('session refs open the session', () => {
  it('opens the session from an agent-written link', async () => {
    render(<MarkdownTextContent isRunning={false} text="Picked up in @session:work/20260101_abc123 last night." />)

    fireEvent.click(await screen.findByTitle('work/20260101_abc123'))

    await vi.waitFor(() => expect(openSession).toHaveBeenCalledWith('20260101_abc123', expect.any(Function), 'tab'))
  })

  it('opens the session from a chip in the user transcript', async () => {
    render(<DirectiveContent text="pick up @session:work/20260101_abc123 please" />)

    const chip = screen.getByTitle('work/20260101_abc123')

    expect(chip.tagName).toBe('BUTTON')
    fireEvent.click(chip)

    await vi.waitFor(() => expect(openSession).toHaveBeenCalledWith('20260101_abc123', expect.any(Function), 'tab'))
  })
})
