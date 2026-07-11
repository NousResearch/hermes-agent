import { render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { TerminalPaneChrome } from './chrome'

vi.mock('./persistent', () => ({
  TerminalSlot: () => <div data-testid="terminal-slot" />
}))

vi.mock('./rail', () => ({
  TerminalRail: () => <div data-testid="terminal-rail" />
}))

vi.mock('./terminals', async () => {
  const { atom } = await import('nanostores')

  return { $terminals: atom([]) }
})

describe('TerminalPaneChrome', () => {
  it('establishes a full-size host when mounted inside a non-flex preview body', () => {
    const { container } = render(<TerminalPaneChrome />)
    const root = container.firstElementChild

    expect(root?.classList.contains('h-full')).toBe(true)
    expect(root?.classList.contains('w-full')).toBe(true)
  })
})
