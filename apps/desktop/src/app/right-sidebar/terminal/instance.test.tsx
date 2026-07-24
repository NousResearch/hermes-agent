import { render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { AgentTerminalInstance, TerminalInstance } from './instance'

vi.mock('./use-terminal-session', () => ({
  useTerminalSession: () => ({
    addSelectionToChat: vi.fn(),
    hostRef: { current: null },
    selection: '',
    selectionStyle: null,
    status: 'ready'
  })
}))

vi.mock('./use-agent-terminal', () => ({
  useAgentTerminal: () => ({ hostRef: { current: null } })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: { rightSidebar: { addToChat: 'Add to chat' } } })
}))

describe('terminal input markers', () => {
  it('marks a user PTY as interactive', () => {
    const { container } = render(<TerminalInstance active cwd="/tmp" id="terminal-1" onAddSelectionToChat={vi.fn()} />)

    const terminal = container.firstElementChild

    expect(terminal?.hasAttribute('data-terminal')).toBe(true)
    expect(terminal?.hasAttribute('data-interactive-terminal')).toBe(true)
  })

  it('keeps an agent terminal read-only', () => {
    const { container } = render(<AgentTerminalInstance active id="terminal-1" procId="process-1" />)

    const terminal = container.firstElementChild

    expect(terminal?.hasAttribute('data-terminal')).toBe(true)
    expect(terminal?.hasAttribute('data-interactive-terminal')).toBe(false)
  })
})
