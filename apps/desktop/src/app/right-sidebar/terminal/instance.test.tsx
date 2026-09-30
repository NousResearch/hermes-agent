import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $currentCwd } from '@/store/session'

import { AgentTerminalInstance, TerminalInstance } from './instance'
import { $activeTerminalId, $terminals } from './terminals'

vi.mock('./use-agent-terminal', () => ({
  useAgentTerminal: () => ({ hostRef: { current: null } })
}))

vi.mock('./use-terminal-session', () => ({
  useTerminalSession: () => ({ hostRef: { current: null }, selection: '', status: 'ready' })
}))

describe('background terminal input guidance (#108233)', () => {
  beforeEach(() => {
    $terminals.set([{ id: 'agent-tab', title: 'installer', auto: false, cwd: '', kind: 'agent', procId: 'process-1' }])
    $activeTerminalId.set('agent-tab')
    $currentCwd.set('/workspace/project')
  })

  afterEach(() => {
    cleanup()
    $terminals.set([])
    $activeTerminalId.set(null)
    $currentCwd.set('')
  })

  it('explains the mirror and opens a separate interactive shell only when requested', () => {
    const original = $terminals.get()[0]
    render(<AgentTerminalInstance active id="agent-tab" procId="process-1" />)

    expect(screen.getByText(en.rightSidebar.terminalReadOnly)).toBeTruthy()
    expect(screen.getByText(en.rightSidebar.terminalReadOnlyHelp)).toBeTruthy()
    expect($terminals.get()).toEqual([original])

    fireEvent.click(screen.getByRole('button', { name: en.rightSidebar.terminalOpenInteractive }))

    const shell = $terminals.get().find(term => term.id === $activeTerminalId.get())
    expect(shell).toMatchObject({ kind: 'user', cwd: '/workspace/project', auto: true })
    expect(shell?.procId).toBeUndefined()
    expect($terminals.get()).toHaveLength(2)
    expect($terminals.get()[0]).toBe(original)
  })

  it('does not create shells or change selection on background mount and does not label user shells read-only', () => {
    const view = render(<AgentTerminalInstance active={false} id="agent-tab" procId="process-1" />)
    expect($activeTerminalId.get()).toBe('agent-tab')
    expect($terminals.get()).toHaveLength(1)

    view.rerender(<TerminalInstance active cwd="/workspace/project" id="user-tab" onAddSelectionToChat={vi.fn()} />)
    expect(screen.queryByText(en.rightSidebar.terminalReadOnly)).toBeNull()
    expect(screen.queryByRole('button', { name: en.rightSidebar.terminalOpenInteractive })).toBeNull()
    expect($terminals.get()).toHaveLength(1)
  })
})
