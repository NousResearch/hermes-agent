import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'
import { $subagentsBySession, type SubagentProgress } from '@/store/subagents'
import type * as WindowsStore from '@/store/windows'

import { DelegateTool } from './delegate'

const openSessionInNewWindow = vi.fn()

vi.mock('@/store/windows', async importOriginal => ({
  ...(await importOriginal<typeof WindowsStore>()),
  openSessionInNewWindow: (...args: unknown[]) => openSessionInNewWindow(...args)
}))

const RUNTIME = 'parent-runtime'
const STORED = 'parent-stored'

const subagent = (): SubagentProgress => ({
  filesRead: [],
  filesWritten: [],
  goal: 'Research docs',
  id: 'sub-1',
  parentId: null,
  sessionId: 'child-stored',
  startedAt: 0,
  status: 'running',
  stream: [],
  taskCount: 1,
  taskIndex: 0,
  updatedAt: 0
})

describe('DelegateTool subagent windows', () => {
  beforeEach(() => {
    openSessionInNewWindow.mockReset()
    $activeGatewayProfile.set('life')
    $activeSessionId.set(RUNTIME)
    $selectedStoredSessionId.set(STORED)
    $subagentsBySession.set({ [RUNTIME]: [subagent()] })
  })

  afterEach(() => {
    cleanup()
    $activeGatewayProfile.set('default')
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $subagentsBySession.set({})
  })

  it('opens a child watch window against the transcript session profile', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <DelegateTool args={{ tasks: [{ goal: 'Research docs' }] }} result={undefined} toolCallId="call-1" />
      </I18nProvider>
    )

    fireEvent.click(screen.getByText('Research docs'))

    expect(openSessionInNewWindow).toHaveBeenCalledWith('child-stored', { profile: 'life', watch: true })
  })
})
