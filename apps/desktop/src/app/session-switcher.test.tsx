import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { $attentionSessions, $workingSessions } from '@/store/session'
import { $switcherIndex, $switcherOpen, $switcherSessions } from '@/store/session-switcher'

import { SessionSwitcher } from './session-switcher'

const listedSession = (profile: string, title: string): SessionInfo =>
  ({ id: 'same-id', profile, title }) as SessionInfo

describe('SessionSwitcher profile indicators', () => {
  afterEach(() => {
    cleanup()
    $switcherOpen.set(false)
    $switcherSessions.set([])
    $workingSessions.set([])
    $attentionSessions.set([])
  })

  it('shows working and attention only on the matching same-id profile row', () => {
    Element.prototype.scrollIntoView = vi.fn()
    $switcherOpen.set(true)
    $switcherIndex.set(0)
    $switcherSessions.set([listedSession('default', 'Default row'), listedSession('work', 'Work row')])
    $workingSessions.set([{ profile: 'work', sessionId: 'same-id' }])
    $attentionSessions.set([{ profile: 'default', sessionId: 'same-id' }])

    render(
      <MemoryRouter>
        <SessionSwitcher />
      </MemoryRouter>
    )

    const defaultDot = screen.getByText('Default row').parentElement?.querySelector('span')
    const workDot = screen.getByText('Work row').parentElement?.querySelector('span')

    expect(defaultDot?.className).toContain('bg-amber-400')
    expect(defaultDot?.className).not.toContain('animate-pulse')
    expect(workDot?.className).toContain('animate-pulse')
    expect(workDot?.className).not.toContain('bg-amber-400')
  })
})
