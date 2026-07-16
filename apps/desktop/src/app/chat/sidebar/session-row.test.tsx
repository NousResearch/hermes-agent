import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import type { SessionInfo } from '@/types/hermes'

import { SidebarSessionRow } from './session-row'

vi.mock('./session-actions-menu', () => ({
  SessionActionsMenu: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  SessionContextMenu: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

function session(overrides: Partial<SessionInfo> = {}): SessionInfo {
  const now = Math.floor(Date.now() / 1000)

  return {
    ended_at: now,
    id: 'session-1',
    input_tokens: 0,
    is_active: false,
    last_active: now - 39 * 60,
    message_count: 2,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: now - 60 * 60,
    title: 'Fix risky issue',
    tool_call_count: 0,
    ...overrides
  }
}

function renderRow(props: Partial<React.ComponentProps<typeof SidebarSessionRow>> = {}) {
  const onArchive = vi.fn()

  render(
    <I18nProvider configClient={null} initialLocale="en">
      <SidebarSessionRow
        data-testid="session-row"
        isPinned={false}
        isSelected={false}
        isWorking={false}
        onArchive={onArchive}
        onDelete={vi.fn()}
        onPin={vi.fn()}
        onResume={vi.fn()}
        session={session()}
        {...props}
      />
    </I18nProvider>
  )

  return { onArchive }
}

describe('SidebarSessionRow archive affordance', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('shows the session age by default', () => {
    renderRow()

    expect(screen.getByText('39m').style.opacity).toBe('1')
    expect(screen.getByRole('button', { name: 'Archive' }).style.opacity).toBe('0')
  })

  it('swaps the session age for archive actions on pointer hover', () => {
    renderRow()

    const row = screen.getByTestId('session-row')
    const age = screen.getByText('39m')
    const archive = screen.getByRole('button', { name: 'Archive' })

    fireEvent.pointerEnter(row)

    expect(age.style.opacity).toBe('0')
    expect(archive.style.opacity).toBe('1')

    fireEvent.pointerLeave(row)

    expect(age.style.opacity).toBe('1')
    expect(archive.style.opacity).toBe('0')
  })

  it('swaps the session age for archive actions on keyboard focus', () => {
    renderRow()

    const sessionButton = screen.getByRole('button', { name: 'Fix risky issue' })
    const age = screen.getByText('39m')
    const archive = screen.getByRole('button', { name: 'Archive' })

    fireEvent.focus(sessionButton)

    expect(age.style.opacity).toBe('0')
    expect(archive.style.opacity).toBe('1')

    fireEvent.blur(sessionButton)

    expect(age.style.opacity).toBe('1')
    expect(archive.style.opacity).toBe('0')
  })

  it('archives from the row shortcut button', () => {
    const { onArchive } = renderRow()

    fireEvent.pointerEnter(screen.getByTestId('session-row'))
    fireEvent.click(screen.getByRole('button', { name: 'Archive' }))

    expect(onArchive).toHaveBeenCalledTimes(1)
  })
})
