import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $hiddenStripTabs, $layoutTree } from '@/components/pane-shell/tree/store'
import { $workspaceMode } from '@/components/pane-shell/workspace-scope'
import { $panesFlipped, setFileBrowserOpen, setSidebarOpen } from '@/store/layout'

import { TitlebarControls } from './titlebar-controls'

vi.mock('@/store/session-dot-state', async () => {
  const { atom } = await import('nanostores')

  return { $unreadSessionCount: atom(3) }
})

const renderControls = () =>
  render(
    <MemoryRouter>
      <TitlebarControls onOpenSettings={vi.fn()} />
    </MemoryRouter>
  )

beforeEach(() => {
  $workspaceMode.set('sessions')
  $panesFlipped.set(false)
  setSidebarOpen(true)
  setFileBrowserOpen(true)
  $hiddenStripTabs.set(new Set())
  $layoutTree.set(split('row', [group(['sessions', 'bots', 'terminal']), group(['workspace'])]))
})

afterEach(() => {
  cleanup()
})

describe('rendered unread badge ownership', () => {
  it('does not attach a count to the visible sidebar hide button', () => {
    renderControls()

    expect(screen.queryByRole('button', { name: /3 unread sessions/ })).toBeNull()
  })

  it('does not label a Bots workspace toggle with a Sessions count', () => {
    $workspaceMode.set('bots')
    setSidebarOpen(false)
    renderControls()

    expect(screen.queryByRole('button', { name: /3 unread sessions/ })).toBeNull()
  })

  it('does not label a Terminal reveal control with a Sessions count', () => {
    $layoutTree.set(split('row', [group(['sessions', 'terminal'], { active: 'terminal' }), group(['workspace'])]))
    setSidebarOpen(false)
    renderControls()

    expect(screen.queryByRole('button', { name: /3 unread sessions/ })).toBeNull()
  })

  it.each([false, true])('puts one count on the hidden Sessions reveal control (flipped=%s)', flipped => {
    $panesFlipped.set(flipped)
    setSidebarOpen(flipped)
    setFileBrowserOpen(!flipped)
    renderControls()

    const controls = screen.getAllByRole('button', { name: /3 unread sessions/ })

    expect(controls).toHaveLength(1)
    expect(controls[0].textContent).toContain('3')
    expect(controls[0].getAttribute('aria-label')).toMatch(/show/i)
  })
})
