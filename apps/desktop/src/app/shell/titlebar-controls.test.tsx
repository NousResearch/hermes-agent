import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as PaneShellTreeStore from '@/components/pane-shell/tree/store'
import { $hapticsMuted } from '@/store/haptics'
import type * as HudStore from '@/store/hud'

import { TitlebarControls } from './titlebar-controls'

afterEach(() => {
  cleanup()
  $hapticsMuted.set(false)
})

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      shell: {
        appControls: 'App controls',
        windowControls: 'Window controls'
      },
      titlebar: {
        enterHud: 'Enter HUD',
        hideRightSidebar: 'Hide right sidebar',
        hideSidebar: 'Hide sidebar',
        layoutEditor: 'Layout editor',
        layoutEditorTitle: (modifier: string) => `Layout editor (${modifier})`,
        muteHaptics: 'Mute haptics',
        openSettings: 'Open settings',
        showRightSidebar: 'Show right sidebar',
        showSidebar: 'Show sidebar',
        swapSidebarSides: 'Swap sidebar sides',
        unmuteHaptics: 'Unmute haptics',
        unreadSessions: (count: number) => `${count} unread`
      }
    }
  })
}))

vi.mock('@/app/hud/handoff', () => ({ hudTargetSessionId: () => null }))
vi.mock('@/components/pane-shell/edit-mode', () => ({ toggleLayoutEditMode: vi.fn() }))
vi.mock('@/components/pane-shell/tree/store', async importOriginal => ({
  ...(await importOriginal<typeof PaneShellTreeStore>()),
  resetLayoutTree: vi.fn()
}))
vi.mock('@/store/hud', async importOriginal => ({
  ...(await importOriginal<typeof HudStore>()),
  toggleHud: vi.fn()
}))

function renderTitlebar() {
  return render(
    <MemoryRouter>
      <TitlebarControls onOpenSettings={vi.fn()} />
    </MemoryRouter>
  )
}

describe('TitlebarControls haptics toggle', () => {
  it('renders a mute-haptics button and flips $hapticsMuted when clicked', () => {
    renderTitlebar()

    const button = screen.getByRole('button', { name: 'Mute haptics' })
    expect($hapticsMuted.get()).toBe(false)

    fireEvent.click(button)

    expect($hapticsMuted.get()).toBe(true)
  })

  it('relabels to unmute once haptics are muted', () => {
    $hapticsMuted.set(true)
    renderTitlebar()

    expect(screen.getByRole('button', { name: 'Unmute haptics' })).toBeTruthy()
  })
})
