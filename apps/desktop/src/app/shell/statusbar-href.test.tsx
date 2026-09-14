import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { StatusbarControls, type StatusbarItem } from '@/app/shell/statusbar-controls'
import { $statusbarHiddenIds, $statusbarVisible, STATUSBAR_HIDDEN_BY_DEFAULT } from '@/store/statusbar-prefs'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
let openExternal: ReturnType<typeof vi.fn>

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
})

beforeEach(() => {
  openExternal = vi.fn().mockResolvedValue(undefined)
  desktopWindow.hermesDesktop = { openExternal } as unknown as Window['hermesDesktop']
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
  $statusbarVisible.set(true)

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('statusbar external href', () => {
  it('opens a link-variant item href in the OS browser', () => {
    const href = 'https://hermes-agent.nousresearch.com/docs/user-guide/desktop'

    const item: StatusbarItem = {
      href,
      id: 'docs-link',
      label: 'Docs',
      lockedVisible: true,
      variant: 'link'
    }

    render(
      <MemoryRouter>
        <StatusbarControls items={[item]} />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole('link', { name: 'Docs' }))

    expect(openExternal).toHaveBeenCalledWith(href)
  })

  it('opens a dropdown menuItem href in the OS browser without falling back to window.open', async () => {
    const href = 'https://hermes-agent.nousresearch.com/docs/user-guide/desktop#operator'

    const item: StatusbarItem = {
      id: 'ops-menu',
      label: 'Ops',
      lockedVisible: true,
      menuItems: [{ href, id: 'ops-docs', label: 'Operator docs' }],
      variant: 'menu'
    }

    const windowOpen = vi.spyOn(window, 'open').mockReturnValue(null)

    try {
      render(
        <MemoryRouter>
          <StatusbarControls items={[item]} />
        </MemoryRouter>
      )

      const trigger = screen.getByRole('button', { name: 'Ops' })

      fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
      fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
      fireEvent.click(trigger)

      const menuLink = await screen.findByRole('link', { name: 'Operator docs' })

      fireEvent.click(menuLink)

      expect(openExternal).toHaveBeenCalledWith(href)
      expect(windowOpen).not.toHaveBeenCalled()
    } finally {
      windowOpen.mockRestore()
    }
  })
})
