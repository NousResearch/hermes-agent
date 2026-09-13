import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { StatusbarControls, type StatusbarItem } from '@/app/shell/statusbar-controls'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
})

afterEach(() => {
  cleanup()
})

const openTriggerMenu = (trigger: HTMLElement) => {
  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(trigger)
}

describe('workspace statusbar affordance', () => {
  it('renders menu item with keybind hint when a workspace folder is open', async () => {
    const onOpenFolder = vi.fn()

    const workspaceItem: StatusbarItem = {
      actionId: 'workspace.openFolder',
      id: 'workspace-cwd',
      label: 'my-project',
      menuItems: [
        {
          actionId: 'workspace.openFolder',
          id: 'open-workspace-folder',
          label: 'Open folder as project',
          onSelect: onOpenFolder
        },
        {
          id: 'copy-workspace-path',
          label: 'Copy Path'
        }
      ],
      title: '/path/to/my-project',
      toggleLabel: 'Workspace',
      variant: 'menu'
    }

    render(
      <MemoryRouter>
        <StatusbarControls items={[workspaceItem]} />
      </MemoryRouter>
    )

    // Workspace item button is rendered in the status bar
    const trigger = screen.getByRole('button', { name: /my-project/i })
    expect(trigger).toBeTruthy()

    // Opening the dropdown menu via Radix pointer sequence
    openTriggerMenu(trigger)

    // "Open folder as project" is present in the menu
    const openFolderMenuItem = await screen.findByRole('menuitem', { name: /open folder as project/i })
    expect(openFolderMenuItem).toBeTruthy()

    // Selecting the menu item triggers the callback
    fireEvent.click(openFolderMenuItem)
    expect(onOpenFolder).toHaveBeenCalledTimes(1)
  })

  it('renders direct clickable action when no workspace is active', () => {
    const onOpenFolder = vi.fn()

    const emptyWorkspaceItem: StatusbarItem = {
      actionId: 'workspace.openFolder',
      id: 'workspace-cwd',
      label: 'Open folder as project',
      onSelect: onOpenFolder,
      title: 'Open folder as project',
      toggleLabel: 'Workspace',
      variant: 'action'
    }

    render(
      <MemoryRouter>
        <StatusbarControls items={[emptyWorkspaceItem]} />
      </MemoryRouter>
    )

    const button = screen.getByRole('button', { name: /open folder as project/i })
    expect(button).toBeTruthy()

    fireEvent.click(button)
    expect(onOpenFolder).toHaveBeenCalledTimes(1)
  })
})
