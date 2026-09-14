// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { AppContextMenu } from '@/app/context-menu/app-context-menu'
import { $contextMenu } from '@/app/context-menu/store'
import { I18nProvider } from '@/i18n'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

function installBridge() {
  desktopWindow.hermesDesktop = {
    openExternal: vi.fn().mockResolvedValue(undefined),
    writeClipboard: vi.fn().mockResolvedValue(undefined)
  } as unknown as Window['hermesDesktop']
}

const { DocsLink } = await import('./flow')

afterEach(() => {
  cleanup()
  $contextMenu.set(null)
  vi.clearAllMocks()
  delete desktopWindow.hermesDesktop
})

describe('DocsLink context menu', () => {
  it('opens the link menu on right-click', async () => {
    installBridge()
    render(
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <AppContextMenu />
          <DocsLink href="https://docs.github.com/en/copilot">GitHub Copilot (ACP) docs</DocsLink>
        </I18nProvider>
      </MemoryRouter>
    )

    fireEvent.contextMenu(screen.getByText('GitHub Copilot (ACP) docs'))

    expect(await screen.findByText('Open in external browser')).toBeTruthy()
  })
})
