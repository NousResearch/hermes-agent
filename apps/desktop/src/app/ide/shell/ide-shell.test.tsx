// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'

import { $ideEditor, closeIdeFile, openIdeFile } from '../regions/editor/tabs'
import { $ideWorkspaceRoot, setIdeWorkspaceRoot } from '../state'

import { $ideLayout, IDE_LAYOUT_DEFAULTS } from './ide-layout'
import { IdeShell } from './ide-shell'

function renderShell() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <IdeShell />
    </I18nProvider>
  )
}

const region = (name: string) => screen.queryByRole('region', { name })

beforeEach(() => {
  window.localStorage.clear()
  $ideLayout.set(IDE_LAYOUT_DEFAULTS)
  setIdeWorkspaceRoot(null)
})

afterEach(() => {
  cleanup()
  $ideEditor.set({ activePath: null, openPaths: [] })
  $ideWorkspaceRoot.set(null)
})

describe('IdeShell', () => {
  it('renders the explorer, chat, and browser regions — and no empty editor column', () => {
    renderShell()

    expect(region('Explorer')).toBeTruthy()
    expect(region('Editor')).toBeNull()
    expect(region('Chat')).toBeTruthy()
    expect(region('Browser')).toBeTruthy()
    expect(screen.getByText('No workspace open')).toBeTruthy()
    expect(screen.queryByText('No file open')).toBeNull()
    expect(screen.getByText('No IDE session yet')).toBeTruthy()
    expect(screen.getByText('No page open')).toBeTruthy()
  })

  it('brings the editor column in when a file opens and removes it when the last tab closes', () => {
    openIdeFile('D:/scratch/hello.txt')
    const view = renderShell()

    expect(region('Editor')).toBeTruthy()
    // With an editor above it the browser is docked: it has a height to be
    // resized against, so the split handle exists.
    expect(screen.getByRole('separator', { name: 'Resize browser' })).toBeTruthy()

    closeIdeFile('D:/scratch/hello.txt')
    view.rerender(
      <I18nProvider configClient={null} initialLocale="en">
        <IdeShell />
      </I18nProvider>
    )

    expect(region('Editor')).toBeNull()
  })

  it('gives the browser the whole column while no editor is open', () => {
    renderShell()

    // No editor: nothing to size the browser against, so the handle is gone
    // and the region stretches (asserted structurally; geometry in the live
    // probe).
    expect(region('Browser')).toBeTruthy()
    expect(screen.queryByRole('separator', { name: 'Resize browser' })).toBeNull()
  })

  it('shows the seeded workspace (basename) and keeps the full path in the status bar', () => {
    setIdeWorkspaceRoot('D:\\My apps\\COAI')

    renderShell()

    expect(screen.queryByText('No workspace open')).toBeNull()
    // The title bar and the explorer row both read the basename…
    expect(screen.getAllByText('COAI').length).toBeGreaterThanOrEqual(2)
    // …while the full path appears exactly once, in the status bar.
    expect(screen.getByText('D:\\My apps\\COAI')).toBeTruthy()
  })

  it('toggles a region off from the status bar', () => {
    renderShell()

    fireEvent.click(screen.getByRole('button', { name: 'Toggle explorer' }))

    expect(region('Explorer')).toBeNull()
    expect(screen.getByRole('button', { name: 'Toggle explorer' }).getAttribute('aria-pressed')).toBe('false')
    expect(region('Chat')).toBeTruthy()
  })
})
