import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $connection } from '@/store/session'

import { FileEntryContextMenu } from './file-actions'

const saveFileAs = vi.fn(async () => ({ canceled: false, path: 'C:\\Downloads\\report.pdf' }))

function renderFileMenu(isDirectory: boolean) {
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <FileEntryContextMenu
        isDirectory={isDirectory}
        name={isDirectory ? 'reports' : 'report.pdf'}
        path={isDirectory ? '/srv/reports' : '/srv/reports/report.pdf'}
      >
        <button type="button">File entry</button>
      </FileEntryContextMenu>
    </I18nProvider>
  )

  fireEvent.contextMenu(screen.getByRole('button', { name: 'File entry' }), { clientX: 10, clientY: 10 })
}

describe('FileEntryContextMenu downloads', () => {
  beforeEach(() => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { saveFileAs }
    })
    vi.stubGlobal(
      'ResizeObserver',
      class {
        disconnect() {}
        observe() {}
        unobserve() {}
      }
    )
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.clearAllMocks()
    $connection.set(null)
  })

  it('offers Download for a remote file', async () => {
    $connection.set({ mode: 'remote' } as never)
    renderFileMenu(false)

    fireEvent.click(await screen.findByRole('menuitem', { name: 'Download…' }))

    await waitFor(() =>
      expect(saveFileAs).toHaveBeenCalledWith({
        path: '/srv/reports/report.pdf',
        profile: undefined,
        remote: true,
        title: 'Download…'
      })
    )
    expect(screen.queryByRole('menuitem', { name: 'Save a Copy…' })).toBeNull()
  })

  it('offers Save a Copy for a local file', async () => {
    $connection.set({ mode: 'local' } as never)
    renderFileMenu(false)

    expect(await screen.findByRole('menuitem', { name: 'Save a Copy…' })).toBeTruthy()
    expect(screen.queryByRole('menuitem', { name: 'Download…' })).toBeNull()
  })

  it('does not offer a download for a directory', async () => {
    $connection.set({ mode: 'remote' } as never)
    renderFileMenu(true)

    expect(await screen.findByRole('menuitem', { name: 'Copy Path' })).toBeTruthy()
    expect(screen.queryByRole('menuitem', { name: 'Download…' })).toBeNull()
  })
})
