// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

const confirmMock = vi.fn()

vi.mock('@/store/confirm', () => ({
  confirm: (...args: unknown[]) => confirmMock(...args)
}))

vi.mock('./file-editor', () => ({
  IdeFileEditor: ({ path }: { path: string }) => <div data-testid="file-editor">{path}</div>
}))

import { $ideDirtyPaths, $ideEditor, openIdeFile, setIdeFileDirty } from './tabs'

import { EditorRegion } from './index'

function renderRegion() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <EditorRegion />
    </I18nProvider>
  )
}

beforeEach(() => {
  window.localStorage.clear()
  $ideEditor.set({ activePath: null, openPaths: [] })
  $ideDirtyPaths.set([])
  confirmMock.mockReset()
})

afterEach(() => {
  cleanup()
})

describe('EditorRegion', () => {
  it('shows the empty state with no open files', () => {
    renderRegion()

    expect(screen.getByText('No file open')).toBeTruthy()
  })

  it('renders a tab per open file and mounts the active editor', () => {
    openIdeFile('/repo/a.ts')
    openIdeFile('/repo/b.ts')

    renderRegion()

    expect(screen.getByRole('tablist')).toBeTruthy()
    expect(screen.getByText('a.ts')).toBeTruthy()
    expect(screen.getByText('b.ts')).toBeTruthy()
    expect(screen.getByTestId('file-editor').textContent).toBe('/repo/b.ts')
  })

  it('activates another tab on click', () => {
    openIdeFile('/repo/a.ts')
    openIdeFile('/repo/b.ts')

    renderRegion()
    fireEvent.click(screen.getByText('a.ts'))

    expect(screen.getByTestId('file-editor').textContent).toBe('/repo/a.ts')
  })

  it('asks before closing a dirty tab and keeps it when declined', async () => {
    openIdeFile('/repo/a.ts')
    setIdeFileDirty('/repo/a.ts', true)
    confirmMock.mockResolvedValue(false)

    renderRegion()
    fireEvent.click(screen.getByLabelText('Close a.ts'))

    await waitFor(() => expect(confirmMock).toHaveBeenCalledTimes(1))
    expect(screen.getByText('a.ts')).toBeTruthy()
  })

  it('closes a dirty tab when the confirm is accepted', async () => {
    openIdeFile('/repo/a.ts')
    setIdeFileDirty('/repo/a.ts', true)
    confirmMock.mockResolvedValue(true)

    renderRegion()
    fireEvent.click(screen.getByLabelText('Close a.ts'))

    await waitFor(() => expect(screen.queryByText('a.ts')).toBeNull())
    expect($ideDirtyPaths.get()).toEqual([])
  })

  it('closes a clean tab without asking', () => {
    openIdeFile('/repo/a.ts')

    renderRegion()
    fireEvent.click(screen.getByLabelText('Close a.ts'))

    expect(confirmMock).not.toHaveBeenCalled()
    expect(screen.queryByText('a.ts')).toBeNull()
  })
})
