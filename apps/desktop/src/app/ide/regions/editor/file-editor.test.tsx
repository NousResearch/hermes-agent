// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

const readMock = vi.fn()
const writeMock = vi.fn()

vi.mock('@/lib/desktop-fs', () => ({
  readDesktopFileText: (...args: unknown[]) => readMock(...args),
  writeDesktopFileText: (...args: unknown[]) => writeMock(...args)
}))

vi.mock('@/components/chat/code-editor', () => ({
  CodeEditor: ({
    initialValue,
    onChange,
    onSave
  }: {
    initialValue: string
    onChange: (value: string) => void
    onSave?: () => void
  }) => (
    <div>
      <div data-testid="editor-initial">{initialValue}</div>
      <button onClick={() => onChange('EDITED')} type="button">
        type
      </button>
      <button onClick={() => onSave?.()} type="button">
        save
      </button>
    </div>
  )
}))

import { IdeFileEditor } from './file-editor'
import { $ideDirtyPaths } from './tabs'

beforeEach(() => {
  readMock.mockReset()
  writeMock.mockReset()
  $ideDirtyPaths.set([])
})

afterEach(() => {
  cleanup()
})

const renderEditor = () =>
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <IdeFileEditor path="/repo/a.ts" />
    </I18nProvider>
  )

describe('IdeFileEditor', () => {
  it('loads the file, tracks edits, and saves them', async () => {
    readMock.mockResolvedValue({ path: '/repo/a.ts', text: 'ORIGINAL', truncated: false })
    writeMock.mockResolvedValue({ path: '/repo/a.ts' })

    renderEditor()

    await screen.findByTestId('editor-initial')
    fireEvent.click(screen.getByText('type'))
    expect($ideDirtyPaths.get()).toContain('/repo/a.ts')

    fireEvent.click(screen.getByText('save'))

    await waitFor(() => expect(writeMock).toHaveBeenCalledWith('/repo/a.ts', 'EDITED'))
    await waitFor(() => expect($ideDirtyPaths.get()).not.toContain('/repo/a.ts'))
  })

  it('blocks a save when the disk changed underneath and offers overwrite', async () => {
    // Default covers the mount reads; the queued once-value lands on the save's
    // stale-on-disk check.
    readMock.mockResolvedValue({ text: 'ORIGINAL', truncated: false })
    writeMock.mockResolvedValue({ path: '/repo/a.ts' })

    renderEditor()

    await screen.findByTestId('editor-initial')
    fireEvent.click(screen.getByText('type'))
    readMock.mockResolvedValueOnce({ text: 'CHANGED-ELSEWHERE', truncated: false })
    fireEvent.click(screen.getByText('save'))

    await screen.findByText('This file changed on disk')
    expect(writeMock).not.toHaveBeenCalled()

    readMock.mockResolvedValue({ text: 'CHANGED-ELSEWHERE', truncated: false })
    fireEvent.click(screen.getByText('Overwrite'))

    await waitFor(() => expect(writeMock).toHaveBeenCalledWith('/repo/a.ts', 'EDITED'))
  })

  it('reloads from disk and drops the edit when asked', async () => {
    readMock.mockResolvedValue({ text: 'ORIGINAL', truncated: false })

    renderEditor()

    await screen.findByTestId('editor-initial')
    fireEvent.click(screen.getByText('type'))
    readMock.mockResolvedValueOnce({ text: 'CHANGED-ELSEWHERE', truncated: false })
    fireEvent.click(screen.getByText('save'))
    await screen.findByText('This file changed on disk')

    readMock.mockResolvedValue({ text: 'CHANGED-ELSEWHERE', truncated: false })
    fireEvent.click(screen.getByText('Reload from disk'))

    await waitFor(() => expect(screen.getByTestId('editor-initial').textContent).toBe('CHANGED-ELSEWHERE'))
    expect(writeMock).not.toHaveBeenCalled()
  })

  it('shows the too-large state for truncated reads', async () => {
    readMock.mockResolvedValue({ text: '', truncated: true })

    renderEditor()

    await screen.findByText('File is too large to edit here')
  })

  it('surfaces a load failure honestly', async () => {
    readMock.mockRejectedValue(new Error('ENOENT: gone'))

    renderEditor()

    await screen.findByText('Could not open this file')
    expect(screen.getByText('ENOENT: gone')).toBeTruthy()
  })
})
