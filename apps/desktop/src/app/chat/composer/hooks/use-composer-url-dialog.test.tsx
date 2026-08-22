import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { useComposerUrlDialog } from './use-composer-url-dialog'

vi.mock('@/lib/haptics', () => ({ triggerHaptic: () => {} }))

// The dialog engine always gets a chip-edit path wired in, even when a given
// surface only uses the attach path — supply inert defaults so the options
// object is complete.
const noopOptions = {
  applyChipEdit: () => true,
  recordChipEditUndo: () => {},
  syncEditorToDraft: () => {}
}

describe('useComposerUrlDialog', () => {
  it('drops an @url: directive into the draft when there is no host onAddUrl', () => {
    const insertText = vi.fn()
    const { result } = renderHook(() => useComposerUrlDialog({ ...noopOptions, insertText }))

    act(() => result.current.setUrlValue('  https://example.dev  '))
    act(() => result.current.submitUrl())

    // The trailing/leading whitespace is trimmed before building the directive.
    expect(insertText).toHaveBeenCalledWith('@url:https://example.dev')
  })

  it('prefers the host onAddUrl handler, then clears + closes the dialog', () => {
    const insertText = vi.fn()
    const onAddUrl = vi.fn()
    const { result } = renderHook(() => useComposerUrlDialog({ ...noopOptions, insertText, onAddUrl }))

    act(() => {
      result.current.openUrlDialog()
      result.current.setUrlValue(' https://example.dev ')
    })
    act(() => result.current.submitUrl())

    expect(onAddUrl).toHaveBeenCalledWith('https://example.dev')
    expect(insertText).not.toHaveBeenCalled()
    expect(result.current.urlValue).toBe('')
    expect(result.current.urlOpen).toBe(false)
  })

  it('no-ops on an empty / whitespace-only URL', () => {
    const insertText = vi.fn()
    const onAddUrl = vi.fn()
    const { result } = renderHook(() => useComposerUrlDialog({ ...noopOptions, insertText, onAddUrl }))

    act(() => result.current.setUrlValue('   '))
    act(() => result.current.submitUrl())

    expect(insertText).not.toHaveBeenCalled()
    expect(onAddUrl).not.toHaveBeenCalled()
  })

  it('rewrites the chip in place on a chip-edit submit, banking undo first', () => {
    const applyChipEdit = vi.fn(() => true)
    const recordChipEditUndo = vi.fn()
    const syncEditorToDraft = vi.fn()
    const insertText = vi.fn()
    const chip = { dataset: {} } as HTMLElement

    const { result } = renderHook(() =>
      useComposerUrlDialog({
        ...noopOptions,
        applyChipEdit,
        insertText,
        recordChipEditUndo,
        syncEditorToDraft
      })
    )

    act(() => result.current.beginChipEdit(chip, 'https://example.dev/a'))
    act(() => result.current.setUrlValue('https://example.dev/b'))
    act(() => result.current.submitUrl())

    expect(recordChipEditUndo).toHaveBeenCalled()
    expect(applyChipEdit).toHaveBeenCalledWith(chip, 'https://example.dev/b')
    expect(syncEditorToDraft).toHaveBeenCalled()
    // Editing never inserts a new directive.
    expect(insertText).not.toHaveBeenCalled()
    expect(result.current.urlOpen).toBe(false)
    expect(result.current.chipEdit).toBeNull()
  })

  it('keeps the dialog open when the chip is gone before the edit lands', () => {
    const applyChipEdit = vi.fn(() => false)
    const recordChipEditUndo = vi.fn()
    const syncEditorToDraft = vi.fn()
    const insertText = vi.fn()
    const chip = { dataset: {} } as HTMLElement

    const { result } = renderHook(() =>
      useComposerUrlDialog({
        ...noopOptions,
        applyChipEdit,
        insertText,
        recordChipEditUndo,
        syncEditorToDraft
      })
    )

    act(() => result.current.beginChipEdit(chip, 'https://example.dev/a'))
    act(() => result.current.setUrlValue('https://example.dev/b'))
    act(() => result.current.submitUrl())

    expect(applyChipEdit).toHaveBeenCalledWith(chip, 'https://example.dev/b')
    // The value survives so the user can attach it or retry, and the dialog stays up.
    expect(result.current.urlOpen).toBe(true)
    expect(result.current.urlValue).toBe('https://example.dev/b')
    expect(syncEditorToDraft).not.toHaveBeenCalled()
  })

  it('does not stack a second chip edit while a dialog is already open', () => {
    const chipA = { dataset: {} } as HTMLElement
    const chipB = { dataset: {} } as HTMLElement
    const { result } = renderHook(() => useComposerUrlDialog({ ...noopOptions, insertText: vi.fn() }))

    act(() => result.current.beginChipEdit(chipA, 'https://example.dev/a'))
    act(() => result.current.beginChipEdit(chipB, 'https://example.dev/b'))

    // The first edit wins; the second is a no-op.
    expect(result.current.chipEdit).toEqual({ chip: chipA, value: 'https://example.dev/a' })
  })
})
