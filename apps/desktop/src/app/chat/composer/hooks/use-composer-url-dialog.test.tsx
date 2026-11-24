import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { useComposerUrlDialog } from './use-composer-url-dialog'

vi.mock('@/lib/haptics', () => ({ triggerHaptic: () => {} }))

describe('useComposerUrlDialog', () => {
  it('drops an @url: directive into the draft when there is no host onAddUrl', () => {
    const insertText = vi.fn()
    const { result } = renderHook(() => useComposerUrlDialog({ insertText }))

    act(() => result.current.setUrlValue('  https://example.dev  '))
    act(() => result.current.submitUrl())

    // The trailing/leading whitespace is trimmed before building the directive.
    expect(insertText).toHaveBeenCalledWith('@url:https://example.dev')
  })

  it('prefers the host onAddUrl handler, then clears + closes the dialog', () => {
    const insertText = vi.fn()
    const onAddUrl = vi.fn()
    const { result } = renderHook(() => useComposerUrlDialog({ insertText, onAddUrl }))

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
    const { result } = renderHook(() => useComposerUrlDialog({ insertText, onAddUrl }))

    act(() => result.current.setUrlValue('   '))
    act(() => result.current.submitUrl())

    expect(insertText).not.toHaveBeenCalled()
    expect(onAddUrl).not.toHaveBeenCalled()
  })
})
