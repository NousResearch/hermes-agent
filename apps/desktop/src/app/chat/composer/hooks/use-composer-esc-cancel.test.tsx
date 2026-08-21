import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useComposerEscCancel } from './use-composer-esc-cancel'
import { resetBinding, setBinding } from '@/store/keybinds'

function Harness({ busy, onCancel }: { busy: boolean; onCancel: () => void }) {
  useComposerEscCancel({ awaitingInput: false, busy, onCancel, target: 'main' })

  return null
}

afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
  resetBinding('composer.cancel')
})

describe('useComposerEscCancel', () => {
  it('cancels the turn on Esc while busy with chat focus', () => {
    const onCancel = vi.fn()
    render(<Harness busy onCancel={onCancel} />)

    fireEvent.keyDown(window, { code: 'Escape', key: 'Escape' })

    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('does not cancel on Esc when the turn is not busy', () => {
    const onCancel = vi.fn()
    render(<Harness busy={false} onCancel={onCancel} />)

    fireEvent.keyDown(window, { code: 'Escape', key: 'Escape' })

    expect(onCancel).not.toHaveBeenCalled()
  })

  it('cancels with a rebound busy-composer key', () => {
    const onCancel = vi.fn()
    setBinding('composer.cancel', ['mod+q'])
    render(<Harness busy onCancel={onCancel} />)

    fireEvent.keyDown(window, { code: 'KeyQ', ctrlKey: true, key: 'q' })

    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('does not cancel on Esc while an overlay surface covers the chat', () => {
    const onCancel = vi.fn()
    render(<Harness busy onCancel={onCancel} />)

    // OverlayView stamps `data-overlay-surface` on its root — the same signal
    // the type-to-focus path uses to stand down while an overlay is open.
    const overlay = document.createElement('div')
    overlay.setAttribute('data-overlay-surface', '')
    document.body.appendChild(overlay)

    fireEvent.keyDown(window, { code: 'Escape', key: 'Escape' })

    expect(onCancel).not.toHaveBeenCalled()
  })
})
