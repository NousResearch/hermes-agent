import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $composerSelectModelAction } from '@/store/session'
import type * as WindowsStore from '@/store/windows'

import { usePublishComposerSelectModel } from './wiring-effects'

const windows = vi.hoisted(() => ({ auxiliary: false }))

vi.mock('@/store/windows', async importOriginal => ({
  ...(await importOriginal<typeof WindowsStore>()),
  isAuxiliaryWindow: () => windows.auxiliary
}))

describe('usePublishComposerSelectModel', () => {
  afterEach(() => {
    windows.auxiliary = false
    $composerSelectModelAction.set(null)
  })

  it("publishes the model menu's selectModel for host.composer.selectModel and clears it on unmount", async () => {
    const selectModel = vi.fn(async () => true)
    const { unmount } = renderHook(() => usePublishComposerSelectModel(selectModel))
    const action = $composerSelectModelAction.get()

    expect(action).not.toBeNull()
    await expect(action!({ model: 'm', provider: 'p', sessionId: null })).resolves.toBe(true)
    expect(selectModel).toHaveBeenCalledWith({ model: 'm', provider: 'p', sessionId: null })

    unmount()
    expect($composerSelectModelAction.get()).toBeNull()
  })

  it('normalizes a non-boolean answer to a boolean', async () => {
    renderHook(() => usePublishComposerSelectModel(async () => undefined))

    await expect($composerSelectModelAction.get()!({ model: 'm', provider: 'p' })).resolves.toBe(false)
  })

  it('does not publish from an auxiliary window (only the main window owns the composer menu)', () => {
    windows.auxiliary = true
    renderHook(() => usePublishComposerSelectModel(async () => true))

    expect($composerSelectModelAction.get()).toBeNull()
  })
})
