import { describe, expect, it, vi } from 'vitest'

import {
  createPreviewUblockBlockCounter,
  PREVIEW_UBLOCK_BLOCKED_REQUEST_ERROR,
  type PreviewUblockBlockCounterGuest
} from './preview-ublock-block-counter'

function target() {
  const listeners = new Map<string, Array<() => void>>()

  return {
    emit(event: 'destroyed' | 'did-navigate') {
      for (const listener of listeners.get(event) ?? []) {
        listener()
      }
    },
    guest: {
      on(event: 'destroyed' | 'did-navigate', listener: () => void) {
        listeners.set(event, [...(listeners.get(event) ?? []), listener])
      },
      removeListener(event: 'destroyed' | 'did-navigate', listener: () => void) {
        listeners.set(
          event,
          (listeners.get(event) ?? []).filter(candidate => candidate !== listener)
        )
      }
    } as PreviewUblockBlockCounterGuest
  }
}

function counter() {
  let completedListener: ((details: { error?: string; id?: number; webContentsId?: number }) => void) | undefined
  let listener: typeof completedListener
  const onUpdate = vi.fn()
  const owner = target()
  const session = {
    webRequest: {
      onCompleted(next: typeof completedListener) {
        completedListener = next
      },
      onErrorOccurred(next: typeof listener) {
        listener = next
      },
      removeCompletedListener() {
        completedListener = undefined
      },
      removeListener() {
        listener = undefined
      }
    }
  }
  const value = createPreviewUblockBlockCounter({ onUpdate, session })

  return {
    emitCompleted(details: { error?: string; id?: number; webContentsId?: number }) {
      completedListener?.(details)
    },
    emitError(details: { error?: string; webContentsId?: number }) {
      listener?.(details)
    },
    onUpdate,
    owner,
    value
  }
}

describe('preview uBlock blocked request counter', () => {
  it('counts exact blocked-by-client failures only for registered guests', () => {
    const state = counter()

    state.value.registerGuest(12, 99, state.owner.guest)
    state.value.setActive(true)
    state.emitError({ error: PREVIEW_UBLOCK_BLOCKED_REQUEST_ERROR, webContentsId: 12 })
    state.emitError({ error: 'net::ERR_FAILED', webContentsId: 12 })
    state.emitError({ error: PREVIEW_UBLOCK_BLOCKED_REQUEST_ERROR, webContentsId: 13 })

    expect(state.onUpdate).toHaveBeenLastCalledWith({
      blockedRequestCount: 1,
      ownerWebContentsId: 99,
      webContentsId: 12
    })
  })

  it('also counts the same cancellation when Electron reports it as completed with an error', () => {
    const state = counter()

    state.value.registerGuest(12, 99, state.owner.guest)
    state.value.setActive(true)
    state.emitCompleted({ error: PREVIEW_UBLOCK_BLOCKED_REQUEST_ERROR, id: 44, webContentsId: 12 })
    state.emitCompleted({ error: PREVIEW_UBLOCK_BLOCKED_REQUEST_ERROR, id: 44, webContentsId: 12 })

    expect(state.onUpdate).toHaveBeenLastCalledWith({
      blockedRequestCount: 1,
      ownerWebContentsId: 99,
      webContentsId: 12
    })
  })

  it('reloads registered guests when uBlock becomes active', () => {
    const state = counter()
    const reload = vi.fn()

    state.owner.guest.reload = reload
    state.value.registerGuest(12, 99, state.owner.guest)
    state.value.setActive(true)

    expect(reload).toHaveBeenCalledOnce()
  })

  it('resets on top-level navigation and publishes zero when disabled', () => {
    const state = counter()

    state.value.registerGuest(12, 99, state.owner.guest)
    state.value.setActive(true)
    state.emitError({ error: PREVIEW_UBLOCK_BLOCKED_REQUEST_ERROR, webContentsId: 12 })
    state.owner.emit('did-navigate')

    expect(state.onUpdate).toHaveBeenLastCalledWith({
      blockedRequestCount: 0,
      ownerWebContentsId: 99,
      webContentsId: 12
    })

    state.emitError({ error: PREVIEW_UBLOCK_BLOCKED_REQUEST_ERROR, webContentsId: 12 })
    state.value.setActive(false)
    expect(state.onUpdate).toHaveBeenLastCalledWith({
      blockedRequestCount: 0,
      ownerWebContentsId: 99,
      webContentsId: 12
    })
  })

  it('clears a guest when it is destroyed or unregistered', () => {
    const state = counter()

    state.value.registerGuest(12, 99, state.owner.guest)
    state.value.setActive(true)
    state.emitError({ error: PREVIEW_UBLOCK_BLOCKED_REQUEST_ERROR, webContentsId: 12 })
    state.owner.emit('destroyed')
    state.emitError({ error: PREVIEW_UBLOCK_BLOCKED_REQUEST_ERROR, webContentsId: 12 })

    expect(state.onUpdate).toHaveBeenLastCalledWith({
      blockedRequestCount: 0,
      ownerWebContentsId: 99,
      webContentsId: 12
    })
  })
})
