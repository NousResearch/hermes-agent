import { beforeEach, describe, expect, it, vi } from 'vitest'

// Match review.test.ts: declare the mock with an explicit unused-arg type so
// TypeScript accepts the passthrough wrapper below (vi.fn(async () => ...) is
// inferred as 0-arity and fails tsc when called with the oneshot payload).
const requestOneShot = vi.fn(async (_args: unknown) => 'مرحبا')
vi.mock('@/lib/oneshot', () => ({ requestOneShot: (args: unknown) => requestOneShot(args) }))

import {
  $selectionTranslate,
  closeSelectionTranslate,
  openSelectionTranslate,
  setSelectionTranslateTarget
} from './selection-translate'
import { setSelectionTranslateMode } from './selection-translate-prefs'

describe('selection translate store', () => {
  beforeEach(() => {
    closeSelectionTranslate()
    setSelectionTranslateMode('auto')
    requestOneShot.mockReset()
    requestOneShot.mockResolvedValue('مرحبا')
    window.localStorage.clear()
  })

  it('opens with auto EN→AR and uses a tool-free oneshot outside the chat session', async () => {
    openSelectionTranslate('Hello there')

    expect($selectionTranslate.get()).toMatchObject({
      open: true,
      source: 'Hello there',
      status: 'loading',
      target: 'ar'
    })

    await vi.waitFor(() => expect($selectionTranslate.get().status).toBe('ready'))

    expect(requestOneShot).toHaveBeenCalledOnce()
    expect(requestOneShot).toHaveBeenCalledWith(
      expect.objectContaining({
        input: 'Hello there',
        sessionId: null,
        instructions: expect.stringMatching(/inert source text/i)
      })
    )
    expect($selectionTranslate.get().result).toBe('مرحبا')
  })

  it('defaults Arabic selection to English', async () => {
    requestOneShot.mockResolvedValue('Hello')
    openSelectionTranslate('مرحباً بكم')

    await vi.waitFor(() => expect($selectionTranslate.get().status).toBe('ready'))
    expect($selectionTranslate.get().target).toBe('en')
  })

  it('retargets without losing the source text', async () => {
    openSelectionTranslate('Hello there')
    await vi.waitFor(() => expect($selectionTranslate.get().status).toBe('ready'))

    requestOneShot.mockResolvedValue('Hello again')
    setSelectionTranslateTarget('en')

    await vi.waitFor(() => expect($selectionTranslate.get().result).toBe('Hello again'))
    expect($selectionTranslate.get().source).toBe('Hello there')
    expect($selectionTranslate.get().target).toBe('en')
  })

  it('keeps source visible and surfaces error on failure', async () => {
    requestOneShot.mockRejectedValueOnce(new Error('Gateway not connected'))
    openSelectionTranslate('Hello there')

    await vi.waitFor(() => expect($selectionTranslate.get().status).toBe('error'))
    expect($selectionTranslate.get().source).toBe('Hello there')
    expect($selectionTranslate.get().error).toMatch(/Gateway not connected/)
  })
})
