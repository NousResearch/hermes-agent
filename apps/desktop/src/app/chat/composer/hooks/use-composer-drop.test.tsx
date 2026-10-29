import { act, cleanup, renderHook } from '@testing-library/react'
import type { DragEvent } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $terminalBackend } from '@/store/session'

import { HERMES_PATHS_MIME } from '../../hooks/use-composer-actions'

import { useComposerDrop } from './use-composer-drop'

vi.mock('@/lib/haptics', () => ({ triggerHaptic: () => {} }))

const bridge = Object.getOwnPropertyDescriptor(window, 'hermesDesktop')
const originalBackend = $terminalBackend.get()

afterEach(() => {
  cleanup()
  $terminalBackend.set(originalBackend)

  if (bridge) {
    Object.defineProperty(window, 'hermesDesktop', bridge)
  } else {
    Reflect.deleteProperty(window, 'hermesDesktop')
  }
})

function transfer(internal = false): DataTransfer {
  const file = new File(['synthetic document'], 'report.txt', { type: 'text/plain' })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { getPathForFile: () => '/tmp/drop-fixture/report.txt' }
  })

  return {
    types: internal ? [HERMES_PATHS_MIME] : ['Files'],
    getData: (mime: string) =>
      internal && mime === HERMES_PATHS_MIME ? JSON.stringify([{ path: 'notes/report.txt' }]) : '',
    items: [],
    files: { length: internal ? 0 : 1, item: () => file }
  } as unknown as DataTransfer
}

describe('composer file drops', () => {
  for (const handler of ['handleDrop', 'handleInputDrop'] as const) {
    it.each([
      { sessionId: undefined, backend: 'local', staged: true },
      { sessionId: undefined, backend: 'docker', staged: true },
      { sessionId: 'known-local', backend: 'local', staged: false },
      { sessionId: 'known-container', backend: 'docker', staged: true }
    ])(`${handler}: session=$sessionId backend=$backend stages=$staged`, async ({ sessionId, backend, staged }) => {
      $terminalBackend.set(backend)
      const attach = vi.fn().mockResolvedValue(true)
      const insert = vi.fn().mockReturnValue(true)

      const { result } = renderHook(() =>
        useComposerDrop({
          cwd: '/tmp/drop-fixture',
          sessionId,
          insertInlineRefs: insert,
          onAttachDroppedItems: attach,
          recordUndoPoint: vi.fn(),
          requestMainFocus: vi.fn()
        })
      )

      const event = { dataTransfer: transfer(), preventDefault: vi.fn(), stopPropagation: vi.fn() }
      await act(async () =>
        result.current[handler](event as unknown as DragEvent<HTMLFormElement> & DragEvent<HTMLDivElement>)
      )

      if (staged) {
        expect(attach).toHaveBeenCalledTimes(1)
        expect(attach.mock.calls[0][0][0].file).toBeInstanceOf(File)
        expect(insert).not.toHaveBeenCalled()
      } else {
        expect(insert).toHaveBeenCalledTimes(1)
        expect(attach).not.toHaveBeenCalled()
      }
    })

    it(`${handler}: preserves in-app references in a fresh draft`, async () => {
      $terminalBackend.set('local')
      const attach = vi.fn().mockResolvedValue(true)
      const insert = vi.fn().mockReturnValue(true)

      const { result } = renderHook(() =>
        useComposerDrop({
          cwd: '/tmp/drop-fixture',
          insertInlineRefs: insert,
          onAttachDroppedItems: attach,
          recordUndoPoint: vi.fn(),
          requestMainFocus: vi.fn()
        })
      )

      const event = { dataTransfer: transfer(true), preventDefault: vi.fn(), stopPropagation: vi.fn() }
      await act(async () =>
        result.current[handler](event as unknown as DragEvent<HTMLFormElement> & DragEvent<HTMLDivElement>)
      )
      expect(insert).toHaveBeenCalledTimes(1)
      expect(attach).not.toHaveBeenCalled()
    })
  }
})
