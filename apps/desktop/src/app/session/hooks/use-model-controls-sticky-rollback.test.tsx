import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { confirm } from '@/store/confirm'
import {
  $activeSessionId,
  $currentModel,
  $currentProvider,
  $currentReasoningEffortWire,
  getComposerModelSelection,
  markComposerSelectionManual,
  setConnection,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentModelTransient,
  setCurrentProvider,
  setCurrentProviderTransient,
  setCurrentReasoningEffortWire
} from '@/store/session'

import { deferred } from '../../../test/deferred'

import { useModelControls } from './use-model-controls'
import { desktopSessionCreateParams } from './use-session-actions/session-create-request'

vi.mock('@/store/confirm', () => ({ confirm: vi.fn() }))
vi.mock('@/store/notifications', () => ({
  dismissNotification: vi.fn(),
  notify: vi.fn(),
  notifyError: vi.fn()
}))
vi.mock('@/store/profile', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn(async () => undefined),
  ensureGatewayProfile: vi.fn(async () => undefined)
}))

function seed(sessionId: string) {
  $activeSessionId.set(sessionId)
  setCurrentModel('chosen-model')
  setCurrentProvider('custom:chosen-endpoint')
  markComposerSelectionManual()
  setCurrentModelTransient('fallback-model')
  setCurrentProviderTransient('custom')
  setCurrentReasoningEffortWire('high')
}

async function expectOriginalSelection() {
  expect(getComposerModelSelection()).toEqual({
    model: 'chosen-model',
    provider: 'custom:chosen-endpoint',
    source: 'manual'
  })
  expect($currentModel.get()).toBe('fallback-model')
  expect($currentProvider.get()).toBe('custom')
  expect($currentReasoningEffortWire.get()).toBe('high')
  await expect(desktopSessionCreateParams('')).resolves.toMatchObject({
    model: 'chosen-model',
    provider: 'custom:chosen-endpoint'
  })
}

beforeEach(() => {
  window.localStorage.clear()
  setConnection({ mode: 'local', baseUrl: '' } as never)
  vi.mocked(confirm).mockReset()
})

afterEach(() => {
  cleanup()
  $activeSessionId.set(null)
  setCurrentModel('')
  setCurrentProvider('')
  setCurrentModelSource('')
  setConnection(null)
  window.localStorage.clear()
})

describe('model switch rollback keeps sticky intent separate from runtime display', () => {
  it('restores both snapshots through overlapping failures without undoing a newer successful pick', async () => {
    for (const order of ['newer-first', 'older-first', 'newer-applied']) {
      seed(`overlap-${order}`)
      const older = deferred<unknown>()
      const newer = deferred<unknown>()
      const requestGateway = vi.fn().mockReturnValueOnce(older.promise).mockReturnValueOnce(newer.promise)

      const { result, unmount } = renderHook(() => useModelControls({ queryClient: new QueryClient(), requestGateway }))

      const olderPick = result.current.selectModel({ model: 'older-model', provider: 'custom:older-endpoint' })
      const newerPick = result.current.selectModel({ model: 'newer-model', provider: 'custom:newer-endpoint' })

      if (order === 'older-first') {
        older.reject(new Error('switch refused'))
        await expect(olderPick).resolves.toBe(false)
        newer.reject(new Error('switch refused'))
        await expect(newerPick).resolves.toBe(false)
      } else {
        if (order === 'newer-applied') {
          newer.resolve({})
          await expect(newerPick).resolves.toBe(true)
        } else {
          newer.reject(new Error('switch refused'))
          await expect(newerPick).resolves.toBe(false)
        }

        older.reject(new Error('switch refused'))
        await expect(olderPick).resolves.toBe(false)
      }

      if (order === 'newer-applied') {
        expect(getComposerModelSelection()).toEqual({
          model: 'newer-model',
          provider: 'custom:newer-endpoint',
          source: 'manual'
        })
        expect($currentModel.get()).toBe('newer-model')
      } else {
        await expectOriginalSelection()
      }

      unmount()
    }
  })

  it('keeps the original sticky route during confirmation and restores it if the confirmed retry fails', async () => {
    for (const outcome of ['declined', 'retry-failed', 'retry-applied']) {
      seed(`confirm-${outcome}`)
      const answer = deferred<boolean>()
      const retry = deferred<unknown>()

      const requestGateway = vi
        .fn()
        .mockResolvedValueOnce({ confirm_required: true, confirm_message: 'Confirm this model switch.' })
        .mockReturnValueOnce(retry.promise)

      vi.mocked(confirm).mockReturnValueOnce(answer.promise)

      const { result, unmount } = renderHook(() => useModelControls({ queryClient: new QueryClient(), requestGateway }))

      await expect(
        result.current.selectModel({ model: 'guarded-model', provider: 'custom:guarded-endpoint' })
      ).resolves.toBe(false)
      await expectOriginalSelection()

      await act(async () => answer.resolve(outcome !== 'declined'))

      if (outcome === 'declined') {
        expect(requestGateway).toHaveBeenCalledTimes(1)
        await expectOriginalSelection()
      } else {
        expect(requestGateway).toHaveBeenCalledTimes(2)
        await act(async () => {
          if (outcome === 'retry-applied') {
            retry.resolve({})
          } else {
            retry.reject(new Error('confirmed switch refused'))
          }
        })

        if (outcome === 'retry-applied') {
          expect(getComposerModelSelection()).toEqual({
            model: 'guarded-model',
            provider: 'custom:guarded-endpoint',
            source: 'manual'
          })
        } else {
          await expectOriginalSelection()
        }
      }

      unmount()
    }
  })
})
