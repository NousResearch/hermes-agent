import { act, renderHook } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

const { request, socket } = vi.hoisted(() => ({ request: vi.fn(), socket: { gateway: null as null | WritableAtom<string> } }))
vi.mock('@hermes/plugin-sdk', async () => {
  const { captureGroupRequests } = await import('./canonical-group-contract-test-utils')
  const { atom } = await import('nanostores')
  socket.gateway = atom('open')

  return { host: { requestProfile: captureGroupRequests(request).request, state: { gateway: socket.gateway } } }
})

import { offersGroupMethod, useGroupMethods } from './canonical-group-capabilities'
import { CANONICAL_GROUP_CAPABILITIES, STANDALONE_GROUP_CAPABILITIES } from './canonical-group-contract-test-utils'

const route = { connectionId: 'remote', profile: 'team' }
afterEach(() => { vi.useRealTimers(); request.mockReset(); socket.gateway?.set('open') })

it('offers no control after a failed capabilities read, and reads again while the room is visible', async () => {
  vi.useFakeTimers()
  request.mockRejectedValueOnce(new Error('request timed out')).mockResolvedValue(STANDALONE_GROUP_CAPABILITIES)
  const { result, rerender } = renderHook(({ visible }) => useGroupMethods(route, visible), { initialProps: { visible: false } })
  await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
  expect(request).not.toHaveBeenCalled()

  rerender({ visible: true })
  await act(async () => { await vi.advanceTimersByTimeAsync(0) })
  expect(request).toHaveBeenCalledOnce()
  expect(['groups.discard', 'groups.disband', 'groups.retry'].map(method => offersGroupMethod(result.current, method))).toEqual([false, false, false])

  await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
  expect(request).toHaveBeenCalledTimes(2)
  expect(['groups.discard', 'groups.disband', 'groups.retry'].map(method => offersGroupMethod(result.current, method))).toEqual([false, true, true])

  // An answer is kept for the room view's lifetime.
  rerender({ visible: false })
  rerender({ visible: true })
  await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
  expect(request).toHaveBeenCalledTimes(2)
})

// An update restarts the gateway: a reopened socket may list other methods (Discard and attachments
// after an upgrade from the standalone server), so each opening reads them again, and a reply that
// started before the socket changed never publishes.
it('reads the methods again when the gateway socket reopens and drops a reply from before', async () => {
  let answerOld: (value: unknown) => void = () => {}
  request.mockResolvedValueOnce(STANDALONE_GROUP_CAPABILITIES)
    .mockImplementationOnce(() => new Promise(resolve => { answerOld = resolve }))
    .mockResolvedValueOnce(CANONICAL_GROUP_CAPABILITIES)
  const { result } = renderHook(() => useGroupMethods(route))
  await act(async () => {})
  expect(offersGroupMethod(result.current, 'groups.discard')).toBe(false)
  expect(offersGroupMethod(result.current, 'groups.disband')).toBe(true)

  // Closed: nothing is offered, and no read goes out until the socket opens again.
  await act(async () => { socket.gateway!.set('closed') })
  expect(result.current).toBeUndefined()
  expect(request).toHaveBeenCalledOnce()

  // A read for one opening that answers after the next change is thrown away.
  await act(async () => { socket.gateway!.set('open') })
  expect(request).toHaveBeenCalledTimes(2)
  await act(async () => { socket.gateway!.set('closed') })
  await act(async () => { answerOld(STANDALONE_GROUP_CAPABILITIES) })
  expect(result.current).toBeUndefined()

  await act(async () => { socket.gateway!.set('open') })
  expect(request).toHaveBeenCalledTimes(3)
  expect(offersGroupMethod(result.current, 'groups.discard')).toBe(true)
})
