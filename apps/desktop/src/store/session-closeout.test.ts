import { afterEach, expect, it, vi } from 'vitest'

import { captureCloseoutTarget, requestSessionCloseout } from './session-closeout'

vi.mock('./session-request-router', async original => ({
  ...(await original<object>()),
  requestForSessionProfile: (_owner: unknown, request: Function, method: string, params: object) =>
    request(method, params)
}))

afterEach(() => vi.useRealTimers())

const receipt = (id: string) => ({
  closeout_id: `close-${id}`,
  status: 'preserved',
  settled: false,
  old: { logical_session_id: id, stored_session_id: id, profile: 'work' },
  archived: true,
  receipt_id: 'durable',
  destinations: { memory: 'pending', tasks: 'pending' },
  next_create_request_id: 'stable-new'
})

const target = (id: string, request: ReturnType<typeof vi.fn>) =>
  captureCloseoutTarget(id, request as never, {
    destination: {
      scopeKey: `cloud-a:work:${id}`,
      owner: { connectionId: 'cloud-a', profile: 'work' },
      requestGateway: request as never
    }
  })

it('waits for preservation, shares in-flight requests, and replays one retained identity without waiting for physical settlement', async () => {
  vi.useFakeTimers()
  const id = 'polling'

  const request = vi
    .fn()
    .mockResolvedValueOnce({ ...receipt(id), status: 'reviewing', archived: false, receipt_id: null })
    .mockResolvedValue(receipt(id))

  const captured = target(id, request)
  const first = requestSessionCloseout(captured, { intent: 'new' })
  const second = requestSessionCloseout(captured, { intent: 'new' })
  expect(first).toBe(second)
  await vi.advanceTimersByTimeAsync(1000)
  await expect(first).resolves.toMatchObject({
    archived: true,
    settled: false,
    destinations: { memory: 'pending', tasks: 'pending' }
  })
  expect(request.mock.calls.map(call => call[1].request_id)).toEqual([
    expect.any(String),
    request.mock.calls[0]![1].request_id
  ])
})

it('does not accept a receipt from another conversation or an archive flag without durable preservation', async () => {
  const wrong = vi.fn().mockResolvedValue(receipt('another'))
  await expect(requestSessionCloseout(target('expected', wrong))).rejects.toThrow('another session')
  const bare = vi.fn().mockResolvedValue({ ...receipt('bare'), receipt_id: null })
  await expect(requestSessionCloseout(target('bare', bare))).rejects.toThrow('durable archived preservation receipt')
})

it('rejects a definitively released receipt even when its preservation checkpoint remains durable', async () => {
  const request = vi.fn().mockResolvedValue({ ...receipt('released'), archived: false, error: {
    code: 'closeout_reopened', message: 'Closeout was released by explicit unarchive.'
  } })

  await expect(requestSessionCloseout(target('released', request))).rejects.toThrow('released by explicit unarchive')
  request.mockResolvedValue(receipt('released'))
  await requestSessionCloseout(target('released', request))
  expect(request.mock.calls[1]![1].request_id).not.toBe(request.mock.calls[0]![1].request_id)
})

it('retains the request id after an uncertain failure and never falls back to plain archive', async () => {
  const request = vi.fn().mockRejectedValueOnce(new Error('transport timeout')).mockResolvedValue(receipt('retry'))
  const captured = target('retry', request)
  await expect(requestSessionCloseout(captured)).rejects.toThrow('timeout')
  await requestSessionCloseout(captured)
  expect(request.mock.calls.every(call => call[0] === 'session.closeout')).toBe(true)
  expect(request.mock.calls[1]![1].request_id).toBe(request.mock.calls[0]![1].request_id)
})

it('retires a completed identity so restored conversations can preserve a new cutoff', async () => {
  const request = vi.fn().mockResolvedValue(receipt('restored'))
  const captured = target('restored', request)
  await requestSessionCloseout(captured)
  await requestSessionCloseout(captured)
  expect(request.mock.calls[1]![1].request_id).not.toBe(request.mock.calls[0]![1].request_id)
})
