import { expect, test, vi } from 'vitest'

import { ensureLocalGateway } from './local-gateway'

const endpoint = {
  profile_id: '/home', instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1,
  api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none'
}

const pending = { code: 5, stdout: JSON.stringify({ state: 'starting', reason_code: 'deadline' }) }
const ready = { code: 0, stdout: JSON.stringify({ state: 'ready', endpoint }) }
const restartWait = { code: 6, stdout: JSON.stringify({ state: 'draining', reason_code: 'restart_wait', endpoint }) }

test.each([ready, restartWait])('startup attaches to the owner that answers the second ensure ($code)', async verdict => {
  const run = vi.fn().mockResolvedValueOnce(pending).mockResolvedValueOnce(verdict)
  const restart = vi.fn()
  const stillCurrent = vi.fn()

  await expect(ensureLocalGateway(run, undefined, restart, stillCurrent)).resolves.toMatchObject({ gatewayEndpoint: endpoint })
  expect(run).toHaveBeenCalledTimes(2)
  expect(stillCurrent).toHaveBeenCalledTimes(2)
  expect(restart).not.toHaveBeenCalled()
})

test('a second startup deadline surfaces without another ensure', async () => {
  const run = vi.fn().mockResolvedValue(pending)

  await expect(ensureLocalGateway(run, undefined, undefined, vi.fn())).rejects.toThrow('Gateway starting (deadline)')
  expect(run).toHaveBeenCalledTimes(2)
})

test('a terminal verdict from the second ensure reports its own reason', async () => {
  const refused = { state: 'inaccessible', reason_code: 'runtime_exited', detail: 'Owner exited' }
  const run = vi.fn().mockResolvedValueOnce(pending).mockResolvedValueOnce({ code: 7, stdout: JSON.stringify(refused) })

  await expect(ensureLocalGateway(run, undefined, undefined, vi.fn())).rejects.toThrow('Gateway inaccessible (runtime_exited). Owner exited.')
  expect(run).toHaveBeenCalledTimes(2)
})

test('a superseded startup cannot re-ensure its former local route', async () => {
  const run = vi.fn().mockResolvedValueOnce(pending).mockResolvedValueOnce(ready)

  const superseded = () => { throw new Error('superseded') }

  await expect(ensureLocalGateway(run, undefined, undefined, superseded)).rejects.toThrow('superseded')
  expect(run).toHaveBeenCalledOnce()
})

test('switching routes during re-ensure cannot restart the former route’s stale owner', async () => {
  let current = true

  const run = vi.fn().mockResolvedValueOnce(pending).mockImplementationOnce(async () => {
    current = false

    return { code: 0, stdout: JSON.stringify({ state: 'ready', endpoint: { ...endpoint, code_sha: 'old' }, client_code_sha: 'new' }) }
  })

  const restart = vi.fn()

  const assertCurrent = () => { if (!current) { throw new Error('superseded') } }

  await expect(ensureLocalGateway(run, undefined, restart, assertCurrent)).rejects.toThrow('superseded')
  expect(restart).not.toHaveBeenCalled()
})

test.each([
  { code: 4, state: 'inaccessible', reason_code: 'authorization' },
  { code: 6, state: 'draining', reason_code: 'update_paused' },
  { code: 5, state: 'starting', reason_code: 'other' },
  { code: 7, state: 'starting', reason_code: 'deadline' }
])('a $state/$reason_code verdict with exit $code remains terminal', async ({ code, state, reason_code }) => {
  const run = vi.fn().mockResolvedValue({ code, stdout: JSON.stringify({ state, reason_code }) })
  const stillCurrent = vi.fn()

  await expect(ensureLocalGateway(run, undefined, undefined, stillCurrent)).rejects.toThrow(`Gateway ${state} (${reason_code})`)
  expect(run).toHaveBeenCalledOnce()
  expect(stillCurrent).not.toHaveBeenCalled()
})

test('non-startup callers still receive the original deadline', async () => {
  const run = vi.fn().mockResolvedValue(pending)

  await expect(ensureLocalGateway(run)).rejects.toThrow('Gateway starting (deadline)')
  expect(run).toHaveBeenCalledOnce()
})
