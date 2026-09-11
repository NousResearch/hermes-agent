import { afterEach, expect, it, vi } from 'vitest'

import { isFirstBuildSession, markFirstBuildSession } from '@/app/contrib/handoff-receipt'
import { deferred } from '@/test/deferred'

import { $firstBuildConnections, type FirstBuildConnectorPart, openFirstBuildLinks, watchFirstBuildWait } from './first-build-connectors'

const part: FirstBuildConnectorPart = {
  toolCallId: 'wait',
  toolName: 'manage_connections',
  args: { action: 'wait', connectors: ['gmail', 'googlecalendar', 'notion'] }
}

afterEach(() => {
  vi.useRealTimers()
  window.localStorage.clear()
  $firstBuildConnections.set({})
})

it.each(['connected', 'timeout', 'interrupted'])('ends first-build connections after a settled %s wait', async status => {
  markFirstBuildSession('build')
  const request = vi.fn()
  watchFirstBuildWait('build', 'runtime', {
    ...part,
    result: { status, connectors: [{ connector: 'gmail', connected: true }], pending: ['notion'] }
  }, request)

  expect(isFirstBuildSession('build')).toBe(false)
  expect(request).not.toHaveBeenCalled()
  const open = vi.fn()
  await openFirstBuildLinks('build', {
    toolCallId: 'later-connect',
    toolName: 'manage_connections',
    args: { action: 'connect', connectors: ['notion'] },
    result: { results: [{ connector: 'notion', status: 'initiated', connect_url: 'https://connect.example/notion' }] }
  }, { open })
  expect(open).not.toHaveBeenCalled()
  expect($firstBuildConnections.get().build.toolCallId).toBe(part.toolCallId)
})

it('polls through a pending bounce, reconciles unavailable apps, and stops when the part is replaced', async () => {
  vi.useFakeTimers()
  markFirstBuildSession('build')

  const request = vi.fn().mockResolvedValue({
    available: true,
    connectors: [
      { connector: 'gmail', enabled: true, connected: false },
      { connector: 'googlecalendar', enabled: false, connected: false }
    ]
  })

  const stop = watchFirstBuildWait('build', 'runtime', part, request)
  await vi.advanceTimersByTimeAsync(0)
  expect(request).toHaveBeenCalledWith('connectors.list', { session_id: 'runtime' })
  expect($firstBuildConnections.get().build.rows.map(row => [row.phase, row.error])).toEqual([
    ['waiting', undefined],
    ['error', 'unavailable'],
    ['error', 'unavailable']
  ])
  stop?.()

  const stopBounce = watchFirstBuildWait(
    'build',
    'runtime',
    {
      ...part,
      result: { status: 'pending', connectors: [], pending: part.args.connectors }
    },
    request
  )

  request.mockResolvedValue({ available: true, connectors: [{ connector: 'gmail', enabled: true, connected: true }] })
  await vi.advanceTimersByTimeAsync(2000)
  expect($firstBuildConnections.get().build.rows[0].phase).toBe('connected')
  stopBounce?.()
  const calls = request.mock.calls.length
  await vi.advanceTimersByTimeAsync(6000)
  expect(request).toHaveBeenCalledTimes(calls)
})

it.each([{ connector: 'gmail', connected: true, enabled: true }, 'gmail'])(
  'keeps interruption phases and prevents a stale poll from overwriting a settled timeout (%j)', async gmail => {
  vi.useFakeTimers()
  markFirstBuildSession('build')
  const pending = deferred<unknown>()
  const request = vi.fn().mockReturnValue(pending.promise)
  const stop = watchFirstBuildWait('build', 'runtime', part, request)
  stop?.()
  watchFirstBuildWait(
    'build',
    'runtime',
    {
      ...part,
      result: {
        status: 'timeout',
        connectors: [
          gmail,
          { connector: 'googlecalendar', connected: false, enabled: true },
          { connector: 'notion', connected: false, enabled: true }
        ],
        pending: ['googlecalendar', 'notion']
      }
    },
    request
  )
  pending.resolve({ available: true, connectors: [] })
  await vi.advanceTimersByTimeAsync(0)
  expect($firstBuildConnections.get().build.rows.map(row => row.phase)).toEqual(['connected', 'timeout', 'timeout'])

  watchFirstBuildWait(
    'build',
    'runtime',
    {
      ...part,
      result: { status: 'interrupted', connectors: [], pending: part.args.connectors }
    },
    request
  )
  expect($firstBuildConnections.get().build.rows.map(row => row.phase)).toEqual(['connected', 'timeout', 'timeout'])
  expect(request).toHaveBeenCalledTimes(1)
})
