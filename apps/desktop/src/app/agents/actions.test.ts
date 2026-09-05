import { afterEach, expect, it, vi } from 'vitest'

import * as gateway from '@/store/gateway'
import { $selectedStoredSessionId } from '@/store/session'

import { overviewActions } from './actions'
afterEach(() => vi.restoreAllMocks())
import type { AgentRow } from '@/store/agent-overview'

import { createOverviewActions } from './actions'

const row: AgentRow = {
  key: '["remote","coder","same"]',
  id: 'same',
  resolvedId: 'tip',
  runtimeId: 'run',
  history: 'unknown',
  connectionId: 'remote',
  sourceLabel: 'Lab',
  profile: 'coder',
  owner: 'Engineer',
  title: 'Bot Chat',
  description: '',
  preview: '',
  model: '',
  provider: '',
  platform: '',
  attention: 'idle',
  canonical: true,
  stale: false,
  lastActive: 1
}

it('resolves canonical Bot Chat on every action and uses the returned owner-bound runtime for reply and stop', async () => {
  const request = vi.fn(async (_c, _p, method) =>
    method === 'session.list'
      ? { sessions: [{ id: 'new-tip', title: 'Bot Chat' }] }
      : method === 'session.resume'
        ? { session_id: 'new-runtime' }
        : { ok: true }
  )

  const open = vi.fn(async () => {})
  const actions = createOverviewActions({ request, open })
  await actions.reply(row, 'hello')
  expect(request.mock.calls[0]).toEqual([
    'remote',
    'coder',
    'session.list',
    { title: 'Bot Chat', include_hidden: true, profile: 'coder' },
    20_000
  ])
  expect(request).toHaveBeenCalledWith(
    'remote',
    'coder',
    'session.resume',
    { session_id: 'new-tip', profile: 'coder' },
    20_000
  )
  expect(request).toHaveBeenCalledWith(
    'remote',
    'coder',
    'prompt.submit',
    { session_id: 'new-runtime', text: 'hello', profile: 'coder' },
    20_000
  )
  await actions.open(row)
  expect(open).toHaveBeenCalledWith(
    'new-tip',
    expect.objectContaining({
      route: { connectionId: 'remote', profile: 'coder', mode: 'remote', targetProfile: 'coder' },
      forceResume: true
    })
  )
  expect(request.mock.calls.filter(call => call[2] === 'session.list')).toHaveLength(2)
  await actions.stop({ ...row, canonical: false, attention: 'working' })
  expect(request).toHaveBeenLastCalledWith(
    'remote',
    'coder',
    'session.interrupt',
    { session_id: 'run', profile: 'coder' },
    20_000
  )
  expect(request.mock.calls.some(call => call[2] === 'session.create')).toBe(false)
})
it('cancels a late canonical lookup before it can open a conversation', async () => {
  let finish!: (value: unknown) => void

  const request = vi.fn(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )

  const open = vi.fn()
  const actions = createOverviewActions({ request, open })
  let current = true
  const pending = actions.open(row, () => current)
  current = false
  finish({ sessions: [{ id: 'late', title: 'Bot Chat' }] })
  await expect(pending).rejects.toThrow(/superseded/i)
  expect(open).not.toHaveBeenCalled()
})
it('cancels embedded overview opens after newer sidebar navigation even while still mounted', async () => {
  const startHash = window.location.hash
  const startSelection = $selectedStoredSessionId.get()
  let finish!: (value: unknown) => void

  const request = vi.fn(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )

  const open = vi.fn()
  const actions = createOverviewActions({ request, open })

  try {
    window.location.hash = '#/foreground-a'
    $selectedStoredSessionId.set('foreground-a')
    const opening = actions.open(row, () => true)
    window.location.hash = '#/sidebar-newer'
    $selectedStoredSessionId.set('sidebar-newer')
    finish({ sessions: [{ id: 'late', title: 'Bot Chat' }] })
    await expect(opening).rejects.toThrow(/superseded/i)
    expect(open).not.toHaveBeenCalled()

    // The same guard must remain live while the native open awaits its dial.
    request.mockResolvedValue({ sessions: [{ id: 'late', title: 'Bot Chat' }] })
    await actions.open(row, () => true)
    const isCurrent = open.mock.calls[0][1].isCurrent
    expect(isCurrent()).toBe(true)
    window.location.hash = '#/sidebar-even-newer'
    expect(isCurrent()).toBe(false)
  } finally {
    window.location.hash = startHash
    $selectedStoredSessionId.set(startSelection)
  }
})
it('uses the existing turn lease beyond ACK and releases only when submission fails', async () => {
  const release = vi.fn()
  const retain = vi.spyOn(gateway, 'retainGatewayForSessionTurn').mockResolvedValue(release)

  const request = vi
    .spyOn(gateway, 'requestGatewayForAgent')
    .mockImplementation(
      async (_c, _p, method) =>
        (method === 'session.resume' ? { session_id: 'reply-runtime' } : { status: 'streaming' }) as never
    )

  await overviewActions.reply({ ...row, canonical: false }, 'continue')
  expect(request).toHaveBeenCalledWith(
    'remote',
    'coder',
    'session.resume',
    { session_id: 'tip', profile: 'coder' },
    20_000,
    undefined
  )
  expect(request).toHaveBeenCalledWith(
    'remote',
    'coder',
    'prompt.submit',
    { session_id: 'reply-runtime', text: 'continue', profile: 'coder' },
    20_000,
    undefined
  )
  expect(retain).toHaveBeenCalledWith('remote', 'coder', 'reply-runtime')
  expect(release).not.toHaveBeenCalled()
  request.mockImplementation(async (_c, _p, method) => {
    if (method === 'prompt.submit') {
      throw new Error('submission failed')
    }

    return { session_id: 'reply-runtime' } as never
  })
  await expect(overviewActions.reply({ ...row, canonical: false }, 'retry')).rejects.toThrow('submission failed')
  expect(release).toHaveBeenCalledOnce()
})
it('connects only an explicitly selected source without changing the foreground workspace', async () => {
  const connect = vi.spyOn(gateway, 'openGatewayForAgent').mockResolvedValue()
  await overviewActions.connect('parked-ssh')
  expect(connect).toHaveBeenCalledWith('parked-ssh', 'default')
})
it('does not mint missing canonical chats or act on stale runtime claims', async () => {
  const request = vi.fn(async () => ({ sessions: [] }))
  const actions = createOverviewActions({ request, open: vi.fn() })
  await expect(actions.open(row)).rejects.toThrow()
  expect(request).toHaveBeenCalledTimes(1)
  await expect(actions.stop({ ...row, stale: true })).rejects.toThrow()
  expect(request).toHaveBeenCalledTimes(1)
})

it('relaxes hydration only for current known-empty identity, never stale or newly resolved canonical identity', async () => {
  const open = vi.fn(async () => {})
  const request = vi.fn(async () => ({ sessions: [{ id: 'new-tip', title: 'Bot Chat' }] }))
  const actions = createOverviewActions({ request, open })

  for (const [overrides, expectHistory] of [
    [{ canonical: false, history: 'known-empty' }, false],
    [{ canonical: false, history: 'unknown' }, true],
    [{ canonical: false, history: 'known-history' }, true],
    [{ canonical: false, history: 'known-empty', stale: true }, true],
    [{ canonical: true, history: 'known-empty' }, true]
  ] as const) {
    await actions.open({ ...row, ...overrides })
    expect(open).toHaveBeenLastCalledWith(expect.any(String), expect.objectContaining({ expectHistory }))
  }
})
