import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { useSubagentLiveness } from './use-subagent-liveness'

type Request = (method: string, params?: Record<string, unknown>) => Promise<unknown>

const entry = (profile: string, request: Request) => ({ gateway: { request }, profile })

const registry =
  (...entries: ReturnType<typeof entry>[]) =>
  () =>
    entries

const resolves = (value: unknown) =>
  vi.fn((_method: string, _params?: Record<string, unknown>) => Promise.resolve(value))

const rejects = () =>
  vi.fn((_method: string, _params?: Record<string, unknown>) => Promise.reject(new Error('offline')))

const spawn = (sid: string, id: string, profile = 'default') =>
  upsertSubagent(
    sid,
    {
      goal: id,
      status: 'running',
      subagent_id: id,
      task_index: 0
    },
    true,
    undefined,
    profile
  )

describe('useSubagentLiveness', () => {
  beforeEach(() => {
    $subagentsBySession.set({})
    vi.spyOn(Date, 'now').mockReturnValue(1_000)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('drops stale rows after an authoritative empty snapshot', async () => {
    spawn('s1', 'stale')
    vi.mocked(Date.now).mockReturnValue(60_000)
    const request = resolves({ active: [] })

    renderHook(() => useSubagentLiveness(registry(entry('default', request))))

    await waitFor(() => expect(request).toHaveBeenCalledWith('delegation.status', {}))
    await waitFor(() => expect($subagentsBySession.get()).toEqual({}))
  })

  it('keeps rows returned by the backend registry', async () => {
    spawn('s1', 'live')
    vi.mocked(Date.now).mockReturnValue(60_000)
    const request = resolves({ active: [{ subagent_id: 'live' }] })

    renderHook(() => useSubagentLiveness(registry(entry('default', request))))

    await waitFor(() => expect(request).toHaveBeenCalledTimes(1))
    expect($subagentsBySession.get().s1?.map(item => item.id)).toEqual(['live'])
  })

  it('unions live ids across foreground and background profiles', async () => {
    spawn('s1', 'background', 'background')
    vi.mocked(Date.now).mockReturnValue(60_000)
    const foreground = resolves({ active: [] })
    const background = resolves({ active: [{ subagent_id: 'background' }] })

    renderHook(() => useSubagentLiveness(registry(entry('default', foreground), entry('background', background))))

    await waitFor(() => expect(background).toHaveBeenCalledTimes(1))
    expect($subagentsBySession.get().s1?.map(item => item.id)).toEqual(['background'])
  })

  it('reconciles a healthy profile while preserving an unavailable profile', async () => {
    spawn('local', 'local')
    spawn('remote', 'remote', 'remote')
    vi.mocked(Date.now).mockReturnValue(60_000)
    const local = resolves({ active: [] })
    const remote = rejects()

    renderHook(() => useSubagentLiveness(registry(entry('default', local), entry('remote', remote))))

    await waitFor(() => expect(remote).toHaveBeenCalledTimes(1))
    await waitFor(() => expect($subagentsBySession.get().local).toBeUndefined())
    expect($subagentsBySession.get().remote).toHaveLength(1)
  })

  it('does not prune when every backend response is unusable', async () => {
    spawn('s1', 'unknown')
    vi.mocked(Date.now).mockReturnValue(60_000)
    const malformed = resolves({ active: [{}] })

    renderHook(() => useSubagentLiveness(registry(entry('default', malformed))))
    await waitFor(() => expect(malformed).toHaveBeenCalledTimes(1))

    expect($subagentsBySession.get().s1).toHaveLength(1)
  })

  it('does not prune a row that starts after the snapshot request', async () => {
    spawn('old', 'old')
    vi.mocked(Date.now).mockReturnValue(60_000)
    let finish!: (value: unknown) => void

    const request = vi.fn(
      (_method: string, _params?: Record<string, unknown>) => new Promise<unknown>(resolve => (finish = resolve))
    )

    renderHook(() => useSubagentLiveness(registry(entry('default', request))))
    await waitFor(() => expect(request).toHaveBeenCalledTimes(1))
    vi.mocked(Date.now).mockReturnValue(120_000)
    act(() => spawn('new', 'new'))
    vi.mocked(Date.now).mockReturnValue(200_000)
    await act(async () => finish({ active: [] }))

    await waitFor(() => expect(Object.keys($subagentsBySession.get())).toEqual(['new']))
  })
})
