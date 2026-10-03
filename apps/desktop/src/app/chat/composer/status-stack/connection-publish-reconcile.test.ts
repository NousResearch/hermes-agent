import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import * as gateway from '@/store/gateway'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { reconcileSubagentsOnConnectionPublish } from './use-subagent-snapshot'

// A connection publish (boot, reconnect, soft profile swap) must clear live
// rows whose child already finished while no pane was polling them (#106686's
// hidden keep-alive tiles skip in-tick polls) — one race-guarded list per
// session with non-terminal rows, no timer, exactly once per publish.
const SID = 'sess-conn-reconcile'
const OWNER = { connectionId: 'remote-owner', profile: 'research' } as const

describe('reconcileSubagentsOnConnectionPublish', () => {
  const request = vi.fn(async (_c: string, _p: string, method: string, _params: Record<string, unknown>) => {
    if (method === 'subagent.list') {
      return { subagents: [] }
    }

    return {}
  })

  const listCalls = () => request.mock.calls.filter(([, , method]) => method === 'subagent.list')

  beforeEach(() => {
    request.mockClear()
    vi.spyOn(gateway, 'requestGatewayForAgent').mockImplementation(request as never)
    setSessionOwnerHint(SID, OWNER)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    $subagentsBySession.set({})
    _resetSessionOwnerHintsForTests()
  })

  it('drops a stale running row on an owned session with exactly one list call', async () => {
    upsertSubagent(SID, { goal: 'frozen child', status: 'running', subagent_id: 'ghost', task_index: 0 })

    await reconcileSubagentsOnConnectionPublish()

    // The roster holds nothing for this session: the live-looking row is gone.
    expect($subagentsBySession.get()[SID]).toEqual([])
    expect(listCalls()).toHaveLength(1)
    expect(listCalls()[0]?.[3]).toMatchObject({ session_id: SID })
  })

  it('preserves terminal rows and issues no request for a session with none live', async () => {
    setSessionOwnerHint('terminal-only', OWNER)
    upsertSubagent(SID, { goal: 'frozen child', status: 'running', subagent_id: 'ghost', task_index: 0 })
    upsertSubagent(SID, { goal: 'finished child', status: 'completed', subagent_id: 'done', task_index: 1 })
    upsertSubagent('terminal-only', { goal: 'history', status: 'completed', subagent_id: 'done-2', task_index: 0 })

    await reconcileSubagentsOnConnectionPublish()

    const remaining = ($subagentsBySession.get()[SID] ?? []).map(row => row.id)

    expect(remaining.sort()).toEqual(['done'])
    // Only the session with a live row is asked.
    expect(listCalls()).toHaveLength(1)
    expect(listCalls()[0]?.[3]).toMatchObject({ session_id: SID })
  })

  it('skips an unowned session without issuing a request, still reconciling its owned sibling', async () => {
    // 'orphan-sess' has no owner recorded: requestForOwnedSession rejects.
    upsertSubagent('orphan-sess', { goal: 'orphan child', status: 'running', subagent_id: 'orphan', task_index: 0 })
    upsertSubagent(SID, { goal: 'frozen child', status: 'running', subagent_id: 'ghost', task_index: 0 })

    await reconcileSubagentsOnConnectionPublish()

    expect($subagentsBySession.get()['orphan-sess']).toHaveLength(1)
    expect($subagentsBySession.get()[SID]).toEqual([])
    expect(listCalls()).toHaveLength(1)
    expect(listCalls()[0]?.[3]).toMatchObject({ session_id: SID })
  })

  it('discards a response whose owner changed while the request was in flight', async () => {
    upsertSubagent(SID, { goal: 'frozen child', status: 'running', subagent_id: 'ghost', task_index: 0 })

    let release: (value: { subagents: unknown[] }) => void = () => undefined

    request.mockImplementationOnce(
      async () =>
        new Promise(resolve => {
          release = resolve
        }) as never
    )

    const run = reconcileSubagentsOnConnectionPublish()
    await vi.waitFor(() => expect(listCalls()).toHaveLength(1))

    setSessionOwnerHint(SID, { connectionId: 'other-owner', profile: 'other' })
    release({ subagents: [] })
    await run

    // The route moved under the response: the stale snapshot must not land.
    expect($subagentsBySession.get()[SID]!.map(row => row.id)).toEqual(['ghost'])
  })

  it('lets a live event landing mid-flight win over the snapshot', async () => {
    upsertSubagent(SID, { goal: 'frozen child', status: 'running', subagent_id: 'ghost', task_index: 0 })

    let release: (value: { subagents: unknown[] }) => void = () => undefined

    request.mockImplementationOnce(
      async () =>
        new Promise(resolve => {
          release = resolve
        }) as never
    )

    const run = reconcileSubagentsOnConnectionPublish()
    await vi.waitFor(() => expect(listCalls()).toHaveLength(1))

    upsertSubagent(SID, { status: 'running', subagent_id: 'ghost', text: 'still here' }, false, 'subagent.progress')
    release({ subagents: [] })
    await run

    // The row array changed while the list was in flight: newer event truth stays.
    expect($subagentsBySession.get()[SID]!.map(row => row.id)).toEqual(['ghost'])
    expect($subagentsBySession.get()[SID]![0]!.stream.at(-1)?.text).toBe('still here')
  })

  it('is one-shot: no timer, nothing to do on an empty store, one list per affected session', async () => {
    vi.useFakeTimers()

    await reconcileSubagentsOnConnectionPublish()
    expect(listCalls()).toHaveLength(0)

    upsertSubagent(SID, { goal: 'frozen child', status: 'running', subagent_id: 'ghost', task_index: 0 })

    await reconcileSubagentsOnConnectionPublish()
    expect(listCalls()).toHaveLength(1)

    await vi.advanceTimersByTimeAsync(60_000)
    expect(listCalls()).toHaveLength(1)
  })
})
