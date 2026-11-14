import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

// The room view follows the room log by this contract: it reads only after its cursor,
// applies a page only from cursor+1 (the view stays a prefix of the log), refreshes on groups.changed, and
// keeps a slow poll because a push can be lost.
const request = vi.hoisted(() => vi.fn())
const listeners = vi.hoisted(() => new Map<string, Set<(event: Record<string, unknown>) => void>>())
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')
  const { ConfirmDialog } = await import('@/components/ui/confirm-dialog')

  const onEvent = (type: string, listener: (event: Record<string, unknown>) => void) => {
    const set = listeners.get(type) ?? new Set()
    set.add(listener)
    listeners.set(type, set)

    return () => set.delete(listener)
  }

  return {
    ...(await pluginSdkMock(gateway.host)),
    atom,
    useValue: useStore,
    ConfirmDialog,
    useQuery: (await import('@tanstack/react-query')).useQuery,
    useI18n: () => ({ t: en }),
    usePluginI18n: () => (key: string) =>
      CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    Input: (p: ComponentProps<'input'>) => <input {...p} />,
    RowButton: (p: ComponentProps<'button'>) => <button type="button" {...p} />,
    host: { ...gateway.host, requestProfile: request, onEvent }
  }
})

import { CanonicalGroupWorkspace, roomLogPageAfter } from './canonical-group-workspace'

const binding = { connectionId: 'local', profile: 'team', roomId: 'feed', roomName: 'Room' }
const message = (seq: number, text = `m${seq}`) => ({
  seq,
  kind: 'message.user',
  actor: { kind: 'user', id: 'u' },
  payload: { text }
})

/** A scripted room: `log` is the server's log, groups.log pages it after since_seq like read_events. */
function serveRoom(room: { log: ReturnType<typeof message>[]; changeEvents?: boolean; pageSize?: number }) {
  request.mockImplementation(async (_route, method, params: { since_seq?: number; limit?: number }) => {
    if (method === 'groups.state') {
      return {
        room: { name: 'Room', members: [], latest_seq: room.log.at(-1)?.seq ?? 0 },
        driver_status: {},
        ...(room.changeEvents ? { change_events: true } : {})
      }
    }

    if (method === 'groups.log') {
      const since = params.since_seq ?? 0
      const after = room.log.filter(event => event.seq > since)
      const events = after.slice(0, room.pageSize ?? 100)

      return { events, has_more: events.length < after.length }
    }

    return {}
  })
}

const stateReads = () => request.mock.calls.filter(call => call[1] === 'groups.state').length
const logReads = () => request.mock.calls.filter(call => call[1] === 'groups.log').map(call => call[2].since_seq)
const emit = (type: string, event: Record<string, unknown> = {}) =>
  act(() => {
    for (const listener of listeners.get(type) ?? []) {
      listener({ type, ...event })
    }
  })

beforeEach(() => {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined })
})
// Tests override host.retainProfile; put the shared mock's own back after each, so a failed assertion can't leak an
// override into the next test.
const sdkHost = async () => (await import('@hermes/plugin-sdk')).host as { retainProfile?: unknown }
const defaultRetain = sdkHost().then(host => host.retainProfile)
afterEach(async () => {
  cleanup()
  request.mockReset()
  listeners.clear()
  localStorage.clear()
  vi.useRealTimers()
  Object.assign(await sdkHost(), { retainProfile: await defaultRetain })
})

it('applies a page only from cursor+1, skipping events the view already holds', () => {
  expect(roomLogPageAfter(2, [message(3), message(4)])?.map(event => event.seq)).toEqual([3, 4])
  expect(roomLogPageAfter(2, [message(2), message(3)])?.map(event => event.seq)).toEqual([3])
  expect(roomLogPageAfter(2, [])).toEqual([])
  expect(roomLogPageAfter(2, [message(4)])).toBeNull()
  expect(roomLogPageAfter(2, [message(3), message(5)])).toBeNull()
})

it('reads only what is new after the first refresh, across pages', async () => {
  const room = { log: [message(1), message(2), message(3)], changeEvents: true, pageSize: 2 }
  serveRoom(room)
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m3')
  expect(logReads()).toEqual([0, 2])

  room.log.push(message(4))
  emit('groups.changed', { connectionId: 'local' })
  await screen.findByText('m4')
  expect(logReads()).toEqual([0, 2, 3])
  expect(screen.getAllByText(/^m\d$/).map(node => node.textContent)).toEqual(['m1', 'm2', 'm3', 'm4'])
})

it('refuses a page that would leave a hole and keeps the view it has', async () => {
  const room = { log: [message(1)], changeEvents: true }
  serveRoom(room)
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m1')

  room.log.push(message(3)) // seq 2 is missing: applying it would show a log the room never had
  emit('groups.changed')
  expect((await screen.findByRole('alert')).textContent).toContain("Couldn't load more of this group chat's history.")
  expect(screen.queryByText('m3')).toBeNull()
  expect(screen.getByText('m1')).toBeTruthy()
})

it('refreshes on its own route’s and room’s groups.changed, not another connection’s or room’s', async () => {
  const room = { log: [message(1)], changeEvents: true }
  serveRoom(room)
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m1')
  room.log.push(message(2))

  // A refresh issues groups.state synchronously, so an ignored signal is visible at once: no read starts.
  const reads = stateReads()
  emit('groups.changed', { connectionId: 'other-machine' })
  emit('groups.changed', { connectionId: 'local', payload: { room_ids: ['another-room'] } })
  expect(stateReads()).toBe(reads)

  emit('groups.changed', { connectionId: 'local', payload: { room_ids: ['another-room', 'feed'] } })
  await screen.findByText('m2')
  room.log.push(message(3))
  emit('groups.changed', { connectionId: 'local', payload: {} }) // process-wide: every view re-reads
  await screen.findByText('m3')
})

it('keeps the 2 s poll on a backend without change events, and a slow backstop on one with them', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  const legacy = { log: [message(1)] }
  serveRoom(legacy)
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m1')
  legacy.log.push(message(2))
  await act(() => vi.advanceTimersByTimeAsync(2100))
  await screen.findByText('m2')
  view.unmount()

  request.mockReset()
  const pushed = { log: [message(1)], changeEvents: true }
  serveRoom(pushed)
  render(<CanonicalGroupWorkspace binding={{ ...binding, roomId: 'pushed' }} />)
  await screen.findByText('m1')
  pushed.log.push(message(2))
  await act(() => vi.advanceTimersByTimeAsync(2100))
  expect(screen.queryByText('m2')).toBeNull()
  // A lost push still lands: the backstop poll reads the delta.
  await act(() => vi.advanceTimersByTimeAsync(15_000))
  await screen.findByText('m2')
})

it('a signal that lands during a read runs one more read after it', async () => {
  const room = { log: [message(1)], changeEvents: true }

  let release: () => void = () => {}
  serveRoom(room)
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m1')

  const serve = request.getMockImplementation()!
  // Hold the read's state answer (already computed without m2, so it reads no log page) until the second
  // signal has landed.
  request.mockImplementation(async (route, method, params) => {
    const answer = await serve(route, method, params)

    if (method === 'groups.state') {
      await new Promise<void>(resolve => {
        release = resolve
      })
    }

    return answer
  })
  emit('groups.changed')
  await waitFor(() => expect(stateReads()).toBe(2))
  room.log.push(message(2))
  emit('groups.changed') // mid-read: must not be dropped
  request.mockImplementation(serve)
  release()
  await screen.findByText('m2')
})

it('reads once for one broadcast that reaches it on every open socket, and again for any other broadcast', async () => {
  const room = { log: [message(1)], changeEvents: true }
  serveRoom(room)
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m1')
  const serve = request.getMockImplementation()!

  let release: () => void = () => {}

  // Hold each state answer: a push that lands meanwhile queues one more read unless it is a copy.
  const hold = () =>
    request.mockImplementation(async (...args: Parameters<typeof serve>) => {
      if (args[1] === 'groups.state') {
        await new Promise<void>(resolve => {
          release = resolve
        })
      }

      return serve(...args)
    })

  const settle = async (text: string) => {
    request.mockImplementation(serve)
    await act(async () => {
      release()
    })
    await screen.findByText(text)
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 20))
    }) // any queued read runs now
  }

  const push = (id?: string) =>
    emit('groups.changed', {
      connectionId: 'local',
      payload: { room_ids: ['feed'], ...(id ? { broadcast_id: id } : {}) }
    })
  let before = stateReads()
  hold()
  room.log.push(message(2))
  push('p1:7')
  push('p1:7')
  push('p1:7')
  push('p1:7') // one frame on the primary, the lease and two more sockets
  await settle('m2')
  expect(stateReads() - before).toBe(1)

  // A later broadcast with the same rooms (or another backend's), landing before the held answer: not a copy.
  before = stateReads()
  hold()
  push('p1:8')
  room.log.push(message(3))
  push('p1:9')
  await settle('m3')
  expect(stateReads() - before).toBe(2)

  // A backend that sends no id: every push queues a read.
  before = stateReads()
  hold()
  room.log.push(message(4))
  push()
  push()
  await settle('m4')
  expect(stateReads() - before).toBe(2)
})

it('rebuilds from the start when the room log is shorter than the view (re-hosted)', async () => {
  const room = { log: [message(1), message(2), message(3)], changeEvents: true }
  serveRoom(room)
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m3')

  room.log = [message(1, 'fresh-1')]
  emit('groups.changed')
  await screen.findByText('fresh-1')
  expect(screen.queryByText('m3')).toBeNull()
  expect(logReads().at(-1)).toBe(0)
})

it('runs one room read at a time for every caller: Refresh joins a push read in flight', async () => {
  const room = { log: [message(1)], changeEvents: true }
  serveRoom(room)
  const serve = request.getMockImplementation()!
  let failLog = true
  request.mockImplementation(async (...args: Parameters<typeof serve>) =>
    args[1] === 'groups.log' && failLog ? Promise.reject(new Error('busy store')) : serve(...args)
  )
  render(<CanonicalGroupWorkspace binding={binding} />)
  const retry = await screen.findByRole('button', { name: 'Refresh' }) // the read-error banner's button

  let release!: () => void
  const held = new Promise<void>(resolve => {
    release = resolve
  })
  let inFlight = 0
  let maxInFlight = 0
  failLog = false
  request.mockImplementation(async (...args: Parameters<typeof serve>) => {
    if (args[1] !== 'groups.state') {
      return serve(...args)
    }
    inFlight++
    maxInFlight = Math.max(maxInFlight, inFlight)

    try {
      await held

      return await serve(...args)
    } finally {
      inFlight--
    }
  })
  room.log.push(message(2))
  emit('groups.changed', { connectionId: 'local' }) // a push read starts and is held
  fireEvent.click(retry) // the user's Refresh lands mid-read
  await act(async () => {
    release()
  })
  await screen.findByText('m2')
  expect(maxInFlight).toBe(1)
})

it('keeps the same log array when a read brings nothing new', async () => {
  const room = { log: [message(1), message(2)], changeEvents: true }
  serveRoom(room)
  const History = await import('./canonical-group-history')
  const render0 = vi.spyOn(History, 'CanonicalGroupHistory')
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m2')
  const before = (render0.mock.calls.at(-1)?.[0] as { events: unknown }).events
  emit('groups.changed', { connectionId: 'local', payload: { room_ids: ['feed'] } })
  await waitFor(() => expect(stateReads()).toBeGreaterThanOrEqual(2))
  expect(logReads()).toEqual([0]) // a status-only read asks for no log page
  await act(async () => {
    await Promise.resolve()
  })
  expect((render0.mock.calls.at(-1)?.[0] as { events: unknown }).events).toBe(before)
  render0.mockRestore()
})

it("holds the room route's socket open while the view is shown, so its pushes arrive", async () => {
  serveRoom({ log: [message(1)], changeEvents: true })
  const sdk = await import('@hermes/plugin-sdk')
  const released = vi.fn()
  const retain = vi.fn(async (): Promise<() => void> => released)
  Object.assign(sdk.host, { retainProfile: retain })
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m1')
  expect(retain).toHaveBeenCalledWith({ connectionId: 'local', profile: 'team', targetProfile: 'team', mode: 'local' })
  expect(released).not.toHaveBeenCalled()
  view.unmount()
  expect(released).toHaveBeenCalledTimes(1)

  // A lease that resolves only after the view closed is released at once.
  let grant!: (release: () => void) => void
  const late = vi.fn()
  retain.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        grant = resolve
      })
  )
  const again = render(<CanonicalGroupWorkspace binding={binding} />)
  again.unmount()
  await act(async () => {
    grant(late)
  })
  expect(late).toHaveBeenCalledTimes(1)
})

it('refuses a page that says there is more but moves the cursor nowhere, instead of spinning or stopping short', async () => {
  request.mockImplementation(async (_route, method) =>
    method === 'groups.state'
      ? { room: { name: 'Room', members: [], latest_seq: 5 }, driver_status: {}, change_events: true }
      : method === 'groups.log'
        ? { events: [], has_more: true }
        : {}
  )
  render(<CanonicalGroupWorkspace binding={binding} />)
  expect(await screen.findByRole('alert')).toBeTruthy()
  expect(logReads()).toEqual([0])
})

it('re-enables the room controls after an action even while pushes keep the room busy', async () => {
  const room = { log: [message(1)], changeEvents: true }
  serveRoom(room)
  const serve = request.getMockImplementation()!
  let storm = false
  let reads = 0
  // While the storm runs, every state read is slow and every read is followed by another push: the room never
  // falls quiet, so a settle that waited for the whole queue would never return.
  request.mockImplementation(async (...args: Parameters<typeof serve>) => {
    if (args[1] === 'groups.state') {
      return { ...((await serve(...args)) as object), driver_status: { working: true, member_activity: [] } }
    }

    if (args[1] === 'groups.stop') {
      return { cancelled: 1 }
    }

    return serve(...args)
  })
  const base = request.getMockImplementation()!
  request.mockImplementation(async (...args: Parameters<typeof base>) => {
    if (storm && args[1] === 'groups.state') {
      reads++
      await new Promise(resolve => setTimeout(resolve, 15))
      setTimeout(() => {
        if (storm) {
          emit('groups.changed', { connectionId: 'local' })
        }
      }, 0)
    }

    return base(...args)
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m1')
  const stop = await screen.findByRole('button', { name: 'Stop the group' })
  storm = true
  emit('groups.changed', { connectionId: 'local' })
  fireEvent.click(stop)
  // Stop stays on screen, disabled, while it runs: the controls are back once it is enabled again.
  await waitFor(
    () => expect((screen.getByRole('button', { name: 'Stop the group' }) as HTMLButtonElement).disabled).toBe(false),
    { timeout: 3000 }
  )
  const during = reads
  storm = false
  expect(during).toBeGreaterThan(0)
})

// A socket tags its events with its registry connection; a legacy primary (no resolved connectionId) tags none.
it.each([
  ['a tagged socket', { connectionId: 'local' }],
  ['a legacy primary socket', {}]
])(
  're-reads at once when its route reconnects on %s, since the pushes sent while the socket was down are lost',
  async (_socket, ready) => {
    const room = { log: [message(1)], changeEvents: true }
    serveRoom(room)
    render(<CanonicalGroupWorkspace binding={binding} />)
    await screen.findByText('m1')
    room.log.push(message(2)) // appended while the socket was down: its groups.changed never arrived

    const reads = stateReads()
    emit('gateway.ready', { connectionId: 'other-machine' })
    expect(stateReads()).toBe(reads)
    emit('gateway.ready', ready)
    expect(stateReads()).toBe(reads + 1) // the read starts now, not at the 15 s backstop
    await screen.findByText('m2')
  }
)

it('retries a failed read on the fast poll, not the backstop, even on a backend that pushes', async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
  const room = { log: [message(1)], changeEvents: true }
  serveRoom(room)
  const sdk = await import('@hermes/plugin-sdk')
  Object.assign(sdk.host, { retainProfile: vi.fn(async () => () => undefined) }) // a held lease: the slow backstop is in play
  render(<CanonicalGroupWorkspace binding={binding} />)
  await vi.waitFor(() => expect(screen.getByText('m1')).toBeTruthy())
  const serve = request.getMockImplementation()!
  let down = true
  request.mockImplementation(async (...args: Parameters<typeof serve>) =>
    down && args[1] === 'groups.state' ? Promise.reject(new Error('socket closed')) : serve(...args)
  )
  emit('groups.changed', { connectionId: 'local' }) // the read that finds the route down
  await act(async () => {
    await vi.advanceTimersByTimeAsync(10)
  })
  down = false
  room.log.push(message(2))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2100)
  }) // one fast poll, far short of 15 s
  expect(screen.getByText('m2')).toBeTruthy()
})

it('retries a failed Refresh on the fast poll too, not only a failed poll', async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
  const room = { log: [message(1)], changeEvents: true }
  serveRoom(room)
  const sdk = await import('@hermes/plugin-sdk')
  Object.assign(sdk.host, { retainProfile: vi.fn(async () => () => undefined) }) // a held lease: the slow backstop is in play
  const serve = request.getMockImplementation()!
  let down = false
  request.mockImplementation(async (...args: Parameters<typeof serve>) =>
    down && args[1] === 'groups.log' ? Promise.reject(new Error('socket closed')) : serve(...args)
  )
  render(<CanonicalGroupWorkspace binding={binding} />)
  await vi.waitFor(() => expect(screen.getByText('m1')).toBeTruthy())
  down = true
  room.log.push(message(2))
  emit('groups.changed', { connectionId: 'local' })
  const refresh = await vi.waitFor(() => screen.getByRole('button', { name: 'Refresh' }))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2100)
  }) // the push's failed read: fast poll, fails again
  room.log.push(message(3))
  fireEvent.click(refresh) // the user's Refresh, also failing
  await act(async () => {
    await vi.advanceTimersByTimeAsync(10)
  })
  down = false
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2100)
  }) // its retry: 2 s, not the 15 s backstop
  expect(screen.getByText('m3')).toBeTruthy()
})

it('drops a read that started before an action settled, so its stale status cannot clear the stop notice', async () => {
  const room = { log: [message(1)], changeEvents: true }
  serveRoom(room)
  const serve = request.getMockImplementation()!

  let release: () => void = () => {}
  let hold = false
  let working = true
  request.mockImplementation(async (...args: Parameters<typeof serve>) => {
    if (args[1] === 'groups.stop') {
      working = false

      return { cancelled: 1 }
    }

    if (args[1] !== 'groups.state') {
      return serve(...args)
    }

    // The status this read reports was taken when it started.
    const snapshot = {
      ...((await serve(...args)) as object),
      driver_status: working
        ? { working: true, member_activity: [] }
        : {
            working: false,
            member_activity: [],
            pending_actions: [{ kind: 'retry', member_id: 'ops', task_id: 'kept', execution_generation: 1 }]
          }
    }

    if (hold && working) {
      hold = false
      // A pre-Stop answer that says nothing is waiting would clear the notice the Stop just set.
      await new Promise<void>(resolve => {
        release = resolve
      })

      return { ...snapshot, driver_status: { working: false, member_activity: [] } }
    }

    return snapshot
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m1')
  hold = true
  emit('groups.changed', { connectionId: 'local' }) // a read starts before the Stop and is held
  await waitFor(() => expect(hold).toBe(false))
  fireEvent.click(await screen.findByRole('button', { name: 'Stop the group' }))
  await waitFor(() => expect(working).toBe(false))
  await screen.findByText(/Stop requested/)
  await act(async () => {
    release()
  }) // the pre-Stop answer lands now
  await screen.findByText('Retry') // the read after the Stop has landed too
  expect(screen.getByText(/Stop requested/)).toBeTruthy()
})

it("asks again for a socket lease that failed when its route reconnects, and a read's own reconnect is not a re-read", async () => {
  const room = { log: [message(1)], changeEvents: true }
  serveRoom(room)
  const sdk = await import('@hermes/plugin-sdk')
  const released = vi.fn()
  const retain = vi
    .fn<() => Promise<() => void>>()
    .mockRejectedValueOnce(new Error('route unreachable'))
    .mockResolvedValue(released)
  Object.assign(sdk.host, { retainProfile: retain })
  const serve = request.getMockImplementation()!
  // An unleased secondary socket reconnects for each request: every read's own dial announces gateway.ready.
  let dials = 0
  request.mockImplementation(async (...args: Parameters<typeof serve>) => {
    // Bounded, so a view that re-reads on its own dial fails the count below instead of spinning forever.
    if (args[1] === 'groups.state' && dials++ < 20) {
      emit('gateway.ready', { connectionId: 'local' })
    }

    return serve(...args)
  })
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m1')
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 50))
  })
  expect(stateReads()).toBe(1) // its own dial's gateway.ready started no second read
  expect(retain).toHaveBeenCalledTimes(1) // and that first lease failed

  emit('gateway.ready', { connectionId: 'local' }) // the route really came back
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 50))
  })
  expect(stateReads()).toBe(2) // one re-read, not a loop of them
  expect(retain).toHaveBeenCalledTimes(2) // the failed lease is asked for again
  view.unmount()
  expect(released).toHaveBeenCalledTimes(1)
})

it("re-reads on a reconnect that lands after its read's state answer, which may have missed pushes", async () => {
  const room = { log: [message(1)], changeEvents: true }
  serveRoom(room)
  const sdk = await import('@hermes/plugin-sdk')
  Object.assign(sdk.host, { retainProfile: vi.fn(async () => () => undefined) }) // the lease holds the socket open
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m1')
  const serve = request.getMockImplementation()!

  let release: () => void = () => {}
  let held = false
  room.log.push(message(2))
  request.mockImplementation(async (...args: Parameters<typeof serve>) => {
    const answer = await serve(...args)

    if (args[1] === 'groups.log' && !held) {
      held = true
      // The route drops and comes back after this read's state answer: a push sent meanwhile is lost.
      emit('gateway.ready', { connectionId: 'local' })
      await new Promise<void>(resolve => {
        release = resolve
      })
    }

    return answer
  })
  const before = stateReads()
  emit('groups.changed', { connectionId: 'local' })
  await waitFor(() => expect(held).toBe(true))
  await act(async () => {
    release()
  })
  await waitFor(() => expect(stateReads()).toBe(before + 2)) // the push's read, then one for the reconnect
})

it("without a socket lease, its log request's own redial is not a reconnect, so a failing log read does not loop", async () => {
  const room = { log: [message(1)], changeEvents: true }
  serveRoom(room)
  const sdk = await import('@hermes/plugin-sdk')
  Object.assign(sdk.host, {
    retainProfile: vi.fn(async () => {
      throw new Error('route unreachable')
    })
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m1')
  const serve = request.getMockImplementation()!
  let logDials = 0
  // An unleased secondary socket closes between requests: the log request after an answered state redials.
  request.mockImplementation(async (...args: Parameters<typeof serve>) => {
    if (args[1] === 'groups.log' && logDials++ < 20) {
      emit('gateway.ready', { connectionId: 'local' })

      throw new Error('socket closed')
    }

    return serve(...args)
  })
  room.log.push(message(2))
  const before = stateReads()
  emit('groups.changed', { connectionId: 'local' })
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 100))
  })
  expect(stateReads() - before).toBe(1) // the push's read, not one more per redial
})

it("stays on the fast poll while its route's socket lease is missing, since no push can reach it", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })

  serveRoom({ log: [message(1)], changeEvents: true })
  const sdk = await import('@hermes/plugin-sdk')
  Object.assign(sdk.host, {
    retainProfile: vi.fn(async () => {
      throw new Error('route unreachable')
    })
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m1')
  const before = stateReads()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2_100)
  })
  expect(stateReads()).toBeGreaterThan(before)
})

it('moves to the slow backstop once the lease holds and the backend pushes', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })

  serveRoom({ log: [message(1)], changeEvents: true })
  const sdk = await import('@hermes/plugin-sdk')
  Object.assign(sdk.host, { retainProfile: vi.fn(async () => () => undefined) })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('m1')
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2_100)
  }) // the first read may end before the lease
  const before = stateReads()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(10_000)
  })
  expect(stateReads()).toBe(before)
})
