import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/gateway', () => ({
  $gateway: atom(null),
  gatewayRouteState: vi.fn(() => ({ gateway: null, key: '', open: false })),
  requestGatewayForAgent: vi.fn()
}))

const gw = await import('@/store/gateway')
const request = vi.mocked(gw.requestGatewayForAgent)
const routeState = vi.mocked(gw.gatewayRouteState)
const looks = await import('./gateway-project-looks')

const STUDIO = { connectionId: 'studio', profile: 'research' }

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(res => (resolve = res))

  return { promise, resolve }
}

const project = (id: string, path: string, color: null | string) => ({
  archived: false,
  color,
  folders: [{ path }],
  icon: null,
  id,
  name: path.split('/').pop(),
  primary_path: path
})

const settle = () => new Promise(resolve => setTimeout(resolve, 0))

/** A socket already open to an owner, as gatewayRouteState hands it to passive reads. */
function openSocket(list: () => Promise<unknown>) {
  const handlers = new Set<(state: string) => void>()

  const socket = {
    connectionState: 'open',
    onState(handler: (state: string) => void) {
      handlers.add(handler)
      handler(socket.connectionState)

      return () => handlers.delete(handler)
    },
    request: vi.fn(list),
    become(state: string) {
      socket.connectionState = state
      handlers.forEach(handler => handler(state))
    }
  }

  routeState.mockReturnValue({ gateway: socket as never, key: 'studio', open: true })

  return socket
}

const listing = (...projects: unknown[]) => ({ active_id: null, projects })

beforeEach(() => {
  vi.clearAllMocks()
  looks.$gatewayProjectLooks.set({})
})

describe('gateway project looks', () => {
  it("keeps a folder's new colour when another folder's write read the owner before it committed", async () => {
    const lists: ReturnType<typeof deferred<unknown>>[] = []
    const creates: ReturnType<typeof deferred<unknown>>[] = []

    request.mockImplementation(((_connection: string, _profile: string, method: string) => {
      const next = deferred<unknown>()

      ;(method === 'projects.list' ? lists : creates).push(next)

      return next.promise
    }) as never)

    const a = looks.setGatewayProjectAppearance(STUDIO, '/code/a', { color: '#aaa' })
    const b = looks.setGatewayProjectAppearance(STUDIO, '/code/b', { color: '#bbb' })
    await settle()
    // Both reads answer with the owner's state from before either create.
    const before = { active_id: null, projects: [] }
    lists[0].resolve(before)
    await settle()
    creates[0].resolve({ project: project('p_a', '/code/a', '#aaa') })
    await a
    lists[1].resolve(before)
    await settle()
    creates[1].resolve({ project: project('p_b', '/code/b', '#bbb') })
    await b

    expect(looks.gatewayProjectLook(STUDIO, '/code/a')).toEqual({ color: '#aaa', icon: null, id: 'p_a' })
    expect(looks.gatewayProjectLook(STUDIO, '/code/b')).toEqual({ color: '#bbb', icon: null, id: 'p_b' })
  })

  it('reads an owner again on a later render after a failed read, a bounded number of times', async () => {
    const socket = openSocket(async () => {
      throw new Error('offline')
    })

    for (let render = 0; render < 6; render += 1) {
      looks.loadGatewayProjectLooks([STUDIO])
      await settle()
    }

    expect(socket.request).toHaveBeenCalledTimes(3)
    expect(request).not.toHaveBeenCalled()

    socket.request.mockResolvedValue(listing(project('p_app', '/code/app', '#f43f5e')))
    socket.become('closed')
    socket.become('open')
    await settle()

    expect(looks.gatewayProjectLook(STUDIO, '/code/app')?.color).toBe('#f43f5e')
  })

  it("re-reads an owner's looks when its socket reconnects", async () => {
    const socket = openSocket(async () => listing(project('p_app', '/code/app', '#aaa')))

    looks.loadGatewayProjectLooks([STUDIO])
    await settle()
    expect(looks.gatewayProjectLook(STUDIO, '/code/app')?.color).toBe('#aaa')

    // The owner's own window recolours while this one is disconnected.
    socket.request.mockResolvedValue(listing(project('p_app', '/code/app', '#bbb')))
    socket.become('closed')
    socket.become('open')
    await settle()

    expect(looks.gatewayProjectLook(STUDIO, '/code/app')?.color).toBe('#bbb')
    expect(socket.request).toHaveBeenCalledTimes(2)
  })

  it('reads again when the socket reconnects while an earlier read is still in flight', async () => {
    const before = deferred<unknown>()
    const socket = openSocket(() => before.promise)

    looks.loadGatewayProjectLooks([STUDIO])
    socket.request.mockResolvedValue(listing(project('p_app', '/code/app', '#bbb')))
    socket.become('closed')
    socket.become('open')
    before.resolve(listing(project('p_app', '/code/app', '#aaa')))
    await settle()

    expect(socket.request).toHaveBeenCalledTimes(2)
    expect(looks.gatewayProjectLook(STUDIO, '/code/app')?.color).toBe('#bbb')
  })

  it("keeps a first colour pick when the sidebar's earlier read lands after it", async () => {
    const sidebarRead = deferred<unknown>()

    openSocket(() => sidebarRead.promise)
    request.mockImplementation((async (_connection: string, _profile: string, method: string) =>
      method === 'projects.list' ? listing() : { project: project('p_app', '/code/app', '#f43f5e') }) as never)

    looks.loadGatewayProjectLooks([STUDIO])
    await looks.setGatewayProjectAppearance(STUDIO, '/code/app', { color: '#f43f5e' })
    sidebarRead.resolve(listing())
    await settle()

    expect(looks.gatewayProjectLook(STUDIO, '/code/app')?.color).toBe('#f43f5e')
  })
})
