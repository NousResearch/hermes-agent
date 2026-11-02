import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/gateway', () => ({
  $gateway: atom(null),
  requestGatewayForAgent: vi.fn()
}))

const gw = await import('@/store/gateway')
const request = vi.mocked(gw.requestGatewayForAgent)
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
})
