import { beforeEach, describe, expect, it, vi } from 'vitest'

const request = vi.fn()

vi.mock('@/store/gateway', () => ({
  requestGatewayForAgent: (...args: unknown[]) => request(...args)
}))

// A window live on a source the registry cannot name has no connection id.
const OWNER = { connectionId: null, profile: 'default' }
const WORK = { connectionId: null, profile: 'work' }
const PAGE_ID = '20260920_101010_abcdef'

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(res => {
    resolve = res
  })

  return { promise, resolve }
}

// Fresh module state (page ids, save queue) per test.
async function loadPageModule() {
  vi.resetModules()

  return import('./page')
}

function saveParams() {
  return request.mock.calls.filter(call => call[2] === 'backworkspace.save').map(call => call[3])
}

beforeEach(() => {
  request.mockReset()
})

describe('back workspace page', () => {
  it('reuses the id minted by the first save for every save queued behind it', async () => {
    const page = await loadPageModule()
    const firstSave = deferred<{ id: string }>()
    let saves = 0

    request.mockImplementation((_connection, _profile, method, params) => {
      if (method === 'backworkspace.open') {
        return Promise.resolve({ page: null })
      }

      saves += 1

      return saves === 1 ? firstSave.promise : Promise.resolve({ id: params.id })
    })
    await page.loadBackworkspacePage(OWNER)

    page.editBackworkspacePage('one')
    void page.flushBackworkspacePage()
    page.editBackworkspacePage('one two')
    const settled = page.flushBackworkspacePage()

    firstSave.resolve({ id: PAGE_ID })
    await settled

    expect(saveParams()).toEqual([
      { content: 'one', id: undefined },
      { content: 'one two', id: PAGE_ID }
    ])
    // The null id reaches the router as-is; `local` would name another machine.
    expect(request.mock.calls.map(call => call[0])).toEqual([null, null, null])
  })

  it('lets only the latest open land, even after switching away and back', async () => {
    const page = await loadPageModule()
    const firstOpen = deferred<{ page: { content: string; id: string } }>()
    let defaultOpens = 0

    request.mockImplementation((_connection, profile) => {
      if (profile === 'work') {
        return Promise.resolve({ page: { content: 'work page', id: PAGE_ID } })
      }

      defaultOpens += 1

      return defaultOpens === 1 ? firstOpen.promise : Promise.resolve({ page: { content: 'fresh', id: PAGE_ID } })
    })

    const first = page.loadBackworkspacePage(OWNER)

    await page.loadBackworkspacePage(WORK)
    await page.loadBackworkspacePage(OWNER)
    firstOpen.resolve({ page: { content: 'stale', id: PAGE_ID } })
    await first

    expect(page.$backworkspacePage.get()).toMatchObject({ content: 'fresh', key: ':default', status: 'ready' })
  })

  it('shows a backend without backworkspace.* as unsupported instead of an error', async () => {
    const page = await loadPageModule()

    request.mockRejectedValue(Object.assign(new Error('method not found'), { code: -32601 }))
    await page.loadBackworkspacePage(OWNER)

    expect(page.$backworkspacePage.get()?.status).toBe('unsupported')
  })

  it('keeps text whose save failed across a profile switch and saves it once the backend recovers', async () => {
    const page = await loadPageModule()
    const files = new Map([['default', 'old file']])
    let diskFull = true

    request.mockImplementation((_connection, profile, method, params) => {
      if (method === 'backworkspace.open') {
        const content = files.get(profile)

        return Promise.resolve({ page: content === undefined ? null : { content, id: PAGE_ID } })
      }

      if (diskFull) {
        return Promise.reject(new Error('disk full'))
      }

      files.set(profile, params.content)

      return Promise.resolve({ id: PAGE_ID })
    })
    await page.loadBackworkspacePage(OWNER)

    page.editBackworkspacePage('draft')
    await page.flushBackworkspacePage()
    expect(page.$backworkspacePage.get()).toMatchObject({ content: 'draft', saveFailed: true })

    await page.loadBackworkspacePage(WORK)
    diskFull = false
    await page.loadBackworkspacePage(OWNER)
    await page.flushBackworkspacePage()

    expect(page.$backworkspacePage.get()).toMatchObject({ content: 'draft', saveFailed: false })
    expect(files.get('default')).toBe('draft')
  })
})
