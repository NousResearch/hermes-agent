import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { $activeGatewayProfile } from '@/store/profile'

import { isSkillItem } from '../composer-utils'

import { useSlashCompletions } from './use-slash-completions'

const CATALOG = {
  categories: [{ name: 'Session', pairs: [['/new', 'Start a new session']] }],
  pairs: [
    ['/new', 'Start a new session'],
    ['/work', 'Kick off a task in a fresh worktree']
  ]
}

// A catalog shaped like a real install: a couple of skills the user lives in,
// a bundled one they have never opened, and one of their own they haven't
// either.
const RANKED_CATALOG = {
  categories: [{ name: 'Session', pairs: [['/new', 'Start a new session']] }],
  pairs: [
    ['/new', 'Start a new session'],
    ['/docx', 'Edit Word documents'],
    ['/research', 'Look it up before answering'],
    ['/research-paper-writing', 'Write an academic paper'],
    ['/work', 'Kick off a task in a fresh worktree']
  ],
  skills: {
    '/docx': { usage: 0, origin: 'local' },
    '/research': { usage: 60, origin: 'local' },
    '/research-paper-writing': { usage: 0, origin: 'bundled' },
    '/work': { usage: 172, origin: 'local' }
  }
}

const commandsOf = (items: readonly Unstable_TriggerItem[]) =>
  items.map(item => (item.metadata as { command?: string })?.command)

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

function harness(gateway: HermesGateway, options: { sessionId?: string | null } = {}) {
  const api: {
    search?: (query: string) => readonly Unstable_TriggerItem[]
    rerenderSession?: (sessionId?: string | null) => void
  } = {}

  function Probe({ sessionId }: { sessionId?: string | null }) {
    const { adapter } = useSlashCompletions({ gateway, sessionId })
    api.search = adapter.search

    return null
  }

  const view = render(<Probe sessionId={options.sessionId} />)
  api.rerenderSession = sessionId => view.rerender(<Probe sessionId={sessionId} />)

  return api as {
    search: (query: string) => readonly Unstable_TriggerItem[]
    rerenderSession: (sessionId?: string | null) => void
  }
}

/** Drive the adapter until its async fetch has settled into `search`'s result. */
async function completions(api: { search: (query: string) => readonly Unstable_TriggerItem[] }, query: string) {
  await act(async () => {
    api.search(query)
    await Promise.resolve()
  })

  // The debounce is skipped only for cached queries; give the timer a beat.
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 120))
  })

  return api.search(query)
}

afterEach(() => {
  cleanup()
  $activeGatewayProfile.set('default')
  queryClient.clear()
})

describe('useSlashCompletions', () => {
  it('scopes a new-session catalog and typed completion to the selected profile', async () => {
    $activeGatewayProfile.set('search')

    const request = vi.fn().mockImplementation((method: string) =>
      Promise.resolve(method === 'commands.catalog' ? CATALOG : { items: [] })
    )

    const api = harness({ request } as unknown as HermesGateway)

    await completions(api, '')
    await completions(api, 'work')

    expect(request).toHaveBeenCalledWith('commands.catalog', { profile: 'search' })
    expect(request).toHaveBeenCalledWith('complete.slash', { profile: 'search', text: '/work' })
  })

  it('sends both the selected profile fallback and authoritative session id', async () => {
    $activeGatewayProfile.set('search')
    const request = vi.fn().mockResolvedValue(CATALOG)
    const api = harness({ request } as unknown as HermesGateway, { sessionId: 'sess-search' })

    await completions(api, '')

    expect(request).toHaveBeenCalledWith('commands.catalog', {
      profile: 'search',
      session_id: 'sess-search'
    })
  })

  it('clears held completions when the session scope changes', async () => {
    const catalogFor = (name: string) => ({ categories: [], pairs: [[`/${name}`, `${name} skill`]] })

    const request = vi.fn().mockImplementation((_method: string, params: { session_id?: string }) =>
      Promise.resolve(catalogFor(params.session_id === 'sess-b' ? 'b-only' : 'a-only'))
    )

    const api = harness({ request } as unknown as HermesGateway, { sessionId: 'sess-a' })

    expect(commandsOf(await completions(api, ''))).toContain('/a-only')

    await act(async () => api.rerenderSession('sess-b'))
    let immediate: readonly Unstable_TriggerItem[] = []
    await act(async () => {
      immediate = api.search('')
    })
    expect(commandsOf(immediate)).not.toContain('/a-only')
    expect(commandsOf(await completions(api, ''))).toContain('/b-only')
  })

  it('ignores an old session request that resolves after a scope change', async () => {
    const oldRequest = deferred<typeof CATALOG>()
    const newCatalog = { categories: [], pairs: [['/b-only', 'B skill']] }
    const newRequest = deferred<typeof newCatalog>()

    const request = vi.fn().mockImplementation((_method: string, params: { session_id?: string }) =>
      params.session_id === 'sess-b' ? newRequest.promise : oldRequest.promise
    )

    const api = harness({ request } as unknown as HermesGateway, { sessionId: 'sess-a' })

    await act(async () => {
      api.search('')
      await new Promise(resolve => setTimeout(resolve, 120))
    })

    await act(async () => api.rerenderSession('sess-b'))
    await act(async () => {
      api.search('')
      await new Promise(resolve => setTimeout(resolve, 120))
    })

    await act(async () => {
      oldRequest.resolve(CATALOG)
      await Promise.resolve()
    })
    expect(commandsOf(api.search(''))).not.toContain('/work')

    await act(async () => {
      newRequest.resolve(newCatalog)
      await Promise.resolve()
    })
    expect(commandsOf(api.search(''))).toContain('/b-only')
  })

  it('serves the bare-slash catalog from cache instead of re-requesting it', async () => {
    const request = vi.fn().mockResolvedValue(CATALOG)
    const api = harness({ request } as unknown as HermesGateway)

    await completions(api, '')
    expect(request).toHaveBeenCalledTimes(1)

    // Reopening `/` must not hit the gateway again.
    queryClient.setQueryData(['unrelated'], 1)
    await completions(api, '')
    expect(request).toHaveBeenCalledTimes(1)

    // …until something that changes the command set invalidates it.
    await act(async () => invalidateSlashCompletions())
    await completions(api, '')
    expect(request).toHaveBeenCalledTimes(2)
  })

  it('offers skill commands on a bare slash, not just built-ins', async () => {
    const request = vi.fn().mockResolvedValue(CATALOG)
    const api = harness({ request } as unknown as HermesGateway)

    const items = await completions(api, '')
    const work = items.find(item => (item.metadata as { command?: string })?.command === '/work')

    expect((work?.metadata as { group?: string })?.group).toBe('Skills')
  })

  // A `/` typed mid-message is a reference dropped into prose, so the trigger
  // filters the list to skills (use-composer-trigger). A bare mid-message `/`
  // resolves to the same empty query as an opening `/`, so that filter runs
  // over the catalog — which listed no skill-group rows at all, leaving the
  // inline popover empty. Asserted through isSkillItem, the real predicate.
  it('leaves only skills for a mid-message slash', async () => {
    const request = vi.fn().mockResolvedValue(CATALOG)
    const api = harness({ request } as unknown as HermesGateway)

    const inline = (await completions(api, '')).filter(isSkillItem)

    expect(inline.map(item => (item.metadata as { command?: string })?.command)).toEqual(['/work'])
  })

  // An alphabetical `/` menu buries the skills someone runs daily under the
  // ones that shipped with Hermes and were never opened.
  it('orders skills by use and hides never-used built-ins on a bare slash', async () => {
    const request = vi.fn().mockResolvedValue(RANKED_CATALOG)
    const api = harness({ request } as unknown as HermesGateway)

    const skills = commandsOf((await completions(api, '')).filter(isSkillItem))

    expect(skills).toEqual(['/work', '/research', '/docx'])
  })

  // Typing is a search, and a search that hides a match is broken — the
  // never-used built-in still shows, just below the one she actually uses.
  it('ranks a typed query by use without hiding anything', async () => {
    const request = vi.fn().mockImplementation((method: string) =>
      Promise.resolve(
        method === 'commands.catalog'
          ? RANKED_CATALOG
          : {
              items: [
                { text: '/research-paper-writing', display: '/research-paper-writing', meta: 'Write a paper' },
                { text: '/research', display: '/research', meta: 'Look it up' }
              ]
            }
      )
    )

    const api = harness({ request } as unknown as HermesGateway)

    // Warm the catalog first: the popover always opens on a bare `/` before a
    // query is typed, which is where the usage map comes from.
    await completions(api, '')

    expect(commandsOf(await completions(api, 'research'))).toEqual(['/research', '/research-paper-writing'])
  })
})
