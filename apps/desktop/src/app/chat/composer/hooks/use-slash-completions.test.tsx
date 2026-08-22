import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'

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

function harness(gateway: HermesGateway, scope?: { cwd?: string; sessionId?: string }) {
  const api: { search?: (query: string) => readonly Unstable_TriggerItem[] } = {}

  function Probe() {
    const { adapter } = useSlashCompletions({ cwd: scope?.cwd ?? null, gateway, sessionId: scope?.sessionId ?? null })
    api.search = adapter.search

    return null
  }

  render(<Probe />)

  return api as { search: (query: string) => readonly Unstable_TriggerItem[] }
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
  queryClient.clear()
})

describe('useSlashCompletions', () => {
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

  // The catalog includes the session's PROJECT skills (`.hermes/skills` /
  // `.agents/skills` in the open repo). One backend process serves every
  // session, so it can only resolve which project that is from the session
  // identity we send — without it the scan ran against the backend's own
  // directory and no repo skill ever reached the popover.
  it('scopes both completion RPCs to the session and its workspace', async () => {
    const request = vi.fn().mockResolvedValue(CATALOG)
    const api = harness({ request } as unknown as HermesGateway, { cwd: '/repo', sessionId: 'sid-1' })

    await completions(api, '')
    expect(request).toHaveBeenCalledWith('commands.catalog', { cwd: '/repo', session_id: 'sid-1' })

    request.mockResolvedValue({ items: [{ text: '/work', display: '/work', meta: '' }] })
    await completions(api, 'wo')
    expect(request).toHaveBeenCalledWith('complete.slash', { cwd: '/repo', session_id: 'sid-1', text: '/wo' })
  })

  it('does not serve one workspace a catalog cached for another', async () => {
    const request = vi.fn().mockResolvedValue(CATALOG)

    const inRepo = harness({ request } as unknown as HermesGateway, { cwd: '/repo-a', sessionId: 'a' })
    await completions(inRepo, '')
    expect(request).toHaveBeenCalledTimes(1)

    // A second session in a different repo must re-ask rather than inherit
    // repo A's answer — the catalog is per-workspace, not per-process.
    cleanup()
    const elsewhere = harness({ request } as unknown as HermesGateway, { cwd: '/repo-b', sessionId: 'b' })
    await completions(elsewhere, '')
    expect(request).toHaveBeenCalledTimes(2)
    expect(request).toHaveBeenLastCalledWith('commands.catalog', { cwd: '/repo-b', session_id: 'b' })
  })

  // A raw `${cwd}|${sessionId}` join lets a delimiter inside the cwd forge
  // another scope's key: ('/repo|x', 'sid') and ('/repo', 'x|sid') both
  // serialize to '/repo|x|sid'. Directories with `|` in their name are legal
  // on POSIX filesystems, so these are two REAL distinct scopes.
  it('does not collide cache keys when a cwd contains the delimiter', async () => {
    const request = vi.fn().mockResolvedValue(CATALOG)

    const first = harness({ request } as unknown as HermesGateway, { cwd: '/repo|x', sessionId: 'sid' })
    await completions(first, '')
    expect(request).toHaveBeenCalledTimes(1)

    cleanup()
    const second = harness({ request } as unknown as HermesGateway, { cwd: '/repo', sessionId: 'x|sid' })
    await completions(second, '')
    expect(request).toHaveBeenCalledTimes(2)
  })
})
