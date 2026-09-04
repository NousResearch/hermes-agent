import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { rememberDesktopCommandsCatalog, type CommandsCatalogLike } from '@/lib/desktop-slash-commands'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'

import { isSkillItem } from '../composer-utils'

import { canonicalizeSlashCommandCompletions, useSlashCompletions } from './use-slash-completions'

const CATALOG: CommandsCatalogLike = {
  categories: [{ name: 'Session', pairs: [['/new', 'Start a new session']] }],
  pairs: [
    ['/new', 'Start a new session'],
    ['/work', 'Kick off a task in a fresh worktree']
  ]
}

const ALIAS_CATALOG: CommandsCatalogLike = {
  ...CATALOG,
  canon: {
    '/background': '/background',
    '/bg': '/background',
    '/btw': '/background'
  }
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

function harness(gateway: HermesGateway) {
  const api: { search?: (query: string) => readonly Unstable_TriggerItem[] } = {}

  function Probe() {
    const { adapter } = useSlashCompletions({ gateway })
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
  rememberDesktopCommandsCatalog(undefined)
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

  it('shows the matched alias beside its canonical command', async () => {
    const request = vi.fn().mockImplementation((method: string) =>
      Promise.resolve(
        method === 'commands.catalog'
          ? ALIAS_CATALOG
          : { items: [{ text: '/btw ', display: '/btw', meta: 'Run a prompt in the background' }] }
      )
    )

    const api = harness({ request } as unknown as HermesGateway)

    const items = await completions(api, 'bt')

    expect(commandsOf(items)).toEqual(['/background'])
    expect(items[0]?.label).toBe('background (btw)')
    expect(items[0]?.description).toBe('Run a prompt in the background')
  })

  it('waits briefly for an in-flight catalog before canonicalizing alias results', async () => {
    let resolveCatalog!: (catalog: CommandsCatalogLike) => void
    const catalogPromise = new Promise<CommandsCatalogLike>(resolve => {
      resolveCatalog = resolve
    })
    const request = vi.fn().mockImplementation((method: string) =>
      method === 'commands.catalog'
        ? catalogPromise
        : Promise.resolve({ items: [{ text: '/btw ', display: '/btw', meta: 'Run a prompt in the background' }] })
    )
    const api = harness({ request } as unknown as HermesGateway)

    await act(async () => {
      api.search('bt')
      await new Promise(resolve => setTimeout(resolve, 120))
    })
    expect(request).toHaveBeenCalledWith('complete.slash', { text: '/bt' })

    await act(async () => {
      resolveCatalog(ALIAS_CATALOG)
      await Promise.resolve()
    })
    const items = api.search('bt')

    expect(commandsOf(items)).toEqual(['/background'])
    expect(items[0]?.label).toBe('background (btw)')
  })

  it('does not withhold command completions when the catalog stalls', async () => {
    const request = vi.fn().mockImplementation((method: string) =>
      method === 'commands.catalog'
        ? new Promise(() => {})
        : Promise.resolve({ items: [{ text: '/help', display: '/help', meta: 'Show help' }] })
    )
    const api = harness({ request } as unknown as HermesGateway)

    await act(async () => {
      api.search('help')
      await new Promise(resolve => setTimeout(resolve, 300))
    })

    expect(commandsOf(api.search('help'))).toEqual(['/help'])
  })

  it('matches aliases case-insensitively', () => {
    rememberDesktopCommandsCatalog(ALIAS_CATALOG)

    expect(
      canonicalizeSlashCommandCompletions(
        [{ text: '/btw', display: '/btw', meta: 'Run a prompt in the background' }],
        '/BT'
      )
    ).toEqual([
      {
        text: '/background',
        display: '/background (btw)',
        meta: 'Run a prompt in the background'
      }
    ])
  })

  it('does not annotate a canonical command match with its aliases', async () => {
    const request = vi.fn().mockResolvedValue({
      items: [{ text: '/background', display: '/background', meta: 'Run a prompt in the background' }]
    })

    const api = harness({ request } as unknown as HermesGateway)

    const items = await completions(api, 'background')

    expect(commandsOf(items)).toEqual(['/background'])
    expect(items[0]?.label).toBe('background')
  })

  it('deduplicates canonical and alias matches into one command row', () => {
    rememberDesktopCommandsCatalog(ALIAS_CATALOG)

    const items = canonicalizeSlashCommandCompletions(
      [
        { text: '/background', display: '/background', meta: 'Run a prompt in the background' },
        { text: '/bg', display: '/bg', meta: 'Run a prompt in the background' },
        { text: '/btw', display: '/btw', meta: 'Run a prompt in the background' }
      ],
      '/b'
    )

    expect(items.map(item => item.text)).toEqual(['/background'])
  })

  // Typing is a search, and a search that hides a match is broken. Order
  // stays the backend's (fuzzy score, then usage) — a second client usage
  // sort is what buried `/review` under skills.
  it('keeps backend order on a typed query and does not hide matches', async () => {
    const request = vi.fn().mockImplementation((method: string) =>
      Promise.resolve(
        method === 'commands.catalog'
          ? RANKED_CATALOG
          : {
              items: [
                {
                  text: '/research-paper-writing',
                  display: '/research-paper-writing',
                  kind: 'skill',
                  meta: 'Write a paper'
                },
                { text: '/research', display: '/research', kind: 'skill', meta: 'Look it up' }
              ]
            }
      )
    )

    const api = harness({ request } as unknown as HermesGateway)

    expect(commandsOf(await completions(api, 'research'))).toEqual(['/research-paper-writing', '/research'])
  })

  it('keeps a registry command in Commands even when the desktop table has no row', async () => {
    const request = vi.fn().mockImplementation((method: string) =>
      Promise.resolve(
        method === 'commands.catalog'
          ? CATALOG
          : {
              items: [
                { text: '/refine', display: '/refine', kind: 'command', meta: 'Review this conversation' },
                { text: '/docx', display: '/docx', kind: 'skill', meta: 'Edit Word documents' },
                { text: '/compress', display: '/compress', kind: 'command', meta: 'Compress context' }
              ]
            }
      )
    )

    const api = harness({ request } as unknown as HermesGateway)
    const items = await completions(api, 're')

    const groupOf = (command: string) =>
      (items.find(item => (item.metadata as { command?: string })?.command === command)?.metadata as { group?: string })
        ?.group

    expect(commandsOf(items)).toEqual(['/refine', '/compress', '/docx'])
    expect(groupOf('/refine')).toBe('Commands')
    expect(groupOf('/compress')).toBe('Commands')
    expect(groupOf('/docx')).toBe('Skills')
  })
})
