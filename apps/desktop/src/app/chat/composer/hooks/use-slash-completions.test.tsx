import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { I18nProvider, useI18n } from '@/i18n'
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

type HarnessApi = {
  search: (query: string) => readonly Unstable_TriggerItem[]
  setLocale: ReturnType<typeof useI18n>['setLocale']
}

function harness(gateway: HermesGateway, initialLocale: 'en' | 'zh-hant' = 'en') {
  const api: Partial<HarnessApi> = {}

  function Probe() {
    const { setLocale } = useI18n()
    const { adapter } = useSlashCompletions({ gateway })

    api.search = adapter.search
    api.setLocale = setLocale

    return null
  }

  render(
    <I18nProvider configClient={null} initialLocale={initialLocale}>
      <Probe />
    </I18nProvider>
  )

  return api as HarnessApi
}

/** Drive the adapter until its async fetch has settled into `search`'s result. */
async function completions(api: Pick<HarnessApi, 'search'>, query: string) {
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

  it('keeps catalog ids and descriptions raw until the render boundary', async () => {
    const request = vi.fn().mockResolvedValue(CATALOG)
    const api = harness({ request } as unknown as HermesGateway, 'zh-hant')

    const items = await completions(api, '')

    const itemFor = (command: string) =>
      items.find(item => (item.metadata as { command?: string })?.command === command)?.metadata as
        { group?: string; meta?: string } | undefined

    expect(itemFor('/new')).toMatchObject({ group: 'Session', meta: 'Start a new desktop chat' })
    expect(itemFor('/work')).toMatchObject({ group: 'Skills', meta: 'Kick off a task in a fresh worktree' })
  })

  it('localizes the Desktop-owned /resume action without changing its group id', async () => {
    const request = vi.fn().mockResolvedValue(CATALOG)
    const api = harness({ request } as unknown as HermesGateway, 'zh-hant')

    const items = await completions(api, 'resume ')
    const browse = items.find(item => (item.metadata as { action?: string })?.action === 'session-picker')

    expect(browse?.metadata).toMatchObject({
      command: '/resume',
      display: '瀏覽所有工作階段…',
      group: 'Sessions',
      meta: ''
    })
  })

  it('refreshes Desktop-owned completion copy after the locale changes', async () => {
    const request = vi.fn().mockResolvedValue(CATALOG)
    const api = harness({ request } as unknown as HermesGateway)

    const browseMeta = (items: readonly Unstable_TriggerItem[]) =>
      items.find(item => (item.metadata as { action?: string })?.action === 'session-picker')?.metadata as
        { display?: string; group?: string } | undefined

    expect(browseMeta(await completions(api, 'resume '))).toMatchObject({
      display: 'Browse all sessions…',
      group: 'Sessions'
    })

    await act(async () => {
      await api.setLocale('zh-hant')
      await Promise.resolve()
    })

    expect(browseMeta(await completions(api, 'resume '))).toMatchObject({
      display: '瀏覽所有工作階段…',
      group: 'Sessions'
    })
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

  it('keeps typed command usage in English metadata until the render boundary', async () => {
    const request = vi.fn().mockImplementation((method: string) =>
      Promise.resolve(
        method === 'commands.catalog'
          ? CATALOG
          : {
              items: [
                {
                  text: '/save',
                  display: '/save',
                  kind: 'command',
                  meta: 'Save the current transcript (usage: /save <json|md|html> [filename] [redact])'
                }
              ]
            }
      )
    )

    const api = harness({ request } as unknown as HermesGateway, 'zh-hant')
    const [save] = await completions(api, 'sav')

    expect(save.metadata).toMatchObject({
      command: '/save',
      group: 'Commands',
      meta: 'Save the current transcript to JSON (usage: /save <json|md|html> [filename] [redact])'
    })
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
