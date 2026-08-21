/**
 * `composer.atCompletions` contributions merge into the `@` popover ahead of
 * the gateway's path results (#88060 — Bot Mode agent handles). Covers: rows
 * appear, prefix filtering is delegated to the source, contributed rows lead,
 * and a throwing source is isolated instead of killing the popover.
 */
import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { COMPOSER_AREAS, type ComposerAtCompletionSource } from '@/app/chat/composer/contrib'
import { registry } from '@/contrib/registry'
import { queryClient } from '@/lib/query-client'

import { useAtCompletions } from './use-at-completions'

const disposers: Array<() => void> = []

function addSource(id: string, provide: ComposerAtCompletionSource['provide']) {
  disposers.push(
    registry.register({
      id,
      area: COMPOSER_AREAS.atCompletions,
      data: { provide } satisfies ComposerAtCompletionSource
    })
  )
}

function gatewayStub(items = [{ text: '@file:src/research-notes.md', display: 'research-notes.md', meta: 'file' }]) {
  return {
    request: vi.fn(async () => ({ items }))
  }
}

/** Drive the adapter like the popover does: search, let the debounce and
 *  fetch settle, then read the settled items from a second search call
 *  (the adapter returns the currently-held rows synchronously). */
async function searchAndRead(
  result: { current: { adapter: { search?: (q: string) => readonly { label: string }[] } } },
  q: string
): Promise<string[]> {
  act(() => {
    result.current.adapter.search?.(q)
  })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(300)
  })

  let rows: readonly { label: string }[] = []
  act(() => {
    rows = result.current.adapter.search?.(q) || []
  })

  return rows.map(r => r.label)
}

afterEach(() => {
  disposers.splice(0).forEach(d => d())
  queryClient.clear()
  vi.useRealTimers()
})

describe('contributed @ completion sources', () => {
  it('merges contributed rows ahead of gateway path results', async () => {
    vi.useFakeTimers()
    addSource('bots', q => ('researcher'.startsWith(q) ? [{ insert: '@researcher', meta: 'Bot · Researcher' }] : []))

    const { result } = renderHook(() =>
      useAtCompletions({ gateway: gatewayStub() as never, sessionId: 's1', cwd: '/repo' })
    )

    const rows = await searchAndRead(result, 'rese')
    expect(rows[0]).toBe('@researcher')
    expect(rows.some(l => l.includes('research-notes.md'))).toBe(true)
  })

  it('deduplicates contributed and core profile handles without collapsing distinct references', async () => {
    vi.useFakeTimers()
    addSource('bots', q => ('researcher'.startsWith(q) ? [{ insert: '@Researcher', meta: 'Bot · Researcher' }] : []))

    const gateway = gatewayStub([
      { text: '@researcher', display: '@researcher', meta: 'Research and intelligence analyst' },
      { text: '@researcher-homelab', display: '@researcher-homelab', meta: 'agent profile' },
      { text: '@file:src/researcher.md', display: 'researcher.md', meta: 'file' }
    ])

    const { result } = renderHook(() => useAtCompletions({ gateway: gateway as never, sessionId: 's1', cwd: '/repo' }))

    const rows = await searchAndRead(result, 'rese')
    expect(rows.filter(label => label.toLowerCase() === '@researcher')).toEqual(['@Researcher'])
    expect(rows).toContain('@researcher-homelab')
    expect(rows.some(label => label.includes('researcher.md'))).toBe(true)
  })

  it('drops rows when the query does not match the source filter', async () => {
    vi.useFakeTimers()
    addSource('bots', q => ('researcher'.startsWith(q) ? [{ insert: '@researcher', meta: 'Bot' }] : []))

    const { result } = renderHook(() =>
      useAtCompletions({ gateway: gatewayStub() as never, sessionId: 's1', cwd: '/repo' })
    )

    const rows = await searchAndRead(result, 'zzz')
    expect(rows).not.toContain('@researcher')
  })

  it('isolates a throwing source instead of killing the popover', async () => {
    vi.useFakeTimers()
    addSource('broken', () => {
      throw new Error('boom')
    })
    addSource('bots', () => [{ insert: '@researcher' }])

    const { result } = renderHook(() =>
      useAtCompletions({ gateway: gatewayStub() as never, sessionId: 's1', cwd: '/repo' })
    )

    const rows = await searchAndRead(result, 'rese')
    expect(rows[0]).toBe('@researcher')
    expect(rows.length).toBeGreaterThan(1)
  })
})
