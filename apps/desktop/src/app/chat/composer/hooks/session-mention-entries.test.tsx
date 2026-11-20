/**
 * `@session:` in the composer offers other chats (kimi-code#5068's session mentions): loaded
 * sidebar sessions of the active profile first, then the server's full-text search for chats
 * outside the loaded page. Picking a row produces the same `@session:<profile>/<id>` reference a
 * sidebar drag makes, and the gateway's path listing is never asked.
 */
import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import { $activeProfile } from '@/store/profile'
import { $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { useAtCompletions } from './use-at-completions'

const search = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', async importOriginal => ({ ...(await importOriginal<object>()), searchSessions: search }))

function session(id: string, title: string, lastActive: number, extra: Partial<SessionInfo> = {}): SessionInfo {
  return { id, title, last_active: lastActive, started_at: lastActive, preview: null, profile: 'default', ...extra } as SessionInfo
}

async function rowsFor(query: string, gateway = { request: vi.fn(async () => ({ items: [] })) }) {
  const { result } = renderHook(() => useAtCompletions({ gateway: gateway as never, sessionId: 's-live', cwd: '/repo' }))

  act(() => {
    result.current.adapter.search?.(query)
  })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(300)
  })

  let rows: readonly { description?: string; label: string; metadata?: unknown }[] = []
  act(() => {
    rows = (result.current.adapter.search?.(query) || []) as typeof rows
  })

  return {
    descriptions: rows.map(r => r.description),
    gateway,
    rows: rows.map(r => ({ label: r.label, raw: (r.metadata as { rawText: string }).rawText }))
  }
}

afterEach(() => {
  $sessions.set([])
  $activeProfile.set('default')
  search.mockReset()
  queryClient.clear()
  vi.useRealTimers()
})

describe('@session: mentions', () => {
  it('lists loaded chats newest first, then search hits, as session references', async () => {
    vi.useFakeTimers()
    $sessions.set([
      session('old-deploy', 'Deploy notes', 100),
      session('new-deploy', 'Deploy rollback', 300),
      session('other', 'Groceries', 400),
      session('archived', 'Deploy archive', 500, { archived: true }),
      session('work-profile', 'Deploy on work', 600, { profile: 'work' })
    ])
    search.mockResolvedValue({
      results: [
        { session_id: 'new-deploy', title: 'Deploy rollback', snippet: 'dup' },
        { session_id: 'far-away', title: 'Deploy from March', snippet: 'we >>>deploy<<<ed' }
      ]
    })

    const { gateway, rows, descriptions } = await rowsFor('session:deploy')

    expect(rows).toEqual([
      { label: 'Deploy rollback', raw: '@session:default/new-deploy' },
      { label: 'Deploy notes', raw: '@session:default/old-deploy' },
      { label: 'Deploy from March', raw: '@session:default/far-away' }
    ])
    expect(descriptions[2]).toBe('we deployed')
    expect(search).toHaveBeenCalledWith('deploy', 'default')
    expect(gateway.request).not.toHaveBeenCalled()
  })

  it('offers the @session: starter beside the other reference kinds', async () => {
    vi.useFakeTimers()

    const { rows } = await rowsFor('ses')

    expect(rows.map(r => r.raw)).toContain('@session:')
  })
})
