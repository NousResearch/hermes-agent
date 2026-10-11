/**
 * #126732: Telegram DM-topic threads surface in Bot Mode.
 *
 * A bot serving a Telegram gateway with DM topic mode holds one session per
 * DM topic (same chat_id, different thread_id). Those rows live in the bot
 * profile's state.db (source=telegram, messaging slice only) and were
 * invisible in Bot Mode: the bot row opens the canonical Bot Chat (by name,
 * never recency), and the Sessions sidebar fetched/filtered the ambient
 * gateway scope instead of the selected bot's profile — while every
 * recents-only session lookup missed messaging rows entirely, so even a
 * visible Telegram row had no working open/rename/delete.
 *
 * This locks in the two halves:
 * - Sidebar fetch + display follow the selected bot's workspace route
 *   (logical profile for override-aware fetch, backend target for row
 *   matching); All-profiles keeps fanning out; blocked/empty targets fall
 *   back to the ambient scope. Lane isolation (one row per thread) and the
 *   hidden-flag contract (plumbing hidden, gateway threads visible) are
 *   untouched.
 * - Every recents-only owner lookup resolves across all slices (recents +
 *   cron + messaging) via the shared owner index, so a Telegram row is
 *   findable/renamable/deletable from any surface.
 */

import { beforeEach, describe, expect, it } from 'vitest'

import {
  $workspaceMode,
  $workspaceNewSessionTarget,
} from '@/components/pane-shell/workspace-scope'
import {
  $messagingSessions,
  $sessions,
  ownerLookupSessionRows,
  setMessagingSessions,
} from '@/store/session'

import { botModeFetchScope } from './use-session-list-actions'

const telegramRow = (id: string, profile = 'mybot') =>
  ({
    id,
    profile,
    source: 'telegram',
    title: `Telegram ${id}`,
  }) as never

const desktopRow = (id: string, profile = 'default') =>
  ({
    id,
    profile,
    source: 'desktop',
    title: `Desk ${id}`,
  }) as never

beforeEach(() => {
  $sessions.set([])
  $messagingSessions.set([])
  $workspaceMode.set('sessions')
  $workspaceNewSessionTarget.set(null)
})

describe('botModeFetchScope', () => {
  it('returns null outside Bot Mode, for All-profiles, and for blocked targets', () => {
    // Sessions mode with a bot route still set (stale target): ambient wins.
    $workspaceMode.set('sessions')
    $workspaceNewSessionTarget.set({
      kind: 'route',
      route: { connectionId: 'local', mode: 'local', profile: 'mybot' },
    } as never)
    expect(botModeFetchScope('default')).toBeNull()

    // Bot Mode but explicit All-profiles: unified fetch already includes the bot.
    $workspaceMode.set('bots')
    expect(botModeFetchScope('__all__')).toBeNull()

    // Bot Mode with no selection (group chat / empty): ambient scope.
    $workspaceNewSessionTarget.set({ kind: 'blocked', message: 'nope' } as never)
    expect(botModeFetchScope('default')).toBeNull()
    $workspaceNewSessionTarget.set(null)
    expect(botModeFetchScope('default')).toBeNull()
  })

  it('scopes fetch to the selected local bot (logical == backend)', () => {
    $workspaceMode.set('bots')
    $workspaceNewSessionTarget.set({
      kind: 'route',
      route: { connectionId: 'local', mode: 'local', profile: 'mybot', targetProfile: 'mybot' },
    } as never)

    expect(botModeFetchScope('default')).toEqual({ displayProfile: 'mybot', fetchProfile: 'mybot' })
  })

  it('fetches via the logical alias but displays the backend target', () => {
    $workspaceMode.set('bots')
    $workspaceNewSessionTarget.set({
      kind: 'route',
      route: { connectionId: 'remote-a', mode: 'remote', profile: 'moxie', targetProfile: 'default' },
    } as never)

    // Fetch routes through the alias override; rows carry the backend stamp.
    expect(botModeFetchScope('default')).toEqual({ displayProfile: 'default', fetchProfile: 'moxie' })
  })
})

describe('messaging rows stay resolvable across slices (#126732)', () => {
  it('ownerLookupSessionRows finds a telegram row that recents alone misses', () => {
    $sessions.set([desktopRow('desk-1')])
    $messagingSessions.set([telegramRow('tg-1')])

    expect($sessions.get().find(s => (s as { id: string }).id === 'tg-1')).toBeUndefined()
    expect(ownerLookupSessionRows().find(s => (s as { id: string }).id === 'tg-1')).toMatchObject({
      id: 'tg-1',
      profile: 'mybot',
      source: 'telegram',
    })
  })

  it('keeps distinct DM-topic threads as distinct rows (lane isolation)', () => {
    setMessagingSessions([
      telegramRow('tg-topic-2'),
      telegramRow('tg-topic-3'),
    ] as never)

    const rows = ownerLookupSessionRows().filter(s => (s as { source: string }).source === 'telegram')

    expect(rows.map(s => (s as { id: string }).id).sort()).toEqual(['tg-topic-2', 'tg-topic-3'])
  })
})
