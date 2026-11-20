/**
 * "New chat with this bot" refusals (#154651): an unroutable row — a local
 * bot with no source-qualified route, or an orphaned row whose connection was
 * deleted — gets its OWN accurate message, not the outdated-app text. The old
 * code surfaced "Update Hermes Desktop…" for both, which sent a user on a
 * current build chasing an update that does not exist, and the strict route
 * resolver threw out of the context menu for the orphaned row.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

interface Store<T> {
  get: () => T
  listen: (listener: (value: T) => void) => () => void
}

const { state } = vi.hoisted(() => ({
  state: {
    connectionId: 'source-a',
    newChat: null as null | ((route: unknown, options: unknown) => void),
    notify: null as null | ((notification: { kind: string; message: string }) => void),
    profile: 'default'
  }
}))

const store = <T>(read: () => T): Store<T> => ({ get: read, listen: () => () => undefined })

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    host: {
      get newChat() {
        return state.newChat ?? undefined
      },
      get notify() {
        return state.notify ?? undefined
      },
      state: {
        connectionId: store(() => state.connectionId),
        profile: store(() => state.profile)
      }
    },
    queryClient: { invalidateQueries: vi.fn() },
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({
  getPluginCtx: () => ({ i18n: { t: (key: string) => key } }),
  ID: 'hermes-bots'
}))

async function load() {
  vi.resetModules()

  return import('./data')
}

beforeEach(() => {
  state.connectionId = 'source-a'
  state.newChat = vi.fn()
  state.notify = vi.fn()
  state.profile = 'default'
})

describe('newBotChat refusals — the message names the real cause (#154651)', () => {
  it('tells a local (not source-scoped) bot has no routable connection, without refusing as outdated', async () => {
    const { newBotChat } = await load()

    newBotChat({ name: 'worker' })

    expect(state.notify).toHaveBeenCalledWith({ kind: 'error', message: 'bot.noRoutableConnection' })
    expect(state.newChat).not.toHaveBeenCalled()
  })

  it('does not throw for an orphaned row whose owner connection is gone', async () => {
    const { newBotChat } = await load()

    expect(() => newBotChat({ name: 'lost', sourceScoped: true })).not.toThrow()

    expect(state.notify).toHaveBeenCalledWith({ kind: 'error', message: 'bot.noRoutableConnection' })
    expect(state.newChat).not.toHaveBeenCalled()
  })

  it('still routes a source-scoped row through host.newChat', async () => {
    const { newBotChat } = await load()

    newBotChat({ connectionId: 'conn-1', name: 'worker', sourceScoped: true })

    expect(state.notify).not.toHaveBeenCalled()
    expect(state.newChat).toHaveBeenCalledTimes(1)
    expect(state.newChat).toHaveBeenCalledWith(
      { connectionId: 'conn-1', mode: 'remote', profile: 'worker', targetProfile: 'worker' },
      { workspaceMode: 'bots', workspaceOwnerKey: 'bot:conn-1::worker' }
    )
  })

  it('keeps the outdated-app message for a host without newChat', async () => {
    state.newChat = null
    const { newBotChat } = await load()

    newBotChat({ connectionId: 'conn-1', name: 'worker', sourceScoped: true })

    expect(state.notify).toHaveBeenCalledWith({ kind: 'error', message: 'bot.openAnotherChatUnsupported' })
  })
})
