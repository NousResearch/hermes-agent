/**
 * A bot row click is "go to this bot", not "open its Bot Chat". Before this,
 * every click resolved the canonical chat by name and opened it as a tab — and
 * with no record of a close anywhere (this plugin keeps no closed set; core's
 * tile bucket only forgets), a Bot Chat the user closed came back beside every
 * newer thread on every bot switch. Now a bot whose workspace already holds
 * tabs comes back to the one the user left; the forever-chat is opened only
 * when nothing is open, or on the explicit ask (the row menu's "Open Bot Chat").
 *
 * Ported from tests/bot-row-keeps-closed-chat.test.mjs, which drove a `vm`
 * copy of plugin.js. Its two source-reading cases are dropped for real
 * assertions: the menu's call site is now a render in bot-row.test.tsx, and
 * the reclaim guard's text is asserted here as the claim-shape invariant the
 * guard actually reads.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { openBotCanonicalChat, prepareBotSource, reconcileBotProfileSessions } = vi.hoisted(() => ({
  openBotCanonicalChat: vi.fn(),
  prepareBotSource: vi.fn(),
  reconcileBotProfileSessions: vi.fn()
}))

vi.mock('./canonical-chat', () => ({
  CANONICAL_CHAT_TITLE: 'Bot Chat',
  ensureBotMetadata: vi.fn(async () => ({})),
  notifyBotOpenFailure: vi.fn(),
  openBotCanonicalChat,
  prepareBotSource,
  PROFILE_SESSION_LIST_LIMIT: 200
}))

vi.mock('./session-sweep', () => ({ reconcileBotProfileSessions }))

const { host } = await import('@hermes/plugin-sdk')
const { $openBotChat, $selectedBot } = await import('./bot-state')
const { openRosterBot } = await import('./roster-actions')

const bot = { connectionId: 'local', name: 'alpha' } as RosterRow

/** Swap in a focus API for one test, restoring whatever the SDK really has —
 *  including its absence, which is the older-shell case. */
function withFocusApi(impl: null | (() => null | string)) {
  const had = Object.hasOwn(host, 'focusOpenWorkspaceSession')
  const original = host.focusOpenWorkspaceSession

  if (impl) {
    host.focusOpenWorkspaceSession = impl
  } else {
    // @ts-expect-error — modelling a Desktop old enough to lack the verb.
    delete host.focusOpenWorkspaceSession
  }

  return () => {
    if (had) {
      host.focusOpenWorkspaceSession = original
    } else {
      // @ts-expect-error — same.
      delete host.focusOpenWorkspaceSession
    }
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  prepareBotSource.mockResolvedValue(undefined)
  reconcileBotProfileSessions.mockResolvedValue(undefined)
  openBotCanonicalChat.mockResolvedValue({ openedId: 'bot-chat', registryId: 'bot-chat' })
  $openBotChat.set(null)
  $selectedBot.set('')
})

describe('a row click returns to the tabs the bot already has open', () => {
  it('fronts the remembered tab and resolves no canonical chat', async () => {
    const focus = vi.fn(() => 'thread-2')
    const restore = withFocusApi(focus)

    try {
      await expect(openRosterBot(bot)).resolves.toBe(true)

      expect(focus).toHaveBeenCalledWith('bot:alpha', expect.any(Function))
      expect(openBotCanonicalChat).not.toHaveBeenCalled()
      // Open tabs need no source activation either — the bot is already live.
      expect(prepareBotSource).not.toHaveBeenCalled()
      expect(reconcileBotProfileSessions).toHaveBeenCalledWith(bot)
    } finally {
      restore()
    }
  })

  it('claims only the fronted tab, with no registry id', async () => {
    const restore = withFocusApi(() => 'thread-2')

    try {
      await openRosterBot(bot)

      expect($openBotChat.get()).toEqual({
        key: 'local::alpha',
        openedRegistryId: '',
        openedSessionId: 'thread-2'
      })
    } finally {
      restore()
    }
  })
})

describe('the canonical chat still opens when it is what was asked for', () => {
  it('opens it when the bot has nothing open', async () => {
    const restore = withFocusApi(() => null)

    try {
      await expect(openRosterBot(bot)).resolves.toBe(true)

      expect(openBotCanonicalChat).toHaveBeenCalled()
      expect(reconcileBotProfileSessions).toHaveBeenCalledWith(bot)
      expect($openBotChat.get()?.openedRegistryId).toBe('bot-chat')
    } finally {
      restore()
    }
  })

  it('skips the open-tab shortcut on the explicit ask', async () => {
    const focus = vi.fn(() => 'thread-2')
    const restore = withFocusApi(focus)

    try {
      await expect(openRosterBot(bot, { canonical: true })).resolves.toBe(true)

      expect(focus).not.toHaveBeenCalled()
      expect($openBotChat.get()?.openedRegistryId).toBe('bot-chat')
    } finally {
      restore()
    }
  })
})

describe('the fronted-tab shortcut reconciles with the canonical registry (#90102)', () => {
  // The stuck shape: a persisted "Bot Chat" tile names a session the
  // registry no longer resolves to (superseded pointer-era row, re-minted
  // canonical chat, stale finished session). The roster click must judge
  // that tile against the server-resolved canonical_session and fall
  // through to the authoritative registry open instead of fronting it.
  const staleBot = {
    connectionId: 'local',
    name: 'alpha',
    canonical_session: { id: 'bot-chat', resolved_id: 'bot-chat-tip' }
  } as RosterRow

  /** The probe openRosterBot hands the focus verb, captured. */
  function captureProbe() {
    let probe: ((tile: { storedSessionId: string; workspaceTabTitle?: string }) => boolean) | undefined

    const focus = vi.fn((_key: string, isStaleTile?: typeof probe) => {
      probe = isStaleTile

      return null
    })

    return { focus, probe: () => probe }
  }

  it('classifies a canonical-titled tile at a foreign id as stale', async () => {
    const { focus, probe } = captureProbe()
    const restore = withFocusApi(focus as unknown as () => null | string)

    try {
      await openRosterBot(staleBot)

      const isStale = probe()!
      expect(isStale({ storedSessionId: 'old-finished-session', workspaceTabTitle: 'Bot Chat' })).toBe(true)
    } finally {
      restore()
    }
  })

  it('keeps the tile that matches the registry row or its lineage tip', async () => {
    const { focus, probe } = captureProbe()
    const restore = withFocusApi(focus as unknown as () => null | string)

    try {
      await openRosterBot(staleBot)

      const isStale = probe()!
      expect(isStale({ storedSessionId: 'bot-chat', workspaceTabTitle: 'Bot Chat' })).toBe(false)
      expect(isStale({ storedSessionId: 'bot-chat-tip', workspaceTabTitle: 'Bot Chat' })).toBe(false)
    } finally {
      restore()
    }
  })

  it('never judges side-chat tabs — only canonical-titled tiles carry registry identity', async () => {
    const { focus, probe } = captureProbe()
    const restore = withFocusApi(focus as unknown as () => null | string)

    try {
      await openRosterBot(staleBot)

      const isStale = probe()!
      expect(isStale({ storedSessionId: 'scratch-thread', workspaceTabTitle: 'Group: writers' })).toBe(false)
      expect(isStale({ storedSessionId: 'scratch-thread' })).toBe(false)
    } finally {
      restore()
    }
  })

  it('an older gateway without canonical_session cannot judge — every tile survives', async () => {
    const { focus, probe } = captureProbe()
    const restore = withFocusApi(focus as unknown as () => null | string)

    try {
      await openRosterBot(bot) // no canonical_session on this row

      const isStale = probe()!
      expect(isStale({ storedSessionId: 'anything', workspaceTabTitle: 'Bot Chat' })).toBe(false)
    } finally {
      restore()
    }
  })

  it('falls through to the authoritative canonical open when the stale tile was the only tab', async () => {
    // The store discards the stale tile and reports null; the click must
    // then resolve the registry — the backend-truth path — not give up.
    const restore = withFocusApi(() => null)

    try {
      await expect(openRosterBot(staleBot)).resolves.toBe(true)

      expect(openBotCanonicalChat).toHaveBeenCalled()
      expect($openBotChat.get()?.openedRegistryId).toBe('bot-chat')
    } finally {
      restore()
    }
  })
})

describe('a shell that cannot report open tabs behaves as it did before', () => {
  it('opens the canonical chat when the verb is missing', async () => {
    const restore = withFocusApi(null)

    try {
      await expect(openRosterBot(bot)).resolves.toBe(true)

      expect(openBotCanonicalChat).toHaveBeenCalled()
    } finally {
      restore()
    }
  })

  it('opens the canonical chat when the verb throws', async () => {
    const restore = withFocusApi(() => {
      throw new Error('no tree yet')
    })

    try {
      await expect(openRosterBot(bot)).resolves.toBe(true)

      expect(openBotCanonicalChat).toHaveBeenCalled()
    } finally {
      restore()
    }
  })
})

describe('the claim a fronted tab records cannot resurrect the closed chat', () => {
  // The reclaim listener re-resolves the canonical chat for a claim it owns,
  // and guards on the registry id to avoid doing so for a fronted tab. That
  // guard is only correct because a fronted-tab claim leaves the id empty
  // while a real canonical open always fills it — the invariant asserted here.
  it('leaves the registry id empty for a fronted tab', async () => {
    const restore = withFocusApi(() => 'thread-2')

    try {
      await openRosterBot(bot)

      expect($openBotChat.get()?.openedRegistryId).toBe('')
      expect($openBotChat.get()?.openedSessionId).toBeTruthy()
    } finally {
      restore()
    }
  })

  it('fills the registry id for a real canonical open', async () => {
    const restore = withFocusApi(() => null)

    try {
      await openRosterBot(bot)

      expect($openBotChat.get()?.openedRegistryId).toBeTruthy()
    } finally {
      restore()
    }
  })
})
