import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

/**
 * A bot row must open the bot's canonical, pinned Bot Chat.
 *
 * This file previously asserted the opposite — that a row opens the user's
 * NEWEST visible conversation — after a report that a freshly started chat
 * seemed to vanish when clicking away and back. That preference was reverted
 * (2026-08-22): canonical Bot Chats are ALWAYS hidden from the Sessions
 * sidebar (see hide-bot-chats.test.mjs), so the bot row is the ONLY door to
 * the forever-chat. Preferring a newer session made the pinned relationship
 * unreachable from anywhere in the UI — a user lost an entire bot-building
 * history behind a row that previewed one session and opened another.
 *
 * The original complaint has a non-destructive answer: scratch sessions from
 * "New chat with this agent" are not plumbing-titled, so the hide sweep leaves
 * them visible in the Sessions sidebar. They are reachable there; they are
 * simply not what the bot row targets.
 *
 * Documented contract (docs/user-guide/bot-mode): "Click a Bot to land in its
 * chat — every Bot has a canonical, persistent Bot Chat conversation that is
 * created (and pinned) the moment the Bot is born."
 */
function loadOpenPath({ openSession, request }) {
  const start = source.indexOf('const canonicalCreations = new Map()')
  const end = source.indexOf('function displayName(', start)

  assert.notEqual(start, -1, 'canonical creation section is missing')
  assert.notEqual(end, -1, 'canonical creation section delimiter is missing')

  const saved = []
  const opened = []
  const context = {
    host: {
      openSession: async (id, options) => {
        opened.push({ id, options })

        return openSession(id, options)
      },
      request: async (method, params) => request(method, params)
    },
    saveBotMeta: (name, patch) => saved.push({ name, patch: JSON.parse(JSON.stringify(patch)) }),
    $hideBotChats: { get: () => false },
    window: { setTimeout: callback => callback() }
  }

  const section = source
    .slice(start, end)
    .concat('\nglobalThis.__open = { openBotCanonicalChat, newerVisibleBotChat };\n')

  vm.runInNewContext(section, context, { filename: 'canonical-open.js' })

  return { ...context.__open, saved, opened }
}

const noRequests = async () => ({})

/** A live, healthy pin: `profiles.list` resolves it to the canonical Bot Chat.
 *  That verification is the gate the newer-conversation preference sits behind
 *  — with a dead or unverified pin the bot must NOT adopt the profile's latest
 *  row (that would claim an unrelated conversation). */
const healthyPin =
  (pinned = 'pinned-bot-chat') =>
  async (method, params) => {
    if (method === 'profiles.list') {
      const name = Object.keys(params?.preferred_session_ids ?? { ops: 1 })[0]

      return {
        profiles: [{ name, preferred_session: { id: pinned, resolved_id: pinned, title: 'Bot Chat' } }]
      }
    }

    return {}
  }

test('a healthy pin wins over a newer conversation — the row lands in the Bot Chat', async () => {
  const runtime = loadOpenPath({ openSession: async () => undefined, request: healthyPin() })

  // The roster's freshest visible session is a real conversation the user
  // started after the pin was made. It must NOT displace the forever-chat:
  // the pinned chat is hidden from Sessions, so the row is its only door,
  // while this newer session remains reachable in the Sessions sidebar.
  const history = { id: 'new-chat', title: '릴시아 카피 회의', message_count: 12, last_active: 9000 }

  const result = await runtime.openBotCanonicalChat('plan', 'pinned-bot-chat', history, history)

  assert.equal(result, 'pinned-bot-chat', 'should return the pinned Bot Chat')
  assert.equal(runtime.opened.length, 1)
  assert.equal(runtime.opened[0].id, 'pinned-bot-chat', 'must open the pinned forever-chat')
  assert.equal(runtime.opened[0].options.profile, 'plan')
  assert.equal(
    runtime.opened[0].options.keepAllProfilesScope,
    false,
    'clicking a bot moves the workspace onto that bot'
  )
})

/**
 * The REAL call shape from the roster row: `previewSession` is
 * `bot.preferred_session || last`, so on a pinned bot it resolves to the PIN.
 * Preview identity and click identity are the same session by construction
 * (#88200) — which is exactly the property the reverted newer-session
 * preference broke.
 */
test('real roster call: preview identity and click identity are the same session', async () => {
  const runtime = loadOpenPath({ openSession: async () => undefined, request: healthyPin('pin-1') })

  const pinnedPreview = { id: 'pin-1', title: 'Bot Chat', preview: 'plumbing' }

  // Mirrors: openBotCanonicalChat(bot.name, pinnedChat, previewSession)
  const result = await runtime.openBotCanonicalChat('plan', 'pin-1', pinnedPreview)

  assert.equal(result, 'pin-1', 'must open the pinned Bot Chat the row previewed')
  assert.equal(runtime.opened[0].id, 'pin-1')
})

/** Regression guard for the revert: the open path must not consult the
 *  newer-visible-session predicate while the pin is alive. Bot Chats are
 *  hidden from Sessions, so a row that prefers a newer session strands the
 *  forever-chat with no reachable entry point. */
test('the healthy-pin branch never prefers a newer visible session', () => {
  const start = source.indexOf('if (preferred && isCanonicalBotChatHistory(preferred)) {')
  const end = source.indexOf('if (preferred) {', start)

  assert.notEqual(start, -1, 'healthy-pin branch is missing')

  const branch = source.slice(start, end)

  assert.equal(
    branch.includes('newerVisibleBotChat('),
    false,
    'a healthy pin must be opened directly — no newer-session preference'
  )
})

test('the canonical Bot Chat itself never counts as "newer" (it IS the pin)', () => {
  const runtime = loadOpenPath({ openSession: async () => undefined, request: noRequests })

  assert.equal(runtime.newerVisibleBotChat('pin-1', { id: 'hidden-plumbing', title: 'Bot Chat' }), null)
  assert.equal(
    runtime.newerVisibleBotChat('pin-1', { id: 'hidden-plumbing', root_title: 'Bot Chat', title: '자동 제목' }),
    null
  )
})

test('an empty draft never displaces the pinned conversation', () => {
  const runtime = loadOpenPath({ openSession: async () => undefined, request: noRequests })

  assert.equal(runtime.newerVisibleBotChat('pin-1', { id: 'blank', title: '', message_count: 0 }), null)
})

test('a gateway that omits message_count still yields the newer session', () => {
  const runtime = loadOpenPath({ openSession: async () => undefined, request: noRequests })

  assert.equal(runtime.newerVisibleBotChat('pin-1', { id: 'legacy', title: '대화' }), 'legacy')
})

test('history that IS the pin changes nothing', () => {
  const runtime = loadOpenPath({ openSession: async () => undefined, request: noRequests })

  assert.equal(runtime.newerVisibleBotChat('same-id', { id: 'same-id', title: '대화', message_count: 5 }), null)
})

/**
 * Every path that mounts a bot's chat must move the workspace onto that bot.
 *
 * `keepAllProfilesScope` defaults to TRUE in the SDK, which keeps
 * `$activeGatewayProfile` pointing at whatever profile was active before the
 * click. Bot Mode wants the opposite: clicking a bot IS a profile switch, and
 * leaving the scope behind meant sessions created afterwards were filed under
 * the previous bot's profile (measured: four new chats started from three
 * different bots all landed in `ops`).
 *
 * The newly-minted-chat path is asserted separately from the stored-chat path
 * because they are different call sites; a guard on only one of them let the
 * other regress silently.
 */
function creationRuntime({ failFirstOpen = false } = {}) {
  let opens = 0

  return loadOpenPath({
    openSession: async () => {
      opens += 1

      if (failFirstOpen && opens === 1) {
        throw new Error('stored row not persisted yet')
      }

      return undefined
    },
    request: async method => {
      if (method === 'session.create') {
        return { stored_session_id: 'fresh-stored', session_id: 'fresh-runtime' }
      }

      return {}
    }
  })
}

test('a newly minted Bot Chat opens with the workspace following the bot', async () => {
  const runtime = creationRuntime()

  // No pin and no adoptable history — the real "first click on a bot" path.
  const result = await runtime.openBotCanonicalChat('plan', null, null, null)

  assert.equal(result, 'fresh-stored')
  assert.ok(runtime.opened.length >= 1, 'the new chat is mounted')

  for (const entry of runtime.opened) {
    assert.equal(entry.options.keepAllProfilesScope, false, 'creating a bot chat must move the workspace onto that bot')
    assert.equal(entry.options.profile, 'plan')
  }
})

test('the post-kickoff retry open also follows the bot', async () => {
  const runtime = creationRuntime({ failFirstOpen: true })

  await runtime.openBotCanonicalChat('plan', null, null, null)

  assert.equal(runtime.opened.length, 2, 'first open fails, retry runs after the kickoff')
  assert.equal(
    runtime.opened[1].options.keepAllProfilesScope,
    false,
    'the retry must not silently fall back to the SDK default'
  )
})

/** With the newer-session preference gone there is no "try the newer chat,
 *  fall back to the pin" dance: a verified pin is opened directly, and a
 *  failed open of a JUST-verified session is transient (reconnect, backend
 *  restart), so it propagates rather than forking the forever-chat. */
test('a failed open of a verified pin surfaces instead of forking the chat', async () => {
  const runtime = loadOpenPath({
    openSession: async () => {
      throw new Error('session not found')
    },
    request: healthyPin('pin-1')
  })

  await assert.rejects(
    () => runtime.openBotCanonicalChat('ops', 'pin-1', { id: 'pin-1', title: 'Bot Chat' }),
    /session not found/
  )

  assert.deepEqual(
    runtime.saved,
    [],
    'a transient failure must not clear the pin or mint a replacement'
  )
})
