import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

/**
 * A bot row must open the conversation the user was LAST having with that bot.
 *
 * Symptom (2026-08-21): every bot was welded to one session. Start a new chat
 * with 기획총괄, click 시스템총괄, click back — and the new chat was gone,
 * replaced by the pinned transcript. "세션을 다시 만들어도 다른 봇 갔다가 다시
 * 누르면 그 전 세션으로 다시 돌아와."
 *
 * The pin still owns plumbing (creation, hide sweep, DM delivery); it just
 * must not override a newer real conversation.
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

test('bot row opens the NEWER real conversation instead of the pinned chat', async () => {
  const runtime = loadOpenPath({ openSession: async () => undefined, request: healthyPin() })

  // The roster's freshest visible session is a real conversation the user
  // started after the pin was made.
  const history = { id: 'new-chat', title: '릴시아 카피 회의', message_count: 12, last_active: 9000 }

  const result = await runtime.openBotCanonicalChat('plan', 'pinned-bot-chat', history, history)

  assert.equal(result, 'new-chat', 'should return the newer conversation')
  assert.equal(runtime.opened.length, 1)
  assert.equal(runtime.opened[0].id, 'new-chat', 'must not reopen the pinned transcript')
  assert.equal(runtime.opened[0].options.profile, 'plan')
  assert.equal(
    runtime.opened[0].options.keepAllProfilesScope,
    false,
    'clicking a bot moves the workspace onto that bot'
  )
})

/**
 * The REAL call shape from the roster row — this is what the first fix got
 * wrong. `previewSession` is `bot.preferred_session || last`, so on a pinned
 * bot it resolves to the PIN (preview identity must match click identity).
 * Feeding that as the "newer" candidate made the whole preference dead code:
 * it always saw the pin and short-circuited on "same id", and the user still
 * got the old session back ("다른 봇 눌렀다가 다시 그 봇 누르면 그 전 세션 열림").
 * The freshest visible session has to arrive as its own argument.
 */
test('real roster call: previewSession is the pin, latest arrives separately', async () => {
  const runtime = loadOpenPath({ openSession: async () => undefined, request: healthyPin('pin-1') })

  const pinnedPreview = { id: 'pin-1', title: 'Bot Chat', preview: 'plumbing' }
  const last = { id: 'user-newest', title: '오늘 기획 회의', message_count: 8, last_active: 9999 }

  // Mirrors: openBotCanonicalChat(bot.name, pinnedChat, previewSession, last)
  const result = await runtime.openBotCanonicalChat('plan', 'pin-1', pinnedPreview, last)

  assert.equal(result, 'user-newest', 'must open the newest real conversation, not the pin')
  assert.equal(runtime.opened[0].id, 'user-newest')
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

test('a failed open of the newer session falls back to the pin (row never dies)', async () => {
  const runtime = loadOpenPath({
    openSession: async id => {
      if (id === 'deleted-chat') {
        throw new Error('session not found')
      }

      return undefined
    },
    request: healthyPin('pin-1')
  })

  const history = { id: 'deleted-chat', title: '지워진 대화', message_count: 3 }

  const result = await runtime.openBotCanonicalChat('ops', 'pin-1', history)

  const ids = runtime.opened.map(entry => entry.id)

  assert.ok(ids.includes('deleted-chat'), 'tries the newer session first')
  assert.ok(ids.includes('pin-1'), 'falls back to the verified pin')
  assert.equal(result, 'pin-1', 'row resolves to the pin rather than failing')
})
