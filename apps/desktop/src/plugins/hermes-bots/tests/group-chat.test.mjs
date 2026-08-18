import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

/** Load the plugin in a vm with a scripted cli.exec so member turns are
 *  deterministic. `turnScript(profile, prompt)` returns the member's reply
 *  text (or throws to simulate a failed turn). */
function load(turnScript) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const calls = []
  const sessions = new Map()
  const runtimeToStored = new Map()
  const titleToStored = new Map()
  let sessionSequence = 0

  const resolveSession = (profile, target) => {
    const stored = runtimeToStored.get(target) || (sessions.has(target) ? target : titleToStored.get(`${profile}::${target}`))
    return stored ? sessions.get(stored) : null
  }
  const context = {
    atom,
    setTimeout: fn => {
      fn()
      return 0
    },
    clearTimeout: () => undefined,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      request: async (method, params) => {
        if (method === 'session.create') {
          sessionSequence += 1
          const stored = `sid-${params.profile}-${sessionSequence}`
          const runtime = `rt-${params.profile}-${sessionSequence}`
          const session = { stored, runtime, profile: params.profile, title: params.title, messages: [] }
          sessions.set(stored, session)
          runtimeToStored.set(runtime, stored)
          titleToStored.set(`${params.profile}::${params.title}`, stored)
          return { session_id: runtime, stored_session_id: stored, message_count: 0, messages: [] }
        }
        if (method === 'session.resume') {
          const session = resolveSession(params.profile, params.session_id)
          if (!session) {
            throw new Error(`session not found: ${params.session_id}`)
          }
          return {
            session_id: session.runtime,
            session_key: session.stored,
            message_count: session.messages.length,
            messages: [...session.messages],
            inflight: false,
            running: false
          }
        }
        if (method === 'prompt.submit') {
          const session = resolveSession(null, params.session_id)
          if (!session) {
            throw new Error(`runtime session not found: ${params.session_id}`)
          }
          session.messages.push({ role: 'user', content: params.text })
          calls.push({
            profile: session.profile,
            prompt: params.text,
            runtime: session.runtime,
            stored: session.stored,
            title: session.title
          })
          const reply = turnScript(session.profile, params.text, calls.length, session)
          session.messages.push({ role: 'assistant', content: reply })
          return {}
        }
        return {}
      },
      state: { profile: { get: () => 'default', listen: () => undefined }, gateway: { listen: () => undefined } },
      notify: () => undefined,
      notifyError: () => undefined
    }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(
      '\nglobalThis.__gc = { sendToGroupChat, runGroupChatRounds, harvestStrandedGroupReply, resolveGroupResponders, parseGroupChatMentions, rotateGroupSpeakers, isGroupPassText, formatGroupChatLine, buildGroupChatTurnPrompt, trimGroupChatLog, disbandGroupChat, updateGroupChat, $groupChats, $groupNeedsYou, $groupChatWorkspace, $botMeta, GROUP_CHAT_MAX_ROUNDS, GROUP_CHAT_MAX_MESSAGES };\n'
    )
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  const storageWrites = new Map()
  context.plugin.register({
    storage: { get: () => null, set: (key, value) => storageWrites.set(key, value) },
    register: () => undefined
  })
  return { ...context.__gc, calls, sessions, storageWrites }
}

const MEMBERS = [{ name: 'research', title: '' }, { name: 'builder', title: '' }, { name: 'ops', title: 'The Ops' }]

function roomLog(gc, group) {
  return (gc.$groupChats.get()[group] || { log: [] }).log
}

test('pass detection: (pass), pass, pass., empty are silence; real text is not', () => {
  const gc = load(() => '(pass)')
  assert.equal(gc.isGroupPassText('(pass)'), true)
  assert.equal(gc.isGroupPassText('pass'), true)
  assert.equal(gc.isGroupPassText('Pass.'), true)
  assert.equal(gc.isGroupPassText('  '), true)
  assert.equal(gc.isGroupPassText('I will pass this to ops'), false)
})

test('mention routing: only @-mentioned members respond; @everyone or none = all', () => {
  const gc = load(() => '(pass)')
  const log = [{ from: { kind: 'user', name: 'You' }, text: '@builder take this one', at: 1 }]
  const one = gc.resolveGroupResponders(log, MEMBERS)
  assert.equal(JSON.stringify(one.map(m => m.name)), JSON.stringify(['builder']))

  const all = gc.resolveGroupResponders([{ from: { kind: 'user', name: 'You' }, text: 'hello team', at: 1 }], MEMBERS)
  assert.equal(all.length, 3)

  const everyone = gc.resolveGroupResponders(
    [{ from: { kind: 'user', name: 'You' }, text: '@everyone standup', at: 1 }],
    MEMBERS
  )
  assert.equal(everyone.length, 3)
})

test('mention routing: display titles resolve to the member and @user never matches a bot', () => {
  const gc = load(() => '(pass)')
  const parsed = gc.parseGroupChatMentions('@theops please check, then ping @user', MEMBERS)
  assert.equal(parsed.mentioned.has('ops'), true)
  assert.equal(parsed.mentioned.size, 1)
})

test('a member @-mentioned by another bot joins the NEXT round', async () => {
  const gc = load((profile, prompt) => {
    if (profile === 'research' && !prompt.includes('(you)')) {
      return 'Interesting — @builder should own this.'
    }
    if (profile === 'builder') {
      return 'On it. OWNER: @builder.'
    }
    return '(pass)'
  })

  gc.sendToGroupChat('Core', [{ name: 'research', title: '' }, { name: 'builder', title: '' }], '@research thoughts?')
  await new Promise(resolve => setTimeout(resolve, 0))
  await new Promise(resolve => setImmediate(resolve))
  // Drain the async loop: poll until running flips false.
  for (let i = 0; i < 200 && (gc.$groupChats.get().Core || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }

  const texts = roomLog(gc, 'Core').map(e => `${e.from.name}: ${e.text}`)
  assert.equal(texts.some(t => t.startsWith('research:')), true)
  assert.equal(texts.some(t => t.startsWith('builder: On it')), true)
})

test('settle: everyone passing ends the room turn with only the user message logged', async () => {
  const gc = load(() => '(pass)')

  gc.sendToGroupChat('Quiet', MEMBERS, 'fyi, deploy went out')
  for (let i = 0; i < 200 && (gc.$groupChats.get().Quiet || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }

  const log = roomLog(gc, 'Quiet')
  assert.equal(log.length, 1)
  assert.equal(log[0].from.kind, 'user')
  // Every member took exactly one turn (round 1), then the settle exit fired.
  assert.equal(gc.calls.length, 3)
})

test('hard caps: chatty members stop at GROUP_CHAT_MAX_MESSAGES total', async () => {
  const gc = load((profile, prompt, n) => `message ${n} — @everyone keep going`)

  gc.sendToGroupChat('Loud', MEMBERS, 'go wild')
  for (let i = 0; i < 400 && (gc.$groupChats.get().Loud || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }

  const memberMessages = roomLog(gc, 'Loud').filter(e => e.from.kind === 'member')
  assert.ok(memberMessages.length <= gc.GROUP_CHAT_MAX_MESSAGES, `posted ${memberMessages.length}`)
})

test('failed member turn is a pass, not a room error', async () => {
  const gc = load(profile => {
    if (profile === 'builder') {
      throw new Error('gateway hiccup')
    }
    return '(pass)'
  })

  gc.sendToGroupChat('Flaky', MEMBERS, 'anyone around?')
  for (let i = 0; i < 200 && (gc.$groupChats.get().Flaky || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }

  const log = roomLog(gc, 'Flaky')
  assert.equal(log.length, 1) // just the user message; no error entries
})

test('delta injection: a second user send only feeds members the NEW messages', async () => {
  const prompts = []
  const gc = load((profile, prompt) => {
    prompts.push({ profile, prompt })
    return '(pass)'
  })

  gc.sendToGroupChat('Delta', [{ name: 'research', title: '' }], 'first message')
  for (let i = 0; i < 200 && (gc.$groupChats.get().Delta || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }
  const firstCount = prompts.length
  gc.sendToGroupChat('Delta', [{ name: 'research', title: '' }], 'second message')
  for (let i = 0; i < 200 && (gc.$groupChats.get().Delta || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }

  const second = prompts.slice(firstCount).find(p => p.prompt.includes('second message'))
  assert.ok(second, 'second turn ran')
  assert.equal(second.prompt.includes('first message'), false, 'first message was already seen — not re-injected')
})

test('concurrent groups sharing one member keep sessions, deltas, and context isolated', async () => {
  const gc = load(() => '(pass)')
  const sharedMember = [{ name: 'research', title: '' }]

  // Start both rooms without waiting for either drive to finish.
  gc.sendToGroupChat('Alpha', sharedMember, 'ALPHA_ONLY_1')
  gc.sendToGroupChat('Beta', sharedMember, 'BETA_ONLY_1')
  for (let i = 0; i < 400; i++) {
    const rooms = gc.$groupChats.get()
    if (!rooms.Alpha?.running && !rooms.Beta?.running) {
      break
    }
    await new Promise(resolve => setImmediate(resolve))
  }

  const alphaFirst = gc.calls.find(call => call.title === 'Group: Alpha')
  const betaFirst = gc.calls.find(call => call.title === 'Group: Beta')
  assert.ok(alphaFirst && betaFirst, 'the shared member took one turn in each room')
  assert.notEqual(alphaFirst.stored, betaFirst.stored, 'each room owns a distinct stored session')
  assert.notEqual(alphaFirst.runtime, betaFirst.runtime, 'each room owns a distinct runtime session')
  assert.equal(alphaFirst.prompt.includes('ALPHA_ONLY_1'), true)
  assert.equal(alphaFirst.prompt.includes('BETA_ONLY_1'), false)
  assert.equal(betaFirst.prompt.includes('BETA_ONLY_1'), true)
  assert.equal(betaFirst.prompt.includes('ALPHA_ONLY_1'), false)

  const roomsAfterFirst = gc.$groupChats.get()
  assert.equal(roomsAfterFirst.Alpha.sessions.research, alphaFirst.stored)
  assert.equal(roomsAfterFirst.Beta.sessions.research, betaFirst.stored)

  // Interleave a second pair. Each room resumes its own session and receives
  // only its unseen room delta, never the sibling room's messages.
  const firstCallCount = gc.calls.length
  gc.sendToGroupChat('Alpha', sharedMember, 'ALPHA_ONLY_2')
  gc.sendToGroupChat('Beta', sharedMember, 'BETA_ONLY_2')
  for (let i = 0; i < 400; i++) {
    const rooms = gc.$groupChats.get()
    if (!rooms.Alpha?.running && !rooms.Beta?.running) {
      break
    }
    await new Promise(resolve => setImmediate(resolve))
  }

  const secondCalls = gc.calls.slice(firstCallCount)
  const alphaSecond = secondCalls.find(call => call.title === 'Group: Alpha')
  const betaSecond = secondCalls.find(call => call.title === 'Group: Beta')
  assert.ok(alphaSecond && betaSecond, 'both rooms resumed for the second pair')
  assert.equal(alphaSecond.stored, alphaFirst.stored)
  assert.equal(betaSecond.stored, betaFirst.stored)
  assert.equal(alphaSecond.prompt.includes('ALPHA_ONLY_2'), true)
  assert.equal(alphaSecond.prompt.includes('ALPHA_ONLY_1'), false, 'Alpha first delta was already seen')
  assert.equal(alphaSecond.prompt.includes('BETA_ONLY_2'), false)
  assert.equal(betaSecond.prompt.includes('BETA_ONLY_2'), true)
  assert.equal(betaSecond.prompt.includes('BETA_ONLY_1'), false, 'Beta first delta was already seen')
  assert.equal(betaSecond.prompt.includes('ALPHA_ONLY_2'), false)

  const alphaSession = gc.sessions.get(alphaFirst.stored)
  const betaSession = gc.sessions.get(betaFirst.stored)
  assert.equal(alphaSession.messages.some(message => String(message.content).includes('BETA_ONLY')), false)
  assert.equal(betaSession.messages.some(message => String(message.content).includes('ALPHA_ONLY')), false)
})

test('needs-you: a member reply mentioning @user badges the group; user send clears it', async () => {
  const gc = load(profile => (profile === 'research' ? 'Blocked on billing access — @user which account?' : '(pass)'))

  gc.sendToGroupChat('Escalate', [{ name: 'research', title: '' }], 'sort out the invoices')
  for (let i = 0; i < 200 && (gc.$groupChats.get().Escalate || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }
  assert.equal(gc.$groupNeedsYou.get().Escalate, true)

  const gc2 = gc // same room: user reply clears
  gc2.sendToGroupChat('Escalate', [{ name: 'research', title: '' }], 'use the ops account')
  assert.equal(gc2.$groupNeedsYou.get().Escalate, false)
})

test('turn transport is gateway-native (session RPCs) and hostile text rides verbatim', async () => {
  const gc = load(() => '(pass)')

  gc.sendToGroupChat('Rpc', [{ name: 'research', title: '' }], 'hello "there" `whoami` $(id)')
  for (let i = 0; i < 200 && (gc.$groupChats.get().Rpc || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }

  const call = gc.calls[0]
  assert.equal(call.profile, 'research')
  // Hostile text is a JSON string in an RPC param — never a shell string.
  assert.equal(call.prompt.includes('hello "there" `whoami` $(id)'), true)
  // The per-group session is created with the room title.
  assert.match(pluginSource, /title,\n/)
  assert.match(pluginSource, /const title = `Group: \$\{group\}`/)
})

test('log trimming keeps watermarks consistent', () => {
  const gc = load(() => '(pass)')
  const log = Array.from({ length: 200 }, (_, i) => ({ from: { kind: 'user', name: 'You' }, text: `m${i}`, at: i }))
  const { log: trimmed, watermarks } = gc.trimGroupChatLog(log, { research: 150, builder: 10 }, 96)
  assert.equal(trimmed.length, 96)
  assert.equal(watermarks.research, 150 - 104)
  assert.equal(watermarks.builder, 0)
})

test('source contract: workspace + main-window door + prompt rules are wired', () => {
  assert.match(pluginSource, /function GroupChatWorkspace\(/)
  // Group rows open through the main-window door, feature-detected with the
  // in-panel room as the older-desktop fallback.
  assert.match(pluginSource, /function openGroupChat\(/)
  assert.match(pluginSource, /typeof host\.openWorkspace === 'function'/)
  assert.match(pluginSource, /\$groupChatWorkspace\.set\(group\)/)
  assert.match(pluginSource, /reply with exactly "\(pass\)"/i)
  assert.match(pluginSource, /\[Group chat: "\$\{groupName\}"\]/)
})

test('disband: removes only this membership, room log, workspace, and needs-you state', async () => {
  const gc = load(() => '(pass)')

  // Two rooms; disband one.
  gc.sendToGroupChat('Keep', [{ name: 'research', title: '' }], 'hello keepers')
  for (let i = 0; i < 200 && (gc.$groupChats.get().Keep || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }
  gc.sendToGroupChat('Gone', [{ name: 'builder', title: '' }], 'hello goners')
  for (let i = 0; i < 200 && (gc.$groupChats.get().Gone || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }

  const rooms = { ...gc.$groupChats.get() }
  rooms.Keep = {
    ...rooms.Keep,
    members: [{ name: 'remote', remoteSource: true, sourceScoped: true, connectionId: 'remote-1' }]
  }
  gc.$groupChats.set(rooms)

  gc.$botMeta.set({
    builder: { groups: ['Gone', 'Keep'], group: 'Gone' },
    research: { groups: ['Keep'], group: 'Keep' }
  })
  gc.$groupChatWorkspace.set('Gone')
  gc.$groupNeedsYou.set({ Gone: true, Keep: true })

  await gc.disbandGroupChat('Gone', [{ name: 'builder' }])

  // Room state: gone from the atom (no running drive, so no tombstone).
  assert.equal(gc.$groupChats.get().Gone, undefined)
  assert.ok(gc.$groupChats.get().Keep, 'other rooms untouched')
  // The open room view closed; needs-you cleared for the disbanded room only.
  assert.equal(gc.$groupChatWorkspace.get(), null)
  assert.equal(gc.$groupNeedsYou.get().Gone, undefined)
  assert.equal(gc.$groupNeedsYou.get().Keep, true)
  // Disband removes only this membership; other groups survive.
  assert.equal(JSON.stringify(gc.$botMeta.get().builder.groups), JSON.stringify(['Keep']))
  assert.equal(gc.$botMeta.get().builder.group, 'Keep')
  assert.equal(JSON.stringify(gc.$botMeta.get().research.groups), JSON.stringify(['Keep']))
  assert.equal(gc.$botMeta.get().research.group, 'Keep')
  // Persisted room map no longer carries the room.
  const durable = gc.storageWrites.get('group-chats')
  assert.ok(durable && !('Gone' in durable), 'disbanded room not persisted')
  assert.ok('Keep' in durable, 'surviving room still persisted')
  assert.equal(durable.Keep.members.length, 1, 'surviving room keeps remote member descriptors')
  assert.equal(durable.Keep.members[0].connectionId, 'remote-1')
})

test('disband: skips source-qualified remote members instead of mutating same-named local metadata', async () => {
  const gc = load(() => '(pass)')
  gc.$botMeta.set({ builder: { groups: ['Keep'], group: 'Keep' } })

  await gc.disbandGroupChat('Remote', [
    { name: 'builder', remoteSource: true, sourceScoped: true, connectionId: 'remote-1' }
  ])

  assert.equal(JSON.stringify(gc.$botMeta.get().builder.groups), JSON.stringify(['Keep']))
  assert.equal(gc.$botMeta.get().builder.group, 'Keep')
  assert.equal(gc.$botMeta.get()['[object Object]'], undefined)
})

test('disband: a running room leaves an epoch-bumped empty tombstone so in-flight turns bail', async () => {
  const gc = load(() => '(pass)')

  gc.sendToGroupChat('Live', [{ name: 'research', title: '' }], 'kick off')
  for (let i = 0; i < 200 && (gc.$groupChats.get().Live || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }

  // Simulate a drive still in flight at disband time.
  const rooms = { ...gc.$groupChats.get() }
  rooms.Live = { ...rooms.Live, running: true, epoch: 3 }
  gc.$groupChats.set(rooms)

  await gc.disbandGroupChat('Live', [{ name: 'research' }])

  const tomb = gc.$groupChats.get().Live
  assert.ok(tomb, 'tombstone present while a drive is mid-turn')
  assert.equal(tomb.log.length, 0)
  assert.equal(tomb.running, false)
  assert.equal(tomb.epoch, 4, 'epoch bumped so the loop bails at its member boundary')
  const durable = gc.storageWrites.get('group-chats')
  assert.ok(!durable || !('Live' in (durable || {})), 'tombstone is never persisted')
})

test('source contract: workspace header offers disband behind a ConfirmDialog', () => {
  assert.match(pluginSource, /function disbandGroupChat\(/)
  assert.match(pluginSource, /Disband group chat\?/)
  assert.match(pluginSource, /title: `Disband the \$\{group\} group chat`/)
})

test('default profile speaks as Hermes in room transcripts, not @default', () => {
  const gc = load(() => '(pass)')
  const line = gc.formatGroupChatLine({ from: { kind: 'member', name: 'default' }, text: 'hello room' }, 'builder')
  assert.equal(line, 'Hermes: hello room')
  assert.doesNotMatch(line, /default/)

  // Other members keep their profile name; the (you) suffix survives.
  const you = gc.formatGroupChatLine({ from: { kind: 'member', name: 'default' }, text: 'hi' }, 'default')
  assert.equal(you, 'Hermes (you): hi')
  const plain = gc.formatGroupChatLine({ from: { kind: 'member', name: 'builder' }, text: 'yo' }, 'research')
  assert.equal(plain, 'builder: yo')
})

test('turn prompt addresses the default profile as @hermes', () => {
  const gc = load(() => '(pass)')
  const prompt = gc.buildGroupChatTurnPrompt({
    groupName: 'Core',
    members: [{ name: 'default', title: '' }, { name: 'builder', title: '' }],
    viewer: { name: 'default', title: '' },
    deltaLines: []
  })
  assert.match(prompt, /You are @hermes,/)
  assert.doesNotMatch(prompt, /@default\b/)

  const peerView = gc.buildGroupChatTurnPrompt({
    groupName: 'Core',
    members: [{ name: 'default', title: '' }, { name: 'builder', title: '' }],
    viewer: { name: 'builder', title: '' },
    deltaLines: []
  })
  assert.match(peerView, /group chat with @hermes/)
})

test('mention routing: @hermes resolves to the default member', () => {
  const gc = load(() => '(pass)')
  const members = [{ name: 'default', title: '' }, { name: 'builder', title: '' }]
  const parsed = gc.parseGroupChatMentions('@hermes take a look', members)
  assert.equal(parsed.mentioned.has('default'), true)
  assert.equal(parsed.mentioned.size, 1)
})

test('source contract: workspace speaker labels use displayName with a click-to-reveal handle', () => {
  // Speaker labels come from the roster displayName (default → Hermes)…
  assert.match(pluginSource, /displayName\(member \|\| \{ name: entry\.from\.name \}, meta\)/)
  // …and clicking a speaker reveals the full disambiguated handle, with the
  // gateway/device name appended for cross-connection speakers.
  assert.match(pluginSource, /setRevealedSpeaker\(revealed \? null : entryKey\)/)
  assert.match(pluginSource, /\$\{display\}\$\{entry\.from\.source \? `-\$\{entry\.from\.source\}` : ''\} \(@\$\{botHandle\(entry\.from\.name, member \|\| undefined\)\}\)/)
})

test('source contract: room messages carry the speaker avatar via the roster appearance pipeline', () => {
  const start = pluginSource.indexOf('function GroupChatWorkspace(')
  const end = pluginSource.indexOf('function BotsPane(')
  const workspace = pluginSource.slice(start, end === -1 ? undefined : end)

  // Per-message avatar: appearance resolved the same way as BotRow (custom
  // image/pet honored, backfilled PNG dropped so the math face animates).
  assert.match(workspace, /botAppearance\(entry\.from\.name, meta\)/)
  assert.match(workspace, /image && !isBackfilledFacePng\(image\)/)
  assert.match(workspace, /jsx\(BotFace, \{\s*shape,\s*color,\s*image: photo \? image : null,\s*size: 24,\s*name: entry\.from\.name/)

  // Header shows the member faces (capped) with a names tooltip.
  assert.match(workspace, /members\.slice\(0, 6\)\.map\(/)
  assert.match(workspace, /title: members\.map\(b => displayName\(b, botRosterMeta\(b, allMeta\)\)\)\.join\(', '\)/)
})

test('stranded harvest: a timed-out turn whose reply landed late posts into the room and clears the marker', async () => {
  const gc = load(() => '(pass)')

  // Room with a stranded marker for research: baseline 0 messages.
  gc.updateGroupChat('Late', r => {
    r.stranded = { research: 0 }
    r.sessions = { research: 'sid-research' }
    return r
  })
  // The member's session finished after we stopped waiting.
  gc.sessions.set('sid-research', {
    stored: 'sid-research',
    runtime: 'rt-research',
    profile: 'research',
    title: 'Group: Late',
    messages: [
      { role: 'user', content: 'the turn prompt' },
      { role: 'assistant', content: 'Here is the full research result, delivered late.' }
    ]
  })

  await gc.harvestStrandedGroupReply('Late', { name: 'research', title: '' })

  const log = roomLog(gc, 'Late')
  assert.equal(log.length, 1)
  assert.equal(log[0].from.name, 'research')
  assert.match(log[0].text, /delivered late/)
  assert.equal(gc.$groupChats.get().Late.stranded.research, undefined, 'marker consumed')
})

test('stranded harvest: a late (pass) or no-new-message consumes the marker without posting', async () => {
  const gc = load(() => '(pass)')

  gc.updateGroupChat('Quiet2', r => {
    r.stranded = { builder: 2 }
    r.sessions = { builder: 'sid-builder' }
    return r
  })
  gc.sessions.set('sid-builder', {
    stored: 'sid-builder',
    runtime: 'rt-builder',
    profile: 'builder',
    title: 'Group: Quiet2',
    messages: [
      { role: 'user', content: 'p1' },
      { role: 'user', content: 'prompt' },
      { role: 'assistant', content: '(pass)' }
    ]
  })

  await gc.harvestStrandedGroupReply('Quiet2', { name: 'builder', title: '' })

  assert.equal(roomLog(gc, 'Quiet2').length, 0)
  assert.equal(gc.$groupChats.get().Quiet2.stranded.builder, undefined)
})

test('stranded markers persist so late replies survive a window reload', async () => {
  const gc = load(() => '(pass)')

  gc.updateGroupChat('Persist', r => {
    r.stranded = { research: 3 }
    return r
  })

  const durable = gc.storageWrites.get('group-chats')
  assert.ok(durable && durable.Persist, 'room persisted')
  assert.equal(durable.Persist.stranded.research, 3, 'stranded marker rides the durable map')
})

test('source contract: long visible turns extend the deadline up to a hard cap', () => {
  assert.match(pluginSource, /const GROUP_TURN_HARD_CAP_MS = /)
  assert.match(pluginSource, /deadline = Math\.min\(started \+ GROUP_TURN_HARD_CAP_MS/)
})

test('source contract: the working line names the member on turn', () => {
  assert.match(pluginSource, /is thinking…/)
  assert.match(pluginSource, /r\.turn = member\.name/)
  assert.match(pluginSource, /r\.turn = null/)
})

test('source contract: creating a group with a taken name mints a fresh room, never reopens the old log', () => {
  assert.match(pluginSource, /const taken = new Set\(Object\.keys\(\$groupChats\.get\(\)\)\)/)
  assert.match(pluginSource, /while \(taken\.has\(`\$\{groupName\} \$\{n\}`\)\)/)
})

test('turn prompt: results are full quality — only chatter is asked to stay short', () => {
  const gc = load(() => '(pass)')
  const prompt = gc.buildGroupChatTurnPrompt({
    groupName: 'Core',
    members: [{ name: 'research', title: '' }, { name: 'builder', title: '' }],
    viewer: { name: 'research', title: '' },
    deltaLines: []
  })
  assert.match(prompt, /never thin out real content/i)
  assert.match(prompt, /Keep chatter short/i)
})

test('threads: room composer mints a new thread; replies land in it', async () => {
  const gc = load(() => '(pass)')

  const t1 = gc.sendToGroupChat('Rooms', [{ name: 'research', title: '' }], 'first topic')
  for (let i = 0; i < 200 && (gc.$groupChats.get().Rooms || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }
  const t2 = gc.sendToGroupChat('Rooms', [{ name: 'research', title: '' }], 'second topic')
  for (let i = 0; i < 200 && (gc.$groupChats.get().Rooms || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }

  assert.ok(t1 && t2 && t1 !== t2, 'each composer send mints a distinct thread')
  const log = roomLog(gc, 'Rooms')
  assert.equal(log[0].thread, t1)
  assert.equal(log[1].thread, t2)
})

test('threads: replying with an explicit thread id continues that thread and scopes the member delta to it', async () => {
  const prompts = []
  const gc = load((profile, prompt) => {
    prompts.push(prompt)
    return prompt.includes('billing') ? 'On the billing fix.' : '(pass)'
  })

  const billing = gc.sendToGroupChat('Scoped', [{ name: 'research', title: '' }], 'fix the billing bug')
  for (let i = 0; i < 200 && (gc.$groupChats.get().Scoped || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }
  gc.sendToGroupChat('Scoped', [{ name: 'research', title: '' }], 'research pricing')
  for (let i = 0; i < 200 && (gc.$groupChats.get().Scoped || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }

  // Continue the BILLING thread explicitly.
  const again = gc.sendToGroupChat('Scoped', [{ name: 'research', title: '' }], 'billing follow-up: ship it', billing)
  for (let i = 0; i < 200 && (gc.$groupChats.get().Scoped || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }

  assert.equal(again, billing, 'explicit thread id is reused, not re-minted')
  const followUpPrompt = prompts.find(p => p.includes('ship it'))
  assert.ok(followUpPrompt, 'follow-up turn ran')
  assert.equal(followUpPrompt.includes('research pricing'), false, 'other thread never leaks into the delta')

  // Member replies carry the thread of the turn that produced them.
  const memberEntries = roomLog(gc, 'Scoped').filter(e => e.from.kind === 'member')
  assert.ok(memberEntries.length >= 1)
  assert.ok(memberEntries.every(e => e.thread === billing), 'replies land in the thread that triggered them')
})

test('threads: hydration assigns legacy thread ids — lull splits, follow-ups stay together', () => {
  const gc = load(() => '(pass)')
  assert.match(pluginSource, /function assignLegacyThreads\(log\)/)

  const fn = new Function(
    `${pluginSource.slice(pluginSource.indexOf('const GROUP_THREAD_GAP_MS'), pluginSource.indexOf('/** Merged room view'))}; return assignLegacyThreads`
  )()
  const M = 60000
  const u = (text, at) => ({ from: { kind: 'user', name: 'You' }, text, at })
  const m = (name, text, at) => ({ from: { kind: 'member', name }, text, at })

  const log = fn([
    u('task one', 0),
    m('a', 'r1', 1 * M),
    u('quick follow-up', 3 * M), // inside the 15-min window: SAME thread
    m('a', 'r2', 4 * M),
    u('new topic much later', 60 * M), // after the lull: new thread
    m('a', 'r3', 61 * M)
  ])
  assert.equal(log[0].thread, log[2].thread, 'follow-up stays in the same thread')
  assert.equal(log[2].thread, log[3].thread)
  assert.notEqual(log[0].thread, log[4].thread, 'post-lull message starts a new thread')
  assert.equal(log[4].thread, log[5].thread)
  void gc
})

test('source contract: thread UI — folded rows, per-thread reply box, new-thread composer', () => {
  assert.match(pluginSource, /Open this thread/)
  assert.match(pluginSource, /Collapse thread/)
  assert.match(pluginSource, /Reply in thread…/)
  assert.match(pluginSource, /children: 'New Thread'/)
  assert.match(pluginSource, /const markKey = `\$\{thread\}::\$\{memberKey\}`/)
})

function groupChatWorkspaceSource() {
  const start = pluginSource.indexOf('function GroupChatWorkspace')
  const end = pluginSource.indexOf('function closeGroupChatMainTab')
  assert.ok(start >= 0, 'GroupChatWorkspace is defined')
  assert.ok(end > start, 'closeGroupChatMainTab follows GroupChatWorkspace')
  return pluginSource.slice(start, end)
}

test('source contract: group chat log lines expose CopyButton on the entry body', () => {
  const src = groupChatWorkspaceSource()
  assert.match(pluginSource, /CopyButton/)
  assert.match(src, /CopyButton/)
  assert.match(src, /text:\s*entry\.text/)
  assert.match(src, /stopPropagation:\s*true/)
  assert.match(src, /appearance:\s*['"]icon['"]/)
  assert.match(src, /buttonSize:\s*['"]icon['"]/)
  assert.match(src, /['"]group /)
  assert.doesNotMatch(src, /playSpeechText/)
  assert.doesNotMatch(src, /RefreshCw/)
  assert.doesNotMatch(src, /readAloud/)
  assert.doesNotMatch(src, /ActionBarPrimitive\.Reload/)
})
