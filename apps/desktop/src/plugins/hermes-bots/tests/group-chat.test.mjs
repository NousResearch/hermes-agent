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
  const transcripts = new Map()
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
          return { session_id: `rt-${params.profile}`, stored_session_id: `sid-${params.profile}`, message_count: 0, messages: [] }
        }
        if (method === 'session.resume') {
          const profile = params.profile
          const transcript = transcripts.get(profile) || []
          return { session_id: `rt-${profile}`, messages: transcript, inflight: false, running: false }
        }
        if (method === 'prompt.submit') {
          const profile = String(params.session_id).replace(/^rt-/, '')
          const transcript = transcripts.get(profile) || []
          transcript.push({ role: 'user', content: params.text })
          calls.push({ profile, prompt: params.text })
          const reply = turnScript(profile, params.text, calls.length)
          transcript.push({ role: 'assistant', content: reply })
          transcripts.set(profile, transcript)
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
  return { ...context.__gc, calls, storageWrites, transcripts }
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

test('disband: clears grouping meta, room log, workspace, needs-you; keeps sessions in storage map only for other rooms', async () => {
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

  gc.$botMeta.set({ builder: { group: 'Gone' }, research: { group: 'Keep' } })
  gc.$groupChatWorkspace.set('Gone')
  gc.$groupNeedsYou.set({ Gone: true, Keep: true })

  await gc.disbandGroupChat('Gone', ['builder'])

  // Room state: gone from the atom (no running drive, so no tombstone).
  assert.equal(gc.$groupChats.get().Gone, undefined)
  assert.ok(gc.$groupChats.get().Keep, 'other rooms untouched')
  // The open room view closed; needs-you cleared for the disbanded room only.
  assert.equal(gc.$groupChatWorkspace.get(), null)
  assert.equal(gc.$groupNeedsYou.get().Gone, undefined)
  assert.equal(gc.$groupNeedsYou.get().Keep, true)
  // Members ungrouped; other bots keep their group.
  assert.equal(gc.$botMeta.get().builder.group, null)
  assert.equal(gc.$botMeta.get().research.group, 'Keep')
  // Persisted room map no longer carries the room.
  const durable = gc.storageWrites.get('group-chats')
  assert.ok(durable && !('Gone' in durable), 'disbanded room not persisted')
  assert.ok('Keep' in durable, 'surviving room still persisted')
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

  await gc.disbandGroupChat('Live', ['research'])

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
  gc.transcripts.set('research', [
    { role: 'user', content: 'the turn prompt' },
    { role: 'assistant', content: 'Here is the full research result, delivered late.' }
  ])

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
  gc.transcripts.set('builder', [
    { role: 'user', content: 'prompt' },
    { role: 'assistant', content: '(pass)' }
  ])

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
