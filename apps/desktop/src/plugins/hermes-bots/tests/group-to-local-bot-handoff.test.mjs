import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #89834 — opening a local bot while a group owns the main workspace must
// retire that group's registered main tab (and clear the selection atom)
// BEFORE any async source prep / canonical open. Remote bots stay put.
//
// Behavioral seam (same style as profile-prewarm): render real BotRow /
// ActiveNow open handlers and assert close-before-open ordering. Source
// regex is not the load-bearing proof.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function sourceBetween(start, end) {
  const from = source.indexOf(start)
  const to = source.indexOf(end, from)

  assert.notEqual(from, -1, `missing ${start}`)
  assert.notEqual(to, -1, `missing ${end}`)

  return source.slice(from, to)
}

// ACTIVE_WINDOW_S through activeBots (includes botActivitySession + workerActiveAt).
function activeBotsSource() {
  return sourceBetween('const ACTIVE_WINDOW_S', '// ── bot row ─')
}

function handoffHelpersSource() {
  return (
    sourceBetween('function closeGroupChatMainTab(group) {', '/** Local-bot open handoff:') +
    sourceBetween('function dismissGroupChatForLocalBotOpen() {', '/** Main-window wrapper:')
  )
}

/** Extract `bot => { ... }` assigned to ActiveNowStrip's onOpen in BotsPane. */
function activeNowOnOpenSource() {
  const strip = source.indexOf('jsx(ActiveNowStrip, {')
  assert.ok(strip >= 0, 'ActiveNowStrip mount must remain in BotsPane')

  const key = source.indexOf('onOpen: bot => {', strip)
  assert.ok(key > strip, 'ActiveNowStrip onOpen handler must remain inline')

  const start = key + 'onOpen: '.length
  assert.equal(source.slice(start, start + 'bot => {'.length), 'bot => {')

  let i = start + 'bot => {'.length
  let depth = 1

  while (i < source.length && depth > 0) {
    const ch = source[i]
    if (ch === '{') depth += 1
    else if (ch === '}') depth -= 1
    i += 1
  }

  assert.equal(depth, 0, 'ActiveNowStrip onOpen braces must balance')
  return source.slice(start, i)
}

function atom(initial) {
  let value = initial
  return {
    get: () => value,
    set: next => {
      value = typeof next === 'function' ? next(value) : next
    }
  }
}

function node(type, props = {}) {
  return { type, props }
}

function findButton(tree, predicate = () => true) {
  const visit = value => {
    if (value == null || typeof value !== 'object') return null
    if (Array.isArray(value)) {
      for (const child of value) {
        const found = visit(child)
        if (found) return found
      }
      return null
    }
    if (value.type === 'button' && predicate(value)) return value
    return visit(value.props?.children)
  }
  return visit(tree)
}

function loadCallers() {
  const timeline = []
  const notifications = []
  const groupChatMainTabs = new Map()
  const $groupChatWorkspace = atom(null)
  const $selectedBot = atom('default')
  const $botUnread = atom({})
  const $botMeta = atom({})
  const $focusedBotProfile = atom('default')
  const $lastRoster = atom([])

  const prepareSource = sourceBetween('async function prepareBotSource(', 'function displayName(')
  const botRowSource = sourceBetween('function BotRow(', '// ── model picker')
  const activeNowSource = sourceBetween(
    'function ActiveNowStrip({ roster, activeProfile, gatewayState, metaByName, onOpen }) {',
    '/** Assign a bot to a group-chat membership'
  )

  const context = {
    BotFace: 'BotFace',
    ContextMenu: 'ContextMenu',
    ContextMenuContent: 'ContextMenuContent',
    ContextMenuItem: 'ContextMenuItem',
    ContextMenuSeparator: 'ContextMenuSeparator',
    ContextMenuTrigger: 'ContextMenuTrigger',
    ROSTER_KEY: ['hermes-bots', 'roster'],
    groupChatMainTabs,
    $groupChatWorkspace,
    $selectedBot,
    $botUnread,
    $botMeta,
    $focusedBotProfile,
    $lastRoster,
    botAppearance: () => ({ shape: 'round', color: '#000', image: null }),
    botGroups: () => [],
    botHandle: value => value,
    botOpenGeneration: 0,
    botRosterKey: bot => bot.name,
    botRosterMeta: (bot, metaByName) => metaByName?.[bot.name] ?? null,
    cn: (...values) => values.filter(Boolean).join(' '),
    createCanonicalChat: async () => null,
    displayName: (bot, _meta) => bot?.title || bot?.name || 'bot',
    duplicateBot: async () => 'copy',
    haptic: () => undefined,
    isBackfilledFacePng: () => false,
    previewKind: () => ({ fromBot: false, sender: null }),
    generatedSessionTitle: () => null,
    openBotCanonicalChat: async (...args) => {
      timeline.push({ type: 'openBotCanonicalChat', args })
      return 'stored-chat'
    },
    ACTIVE_WINDOW_S: 90,
    A2A_PREFIX_RE: /^$/,
    useEffect: () => undefined,
    useState: initial => [typeof initial === 'function' ? initial() : initial, () => undefined],
    host: {
      state: { gateway: atom('open'), profile: atom('default') },
      ensureAgent: async () => undefined,
      activeConnectionId: () => 'local',
      warmAgent: () => undefined,
      warmProfile: () => undefined,
      request: async () => ({ profiles: [], sessions: [] }),
      notify: payload => notifications.push(payload),
      notifyError: () => undefined,
      newChat: () => timeline.push({ type: 'newChat' }),
      navigate: () => timeline.push({ type: 'navigate' })
    },
    jsx: node,
    jsxs: node,
    onEdit: () => undefined,
    queryClient: { invalidateQueries: () => undefined },
    relativeTime: () => 'now',
    saveBotMeta: () => undefined,
    showsHandle: () => false,
    stripPreviewMarkdown: text => String(text || ''),
    useValue: store => store.get(),
    allMeta: {},
    timeline,
    notifications
  }

  // Bind Map + atoms on the sandbox so helper source closes over the live bindings.
  context.groupChatMainTabs = groupChatMainTabs
  context.$groupChatWorkspace = $groupChatWorkspace

  const code = [
    activeBotsSource(),
    handoffHelpersSource(),
    prepareSource,
    botRowSource,
    activeNowSource,
    'const openActiveNow = ' + activeNowOnOpenSource() + ';',
    'globalThis.__callers = { BotRow, ActiveNowStrip, openActiveNow, dismissGroupChatForLocalBotOpen, closeGroupChatMainTab };'
  ].join('\n')

  vm.runInNewContext(code, context)

  const registerGroupMainTab = group => {
    let closed = 0
    $groupChatWorkspace.set(group)
    groupChatMainTabs.set(group, () => {
      closed += 1
      timeline.push({ type: 'mainTabClose', group })
    })
    return {
      closed: () => closed,
      isRegistered: () => groupChatMainTabs.has(group)
    }
  }

  return {
    ...context.__callers,
    $groupChatWorkspace,
    $selectedBot,
    groupChatMainTabs,
    timeline,
    notifications,
    registerGroupMainTab,
    renderBotRow(bot) {
      const tree = context.__callers.BotRow({ bot, onEdit: () => undefined })
      const row = tree.type === 'button' ? tree : tree.props.children[0].props.children
      return row
    },
    renderActiveNowChip(bot, { gatewayState = 'open', activeProfile = 'other' } = {}) {
      // Force the bot into the strip via a fresh last_session inside the window.
      const rosterBot = {
        ...bot,
        last_session: bot.last_session || {
          id: 'live',
          last_active: Math.floor(Date.now() / 1000) - 5
        }
      }
      const tree = context.__callers.ActiveNowStrip({
        roster: [rosterBot],
        activeProfile,
        gatewayState,
        metaByName: {},
        onOpen: context.__callers.openActiveNow
      })
      const chip = findButton(tree, button => String(button.props?.title || '').includes(bot.name) ||
        String(button.props?.title || '').includes(bot.title || ''))
      assert.ok(chip, 'Active Now chip must render for the seeded bot')
      return chip
    }
  }
}

test('BotRow local open closes the group main tab before canonical open', async () => {
  const runtime = loadCallers()
  const tab = runtime.registerGroupMainTab('Core')
  const row = runtime.renderBotRow({ name: 'alpha', title: 'Alpha' })

  await row.props.onClick()

  assert.equal(tab.closed(), 1, 'registered main-tab closer must run')
  assert.equal(runtime.$groupChatWorkspace.get(), null, 'group selection atom must clear')
  assert.equal(tab.isRegistered(), false, 'closer entry must be retired')
  assert.ok(
    runtime.timeline.some(entry => entry.type === 'openBotCanonicalChat'),
    'local open must proceed to canonical chat'
  )

  const closeAt = runtime.timeline.findIndex(entry => entry.type === 'mainTabClose')
  const openAt = runtime.timeline.findIndex(entry => entry.type === 'openBotCanonicalChat')
  assert.ok(closeAt >= 0 && openAt >= 0 && closeAt < openAt, 'close must precede canonical open')
})

test('BotRow remote open leaves the group main tab open', async () => {
  const runtime = loadCallers()
  const tab = runtime.registerGroupMainTab('Core')
  const row = runtime.renderBotRow({
    name: 'research',
    title: 'Research',
    remoteSource: true,
    sourceScoped: true,
    connectionId: 'work',
    connectionLabel: 'Work'
  })

  await row.props.onClick()

  assert.equal(tab.closed(), 0, 'remote open must not dismiss the group tab')
  assert.equal(runtime.$groupChatWorkspace.get(), 'Core')
  assert.equal(tab.isRegistered(), true)
  assert.equal(
    runtime.timeline.some(entry => entry.type === 'openBotCanonicalChat'),
    false,
    'remote open must not hop into a canonical chat'
  )
  assert.ok(runtime.notifications.some(n => /Stay in this chat/.test(n.message || '')))
})

test('Active Now local open closes the group main tab before canonical open', async () => {
  const runtime = loadCallers()
  const tab = runtime.registerGroupMainTab('Ops')
  const chip = runtime.renderActiveNowChip({ name: 'alpha', title: 'Alpha' })

  await chip.props.onClick()
  // Active Now fires canonical open inside an async IIFE — drain microtasks.
  await new Promise(resolve => setImmediate(resolve))
  await new Promise(resolve => setImmediate(resolve))

  assert.equal(tab.closed(), 1, 'registered main-tab closer must run')
  assert.equal(runtime.$groupChatWorkspace.get(), null)
  assert.equal(tab.isRegistered(), false)
  assert.ok(runtime.timeline.some(entry => entry.type === 'openBotCanonicalChat'))

  const closeAt = runtime.timeline.findIndex(entry => entry.type === 'mainTabClose')
  const openAt = runtime.timeline.findIndex(entry => entry.type === 'openBotCanonicalChat')
  assert.ok(closeAt >= 0 && openAt >= 0 && closeAt < openAt, 'close must precede canonical open')
})

test('Active Now remote open leaves the group main tab open', async () => {
  const runtime = loadCallers()
  const tab = runtime.registerGroupMainTab('Ops')
  const chip = runtime.renderActiveNowChip({
    name: 'research',
    title: 'Research',
    remoteSource: true,
    sourceScoped: true,
    connectionId: 'work',
    connectionLabel: 'Work'
  })

  await chip.props.onClick()
  await new Promise(resolve => setImmediate(resolve))

  assert.equal(tab.closed(), 0)
  assert.equal(runtime.$groupChatWorkspace.get(), 'Ops')
  assert.equal(tab.isRegistered(), true)
  assert.equal(
    runtime.timeline.some(entry => entry.type === 'openBotCanonicalChat'),
    false
  )
  assert.ok(runtime.notifications.some(n => /Stay in this chat/.test(n.message || '')))
})

test('local BotRow open with no main-tab registration still clears in-panel selection', async () => {
  const runtime = loadCallers()
  runtime.$groupChatWorkspace.set('Legacy')
  assert.equal(runtime.groupChatMainTabs.has('Legacy'), false)

  const row = runtime.renderBotRow({ name: 'alpha', title: 'Alpha' })
  await row.props.onClick()

  assert.equal(runtime.$groupChatWorkspace.get(), null)
  assert.ok(runtime.timeline.some(entry => entry.type === 'openBotCanonicalChat'))
})

test('local BotRow open with no group selected is safe and still opens chat', async () => {
  const runtime = loadCallers()
  assert.equal(runtime.$groupChatWorkspace.get(), null)

  const row = runtime.renderBotRow({ name: 'alpha', title: 'Alpha' })
  await assert.doesNotReject(async () => row.props.onClick())
  assert.equal(runtime.$groupChatWorkspace.get(), null)
  assert.ok(runtime.timeline.some(entry => entry.type === 'openBotCanonicalChat'))
})
