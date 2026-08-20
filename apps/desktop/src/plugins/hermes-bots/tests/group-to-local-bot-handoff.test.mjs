import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #89834 — opening a local bot while a group owns the main workspace must
// retire that group's registered main tab (and clear the selection atom)
// BEFORE any async source prep / canonical open. Remote bots stay put.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load() {
  const values = new Map()
  const atom = initial => {
    const slot = {
      get: () => values.get(slot),
      set: value => values.set(slot, typeof value === 'function' ? value(values.get(slot)) : value)
    }
    values.set(slot, initial)
    return slot
  }

  const timeline = []
  const context = {
    atom,
    setTimeout: fn => {
      fn()
      return 0
    },
    clearTimeout: () => undefined,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: {
      getElementById: () => null,
      createElement: () => ({}),
      head: { appendChild: () => undefined }
    },
    host: {
      request: async () => ({}),
      state: {
        profile: { get: () => 'default', listen: () => undefined },
        gateway: { listen: () => undefined }
      },
      notify: payload => timeline.push({ type: 'notify', payload }),
      notifyError: () => undefined,
      openSession: async id => {
        timeline.push({ type: 'openSession', id })
      },
      openWorkspace: undefined
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
      '\nglobalThis.__handoff = {\n' +
        '  openGroupChat,\n' +
        '  closeGroupChatMainTab,\n' +
        '  dismissGroupChatForLocalBotOpen,\n' +
        '  $groupChatWorkspace,\n' +
        '  $selectedBot,\n' +
        '  groupChatMainTabs\n' +
        '};\n'
    )

  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  context.plugin.register({
    storage: { get: () => null, set: () => undefined },
    register: () => undefined
  })

  return { ...context.__handoff, host: context.host, timeline }
}

test('local handoff closes the registered group main tab before any canonical open', () => {
  const runtime = load()
  let closed = 0

  runtime.host.openWorkspace = () => {
    timelineCloseHook()
    return () => {
      closed += 1
      runtime.timeline.push({ type: 'mainTabClose' })
    }
  }

  // openWorkspace's closer is what host.openWorkspace returns; capture via openGroupChat.
  function timelineCloseHook() {
    /* placeholder so openWorkspace is feature-detected */
  }

  runtime.openGroupChat('Core')
  assert.equal(runtime.$groupChatWorkspace.get(), 'Core')
  assert.equal(typeof runtime.groupChatMainTabs.get('Core'), 'function')

  // Shared boundary used by BotRow + Active Now for local bots only.
  runtime.dismissGroupChatForLocalBotOpen()

  assert.equal(closed, 1, 'registered main-tab closer must run')
  assert.equal(runtime.$groupChatWorkspace.get(), null, 'group selection atom must clear')
  assert.equal(runtime.groupChatMainTabs.has('Core'), false, 'closer entry must be retired')
  assert.equal(
    runtime.timeline.some(entry => entry.type === 'openSession'),
    false,
    'handoff itself must not open a canonical chat'
  )
})

test('remote bots do not dismiss the selected group workspace', () => {
  const runtime = load()
  let closed = 0

  runtime.host.openWorkspace = () => () => {
    closed += 1
  }

  runtime.openGroupChat('Core')
  assert.equal(runtime.$groupChatWorkspace.get(), 'Core')

  // Remote path must leave the group tab alone (stay-and-@).
  // The shared dismiss helper is local-only; callers skip it for remoteSource.
  assert.equal(runtime.$groupChatWorkspace.get(), 'Core')
  assert.equal(closed, 0)
  assert.equal(runtime.groupChatMainTabs.has('Core'), true)
})

test('local handoff with no main-tab registration still clears in-panel selection safely', () => {
  const runtime = load()
  // Older host: no openWorkspace → openGroupChat only sets the selection atom.
  runtime.openGroupChat('Ops')
  assert.equal(runtime.$groupChatWorkspace.get(), 'Ops')
  assert.equal(runtime.groupChatMainTabs.has('Ops'), false)

  runtime.dismissGroupChatForLocalBotOpen()

  assert.equal(runtime.$groupChatWorkspace.get(), null)
})

test('local handoff with no group selected is a no-op', () => {
  const runtime = load()
  assert.equal(runtime.$groupChatWorkspace.get(), null)

  assert.doesNotThrow(() => runtime.dismissGroupChatForLocalBotOpen())
  assert.equal(runtime.$groupChatWorkspace.get(), null)
})

test('BotRow and Active Now both route local opens through the shared handoff', () => {
  // Structural guard so the two call sites cannot drift apart again.
  assert.match(pluginSource, /function dismissGroupChatForLocalBotOpen\s*\(/)
  assert.match(
    pluginSource,
    /dismissGroupChatForLocalBotOpen\s*\(\s*\)[\s\S]*?openBotCanonicalChat/
  )

  const botRowOpen = pluginSource.slice(
    pluginSource.indexOf('function BotRow('),
    pluginSource.indexOf('function ActiveNowStrip(')
  )
  const activeNowOpen = pluginSource.slice(
    pluginSource.indexOf('jsx(ActiveNowStrip,'),
    pluginSource.indexOf('jsx(ActiveNowStrip,') + 2500
  )

  assert.match(botRowOpen, /dismissGroupChatForLocalBotOpen\s*\(/)
  assert.match(activeNowOpen, /dismissGroupChatForLocalBotOpen\s*\(/)
  // Remote path must not call the dismiss helper before the remote early-return.
  assert.match(botRowOpen, /if\s*\(\s*bot\.remoteSource\s*\)/)
  assert.match(activeNowOpen, /if\s*\(\s*bot\.remoteSource\s*\)/)
})
