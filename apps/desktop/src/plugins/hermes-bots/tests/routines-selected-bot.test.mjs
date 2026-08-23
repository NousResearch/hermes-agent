import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #89625 follow-up (PR #89637 residual): nanostores' `.listen()` never replays
// the current value the way `.subscribe()` does, so the $focusedBotProfile
// listener in register() only kept $selectedBot current from the moment it was
// attached. A disable -> profile switch -> re-enable cycle (Settings > Plugins)
// left $selectedBot pointed at whichever bot was active before the plugin was
// disabled. bindProfileSync() reseeds $selectedBot from the store's CURRENT
// value before attaching the listener, so every register() starts in sync.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load({ focusedSessionOwnerSupported = false, focusedSessionOwner = null } = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const state = { profile: { listen: () => undefined } }
  if (focusedSessionOwnerSupported) {
    state.focusedSessionOwner = atom(focusedSessionOwner)
  }
  const context = {
    atom,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: { state }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(`
      globalThis.__api = {
        bindProfileSync,
        resolveRoutineOwner: typeof resolveRoutineOwner === 'function' ? resolveRoutineOwner : undefined,
        $selectedBot
      };
    `)
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context
}

// Mimic real nanostores atom semantics (get + non-replaying listen) — not a
// mock that assumes the fix — so this fails honestly if bindProfileSync
// regresses to a plain, non-reseeding listen().
function fakeProfileStore(initial) {
  let value = initial
  const listeners = new Set()
  return {
    get: () => value,
    set: next => {
      value = next
      for (const listener of listeners) listener(value)
    },
    listen: listener => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    }
  }
}

test('regression: re-enabling after a profile switch while disabled resyncs, not stays stale', () => {
  const { bindProfileSync, $selectedBot } = load().__api
  const profile = fakeProfileStore('blog-writer')

  const unbind1 = bindProfileSync(profile)
  assert.equal($selectedBot.get(), 'blog-writer', 'reseeded to the live profile on first bind')

  unbind1() // plugin disabled: listener torn down
  profile.set('researcher') // profile changes while nothing is listening

  bindProfileSync(profile) // plugin re-enabled: register() runs again
  assert.equal($selectedBot.get(), 'researcher', 'reseeded to the live profile on re-bind, not left stale')
})

test('unit: bindProfileSync keeps forwarding live profile changes after (re)binding', () => {
  const { bindProfileSync, $selectedBot } = load().__api
  const profile = fakeProfileStore('default')

  bindProfileSync(profile)
  profile.set('researcher')
  assert.equal($selectedBot.get(), 'researcher', 'live changes still flow through the listener')
})

test('unit: bindProfileSync follows connection changes for the same profile name', () => {
  const { bindProfileSync, $selectedBot } = load().__api
  const owner = fakeProfileStore({ connectionId: 'source-a', profile: 'default' })

  bindProfileSync(owner)
  assert.equal($selectedBot.get(), 'source-a::default')

  owner.set({ connectionId: 'source-b', profile: 'default' })
  assert.equal($selectedBot.get(), 'source-b::default')
})

test('unit: an empty/non-string current value is not seeded into $selectedBot', () => {
  const { bindProfileSync, $selectedBot } = load().__api
  const before = $selectedBot.get()
  bindProfileSync(fakeProfileStore(''))
  assert.equal($selectedBot.get(), before, 'falsy current value leaves $selectedBot untouched')
})

test('unit: an authoritative focused owner missing from the roster fails closed', () => {
  const { resolveRoutineOwner } = load().__api
  const roster = [{
    connectionId: 'source-a',
    name: 'default',
    remoteSource: true,
    sourceScoped: true
  }]
  const focusedOwner = {
    authoritative: true,
    connectionId: 'source-b',
    name: 'worker'
  }

  assert.equal(resolveRoutineOwner(roster, focusedOwner, 'source-a::default'), null)
})

test('unit: an authoritative null focused owner disables routines instead of using selection', () => {
  const { resolveRoutineOwner } = load({ focusedSessionOwnerSupported: true }).__api
  const roster = [{
    connectionId: 'source-a',
    name: 'default',
    remoteSource: true,
    sourceScoped: true
  }]

  assert.equal(resolveRoutineOwner(roster, null, 'source-a::default'), null)
})

test('unit: a legacy SDK without focused owner support may use selection', () => {
  const { resolveRoutineOwner } = load().__api
  const roster = [{
    connectionId: 'source-a',
    name: 'default',
    remoteSource: true,
    sourceScoped: true
  }]

  assert.equal(resolveRoutineOwner(roster, null, 'source-a::default'), roster[0])
})

test('unit: register() wires the focused-owner ladder through bindProfileSync', () => {
  assert.match(
    pluginSource,
    /const unbindProfileListener = bindProfileSync\(\$focusedBotOwner\)/,
    'the reseeding helper must own the connection-qualified focused owner subscription'
  )
})
