import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Locks the plugin-i18n contract (#67303) for hermes-bots:
//
//  1. The bundle MAY contain function-valued leaves (`PluginMessageValue =
//     string | ((...args) => string)`) — the SDK's translateFrom invokes them
//     with positional args, exactly like core. bindBotsText therefore forwards
//     call-site args through t(path, ...args) — NOT named placeholders.
//  2. A host WITH the feature composes parameterized strings byte-identically
//     to the en template (the scenario a graceful-degradation fallback can't
//     protect — it never runs there).
//  3. useBots() memoizes the bound wrapper tree on the translator identity
//     (mirrors kanban's useKanban: useMemo(() => bind(t, en), [t])).
//  4. Hosts WITHOUT the feature (or the SDK-stripped sandbox) get BOTS_EN
//     verbatim — byte-identical to the pre-bundle literals.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

/** Resolution policy copied from the SDK (apps/desktop/src/i18n/runtime.ts
 *  translateFrom): descend the dot-path, call function leaves with the args. */
function registryT(bundles, locale = 'en') {
  const messages = bundles[locale] || bundles.en
  return (key, ...args) => {
    let value = messages
    for (const part of String(key).split('.')) {
      value = value?.[part]
      if (value === undefined) return key
    }
    if (typeof value === 'function') return value(...args)
    return typeof value === 'string' ? value : key
  }
}

/** Tiny correct-enough useMemo stub: caches on element-wise dep identity and
 *  counts real rebuilds so memoization is observable. */
function makeUseMemo() {
  let cached
  let cachedDeps
  let runs = 0
  const useMemo = (fn, deps) => {
    const same =
      cachedDeps && cachedDeps.length === deps.length && deps.every((dep, i) => dep === cachedDeps[i])
    if (!same) {
      cached = fn()
      cachedDeps = deps
      runs += 1
    }
    return cached
  }
  useMemo.runs = () => runs
  return useMemo
}

function load({ withI18n = false } = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const useMemo = makeUseMemo()
  let currentT = null
  const context = {
    atom,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware', atCompletions: 'at-completions' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      state: {
        profile: { get: () => 'default', listen: () => undefined },
        gateway: { get: () => 'open', listen: () => undefined }
      },
      request: () => Promise.resolve({}),
      notify: () => undefined
    },
    // Feature-detectable SDK: usePluginI18n only exists when withI18n is set.
    sdk: new Proxy(
      {},
      {
        get: (_target, prop) =>
          prop === 'usePluginI18n' && withI18n ? () => currentT : undefined
      }
    ),
    useMemo
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(`
globalThis.__i18n = { useBots, bindBotsText, BOTS_EN, BOTS_LOCALES };
`)
  vm.runInNewContext(source, context, { filename: 'plugin.js' })

  return {
    ...context.__i18n,
    plugin: context.plugin,
    useMemo,
    setT: next => {
      currentT = next
    }
  }
}

test('i18n contract: bindBotsText forwards positional args through t(path, ...args)', () => {
  const env = load({ withI18n: true })
  const calls = []
  env.setT((path, ...args) => {
    calls.push([path, ...args])
    return `«${path}»${args.length ? `(${args.map(String).join('|')})` : ''}`
  })

  const k = env.useBots()

  assert.equal(k.pane.pinToggle('Fred', false), '«pane.pinToggle»(Fred|false)')
  assert.equal(k.pane.showHiddenBots(3), '«pane.showHiddenBots»(3)')
  assert.equal(k.create.takenRemote('x', 'target-device'), '«create.takenRemote»(x|target-device)')
  assert.equal(k.edit.sectionsFailed('a, b'), '«edit.sectionsFailed»(a, b)')
  assert.equal(k.pane.title, '«pane.title»')

  // Function leaves forward their args at call time (flat leaves resolve at
  // bind time, so find the parameterized calls rather than assuming order).
  assert.deepEqual(
    calls.find(call => call[0] === 'pane.pinToggle'),
    ['pane.pinToggle', 'Fred', false]
  )
  assert.deepEqual(
    calls.find(call => call[0] === 'pane.showHiddenBots'),
    ['pane.showHiddenBots', 3]
  )
  assert.deepEqual(
    calls.find(call => call[0] === 'create.takenRemote'),
    ['create.takenRemote', 'x', 'target-device']
  )
})

test('i18n contract: a host WITH the feature composes parameterized strings byte-identically', () => {
  const env = load({ withI18n: true })
  // The real registered bundle (en) + the SDK's resolution policy: function
  // leaves get called with the positional args the client passed.
  env.setT(registryT(env.BOTS_LOCALES, 'en'))

  const k = env.useBots()

  // Parameterized leaves — the exact surfaces the review flagged.
  assert.equal(k.pane.pinToggle('Fred', false), 'Fred pinned to top')
  assert.equal(k.pane.pinToggle('Fred', true), 'Fred unpinned')
  assert.equal(k.pane.showHiddenBots(1), 'Show 1 hidden bot')
  assert.equal(k.pane.showHiddenBots(2), 'Show 2 hidden bots')
  assert.equal(
    k.create.draftCleanupFailed('draft-1'),
    'Could not clean up draft profile "draft-1"'
  )
  assert.equal(
    k.pane.rosterUnavailable('boom'),
    'Roster unavailable: boom. If your gateway predates profiles.list, update Hermes and restart the gateway.'
  )
  assert.equal(k.menu.hiddenNotice('Fred'), 'Fred hidden — use the eye button in the Bots header to see hidden bots')

  // Flat leaves resolve through the same translator.
  assert.equal(k.menu.pin, 'Pin to top')
  assert.equal(k.menu.unpin, 'Unpin')
  assert.equal(k.common.deleted, 'Deleted')
  assert.equal(k.pane.searchBotsPlaceholder, 'Search bots…')
  assert.equal(k.pane.emptyStateDescription, 'Create your first teammate.')
})

test('i18n contract: hosts without the feature get the en template verbatim', () => {
  const env = load() // sdk.usePluginI18n absent → graceful degradation

  const k = env.useBots()

  assert.equal(k, env.BOTS_EN) // the raw template, not a bound copy
  assert.equal(k.pane.title, 'Bots')
  assert.equal(k.pane.pinToggle('Fred', false), 'Fred pinned to top')
  assert.equal(k.pane.showHiddenBots(2), 'Show 2 hidden bots')
  assert.equal(k.menu.pin, 'Pin to top')
  assert.equal(k.menu.unpin, 'Unpin')
  assert.equal(k.common.deleted, 'Deleted')
  assert.equal(k.pane.emptyStateDescription, 'Create your first teammate.')
  assert.equal(k.pane.searchBotsPlaceholder, 'Search bots…')
  assert.equal(env.plugin.name, 'Bots')
  assert.equal(env.plugin.description, 'Bot Mode — a one-chat-per-agent roster with avatars, routines, group chats, and bot-to-bot messaging. Ships with the app; disable here if unwanted.')
})

test('i18n contract: useBots memoizes the bound tree on the translator identity', () => {
  const env = load({ withI18n: true })
  env.setT(registryT(env.BOTS_LOCALES, 'en'))

  const first = env.useBots()
  const again = env.useBots()
  assert.equal(first, again, 'same translator → same bound tree (no rebuild)')
  assert.equal(env.useMemo.runs(), 1)

  // Locale switch / re-registration hands usePluginI18n a NEW t — rebuild once.
  env.setT(registryT(env.BOTS_LOCALES, 'en'))
  const rebuilt = env.useBots()
  assert.notEqual(rebuilt, first)
  assert.equal(env.useMemo.runs(), 2)

  const stable = env.useBots()
  assert.equal(stable, rebuilt, 'same new translator → stable again')
  assert.equal(env.useMemo.runs(), 2)
})
