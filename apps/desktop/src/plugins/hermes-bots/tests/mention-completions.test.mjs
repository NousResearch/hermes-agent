import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// The mention-completions source (#88060): typing @ offers roster handles.
// Extract the registered provide() by running plugin registration with a
// stubbed ctx/queryClient and exercising the captured contribution.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadProvide({ roster, active = 'default', focused = active, meta = {} } = {}) {
  const registrations = []
  const atom = value => {
    let current = value

    return { get: () => current, set: next => (current = next), listen: () => () => undefined }
  }
  const jsx = (type, props = {}) => ({ type, props })
  const context = {
    atom,
    jsx,
    jsxs: jsx,
    useQuery: () => ({}),
    useValue: v => (v?.get ? v.get() : v),
    useState: v => [v, () => undefined],
    setTimeout,
    clearTimeout,
    performance,
    window: {
      setTimeout,
      clearTimeout,
      performance,
      requestAnimationFrame: () => 0,
      cancelAnimationFrame: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined
    },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    queryClient: {
      // Key-exact semantics like the real QueryClient (v5): a 2-element
      // ROSTER_KEY lookup MUST miss the 3-element cache keys useRoster
      // writes — the bug behind issue #89303 was invisible under the old
      // key-ignoring mock.
      getQueryData: key =>
        key.length === 3 && key[0] === 'hermes-bots' && key[1] === 'roster'
          ? roster
          : undefined,
      getQueriesData: () => (Array.isArray(roster?.profiles) ? [[['hermes-bots', 'roster', 'local'], roster]] : []),
      invalidateQueries: () => undefined,
      setQueryData: () => undefined
    },
    host: {
      state: {
        profile: { get: () => active, listen: () => () => undefined },
        focusedSessionProfile: { get: () => focused, listen: () => () => undefined },
        gateway: { get: () => null, listen: () => () => undefined },
        connectionId: { get: () => 'local', listen: () => () => undefined }
      },
      request: async () => ({}),
      onEvent: () => () => undefined
    },
    COMPOSER_AREAS: { middleware: 'composer.middleware', atCompletions: 'composer.atCompletions' },
    sdk: new Proxy({}, { get: () => undefined })
  }
  const code = source
    .replace(/^import \* as sdk from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__botMeta = $botMeta;')
  vm.runInNewContext(code, context)
  context.__botMeta.set(meta)

  const ctx = {
    register: c => registrations.push(c),
    storage: { get: async () => undefined, set: async () => undefined }
  }

  try {
    context.plugin.register(ctx)
  } catch {
    /* registration walks UI surfaces the stub doesn't fully model — the
       contributions registered before any throw are what we assert on */
  }

  const contribution = registrations.find(c => c.area === 'composer.atCompletions')
  assert.ok(contribution, 'mention-completions contribution registered')

  return contribution.data.provide
}

const ROSTER = {
  profiles: [
    { name: 'default' },
    { name: 'researcher' },
    { name: 'writer', connectionLabel: 'Homelab', handle: 'writer-homelab' }
  ]
}

test('offers roster handles for a matching prefix, excluding the active profile', () => {
  const provide = loadProvide({ roster: ROSTER, active: 'researcher' })
  const all = provide('')
  const inserts = all.map(i => i.insert)

  assert.ok(inserts.includes('@hermes'), 'default surfaces as @hermes')
  assert.ok(inserts.includes('@writer-homelab'), 'multi-source handle used')
  assert.ok(!inserts.includes('@researcher'), 'active profile excluded')
})

test('prefix-filters against the handle', () => {
  const provide = loadProvide({ roster: ROSTER, active: 'default' })

  assert.deepStrictEqual([...provide('wri').map(i => i.insert)], ['@writer-homelab'])
  assert.strictEqual(provide('zzz').length, 0)
})

test('filters self by the focused Bot Chat owner, not the gateway socket profile', () => {
  const provide = loadProvide({
    active: 'default',
    focused: 'renametest',
    roster: {
      profiles: [
        { name: 'default', display_name: 'Lucy' },
        { name: 'renametest' }
      ]
    }
  })

  const inserts = provide('').map(item => item.insert)
  assert.ok(
    inserts.includes('@lucy'),
    'renamed default profile remains mentionable from the focused bot chat'
  )
  assert.ok(
    !inserts.includes('@renametest'),
    'focused bot excludes itself even when the socket stays on default'
  )
})

test('empty roster cache yields no rows (never throws)', () => {
  const provide = loadProvide({ roster: undefined })

  assert.strictEqual(provide('a').length, 0)
})
