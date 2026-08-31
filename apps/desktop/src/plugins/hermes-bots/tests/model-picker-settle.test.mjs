import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #95279 regression: the Bot Mode model picker's catalog read must always
// SETTLE and must not churn the network on every surface remount.
//
// The picker's query rides the BOT's own socket (a second, lazily dialed pool
// backend — exactly the Bots-only context of #95279). Two defects there left
// the picker unusable while Bots was active:
//
//   1. The RPC promise had NO deadline. A wedged dial left it pending forever
//      → the spinner never settled and the picker was dead ("model picker
//      never settles").
//   2. Every fetch forced `refresh: true`, bypassing react-query's cache, so
//      each Bots view remount (tab re-front, dialog reopen, pane visibility
//      flip) knocked the picker back into its loading state and discarded the
//      staged selection mid-edit.
//
// These tests pin the contract: bounded settlement (reject → free-text
// fallback) and a cached, unforced catalog read.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadPickerHarness({ requestProfile } = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = {
      get: () => values.get(slot),
      set: value => values.set(slot, value),
      listen: () => undefined
    }
    values.set(slot, initial)
    return slot
  }
  const jsx = (type, props = {}) => ({ type, props })
  const routed = []
  let query
  const stateValues = [true, false, '']
  const context = {
    atom,
    Checkbox: 'Checkbox',
    GlyphSpinner: 'GlyphSpinner',
    Input: 'Input',
    ScrollArea: 'ScrollArea',
    Textarea: 'Textarea',
    document: { createElement: () => ({}), getElementById: () => null, head: { appendChild: () => undefined } },
    host: {
      getGateway: () => 'ambient-gateway',
      requestProfile: async (...args) => {
        routed.push(args)

        return requestProfile ? requestProfile(...args) : new Promise(() => {})
      },
      request: async (...args) => {
        routed.push(['ambient', ...args])

        return requestProfile ? requestProfile(...args) : new Promise(() => {})
      },
      state: {
        gateway: { get: () => 'open', listen: () => undefined },
        profile: { get: () => 'default', listen: () => undefined }
      }
    },
    jsx,
    jsxs: jsx,
    queryClient: { invalidateQueries: () => undefined },
    sdk: {},
    useQuery: options => {
      query = options

      return { data: undefined, isLoading: false, error: null }
    },
    useState: initial => [stateValues.length ? stateValues.shift() : initial, () => undefined],
    window: { setTimeout, clearTimeout }
  }
  const code = source
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__useModelOptions = useModelOptions;')
  vm.runInNewContext(code, context, { filename: 'plugin.js' })

  return { context, query: () => query, routed }
}

const remoteBot = {
  connectionId: 'remote-a',
  name: 'default',
  remoteSource: true,
  route: {
    connectionId: 'remote-a',
    mode: 'remote',
    profile: 'default',
    targetProfile: 'backend-default'
  },
  sourceScoped: true,
  targetProfile: 'backend-default'
}

test('#95279: the picker catalog query settles when the bot gateway never answers', async t => {
  // Fake clocks: tick past the settle budget without waiting real time. The
  // harness captured the mocked setTimeout above (enable runs first).
  t.mock.timers.enable({ apis: ['setTimeout'] })

  const runtime = loadPickerHarness({ requestProfile: () => new Promise(() => {}) })
  runtime.context.__useModelOptions(remoteBot)
  const query = runtime.query()

  assert.equal(typeof query.queryFn, 'function')

  let settled = false

  query.queryFn().then(
    () => {
      settled = 'resolved'
    },
    () => {
      settled = 'rejected'
    }
  )

  // Past the settle budget the bounded fetch must have rejected (the picker
  // then renders its free-text fallback). A macrotask boundary drains every
  // microtask hop of the rejection chain before we look.
  t.mock.timers.tick(30_000)
  await new Promise(resolve => setImmediate(resolve))

  assert.equal(settled, 'rejected')
})

test('#95279: the picker catalog read is cached, not force-refreshed on every mount', async () => {
  const runtime = loadPickerHarness({
    requestProfile: async () => ({ providers: [{ models: ['m1'], name: 'Prov', slug: 'prov' }] })
  })
  runtime.context.__useModelOptions(remoteBot)
  const query = runtime.query()

  // Same bot + route ⇒ one stable cache key, so Bots view remounts reuse the
  // fetched catalog instead of knocking the picker back into its spinner.
  assert.deepEqual(query.queryKey[1], 'model-options')
  assert.deepEqual(query.queryKey[2], 'remote-a::default')

  const data = await query.queryFn()

  assert.deepEqual(JSON.parse(JSON.stringify(data)), {
    providers: [{ models: ['m1'], name: 'Prov', slug: 'prov' }]
  })

  // No forced refresh: the read participates in the staleTime cache. A forced
  // refresh here is what made every remount re-enter the loading state and
  // wipe the user's in-progress pick (#95279).
  assert.deepEqual(JSON.parse(JSON.stringify(runtime.routed)), [
    [remoteBot.route, 'model.options', { include_unconfigured: true, explicit_only: false }]
  ])
})

test('#95279: picker query stays disabled for orphaned rows and keyed active for local bots', () => {
  const runtime = loadPickerHarness()

  // An orphaned row must not dispatch at all (renders free-text fallback).
  runtime.context.__useModelOptions({ name: 'ghost', remoteSource: true })
  const orphanQuery = runtime.query()
  assert.equal(orphanQuery.enabled, false)

  // A local (non source-scoped) bot reads the ambient gateway under the
  // stable 'active' key.
  runtime.context.__useModelOptions({ name: 'default' })
  const localQuery = runtime.query()
  assert.equal(localQuery.enabled, true)
  assert.deepEqual(localQuery.queryKey[2], 'active')
})
