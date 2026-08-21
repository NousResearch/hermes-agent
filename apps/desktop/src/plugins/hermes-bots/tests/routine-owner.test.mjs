import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')
const plain = value => JSON.parse(JSON.stringify(value))

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
  const invalidations = []
  const context = {
    atom,
    host: {
      state: {
        profile: { get: () => 'ops', listen: () => undefined },
        gateway: { get: () => 'open', listen: () => undefined }
      },
      request: async () => ({ jobs: [] }),
      notify: () => undefined,
      notifyError: () => undefined
    },
    queryClient: {
      invalidateQueries: async options => invalidations.push(options)
    },
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__routineOwners = { routineCreateTarget, invalidateRoutineOwner };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return { ...context, invalidations }
}

test('routine creation keeps its captured owner while another bot becomes active', () => {
  const runtime = load()
  assert.equal(runtime.__routineOwners.routineCreateTarget('ops', 'ops'), 'ops')
  assert.equal(runtime.__routineOwners.routineCreateTarget('ops', 'default'), 'ops')
  assert.equal(runtime.__routineOwners.routineCreateTarget(null, 'default'), 'default')
})

test('routine mutation invalidates only its immutable owner cache', async () => {
  const runtime = load()
  await runtime.__routineOwners.invalidateRoutineOwner('ops')
  assert.deepEqual(plain(runtime.invalidations), [{
    queryKey: ['hermes-bots', 'routines', 'ops'],
    exact: true
  }])
})

test('source contract: row mutations invalidate the profile that owned the request', () => {
  assert.match(
    pluginSource,
    /function RoutineRow\(\{ job, profile \}\)[\s\S]*await host\.request\('cron\.manage', \{ action, name: job\.job_id, \.\.\.\(profile \? \{ profile \} : \{\}\) \}\)[\s\S]*await invalidateRoutineOwner\(profile\)/
  )
  assert.doesNotMatch(pluginSource, /function RoutineRow\(\{ job, onChanged, profile \}\)/)
})

test('source contract: create mutations and dialog state retain one owner', () => {
  assert.match(
    pluginSource,
    /function CreateRoutineDialog\(\{ bot, open, onClose \}\)[\s\S]*await host\.request\('cron\.manage', \{[\s\S]*\.\.\.\(bot \? \{ profile: bot \} : \{\}\)[\s\S]*await invalidateRoutineOwner\(bot\)/
  )
  assert.match(pluginSource, /const \[createOwner, setCreateOwner\] = useState\(null\)/)
  assert.match(pluginSource, /const openCreate = \(\) => \{[\s\S]*setCreateOwner\(bot\)[\s\S]*setCreateOpen\(true\)/)
  assert.match(pluginSource, /const createTarget = routineCreateTarget\(createOwner, bot\)/)
  // key must be the jsx() THIRD argument (a `key:` prop is silently ignored
  // by the react/jsx-runtime and the dialog would keep stale per-bot state).
  assert.match(pluginSource, /jsx\(CreateRoutineDialog, \{[\s\S]*?\}, createTarget\)/)
  assert.doesNotMatch(pluginSource, /key: createTarget/)
  assert.match(pluginSource, /bot: createTarget/)
  assert.doesNotMatch(pluginSource, /setCreateOwner\(owner =>/)
  assert.doesNotMatch(pluginSource, /onChanged: \(\) => void refetch\(\)/)
})
