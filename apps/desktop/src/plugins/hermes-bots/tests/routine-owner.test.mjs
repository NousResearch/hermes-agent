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
  const requests = []
  const node = (type, props = {}) => ({ props, type })
  const context = {
    atom,
    Button: 'Button',
    Codicon: 'Codicon',
    cn: (...parts) => parts.filter(Boolean).join(' '),
    Dialog: 'Dialog',
    DialogContent: 'DialogContent',
    DialogDescription: 'DialogDescription',
    DialogFooter: 'DialogFooter',
    DialogHeader: 'DialogHeader',
    DialogTitle: 'DialogTitle',
    host: {
      state: {
        profile: { get: () => 'ops', listen: () => undefined },
        gateway: { get: () => 'open', listen: () => undefined }
      },
      request: async (method, params) => {
        requests.push({ method, params })
        return { jobs: [] }
      },
      requestProfile: async (route, method, params) => {
        requests.push({ method, params, route })
        return { jobs: [] }
      },
      notify: () => undefined,
      notifyError: () => undefined
    },
    jsx: node,
    jsxs: node,
    queryClient: {
      invalidateQueries: async options => invalidations.push(options)
    },
    relativeTime: () => 'in 5 min',
    Switch: 'Switch',
    Tip: 'Tip',
    useState: initial => [typeof initial === 'function' ? initial() : initial, () => undefined],
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
    .concat('\nglobalThis.__routineOwners = { RoutineRow, routineCreateTarget, invalidateRoutineOwner };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return { ...context, invalidations, requests }
}

/** Depth-first search over the stubbed element tree. */
function find(node, match) {
  if (Array.isArray(node)) {
    for (const child of node) {
      const hit = find(child, match)
      if (hit) return hit
    }
    return null
  }

  if (!node || typeof node !== 'object') {
    return null
  }

  return match(node) ? node : find(node.props?.children, match)
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

test('a row mutation addresses — and invalidates — the owner that rendered it', async () => {
  const runtime = load()
  const row = runtime.__routineOwners.RoutineRow({
    job: { enabled: true, job_id: 'digest', name: '[bot:ops] Digest', schedule: 'every 1h' },
    onOpen: () => undefined,
    owner: 'ops'
  })

  const toggle = find(row, node => node.type === 'Switch')
  assert.ok(toggle, 'the row still exposes its enable switch')

  // 'ops' is the row's owner while the live gateway may already be on another
  // bot: both the RPC and the cache eviction must name the owner, not the
  // ambient profile.
  await toggle.props.onCheckedChange(false)

  assert.deepEqual(plain(runtime.requests), [
    { method: 'cron.manage', params: { action: 'pause', name: 'digest', profile: 'ops' } }
  ])
  assert.deepEqual(plain(runtime.invalidations), [
    { queryKey: ['hermes-bots', 'routines', 'ops'], exact: true }
  ])
})

test('source contract: create mutations and dialog state retain one owner', () => {
  assert.match(
    pluginSource,
    /function CreateRoutineDialog\(\{ bot, open, onClose \}\)[\s\S]*await requestForBot\(bot, 'cron\.manage',[\s\S]*\.\.\.\(profile \? \{ profile \} : \{\}\)[\s\S]*await invalidateRoutineOwner\(bot\)/
  )
  assert.match(pluginSource, /const \[createOwner, setCreateOwner\] = useState\(null\)/)
  assert.match(pluginSource, /const openCreate = \(\) => \{[\s\S]*setCreateOwner\(owner\)[\s\S]*setCreateOpen\(true\)/)
  assert.match(pluginSource, /const createTarget = owner \? routineCreateTarget\(createOwner, bot\) : null/)
  // key must be the jsx() THIRD argument (a `key:` prop is silently ignored
  // by the react/jsx-runtime and the dialog would keep stale per-bot state).
  assert.match(pluginSource, /jsx\(CreateRoutineDialog, \{[\s\S]*?\}, createTarget\)/)
  assert.doesNotMatch(pluginSource, /key: createTarget/)
  assert.match(pluginSource, /bot: createTarget/)
  assert.doesNotMatch(pluginSource, /setCreateOwner\(owner =>/)
  assert.doesNotMatch(pluginSource, /onChanged: \(\) => void refetch\(\)/)
})
