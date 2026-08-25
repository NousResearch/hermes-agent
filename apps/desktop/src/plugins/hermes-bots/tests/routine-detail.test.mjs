import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Bot Mode's Cronjobs rows were inert: the only interactive controls were the
// enable switch and the hover-only delete button, so clicking a cronjob to see
// what it runs, when it runs next, or why it stopped did nothing at all. The
// gateway already ships every one of those facts with `cron.manage list`, so
// the inspector reads the record the pane is holding — no extra RPC, and no
// second mutation path beside the row's own switch and delete.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load() {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const node = (type, props = {}) => ({ props, type })
  const context = {
    atom,
    Button: 'Button',
    Codicon: 'Codicon',
    COMPOSER_AREAS: { middleware: 'middleware' },
    cn: (...parts) => parts.filter(Boolean).join(' '),
    Dialog: 'Dialog',
    DialogContent: 'DialogContent',
    DialogDescription: 'DialogDescription',
    DialogFooter: 'DialogFooter',
    DialogHeader: 'DialogHeader',
    DialogTitle: 'DialogTitle',
    document: { createElement: () => ({}), getElementById: () => null, head: { appendChild: () => undefined } },
    host: { state: { profile: { listen: () => undefined } } },
    jsx: node,
    jsxs: node,
    PALETTE_AREA: 'palette',
    relativeTime: () => 'in 5 min',
    Switch: 'Switch',
    Tip: 'Tip',
    useState: initial => [typeof initial === 'function' ? initial() : initial, () => undefined]
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(`
      globalThis.__api = { RoutineDetailDialog, RoutineRow, routineDetailIssue, routineDetailRows };
    `)
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context.__api
}

/** Depth-first walk over the stubbed element tree. */
function collect(node, match, found = []) {
  if (Array.isArray(node)) {
    for (const child of node) collect(child, match, found)
    return found
  }

  if (!node || typeof node !== 'object') {
    return found
  }

  if (match(node)) {
    found.push(node)
  }

  return collect(node.props?.children, match, found)
}

function textOf(node) {
  return collect(node, entry => typeof entry.props?.children === 'string')
    .map(entry => entry.props.children)
    .join(' | ')
}

const activeJob = {
  deliver: 'bot-chat',
  enabled: true,
  job_id: 'job-1',
  last_run_at: '2026-08-23T09:00:00Z',
  last_status: 'success',
  name: '[bot:notetaker] Morning digest',
  next_run_at: '2026-08-23T10:00:00Z',
  prompt_preview: 'Summarize yesterday and post it.',
  repeat: 'forever',
  schedule: 'every 1440m'
}

// ── the facts the row never showed ──────────────────────────────────────────

test('detail rows only carry fields the gateway actually sent', () => {
  const { routineDetailRows } = load()
  const rows = routineDetailRows({ enabled: true, job_id: 'bare', name: 'Bare', schedule: 'every 1h' })
  const labels = rows.map(row => row.label)

  // A job that has never run carries no last_run_at/last_status/model; those
  // rows must be absent rather than rendering an empty or "undefined" value.
  assert.ok(!labels.includes('Last run'))
  assert.ok(!labels.includes('Last result'))
  assert.ok(!labels.includes('Model'))
  assert.ok(rows.every(row => row.value.trim().length > 0))
})

test('an active job reports its next run; a paused one does not claim one', () => {
  const { routineDetailRows } = load()
  const value = (rows, label) => rows.find(row => row.label === label)?.value

  const active = routineDetailRows(activeJob)
  assert.equal(value(active, 'Status'), 'Active')
  assert.ok(value(active, 'Next run'))

  const paused = routineDetailRows({ ...activeJob, enabled: false, state: 'paused' })
  assert.equal(value(paused, 'Status'), 'Paused')
  assert.equal(value(paused, 'Next run'), undefined, 'a paused job has no next run to promise')
  assert.ok(value(paused, 'Last run'), 'history it does have is still shown')
})

test('the raw schedule appears only when the humanized label dropped something', () => {
  const { routineDetailRows } = load()
  const raw = rows => rows.find(row => row.label === 'Schedule (raw)')?.value

  // "every 1440m" humanizes to "Daily" — the raw form still carries the cadence.
  assert.equal(raw(routineDetailRows(activeJob)), 'every 1440m')
  // A schedule the label passes through unchanged would only be duplicated.
  const passthrough = routineDetailRows({ ...activeJob, schedule: '0 9 * * 1-5' })
  assert.equal(raw(passthrough), undefined)
  assert.equal(passthrough.find(row => row.label === 'Schedule')?.value, '0 9 * * 1-5')
})

test('a failing or scheduler-paused job explains itself, in failure order', () => {
  const { routineDetailIssue } = load()

  assert.equal(routineDetailIssue(activeJob), null)
  assert.equal(routineDetailIssue({ ...activeJob, paused_reason: 'too many failures' }), 'too many failures')
  assert.equal(
    routineDetailIssue({ ...activeJob, last_delivery_error: 'telegram 401', paused_reason: 'too many failures' }),
    'telegram 401'
  )
  assert.equal(
    routineDetailIssue({
      ...activeJob,
      last_delivery_error: 'telegram 401',
      last_fire_error: 'model timeout',
      paused_reason: 'too many failures'
    }),
    'model timeout',
    'the run that never happened outranks the delivery of a run that did'
  )
})

// ── the row is reachable ────────────────────────────────────────────────────

test('the row title is a button that opens THIS job', () => {
  const { RoutineRow } = load()
  const opened = []
  const tree = RoutineRow({ job: activeJob, onOpen: job => opened.push(job), owner: 'notetaker' })
  const buttons = collect(tree, entry => entry.type === 'button' && typeof entry.props.onClick === 'function')

  assert.ok(buttons.length >= 1, 'the row exposes at least one activation target')
  const [title] = buttons
  title.props.onClick()
  assert.deepEqual(opened, [activeJob])
})

test('opening the details cannot swallow the switch or the delete control', () => {
  const { RoutineRow } = load()
  const tree = RoutineRow({ job: activeJob, onOpen: () => undefined, owner: 'notetaker' })
  const openers = collect(tree, entry => entry.props?.title === 'Cronjob details')

  assert.equal(openers.length, 1)
  // The switch and the delete button must not live INSIDE the opener: a click
  // on them would otherwise also open the inspector (and nested interactive
  // elements are invalid markup).
  assert.deepEqual(collect(openers[0], entry => entry.type === 'Switch'), [])
  assert.deepEqual(collect(openers[0], entry => entry.type === 'Tip'), [])
  assert.equal(collect(tree, entry => entry.type === 'Switch').length, 1)
})

// ── the inspector ───────────────────────────────────────────────────────────

test('the inspector renders the job’s instruction and its failure', () => {
  const { RoutineDetailDialog } = load()
  const job = { ...activeJob, last_fire_error: 'model timeout' }
  const rendered = textOf(RoutineDetailDialog({ job, onClose: () => undefined, open: true }))

  assert.match(rendered, /Morning digest/)
  assert.match(rendered, /Summarize yesterday and post it\./)
  assert.match(rendered, /model timeout/)
  assert.doesNotMatch(rendered, /\[bot:notetaker\]/, 'the routing tag is plumbing, not a title')
})

test('the inspector stays shut without a job to inspect', () => {
  const { RoutineDetailDialog } = load()

  assert.equal(RoutineDetailDialog({ job: null, onClose: () => undefined, open: true }).props.open, false)
  assert.equal(RoutineDetailDialog({ job: activeJob, onClose: () => undefined, open: false }).props.open, false)
  assert.equal(RoutineDetailDialog({ job: activeJob, onClose: () => undefined, open: true }).props.open, true)
})
