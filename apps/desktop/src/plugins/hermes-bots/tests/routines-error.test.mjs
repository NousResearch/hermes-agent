import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load() {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const context = {
    atom,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: { state: { profile: { listen: () => undefined } } }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__api = { selectRoutineJobs };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context
}

const jobs = [
  { name: '[bot:ops] Morning', job_id: '1' },
  { name: '[bot:research] Digest', job_id: '2' }
]

test('unit: a live list is filtered to the current bot', () => {
  const { jobs: shown, live } = load().__api.selectRoutineJobs({ jobs }, null, [], 'ops')
  assert.equal(live.length, 2)
  assert.equal(shown.length, 1)
  assert.equal(shown[0].job_id, '1')
})

test('regression: a profile-scoped list shows every job owned by that bot profile', () => {
  const profileJobs = [
    { name: 'ordinary profile cronjob', job_id: 'legacy' },
    { name: '[bot:ops] Bot Mode routine', job_id: 'routine' }
  ]
  const view = load().__api.selectRoutineJobs({ jobs: profileJobs, scoped: 'ops' }, null, [], 'ops')

  assert.deepEqual(
    Array.from(view.jobs, job => job.job_id),
    ['legacy', 'routine']
  )
})

test('compatibility: an unmarked list keeps tag filtering for older gateways', () => {
  const profileJobs = [
    { name: 'ordinary launch-profile cronjob', job_id: 'legacy' },
    { name: '[bot:ops] Bot Mode routine', job_id: 'routine' }
  ]
  const view = load().__api.selectRoutineJobs({ jobs: profileJobs }, null, [], 'ops')

  assert.deepEqual(
    Array.from(view.jobs, job => job.job_id),
    ['routine']
  )
})

test('compatibility: a stale scope marker cannot leak another profile\'s jobs', () => {
  const profileJobs = [
    { name: 'ordinary research cronjob', job_id: 'research' },
    { name: '[bot:ops] Bot Mode routine', job_id: 'routine' }
  ]
  const view = load().__api.selectRoutineJobs({ jobs: profileJobs, scoped: 'research' }, null, [], 'ops')

  assert.deepEqual(
    Array.from(view.jobs, job => job.job_id),
    ['routine']
  )
})

test('unit: a failed refresh keeps the last good list', () => {
  const view = load().__api.selectRoutineJobs(undefined, new Error('down'), jobs, 'ops')
  assert.equal(view.live, null)
  assert.equal(view.all.length, 2)
  assert.equal(view.jobs.length, 1)
})

test('regression: a true empty list is still empty', () => {
  const view = load().__api.selectRoutineJobs({ jobs: [] }, null, jobs, 'ops')
  assert.ok(view.live)
  assert.equal(view.jobs.length, 0)
})

test('regression: a first load error has no jobs to show', () => {
  const view = load().__api.selectRoutineJobs(undefined, new Error('down'), [], 'ops')
  assert.equal(view.all.length, 0)
  assert.equal(view.jobs.length, 0)
})
