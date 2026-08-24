import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// loadRoutines pauses persisted delegated routines inline before returning the
// list. A single pause RPC failing used to reject the whole query — the pane
// showed "Could not load cronjobs" (or the stale notice) over a list that
// loaded fine, and the 20s poll retried the failing pause inside a failing
// query forever. Each pause now swallows its own error, and the paused-state
// overlay only claims jobs the gateway actually paused.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load(request) {
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
    host: { request, state: { profile: { listen: () => undefined } } }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__routines = { loadRoutines };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context
}

const LEGACY_PREFIX = 'You are running the scheduled routine "'

test('regression: a failed legacy-routine pause does not fail the routines list', async () => {
  const jobs = [
    { job_id: 'legacy-fails', name: '[bot:research] Audit', prompt_preview: `${LEGACY_PREFIX}Audit" for agent 'research'`, enabled: true, state: 'scheduled' },
    { job_id: 'legacy-pauses', name: '[bot:research] Build', prompt_preview: `${LEGACY_PREFIX}Build" for agent 'research'`, enabled: true, state: 'scheduled' },
    { job_id: 'normal', name: '[bot:research] Report', prompt: 'Summarize the day', enabled: true, state: 'scheduled' }
  ]
  const requests = []
  const runtime = load(async (method, params) => {
    requests.push([method, params])

    if (params.action === 'list') {
      return { jobs }
    }
    if (params.action === 'pause' && params.name === 'legacy-fails') {
      throw new Error('gateway rejected the pause')
    }
    if (params.action === 'pause') {
      return { success: true }
    }
    throw new Error(`Unexpected request: ${method} ${JSON.stringify(params)}`)
  })

  // Pre-fix this rejected with the pause error.
  const result = await runtime.__routines.loadRoutines('research')
  assert.equal(result.jobs.length, 3)

  const byId = Object.fromEntries(result.jobs.map(job => [job.job_id, job]))
  // Only the pause the gateway confirmed is overlaid as paused; the failed
  // one keeps its server state and is retried on the next poll.
  assert.equal(byId['legacy-fails'].enabled, true)
  assert.equal(byId['legacy-fails'].state, 'scheduled')
  assert.equal(byId['legacy-pauses'].enabled, false)
  assert.equal(byId['legacy-pauses'].state, 'paused')
  assert.equal(byId.normal.enabled, true)

  assert.deepEqual(
    requests.map(([method, params]) => [method, params.action, params.name]),
    [
      ['cron.manage', 'list', undefined],
      ['cron.manage', 'pause', 'legacy-fails'],
      ['cron.manage', 'pause', 'legacy-pauses']
    ]
  )
})

test('regression: all-pauses-failing still resolves with the list', async () => {
  const runtime = load(async (method, params) => {
    if (params.action === 'list') {
      return { jobs: [{ job_id: 'only', name: '[bot:ops] Watch', prompt_preview: `${LEGACY_PREFIX}Watch" for agent 'ops'`, enabled: true, state: 'scheduled' }] }
    }
    throw new Error('gateway down')
  })

  const result = await runtime.__routines.loadRoutines('ops')
  assert.equal(result.jobs[0].job_id, 'only')
  assert.equal(result.jobs[0].enabled, true)
})
