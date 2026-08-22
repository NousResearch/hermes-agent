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
    atom, PALETTE_AREA: 'palette', COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      request: async () => ({ jobs: [] }),
      state: {
        profile: { listen: () => undefined },
        gateway: { listen: () => undefined }
      }
    }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__sched = { scheduleLabel, composeSchedule };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context
}

test('unit: minute intervals keep their existing labels', () => {
  const { __sched } = load()
  assert.equal(__sched.scheduleLabel('every 45m'), 'Every 45m')
  assert.equal(__sched.scheduleLabel('every 60m'), 'Hourly')
  assert.equal(__sched.scheduleLabel('every 1440m'), 'Daily')
  assert.equal(__sched.scheduleLabel('every 2880m'), 'Every 2 days')
})

test("unit: hour/day intervals composeSchedule emits get friendly labels, not the raw string", () => {
  const { __sched } = load()
  // 'hourly' frequency in the picker emits exactly 'every 1h'
  assert.equal(__sched.scheduleLabel('every 1h'), 'Hourly')
  assert.equal(__sched.scheduleLabel('every 2h'), 'Every 2h')
  assert.equal(__sched.scheduleLabel('every 1d'), 'Daily')
  assert.equal(__sched.scheduleLabel('every 3d'), 'Every 3 days')
})

test('integration: every schedule the picker can emit round-trips to a friendly label', () => {
  const { __sched } = load()
  const states = [
    { freq: 'hourly' },
    { freq: 'interval', intervalN: '2', intervalUnit: 'h' },
    { freq: 'interval', intervalN: '30', intervalUnit: 'm' },
    { freq: 'interval', intervalN: '2', intervalUnit: 'd' }
  ]
  for (const state of states) {
    const schedule = __sched.composeSchedule({ time: '9:0', ...state })
    const label = __sched.scheduleLabel(schedule)
    assert.notEqual(label, schedule, `raw schedule leaked into the label for ${JSON.stringify(state)}`)
  }
})

test('unit: cron and once schedules are untouched', () => {
  const { __sched } = load()
  assert.equal(__sched.scheduleLabel('30m'), 'Once (30m)')
  assert.equal(__sched.scheduleLabel('once in 2h'), 'Once (2h)')
  assert.equal(__sched.scheduleLabel('0 9 * * *'), '0 9 * * *')
})
