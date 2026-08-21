import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')
const start = source.indexOf('function RoutineRow(')
const end = source.indexOf('// Structured schedule picker:', start)

assert.ok(start >= 0 && end > start, 'RoutineRow must remain extractable')

function renderRoutineRow() {
  const context = {
    Codicon: 'Codicon',
    Switch: 'Switch',
    Tip: 'Tip',
    cn: (...values) => values.filter(Boolean).join(' '),
    host: { request: async () => undefined, notifyError: () => undefined },
    invalidateRoutineOwner: async () => undefined,
    isLegacyDelegatedRoutine: () => false,
    jsx: (type, props) => ({ type, props }),
    jsxs: (type, props) => ({ type, props }),
    relativeTime: () => 'in 4 days',
    routineTitle: () => 'Daily digest',
    scheduleLabel: () => 'Every day',
    useState: initial => [initial, () => undefined]
  }

  vm.createContext(context)
  vm.runInContext(`${source.slice(start, end)}\nglobalThis.__RoutineRow = RoutineRow`, context)

  return context.__RoutineRow({
    job: {
      enabled: true,
      job_id: 'daily-digest',
      next_run_at: '2026-08-23T09:00:00Z',
      schedule: '0 9 * * *'
    },
    profile: 'default'
  })
}

test('routine metadata wraps without truncating the schedule or next-run time', () => {
  const row = renderRoutineRow()
  const metadata = row.props.children[1]
  const [schedule, nextRun] = metadata.props.children

  assert.match(metadata.props.className, /\bflex-wrap\b/)
  assert.match(schedule.props.className, /\bshrink-0\b/)
  assert.match(schedule.props.className, /\bwhitespace-nowrap\b/)
  assert.match(nextRun.props.className, /\bshrink-0\b/)
  assert.match(nextRun.props.className, /\bwhitespace-nowrap\b/)
  assert.doesNotMatch(nextRun.props.className, /\btruncate\b/)
  assert.equal(nextRun.props.children, 'next in 4 days')
})
