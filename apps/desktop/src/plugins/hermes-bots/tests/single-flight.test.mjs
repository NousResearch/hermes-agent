import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadSingleFlight() {
  const start = pluginSource.indexOf('function singleFlight(')
  const end = pluginSource.indexOf('/** The agent-to-agent messaging protocol', start)
  assert.notEqual(start, -1, 'plugin exports the single-flight helper source')
  assert.notEqual(end, -1, 'single-flight helper has a stable source boundary')
  const context = {}
  vm.runInNewContext(`${pluginSource.slice(start, end)}\nglobalThis.singleFlight = singleFlight`, context)
  return context.singleFlight
}

test('concurrent callers share one in-flight profile creation', async () => {
  const singleFlight = loadSingleFlight()
  const ref = { current: null }
  let calls = 0
  let release
  const pending = new Promise(resolve => {
    release = resolve
  })
  const create = async () => {
    calls += 1
    await pending
    return 'researcher'
  }

  const first = singleFlight(ref, create)
  const second = singleFlight(ref, create)

  assert.equal(calls, 1)
  assert.equal(first, second)
  release()
  assert.equal(await first, 'researcher')
  assert.equal(ref.current, first)
})

test('a failed creation clears the flight so retry can succeed', async () => {
  const singleFlight = loadSingleFlight()
  const ref = { current: null }
  let calls = 0

  await assert.rejects(
    singleFlight(ref, async () => {
      calls += 1
      throw new Error('gateway unavailable')
    }),
    /gateway unavailable/
  )

  assert.equal(ref.current, null)
  assert.equal(await singleFlight(ref, async () => {
    calls += 1
    return 'researcher'
  }), 'researcher')
  assert.equal(calls, 2)
})
