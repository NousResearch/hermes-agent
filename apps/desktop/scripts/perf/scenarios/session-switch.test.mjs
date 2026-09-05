import assert from 'node:assert/strict'
import test from 'node:test'

import scenario from './session-switch.mjs'

function fixtureCdp({ cacheHit = false, cachePaintDelay = 25, restBeforeResume = true } = {}) {
  let rounds = 0
  let result = null

  return {
    async eval(expression) {
      if (expression.includes('typeof window.__PERF_DRIVE__')) {
        return true
      }

      if (expression.includes('__HERMES_PERF_SESSION_SWITCH__') && expression.includes('sessionSwitch(')) {
        rounds += 1
        const select = rounds * 100
        const rest = select + 4
        const resume = select + (restBeforeResume ? 2004 : 3)

        const marks = {
          'hermes.session.cache.commit': select + 1,
          'hermes.session.resume.ready': resume,
          'hermes.session.select': select
        }

        if (!cacheHit) {
          marks['hermes.session.rest.commit'] = rest
        }

        result = {
          agentReady: null,
          cachePaintReady: select + (typeof cachePaintDelay === 'function' ? cachePaintDelay(rounds) : cachePaintDelay),
          marks
        }

        return true
      }

      if (expression.includes('__HERMES_PERF_SESSION_SWITCH__') && expression.includes('delete window')) {
        result = null

        return true
      }

      return { state: 'fulfilled', value: result }
    }
  }
}

test('verified-cache fixture measures the real unchanged path without fabricating a REST commit', async () => {
  const result = await scenario.run(fixtureCdp({ cacheHit: true }), { 'verified-cache': true, rounds: 20 })

  assert.deepEqual(result.metrics, {
    agent_ready_p95_ms: null,
    cache_first_paint_p95_ms: 25,
    rest_before_resume_count: 0,
    rest_commit_p95_ms: null,
    resume_ready_p95_ms: 2004
  })
  assert.equal(result.detail.fixture, 'synthetic-local-no-network')
  assert.equal(result.detail.agent_ready, 'unavailable-current-backend-protocol')
})

test('delayed-runtime fixture rejects a round where REST does not precede runtime readiness', async () => {
  await assert.rejects(
    scenario.run(fixtureCdp({ restBeforeResume: false }), { 'delay-runtime': 2000, rounds: 5 }),
    /REST before resume in every round \(0\/5 observed\)/
  )
})

test('verified-cache requires at least the planned 20 controlled rounds', async () => {
  await assert.rejects(
    scenario.run(fixtureCdp(), { 'verified-cache': true, rounds: 19 }),
    /--verified-cache requires at least 20 rounds/
  )
})

test('controlled fixture rejects a negative inter-round settle interval', async () => {
  await assert.rejects(
    scenario.run(fixtureCdp(), { 'delay-runtime': 1, 'round-settle-ms': -1, rounds: 2 }),
    /--round-settle-ms must be a non-negative number/
  )
})

test('verified-cache gate reports the measured distribution when it fails', async () => {
  await assert.rejects(
    scenario.run(
      fixtureCdp({ cacheHit: true, cachePaintDelay: round => (round === 20 ? 140 : 120) }),
      { 'verified-cache': true, rounds: 20 }
    ),
    /min=120ms, p50=120ms, p90=120ms, p95=120ms, max=140ms/
  )
})

test('retains the controlled fixture promise in the renderer across a route change', async () => {
  let retained = false

  const cdp = {
    async eval(expression) {
      if (expression.includes('typeof window.__PERF_DRIVE__')) {
        return true
      }

      if (expression.includes('__HERMES_PERF_SESSION_SWITCH__') && expression.includes('sessionSwitch(')) {
        retained = true

        return true
      }

      if (expression.includes('__HERMES_PERF_SESSION_SWITCH__') && expression.includes('delete window')) {
        return true
      }

      if (expression.includes('__HERMES_PERF_SESSION_SWITCH__') && retained) {
        return {
          state: 'fulfilled',
          value: {
            agentReady: null,
            marks: {
              'hermes.session.rest.commit': 4,
              'hermes.session.resume.ready': 2004,
              'hermes.session.select': 0
            }
          }
        }
      }

      throw new Error('Promise was collected')
    }
  }

  const result = await scenario.run(cdp, { 'delay-runtime': 2000, rounds: 1 })

  assert.equal(result.metrics.rest_before_resume_count, 1)
})
