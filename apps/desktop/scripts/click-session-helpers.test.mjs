import vm from 'node:vm'

import { expect, test } from 'vitest'

import { buildClickSessionExpression, readEvaluationValue } from './click-session-helpers.mjs'

function runExpression(expression, document, window) {
  return vm.runInNewContext(expression, {
    Date,
    JSON,
    Promise,
    document,
    setTimeout,
    window
  })
}

test('click-session scrolls with explicit positioning and waits for visibility', async () => {
  let measurements = 0
  let scrollOptions
  let scrollCalls = 0
  let clicked = false
  let clickedAtMeasurement

  const session = {
    textContent: 'Phaser particle',
    scrollIntoView(options) {
      scrollCalls += 1
      scrollOptions = options
    },
    getBoundingClientRect() {
      measurements += 1
      return measurements === 1
        ? { top: 900, right: 120, bottom: 940, left: 20 }
        : { top: 280, right: 120, bottom: 320, left: 20 }
    },
    click() {
      clicked = true
      clickedAtMeasurement = measurements
    }
  }

  const result = await runExpression(
    buildClickSessionExpression('Phaser particle'),
    {
      documentElement: { clientHeight: 600, clientWidth: 800 },
      querySelectorAll: () => [session]
    },
    { innerHeight: 600, innerWidth: 800 }
  )

  expect(scrollOptions).toEqual({ behavior: 'auto', block: 'center' })
  expect(scrollCalls).toBe(2)
  expect(result.found).toBe(true)
  expect(result.visible).toBe(true)
  expect(clicked).toBe(true)
  expect(clickedAtMeasurement).toBe(2)
  expect(measurements).toBe(2)
})

test('reads values from the nested CDP Runtime.evaluate response', () => {
  const value = { found: true, visible: true }
  const response = { result: { result: { value } } }

  expect(readEvaluationValue(response)).toBe(value)
  expect(readEvaluationValue({ result: { value } })).toBeUndefined()
})

test('click-session reports a missing title without touching the DOM', async () => {
  const result = await runExpression(
    buildClickSessionExpression('missing'),
    {
      documentElement: { clientHeight: 600, clientWidth: 800 },
      querySelectorAll: () => []
    },
    { innerHeight: 600, innerWidth: 800 }
  )

  expect(result).toEqual({ found: false, tried: 'missing' })
})
