const assert = require('node:assert/strict')
const test = require('node:test')

const {
  ZOOM_FACTORS,
  clampZoomFactor,
  getNextZoomFactor,
  getZoomPercent,
  isZoomAccelerator,
  normalizeZoomFactor
} = require('./zoom-controller.cjs')

test('normalizes zoom factors to the supported range and precision', () => {
  assert.equal(clampZoomFactor(0.1), ZOOM_FACTORS[0])
  assert.equal(clampZoomFactor(9), ZOOM_FACTORS.at(-1))
  assert.equal(normalizeZoomFactor(1.234), 1.23)
  assert.equal(normalizeZoomFactor('bad'), 1)
  assert.equal(getZoomPercent(1.25), 125)
})

test('steps zoom through modern desktop zoom stops', () => {
  assert.equal(getNextZoomFactor(1, 1), 1.1)
  assert.equal(getNextZoomFactor(1.1, 1), 1.25)
  assert.equal(getNextZoomFactor(1.25, -1), 1.1)
  assert.equal(getNextZoomFactor(1.24, 1), 1.25)
  assert.equal(getNextZoomFactor(2, 1), 2)
  assert.equal(getNextZoomFactor(0.5, -1), 0.5)
})

test('recognizes browser-style zoom accelerators', () => {
  assert.equal(isZoomAccelerator({ control: true, meta: false, alt: false, key: '+', code: 'Equal' }), 'in')
  assert.equal(isZoomAccelerator({ control: true, meta: false, alt: false, key: '=', code: 'Equal' }), 'in')
  assert.equal(isZoomAccelerator({ control: true, meta: false, alt: false, key: '-', code: 'Minus' }), 'out')
  assert.equal(isZoomAccelerator({ control: true, meta: false, alt: false, key: '0', code: 'Digit0' }), 'reset')
  assert.equal(isZoomAccelerator({ control: false, meta: true, alt: false, key: '+', code: 'Equal' }), 'in')
  assert.equal(isZoomAccelerator({ control: true, meta: false, alt: true, key: '+', code: 'Equal' }), null)
  assert.equal(isZoomAccelerator({ control: false, meta: false, alt: false, key: '+', code: 'Equal' }), null)
})
