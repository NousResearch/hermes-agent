import assert from 'node:assert/strict'
import test from 'node:test'

import { percentile, summarize } from './stats.mjs'

test('percentile uses the documented nearest-rank index', () => {
  const values = Array.from({ length: 20 }, (_, index) => index)

  assert.equal(percentile(values, 0.95), 18)
  assert.equal(summarize(values).p95, 18)
})
