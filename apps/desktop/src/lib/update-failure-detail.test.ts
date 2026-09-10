import assert from 'node:assert/strict'

import { test } from 'vitest'

import { updateFailureDetail } from './update-failure-detail'

test("a backend/bridge diagnostic is shown verbatim instead of the surface's generic line", () => {
  const detail = updateFailureDetail(
    { message: 'git failed (exit 128) talking to origin — remote is not a git repository' },
    "We couldn't reach the update server."
  )

  assert.match(detail, /exit 128/)
  assert.doesNotMatch(detail, /couldn't reach the update server/i)
})

test('the generic line is the fallback only when there is nothing to report', () => {
  const fallback = "We couldn't reach the update server."

  assert.equal(updateFailureDetail(null, fallback), fallback)
  assert.equal(updateFailureDetail(undefined, fallback), fallback)
  assert.equal(updateFailureDetail({}, fallback), fallback)
  assert.equal(updateFailureDetail({ message: '' }, fallback), fallback)
  assert.equal(updateFailureDetail({ message: '   \n  ' }, fallback), fallback)
})

test('the detail is trimmed so a trailing newline from git stderr cannot break layout', () => {
  assert.equal(updateFailureDetail({ message: ' fatal: boom\n' }, 'fallback'), 'fatal: boom')
})
