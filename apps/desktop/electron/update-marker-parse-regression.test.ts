import assert from 'node:assert/strict'

import { test } from 'vitest'

import { judgeMarkerText, parseUpdateMarker } from './update-marker'

test('rejects numeric prefixes in positional update-marker fields', () => {
  assert.equal(parseUpdateMarker('4242junk\n100\n'), null)
  assert.equal(parseUpdateMarker('4242\n100junk\n'), null)
})

test('keeps a future-dated live owner without rewriting its acquisition time', async () => {
  const startedAt = 1_060
  const judgement = await judgeMarkerText(`4242\n${startedAt}\n`, {
    ourPid: 7,
    ourCt: () => null,
    isAlive: pid => pid === 4242,
    createTime: () => null,
    nowS: 1_000
  })

  assert.equal(judgement.verdict, 'live')
  assert.equal(judgement.owner, 4242)
  assert.equal(judgement.marker?.startedAt, startedAt)
})
