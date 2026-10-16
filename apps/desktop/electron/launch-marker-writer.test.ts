import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createLaunchMarkerWriter } from './launch-marker-writer'

test('a primary instance persists every queued marker write', () => {
  const writer = createLaunchMarkerWriter()
  const written: string[] = []

  writer.queue(() => written.push('linux-gpu'))
  writer.queue(() => written.push('sandbox'))

  assert.equal(writer.pending(), 2)
  assert.equal(writer.flush(true), 2)
  assert.deepEqual(written, ['linux-gpu', 'sandbox'])
  assert.equal(writer.pending(), 0)
})

test('a second launch that loses the lock writes no marker at all', () => {
  const writer = createLaunchMarkerWriter()
  const written: string[] = []

  writer.queue(() => written.push('booting'))
  writer.queue(() => written.push('booting'))

  // #131055: the launch exits at the lock without ever starting a GPU or
  // sandbox child. Its `booting` write is what the next launch would read as
  // an aborted boot, so nothing may be persisted — and the running instance's
  // markers must be left exactly as they are.
  assert.equal(writer.flush(false), 0)
  assert.deepEqual(written, [])
  assert.equal(writer.pending(), 0)
})

test('a marker write that throws does not strand the ones behind it', () => {
  const writer = createLaunchMarkerWriter()
  const written: string[] = []

  writer.queue(() => {
    throw new Error('EACCES')
  })
  writer.queue(() => written.push('sandbox'))

  assert.equal(writer.flush(true), 1)
  assert.deepEqual(written, ['sandbox'])
})

test('a dropped launch cannot flush twice', () => {
  const writer = createLaunchMarkerWriter()

  writer.queue(() => undefined)

  assert.equal(writer.flush(false), 0)
  assert.equal(writer.flush(false), 0)
  assert.equal(writer.pending(), 0)
})

test('non-callable entries are ignored rather than failing the flush', () => {
  const writer = createLaunchMarkerWriter()

  writer.queue(undefined as unknown as () => void)

  assert.equal(writer.pending(), 0)
  assert.equal(writer.flush(true), 0)
})
