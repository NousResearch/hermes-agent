import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import test from 'node:test'

import { installStdioPipeErrorGuards, isIgnorablePipeError } from './stdio-pipe-guards'

test('identifies broken standard pipe errors', () => {
  assert.equal(isIgnorablePipeError(Object.assign(new Error('closed'), { code: 'EPIPE' })), true)
  assert.equal(
    isIgnorablePipeError(Object.assign(new Error('destroyed'), { code: 'ERR_STREAM_DESTROYED' })),
    true
  )
  assert.equal(isIgnorablePipeError(new Error('broken pipe')), true)
  assert.equal(isIgnorablePipeError(new Error('unrelated failure')), false)
})

test('only absorbs broken pipe errors and installs once', () => {
  const stdout = new EventEmitter()
  const stderr = new EventEmitter()

  assert.equal(installStdioPipeErrorGuards({ stdout, stderr }), 2)
  assert.equal(installStdioPipeErrorGuards({ stdout, stderr }), 0)
  assert.doesNotThrow(() => stdout.emit('error', Object.assign(new Error('closed'), { code: 'EPIPE' })))
  assert.throws(() => stderr.emit('error', new Error('unexpected')), /unexpected/)
})
