import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import test from 'node:test'

import { installStdioPipeErrorGuards, isIgnorablePipeError } from './stdio-pipe-guards'

test('isIgnorablePipeError accepts only EPIPE and ERR_STREAM_DESTROYED codes', () => {
  assert.equal(isIgnorablePipeError(Object.assign(new Error('closed'), { code: 'EPIPE' })), true)
  assert.equal(
    isIgnorablePipeError(Object.assign(new Error('gone'), { code: 'ERR_STREAM_DESTROYED' })),
    true
  )
})

test('isIgnorablePipeError rejects message-only broken-pipe errors', () => {
  assert.equal(isIgnorablePipeError(new Error('broken pipe')), false)
  assert.equal(isIgnorablePipeError(new Error('EPIPE')), false)
  assert.equal(isIgnorablePipeError(new Error('ERR_STREAM_DESTROYED')), false)
  assert.equal(isIgnorablePipeError(Object.assign(new Error('write'), { code: 'EIO' })), false)
})

test('installStdioPipeErrorGuards swallows coded pipe errors on stdout/stderr', () => {
  const stdout = new EventEmitter()
  const stderr = new EventEmitter()

  assert.equal(installStdioPipeErrorGuards({ stdout, stderr }), 2)

  assert.doesNotThrow(() => stdout.emit('error', Object.assign(new Error('closed'), { code: 'EPIPE' })))
  assert.doesNotThrow(() =>
    stderr.emit('error', Object.assign(new Error('gone'), { code: 'ERR_STREAM_DESTROYED' }))
  )
})
