import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createTerminalOutputGate } from './terminal-output-gate'

test('holds the initial shell prompt until the renderer attaches', () => {
  const events: string[] = []

  const gate = createTerminalOutputGate({
    onExitFlushed: () => events.push('cleanup'),
    sendData: data => events.push(`data:${data}`),
    sendExit: payload => events.push(`exit:${payload.code}`)
  })

  gate.data('$ ')
  assert.deepEqual(events, [])

  gate.attach()
  assert.deepEqual(events, ['data:$ '])

  gate.data('pwd\r\n')
  gate.flush()
  assert.deepEqual(events, ['data:$ ', 'data:pwd\r\n'])
})

test('flushes buffered output before an early shell exit', () => {
  const events: string[] = []

  const gate = createTerminalOutputGate({
    onExitFlushed: () => events.push('cleanup'),
    sendData: data => events.push(`data:${data}`),
    sendExit: payload => events.push(`exit:${payload.code}:${payload.signal}`)
  })

  gate.data('startup failed\r\n')
  gate.exit({ code: 1, signal: null })
  assert.deepEqual(events, [])

  gate.attach()
  assert.deepEqual(events, ['data:startup failed\r\n', 'exit:1:null', 'cleanup'])
})

test('attach is idempotent and never replays startup output twice', () => {
  const events: string[] = []

  const gate = createTerminalOutputGate({
    onExitFlushed: () => events.push('cleanup'),
    sendData: data => events.push(`data:${data}`),
    sendExit: payload => events.push(`exit:${payload.code}`)
  })

  gate.data('$ ')
  gate.attach()
  gate.attach()

  assert.deepEqual(events, ['data:$ '])
})

test('coalesces a fast burst into one IPC instead of one per chunk', () => {
  const events: string[] = []

  const gate = createTerminalOutputGate({
    onExitFlushed: () => events.push('cleanup'),
    sendData: data => events.push(`data:${data}`),
    sendExit: payload => events.push(`exit:${payload.code}`)
  })

  gate.attach()

  for (let i = 0; i < 100; i += 1) {
    gate.data(`chunk-${i};`)
  }

  assert.deepEqual(events, [])
  gate.flush()
  assert.equal(events.length, 1)
  assert.ok(events[0].startsWith('data:chunk-0;'))
  assert.ok(events[0].endsWith('chunk-99;'))
})

test('an exit flushes coalesced output before the exit payload', () => {
  const events: string[] = []

  const gate = createTerminalOutputGate({
    onExitFlushed: () => events.push('cleanup'),
    sendData: data => events.push(`data:${data}`),
    sendExit: payload => events.push(`exit:${payload.code}:${payload.signal}`)
  })

  gate.attach()
  gate.data('tail output')
  gate.exit({ code: 0, signal: null })
  assert.deepEqual(events, ['data:tail output', 'exit:0:null', 'cleanup'])
})

test('dispose drops buffered output and silences later chunks', () => {
  const events: string[] = []

  const gate = createTerminalOutputGate({
    onExitFlushed: () => events.push('cleanup'),
    sendData: data => events.push(`data:${data}`),
    sendExit: payload => events.push(`exit:${payload.code}`)
  })

  gate.attach()
  gate.data('held')
  gate.dispose()
  gate.flush()
  gate.data('late')
  gate.flush()
  gate.exit({ code: 0, signal: null })
  assert.deepEqual(events, [])
})
