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
