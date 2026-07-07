const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const {
  createDiagnosticsRecorder,
  normalizeDiagnosticEvent,
  planDiagnosticsRotation,
  redactDiagnosticValue
} = require('./diagnostics.cjs')

test('redactDiagnosticValue redacts sensitive keys recursively', () => {
  assert.deepEqual(
    redactDiagnosticValue({
      nested: {
        apiKey: 'sk-secret',
        ok: 'value',
        tokenPreview: 'abc'
      }
    }),
    {
      nested: {
        apiKey: '[Redacted]',
        ok: 'value',
        tokenPreview: '[Redacted]'
      }
    }
  )
})

test('normalizeDiagnosticEvent fills safe defaults', () => {
  const event = normalizeDiagnosticEvent({ component: 'renderer', event: 'heartbeat', severity: 'nope' })

  assert.equal(event.component, 'renderer')
  assert.equal(event.event, 'heartbeat')
  assert.equal(event.severity, 'info')
  assert.equal(typeof event.ts, 'string')
})

test('planDiagnosticsRotation cascades backups', () => {
  assert.deepEqual(planDiagnosticsRotation('/tmp/desktop-events.jsonl', 10, 100, 2), [])
  assert.deepEqual(planDiagnosticsRotation('/tmp/desktop-events.jsonl', 100, 100, 2), [
    ['rm', '/tmp/desktop-events.jsonl.2'],
    ['mv', '/tmp/desktop-events.jsonl.1', '/tmp/desktop-events.jsonl.2'],
    ['mv', '/tmp/desktop-events.jsonl', '/tmp/desktop-events.jsonl.1']
  ])
})

test('createDiagnosticsRecorder writes jsonl and keeps recent events', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-diagnostics-'))
  const filePath = path.join(dir, 'desktop-events.jsonl')
  const recorder = createDiagnosticsRecorder({ filePath, recentLimit: 2 })

  recorder.record({ component: 'backend', event: 'spawn', message: 'one' })
  recorder.record({ component: 'backend', event: 'exit', message: 'two' })
  recorder.record({ component: 'renderer', event: 'heartbeat.missed', message: 'three', severity: 'warn' })

  const lines = fs.readFileSync(filePath, 'utf8').trim().split('\n').map(line => JSON.parse(line))
  assert.equal(lines.length, 3)
  assert.deepEqual(
    recorder.recent().map(event => event.message),
    ['two', 'three']
  )
})
