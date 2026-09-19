import assert from 'node:assert/strict'

import { test } from 'vitest'

import { bootstrapStageLabel, describeBootstrapFailure, missingInstallPartMessage } from './bootstrap-failure-copy'

test('known stage names get everyday labels; unknown ones are humanized', () => {
  assert.equal(bootstrapStageLabel('venv'), 'Python environment')
  assert.equal(bootstrapStageLabel('system-packages'), 'System packages')
  assert.equal(bootstrapStageLabel('some-new_stage'), 'Some new stage')
  assert.equal(bootstrapStageLabel(null), null)
})

test('lead sentence is plain and actionable; raw error is confined to the Details line', () => {
  const raw = "install.ps1 exited 1: spawn ENOENT (stage 'venv')"
  const message = describeBootstrapFailure('venv', raw)
  const [lead, ...rest] = message.split('\n')

  assert.match(lead, /'Python environment' step/)
  assert.match(lead, /Reload and retry/)
  assert.match(lead, /open the logs/)
  assert.doesNotMatch(lead, /bootstrap|stage|venv|ENOENT|exited|desktop\.log/)
  assert.equal(rest.join('\n'), `Details: ${raw}`)
})

test('missing stage and missing error still produce a complete message', () => {
  const message = describeBootstrapFailure(null, undefined)

  assert.match(message, /^Setting up Hermes stopped before it could finish\./)
  assert.match(message, /\nDetails: unknown error$/)
})

test('missing-install-part copy names Repair install and keeps the path in Details', () => {
  const message = missingInstallPartMessage('Python environment missing at /home/me/.hermes/venv')
  const [lead, details] = message.split('Details: ')

  assert.match(lead, /Repair install/)
  assert.doesNotMatch(lead, /venv|install\.ps1|\//)
  assert.equal(details, 'Python environment missing at /home/me/.hermes/venv')
})
