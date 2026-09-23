import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import { linuxCrashDiagnostics } from './linux-crash-diagnostics'

// Regression for #100573: the Linux shell died with SIGTRAP at Chromium's
// shared fatal-handler address and no launcher kept the FATAL message. The
// fix is not a guess at the cause; it is making the next crash legible.

test('on linux, fatal Chromium output lands in a file under the Hermes logs dir', () => {
  const plan = linuxCrashDiagnostics('/home/u/.hermes/logs', 'linux')

  assert.ok(plan)

  const switches = new Map(plan.switches)

  assert.equal(switches.get('enable-logging'), 'file')

  const logFile = switches.get('log-file')

  assert.ok(logFile)
  assert.equal(path.dirname(logFile), '/home/u/.hermes/logs')
  // FATAL (3) must survive the level filter; anything stricter would drop it.
  assert.ok(Number(switches.get('log-level')) <= 3)
  assert.equal(plan.crashReporter.uploadToServer, false)
})

test('other platforms get no Chromium logging switches and no crash reporter', () => {
  assert.equal(linuxCrashDiagnostics('/Users/u/.hermes/logs', 'darwin'), null)
  assert.equal(linuxCrashDiagnostics('C:\\Users\\u\\.hermes\\logs', 'win32'), null)
})
