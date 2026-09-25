import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import { CHROMIUM_LOG_FILENAME, enableLinuxCrashDiagnostics, linuxCrashDiagnostics } from './linux-crash-diagnostics'

// Regression for #100573: the Linux shell died with SIGTRAP at Chromium's
// shared fatal-handler address and no launcher kept the FATAL message. The
// fix is not a guess at the cause; it is making the next crash legible.

test.skipIf(process.platform !== 'linux')('on linux, fatal Chromium output lands in a file under the Hermes logs dir', () => {
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

test('a logs dir that cannot be created degrades to no logging, never a dead shell', () => {
  const switches: string[] = []
  let reporterStarted = false

  // Read-only or invalid HERMES_HOME/logs: mkdir throws before app readiness.
  enableLinuxCrashDiagnostics(linuxCrashDiagnostics('/read-only/logs', 'linux'), '/read-only/logs', {
    ensureLogsDir: () => {
      throw new Error('EROFS: read-only file system')
    },
    reclaimChromiumLog: () => assert.fail('must not touch a log dir that does not exist'),
    appendSwitch: name => switches.push(name),
    startCrashReporter: () => {
      reporterStarted = true
    }
  })

  // No log-file switch (Chromium could not have opened it anyway), and the
  // crash reporter — which writes elsewhere — still runs.
  assert.deepEqual(switches, [])
  assert.equal(reporterStarted, true)
})

test('a crash reporter that refuses to start is not fatal either', () => {
  const switches: string[] = []

  enableLinuxCrashDiagnostics(linuxCrashDiagnostics('/home/u/.hermes/logs', 'linux'), '/home/u/.hermes/logs', {
    ensureLogsDir: () => {},
    reclaimChromiumLog: () => {},
    appendSwitch: name => switches.push(name),
    startCrashReporter: () => {
      throw new Error('crashpad handler missing')
    }
  })

  assert.ok(switches.includes('log-file'))
})

test('the Chromium log is bounded before Chromium appends to it', () => {
  const reclaimed: string[] = []

  enableLinuxCrashDiagnostics(linuxCrashDiagnostics('/home/u/.hermes/logs', 'linux'), '/home/u/.hermes/logs', {
    ensureLogsDir: () => {},
    reclaimChromiumLog: file => reclaimed.push(file),
    appendSwitch: () => {},
    startCrashReporter: () => {}
  })

  // Electron opens an explicit --log-file with APPEND_TO_OLD_LOG_FILE, so the
  // file it is about to append to is exactly the one that must be reclaimed.
  assert.deepEqual(reclaimed, [path.join('/home/u/.hermes/logs', CHROMIUM_LOG_FILENAME)])
})
