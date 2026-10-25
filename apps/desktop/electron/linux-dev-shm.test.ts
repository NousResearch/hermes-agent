import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { applyLinuxDevShmFallback, DEV_SHM_FALLBACK_SWITCH, probeDevShmUsable } from './linux-dev-shm'

// Regression for #138277: a /dev/shm mount that refuses creates kills every
// Chromium renderer pre-main (platform_shared_memory_region_posix FATAL) and
// the desktop crash-loops. The fix probes the mount with a real create before
// `app` is ready and switches renderers to temp-file-backed shared memory.

test('a usable shm dir probes ok and leaves no probe file behind', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-shm-ok-'))

  const outcome = probeDevShmUsable(dir)

  assert.equal(outcome.ok, true)
  assert.equal(outcome.error, undefined)
  assert.deepEqual(fs.readdirSync(dir), [])
})

test('a missing shm dir probes not-ok and carries the errno for the log line', () => {
  const outcome = probeDevShmUsable(path.join(os.tmpdir(), `hermes-shm-missing-${Date.now()}`))

  assert.equal(outcome.ok, false)
  assert.match(outcome.error ?? '', /ENOENT/)
})

test('a create-refusing shm dir (0o500) probes not-ok', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-shm-readonly-'))
  fs.writeFileSync(path.join(dir, 'sentinel'), 'x')
  fs.chmodSync(dir, 0o500)

  const outcome = probeDevShmUsable(dir)

  fs.chmodSync(dir, 0o700)
  assert.equal(outcome.ok, false)
})

test('on linux, an unusable shm mount appends disable-dev-shm-usage and logs why', () => {
  const switches: string[] = []
  const logs: string[] = []

  applyLinuxDevShmFallback('linux', {
    probe: () => ({ ok: false, error: 'ENOENT: no such file or directory, open \'/dev/shm/.hermes-shm-probe-1\'' }),
    appendSwitch: name => switches.push(name),
    log: message => logs.push(message)
  })

  assert.deepEqual(switches, [DEV_SHM_FALLBACK_SWITCH])
  assert.match(logs[0] ?? '', /#138277/)
  assert.match(logs[0] ?? '', /--disable-dev-shm-usage/)
})

test('on linux, a usable shm mount keeps POSIX shm and stays silent', () => {
  const switches: string[] = []

  applyLinuxDevShmFallback('linux', {
    probe: () => ({ ok: true }),
    appendSwitch: name => switches.push(name),
    log: () => assert.fail('a healthy host must not announce a fallback')
  })

  assert.deepEqual(switches, [])
})

test('off linux the mount is never probed and no switch is set', () => {
  applyLinuxDevShmFallback('darwin', {
    probe: () => assert.fail('must not probe off linux'),
    appendSwitch: () => assert.fail('must not switch off linux'),
    log: () => assert.fail('must not log off linux')
  })
})

test('a host side effect that throws never breaks startup', () => {
  applyLinuxDevShmFallback('linux', {
    probe: () => {
      throw new Error('EIO: I/O error')
    },
    appendSwitch: () => assert.fail('must not switch when the probe itself dies'),
    log: () => assert.fail('must not log when the probe itself dies')
  })
})
