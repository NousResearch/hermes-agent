/**
 * Tests for electron/update-marker.ts — the in-app update mutual-exclusion
 * marker that prevents a desktop relaunched mid-update from spawning a backend
 * the updater then kills in a loop (#50238).
 *
 * Run with: node --test electron/update-marker.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Why this matters: the gate must (a) report a live update only when the
 * updater pid is alive AND the marker is fresh, (b) treat absent/malformed/
 * dead-pid/expired markers as "no live update" so a crashed updater can't
 * strand future launches, and (c) self-heal by deleting a stale marker file.
 */

import fs from 'fs'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import os from 'os'
import path from 'path'

import { test } from 'vitest'

import {
  hasLiveUpdateMarker,
  isMacUpdateProcess,
  isPidAlive,
  markerPath,
  posixProcessState,
  readLiveUpdateMarker,
  UPDATE_MARKER_MAX_AGE_MS,
  updateHandoffConflict,
  writeUpdateMarker
} from './update-marker'

function tmpHome(tag) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-marker-${tag}-`))

  return dir
}

function writeMarker(home, pid, startedAtSec) {
  fs.writeFileSync(markerPath(home), `${pid}\n${startedAtSec}`)
}

const ALIVE: typeof process.kill = () => true // injected kill that "succeeds" => pid alive

const DEAD: typeof process.kill = () => {
  const err = new Error('no such process')

  ;(err as any).code = 'ESRCH'
  throw err
}

test('macOS update-process check accepts only update hand-off commands', () => {
  const inspect = (_command: string, _args: string[], _options: object) =>
    JSON.stringify(['/bin/bash', '/Users/me/Hermes Agent/scripts/desktop-update/posix.sh', '--daemonized'])

  const unrelated = (_command: string, _args: string[], _options: object) =>
    JSON.stringify(['/Applications/Notes.app/Contents/MacOS/Notes'])

  assert.equal(isMacUpdateProcess(4242, inspect), true)
  assert.equal(isMacUpdateProcess(4242, unrelated), false)
  assert.equal(
    isMacUpdateProcess(4242, () => {
      throw new Error('ps unavailable')
    }),
    true,
    'inspection failures must preserve the update gate'
  )
})

test.skipIf(process.platform !== 'darwin')('update identity uses entrypoint and subcommand boundaries, never arbitrary arguments', () => {
  for (const argv of [
    ['/usr/bin/python3', '-m', 'hermes_cli.main', 'update'],
    ['/opt/Tools With Spaces/bin/python3.14', '/opt/Tools With Spaces/bin/hermes', '--profile', 'work', 'update'],
    ['/opt/Tools With Spaces/bin/hermes', 'update']
  ]) {
    assert.equal(isMacUpdateProcess(4242, inspectArgv(argv)), true)
  }

  for (const argv of [
    ['/usr/bin/printf', 'hermes update'],
    ['/bin/bash', '/tmp/ordinary.sh', '/Users/me/scripts/desktop-update/posix.sh'],
    ['/usr/bin/python3', '-c', 'print("hermes update")'],
    ['/usr/bin/python3', '/tmp/ordinary.py', '-m', 'hermes_cli.main', 'update'],
    ['/usr/bin/hermes', 'chat', 'hermes_cli.main', 'update']
  ]) {
    assert.equal(isMacUpdateProcess(4242, inspectArgv(argv)), false)
  }

  assert.equal(isMacUpdateProcess(4242, () => 'unparseable arguments'), true)
  assert.equal(isMacUpdateProcess(4242, () => JSON.stringify([])), true)
})

function inspectArgv(argv: string[]) {
  return (command: string, args: string[], options: object) =>
    args.length === 5 ? JSON.stringify(argv) : execFileSync(command, args, options)
}

test.skipIf(process.platform !== 'darwin')('formal inline launchers retain their real CLI update owner', () => {
  const root = path.resolve(import.meta.dirname, '../../..')

  const fixture = `import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli._launchers import runtime_command, _launcher_script
root = Path(sys.argv[1])
print(json.dumps([runtime_command(root, python="/usr/bin/python3"),
    ["/usr/bin/python3", "-I", "-c", _launcher_script("hermes", root, None)]]))`

  const factoryPython = process.env.HERMES_PYTHON || path.join(root, '.venv', 'bin', 'python')

  const launchers: string[][] = JSON.parse(
    execFileSync(factoryPython, ['-I', '-c', fixture, root], { encoding: 'utf8' })
  )

  for (const launcher of launchers) {
    for (const args of [['update'], ['--profile', 'work', 'update'], ['-p', 'work', 'update'], ['--profile=work', 'update']]) {
      const home = tmpHome('formal-inline-owner')
      const now = 1_000_000_000_000
      writeMarker(home, 4242, Math.floor(now / 1000) - 5)
      const inspect = inspectArgv([...launcher, ...args])

      const live = readLiveUpdateMarker(home, {
        kill: ALIVE,
        now: () => now,
        processState: () => 'S',
        isExpectedUpdateProcess: pid => isMacUpdateProcess(pid, inspect)
      })

      assert.equal(live?.pid, 4242, 'formal managed launcher must not lose its update claim')
      assert.ok(fs.existsSync(markerPath(home)))
    }

    assert.equal(
      isMacUpdateProcess(4242, inspectArgv([...launcher, '--profile', 'work', 'chat'])),
      true,
      'unproved inline identity remains unknown even when argv suggests a non-update command'
    )
  }
})

test.skipIf(process.platform !== 'darwin')('inline source inspection rejects only a proven harmless literal print', () => {
  assert.equal(isMacUpdateProcess(4242, inspectArgv(['/usr/bin/python3', '-c', 'print("hermes update")'])), false)

  for (const argv of [
    ['/usr/bin/python3', '-I', '-c', 'arbitrary_launcher()', 'update'],
    ['/usr/bin/python3', '-X', 'utf8', '-c', 'arbitrary_launcher()', 'update'],
    ['/usr/bin/python3', '-Iu', '-c', 'arbitrary_launcher()', 'update'],
    ['/usr/bin/python3', '-I', '-c', 'arbitrary_launcher()', '--profile', 'work', 'chat'],
    ['/usr/bin/python3', '-c', 'print("hermes update"); acquire_update_lock()'],
    ['/usr/bin/python3', '-c', 'print(*make_args())'],
    ['/usr/bin/python3', '-c', 'print("hermes update", file=open("/tmp/output", "w"))']
  ]) {
    assert.equal(isMacUpdateProcess(4242, inspectArgv(argv)), true, 'unproved inline identity remains unknown')
  }
})

test('the host boot gate handles a real non-updater process', () => {
  const home = tmpHome('host-boot-gate')
  writeMarker(home, process.pid, Math.floor(Date.now() / 1000))

  assert.equal(hasLiveUpdateMarker(home), process.platform !== 'darwin')
  assert.equal(fs.existsSync(markerPath(home)), process.platform !== 'darwin')
})

test('absent marker => no live update', () => {
  const home = tmpHome('absent')
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE }), null)
})

test('live pid within age ceiling => live update reported', () => {
  const home = tmpHome('live')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5) // 5s old
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.ok(res, 'a fresh, alive marker is a live update')
  assert.equal(res.pid, 4242)
  assert.ok(res.ageMs >= 0 && res.ageMs < 10_000)
  assert.ok(fs.existsSync(markerPath(home)), 'a live marker is NOT deleted')
})

test('a live macOS pid outside the update hand-off is stale and is pruned', () => {
  const home = tmpHome('unrelated-live-macos-pid')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5)

  assert.equal(
    readLiveUpdateMarker(home, {
      kill: ALIVE,
      now: () => now,
      isExpectedUpdateProcess: () => false
    }),
    null,
    'PID reuse must not keep the Desktop update gate closed'
  )
  assert.ok(fs.existsSync(markerPath(home)) === false, 'an unrelated live pid marker self-heals')
})

test('an expected macOS hand-off pid remains a live update', () => {
  const home = tmpHome('expected-live-macos-handoff')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5)

  const res = readLiveUpdateMarker(home, {
    kill: ALIVE,
    now: () => now,
    isExpectedUpdateProcess: pid => pid === 4242
  })

  assert.equal(res?.pid, 4242)
  assert.ok(fs.existsSync(markerPath(home)), 'a live hand-off marker is retained')
})

test('an expected hand-off that is a zombie is stale and is pruned', () => {
  const home = tmpHome('expected-zombie-handoff')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5)

  assert.equal(
    readLiveUpdateMarker(home, {
      kill: ALIVE,
      now: () => now,
      isExpectedUpdateProcess: () => {
        assert.fail('a zombie owner must be cleared without inspecting identity')
      },
      processState: () => 'Z'
    }),
    null
  )
  assert.ok(!fs.existsSync(markerPath(home)), 'identity must not revive an exited owner')
})

test('identity inspection errors retain a live owner with unknown process state', () => {
  const home = tmpHome('inspection-failed-live-owner')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5)

  const res = readLiveUpdateMarker(home, {
    kill: ALIVE,
    now: () => now,
    isExpectedUpdateProcess: () => {
      throw new Error('ps unavailable')
    },
    processState: () => null
  })

  assert.equal(res?.pid, 4242)
  assert.ok(fs.existsSync(markerPath(home)), 'inspection failure must preserve the update gate')
})

test('dead pid => no live update and marker is pruned', () => {
  const home = tmpHome('dead')
  writeMarker(home, 999999, Math.floor(Date.now() / 1000))
  assert.equal(readLiveUpdateMarker(home, { kill: DEAD }), null)
  assert.ok(!fs.existsSync(markerPath(home)), 'a dead-pid marker self-heals (deleted)')
})

test('zombie pid => no live update and marker is pruned', () => {
  // The kill(pid, 0) false positive: a process that exited but is still in the
  // table (parent has not reaped it) answers signal 0 like a live one. The
  // state probe must turn that into "dead" so the boot gate self-heals in
  // seconds instead of parking for the whole 20-minute ceiling.
  const home = tmpHome('zombie')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5)
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now, processState: () => 'Z' })
  assert.equal(res, null, 'a zombie owner is not a live update')
  assert.ok(!fs.existsSync(markerPath(home)), 'a zombie-owned marker self-heals (deleted)')
})

test('a live state keeps the marker (probe answers non-Z)', () => {
  const home = tmpHome('state-live')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5)
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now, processState: () => 'S' })
  assert.ok(res, 'an alive, non-zombie owner keeps the gate closed')
  assert.ok(fs.existsSync(markerPath(home)), 'a live marker is NOT deleted')
})

test('an unknown process state fails open to alive (keeps the marker)', () => {
  const home = tmpHome('state-unknown')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5)
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now, processState: () => null })
  assert.ok(res, 'probe failure must keep the conservative signal-0 verdict')
  assert.ok(fs.existsSync(markerPath(home)))
})

test('expired marker (past age ceiling) => no live update and pruned', () => {
  const home = tmpHome('expired')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  // Even though the pid is "alive", the marker is too old to trust.
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE, now: () => now }), null)
  assert.ok(!fs.existsSync(markerPath(home)), 'an expired marker self-heals (deleted)')
})

test('malformed marker => no live update and pruned', () => {
  const home = tmpHome('malformed')
  fs.writeFileSync(markerPath(home), 'not-a-pid\nnonsense')
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE }), null)
  assert.ok(!fs.existsSync(markerPath(home)))
})

test('isPidAlive: own pid is alive, impossible pid is dead', () => {
  assert.equal(isPidAlive(process.pid), true)
  assert.equal(isPidAlive(-1), false)
  assert.equal(isPidAlive(0), false)
  assert.equal(isPidAlive(NaN), false)
})

test('isPidAlive: EPERM counts as alive (process owned by another user)', () => {
  const eperm = () => {
    const err = new Error('operation not permitted')

    ;(err as any).code = 'EPERM'
    throw err
  }

  assert.equal(isPidAlive(4242, eperm), true)
})

test('posixProcessState: own pid is probeable and not a zombie; dead pid is unknown', () => {
  if (process.platform === 'win32') {
    // Windows has no zombie state and no ps stat lane; the probe is a no-op.
    assert.equal(posixProcessState(process.pid), null)

    return
  }

  const own = posixProcessState(process.pid)
  assert.ok(own, 'a live pid must be probeable on linux/darwin')
  assert.ok(!own.toUpperCase().startsWith('Z'), 'this process is not a zombie')

  // A pid nothing owns (and that kill(0) would reject) is simply unknowable —
  // callers keep their signal-0 verdict in that case.
  assert.equal(posixProcessState(2147483647), null)
})

test('writeUpdateMarker writes a marker that readLiveUpdateMarker accepts', () => {
  const home = tmpHome('write')
  const now = 1_000_000_000_000
  writeUpdateMarker(home, 4242, { now: () => now })
  // The marker should be readable and report the same pid.
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.ok(res, 'marker written by writeUpdateMarker should be detected as live')
  assert.equal(res.pid, 4242)
  assert.ok(fs.existsSync(markerPath(home)), 'marker file should exist after write')
})

test('writeUpdateMarker preserves a live holder age across pid hand-off', () => {
  const home = tmpHome('write-handoff-age')
  const now = 1_000_000_000_000
  const startedAt = Math.floor(now / 1000) - 300

  writeMarker(home, 1010, startedAt)
  writeUpdateMarker(home, 2020, { kill: ALIVE, now: () => now })

  const [pidLine, startedLine] = fs.readFileSync(markerPath(home), 'utf8').split('\n')
  assert.equal(Number.parseInt(pidLine, 10), 2020, 'the hand-off records the new owner')
  assert.equal(Number.parseInt(startedLine, 10), startedAt, 'the holder age must not restart during hand-off')
})

test('writeUpdateMarker uses the acquisition time passed to a detached script', () => {
  const home = tmpHome('write-script-acquired-at')
  const now = 1_000_000_000_000
  const startedAt = Math.floor(now / 1000) - 300

  writeUpdateMarker(home, 2020, { now: () => now, startedAt })

  const [, startedLine] = fs.readFileSync(markerPath(home), 'utf8').split('\n')
  assert.equal(Number.parseInt(startedLine, 10), startedAt)
})

test('writeUpdateMarker is best-effort (no throw on bad path)', () => {
  // A non-existent directory should not throw.
  const badHome = path.join(os.tmpdir(), 'hermes-marker-nonexistent-' + Date.now())
  assert.doesNotThrow(() => writeUpdateMarker(badHome, 4242))
})

test('writeUpdateMarker + dead pid => self-heals on read', () => {
  const home = tmpHome('write-dead')
  writeUpdateMarker(home, 999999, { now: () => Date.now() })
  // PID 999999 is almost certainly not alive.
  const res = readLiveUpdateMarker(home, { kill: DEAD })
  assert.equal(res, null, 'a dead-pid marker from writeUpdateMarker self-heals')
  assert.ok(!fs.existsSync(markerPath(home)), 'marker file is pruned')
})

// ---------------------------------------------------------------------------
// updateHandoffConflict (#75778)
//
// A retried "Update" click must not spawn a second updater over a still-live
// one — writeUpdateMarker unconditionally overwrites the marker, so an
// unchecked hand-off clobbers the original updater's claim while it is still
// alive and mutating the checkout.
// ---------------------------------------------------------------------------

test('no marker => hand-off is not blocked', () => {
  const home = tmpHome('conflict-none')
  assert.equal(updateHandoffConflict(home, { kill: ALIVE }), null)
})

test('a different live updater already owns the marker => hand-off is blocked', () => {
  const home = tmpHome('conflict-live')
  const now = 1_000_000_000_000
  writeMarker(home, 1010, Math.floor(now / 1000) - 6) // 6s old
  const conflict = updateHandoffConflict(home, { kill: ALIVE, now: () => now })
  assert.ok(conflict, 'a live foreign updater must block a new hand-off')
  assert.equal(conflict.pid, 1010)
  assert.match(conflict.message, /already running/)
  assert.match(conflict.message, /PID 1010/)
  assert.match(conflict.message, /6s/)
})

test('a dead-pid marker does not block a hand-off (self-heals)', () => {
  const home = tmpHome('conflict-dead')
  writeMarker(home, 999999, Math.floor(Date.now() / 1000))
  assert.equal(updateHandoffConflict(home, { kill: DEAD }), null)
})

test('an expired marker does not block a hand-off (self-heals)', () => {
  const home = tmpHome('conflict-expired')
  const now = 1_000_000_000_000
  writeMarker(home, 1010, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  assert.equal(updateHandoffConflict(home, { kill: ALIVE, now: () => now }), null)
})
