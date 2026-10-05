import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test, vi } from 'vitest'

import { loadOrCreateInstallationId, parseInstallationId, sshOwnershipId } from './desktop-installation'
import { formatCreateTime, processCreateTimeSync } from './update-marker'

const ID_A = '11111111-1111-4111-8111-111111111111'
const ID_B = '22222222-2222-4222-8222-222222222222'

function withTempDir(run) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-installation-'))

  try {
    return run(directory)
  } finally {
    fs.rmSync(directory, { recursive: true, force: true })
  }
}

test('parseInstallationId accepts only a version-4 UUID record', () => {
  assert.equal(parseInstallationId(JSON.stringify({ installationId: ID_A.toUpperCase() })), ID_A)
  assert.equal(parseInstallationId(JSON.stringify({ installationId: 'not-an-id' })), '')
  assert.equal(parseInstallationId('{}'), '')
  assert.equal(parseInstallationId('{'), '')
})

test('loadOrCreateInstallationId persists and reuses one installation ID', () =>
  withTempDir(directory => {
    const filePath = path.join(directory, 'desktop-installation.json')
    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_A),
      ID_A
    )
    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_B),
      ID_A
    )

    if (process.platform !== 'win32') {
      assert.equal(fs.statSync(filePath).mode & 0o777, 0o600)
    }
  }))

test('loadOrCreateInstallationId tightens an existing identity file', () =>
  withTempDir(directory => {
    const filePath = path.join(directory, 'desktop-installation.json')
    fs.writeFileSync(filePath, JSON.stringify({ installationId: ID_A }), { mode: 0o644 })
    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_B),
      ID_A
    )

    if (process.platform !== 'win32') {
      assert.equal(fs.statSync(filePath).mode & 0o777, 0o600)
    }
  }))

test('loadOrCreateInstallationId replaces a malformed existing record', () =>
  withTempDir(directory => {
    const filePath = path.join(directory, 'desktop-installation.json')
    fs.writeFileSync(filePath, '{', { mode: 0o600 })
    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_A),
      ID_A
    )
    assert.equal(JSON.parse(fs.readFileSync(filePath, 'utf8')).installationId, ID_A)
  }))

test('loadOrCreateInstallationId replaces an existing symlink', () =>
  withTempDir(directory => {
    if (process.platform === 'win32') {
      return
    }

    const target = path.join(directory, 'target.json')
    const filePath = path.join(directory, 'desktop-installation.json')
    fs.writeFileSync(target, JSON.stringify({ installationId: ID_B }), { mode: 0o600 })
    fs.symlinkSync(target, filePath)
    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_A),
      ID_A
    )
    assert.equal(fs.lstatSync(filePath).isSymbolicLink(), false)
    assert.equal(JSON.parse(fs.readFileSync(target, 'utf8')).installationId, ID_B)
  }))

test('loadOrCreateInstallationId replaces a malformed destination without a repair lock', () =>
  withTempDir(directory => {
    const filePath = path.join(directory, 'desktop-installation.json')
    fs.writeFileSync(filePath, '{', { mode: 0o600 })
    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_A),
      ID_A
    )
    assert.equal(fs.existsSync(`${filePath}.lock`), false)
  }))

test('sshOwnershipId is stable, scoped, and does not disclose the UUID', () => {
  const global = sshOwnershipId(ID_A, '')
  assert.match(global, /^[0-9a-f]{32}$/)
  assert.equal(global, sshOwnershipId(ID_A, ''))
  assert.notEqual(global, sshOwnershipId(ID_A, 'worker'))
  assert.ok(!global.includes(ID_A.slice(0, 8)))
  assert.throws(() => sshOwnershipId('bad', ''))
})

// V20: a repair lock left by a crashed launch used to make every later launch
// throw at module init. It now names its holder and is reclaimed by liveness.
test('a repair lock whose holder died is reclaimed instead of bricking launch', () =>
  withTempDir(directory => {
    const filePath = path.join(directory, 'desktop-installation.json')
    // A REAL process that ran and exited: its pid now names nobody.
    const gone = spawnSync(process.execPath, ['-e', 'process.stdout.write(String(process.pid))'], { encoding: 'utf8' })
    fs.writeFileSync(`${filePath}.repair.lock`, `${gone.stdout}\n`)

    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_A),
      ID_A
    )
    assert.equal(fs.existsSync(`${filePath}.repair.lock`), false)

    // A pre-fix crash left an EMPTY lock (no owner to probe): reclaimed too.
    fs.rmSync(filePath)
    fs.writeFileSync(`${filePath}.repair.lock`, '')
    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_B),
      ID_B
    )
  }))

// Review G5: reclaiming a dead repair lock by read/compare/unlink-by-name could
// delete the fresh lock a peer created after reclaiming the same dead one (the
// peer swaps the file between our compare and our unlink). The reclaim must
// only ever remove the file it judged dead.
test('reclaiming a dead repair lock never deletes a lock a peer created meanwhile', () =>
  withTempDir(directory => {
    const filePath = path.join(directory, 'installation.json')
    const repairPath = `${filePath}.repair.lock`
    const gone = spawnSync(process.execPath, ['-p', 'process.pid'], { encoding: 'utf8' })
    const peerBody = `${process.pid}\nct:${formatCreateTime(processCreateTimeSync(process.pid)!)}\n`
    fs.writeFileSync(repairPath, `${gone.stdout.trim()}\n`)

    const realRead = fs.readFileSync
    let lockReads = 0

    const spy = vi.spyOn(fs, 'readFileSync').mockImplementation(((target: any, ...rest: any[]) => {
      const value = (realRead as any)(target, ...rest)

      // The second read of the lock (the old compare, or the read of our own
      // rename claim): a peer reclaims the dead lock and takes it, live.
      if (String(target).startsWith(repairPath) && ++lockReads === 2) {
        fs.rmSync(repairPath, { force: true })
        fs.writeFileSync(repairPath, peerBody, { flag: 'wx' })
      }

      return value
    }) as any)

    try {
      assert.throws(() => loadOrCreateInstallationId(filePath, () => ID_A), /Could not repair/)
    } finally {
      spy.mockRestore()
    }

    assert.equal(fs.readFileSync(repairPath, 'utf8'), peerBody, "the peer's live lock survives")
    assert.deepEqual(
      fs.readdirSync(directory).filter(name => name.endsWith('.reclaim')),
      [],
      'no rename claim is left behind'
    )
  }))

// Review 5411222842: the repair lock judged its holder with a second copy of
// the marker identity rule that had no own-pid check and no v1 age ceiling, so
// a lock naming our own pid (a previous incarnation reusing it) or a live
// unrelated pid with no creation time wedged every launch. It now runs the
// judge's rule (identityStateSync).
test('a repair lock naming our pid without our creation time is a previous incarnation, reclaimed', () =>
  withTempDir(directory => {
    const filePath = path.join(directory, 'desktop-installation.json')
    fs.writeFileSync(`${filePath}.repair.lock`, `${process.pid}\n`)

    assert.equal(
      loadOrCreateInstallationId(filePath, () => ID_A),
      ID_A
    )
    assert.equal(fs.existsSync(`${filePath}.repair.lock`), false)
  }))

test('a live holder with no creation time is live only within the v1 ceiling', async () => {
  const holder = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' })

  try {
    await new Promise(resolve => holder.once('spawn', resolve))

    withTempDir(directory => {
      const filePath = path.join(directory, 'desktop-installation.json')
      const repairPath = `${filePath}.repair.lock`
      fs.writeFileSync(repairPath, `${holder.pid}\n`)

      // Fresh: a live unknown identity is waited on, never reclaimed.
      assert.throws(() => loadOrCreateInstallationId(filePath, () => ID_A), /Could not repair/)
      assert.equal(fs.readFileSync(repairPath, 'utf8'), `${holder.pid}\n`)

      // Past the v1 ceiling the pid may be reused by anything: reclaimed.
      const old = Date.now() / 1000 - 2 * 1200
      fs.utimesSync(repairPath, old, old)
      assert.equal(
        loadOrCreateInstallationId(filePath, () => ID_B),
        ID_B
      )
    })
  } finally {
    holder.kill()
  }
})
