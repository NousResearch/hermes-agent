import assert from 'node:assert/strict'
import { exec as execCallback, spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { test } from 'vitest'

import { buildRemoteUpdateObservationCommand, parseRemoteUpdateObservation } from './managed-ssh-update'
import { assertRemoteInstallUpdateClear } from './remote-lifecycle'

const exec = promisify(execCallback)

// The real client-supplied relaunch program, run locally against v2 claims (the SSH hop is the
// only thing elided): the updater writes a creation-time line 3 and the hand-off scripts tagged
// lines 4+, which older two-line readers judged UNCERTAIN forever.
test.runIf(process.platform === 'linux')(
  'POSIX relaunch gate clears a dead v2 claim but keeps one a live delegate or held mutex still guards',
  async () => {
    const home = await mkdtemp(path.join(os.tmpdir(), 'hermes-v2-marker-'))
    const marker = path.join(home, '.hermes-update-in-progress')
    const shell = (await exec('command -v bash')).stdout.trim()
    const ssh = { exec: async (command: string) => (await exec(command, { shell })).stdout }
    const exited = spawn(process.execPath, ['-e', ''])
    await new Promise(resolve => exited.once('exit', resolve))
    // The updater's v2 claim: pid, started_at, creation-time line (A2), then tagged lines.
    const deadClaim = `${exited.pid}\n${Math.floor(Date.now() / 1000)}\nct:1700000000.125\n`
    // A delegate is live only at its real creation time (the judge checks pid AND ct, so a reused
    // pid cannot impersonate it): record the spawn time, well inside the 2 s tolerance.
    const delegateCt = (Date.now() / 1000).toFixed(3)
    const delegate = spawn('sleep', ['30'], { argv0: 'hermes-update', stdio: 'ignore' })
    let mutexHolder: ReturnType<typeof spawn> | undefined

    const refused = (error: any) => error.kind === 'update-in-progress'

    try {
      await writeFile(marker, deadClaim)
      await assertRemoteInstallUpdateClear(ssh, home)
      await assert.rejects(readFile(marker), 'a confirmed-dead v2 claim is cleared, not left UNCERTAIN')

      const delegated = `${deadClaim}run:abc\ndelegate:${delegate.pid} ct:${delegateCt}\n`
      await writeFile(marker, delegated)
      await assert.rejects(() => assertRemoteInstallUpdateClear(ssh, home), refused)
      assert.equal(await readFile(marker, 'utf8'), delegated, 'a live delegate keeps the claim')

      mutexHolder = spawn(
        'python3',
        [
          '-c',
          'import fcntl,sys,time;fd=open(sys.argv[1],"a");fcntl.flock(fd,fcntl.LOCK_EX);print(1,flush=True);time.sleep(30)',
          `${marker}.lock`
        ],
        { stdio: ['ignore', 'pipe', 'ignore'] }
      )
      await new Promise(resolve => mutexHolder!.stdout!.once('data', resolve))
      await writeFile(marker, deadClaim)
      await assert.rejects(() => assertRemoteInstallUpdateClear(ssh, home), refused)
      assert.equal(await readFile(marker, 'utf8'), deadClaim, 'no delete while an updater holds the marker mutex')
    } finally {
      delegate.kill()
      mutexHolder?.kill()
      await rm(home, { force: true, recursive: true })
    }
  },
  // A held marker mutex makes the gate wait out its 10 s acquisition window before refusing.
  30_000
)

// The shapes update_lock._parse_marker accepts (tests/fixtures/update_marker_corpus.json) through
// the real relaunch gate and managed observer: a dead claim is dead whatever its padding, and the
// first in-range delegate names the live holder (the embedded readers used to say UNCERTAIN).
test.runIf(process.platform === 'linux')(
  'POSIX relaunch gate and managed observer read every corpus marker shape like update_lock',
  async () => {
    const home = await mkdtemp(path.join(os.tmpdir(), 'hermes-v2-shapes-'))
    const marker = path.join(home, '.hermes-update-in-progress')
    const ssh = { exec: async (command: string) => (await exec(command, { shell: 'sh' })).stdout }
    const target = { ssh, platform: 'Linux', hermesPath: '/opt/hermes/hermes', hermesHome: home }
    const correlation = randomUUID()

    const observe = async () =>
      parseRemoteUpdateObservation(
        await ssh.exec(buildRemoteUpdateObservationCommand(target as any, correlation)),
        correlation
      )

    const exited = spawn(process.execPath, ['-e', ''])
    await new Promise(resolve => exited.once('exit', resolve))
    const dead = exited.pid
    const now = Math.floor(Date.now() / 1000)

    const holder = spawn('python3', ['-c', 'import time;print(time.time(),flush=True);time.sleep(30)'], {
      stdio: ['ignore', 'pipe', 'ignore']
    })

    try {
      const holderCt = await new Promise<string>(resolve =>
        holder.stdout!.once('data', chunk => resolve(String(chunk).trim()))
      )

      for (const claim of [
        `\ufeff${dead}\r\n${now}\r\nct:1700000000.125\r\n`,
        ` ${dead} \t\n\t${now}  \nct:1700000000.125\n`,
        `0000${dead}\n${now}\nct:1700000000.125\n`,
        `${dead}\n18446744073709551615\nct:1700000000.125\n`
      ]) {
        await writeFile(marker, claim)
        assert.equal((await observe()).marker, 'dead', JSON.stringify(claim))
        await assertRemoteInstallUpdateClear(ssh, home)
        await assert.rejects(readFile(marker), `a dead claim is reclaimed: ${JSON.stringify(claim)}`)
      }

      const delegated = `${dead}\n${now}\nct:1700000000.125\ndelegate:4294967296 ct:1\ndelegate:${holder.pid} ct:${holderCt}\n`
      await writeFile(marker, delegated)
      assert.deepEqual(await observe().then(o => [o.marker, o.markerPid]), ['live', holder.pid])
      await assert.rejects(
        () => assertRemoteInstallUpdateClear(ssh, home),
        new RegExp(`process ${holder.pid} is still running`)
      )
      assert.equal(await readFile(marker, 'utf8'), delegated)
    } finally {
      holder.kill()
      await rm(home, { force: true, recursive: true })
    }
  },
  30_000
)
