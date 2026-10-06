import assert from 'node:assert/strict'
import { exec as execCallback, spawn } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { test } from 'vitest'

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
