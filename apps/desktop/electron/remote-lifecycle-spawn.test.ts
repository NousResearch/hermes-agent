import assert from 'node:assert/strict'
import { type ChildProcess, exec as execCallback, type ExecOptionsWithStringEncoding, spawn } from 'node:child_process'
import { mkdir, mkdtemp, readdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { expect, test } from 'vitest'

import {
  assertRemoteInstallUpdateClear,
  buildSpawnCommand,
  cleanupStale,
  connectSpawnLockPath,
  expandRemotePath,
  fingerprintToken,
  LOCKFILE_SCHEMA_VERSION,
  lockfilePath,
  pidIsOurDashboard,
  PROTOCOL_VERSION,
  readLockfile,
  REMOTE_SPAWN_RESERVATION_PY,
  spawnLogPath,
  spawnTokenPath
} from './remote-lifecycle'
import { shellExec, shellSshDouble } from './remote-ssh-exec.test-helpers'
import type { SshConnection } from './ssh-connection'

const exec: (command: string, options?: ExecOptionsWithStringEncoding) => Promise<{ stdout: string; stderr: string }> =
  promisify(execCallback)

const ownershipId: string = '0123456789abcdef0123456789abcdef'
const spawnNonce: string = '0123456789abcdef'
interface SpawnVariant {
  launcher: 'setsid' | 'nohup'
  owned: boolean
}

const variants: SpawnVariant[] = [
  ...(process.platform === 'linux'
    ? [
        { launcher: 'setsid' as const, owned: false },
        { launcher: 'setsid' as const, owned: true }
      ]
    : []),
  { launcher: 'nohup', owned: false },
  { launcher: 'nohup', owned: true }
]

interface ChildReport {
  pid: number
  sessionId: number
  hupIgnored: boolean
  mutexFds: number[]
}

interface SpawnLock {
  schemaVersion: number
  ownershipId: string
  spawnNonce: string
  pid: number
  port: number
  profile: string
  hermesPath: string
  hermesHome: string
  logPath: string
  tokenFingerprint: string
  protocolVersion: number
  startedAt: string
}

interface SpawnFixture {
  root: string
  shell: string
  env: NodeJS.ProcessEnv
  command: string
  lock: SpawnLock
  marker: string
  run: (command: string) => Promise<{ stdout: string; stderr: string }>
  ssh: Pick<SshConnection, 'exec'>
  localPath: (remotePath: string) => string
  dispose: () => Promise<void>
}

async function spawnFixture(launcher: 'setsid' | 'nohup', owned: boolean): Promise<SpawnFixture> {
  const root: string = await mkdtemp(path.join(os.tmpdir(), 'hermes-spawn-'))
  const bin: string = path.join(root, 'bin')
  const shell: string = (await exec('command -v sh', { shell: 'sh' })).stdout.trim()
  const localPath = (remotePath: string): string => remotePath.replace(/^~/, root)
  const env: NodeJS.ProcessEnv = { HOME: root, HERMES_HOME: path.join(root, '.hermes'), PATH: bin, LANG: 'C.UTF-8' }

  const run = async (command: string): Promise<{ stdout: string; stderr: string }> => shellExec(command, { shell, env })

  const ssh: Pick<SshConnection, 'exec'> = shellSshDouble({ shell, env })

  const hermesPath: string = path.join(root, 'fake hermes')
  const hermesHome: string = path.join(root, '.hermes')
  const marker: string = path.join(hermesHome, '.hermes-update-in-progress')

  const lock: SpawnLock = {
    schemaVersion: LOCKFILE_SCHEMA_VERSION,
    ownershipId,
    spawnNonce,
    pid: 0,
    port: 0,
    profile: 'ops__PID__',
    hermesPath,
    hermesHome,
    logPath: spawnLogPath(ownershipId, spawnNonce),
    tokenFingerprint: fingerprintToken('fixture-token'),
    protocolVersion: PROTOCOL_VERSION,
    startedAt: new Date().toISOString()
  }

  await mkdir(bin)
  await mkdir(hermesHome)
  await mkdir(path.join(root, 'reports'))

  // Restrict capability discovery, not the reported host OS. The fallback runs
  // real nohup with no setsid executable anywhere on the child's PATH.
  for (const tool of [
    'sh',
    'python3',
    'mkdir',
    'dirname',
    'rm',
    'cat',
    'sed',
    'head',
    'sleep',
    'mv',
    'ln',
    'env',
    'ps',
    // Neither `flock(1)` nor `lockf(1)` is on the payload's PATH: the
    // reservation is `fcntl.flock` in python3, which is already required by the
    // marker gate that wraps every spawn. A payload that shelled out to either
    // CLI would exit 75 here rather than reserve.
    launcher
  ]) {
    const executable: string = (await exec(`command -v ${tool}`, { shell })).stdout.trim()
    await symlink(executable, path.join(bin, tool))
  }

  await writeFile(
    hermesPath,
    `#!${path.join(bin, 'python3')}
import json,os,signal,time
from pathlib import Path
pid=os.getpid()
mutex=Path(os.environ['HERMES_HOME'])/'.hermes-update-in-progress.lock'
identity=mutex.stat()
fds=[]
for fd in range(3,256):
 try:
  candidate=os.fstat(fd)
  if (candidate.st_dev,candidate.st_ino)==(identity.st_dev,identity.st_ino):fds.append(fd)
 except OSError:pass
report={'pid':pid,'sessionId':os.getsid(0),'hupIgnored':signal.getsignal(signal.SIGHUP)==signal.SIG_IGN,'mutexFds':fds}
p=Path(os.environ['HOME'])/'reports'/str(pid)
p.with_suffix('.tmp').write_text(json.dumps(report),encoding='utf-8')
p.with_suffix('.tmp').replace(p)
time.sleep(30)
`,
    { encoding: 'utf8', mode: 0o700 }
  )

  const command: string = buildSpawnCommand(hermesPath, lock.profile, {
    hermesHome,
    logPath: lock.logPath,
    spawnNonce,
    tokenFilePath: spawnTokenPath(ownershipId, spawnNonce),
    ownershipId: owned ? ownershipId : undefined,
    reservationNonce: spawnNonce,
    lockMetadata: owned ? lock : undefined
  })

  const dispose = async (): Promise<void> => {
    for (const name of await readdir(path.join(root, 'reports'))) {
      if (!/^[1-9][0-9]*$/.test(name)) {
        continue
      }

      const report: ChildReport = JSON.parse(await readFile(path.join(root, 'reports', name), 'utf8'))

      try {
        if (await pidIsOurDashboard(ssh, report.pid, spawnNonce, hermesPath, hermesHome, ownershipId, lock.profile)) {
          process.kill(report.pid, 'SIGKILL')
        }
      } catch (error) {
        if (!(error instanceof Error && 'code' in error && error.code === 'ESRCH')) {
          throw error
        }
      }
    }

    await rm(root, { recursive: true, force: true })
  }

  return { root, shell, env, command, lock, marker, run, ssh, localPath, dispose }
}

async function reports(fixture: SpawnFixture): Promise<ChildReport[]> {
  const names: string[] = (await readdir(path.join(fixture.root, 'reports'))).filter((name: string): boolean =>
    /^[1-9][0-9]*$/.test(name)
  )

  return Promise.all(
    names.map(async (name: string): Promise<ChildReport> =>
      JSON.parse(await readFile(path.join(fixture.root, 'reports', name), 'utf8'))
    )
  )
}

/** Exit 0 only while nobody holds the spawn reservation at `$1`. */
function lockProbe(): string {
  // A lock file that does not exist holds nobody (the non-owned spawn path never
  // touches it). An existing one is free only when the kernel grants a fresh
  // exclusive lock immediately - which no live holder allows, including one this
  // probe did not create. The probe takes the same primitive as the payload,
  // `fcntl.flock`, so it needs no CLI on the host either.
  const exclusive: string =
    `python3 -c "import fcntl,os,sys;fcntl.flock(os.open(sys.argv[1],os.O_RDWR),fcntl.LOCK_EX|fcntl.LOCK_NB)" "$1"`

  return `[ -e "$1" ] || exit 0\n${exclusive}`
}

/**
 * The reservation must be free between spawns: the holder's descriptor is gone
 * once its shell exits, so a fresh exclusive lock on the same path succeeds.
 * This is the property `rm -rf` of a reservation directory used to assert.
 */
async function assertSpawnLockFree(root: string): Promise<void> {
  const lock: string = root + connectSpawnLockPath(ownershipId).replace(/^~/, '')

  await exec(`sh -c '${lockProbe()}' hermes-lock-probe '${lock}'`, { shell: 'sh' })
}

test.skipIf(process.platform === 'win32').each(variants)(
  '$launcher spawn (ownership=$owned) returns only the detached child PID and preserves ownership',
  async ({ launcher, owned }: SpawnVariant): Promise<void> => {
    const fixture: SpawnFixture = await spawnFixture(launcher, owned)

    const foreign: ChildProcess = spawn('python3', ['-c', 'import time;time.sleep(30)'], {
      env: fixture.env,
      stdio: 'ignore'
    })

    assert.ok(foreign.pid)

    try {
      const parentSession: number = Number(
        (await fixture.run('python3 -c "import os;print(os.getsid(0))"')).stdout.trim()
      )

      const results: { stdout: string; stderr: string }[] = await Promise.all(
        Array.from({ length: owned ? 2 : 1 }, (): Promise<{ stdout: string; stderr: string }> =>
          fixture.run(fixture.command)
        )
      )

      const result: { stdout: string; stderr: string } | undefined = results.find(
        (output: { stdout: string }): boolean => output.stdout !== 'EXISTING'
      )

      assert.ok(result)

      if (owned) {
        assert.equal(results.filter((output: { stdout: string }): boolean => output.stdout === 'EXISTING').length, 1)
      }

      await expect.poll(async (): Promise<number> => (await reports(fixture)).length, { timeout: 5_000 }).toBe(1)
      const [report]: ChildReport[] = await reports(fixture)
      assert.match(result.stdout, /^[1-9][0-9]*\n$/)
      assert.equal(result.stdout.trim(), String(report.pid))
      assert.equal(result.stderr, '')
      assert.deepEqual(report.mutexFds, [])
      assert.ok(process.kill(report.pid, 0), 'the reported child survives the spawning shell exiting')

      if (launcher === 'setsid') {
        assert.notEqual(report.sessionId, parentSession)
      } else {
        assert.equal(report.sessionId, parentSession)
        assert.equal(report.hupIgnored, true)
        process.kill(report.pid, 'SIGHUP')
      }

      // The child remains alive while another process acquires the exact mutex.
      await fixture.run(
        `python3 -c 'import fcntl,sys;f=open(sys.argv[1],"a");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)' ${expandRemotePath(`${fixture.marker}.lock`)}`
      )
      // The published ownership record carries the profile the spawn argv
      // actually pinned — normalized like the CLI would (#88842) — never the
      // raw roster/bridge label the fixture passed in. Derive the working lock
      // from the published record exactly the way a reconnecting client does.
      const lock: SpawnLock = { ...fixture.lock, pid: report.pid, profile: 'ops__pid__' }

      if (owned) {
        assert.deepEqual(JSON.parse(await readFile(fixture.localPath(lockfilePath(ownershipId)), 'utf8')), lock)
        assert.deepEqual(await readLockfile(fixture.ssh, ownershipId), lock)
        assert.equal((await fixture.run(fixture.command)).stdout, 'EXISTING')
        assert.equal((await reports(fixture)).length, 1)
        assert.ok(
          !(await readdir(path.dirname(fixture.localPath(lockfilePath(ownershipId))))).some((name: string): boolean =>
            name.endsWith('.tmp')
          )
        )
      }

      // The reservation's descriptor lock is released when the shell exits, and
      // the lock file itself is never unlinked (a re-created lock path is a new
      // inode another caller could lock while a holder is still live).
      await assertSpawnLockFree(fixture.root)
      await cleanupStale(fixture.ssh, ownershipId, { ...lock, pid: foreign.pid, logPath: '' })
      assert.ok(process.kill(foreign.pid, 0), 'a foreign process is never killed by cleanup')
      await cleanupStale(fixture.ssh, ownershipId, lock)
      assert.ok(process.kill(foreign.pid, 0), 'owned cleanup leaves the unrelated child alive')
      assert.throws((): boolean => process.kill(report.pid, 0), { code: 'ESRCH' })
      await assert.rejects(readFile(fixture.localPath(lockfilePath(ownershipId))), { code: 'ENOENT' })
    } finally {
      foreign.kill('SIGKILL')
      await fixture.dispose()
    }
  },
  20_000
)

test.skipIf(process.platform === 'win32').each(variants)(
  '$launcher spawn (ownership=$owned) refuses live or malformed update markers without reserving an owner',
  async ({ launcher, owned }: SpawnVariant): Promise<void> => {
    const fixture: SpawnFixture = await spawnFixture(launcher, owned)

    try {
      for (const marker of [`${process.pid}\n${Math.floor(Date.now() / 1000)}\n`, 'not-a-pid\n']) {
        await writeFile(fixture.marker, marker, 'utf8')
        await assert.rejects(fixture.run(fixture.command), { code: 75, stdout: '' })
        assert.deepEqual(await reports(fixture), [])
        // A refused spawn leaves the reservation free, not deleted: nothing
        // about the mutex is state another caller has to reclaim.
        await assertSpawnLockFree(fixture.root)
        await assert.rejects(readFile(fixture.localPath(lockfilePath(ownershipId))), { code: 'ENOENT' })
      }
    } finally {
      await fixture.dispose()
    }
  },
  20_000
)

/** A python3 child that prints one line (after its setup) and stays alive. */
async function pythonChild(fixture: SpawnFixture, script: string, ...args: string[]): Promise<[ChildProcess, string]> {
  const child: ChildProcess = spawn('python3', ['-c', script, ...args], {
    env: fixture.env,
    stdio: ['ignore', 'pipe', 'inherit']
  })

  const line: string = await new Promise((resolve, reject) => {
    child.stdout!.once('data', (chunk: Buffer) => resolve(chunk.toString().trim()))
    child.once('exit', code => reject(new Error(`python child exited ${code}`)))
  })

  return [child, line]
}

async function deadPid(): Promise<number> {
  const child: ChildProcess = spawn('true')
  await new Promise(resolve => child.once('exit', resolve))

  return child.pid!
}

/** Wait for a file a gated racer creates. The gate, never a sleep, orders the race. */
async function waitFor(file: string, timeoutMs: number): Promise<boolean> {
  const deadline: number = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    try {
      await readFile(file)

      return true
    } catch {
      await new Promise(resolve => setTimeout(resolve, 20))
    }
  }

  return false
}

test.skipIf(process.platform === 'win32')(
  'v2 claims are judged by pid + creation time: a dead claim is cleared, a live delegate refuses',
  async (): Promise<void> => {
    const fixture: SpawnFixture = await spawnFixture(variants[0].launcher, false)
    const [delegate, ct] = await pythonChild(fixture, 'import time;print(time.time(),flush=True);time.sleep(30)')
    const dead: number = await deadPid()
    const now: number = Math.floor(Date.now() / 1000)

    try {
      await writeFile(fixture.marker, `${dead}\n${now}\nct:${ct}\ndelegate:${dead} ct:${ct}\nrun:desk-1\n`)
      await assertRemoteInstallUpdateClear(fixture.ssh, fixture.lock.hermesHome)
      await assert.rejects(readFile(fixture.marker), { code: 'ENOENT' })

      const live: string = `${dead}\n${now}\nct:${ct}\ndelegate:${delegate.pid} ct:${ct}\n`
      await writeFile(fixture.marker, live)
      await assert.rejects(
        assertRemoteInstallUpdateClear(fixture.ssh, fixture.lock.hermesHome),
        (error: any) => error.kind === 'update-in-progress' && error.message.includes(`process ${delegate.pid}`)
      )
      await assert.rejects(fixture.run(fixture.command), { code: 75, stdout: '' })
      assert.equal(await readFile(fixture.marker, 'utf8'), live)

      // The live pid at another incarnation (creation time 60 s off) is a dead claim.
      await writeFile(fixture.marker, `${delegate.pid}\n${now}\nct:${Number(ct) - 60}\n`)
      const spawned: string = (await fixture.run(fixture.command)).stdout
      assert.match(spawned, /^[1-9][0-9]*\n$/)
      // Kill the child it detached now: left running, it writes its report into the fixture
      // while dispose's rm removes the directory (ENOTEMPTY), then sleeps on for 30 s.
      process.kill(Number(spawned), 'SIGKILL')
      await assert.rejects(readFile(fixture.marker), { code: 'ENOENT' })
    } finally {
      delegate.kill('SIGKILL')
      await fixture.dispose()
    }
  },
  20_000
)

test.skipIf(process.platform === 'win32')(
  "spawn waits on the updaters' <marker>.lock, so a claim made under it is never raced",
  async (): Promise<void> => {
    const fixture: SpawnFixture = await spawnFixture(variants[0].launcher, false)

    // An updater: hold <marker>.lock (update_lock.py / marker.sh flock), then
    // write its v2 claim and release, staying alive as the live owner.
    const [updater] = await pythonChild(
      fixture,
      `import fcntl,os,sys,time
t=time.time()
fd=os.open(sys.argv[1]+'.lock',os.O_RDWR|os.O_CREAT)
fcntl.flock(fd,fcntl.LOCK_EX)
print('held',flush=True)
time.sleep(0.5)
open(sys.argv[1],'w').write('%d\\n%d\\nct:%.3f\\n'%(os.getpid(),int(t),t))
time.sleep(0.5)
fcntl.flock(fd,fcntl.LOCK_UN)
time.sleep(30)`,
      fixture.marker
    )

    try {
      await assert.rejects(fixture.run(fixture.command), { code: 75, stdout: '' })
      assert.deepEqual(await reports(fixture), [])
      assert.match(await readFile(fixture.marker, 'utf8'), new RegExp(`^${updater.pid}\\n`))
    } finally {
      updater.kill('SIGKILL')
      await fixture.dispose()
    }
  },
  20_000
)

interface ReservationRace {
  root: string
  home: string
  /** The reservation block `buildSpawnCommand` composes, byte for byte. */
  payload: string
  /** The one file the reservation locks, for the life of the caller. */
  spawnLock: string
}

/**
 * Capture the reservation payload the app really composes: the argv the remote
 * shell hands to python3 is recorded and the payload is argv[3] of that call.
 * Every race below runs those bytes verbatim instead of a copy of them.
 */
async function reservationRace(): Promise<ReservationRace> {
  const root: string = await mkdtemp(path.join(os.tmpdir(), 'hermes-reservation-race-'))
  const home: string = path.join(root, 'home')
  const argvShim: string = path.join(root, 'argv-shim')
  const argvFile: string = path.join(root, 'argv.bin')

  await mkdir(home)
  await mkdir(argvShim)

  await writeFile(path.join(argvShim, 'python3'), `#!/bin/sh\nprintf '%s\\0' "$@" > '${argvFile}'\n`, {
    mode: 0o755
  })

  const command: string = buildSpawnCommand('/x/hermes', 'work', {
    hermesHome: '~/.hermes',
    lockMetadata: { ownershipId, spawnNonce },
    logPath: spawnLogPath(ownershipId, spawnNonce),
    ownershipId,
    reservationNonce: spawnNonce,
    spawnNonce,
    tokenFilePath: spawnTokenPath(ownershipId, spawnNonce)
  })

  await exec(command, { env: { ...process.env, HOME: home, PATH: `${argvShim}:${process.env.PATH}` }, shell: 'sh' })

  const payload: string = (await readFile(argvFile, 'utf8')).split('\0')[3]
  const spawnLock: string = home + connectSpawnLockPath(ownershipId).replace(/^~/, '')

  return { root, home, payload, spawnLock }
}

/** shell-single-quote, the same escaping `remote-lifecycle.ts`'s shq() does. */
function shq(value: string): string {
  return `'${String(value).replace(/'/g, `'\\''`)}'`
}

/** Everything the payload does before it takes the reservation. */
function reservationPrefix(payload: string): string {
  // The reservation starts at whichever primitive the payload reserves with, so
  // the same racer runs the shipped bytes on both sides of the change.
  const starts: number[] = ['python3 -c', 'exec 9>>', 'reservation='].map((needle: string): number =>
    payload.indexOf(needle)
  )

  const reserve: number = Math.min(...starts.filter((at: number): boolean => at > 0))

  assert.ok(reserve > 0, 'the reservation moved; teach this test its new shape')

  return payload.slice(0, reserve)
}

/**
 * A caller that runs the shipped reservation program with a racer's critical
 * section instead of the spawn: the lock, its 30 s bound and its fail-closed
 * exit are the bytes the app ships, and only the body differs.
 */
function reservationRacer(payload: string, spawnLock: string, body: string): string {
  return (
    `${reservationPrefix(payload)}` +
    `python3 -c ${shq(REMOTE_SPAWN_RESERVATION_PY)} ${shq(spawnLock)} '' ${shq(`sh ${shq(body)}`)}`
  )
}

/** The lock probe as a script the racers can run from inside their hold. */
async function lockProbeScript(root: string): Promise<string> {
  const probe: string = path.join(root, 'probe.sh')

  await writeFile(probe, `#!/bin/sh\n${lockProbe()}\n`, { mode: 0o755 })

  return probe
}

/**
 * A caller that runs the shipped reservation program, then behaves like the
 * holder: it records the reservation it took, proves the mutex is genuinely
 * held (an independent non-blocking lock on the same path must fail while it is
 * inside), holds long enough for a second holder to be observable, and records
 * its entry.
 */
async function holderRacer(root: string, payload: string, spawnLock: string, name = 'race'): Promise<string> {
  const race: string = path.join(root, `${name}.sh`)
  const probe: string = await lockProbeScript(root)
  const body: string = path.join(root, `${name}-body.sh`)

  await writeFile(
    body,
    `printf '%s\\n' "$$" >> "$MARK_DIR/entries"; ` +
      `if [ -e "$MARK_DIR/held" ]; then : > "$MARK_DIR/double"; fi; ` +
      `: > "$MARK_DIR/held"; ` +
      // A lock probe that succeeds while this caller is inside is proof the
      // mutex is not exclusive. It only says something when the lock file
      // exists at all: the non-owned path never creates one.
      `if [ -e '${spawnLock}' ] && sh '${probe}' '${spawnLock}' 2>/dev/null; then : > "$MARK_DIR/unheld"; fi; ` +
      `sleep 0.15; ` +
      `command -p rm -f "$MARK_DIR/held"; `,
    { mode: 0o755 }
  )

  await writeFile(race, reservationRacer(payload, spawnLock, body), { mode: 0o755 })

  return race
}

/** Run one round of `callers` real callers and fail on any overlap/unheld marker. */
async function raceRound(race: string, home: string, mark: string, callers: number, label: string): Promise<string[]> {
  await rm(mark, { force: true, recursive: true })
  await mkdir(mark)

  await Promise.all(
    Array.from({ length: callers }, (): Promise<{ stdout: string }> =>
      exec(race, { env: { ...process.env, HOME: home, MARK_DIR: mark }, shell: 'sh' })
    )
  )

  for (const name of ['double', 'unheld']) {
    await assert.rejects(
      readFile(path.join(mark, name)),
      { code: 'ENOENT' },
      `${label}: ${
        name === 'double'
          ? 'two callers held the ownership reservation at once'
          : 'a lock probe succeeded while a live holder was inside the reservation'
      }`
    )
  }

  return (await readFile(path.join(mark, 'entries'), 'utf8')).trim().split('\n')
}

// The reservation must never be decided by the exit status of `mkdir`. Ubuntu
// 26.04 ships uutils coreutils 0.10.0, whose `mkdir` can report a create it lost
// as a success; with mkdir as the mutex the loser never took the "already
// reserved" path, two backends spawned for one ownership id, and a reaper could
// delete a reservation whose owner record was not written yet. The mutex is a
// kernel lock now, so a `mkdir` that lies cannot admit a second holder — and
// this test stays as the guard against ever going back to a directory create.
test.skipIf(process.platform === 'win32')(
  'the ownership reservation admits one holder when a mkdir reports a create it lost',
  async (): Promise<void> => {
    const { root, home, payload, spawnLock }: ReservationRace = await reservationRace()

    try {
      const round: string = path.join(root, 'round')
      const raceShim: string = path.join(root, 'race-shim')
      const race: string = path.join(root, 'race.sh')

      await mkdir(raceShim)

      await writeFile(
        path.join(raceShim, 'mkdir'),
        '#!/bin/sh\n' +
          // `-p` is idempotent container creation, not an ownership claim; every
          // bare `mkdir <path>` reports success whether or not it created it.
          'for a in "$@"; do case "$a" in -p|--parents) command -p mkdir "$@" ;; esac; done\n' +
          'command -p mkdir "$@" 2>/dev/null\n' +
          'exit 0\n',
        { mode: 0o755 }
      )

      // Racer protocol: whoever holds the reservation sleeps; a second holder
      // inside that window is the double-win the mkdir mutex allowed, and the
      // one the reaper's reservation delete raced against.
      const body: string = path.join(root, 'race-body.sh')

      await writeFile(
        body,
        `if [ -e "$MARK_DIR/held" ]; then : > "$MARK_DIR/double"; fi; ` +
          `: > "$MARK_DIR/held"; sleep 0.3; rm -f "$MARK_DIR/held"; `,
        { mode: 0o755 }
      )
      await writeFile(race, reservationRacer(payload, spawnLock, body), { mode: 0o755 })

      const env: NodeJS.ProcessEnv = { ...process.env, HOME: home, PATH: `${raceShim}:${process.env.PATH}` }

      for (let index = 0; index < 8; index++) {
        await rm(round, { force: true, recursive: true })
        await mkdir(round)

        await Promise.all(
          [1, 2].map((): Promise<{ stdout: string }> => exec(race, { env: { ...env, MARK_DIR: round }, shell: 'sh' }))
        )

        await assert.rejects(
          readFile(path.join(round, 'double')),
          { code: 'ENOENT' },
          `round ${index + 1}: two spawns held the ownership reservation at once`
        )
      }
    } finally {
      await rm(root, { force: true, recursive: true })
    }
  },
  60_000
)

// The reservation is one kernel lock on one file, so every caller that takes it
// is the only holder until it exits. K real callers on one ownership id take it
// in turn: no owner record, no ~2 s grace for an unpublished record, and no
// reclaim step at all.
test.skipIf(process.platform === 'win32')(
  'K callers on one ownership id take the spawn reservation in turn',
  async (): Promise<void> => {
    const { root, home, payload, spawnLock }: ReservationRace = await reservationRace()
    const race: string = await holderRacer(root, payload, spawnLock)

    try {
      for (let index = 0; index < 4; index++) {
        const entries: string[] = await raceRound(race, home, path.join(root, `turn-${index}`), 8, `round ${index + 1}`)

        assert.equal(entries.length, 8, `round ${index + 1}: every caller should have taken the reservation in turn`)
        assert.equal(new Set(entries).size, 8, `round ${index + 1}: a caller took the reservation twice`)
      }
    } finally {
      await rm(root, { force: true, recursive: true })
    }
  },
  120_000
)

// SIGKILL cannot be trapped, and the kernel drops a descriptor lock when its
// holder dies. A spawn killed between taking the reservation and publishing its
// lockfile therefore leaves no state behind: the next caller takes the same lock
// at once, with no pid to judge, no grace to wait out and nothing to reclaim.
test.skipIf(process.platform === 'win32')(
  'a holder killed with SIGKILL releases the reservation at once and leaves nothing to reclaim',
  async (): Promise<void> => {
    const { root, home, payload, spawnLock }: ReservationRace = await reservationRace()
    const race: string = await holderRacer(root, payload, spawnLock, 'killed')
    const mark: string = path.join(root, 'killed-mark')

    await mkdir(mark)

    try {
      const child: ChildProcess = spawn('sh', [race], {
        env: { ...process.env, HOME: home, MARK_DIR: mark },
        stdio: 'ignore'
      })

      assert.ok(await waitFor(path.join(mark, 'held'), 5_000), 'the killed caller never took the reservation')
      child.kill('SIGKILL')
      await new Promise(resolve => child.once('exit', resolve))

      const next: string = await holderRacer(root, payload, spawnLock, 'next')
      const started: number = Date.now()
      const entries: string[] = await raceRound(next, home, path.join(root, 'next'), 4, 'after the kill')

      assert.ok(Date.now() - started < 5_000, 'the next callers waited on a reservation whose holder was SIGKILLed')
      assert.equal(entries.length, 4, 'the killed holder must not wedge the next callers')
      assert.equal(new Set(entries).size, 4, 'a next caller took the reservation twice')
    } finally {
      await rm(root, { force: true, recursive: true })
    }
  },
  60_000
)

// THE REGRESSION for the reclaim-marker race: two waiters read the same dead
// marker, one removes it, a reaper publishes its own marker, and the delayed
// waiter's unchecked `rm -f` deletes the marker that reaper is holding — so a
// second reaper enters the section while the first one is still inside it.
//
// The interleaving is forced with gates, never with a timing assumption: the
// delayed waiter's `rm` stops before it deletes, the first reaper's `mv` stops
// inside the section, and the third caller starts only after the delete landed.
// Those gates exist only while the payload still has the marker protocol; once
// the marker is gone the seeded files are inert, no gate fires, and the three
// callers take the one lock in turn. Either way the test fails if two callers
// are ever inside the reservation at once.
test.skipIf(process.platform === 'win32')(
  'a waiter that judged a stale reclaim marker cannot admit a second reaper',
  async (): Promise<void> => {
    const { root, home, payload, spawnLock }: ReservationRace = await reservationRace()
    const mark: string = path.join(root, 'marker-race')
    const legacy: string = path.join(home, '.hermes', 'desktop-ssh', ownershipId, '.connect.lock')
    const dead: number = await deadPid()
    const roles: string[] = ['w', 'r', 'l']
    const shims: Record<string, string> = {}

    await mkdir(mark)
    await mkdir(legacy, { recursive: true })
    // The state a marker-era reaper judges: a dead owner record and a dead
    // reaper's marker, both from a real, reaped pid.
    await writeFile(path.join(legacy, 'owner'), `${dead}:${spawnNonce}`)
    await writeFile(path.join(legacy, '.reclaim'), String(dead))

    for (const role of roles) {
      shims[role] = path.join(root, `${role}-shim`)
      await mkdir(shims[role])
    }

    const mvShim: string = [
      '#!/bin/sh',
      ': > "$MARK_DIR/${ROLE}_inside"',
      'for f in "$MARK_DIR"/*_inside; do',
      '  [ -e "$f" ] || continue',
      '  case "$f" in *"/${ROLE}_inside") continue;; esac',
      '  o=${f##*/}; o=${o%_inside}',
      '  [ -e "$MARK_DIR/${o}_done" ] || : > "$MARK_DIR/overlap"',
      'done',
      // The first reaper holds the section open until the third caller is in, so
      // the overlap is a fact about the section and not about scheduling.
      'if [ "${ROLE}" = r ]; then i=0; while [ ! -e "$MARK_DIR/l_inside" ] && [ "$i" -lt 500 ]; do sleep 0.02; i=$((i+1)); done; fi',
      'command -p mv "$@"',
      's=$?; : > "$MARK_DIR/${ROLE}_done"; exit $s',
      ''
    ].join('\n')

    const rmShim: string = [
      '#!/bin/sh',
      'last=""; for a in "$@"; do last=$a; done',
      'case "$last" in',
      '  */.reclaim)',
      // The waiter has already read the dead pid into its own variable here.
      // Hold it until the reaper that replaced the marker is inside.
      '    : > "$MARK_DIR/w_at_rm"',
      '    i=0; while [ ! -e "$MARK_DIR/r_inside" ] && [ "$i" -lt 500 ]; do sleep 0.02; i=$((i+1)); done',
      '    command -p rm "$@"; s=$?; : > "$MARK_DIR/w_done"; exit $s;;',
      'esac',
      'command -p rm "$@"',
      ''
    ].join('\n')

    for (const role of roles) {
      await writeFile(path.join(shims[role], 'mv'), mvShim, { mode: 0o755 })
    }

    await writeFile(path.join(shims.w, 'rm'), rmShim, { mode: 0o755 })

    const races: Record<string, string> = {}

    for (const role of roles) {
      races[role] = await holderRacer(root, payload, spawnLock, role)
    }

    const run = (role: string): Promise<{ stdout: string }> =>
      exec(races[role], {
        env: { ...process.env, HOME: home, MARK_DIR: mark, ROLE: role, PATH: `${shims[role]}:${process.env.PATH}` },
        shell: 'sh'
      })

    try {
      if (payload.includes('reclaim_marker')) {
        const waiter: Promise<{ stdout: string }> = run('w')

        assert.ok(await waitFor(path.join(mark, 'w_at_rm'), 10_000), 'the delayed waiter never reached the marker delete')

        const reaper: Promise<{ stdout: string }> = run('r')

        assert.ok(
          await waitFor(path.join(mark, 'r_inside'), 10_000),
          'the first reaper never entered the reclaim section'
        )
        assert.ok(await waitFor(path.join(mark, 'w_done'), 10_000), 'the delayed waiter never deleted the marker')

        const late: Promise<{ stdout: string }> = run('l')

        await Promise.all([waiter, reaper, late])
      } else {
        // The marker does not exist in this design: the seeded files are inert
        // and the three callers contend for the one lock directly.
        await Promise.all(roles.map(run))
      }

      await assert.rejects(
        readFile(path.join(mark, 'overlap')),
        { code: 'ENOENT' },
        'two callers were inside the reclaim section at once: a delayed waiter deleted the marker a live reaper held'
      )
      await assert.rejects(readFile(path.join(mark, 'double')), { code: 'ENOENT' })
      await assert.rejects(readFile(path.join(mark, 'unheld')), { code: 'ENOENT' })
    } finally {
      await rm(root, { force: true, recursive: true })
    }
  },
  120_000
)

