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
  connectReservationPath,
  expandRemotePath,
  fingerprintToken,
  LOCKFILE_SCHEMA_VERSION,
  lockfilePath,
  pidIsOurDashboard,
  PROTOCOL_VERSION,
  readLockfile,
  spawnLogPath,
  spawnTokenPath
} from './remote-lifecycle'
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

  const run = async (command: string): Promise<{ stdout: string; stderr: string }> =>
    exec(command, { shell, env, timeout: 10_000 })

  const ssh: Pick<SshConnection, 'exec'> = {
    exec: async (command: string): Promise<string> => (await run(command)).stdout
  }

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
    'env',
    'ps',
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

      await assert.rejects(readFile(fixture.localPath(connectReservationPath(ownershipId))), { code: 'ENOENT' })
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
        await assert.rejects(readFile(fixture.localPath(connectReservationPath(ownershipId))), { code: 'ENOENT' })
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
      assert.match((await fixture.run(fixture.command)).stdout, /^[1-9][0-9]*\n$/)
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
