import { spawn } from 'node:child_process'
import { execFileSync } from 'node:child_process'
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

import type { BrowserWindow } from 'electron'
import { afterEach, describe, expect, it } from 'vitest'

import { PlayerMediaJobs } from './player-media-jobs'

const roots: string[] = []
const hosts: PlayerMediaJobs[] = []
const owner = {} as BrowserWindow
const scope = { connectionId: 'local', profileId: 'test', sessionId: 'one' }
const other = { ...scope, sessionId: 'two' }
const target = { source: 'youtube' as const, videoId: 'abcdefghijk', canonicalUrl: 'https://www.youtube.com/watch?v=abcdefghijk', title: { text: 'A / video', trust: 'untrusted-page' as const }, capturedTimeSeconds: 12, adStatus: 'content' as const }
const python = execFileSync('which', ['python3'], { encoding: 'utf8' }).trim()
const workerEntrypoint = path.resolve(import.meta.dirname, '../public/hermestv-worker.py')

function fixture(slow = false, descendant = false, chooseDestination?: (destination: string) => Promise<string | null>) {
  const root = mkdtempSync(path.join(tmpdir(), 'hermestv-jobs-')); roots.push(root)
  const helperDir = path.join(root, 'helpers', 'venv', process.platform === 'win32' ? 'Scripts' : 'bin')
  mkdirSync(helperDir, { recursive: true })
  symlinkSync(python, path.join(helperDir, process.platform === 'win32' ? 'python.exe' : 'python'))
  writeFileSync(path.join(root, 'helpers', 'ready.json'), '{}')
  const worker = path.join(root, 'worker.py')
  const descendantMarker = path.join(root, 'descendant-stopped')
  const descendantStartedMarker = path.join(root, 'descendant-started')
  const childCode = `import signal,sys\nfrom pathlib import Path\nPath(${JSON.stringify(descendantStartedMarker)}).write_text('started')\ndef stop(*_):\n Path(${JSON.stringify(descendantMarker)}).write_text('terminated')\n raise SystemExit(0)\nsignal.signal(signal.SIGTERM, stop)\nimport time\ntime.sleep(30)`

  const descendantSetup = descendant && process.platform !== 'win32'
    ? `import os,signal,subprocess,sys\nif os.getpgrp()!=os.getpid(): raise SystemExit(90)\ndef stop_group(sig,_):\n signal.signal(sig,signal.SIG_DFL)\n os.killpg(os.getpgrp(),sig)\nsignal.signal(signal.SIGTERM,stop_group)\nsubprocess.Popen([sys.executable,'-c',${JSON.stringify(childCode)}])\n`
    : ''

  writeFileSync(worker, `import sys,json,time\n${descendantSetup}r=json.loads(sys.stdin.readline())\n${slow ? 'time.sleep(10)' : ''}\nif r['operation']=='download':\n open(r['output'],'wb').write(b'\\x00\\x00\\x00\\x18ftypisomfixture')\n print(json.dumps({'type':'result','height':360}),flush=True)\nelse:\n print(json.dumps({'type':'result','language':'en','trackKind':'automatic','segments':[{'start':0,'end':3,'text':'caption text'}]}),flush=True)\n`)
  const destination = path.join(root, 'result.mp4')
  const revealed: string[] = []
  const host = new PlayerMediaJobs({ stateDirectory: root, workerPath: worker, nodeExecutable: process.execPath, chooseDestination: async () => chooseDestination ? chooseDestination(destination) : destination, revealFile: file => revealed.push(file) }); hosts.push(host)

  return { root, host, destination, revealed, worker, descendantMarker, descendantStartedMarker }
}

afterEach(() => { for (const host of hosts.splice(0)) {host.dispose();}

 for (const root of roots.splice(0)) {rmSync(root, { recursive: true, force: true })} })

describe('local media jobs', () => {
  it('retrieves timestamped untrusted captions through a real subprocess and scopes results', async () => {
    const { host } = fixture()
    const job = await host.start(owner, scope, target, { kind: 'transcript', language: 'en' })
    await expect.poll(() => host.get(scope, job.id)?.state).toBe('completed')
    expect(host.get(scope, job.id)?.transcript).toMatchObject({ language: 'en', trackKind: 'automatic', segments: [{ startSeconds: 0, endSeconds: 3, text: { text: 'caption text', trust: 'untrusted-page' } }] })
    expect(host.get(other, job.id)).toBeNull(); expect(host.list(other)).toEqual([]); expect(host.cancel(other, job.id)).toBe(false)
  })
  it('validates actual file output and reveals it only to its owning chat', async () => {
    const { host, destination, revealed } = fixture()
    const job = await host.start(owner, scope, target, { kind: 'download', quality: '720p' })
    await expect.poll(() => host.get(scope, job.id)?.state).toBe('completed')
    expect(existsSync(destination)).toBe(true)
    expect(host.get(scope, job.id)?.artifact).toMatchObject({ fileName: 'result.mp4', format: 'mp4 (360p)' })
    expect(JSON.stringify(host.get(scope, job.id))).not.toContain(destination)
    expect(host.reveal(owner, other, job.id)).toBe(false); expect(host.reveal(owner, scope, job.id)).toBe(true); expect(revealed).toEqual([realpathSync(destination)])
    await expect(host.start(owner, scope, target, { kind: 'download' })).rejects.toThrow('destination-exists-or-invalid')
  })
  it('cancels only its own worker without creating a completed artifact', async () => {
    const { host, destination } = fixture(true)
    const job = await host.start(owner, scope, target, { kind: 'download' })
    await expect.poll(() => host.get(scope, job.id)?.state).toBe('running')
    expect(host.cancel(scope, job.id)).toBe(true)
    await expect.poll(() => host.get(scope, job.id)?.state).toBe('cancelled')
    expect(existsSync(destination)).toBe(false)
  })
  it.skipIf(process.platform === 'win32')('terminates a worker descendant through its verified private process group', async () => {
    const { host, descendantMarker, descendantStartedMarker } = fixture(true, true)
    const job = await host.start(owner, scope, target, { kind: 'transcript' })
    await expect.poll(() => host.get(scope, job.id)?.state).toBe('running')
    await expect.poll(() => existsSync(descendantStartedMarker)).toBe(true)
    expect(host.cancel(scope, job.id)).toBe(true)
    await expect.poll(() => host.get(scope, job.id)?.state).toBe('cancelled')
    await expect.poll(() => existsSync(descendantMarker)).toBe(true)
  })
  it.skipIf(process.platform === 'win32')('setup supervisor terminates its subprocess tree after exact-child SIGTERM', async () => {
    const root = mkdtempSync(path.join(tmpdir(), 'hermestv-supervisor-')); roots.push(root)
    const started = path.join(root, 'descendant-started')
    const stopped = path.join(root, 'descendant-stopped')
    const descendant = `import signal,sys\nfrom pathlib import Path\nPath(${JSON.stringify(started)}).write_text('started')\ndef stop(*_):\n Path(${JSON.stringify(stopped)}).write_text('stopped')\n raise SystemExit(0)\nsignal.signal(signal.SIGTERM,stop)\nimport time\ntime.sleep(30)`
    const command = `import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c',${JSON.stringify(descendant)}])\ntime.sleep(30)`
    const supervisor = spawn(python, ['-I', workerEntrypoint, '--supervise', JSON.stringify({ executable: python, args: ['-c', command] })], { detached: true, stdio: 'ignore' })
    const exited = new Promise<void>(resolve => supervisor.once('exit', () => resolve()))

    try {
      await expect.poll(() => existsSync(started)).toBe(true)
      expect(supervisor.kill('SIGTERM')).toBe(true)
      await exited
      await expect.poll(() => existsSync(stopped)).toBe(true)
    } finally {
      if (supervisor.exitCode === null && supervisor.signalCode === null && supervisor.pid) {
        try { process.kill(-supervisor.pid, 'SIGKILL') } catch { try { supervisor.kill('SIGKILL') } catch { /* The owned fixture is already gone. */ } }
      }
    }
  })
  it('does not start a download if its scope is unlinked while the save dialog is open', async () => {
    let releaseDialog!: (destination: string | null) => void
    let dialogOpened = false
    const dialog = new Promise<string | null>(resolve => { releaseDialog = resolve })

    const { host, destination } = fixture(false, false, () => { dialogOpened = true;

 return dialog })

    const pending = host.start(owner, scope, target, { kind: 'download' })
    await expect.poll(() => dialogOpened).toBe(true)
    host.disposeScope(scope)
    releaseDialog(destination)
    await expect(pending).rejects.toThrow('client-unavailable')
    expect(host.list(scope)).toEqual([])
    expect(existsSync(destination)).toBe(false)
  })
  it('rejects ads, forged canonical targets and concurrent same-kind jobs', async () => {
    const { host } = fixture(true)
    await expect(host.start(owner, scope, { ...target, adStatus: 'ad' }, { kind: 'download' })).rejects.toThrow('unsupported-source')
    await expect(host.start(owner, scope, { ...target, canonicalUrl: 'https://evil.example/' }, { kind: 'transcript' })).rejects.toThrow('unsupported-source')
    const job = await host.start(owner, scope, target, { kind: 'transcript' })
    await expect(host.start(owner, other, target, { kind: 'transcript' })).rejects.toThrow('media-job-busy')
    host.cancel(scope, job.id)
  })
  it('reports interrupted work after restart without automatically rerunning it', async () => {
    const original = fixture(true)
    const job = await original.host.start(owner, scope, target, { kind: 'transcript' })
    await expect.poll(() => original.host.get(scope, job.id)?.state).toBe('running')
    const restarted = fixture()
    copyFileSync(path.join(original.root, 'jobs.json'), path.join(restarted.root, 'jobs.json'))
    copyFileSync(path.join(original.root, 'jobs.key'), path.join(restarted.root, 'jobs.key'))
    const host = new PlayerMediaJobs({ stateDirectory: restarted.root, workerPath: restarted.worker, nodeExecutable: process.execPath, chooseDestination: async () => null, revealFile: () => undefined }); hosts.push(host)
    expect(host.get(scope, job.id)?.state).toBe('interrupted')
    expect(JSON.parse(readFileSync(path.join(restarted.root, 'jobs.json'), 'utf8')).records[0].job.state).toBe('running')
    original.host.cancel(scope, job.id)
    await expect.poll(() => original.host.get(scope, job.id)?.state).toBe('cancelled')
  })
  it('quarantines corrupt or forged ledgers instead of crashing or trusting paths', () => {
    const { root, worker } = fixture()
    writeFileSync(path.join(root, 'jobs.json'), '{bad json')
    const host = new PlayerMediaJobs({ stateDirectory: root, workerPath: worker, nodeExecutable: process.execPath, chooseDestination: async () => null, revealFile: () => undefined }); hosts.push(host)
    expect(host.list(scope)).toEqual([])
    expect(existsSync(path.join(root, 'jobs.json'))).toBe(false)
  })
  it('redacts local filesystem paths when another file wins the destination race', async () => {
    const { host, worker, destination } = fixture()
    writeFileSync(worker, `import sys,json\nr=json.loads(sys.stdin.readline())\nopen(${JSON.stringify(destination)},'wb').write(b'existing')\nopen(r['output'],'wb').write(b'\\x00\\x00\\x00\\x18ftypisomfixture')\nprint(json.dumps({'type':'result','height':360}),flush=True)\n`)
    const job = await host.start(owner, scope, target, { kind: 'download' })
    await expect.poll(() => host.get(scope, job.id)?.state).toBe('failed')
    expect(host.get(scope, job.id)?.error?.code).toBe('media-worker-failed')
    expect(JSON.stringify(host.get(scope, job.id))).not.toContain(destination)
    expect(readFileSync(destination, 'utf8')).toBe('existing')
  })

  it('does not reveal a completed file replaced with another file', async () => {
    const { host, destination } = fixture()
    const job = await host.start(owner, scope, target, { kind: 'download' })
    await expect.poll(() => host.get(scope, job.id)?.state).toBe('completed')
    writeFileSync(destination, 'replacement content')
    expect(host.reveal(owner, scope, job.id)).toBe(false)
  })
})
