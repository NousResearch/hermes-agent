import { type ChildProcess, spawn } from 'node:child_process'
import { createHmac, randomBytes, randomUUID, timingSafeEqual } from 'node:crypto'
import { closeSync, constants, existsSync, linkSync, lstatSync, mkdirSync, openSync, readFileSync, readSync, realpathSync, renameSync, rmSync, statSync, unlinkSync, writeFileSync } from 'node:fs'
import path from 'node:path'

import type { BrowserWindow } from 'electron'

import type { PlayerAgentScope, PlayerMediaJob, PlayerMediaJobCapabilities, PlayerMediaJobStartOptions, PlayerMediaTarget } from '../src/lib/player-context-contract'

interface MediaJobsOptions {
  stateDirectory: string
  workerPath: string
  nodeExecutable: string
  chooseDestination(owner: BrowserWindow, defaultFileName: string): Promise<string | null>
  revealFile(file: string): void
  choosePythonExecutable?(owner: BrowserWindow): Promise<string | null>
}
interface StoredJob { scope: PlayerAgentScope; job: PlayerMediaJob; destination: string | null; workDirectory: string | null; artifactIdentity?: string }
const terminal = new Set(['completed', 'failed', 'cancelled', 'interrupted'])
const sameScope = (a: PlayerAgentScope, b: PlayerAgentScope) => a.connectionId === b.connectionId && a.profileId === b.profileId && a.sessionId === b.sessionId

const fail = (code: string): never => { throw new Error(code) }
const stamp = () => new Date().toISOString()

// Control characters are forbidden in a suggested filename.
const safeMediaTitle = (target: PlayerMediaTarget): string => (target.title?.text ?? target.videoId)
  // eslint-disable-next-line no-control-regex
  .replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').replace(/[. ]+$/, '').slice(0, 80) || target.videoId

/** Local workers receive a captured watch ID, never service cookies, page code or arbitrary URLs. */
export class PlayerMediaJobs {
  private records = new Map<string, StoredJob>()
  private children = new Map<string, ChildProcess>()
  private settingUp = false
  private disposed = false
  private starting = new Set<string>()
  private setupChildren = new Set<ChildProcess>()
  private scopeEpochs = new Map<string, number>()
  private python: string
  private readyFile: string
  private ledger: string
  private ledgerKey: string

  constructor(private options: MediaJobsOptions) {
    this.python = path.join(options.stateDirectory, 'helpers', 'venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')
    this.readyFile = path.join(options.stateDirectory, 'helpers', 'ready.json')
    this.ledger = path.join(options.stateDirectory, 'jobs.json')
    this.ledgerKey = path.join(options.stateDirectory, 'jobs.key')

    if (existsSync(this.ledger)) {
      try {
      if (!lstatSync(this.ledger).isFile() || lstatSync(this.ledger).isSymbolicLink() || statSync(this.ledger).size > 4 * 1024 * 1024) {fail('invalid-ledger')}
      const document = JSON.parse(readFileSync(this.ledger, 'utf8'))
      const key = this.readKey(false)
      const expected = createHmac('sha256', key).update(JSON.stringify(document.records)).digest('hex')

      if (document.schema !== 1 || typeof document.mac !== 'string' || document.mac.length !== expected.length || !timingSafeEqual(Buffer.from(document.mac), Buffer.from(expected))) {fail('invalid-ledger')}
      const saved: unknown = document.records

      if (Array.isArray(saved)) {for (const record of saved as StoredJob[]) {
        if (!record?.job?.id || !/^[a-f0-9-]{36}$/.test(record.job.id) || !record.scope || !['connectionId', 'profileId', 'sessionId'].every(k => typeof (record.scope as unknown as Record<string, unknown>)[k] === 'string') || !['transcript', 'download'].includes(record.job.kind)) {continue}

        if (!terminal.has(record.job.state)) {
          record.job.state = 'interrupted'
          record.job.error = { code: 'interrupted', message: 'The app exited before this job finished. Start a new request to retry.' }
          record.job.updatedAt = stamp()
        }

        this.records.set(record.job.id, record)
      }}
      } catch {
        // A corrupt local ledger must not prevent Hermes startup or rerun old work.
        renameSync(this.ledger, `${this.ledger}.invalid-${randomUUID()}`)
      }
    }
  }

  capabilities(): PlayerMediaJobCapabilities {
    return { available: !this.settingUp && existsSync(this.python) && existsSync(this.readyFile) && existsSync(this.options.workerPath), reason: this.settingUp ? 'media-setup-running' : existsSync(this.python) && existsSync(this.readyFile) && existsSync(this.options.workerPath) ? null : 'media-helpers-not-installed' }
  }

  async setup(owner: BrowserWindow): Promise<PlayerMediaJobCapabilities> {
    if (this.settingUp || this.disposed) {return this.capabilities()}

    if (this.children.size || this.starting.size) {return { available: false, reason: 'media-job-busy' }}
    this.settingUp = true

    try {
      const systemPython = await this.selectPython(owner)

      if (!systemPython || this.disposed) {fail('python-3.10-required')}
      await this.installHelpers(systemPython)

      return { available: true, reason: null }
    } catch {
      return { available: false, reason: 'media-setup-failed' }
    } finally { this.settingUp = false }
  }

  private async selectPython(owner: BrowserWindow): Promise<string | null> {
    for (const candidate of this.pythonCandidates()) {
      if (await this.supportsPython(candidate)) {return candidate}
    }

    const selected = await this.options.choosePythonExecutable?.(owner)

    if (!selected) {return null}

    if (!await this.supportsPython(selected)) {fail('python-3.10-required')}

    return selected
  }

  private pythonCandidates(): string[] {
    const names = process.platform === 'win32' ? ['python.exe', 'python3.exe'] : ['python3', 'python3.13', 'python3.12', 'python3.11']
    const candidates = new Set<string>()

    for (const directory of (process.env.PATH ?? '').split(path.delimiter)) {
      if (!directory || /WindowsApps/i.test(directory)) {continue}

      for (const name of names) {
        const candidate = path.join(directory, name)

        if (existsSync(candidate)) {candidates.add(candidate)}
      }
    }

    return [...candidates]
  }

  private async supportsPython(executable: string): Promise<boolean> {
    try {
      await this.run(executable, ['-I', '-c', 'import sys; assert (3,10) <= sys.version_info < (3,15)'], 10_000)

      return true
    } catch {
      return false
    }
  }

  private async installHelpers(systemPython: string): Promise<void> {
    mkdirSync(this.options.stateDirectory, { recursive: true, mode: 0o700 })
    mkdirSync(path.dirname(this.readyFile), { recursive: true, mode: 0o700 })
    this.validateHelperDirectories()

    if (existsSync(this.readyFile)) {unlinkSync(this.readyFile)}
    await this.run(systemPython, ['-I', '-m', 'venv', '--clear', path.dirname(path.dirname(this.python))], 120_000)
    await this.run(this.python, ['-I', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'install', '--require-virtualenv', '--only-binary=:all:', 'youtube-transcript-api==1.2.4', 'yt-dlp[default]==2026.10.21'], 300_000)
    await this.run(this.python, ['-I', '-c', 'import youtube_transcript_api, yt_dlp'], 15_000)
    writeFileSync(this.readyFile, JSON.stringify({ schema: 1, installedAt: stamp(), transcript: '1.2.4', ytDlp: '2026.10.21' }), { mode: 0o600 })
  }

  private validateHelperDirectories(): void {
    const directories = [path.dirname(this.readyFile), path.dirname(path.dirname(this.python))]

    if (directories.some(directory => existsSync(directory) && lstatSync(directory).isSymbolicLink())) {fail('invalid-helper-directory')}
  }

  async start(owner: BrowserWindow, scope: PlayerAgentScope, target: PlayerMediaTarget, options: PlayerMediaJobStartOptions): Promise<PlayerMediaJob> {
    const { scopeKey, scopeEpoch } = this.validateStartRequest(scope, target, options)
    const id = randomUUID()
    const extension = options.quality === 'audio' ? '.m4a' : '.mp4'
    this.starting.add(options.kind)

    try {
      const destination = await this.chooseDestination(owner, scopeKey, scopeEpoch, target, options, extension)
      const workDirectory = destination ? path.join(path.dirname(destination), `.hermestv-${id}`) : null

      if (workDirectory) {mkdirSync(workDirectory, { mode: 0o700 })}

      return this.createQueuedJob(scope, target, options, id, destination, workDirectory)
    } finally { this.starting.delete(options.kind) }
  }

  private validateStartRequest(scope: PlayerAgentScope, target: PlayerMediaTarget, options: PlayerMediaJobStartOptions): { scopeKey: string; scopeEpoch: number } {
    const scopeKey = JSON.stringify([scope.connectionId, scope.profileId, scope.sessionId])
    const scopeEpoch = this.scopeEpochs.get(scopeKey) ?? 0

    if (this.disposed || !this.capabilities().available) {fail('media-helpers-not-installed')}
    this.validateCapturedTarget(target)
    this.validateJobKind(options.kind)
    this.validateJobOptions(options)

    return { scopeKey, scopeEpoch }
  }

  private validateCapturedTarget(target: PlayerMediaTarget): void {
    if (target.source !== 'youtube' || !/^[A-Za-z0-9_-]{11}$/.test(target.videoId) || target.canonicalUrl !== `https://www.youtube.com/watch?v=${target.videoId}` || target.adStatus === 'ad') {fail('unsupported-source')}
  }

  private validateJobKind(kind: PlayerMediaJobStartOptions['kind']): void {
    if (!['transcript', 'download'].includes(kind)) {fail('unsupported-operation')}

    if (this.starting.has(kind) || [...this.records.values()].some(record => record.job.kind === kind && !terminal.has(record.job.state))) {fail('media-job-busy')}
  }

  private validateJobOptions(options: PlayerMediaJobStartOptions): void {
    if (options.language && !/^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$/.test(options.language)) {fail('invalid-language')}
    this.validateRange(options)

    if (options.quality && !['best', '720p', '1080p', 'audio'].includes(options.quality)) {fail('invalid-quality')}
  }

  private validateRange(options: PlayerMediaJobStartOptions): void {
    for (const value of [options.range?.startSeconds, options.range?.endSeconds]) {
      if (value !== undefined && (!Number.isFinite(value) || value < 0 || value > 86400)) {fail('invalid-range')}
    }

    if (options.range?.startSeconds !== undefined && options.range.endSeconds !== undefined && options.range.endSeconds < options.range.startSeconds) {fail('invalid-range')}
  }

  private async chooseDestination(owner: BrowserWindow, scopeKey: string, scopeEpoch: number, target: PlayerMediaTarget, options: PlayerMediaJobStartOptions, extension: string): Promise<string | null> {
    if (options.kind !== 'download') {return null}
    const chosen = await this.options.chooseDestination(owner, `HermesTV-${safeMediaTitle(target)}${extension}`)

    if (!chosen) {fail('destination-cancelled')}

    if (this.disposed || (this.scopeEpochs.get(scopeKey) ?? 0) !== scopeEpoch) {fail('client-unavailable')}

    if (!path.isAbsolute(chosen)) {fail('invalid-destination')}
    const destination = path.join(realpathSync(path.dirname(chosen)), path.basename(chosen))

    if (path.extname(destination).toLowerCase() !== extension || existsSync(destination)) {fail('destination-exists-or-invalid')}

    return destination
  }

  private createQueuedJob(scope: PlayerAgentScope, target: PlayerMediaTarget, options: PlayerMediaJobStartOptions, id: string, destination: string | null, workDirectory: string | null): PlayerMediaJob {
    const job: PlayerMediaJob = { id, kind: options.kind, state: 'queued', progress: null, createdAt: stamp(), updatedAt: stamp(), source: structuredClone(target), error: null, transcript: null, artifact: null }
    const record = { scope: { ...scope }, job, destination, workDirectory }
    this.records.set(id, record)
    this.persist()
    queueMicrotask(() => this.execute(record, options))

    return structuredClone(job)
  }

  list(scope: PlayerAgentScope): PlayerMediaJob[] { return [...this.records.values()].filter(record => sameScope(record.scope, scope)).map(record => structuredClone(record.job)) }
  get(scope: PlayerAgentScope, id: string): PlayerMediaJob | null { const record = this.records.get(id);

 return record && sameScope(record.scope, scope) ? structuredClone(record.job) : null }
  cancel(scope: PlayerAgentScope, id: string): boolean {
    const record = this.records.get(id)

    if (!record || !sameScope(record.scope, scope)) {return false}

    if (terminal.has(record.job.state)) {return true}
    record.job.state = 'cancelling'; record.job.updatedAt = stamp(); this.persist()
    const child = this.children.get(id)

    if (child) {this.terminateWorker(child, () => this.children.get(id) === child)}
    else { record.job.state = 'cancelled'; this.persist() }

    return true
  }
  reveal(_owner: BrowserWindow, scope: PlayerAgentScope, id: string): boolean {
    const record = this.records.get(id)

    if (!record || !sameScope(record.scope, scope) || record.job.state !== 'completed' || !record.destination || !existsSync(record.destination)) {return false}
    const file = lstatSync(record.destination)

    if (!file.isFile() || file.isSymbolicLink() || !record.artifactIdentity || this.fileIdentity(record.destination) !== record.artifactIdentity) {return false}
    this.options.revealFile(record.destination);

 return true
  }
  disposeScope(scope: PlayerAgentScope): void {
    const key = JSON.stringify([scope.connectionId, scope.profileId, scope.sessionId])
    this.scopeEpochs.set(key, (this.scopeEpochs.get(key) ?? 0) + 1)

    for (const record of this.records.values()) {if (sameScope(record.scope, scope) && !terminal.has(record.job.state)) {this.cancel(scope, record.job.id)}}
  }
  dispose(): void { this.disposed = true;

 for (const child of this.setupChildren) {this.terminateWorker(child);}

 for (const record of this.records.values()) {if (!terminal.has(record.job.state)) {this.cancel(record.scope, record.job.id)}} }

  private persist(): void {
    mkdirSync(this.options.stateDirectory, { recursive: true, mode: 0o700 })
    const records = [...this.records.values()].slice(-100)
    const mac = createHmac('sha256', this.readKey(true)).update(JSON.stringify(records)).digest('hex')
    const temp = `${this.ledger}.${randomUUID()}.tmp`
    const descriptor = openSync(temp, 'wx', 0o600)

    try { writeFileSync(descriptor, JSON.stringify({ schema: 1, records, mac })) } finally { closeSync(descriptor) }
    renameSync(temp, this.ledger)
  }
  private readKey(create: boolean): Buffer {
    if (!existsSync(this.ledgerKey) && create) {
      const descriptor = openSync(this.ledgerKey, 'wx', 0o600)

      try { writeFileSync(descriptor, randomBytes(32)) } finally { closeSync(descriptor) }
    }

    if (!lstatSync(this.ledgerKey).isFile() || lstatSync(this.ledgerKey).isSymbolicLink() || statSync(this.ledgerKey).size !== 32) {fail('invalid-ledger-key')}
    const descriptor = openSync(this.ledgerKey, constants.O_RDONLY | (constants.O_NOFOLLOW || 0))

    try { return readFileSync(descriptor) } finally { closeSync(descriptor) }
  }
  private fileIdentity(file: string): string {
    const stat = lstatSync(file, { bigint: true })

    return `${stat.dev}:${stat.ino}:${stat.size}:${stat.mtimeNs}`
  }
  private stopWorker(child: ChildProcess, signal: NodeJS.Signals): void {
    if (child.exitCode !== null || child.signalCode !== null) {return}
    // Signal only the exact child PID. POSIX workers propagate SIGTERM from
    // inside their verified private process group; Windows workers attach
    // descendants to a kill-on-close Job Object before spawning them.
    child.kill(signal)
  }
  private terminateWorker(child: ChildProcess, stillOwned: () => boolean = () => true): void {
    this.stopWorker(child, 'SIGTERM')

    const timer = setTimeout(() => {
      if (stillOwned() && child.exitCode === null && child.signalCode === null) {this.stopWorker(child, 'SIGKILL')}
    }, 3000)

    timer.unref()
  }
  private environment(): NodeJS.ProcessEnv {
    const env: NodeJS.ProcessEnv = { HOME: this.options.stateDirectory, USERPROFILE: this.options.stateDirectory, PYTHONNOUSERSITE: '1', ELECTRON_RUN_AS_NODE: '1' }

    for (const key of ['PATH', 'SystemRoot', 'WINDIR', 'COMSPEC', 'PATHEXT', 'TEMP', 'TMP', 'TMPDIR', 'LANG']) {if (process.env[key]) {env[key] = process.env[key]}}

    return env
  }
  private run(executable: string, args: string[], timeout: number): Promise<void> {
    return new Promise((resolve, reject) => {
      // A small Python supervisor owns nested setup subprocesses (venv/pip).
      // Its private process group or Windows Job Object makes exact-child
      // termination cover the full setup tree without host-wide PID signaling.
      const supervisorRequest = JSON.stringify({ executable, args })
      const child = spawn(executable, ['-I', this.options.workerPath, '--supervise', supervisorRequest], { shell: false, detached: process.platform !== 'win32', windowsHide: true, stdio: 'ignore', env: this.environment() })
      this.setupChildren.add(child)
      let settled = false

      const timer = setTimeout(() => {
        if (settled) {return}
        settled = true
        this.terminateWorker(child, () => this.setupChildren.has(child))
        reject(new Error('helper-timeout'))
      }, timeout)

      child.once('error', () => {
        clearTimeout(timer); this.setupChildren.delete(child)

        if (!settled) { settled = true; reject(new Error('helper-start-failed')) }
      })
      child.once('exit', code => {
        clearTimeout(timer); this.setupChildren.delete(child)

        if (!settled) { settled = true; code === 0 && !this.disposed ? resolve() : reject(new Error('helper-setup-failed')) }
      })
    })
  }
  private execute(record: StoredJob, options: PlayerMediaJobStartOptions): void {
    if (record.job.state !== 'queued' || this.disposed) {return}
    record.job.state = 'running'; record.job.updatedAt = stamp(); this.persist()
    const output = record.workDirectory ? path.join(record.workDirectory, options.quality === 'audio' ? 'media.m4a' : 'media.mp4') : null
    const child = spawn(this.python, ['-I', this.options.workerPath], { shell: false, detached: process.platform !== 'win32', windowsHide: true, stdio: ['pipe', 'pipe', 'ignore'], env: this.environment() })
    this.children.set(record.job.id, child)
    let buffer = ''; let result: Record<string, any> | null = null; let received = 0; let finished = false
    const timer = setTimeout(() => { record.job.error = { code: 'timeout', message: 'The media worker timed out.' }; this.terminateWorker(child, () => this.children.get(record.job.id) === child) }, 30 * 60_000)
    child.stdin.end(JSON.stringify({ operation: options.kind, videoId: record.job.source.videoId, language: options.language, range: options.range, quality: options.quality ?? 'best', output, nodeExecutable: this.options.nodeExecutable }))
    child.stdin.on('error', () => { this.terminateWorker(child, () => this.children.get(record.job.id) === child) })
    child.stdout.on('data', chunk => {
      received += chunk.length

      if (received > 4 * 1024 * 1024) { this.terminateWorker(child, () => this.children.get(record.job.id) === child);

 return }

      buffer += chunk.toString('utf8')
      let newline: number

      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline); buffer = buffer.slice(newline + 1)

        try {
          const value = JSON.parse(line)

          if (value.type === 'progress' && typeof value.progress === 'number' && Number.isFinite(value.progress)) {record.job.progress = Math.max(0, Math.min(1, value.progress))}

          if (value.type === 'result') {result = value}
        } catch { /* Worker protocol admits JSON lines only. */ }
      }
    })

    const finish = (code: number | null) => {
      if (finished) {return}
      finished = true
      clearTimeout(timer); this.children.delete(record.job.id)

      try {
        this.finishJob(record, options, output, result as Record<string, any> | null, code)
      } catch (error) { record.job.state = 'failed'; record.job.error ??= { code: error instanceof Error && /^[a-z-]{1,60}$/.test(error.message) ? error.message : 'media-worker-failed', message: 'The requested media job could not complete. No account cookies or protected streams were exported.' } }
      finally {
        if (record.workDirectory) {rmSync(record.workDirectory, { recursive: true, force: true })}
        record.job.updatedAt = stamp(); this.persist()
      }
    }

    child.once('error', () => finish(1))
    child.once('exit', finish)
  }

  private finishJob(record: StoredJob, options: PlayerMediaJobStartOptions, output: string | null, result: Record<string, any> | null, code: number | null): void {
    if (record.job.state === 'cancelling') {
      record.job.state = 'cancelled'

      return
    }

    const value = this.requireWorkerResult(code, result)

    if (record.job.kind === 'download') {this.finishDownload(record, options, output, value)}
    else {this.finishTranscript(record, value)}

    record.job.state = 'completed'
    record.job.progress = 1
  }

  private requireWorkerResult(code: number | null, result: Record<string, any> | null): Record<string, any> {
    if (code !== 0 || !result || result.error) {
      const error = typeof result?.error === 'string' && /^[a-z-]{1,60}$/.test(result.error) ? result.error : 'media-worker-failed'
      fail(error)
    }

    return result
  }

  private finishDownload(record: StoredJob, options: PlayerMediaJobStartOptions, output: string | null, result: Record<string, any>): void {
    const destination = record.destination

    if (!output || !destination || !existsSync(output) || !lstatSync(output).isFile() || lstatSync(output).isSymbolicLink() || statSync(output).size < 12) {fail('output-invalid')}
    const bytes = Buffer.alloc(12)
    const descriptor = openSync(output, constants.O_RDONLY | (constants.O_NOFOLLOW || 0))

    try { readSync(descriptor, bytes, 0, bytes.length, 0) } finally { closeSync(descriptor) }

    if (bytes.subarray(4, 8).toString('ascii') !== 'ftyp') {fail('output-invalid')}
    linkSync(output, destination)
    unlinkSync(output)
    record.artifactIdentity = this.fileIdentity(destination)
    record.job.artifact = {
      fileName: path.basename(destination),
      sizeBytes: statSync(destination).size,
      format: options.quality === 'audio' ? 'm4a' : `mp4${Number.isFinite(result.height) ? ` (${result.height}p)` : ''}`
    }
  }

  private finishTranscript(record: StoredJob, result: Record<string, any>): void {
    if (!Array.isArray(result.segments) || result.segments.length > 10000 || typeof result.language !== 'string') {fail('transcript-invalid')}
    record.job.transcript = {
      language: result.language.slice(0, 32),
      trackKind: result.trackKind === 'automatic' ? 'automatic' : result.trackKind === 'manual' ? 'manual' : 'unknown',
      retrievedAt: stamp(),
      segments: result.segments.map((segment: any) => this.toTranscriptSegment(segment))
    }
  }

  private toTranscriptSegment(segment: any): NonNullable<PlayerMediaJob['transcript']>['segments'][number] {
    if (!Number.isFinite(segment.start) || segment.start < 0 || !Number.isFinite(segment.end) || segment.end < segment.start || typeof segment.text !== 'string' || segment.text.length > 20000) {fail('transcript-invalid')}

    return { startSeconds: segment.start, endSeconds: segment.end, text: { text: segment.text, trust: 'untrusted-page' } }
  }
}
