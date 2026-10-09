/**
 * ssh-connection.ts
 *
 * Pure, electron-free OpenSSH ControlMaster connection manager for Desktop SSH
 * remote mode. Uses the system `ssh` client (not a JS SSH library) so it
 * inherits ~/.ssh/config, the agent, jump hosts (ProxyJump), and hardware keys
 * for free — the same rationale as tools/environments/ssh.py.
 *
 * No `import 'electron'` so it is unit-testable without Electron. main.ts
 * wires it into the electron-coupled lifecycle.
 *
 * Conventions mirrored from tools/environments/ssh.py:
 *   - ControlMaster=auto + ControlPersist so one TCP/auth handshake is reused
 *     across exec/forward operations.
 *   - Hashed control-socket filename under a short tmpdir to stay under the
 *     104-byte sun_path limit macOS enforces on Unix domain sockets.
 *   - BatchMode=yes for every programmatic invocation — a spawned ssh must
 *     never hang on an interactive prompt (passphrase / 2FA). If auth needs
 *     interactivity we fail fast and tell the user to load the key into their
 *     agent.
 *
 * Host-key policy: StrictHostKeyChecking=accept-new (trust-on-first-use, log
 * the fingerprint), never `no`. A host-key *change* fails closed with the
 * verbatim OpenSSH error surfaced to the UI.
 *
 * Every operation is raced against a hard timeout. A half-open TCP connection
 * after laptop sleep can leave ssh hanging indefinitely rather than erroring;
 * timeout is treated as connection-dead so the caller does a full reconnect
 * rather than retrying in place.
 */

import { spawn } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'

const DEFAULT_CONNECT_TIMEOUT_MS = 15_000
const DEFAULT_EXEC_TIMEOUT_MS = 20_000
const DEFAULT_FORWARD_TIMEOUT_MS = 15_000
const CONTROL_PERSIST_SECONDS = 300

// eslint-disable-next-line no-control-regex -- deliberately reject control chars in ssh targets
const _CONTROL_CHAR_RE = /[\x00-\x1f\x7f]/

function validateSshTarget(host, user, port) {
  if (!host || typeof host !== 'string') {
    throw new Error('Unsafe SSH target: host is required.')
  }

  if (host.startsWith('-')) {
    throw new Error(`Unsafe SSH target: host must not start with a dash ("${host}").`)
  }

  if (_CONTROL_CHAR_RE.test(host)) {
    throw new Error('Unsafe SSH target: host contains control characters.')
  }

  if (user && _CONTROL_CHAR_RE.test(user)) {
    throw new Error('Unsafe SSH target: user contains control characters.')
  }

  if (user && user.startsWith('-')) {
    throw new Error(`Unsafe SSH target: user must not start with a dash ("${user}").`)
  }

  const p = Number(port)

  if (!Number.isInteger(p) || p < 1 || p > 65535) {
    throw new Error(`Unsafe SSH port: ${port} (must be 1-65535).`)
  }
}

function validateKeyPath(keyPath) {
  if (!keyPath) {
    return
  }

  if (_CONTROL_CHAR_RE.test(keyPath)) {
    throw new Error('Unsafe SSH key path: contains control characters.')
  }

  if (keyPath.startsWith('-')) {
    throw new Error(`Unsafe SSH key path: must not start with a dash ("${keyPath}").`)
  }
}

// Token / secret redaction

const _REDACTIONS: Array<[RegExp, string]> = [
  [/(HERMES_DASHBOARD_SESSION_TOKEN=)(\S+)/g, '$1<redacted>'],
  [/(X-Hermes-Session-Token["']?\s*[:=]\s*["']?)([^\s"'&]+)/gi, '$1<redacted>'],
  [/(Authorization["']?\s*:\s*Bearer\s+)(\S+)/gi, '$1<redacted>'],
  [/([?&](?:token|ticket)=)([^\s&"']+)/gi, '$1<redacted>']
]

function redactSecrets(text) {
  let out = String(text == null ? '' : text)

  for (const [re, repl] of _REDACTIONS) {
    out = out.replace(re, repl)
  }

  return out
}

// Control-socket path

// Hash user@host:port to a short, stable, filesystem-safe socket id — stable
// across reconnects so ControlMaster reuse works, short so the full path stays
// under sun_path's 104-byte limit.
//
// CRITICAL (macOS): the base dir must be SHORT. os.tmpdir() on macOS is the
// per-user `/var/folders/xx/yyyy…/T/` (~49 bytes), and OpenSSH binds a
// TEMPORARY listener at `<ControlPath>.<16 random chars>` while establishing
// the master — so a path that itself fits 104 still overflows at bind time. We
// root under a short per-user base (`~/.hermes/desktop-ssh`) so even worst case
// (~72 bytes on macOS) stays clear. Windows has no AF_UNIX sun_path limit.
function controlSocketPath(user, host, port, baseDir?, identity: any = {}) {
  const dir = baseDir || defaultControlDir()
  const keyPathIdentity = path.normalize(String(identity.keyPath || ''))

  const parts = [
    identity.ownershipId || '',
    identity.scope || '',
    user || '',
    host,
    Number(port),
    keyPathIdentity,
    identity.effectiveConfigFingerprint || ''
  ]

  const id = crypto.createHash('sha256').update(JSON.stringify(parts)).digest('hex').slice(0, 16)

  return path.join(dir, `${id}.sock`)
}

function defaultControlDir() {
  // POSIX: a SHORT, PER-USER base stays under the socket limit AND avoids a
  // world-shared /tmp dir (no symlink-hijack surface). Created 0700 in open().
  if (process.platform === 'win32') {
    return path.join(os.tmpdir(), 'hermes-desktop-ssh')
  }

  return path.join(os.homedir(), '.hermes', 'desktop-ssh')
}

// Command construction (pure — the unit tests exercise these directly)

// Mux (POSIX): ControlMaster options so exec/forward share one authenticated
// connection. No-mux (Windows OpenSSH never implemented mux sockets): plain
// per-invocation options — each ssh call authenticates on its own.
function baseSshOptions(controlPath, connectTimeoutMs?) {
  const connectSecs = Math.max(1, Math.round((connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS) / 1000))

  const mux = controlPath
    ? [
        '-o',
        `ControlPath=${controlPath}`,
        '-o',
        'ControlMaster=auto',
        '-o',
        `ControlPersist=${CONTROL_PERSIST_SECONDS}`
      ]
    : []

  return [
    ...mux,
    '-o',
    'BatchMode=yes',
    '-o',
    'StrictHostKeyChecking=accept-new',
    '-o',
    'ExitOnForwardFailure=yes',
    '-o',
    `ConnectTimeout=${connectSecs}`
  ]
}

// Non-default port and explicit identity file, shared by exec/master/forward.
function hostArgs({ port, keyPath }: { port?: number | string; keyPath?: string } = {}) {
  const args: string[] = []

  if (port && Number(port) !== 22) {
    args.push('-p', String(port))
  }

  if (keyPath) {
    validateKeyPath(keyPath)
    args.push('-i', keyPath)
  }

  return args
}

function target(user, host) {
  return user ? `${user}@${host}` : host
}

function buildExecArgs(conn, remoteCommand, connectTimeoutMs?) {
  return [
    ...baseSshOptions(conn.controlPath, connectTimeoutMs),
    ...hostArgs(conn),
    '--',
    target(conn.user, conn.host),
    remoteCommand
  ]
}

function buildControlArgs(conn, op, extra: string[] = [], connectTimeoutMs?) {
  return [
    '-O',
    op,
    ...extra,
    ...baseSshOptions(conn.controlPath, connectTimeoutMs),
    ...hostArgs(conn),
    '--',
    target(conn.user, conn.host)
  ]
}

// Open the master explicitly: `-M -N -f` backgrounds ssh once the master is up,
// so the spawn resolves when the connection is established (or fails fast under
// BatchMode if auth is non-interactive-only).
function buildMasterArgs(conn, connectTimeoutMs?) {
  return [
    '-M',
    '-N',
    '-f',
    ...baseSshOptions(conn.controlPath, connectTimeoutMs),
    ...hostArgs(conn),
    '--',
    target(conn.user, conn.host)
  ]
}

// Interactive `ssh -tt` for the INTERIM remote terminal (SSH mode only). Reuses
// the existing ControlMaster socket so NO new auth handshake happens — the
// master is already open, so this attaches instantly and never prompts.
//
// NOTE(remote-terminal): interim until the dashboard /api/terminal WebSocket
// lands (specs/desktop-remote-terminal.md); delete this path then.
function buildInteractiveSshArgs(conn, remoteCwd, connectTimeoutMs?, remoteCommand?) {
  const args = [
    '-tt',
    ...baseSshOptions(conn.controlPath, connectTimeoutMs),
    ...hostArgs(conn),
    '--',
    target(conn.user, conn.host)
  ]

  if (remoteCommand) {
    args.push(remoteCommand)

    return args
  }

  const cwd = String(remoteCwd || '').trim()

  if (cwd) {
    const q = `'${cwd.replace(/'/g, `'\\''`)}'`
    args.push(`cd ${q} 2>/dev/null; exec "$SHELL" -l`)
  } else {
    args.push('exec "$SHELL" -l')
  }

  return args
}

// Bind the local end to 127.0.0.1 ONLY — never 0.0.0.0 — so the tunnel does not
// re-expose the remote dashboard to the client's LAN.
function forwardSpec(localPort, remotePort, remoteHost = '127.0.0.1') {
  return `127.0.0.1:${localPort}:${remoteHost}:${remotePort}`
}

// Error classification — distinct, actionable messages for the UI

const SSH_ERROR = {
  UNREACHABLE: 'unreachable',
  AUTH_FAILED: 'auth-failed',
  HOST_KEY_CHANGED: 'host-key-changed',
  TIMEOUT: 'timeout',
  UNKNOWN: 'unknown'
}

// Order matters: the host-key-change banner also contains "WARNING"/"Offending",
// so check it before generic auth.
function classifySshError(stderr) {
  const text = String(stderr || '')

  if (
    /REMOTE HOST IDENTIFICATION HAS CHANGED|Host key verification failed|Offending (?:key|ECDSA|RSA|ED25519)/i.test(
      text
    )
  ) {
    return SSH_ERROR.HOST_KEY_CHANGED
  }

  if (
    /Permission denied|Too many authentication failures|no matching host key|publickey|password|keyboard-interactive/i.test(
      text
    )
  ) {
    return SSH_ERROR.AUTH_FAILED
  }

  if (
    /Could not resolve hostname|Connection refused|Connection timed out|No route to host|Network is unreachable|Operation timed out|port \d+: Connection/i.test(
      text
    )
  ) {
    return SSH_ERROR.UNREACHABLE
  }

  return SSH_ERROR.UNKNOWN
}

function sshErrorMessage(kind, conn, stderr?) {
  const host = target(conn.user, conn.host)

  switch (kind) {
    case SSH_ERROR.HOST_KEY_CHANGED:
      return (
        `The host key for ${host} has CHANGED since you last connected. ` +
        `This could be a man-in-the-middle attack, or the server was reinstalled. ` +
        `SSH refused to connect. Verify the change is expected, then remove the old key ` +
        `with \`ssh-keygen -R ${conn.host}\` and reconnect.\n\n${String(stderr || '').trim()}`
      )

    case SSH_ERROR.AUTH_FAILED:
      return (
        `SSH authentication to ${host} failed. Desktop runs ssh non-interactively ` +
        `(BatchMode), so a key requiring a passphrase or 2FA must be loaded into your ` +
        `ssh-agent first (e.g. \`ssh-add ~/.ssh/id_ed25519\`), or set an IdentityFile in ` +
        `~/.ssh/config. Original error: ${String(stderr || '').trim()}`
      )

    case SSH_ERROR.UNREACHABLE:
      return `Could not reach ${host} over SSH. Check the host, port, and your network. Original error: ${String(stderr || '').trim()}`

    case SSH_ERROR.TIMEOUT:
      return `SSH operation to ${host} timed out. The connection may be half-open (e.g. after sleep); reconnecting.`

    default:
      return `SSH error connecting to ${host}: ${String(stderr || '').trim() || 'unknown failure'}`
  }
}

// Spawn helper — runs an ssh invocation, races it against a hard timeout

// Resolves { code, stdout, stderr }. On timeout the child is SIGKILLed and the
// promise rejects with err.kind = TIMEOUT. `spawnFn` is injectable for tests.
function runSsh(args, { timeoutMs, spawnFn = spawn, stdin = 'ignore', stdinData }: any = {}) {
  return new Promise((resolve, reject) => {
    const useStdinPipe = stdinData != null || stdin !== 'ignore'
    let child

    try {
      child = spawnFn('ssh', args, { stdio: [useStdinPipe ? 'pipe' : 'ignore', 'pipe', 'pipe'] })
    } catch (error) {
      reject(error)

      return
    }

    if (stdinData != null && child.stdin) {
      child.stdin.end(stdinData)
    }

    let stdout = ''
    let stderr = ''
    let settled = false

    const timer = setTimeout(() => {
      if (settled) {
        return
      }

      settled = true

      try {
        child.kill('SIGKILL')
      } catch {
        // already gone
      }

      const err: any = new Error(`ssh timed out after ${timeoutMs}ms`)
      err.kind = SSH_ERROR.TIMEOUT
      reject(err)
    }, timeoutMs)

    child.stdout?.on('data', d => {
      stdout += d.toString()
    })
    child.stderr?.on('data', d => {
      stderr += d.toString()
    })
    child.on('error', error => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      reject(error)
    })
    child.on('close', code => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      resolve({ code, stdout, stderr })
    })
  })
}

function stopTunnelChild(child, timeoutMs = 5_000) {
  if (!child || child.exitCode != null || child.signalCode != null) {
    return Promise.resolve()
  }

  return new Promise<void>((resolve, reject) => {
    let settled = false

    const finish = (error?: unknown) => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      child.off?.('exit', onExit)
      child.off?.('error', onError)
      error ? reject(error) : resolve()
    }

    const onExit = () => finish()
    const onError = error => finish(error)
    const timer = setTimeout(() => finish(new Error('SSH tunnel did not exit after termination.')), timeoutMs)
    child.once('exit', onExit)
    child.once('error', onError)

    try {
      if (!child.kill()) {
        finish(new Error('SSH tunnel termination was refused.'))
      }
    } catch (error) {
      finish(error)
    }
  })
}

// SshConnection — the public manager

class SshConnection {
  host: string
  user: string
  port: number
  keyPath: string
  controlPath: string
  _spawnFn: any
  _log: (msg: string) => void
  _connectTimeoutMs: number
  _execTimeoutMs: number
  _forwardTimeoutMs: number
  _opened: boolean
  _mux: boolean
  _tunnels: Map<string, any>

  constructor(cfg, opts: any = {}) {
    if (!cfg || !cfg.host) {
      throw new Error('SshConnection requires a host.')
    }

    const port = cfg.port ? Number(cfg.port) : 22
    validateSshTarget(cfg.host, cfg.user || '', port)

    if (cfg.keyPath) {
      validateKeyPath(cfg.keyPath)
    }

    this.host = cfg.host
    this.user = cfg.user || ''
    this.port = port
    this.keyPath = cfg.keyPath || ''
    // Windows OpenSSH has no ControlMaster (mux sockets were never implemented
    // on Win32) — fall back to one ssh invocation per operation and a
    // persistent `ssh -N -L` child per tunnel. Empty controlPath routes the
    // pure builders onto their no-mux form.
    this._mux = opts.mux ?? process.platform !== 'win32'
    this.controlPath = this._mux
      ? controlSocketPath(this.user, this.host, this.port, opts.controlDir, {
          keyPath: this.keyPath,
          ownershipId: opts.ownershipId,
          scope: opts.scope,
          effectiveConfigFingerprint: opts.effectiveConfigFingerprint
        })
      : ''
    this._tunnels = new Map()

    this._spawnFn = opts.spawnFn || spawn

    this._log = typeof opts.rememberLog === 'function' ? opts.rememberLog : () => {}
    this._connectTimeoutMs = opts.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS
    this._execTimeoutMs = opts.execTimeoutMs ?? DEFAULT_EXEC_TIMEOUT_MS
    this._forwardTimeoutMs = opts.forwardTimeoutMs ?? DEFAULT_FORWARD_TIMEOUT_MS
    this._opened = false
  }

  // Lifecycle logging — ALWAYS through redaction.
  _logLine(msg) {
    this._log(redactSecrets(`[ssh] ${msg}`))
  }

  _fail(stderrOrErr, fallbackKind = SSH_ERROR.UNKNOWN) {
    if (stderrOrErr && stderrOrErr.kind === SSH_ERROR.TIMEOUT) {
      const err: any = new Error(sshErrorMessage(SSH_ERROR.TIMEOUT, this))
      err.kind = SSH_ERROR.TIMEOUT

      return err
    }

    const stderr = typeof stderrOrErr === 'string' ? stderrOrErr : stderrOrErr?.message || ''
    const kind = stderr ? classifySshError(stderr) : fallbackKind
    const err: any = new Error(sshErrorMessage(kind, this, stderr))
    err.kind = kind

    return err
  }

  // Open the connection. Mux: start the persistent ControlMaster (idempotent —
  // a live master is a no-op). No-mux: there is no master; validate auth +
  // reachability with a one-shot `ssh true` so failures classify identically.
  async open() {
    if (await this.isAlive()) {
      // -O check passing is not proof the master works: a ControlPersist master
      // can survive a failed teardown with wedged channels (observed on macOS
      // after a mode switch — check succeeds, every exec times out). Verify with
      // a real exec before trusting it; on failure, evict and dial fresh.
      if (!this._mux || (await this._verifyMuxChannel())) {
        this._opened = true

        return
      }

      this._logLine('existing control master failed exec verification; evicting stale master')
      await this._evictStaleMaster()
    }

    if (!this._mux) {
      this._logLine(`connecting (no-mux) to ${target(this.user, this.host)}:${this.port}`)
      let result

      try {
        result = await runSsh(buildExecArgs(this, 'exit 0', this._connectTimeoutMs), {
          timeoutMs: this._connectTimeoutMs,
          spawnFn: this._spawnFn
        })
      } catch (error) {
        throw this._fail(error, SSH_ERROR.UNREACHABLE)
      }

      if (result.code !== 0) {
        throw this._fail(result.stderr, SSH_ERROR.UNREACHABLE)
      }

      this._opened = true
      this._logLine('connection verified (no-mux; per-operation ssh)')

      return
    }

    const controlDir = path.dirname(this.controlPath)

    try {
      fs.mkdirSync(controlDir, { recursive: true, mode: 0o700 })
    } catch {
      void 0
    }

    if (process.platform !== 'win32') {
      const st = fs.lstatSync(controlDir)

      if (st.isSymbolicLink()) {
        throw new Error(`Unsafe SSH control dir: ${controlDir} is a symlink.`)
      }

      if (!st.isDirectory()) {
        throw new Error(`Unsafe SSH control dir: ${controlDir} is not a directory.`)
      }

      if (st.uid !== process.getuid!()) {
        throw new Error(`Unsafe SSH control dir: ${controlDir} is owned by uid ${st.uid}, not ${process.getuid!()}.`)
      }

      if ((st.mode & 0o777) !== 0o700) {
        fs.chmodSync(controlDir, 0o700)
      }
    }

    const args = buildMasterArgs(this, this._connectTimeoutMs)
    this._logLine(`opening control master to ${target(this.user, this.host)}:${this.port}`)
    let result

    try {
      result = await runSsh(args, { timeoutMs: this._connectTimeoutMs, spawnFn: this._spawnFn })
    } catch (error) {
      throw this._fail(error, SSH_ERROR.UNREACHABLE)
    }

    if (result.code !== 0) {
      throw this._fail(result.stderr, SSH_ERROR.UNREACHABLE)
    }

    this._opened = true
    this._logLine('control master established')
  }

  // Liveness. Mux: `-O check` against the master socket. No-mux: a cheap
  // one-shot exec — "alive" means "we can still authenticate and run".
  async isAlive() {
    if ([...this._tunnels.values()].some(tunnel => tunnel.alive === false)) {
      return false
    }

    const args = this._mux
      ? buildControlArgs(this, 'check', [], this._connectTimeoutMs)
      : buildExecArgs(this, 'exit 0', this._connectTimeoutMs)

    try {
      const result: any = await runSsh(args, { timeoutMs: this._connectTimeoutMs, spawnFn: this._spawnFn })

      return result.code === 0
    } catch {
      return false
    }
  }

  // A real exec through the master (`exit 0` works under POSIX shells and
  // cmd.exe); a wedged mux hangs to the timeout.
  async _verifyMuxChannel() {
    try {
      const result: any = await runSsh(buildExecArgs(this, 'exit 0', this._connectTimeoutMs), {
        timeoutMs: this._connectTimeoutMs,
        spawnFn: this._spawnFn
      })

      return result.code === 0
    } catch {
      return false
    }
  }

  // -O exit (best-effort) then drop the socket so ControlMaster=auto cannot
  // re-attach to the corpse. (The orphaned master process is left to
  // ControlPersist; a wedged channel can pin it, but without its socket it is
  // inert.)
  async _evictStaleMaster() {
    try {
      await runSsh(buildControlArgs(this, 'exit', [], this._connectTimeoutMs), {
        timeoutMs: this._connectTimeoutMs,
        spawnFn: this._spawnFn
      })
    } catch {
      void 0
    }

    try {
      fs.unlinkSync(this.controlPath)
    } catch (error: any) {
      if (error?.code !== 'ENOENT') {
        this._logLine(`could not remove stale control socket (${error.code}); a fresh master may not dial`)
      }
    }
  }

  // One-shot remote command over the control connection. Resolves stdout;
  // rejects with a classified error on non-zero exit or timeout.
  async exec(remoteCommand, { timeoutMs, stdinData }: any = {}) {
    const args = buildExecArgs(this, remoteCommand, this._connectTimeoutMs)
    let result

    try {
      result = await runSsh(args, {
        timeoutMs: timeoutMs ?? this._execTimeoutMs,
        spawnFn: this._spawnFn,
        ...(stdinData != null ? { stdinData } : {})
      })
    } catch (error) {
      throw this._fail(error)
    }

    if (result.code !== 0) {
      throw this._fail(result.stderr)
    }

    return result.stdout
  }

  // Establish a local→remote forward. Mux: `-O forward` against the master.
  // No-mux: spawn a persistent `ssh -N -L` child that IS the tunnel; ready when
  // the local port accepts. The child dying = tunnel down (isAlive of the
  // backend catches it upstream).
  async forward(localPort, remotePort, remoteHost = '127.0.0.1') {
    const spec = forwardSpec(localPort, remotePort, remoteHost)
    this._logLine(`forwarding 127.0.0.1:${localPort} -> ${remoteHost}:${remotePort}`)

    if (!this._mux) {
      const args = [
        ...baseSshOptions('', this._connectTimeoutMs),
        ...hostArgs(this),
        '-v',
        '-N',
        '-L',
        spec,
        '--',
        target(this.user, this.host)
      ]

      const child = this._spawnFn('ssh', args, { stdio: ['ignore', 'ignore', 'pipe'] })
      const tunnel = { child, alive: true }
      this._tunnels.set(spec, tunnel)
      let stderr = ''
      let readyConfirmed = false
      let readyResolve
      let readyReject

      const ready = new Promise<void>((resolve, reject) => {
        readyResolve = resolve
        readyReject = reject
      })

      const readyPattern = new RegExp(`Local forwarding listening on .* port ${localPort}\\b`)
      child.stderr?.on('data', d => {
        if (readyConfirmed) {
          return
        }

        stderr = `${stderr}${String(d)}`.slice(-16_384)

        if (readyPattern.test(stderr)) {
          readyConfirmed = true
          readyResolve()
        }
      })
      child.on('error', error => {
        tunnel.alive = false
        readyReject(error)
      })
      child.on('exit', code => {
        tunnel.alive = false
        readyReject(new Error(`tunnel process exited with code ${code}`))
      })
      child.on('close', code => {
        tunnel.alive = false
        readyReject(new Error(`tunnel process closed with code ${code}`))
      })
      let readyTimeout

      try {
        await Promise.race([
          ready,
          new Promise((_, reject) => {
            readyTimeout = setTimeout(
              () => reject(new Error('tunnel did not confirm local forwarding')),
              this._forwardTimeoutMs
            )
          })
        ])
      } catch (error: any) {
        try {
          await stopTunnelChild(child)
          this._tunnels.delete(spec)
        } catch (stopError) {
          throw this._fail(stopError, SSH_ERROR.UNKNOWN)
        }

        throw this._fail(stderr || error, SSH_ERROR.UNKNOWN)
      } finally {
        clearTimeout(readyTimeout)
      }

      return
    }

    const args = buildControlArgs(this, 'forward', ['-L', spec], this._connectTimeoutMs)
    let result

    try {
      result = await runSsh(args, { timeoutMs: this._forwardTimeoutMs, spawnFn: this._spawnFn })
    } catch (error) {
      throw this._fail(error)
    }

    if (result.code !== 0) {
      throw this._fail(result.stderr)
    }
  }

  // Cancel a previously-established forward. Best-effort: a failure here is
  // logged but not thrown (close tears everything down anyway).
  async cancelForward(localPort, remotePort, remoteHost = '127.0.0.1') {
    const spec = forwardSpec(localPort, remotePort, remoteHost)

    if (!this._mux) {
      const tunnel = this._tunnels.get(spec)

      if (tunnel) {
        await stopTunnelChild(tunnel.child)
        this._tunnels.delete(spec)
        this._logLine(`cancelled forward 127.0.0.1:${localPort}`)
      }

      return
    }

    const args = buildControlArgs(this, 'cancel', ['-L', spec], this._connectTimeoutMs)

    try {
      await runSsh(args, { timeoutMs: this._forwardTimeoutMs, spawnFn: this._spawnFn })
      this._logLine(`cancelled forward 127.0.0.1:${localPort}`)
    } catch (error: any) {
      this._logLine(`cancelForward failed (ignored): ${error.message}`)
    }
  }

  // Tear down. Mux: exit the master (drops every forward with it). No-mux:
  // kill the tunnel children. Best-effort; never throws.
  async close() {
    if (!this._opened) {
      return
    }

    if (!this._mux) {
      for (const [spec, tunnel] of this._tunnels) {
        await stopTunnelChild(tunnel.child)
        this._tunnels.delete(spec)
      }

      this._opened = false
      this._logLine('connection closed (no-mux tunnels killed)')

      return
    }

    const args = buildControlArgs(this, 'exit', [], this._connectTimeoutMs)

    try {
      const result: any = await runSsh(args, { timeoutMs: this._connectTimeoutMs, spawnFn: this._spawnFn })

      if (result.code !== 0) {
        throw this._fail(result.stderr)
      }

      this._logLine('control master closed')
    } catch (error: any) {
      // A master that refuses -O exit is the wedge that poisons re-attach;
      // disown it. (Without its socket the orphan is inert; ControlPersist may
      // not reap it if a wedged channel never idles.)
      this._logLine(`close failed; removing control socket: ${error.message}`)

      try {
        fs.unlinkSync(this.controlPath)
      } catch {
        void 0
      }
    }

    this._opened = false
  }
}

// Free local port for the tunnel's local end. Bind 127.0.0.1:0, read the
// kernel-assigned port, release. The benign TOCTOU window (release → forward
// grabs it) is caught upstream and retried with a fresh port.

function pickLocalPort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as net.AddressInfo
      server.close(() => resolve(port))
    })
  })
}

function createSshProbeConnection(config, options: any = {}) {
  return new SshConnection(config, { ...options, mux: false })
}

export {
  baseSshOptions,
  buildControlArgs,
  buildExecArgs,
  buildInteractiveSshArgs,
  buildMasterArgs,
  classifySshError,
  CONTROL_PERSIST_SECONDS,
  controlSocketPath,
  createSshProbeConnection,
  DEFAULT_CONNECT_TIMEOUT_MS,
  DEFAULT_EXEC_TIMEOUT_MS,
  DEFAULT_FORWARD_TIMEOUT_MS,
  forwardSpec,
  hostArgs,
  pickLocalPort,
  redactSecrets,
  runSsh,
  SSH_ERROR,
  SshConnection,
  sshErrorMessage,
  stopTunnelChild,
  target,
  validateKeyPath,
  validateSshTarget
}
