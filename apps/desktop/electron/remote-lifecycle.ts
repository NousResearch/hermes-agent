/**
 * remote-lifecycle.ts
 *
 * Pure, electron-free remote Hermes dashboard lifecycle over SSH for Desktop
 * SSH remote mode. Composes an SshConnection (injected) with HTTP probes
 * through the established tunnel (injected fetch) and the served-token adoption
 * step (injected). Knows how to:
 *
 *   - locate the Hermes install on the remote (login-shell probe),
 *   - gate the remote platform to Linux/macOS via `uname`,
 *   - reuse an existing desktop-dedicated dashboard via a lockfile + an
 *     AUTHENTICATED /api/status probe (pid liveness alone is insufficient),
 *   - spawn a fresh detached `--isolated --port 0` dashboard and scrape its
 *     `HERMES_DASHBOARD_READY port=<n>` readiness line,
 *   - adopt the token the dashboard actually serves (served-token adoption),
 *   - clean up a stale dashboard only when it is provably ours.
 *
 * No `import 'electron'` so it's unit-testable with `node --test`. main.ts wires
 * the real SshConnection, fetch, adoptServedDashboardToken, and waitForHermes in.
 *
 * The minted HERMES_DASHBOARD_SESSION_TOKEN is the SPAWN credential. After
 * readiness the caller runs served-token adoption against the tunneled baseUrl
 * and the SERVED token's fingerprint is what lands in the lockfile — so the
 * reuse probe checks the credential that actually authenticates /api/ws, not
 * the minted one (which the dashboard may regen).
 */

import crypto from 'node:crypto'

const LOCKFILE_SCHEMA_VERSION = 2
// Bumped when the desktop<->dashboard reuse contract changes in a way that makes
// an old running dashboard unsafe to reattach to (token handling, readiness/spawn
// args, served-token reconciliation). A mismatch forces a clean respawn.
const PROTOCOL_VERSION = 1
const READY_RE = /^HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)/m
const REMOTE_LOCK_DIR = '~/.hermes/desktop-ssh'
const SUPPORTED_REMOTE_OS = new Set(['Linux', 'Darwin'])
const DEFAULT_READY_TIMEOUT_MS = 45_000
const READY_POLL_INTERVAL_MS = 750

function mintToken() {
  return crypto.randomBytes(32).toString('hex')
}

// Fingerprint a token for the lockfile — never store the raw secret on the
// remote. SHA256, truncated.
function fingerprintToken(token) {
  return crypto
    .createHash('sha256')
    .update(String(token || ''))
    .digest('hex')
    .slice(0, 32)
}

function validateOwnershipId(ownershipId) {
  const value = String(ownershipId || '')

  if (!/^[0-9a-f]{32}$/.test(value)) {
    throw new Error('SSH ownership ID is invalid.')
  }

  return value
}

function validateSpawnNonce(spawnNonce) {
  const value = String(spawnNonce || '')

  if (!/^[0-9a-f]{16}$/.test(value)) {
    throw new Error('SSH spawn nonce is invalid.')
  }

  return value
}

function ownershipDirectory(ownershipId) {
  return `${REMOTE_LOCK_DIR}/${validateOwnershipId(ownershipId)}`
}

function lockfilePath(ownershipId) {
  return `${ownershipDirectory(ownershipId)}/backend.lock.json`
}

function spawnLogPath(ownershipId, spawnNonce) {
  return `${ownershipDirectory(ownershipId)}/${validateSpawnNonce(spawnNonce)}.log`
}

// shell-single-quote a value for safe interpolation into a remote command.
function shq(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`
}

function validateRemotePath(p) {
  const s = String(p || '')

  if (!s) {
    throw new Error('Remote path must not be empty.')
  }

  // eslint-disable-next-line no-control-regex -- deliberately reject NUL in remote paths
  if (/[\x00\n\r]/.test(s)) {
    throw new Error('Unsafe remote path: contains NUL or newline.')
  }

  if (s === '~' || s.startsWith('~/') || s.startsWith('/')) {
    return
  }

  throw new Error(`Remote path must be absolute or start with ~/: "${s}"`)
}

function expandRemotePath(p) {
  validateRemotePath(p)

  if (p === '~') {
    return '"$HOME"'
  }

  if (p.startsWith('~/')) {
    return '"$HOME"' + shq(p.slice(1))
  }

  return shq(p)
}

// Resolve the remote hermes executable. An EXPLICIT path is honored strictly
// (throws a path-naming error if not executable — never silently falls back to a
// different install). A BLANK path auto-detects: login-shell `command -v` (a
// non-login `ssh host cmd` PATH misses user installs), then known install paths.
async function locateHermes(ssh, remoteHermesPath) {
  const resolveLauncher = async (candidate: string) => {
    const script =
      'import os,shlex,sys\n' +
      `p=os.path.expanduser(${shq(candidate)})\n` +
      'out=p\n' +
      'try:\n' +
      ' data=open(p,"r",encoding="utf-8",errors="ignore").read(4096)\n' +
      ' for line in data.splitlines():\n' +
      '  words=shlex.split(line)\n' +
      '  if len(words)>1 and words[0]=="exec":\n' +
      '   target=os.path.expanduser(words[1])\n' +
      '   if os.path.isabs(target) and os.access(target,os.X_OK):out=target\n' +
      '   break\n' +
      'except (OSError,ValueError):pass\n' +
      'print(out)'

    const resolved = (await ssh.exec(`python3 -c ${shq(script)}`)).trim()

    return resolved || candidate
  }

  const isExecutable = async (candidate: string) => {
    try {
      validateRemotePath(candidate)
      const ok = (await ssh.exec(`[ -x ${expandRemotePath(candidate)} ] && echo OK || true`)).trim()

      return ok === 'OK'
    } catch {
      return false
    }
  }

  if (remoteHermesPath) {
    if (await isExecutable(remoteHermesPath)) {
      return resolveLauncher(remoteHermesPath)
    }

    const err: any = new Error(
      `The Hermes path you set is not an executable on the remote host: "${remoteHermesPath}". ` +
        'Check the path (it must be the full path to the `hermes` binary on the remote, e.g. ' +
        '~/hermes-agent/.venv/bin/hermes), or clear it to auto-detect.'
    )

    err.kind = 'hermes-not-found'
    throw err
  }

  const candidates: string[] = []

  try {
    const found = (await ssh.exec(`bash -lc ${shq('command -v hermes')}`)).trim()

    if (found) {
      candidates.push(found.split('\n').pop().trim())
    }
  } catch {
    // ignore
  }

  // Fallback candidates when the login-shell probe misses: the installer's
  // command locations (scripts/install.sh) — per-user, root/FHS, legacy venv.
  candidates.push('~/.local/bin/hermes')
  candidates.push('/usr/local/bin/hermes')
  candidates.push('~/.hermes/hermes-agent/venv/bin/hermes')

  for (const candidate of candidates) {
    if (!candidate) {
      continue
    }

    if (await isExecutable(candidate)) {
      return resolveLauncher(candidate)
    }
  }

  const err: any = new Error(
    'Hermes is not installed on the remote host (could not find a `hermes` executable). ' +
      'Install it on the remote with:  curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh  ' +
      '— or set the Hermes path explicitly in the SSH connection settings.'
  )

  err.kind = 'hermes-not-found'
  throw err
}

// Probe the resolved binary's version string (first line of `<hermes> --version`,
// e.g. "Hermes Agent v0.18.2 ..."), or '' on failure. Surfaces WHICH hermes a
// connection uses, so a stale/unexpected install is visible.
async function probeHermesVersion(ssh, hermesPath) {
  try {
    const out = (await ssh.exec(`${expandRemotePath(hermesPath)} --version 2>&1`)).trim()

    return (out.split('\n')[0] || '').trim()
  } catch {
    return ''
  }
}

async function probeRemotePlatform(ssh) {
  const out = (await ssh.exec('uname -s; uname -m')).trim().split('\n')
  const osName = (out[0] || '').trim()
  const arch = (out[1] || '').trim()

  if (!SUPPORTED_REMOTE_OS.has(osName)) {
    const err: any = new Error(
      `Unsupported remote platform "${osName || 'unknown'}". Hermes Desktop SSH mode supports Linux, macOS, and Windows remote hosts.`
    )

    err.kind = 'unsupported-platform'
    throw err
  }

  return { os: osName, arch }
}

// The HERMES_HOME the remote dashboard will use (explicit env wins, else
// ~/.hermes). Recorded in the lockfile so a future reuse can tell it's the same
// state store; best-effort.
async function probeRemoteHermesHome(ssh) {
  try {
    const out = (await ssh.exec('echo "${HERMES_HOME:-$HOME/.hermes}"')).trim().split('\n').pop()

    return out || '~/.hermes'
  } catch (cause) {
    const error: any = new Error('Could not resolve the remote Hermes home.')
    error.kind = 'transient-transport-error'
    error.cause = cause
    throw error
  }
}

async function readLockfile(ssh, ownershipId) {
  const lpath = lockfilePath(ownershipId)
  let raw

  try {
    raw = await ssh.exec(`if [ ! -e ${expandRemotePath(lpath)} ]; then exit 0; fi; cat ${expandRemotePath(lpath)}`)
  } catch (cause) {
    const error: any = new Error('Could not read the SSH backend ownership record.')
    error.kind = 'transient-transport-error'
    error.cause = cause
    throw error
  }

  const text = String(raw || '').trim()

  if (!text) {
    return null
  }

  let parsed

  try {
    parsed = JSON.parse(text)
  } catch {
    return null
  }

  if (!parsed || parsed.schemaVersion !== LOCKFILE_SCHEMA_VERSION) {
    return null
  }

  const pid = parsed.pid
  const port = parsed.port

  if (!Number.isInteger(pid) || pid <= 0 || pid > 4194304) {
    return null
  }

  // port 0 = spawn-in-progress record (written before readiness); valid
  // ownership proof for cleanup, but never reusable.
  if (!Number.isInteger(port) || port < 0 || port > 65535) {
    return null
  }

  if (parsed.ownershipId !== ownershipId || !/^[0-9a-f]{16}$/.test(parsed.spawnNonce || '')) {
    return null
  }

  if (!/^[0-9a-f]{32}$/.test(parsed.tokenFingerprint || '')) {
    return null
  }

  if (parsed.protocolVersion !== PROTOCOL_VERSION) {
    return null
  }

  if (parsed.logPath !== spawnLogPath(ownershipId, parsed.spawnNonce)) {
    return null
  }

  for (const field of ['profile', 'hermesPath', 'hermesHome', 'logPath', 'startedAt']) {
    if (typeof parsed[field] !== 'string' || parsed[field].length > 1024) {
      return null
    }
  }

  return parsed
}

async function writeLockfile(ssh, ownershipId, lock) {
  const directory = ownershipDirectory(ownershipId)
  const lpath = lockfilePath(ownershipId)
  const temporaryPath = `${directory}/.${crypto.randomBytes(8).toString('hex')}.lock.tmp`
  const json = JSON.stringify({ ...lock, schemaVersion: LOCKFILE_SCHEMA_VERSION })
  await ssh.exec(
    `umask 077 && mkdir -p ${expandRemotePath(directory)} && ` +
      `printf '%s' ${shq(json)} > ${expandRemotePath(temporaryPath)} && ` +
      `mv -f ${expandRemotePath(temporaryPath)} ${expandRemotePath(lpath)}`
  )
}

async function removeLockfile(ssh, ownershipId) {
  const lpath = lockfilePath(ownershipId)

  try {
    await ssh.exec(`rm -f ${expandRemotePath(lpath)}`)
  } catch {
    // best effort
  }
}

async function remotePidAlive(ssh, pid) {
  if (!pid || !Number.isInteger(Number(pid))) {
    return false
  }

  try {
    const out = (await ssh.exec(`kill -0 ${Number(pid)} 2>/dev/null && echo ALIVE || echo DEAD`)).trim()

    return out === 'ALIVE'
  } catch (cause) {
    const error: any = new Error('Could not verify the SSH backend process.')
    error.kind = 'transient-transport-error'
    error.cause = cause
    throw error
  }
}

// A pid is "provably ours" only if its remote cmdline carries our dashboard
// args — never kill a pid we can't positively identify as our dashboard.
async function pidIsOurDashboard(ssh, pid, spawnNonce, hermesPath = '') {
  if (!pid || !/^[0-9a-f]{16}$/.test(String(spawnNonce || '')) || !hermesPath) {
    return false
  }

  try {
    const script =
      'import os,shlex,subprocess,sys\n' +
      `pid=${Number(pid)}\n` +
      `expected=os.path.expanduser(${shq(hermesPath)})\n` +
      `nonce=${shq(spawnNonce)}\n` +
      'try:\n' +
      ' raw=open(f"/proc/{pid}/cmdline","rb").read()\n' +
      ' args=[x.decode("utf-8","surrogateescape") for x in raw.split(b"\\0") if x]\n' +
      'except OSError:\n' +
      ' line=subprocess.check_output(["ps","-o","command=","-p",str(pid)],text=True).strip()\n' +
      ' args=shlex.split(line)\n' +
      'ok=False\n' +
      'try:\n' +
      ' serve=args.index("serve")\n' +
      ' owner=args.index("--ssh-owner-nonce",serve+1)\n' +
      ' direct=args[0]==expected\n' +
      ' python_entry=len(args)>1 and args[1]==expected and os.path.basename(args[0]).startswith("python")\n' +
      ' ok=(direct or python_entry) and "--isolated" in args[serve+1:] and args[owner+1]==nonce\n' +
      'except (ValueError,IndexError):pass\n' +
      'print("OWNED" if ok else "FOREIGN")'

    const out = await ssh.exec(`python3 -c ${shq(script)}`)

    return String(out || '').trim() === 'OWNED'
  } catch (cause) {
    const error: any = new Error('Could not verify SSH backend process ownership.')
    error.kind = 'transient-transport-error'
    error.cause = cause
    throw error
  }
}

// Kill the stale dashboard ONLY if provably ours, then drop the lockfile.
async function cleanupStale(ssh, ownershipId, lock, pidAlive = true) {
  if (pidAlive && lock && (await pidIsOurDashboard(ssh, lock.pid, lock.spawnNonce, lock.hermesPath))) {
    try {
      const result = (
        await ssh.exec(
          `kill ${Number(lock.pid)} && ` +
            `i=0; while kill -0 ${Number(lock.pid)} 2>/dev/null; do ` +
            `i=$((i+1)); [ "$i" -ge 50 ] && exit 1; sleep 0.1; done`
        )
      ).trim()

      void result
    } catch (cause) {
      const error: any = new Error('Could not terminate the stale SSH backend.')
      error.kind = 'transient-transport-error'
      error.cause = cause
      throw error
    }
  }

  const expectedLogPath = lock?.spawnNonce ? spawnLogPath(ownershipId, lock.spawnNonce) : ''

  if (lock?.logPath === expectedLogPath) {
    try {
      await ssh.exec(`rm -f ${expandRemotePath(lock.logPath)}`)
    } catch {
      void 0
    }
  }

  await removeLockfile(ssh, ownershipId)
}

// Detach so the backend survives the SSH channel closing: setsid (Linux)
// starts a new session; macOS has no setsid, so fall back to nohup (HUP-immune;
// fd-detachment is already handled by </dev/null + redirect + &).
function buildSpawnCommand(hermesPath, profile, opts: any = {}) {
  const hermes = expandRemotePath(hermesPath)
  const profileArgs = profile ? `--profile ${shq(profile)} ` : ''
  const logPath = expandRemotePath(opts.logPath)
  const tokenFilePath = opts.tokenFilePath
  const tokenArg = tokenFilePath ? ` --ssh-session-token-file ${expandRemotePath(tokenFilePath)}` : ''
  const ownerArg = opts.spawnNonce ? ` --ssh-owner-nonce ${validateSpawnNonce(opts.spawnNonce)}` : ''
  const subCmd = `serve --isolated --host 127.0.0.1 --port 0${tokenArg}${ownerArg}`
  const dashCmd = `env HERMES_DESKTOP=1 ${hermes} ${profileArgs}${subCmd}`

  return (
    `mkdir -p "$(dirname ${logPath})" && ` +
    `"$(command -v setsid || echo nohup)" sh -c ${shq(`${dashCmd} </dev/null >> ${logPath} 2>&1 & echo $!`)}`
  )
}

async function remoteSupportsSshOwnership(ssh, hermesPath) {
  const hermes = expandRemotePath(hermesPath)

  const out = await ssh.exec(
    `help="$(${hermes} serve --help 2>&1)"; ` +
      `printf '%s' "$help" | grep -q ssh-session-token-file && ` +
      `printf '%s' "$help" | grep -q ssh-owner-nonce && echo YES || echo NO`
  )

  return String(out || '')
    .trim()
    .endsWith('YES')
}

async function scrapeReadyPort(ssh, logPath, { timeoutMs = DEFAULT_READY_TIMEOUT_MS, isAlive, signal }: any = {}) {
  const deadline = Date.now() + timeoutMs
  const remoteLog = expandRemotePath(logPath)

  while (Date.now() < deadline) {
    assertNotAborted(signal)

    if (isAlive && !(await isAlive())) {
      const err: any = new Error('Remote dashboard process exited before announcing its port.')
      err.kind = 'spawn-failed'
      throw err
    }

    let tail

    try {
      tail = await ssh.exec(`cat ${remoteLog} 2>/dev/null || true`)
    } catch {
      tail = ''
    }

    const m = READY_RE.exec(String(tail || ''))

    if (m) {
      return parseInt(m[1], 10)
    }

    await new Promise(r => setTimeout(r, READY_POLL_INTERVAL_MS))
  }

  const err: any = new Error(`Timed out waiting for the remote dashboard to announce its port (${timeoutMs}ms).`)
  err.kind = 'ready-timeout'
  throw err
}

async function spawnRemoteDashboard(ssh, { hermesPath, profile, token, ownershipId }) {
  if (!(await remoteSupportsSshOwnership(ssh, hermesPath))) {
    const err: any = new Error(
      'The remote Hermes install does not support --ssh-session-token-file and --ssh-owner-nonce. ' +
        'Update Hermes on the remote host to continue using Desktop SSH mode.'
    )

    err.kind = 'update-required'
    throw err
  }

  const spawnNonce = crypto.randomBytes(8).toString('hex')
  const tokenDir = ownershipDirectory(ownershipId)
  const tokenFilePath = `${tokenDir}/${spawnNonce}.token`
  const logPath = spawnLogPath(ownershipId, spawnNonce)

  const tokenUploadPy =
    'import os,sys,stat\n' +
    `p=os.path.expanduser(${shq(tokenFilePath)})\n` +
    'd=os.path.dirname(p)\n' +
    'n=os.path.basename(p)\n' +
    'os.makedirs(d,mode=0o700,exist_ok=True)\n' +
    'df=os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0)\n' +
    'dd=os.open(d,df)\n' +
    'try:\n' +
    ' s=os.fstat(dd)\n' +
    ' if not stat.S_ISDIR(s.st_mode):raise SystemExit("unsafe token directory")\n' +
    ' if hasattr(os,"getuid") and s.st_uid!=os.getuid():raise SystemExit("token directory owner mismatch")\n' +
    ' if (s.st_mode&0o777)!=0o700:os.fchmod(dd,0o700)\n' +
    ' fl=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0)\n' +
    ' now=__import__("time").time()\n' +
    ' for stale in os.listdir(dd):\n' +
    '  if stale.endswith(".token") and len(stale)==22:\n' +
    '   try:\n' +
    '    ss=os.stat(stale,dir_fd=dd,follow_symlinks=False)\n' +
    '    if stat.S_ISREG(ss.st_mode) and now-ss.st_mtime>3600:os.unlink(stale,dir_fd=dd)\n' +
    '   except OSError:pass\n' +
    ' fd=os.open(n,fl,0o600,dir_fd=dd)\n' +
    ' try:os.write(fd,sys.stdin.buffer.read())\n' +
    ' except BaseException:\n' +
    '  try:os.unlink(n,dir_fd=dd)\n' +
    '  except OSError:pass\n' +
    '  raise\n' +
    ' finally:os.close(fd)\n' +
    'finally:os.close(dd)'

  try {
    await ssh.exec(`python3 -c ${shq(tokenUploadPy)}`, { stdinData: token })
  } catch (error) {
    try {
      await ssh.exec(`rm -f ${expandRemotePath(tokenFilePath)}`)
    } catch {
      void 0
    }

    throw error
  }

  let out

  try {
    out = await ssh.exec(buildSpawnCommand(hermesPath, profile, { spawnNonce, tokenFilePath, logPath }))
  } catch (error) {
    try {
      await ssh.exec(`rm -f ${expandRemotePath(tokenFilePath)}`)
    } catch {
      void 0
    }

    throw error
  }

  const pid = parseInt(
    String(out || '')
      .trim()
      .split('\n')
      .pop(),
    10
  )

  if (!Number.isInteger(pid) || pid <= 0) {
    try {
      await ssh.exec(`rm -f ${expandRemotePath(tokenFilePath)}`)
    } catch {
      void 0
    }

    const err: any = new Error('Failed to launch the remote dashboard (no pid returned).')
    err.kind = 'spawn-failed'
    throw err
  }

  return { pid, spawnNonce, logPath, tokenFilePath }
}

// Best-effort forward teardown when a reuse attempt fails mid-flight, so we
// don't leak a forward before respawning. `deps.cancelForward` is optional.
async function cancelForwardSafe(deps, localPort, remotePort) {
  if (typeof deps.cancelForward !== 'function') {
    return
  }

  try {
    await deps.cancelForward(localPort, remotePort)
  } catch {
    // best effort
  }
}

function assertNotAborted(signal) {
  if (signal?.aborted) {
    const error: any = new Error('SSH bootstrap was cancelled.')
    error.kind = 'superseded'
    throw error
  }
}

function isForwardBindCollision(error) {
  return /address already in use|cannot listen to port|bind.*failed/i.test(String(error?.message || error || ''))
}

async function openForward(deps, remotePort, attempts = 3) {
  let lastError

  for (let attempt = 0; attempt < attempts; attempt++) {
    const localPort = await deps.pickLocalPort()

    try {
      await deps.forward(localPort, remotePort)

      return localPort
    } catch (error) {
      lastError = error

      if (!isForwardBindCollision(error) || attempt === attempts - 1) {
        throw error
      }
    }
  }

  throw lastError
}

/**
 * Establish (or reuse) a remote dashboard and a tunnel to it. `deps` injects the
 * opened SshConnection, forward/pickLocalPort/waitForHermes, a token-gated
 * probeReuseProof, and adoptServedToken. Returns the connection descriptor
 * { baseUrl, token, tokenFingerprint, remotePort, localPort, pid, reused, platform }.
 */
async function adoptOwnedServedToken(adoptServedToken, baseUrl, expectedToken, ssh, pid, label) {
  const token = await adoptServedToken(baseUrl, expectedToken, {
    childAlive: () => true,
    label
  })

  if (!(await remotePidAlive(ssh, pid))) {
    const error: any = new Error(`${label} exited while its served token was being resolved.`)
    error.kind = token === expectedToken ? 'spawn-failed' : 'foreign-backend'
    throw error
  }

  return token
}

async function connect(deps) {
  const {
    ssh,
    profile = '',
    remoteHermesPath = '',
    ownershipId,
    forward,
    pickLocalPort,
    waitForHermes,
    probeReuseProof,
    adoptServedToken,
    rememberLog = () => {},
    readyTimeoutMs = DEFAULT_READY_TIMEOUT_MS,
    signal
  } = deps

  const log = msg => rememberLog(`[ssh-lifecycle] ${msg}`)

  assertNotAborted(signal)
  const platform = await probeRemotePlatform(ssh)
  log(`remote platform ${platform.os}/${platform.arch}`)
  const hermesPath = await locateHermes(ssh, remoteHermesPath)
  log(`located hermes at ${hermesPath}`)
  const hermesVersion = await probeHermesVersion(ssh, hermesPath)

  if (hermesVersion) {
    log(`remote hermes version: ${hermesVersion}`)
  }

  const reuseToken = deps.reuseToken || ''
  const hermesHome = await probeRemoteHermesHome(ssh)
  const lock = await readLockfile(ssh, ownershipId)

  if (lock) {
    const pidAlive = await remotePidAlive(ssh, lock.pid)
    const owned = pidAlive && (await pidIsOurDashboard(ssh, lock.pid, lock.spawnNonce, lock.hermesPath))

    const reusable =
      pidAlive &&
      owned &&
      lock.port > 0 &&
      Boolean(reuseToken) &&
      lock.tokenFingerprint === fingerprintToken(reuseToken) &&
      lock.hermesPath === hermesPath &&
      lock.hermesHome === hermesHome

    if (reusable) {
      assertNotAborted(signal)
      const localPort = await openForward(deps, lock.port)

      try {
        const baseUrl = `http://127.0.0.1:${localPort}`
        let reuseClassification

        try {
          reuseClassification = await probeReuseProof(baseUrl, reuseToken, lock.spawnNonce)
        } catch (cause) {
          const error: any = new Error('Could not verify the existing SSH backend.')
          error.kind = 'transient-transport-error'
          error.cause = cause
          throw error
        }

        if (reuseClassification === 'authenticated-stale') {
          assertNotAborted(signal)
          await cancelForwardSafe(deps, localPort, lock.port)
          await cleanupStale(ssh, ownershipId, lock)
        } else if (reuseClassification === 'authenticated-ok') {
          const token = await adoptOwnedServedToken(
            adoptServedToken,
            baseUrl,
            reuseToken,
            ssh,
            lock.pid,
            'reused remote dashboard'
          )

          assertNotAborted(signal)
          log(`reusing remote dashboard pid=${lock.pid} port=${lock.port}`)

          return {
            baseUrl,
            token,
            tokenFingerprint: fingerprintToken(token),
            remotePort: lock.port,
            localPort,
            pid: lock.pid,
            reused: true,
            platform,
            hermesPath,
            hermesVersion,
            ownershipId,
            spawnNonce: lock.spawnNonce,
            logPath: lock.logPath
          }
        } else {
          const error: any = new Error('SSH reuse proof returned an invalid classification.')
          error.kind = 'transient-transport-error'
          throw error
        }
      } catch (error) {
        await cancelForwardSafe(deps, localPort, lock.port)
        throw error
      }
    } else {
      assertNotAborted(signal)
      await cleanupStale(ssh, ownershipId, lock, pidAlive)
    }
  }

  assertNotAborted(signal)
  const spawnToken = mintToken()

  const { pid, spawnNonce, logPath, tokenFilePath } = await spawnRemoteDashboard(ssh, {
    hermesPath,
    profile,
    token: spawnToken,
    ownershipId
  })

  log(`spawned remote dashboard pid=${pid}`)

  const ownedSpawn = {
    ownershipId,
    spawnNonce,
    pid,
    port: 0,
    profile,
    hermesPath,
    hermesHome,
    logPath,
    tokenFingerprint: fingerprintToken(spawnToken),
    protocolVersion: PROTOCOL_VERSION,
    startedAt: new Date().toISOString()
  }

  let localPort = 0
  let remotePort = 0

  try {
    // Write the ownership record IMMEDIATELY (port=0): a supersede between
    // spawn and readiness whose cleanup cannot reach the box must not leave a
    // lockless orphan — the next connect reaps it by exact ownership via this
    // record. Inside the try: if this write itself fails, the catch still
    // kills the just-spawned process via the in-memory record.
    await writeLockfile(ssh, ownershipId, ownedSpawn)
    remotePort = await scrapeReadyPort(ssh, logPath, {
      timeoutMs: readyTimeoutMs,
      isAlive: () => remotePidAlive(ssh, pid),
      signal
    })
    assertNotAborted(signal)
    log(`remote dashboard bound port ${remotePort}`)

    localPort = await openForward(deps, remotePort)
    assertNotAborted(signal)
    const baseUrl = `http://127.0.0.1:${localPort}`
    await waitForHermes(baseUrl, spawnToken)
    assertNotAborted(signal)

    const token = await adoptOwnedServedToken(adoptServedToken, baseUrl, spawnToken, ssh, pid, 'remote dashboard')

    assertNotAborted(signal)
    const tokenFingerprint = fingerprintToken(token)
    await writeLockfile(ssh, ownershipId, { ...ownedSpawn, port: remotePort, tokenFingerprint })
    assertNotAborted(signal)

    return {
      baseUrl,
      token,
      tokenFingerprint,
      remotePort,
      localPort,
      pid,
      reused: false,
      platform,
      hermesPath,
      hermesVersion,
      ownershipId,
      spawnNonce,
      logPath
    }
  } catch (error) {
    if (localPort && remotePort) {
      await cancelForwardSafe(deps, localPort, remotePort)
    }

    try {
      await ssh.exec(`rm -f ${expandRemotePath(tokenFilePath)}`)
    } catch {
      void 0
    }

    await cleanupStale(ssh, ownershipId, ownedSpawn)
    throw error
  }
}

export {
  adoptOwnedServedToken,
  buildSpawnCommand,
  cleanupStale,
  connect,
  DEFAULT_READY_TIMEOUT_MS,
  expandRemotePath,
  fingerprintToken,
  isForwardBindCollision,
  locateHermes,
  LOCKFILE_SCHEMA_VERSION,
  lockfilePath,
  mintToken,
  openForward,
  ownershipDirectory,
  pidIsOurDashboard,
  probeHermesVersion,
  probeRemoteHermesHome,
  probeRemotePlatform,
  PROTOCOL_VERSION,
  readLockfile,
  READY_RE,
  REMOTE_LOCK_DIR,
  remotePidAlive,
  remoteSupportsSshOwnership,
  removeLockfile,
  scrapeReadyPort,
  shq,
  spawnLogPath,
  spawnRemoteDashboard,
  SUPPORTED_REMOTE_OS,
  validateRemotePath,
  writeLockfile
}
