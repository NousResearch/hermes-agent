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

import { parseRemoteProfileListing } from './connection-registry'
import { assertBootstrapNotSuperseded } from './ssh-connection'

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
// macOS sshd starts non-interactive shells with a 256-FD soft limit even when
// the hard limit is unlimited. A Desktop backend can legitimately exceed that
// while serving several profiles/tools, so raise only the child process limit.
// Keep startup portable: restricted hosts retain their existing limit.
const REMOTE_NOFILE_SOFT_LIMIT = 65_536

function classifySshReuseProof(proof, spawnNonce) {
  return proof?.ok === true &&
    proof.sshOwnerNonce === spawnNonce &&
    proof.protocolVersion === PROTOCOL_VERSION &&
    proof.runtimeIntact !== false
    ? 'authenticated-ok'
    : 'authenticated-stale'
}

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

// #95532 fail-closed skew sentinel. A backend.lock.json that EXISTS but does
// not match what this build writes (unknown schemaVersion, missing/foreign
// ownershipId, truncated JSON, malformed shape) is "skew" — most likely a
// different desktop build (fork) owns this remote, or the file is corrupt.
// Skew must never be conflated with "no lockfile": every reap/cleanup path
// (#78872 ownership guard) must SKIP on skew, because killing or overwriting
// on unparseable/foreign state is exactly the wrong-way failure — it murders
// a live tunnel some other build is depending on.
function lockfileSkew(reason) {
  return { skew: true, reason: String(reason) }
}

function isLockfileSkew(lock) {
  return Boolean(lock) && (lock as any).skew === true
}

function connectReservationPath(ownershipId) {
  return `${ownershipDirectory(ownershipId)}/.connect.lock`
}

function spawnLogPath(ownershipId, spawnNonce) {
  return `${ownershipDirectory(ownershipId)}/${validateSpawnNonce(spawnNonce)}.log`
}

function spawnTokenPath(ownershipId, spawnNonce) {
  return `${ownershipDirectory(ownershipId)}/${validateSpawnNonce(spawnNonce)}.token`
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
    // Return the candidate path directly. The hermes binary or wrapper script
    // is executable and handles argument forwarding (e.g. `exec <python> <script> "$@"`)
    // correctly on its own. Previously, this function followed `exec` wrappers and
    // returned only the python interpreter, which broke:
    //   - version checking: `<python> --version` printed "Python x.y.z" instead of
    //     the Hermes version, and
    //   - capability probing: `<python> serve --help` failed entirely.
    // See https://github.com/NousResearch/hermes-agent/issues/74411
    return candidate
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

const REMOTE_UPDATE_MARKER_PROBE = String.raw`
import errno,os,re,sys
from pathlib import Path

home=Path(os.path.expanduser(sys.argv[1]))
if home.parent.name=='profiles':home=home.parent.parent
marker=home/'.hermes-update-in-progress'
try:
    with marker.open('rb') as stream:raw=stream.read(257)
except FileNotFoundError:
    print('CLEAR');raise SystemExit
except OSError:
    print('UNCERTAIN');raise SystemExit
if len(raw)>256:
    print('UNCERTAIN');raise SystemExit
match=re.fullmatch(rb'([1-9][0-9]*)\r?\n([0-9]+)(?:\r?\n)?',raw)
if not match:
    print('UNCERTAIN');raise SystemExit
try:
    owner=int(match.group(1));lease=int(match.group(2))
    if owner<1 or owner>4294967295 or lease>9007199254740991:raise ValueError()
except ValueError:
    print('UNCERTAIN');raise SystemExit
try:
    os.kill(owner,0)
except ProcessLookupError:
    print('CLEAR')
except PermissionError:
    print('LIVE:'+str(owner))
except OSError as error:
    if error.errno==errno.ESRCH:print('CLEAR')
    elif error.errno==errno.EPERM:print('LIVE:'+str(owner))
    else:print('UNCERTAIN')
else:
    print('LIVE:'+str(owner))
`

/**
 * Refuse normal SSH reuse/spawn while the remote install is being mutated.
 *
 * This probe intentionally uses only the host's system Python and raw marker
 * bytes; it never imports or executes code from the changing Hermes checkout.
 * Absence or a well-formed, confirmed-dead owner is clear. Every parse, read,
 * probe, or transport uncertainty fails closed so a Desktop relaunch cannot
 * start `serve` beside an updater that survived the old app process.
 */
async function assertRemoteInstallUpdateClear(ssh, hermesHome) {
  const home = assertSafeRemoteHome(hermesHome)
  let observation = ''

  try {
    observation =
      String(await ssh.exec(`python3 -c ${shq(REMOTE_UPDATE_MARKER_PROBE)} ${expandRemotePath(home)}`))
        .trim()
        .split(/\r?\n/)
        .pop() || ''
  } catch (cause) {
    const error: any = new Error('Could not prove that the remote Hermes install is clear for SSH startup.')
    error.kind = 'update-in-progress'
    error.cause = cause
    throw error
  }

  if (observation === 'CLEAR') {
    return
  }

  const live = /^LIVE:([1-9][0-9]*)$/.exec(observation)

  const error: any = new Error(
    live
      ? `Remote Hermes update process ${live[1]} is still running; SSH startup is paused.`
      : 'The remote Hermes update marker is unreadable or malformed; refusing SSH startup.'
  )

  error.kind = 'update-in-progress'
  throw error
}

async function listRemoteHermesProfiles(ssh) {
  const home = assertSafeRemoteHome(await probeRemoteHermesHome(ssh))
  const dir = expandRemotePath(`${home}/profiles`)
  let listing = ''

  try {
    listing = await ssh.exec(`if [ -d ${dir} ]; then ls -1 ${dir}; fi`)
  } catch (cause) {
    const error: any = new Error('Could not list remote Hermes profiles.')
    error.kind = 'transient-transport-error'
    error.cause = cause
    throw error
  }

  return parseRemoteProfileListing(listing)
}

function assertSafeRemoteHome(home) {
  const value = String(home || '').trim()

  if (!/^(\/|~\/)[A-Za-z0-9._/+-]+$/.test(value) || value.includes('..')) {
    const error: any = new Error('Unsafe remote Hermes home.')
    error.kind = 'unsafe-path'
    throw error
  }

  return value.replace(/\/+$/, '')
}

function remoteInstallRoot(home) {
  const value = assertSafeRemoteHome(home)
  const profile = value.match(/^(.*)\/profiles\/[^/]+$/)

  return profile ? profile[1] : value
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
    // Exists but doesn't parse: truncated write or a foreign format. NOT the
    // same as "no lockfile" — see lockfileSkew().
    return lockfileSkew('unparseable-json')
  }

  if (!parsed || typeof parsed !== 'object') {
    return lockfileSkew('non-object')
  }

  if (parsed.schemaVersion !== LOCKFILE_SCHEMA_VERSION) {
    return lockfileSkew(`schema-version ${JSON.stringify(parsed.schemaVersion ?? null)}`)
  }

  const pid = parsed.pid
  const port = parsed.port

  if (!Number.isInteger(pid) || pid <= 0 || pid > 4194304) {
    return lockfileSkew('malformed-pid')
  }

  // port 0 = spawn-in-progress record (written before readiness); valid
  // ownership proof for cleanup, but never reusable.
  if (!Number.isInteger(port) || port < 0 || port > 65535) {
    return lockfileSkew('malformed-port')
  }

  if (parsed.ownershipId !== ownershipId || !/^[0-9a-f]{16}$/.test(parsed.spawnNonce || '')) {
    return lockfileSkew('ownership-mismatch')
  }

  if (!/^[0-9a-f]{32}$/.test(parsed.tokenFingerprint || '')) {
    return lockfileSkew('malformed-token-fingerprint')
  }

  if (parsed.protocolVersion !== PROTOCOL_VERSION) {
    // Fully validated ownership (our schema, our ownershipId, our shape) from
    // a protocol-incompatible build of OUR OWN lineage: the record is not
    // reusable and readLockfile keeps its historical contract of hiding it,
    // which routes connect() to a fresh spawn.
    return null
  }

  if (parsed.logPath !== spawnLogPath(ownershipId, parsed.spawnNonce)) {
    return lockfileSkew('log-path-mismatch')
  }

  for (const field of ['profile', 'hermesPath', 'hermesHome', 'logPath', 'startedAt']) {
    if (typeof parsed[field] !== 'string' || parsed[field].length > 1024) {
      return lockfileSkew(`malformed-field ${field}`)
    }
  }

  if (
    parsed.creationTime !== undefined &&
    (typeof parsed.creationTime !== 'string' || !/^(?:linux:[0-9]+|darwin:[A-Za-z0-9 :+-]+)$/.test(parsed.creationTime))
  ) {
    return null
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

// Stable kernel process-start identity used to fence a later managed-update
// termination against PID recycling. Linux exposes boot-relative start ticks;
// Darwin's `ps lstart` is only second-resolution, so the signal boundary also
// re-reads the complete argv and random ownership nonce in the same remote
// command. A same-second PID reuse with a forged/repeated nonce remains a
// residual limitation. Failure is represented as an empty string so SSH mode
// remains compatible on unusual POSIX hosts, but a managed update then refuses
// to kill that unproved serve.
async function remoteProcessCreationTime(ssh, pid) {
  if (!pid || !Number.isInteger(Number(pid))) {
    return ''
  }

  const script =
    'import subprocess,sys\n' +
    `pid=${Number(pid)}\n` +
    'value=""\n' +
    'if sys.platform.startswith("linux"):\n' +
    ' try:\n' +
    '  raw=open(f"/proc/{pid}/stat","r",encoding="ascii").read()\n' +
    '  fields=raw[raw.rfind(")")+2:].split()\n' +
    '  value="linux:"+fields[19]\n' +
    ' except (OSError,IndexError,UnicodeError):pass\n' +
    'elif sys.platform=="darwin":\n' +
    ' try:\n' +
    '  started=subprocess.check_output(["ps","-o","lstart=","-p",str(pid)],text=True).strip()\n' +
    '  if started:value="darwin:"+started\n' +
    ' except (OSError,subprocess.CalledProcessError):pass\n' +
    'print(value)'

  try {
    const value = String(await ssh.exec(`python3 -c ${shq(script)}`)).trim()

    return /^(?:linux:[0-9]+|darwin:[A-Za-z0-9 :+-]+)$/.test(value) ? value : ''
  } catch {
    return ''
  }
}

// A pid is "provably ours" only if its remote cmdline carries our dashboard
// args — never kill a pid we can't positively identify as our dashboard.
async function pidIsOurDashboard(
  ssh,
  pid,
  spawnNonce,
  hermesPath = '',
  hermesHome = '',
  ownershipId = '',
  profile = ''
) {
  if (!pid || !/^[0-9a-f]{16}$/.test(String(spawnNonce || '')) || !hermesPath) {
    return false
  }

  try {
    const script =
      'import os,shlex,subprocess,sys\n' +
      `pid=${Number(pid)}\n` +
      `expected=os.path.expanduser(${shq(hermesPath)})\n` +
      // The installer-facing launcher is intentionally preserved for invocation
      // (#74411), but it may `exec python <install-dir>/hermes`, leaving neither
      // launcher nor HERMES_HOME-derived entrypoint in argv. The ownership-scoped
      // token path + random nonce + exact profile below are the alternative proof.
      `hermes_home=os.path.expanduser(${shq(hermesHome)}) if ${shq(hermesHome)} else ""\n` +
      'expected_entries={expected}\n' +
      'if hermes_home:\n' +
      ' expected_entries.add(os.path.join(hermes_home,"hermes-agent","venv","bin","hermes"))\n' +
      `expected_token=os.path.expanduser(${shq(ownershipId ? spawnTokenPath(ownershipId, spawnNonce) : '')})\n` +
      `expected_profile=${shq(profile)}\n` +
      `nonce=${shq(spawnNonce)}\n` +
      'try:\n' +
      ' raw=open(f"/proc/{pid}/cmdline","rb").read()\n' +
      ' args=[x.decode("utf-8","surrogateescape") for x in raw.split(b"\\0") if x]\n' +
      'except OSError:\n' +
      ' try:\n' +
      '  line=subprocess.check_output(["ps","-ww","-o","command=","-p",str(pid)],text=True).strip()\n' +
      ' except subprocess.CalledProcessError:\n' +
      '  # pid already gone — a dead process is FOREIGN, not a transport error\n' +
      '  print("FOREIGN");sys.exit(0)\n' +
      ' args=shlex.split(line)\n' +
      'ok=False\n' +
      'try:\n' +
      ' serve=args.index("serve")\n' +
      ' owner=args.index("--ssh-owner-nonce",serve+1)\n' +
      ' token=args.index("--ssh-session-token-file",serve+1) if expected_token else -1\n' +
      ' isolated=args.index("--isolated",serve+1)\n' +
      ' profile_arg=args.index("--profile") if expected_profile else -1\n' +
      ' serve_count=args.count("serve")\n' +
      ' owner_count=args.count("--ssh-owner-nonce")\n' +
      ' token_count=args.count("--ssh-session-token-file")\n' +
      ' isolated_count=args.count("--isolated")\n' +
      ' profile_count=args.count("--profile")\n' +
      ' direct=args[0] in expected_entries\n' +
      ' python_entry=len(args)>1 and args[1] in expected_entries and os.path.basename(args[0]).startswith("python")\n' +
      ' token_ok=not expected_token or args[token+1]==expected_token\n' +
      ' isolated_ok=isolated_count==1 and isolated>serve\n' +
      ' profile_ok=(profile_count==1 and profile_arg<serve and args[profile_arg+1]==expected_profile) if expected_profile else profile_count==0\n' +
      ' spawn_proof=bool(expected_token) and owner_count==1 and token_count==1 and token_ok and profile_ok\n' +
      ' ok=(direct or python_entry or spawn_proof) and serve_count==1 and isolated_ok and owner_count==1 and args[owner+1]==nonce and token_ok and profile_ok\n' +
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
  // Defense in depth (#95532): a skew sentinel is foreign/corrupt state, not
  // an ownership record — never reap or remove anything based on it.
  if (isLockfileSkew(lock)) {
    return
  }

  if (
    pidAlive &&
    lock &&
    (await pidIsOurDashboard(
      ssh,
      lock.pid,
      lock.spawnNonce,
      lock.hermesPath,
      lock.hermesHome,
      ownershipId,
      lock.profile
    ))
  ) {
    try {
      const result = (
        await ssh.exec(
          `kill ${Number(lock.pid)} && ` +
            `i=0; while kill -0 ${Number(lock.pid)} 2>/dev/null; do ` +
            `i=$((i+1)); [ "$i" -ge 50 ] && exit 1; sleep 0.1; done`
        )
      ).trim()

      void result
    } catch {
      // A backend mid-turn (in-flight LLM call, live MCP children) can ride
      // out SIGTERM past the 5s graceful wait — and before-quit races this
      // whole teardown against 6s before closing SSH, so giving up here
      // reparents the still-running serve to pid 1: the #91668 leak, now on
      // the quit-during-active-turn path. Escalate to SIGKILL and require a
      // confirmed exit before treating the record as reclaimed.
      try {
        await ssh.exec(
          `kill -9 ${Number(lock.pid)} 2>/dev/null; ` +
            `i=0; while kill -0 ${Number(lock.pid)} 2>/dev/null; do ` +
            `i=$((i+1)); [ "$i" -ge 20 ] && exit 1; sleep 0.1; done`
        )
      } catch (cause) {
        // Even SIGKILL could not confirm death (D-state, permissions). Keep
        // the lockfile so the next connect's reap pass retries.
        const error: any = new Error('Could not terminate the stale SSH backend.')
        error.kind = 'transient-transport-error'
        error.cause = cause
        throw error
      }
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

// Normal disconnect (quit, connection switch): reuse cleanupStale so we
// kill only a provably-owned serve --isolated and drop our lockfile.
// Closing the SSH transport first is not enough — spawn detaches with
// setsid/nohup, so the backend reparents to pid 1 and keeps state.db
// open (#91668).
async function disconnect(ssh, ownershipId) {
  if (!ssh || !ownershipId) {
    return
  }

  const lock = await readLockfile(ssh, ownershipId)

  if (!lock || isLockfileSkew(lock)) {
    // Skew (#95532): fail closed — this is not our record, so there is
    // nothing we may safely reap or remove here.
    return
  }

  const pidAlive = await remotePidAlive(ssh, lock.pid)
  await cleanupStale(ssh, ownershipId, lock, pidAlive)
}

function buildOwnedStaleTerminationCommand(lock, ownershipId) {
  const pid = Number(lock.pid)
  // expandRemotePath() output is already a shell-quoted fragment; embed it
  // raw so $HOME expands at assignment. Double-quoting stores the quote
  // characters in the variable and every identity match below REFUSEs.
  const expectedPath = expandRemotePath(lock.hermesPath)
  const expectedHome = lock.hermesHome ? expandRemotePath(lock.hermesHome) : "''"
  const expectedToken = expandRemotePath(spawnTokenPath(ownershipId, lock.spawnNonce))
  const nonce = shq(lock.spawnNonce)
  const profile = shq(lock.profile || '')
  const command = `$(ps -ww -o command= -p ${pid} 2>/dev/null || true)`

  const executableMatch = lock.hermesHome
    ? `case "$cmd" in *"$path"*|*"$home"*) ;; *) printf REFUSED; exit 0;; esac; `
    : `case "$cmd" in *"$path"*) ;; *) printf REFUSED; exit 0;; esac; `

  const identity =
    `cmd=${command}; ` +
    `path=${expectedPath}; home=${expectedHome}; token=${expectedToken}; nonce=${nonce}; profile=${profile}; ` +
    executableMatch +
    `case "$cmd" in *" serve"*|*" serve "*) ;; *) printf REFUSED; exit 0;; esac; ` +
    `case "$cmd" in *"--ssh-owner-nonce $nonce"*) ;; *) printf REFUSED; exit 0;; esac; ` +
    `case "$cmd" in *"--ssh-session-token-file $token"*) ;; *) printf REFUSED; exit 0;; esac; ` +
    `[ -n "$profile" ] && case "$cmd" in *"--profile $profile"*) ;; *) printf REFUSED; exit 0;; esac; `

  // Legacy records do not have creationTime. Re-read argv immediately before
  // signaling in this same shell command; never use the earlier probe's PID
  // verdict as authority for the kill.
  return (
    `${identity} kill ${pid} && ` +
    `i=0; while kill -0 ${pid} 2>/dev/null; do ` +
    `i=$((i+1)); [ "$i" -ge 50 ] && printf TIMEOUT && exit 0; sleep 0.1; done; printf TERMINATED`
  )
}

function lockMatchesManagedUpdateScope(lock, expected) {
  return Boolean(
    lock &&
    expected &&
    lock.ownershipId === expected.ownershipId &&
    lock.pid === expected.pid &&
    lock.spawnNonce === expected.spawnNonce &&
    lock.startedAt === expected.startedAt &&
    lock.creationTime === expected.creationTime &&
    lock.profile === expected.profile &&
    lock.hermesPath === expected.hermesPath &&
    lock.hermesHome === expected.hermesHome
  )
}

function buildOwnedTerminationCommand(lock, ownershipId) {
  const pid = Number(lock.pid)
  const py = value => JSON.stringify(String(value || ''))
  const expectedToken = spawnTokenPath(ownershipId, lock.spawnNonce)

  const script = `
import os,select,shlex,signal,subprocess,sys,time
pid=${pid}
expected_creation=${py(lock.creationTime)}
expected_path=os.path.expanduser(${py(lock.hermesPath)})
hermes_home=os.path.expanduser(${py(lock.hermesHome)})
expected_entries={expected_path,os.path.join(hermes_home,"hermes-agent","venv","bin","hermes")}
expected_token=os.path.expanduser(${py(expectedToken)})
expected_profile=${py(lock.profile)}
nonce=${py(lock.spawnNonce)}

def creation():
 if sys.platform.startswith("linux"):
  try:
   raw=open(f"/proc/{pid}/stat","r",encoding="ascii").read()
   return "linux:"+raw[raw.rfind(")")+2:].split()[19]
  except (OSError,IndexError,UnicodeError):return ""
 if sys.platform=="darwin":
  try:
   value=subprocess.check_output(["ps","-o","lstart=","-p",str(pid)],text=True).strip()
   return "darwin:"+value if value else ""
  except (OSError,subprocess.CalledProcessError):return ""
 return ""

def argv():
 try:
  raw=open(f"/proc/{pid}/cmdline","rb").read()
  return [part.decode("utf-8","surrogateescape") for part in raw.split(b"\\0") if part]
 except OSError:
  try:return shlex.split(subprocess.check_output(["ps","-ww","-o","command=","-p",str(pid)],text=True).strip())
  except (OSError,subprocess.CalledProcessError,ValueError):return []

def identity_before_signal():
 # Darwin's lstart has one-second resolution. Read the start time and the
 # complete argv in one ps call immediately before signalling; the random
 # ownership nonce is the discriminator for a same-second PID reuse. A
 # same-second reuse with a forged/repeated nonce remains a residual limitation.
 if sys.platform=="darwin":
  try:
   line=subprocess.check_output(["ps","-ww","-p",str(pid),"-o","lstart=","-o","command="],text=True).strip()
   prefix=expected_creation.removeprefix("darwin:")
   if not prefix or not line.startswith(prefix):return "",[]
   return "darwin:"+prefix,shlex.split(line[len(prefix):].strip())
  except (OSError,subprocess.CalledProcessError,ValueError):return "",[]
 return creation(),argv()

def owned(args):
 try:
  serve=args.index("serve")
  owner=args.index("--ssh-owner-nonce",serve+1)
  token=args.index("--ssh-session-token-file",serve+1)
  isolated=args.index("--isolated",serve+1)
  profile_arg=args.index("--profile") if expected_profile else -1
  direct=args[0] in expected_entries
  python_entry=len(args)>1 and args[1] in expected_entries and os.path.basename(args[0]).startswith("python")
  profile_ok=(args.count("--profile")==1 and profile_arg<serve and args[profile_arg+1]==expected_profile) if expected_profile else args.count("--profile")==0
  return ((direct or python_entry or (args[token+1]==expected_token and profile_ok)) and
          args.count("serve")==1 and args.count("--ssh-owner-nonce")==1 and
          args.count("--ssh-session-token-file")==1 and args.count("--isolated")==1 and
          isolated>serve and args[owner+1]==nonce and args[token+1]==expected_token and profile_ok)
 except (ValueError,IndexError):return False

pidfd=None
if sys.platform.startswith("linux"):
 if not hasattr(os,"pidfd_open") or not hasattr(signal,"pidfd_send_signal"):
  print("UNAVAILABLE");sys.exit(2)
 try:pidfd=os.pidfd_open(pid,0)
 except ProcessLookupError:print("ALREADY_STOPPED");sys.exit(0)
 except (OSError,PermissionError):print("UNAVAILABLE");sys.exit(2)

try:
 live_creation,live_args=identity_before_signal()
 if live_creation!=expected_creation or not owned(live_args):
  print("REFUSED");sys.exit(3)
 if (sys.platform=="darwin"):
  # Darwin has no pidfd-style signal binding. Refuse instead of accepting the
  # residual PID-reuse window between ps and os.kill; reconnect will surface
  # the still-running remote owner for an explicit retry.
  print("DARWIN_UNAVAILABLE");sys.exit(2)
 try:
  if pidfd is not None:signal.pidfd_send_signal(pidfd,signal.SIGTERM)
  else:os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:print("ALREADY_STOPPED");sys.exit(0)
 if pidfd is not None:
  poller=select.poll();poller.register(pidfd,select.POLLIN)
  if not poller.poll(10000):print("TIMEOUT");sys.exit(4)
 else:
  deadline=time.monotonic()+10
  while time.monotonic()<deadline:
   try:os.kill(pid,0)
   except ProcessLookupError:break
   except PermissionError:print("UNAVAILABLE");sys.exit(2)
   time.sleep(.1)
  else:print("TIMEOUT");sys.exit(4)
 print("TERMINATED")
finally:
 if pidfd is not None:os.close(pidfd)
`.trim()

  return `python3 -c ${shq(script)}`
}

// The updater's Python _MarkerMutex uses the marker's .mutex sidecar and an
// advisory flock. Keep that same descriptor locked while the remote shell does
// the marker check, spawns the backend, and publishes its initial lockfile.
// Python keeps the descriptor close-on-exec by default and passes it explicitly
// only to the intended outer shell; each detached child closes it before
// execing Hermes.
function withRemoteUpdateMutex(command, mutexPath) {
  const script = `
import fcntl,os,subprocess,sys
mutex_path=sys.argv[1]
payload=sys.argv[2]
parent=os.path.dirname(mutex_path)
if parent:os.makedirs(parent,exist_ok=True)
fd=os.open(mutex_path,os.O_RDWR|os.O_CREAT|os.O_CLOEXEC,0o600)
fcntl.flock(fd,fcntl.LOCK_EX)
result=None
try:
 result=subprocess.run(["sh","-c",payload,"hermes-update-mutex",str(fd)],pass_fds=(fd,),check=False)
finally:
 os.close(fd)
sys.exit(result.returncode if result is not None else 1)
`.trim()

  // mutexPath is an expandRemotePath fragment — already quoted; do not shq again.
  return `python3 -c ${shq(script)} ${mutexPath} ${shq(command)}`
}

/**
 * Stop one Desktop-owned POSIX serve before an install update.
 *
 * This is deliberately stricter than stale cleanup. The in-memory scope is a
 * snapshot of the ownership record that established the forward; immediately
 * before signalling we re-read that record, compare its PID + kernel creation
 * identity and random argv nonce, and prove the live argv is the exact
 * isolated serve Desktop launched. Any absence, parse failure, replacement,
 * or transport uncertainty refuses the kill. The lock is intentionally left
 * behind: the post-update reconnect reclaims the now-dead exact record, while
 * an old cleanup can never unlink a replacement owner.
 */
async function terminateOwnedDashboardForUpdate(ssh, expected) {
  const ownershipId = validateOwnershipId(expected?.ownershipId)

  if (!expected?.creationTime) {
    const error: any = new Error('The remote POSIX serve has no process creation-time proof.')
    error.kind = 'ownership-changed'
    throw error
  }

  let lock = await readLockfile(ssh, ownershipId)

  if (!lock || !lockMatchesManagedUpdateScope(lock, expected)) {
    const error: any = new Error('The remote POSIX ownership record changed before the managed update.')
    error.kind = 'ownership-changed'
    throw error
  }

  if (!(await remotePidAlive(ssh, lock.pid))) {
    return { pid: lock.pid, terminated: false, alreadyStopped: true }
  }

  if ((await remoteProcessCreationTime(ssh, lock.pid)) !== lock.creationTime) {
    const error: any = new Error('The remote POSIX PID creation time no longer matches its ownership record.')
    error.kind = 'ownership-changed'
    throw error
  }

  if (
    !(await pidIsOurDashboard(
      ssh,
      lock.pid,
      lock.spawnNonce,
      lock.hermesPath,
      lock.hermesHome,
      ownershipId,
      lock.profile
    ))
  ) {
    const error: any = new Error('Refusing to terminate a remote process whose Desktop ownership is unproven.')
    error.kind = 'foreign-backend'
    throw error
  }

  // Re-read after the argv proof. A concurrent/replacement writer cannot turn
  // proof of the old record into authority over its new PID.
  lock = await readLockfile(ssh, ownershipId)

  if (!lock || !lockMatchesManagedUpdateScope(lock, expected)) {
    const error: any = new Error('The remote POSIX ownership record changed during process verification.')
    error.kind = 'ownership-changed'
    throw error
  }

  if (
    (await remoteProcessCreationTime(ssh, lock.pid)) !== lock.creationTime ||
    !(await pidIsOurDashboard(
      ssh,
      lock.pid,
      lock.spawnNonce,
      lock.hermesPath,
      lock.hermesHome,
      ownershipId,
      lock.profile
    ))
  ) {
    const error: any = new Error('The remote POSIX process identity changed during managed update drain.')
    error.kind = 'ownership-changed'
    throw error
  }

  try {
    const result = String(await ssh.exec(buildOwnedTerminationCommand(lock, ownershipId))).trim()

    if (result === 'ALREADY_STOPPED') {
      return { pid: lock.pid, terminated: false, alreadyStopped: true }
    }

    if (result !== 'TERMINATED') {
      const error: any = new Error(
        result === 'REFUSED'
          ? 'The remote POSIX process identity changed at the signal boundary.'
          : result === 'DARWIN_UNAVAILABLE'
            ? 'Darwin cannot atomically bind a signal to the verified PID; refusing termination.'
            : 'The remote POSIX signal boundary was unavailable.'
      )

      error.kind =
        result === 'REFUSED' || result === 'DARWIN_UNAVAILABLE' ? 'ownership-changed' : 'transient-transport-error'
      throw error
    }
  } catch (cause: any) {
    if (cause?.kind === 'ownership-changed') {
      throw cause
    }

    const error: any = new Error('Could not terminate the Desktop-owned remote serve for update.')
    error.kind = 'transient-transport-error'
    error.cause = cause
    throw error
  }

  return { pid: lock.pid, terminated: true, alreadyStopped: false }
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
  const marker = expandRemotePath(`${remoteInstallRoot(opts.hermesHome || '~/.hermes')}/.hermes-update-in-progress`)

  const updateMutex = expandRemotePath(
    `${remoteInstallRoot(opts.hermesHome || '~/.hermes')}/.hermes-update-in-progress.mutex`
  )

  // The marker probe, ownership reservation, process creation, and initial
  // lockfile publication must be one remote command. A second Desktop process
  // can therefore never observe an empty lock and spawn before this one records
  // its PID. The reservation is an atomic mkdir and is reclaimed only when its
  // owning remote shell is dead.
  const markerClear =
    `marker_clear() { if [ ! -e ${marker} ]; then return 0; fi; ` +
    `if [ ! -r ${marker} ]; then return 1; fi; ` +
    `owner=$(IFS= read -r owner < ${marker} && printf '%s' "$owner"); ` +
    `case "$owner" in ''|*[!0-9]*) return 1;; esac; if kill -0 "$owner" 2>/dev/null; then return 1; fi; return 0; }`

  const dashCmd =
    `ulimit -n ${REMOTE_NOFILE_SOFT_LIMIT} 2>/dev/null || true; ` +
    `exec env HERMES_DESKTOP=1 ${hermes} ${profileArgs}${subCmd}`

  const detachedShell = `eval "exec $1>&-"; ${dashCmd} </dev/null >> ${logPath} 2>&1 & echo $!`
  const detachedSpawn = `child=$("$(command -v setsid || echo nohup)" sh -c ${shq(detachedShell)} hermes-update-child "$1" & echo $!)`

  if (!opts.ownershipId || !opts.lockMetadata) {
    return withRemoteUpdateMutex(
      `${markerClear}; marker_clear || exit 75; ` +
        `mkdir -p "$(dirname ${logPath})" && ` +
        `${detachedSpawn}; ` +
        `marker_clear || { kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; exit 75; }; echo "$child"`,
      updateMutex
    )
  }

  const reservation = expandRemotePath(connectReservationPath(opts.ownershipId))
  const lockPath = expandRemotePath(lockfilePath(opts.ownershipId))
  const tokenPath = tokenFilePath ? expandRemotePath(tokenFilePath) : ''
  const ownerPath = `${reservation}/owner`
  const metadata = JSON.stringify({ schemaVersion: LOCKFILE_SCHEMA_VERSION, ...opts.lockMetadata, pid: '__PID__' })
  const reservationNonce = validateSpawnNonce(opts.reservationNonce || crypto.randomBytes(8).toString('hex'))

  return withRemoteUpdateMutex(
    `umask 077 && mkdir -p "$(dirname ${reservation})"; ` +
      // reservation/lockPath/ownerPath are expandRemotePath() output — already
      // shell-quoted fragments ("$HOME"'/…'). Embed raw so the assignment
      // expands $HOME; shq() here would store the quote characters literally
      // and every mkdir/cat against the variable fails forever.
      `reservation=${reservation}; lock=${lockPath}; owner_file=${ownerPath}; ` +
      `reservation_nonce=${shq(reservationNonce)}; ` +
      `i=0; while ! mkdir "$reservation" 2>/dev/null; do ` +
      `owner_data=$(cat "$owner_file" 2>/dev/null || true); owner_pid=${'${owner_data%%:*}'}; ` +
      `case "$owner_pid" in ''|*[!0-9]*) ;; *) kill -0 "$owner_pid" 2>/dev/null || { rm -rf "$reservation"; continue; };; esac; ` +
      `i=$((i+1)); [ "$i" -ge 600 ] && exit 75; sleep 0.05; done; ` +
      `printf '%s:%s' "$$" "$reservation_nonce" > "$owner_file"; ` +
      `trap 'rm -rf "$reservation"' EXIT; ` +
      `if [ -f "$lock" ]; then ` +
      `existing_pid=$(sed -n 's/.*"pid":\\([0-9][0-9]*\\).*/\\1/p' "$lock" | head -n 1); ` +
      `case "$existing_pid" in ''|*[!0-9]*) rm -f "$lock";; *) ` +
      `if kill -0 "$existing_pid" 2>/dev/null; then ${tokenPath ? `rm -f ${tokenPath}; ` : ''}printf EXISTING; exit 0; fi; rm -f "$lock";; esac; fi; ` +
      `${markerClear}; marker_clear || exit 75; mkdir -p "$(dirname ${logPath})" && ` +
      `${detachedSpawn}; ` +
      `marker_clear || { kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; exit 75; }; ` +
      // ${var//pat/rep} is a bashism — this payload runs under plain sh (dash
      // on Ubuntu), which aborts the whole script on it with "Bad
      // substitution" AFTER the child was spawned, orphaning the backend and
      // skipping the lockfile publication. Substitute with sed instead.
      `lock_json=$(printf '%s' ${shq(metadata)} | sed "s/__PID__/\${child}/"); ` +
      `temporary_lock="\${lock}.${reservationNonce}.tmp"; ` +
      `printf '%s' "$lock_json" > "$temporary_lock" && mv -f "$temporary_lock" "$lock" || { kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; exit 76; }; ` +
      `echo "$child"`,
    updateMutex
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
    assertBootstrapNotSuperseded(signal)

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

async function spawnRemoteDashboard(
  ssh,
  { hermesPath, profile, token, ownershipId, hermesHome = '~/.hermes', assertInstallClear = async () => {} }
) {
  if (!(await remoteSupportsSshOwnership(ssh, hermesPath))) {
    const err: any = new Error(
      'The remote Hermes install does not support --ssh-session-token-file and --ssh-owner-nonce. ' +
        'Update Hermes on the remote host to continue using Desktop SSH mode.'
    )

    err.kind = 'update-required'
    throw err
  }

  const spawnNonce = crypto.randomBytes(8).toString('hex')
  const tokenFilePath = spawnTokenPath(ownershipId, spawnNonce)
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
    // Close the marker race after the token-file write and immediately before
    // process creation. The caller's probe imports no changing checkout code.
    await assertInstallClear()
    out = await ssh.exec(
      buildSpawnCommand(hermesPath, profile, {
        spawnNonce,
        tokenFilePath,
        logPath,
        hermesHome,
        ownershipId,
        reservationNonce: spawnNonce,
        lockMetadata: {
          ownershipId,
          spawnNonce,
          port: 0,
          profile,
          hermesPath,
          hermesHome,
          logPath,
          tokenFingerprint: fingerprintToken(token),
          protocolVersion: PROTOCOL_VERSION,
          startedAt: new Date().toISOString()
        }
      })
    )
  } catch (error) {
    try {
      await ssh.exec(`rm -f ${expandRemotePath(tokenFilePath)}`)
    } catch {
      void 0
    }

    throw error
  }

  const outputLines = String(out || '')
    .trim()
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)

  if (outputLines.at(-1) === 'EXISTING') {
    return { existing: true }
  }

  const pid = parseInt(outputLines.at(-1) || '', 10)

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

async function waitForRemoteSpawnCompletion(ssh, ownershipId, timeoutMs) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    const lock = await readLockfile(ssh, ownershipId)

    if (!lock) {
      return false
    }

    if (lock.port > 0) {
      return true
    }

    await new Promise(resolve => setTimeout(resolve, READY_POLL_INTERVAL_MS))
  }

  const error: any = new Error('Timed out waiting for the concurrent SSH connection to publish its backend.')
  error.kind = 'spawn-failed'
  throw error
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

  assertBootstrapNotSuperseded(signal)
  const platform = await probeRemotePlatform(ssh)
  log(`remote platform ${platform.os}/${platform.arch}`)
  const hermesHome = await probeRemoteHermesHome(ssh)
  await assertRemoteInstallUpdateClear(ssh, hermesHome)
  const hermesPath = await locateHermes(ssh, remoteHermesPath)
  log(`located hermes at ${hermesPath}`)
  const hermesVersion = await probeHermesVersion(ssh, hermesPath)

  if (hermesVersion) {
    log(`remote hermes version: ${hermesVersion}`)
  }

  const reuseToken = deps.reuseToken || ''
  const lock = await readLockfile(ssh, ownershipId)

  if (isLockfileSkew(lock)) {
    // #95532: the lockfile exists but was written by a different (fork) build
    // or is corrupt. FAIL CLOSED: no reap, no removal, no overwrite, no spawn
    // on top of foreign live state — reaping here is how live tunnels die.
    const lpath = lockfilePath(ownershipId)
    log(
      `lockfile schema/ownership skew (${lock.reason}) at ${lpath} — failing closed: skipping reap, leaving remote state untouched`
    )

    const error: any = new Error(
      `The remote ownership record ${lpath} does not match this Hermes Desktop build (${lock.reason}). ` +
        'It was probably written by a different or modified desktop build sharing this remote, or the file is corrupt. ' +
        'Refusing to reap or overwrite it — that could kill a live SSH backend owned by another build. ' +
        'If nothing else uses this remote, delete that file on the remote host and reconnect.'
    )

    error.kind = 'remote-lockfile-skew'
    throw error
  }

  if (lock) {
    const pidAlive = await remotePidAlive(ssh, lock.pid)

    const owned =
      pidAlive &&
      (await pidIsOurDashboard(
        ssh,
        lock.pid,
        lock.spawnNonce,
        lock.hermesPath,
        lock.hermesHome,
        ownershipId,
        lock.profile
      ))

    const reusable =
      pidAlive &&
      owned &&
      lock.port > 0 &&
      lock.profile === profile &&
      Boolean(reuseToken) &&
      lock.tokenFingerprint === fingerprintToken(reuseToken) &&
      lock.hermesPath === hermesPath &&
      lock.hermesHome === hermesHome

    if (reusable) {
      const creationTime = lock.creationTime || (await remoteProcessCreationTime(ssh, lock.pid))

      if (creationTime && !lock.creationTime) {
        await writeLockfile(ssh, ownershipId, { ...lock, creationTime })
        lock.creationTime = creationTime
      }

      assertBootstrapNotSuperseded(signal)
      await assertRemoteInstallUpdateClear(ssh, hermesHome)
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
          assertBootstrapNotSuperseded(signal)
          await cancelForwardSafe(deps, localPort, lock.port)
          await assertRemoteInstallUpdateClear(ssh, hermesHome)
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

          assertBootstrapNotSuperseded(signal)
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
            logPath: lock.logPath,
            hermesHome,
            startedAt: lock.startedAt,
            creationTime: lock.creationTime || ''
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
      assertBootstrapNotSuperseded(signal)
      await assertRemoteInstallUpdateClear(ssh, hermesHome)
      await cleanupStale(ssh, ownershipId, lock, pidAlive)
    }
  }

  assertBootstrapNotSuperseded(signal)
  await assertRemoteInstallUpdateClear(ssh, hermesHome)
  const spawnToken = mintToken()

  const spawned = await spawnRemoteDashboard(ssh, {
    hermesPath,
    profile,
    token: spawnToken,
    ownershipId,
    hermesHome,
    assertInstallClear: () => assertRemoteInstallUpdateClear(ssh, hermesHome)
  })

  if (spawned.existing) {
    if (!reuseToken) {
      const error: any = new Error(
        'Another SSH connection owns this remote dashboard; a session token is required to reuse it.'
      )

      error.kind = 'remote-ownership-contended'
      throw error
    }

    const published = await waitForRemoteSpawnCompletion(ssh, ownershipId, readyTimeoutMs)

    if (!published) {
      return connect({ ...deps, reuseToken })
    }

    return connect({ ...deps, reuseToken })
  }

  const { pid, spawnNonce, logPath, tokenFilePath } = spawned
  log(`spawned remote dashboard pid=${pid}`)
  const creationTime = await remoteProcessCreationTime(ssh, pid)

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
    startedAt: new Date().toISOString(),
    ...(creationTime ? { creationTime } : {})
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
    assertBootstrapNotSuperseded(signal)
    log(`remote dashboard bound port ${remotePort}`)

    localPort = await openForward(deps, remotePort)
    assertBootstrapNotSuperseded(signal)
    const baseUrl = `http://127.0.0.1:${localPort}`
    await waitForHermes(baseUrl, spawnToken)
    assertBootstrapNotSuperseded(signal)

    const token = await adoptOwnedServedToken(adoptServedToken, baseUrl, spawnToken, ssh, pid, 'remote dashboard')

    assertBootstrapNotSuperseded(signal)
    const tokenFingerprint = fingerprintToken(token)
    await writeLockfile(ssh, ownershipId, { ...ownedSpawn, port: remotePort, tokenFingerprint })
    assertBootstrapNotSuperseded(signal)

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
      logPath,
      hermesHome,
      startedAt: ownedSpawn.startedAt,
      creationTime: ownedSpawn.creationTime || ''
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
  assertRemoteInstallUpdateClear,
  buildSpawnCommand,
  classifySshReuseProof,
  cleanupStale,
  connect,
  connectReservationPath,
  DEFAULT_READY_TIMEOUT_MS,
  disconnect,
  expandRemotePath,
  fingerprintToken,
  isForwardBindCollision,
  isLockfileSkew,
  listRemoteHermesProfiles,
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
  remoteProcessCreationTime,
  remoteSupportsSshOwnership,
  removeLockfile,
  scrapeReadyPort,
  shq,
  spawnLogPath,
  spawnRemoteDashboard,
  spawnTokenPath,
  SUPPORTED_REMOTE_OS,
  terminateOwnedDashboardForUpdate,
  validateRemotePath,
  writeLockfile
}
