/**
 * Desktop-managed SSH update transaction.
 *
 * This module is intentionally Electron-free. main.ts owns the concrete pool
 * maps, while this file owns the security-sensitive ordering and the remote
 * wire protocol:
 *
 *   gate dials -> drain every captured scope -> detached `hermes update`
 *   -> correlated terminal marker + durable receipt -> restore every scope
 *   -> lift the gate
 *
 * A remote URL/cloud connection never enters this lifecycle. Only SSH scopes
 * whose serve process is proved by the Desktop ownership record are supplied
 * by main.ts.
 */

import { expandRemotePath, shq } from './remote-lifecycle'
import { encodedPowerShell, powerShellCommand, psLiteral } from './windows-remote-lifecycle'

const UPDATE_EXIT_INDEPENDENT_HANDOFF = 75
const DEFAULT_REMOTE_UPDATE_TIMEOUT_MS = 60 * 60 * 1000
const DEFAULT_REMOTE_CLEARANCE_TIMEOUT_MS = 5 * 60 * 1000
const DEFAULT_REMOTE_UPDATE_POLL_MS = 1_000
const RECEIPT_GRACE_MS = 15_000
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

type ManagedUpdateOutcome = 'updated' | 'update-failed' | 'restore-failed' | 'update-and-restore-failed' | 'refused'

interface ManagedUpdateReceiptSummary {
  correlationId: string
  outcome: string
  startedAt?: string
  finishedAt?: string
  preSha?: string
  postSha?: string
  preVersion?: string
  postVersion?: string
  stopReason?: string
}

interface ManagedUpdateScopeResult {
  profile: string
  restored: boolean
  error?: string
}

interface ManagedConnectionUpdateResult {
  connectionId: string
  correlationId: string
  ok: boolean
  updateOk: boolean
  restoreOk: boolean
  outcome: ManagedUpdateOutcome
  exitCode: null | number
  receipt: ManagedUpdateReceiptSummary | null
  scopes: ManagedUpdateScopeResult[]
  error?: string
  message?: string
}

interface ManagedSshScope {
  key: string
  profile: string
}

interface RemoteUpdateTarget {
  ssh: {
    exec: (command: string, options?: { timeoutMs?: number; stdinData?: string }) => Promise<string>
  }
  platform: 'Darwin' | 'Linux' | 'Windows'
  hermesPath: string
  hermesHome: string
  pythonPath?: string
}

type RemoteMarkerState = 'absent' | 'dead' | 'live' | 'malformed' | 'unavailable'

interface RemoteUpdateObservation {
  marker: RemoteMarkerState
  markerPid?: number
  launchIntent: 'absent' | 'dead' | 'present' | 'malformed' | 'unavailable'
  exitCode: null | number
  receipt: ManagedUpdateReceiptSummary | null
  coordinatorReady: null | { correlationId: string; pid: number }
}

interface RemoteUpdateProof {
  exitCode: number
  receipt: ManagedUpdateReceiptSummary
}

interface ManagedUpdateDeps<TScope extends ManagedSshScope = ManagedSshScope> {
  connectionId: string
  correlationId: string
  scopes: TScope[]
  preflightRemote: () => Promise<void>
  drainScope: (scope: TScope) => Promise<void>
  updateRemote: () => Promise<RemoteUpdateProof>
  awaitRestoreClearance: () => Promise<void>
  closeTransports: () => Promise<void>
  restoreScope: (scope: TScope) => Promise<unknown>
  releaseGate: () => void
  prepareRecovery?: () => Promise<void>
  completeRecovery?: () => Promise<void>
}

function validateCorrelationId(correlationId: string): string {
  const value = String(correlationId || '')
    .trim()
    .toLowerCase()

  if (!UUID_RE.test(value)) {
    throw new Error('Managed SSH update correlation ID is invalid.')
  }

  return value
}

function validateRemoteValue(value: string, label: string): string {
  const normalized = String(value || '').trim()

  // eslint-disable-next-line no-control-regex -- remote shell values may not contain control bytes
  if (!normalized || /[\x00\r\n]/.test(normalized)) {
    throw new Error(`Managed SSH update ${label} is invalid.`)
  }

  return normalized
}

function managedSshTokenPersistencePlan(
  source: unknown,
  registryConnectionId = ''
): {
  legacySource: 'global' | 'profile' | null
  registryConnectionId: string
} {
  const value = typeof source === 'string' ? source : ''
  const sourceRegistryId = value.startsWith('registry:') ? value.slice('registry:'.length) : ''

  return {
    legacySource: sourceRegistryId ? null : value === 'profile' ? 'profile' : 'global',
    registryConnectionId: String(registryConnectionId || sourceRegistryId || '').trim()
  }
}

function managedSshScopeRole(input: {
  connectionId: string
  key: string
  prefix: string
  routeConnectionId?: string
  state?: { primaryRegistryScope?: boolean; registryConnectionId?: string } | null
}): 'pool' | 'primary' | null {
  if (input.state?.registryConnectionId === input.connectionId) {
    return input.state.primaryRegistryScope === true ? 'primary' : 'pool'
  }
  if (input.key.startsWith(input.prefix) || input.routeConnectionId === input.connectionId) {
    return 'pool'
  }

  return null
}

type ManagedSshRecoveryScope = {
  key: string
  kind: 'legacy' | 'primary' | 'registry'
  profile: string
}

function managedSshRecoveryScopes(
  scopes: Iterable<{ key: string; primary?: boolean; profile: string }>,
  registryPrefix: string
): ManagedSshRecoveryScope[] {
  const unique = new Map<string, ManagedSshRecoveryScope>()

  for (const scope of scopes) {
    const key = String(scope.key)
    const existing = unique.get(key)
    const next = {
      key,
      kind: scope.primary ? 'primary' : key.startsWith(registryPrefix) ? 'registry' : 'legacy',
      profile: String(scope.profile || 'default')
    } as ManagedSshRecoveryScope

    // update-all can collect a primary scope through both the legacy and
    // registry indexes. Restore it once, with primary taking precedence, so a
    // single connection is never drained/restarted twice in one transaction.
    if (!existing || next.kind === 'primary') {
      unique.set(key, next)
    }
  }

  return [...unique.values()]
}

function posixChildPath(home: string, name: string): string {
  return `${home.replace(/\/+$/, '')}/${name}`
}

function windowsChildPath(home: string, name: string): string {
  return `${home.replace(/[\\/]+$/, '')}\\${name}`
}

/**
 * Build the POSIX launch command. The SSH channel waits only for the tiny
 * launcher shell; the updater runs in a new session. Exit 75 is deliberately
 * not published by the wrapper because it means an independently-supervised
 * rollout worker owns the eventual terminal marker.
 */
function buildPosixManagedUpdateLaunch(target: RemoteUpdateTarget, correlationId: string): string {
  const correlation = validateCorrelationId(correlationId)
  const home = validateRemoteValue(target.hermesHome, 'Hermes home')
  const hermesPath = validateRemoteValue(target.hermesPath, 'launcher path')
  const statusPath = posixChildPath(home, `.update_exit_code.${correlation}`)
  const intentPath = posixChildPath(home, `.update_launch_intent.${correlation}`)
  const outputPath = posixChildPath(home, `logs/desktop-update-${correlation}.log`)
  const homeWord = expandRemotePath(home)
  const statusWord = expandRemotePath(statusPath)
  const intentWord = expandRemotePath(intentPath)
  const outputWord = expandRemotePath(outputPath)
  const launcherWord = expandRemotePath(hermesPath)
  const updateCommand =
    `env HERMES_HOME=${homeWord} ` +
    `HERMES_UPDATE_CORRELATION_ID=${shq(correlation)} ` +
    'HERMES_UPDATE_ORIGIN_PROFILE=default ' +
    `HERMES_UPDATE_ORIGIN_HOME=${homeWord} ` +
    `HERMES_UPDATE_OUTPUT_PATH=${outputWord} ` +
    `${launcherWord} update --yes`
  const inner =
    `set +e; if [ -r "/proc/$$/stat" ]; then ` +
    `intent_creation="linux:$(awk '{print $22}' "/proc/$$/stat")"; ` +
    `else intent_creation="darwin:$(ps -o lstart= -p "$$" | sed 's/^ *//')"; fi; ` +
    `intent_tmp=${intentWord}."$$".tmp; ` +
    `printf '{"correlation":"%s","pid":%s,"creation":"%s"}' ${shq(correlation)} "$$" "$intent_creation" > "$intent_tmp" && ` +
    `mv -f "$intent_tmp" ${intentWord} || exit 70; ` +
    `${updateCommand}; rc=$?; ` +
    `if [ "$rc" -ne ${UPDATE_EXIT_INDEPENDENT_HANDOFF} ] && [ ! -e ${statusWord} ]; then ` +
    `tmp=${statusWord}."$$".tmp; umask 077; ` +
    `printf "%s" "$rc" > "$tmp" && mv -f "$tmp" ${statusWord}; fi; ` +
    'exit "$rc"'

  return (
    `umask 077 && mkdir -p "$(dirname ${outputWord})" && rm -f ${statusWord} ${intentWord} && ` +
    `if command -v setsid >/dev/null 2>&1; then ` +
    `setsid sh -c ${shq(inner)} </dev/null >>${outputWord} 2>&1 & ` +
    `else nohup sh -c ${shq(inner)} </dev/null >>${outputWord} 2>&1 & fi; child=$!; ` +
    `i=0; while [ ! -e ${intentWord} ]; do kill -0 "$child" 2>/dev/null || exit 1; ` +
    'i=$((i+1)); [ "$i" -ge 200 ] && exit 1; sleep 0.05; done; printf MANAGED_UPDATE_STARTED'
  )
}

/** Windows equivalent of buildPosixManagedUpdateLaunch. */
function buildWindowsManagedUpdateLaunch(target: RemoteUpdateTarget, correlationId: string): string {
  const correlation = validateCorrelationId(correlationId)
  const home = validateRemoteValue(target.hermesHome, 'Hermes home')
  const hermesPath = validateRemoteValue(target.hermesPath, 'launcher path')
  const statusPath = windowsChildPath(home, `.update_exit_code.${correlation}`)
  const readyPath = windowsChildPath(home, `.update_coordinator_ready.${correlation}`)
  const intentPath = windowsChildPath(home, `.update_launch_intent.${correlation}`)
  const outputPath = windowsChildPath(home, `logs\\desktop-update-${correlation}.log`)
  const wrapper = [
    '$ErrorActionPreference="Continue"',
    `$env:HERMES_HOME=${psLiteral(home)}`,
    `$env:HERMES_UPDATE_CORRELATION_ID=${psLiteral(correlation)}`,
    '$env:HERMES_UPDATE_ORIGIN_PROFILE="default"',
    `$env:HERMES_UPDATE_ORIGIN_HOME=${psLiteral(home)}`,
    `$env:HERMES_UPDATE_OUTPUT_PATH=${psLiteral(outputPath)}`,
    // The copied Windows coordinator verifies this correlation AND its actual
    // breakaway state before accepting it; the string alone grants nothing.
    `$env:HERMES_UPDATE_WINDOWS_DETACHED=${psLiteral(correlation)}`,
    `$env:HERMES_UPDATE_TAURI_OUTCOME_PATH=${psLiteral(statusPath)}`,
    `$env:HERMES_UPDATE_TAURI_READY_PATH=${psLiteral(readyPath)}`,
    `$intentTmp=${psLiteral(intentPath)}+"."+$PID+".tmp"`,
    '$intentCreation="windows:"+[string]([Diagnostics.Process]::GetCurrentProcess().StartTime.ToUniversalTime().ToFileTimeUtc())',
    `$intentPayload=[ordered]@{correlation=${psLiteral(correlation)};pid=$PID;creation=$intentCreation}|ConvertTo-Json -Compress`,
    '[IO.File]::WriteAllText($intentTmp,$intentPayload,[Text.UTF8Encoding]::new($false))',
    `Move-Item -LiteralPath $intentTmp -Destination ${psLiteral(intentPath)} -Force`,
    `& ${psLiteral(hermesPath)} update --yes *>> ${psLiteral(outputPath)}`,
    '$rc=$LASTEXITCODE',
    // A non-gateway Windows coordinator parent returns 0 once its copied
    // child owns the marker. That is acceptance, not completion: suppress the
    // wrapper status when the correlated readiness proof exists and let the
    // durable child receipt + released marker provide terminal truth.
    `$handoffAccepted=($rc -eq 0 -and (Test-Path -LiteralPath ${psLiteral(readyPath)}))`,
    `if($rc -ne ${UPDATE_EXIT_INDEPENDENT_HANDOFF} -and -not $handoffAccepted -and -not (Test-Path -LiteralPath ${psLiteral(statusPath)})){`,
    `  $tmp=${psLiteral(statusPath)}+"."+$PID+".tmp"`,
    '  [IO.File]::WriteAllText($tmp,[string]$rc,[Text.UTF8Encoding]::new($false))',
    `  Move-Item -LiteralPath $tmp -Destination ${psLiteral(statusPath)} -Force`,
    '}',
    'exit $rc'
  ].join(';')
  const outer = [
    '$ErrorActionPreference="Stop"',
    `$output=${psLiteral(outputPath)}`,
    'New-Item -ItemType Directory -Force -Path (Split-Path -Parent $output)|Out-Null',
    `Remove-Item -LiteralPath ${psLiteral(statusPath)},${psLiteral(readyPath)},${psLiteral(intentPath)} -Force -ErrorAction SilentlyContinue`,
    `$args=@("-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-EncodedCommand",${psLiteral(encodedPowerShell(wrapper))})`,
    '$child=Start-Process -FilePath "powershell.exe" -ArgumentList $args -WindowStyle Hidden -PassThru',
    'if(-not $child){throw "managed update launcher did not start"}',
    '$deadline=[DateTime]::UtcNow.AddSeconds(10)',
    `while(-not [IO.File]::Exists(${psLiteral(intentPath)})){if($child.HasExited){throw "managed update launcher exited before intent proof"};if([DateTime]::UtcNow -ge $deadline){throw "managed update launcher intent timed out"};Start-Sleep -Milliseconds 50}`,
    '[ordered]@{started=$true;pid=$child.Id}|ConvertTo-Json -Compress'
  ].join(';')

  return powerShellCommand(outer)
}

const OBSERVATION_SCRIPT = String.raw`
import ctypes,json,os,re,sys
from pathlib import Path

home=Path(os.path.expanduser(sys.argv[1]))
correlation=sys.argv[2]
# The update marker is install-wide even when the launcher was invoked with a
# named profile home (<root>/profiles/<name>). Correlated status/intent/receipt
# remain under the launch home, matching the CLI writer.
profile_parent=home.parent.name
is_profile_home=(profile_parent.lower()=='profiles') if os.name=='nt' else (profile_parent=='profiles')
install_root=home.parent.parent if is_profile_home else home
marker_path=install_root/'.hermes-update-in-progress'
status_path=home/('.update_exit_code.'+correlation)
ready_path=home/('.update_coordinator_ready.'+correlation)
intent_path=home/('.update_launch_intent.'+correlation)
marker_re=re.compile(rb'([1-9][0-9]*)\r?\n([0-9]+)(?:\r?\n)?\Z')

def pid_alive(pid):
    if os.name!='nt':
        try:
            os.kill(pid,0);return True
        except ProcessLookupError:return False
        except PermissionError:return True
        except OSError:return None
    try:
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype=wintypes.BOOL
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:
            return False if ctypes.get_last_error()==87 else None
        try:
            code=wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):return None
            return code.value==259
        finally:kernel.CloseHandle(handle)
    except Exception:return None

def process_creation(pid):
    if os.name!='nt':
        if sys.platform.startswith('linux'):
            try:
                raw=Path('/proc/'+str(pid)+'/stat').read_text(encoding='ascii')
                return 'linux:'+raw[raw.rfind(')')+2:].split()[19]
            except (OSError,UnicodeError,IndexError):return None
        if sys.platform=='darwin':
            try:
                import subprocess
                value=subprocess.check_output(['ps','-o','lstart=','-p',str(pid)],text=True).strip()
                return 'darwin:'+value if value else None
            except (OSError,subprocess.CalledProcessError):return None
        return None
    try:
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME)]
        kernel.GetProcessTimes.restype=wintypes.BOOL
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return None
        try:
            created=wintypes.FILETIME();exited=wintypes.FILETIME();kernel_time=wintypes.FILETIME();user_time=wintypes.FILETIME()
            if not kernel.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(exited),ctypes.byref(kernel_time),ctypes.byref(user_time)):return None
            value=(created.dwHighDateTime<<32)|created.dwLowDateTime
            return 'windows:'+str(value)
        finally:kernel.CloseHandle(handle)
    except Exception:return None

def marker_state():
    try:raw=marker_path.read_bytes()
    except FileNotFoundError:return {'state':'absent'}
    except OSError:return {'state':'unavailable'}
    match=marker_re.fullmatch(raw)
    if not match:return {'state':'malformed'}
    try:
        pid=int(match.group(1));lease=int(match.group(2))
        if pid<1 or pid>4294967295 or lease>9007199254740991:raise ValueError()
    except ValueError:return {'state':'malformed'}
    live=pid_alive(pid)
    if live is None:return {'state':'unavailable','pid':pid}
    return {'state':'live' if live else 'dead','pid':pid}

def terminal_code():
    try:raw=status_path.read_bytes()
    except FileNotFoundError:return None
    except OSError:return 'unavailable'
    if not re.fullmatch(rb'[0-9]+',raw):return 'malformed'
    try:return int(raw)
    except ValueError:return 'malformed'

def receipt():
    directory=home/'logs'/'update_receipts'
    try:paths=sorted(directory.glob('update_*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)
    except OSError:return None
    for path in paths:
        try:payload=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,UnicodeError,ValueError):continue
        if not isinstance(payload,dict) or payload.get('correlation_id')!=correlation:continue
        if payload.get('outcome')=='running' or not payload.get('finished_at'):continue
        pre=payload.get('pre_update') if isinstance(payload.get('pre_update'),dict) else {}
        post=payload.get('post_update') if isinstance(payload.get('post_update'),dict) else {}
        return {
            'correlationId':correlation,'outcome':str(payload.get('outcome') or 'unknown'),
            'startedAt':payload.get('started_at'),'finishedAt':payload.get('finished_at'),
            'preSha':pre.get('sha'),'postSha':post.get('sha'),
            'preVersion':pre.get('version'),'postVersion':post.get('version'),
            'stopReason':payload.get('stop_reason'),
        }
    return None

def ready():
    try:payload=json.loads(ready_path.read_text(encoding='utf-8'))
    except (FileNotFoundError,OSError,UnicodeError,ValueError):return None
    if not isinstance(payload,dict) or payload.get('correlation_id')!=correlation:return None
    pid=payload.get('pid')
    return {'correlationId':correlation,'pid':pid} if isinstance(pid,int) and pid>0 else None

def launch_intent():
    try:payload=json.loads(intent_path.read_text(encoding='utf-8'))
    except FileNotFoundError:return 'absent'
    except OSError:return 'unavailable'
    except (UnicodeError,ValueError):return 'malformed'
    if not isinstance(payload,dict) or payload.get('correlation')!=correlation:return 'malformed'
    pid=payload.get('pid');creation=payload.get('creation')
    if not isinstance(pid,int) or pid<1 or pid>4294967295 or not isinstance(creation,str):return 'malformed'
    live=pid_alive(pid)
    if live is None:return 'unavailable'
    if not live:return 'dead'
    current=process_creation(pid)
    if current is None:return 'unavailable'
    return 'present' if current==creation else 'dead'

state=marker_state()
print(json.dumps({'marker':state['state'],'markerPid':state.get('pid'),'launchIntent':launch_intent(),'exitCode':terminal_code(),'receipt':receipt(),'coordinatorReady':ready()},separators=(',',':')))
`.trim()

function buildRemoteUpdateObservationCommand(target: RemoteUpdateTarget, correlationId: string): string {
  const correlation = validateCorrelationId(correlationId)
  const home = validateRemoteValue(target.hermesHome, 'Hermes home')

  if (target.platform === 'Windows') {
    const python = validateRemoteValue(target.pythonPath || '', 'Python path')
    const script = [
      '$ErrorActionPreference="Stop"',
      `& ${psLiteral(python)} -c ${psLiteral(OBSERVATION_SCRIPT)} ${psLiteral(home)} ${psLiteral(correlation)}`,
      'if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}'
    ].join(';')

    return powerShellCommand(script)
  }

  return `python3 -c ${shq(OBSERVATION_SCRIPT)} ${shq(home)} ${shq(correlation)}`
}

function parseRemoteUpdateObservation(raw: string, correlationId: string): RemoteUpdateObservation {
  const correlation = validateCorrelationId(correlationId)
  let parsed: any

  try {
    const lines = String(raw || '')
      .replace(/^\uFEFF/, '')
      .trim()
      .split(/\r?\n/)
      .filter(Boolean)

    parsed = JSON.parse(lines.at(-1) || 'null')
  } catch {
    throw new Error('Remote update observation was malformed.')
  }

  if (!parsed || !['absent', 'dead', 'live', 'malformed', 'unavailable'].includes(parsed.marker)) {
    throw new Error('Remote update marker state was malformed.')
  }

  if (!['absent', 'dead', 'present', 'malformed', 'unavailable'].includes(parsed.launchIntent)) {
    throw new Error('Remote update launch intent was malformed.')
  }

  if (parsed.exitCode === 'malformed' || parsed.exitCode === 'unavailable') {
    throw new Error(`Remote update terminal status was ${parsed.exitCode}.`)
  }

  const exitCode = parsed.exitCode == null ? null : Number(parsed.exitCode)

  if (exitCode !== null && (!Number.isSafeInteger(exitCode) || exitCode < 0)) {
    throw new Error('Remote update terminal status was malformed.')
  }

  let receipt: ManagedUpdateReceiptSummary | null = null

  if (parsed.receipt != null) {
    if (
      typeof parsed.receipt !== 'object' ||
      parsed.receipt.correlationId !== correlation ||
      typeof parsed.receipt.outcome !== 'string'
    ) {
      throw new Error('Remote update receipt did not match this transaction.')
    }
    receipt = parsed.receipt as ManagedUpdateReceiptSummary
  }

  let coordinatorReady: RemoteUpdateObservation['coordinatorReady'] = null

  if (parsed.coordinatorReady != null) {
    const pid = Number(parsed.coordinatorReady.pid)

    if (
      parsed.coordinatorReady.correlationId !== correlation ||
      !Number.isSafeInteger(pid) ||
      pid <= 0 ||
      pid > 4_294_967_295
    ) {
      throw new Error('Remote update coordinator readiness proof was malformed.')
    }
    coordinatorReady = { correlationId: correlation, pid }
  }

  const markerPid = parsed.markerPid == null ? undefined : Number(parsed.markerPid)

  if (markerPid !== undefined && (!Number.isSafeInteger(markerPid) || markerPid <= 0 || markerPid > 4_294_967_295)) {
    throw new Error('Remote update marker PID was malformed.')
  }

  return {
    marker: parsed.marker,
    ...(markerPid === undefined ? {} : { markerPid }),
    launchIntent: parsed.launchIntent,
    exitCode,
    receipt,
    coordinatorReady
  }
}

async function observeManagedRemoteUpdate(
  target: RemoteUpdateTarget,
  correlationId: string
): Promise<RemoteUpdateObservation> {
  const command = buildRemoteUpdateObservationCommand(target, correlationId)
  const raw = await target.ssh.exec(command, { timeoutMs: 30_000 })

  return parseRemoteUpdateObservation(raw, correlationId)
}

function markerIsClear(observation: RemoteUpdateObservation): boolean {
  return observation.marker === 'absent' || observation.marker === 'dead'
}

async function assertManagedUpdatePreflightClear(target: RemoteUpdateTarget, correlationId: string): Promise<void> {
  const observation = await observeManagedRemoteUpdate(target, correlationId)

  if (!markerIsClear(observation)) {
    const owner = observation.markerPid ? ` (PID ${observation.markerPid})` : ''
    throw new Error(`The remote install update marker is ${observation.marker}${owner}; refusing to drain its serves.`)
  }
}

async function launchManagedRemoteUpdate(target: RemoteUpdateTarget, correlationId: string): Promise<void> {
  const command =
    target.platform === 'Windows'
      ? buildWindowsManagedUpdateLaunch(target, correlationId)
      : buildPosixManagedUpdateLaunch(target, correlationId)
  const output = await target.ssh.exec(command, { timeoutMs: 30_000 })

  if (target.platform === 'Windows') {
    let parsed: any

    try {
      const lines = String(output || '')
        .replace(/^\uFEFF/, '')
        .trim()
        .split(/\r?\n/)
        .filter(Boolean)
      parsed = JSON.parse(lines.at(-1) || 'null')
    } catch {
      parsed = null
    }

    if (parsed?.started !== true || !Number.isInteger(parsed?.pid) || parsed.pid <= 0) {
      throw new Error('Remote Windows update launcher did not acknowledge its detached child.')
    }
  } else if (!String(output || '').includes('MANAGED_UPDATE_STARTED')) {
    throw new Error('Remote update launcher did not acknowledge its detached child.')
  }
}

async function waitForManagedRemoteUpdate(
  target: RemoteUpdateTarget,
  correlationId: string,
  options: {
    timeoutMs?: number
    pollMs?: number
    now?: () => number
    sleep?: (ms: number) => Promise<void>
  } = {}
): Promise<RemoteUpdateProof> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_REMOTE_UPDATE_TIMEOUT_MS
  const pollMs = options.pollMs ?? DEFAULT_REMOTE_UPDATE_POLL_MS
  const now = options.now || Date.now
  const sleep = options.sleep || (ms => new Promise(resolve => setTimeout(resolve, ms)))
  const deadline = now() + timeoutMs
  let terminalSeenAt: null | number = null

  while (true) {
    let observation: RemoteUpdateObservation

    try {
      observation = await observeManagedRemoteUpdate(target, correlationId)
    } catch (error) {
      if (now() >= deadline) {
        throw error
      }
      await sleep(pollMs)
      continue
    }

    if (observation.marker === 'malformed' || observation.marker === 'unavailable') {
      if (now() >= deadline) {
        throw new Error(
          `Remote update marker remained ${observation.marker} through the update timeout; ` +
            'services remain stopped and Desktop recorded them for safe recovery.'
        )
      }
      await sleep(pollMs)
      continue
    }

    if (
      observation.coordinatorReady &&
      observation.marker === 'live' &&
      observation.markerPid !== observation.coordinatorReady.pid
    ) {
      throw new Error(
        `Remote update marker PID ${observation.markerPid || 'unknown'} did not match coordinator PID ${observation.coordinatorReady.pid}.`
      )
    }

    if (observation.exitCode !== null) {
      terminalSeenAt ??= now()

      if (observation.receipt && markerIsClear(observation)) {
        return { exitCode: observation.exitCode, receipt: observation.receipt }
      }

      if (!observation.receipt && markerIsClear(observation) && now() - terminalSeenAt >= RECEIPT_GRACE_MS) {
        throw new Error('Remote update finished without a correlated durable receipt.')
      }
    }

    if (observation.coordinatorReady && observation.receipt && markerIsClear(observation)) {
      return {
        exitCode: observation.receipt.outcome === 'success' ? 0 : 1,
        receipt: observation.receipt
      }
    }

    if (now() >= deadline) {
      throw new Error(
        markerIsClear(observation)
          ? 'Remote update did not publish a correlated terminal result before the timeout.'
          : 'Remote update was still active at the timeout; services remain stopped and Desktop recorded them for safe recovery.'
      )
    }

    await sleep(pollMs)
  }
}

async function executeManagedRemoteUpdate(
  target: RemoteUpdateTarget,
  correlationId: string,
  options: Parameters<typeof waitForManagedRemoteUpdate>[2] = {},
  onLaunchProved: () => Promise<void> = async () => {}
): Promise<RemoteUpdateProof> {
  await assertManagedUpdatePreflightClear(target, correlationId)
  await launchManagedRemoteUpdate(target, correlationId)
  await onLaunchProved()

  return waitForManagedRemoteUpdate(target, correlationId, options)
}

async function waitForManagedRemoteClearance(
  target: RemoteUpdateTarget,
  correlationId: string,
  options: {
    timeoutMs?: number
    pollMs?: number
    now?: () => number
    sleep?: (ms: number) => Promise<void>
    requireTerminal?: boolean
  } = {}
): Promise<void> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_REMOTE_CLEARANCE_TIMEOUT_MS
  const pollMs = options.pollMs ?? DEFAULT_REMOTE_UPDATE_POLL_MS
  const now = options.now || Date.now
  const sleep = options.sleep || (ms => new Promise(resolve => setTimeout(resolve, ms)))
  const deadline = now() + timeoutMs
  let lastState = 'unavailable'
  let observedLiveOwner = false

  while (true) {
    try {
      const observation = await observeManagedRemoteUpdate(target, correlationId)
      lastState = observation.marker
      observedLiveOwner ||= observation.marker === 'live'

      if (observation.launchIntent === 'malformed' || observation.launchIntent === 'unavailable') {
        lastState = `launch-intent-${observation.launchIntent}`
      } else if (markerIsClear(observation)) {
        if (
          observation.launchIntent === 'dead' ||
          (!options.requireTerminal && observation.launchIntent === 'absent') ||
          observedLiveOwner ||
          observation.exitCode !== null ||
          observation.receipt !== null
        ) {
          return
        }
      }
    } catch {
      // Transport/parse uncertainty is not clearance. A reconnecting observer
      // may eventually prove absent/dead; until then startup remains fenced.
      lastState = 'unavailable'
    }

    if (now() >= deadline) {
      throw new Error(
        `Could not prove remote update clearance before the recovery timeout (marker: ${lastState}); ` +
          'services were not restarted and Desktop retained a durable recovery record.'
      )
    }
    await sleep(pollMs)
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function resultOutcome(updateOk: boolean, restoreOk: boolean): ManagedUpdateOutcome {
  if (updateOk && restoreOk) {
    return 'updated'
  }
  if (!updateOk && !restoreOk) {
    return 'update-and-restore-failed'
  }

  return updateOk ? 'restore-failed' : 'update-failed'
}

async function runManagedSshUpdate<TScope extends ManagedSshScope>(
  deps: ManagedUpdateDeps<TScope>
): Promise<ManagedConnectionUpdateResult> {
  const { connectionId, correlationId, scopes } = deps
  let proof: RemoteUpdateProof | null = null
  let updateError = ''
  let restorationBlocked = ''
  let recoveryComplete = true
  let recoveryPrepared = false
  let restoreNeedsMarkerFence = false
  const restoreResults: ManagedUpdateScopeResult[] = []
  const restoreCandidates = new Set<TScope>()

  try {
    // Refuse on a live/malformed/unreadable install marker before disrupting
    // any currently healthy forward or serve.
    await deps.preflightRemote()
    await deps.prepareRecovery?.()
    recoveryPrepared = true
    const drainErrors: string[] = []

    restoreNeedsMarkerFence = scopes.length > 0
    for (const scope of scopes) {
      restoreCandidates.add(scope)
      try {
        await deps.drainScope(scope)
      } catch (error) {
        drainErrors.push(`${scope.profile}: ${errorMessage(error)}`)
      }
    }
    if (drainErrors.length) {
      throw new Error(`Could not safely drain every managed SSH scope (${drainErrors.join('; ')}).`)
    }
    // A zero-scope update still launches a mutator, so any subsequent error
    // must wait for its marker to clear before closing the update transport.
    restoreNeedsMarkerFence = true
    proof = await deps.updateRemote()
  } catch (error) {
    updateError = errorMessage(error)
  } finally {
    // A launch acknowledgement can be lost after the remote child starts, and
    // a competing updater can claim the install between preflight and drain.
    // Never restart serves until the shared install marker is positively
    // absent/dead, even on an error path.
    if (restoreNeedsMarkerFence) {
      try {
        await deps.awaitRestoreClearance()
      } catch (error) {
        restorationBlocked = errorMessage(error)
        updateError = [updateError, restorationBlocked].filter(Boolean).join(' ')
      }
    }
    try {
      await deps.closeTransports()
    } catch (error) {
      updateError ||= `Could not close the drained SSH transport: ${errorMessage(error)}`
    }

    for (const scope of scopes) {
      if (!restoreCandidates.has(scope)) {
        // Preflight/journal refusal occurred before this healthy scope was
        // touched. Report it ready without cycling its primary/pool backend.
        restoreResults.push({ profile: scope.profile, restored: true })
      } else if (restorationBlocked) {
        restoreResults.push({ profile: scope.profile, restored: false, error: restorationBlocked })
      } else {
        try {
          await deps.restoreScope(scope)
          restoreResults.push({ profile: scope.profile, restored: true })
        } catch (error) {
          restoreResults.push({ profile: scope.profile, restored: false, error: errorMessage(error) })
        }
      }
    }

    try {
      if (recoveryPrepared && !restorationBlocked && restoreResults.every(result => result.restored)) {
        await deps.completeRecovery?.()
      }
    } catch (error) {
      recoveryComplete = false
      updateError ||= `Could not clear the managed SSH recovery record: ${errorMessage(error)}`
    } finally {
      deps.releaseGate()
    }
  }

  const updateOk = Boolean(proof && proof.exitCode === 0 && proof.receipt.outcome === 'success')
  const restoreOk = recoveryComplete && restoreResults.every(result => result.restored)
  const outcome = resultOutcome(updateOk, restoreOk)
  const restoreErrors = restoreResults.filter(result => !result.restored).map(result => result.error)
  const error = [updateError, ...restoreErrors].filter(Boolean).join(' ')

  return {
    connectionId,
    correlationId,
    ok: updateOk && restoreOk,
    updateOk,
    restoreOk,
    outcome,
    exitCode: proof?.exitCode ?? null,
    receipt: proof?.receipt ?? null,
    scopes: restoreResults,
    ...(error ? { error } : {}),
    message:
      outcome === 'updated'
        ? 'Remote Hermes updated and every managed SSH profile is ready.'
        : restoreOk
          ? 'The remote update failed, but every managed SSH profile was restored.'
          : 'The remote update transaction could not restore every managed SSH profile.'
  }
}

function refusedManagedSshUpdate(connectionId: string, correlationId: string, error: string) {
  const message = String(error || 'This connection is not managed by Desktop SSH.')

  return {
    connectionId,
    correlationId,
    ok: false,
    updateOk: false,
    restoreOk: true,
    outcome: 'refused' as const,
    exitCode: null,
    receipt: null,
    scopes: [],
    error: message,
    message
  }
}

// before-quit uses this to join remote update transactions before it starts
// tearing down their SSH transports. Re-read after every batch so an operation
// registered while the first batch settles is joined too.
async function waitForManagedUpdateOperations(getOperations: () => Iterable<Promise<unknown>>): Promise<void> {
  for (;;) {
    const pending = [...getOperations()]

    if (pending.length === 0) {
      return
    }

    await Promise.allSettled(pending)
  }
}

async function recoverManagedSshScopes<TScope>(deps: {
  afterClearance?: () => Promise<void>
  awaitClearance: () => Promise<void>
  completeRecovery: () => Promise<void>
  restoreScope: (scope: TScope) => Promise<unknown>
  scopes: TScope[]
}): Promise<PromiseSettledResult<unknown>[]> {
  await deps.awaitClearance()
  await deps.afterClearance?.()
  const results = await Promise.allSettled(deps.scopes.map(scope => deps.restoreScope(scope)))

  if (results.every(result => result.status === 'fulfilled')) {
    // This intentionally runs for an empty scope list. An inactive connection
    // still journals the detached mutator so a crash/relaunch remains fenced;
    // positive marker clearance is what authorizes removing that durable gate.
    await deps.completeRecovery()
  }

  return results
}

async function fenceManagedSshBootstrapPublication<T>(deps: {
  assertCanPublish: () => void
  publish: () => T
  rollback: (cause: unknown) => Promise<void>
}): Promise<T> {
  try {
    deps.assertCanPublish()
    // Deliberately no await between the final gate assertion and publication:
    // both run in this JavaScript turn, so a queued update claim either lands
    // before the assertion (and triggers rollback) or after the scope is
    // visible to capture/drain.
    return deps.publish()
  } catch (error) {
    await deps.rollback(error)
    throw error
  }
}

async function waitForManagedSshBootstrapFence(
  entries: Iterable<{ metadata?: { registryConnectionId?: string }; promise: Promise<unknown> }>,
  connectionId: string
): Promise<void> {
  const pending = [...entries].filter(entry => entry.metadata?.registryConnectionId === connectionId)
  const results = await Promise.allSettled(pending.map(entry => entry.promise))
  const unsafe = results.find(
    result => result.status === 'rejected' && (result.reason as any)?.unsafeManagedBootstrap === true
  )

  if (unsafe?.status === 'rejected') {
    throw unsafe.reason
  }
}

class ManagedConnectionUpdateGate {
  private claims = new Map<string, string>()

  constructor(private readonly durableOwner: (connectionId: string) => string | null = () => null) {}

  claim(connectionId: string, correlationId: string): boolean {
    const id = String(connectionId || '').trim()
    const correlation = validateCorrelationId(correlationId)
    const durable = this.durableOwner(id)

    if (!id || this.claims.has(id) || (durable && durable !== correlation)) {
      return false
    }
    this.claims.set(id, correlation)

    return true
  }

  release(connectionId: string, correlationId: string): void {
    const id = String(connectionId || '').trim()

    if (this.claims.get(id) === correlationId) {
      this.claims.delete(id)
    }
  }

  assertCanDial(connectionId: string, correlationId = ''): void {
    const id = String(connectionId || '').trim()
    const owner = this.claims.get(id) || this.durableOwner(id)

    if (owner && owner !== correlationId) {
      const error: any = new Error(`SSH connection "${id}" is paused while its managed update is in progress.`)
      error.code = 'managed-update-in-progress'
      throw error
    }
  }

  assertCanMutate(connectionId: string): void {
    const id = String(connectionId || '').trim()
    const owner = this.claims.get(id) || this.durableOwner(id)

    if (owner) {
      const error: any = new Error(
        `SSH connection "${id}" cannot be edited or removed while its managed update recovery is pending.`
      )
      error.code = 'managed-update-in-progress'
      throw error
    }
  }

  owner(connectionId: string): string | null {
    const id = String(connectionId || '').trim()

    return this.claims.get(id) || this.durableOwner(id) || null
  }
}

export {
  assertManagedUpdatePreflightClear,
  buildPosixManagedUpdateLaunch,
  buildRemoteUpdateObservationCommand,
  buildWindowsManagedUpdateLaunch,
  DEFAULT_REMOTE_CLEARANCE_TIMEOUT_MS,
  DEFAULT_REMOTE_UPDATE_POLL_MS,
  DEFAULT_REMOTE_UPDATE_TIMEOUT_MS,
  executeManagedRemoteUpdate,
  fenceManagedSshBootstrapPublication,
  launchManagedRemoteUpdate,
  ManagedConnectionUpdateGate,
  type ManagedConnectionUpdateResult,
  type ManagedSshRecoveryScope,
  managedSshRecoveryScopes,
  type ManagedSshScope,
  managedSshScopeRole,
  managedSshTokenPersistencePlan,
  type ManagedUpdateDeps,
  type ManagedUpdateOutcome,
  type ManagedUpdateReceiptSummary,
  type ManagedUpdateScopeResult,
  markerIsClear,
  observeManagedRemoteUpdate,
  parseRemoteUpdateObservation,
  RECEIPT_GRACE_MS,
  recoverManagedSshScopes,
  refusedManagedSshUpdate,
  type RemoteUpdateObservation,
  type RemoteUpdateProof,
  type RemoteUpdateTarget,
  runManagedSshUpdate,
  UPDATE_EXIT_INDEPENDENT_HANDOFF,
  validateCorrelationId,
  waitForManagedRemoteClearance,
  waitForManagedRemoteUpdate,
  waitForManagedSshBootstrapFence,
  waitForManagedUpdateOperations
}
