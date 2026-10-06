/**
 * The update-marker judge as programs that run ON an SSH remote.
 *
 * Same contract as `update-marker-judge.ts` (`tests/fixtures/update_marker_corpus.json`),
 * with its constants interpolated so no copy can drift: an identity (pid, ct)
 * is live when the pid is alive and its creation time is within
 * CREATE_TIME_TOLERANCE_S of the recorded `ct:`; with no recorded/readable ct it
 * is live only while `now - started_at <= V1_MAX_AGE_S`. Owner or delegate live => LIVE, both dead =>
 * CLEAR, malformed => UNCERTAIN. Only the host's system Python (POSIX) or
 * PowerShell/.NET (Windows relaunch/spawn) runs these: nothing imports the
 * checkout an updater may be replacing.
 *
 * The Python stays free of double quotes: managed-ssh-update ships it to Windows
 * as a PowerShell native argument, and PowerShell 5.1 does not escape them.
 */

import { CREATE_TIME_TOLERANCE_S, OWN_CT_EPSILON_S, V1_MAX_AGE_S } from './update-marker-judge'

/** Defines `marker_judge(text, env)` (corpus-shaped, injectable facts) and `marker_verdict(raw_bytes_or_None)`. */
export const REMOTE_MARKER_JUDGE_PY = String.raw`
import os,re,sys,time
MARKER_INT_RE=re.compile(r'[0-9]+')
MARKER_CT_RE=re.compile(r'ct:([0-9]+(?:\.[0-9]+)?)')
MARKER_DELEGATE_RE=re.compile(r'delegate:([0-9]+) ct:([0-9]+(?:\.[0-9]+)?)')

def marker_int(text):
    # int() refuses >4300 digits; past 20 significant digits nothing fits u64 anyway.
    text=text.lstrip('0') or '0'
    return int(text) if len(text)<=20 else None

def marker_identity_state(pid,ct,started,env):
    if pid==0:return 'dead'
    if pid==env['our_pid']:
        own=env['our_ct']()
        return 'ours' if ct is not None and own is not None and abs(ct-own)<=${OWN_CT_EPSILON_S} else 'dead'
    if not env['alive'](pid):return 'dead'
    actual=None if ct is None else env['ct'](pid)
    if ct is None or actual is None:return 'unknown' if env['now']-started<=${V1_MAX_AGE_S} else 'dead'
    return 'match' if abs(ct-actual)<=${CREATE_TIME_TOLERANCE_S} else 'dead'

def marker_judge(text,env):
    if text.startswith('\ufeff'):text=text[1:]
    lines=[(line[:-1] if line.endswith('\r') else line).strip(' \t') for line in text.split('\n')]
    if len(lines)<2 or not MARKER_INT_RE.fullmatch(lines[0]) or not MARKER_INT_RE.fullmatch(lines[1]):return 'malformed',None
    pid=marker_int(lines[0]);started=marker_int(lines[1])
    if pid is None or pid>4294967295 or started is None or started>18446744073709551615:return 'malformed',None
    ct=MARKER_CT_RE.fullmatch(lines[2]) if len(lines)>2 else None
    ids=[(pid,float(ct.group(1)) if ct else None)]
    for line in lines[3:]:
        delegate=MARKER_DELEGATE_RE.fullmatch(line)
        delegate_pid=marker_int(delegate.group(1)) if delegate else None
        if delegate_pid is not None and delegate_pid<=4294967295:
            ids.append((delegate_pid,float(delegate.group(2))));break
    states=[marker_identity_state(p,c,started,env) for p,c in ids]
    live=[p for (p,_),state in zip(ids,states) if state!='dead']
    return ('ours' if 'ours' in states else 'live' if live else 'dead'),(live[0] if live else None)

def marker_stat(pid):
    raw=open('/proc/%d/stat'%pid).read()
    return raw[raw.rfind(')')+2:].split()

def marker_win(pid):
    # (alive, creation unix seconds or None); an open we are denied is alive with no ct.
    import ctypes
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    k.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=k.OpenProcess(0x1000,False,pid)
    if not handle:return ctypes.get_last_error()!=87,None
    try:
        code=wintypes.DWORD();times=[wintypes.FILETIME() for _ in range(4)]
        if k.GetExitCodeProcess(handle,ctypes.byref(code)) and code.value!=259:return False,None
        if not k.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]):return True,None
        return True,((times[0].dwHighDateTime<<32)|times[0].dwLowDateTime)/1e7-11644473600
    finally:k.CloseHandle(handle)

def marker_alive(pid):
    if os.name=='nt':return marker_win(pid)[0]
    try:os.kill(pid,0)
    except (ProcessLookupError,OverflowError):return False
    except OSError:pass  # EPERM or unprovable: alive (fail closed)
    try:return marker_stat(pid)[0]!='Z'
    except (OSError,IndexError):return True

def marker_ct(pid):
    # psutil.create_time() without psutil; unreadable => None => the v1 age ceiling.
    try:
        if os.name=='nt':return marker_win(pid)[1]
        if sys.platform.startswith('linux'):
            with open('/proc/stat') as stat:btime=next(int(line.split()[1]) for line in stat if line.startswith('btime '))
            return btime+int(marker_stat(pid)[19])/os.sysconf('SC_CLK_TCK')
        # UTC like update_lock._stdlib_create_time: a local-time lstart is ambiguous in the repeated DST hour.
        import calendar,subprocess
        out=subprocess.check_output(['ps','-o','lstart=','-p',str(pid)],env=dict(os.environ,LC_ALL='C',TZ='UTC0'),universal_newlines=True)
        return float(calendar.timegm(time.strptime(' '.join(out.split()),'%a %b %d %H:%M:%S %Y')))
    except Exception:return None

MARKER_ENV={'our_pid':os.getpid(),'our_ct':lambda:marker_ct(os.getpid()),'alive':marker_alive,'ct':marker_ct,'now':time.time()}

def marker_verdict(raw):
    if raw is None:return 'CLEAR'
    if len(raw)>4096:return 'UNCERTAIN'
    verdict,owner=marker_judge(raw.decode('utf-8','replace'),MARKER_ENV)
    return 'CLEAR' if verdict=='dead' else 'UNCERTAIN' if verdict=='malformed' else 'LIVE:%d'%owner
`

/**
 * POSIX gate: `python3 -c GATE <marker> [payload] [hermes...]` (an empty payload
 * is the probe). Holds the updaters' kernel lock `<marker>.lock` (A7 rule 1:
 * Python update_lock flock, marker.sh flock) for a bounded 10 s, judges the
 * marker, and unlinks a dead claim inside that hold only while the install's
 * CHECKOUT lock is free (update_lock._reclaim_dead): a killed updater's
 * completion/build child still holding it answers HELD and keeps the marker.
 * The checkouts probed are `<marker dir>/hermes-agent` plus the checkout of
 * each `hermes` executable argument (`<root>/venv/bin/hermes`, symlinks
 * resolved). Then it either prints the verdict (no payload: the relaunch probe)
 * or runs the payload as `sh -c payload hermes-update-mutex <fd>` still holding
 * the lock. A refused payload exits 75 with the verdict on stderr. The probe
 * skips the lock when there is no marker, so it never creates files on a clean host.
 */
export const REMOTE_MARKER_GATE_PY = `${REMOTE_MARKER_JUDGE_PY}
import fcntl,subprocess
marker=sys.argv[1]
payload=sys.argv[2] if len(sys.argv)>2 and sys.argv[2] else None
if payload is None and not os.path.lexists(marker):
    print('CLEAR');sys.exit(0)

def hold(fd):
    deadline=time.monotonic()+10
    while True:
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
        except BlockingIOError:
            if time.monotonic()>=deadline:return False
            time.sleep(0.02)

def read_marker():
    try:
        with open(marker,'rb') as stream:return stream.read(4097)
    except FileNotFoundError:return None

def checkout_lock(root):
    # hermes_cli/update_lock.py::checkout_lock_path: <git common dir>/hermes-update.lock, else <root>/.hermes-update.lock.
    plain=os.path.join(root,'.hermes-update.lock');dot=os.path.join(root,'.git')
    try:
        if os.path.isdir(dot):gitdir=dot
        elif os.path.isfile(dot):
            with open(dot,encoding='utf-8-sig') as stream:text=stream.read().strip()
            if not text.startswith('gitdir:'):return plain
            gitdir=os.path.join(root,text[len('gitdir:'):].strip())
        else:return plain
        common=os.path.join(gitdir,'commondir')
        if os.path.isfile(common):
            with open(common,encoding='utf-8-sig') as stream:gitdir=os.path.join(gitdir,stream.read().strip())
    except (OSError,ValueError):return plain
    return os.path.join(os.path.normpath(gitdir),'hermes-update.lock')

def checkout_held():
    # update_lock.checkout_lock_held: take the flock for one try and drop it. A lock file that
    # exists but cannot be opened or probed counts as held: reclaim needs a provably free checkout.
    roots=[os.path.join(os.path.dirname(marker),'hermes-agent')]
    roots+=[os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(os.path.expanduser(exe))))) for exe in sys.argv[3:] if exe]
    for root in roots:
        try:lock=os.open(checkout_lock(root),os.O_RDONLY|os.O_CLOEXEC)
        except (FileNotFoundError,NotADirectoryError):continue
        except OSError:return True
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:return True
        finally:os.close(lock)
    return False

os.makedirs(os.path.dirname(marker),exist_ok=True)
try:fd=os.open(marker+'.lock',os.O_RDWR|os.O_CREAT|os.O_CLOEXEC,0o644)
except PermissionError:fd=os.open(marker+'.lock',os.O_RDONLY|os.O_CLOEXEC)
verdict='UNCERTAIN'
if hold(fd):
    raw=read_marker();verdict=marker_verdict(raw)
    if verdict=='CLEAR' and raw is not None:
        if checkout_held():verdict='HELD'
        else:
            try:os.unlink(marker)
            except FileNotFoundError:pass
if payload is None or verdict!='CLEAR':
    print(verdict,file=sys.stderr if payload else sys.stdout);sys.exit(75 if payload else 0)
sys.exit(subprocess.run(['sh','-c',payload,'hermes-update-mutex',str(fd)],pass_fds=(fd,)).returncode)
`

/**
 * PowerShell `Get-MarkerVerdict $text` -> CLEAR | LIVE:<pid> | UNCERTAIN, the
 * same rule via .NET process facts (Process.StartTime = psutil create_time on
 * Windows; a process we may not query is alive with no ct). Windows runs this
 * instead of the venv python so a probe never holds the runtime's python.exe
 * open while an updater replaces it. The process facts and clock sit in
 * `Get-MarkerProcessFacts` / `Get-MarkerNow` so the corpus replay can inject them.
 */
export const WINDOWS_MARKER_JUDGE_PS = [
  'function Get-MarkerProcessFacts([int]$ownerId){',
  // Our own pid is never a remote owner.
  'if($ownerId -eq $PID){return @{Alive=$false}}',
  'try{$proc=[Diagnostics.Process]::GetProcessById($ownerId)}catch [ArgumentException]{return @{Alive=$false}}',
  'try{if($proc.HasExited){return @{Alive=$false}};return @{Alive=$true;Ct=([DateTimeOffset]$proc.StartTime).ToUnixTimeMilliseconds()/1000.0}}catch{return @{Alive=$true}}finally{$proc.Dispose()}',
  '}',
  'function Get-MarkerNow{[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()}',
  'function Test-MarkerIdentity($ownerId,$ct,$started,$now){',
  'if($ownerId -eq 0 -or $ownerId -gt [int]::MaxValue){return $false}',
  '$facts=Get-MarkerProcessFacts $ownerId',
  'if(-not $facts.Alive){return $false}',
  `if($null -eq $ct -or $null -eq $facts.Ct){return ($now-[double]$started) -le ${V1_MAX_AGE_S}}`,
  `return [Math]::Abs($ct-$facts.Ct) -le ${CREATE_TIME_TOLERANCE_S}`,
  '}',
  'function Get-MarkerCt([string]$digits){$value=0.0;if([double]::TryParse($digits,[Globalization.NumberStyles]::AllowDecimalPoint,[Globalization.CultureInfo]::InvariantCulture,[ref]$value)){return $value};return [double]::PositiveInfinity}',
  'function Get-MarkerVerdict([string]$text){',
  '$lines=@(($text -replace "^\\uFEFF","") -split "`n" | ForEach-Object {($_ -replace "`r$","").Trim([char[]]" `t")})',
  '$style=[Globalization.NumberStyles]::None;$culture=[Globalization.CultureInfo]::InvariantCulture;[uint32]$ownerId=0;[uint64]$started=0',
  'if($lines.Count -lt 2 -or $lines[0] -cnotmatch "^[0-9]+$" -or $lines[1] -cnotmatch "^[0-9]+$" -or -not [uint32]::TryParse($lines[0],$style,$culture,[ref]$ownerId) -or -not [uint64]::TryParse($lines[1],$style,$culture,[ref]$started)){return "UNCERTAIN"}',
  '$ct=$null;if($lines.Count -gt 2 -and $lines[2] -cmatch "^ct:([0-9]+(\\.[0-9]+)?)$"){$ct=Get-MarkerCt $Matches[1]}',
  '$ids=@(,@($ownerId,$ct))',
  'foreach($line in @($lines | Select-Object -Skip 3)){[uint32]$delegateId=0;if($line -cmatch "^delegate:([0-9]+) ct:([0-9]+(\\.[0-9]+)?)$" -and [uint32]::TryParse($Matches[1],$style,$culture,[ref]$delegateId)){$ids+=,@($delegateId,(Get-MarkerCt $Matches[2]));break}}',
  '$now=Get-MarkerNow',
  'foreach($id in $ids){if(Test-MarkerIdentity $id[0] $id[1] $started $now){return "LIVE:$($id[0])"}}',
  'return "CLEAR"',
  '}'
].join('\n')

/**
 * PowerShell `Test-CheckoutLockHeld $roots` -> $true while any checkout in
 * `$roots` has its update kernel lock held: the Python update_lock byte range on
 * `checkout_lock_path` (offset 1048576, the owner byte plus the 16 R5b lease
 * bytes a refused completion child keeps while it outlives its owner), taken for
 * one try and dropped. A lock file that exists but cannot be opened or locked
 * counts as held. Gates delete a dead marker only when this is $false.
 */
export const WINDOWS_CHECKOUT_LOCK_PS = [
  'function Get-CheckoutLockPath([string]$root){',
  '$plain=[IO.Path]::Combine($root,".hermes-update.lock");$dot=[IO.Path]::Combine($root,".git");$gitdir=$null',
  'try{',
  'if([IO.Directory]::Exists($dot)){$gitdir=$dot}elseif([IO.File]::Exists($dot)){$text=[IO.File]::ReadAllText($dot).Trim();if($text.StartsWith("gitdir:")){$gitdir=[IO.Path]::Combine($root,$text.Substring(7).Trim())}}',
  'if($gitdir){$common=[IO.Path]::Combine($gitdir,"commondir");if([IO.File]::Exists($common)){$gitdir=[IO.Path]::Combine($gitdir,[IO.File]::ReadAllText($common).Trim())}}',
  '}catch{return $plain}',
  'if(-not $gitdir){return $plain}',
  'return [IO.Path]::GetFullPath([IO.Path]::Combine($gitdir,"hermes-update.lock"))',
  '}',
  'function Test-CheckoutLockHeld([object[]]$roots){',
  'foreach($root in $roots){',
  'if([string]::IsNullOrWhiteSpace([string]$root)){continue}',
  '$lockFile=$null',
  'try{$lockFile=[IO.File]::Open((Get-CheckoutLockPath ([string]$root)),[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)}catch [IO.FileNotFoundException],[IO.DirectoryNotFoundException]{continue}catch{return $true}',
  'try{$lockFile.Lock(1048576,17);$lockFile.Unlock(1048576,17)}catch{return $true}finally{$lockFile.Dispose()}',
  '}',
  'return $false',
  '}'
].join('\n')
