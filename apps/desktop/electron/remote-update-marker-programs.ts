/**
 * The update-marker judge as programs that run ON an SSH remote.
 *
 * Same contract as `update-marker-judge.ts` (`tests/fixtures/update_marker_corpus.json`):
 * an identity (pid, ct) is live when the pid is alive and its creation time is
 * within 2 s of the recorded `ct:`; with no recorded/readable ct it is live only
 * while `now - started_at <= 1200`. Owner or delegate live => LIVE, both dead =>
 * CLEAR, malformed => UNCERTAIN. Only the host's system Python (POSIX) or
 * PowerShell/.NET (Windows relaunch/spawn) runs these: nothing imports the
 * checkout an updater may be replacing.
 *
 * The Python stays free of double quotes: managed-ssh-update ships it to Windows
 * as a PowerShell native argument, and PowerShell 5.1 does not escape them.
 */

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
        return 'ours' if ct is not None and own is not None and abs(ct-own)<=0.005 else 'dead'
    if not env['alive'](pid):return 'dead'
    actual=None if ct is None else env['ct'](pid)
    if ct is None or actual is None:return 'unknown' if env['now']-started<=1200 else 'dead'
    return 'match' if abs(ct-actual)<=2.0 else 'dead'

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
        import subprocess
        out=subprocess.check_output(['ps','-o','lstart=','-p',str(pid)],env=dict(os.environ,LC_ALL='C'),universal_newlines=True)
        return time.mktime(time.strptime(out.strip(),'%a %b %d %H:%M:%S %Y'))
    except Exception:return None

MARKER_ENV={'our_pid':os.getpid(),'our_ct':lambda:marker_ct(os.getpid()),'alive':marker_alive,'ct':marker_ct,'now':time.time()}

def marker_verdict(raw):
    if raw is None:return 'CLEAR'
    if len(raw)>4096:return 'UNCERTAIN'
    verdict,owner=marker_judge(raw.decode('utf-8','replace'),MARKER_ENV)
    return 'CLEAR' if verdict=='dead' else 'UNCERTAIN' if verdict=='malformed' else 'LIVE:%d'%owner
`

/**
 * POSIX gate: `python3 -c GATE <marker> [payload]`. Holds the updaters' kernel
 * lock `<marker>.lock` (A7 rule 1: Python update_lock flock, marker.sh flock)
 * for a bounded 10 s, judges the marker, unlinks a dead claim inside that hold,
 * then either prints the verdict (no payload: the relaunch probe) or runs the
 * payload as `sh -c payload hermes-update-mutex <fd>` still holding the lock.
 * A refused payload exits 75 with the verdict on stderr. The probe skips the
 * lock when there is no marker, so it never creates files on a clean host.
 */
export const REMOTE_MARKER_GATE_PY = `${REMOTE_MARKER_JUDGE_PY}
import fcntl,subprocess
marker=sys.argv[1]
payload=sys.argv[2] if len(sys.argv)>2 else None
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

os.makedirs(os.path.dirname(marker),exist_ok=True)
try:fd=os.open(marker+'.lock',os.O_RDWR|os.O_CREAT|os.O_CLOEXEC,0o644)
except PermissionError:fd=os.open(marker+'.lock',os.O_RDONLY|os.O_CLOEXEC)
verdict='UNCERTAIN'
if hold(fd):
    raw=read_marker();verdict=marker_verdict(raw)
    if verdict=='CLEAR' and raw is not None:
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
 * open while an updater replaces it. Our own pid is never a remote owner.
 */
export const WINDOWS_MARKER_JUDGE_PS = [
  'function Test-MarkerIdentity($ownerId,$ct,$started,$now){',
  'if($ownerId -eq 0 -or $ownerId -eq $PID -or $ownerId -gt [int]::MaxValue){return $false}',
  'try{$proc=[Diagnostics.Process]::GetProcessById([int]$ownerId)}catch [ArgumentException]{return $false}',
  '$actual=$null',
  'try{if($proc.HasExited){return $false};if($null -ne $ct){$actual=([DateTimeOffset]$proc.StartTime).ToUnixTimeMilliseconds()/1000.0}}catch{}finally{$proc.Dispose()}',
  'if($null -eq $ct -or $null -eq $actual){return ($now-[double]$started) -le 1200}',
  'return [Math]::Abs($ct-$actual) -le 2.0',
  '}',
  'function Get-MarkerCt([string]$digits){$value=0.0;if([double]::TryParse($digits,[Globalization.NumberStyles]::AllowDecimalPoint,[Globalization.CultureInfo]::InvariantCulture,[ref]$value)){return $value};return [double]::PositiveInfinity}',
  'function Get-MarkerVerdict([string]$text){',
  '$lines=@(($text -replace "^\\uFEFF","") -split "`n" | ForEach-Object {($_ -replace "`r$","").Trim([char[]]" `t")})',
  '$style=[Globalization.NumberStyles]::None;$culture=[Globalization.CultureInfo]::InvariantCulture;[uint32]$ownerId=0;[uint64]$started=0',
  'if($lines.Count -lt 2 -or $lines[0] -cnotmatch "^[0-9]+$" -or $lines[1] -cnotmatch "^[0-9]+$" -or -not [uint32]::TryParse($lines[0],$style,$culture,[ref]$ownerId) -or -not [uint64]::TryParse($lines[1],$style,$culture,[ref]$started)){return "UNCERTAIN"}',
  '$ct=$null;if($lines.Count -gt 2 -and $lines[2] -cmatch "^ct:([0-9]+(\\.[0-9]+)?)$"){$ct=Get-MarkerCt $Matches[1]}',
  '$ids=@(,@($ownerId,$ct))',
  'foreach($line in @($lines | Select-Object -Skip 3)){[uint32]$delegateId=0;if($line -cmatch "^delegate:([0-9]+) ct:([0-9]+(\\.[0-9]+)?)$" -and [uint32]::TryParse($Matches[1],$style,$culture,[ref]$delegateId)){$ids+=,@($delegateId,(Get-MarkerCt $Matches[2]));break}}',
  '$now=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()',
  'foreach($id in $ids){if(Test-MarkerIdentity $id[0] $id[1] $started $now){return "LIVE:$($id[0])"}}',
  'return "CLEAR"',
  '}'
].join('\n')
