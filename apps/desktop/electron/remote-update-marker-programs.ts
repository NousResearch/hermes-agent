/**
 * The update-marker judge as a program that runs ON an SSH remote.
 *
 * One parser and one identity rule for every Desktop remote reader (the POSIX
 * relaunch gate, the spawn recheck, the managed-update observer), the same as
 * `hermes_cli/update_lock.py::judge_marker` and `tests/fixtures/update_marker_corpus.json`:
 * an identity (pid, ct) is live when the pid is alive and its creation time is
 * within 2 s of the recorded `ct:`; with no recorded/readable ct it is live only
 * while `now - started_at <= 1200`. Owner or delegate live => LIVE, both dead =>
 * CLEAR, malformed => UNCERTAIN. Only the host's system Python runs it: nothing
 * imports the checkout an updater may be replacing.
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
