"""JSON-lines local media adapter. No browser cookies, account profiles or protected services."""
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

_windows_job_handle = None


def isolate_windows_children():
    """A dying/cancelled worker must also terminate all owned subprocesses."""
    global _windows_job_handle
    import ctypes
    from ctypes import wintypes
    size_t = ctypes.c_size_t

    class BasicLimits(ctypes.Structure):
        _fields_ = [('processTime', ctypes.c_int64), ('jobTime', ctypes.c_int64), ('flags', wintypes.DWORD), ('minimumWorkingSet', size_t), ('maximumWorkingSet', size_t), ('activeProcesses', wintypes.DWORD), ('affinity', size_t), ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ['readOperations', 'writeOperations', 'otherOperations', 'readBytes', 'writeBytes', 'otherBytes']]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [('basic', BasicLimits), ('io', IoCounters), ('processMemory', size_t), ('jobMemory', size_t), ('peakProcessMemory', size_t), ('peakJobMemory', size_t)]

    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    api.CreateJobObjectW.restype = wintypes.HANDLE
    api.GetCurrentProcess.restype = wintypes.HANDLE
    api.GetCurrentProcess.argtypes = []
    api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    api.SetInformationJobObject.restype = wintypes.BOOL
    api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    api.AssignProcessToJobObject.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    handle = api.CreateJobObjectW(None, None)
    if not handle:
        raise ValueError('worker-isolation-unavailable')
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    configured = api.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))
    assigned = configured and api.AssignProcessToJobObject(handle, api.GetCurrentProcess())
    if not assigned:
        api.CloseHandle(handle)
        raise ValueError('worker-isolation-unavailable')
    _windows_job_handle = handle  # OS closes this handle when the worker exits.


def isolate_posix_children():
    """Verify our detached group, then let the worker terminate only that group."""
    if os.getpgrp() != os.getpid():
        raise ValueError('worker-isolation-unavailable')
    try:
        os.killpg(os.getpgrp(), 0)
    except OSError:
        raise ValueError('worker-isolation-unavailable') from None

    def stop_private_group(signum, _frame):
        # Reset first so the group signal terminates this worker too instead of
        # re-entering the handler. All descendants inherit this private group.
        signal.signal(signum, signal.SIG_DFL)
        os.killpg(os.getpgrp(), signum)

    signal.signal(signal.SIGTERM, stop_private_group)


def supervise(payload):
    """Run setup commands under the same owned-tree cancellation boundary."""
    request = json.loads(payload)
    if (not isinstance(request, dict) or not isinstance(request.get('executable'), str)
            or not os.path.isabs(request['executable']) or not isinstance(request.get('args'), list)
            or any(not isinstance(argument, str) for argument in request['args'])):
        raise ValueError('invalid-supervisor-request')
    child = subprocess.Popen([request['executable'], *request['args']])
    try:
        return child.wait(timeout=600)
    except subprocess.TimeoutExpired:
        # main() established an owned POSIX group or a Windows kill-on-close
        # job before spawning. Terminating this supervisor tears down that
        # owned tree, rather than leaving grandchildren after a child-only kill.
        os.kill(os.getpid(), signal.SIGTERM)
        raise ValueError('supervisor-timeout') from None


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def transcript(request):
    from youtube_transcript_api import YouTubeTranscriptApi
    language = request.get('language') or 'en'
    if not re.fullmatch(r'[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*', language):
        raise ValueError('invalid-language')
    fetched = YouTubeTranscriptApi().fetch(request['videoId'], languages=[language])
    window = request.get('range') or {}
    start = window.get('startSeconds', 0)
    end = window.get('endSeconds', 86400)
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [start, end]) or not 0 <= start <= end <= 86400:
        raise ValueError('invalid-range')
    segments = []
    for item in fetched:
        if item.start + item.duration > start and item.start < end:
            segments.append({'start': item.start, 'end': item.start + item.duration, 'text': item.text})
    if len(segments) > 10000 or sum(len(x['text']) for x in segments) > 150000:
        raise ValueError('transcript-too-large-use-range')
    emit({'type': 'result', 'language': fetched.language_code, 'trackKind': 'automatic' if fetched.is_generated else 'manual', 'segments': segments})


def download(request):
    from yt_dlp import YoutubeDL
    output = Path(request['output'])
    if not output.is_absolute() or output.exists():
        raise ValueError('invalid-output')
    quality = request.get('quality', 'best')
    if quality not in ['best', '720p', '1080p', 'audio']:
        raise ValueError('invalid-quality')
    if output.suffix != ('.m4a' if quality == 'audio' else '.mp4'):
        raise ValueError('invalid-output')
    # Single-file formats avoid leaving a separate FFmpeg process on cancellation.
    # Requested video quality is an upper bound; actual resolution is reported.
    fmt = 'bestaudio[ext=m4a]' if quality == 'audio' else 'best[ext=mp4][vcodec!=none][acodec!=none]' + (f'[height<={quality[:-1]}]' if quality in ['720p', '1080p'] else '')
    last = [0.0]

    def progress(value):
        now = time.monotonic()
        if now - last[0] < 0.5:
            return
        last[0] = now
        total = value.get('total_bytes') or value.get('total_bytes_estimate')
        done = value.get('downloaded_bytes')
        if isinstance(total, (int, float)) and total > 0 and isinstance(done, (int, float)):
            emit({'type': 'progress', 'progress': min(1, max(0, done / total))})

    class QuietLogger:
        def debug(self, *_): pass
        def warning(self, *_): pass
        def error(self, *_): pass

    options = {
        'format': fmt, 'outtmpl': str(output), 'noplaylist': True,
        'cachedir': False, 'cookiefile': None, 'usenetrc': False,
        'plugin_dirs': [], 'remote_components': [],
        'quiet': True, 'no_warnings': True, 'logger': QuietLogger(),
        'progress_hooks': [progress], 'max_filesize': 2 * 1024**3,
        'socket_timeout': 30, 'retries': 2, 'fragment_retries': 2,
        'overwrites': False, 'restrictfilenames': True,
        'js_runtimes': {'node': {'path': request['nodeExecutable']}},
    }
    url = 'https://www.youtube.com/watch?v=' + request['videoId']
    with YoutubeDL(options) as client:
        info = client.extract_info(url, download=False)
        if not isinstance(info, dict) or info.get('id') != request['videoId'] or info.get('_type', 'video') != 'video' or info.get('is_live') or info.get('has_drm'):
            raise ValueError('download-unavailable')
        if not info.get('url') or info.get('protocol') == 'mhtml':
            raise ValueError('download-format-unavailable')
        # Reuse this exact extracted item rather than resolving a playlist or another target.
        client.process_info(info)
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError('output-invalid')
    emit({'type': 'result', 'height': info.get('height')})


def main():
    if os.name == 'nt':
        isolate_windows_children()
    else:
        isolate_posix_children()
    if len(sys.argv) == 3 and sys.argv[1] == '--supervise':
        sys.exit(supervise(sys.argv[2]))
    request = json.loads(sys.stdin.buffer.readline(32769))
    if not isinstance(request, dict) or not re.fullmatch(r'[A-Za-z0-9_-]{11}', str(request.get('videoId', ''))):
        raise ValueError('invalid-video')
    if request.get('operation') == 'transcript':
        transcript(request)
    elif request.get('operation') == 'download':
        download(request)
    else:
        raise ValueError('unsupported-operation')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # health: allow BLE001 -- Extractor errors may contain account URLs; only fixed codes may cross this IPC stream.
        # Upstream exception strings can contain URLs/account information; send fixed codes only.
        names = {'TranscriptsDisabled': 'transcript-unavailable', 'NoTranscriptFound': 'transcript-language-unavailable', 'RequestBlocked': 'transcript-rate-limited', 'IpBlocked': 'transcript-rate-limited', 'VideoUnavailable': 'video-unavailable'}
        code = str(exc) if isinstance(exc, ValueError) and re.fullmatch(r'[a-z-]{1,60}', str(exc)) else names.get(type(exc).__name__, 'media-extractor-failed')
        emit({'type': 'result', 'error': code})
        sys.exit(1)
