"""Backend-neutral local evidence and single-attempt process recording. Python 3.9+.

No shell, retries, credential discovery, classification or inferred authorization.
Callers decide which commands are authorized. Output may contain private mail.
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False).encode('utf-8'))


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def require_text(value, name):
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError(name + ' must be a nonempty string without NUL; never repair an ID')
    return value


def unique_record(payload, collection, name_field, name):
    """Select from actual parsed JSON; never generate IDs or choose first duplicate."""
    if not isinstance(payload, dict) or 'error' in payload or not isinstance(payload.get(collection), list):
        raise ValueError('Expected successful backend-specific collection JSON')
    matches = [r for r in payload[collection] if isinstance(r, dict) and r.get(name_field) == name]
    if len(matches) != 1:
        raise ValueError('Need exactly one matching target in the supplied scope')
    require_text(matches[0].get('id'), 'Target ID')
    return matches[0]


def run_recorded(argv, cwd, directory, timeout=30, stopped=lambda: False,
                 before_launch=lambda invocation, path: None):
    """Record actual resolved argv/cwd before one process launch; save raw outputs.

    A new exclusive directory prevents replay into the same capture. A crash can
    leave invocation.json without result.json: reconcile, never assume no effect.
    before_launch lets an orchestrator durably journal submission after recording
    argv and before launching. It must not itself execute the command.
    """
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or '\x00' in x for x in argv):
        raise ValueError('Supply a string argv array, not shell text')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Timeout must be finite and positive')
    cwd, directory = Path(cwd), Path(directory)
    if not cwd.is_absolute() or not cwd.is_dir() or not directory.is_absolute():
        raise ValueError('Use an existing absolute native cwd and a new absolute capture directory')
    executable = shutil.which(argv[0])
    if executable is None:
        raise ValueError('Executable cannot be resolved')
    actual = [str(Path(executable).resolve())] + list(argv[1:])
    directory.mkdir(mode=0o700)  # No reuse; parent must already exist.
    invocation = {'schema_version': 1, 'argv': actual, 'cwd': str(cwd.resolve()),
                  'shell': False, 'timeout_seconds': timeout, 'recorded_at': now()}
    save_new(directory/'invocation.json', invocation)
    result = {'status': 'not_launched', 'returncode': None, 'started_at': None,
              'invocation_sha256': digest((directory/'invocation.json').read_bytes())}
    process = None
    try:
        with (directory/'stdout.bin').open('xb') as out, (directory/'stderr.bin').open('xb') as err:
            if stopped():
                result['status'] = 'stopped_before_launch'
            else:
                before_launch(invocation, directory/'invocation.json')
                if stopped():
                    result['status'] = 'stopped_before_launch'
                else:
                    process = subprocess.Popen(actual, cwd=cwd, shell=False, stdin=subprocess.DEVNULL,
                                               stdout=out, stderr=err)
                    result['started_at'] = now()
                    deadline = time.monotonic() + timeout
                    while True:
                        if stopped() or time.monotonic() >= deadline:
                            result['status'] = 'stopped' if stopped() else 'timeout'
                            process.kill()
                            process.wait()
                            break
                        try:
                            process.wait(timeout=min(.1, max(.001, deadline-time.monotonic())))
                            result['status'] = 'completed'
                            break
                        except subprocess.TimeoutExpired:
                            pass
                    result['returncode'] = process.returncode
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        result['status'] = 'interrupted' if process is not None else 'launch_failed'
        raise
    finally:
        result['finished_at'] = now()
        for name in ('stdout', 'stderr'):
            path = directory/(name+'.bin')
            if path.exists():
                result[name+'_sha256'] = digest(path.read_bytes())
        save_new(directory/'result.json', result)
    return str(directory)


def load_capture(directory):
    directory = Path(directory)
    invocation_bytes = (directory/'invocation.json').read_bytes()
    invocation = json.loads(invocation_bytes)
    result = json.loads((directory/'result.json').read_bytes())
    stdout, stderr = (directory/'stdout.bin').read_bytes(), (directory/'stderr.bin').read_bytes()
    if (result.get('invocation_sha256') != digest(invocation_bytes)
            or result.get('stdout_sha256') != digest(stdout)
            or result.get('stderr_sha256') != digest(stderr)):
        raise ValueError('Capture bytes changed')
    return invocation, result, stdout


def successful_json(directory):
    invocation, result, stdout = load_capture(directory)
    if result.get('status') != 'completed' or type(result.get('returncode')) is not int or result['returncode'] != 0:
        raise ValueError('Command did not complete successfully; inspect private capture')
    payload = json.loads(stdout)
    if not isinstance(payload, dict) or 'error' in payload:
        raise ValueError('Expected successful object JSON')
    return invocation, result, payload
