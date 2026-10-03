"""Tiny JSON/multipart HTTP helpers over urllib.

Deliberately not `requests`: this is a skill other people install, and the
fewer things it drags into their environment the better.
"""
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request


def post_json(url, payload, headers=None, timeout=180, retries=3):
    body = json.dumps(payload).encode()
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    return _send(url, body, h, "POST", timeout, retries)


def get_json(url, headers=None, timeout=60, retries=3):
    return _send(url, None, headers or {}, "GET", timeout, retries)


def post_multipart(url, fields, files, headers=None, timeout=300, retries=3):
    """fields: {name: str}. files: [(name, path)] — repeated names become arrays."""
    boundary = "----iconloop" + os.urandom(8).hex()
    out = b""
    for k, v in fields.items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, p in files:
        mime = mimetypes.guess_type(p)[0] or "application/octet-stream"
        out += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; '
            f'filename="{os.path.basename(p)}"\r\nContent-Type: {mime}\r\n\r\n'
        ).encode()
        out += open(p, "rb").read() + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    h = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    h.update(headers or {})
    return _send(url, out, h, "POST", timeout, retries)


# urllib's default User-Agent ("Python-urllib/3.x") is blocked outright by
# Cloudflare in front of Replicate — it returns 403 with error code 1010, which
# looks exactly like a bad token and is not. Send a real one.
UA = "icon-loop/0.1 (+https://github.com/samyost/animated-3d-icon)"


def _send(url, body, headers, method, timeout, retries):
    last = None
    headers = {"User-Agent": UA, **headers}
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="ignore")[:400]
            last = RuntimeError(f"HTTP {e.code} from {url.split('?')[0]}: {detail}")
            # 4xx other than rate-limiting will not fix itself on a retry.
            if e.code < 500 and e.code != 429:
                raise last
        except Exception as e:  # noqa: BLE001 - network flakiness of any shape
            last = e
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise last


def download(url, path, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(path, "wb") as f:
        f.write(r.read())
    return path
