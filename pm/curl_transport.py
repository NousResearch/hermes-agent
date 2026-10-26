"""curl-backed download transport, used when urllib cannot complete a request.

Python's urllib negotiates TLS through OpenSSL. Some upstream hosts on some networks
reset that handshake outright — observed on one Windows host, where
``huggingface.co`` reset ~90% of OpenSSL handshakes (5 of 60 succeeded) while the
same host's Schannel-backed ``curl.exe`` succeeded 100% (60 of 60). The reset lands
during the handshake, before a single byte moves, so no amount of retrying inside
urllib helps: every probe and every range worker dies the same way.

curl is the transport that does work there, and it ships with every platform Hermes
supports (Windows 10+ ``System32\\curl.exe``, macOS, Linux). The downloader keeps
curl as a *fallback only* — a healthy network never sees it — so behaviour on a
working connection is unchanged.

This module is deliberately transport-only: it returns plain tuples, owns no
partial state, and never imports :mod:`pm.downloader`. Probing policy, partials,
hash verification and atomic publication stay in the downloader, so the curl path
publishes bytes that are exactly as trusted as the urllib path's.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Same browser-shaped UA the urllib path sends; some CDNs reject unknown tool UAs.
UA = "Mozilla/5.0 (X11; Linux x86_64) hermes-pm/1.0"

# curl's own in-run retry budget. It covers a dropped connection mid-transfer; the
# downloader's outer retry still bounds a run that never succeeds.
_RETRIES = 20

# Abort a transfer whose throughput collapses, so a stalled socket cannot hold a
# download open indefinitely. Generous: a 16 GB model at 5 MB/s is fine.
_STALL_SECONDS = 120
_STALL_MIN_BYTES = 1024

# Probe budget. The probe asks for a single byte, so this only bounds a hung connect.
_PROBE_TIMEOUT = 60

# A probe that gets a 200 (server ignored Range) would otherwise make curl write the
# whole file just to learn it is not ranged; cap it well above the 1 byte asked for.
_PROBE_MAX_BYTES = 1 << 20

# (total_bytes, ranged, strong_etag); None when curl cannot answer at all.
ProbeResult = tuple[int, bool, str]

_PROGRESS_POLL_SECONDS = 0.5


class CurlError(RuntimeError):
    """curl was invoked and did not produce the bytes it was asked for."""


def executable() -> Optional[str]:
    """Locate a usable curl, or None when this host has none.

    ``shutil.which`` first (PATH, including a mingw/MSYS curl), then the curl Windows
    ships in System32, then anything bundled under the Hermes tools directory.
    """
    for name in ("curl", "curl.exe"):
        found = shutil.which(name)
        if found:
            return found
    roots = []
    system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
    if system_root:
        roots.append(Path(system_root) / "System32" / "curl.exe")
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        roots.extend(sorted(Path(hermes_home).glob("tools/*/curl.exe")))
        roots.extend(sorted(Path(hermes_home).glob("tools/*/bin/curl.exe")))
    for candidate in roots:
        if candidate.is_file():
            return str(candidate)
    return None


def _config_allows() -> bool:
    """``download.allow_curl_fallback`` from config.yaml; True when absent or unreadable.

    Behaviour belongs in config, not in an env var — ``.env`` is secrets only. A
    missing or broken config leaves the fallback ON on purpose: an unreachable host
    is exactly when it earns its keep, and a malformed config file must not take
    downloads down with it.
    """
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
    except ModuleNotFoundError as exc:
        # pm can run where the CLI package is not importable (a bare PM install).
        return exc.name not in {"hermes_cli", "hermes_cli.config"}
    except ImportError:
        return True
    try:
        value = cfg_get(load_config_readonly(), "download", "allow_curl_fallback", default=True)
    except Exception:
        return True
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def enabled() -> bool:
    """True when the curl fallback may be used.

    ``download.allow_curl_fallback: false`` turns it off for a run — useful to prove
    a failure is not the fallback masking one.
    """
    return _config_allows() and executable() is not None


def _strong_etag(value: str) -> str:
    return value if value.startswith('"') and value.endswith('"') else ""


def _last_header_block(raw: str) -> tuple[dict[str, str], Optional[int]]:
    """Headers and status of the FINAL response hop.

    ``-L`` writes every hop's headers to the dump file; only the last block describes
    the representation actually being downloaded.
    """
    blocks = [block for block in re.split(r"\r?\n\r?\n", raw.strip()) if block.strip()]
    if not blocks:
        return {}, None
    lines = blocks[-1].splitlines()
    status: Optional[int] = None
    if lines:
        match = re.match(r"HTTP/[\d.]+\s+(\d{3})", lines[0])
        if match:
            status = int(match.group(1))
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return headers, status


def probe(url: str, *, timeout: float = _PROBE_TIMEOUT) -> Optional[ProbeResult]:
    """Ask the server how big ``url`` is and whether it honors Range.

    Mirrors the urllib probe's contract: ``Range: bytes=0-0`` for a strong validator
    and a total, or a plain ``Content-Length`` when the server ignores Range. Returns
    None when curl is missing or the server will not answer — the caller then keeps
    its own failure rather than substituting a guess.
    """
    exe = executable()
    if not exe:
        return None
    with tempfile.TemporaryDirectory(prefix="hermes-curl-probe-") as tmp:
        header_file = Path(tmp) / "headers.txt"
        body_file = Path(tmp) / "body"
        try:
            proc = subprocess.run(
                [exe, "-sS", "-L", "--max-time", str(int(timeout)),
                 "--connect-timeout", "30", "--max-filesize", str(_PROBE_MAX_BYTES),
                 # A reset handshake is bursty: a single attempt fails often enough
                 # that an unretried probe would abandon a download curl could make.
                 "--retry", str(_RETRIES), "--retry-delay", "2", "--retry-all-errors",
                 "-r", "0-0", "-A", UA, "-D", str(header_file), "-o", str(body_file), url],
                capture_output=True, text=True, timeout=timeout + 30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("curl probe of %s failed to run: %s", url, exc)
            return None
        try:
            raw = header_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    # 0 = ranged single byte. 63 = the server ignored Range and the body overran the
    # probe cap; the headers are still the ones we need.
    if proc.returncode not in (0, 63):
        logger.debug("curl probe of %s exited %s: %s", url, proc.returncode,
                     (proc.stderr or "").strip()[:200])
        return None
    headers, status = _last_header_block(raw)
    if status is None:
        return None
    ranged = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers.get("content-range", ""))
    if ranged:
        return int(ranged.group(3)), True, _strong_etag(headers.get("etag", ""))
    length = headers.get("content-length", "")
    if status == 200 and length.isdecimal():
        return int(length), False, _strong_etag(headers.get("etag", ""))
    return None


def fetch(
    url: str,
    dest: Path,
    *,
    timeout: float = 0,
    on_progress: Optional[Callable[[int], None]] = None,
) -> int:
    """Download ``url`` to ``dest`` and return the number of bytes written.

    ``--fail`` turns an HTTP error status into a non-zero exit instead of writing an
    error page into the destination. ``-C -`` resumes a genuine partial; the caller
    must ensure ``dest`` holds real bytes, not a preallocated skeleton, before asking
    for a resume.

    ``on_progress`` is called roughly twice a second with the bytes on disk, so a long
    transfer keeps reporting a live bar instead of appearing stuck.
    """
    exe = executable()
    if not exe:
        raise CurlError("curl is not available")
    dest = Path(dest)
    args = [
        exe, "-sS", "-L", "--fail",
        "--retry", str(_RETRIES), "--retry-delay", "2", "--retry-all-errors",
        "--connect-timeout", "30",
        "--speed-time", str(_STALL_SECONDS), "--speed-limit", str(_STALL_MIN_BYTES),
        "-A", UA,
    ]
    if dest.exists():
        args += ["-C", "-"]
    args += ["-o", str(dest)]
    if timeout:
        args += ["--max-time", str(int(timeout))]
    args.append(url)

    try:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        raise CurlError(f"could not run curl: {exc}") from exc

    try:
        while proc.poll() is None:
            if on_progress is not None and dest.exists():
                try:
                    on_progress(dest.stat().st_size)
                except OSError:
                    pass
            time.sleep(_PROGRESS_POLL_SECONDS)
        stderr = proc.stderr.read() if proc.stderr else ""
    finally:
        if proc.poll() is None:  # pragma: no cover - only on an unhandled signal
            proc.kill()
            proc.wait()

    if proc.returncode != 0:
        raise CurlError(f"curl exited {proc.returncode}: {(stderr or '').strip()[:300]}")
    size = dest.stat().st_size if dest.exists() else 0
    if on_progress is not None:
        on_progress(size)
    return size
