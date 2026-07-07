"""hc-452: prerequisites-stage network connectivity check caching.

``check_network_prerequisites`` probes two URLs (pypi.org, duckduckgo.com)
with an 8s max-time each on every bootstrap run -- including a runtime
version *update*, where connectivity essentially never changes between two
updates minutes or hours apart. This is one of the two network round-trips
Kael's 2026-07-08 report measured as part of the observed 8.6s
"prerequisites" stage on a re-run.

These tests pin a time-boxed cache keyed by a lightweight machine fingerprint
(OS/DISTRO/hostname) under ``$HERMES_HOME/bootstrap-cache/network-check.marker``:
a fresh, matching cache entry skips both curl probes entirely; a stale,
mismatched, or absent one always falls through to a real probe (fail-open --
this cache can only ever make a would-be-slow check fast, never make a real
failure invisible, since a failed probe is never cached).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

pytestmark = pytest.mark.skipif(
    subprocess.run(["bash", "--version"], capture_output=True).returncode != 0,
    reason="needs bash",
)


def _extract_function(name: str) -> str:
    text = INSTALL_SH.read_text()
    match = re.search(rf"^{re.escape(name)}\(\) \{{.*?\n\}}", text, re.DOTALL | re.MULTILINE)
    assert match is not None, f"{name}() not found in install.sh"
    return match.group(0)


def _harness(ttl_seconds: int = 3600, distro: str = "macos", os_name: str = "macos", curl_ok: bool = True) -> str:
    curl_stub = "curl() { return 0; }" if curl_ok else 'curl() { return 1; }'
    return "\n".join(
        [
            'log_info() { echo "INFO: $*"; }',
            'log_success() { echo "SUCCESS: $*"; }',
            'log_warn() { echo "WARN: $*"; }',
            'log_error() { echo "ERROR: $*"; }',
            'mark_stage_skipped() { : > "${STAGE_SKIP_MARKER:-/dev/null}" 2>/dev/null || true; }',
            f"DISTRO={distro}",
            f"OS={os_name}",
            curl_stub,
            _extract_function("_network_check_fingerprint"),
            f"NETWORK_CHECK_CACHE_TTL_SECONDS={ttl_seconds}",
            _extract_function("check_network_prerequisites"),
        ]
    )


def _run(script: str, cwd: Path, hermes_home: Path, **harness_kwargs) -> subprocess.CompletedProcess:
    env_prefix = f'export HERMES_HOME="{hermes_home}"\n'
    full = f"set -e\n{env_prefix}{_harness(**harness_kwargs)}\n{script}"
    return subprocess.run(["bash", "-c", full], cwd=cwd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Static guards
# ---------------------------------------------------------------------------


def test_check_network_prerequisites_still_has_both_probe_urls() -> None:
    fn = _extract_function("check_network_prerequisites")
    assert "https://pypi.org/simple/" in fn
    assert "https://duckduckgo.com/" in fn


def test_check_network_prerequisites_consults_cache_before_probing() -> None:
    fn = _extract_function("check_network_prerequisites")
    idx_cache_read = fn.find('if [ -f "$cache_file" ]')
    idx_probe_loop = fn.find('for url in "${checks[@]}"')
    assert idx_cache_read != -1, "must check for a cached result"
    assert idx_probe_loop != -1
    assert idx_cache_read < idx_probe_loop, "cache check must happen before the curl probe loop"


def test_check_network_prerequisites_never_caches_a_failure() -> None:
    fn = _extract_function("check_network_prerequisites")
    idx_failed_branch = fn.find('if [ "$failed" = false ]')
    assert idx_failed_branch != -1
    # After the failure branch (i.e. once we know at least one probe failed),
    # the function must remove any cache file rather than writing one.
    tail = fn[idx_failed_branch:]
    assert 'rm -f "$cache_file"' in tail, (
        "a failed connectivity check must never leave a stale success cached "
        "for the rest of the TTL window"
    )


# ---------------------------------------------------------------------------
# Behavioral
# ---------------------------------------------------------------------------


def test_first_call_probes_and_writes_cache(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    res = _run("check_network_prerequisites", tmp_path, home)
    assert res.returncode == 0, res.stderr
    assert "cached" not in res.stdout
    marker = home / "bootstrap-cache" / "network-check.marker"
    assert marker.exists(), "a successful check must write the cache marker"
    lines = marker.read_text().splitlines()
    assert len(lines) == 2, f"marker should have fingerprint + epoch lines, got: {lines}"
    assert lines[1].isdigit(), "second line must be a numeric epoch timestamp"


def test_second_call_with_fresh_cache_skips_curl_entirely(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    script = (
        "check_network_prerequisites\n"
        # Redefine curl to prove it is never invoked on the second call.
        'curl() { echo "CURL_INVOKED"; return 1; }\n'
        "check_network_prerequisites"
    )
    res = _run(script, tmp_path, home)
    assert res.returncode == 0, res.stderr
    assert "CURL_INVOKED" not in res.stdout, "second call must not re-probe when the cache is fresh"
    assert res.stdout.count("cached") == 1, "exactly the second call should report a cache hit"


def test_different_fingerprint_forces_reprobe(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    marker_dir = home / "bootstrap-cache"
    marker_dir.mkdir(parents=True)
    # Seed a cache entry for a different (fake) machine fingerprint.
    (marker_dir / "network-check.marker").write_text("some-other-machine|fingerprint\n9999999999\n")

    res = _run("check_network_prerequisites", tmp_path, home)
    assert res.returncode == 0, res.stderr
    assert "cached" not in res.stdout, "a fingerprint mismatch must force a real probe"


def test_expired_ttl_forces_reprobe(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    res = _run("check_network_prerequisites", tmp_path, home, ttl_seconds=0)
    assert res.returncode == 0, res.stderr
    # Immediately call again with the same (expired-instantly) TTL.
    res2 = _run("check_network_prerequisites", tmp_path, home, ttl_seconds=0)
    assert res2.returncode == 0, res2.stderr
    assert "cached" not in res2.stdout, "TTL=0 must never produce a cache hit"


def test_malformed_cache_file_fails_open_to_a_real_probe(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    marker_dir = home / "bootstrap-cache"
    marker_dir.mkdir(parents=True)
    (marker_dir / "network-check.marker").write_text("garbage\nnot-a-number\n")

    res = _run("check_network_prerequisites", tmp_path, home)
    assert res.returncode == 0, res.stderr
    assert "cached" not in res.stdout, "a malformed cache entry must never be trusted"


def test_failed_probe_does_not_write_or_keep_a_cache_entry(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    res = _run("check_network_prerequisites", tmp_path, home, curl_ok=False)
    assert res.returncode == 0, res.stderr
    marker = home / "bootstrap-cache" / "network-check.marker"
    assert not marker.exists(), "a failed connectivity check must not leave a cache entry behind"


def test_cache_write_failure_does_not_abort_the_check(tmp_path: Path) -> None:
    """A read-only HERMES_HOME (disk full, permissions) must degrade to
    'skip caching, but the check itself still succeeds' -- never a hard
    failure of the prerequisites stage."""
    home = tmp_path / "home"
    home.mkdir(mode=0o555)
    try:
        res = _run("check_network_prerequisites", tmp_path, home)
        assert res.returncode == 0, res.stderr
        assert "Internet connectivity looks good" in res.stdout
    finally:
        home.chmod(0o755)
