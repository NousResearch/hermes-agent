"""Build-level wheel/sdist coverage for Research Protocol data files."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from uuid import uuid4
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "plugins" / "research_protocol"
DATA_SUFFIXES = {".json", ".yaml", ".sql", ".md"}

SECRET_PATTERNS = (
    ("postgresql_dsn", re.compile(rb"postgres(?:ql)?://[^\s\"'<>]+", re.IGNORECASE)),
    ("private_key_pem", re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    (
        "credential_assignment",
        re.compile(
            rb"\b(?:token|password|secret|api[_-]?key)\s*[:=]\s*"
            rb"[\"']?[^\s,;\"']+",
            re.IGNORECASE,
        ),
    ),
)


def _secret_canaries() -> tuple[bytes, bytes]:
    nonce = uuid4().hex.encode("ascii")
    return b"token-" + nonce, b"postgresql://" + nonce


def _required_data_paths() -> set[str]:
    required = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in PLUGIN_ROOT.rglob("*")
        if path.is_file() and path.suffix in DATA_SUFFIXES
    }
    required.add("plugins/research_protocol/plugin.yaml")
    return required


def _wheel_entries(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }


def _sdist_entries(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        entries = {}
        for member in members:
            extracted = archive.extractfile(member)
            assert extracted is not None
            entries[member.name.split("/", 1)[1]] = extracted.read()
    return entries


def _scan_research_protocol_payloads(entries: dict[str, bytes]) -> list[str]:
    """Scan only packaged payloads below the plugin subtree.

    Keeping the path filter here prevents the scanner's own test regexes, which
    are included in the sdist under ``tests/``, from matching themselves.
    """

    findings = []
    for name, payload in entries.items():
        if not name.startswith("plugins/research_protocol/"):
            continue
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(payload):
                findings.append(f"{name}:{label}")
    return findings


def test_secret_scanner_detects_synthetic_payloads_without_scanning_other_subtrees():
    malicious = {
        "plugins/research_protocol/fixtures/dsn.txt": b"postgresql://user:pw@host/db",
        "plugins/research_protocol/fixtures/key.txt": (
            b"-----BEGIN PRIVATE KEY-----\\nsecret\\n-----END PRIVATE KEY-----"
        ),
        "plugins/research_protocol/fixtures/token.txt": b"api_key=synthetic-token",
        "tests/test_research_protocol_packaging.py": (
            b"postgresql://source-code-must-not-be-scanned"
        ),
    }

    findings = _scan_research_protocol_payloads(malicious)

    assert {
        "plugins/research_protocol/fixtures/dsn.txt:postgresql_dsn",
        "plugins/research_protocol/fixtures/key.txt:private_key_pem",
        "plugins/research_protocol/fixtures/token.txt:credential_assignment",
    } <= set(findings)
    assert all(not finding.startswith("tests/") for finding in findings)


def test_wheel_and_sdist_ship_every_versioned_protocol_data_file(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None, "uv is required by the Hermes development workflow"
    canaries = _secret_canaries()
    env = os.environ.copy()
    env.update(
        {
            "RESEARCH_PROTOCOL_TOKEN": canaries[0].decode(),
            "RESEARCH_PROTOCOL_DSN": canaries[1].decode(),
        }
    )
    subprocess.run(
        [uv, "build", "--offline", "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(tmp_path.glob("*.whl"))
    sdist = next(tmp_path.glob("*.tar.gz"))
    required = _required_data_paths()
    wheel_entries = _wheel_entries(wheel)
    sdist_entries = _sdist_entries(sdist)

    assert required <= wheel_entries.keys()
    assert required <= sdist_entries.keys()
    for canary in canaries:
        assert all(canary not in payload for payload in wheel_entries.values())
        assert all(canary not in payload for payload in sdist_entries.values())
    assert _scan_research_protocol_payloads(wheel_entries) == []
    assert _scan_research_protocol_payloads(sdist_entries) == []
