"""Local validation harness for Auto Safe PR path policy.

Covers allowlist acceptance, unauthorized mixes, broad globs, secrets,
cache/build/db artifacts, and no-change runs — without network or secrets.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Import from repository scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from auto_safe_pr_path_policy import (  # noqa: E402
    ALLOWED_GENERATED_PREFIXES,
    classify_path,
    is_allowed_prefix,
    is_broad_glob,
    main,
    validate_paths,
)


def test_allowlist_is_fixed_and_non_empty():
    assert ALLOWED_GENERATED_PREFIXES
    for prefix in ALLOWED_GENERATED_PREFIXES:
        assert prefix.endswith("/"), "prefixes must end with /"
        assert "*" not in prefix
        assert not prefix.startswith("/")


def test_allowed_generated_paths():
    allowed_examples = [
        "website/static/api/skills-index.json",
        "website/docs/guide.md",
        "docs/generated/index.md",
        "docs/README.md",
    ]
    for path in allowed_examples:
        assert classify_path(path) == "ok", path
        assert is_allowed_prefix(path)


def test_unauthorized_path_mixed_with_allowed():
    paths = [
        "docs/ok.md",
        "src/hermes_cli/config.py",  # unauthorized — not generated prefix
    ]
    failures = validate_paths(paths)
    assert len(failures) == 1
    assert failures[0][0] == "src/hermes_cli/config.py"
    assert failures[0][1] == "outside_allowlist"


@pytest.mark.parametrize(
    "glob",
    ["*", "**", "**/", "path/*", "path/**", "docs/*", "website/**", ".", "./", "/", "**/*"],
)
def test_rejects_broad_globs(glob: str):
    assert is_broad_glob(glob) or classify_path(glob) in {
        "broad",
        "outside_allowlist",
    }
    # Explicit star forms must never be ok
    assert classify_path(glob) != "ok"


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "secrets/token.txt",
        "secret/api.key",
        "credentials/aws.json",
        "id_rsa",
        "keys/id_ed25519",
        "certs/server.pem",
        "auth/service.key",
        "store/client.p12",
    ],
)
def test_rejects_secret_key_paths(path: str):
    assert classify_path(path) == "secret"


@pytest.mark.parametrize(
    "path",
    [
        "node_modules/pkg/index.js",
        "dist/bundle.js",
        "build/output.o",
        "target/release/app",
        "src/__pycache__/x.pyc",
        ".pytest_cache/v/cache",
        ".mypy_cache/3.11/x.json",
        ".ruff_cache/c/x",
        "logs/app.log",
        "data/local.sqlite",
        "data/app.sqlite3",
        "state/app.db",
        "backup/dump.bak",
    ],
)
def test_rejects_cache_build_database_artifacts(path: str):
    assert classify_path(path) == "artifact"


def test_no_change_run_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Empty path set is a successful no-op."""
    out = tmp_path / "allowed.txt"
    code = main(["--write-allowed", str(out)])
    assert code == 0
    assert out.read_text(encoding="utf-8") == ""


def test_cli_from_paths_file_fails_closed(tmp_path: Path):
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text(
        "docs/ok.md\n"
        "website/static/api/x.json\n"
        "src/unauthorized.py\n",
        encoding="utf-8",
    )
    code = main(["--paths-file", str(paths_file)])
    assert code == 1


def test_cli_from_paths_file_all_allowed(tmp_path: Path):
    paths_file = tmp_path / "paths.txt"
    allowed_out = tmp_path / "allowed.txt"
    paths_file.write_text(
        "docs/ok.md\n"
        "website/static/api/x.json\n",
        encoding="utf-8",
    )
    code = main(["--paths-file", str(paths_file), "--write-allowed", str(allowed_out)])
    assert code == 0
    written = [line for line in allowed_out.read_text(encoding="utf-8").splitlines() if line]
    assert written == ["docs/ok.md", "website/static/api/x.json"]


def test_script_is_executable_via_python():
    """Smoke: scripts/auto_safe_pr_path_policy.py --print-allowlist."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "auto_safe_pr_path_policy.py"), "--print-allowlist"],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    for prefix in ALLOWED_GENERATED_PREFIXES:
        assert prefix in result.stdout
