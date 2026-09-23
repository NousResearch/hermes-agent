"""Runtime qualification for SQLite in the published Docker image."""

from __future__ import annotations

import json
import subprocess


_SQLITE_PROBE = r"""
import json
import sqlite3

from hermes_cli.sqlite_runtime import is_sqlite_wal_reset_vulnerable

db = sqlite3.connect(":memory:")
try:
    db.execute("CREATE VIRTUAL TABLE docs USING fts5(content, tokenize='trigram')")
    db.execute("INSERT INTO docs VALUES ('hermes')")
    matches = db.execute(
        "SELECT count(*) FROM docs WHERE docs MATCH 'erm'"
    ).fetchone()[0]
finally:
    db.close()

print(json.dumps({
    "sqlite_version": sqlite3.sqlite_version,
    "wal_reset_vulnerable": is_sqlite_wal_reset_vulnerable(
        sqlite3.sqlite_version_info
    ),
    "trigram_matches": matches,
}))
"""


_CJK_TOKENIZER_PROBE = r"""
import json
import os
import sqlite3
from pathlib import Path

path = os.environ["HERMES_FTS5_CJK_SO"]
artifact = Path(path)
db = sqlite3.connect(":memory:")
try:
    db.enable_load_extension(True)
    db.load_extension(path)
    db.enable_load_extension(False)
    db.execute(
        "CREATE VIRTUAL TABLE docs USING "
        "fts5(content, tokenize='cjk_unicode61')"
    )
    db.execute("INSERT INTO docs VALUES ('웅기가 말했다')")
    matches = db.execute(
        "SELECT count(*) FROM docs WHERE docs MATCH '웅기'"
    ).fetchone()[0]
finally:
    db.close()

print(json.dumps({
    "hermes_home": os.environ["HERMES_HOME"],
    "cjk_so": path,
    "artifact_is_file": artifact.is_file(),
    "artifact_readable": os.access(artifact, os.R_OK),
    "artifact_mode": oct(artifact.stat().st_mode & 0o777),
    "matches": matches,
}))
"""


def test_image_links_fixed_sqlite_with_fts5_trigram(built_image: str) -> None:
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "hermes",
            "--entrypoint",
            "/opt/hermes/.venv/bin/python",
            built_image,
            "-c",
            _SQLITE_PROBE,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"SQLite runtime probe failed: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    payload = json.loads(result.stdout)
    assert payload["wal_reset_vulnerable"] is False, payload
    assert payload["trigram_matches"] == 1, payload


def test_image_ships_loadable_cjk_tokenizer_outside_data_volume(
    built_image: str,
) -> None:
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "hermes",
            "--volume",
            "/opt/data",
            "--entrypoint",
            "/opt/hermes/.venv/bin/python",
            built_image,
            "-c",
            _CJK_TOKENIZER_PROBE,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"CJK tokenizer probe failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    payload = json.loads(result.stdout)
    assert payload == {
        "hermes_home": "/opt/data",
        "cjk_so": "/opt/hermes/lib/libfts5_cjk.so",
        "artifact_is_file": True,
        "artifact_readable": True,
        "artifact_mode": "0o644",
        "matches": 1,
    }
