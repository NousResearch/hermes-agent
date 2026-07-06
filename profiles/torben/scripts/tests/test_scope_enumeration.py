from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", "/Users/ericfreeman/.hermes/hermes-agent"))


def _uv_python(code: str, *args: str) -> dict:
    env = dict(os.environ)
    env["HERMES_REPO_ROOT"] = str(REPO_ROOT)
    result = subprocess.run(
        ["uv", "run", "python", "-c", code, *args],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(result.stdout)


def _write_fixture_config(tmp_path: Path, *, work_config_scope: str, personal_config_scope: str) -> Path:
    personal_token = tmp_path / "personal-token.json"
    work_token = tmp_path / "work-token.json"
    for path, token, scope in [
        (personal_token, "personal-token", personal_config_scope),
        (work_token, "work-token", work_config_scope),
    ]:
        path.write_text(
            json.dumps(
                {
                    "token": token,
                    "expiry": "2027-01-01T00:00:00+00:00",
                    "scopes": [scope],
                }
            ),
            encoding="utf-8",
        )
    config = tmp_path / "google_accounts.yaml"
    config.write_text(
        f"""
accounts:
  personal:
    alias: personal
    email: personal@example.com
    enabled: true
    role: personal
    scopes:
      - {personal_config_scope}
    token_path: {personal_token}
    client_secret_path: {tmp_path / "personal-client.json"}
  work:
    alias: work
    email: eric@work.example
    enabled: true
    role: work
    scopes:
      - {work_config_scope}
    token_path: {work_token}
    client_secret_path: {tmp_path / "work-client.json"}
""".lstrip(),
        encoding="utf-8",
    )
    return config


def _inventory(config: Path, token_scope_map: dict[str, str]) -> dict:
    code = r'''
import json
import sys
from pathlib import Path

scripts_dir = Path(sys.argv[1])
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from torben_oauth_scope_inventory import build_scope_inventory

config = Path(sys.argv[2])
scope_map = json.loads(sys.argv[3])

def fake_tokeninfo(token):
    return {"scope": scope_map[token]}

inventory = build_scope_inventory(config_path=config, tokeninfo_fetcher=fake_tokeninfo)
print(json.dumps(inventory, sort_keys=True))
'''
    return _uv_python(code, str(SCRIPTS_DIR), str(config), json.dumps(token_scope_map))


def test_work_write_scope_raises_type_1_finding_and_gates_categories(tmp_path: Path) -> None:
    config = _write_fixture_config(
        tmp_path,
        personal_config_scope="https://www.googleapis.com/auth/gmail.modify",
        work_config_scope="https://www.googleapis.com/auth/gmail.modify",
    )

    inventory = _inventory(
        config,
        {
            "personal-token": "https://www.googleapis.com/auth/gmail.modify",
            "work-token": "https://www.googleapis.com/auth/gmail.modify",
        },
    )

    assert inventory["status"] == "type_1_findings"
    assert inventory["type_1_findings"][0]["account"] == "work"
    assert inventory["type_1_findings"][0]["service"] == "gmail"
    assert inventory["category_gates"]["gmail_archive"]["status"] == "blocked_type_1"
    assert inventory["category_gates"]["gmail_trash"]["floor"] == "packet_only"
    assert inventory["category_gates"]["calendar_edit"]["status"] == "blocked_type_1"


def test_work_read_only_scope_is_clean(tmp_path: Path) -> None:
    config = _write_fixture_config(
        tmp_path,
        personal_config_scope="https://www.googleapis.com/auth/gmail.modify",
        work_config_scope="https://www.googleapis.com/auth/gmail.readonly",
    )

    inventory = _inventory(
        config,
        {
            "personal-token": "https://www.googleapis.com/auth/gmail.modify",
            "work-token": "https://www.googleapis.com/auth/gmail.readonly",
        },
    )

    assert inventory["status"] == "clean"
    assert inventory["type_1_findings"] == []
    assert inventory["category_gates"]["gmail_archive"]["status"] == "clear"
