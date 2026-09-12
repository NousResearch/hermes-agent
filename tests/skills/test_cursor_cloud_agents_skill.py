"""Contract tests for the bundled `cursor-cloud-agents` skill.

Feature request: #107480 — first-class Cursor Cloud Agents support under
`skills/autonomous-ai-agents/`, alongside the bundled claude-code / codex skills.

Two layers, stdlib + pytest only, no live network:

  1. Structural contract on SKILL.md (skills/AGENTS.md authoring standards):
     frontmatter fields, description hardline, section order, credential hygiene.
  2. Behavioral: drive the real request path in `scripts/cursor_cloud.py`
     against a stubbed `_urlopen` seam, asserting the URL, method, auth header,
     payload shape and CLI output for each lifecycle command.
"""

import base64
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "autonomous-ai-agents" / "cursor-cloud-agents"
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPT = SKILL_DIR / "scripts" / "cursor_cloud.py"
API_REFERENCE = SKILL_DIR / "references" / "api.md"

API_KEY = "crsr_test_key_do_not_use"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("cursor_cloud_undertest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")
        self.status = 200

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Transport:
    """Records every Request object and replays canned JSON responses."""

    def __init__(self, *payloads):
        self._payloads = list(payloads)
        self.requests = []

    def __call__(self, req, timeout):
        self.requests.append(req)
        payload = self._payloads.pop(0) if self._payloads else {}
        return _FakeResponse(payload)

    def headers(self, index=0):
        return {k.lower(): v for k, v in self.requests[index].headers.items()}

    def body(self, index=0):
        return json.loads(self.requests[index].data.decode("utf-8"))


@pytest.fixture
def transport(monkeypatch, mod):
    def _install(*payloads):
        t = _Transport(*payloads)
        monkeypatch.setattr(mod, "_urlopen", t)
        monkeypatch.setenv(mod.ENV_VAR, API_KEY)
        return t

    return _install


# --- structural contract ---------------------------------------------------


def test_skill_files_exist():
    assert SKILL_MD.is_file(), "SKILL.md must ship with the bundled skill"
    assert SCRIPT.is_file(), "the helper script must ship with the skill"
    assert API_REFERENCE.is_file(), "the endpoint reference must ship with the skill"


@pytest.fixture(scope="module")
def skill_text():
    return SKILL_MD.read_text(encoding="utf-8")


def test_frontmatter_declares_required_fields(skill_text):
    assert skill_text.startswith("---\n")
    head = skill_text.split("---", 2)[1]
    for field in ("name:", "description:", "version:", "author:", "license:", "platforms:"):
        assert field in head, f"frontmatter missing {field}"
    assert "name: cursor-cloud-agents" in head
    assert "tags:" in head


def test_description_is_one_short_sentence(skill_text):
    line = next(l for l in skill_text.splitlines() if l.startswith("description:"))
    desc = line.split(":", 1)[1].strip().strip('"')
    assert len(desc) <= 60, f"description is {len(desc)} chars: {desc!r}"
    assert desc.endswith(".")


def test_required_sections_in_order(skill_text):
    headings = ["## When to Use", "## Prerequisites", "## How to Run",
                "## Quick Reference", "## Procedure", "## Pitfalls", "## Verification"]
    positions = [skill_text.find(h) for h in headings]
    assert all(p != -1 for p in positions), f"missing section: {headings[positions.index(-1)]}"
    assert positions == sorted(positions), "sections must follow the documented order"


def test_credentials_come_from_the_environment(skill_text):
    assert "CURSOR_API_KEY" in skill_text, "the skill must document the env var"
    assert "https://api.cursor.com" in skill_text


# --- behavioral: request construction --------------------------------------


def test_api_key_missing_fails_closed(monkeypatch, mod, capsys):
    monkeypatch.delenv(mod.ENV_VAR, raising=False)
    rc = mod.main(["models"])
    err = capsys.readouterr().err
    assert rc == 2
    assert mod.ENV_VAR in err


def test_auth_header_is_basic_with_empty_password(mod):
    header = mod.auth_header(API_KEY)
    assert header.startswith("Basic ")
    decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
    assert decoded == f"{API_KEY}:"


def test_models_lists_from_the_fixed_host(transport, mod, capsys):
    t = transport({"items": [{"id": "composer-2", "name": "Composer 2"}]})
    assert mod.main(["models"]) == 0
    assert t.requests[0].full_url == "https://api.cursor.com/v1/models"
    assert t.requests[0].get_method() == "GET"
    assert t.headers()["authorization"] == mod.auth_header(API_KEY)
    assert "composer-2" in capsys.readouterr().out


def test_host_cannot_be_overridden_from_the_cli(mod):
    with pytest.raises(SystemExit):
        mod.main(["models", "--base-url", "https://attacker.example"])


def test_launch_builds_the_documented_create_agent_body(transport, mod, capsys):
    t = transport({"agent": {"id": "bc-1"}, "run": {"id": "run-1", "status": "CREATING"}})
    rc = mod.main([
        "launch", "--prompt", "Add a README", "--repo", "acme/widgets",
        "--ref", "main", "--model", "composer-2", "--auto-create-pr",
    ])
    assert rc == 0
    req = t.requests[0]
    assert req.get_method() == "POST"
    assert req.full_url == "https://api.cursor.com/v1/agents"
    body = t.body()
    assert body["prompt"]["text"] == "Add a README"
    assert body["repos"][0]["url"] == "https://github.com/acme/widgets"
    assert body["repos"][0]["startingRef"] == "main"
    assert body["model"] == {"id": "composer-2"}
    assert body["autoCreatePR"] is True
    # Branching default must stay on the documented "new cursor/... branch" path.
    assert body["workOnCurrentBranch"] is False
    out = capsys.readouterr().out
    assert "bc-1" in out and "run-1" in out
    assert API_KEY not in out, "the API key must never be echoed"


def test_launch_can_opt_into_the_current_branch(transport, mod):
    t = transport({"agent": {"id": "bc-1"}, "run": {"id": "run-1"}})
    mod.main(["launch", "--prompt", "p", "--repo", "https://github.com/acme/widgets.git",
              "--work-on-current-branch"])
    assert t.body()["workOnCurrentBranch"] is True
    assert t.body()["repos"][0]["url"] == "https://github.com/acme/widgets"


def test_followup_posts_a_new_run_with_mode(transport, mod):
    t = transport({"run": {"id": "run-2", "status": "CREATING"}})
    rc = mod.main(["followup", "bc-1", "--prompt", "also document it", "--mode", "plan"])
    assert rc == 0
    assert t.requests[0].get_method() == "POST"
    assert t.requests[0].full_url == "https://api.cursor.com/v1/agents/bc-1/runs"
    assert t.body() == {"prompt": {"text": "also document it"}, "mode": "plan"}


def test_status_reads_agent_and_latest_run(transport, mod, capsys):
    t = transport(
        {"id": "bc-1", "name": "README"},
        {"items": [{
            "id": "run-9",
            "status": "FINISHED",
            "result": "done",
            "git": {"branches": ["cursor/readme"],
                    "prs": ["https://github.com/acme/widgets/pull/7"]},
        }]},
    )
    assert mod.main(["status", "bc-1"]) == 0
    urls = [r.full_url for r in t.requests]
    assert urls == [
        "https://api.cursor.com/v1/agents/bc-1",
        "https://api.cursor.com/v1/agents/bc-1/runs?limit=1",
    ]
    out = capsys.readouterr().out
    assert "FINISHED" in out
    assert "https://github.com/acme/widgets/pull/7" in out


def test_cancel_targets_the_run(transport, mod):
    t = transport({"id": "run-9"})
    assert mod.main(["cancel", "bc-1", "--run", "run-9"]) == 0
    assert t.requests[0].get_method() == "POST"
    assert t.requests[0].full_url == "https://api.cursor.com/v1/agents/bc-1/runs/run-9/cancel"


def test_watch_polls_until_the_run_is_terminal(transport, mod, capsys):
    t = transport(
        {"id": "run-9", "status": "CREATING"},
        {"id": "run-9", "status": "RUNNING"},
        {"id": "run-9", "status": "FINISHED", "result": "shipped"},
    )
    rc = mod.main(["watch", "bc-1", "--run", "run-9", "--interval", "0", "--timeout", "30"])
    assert rc == 0
    assert len(t.requests) == 3, "polling must stop at the first terminal status"
    assert all(r.get_method() == "GET" for r in t.requests)
    assert t.requests[-1].full_url == "https://api.cursor.com/v1/agents/bc-1/runs/run-9"
    assert "shipped" in capsys.readouterr().out


def test_watch_times_out_without_a_terminal_status(transport, mod, capsys):
    t = transport(*[{"id": "run-9", "status": "RUNNING"}] * 50)
    rc = mod.main(["watch", "bc-1", "--run", "run-9", "--interval", "0", "--timeout", "0"])
    assert rc == 1
    assert "RUNNING" in capsys.readouterr().out
    assert t.requests, "watch must poll at least once before timing out"


def test_terminal_status_set_matches_the_api_enum(mod):
    assert {s for s in ("CREATING", "RUNNING") if mod.is_terminal(s)} == set()
    assert all(mod.is_terminal(s) for s in ("FINISHED", "ERROR", "CANCELLED", "EXPIRED"))


def test_repo_url_normalisation(mod):
    cases = {
        "acme/widgets": "https://github.com/acme/widgets",
        "https://github.com/acme/widgets": "https://github.com/acme/widgets",
        "https://github.com/acme/widgets.git": "https://github.com/acme/widgets",
        "git@github.com:acme/widgets.git": "https://github.com/acme/widgets",
    }
    for raw, expected in cases.items():
        assert mod.normalize_repo(raw) == expected
    for bad in ("gitlab.com/acme/widgets", "acme", "https://example.com/a/b"):
        with pytest.raises(mod.CursorCloudError):
            mod.normalize_repo(bad)
