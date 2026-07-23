"""Tests for optional-skills/productivity/here-now.

Covers SKILL.md frontmatter standards and publish.sh request-body behavior:
the claim-token handling (anonymous and API-key updates), the removal of
the deprecated --forkable / fork-meta.json surface (forking was removed from
the here.now product; the API no longer accepts a `forkable` field), and the
--workspace account selector (sent as an x-herenow-account header on the
create and finalize calls; requires an API key; incompatible with
--from-drive).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "productivity"
    / "here-now"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
PUBLISH_SH = SKILL_DIR / "scripts" / "publish.sh"

requires_shell_deps = pytest.mark.skipif(
    shutil.which("bash") is None
    or shutil.which("jq") is None
    or shutil.which("file") is None,
    reason="requires bash, jq, and file",
)


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------


def read_frontmatter() -> str:
    text = SKILL_MD.read_text()
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter"
    return match.group(1)


def test_description_is_single_line_scalar():
    frontmatter = read_frontmatter()
    match = re.search(r"^description: (.+)$", frontmatter, re.MULTILINE)
    assert match, "description must be a single-line scalar"
    assert match.group(1).strip() not in {">", "|", ">-", "|-"}, (
        "description must not be a block scalar"
    )


def test_description_meets_hardline_standard():
    frontmatter = read_frontmatter()
    match = re.search(r"^description: (.+)$", frontmatter, re.MULTILINE)
    assert match
    description = match.group(1).strip()
    assert len(description) <= 60, f"description is {len(description)} chars"
    assert description.endswith("."), "description must end with a period"
    assert description.count(".") == 1, "description must be one sentence"
    assert "here.now" not in description.lower(), (
        "description must not repeat the skill name"
    )


def test_version_is_semver():
    frontmatter = read_frontmatter()
    match = re.search(r"^version: (.+)$", frontmatter, re.MULTILINE)
    assert match, "version missing from frontmatter"
    assert re.fullmatch(r"\d+\.\d+\.\d+", match.group(1).strip())


def test_forkable_flag_removed_from_docs():
    text = SKILL_MD.read_text()
    assert "--forkable" not in text
    assert "forkable" not in text.lower()


# ---------------------------------------------------------------------------
# publish.sh request-body behavior
# ---------------------------------------------------------------------------

CREATE_RESPONSE = {
    "slug": "test-slug-a1b2",
    "siteUrl": "https://test-slug-a1b2.here.now/",
    "claimToken": "server-issued-token",
    "claimUrl": "https://here.now/claim?slug=test-slug-a1b2&token=server-issued-token",
    "expiresAt": "2099-01-01T00:00:00.000Z",
    "upload": {
        "versionId": "ver-1",
        "finalizeUrl": "https://here.now/api/v1/publish/test-slug-a1b2/finalize",
        "uploads": [
            {
                "path": "index.html",
                "url": "https://uploads.test/index.html",
                "headers": {"Content-Type": "text/html; charset=utf-8"},
            }
        ],
        "skipped": [],
    },
}

CURL_STUB = """#!/usr/bin/env bash
# Records each invocation's args and -d body, and returns canned responses.
set -euo pipefail
log_dir="${CURL_STUB_DIR:?}"
n=$(find "$log_dir" -name 'call-*.args' | wc -l | tr -d ' ')
i=$((n + 1))
printf '%s\\n' "$@" > "$log_dir/call-$i.args"

body=""
prev=""
url=""
wants_http_code=0
for a in "$@"; do
  if [[ "$prev" == "-d" ]]; then body="$a"; fi
  if [[ "$a" == "-w" ]]; then wants_http_code=1; fi
  case "$a" in http*) url="$a" ;; esac
  prev="$a"
done
printf '%s' "$body" > "$log_dir/call-$i.body"

# File-upload PUTs use `-o /dev/null -w "%{http_code}"`; answer with a 200.
if [[ "$wants_http_code" -eq 1 ]]; then
  printf '200'
  exit 0
fi

if [[ "$url" == */finalize ]]; then
  echo '{}'
elif [[ "$url" == */api/v1/publish* ]]; then
  cat "$log_dir/create-response.json"
else
  echo '{}'
fi
"""


class PublishHarness:
    """Runs the real publish.sh with a stubbed curl and isolated HOME/cwd."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.bin_dir = tmp_path / "bin"
        self.log_dir = tmp_path / "curl-log"
        self.home = tmp_path / "home"
        self.workdir = tmp_path / "work"
        for d in (self.bin_dir, self.log_dir, self.home, self.workdir):
            d.mkdir()

        curl = self.bin_dir / "curl"
        curl.write_text(CURL_STUB)
        curl.chmod(0o755)
        (self.log_dir / "create-response.json").write_text(json.dumps(CREATE_RESPONSE))

        site = self.workdir / "site"
        site.mkdir()
        (site / "index.html").write_text("<h1>hello</h1>\n")
        self.site = site

    def write_state(self, slug: str, claim_token: str) -> None:
        state_dir = self.workdir / ".herenow"
        state_dir.mkdir(exist_ok=True)
        (state_dir / "state.json").write_text(
            json.dumps({"publishes": {slug: {"claimToken": claim_token}}})
        )

    def run(self, *args: str, api_key: str | None = None):
        env = {
            **os.environ,
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "HOME": str(self.home),
            "CURL_STUB_DIR": str(self.log_dir),
        }
        env.pop("HERENOW_API_KEY", None)
        if api_key is not None:
            env["HERENOW_API_KEY"] = api_key
        return subprocess.run(
            ["bash", str(PUBLISH_SH), *args],
            cwd=self.workdir,
            env=env,
            capture_output=True,
            text=True,
        )

    def request(self, i: int = 1) -> tuple[list[str], dict]:
        args = (self.log_dir / f"call-{i}.args").read_text().splitlines()
        body_text = (self.log_dir / f"call-{i}.body").read_text()
        return args, json.loads(body_text) if body_text else {}


@pytest.fixture()
def harness(tmp_path: Path) -> PublishHarness:
    return PublishHarness(tmp_path)


@requires_shell_deps
def test_forkable_flag_rejected(harness: PublishHarness):
    result = harness.run(str(harness.site), "--forkable")
    assert result.returncode != 0
    assert "unknown option: --forkable" in result.stderr


@requires_shell_deps
def test_create_body_has_no_forkable_or_claim_token(harness: PublishHarness):
    result = harness.run(str(harness.site))
    assert result.returncode == 0, result.stderr
    _, body = harness.request(1)
    assert "forkable" not in body
    assert "claimToken" not in body
    assert [f["path"] for f in body["files"]] == ["index.html"]


@requires_shell_deps
def test_fork_meta_excluded_from_manifest(harness: PublishHarness):
    meta_dir = harness.site / ".herenow"
    meta_dir.mkdir()
    (meta_dir / "fork-meta.json").write_text('{"forkable": true}')
    result = harness.run(str(harness.site))
    assert result.returncode == 0, result.stderr
    _, body = harness.request(1)
    assert [f["path"] for f in body["files"]] == ["index.html"]
    assert "forkable" not in body


@requires_shell_deps
def test_anonymous_update_autoloads_claim_token(harness: PublishHarness):
    harness.write_state("test-slug-a1b2", "state-token")
    result = harness.run(str(harness.site), "--slug", "test-slug-a1b2")
    assert result.returncode == 0, result.stderr
    args, body = harness.request(1)
    assert body["claimToken"] == "state-token"
    assert not any(a.startswith("authorization:") for a in args)


@requires_shell_deps
def test_api_key_update_still_sends_claim_token(harness: PublishHarness):
    # The server only consults claimToken for anonymous sites, so sending it
    # alongside an API key is harmless; auto-loading it regardless of auth
    # mode keeps slug updates working when a site is still unclaimed.
    harness.write_state("test-slug-a1b2", "state-token")
    result = harness.run(
        str(harness.site), "--slug", "test-slug-a1b2", api_key="hn_test_key"
    )
    assert result.returncode == 0, result.stderr
    args, body = harness.request(1)
    assert body["claimToken"] == "state-token"
    assert "authorization: Bearer hn_test_key" in args


@requires_shell_deps
def test_workspace_requires_api_key(harness: PublishHarness):
    result = harness.run(str(harness.site), "--workspace", "acme")
    assert result.returncode != 0
    assert "--workspace requires an account API key" in result.stderr


@requires_shell_deps
def test_workspace_rejects_from_drive(harness: PublishHarness):
    result = harness.run(
        "--workspace",
        "acme",
        "--from-drive",
        "drv_0123456789abcdef",
        api_key="hn_test_key",
    )
    assert result.returncode != 0
    assert "--workspace cannot be combined with --from-drive" in result.stderr


@requires_shell_deps
def test_workspace_sends_account_selector_on_create_and_finalize(
    harness: PublishHarness,
):
    result = harness.run(
        str(harness.site), "--workspace", "acme", api_key="hn_test_key"
    )
    assert result.returncode == 0, result.stderr

    create_args, body = harness.request(1)
    assert "x-herenow-account: acme" in create_args
    assert "authorization: Bearer hn_test_key" in create_args
    assert "forkable" not in body

    call_count = len(list(harness.log_dir.glob("call-*.args")))
    finalize_args, _ = harness.request(call_count)
    assert any(a.endswith("/finalize") for a in finalize_args)
    assert "x-herenow-account: acme" in finalize_args


@requires_shell_deps
def test_explicit_claim_token_overrides_state(harness: PublishHarness):
    harness.write_state("test-slug-a1b2", "state-token")
    result = harness.run(
        str(harness.site),
        "--slug",
        "test-slug-a1b2",
        "--claim-token",
        "explicit-token",
    )
    assert result.returncode == 0, result.stderr
    _, body = harness.request(1)
    assert body["claimToken"] == "explicit-token"
