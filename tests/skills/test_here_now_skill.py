"""Tests for optional-skills/productivity/here-now.

Covers SKILL.md frontmatter standards and publish.sh behavior:
the claim-token handling (anonymous and API-key updates), the removal of
the deprecated --forkable / fork-meta.json surface (forking was removed from
the here.now product; the API no longer accepts a `forkable` field), and the
--workspace account selector (sent as an x-herenow-account header on the
create and finalize calls; requires an API key; incompatible with
--from-drive). Since 1.35.0 also: the base-URL allowlist (https://api.here.now,
the same API through Cloudflare, takes credentials without the override flag),
the finalize retry (an HTML 524 or a 409 finalize_in_flight repeats the
idempotent finalize call instead of being misread as success), and the
--claim-token value being saved to the state file.
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

# Finalize is called with `-w '\\n%{http_code}'`: body, newline, status.
# FINALIZE_SEQUENCE (optional, in the log dir) scripts one answer per line
# for successive finalize calls, "<status> <json|html>"; after the lines
# run out, or when the file is absent, finalize succeeds.
if [[ "$url" == */finalize ]]; then
  status=200; kind=json
  seq_file="$log_dir/finalize-sequence"
  if [[ -s "$seq_file" ]]; then
    read -r status kind < "$seq_file"
    tail -n +2 "$seq_file" > "$seq_file.tmp" && mv "$seq_file.tmp" "$seq_file"
  fi
  case "$status $kind" in
    "200 json") printf '%s' '{"success":true,"slug":"test-slug-a1b2","siteUrl":"https://test-slug-a1b2.here.now/","currentVersionId":"ver-1"}' ;;
    "409 json") printf '%s' '{"error":"A finalize is already in flight for this version.","code":"finalize_in_flight","retry_after":0}' ;;
    "524 html") printf '%s' '<!DOCTYPE html><html><body>error code: 524</body></html>' ;;
    *) printf '%s' '{}' ;;
  esac
  if [[ "$wants_http_code" -eq 1 ]]; then printf '\\n%s' "$status"; fi
  exit 0
fi

# File-upload PUTs use `-o /dev/null -w "%{http_code}"`; answer with a 200.
if [[ "$wants_http_code" -eq 1 ]]; then
  printf '200'
  exit 0
fi

if [[ "$url" == */api/v1/publish* ]]; then
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

    def read_state(self) -> dict:
        return json.loads((self.workdir / ".herenow" / "state.json").read_text())

    def script_finalize(self, *answers: str) -> None:
        """Queue finalize answers, e.g. "524 html", "409 json"; then success."""
        (self.log_dir / "finalize-sequence").write_text("".join(a + "\n" for a in answers))

    def finalize_calls(self) -> int:
        return sum(
            1
            for f in self.log_dir.glob("call-*.args")
            if any(line.endswith("/finalize") for line in f.read_text().splitlines())
        )

    def run(self, *args: str, api_key: str | None = None, env_extra: dict | None = None):
        env = {
            **os.environ,
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "HOME": str(self.home),
            "CURL_STUB_DIR": str(self.log_dir),
        }
        env.pop("HERENOW_API_KEY", None)
        env.pop("HERENOW_FINALIZE_WAIT_SECONDS", None)
        if api_key is not None:
            env["HERENOW_API_KEY"] = api_key
        if env_extra:
            env.update(env_extra)
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
def test_state_json_excluded_from_manifest(harness: PublishHarness):
    # publish.sh writes its own .herenow/state.json next to published
    # directories; it is local cache (and carries claim tokens for anonymous
    # sites), so collection must never upload it as site content.
    meta_dir = harness.site / ".herenow"
    meta_dir.mkdir()
    (meta_dir / "state.json").write_text('{"publishes": {}}')
    result = harness.run(str(harness.site))
    assert result.returncode == 0, result.stderr
    _, body = harness.request(1)
    assert [f["path"] for f in body["files"]] == ["index.html"]


@requires_shell_deps
def test_zero_upload_republish_still_finalizes(harness: PublishHarness):
    # When every file is unchanged the server returns no upload URLs and
    # lists the files in skipped[]. The script must proceed to finalize:
    # a BSD-seq loop bug (`seq 0 -1` counts down on macOS) used to crash
    # between create and finalize here, stranding the site in pending.
    no_upload = dict(CREATE_RESPONSE)
    no_upload["upload"] = {
        **CREATE_RESPONSE["upload"],
        "uploads": [],
        "skipped": [{"path": "index.html"}],
    }
    (harness.log_dir / "create-response.json").write_text(json.dumps(no_upload))
    result = harness.run(
        str(harness.site), "--slug", "test-slug-a1b2", api_key="hn_test_key"
    )
    assert result.returncode == 0, result.stderr
    finalize_args, finalize_body = harness.request(2)
    assert any(a.endswith("/finalize") for a in finalize_args)
    assert finalize_body["versionId"] == "ver-1"


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


@requires_shell_deps
def test_claim_token_flag_is_saved_to_state(harness: PublishHarness):
    # 1.35.0: a token passed by flag (a Site the user created at here.new and
    # handed over) is saved, so the next update from this directory needs no
    # flag. An update response carries no claimToken of its own (only a
    # create does), so the fixture drops it here.
    update_response = {k: v for k, v in CREATE_RESPONSE.items() if k not in {"claimToken", "claimUrl", "expiresAt"}}
    (harness.log_dir / "create-response.json").write_text(json.dumps(update_response))
    result = harness.run(
        str(harness.site), "--slug", "test-slug-a1b2", "--claim-token", "explicit-token"
    )
    assert result.returncode == 0, result.stderr
    assert harness.read_state()["publishes"]["test-slug-a1b2"]["claimToken"] == "explicit-token"


# ---------------------------------------------------------------------------
# Base URL allowlist (1.35.0): https://api.here.now is the same API through
# Cloudflare for networks that cannot reach here.now; credentials may go
# there without the override flag, and nowhere else.
# ---------------------------------------------------------------------------


@requires_shell_deps
def test_api_host_accepts_credentials_without_override(harness: PublishHarness):
    result = harness.run(
        str(harness.site), "--base-url", "https://api.here.now", api_key="hn_test_key"
    )
    assert result.returncode == 0, result.stderr
    args, _ = harness.request(1)
    assert "https://api.here.now/api/v1/publish" in args
    assert "authorization: Bearer hn_test_key" in args


@requires_shell_deps
@pytest.mark.parametrize(
    "base_url",
    ["https://evil.example", "https://api.here.now:8443", "https://api.here.now.evil.example", "http://api.here.now"],
)
def test_other_base_urls_refuse_credentials(harness: PublishHarness, base_url: str):
    result = harness.run(str(harness.site), "--base-url", base_url, api_key="hn_test_key")
    assert result.returncode != 0
    assert "refusing to send API key" in result.stderr
    assert not list(harness.log_dir.glob("call-*.args")), "no request may leave before the guard"


# ---------------------------------------------------------------------------
# Finalize retry (1.35.0): the server's finalize is idempotent, so when no
# real answer arrives (a proxy's 524 with an HTML body, a dropped connection)
# or it reports the finalize still running (409 finalize_in_flight), the
# script repeats the same call instead of misreading the reply.
# ---------------------------------------------------------------------------


@requires_shell_deps
def test_finalize_repeats_while_in_flight_then_reads_the_real_answer(harness: PublishHarness):
    harness.script_finalize("409 json", "409 json")
    result = harness.run(str(harness.site))
    assert result.returncode == 0, result.stderr
    assert harness.finalize_calls() == 3
    assert result.stderr.count("still running on the server") == 2
    assert "publish_result.live_version_id=ver-1" in result.stderr


@requires_shell_deps
def test_finalize_gateway_timeout_is_not_success(harness: PublishHarness):
    # Before 1.35.0 an HTML 524 parsed as a successful finalize with an empty
    # live version. With a wait budget too small for a retry, the script must
    # stop with the timeout marker and no site URL on stdout.
    harness.script_finalize("524 html", "524 html", "524 html")
    result = harness.run(
        str(harness.site), env_extra={"HERENOW_FINALIZE_WAIT_SECONDS": "1"}
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert "no answer (HTTP 524" in result.stderr
    assert "publish_result.finalize=timeout" in result.stderr
    assert "/api/v1/publish/test-slug-a1b2/finalize" in result.stderr
    assert harness.finalize_calls() == 1
