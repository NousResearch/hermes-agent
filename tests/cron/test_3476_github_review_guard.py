"""Tests for ``cron.github_review_guard`` (issue #3476 guard module).

The module is **unwired** (no lane calls it yet — wiring is follow-up work
in the po-2026 container). These tests pin its contract so the future wiring
inherits a verified primitive:

1. **Attribution marker is appended exactly once**, with the right
   lane / host / cycle format, idempotent only for a *full* marker at the
   **end** of the body (a mid-body quote, a bare ``[Hermes]``, or a partial
   marker must not suppress the fresh one).
2. **The guard's subprocess is one ``sh -c`` call** — and that call is
   exercised for real against a stub ``gh`` that records its arguments
   (issue #3476 review: the mock-only tests never ran the shell).
3. **Paginated counting** uses ``--slurp``: two pages with zero matching
   review must NOT false-skip the POST.
4. **GET-reviews skipping** is honored: a same-``commit_id`` review means
   the POST is not issued.
5. **Fail-closed**: subprocess errors surface as :class:`PostResult` with
   ``posted=False, reason="error"`` and never raise into the cron tick.
6. ``GH_PATH`` override is honored, and ``elapsed_s`` is measured.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path

import pytest

# Make the worktree importable when the test runner is invoked from the repo
# root or from ``tests/``.  ``tests/cron/__init__.py`` is empty so we add the
# repo root explicitly.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cron import github_review_guard as guard  # noqa: E402

_HAS_SH = shutil.which("sh") is not None


# --- Attribution marker ----------------------------------------------------


def test_append_attribution_marker_basic_format():
    body = "Verdict: LGTM-side. Table impact recomputed."
    out = guard.append_attribution_marker(
        body,
        lane="hermes-pr-review",
        host="myia-po-2026",
        cycle=":07 06/09",
    )
    assert "[Hermes hermes-pr-review, cycle :07 06/09, host myia-po-2026]" in out
    # The marker is the LAST non-blank line in the body.
    assert out.rstrip().endswith(
        "[Hermes hermes-pr-review, cycle :07 06/09, host myia-po-2026]"
    )


def test_append_attribution_marker_idempotent_when_already_marked():
    body = "Body text\n\n[Hermes foo, cycle :07 06/09, host bar]\n"
    out = guard.append_attribution_marker(
        body,
        lane="other-lane",
        host="other-host",
        cycle=":08 06/09",
    )
    # No second marker line, original marker preserved verbatim.
    markers = re.findall(r"\[Hermes[^\]]*\]", out)
    assert len(markers) == 1
    assert markers[0] == "[Hermes foo, cycle :07 06/09, host bar]"


def test_append_attribution_marker_handles_empty_body():
    out = guard.append_attribution_marker(
        "", lane="L", host="H", cycle=":01 01/01"
    )
    assert "[Hermes L, cycle :01 01/01, host H]" in out


def test_append_attribution_marker_default_lane_and_host(monkeypatch):
    monkeypatch.setenv("HERMES_LANE", "hermes-inbox-poll")
    monkeypatch.setenv("HERMES_HOSTNAME", "myia-po-2026-container")
    out = guard.append_attribution_marker("hello")
    assert "[Hermes hermes-inbox-poll," in out
    assert "host myia-po-2026-container]" in out


def test_cycle_label_format_matches_datapoint16():
    """Datapoint #16 §2: tracked Hermes uses ``:XX DD/MM`` cycle markers."""
    fixed = _dt.datetime(2026, 9, 13, 21, 7, 32, tzinfo=_dt.timezone.utc)
    assert guard._cycle_label(now=fixed) == ":21 13/09"


def test_marker_not_suppressed_by_mid_body_quote():
    """A quoted ``[Hermes …]`` line in the middle of the body is NOT the
    fresh marker — idempotency must not be fooled by a citation/template."""
    body = (
        "> prior review said:\n"
        "> [Hermes quoted-lane, cycle :07 06/09, host quoted-host]\n"
        "\n"
        "New verdict text."
    )
    out = guard.append_attribution_marker(
        body, lane="L", host="H", cycle=":09 09/09"
    )
    assert out.rstrip().endswith("[Hermes L, cycle :09 09/09, host H]")
    assert "[Hermes quoted-lane, cycle :07 06/09, host quoted-host]" in out


def test_marker_not_suppressed_by_bare_hermes_tag():
    """``[Hermes]`` alone at the end lacks lane/cycle/host — not a marker."""
    body = "Verdict text\n\n[Hermes]"
    out = guard.append_attribution_marker(
        body, lane="L", host="H", cycle=":09 09/09"
    )
    assert out.rstrip().endswith("[Hermes L, cycle :09 09/09, host H]")


def test_marker_not_suppressed_when_fields_missing():
    """A partial marker (no host) at the end must not block the real one."""
    body = "Verdict text\n\n[Hermes some-lane, cycle :01 01/01]"
    out = guard.append_attribution_marker(
        body, lane="L", host="H", cycle=":09 09/09"
    )
    assert out.rstrip().endswith("[Hermes L, cycle :09 09/09, host H]")


# --- Single-subprocess invariant -------------------------------------------


def test_atomic_invoking_uses_one_sh_subprocess(monkeypatch):
    """The guard issues ONE ``sh -c`` call, not two."""
    monkeypatch.delenv("GH_PATH", raising=False)
    captured: list[tuple] = []

    class _FakeCompleted:
        def __init__(self, returncode=0, stdout="{}", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(cmd, **kwargs):
        captured.append((tuple(cmd), kwargs))
        return _FakeCompleted(stdout='{"id": 999}')

    monkeypatch.setattr(guard.subprocess, "run", _fake_run)
    monkeypatch.setattr(guard.shutil, "which", lambda _: "/usr/bin/gh")

    guard.post_review_if_unique(
        "jsboige", "CoursIA", 14863,
        "38287ac491cc8b9edcd792a6b4856e6e377dcba4",
        "verdict LGTM-side",
        lane="L", host="H", cycle=":01 01/01",
    )

    # Exactly one subprocess invocation — the single sh -c call.
    assert len(captured) == 1, (
        f"guard must invoke ONE subprocess; saw {len(captured)}"
    )
    cmd, kwargs = captured[0]
    assert cmd[0] == "sh"
    assert cmd[1] == "-c"
    # The shell snippet must contain BOTH the GET-reviews and the POST.
    snippet = cmd[2]
    assert "/reviews" in snippet
    assert "X POST" in snippet
    # The verified gh path is passed through as the snippet's $3.
    assert cmd[6] == "/usr/bin/gh"
    # timeout is forwarded so a hung gh doesn't block the cron tick.
    assert "timeout" in kwargs


# --- Subprocess dispatch: skip vs post -------------------------------------


def _run_with(monkeypatch, *, returncode, stdout, stderr=""):
    monkeypatch.delenv("GH_PATH", raising=False)
    monkeypatch.setattr(guard.shutil, "which", lambda _: "/usr/bin/gh")

    def _fake_run(cmd, **kwargs):
        class _C:
            pass

        c = _C()
        c.returncode = returncode
        c.stdout = stdout
        c.stderr = stderr
        return c

    monkeypatch.setattr(guard.subprocess, "run", _fake_run)


def test_skip_when_duplicate_review_exists(monkeypatch):
    _run_with(monkeypatch, returncode=0, stdout="SKIP_DUP:2\n")
    result = guard.post_review_if_unique(
        "jsboige", "CoursIA", 14863,
        "38287ac491cc8b9edcd792a6b4856e6e377dcba4",
        "verdict",
        lane="L", host="H", cycle=":01 01/01",
    )
    assert result.posted is False
    assert result.reason == "skipped_duplicate"
    assert result.commit_id == (
        "38287ac491cc8b9edcd792a6b4856e6e377dcba4"
    )
    # Attribution marker is still appended for downstream logs even on skip.
    assert "[Hermes L," in result.body_with_marker


def test_post_when_no_duplicate(monkeypatch):
    payload = {"id": 5124644669, "commit_id": "38287ac4"}
    _run_with(monkeypatch, returncode=0, stdout=json.dumps(payload))
    result = guard.post_review_if_unique(
        "jsboige", "CoursIA", 14863,
        "38287ac491cc8b9edcd792a6b4856e6e377dcba4",
        "verdict",
        lane="L", host="H", cycle=":01 01/01",
    )
    assert result.posted is True
    assert result.reason == "posted"
    assert result.review_id == 5124644669


def test_subprocess_nonzero_returns_error_not_raises(monkeypatch):
    _run_with(
        monkeypatch, returncode=1,
        stdout="",
        stderr="gh: API request failed (404)",
    )
    result = guard.post_review_if_unique(
        "jsboige", "CoursIA", 14863,
        "38287ac491cc8b9edcd792a6b4856e6e377dcba4",
        "verdict",
        lane="L", host="H", cycle=":01 01/01",
    )
    assert result.posted is False
    assert result.reason == "error"
    assert "404" in result.stderr


# --- Validation -----------------------------------------------------------


def test_invalid_event_raises_value_error():
    with pytest.raises(ValueError):
        guard.post_review_if_unique(
            "o", "r", 1, "deadbeef", "b", event="BANANA",
        )


def test_missing_gh_binary_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("GH_PATH", raising=False)
    monkeypatch.setattr(guard.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="GH_PATH"):
        guard.post_review_if_unique(
            "o", "r", 1, "deadbeef", "b",
        )


def test_gh_path_env_is_honored(monkeypatch):
    """``GH_PATH`` wins over ``PATH`` and must actually be resolved."""

    def _boom(_):
        raise AssertionError(
            "shutil.which must not be consulted when GH_PATH is set"
        )

    monkeypatch.setenv("GH_PATH", "/opt/fake/gh")
    monkeypatch.setattr(guard.shutil, "which", _boom)
    assert guard._require_gh() == "/opt/fake/gh"


# --- Real-shell execution against a stub gh --------------------------------

_STUB_GH = r"""#!/bin/sh
# Stub gh for the #3476 guard tests. Emulates gh's --paginate semantics:
#  - GET with --slurp  -> one aggregated jq result  ($GH_STUB_COUNT, def 0)
#  - GET without slurp -> per-page jq results, two empty pages -> "0\n0"
#  - POST              -> records the --input payload, emits a review JSON
# Every invocation's argv is appended to $GH_STUB_LOG (one arg per line,
# "---CALL---" separators).
printf '%s\n' "---CALL---" >> "$GH_STUB_LOG"
for a in "$@"; do printf '%s\n' "$a" >> "$GH_STUB_LOG"; done
case " $* " in
  *" -X POST "*)
    if [ -n "${GH_STUB_SLEEP:-}" ]; then sleep "$GH_STUB_SLEEP"; fi
    if [ -n "${GH_STUB_POST_FAIL:-}" ]; then
      echo "stub POST failure" >&2
      exit 1
    fi
    prev=""
    for a in "$@"; do
      if [ "$prev" = "--input" ] && [ -n "${GH_STUB_PAYLOAD_COPY:-}" ]; then
        cp "$a" "$GH_STUB_PAYLOAD_COPY"
      fi
      prev="$a"
    done
    echo '{"id": 424242}'
    exit 0
    ;;
  *" --slurp "*)
    echo "${GH_STUB_COUNT:-0}"
    exit 0
    ;;
  *)
    printf '0\n0\n'
    exit 0
    ;;
esac
"""


@pytest.fixture()
def stub_gh(tmp_path, monkeypatch):
    """Install a recording ``gh`` stub and point ``GH_PATH`` at it."""
    stub = tmp_path / "gh-stub.sh"
    log = tmp_path / "gh-calls.log"
    payload_copy = tmp_path / "posted-payload.json"
    stub.write_text(_STUB_GH, encoding="utf-8", newline="\n")
    log.write_text("", encoding="utf-8")
    os.chmod(stub, os.stat(stub).st_mode | stat.S_IEXEC)
    monkeypatch.setenv("GH_STUB_LOG", str(log))
    monkeypatch.setenv("GH_STUB_PAYLOAD_COPY", str(payload_copy))
    monkeypatch.setenv("GH_PATH", str(stub).replace("\\", "/"))
    monkeypatch.delenv("GH_STUB_COUNT", raising=False)
    monkeypatch.delenv("GH_STUB_SLEEP", raising=False)
    monkeypatch.delenv("GH_STUB_POST_FAIL", raising=False)

    class _Stub:
        def __init__(self):
            self.log = log
            self.payload_copy = payload_copy

        def calls(self):
            """List of argv lists, one per recorded invocation."""
            out: list[list[str]] = []
            for block in self.log.read_text(encoding="utf-8").split(
                "---CALL---\n"
            ):
                if block.strip():
                    out.append(block.splitlines())
            return out

    return _Stub()


@pytest.mark.skipif(not _HAS_SH, reason="sh not on PATH")
def test_real_snippet_posts_with_stub_gh(stub_gh, monkeypatch):
    """End-to-end through the real ``sh -c`` snippet (issue #3476 review
    blocker 1): the POST must actually be reachable, must use ``--input``
    (``--input-file`` does not exist in ``gh``), and must carry the payload
    from the tempfile."""
    result = guard.post_review_if_unique(
        "jsboige", "CoursIA", 14863,
        "38287ac491cc8b9edcd792a6b4856e6e377dcba4",
        "verdict LGTM-side",
        lane="L", host="H", cycle=":01 01/01",
    )
    assert result.posted is True, f"reason={result.reason!r} stderr={result.stderr!r}"
    assert result.reason == "posted"
    assert result.review_id == 424242

    calls = stub_gh.calls()
    assert len(calls) == 2, f"expected GET + POST, saw {calls}"
    get_args, post_args = calls
    # GET: reviews endpoint, paginated AND slurped (single aggregated count).
    assert "/repos/jsboige/CoursIA/pulls/14863/reviews" in get_args
    assert "--paginate" in get_args
    assert "--slurp" in get_args
    # POST: reviews endpoint via --input (the only real gh flag).
    assert "/repos/jsboige/CoursIA/pulls/14863/reviews" in post_args
    assert "--input" in post_args
    assert "--input-file" not in post_args
    # The payload tempfile carried the body WITH the attribution marker.
    payload = json.loads(stub_gh.payload_copy.read_text(encoding="utf-8"))
    assert payload["body"].rstrip().endswith(
        "[Hermes L, cycle :01 01/01, host H]"
    )
    assert payload["commit_id"] == (
        "38287ac491cc8b9edcd792a6b4856e6e377dcba4"
    )


@pytest.mark.skipif(not _HAS_SH, reason="sh not on PATH")
def test_real_snippet_two_pages_zero_match_still_posts(stub_gh, monkeypatch):
    """Issue #3476 review blocker 2: two pages with zero matching review
    must NOT false-skip. The stub emits the per-page result ``0\\n0`` on any
    GET without ``--slurp`` — the exact shape that used to make
    ``[ "$N" != "0" ]`` true — so this test fails if ``--slurp`` is dropped
    from the snippet."""
    monkeypatch.setenv("GH_STUB_COUNT", "0")
    result = guard.post_review_if_unique(
        "jsboige", "CoursIA", 14863,
        "38287ac491cc8b9edcd792a6b4856e6e377dcba4",
        "verdict",
        lane="L", host="H", cycle=":01 01/01",
    )
    assert result.posted is True, (
        f"false SKIP on two empty pages: reason={result.reason!r} "
        f"stdout={result.stdout!r}"
    )
    assert result.reason == "posted"
    # And the aggregated count really flowed through --slurp.
    assert "--slurp" in stub_gh.calls()[0]


@pytest.mark.skipif(not _HAS_SH, reason="sh not on PATH")
def test_real_snippet_skips_when_aggregated_count_nonzero(stub_gh, monkeypatch):
    monkeypatch.setenv("GH_STUB_COUNT", "2")
    result = guard.post_review_if_unique(
        "jsboige", "CoursIA", 14863,
        "38287ac491cc8b9edcd792a6b4856e6e377dcba4",
        "verdict",
        lane="L", host="H", cycle=":01 01/01",
    )
    assert result.posted is False
    assert result.reason == "skipped_duplicate"
    assert result.stdout.strip() == "SKIP_DUP:2"
    # No POST call was issued at all.
    assert len(stub_gh.calls()) == 1


@pytest.mark.skipif(not _HAS_SH, reason="sh not on PATH")
def test_real_snippet_elapsed_s_is_measured(stub_gh, monkeypatch):
    """``elapsed_s`` must be measured, not a hardcoded ``0.0``."""
    monkeypatch.setenv("GH_STUB_SLEEP", "0.3")
    result = guard.post_review_if_unique(
        "jsboige", "CoursIA", 14863,
        "38287ac491cc8b9edcd792a6b4856e6e377dcba4",
        "verdict",
        lane="L", host="H", cycle=":01 01/01",
    )
    assert result.posted is True
    assert result.elapsed_s >= 0.2, f"elapsed_s={result.elapsed_s!r}"


@pytest.mark.skipif(not _HAS_SH, reason="sh not on PATH")
def test_real_snippet_post_failure_is_fail_closed(stub_gh, monkeypatch):
    monkeypatch.setenv("GH_STUB_POST_FAIL", "1")
    result = guard.post_review_if_unique(
        "jsboige", "CoursIA", 14863,
        "38287ac491cc8b9edcd792a6b4856e6e377dcba4",
        "verdict",
        lane="L", host="H", cycle=":01 01/01",
    )
    assert result.posted is False
    assert result.reason == "error"
    assert "stub POST failure" in result.stderr


# --- Convenience wrapper --------------------------------------------------


def test_post_review_resolves_head_and_delegates(monkeypatch):
    """``post_review`` looks up headRefOid then calls the guard."""
    monkeypatch.delenv("GH_PATH", raising=False)
    calls: list[tuple] = []

    def _fake_run(cmd, **kwargs):
        calls.append((tuple(cmd), kwargs))

        class _C:
            returncode = 0

        c = _C()
        # First call: gh pr view -> head SHA.
        # Second call: sh -c guarded snippet -> POST.
        if cmd and cmd[0] == "gh" and "pr" in cmd and "view" in cmd:
            c.stdout = "38287ac491cc8b9edcd792a6b4856e6e377dcba4\n"
        else:
            c.stdout = json.dumps({"id": 1001})
        c.stderr = ""
        return c

    monkeypatch.setattr(guard.subprocess, "run", _fake_run)
    monkeypatch.setattr(guard.shutil, "which", lambda _: "/usr/bin/gh")

    result = guard.post_review(
        "jsboige", "CoursIA", 14863, "verdict",
        lane="L", host="H", cycle=":01 01/01",
    )
    assert result.posted is True
    assert result.review_id == 1001
    # Two subprocess calls: head lookup + guarded POST.
    assert len(calls) == 2
    # First call resolves the PR head SHA via `gh pr view`.
    assert calls[0][0][0].endswith("gh")
    assert list(calls[0][0][1:4]) == ["pr", "view", "14863"]
    # Second call is the guarded sh -c snippet.
    assert calls[1][0][0] == "sh"
    assert calls[1][0][1] == "-c"


def test_post_review_propagates_head_lookup_failure(monkeypatch):
    monkeypatch.delenv("GH_PATH", raising=False)

    def _fake_run(cmd, **kwargs):
        class _C:
            returncode = 1
            stdout = ""
            stderr = "gh: not found"
        return _C()

    monkeypatch.setattr(guard.subprocess, "run", _fake_run)
    monkeypatch.setattr(guard.shutil, "which", lambda _: "/usr/bin/gh")

    result = guard.post_review(
        "jsboige", "CoursIA", 14863, "verdict",
        lane="L", host="H", cycle=":01 01/01",
    )
    assert result.posted is False
    assert result.reason == "error"
    assert "not found" in result.stderr
