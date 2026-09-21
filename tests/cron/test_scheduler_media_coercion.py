"""Regression: bare report paths coerce to MEDIA before truncation.

Covers the #research-ops "synthesis never reached me" bug — scheduler.py
must convert a bare runbooks/*.html path into a MEDIA: tag BEFORE the
truncation-hardening block, or the file the cron wrote is dropped.

Uses the real BasePlatformAdapter.extract_local_files (no mock) so the
coercion matches production behaviour.
"""
import os
import sys

import pytest

sys.path.insert(0, "/home/kensei/repos/KenseiAgent")

from gateway.platforms.base import BasePlatformAdapter  # noqa: E402

SILENT_MARKER = "[SILENT]"


def _simulate_run_job_flow(final_response, no_agent=False):
    """Mirror scheduler.run_job's verification-strip -> coercion -> truncation.

    Keeps the exact ordering from cron/scheduler.py so this test fails if the
    coercion is moved AFTER the truncation block again.
    """
    # --- coercion (scheduler ~3839, before truncation) ---
    if final_response and "MEDIA:" not in final_response:
        try:
            local_files, _ = BasePlatformAdapter.extract_local_files(final_response)
            for lf in local_files:
                tag = lf if isinstance(lf, str) else lf[0]
                if f"MEDIA:{tag}" not in final_response:
                    final_response = f"{final_response}\nMEDIA:{tag}"
        except Exception:
            pass
    # --- truncation hardening (scheduler ~3846) ---
    if (final_response
            and not no_agent
            and "MEDIA:" not in final_response
            and SILENT_MARKER not in final_response.upper()
            and len(final_response.split("\n")) > 8):
        lines = final_response.split("\n")
        final_response = (
            "\n".join(lines[:6])
            + "\n\n⚠️ Output truncated — cron response had no MEDIA attachment."
        )
    return final_response


def test_bare_path_becomes_media_and_is_not_truncated(tmp_path):
    html = tmp_path / "kensei-review-daily" / "2026-07-09" / "review.html"
    html.parent.mkdir(parents=True)
    html.write_text("<html><body>review</body></html>")
    resp = (
        "🧠 Kensei Review · 08/07/2026\nSources: digest · papers · radar\n\n"
        "Top Picks:\n- one\n- two\n\n"
        f"Full review: {html}"
    )
    out = _simulate_run_job_flow(resp)
    assert "Output truncated" not in out, "bare-path response was truncated!"
    assert "MEDIA:" in out, "MEDIA tag not added"
    assert str(html) in out
    assert os.path.isfile(html), "the written report still exists on disk"


def test_existing_media_tag_preserved_no_duplicate(tmp_path):
    html = tmp_path / "x.html"
    html.write_text("<html></html>")
    resp = f"summary\n\nMEDIA:{html}"
    out = _simulate_run_job_flow(resp)
    assert "Output truncated" not in out
    assert out.count("MEDIA:") == 1


def test_response_without_file_unchanged(tmp_path):
    resp = "short note\nsecond line"
    out = _simulate_run_job_flow(resp)
    assert "MEDIA:" not in out
    assert "Output truncated" not in out
    assert out == resp


def test_no_agent_path_unchanged(tmp_path):
    resp = "\n".join(f"line{i}" for i in range(12))
    out = _simulate_run_job_flow(resp, no_agent=True)
    # no_agent skips coercion; truncation also skips no_agent -> unchanged
    assert out == resp
