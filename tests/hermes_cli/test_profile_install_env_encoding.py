"""Regression: the distribution-install preview must read .env as UTF-8.

`_render_distribution_plan` inspects the target profile's `.env` to decide
whether a required env var is already set, so it doesn't nag the user. It
used `Path.read_text()` with no encoding, which:

  1. defaults to the system locale (cp1251/GBK on Windows) and raises
     `UnicodeDecodeError` on any non-ASCII byte — and the surrounding
     `except OSError` does NOT catch that (it's a `ValueError`), so a
     mis-encoded .env aborts the whole install preview; and
  2. even on a UTF-8 locale, a Notepad-added BOM prefixes the first key,
     so the very first env var is mis-detected as "needs setting".

The fix reads `utf-8-sig` and also catches `UnicodeDecodeError`.
"""

from types import SimpleNamespace

import pytest

from hermes_cli.main import _render_distribution_plan


def _make_plan(target_dir, env_requires):
    manifest = SimpleNamespace(
        name="demo",
        version="1.0.0",
        description="",
        author="",
        hermes_requires="",
        env_requires=env_requires,
    )
    return SimpleNamespace(
        manifest=manifest,
        provenance="local",
        target_dir=target_dir,
        existing=False,
        has_cron=False,
    )


def test_bom_prefixed_env_first_key_is_detected_as_set(tmp_path, monkeypatch, capsys):
    """A required key on the first line of a BOM-prefixed .env reads as set."""
    monkeypatch.delenv("FOO_TOKEN", raising=False)
    profile = tmp_path / "profile"
    profile.mkdir()
    # utf-8-sig write == UTF-8 with a leading BOM, exactly what Notepad emits.
    (profile / ".env").write_text("FOO_TOKEN=abc\nBAR=1\n", encoding="utf-8-sig")

    plan = _make_plan(profile, [SimpleNamespace(name="FOO_TOKEN", required=True, description="")])
    _render_distribution_plan(plan)

    out = capsys.readouterr().out
    foo_line = next(ln for ln in out.splitlines() if "FOO_TOKEN" in ln)
    assert "✓ set" in foo_line, f"BOM must not hide the first key: {foo_line!r}"


def test_non_utf8_env_does_not_abort_the_preview(tmp_path, monkeypatch, capsys):
    """A .env with invalid UTF-8 bytes must not crash the install preview."""
    monkeypatch.delenv("FOO_TOKEN", raising=False)
    profile = tmp_path / "profile"
    profile.mkdir()
    # \xff is an invalid UTF-8 start byte -> UnicodeDecodeError on read.
    (profile / ".env").write_bytes(b"FOO_TOKEN=\xff\xfe bad\n")

    plan = _make_plan(profile, [SimpleNamespace(name="FOO_TOKEN", required=True, description="")])

    # Must return normally (old `except OSError` let UnicodeDecodeError escape).
    _render_distribution_plan(plan)

    out = capsys.readouterr().out
    assert any("FOO_TOKEN" in ln for ln in out.splitlines())
