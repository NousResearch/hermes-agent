"""Regression tests for the esbuild / ignore-scripts build-failure diagnosis (#53082)."""

import io
import contextlib

from hermes_cli.main_desktop import _diagnose_esbuild_ignore_scripts


def _hintprinted(output: str | None) -> bool:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _diagnose_esbuild_ignore_scripts(output)
    return "esbuild" in buf.getvalue()


def test_diagnose_triggers_on_missing_esbuild_binary() -> None:
    output = (
        "Error: The package \"@esbuild/darwin-arm64\" could not be found, "
        "and is needed by esbuild."
    )
    assert _hintprinted(output)


def test_diagnose_triggers_on_ignore_scripts_mention() -> None:
    assert _hintprinted("npm error code ERESOLVE\nnpm warn config ignore-scripts=true")


def test_diagnose_silent_on_unrelated_failure() -> None:
    assert not _hintprinted("vite build failed: unexpected token in index.html")


def test_diagnose_silent_on_none_output() -> None:
    assert not _hintprinted(None)
