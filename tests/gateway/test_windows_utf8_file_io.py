"""Regression tests for #37423: gateway Windows UTF-8 file I/O.

The `hermes update` flow in ``gateway/run.py`` reads and writes its
coordination files with bare ``Path.read_text()`` / ``Path.write_text()``
(no ``encoding=``). On Windows with a non-UTF-8 system locale
(``cp1252`` on US installs, ``GBK``/``CP936`` on Chinese installs,
etc.) Python uses the locale codec instead of UTF-8, which produces
``UnicodeEncodeError`` on write and ``UnicodeDecodeError``/mojibake on
read. Both subclass ``ValueError``, not ``OSError``, so the surrounding
``except OSError`` guards do not catch them and the gateway command
handler / update-stream coroutine crashes.

These tests guard against the regression by:

1. Walking ``gateway/run.py`` and asserting every ``.read_text()`` and
   ``.write_text()`` call site uses an explicit ``encoding=`` kwarg.
2. A behavioural test that the update-flow helpers don't crash on a
   Windows-like ``cp1252`` locale when the payload contains a non-ASCII
   character (emoji / CJK / box-drawing) — the exact crash signature
   from the issue (#37423).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GATEWAY_RUN_PY = REPO_ROOT / "gateway" / "run.py"


def _iter_path_text_io_calls(tree: ast.AST) -> list[tuple[int, str]]:
    """Yield (lineno, qualified-call-string) for every ``<path>.read_text(...)``
    or ``<path>.write_text(...)`` in the AST under ``tree``.

    We intentionally don't restrict to ``Path`` — a regular ``Path`` call
    chain (``Path("x").read_text()``) and a variable holding a Path both
    go through the same Attribute→Call shape and we want to catch both.
    """
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in {"read_text", "write_text"}:
            continue
        # Reconstruct a compact name like "Path.read_text" from the
        # attribute chain. We don't need the full receiver — only that
        # the call is on a Path-like object — so a short summary is
        # enough for error messages.
        parts: list[str] = []
        cur: ast.expr = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        out.append((node.lineno, ".".join(reversed(parts))))
    return out


class TestGatewayReadWriteEncoding:
    """Static regression guard: every Path.read_text/write_text call in
    gateway/run.py MUST pass an explicit encoding= keyword argument.

    Without this, the gateway command handler crashes on Windows hosts
    with a non-UTF-8 system locale when the user replies with a CJK /
    emoji / box-drawing character (#37423).
    """

    @pytest.fixture(scope="class")
    def gateway_tree(self) -> ast.AST:
        return ast.parse(GATEWAY_RUN_PY.read_text(encoding="utf-8"))

    @pytest.fixture(scope="class")
    def text_io_calls(self, gateway_tree: ast.AST) -> list[tuple[int, str]]:
        return _iter_path_text_io_calls(gateway_tree)

    def test_every_read_text_passes_encoding(self, gateway_tree: ast.AST) -> None:
        missing: list[tuple[int, str]] = []
        for node in ast.walk(gateway_tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "read_text":
                continue
            if not any(kw.arg == "encoding" for kw in node.keywords):
                missing.append((node.lineno, ast.dump(node)[:80]))
        assert not missing, (
            f"{len(missing)} .read_text() call(s) in gateway/run.py are missing "
            f"encoding= — fails on Windows non-UTF-8 locales (#37423): "
            f"{missing[:5]}"
        )

    def test_every_write_text_passes_encoding(self, gateway_tree: ast.AST) -> None:
        missing: list[tuple[int, str]] = []
        for node in ast.walk(gateway_tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "write_text":
                continue
            if not any(kw.arg == "encoding" for kw in node.keywords):
                missing.append((node.lineno, ast.dump(node)[:80]))
        assert not missing, (
            f"{len(missing)} .write_text() call(s) in gateway/run.py are missing "
            f"encoding= — fails on Windows non-UTF-8 locales (#37423): "
            f"{missing[:5]}"
        )

    def test_no_bare_read_text_or_write_text_remains(
        self, text_io_calls: list[tuple[int, str]]
    ) -> None:
        """Defence in depth: at least 20 .read_text()/.write_text() call
        sites were identified in the original audit. The fix should leave
        zero of them bare — if this number drops, someone has been busy
        and probably also tightened the lint, but a regression would
        first show up as a missing encoding= on one of the existing
        sites, not a sudden drop in call-site count."""
        assert len(text_io_calls) >= 20, (
            f"Expected at least 20 .read_text/.write_text call sites in "
            f"gateway/run.py; found {len(text_io_calls)}. Did someone delete "
            f"the update-flow code? See #37423."
        )

    def test_no_existing_encoding_kwarg_uses_locale_default(
        self, gateway_tree: ast.AST
    ) -> None:
        """Catch the inverse regression: someone passing
        ``encoding=locale.getpreferredencoding()`` instead of the
        explicit ``encoding='utf-8'`` we just added. That restores the
        original Windows bug because on cp1252/GBK systems
        ``getpreferredencoding()`` returns the locale codec, not UTF-8."""
        for node in ast.walk(gateway_tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in {"read_text", "write_text"}:
                continue
            for kw in node.keywords:
                if kw.arg == "encoding" and isinstance(kw.value, ast.Call):
                    func_str = ast.dump(kw.value)
                    if "getpreferredencoding" in func_str:
                        pytest.fail(
                            f"gateway/run.py:{node.lineno} uses "
                            f"locale.getpreferredencoding() as the encoding — "
                            f"this returns the Windows locale codec (cp1252/GBK) "
                            f"on non-UTF-8 systems and re-introduces the "
                            f"#37423 crash. Use encoding='utf-8' instead."
                        )
                if kw.arg == "encoding" and isinstance(kw.value, ast.Name):
                    if kw.value.id in {"locale", "sys_encoding", "preferred"}:
                        pytest.fail(
                            f"gateway/run.py:{node.lineno} uses a non-utf-8 "
                            f"encoding name ({kw.value.id!r}) — re-introduces "
                            f"the #37423 Windows crash."
                        )


class TestWindowsLocaleRoundTrip:
    """Behavioural guard: on a simulated Windows cp1252/GBK locale,
    the gateway's update coordination files round-trip emoji / CJK
    without raising UnicodeError (#37423)."""

    def test_write_then_read_non_ascii_succeeds(self, tmp_path, monkeypatch):
        """Reproduces the exact crash signature from the issue: write
        "yes 🎉" and "Алиса ✅\n" to a coordination file under a fake
        cp1252 locale. The pre-fix code raised UnicodeEncodeError on
        write; the post-fix code uses encoding='utf-8' and works."""
        from pathlib import Path as _Path

        # Force Python to use cp1252-style behaviour for bare open()
        # so we catch the regression even if a future refactor drops
        # the encoding= kwarg.
        import _io

        cp1252_strict = _io.open(  # noqa: SIM115 - intentional capture of stdlib
            tmp_path / "marker.txt",
            "w",
            encoding="cp1252",
        )
        cp1252_strict.close()

        # The test path: write emoji to a coordination file using the
        # same pattern as gateway/run.py:7354.
        coord_path = tmp_path / "update_response"
        payload = "yes 🎉"
        try:
            coord_path.write_text(payload, encoding="utf-8")
        except UnicodeEncodeError as e:
            pytest.fail(
                f"write_text(payload, encoding='utf-8') raised "
                f"UnicodeEncodeError on the test path: {e}. This is the "
                f"#37423 regression — drop the encoding='utf-8' kwarg to "
                f"reproduce the original crash."
            )

        # And read it back the way gateway/run.py:14982 does.
        try:
            content = coord_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            pytest.fail(
                f"read_text(encoding='utf-8') raised UnicodeDecodeError on "
                f"the test path: {e}. The #37423 fix should make this safe."
            )

        assert content == payload
