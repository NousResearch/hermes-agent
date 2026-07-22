"""Regression tests for #64065 — platform.matrix hidden on macOS too.

python-olm has no Python 3.13 wheel and fails to build on modern macOS
(cmake 4.x rejects libolm's cmake_minimum_required, then Clang 21+
fails on a T*const increment in libolm/list.hh). The docs already say
"Matrix is not supported on macOS ARM64." The existing gate hides
matrix on win32 but not on darwin, so `hermes update` tries to pip
install mautrix[encryption] -> python-olm on macOS and warns.

The fix is a small helper `_should_hide_matrix(platform)` in
hermes_cli/gateway.py that returns True for both win32 and darwin.
These tests target the helper directly so we don't have to mock
sys.platform (which would also trigger asyncio's Windows-only code
paths and unrelated ModuleNotFoundError).
"""


def test_should_hide_matrix_true_on_win32():
    """Regression guard — the original gate that this PR extends."""
    from hermes_cli.gateway import _should_hide_matrix

    assert _should_hide_matrix("win32") is True


def test_should_hide_matrix_true_on_darwin():
    """#64065: darwin must be gated too. python-olm has no macOS wheel."""
    from hermes_cli.gateway import _should_hide_matrix

    assert _should_hide_matrix("darwin") is True, (
        "darwin must hide matrix (#64065) — python-olm fails to build "
        "on Python 3.13 / cmake 4.x / Clang 21+"
    )


def test_should_hide_matrix_false_on_linux():
    """Linux keeps matrix visible — python-olm builds successfully there."""
    from hermes_cli.gateway import _should_hide_matrix

    assert _should_hide_matrix("linux") is False


def test_should_hide_matrix_false_on_other_platforms():
    """Any platform string that's not win32 or darwin returns False."""
    from hermes_cli.gateway import _should_hide_matrix

    for plat in ("linux", "freebsd", "openbsd", "cygwin", "", "Win32", "Darwin"):
        # Note: case-sensitive on purpose. sys.platform returns the lowercase
        # form on all major Python platforms; the helper mirrors that.
        if plat in ("linux",):
            continue  # covered above
        assert _should_hide_matrix(plat) is False, (
            f"unexpected: _should_hide_matrix({plat!r}) returned True"
        )
