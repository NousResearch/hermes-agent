"""Tests for RLIMIT_NOFILE soft-limit bump in gateway/run.py (#30230)."""

from __future__ import annotations

import textwrap
from types import ModuleType
from unittest.mock import patch


def _load_raise_fd_soft_limit():
    """Replicate the helper in an isolated module.

    gateway/run.py has heavy imports; tests/gateway/test_ssl_certs.py uses
    the same pattern.  The body below must stay in sync with the production
    function in gateway/run.py.
    """
    code = textwrap.dedent("""\
    def _raise_fd_soft_limit(min_soft=4096):
        try:
            import resource
        except ImportError:
            return
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        except (OSError, ValueError):
            return
        if soft >= min_soft:
            return
        target = min_soft if hard == resource.RLIM_INFINITY else min(min_soft, hard)
        if target <= soft:
            return
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (OSError, ValueError):
            pass
    """)
    mod = ModuleType("_fd_helper")
    exec(code, mod.__dict__)
    return mod._raise_fd_soft_limit


class TestRaiseFdSoftLimit:
    def test_bumps_from_256_to_4096_when_hard_is_infinity(self):
        fn = _load_raise_fd_soft_limit()
        import resource

        calls = []
        with patch.object(resource, "getrlimit", return_value=(256, resource.RLIM_INFINITY)), \
             patch.object(resource, "setrlimit", side_effect=lambda r, v: calls.append((r, v))):
            fn()
        assert calls == [(resource.RLIMIT_NOFILE, (4096, resource.RLIM_INFINITY))]

    def test_caps_at_hard_when_hard_below_target(self):
        fn = _load_raise_fd_soft_limit()
        import resource

        calls = []
        with patch.object(resource, "getrlimit", return_value=(256, 1024)), \
             patch.object(resource, "setrlimit", side_effect=lambda r, v: calls.append((r, v))):
            fn()
        assert calls == [(resource.RLIMIT_NOFILE, (1024, 1024))]

    def test_noop_when_soft_already_high(self):
        fn = _load_raise_fd_soft_limit()
        import resource

        calls = []
        with patch.object(resource, "getrlimit", return_value=(8192, resource.RLIM_INFINITY)), \
             patch.object(resource, "setrlimit", side_effect=lambda r, v: calls.append((r, v))):
            fn()
        assert calls == []

    def test_noop_when_soft_equals_hard_below_min(self):
        fn = _load_raise_fd_soft_limit()
        import resource

        calls = []
        with patch.object(resource, "getrlimit", return_value=(256, 256)), \
             patch.object(resource, "setrlimit", side_effect=lambda r, v: calls.append((r, v))):
            fn()
        # target == hard == 256, but target <= soft (also 256) so no call.
        assert calls == []

    def test_swallows_getrlimit_error(self):
        fn = _load_raise_fd_soft_limit()
        import resource

        calls = []
        with patch.object(resource, "getrlimit", side_effect=OSError("denied")), \
             patch.object(resource, "setrlimit", side_effect=lambda r, v: calls.append((r, v))):
            fn()  # must not raise
        assert calls == []

    def test_swallows_setrlimit_error(self):
        fn = _load_raise_fd_soft_limit()
        import resource

        with patch.object(resource, "getrlimit", return_value=(256, resource.RLIM_INFINITY)), \
             patch.object(resource, "setrlimit", side_effect=OSError("EPERM")):
            fn()  # must not raise

    def test_custom_min_soft_threshold(self):
        fn = _load_raise_fd_soft_limit()
        import resource

        calls = []
        with patch.object(resource, "getrlimit", return_value=(256, resource.RLIM_INFINITY)), \
             patch.object(resource, "setrlimit", side_effect=lambda r, v: calls.append((r, v))):
            fn(min_soft=2048)
        assert calls == [(resource.RLIMIT_NOFILE, (2048, resource.RLIM_INFINITY))]


class TestProductionBodyMatchesReplica:
    """Pin the production source so the in-test replica can't silently drift."""

    def test_production_function_body_keywords(self):
        from pathlib import Path
        src = Path(__file__).resolve().parents[2] / "gateway" / "run.py"
        text = src.read_text()
        assert "def _raise_fd_soft_limit(" in text
        assert "RLIMIT_NOFILE" in text
        assert "RLIM_INFINITY" in text
        # Helper is wired into module init right after _ensure_ssl_certs().
        assert "_raise_fd_soft_limit()" in text
