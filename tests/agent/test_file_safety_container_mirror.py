"""Tests for the container-context sandbox-mirror guard (#32049 follow-up).

Brian's shape-based guard (#32213) catches paths that carry the full
``…/sandboxes/<backend>/<task>/home/.hermes/…`` prefix. This covers the
complementary inner-container case: when file tools execute inside Docker,
the bind-mount strips that prefix and the guard sees plain ``/root/.hermes/…``.
The root:root ownership on the divergent SOUL.md in #32049 confirms this
is the primary failure mode.
"""
from __future__ import annotations

import pytest


class TestClassifyContainerMirrorTarget:
    def test_returns_none_without_context(self):
        """No Docker context — /root/.hermes/… must not be flagged."""
        from agent.file_safety import classify_container_mirror_target

        assert classify_container_mirror_target("/root/.hermes/profiles/group1/SOUL.md") is None

    def test_catches_soul_md_with_context(self):
        """Primary failure mode from #32049: agent writes SOUL.md via container path."""
        from agent.file_safety import _CONTAINER_HERMES_MIRROR, classify_container_mirror_target

        token = _CONTAINER_HERMES_MIRROR.set("/root/.hermes")
        try:
            result = classify_container_mirror_target("/root/.hermes/profiles/group1/SOUL.md")
            assert result is not None
            assert result["mirror_root"].endswith("root/.hermes")
            assert result["inner_path"] == "profiles/group1/SOUL.md"
        finally:
            _CONTAINER_HERMES_MIRROR.reset(token)

    @pytest.mark.parametrize("inner", [
        "SOUL.md",
        "memories/MEMORY.md",
    ])
    def test_catches_authoritative_profile_files(self, inner):
        from agent.file_safety import _CONTAINER_HERMES_MIRROR, classify_container_mirror_target

        token = _CONTAINER_HERMES_MIRROR.set("/root/.hermes")
        try:
            result = classify_container_mirror_target(f"/root/.hermes/{inner}")
            assert result is not None
            assert result["inner_path"] == inner
        finally:
            _CONTAINER_HERMES_MIRROR.reset(token)

    def test_non_hermes_path_not_flagged(self):
        """/root/workspace/… is not .hermes state and must not be blocked."""
        from agent.file_safety import _CONTAINER_HERMES_MIRROR, classify_container_mirror_target

        token = _CONTAINER_HERMES_MIRROR.set("/root/.hermes")
        try:
            assert classify_container_mirror_target("/root/workspace/main.py") is None
        finally:
            _CONTAINER_HERMES_MIRROR.reset(token)


class TestGetContainerMirrorWarning:
    def test_warning_names_inner_path_and_bypass(self):
        from agent.file_safety import _CONTAINER_HERMES_MIRROR, get_container_mirror_warning

        token = _CONTAINER_HERMES_MIRROR.set("/root/.hermes")
        try:
            warn = get_container_mirror_warning("/root/.hermes/profiles/group1/SOUL.md")
            assert warn is not None
            assert "profiles/group1/SOUL.md" in warn
            assert "cross_profile=True" in warn
        finally:
            _CONTAINER_HERMES_MIRROR.reset(token)


class TestOrthogonality:
    """Container-context guard catches what the shape-based guard (#32213) misses."""

    def test_inner_container_path_caught_by_context_guard(self):
        """No sandboxes/ segment — shape guard passes, context guard blocks."""
        from agent.file_safety import _CONTAINER_HERMES_MIRROR, classify_container_mirror_target

        path = "/root/.hermes/profiles/group1/SOUL.md"

        assert classify_container_mirror_target(path) is None  # no context

        token = _CONTAINER_HERMES_MIRROR.set("/root/.hermes")
        try:
            assert classify_container_mirror_target(path) is not None
        finally:
            _CONTAINER_HERMES_MIRROR.reset(token)
