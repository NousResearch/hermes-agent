"""Regression test for #34192 — Dockerfile must keep the tini compat shim
for orchestration templates that still reference /usr/bin/tini.

This is a documentation-as-test guard: removing the shim is a real
choice, but it should be done deliberately (e.g. once Hostinger's
'Hermes WebUI' catalog updates to /init) and not by accident.
"""

from __future__ import annotations

from pathlib import Path


def _dockerfile_text() -> str:
    return (Path(__file__).parent.parent / "Dockerfile").read_text(encoding="utf-8")


def test_tini_compat_symlink_present():
    """The /usr/bin/tini -> /init symlink line must exist for #34192."""
    df = _dockerfile_text()
    assert "ln -sf /init /usr/bin/tini" in df, (
        "Dockerfile must keep the tini compat symlink (#34192). "
        "Removing it breaks orchestration templates that still pin "
        "/usr/bin/tini as the entrypoint (Hostinger 'Hermes WebUI' "
        "catalog as of v0.14.x)."
    )


def test_tini_compat_comment_explains_why():
    """The symlink line is comment-anchored to #34192 so a future reader
    knows why it exists. Removing the comment makes it look like dead
    code worth deleting."""
    df = _dockerfile_text()
    assert "#34192" in df, (
        "The Dockerfile tini compat shim must keep its #34192 anchor "
        "comment so future maintainers know why the symlink is there."
    )


def test_entrypoint_still_init_not_tini():
    """Sanity check: the image ENTRYPOINT stays on the dispatcher, which
    preserves the `/init` path when PID 1 and falls back only when the
    runtime wraps the entrypoint under another init."""
    df = _dockerfile_text()
    assert 'ENTRYPOINT [ "/opt/hermes/docker/entrypoint-dispatch.sh" ]' in df, (
        "Dockerfile ENTRYPOINT must stay on the dispatcher so the image "
        "can preserve /init under PID 1 while still booting on non-PID-1 "
        "platforms like Fly Machines."
    )
