"""The Docker image must keep uv and pip reachable from interactive shells.

The pinned uv is installed into the PM store (``/opt/hermes/tools``), which a
plain ``docker exec`` shell never sees: the image's ENV PATH covers
``/opt/hermes/bin`` and the app venv, neither of which holds uv or pip. Only
what the toolchain layer exports to ``/usr/local/bin`` works there, and
python-build-standalone ships no pip at all — the pre-PM base image inherited
both from its uv sidecar image (#143108).
"""
from __future__ import annotations

import re
from pathlib import Path

_DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile"


def _toolchain_layer() -> str:
    """The RUN block that provisions the pinned toolchain and exports commands."""
    text = _DOCKERFILE.read_text(encoding="utf-8")
    anchor = "from pm import ensure"
    assert anchor in text, "the PM toolchain RUN layer is gone from the Dockerfile"
    start = text.index(anchor)
    return text[start:text.index("\n\n", start)]


def test_pinned_uv_is_exported_to_usr_local_bin() -> None:
    """uv is ensured into the store but only the symlink map reaches shells."""
    layer = _toolchain_layer()
    assert '("uv", "uv")' in layer, (
        "the pinned uv is ensured into the PM store but never symlinked to "
        "/usr/local/bin — `uv` stays command-not-found in docker exec shells"
    )


def test_pip_is_a_uv_backed_shim_with_a_system_fallback() -> None:
    """python-build-standalone ships no pip, so pip must be shimmed to `uv pip`.

    The shim keeps pip's own semantics: an active VIRTUAL_ENV receives the
    packages, everything else lands in the exported interpreter's environment.
    UV_SYSTEM_PYTHON (not a ``--system`` argument — uv only accepts it after
    the subcommand) selects the system interpreter without reordering "$@".
    """
    layer = _toolchain_layer()
    match = re.search(r"printf '([^']+)' > /usr/local/bin/pip", layer)
    assert match, "the /usr/local/bin/pip shim is gone from the toolchain layer"
    shim = match.group(1)
    assert "exec uv pip" in shim, "the pip shim must forward to `uv pip`"
    assert "UV_SYSTEM_PYTHON" in shim and "VIRTUAL_ENV" in shim, (
        "the pip shim must mirror pip's target selection: the active "
        "VIRTUAL_ENV when set, the exported interpreter otherwise"
    )


def test_smoke_check_covers_uv_and_pip() -> None:
    """The layer must fail the build if either command stops resolving."""
    layer = _toolchain_layer()
    assert re.search(r"uv --version", layer), "no uv smoke check in the toolchain layer"
    assert re.search(r"pip --help", layer), "no pip smoke check in the toolchain layer"
