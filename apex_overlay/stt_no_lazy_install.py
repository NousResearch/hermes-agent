"""hc-401 (was patch_native_agent_stt.py / hc-254) — disable faster-whisper
lazy-install on the *managed* runtime, as a zero-in-place overlay seam.

What this replaces
==================
``tools/transcription_tools.py:_try_lazy_install_stt()`` lazy-installs
``faster_whisper`` on first use (via ``tools.lazy_deps.ensure(...)``). On a
768MB-budgeted managed container that download + CPU inference is exactly the
kind of work that OOM-kills the gateway (hc-205/hc-212). Managed/hosted agents
must transcribe through the platform media pipeline (Volcengine ASR via the
internal ``/media/transcribe`` endpoint) only, never by self-installing a local
transcriber. The original cloud fix neutered the function in-place at image
build time (an early ``return False``).

Desktop vs. managed — why this seam is *gated*
==============================================
The original cloud patch unconditionally forced ``return False``. That is
correct for the cloud image but would silently change **desktop** STT behavior
if applied verbatim in the fork (desktop legitimately lazy-installs
faster-whisper for local transcription). So this seam gates on the managed
runtime signal: ``_try_lazy_install_stt`` returns ``False`` *only when*
``HERMES_MANAGED_RUNTIME`` is truthy (``1/true/yes/on``); otherwise it delegates
to the original upstream implementation. The cloud image sets
``HERMES_MANAGED_RUNTIME=1``; desktop images (built from this same fork) don't,
so desktop keeps upstream lazy-install behavior unchanged.

Relationship to ``HERMES_DISABLE_LAZY_INSTALLS`` (not redundant)
===============================================================
The fork Dockerfile sets ``HERMES_DISABLE_LAZY_INSTALLS=1`` unconditionally on
BOTH the cloud and desktop images. That global flag is honored by
``tools.lazy_deps`` — but it only *seals the base venv*: when a durable
lazy-install target exists (the managed container's ``/opt/data`` volume is
exactly such a target), ``lazy_deps`` **redirects the install there and still
proceeds** (see ``tools/lazy_deps.py`` — the seal returns
``_lazy_install_target() is not None``). So on a managed container with a data
volume, ``HERMES_DISABLE_LAZY_INSTALLS=1`` alone would NOT stop faster-whisper
from installing (into the volume) and OOM-killing the gateway. This seam is
therefore the real guarantee, not defense-in-depth: it forces
``_try_lazy_install_stt`` to return ``False`` outright on managed runtimes,
before ``lazy_deps`` is consulted at all. Gating on ``HERMES_MANAGED_RUNTIME``
(rather than reusing the global seal) keeps the intent explicit (managed-only)
and keeps desktop STT behavior at the upstream default at the seam level.

The wrapper is idempotent (``_MARK`` sentinel on the installed function) and
fail-safe — it never raises into the host, and if the managed-env check itself
misfires it errs toward *delegating* to upstream (the safe, behavior-preserving
default). ``apply()`` returns False only if the target symbol is missing (the
seam-test turns that into a loud CI failure).
"""

from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger(__name__)

_TARGET_MODULE = "tools.transcription_tools"
_TARGET_FN = "_try_lazy_install_stt"

# Truthy spellings for the managed-runtime env flag. Mirrors apex_overlay.region
# / the shell detector's truthy set so behavior is consistent across the overlay.
_TRUE_ENV = {"1", "true", "yes", "on"}

_APPLIED = False
_MARK = "_apex_overlay_stt_no_lazy_install"


def _is_managed_runtime() -> bool:
    """True when this process is a managed/hosted container (HERMES_MANAGED_RUNTIME).

    Read fresh per call and fully defensive: any error resolves to False, i.e.
    "not managed" → delegate to upstream lazy-install (the desktop-preserving
    default). The cloud image sets this to ``1``.
    """
    try:
        return (os.environ.get("HERMES_MANAGED_RUNTIME", "") or "").strip().lower() in _TRUE_ENV
    except Exception:
        return False


def _wrap_try_lazy_install_stt(orig):
    """Return a wrapper that forces False on managed runtimes, else delegates.

    On a managed container the faster-whisper download/build must never run
    (OOM risk); returning False makes the local STT provider report unavailable
    so the agent falls back to the platform ASR pipeline. Everywhere else the
    original upstream implementation is called unchanged.
    """

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        try:
            if _is_managed_runtime():
                logger.debug(
                    "apex_overlay: skipping faster-whisper lazy-install on managed "
                    "runtime (hc-401/hc-254); platform ASR is the sanctioned path.",
                )
                return False
        except Exception:
            # Never let the gate itself break the host — fall through to upstream.
            pass
        return orig(*args, **kwargs)

    setattr(wrapper, _MARK, True)
    return wrapper


def apply() -> bool:
    """Install the managed-runtime STT lazy-install guard. Idempotent, fail-safe.

    Returns True when installed (or already present), False if the target symbol
    is missing (seam-test → hard CI failure). Never raises into plugin discovery.
    """
    global _APPLIED
    if _APPLIED:
        return True

    import importlib

    try:
        mod = importlib.import_module(_TARGET_MODULE)
        orig = getattr(mod, _TARGET_FN)
        if not getattr(orig, _MARK, False):
            setattr(mod, _TARGET_FN, _wrap_try_lazy_install_stt(orig))
    except (ImportError, AttributeError) as exc:
        logger.error(
            "apex_overlay: could not patch %s.%s — managed-runtime STT lazy-install "
            "guard is NOT active. A managed container may OOM lazy-installing "
            "faster-whisper. Upstream may have renamed/moved it. (%s)",
            _TARGET_MODULE, _TARGET_FN, exc,
        )
        return False

    _APPLIED = True
    logger.debug("apex_overlay: managed-runtime STT lazy-install guard seam applied")
    return True
