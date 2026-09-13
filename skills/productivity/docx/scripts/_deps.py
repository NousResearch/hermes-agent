"""Lazy-install this skill's third-party libraries on first use.

The scripts here are standalone CLIs invoked through the ``terminal`` tool, so
they run in their own process and cannot rely on the agent having imported
anything. ``tools.lazy_deps`` is Hermes' own install-at-first-use allowlist;
``ensure`` is a no-op once the packages are present, so calling it at import
time costs nothing on a warm install.

Outside a Hermes checkout (a bare ``python3 scripts/...`` on a machine without
Hermes on ``sys.path``) the import fails and each script's own
``except ImportError`` guard prints an install hint instead.
"""

FEATURE = "skill.docx"


def ensure_ready() -> None:
    """Install this skill's libraries if they are missing. Never raises."""
    try:
        from tools.lazy_deps import ensure
    except Exception:
        return  # not running inside a Hermes install; fall through to the guard
    try:
        ensure(FEATURE, prompt=False)
    except Exception:
        return  # installs disabled or failed; fall through to the guard
