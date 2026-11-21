"""Resolved-home test for agent.redact's ``config.yaml`` secret-file detection (moved out unchanged)."""

import os


def _is_under_hermes_home(path: str) -> bool:
    """True when an absolute ``config.yaml`` path sits under the active Hermes home or root.

    The default home's basename is an installation detail — ``.hermes`` on POSIX, ``hermes``
    under ``AppData/Local`` on Windows — and a resolved path never spells ``$HERMES_HOME``,
    so the literal-segment test in ``_is_secret_file_arg`` cannot see a native Windows path.
    Compare against the resolved homes instead. Only reached for a ``config.yaml`` basename,
    so the resolve cost stays off the per-token command scan.
    """
    from agent.file_safety import _hermes_dirs
    from tools.environments.sandbox_scope import sandbox_lexical_path
    # A hosted path names a file in the sandbox: judged as written (never empty), not resolved on this host.
    try:
        target = os.path.normcase(sandbox_lexical_path(path) or os.path.realpath(os.path.expanduser(path)))
    except (OSError, ValueError):
        return False
    for home in _hermes_dirs():
        try:
            base = os.path.normcase(os.path.realpath(str(home)))
        except (OSError, ValueError):
            continue
        if target == base or target.startswith(base + os.sep):
            return True
    return False


def is_hermes_credential_store(path: str, hermes_home: bool, parts: list[str]) -> bool:
    """True for a Hermes-home credential store (``auth.json``, ``.anthropic_oauth.json``,
    ``auth/google_oauth.json``, ...): the same inventory the file tools hard-deny
    (``agent/file_safety._CREDENTIAL_FILE_NAMES``), so the redaction and file-tool lists
    cannot drift. Anchored like ``config.yaml`` — a ``.hermes`` segment, ``$HERMES_HOME``,
    or the resolved Hermes home — so a project's own ``auth.json`` never matches (#157915).
    """
    from agent.file_safety import _CREDENTIAL_FILE_NAMES

    if not (hermes_home or ".hermes" in parts[:-1] or _is_under_hermes_home(path)):
        return False
    for name in _CREDENTIAL_FILE_NAMES:
        segs = [part.lower() for part in name.replace("\\", "/").split("/") if part]
        if segs and parts[-len(segs):] == segs:
            return True
    return False
