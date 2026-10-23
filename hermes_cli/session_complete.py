"""Shell completion source for session names: ``handle<TAB>title`` lines, newest first.

Runs on every Tab press, so it imports nothing from Hermes (the shell calls it as ``python -I -S``)
and opens state.db read-only. The home is the one the hermes command line would use: ``-p NAME`` in
the words typed so far, else ``$HERMES_HOME``, else the sticky ``active_profile``.
"""

import os
import sqlite3
import sys
from pathlib import Path

_LIMIT = 200
_QUERY = """
    SELECT handle, COALESCE(title, '') FROM sessions  -- what `sessions list` shows: no compression
                                                      -- ancestors (their tip is listed) and no sub-agent runs
    WHERE handle IS NOT NULL AND handle LIKE ? ESCAPE '\\' AND COALESCE(hidden, 0) = 0
      AND COALESCE(end_reason, '') != 'compression' AND COALESCE(source, '') != 'tool'
      AND (CASE WHEN json_valid(model_config) THEN json_extract(model_config, '$._delegate_from') END) IS NULL
    ORDER BY COALESCE(last_activity_at, started_at) DESC LIMIT ?
"""


def _native_root(environ) -> Path:
    suffix = environ.get("HERMES_DATA_DIR_SUFFIX", "")
    if os.name == "nt":
        base = environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / ("hermes" + suffix)
    return Path.home() / (".hermes" + suffix)


def _profile_name(value):
    return value.strip().casefold() or None  # as profiles.normalize_profile_name: `-p Work` is `work`


def _profile_from_words(words):
    for i, word in enumerate(words):
        if word in ("-p", "--profile") and i + 1 < len(words):
            return _profile_name(words[i + 1])
        if word.startswith("--profile="):
            return _profile_name(word.split("=", 1)[1])
    return None


def home_for(words, environ=os.environ) -> Path:
    """Same order as ``hermes_cli.main._apply_profile_override``."""
    profile = _profile_from_words(words)
    env = Path(os.path.expandvars(environ["HERMES_HOME"])).expanduser() if environ.get("HERMES_HOME") else None
    if profile is None and env is not None and env.parent.name == "profiles":
        return env
    root = (env.parent.parent if env.parent.name == "profiles" else env) if env else _native_root(environ)
    if profile is None:
        try:
            profile = _profile_name((root / "active_profile").read_text(encoding="utf-8-sig"))
        except OSError:
            profile = None
    return root if profile in (None, "default") else root / "profiles" / profile


def candidates(db_path: Path, prefix: str):
    if not db_path.exists():
        return []
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True, timeout=0.2)
    try:
        return conn.execute(_QUERY, (escaped + "%", _LIMIT)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def main(argv) -> int:
    prefix, words = (argv[0] if argv else ""), argv[1:]
    for handle, title in candidates(home_for(words) / "state.db", prefix):
        title = " ".join(title.split())
        print(f"{handle}\t{title}" if title else handle)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
