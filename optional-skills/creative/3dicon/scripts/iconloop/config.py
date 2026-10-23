"""Key loading and fail-fast checks.

Keys are read from the environment first, then a .env beside the project. They
are never logged: a run that cannot find a key says which name is missing and
stops before spending anything.
"""
import os
import sys

DOTENV_NAMES = (".env", ".env.local")


def load_dotenv(start=None):
    """Read the nearest .env into os.environ without overwriting real env vars."""
    d = os.path.abspath(start or os.getcwd())
    while True:
        for name in DOTENV_NAMES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                for line in open(p, encoding="utf-8", errors="ignore"):
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
                return p
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def key(name, why):
    """Fetch a required key or exit with a message naming it and what it is for."""
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(
            f"\n{name} is not set — needed to {why}.\n"
            f"Copy .env.example to .env and fill it in, or export the variable.\n"
        )
    return v


def opt(name, default=None):
    v = os.environ.get(name, "").strip()
    return v or default


def require_tool(binary, hint):
    """Fail before any paid call if a required external binary is missing."""
    from shutil import which
    if which(binary) is None:
        sys.exit(f"\n`{binary}` not found on PATH — {hint}\n")
