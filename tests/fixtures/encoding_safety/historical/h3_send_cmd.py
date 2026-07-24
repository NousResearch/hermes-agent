"""Historical instance #3 — send_cmd._load_hermes_env (pre-#65124 send_cmd fix).

Private dotenv loader: plain utf-8 primary + latin-1 fallback, same BOM drop
as env_loader. Intentionally non-equivalent to the shared loader (profile
path resolution; no sanitize/secret-source side effects).
"""
from pathlib import Path


def _load_hermes_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None  # type: ignore[assignment]

    home = Path.home() / ".hermes"
    env_path = home / ".env"
    if load_dotenv and env_path.exists():
        try:
            load_dotenv(str(env_path), override=True, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                load_dotenv(str(env_path), override=True, encoding="latin-1")
            except Exception:
                pass
        except Exception:
            pass
