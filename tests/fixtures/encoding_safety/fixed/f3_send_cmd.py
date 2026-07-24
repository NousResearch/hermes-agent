"""Fixed form of H3 — send_cmd private loader BOM-safe (#65124)."""
import codecs
import io
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
            load_dotenv(str(env_path), override=True, encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:
                raw = Path(env_path).read_bytes()
                if raw.startswith(codecs.BOM_UTF8):
                    raw = raw[len(codecs.BOM_UTF8) :]
                load_dotenv(
                    stream=io.StringIO(raw.decode("latin-1")), override=True
                )
            except Exception:
                pass
        except Exception:
            pass
