"""Publish a local file through the running gateway so an out-of-process sender
can hand it to the user.

Why this exists
---------------
Hermes' LINE adapter serves local files over ``/line/media/<token>/<name>`` so
the bot can send images / audio / video. The **token registry lives in the
gateway process** (``LineAdapter._media_tokens``), and a LINE ``document`` push
message does not exist — the API only accepts
``text, textV2, image, video, audio, location, sticker, template, imagemap, flex``
— so a file is handed over as a flex message with a download button pointing at
that route.

That works for the *live* lane (a cron job fired by the gateway's own ticker:
the adapter is in-process and can mint a token). But an **out-of-process**
sender — ``hermes cron run`` from a CLI, the standalone lane — has no adapter
and no token registry. Before this module it could only tell the user
``1 attachment(s) generated; not deliverable from cron``.

``publish_media_file`` bridges the gap: it asks the running gateway (over
loopback) to register the file and returns the public URL.

The gateway side is the LINE plugin's loopback-only endpoint
``POST /line/media/publish`` (see ``plugins/platforms/line/adapter.py``).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

# The LINE adapter's own webhook server (Hermes runs it on 8646 by default).
DEFAULT_LOCAL_BASE = "http://127.0.0.1:8646"

# Must be short: the gateway's media token TTL is effectively unbounded, but the
# caller is a foreground send and should not hang.
_TIMEOUT_SECONDS = 10


def _line_secret() -> str:
    """The LINE channel secret, used as the bearer credential for the publish endpoint."""
    try:
        from gateway.platforms._shared import get_scoped_secret
        secret = str(get_scoped_secret("LINE_CHANNEL_SECRET", "") or "").strip()
    except ImportError:
        secret = ""
    if secret:
        return secret
    try:
        from hermes_constants import get_hermes_home
        env_path = os.path.join(str(get_hermes_home()), ".env")
    except ImportError:
        env_path = os.path.expanduser("~/.hermes/.env")  # health: allow HX001 -- fallback when hermes_constants is unavailable; the profile home is the default location
    try:
        with open(env_path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line.startswith("LINE_CHANNEL_SECRET="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def _local_base() -> str:
    try:
        from gateway.platforms._shared import get_scoped_secret
        scoped = str(get_scoped_secret("LINE_LOCAL_BASE", "") or "")
    except ImportError:
        scoped = ""
    return (scoped or os.environ.get("LINE_LOCAL_BASE", DEFAULT_LOCAL_BASE)).rstrip("/")


def publish_media_file(path: str, timeout: float = _TIMEOUT_SECONDS):
    """Register ``path`` with the running gateway and return ``{"url":..., "path":...}``.

    Returns ``None`` when the gateway cannot be reached or refuses the file — the
    caller then falls back to a plain-text notice instead of dropping the
    attachment silently. Raises ``RuntimeError`` when the gateway answers with
    an error the caller should surface.
    """
    if not path or not os.path.isfile(path):
        return None
    secret = _line_secret()
    if not secret:
        return None
    url = f"{_local_base()}/line/media/publish"
    payload = json.dumps({"path": os.path.abspath(path)}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {secret}",
            "X-Line-Secret": secret,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
        return json.loads(body)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "ignore")[:200]
        except (OSError, ValueError):
            detail = ""
        raise RuntimeError(f"media publish HTTP {exc.code}: {detail}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"media publish failed: {exc}") from exc


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 -m gateway.media_publish <file>")
        raise SystemExit(2)
    try:
        print(json.dumps(publish_media_file(sys.argv[1]), ensure_ascii=False, indent=2))
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
