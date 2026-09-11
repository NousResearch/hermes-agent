"""Fail-closed ingress bridge for the fixed Max English material wrapper."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IngressDispatchResult:
    handled: bool
    status: str


def dispatch(*, settings: object, chat_id: str, user_id: str, message_ids: list[str],
             media_group_id: str | None, cached_paths: list[str]) -> IngressDispatchResult:
    """Stage one authorized JPEG/PNG batch and invoke no configurable shell."""
    if not isinstance(settings, dict) or settings.get("enabled") is not True:
        return IngressDispatchResult(False, "disabled")
    if str(settings.get("chat_id")) != chat_id or str(settings.get("user_id")) != user_id:
        return IngressDispatchResult(False, "unauthorized")
    root_value = settings.get("staging_root")
    if not isinstance(root_value, str) or not message_ids or len(message_ids) != len(cached_paths):
        return IngressDispatchResult(True, "failed")
    root = Path(root_value)
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.stat().st_mode & 0o077:
            return IngressDispatchResult(True, "failed")
        batch = root / ("album-" + media_group_id if media_group_id else "single-" + message_ids[0])
        batch.mkdir(mode=0o700, exist_ok=True)
        files = []
        for position, (message_id, raw) in enumerate(zip(message_ids, cached_paths)):
            source = Path(raw)
            if not source.is_file() or source.suffix.lower() not in {".jpg", ".png"}:
                return IngressDispatchResult(True, "needs_resubmission")
            target = batch / f"{position:02d}{source.suffix.lower()}"
            if not target.exists():
                shutil.copy2(source, target)
                os.chmod(target, 0o600)
            files.append({"message_id": message_id, "staged_path": str(target.relative_to(root))})
        key = f"max:album:{media_group_id}" if media_group_id else f"max:single:{message_ids[0]}"
        payload = {"ingress_key": key, "media_group_id": media_group_id, "files": files}
        fd, temporary = tempfile.mkstemp(prefix="receipt-", suffix=".json", dir=batch)
        try:
            os.write(fd, json.dumps(payload, separators=(",", ":")).encode())
        finally:
            os.close(fd)
        manifest = Path(temporary); os.chmod(manifest, 0o600)
        wrapper = "/home/alan/.hermes/profiles/max/skills/max-english-learning/bin/receive-telegram-material"
        completed = subprocess.run([wrapper], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True, timeout=30,
                                   env={"PATH": os.defpath, "HOME": "/home/alan", "LANG": "C.UTF-8",
                                        "HERMES_ENGLISH_TELEGRAM_MANIFEST": str(manifest)})
        if completed.returncode:
            return IngressDispatchResult(True, "retrying")
        result = json.loads(completed.stdout)
        return IngressDispatchResult(True, str(result.get("status", "failed")))
    except Exception:
        return IngressDispatchResult(True, "failed")
