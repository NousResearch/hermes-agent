"""Install-scoped update channel record (root decision 1 of the stable-channel contract).

The channel is an *installation* property, not a per-profile preference: one
JSON record at ``get_default_hermes_root()/update-channel.json`` is resolved at
call time and shared by every profile using that install (the CLI updater, the
banner's passive check, the dashboard routes, and the Electron app, which
computes the same path for its local installation root).

Contract:

- Schema ``{"schema_version": 1, "channel": "stable" | "beta"}``.
- A missing or malformed record reads as **stable** — the safe default — and a
  read NEVER writes (no startup mutation of a missing record).
- ``beta`` means tracking the moving ``main`` branch. It is NOT a release
  candidate channel.
- Only an explicit selection (``hermes update --channel …`` or the Desktop
  selector) persists a record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hermes_constants import get_default_hermes_root

CHANNEL_RECORD_SCHEMA_VERSION = 1
CHANNEL_RECORD_FILENAME = "update-channel.json"
STABLE_CHANNEL = "stable"
BETA_CHANNEL = "beta"
VALID_CHANNELS = (STABLE_CHANNEL, BETA_CHANNEL)


def channel_record_path(hermes_root: Path | None = None) -> Path:
    """Path of the install-scoped channel record (resolved at call time)."""
    root = Path(hermes_root) if hermes_root is not None else get_default_hermes_root()
    return root / CHANNEL_RECORD_FILENAME


def read_channel_record(hermes_root: Path | None = None) -> dict[str, Any] | None:
    """Return the validated record dict, or ``None`` when absent/invalid.

    Never writes, never raises for the common failure modes (missing file,
    unreadable, malformed JSON, wrong shape). An invalid record is treated as
    absent — callers fall back to the stable default.
    """
    path = channel_record_path(hermes_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    channel = raw.get("channel")
    if channel not in VALID_CHANNELS:
        return None
    return {"schema_version": CHANNEL_RECORD_SCHEMA_VERSION, "channel": channel}


def read_update_channel(hermes_root: Path | None = None) -> str:
    """The effective channel for this installation: the record's value or stable.

    Read-only; a missing or invalid record means stable, never a write.

    Migration (root decision 1): when no record exists yet, an explicit legacy
    config choice (``updates.channel``/``check_strategy``/``strategy``) still
    counts as consent — it is honored read-only until an explicit selection
    persists the install-scoped record, which then wins for every profile.
    A mere main checkout is NOT consent and reads as stable.
    """
    record = read_channel_record(hermes_root)
    if record is not None:
        return record["channel"]
    legacy = _legacy_config_channel()
    return legacy if legacy in VALID_CHANNELS else STABLE_CHANNEL


def _legacy_config_channel() -> str | None:
    """Explicit legacy per-profile channel choice, or ``None`` when absent/implicit."""
    try:
        from hermes_cli.config import load_config
        from hermes_cli.stable_update import normalize_update_channel

        config = load_config() or {}
    except Exception:
        return None
    updates = config.get("updates", {}) if isinstance(config, dict) else {}
    if not isinstance(updates, dict):
        return None
    raw = updates.get("channel") or updates.get("check_strategy") or updates.get("strategy")
    if not str(raw or "").strip():
        return None  # no explicit choice: stable default, not a migration
    normalized = normalize_update_channel(raw, default=STABLE_CHANNEL)
    return BETA_CHANNEL if normalized == "main" else STABLE_CHANNEL


def write_channel_record(channel: str, hermes_root: Path | None = None) -> dict[str, Any]:
    """Persist an explicit channel selection; returns the record written.

    Raises ``ValueError`` for a channel outside ``("stable", "beta")``. The
    write is atomic (temp file + rename) so a crash cannot strand a torn JSON
    record that would silently flip an install back to stable.
    """
    normalized = str(channel or "").strip().lower()
    if normalized in {"beta", "main", "fast", "fast-track", "fast_track"}:
        normalized = BETA_CHANNEL
    elif normalized in {"stable", "release"} or not normalized:
        normalized = STABLE_CHANNEL
    if normalized not in VALID_CHANNELS:
        raise ValueError(f"Unknown update channel: {channel!r}")

    record = {"schema_version": CHANNEL_RECORD_SCHEMA_VERSION, "channel": normalized}
    path = channel_record_path(hermes_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return record
