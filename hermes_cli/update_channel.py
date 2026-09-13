"""Per-install update-channel records.

Channel storage — per install, never home-global::

    update:
      installs:
        a4f3b2c1d0e9f8a7:                      # install id (sha16 of the
          path: /home/u/.hermes/hermes-agent   #   canonical install root)
          channel: canary

One config.yaml serves many installs (host + docker gateway + desktop all
bind-mount one ``~/.hermes``), so a home-global ``update.channel`` key is
UNSAFE and does not exist: setting canary for a dev checkout must not
flip the desktop app's feed. The id is sha16 of the canonical
install-root PATH — the same key that names the ``installs/<sha16>/``
state folder (``boot_bootstrap._install_key``; a byte-identical helper is
inlined below until that module lands). Path-derived on purpose: an
electron-updater update replaces the artifact (new stamp bytes) at the
same path, and the channel opt-in must survive that.

* Written by ``hermes update --set-channel <x>`` from inside an install
  (it knows its own id — the user never types a sha).
* Shown by ``hermes update --install-id`` and the desktop About page.
* Source installs select main or a published stable/canary release. Bundles
  derive their channel from the baked tag, never from these records.
  ``external`` installs have no configurable channel; the steward owns updates.

Pure-stdlib leaf module (plus hermes-internal imports done lazily): the
installers and boot paths read it before the full config machinery loads.
"""

from __future__ import annotations

from hermes_cli.runtime_paths import install_key, installs_root
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CHANNEL_MAIN = "main"
CHANNEL_STABLE = "stable"
CHANNEL_CANARY = "canary"
VALID_CHANNELS = (CHANNEL_MAIN, CHANNEL_STABLE, CHANNEL_CANARY)

# A canary release tag: v<major>.<minor>.<patch>-canary.<YYYYMMDDHHMMSS>,
# or the legacy date-only shape. THIS is the single authority for the
# canary tag shape — scripts/release.py (produces them) and
# scripts/write_install_stamp.py (validates the feed key) import it rather
# than re-typing the rule. Canaries are current-stable patch+1, so any
# patch is accepted here.
_CANARY_TAG_RE = re.compile(r"^v(?:0|[1-9]\d{0,2})\.\d+\.\d+-canary\.20\d{6}(?:\d{6})?$")


def is_canary_tag(tag: Any) -> bool:
    """True when ``tag`` is a canary release tag."""
    return isinstance(tag, str) and bool(_CANARY_TAG_RE.match(tag.strip()))


def canary_tag_for_date(version: str, date_utc: str) -> str:
    """The canary tag name for a UTC timestamp: next PATCH over ``version``
    (the newest stable's patch + 1), second-precision UTC suffix —
    v0.27.5-canary.20260818103000 when stable is v0.27.4. A canary
    outversions every stable at or below its patch and loses to the next
    stable patch, which is exactly the channel-switch upgrade path
    (canary→stable = wait for that patch bump to ship as stable).
    """
    parts = version.lstrip("v").split(".")
    major, minor = int(parts[0]), int(parts[1])
    patch = int(parts[2]) if len(parts) >= 3 else 0
    return f"v{major}.{minor}.{patch + 1}-canary.{date_utc}"


def _default_root() -> Path:
    """This process's install root.

    Mirrors version_info's stamp resolution: HERMES_INSTALL_ROOT when the
    steward wrapper sets it (Nix points it at the sealed tree), else the
    code root of the executing checkout.
    """
    root = os.environ.get("HERMES_INSTALL_ROOT")
    return Path(root) if root else Path(__file__).parent.parent.resolve()


def install_id(project_root: Optional[Path] = None) -> str:
    """The sha16 id of the install at ``project_root`` (default: this one).

    Same identity as the ``installs/<sha16>/`` state folder key.
    """
    if project_root is None:
        project_root = _default_root()
    return install_key(Path(project_root))


def _read_stamp(root: Path) -> dict:
    """The install stamp of ``root``, or ``{}`` (tolerant, like steward.py)."""
    from hermes_cli.steward import read_install_stamp

    return read_install_stamp(root)


def _install_records(config: Optional[dict]) -> dict:
    if not isinstance(config, dict):
        return {}
    update_cfg = config.get("update")
    if not isinstance(update_cfg, dict):
        return {}
    installs = update_cfg.get("installs")
    return installs if isinstance(installs, dict) else {}


def channel_record(config: Optional[dict], project_root: Optional[Path] = None) -> dict:
    """This install's ``{path, channel}`` record from config, or ``{}``."""
    record = _install_records(config).get(install_id(project_root))
    return record if isinstance(record, dict) else {}


def _package_channel(stamp: dict) -> bool:
    return stamp.get("payload") in ("bundled", "light") or stamp.get("updateMechanism") in (
        "electron-updater", "app-installer", "microsoft-store"
    )


def default_channel(project_root: Optional[Path] = None) -> str:
    """The channel an unconfigured install tracks.

    ``self`` source installs follow main (historical behavior).
    ``electron-updater`` and ``app-installer`` bundles report their artifact
    channel: a canary artifact tracks canary, every other bundle
    tracks stable. The stamp's ``tag`` is the authority, the same fact
    apps/desktop/product-identity.cjs keys the published feed name on — so
    the feed a canary artifact asks for and the feed it was published to
    can never disagree. Deriving stable here instead would send a fresh
    canary install to look for its ``canary.yml`` feed file under the
    newest STABLE release, where that file does not exist (404), leaving
    the install unable to update at all.
    """
    root = Path(project_root) if project_root is not None else _default_root()
    stamp = _read_stamp(root)
    if not _package_channel(stamp):
        return CHANNEL_MAIN
    return CHANNEL_CANARY if is_canary_tag(stamp.get("tag")) else CHANNEL_STABLE


def resolve_update_channel(
    config: Optional[dict] = None,
    project_root: Optional[Path] = None,
) -> str:
    """Source records select releases or main; package tags fix bundle identity."""
    root = Path(project_root) if project_root is not None else _default_root()
    if _package_channel(_read_stamp(root)):
        return default_channel(root)
    configured: Any = channel_record(config, root).get("channel")
    if isinstance(configured, str) and configured.strip().lower() in VALID_CHANNELS:
        return configured.strip().lower()
    return default_channel(root)


def set_install_channel(
    channel: str,
    project_root: Optional[Path] = None,
) -> str:
    """Persist ``channel`` for THIS install in config.yaml. Returns the id.

    Refuses when the update source belongs to the OS or package owner,
    including Microsoft Store, rather than this configuration.
    Raises ``ValueError`` for an invalid channel or an OS-owned install.
    """
    from hermes_cli.update_contract import COMMIT_BUILD_UPDATE_MESSAGE, is_commit_build

    root = Path(project_root) if project_root is not None else _default_root()
    if is_commit_build(root):
        raise ValueError(COMMIT_BUILD_UPDATE_MESSAGE)
    channel = (channel or "").strip().lower()
    if channel not in VALID_CHANNELS:
        raise ValueError(
            f"unknown channel {channel!r} (one of {', '.join(VALID_CHANNELS)})"
        )

    stamp = _read_stamp(root)
    if _package_channel(stamp) or stamp.get("updateMechanism") == "external":
        distribution = stamp.get("distribution") or "an external steward"
        raise ValueError(
            f"channels don't apply here; updates are owned by {distribution}"
        )

    sha16 = install_id(root)
    _write_channel_record(sha16, str(root), channel)
    return sha16


def handle_metadata_args(args, project_root: Path) -> bool:
    """Handle metadata-only update commands before any update side effect."""
    if getattr(args, "install_id", False):
        print(install_id(project_root))
        return True
    channel = getattr(args, "set_channel", None)
    if channel is None:
        return False
    try:
        key = set_install_channel(channel, project_root)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(2) from exc
    print(f"Update channel for {key}: {channel}")
    if channel == CHANNEL_CANARY:
        print("Canary builds can write forward-incompatible state. Back up your data before switching.")
    elif channel == CHANNEL_STABLE:
        print("Switching to an older stable release may not read state written by canary. Back up your data first.")
    return True


def _write_channel_record(sha16: str, path: str, channel: str) -> None:
    """Write ``update.installs.<sha16>`` into config.yaml, preserving the rest.

    Persists through the shared comment-preserving atomic writer
    (:func:`utils.atomic_roundtrip_yaml_update` — the same ruamel round-trip
    path ``hermes config set`` uses), fail-closed via
    :func:`hermes_cli.config.require_readable_config_before_write`. Malformed
    ``update`` / ``update.installs`` values are refused, never replaced —
    the dotted writer would otherwise turn a scalar into a mapping and
    destroy whatever the user had there.
    """
    from utils import atomic_roundtrip_yaml_update

    from hermes_cli.config import (
        get_config_path,
        require_readable_config_before_write,
    )

    config_path = get_config_path()
    existing = require_readable_config_before_write(config_path)
    update_cfg = existing.get("update")
    if update_cfg is not None and not isinstance(update_cfg, dict):
        raise ValueError("config key 'update' is not a mapping")
    installs = update_cfg.get("installs") if isinstance(update_cfg, dict) else None
    if installs is not None and not isinstance(installs, dict):
        raise ValueError("config key 'update.installs' is not a mapping")
    record = installs.get(sha16) if isinstance(installs, dict) else None
    new_record = dict(record) if isinstance(record, dict) else {}
    new_record["path"] = path  # DATA, for humans + doctor GC
    new_record["channel"] = channel
    atomic_roundtrip_yaml_update(config_path, f"update.installs.{sha16}", new_record)


def stale_channel_records(config: Optional[dict]) -> list[tuple[str, dict, str]]:
    """Doctor's staleness triad over ``update.installs``.

    Returns ``(sha16, record, reason)`` where reason is one of:

    * ``"replaced"`` — the recorded path exists but the install there keys
      to a DIFFERENT sha16 (the tree moved / was recreated elsewhere and a
      new record claimed it; this one is a leftover).
    * ``"missing"``  — nothing at the recorded path: offer GC (keep-on-doubt).
    * ``"unclaimed"`` — the sha16 matches no live install record
      (``installs/<sha16>/install.json``): offer GC.
    """
    stale: list[tuple[str, dict, str]] = []
    for sha16, record in _install_records(config).items():
        if not isinstance(record, dict):
            continue
        recorded_path = record.get("path")
        if not isinstance(recorded_path, str) or not recorded_path:
            # No path fact — fall through to the live-record check only.
            recorded_path = None

        if recorded_path is not None:
            path = Path(recorded_path)
            if not path.exists():
                stale.append((sha16, record, "missing"))
                continue
            if install_key(path) != sha16:
                stale.append((sha16, record, "replaced"))
                continue

        # Cross-check against the live install-state records: a channel
        # record whose sha16 has no installs/<sha16>/install.json was
        # either hand-written or its install never booted post-record.
        try:
            if not (installs_root() / sha16 / "install.json").is_file():
                stale.append((sha16, record, "unclaimed"))
        except Exception as exc:  # noqa: BLE001 — doctor sweep must not raise
            logger.debug("installs root unavailable: %s", exc)
    return stale
