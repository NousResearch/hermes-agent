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
* Source installs select an R2 channel name, or use an explicit branch override.
  Bundles derive their channel from their baked identity, never these records.
  ``external`` installs have no configurable channel; the steward owns updates.

Pure-stdlib leaf module (plus hermes-internal imports done lazily): the
installers and boot paths read it before the full config machinery loads.
"""

from __future__ import annotations

from pm.environments import install_key, installs_root
from hermes_cli.release_channels import validate_name
from contextlib import contextmanager
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional
from pm.paths import install_root

logger = logging.getLogger(__name__)

CHANNEL_MAIN = "main"
CHANNEL_STABLE = "stable"
CHANNEL_CANARY = "canary"


# A canary source identity: the exact stable version plus a full UTC build
# timestamp in SemVer build metadata. Build metadata is precedence-invisible,
# so channel movement comes only from the R2 head, never version comparison.
# THIS is the single authority; producers and consumers import it rather than
# re-typing the shape.
_CANARY_TAG_RE = re.compile(
    r"^v(?:0|[1-9]\d{0,2})\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"\+canary\.20\d{6}T\d{6}Z$"
)

# A stable release tag: v<major>.<minor>.<patch>, no suffix. The major is
# capped at three digits so the historical CalVer tags (v2026.7.20) can never
# pass as SemVer and reach a stable feed, Docker publish, or the source
# updater. THIS is the single authority for the stable shape; every stable
# selector imports it rather than re-typing the rule.
STABLE_TAG_RE = re.compile(r"^v(?:0|[1-9]\d{0,2})\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$")


def is_canary_tag(tag: Any) -> bool:
    """True for a canonical canary source identity."""
    if not isinstance(tag, str):
        return False
    value = tag.strip()
    return bool(_CANARY_TAG_RE.fullmatch(value))


def canary_timestamp(tag: Any) -> str | None:
    """Return the UTC receipt stamp from a canonical canary tag."""
    if not is_canary_tag(tag):
        return None
    return tag.strip().split("+canary.", 1)[1]


def canary_tag_for_date(version: str, date_utc: str) -> str:
    """Return ``v<stable>+canary.<full UTC timestamp>``.

    The stable core is unchanged. The suffix is build metadata, so the source
    identity compares equal to the stable and only the R2 channel head moves a
    canary subscriber.
    """
    core = version.removeprefix("v")
    if not STABLE_TAG_RE.fullmatch("v" + core):
        raise ValueError(f"invalid stable version for canary: {version!r}")
    if not re.fullmatch(r"20\d{6}T\d{6}Z", date_utc):
        raise ValueError(f"invalid canary UTC timestamp: {date_utc!r}")
    return f"v{core}+canary.{date_utc}"



def install_id(project_root: Optional[Path] = None) -> str:
    """The sha16 id of the install at ``project_root`` (default: this one).

    Same identity as the ``installs/<sha16>/`` state folder key.
    """
    if project_root is None:
        project_root = install_root()
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
    return stamp.get("payload") in ("bundled", "light", "runtime") or stamp.get("updateMechanism") in (
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
    root = Path(project_root) if project_root is not None else install_root()
    stamp = _read_stamp(root)
    if not _package_channel(stamp):
        return CHANNEL_MAIN
    if stamp.get("source") == "channel-build":
        request = stamp.get("channelBuild")
        if not isinstance(request, dict):
            raise ValueError("Channel bundle has no baked subscription")
        return validate_name(request.get("channel"))
    return CHANNEL_CANARY if is_canary_tag(stamp.get("tag")) else CHANNEL_STABLE


def resolve_update_channel(
    config: Optional[dict] = None,
    project_root: Optional[Path] = None,
) -> str:
    """Source records select releases or main; package tags fix bundle identity."""
    root = Path(project_root) if project_root is not None else install_root()
    if _package_channel(_read_stamp(root)):
        return default_channel(root)
    configured: Any = channel_record(config, root).get("channel")
    if configured is not None:
        return validate_name(configured)
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

    root = Path(project_root) if project_root is not None else install_root()
    if is_commit_build(root):
        raise ValueError(COMMIT_BUILD_UPDATE_MESSAGE)
    channel = validate_name(channel)

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


@contextmanager
def _channel_write_lock(config_path: Path):
    """Serialize channel selection and retirement across CLI processes."""
    from hermes_cli.config import _CONFIG_LOCK

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with _CONFIG_LOCK, config_path.with_suffix(".channels.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.write(b"\0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def adopt_retired_channel(request: dict) -> bool:
    """After verified completion, adopt only an unchanged persistent subscription."""
    retirement = request.get("channel_retirement")
    if retirement is None:
        return False
    return _write_channel_record(
        install_id(Path(request["source"])), request["source"],
        validate_name(retirement["destination"]),
        expected=retirement["original"], config_path=Path(request["home"]) / "config.yaml")


def _write_channel_record(sha16: str, path: str, channel: str, *,
                          expected: dict | None = None, config_path: Path | None = None) -> bool:
    from hermes_cli.config import get_config_path

    config_path = config_path if config_path is not None else get_config_path()
    with _channel_write_lock(config_path):
        return _write_channel_record_locked(sha16, path, channel, expected, config_path)


def _write_channel_record_locked(sha16: str, path: str, channel: str,
                                 expected: dict | None, config_path: Path) -> bool:
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

    from hermes_cli.config import require_readable_config_before_write
    existing = require_readable_config_before_write(config_path)
    update_cfg = existing.get("update")
    if update_cfg is not None and not isinstance(update_cfg, dict):
        raise ValueError("config key 'update' is not a mapping")
    installs = update_cfg.get("installs") if isinstance(update_cfg, dict) else None
    if installs is not None and not isinstance(installs, dict):
        raise ValueError("config key 'update.installs' is not a mapping")
    record = installs.get(sha16) if isinstance(installs, dict) else None
    if expected is not None and (record or {}) != expected:
        return False
    new_record = dict(record) if isinstance(record, dict) else {}
    new_record["path"] = path  # DATA, for humans + doctor GC
    new_record["channel"] = channel
    atomic_roundtrip_yaml_update(config_path, f"update.installs.{sha16}", new_record)
    return True


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
