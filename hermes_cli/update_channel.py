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
  Unconfigured official checkouts track ``stable``; forks and mirrors keep ``main``.
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


# A canary tag: the build kind as a prefix, then a full UTC build timestamp, then the exact stable
# version it builds on -- ``canary.<YYYYMMDDTHHMMSSZ>-v<stable>``, the same claim shape as a stable
# release's ``rc.N-v<stable>``. It never matches ``v*``, so nothing that lists release tags mistakes
# it for one. The canary's *version* (what the app, the Termux package and the feeds report) stays
# SemVer: ``<stable>+canary.<stamp>`` (release_version). THIS is the single authority; producers and
# consumers import it rather than re-typing the shape.
_VERSION_CORE = r"(?:0|[1-9]\d{0,2})\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
_CANARY_TAG_RE = re.compile(rf"^canary\.(?P<stamp>20\d{{6}}T\d{{6}}Z)-v(?P<core>{_VERSION_CORE})$")
# A commit build's claim tag: the same shape with the ``commit`` kind --
# ``commit.<YYYYMMDDTHHMMSSZ>-v<pyproject version at the commit>``. The tag points at the built
# commit (any pushed commit, not only main's), and its version is ``<core>+commit.<stamp>``. Commit
# builds publish nowhere and their installs refuse updates (install stamp ``source: commit-build``).
_COMMIT_TAG_RE = re.compile(rf"^commit\.(?P<stamp>20\d{{6}}T\d{{6}}Z)-v(?P<core>{_VERSION_CORE})$")
# A channel build's claim tag: ``channel.<YYYYMMDDTHHMMSSZ>-v<pyproject version at the commit>``.
# The channel name, branding and bundle defaults live in the annotated claim. Desktop packages use the
# admitted request's identity and the claim's version (<core>+channel.<stamp>); native Windows
# versions derive from that claim's timestamp, not the channel allocation sequence.
_CHANNEL_TAG_RE = re.compile(rf"^channel\.(?P<stamp>20\d{{6}}T\d{{6}}Z)-v(?P<core>{_VERSION_CORE})$")
# Every prefix-kind claim tag that is its own release identity: kind, stamp, core.
_KIND_TAG_RE = re.compile(rf"^(?P<kind>canary|commit|channel)\.(?P<stamp>20\d{{6}}T\d{{6}}Z)-v(?P<core>{_VERSION_CORE})$")

# A stable release tag: v<major>.<minor>.<patch>, no suffix. The major is
# capped at three digits so the historical CalVer tags (v2026.7.20) can never
# pass as SemVer and reach a stable feed, Docker publish, or the source
# updater. THIS is the single authority for the stable shape; every stable
# selector imports it rather than re-typing the rule.
STABLE_TAG_RE = re.compile(rf"^v{_VERSION_CORE}$")


# The retired canary shape, ``v<stable>+canary.<stamp>``. Nothing produces it; release tooling reads
# it only in records published before the prefix shape (a channel head's previous manifest, a live
# macOS feed), so the first prefix-shaped canary can order itself after them and replace them.
_LEGACY_CANARY_TAG_RE = re.compile(rf"^v(?P<core>{_VERSION_CORE})\+canary\.(?P<stamp>20\d{{6}}T\d{{6}}Z)$")


def published_canary_timestamp(tag: Any) -> str | None:
    """The build stamp of a published canary tag of either shape (ordering against old records)."""
    if not isinstance(tag, str):
        return None
    match = _CANARY_TAG_RE.fullmatch(tag.strip()) or _LEGACY_CANARY_TAG_RE.fullmatch(tag.strip())
    return match["stamp"] if match else None


def published_release_version(tag: Any) -> str | None:
    """release_version, also reading the retired canary shape of already-published records."""
    version = release_version(tag)
    if version is None and isinstance(tag, str) and _LEGACY_CANARY_TAG_RE.fullmatch(tag.strip()):
        version = tag.strip()[1:]
    return version


def is_canary_tag(tag: Any) -> bool:
    """True for a canonical canary tag (``canary.<stamp>-vX.Y.Z``)."""
    if not isinstance(tag, str):
        return False
    return bool(_CANARY_TAG_RE.fullmatch(tag.strip()))


def canary_timestamp(tag: Any) -> str | None:
    """Return the UTC build stamp from a canonical canary tag."""
    match = _CANARY_TAG_RE.fullmatch(tag.strip()) if isinstance(tag, str) else None
    return match["stamp"] if match else None


def is_commit_tag(tag: Any) -> bool:
    """True for a canonical commit build claim tag (``commit.<stamp>-vX.Y.Z``)."""
    if not isinstance(tag, str):
        return False
    return bool(_COMMIT_TAG_RE.fullmatch(tag.strip()))


def commit_timestamp(tag: Any) -> str | None:
    """Return the UTC claim stamp from a canonical commit build tag."""
    match = _COMMIT_TAG_RE.fullmatch(tag.strip()) if isinstance(tag, str) else None
    return match["stamp"] if match else None


def is_channel_tag(tag: Any) -> bool:
    """True for a canonical channel build claim tag (``channel.<stamp>-vX.Y.Z``)."""
    if not isinstance(tag, str):
        return False
    return bool(_CHANNEL_TAG_RE.fullmatch(tag.strip()))


def channel_timestamp(tag: Any) -> str | None:
    """Return the UTC claim stamp from a canonical channel build tag."""
    match = _CHANNEL_TAG_RE.fullmatch(tag.strip()) if isinstance(tag, str) else None
    return match["stamp"] if match else None


def claim_stamp(tag: Any) -> str | None:
    """The UTC stamp of a canary, commit or channel claim tag (its build time), else None."""
    match = _KIND_TAG_RE.fullmatch(tag.strip()) if isinstance(tag, str) else None
    return match["stamp"] if match else None


def stamped_windows_version(tag: Any) -> str:
    """The package quad ``yy.mmdd.hh.mmss`` of a stamped claim, from its tag stamp (the same value as
    scripts/msix-shared.mjs canaryPackageVersionAt). Raises ValueError for any other tag."""
    from datetime import UTC, datetime

    stamp = claim_stamp(tag)
    if stamp is None:
        raise ValueError(f"Not a stamped claim tag: {tag!r}")
    try:
        instant = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"Invalid claim calendar timestamp: {stamp}") from error
    if instant.strftime("%Y%m%dT%H%M%SZ") != stamp:
        raise ValueError(f"Invalid claim calendar timestamp: {stamp}")
    return (f"{instant.year % 100}.{instant.month * 100 + instant.day}.{instant.hour}."
            f"{instant.minute * 100 + instant.second}")


def is_preview_tag(tag: Any) -> bool:
    """True for a commit or channel build claim: it publishes to no release channel, feed, Docker
    tag or Store (a channel build moves only its own R2 channel head, through the channel CAS)."""
    return is_commit_tag(tag) or is_channel_tag(tag)


def tag_kind(tag: Any) -> str | None:
    """``stable``, ``canary``, ``commit`` or ``channel`` for a release identity tag, else None."""
    if not isinstance(tag, str):
        return None
    value = tag.strip()
    if STABLE_TAG_RE.fullmatch(value):
        return "stable"
    match = _KIND_TAG_RE.fullmatch(value)
    return match["kind"] if match else None


def tag_core(tag: Any) -> str | None:
    """The ``X.Y.Z`` a stable, canary, commit or channel tag builds on, else None."""
    if not isinstance(tag, str):
        return None
    value = tag.strip()
    if STABLE_TAG_RE.fullmatch(value):
        return value[1:]
    match = _KIND_TAG_RE.fullmatch(value)
    return match["core"] if match else None


def release_version(tag: Any) -> str | None:
    """The version a release identity tag carries: ``X.Y.Z`` (stable), ``X.Y.Z+canary.<stamp>``,
    ``X.Y.Z+commit.<stamp>`` or ``X.Y.Z+channel.<stamp>`` (a channel claim's source version; its
    packages are versioned by the channel request).

    Never derive it by stripping a leading ``v``: prefix tags start with their kind.
    """
    if not isinstance(tag, str):
        return None
    value = tag.strip()
    if STABLE_TAG_RE.fullmatch(value):
        return value[1:]
    match = _KIND_TAG_RE.fullmatch(value)
    return f"{match['core']}+{match['kind']}.{match['stamp']}" if match else None


def canary_tag_for_date(version: str, date_utc: str) -> str:
    """Return ``canary.<full UTC timestamp>-v<stable>``.

    The stable core is unchanged; the version it publishes carries the stamp only as SemVer build
    metadata, so only the R2 channel head moves a canary subscriber.
    """
    core = version.removeprefix("v")
    if not STABLE_TAG_RE.fullmatch("v" + core):
        raise ValueError(f"invalid stable version for canary: {version!r}")
    if not re.fullmatch(r"20\d{6}T\d{6}Z", date_utc):
        raise ValueError(f"invalid canary UTC timestamp: {date_utc!r}")
    return f"canary.{date_utc}-v{core}"


def claim_tag_for_date(kind: str, version: str, date_utc: str) -> str:
    """Return ``<kind>.<full UTC timestamp>-v<version>`` for a commit or channel build claim."""
    if kind not in {"commit", "channel"}:
        raise ValueError(f"not a stamped build kind: {kind!r}")
    core = version.removeprefix("v")
    if not STABLE_TAG_RE.fullmatch("v" + core):
        raise ValueError(f"invalid version for a {kind} build: {version!r}")
    if not re.fullmatch(r"20\d{6}T\d{6}Z", date_utc):
        raise ValueError(f"invalid {kind} build UTC timestamp: {date_utc!r}")
    return f"{kind}.{date_utc}-v{core}"


def commit_tag_for_date(version: str, date_utc: str) -> str:
    """Return ``commit.<full UTC timestamp>-v<version>`` for a commit build claim."""
    return claim_tag_for_date("commit", version, date_utc)


def channel_tag_for_date(version: str, date_utc: str) -> str:
    """Return ``channel.<full UTC timestamp>-v<version>`` for a channel build claim."""
    return claim_tag_for_date("channel", version, date_utc)


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

    A source checkout of the official repository follows the latest published
    stable release. Forks, mirrors and checkout-less trees keep following main.
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
        return CHANNEL_STABLE if _tracks_official_origin(root) else CHANNEL_MAIN
    if stamp.get("source") == "channel-build":
        request = stamp.get("channelBuild")
        if not isinstance(request, dict):
            raise ValueError("Channel bundle has no baked subscription")
        return validate_name(request.get("channel"))
    return CHANNEL_CANARY if is_canary_tag(stamp.get("tag")) else CHANNEL_STABLE


def _tracks_official_origin(root: Path) -> bool:
    """True when ``root`` is a git checkout whose ``remote.origin.url`` is the official repo.

    Read from the git config file (no git process: the banner and boot paths call this).
    Forks, mirrors and local-path origins publish no official releases.
    """
    from hermes_cli.source_releases import _GITHUB_ORIGIN, OFFICIAL_REPOSITORY
    from hermes_cli.update_lock import _git_common_dir

    common = _git_common_dir(root)
    try:
        lines = (common / "config").read_text(encoding="utf-8-sig").splitlines() if common else []
    except OSError:
        return False
    in_origin = False
    for line in lines:
        text = line.strip()
        if text.startswith("["):
            in_origin = text == '[remote "origin"]'
        elif in_origin and text.partition("=")[0].strip() == "url":
            match = _GITHUB_ORIGIN.fullmatch(text.partition("=")[2].strip())
            return bool(match) and match[1].lower() == OFFICIAL_REPOSITORY.lower()
    return False


def rides_default_channel(record: dict, channel: str, project_root: Optional[Path] = None) -> bool:
    """True when no channel was ever persisted (``record``) and ``channel`` is the default.

    Such an install never consented to a release switch, so it only moves forward: a
    checkout already newer than the latest release waits for the next one. A transient
    ``--channel`` naming the default (the Desktop passes the channel its check reported)
    is still the default; only ``--set-channel`` consents to a downgrade.
    """
    return record.get("channel") is None and channel == default_channel(project_root)


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
    from hermes_cli.config_backend import write_config_key

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
    write_config_key(config_path, f"update.installs.{sha16}", new_record)
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
        except Exception as exc:
            logger.debug("installs root unavailable: %s", exc)
    return stale
