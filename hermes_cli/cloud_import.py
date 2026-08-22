"""First-boot cloud migration importer for hosted Hermes instances.

Triggered once per instance by the startup env var
``HERMES_MIGRATION_BUNDLE_URL`` (a short-lived signed URL to a migration
bundle produced by ``hermes cloud export``). Runs from the container-boot
reconciler before the gateway comes up, and only when the home is fresh:

- ``HERMES_MIGRATION_BUNDLE_URL`` is set
- no import marker exists yet
- ``state.db`` does not exist (the instance has not been used)

Behavior:

1. Stream the bundle to a temp file beside the home (same filesystem, and
   ``/opt/data`` is a real disk, unlike ``/tmp`` on some images).
2. Refuse, before touching the home, when the bundle's manifest carries a
   ``schema_version`` this build cannot import.
3. Extract through the real ``hermes_cli.backup.run_import`` path, with
   ``.env`` and ``auth.json`` dropped extra on top of the built-in
   machine-local runtime filters. A cloud instance must never absorb
   credentials from a bundle, even when it was exported with
   ``--include-secrets``.
4. On success: write the import marker, delete the local bundle copy, and
   report. On any failure: log loudly, delete the bundle copy, and let the
   boot continue (an empty instance is better than a bricked one).

``MIGRATION_SCHEMA_VERSION`` must stay in lockstep with
``hermes_cli.cloud_migrate.MIGRATION_SCHEMA_VERSION`` (the exporting side).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

from hermes_constants import get_default_hermes_root

logger = logging.getLogger(__name__)

MIGRATION_BUNDLE_URL_ENV = "HERMES_MIGRATION_BUNDLE_URL"
IMPORT_MARKER_NAME = ".cloud-migration-import.json"

# Must match hermes_cli.cloud_migrate.MIGRATION_SCHEMA_VERSION.
MIGRATION_SCHEMA_VERSION = 1

MAX_BUNDLE_BYTES = 512 * 1024 * 1024  # 512 MiB, mirrors the portal upload cap
FETCH_TIMEOUT_SECONDS = 60
FETCH_CHUNK_BYTES = 64 * 1024

# Basenames the cloud importer drops beyond the built-in import filters.
# Secrets belong to the instance's own env plumbing, never to a bundle.
CLOUD_SKIP_NAMES = frozenset({".env", "auth.json"})


def _marker_path(hermes_root: Path) -> Path:
    return hermes_root / IMPORT_MARKER_NAME


def has_import_marker(hermes_root: Path) -> bool:
    return _marker_path(hermes_root).is_file()


def home_is_fresh(hermes_root: Path) -> bool:
    """True when the home has not yet accumulated session state."""
    return not (hermes_root / "state.db").is_file()


def _bundle_url() -> Optional[str]:
    url = os.environ.get(MIGRATION_BUNDLE_URL_ENV, "").strip()
    return url or None


def _safe_hermes_root() -> Optional[Path]:
    try:
        root = get_default_hermes_root()
    except Exception:
        return None
    try:
        root.mkdir(parents=True, exist_ok=True)
        return root
    except OSError as exc:
        logger.warning("cloud_import: cannot prepare hermes root: %s", exc)
        return None


def _fetch_bundle(url: str, hermes_root: Path) -> Optional[Path]:
    """Stream the bundle to a temp file beside the home.

    Returns the staged path, or None on any transport failure (the boot
    must continue; a failed fetch is logged, not raised).
    """
    try:
        req = urllib.request.Request(url, method="GET")
        resp = urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_SECONDS)
    except (urllib.error.URLError, ValueError, OSError) as exc:
        logger.warning("cloud_import: bundle fetch failed: %s", exc)
        return None

    content_length = resp.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_BUNDLE_BYTES:
                logger.warning(
                    "cloud_import: bundle too large (%s bytes, cap %d)",
                    content_length,
                    MAX_BUNDLE_BYTES,
                )
                resp.close()
                return None
        except ValueError:
            pass

    fd, tmp_name = tempfile.mkstemp(
        prefix=".cloud-migration-bundle.", suffix=".zip", dir=str(hermes_root)
    )
    staged = Path(tmp_name)
    total = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = resp.read(FETCH_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BUNDLE_BYTES:
                    logger.warning("cloud_import: bundle exceeded cap while streaming")
                    resp.close()
                    return None
                out.write(chunk)
    except OSError as exc:
        logger.warning("cloud_import: bundle stream failed: %s", exc)
        staged.unlink(missing_ok=True)
        return None
    finally:
        resp.close()

    if total == 0:
        logger.warning("cloud_import: bundle is empty")
        staged.unlink(missing_ok=True)
        return None

    logger.info("cloud_import phase=fetch status=complete bytes=%d", total)
    return staged


def _read_manifest_schema_version(bundle: Path) -> Tuple[bool, Optional[int]]:
    """Return (has_manifest, schema_version_or_None)."""
    try:
        with zipfile.ZipFile(bundle, "r") as zf:
            raw = zf.read("migration-manifest.json")
    except (KeyError, OSError, zipfile.BadZipFile, RuntimeError):
        return False, None
    try:
        manifest = json.loads(raw.decode("utf-8"))
        return True, manifest.get("schema_version")
    except (ValueError, UnicodeDecodeError):
        return True, None


def _schema_is_compatible(bundle: Path) -> Tuple[bool, Optional[str]]:
    """Return (compatible, human_reason_when_not)."""
    if not zipfile.is_zipfile(bundle):
        return False, "not a zip archive"
    has_manifest, schema_version = _read_manifest_schema_version(bundle)
    if not has_manifest:
        # A plain ``hermes backup`` zip has no manifest. It still restores
        # through the same path; log the downgrade so support can tell the
        # two apart in the marker trail.
        return True, None
    if schema_version is None:
        return False, "migration-manifest.json is unreadable"
    if int(schema_version) > MIGRATION_SCHEMA_VERSION:
        return False, (
            f"bundle schema_version {schema_version} is newer than this "
            f"build supports ({MIGRATION_SCHEMA_VERSION})"
        )
    return True, None


def _apply_bundle(bundle: Path, hermes_root: Path) -> bool:
    from hermes_cli.backup import run_import

    try:
        run_import(
            SimpleNamespace(zipfile=str(bundle), force=True),
            extra_skip_names=CLOUD_SKIP_NAMES,
        )
    except SystemExit:
        logger.warning("cloud_import: import aborted (see output above)")
        return False
    except Exception:
        logger.exception("cloud_import: import raised unexpectedly")
        return False
    return True


def _write_marker(hermes_root: Path, bundle_manifest: Optional[dict]) -> bool:
    marker = {
        "imported_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest": bundle_manifest,
        "skipped_secret_files": sorted(CLOUD_SKIP_NAMES),
    }
    try:
        _marker_path(hermes_root).write_text(
            json.dumps(marker, indent=2), encoding="utf-8"
        )
        return True
    except OSError as exc:
        logger.warning("cloud_import: marker write failed: %s", exc)
        return False


def _boot(msg: str) -> None:
    """One-shot boot-visible line. Stdout is the cont-init channel
    (container_boot.main() prints its reconcile lines the same way);
    module-level INFO lines are invisible under the container logging
    config, so the support-visible story must not ride on logging."""
    print(f"[cloud-import] {msg}")


def maybe_run() -> bool:
    """Run the first-boot migration import when the gates open.

    Never raises: hosted boot must continue even when the migration path
    fails. Returns True when an import was attempted.
    """
    url = _bundle_url()
    if not url:
        return False

    hermes_root = _safe_hermes_root()
    if hermes_root is None:
        _boot("cannot prepare hermes root; skipping")
        return False

    if has_import_marker(hermes_root):
        _boot("marker present, skipping (already imported)")
        return False

    if not home_is_fresh(hermes_root):
        # Not fresh and no marker: a previous attempt half-imported, or the
        # instance was used before the bundle URL was injected. Importing
        # now would overwrite live state, so we do not. This state needs a
        # support look, not an automatic second pass.
        _boot(
            "home has session state but no import marker; "
            "refusing to import over a live instance"
        )
        return False

    _boot("first boot, fetching migration bundle")
    bundle = _fetch_bundle(url, hermes_root)
    if bundle is None:
        _boot("bundle fetch failed; booting without migration")
        return True  # attempted; transport failure is logged

    ok, reason = _schema_is_compatible(bundle)
    if not ok:
        _boot(f"refusing bundle: {reason}")
        bundle.unlink(missing_ok=True)
        return True

    schema_note = None
    try:
        with zipfile.ZipFile(bundle, "r") as zf:
            data = zf.read("migration-manifest.json")
            schema_note = json.loads(data.decode("utf-8"))
    except Exception:
        schema_note = None

    success = _apply_bundle(bundle, hermes_root)
    if success:
        marker_ok = _write_marker(hermes_root, schema_note)
        _boot(f"import complete; marker={'written' if marker_ok else 'write failed'}")
    else:
        _boot("import failed; no marker written (a fresh bundle can retry)")

    bundle.unlink(missing_ok=True)
    return True