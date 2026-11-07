"""One dependency-readiness contract for the WhatsApp setup flows and gateway."""

import hashlib
import json
import secrets
import threading
from contextlib import contextmanager
from pathlib import Path

_install_lock_holder = threading.local()


def whatsapp_activation_signature(home: Path, session_path: Path) -> tuple:
    """Identify saved policy and completed pairing without tracking mutable Baileys credentials."""
    from gateway.platforms._shared import get_scoped_secret

    signature = []
    for path in (Path(home) / ".env", Path(session_path) / ".hermes-pairing-generation"):
        try:
            signature.append(hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError:
            signature.append(None)
    for key in ("WHATSAPP_MODE", "WHATSAPP_DM_POLICY", "WHATSAPP_ALLOWED_USERS",
                "WHATSAPP_ALLOW_FROM", "WHATSAPP_GROUP_POLICY", "WHATSAPP_GROUP_ALLOWED_USERS",
                "WHATSAPP_GROUP_ALLOW_FROM"):
        signature.append(get_scoped_secret(key, ""))
    return tuple(signature)


def stamp_whatsapp_pairing(session_path: Path) -> None:
    """Advance identity only after a new QR pairing completed successfully."""
    (Path(session_path) / ".hermes-pairing-generation").write_text(
        secrets.token_hex(16), encoding="utf-8")


@contextmanager
def bridge_dependency_install_lock(bridge_dir: Path):
    """Serialize npm installs across gateway, CLI, dashboard, and timed-out workers."""
    from hermes_cli.auth import _file_lock

    with _file_lock(Path(bridge_dir) / ".hermes-npm-install.lock", _install_lock_holder,
                    360.0, "Timed out waiting for WhatsApp bridge dependency install"):
        yield


def _package_hash(bridge_dir: Path) -> str:
    try:
        return hashlib.sha256((bridge_dir / "package.json").read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def bridge_dependencies_ready(bridge_dir: Path) -> bool:
    """Recognize stamped installs and complete legacy installs made before setup stamped them."""
    package_hash = _package_hash(bridge_dir)
    if not package_hash:
        return False
    try:
        stamped = (bridge_dir / "node_modules" / ".hermes-pkg-hash").read_text(
            encoding="utf-8-sig").strip()
    except OSError:
        stamped = None
    if stamped is not None:
        return stamped == package_hash
    if _legacy_bridge_dependencies_ready(bridge_dir):
        stamp_bridge_dependencies(bridge_dir)
        return True
    return False


def _legacy_bridge_dependencies_ready(bridge_dir: Path) -> bool:
    """A prior npm install is usable when its lock matches every required pinned package."""
    try:
        manifest = json.loads((bridge_dir / "package.json").read_text(encoding="utf-8-sig"))
        expected = json.loads((bridge_dir / "package-lock.json").read_text(encoding="utf-8-sig"))
        installed = json.loads((bridge_dir / "node_modules" / ".package-lock.json").read_text(
            encoding="utf-8-sig"))
        packages = expected["packages"]
        installed_packages = installed["packages"]
        if expected.get("lockfileVersion") != installed.get("lockfileVersion"):
            return False
        if packages[""].get("dependencies") != manifest.get("dependencies"):
            return False
        for name, record in packages.items():
            if not name:
                continue
            actual = installed_packages.get(name)
            if actual is None:
                if record.get("optional"):
                    continue
                return False
            if (record.get("version") != actual.get("version")
                    or record.get("integrity") != actual.get("integrity")
                    or json.loads((bridge_dir / name / "package.json").read_text(
                        encoding="utf-8-sig")).get("version") != record.get("version")):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def stamp_bridge_dependencies(bridge_dir: Path) -> None:
    """Record a completed npm install; a failed stamp only costs a later reinstall."""
    package_hash = _package_hash(bridge_dir)
    if package_hash:
        try:
            (bridge_dir / "node_modules" / ".hermes-pkg-hash").write_text(
                package_hash, encoding="utf-8")
        except OSError:
            pass
