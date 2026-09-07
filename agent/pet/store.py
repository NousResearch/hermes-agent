"""On-disk pet store — install / list / resolve pets.

Pets live under ``get_hermes_home()/pets/<slug>/`` (profile-scoped; NOT petdex's
``~/.codex/pets``, which its CLI owns): ``pet.json`` ({id, displayName,
description, spritesheetPath}) plus ``spritesheet.webp`` (or .png). The active
pet comes from the caller-supplied ``display.pet.slug`` (no config loader here).
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT = 60.0
_HTTP_HEADERS = {"User-Agent": "hermes-agent-petdex"}
_THUMB_FRAME_W = 192
_THUMB_FRAME_H = 208
_THUMB_W = 96  # rendered ~40px; 2x+ keeps it crisp on HiDPI


class PetStoreError(RuntimeError):
    """Raised on install/IO failures."""


@dataclass(frozen=True)
class InstalledPet:
    """A pet present on disk."""

    slug: str
    display_name: str
    description: str
    directory: Path
    spritesheet: Path
    created_by: str = ""  # "generator" for pets hatched locally; "" for petdex installs

    @property
    def exists(self) -> bool:
        return self.spritesheet.is_file()

    @property
    def generated(self) -> bool:
        return self.created_by == "generator"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def pets_dir() -> Path:
    """Return the profile-scoped pets directory (created on demand)."""
    return _ensure_dir(get_hermes_home() / "pets")


def _thumb_path(slug: str) -> Path:
    """Cached thumbnail for *slug* (lives OUTSIDE the pet dir, under ``pets/.thumbs/``)."""
    return _ensure_dir(pets_dir() / ".thumbs") / f"{slug}.png"


def _read_pet_json(directory: Path) -> dict:
    pet_json = directory / "pet.json"
    try:
        return json.loads(pet_json.read_text(encoding="utf-8")) if pet_json.is_file() else {}
    except (OSError, ValueError) as exc:
        logger.debug("unreadable pet.json in %s: %s", directory, exc)
        return {}


def _write_pet_json(directory: Path, meta: dict) -> None:
    (directory / "pet.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _resolve_spritesheet(directory: Path, meta: dict) -> Path:
    """Honor ``spritesheetPath``, else probe conventional names (incl. petdex R2's ``sprite.webp``)."""
    declared = str(meta.get("spritesheetPath", "") or "").strip()
    names = ([declared] if declared else []) + ["spritesheet.webp", "spritesheet.png", "sprite.webp", "sprite.png"]
    return next((directory / n for n in names if (directory / n).is_file()), directory / "spritesheet.webp")  # stable default


def _safe_slug(slug: str) -> str:
    """Normalize a slug to one bare path segment so ``pets_dir()/<slug>`` can never escape the pets directory."""
    segment = Path(str(slug).strip()).name
    return "" if segment in ("", ".", "..") else segment


def load_pet(slug: str) -> InstalledPet | None:
    """Return the :class:`InstalledPet` for *slug*, or ``None`` if absent."""
    if not (slug := _safe_slug(slug)) or not (directory := pets_dir() / slug).is_dir():
        return None
    meta = _read_pet_json(directory)
    name, desc, by = (str(meta.get(k, "") or d) for k, d in (("displayName", slug), ("description", ""), ("createdBy", "")))
    return InstalledPet(slug, name, desc, directory, _resolve_spritesheet(directory, meta), by)


def _usable_pet(slug: str, error: str = "") -> InstalledPet | None:
    """:func:`load_pet` result only when its spritesheet is on disk; raises :class:`PetStoreError` *error* if given."""
    if (pet := load_pet(slug)) and pet.exists:
        return pet
    if error:
        raise PetStoreError(error)
    return None


def installed_pets() -> list[InstalledPet]:
    """Return every installed pet (dirs containing a usable spritesheet)."""
    return [pet for child in sorted(pets_dir().iterdir()) if child.is_dir() and (pet := _usable_pet(child.name))]


def resolve_active_pet(configured_slug: str | None = None) -> InstalledPet | None:
    """The configured slug (``display.pet.slug``) if installed, else the first pet alphabetically."""
    if configured_slug and (pet := _usable_pet(configured_slug.strip())):
        return pet
    return next(iter(installed_pets()), None)


def install_pet(slug: str, *, force: bool = False, timeout: float = _DOWNLOAD_TIMEOUT) -> InstalledPet:
    """Download *slug* from the manifest; idempotent unless *force*. Raises :class:`PetStoreError` / ``ManifestError``."""
    from agent.pet.manifest import find_entry

    slug = _safe_slug(slug)
    if not slug:
        raise PetStoreError("invalid pet slug")
    if not force and (existing := _usable_pet(slug)):
        return existing
    entry = find_entry(slug, timeout=timeout)
    if entry is None:
        raise PetStoreError(f"pet '{slug}' is not in the petdex manifest")
    # Host-pin asset URLs so a compromised/spoofed manifest can't redirect the
    # download to an arbitrary host (matches thumbnail_png).
    if not _is_petdex_host(entry.spritesheet_url):
        raise PetStoreError(f"refusing non-petdex spritesheet host for '{slug}'")

    directory = _ensure_dir(pets_dir() / slug)
    sprite_ext = ".png" if entry.spritesheet_url.lower().split("?")[0].endswith(".png") else ".webp"
    sprite_path = directory / f"spritesheet{sprite_ext}"
    _download(entry.spritesheet_url, sprite_path, timeout=timeout)

    # Prefer the upstream pet.json; else synthesize one so the layout is self-describing.
    meta: dict = {}
    if entry.pet_json_url and _is_petdex_host(entry.pet_json_url):
        try:
            meta = data if isinstance(data := _http_get(entry.pet_json_url, timeout).json(), dict) else {}
        except Exception as exc:  # noqa: BLE001 - non-fatal, fall back below
            logger.debug("pet.json fetch failed for %s: %s", slug, exc)
    meta = meta or {"id": slug, "displayName": entry.display_name, "description": ""}
    meta["spritesheetPath"] = sprite_path.name  # key order matters: pet.json is written verbatim
    meta.setdefault("id", slug)
    meta.setdefault("displayName", entry.display_name)
    _write_pet_json(directory, meta)
    return _usable_pet(slug, f"install of '{slug}' did not produce a spritesheet")


def slugify(name: str) -> str:
    """Lowercase, hyphenate, and strip a display name into a filesystem slug."""
    return re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-") or "pet"


def unique_slug(name: str) -> str:
    """A :func:`slugify` result that doesn't collide with an existing pet dir."""
    base = slugify(name)
    candidates = (base if i == 1 else f"{base}-{i}" for i in itertools.count(1))
    return next(slug for slug in candidates if not (pets_dir() / slug).exists())


def register_local_pet(spritesheet, *, slug: str, display_name: str = "", description: str = "") -> InstalledPet:
    """Write a locally-generated pet (PIL image, WebP/PNG bytes, or path) into the store as lossless WebP.

    Appears in :func:`installed_pets` immediately (no manifest entry needed).
    """
    slug = slugify(slug)
    directory = _ensure_dir(pets_dir() / slug)
    sprite_path = directory / "spritesheet.webp"
    try:
        if isinstance(spritesheet, (bytes, bytearray)):
            sprite_path.write_bytes(bytes(spritesheet))
        else:
            from agent.pet.generate.atlas import _load_rgba

            _load_rgba(spritesheet).save(sprite_path, format="WEBP", lossless=True, quality=100, method=6, exact=True)
    except Exception as exc:  # noqa: BLE001 - normalize to one error type
        raise PetStoreError(f"could not write spritesheet for '{slug}': {exc}") from exc
    meta = {"id": slug, "displayName": display_name or slug, "description": description or "", "spritesheetPath": sprite_path.name}
    _write_pet_json(directory, {**meta, "createdBy": "generator"})
    return _usable_pet(slug, f"register of generated pet '{slug}' did not produce a spritesheet")


def export_pet(slug: str) -> tuple[str, bytes]:
    """Zip an installed pet's folder → ``(filename, bytes)``; dotfiles (thumbs, backups) skipped."""
    root = pets_dir()
    directory = root / slug.strip()
    # Traversal guard: the target must be a direct child of pets_dir.
    if directory.resolve().parent != root.resolve() or not directory.is_dir():
        raise PetStoreError(f"pet '{slug}' is not installed")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                archive.write(path, f"{directory.name}/{path.name}")
    return f"{directory.name}.zip", buf.getvalue()


def _is_petdex_host(url: str) -> bool:
    """True only for petdex.dev hosts — bounds server-side fetch (anti-SSRF)."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host == "petdex.dev" or host.endswith(".petdex.dev")


def thumbnail_png(slug: str, *, source_url: str = "", timeout: float = 30.0) -> bytes | None:
    """Small idle-frame (top-left cell) PNG for *slug*, cached on disk; ``None`` on any failure.

    Source: the installed sheet, else *source_url* only when petdex-hosted (the gateway
    never fetches arbitrary client URLs). Server-side so it rides the authenticated
    gateway as a same-origin data URL, sidestepping CSP/hotlink limits.
    """
    if not (slug := slug.strip()):
        return None
    if (cache := _thumb_path(slug)).is_file():
        with contextlib.suppress(OSError):
            return cache.read_bytes()
    sheet_bytes = None
    if pet := _usable_pet(slug):
        with contextlib.suppress(OSError):
            sheet_bytes = pet.spritesheet.read_bytes()
    if sheet_bytes is None and source_url and _is_petdex_host(source_url):
        try:
            sheet_bytes = _http_get(source_url, timeout).content
        except Exception as exc:  # noqa: BLE001 - cosmetic, degrade to placeholder
            logger.debug("thumb fetch failed for %s: %s", slug, exc)
    if not sheet_bytes:
        return None
    try:
        from PIL import Image

        with Image.open(io.BytesIO(sheet_bytes)) as im:
            frame = im.convert("RGBA").crop((0, 0, min(_THUMB_FRAME_W, im.width), min(_THUMB_FRAME_H, im.height)))
        frame = frame.resize((_THUMB_W, round(_THUMB_W * _THUMB_FRAME_H / _THUMB_FRAME_W)), Image.NEAREST)
        buf = io.BytesIO()
        frame.save(buf, format="PNG")
    except Exception as exc:  # noqa: BLE001
        logger.debug("thumb crop failed for %s: %s", slug, exc)
        return None
    with contextlib.suppress(OSError):
        cache.write_bytes(buf.getvalue())
    return buf.getvalue()


def remove_pet(slug: str) -> bool:
    """Delete an installed pet directory. Returns True if anything was removed."""
    slug = _safe_slug(slug)
    if not slug:
        return False
    # Drop the cached thumb too or a later pet reusing this slug shows the stale one.
    with contextlib.suppress(OSError):
        _thumb_path(slug).unlink(missing_ok=True)
    if not (directory := pets_dir() / slug).is_dir():
        return False
    shutil.rmtree(directory, ignore_errors=True)
    return not directory.exists()


def rename_pet(slug: str, display_name: str) -> str | None:
    """Rename a pet's ``displayName`` AND move its dir/thumb to ``slugify(name)`` when that's a free, different slug.

    Generated pets hatch under a provisional slug; naming makes that the identity. Returns the resulting slug, or ``None``.
    """
    slug = _safe_slug(slug)
    display_name = (display_name or "").strip()
    directory = pets_dir() / slug
    if not slug or not display_name or not (directory / "pet.json").is_file():
        return None
    meta = _read_pet_json(directory)
    meta = meta if isinstance(meta, dict) else {}
    meta["displayName"] = display_name
    new_slug = slug
    desired = slugify(display_name)
    if desired and desired != slug and not (pets_dir() / desired).exists():
        with contextlib.suppress(OSError):  # keep the provisional slug if the move fails
            directory.rename(pets_dir() / desired)
            with contextlib.suppress(OSError):
                _thumb_path(slug).rename(_thumb_path(desired))
            directory = pets_dir() / desired
            new_slug = meta["id"] = desired
    try:
        _write_pet_json(directory, meta)
    except OSError:
        return None
    return new_slug


def _http_get(url: str, timeout: float):
    """GET *url* with the petdex UA, following redirects; raises on HTTP errors."""
    import httpx

    resp = httpx.get(url, timeout=timeout, follow_redirects=True, headers=_HTTP_HEADERS)
    resp.raise_for_status()
    return resp


def _download(url: str, dest: Path, *, timeout: float) -> None:
    """Stream *url* to *dest* via a ``.part`` temp file so a failed download never leaves a truncated sheet."""
    import httpx

    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True, headers=_HTTP_HEADERS) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
            tmp.replace(dest)
    except Exception as exc:  # noqa: BLE001
        raise PetStoreError(f"download failed for {url}: {exc}") from exc
