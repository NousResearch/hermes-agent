"""On-disk pet store — install / list / resolve pets.

Pets live under ``get_hermes_home()/pets/<slug>/`` so every profile gets its
own set (we deliberately do **not** reuse petdex's ``~/.codex/pets`` default —
that's owned by the petdex npm CLI and isn't profile-aware).  Each installed
pet directory holds:

    pets/<slug>/
        pet.json            # {id, displayName, description, spritesheetPath}
        spritesheet.webp    # (or .png)

The active pet is resolved from the caller-supplied ``display.pet.slug`` config
value (falling back to the first installed pet), so this module stays free of
the config loader.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT = 60.0


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
    sprite_version_number: int = 1

    @property
    def exists(self) -> bool:
        return self.spritesheet.is_file()

    @property
    def generated(self) -> bool:
        return self.created_by == "generator"


def pets_dir() -> Path:
    """Return the profile-scoped pets directory (created on demand)."""
    path = get_hermes_home() / "pets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_pet_json(directory: Path) -> dict:
    pet_json = directory / "pet.json"
    if not pet_json.is_file():
        return {}
    try:
        return json.loads(pet_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("unreadable pet.json in %s: %s", directory, exc)
        return {}


def _sprite_version_number(meta: dict) -> int:
    """Return a positive manifest sprite version, defaulting malformed values to v1."""
    raw = meta.get("spriteVersionNumber", 1)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        return 1
    return raw


def _resolve_spritesheet(directory: Path, meta: dict) -> Path:
    """Find the spritesheet for a pet dir.

    Honors ``spritesheetPath`` from pet.json, else probes the conventional
    filenames (``spritesheet.{webp,png}`` and petdex R2's ``sprite.webp``).
    """
    declared = str(meta.get("spritesheetPath", "") or "").strip()
    if declared:
        candidate = directory / declared
        if candidate.is_file():
            return candidate
    for name in ("spritesheet.webp", "spritesheet.png", "sprite.webp", "sprite.png"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    # Default expectation even if missing, so callers get a stable path.
    return directory / "spritesheet.webp"


def _safe_slug(slug: str) -> str:
    """Normalize a slug to a single bare path segment.

    Pet slugs index into ``pets_dir()/<slug>/`` for load/remove, so a value
    carrying path separators (``../``, absolute paths) could escape the pets
    directory. Strip every separator and reject ``.``/``..`` so callers can
    only ever name a direct child of the pets directory.
    """
    segment = Path(str(slug).strip()).name
    if segment in ("", ".", ".."):
        return ""
    return segment


def load_pet(slug: str) -> InstalledPet | None:
    """Return the :class:`InstalledPet` for *slug*, or ``None`` if absent."""
    slug = _safe_slug(slug)
    if not slug:
        return None
    directory = pets_dir() / slug
    if not directory.is_dir():
        return None
    meta = _read_pet_json(directory)
    return InstalledPet(
        slug=slug,
        display_name=str(meta.get("displayName", "") or slug),
        description=str(meta.get("description", "") or ""),
        directory=directory,
        spritesheet=_resolve_spritesheet(directory, meta),
        created_by=str(meta.get("createdBy", "") or ""),
        sprite_version_number=_sprite_version_number(meta),
    )


def installed_pets() -> list[InstalledPet]:
    """Return every installed pet (dirs containing a usable spritesheet)."""
    out: list[InstalledPet] = []
    for child in sorted(pets_dir().iterdir()):
        if not child.is_dir():
            continue
        pet = load_pet(child.name)
        if pet and pet.exists:
            out.append(pet)
    return out


def resolve_active_pet(configured_slug: str | None = None) -> InstalledPet | None:
    """Resolve which pet to display.

    Precedence: the configured slug (``display.pet.slug``) if it's installed,
    otherwise the first installed pet alphabetically, otherwise ``None``.
    """
    if configured_slug:
        pet = load_pet(configured_slug.strip())
        if pet and pet.exists:
            return pet
    pets = installed_pets()
    return pets[0] if pets else None


def install_pet(slug: str, *, force: bool = False, timeout: float = _DOWNLOAD_TIMEOUT) -> InstalledPet:
    """Download *slug* from the manifest into the pets directory.

    Idempotent: a fully-installed pet is returned as-is unless *force*.  Raises
    :class:`PetStoreError` / :class:`~agent.pet.manifest.ManifestError` on
    failure.
    """
    from agent.pet.manifest import find_entry

    slug = _safe_slug(slug)
    if not slug:
        raise PetStoreError("invalid pet slug")
    existing = load_pet(slug)
    if existing and existing.exists and not force:
        return existing

    entry = find_entry(slug, timeout=timeout)
    if entry is None:
        raise PetStoreError(f"pet '{slug}' is not in the petdex manifest")

    # Host-pin every asset URL to petdex. The manifest is trusted (HTTPS from
    # petdex.dev), but pin the asset hosts too so a compromised/spoofed manifest
    # can't redirect the download at an arbitrary host. Matches thumbnail_png.
    if not _is_petdex_host(entry.spritesheet_url):
        raise PetStoreError(f"refusing non-petdex spritesheet host for '{slug}'")

    directory = pets_dir() / slug
    directory.mkdir(parents=True, exist_ok=True)

    sprite_ext = ".png" if entry.spritesheet_url.lower().split("?")[0].endswith(".png") else ".webp"
    sprite_path = directory / f"spritesheet{sprite_ext}"

    _download(entry.spritesheet_url, sprite_path, timeout=timeout)

    # Fetch the upstream pet.json if present; otherwise synthesize a minimal
    # one so the local layout is self-describing.
    meta: dict = {}
    if entry.pet_json_url and _is_petdex_host(entry.pet_json_url):
        try:
            meta = _download_json(entry.pet_json_url, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - non-fatal, fall back below
            logger.debug("pet.json fetch failed for %s: %s", slug, exc)
    if not isinstance(meta, dict) or not meta:
        meta = {"id": slug, "displayName": entry.display_name, "description": ""}
    meta["spritesheetPath"] = sprite_path.name
    meta.setdefault("id", slug)
    meta.setdefault("displayName", entry.display_name)
    (directory / "pet.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pet = load_pet(slug)
    if pet is None or not pet.exists:
        raise PetStoreError(f"install of '{slug}' did not produce a spritesheet")
    return pet


def slugify(name: str) -> str:
    """Lowercase, hyphenate, and strip a display name into a filesystem slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug or "pet"


def unique_slug(name: str) -> str:
    """A :func:`slugify` result that doesn't collide with an existing pet dir."""
    base = slugify(name)
    slug = base
    counter = 2
    while (pets_dir() / slug).exists():
        slug = f"{base}-{counter}"
        counter += 1
    return slug


def _write_spritesheet(source, dest: Path) -> None:
    """Write *source* (PIL image, bytes, or path) as a lossless WebP at *dest*."""
    if isinstance(source, (bytes, bytearray)):
        dest.write_bytes(bytes(source))
        return

    from PIL import Image

    if isinstance(source, (str, Path)):
        with Image.open(source) as opened:
            image = opened.convert("RGBA")
    else:
        image = source.convert("RGBA")
    image.save(dest, format="WEBP", lossless=True, quality=100, method=6, exact=True)


def register_local_pet(
    spritesheet,
    *,
    slug: str,
    display_name: str = "",
    description: str = "",
) -> InstalledPet:
    """Write a locally-generated pet into the store and return it.

    *spritesheet* may be a PIL image, raw WebP/PNG bytes, or a path. The pet
    appears in :func:`installed_pets` immediately, and because :func:`install_pet`
    returns an already-on-disk pet before consulting the manifest, it can be
    adopted (``pet.select`` / ``/pet <slug>``) without a manifest entry.
    """
    slug = slugify(slug)
    directory = pets_dir() / slug
    directory.mkdir(parents=True, exist_ok=True)
    sprite_path = directory / "spritesheet.webp"
    try:
        _write_spritesheet(spritesheet, sprite_path)
    except Exception as exc:  # noqa: BLE001 - normalize to one error type
        raise PetStoreError(f"could not write spritesheet for '{slug}': {exc}") from exc

    meta = {
        "id": slug,
        "displayName": display_name or slug,
        "description": description or "",
        "spritesheetPath": sprite_path.name,
        "createdBy": "generator",
    }
    (directory / "pet.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pet = load_pet(slug)
    if pet is None or not pet.exists:
        raise PetStoreError(f"register of generated pet '{slug}' did not produce a spritesheet")
    return pet


def export_pet(slug: str) -> tuple[str, bytes]:
    """Zip an installed pet's folder (pet.json + spritesheet) → (filename, bytes).

    Dotfiles (cached thumbs, backups) are skipped so the archive is a clean,
    re-importable pet package. Raises :class:`PetStoreError` if not installed.
    """
    import io
    import zipfile

    root = pets_dir()
    directory = root / slug.strip()
    # Guard against traversal: the target must be a direct child of pets_dir.
    if directory.resolve().parent != root.resolve() or not directory.is_dir():
        raise PetStoreError(f"pet '{slug}' is not installed")

    name = directory.name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                archive.write(path, f"{name}/{path.name}")
    return f"{name}.zip", buf.getvalue()


_THUMB_FRAME_W = 192
_THUMB_FRAME_H = 208
_THUMB_W = 96  # rendered ~40px; 2x+ keeps it crisp on HiDPI


def _thumbs_dir() -> Path:
    path = pets_dir() / ".thumbs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_petdex_host(url: str) -> bool:
    """True only for petdex.dev hosts — bounds server-side fetch (anti-SSRF)."""
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host == "petdex.dev" or host.endswith(".petdex.dev")


def thumbnail_png(slug: str, *, source_url: str = "", timeout: float = 30.0) -> bytes | None:
    """Return a small idle-frame PNG for *slug*, cached on disk.

    Crops the top-left (idle, frame 0) cell of the spritesheet and downsamples
    it to a thumbnail. Source preference: an installed spritesheet on disk, else
    *source_url* — but only when it points at petdex (so the gateway never
    fetches an arbitrary client-supplied URL). Returns ``None`` when there's no
    usable source or Pillow/network fails; callers render a placeholder.

    Doing this server-side sidesteps the renderer's CSP / R2 hotlink limits that
    break a direct ``<img src=cdn>`` and lets the result ride the authenticated
    gateway as a same-origin data URL.
    """
    slug = slug.strip()
    if not slug:
        return None

    cache = _thumbs_dir() / f"{slug}.png"
    if cache.is_file():
        try:
            return cache.read_bytes()
        except OSError:
            pass

    sheet_bytes: bytes | None = None
    pet = load_pet(slug)
    if pet and pet.exists:
        try:
            sheet_bytes = pet.spritesheet.read_bytes()
        except OSError:
            sheet_bytes = None

    if sheet_bytes is None and source_url and _is_petdex_host(source_url):
        try:
            import httpx

            resp = httpx.get(
                source_url,
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": "hermes-agent-petdex"},
            )
            resp.raise_for_status()
            sheet_bytes = resp.content
        except Exception as exc:  # noqa: BLE001 - cosmetic, degrade to placeholder
            logger.debug("thumb fetch failed for %s: %s", slug, exc)

    if not sheet_bytes:
        return None

    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(sheet_bytes)) as im:
            frame = im.convert("RGBA").crop(
                (0, 0, min(_THUMB_FRAME_W, im.width), min(_THUMB_FRAME_H, im.height))
            )
            height = round(_THUMB_W * _THUMB_FRAME_H / _THUMB_FRAME_W)
            frame = frame.resize((_THUMB_W, height), Image.NEAREST)
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            data = buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.debug("thumb crop failed for %s: %s", slug, exc)
        return None

    try:
        cache.write_bytes(data)
    except OSError:
        pass
    return data


def remove_pet(slug: str) -> bool:
    """Delete an installed pet directory.  Returns True if anything was removed."""
    import shutil

    slug = _safe_slug(slug)
    if not slug:
        return False

    # The cached thumbnail lives in pets/.thumbs/<slug>.png — OUTSIDE the pet
    # dir, so rmtree won't catch it. Drop it too, or a later pet that reuses this
    # slug renders this one's stale thumbnail.
    try:
        (_thumbs_dir() / f"{slug}.png").unlink(missing_ok=True)
    except OSError:
        pass

    directory = pets_dir() / slug
    if not directory.is_dir():
        return False
    shutil.rmtree(directory, ignore_errors=True)
    return not directory.exists()


def rename_pet(slug: str, display_name: str) -> str | None:
    """Rename a pet's ``displayName`` AND realign its slug/dir to match.

    Generated pets are hatched under a provisional, prompt-derived slug; when
    the user names the pet on the reveal screen we make that name the real
    identity so lists/subtitles show what they typed, not the prompt. The dir is
    renamed to ``slugify(name)`` (and the cached thumbnail moved alongside it)
    whenever that yields a free, different slug — otherwise the slug is left as
    is. Returns the resulting slug on success, or ``None`` on failure.
    """
    slug = _safe_slug(slug)
    display_name = (display_name or "").strip()
    if not slug or not display_name:
        return None
    directory = pets_dir() / slug
    pet_json = directory / "pet.json"
    if not pet_json.is_file():
        return None
    try:
        meta = json.loads(pet_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    meta["displayName"] = display_name

    new_slug = slug
    desired = slugify(display_name)
    if desired and desired != slug and not (pets_dir() / desired).exists():
        try:
            directory.rename(pets_dir() / desired)
            try:
                (_thumbs_dir() / f"{slug}.png").rename(_thumbs_dir() / f"{desired}.png")
            except OSError:
                pass
            directory = pets_dir() / desired
            pet_json = directory / "pet.json"
            new_slug = desired
            meta["id"] = new_slug
        except OSError:
            new_slug = slug  # keep the provisional slug if the move fails

    try:
        pet_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError:
        return None
    return new_slug


def _download(url: str, dest: Path, *, timeout: float) -> None:
    import httpx

    try:
        with httpx.stream(
            "GET",
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "hermes-agent-petdex"},
        ) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
            tmp.replace(dest)
    except Exception as exc:  # noqa: BLE001
        raise PetStoreError(f"download failed for {url}: {exc}") from exc


def _download_json(url: str, *, timeout: float) -> dict:
    import httpx

    resp = httpx.get(
        url,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "hermes-agent-petdex"},
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}
