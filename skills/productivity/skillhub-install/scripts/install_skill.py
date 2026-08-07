#!/usr/bin/env python3
"""
SkillHub source adapter + installer for Hermes.

This module implements SkillHub (skillhub.cn) as a first-class ``SkillSource``
and installs skills by routing the downloaded bundle through Hermes' *existing*
shared installer pipeline (quarantine -> security scan -> install policy ->
confirmation -> lockfile -> audit).

Design notes:
  * Implements the full SkillSource interface (search, fetch, inspect, source_id).
  * No hard-coded ``~/.hermes``. Install paths come from the shared installer,
    which resolves them via ``hermes_constants.get_hermes_home()`` — so profile
    isolation and the native Windows layout are respected.
  * ZIP members are validated one by one with ``_validate_bundle_rel_path``
    (the same helper the ClawHub ZIP path uses) instead of a blind
    ``extractall`` of an unreviewed archive.
  * All validated bundle assets are preserved (scripts/, references/,
    templates/, ...), not just ``scripts/``.
  * Installation goes through quarantine, the security scanner, install policy,
    confirmation, the lockfile, and the audit log — nothing is written to the
    skills directory directly.
  * Prefers the Hermes core scanner (scan_skill_cached) for parity with
    do_install; degrades gracefully to scan_skill on older cores.

Usage:
    python install_skill.py <skill-slug|skillhub-url> [--category CAT] [--yes] [--force]

Examples:
    python install_skill.py baidu-search
    python install_skill.py https://skillhub.cn/skills/baidu-search --category productivity
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote, urlparse


# ---------------------------------------------------------------------------
# Locate the hermes-agent repo so ``tools`` / ``hermes_cli`` are importable
# when this script is run standalone (inside Hermes they already are).
# ---------------------------------------------------------------------------
def _bootstrap_sys_path() -> None:
    try:
        import tools.skills_hub  # noqa: F401
        return
    except Exception:
        pass
    candidates: List[Path] = []
    here = Path(__file__).resolve()
    # skills/<category>/skillhub-install/scripts/install_skill.py -> repo root
    candidates.extend(here.parents[: min(6, len(here.parents))])
    candidates.append(Path.home() / ".hermes" / "hermes-agent")
    for root in candidates:
        if (root / "tools" / "skills_hub.py").exists():
            sys.path.insert(0, str(root))
            return


_bootstrap_sys_path()

import httpx  # noqa: E402  (available in the Hermes runtime)

from tools.skills_hub import (  # noqa: E402
    SkillBundle,
    SkillMeta,
    SkillSource,
    _validate_bundle_rel_path,
    _validate_skill_name,
)

try:  # cached scanner — parity with the shared do_install (newer cores)
    from tools.skills_guard import scan_skill_cached as _scan_skill_cached  # noqa: E402
except Exception:  # pragma: no cover - older cores without the cached scanner
    _scan_skill_cached = None


# ---------------------------------------------------------------------------
# SkillHub source adapter
# ---------------------------------------------------------------------------
class SkillHubSource(SkillSource):
    """Fetch skills from SkillHub (skillhub.cn) via its HTTP API.

    SkillHub is a community marketplace, so every skill is treated as
    ``community`` trust and runs through the same security scan as every other
    source. ``fetch`` returns a validated :class:`SkillBundle`; it never writes
    to disk — the shared installer quarantines, scans, and installs it.
    """

    BASE_URL = "https://api.skillhub.cn"
    _SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    _MAX_FILE_BYTES = 500_000

    def source_id(self) -> str:
        return "skillhub"

    def trust_level_for(self, identifier: str) -> str:
        return "community"

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def slug_of(identifier: str) -> str:
        """Extract a slug from a bare slug or a SkillHub URL."""
        ident = (identifier or "").strip()
        if ident.lower().startswith(("http://", "https://")):
            ident = urlparse(ident).path
        return ident.rstrip("/").split("/")[-1]

    def _get_json(self, url: str, timeout: int = 20) -> Optional[dict]:
        try:
            resp = httpx.get(url, timeout=timeout, follow_redirects=True)
            if resp.status_code != 200:
                print(f"  [HTTP {resp.status_code}] {url}")
                return None
            return resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            print(f"  [Request failed: {e}]")
            return None

    @staticmethod
    def _payload(data: Optional[dict]) -> Optional[dict]:
        """SkillHub wraps the skill under ``data``; tolerate a flat shape too."""
        if not isinstance(data, dict):
            return None
        nested = data.get("data")
        if isinstance(nested, dict):
            return nested
        return data

    @staticmethod
    def _normalize_tags(tags) -> List[str]:
        if isinstance(tags, list):
            return [str(t) for t in tags]
        if isinstance(tags, dict):
            return [str(k) for k in tags if str(k) != "latest"]
        return []

    # -- SkillSource interface -------------------------------------------

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        slug = self.slug_of(identifier)
        if not slug or not self._SLUG_RE.match(slug):
            return None
        payload = self._payload(self._get_json(f"{self.BASE_URL}/api/v1/skills/{slug}"))
        if not isinstance(payload, dict):
            return None
        return SkillMeta(
            name=payload.get("displayName") or payload.get("name") or slug,
            description=payload.get("summary") or payload.get("description") or "",
            source="skillhub",
            identifier=slug,
            trust_level="community",
            tags=self._normalize_tags(payload.get("tags", [])),
            extra={
                "url": f"https://skillhub.cn/skills/{slug}",
                "version": str(payload.get("version", "") or ""),
            },
        )

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        query = (query or "").strip()
        results: List[SkillMeta] = []
        seen: set[str] = set()

        # A slug-shaped query resolves directly to the skill detail.
        candidate = self.slug_of(query)
        if candidate and self._SLUG_RE.match(candidate):
            meta = self.inspect(candidate)
            if meta:
                results.append(meta)
                seen.add(meta.identifier)

        data = self._get_json(
            f"{self.BASE_URL}/api/skills?keyword={quote(query)}&pageSize={max(1, limit)}"
        )
        items = None
        if isinstance(data, dict):
            items = data.get("items") or data.get("data") or data.get("list")
        if isinstance(items, dict):
            items = items.get("items") or items.get("list")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                slug = item.get("slug") or item.get("name")
                if not slug or slug in seen:
                    continue
                seen.add(slug)
                results.append(SkillMeta(
                    name=item.get("displayName") or item.get("name") or slug,
                    description=item.get("summary") or item.get("description") or "",
                    source="skillhub",
                    identifier=slug,
                    trust_level="community",
                    tags=self._normalize_tags(item.get("tags", [])),
                ))
                if len(results) >= limit:
                    break

        return results[:limit]

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        slug = self.slug_of(identifier)
        if not slug or not self._SLUG_RE.match(slug):
            return None

        files = self._download_bundle(slug)
        if "SKILL.md" not in files:
            print(f"  [Error] SKILL.md not found in bundle for '{slug}'")
            return None

        # Normalize the frontmatter to the Hermes format, then let the shared
        # installer handle everything else. Do this only to SKILL.md; all other
        # assets are preserved verbatim.
        files["SKILL.md"] = _convert_frontmatter(files["SKILL.md"])

        # Prefer the declared frontmatter name; fall back to the slug. Validate
        # so we never hand the installer an unsafe skill name.
        name = _frontmatter_name(files["SKILL.md"]) or slug
        try:
            name = _validate_skill_name(name)
        except ValueError:
            try:
                name = _validate_skill_name(slug)
            except ValueError:
                print(f"  [Error] Invalid skill name: {name!r}")
                return None

        return SkillBundle(
            name=name,
            files=files,
            source="skillhub",
            identifier=slug,
            trust_level="community",
            metadata={"url": f"https://skillhub.cn/skills/{slug}"},
        )

    # -- download ---------------------------------------------------------

    def _download_bundle(self, slug: str) -> Dict[str, str]:
        """Download the skill ZIP and return ``{validated_rel_path: text}``.

        Every archive member is validated with ``_validate_bundle_rel_path``
        before it is accepted; unsafe or oversized members are skipped. All
        text files are preserved (not just ``scripts/``).
        """
        files: Dict[str, str] = {}
        url = f"{self.BASE_URL}/api/v1/download?slug={quote(slug)}"
        try:
            resp = httpx.get(url, timeout=60, follow_redirects=True)
        except httpx.HTTPError as exc:
            print(f"  [download failed: {exc}]")
            return files
        if resp.status_code != 200:
            print(f"  [download failed: HTTP {resp.status_code}]")
            return files

        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    try:
                        safe_name = _validate_bundle_rel_path(info.filename)
                    except ValueError:
                        print(f"  [skip unsafe path: {info.filename}]")
                        continue
                    if info.file_size > self._MAX_FILE_BYTES:
                        print(f"  [skip large file: {safe_name} ({info.file_size} bytes)]")
                        continue
                    try:
                        files[safe_name] = zf.read(info.filename).decode("utf-8")
                    except (UnicodeDecodeError, KeyError):
                        print(f"  [skip non-text file: {safe_name}]")
                        continue
        except zipfile.BadZipFile:
            print("  [download failed: invalid ZIP]")
            return {}

        return files


# ---------------------------------------------------------------------------
# Frontmatter helpers (OpenClaw -> Hermes)
# ---------------------------------------------------------------------------
_FM_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def _frontmatter_name(md_content: str) -> Optional[str]:
    m = _FM_RE.match(md_content or "")
    if not m:
        return None
    name_match = re.search(r"^name:\s*(.+)$", m.group(1), re.MULTILINE)
    if not name_match:
        return None
    # Strip any surrounding quotes the converter may have added for YAML safety.
    return name_match.group(1).strip().strip('"').strip("'")


def _title_tag(tag: str) -> str:
    """Title-Case a tag while preserving intra-word casing (skillHub -> SkillHub)."""
    return "-".join(seg[:1].upper() + seg[1:] if seg else seg for seg in tag.split("-"))


def _extract_tags(old_fm: str) -> List[str]:
    """Collect tags from an existing frontmatter block.

    Handles a top-level ``tags: [...]`` list, a ``metadata.hermes.tags`` list
    (both match the same inline-list pattern), and OpenClaw ``"bins"`` (which
    become ``requires-<bin>`` tags). Results are de-duplicated and Title-Cased.
    """
    raw: List[str] = []
    tag_match = re.search(r"tags:\s*\[(.*?)\]", old_fm, re.DOTALL)
    if tag_match:
        raw.extend(v.strip().strip('"').strip("'") for v in tag_match.group(1).split(",") if v.strip())
    bins_match = re.search(r'"bins"\s*:\s*\[(.*?)\]', old_fm, re.DOTALL)
    if bins_match:
        bins = [v.strip().strip('"').strip("'") for v in bins_match.group(1).split(",") if v.strip()]
        raw.extend(f"requires-{b.lower()}" for b in bins[:2])
    out: List[str] = []
    seen: set[str] = set()
    for tag in raw:
        tc = _title_tag(tag)
        if tc and tc.lower() not in seen:
            seen.add(tc.lower())
            out.append(tc)
    return out


def _yaml_scalar(value: str) -> str:
    """Return a YAML-safe representation of a plain string scalar.

    Downloaded frontmatter is untrusted: a value containing a colon, ``#``, a
    flow indicator, or leading/trailing whitespace would break the rebuilt
    block if emitted bare. Such values are double-quoted (with backslashes and
    quotes escaped); clean values are emitted unquoted to keep the block tidy.
    """
    if (value == ""
            or value != value.strip()
            or value[:1] in "!&*[]{}#|>@`\"'%,-?:"
            or ": " in value
            or value.endswith(":")
            or " #" in value):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def _convert_frontmatter(md_content: str) -> str:
    """Rewrite a downloaded SKILL.md frontmatter into the native Hermes format.

    Emits a single valid frontmatter block with tags under
    ``metadata.hermes.tags`` (Title-Cased). Provided ``name``/``description``
    and any existing ``version``/``license``/``author``/``platforms``/
    ``prerequisites`` are preserved; ``version``/``license``/``platforms`` are
    defaulted only when absent, and ``author`` is never fabricated. Scalar
    values are quoted when needed so untrusted input can't produce invalid
    YAML. If no frontmatter is present the content is returned unchanged.
    """
    m = _FM_RE.match(md_content or "")
    if not m:
        return md_content

    old_fm = m.group(1)

    def _field(key: str) -> Optional[str]:
        fm = re.search(rf"^{key}:\s*(.+)$", old_fm, re.MULTILINE)
        return fm.group(1).strip() if fm else None

    name = _field("name") or "unknown"
    desc = re.sub(r"OpenClaw", "Hermes", _field("description") or "")
    version = _field("version") or "1.0.0"
    license_ = _field("license") or "MIT"
    author = _field("author")  # never fabricated
    # ``platforms`` is a native field; preserve the source inline list or
    # default so downloaded skills match the shipped-skill convention.
    platforms = _field("platforms") or "[linux, macos, windows]"
    tags = _extract_tags(old_fm)

    # Preserve a nested ``prerequisites:`` block verbatim if the source has one
    # (its indented child lines run until the next top-level key).
    prereq_match = re.search(r"^prerequisites:.*(?:\n[ \t]+.*)*", old_fm, re.MULTILINE)

    lines = [
        "---",
        f"name: {_yaml_scalar(name)}",
        f"description: {_yaml_scalar(desc)}",
        f"version: {_yaml_scalar(version)}",
    ]
    if author:
        lines.append(f"author: {_yaml_scalar(author)}")
    lines.append(f"license: {_yaml_scalar(license_)}")
    lines.append(f"platforms: {platforms}")
    if prereq_match:
        lines.append(prereq_match.group(0).rstrip())
    lines.append("metadata:")
    lines.append("  hermes:")
    lines.append(f"    tags: [{', '.join(tags)}]")
    lines.append("---")
    new_fm = "\n".join(lines)

    body = md_content[m.end():]
    body = re.sub(r"OpenClaw", "Hermes", body)
    body = re.sub(r"openclaw skills install", "skill_manage", body)
    return new_fm + body


def _scan_quarantine(q_path, bundle):
    """Scan a quarantined bundle.

    Prefers ``scan_skill_cached`` for parity with the shared ``do_install``
    (same content-hash scan cache); degrades to ``scan_skill`` on older cores
    that do not ship the cached entry point.
    """
    if _scan_skill_cached is not None:
        try:
            from tools.skills_hub import HUB_DIR
            cache_dir = HUB_DIR / "scan-cache"
        except Exception:
            cache_dir = None
        result, _provenance = _scan_skill_cached(
            q_path, source=bundle.identifier, cache_dir=cache_dir
        )
        return result
    from tools.skills_guard import scan_skill
    return scan_skill(q_path, source=bundle.identifier)


# ---------------------------------------------------------------------------
# Install: route the bundle through the shared installer pipeline
# ---------------------------------------------------------------------------
def _install_via_do_install(slug: str, category: str, force: bool, skip_confirm: bool) -> bool:
    """Register SkillHub in the router and delegate to the shared do_install.

    Returns True if do_install ran, False if it could not be used (caller then
    falls back to the explicit shared-pipeline path).
    """
    try:
        import tools.skills_hub as hub
        from hermes_cli.skills_hub import do_install
    except Exception:
        return False

    original = hub.create_source_router

    def _router_with_skillhub(auth=None):
        sources = original(auth)
        if not any(getattr(s, "source_id", lambda: "")() == "skillhub" for s in sources):
            # Insert next to the other community marketplaces.
            sources.append(SkillHubSource())
        return sources

    hub.create_source_router = _router_with_skillhub
    try:
        do_install(slug, category=category, force=force, skip_confirm=skip_confirm)
        return True
    finally:
        hub.create_source_router = original


def _install_direct(slug: str, category: str, force: bool, skip_confirm: bool) -> int:
    """Fallback: drive the shared installer building blocks directly.

    Uses exactly the same functions do_install uses, so profile-aware paths,
    quarantine, scanning, install policy, lockfile, and audit all apply.
    """
    import shutil

    from tools.skills_hub import (
        HubLockFile,
        append_audit_log,
        ensure_hub_dirs,
        install_from_quarantine,
        quarantine_bundle,
    )
    from tools.skills_guard import should_allow_install, format_scan_report

    ensure_hub_dirs()
    src = SkillHubSource()

    print(f"Fetching: {slug}")
    bundle = src.fetch(slug)
    if not bundle:
        print(f"Error: could not fetch '{slug}' from SkillHub.")
        return 1

    lock = HubLockFile()
    if lock.get_installed(bundle.name) and not force:
        print(f"Warning: '{bundle.name}' is already installed. Use --force to reinstall.")
        return 0

    # Quarantine (validates skill name + every member path again).
    try:
        q_path = quarantine_bundle(bundle)
    except ValueError as exc:
        append_audit_log("BLOCKED", bundle.name, bundle.source,
                         bundle.trust_level, "invalid_path", str(exc))
        print(f"Installation blocked: {exc}")
        return 1

    print("Running security scan...")
    result = _scan_quarantine(q_path, bundle)
    print(format_scan_report(result))

    allowed, reason = should_allow_install(result, force=force)
    if not allowed:
        shutil.rmtree(q_path, ignore_errors=True)
        append_audit_log("BLOCKED", bundle.name, bundle.source,
                         bundle.trust_level, result.verdict,
                         f"{len(result.findings)}_findings")
        print(f"Installation blocked: {reason}")
        return 1

    if not force and not skip_confirm:
        print(
            "\nYou are installing a third-party skill at your own risk. "
            "Review the files before use."
        )
        try:
            answer = input(f"Install '{bundle.name}'? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer not in {"y", "yes"}:
            shutil.rmtree(q_path, ignore_errors=True)
            print("Installation cancelled.")
            return 0

    try:
        install_dir = install_from_quarantine(q_path, bundle.name, category, bundle, result)
    except ValueError as exc:
        shutil.rmtree(q_path, ignore_errors=True)
        append_audit_log("BLOCKED", bundle.name, bundle.source,
                         bundle.trust_level, "invalid_path", str(exc))
        print(f"Installation blocked: {exc}")
        return 1

    print(f"Installed: {install_dir}")
    print(f"Files: {', '.join(bundle.files.keys())}")
    return 0


def install(slug: str, category: str = "", force: bool = False, skip_confirm: bool = False) -> int:
    """Install a SkillHub skill through Hermes' shared installer."""
    if _install_via_do_install(slug, category, force, skip_confirm):
        return 0
    return _install_direct(slug, category, force, skip_confirm)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install a skill from SkillHub (skillhub.cn) into Hermes."
    )
    parser.add_argument("skill", help="Skill slug or SkillHub URL")
    parser.add_argument("--category", default="", help="Optional category bucket")
    parser.add_argument("--force", action="store_true", help="Reinstall if already present")
    parser.add_argument("--yes", "-y", dest="yes", action="store_true",
                        help="Skip the confirmation prompt (non-interactive)")
    args = parser.parse_args(argv)

    slug = SkillHubSource.slug_of(args.skill)
    if not slug or not SkillHubSource._SLUG_RE.match(slug):
        print(f"Error: could not parse a skill slug from '{args.skill}'")
        return 1

    return install(slug, category=args.category, force=args.force, skip_confirm=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
