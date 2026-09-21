"""Blog illustrator — art-directed image set with a provider-free P11 seam.

P11 persists an immutable visual plan and a planned provenance manifest before
any legacy image generator runs. It does not change the current provider,
provider configuration or publishing path; P10 owns that future replacement.
"""
from __future__ import annotations


import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import config
from blog import art_director
from blog.art_director import build_art_brief, fallback_brief, compose_prompt
from blog.asset_manifest import AssetManifestError, save_asset_manifest
from blog.reference_catalog import CatalogIntegrityError
from blog.visual_plan import VisualPlanError, save_visual_plan


# ── Rotation state (persisted so variety survives cron restarts) ──
ROTATION_STATE_PATH = Path(__file__).parent.parent / "blog_topics" / "skill_rotation.json"
ROTATION_HISTORY_LIMIT = 6


def _load_rotation_state() -> dict:
    try:
        data = json.loads(ROTATION_STATE_PATH.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {"history": data}


def _load_recent_styles() -> list[str]:
    history = _load_rotation_state().get("history", [])
    if not isinstance(history, list):
        return []
    valid = art_director.STYLE_IDS
    return [x for x in history if isinstance(x, str) and x in valid][-ROTATION_HISTORY_LIMIT:]


def _load_recent_concept_fingerprints() -> list[str]:
    values = _load_rotation_state().get("concept_fingerprints", [])
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, str) and value.strip()][-ROTATION_HISTORY_LIMIT:]


def _record_selection(style_id: str, fingerprint: object = None) -> None:
    history = _load_recent_styles()
    history.append(style_id)
    fingerprints = _load_recent_concept_fingerprints()
    if isinstance(fingerprint, (list, tuple)):
        key = "|".join(str(term).strip().lower() for term in fingerprint if str(term).strip())
        if key:
            fingerprints.append(key)
    ROTATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"history": history[-ROTATION_HISTORY_LIMIT:],
               "concept_fingerprints": fingerprints[-ROTATION_HISTORY_LIMIT:],
               "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    ROTATION_STATE_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def _record_style(style_id: str) -> None:
    """Backward-compatible style-only recorder for callers outside illustrate()."""
    _record_selection(style_id)


# ── Codex CLI generation ────────────────────────────────────────
# Substrings that mean the ChatGPT/Codex image quota is exhausted (both the raw
# HTTP 429 form and the friendly weekly-cap CLI message). A capped run is NOT a
# per-image failure — the whole batch should defer, not count attempts.
_CODEX_CAP_SIGNALS = (
    "usage_limit_reached",
    "hit your usage limit",
    "usage limit has been reached",
    "resets_in_seconds",
)


class CodexCapExceeded(Exception):
    """Raised (when raise_on_cap=True) when Codex reports its usage cap."""


def _output_shows_cap(text: str) -> bool:
    low = (text or "").lower()
    return any(sig in low for sig in _CODEX_CAP_SIGNALS)


_CODEX_AUTH_FAILURE_SIGNALS = (
    "refresh token was already used",
    "could not be refreshed",
    "log out and sign in",
    "401 unauthorized",
)


def _output_shows_auth_failure(text: Optional[str]) -> bool:
    """True when Codex output shows the grant itself is dead (not quota).

    These runs burn an attempt on a pool entry that can never succeed until
    re-login, so the entry is marked exhausted (401 cooldown) to stop future
    runs retrying it and to surface the re-auth need in pool state.
    """
    low = (text or "").lower()
    return any(sig in low for sig in _CODEX_AUTH_FAILURE_SIGNALS)


def _load_codex_pool():
    """Return Hermes' native Codex credential pool."""
    from agent.credential_pool import load_pool

    return load_pool("openai-codex")


def _account_id_from_id_token(id_token: Optional[str]) -> Optional[str]:
    """ChatGPT account id from an id_token's OpenAI auth claim, if present.

    The Codex CLI requires an explicit ``account_id`` in auth.json and sends
    it as the ChatGPT-Account-Id header; without it the websocket handshake
    fails 401 and the CLI misreports the session as superseded. Pool entries
    and refresh responses do not reliably carry the field, but the id_token
    we already have does (same claim shape used for quota probes).
    """
    if not id_token:
        return None
    try:
        from hermes_cli.auth_constants import _decode_jwt_claims

        claims = _decode_jwt_claims(id_token) or {}
    except Exception:
        return None
    auth_claims = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        value = auth_claims.get("chatgpt_account_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _codex_auth_payload(entry, *, refresher=None) -> dict:
    """Build a Codex CLI auth payload for an isolated CODEX_HOME.

    Recent Codex CLI versions refuse an auth file that lacks ``id_token`` at
    parse time ("missing field id_token"). Pool entries only carry
    access/refresh, and shared-account grants are single-use rotated by the
    Codex CLI (or another Hermes process) — so the stored copy is often
    already stale even when it looks fresh. We therefore always route the
    token set through ``refresher`` when one is provided: success rotates and
    persists the grant; a relogin-required rejection is self-healed by the
    shared refresh path adopting the canonical ~/.codex/auth.json token
    before surfacing a hard 401. The refresher keeps a short-lived local
    cache so hero + section images share one rotation per run. On refresher
    failure or absence we degrade to the entry's stored tokens.
    """
    tokens = {
        "access_token": entry.access_token,
        "refresh_token": entry.refresh_token,
    }
    for key in ("account_id", "id_token"):
        value = getattr(entry, key, None)
        if value:
            tokens[key] = value
    if refresher is not None:
        try:
            fresh = refresher(entry)
        except Exception:
            fresh = None
        if fresh:
            for key in ("access_token", "refresh_token", "id_token", "account_id"):
                value = fresh.get(key)
                if value:
                    tokens[key] = value
    if not tokens.get("account_id"):
        derived = _account_id_from_id_token(tokens.get("id_token"))
        if derived:
            tokens["account_id"] = derived
    # Codex CLI refreshes (and burns the single-use refresh token) whenever
    # the payload lacks ``last_refresh`` — even when the access token is
    # long-lived and valid. Emitting the entry's last_refresh keeps the CLI
    # on the stored access token instead of a doomed rotation.
    last_refresh = getattr(entry, "last_refresh", None)
    if last_refresh:
        return {"auth_mode": "chatgpt", "last_refresh": last_refresh,
                "tokens": tokens}
    return {"tokens": tokens}


# One rotation per entry per short window: a backlog run renders hero + inline
# sections in the same process and each image would otherwise rotate again.
_CODEX_IMAGE_TOKEN_CACHE: dict[str, tuple[float, dict]] = {}
_CODEX_IMAGE_TOKEN_TTL_SECONDS = 600


def _refresh_entry_tokens(entry) -> Optional[dict]:
    """Refresh one pool entry's tokens; rotation persists via hermes auth."""
    key = str(getattr(entry, "id", "") or entry.refresh_token)
    now = time.time()
    cached = _CODEX_IMAGE_TOKEN_CACHE.get(key)
    if cached and now - cached[0] < _CODEX_IMAGE_TOKEN_TTL_SECONDS:
        return cached[1]
    try:
        from hermes_cli.auth_codex import _refresh_codex_auth_tokens

        fresh = _refresh_codex_auth_tokens(
            {
                "access_token": str(entry.access_token or ""),
                "refresh_token": str(entry.refresh_token or ""),
            },
            timeout_seconds=30.0,
        )
    except Exception as exc:
        print(f"[blog_illustrator] codex token refresh failed for {key}: {exc}")
        return None
    if isinstance(fresh, dict) and fresh.get("access_token"):
        _CODEX_IMAGE_TOKEN_CACHE[key] = (now, fresh)
        return fresh
    return None


def _write_private_json(path: Path, payload: dict) -> None:
    """Atomically create a mode-0600 auth file inside an isolated directory."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _generate_codex_image(full_prompt: str, out_path: str,
                          timeout: int = 300,
                          retry_timeout: int = 360,
                          raise_on_cap: bool = False) -> Optional[str]:
    """Generate an image via Codex CLI and copy it to out_path.

    Codex generates internally then we copy the newest image out of its
    isolated CODEX_HOME. Every configured pool account may be attempted.
    Returns out_path on success, None on failure. When raise_on_cap is set and
    Codex reports its usage cap, raises CodexCapExceeded so batch callers can
    defer instead of burning retries (a capped 429 does not consume quota).
    """
    try:
        pool = _load_codex_pool()
    except Exception as exc:
        print(f"[blog_illustrator] Codex credential pool unavailable: {exc}")
        return None

    entries = list(pool.entries())
    timeouts = [timeout, *([retry_timeout] * max(0, len(entries) - 1))]
    exhausted = False
    for attempt, current_timeout in enumerate(timeouts, 1):
        credential_id = pool.acquire_lease(entries[attempt - 1].id)
        if not credential_id:
            break
        entry = next((item for item in entries if item.id == credential_id), None)
        if entry is None:
            pool.release_lease(credential_id)
            break
        before_ts = time.time()
        copied_image = False
        try:
            with tempfile.TemporaryDirectory(prefix="hermes-codex-image-") as codex_home:
                codex_home_path = Path(codex_home)
                auth_path = codex_home_path / "auth.json"
                _write_private_json(
                    auth_path,
                    _codex_auth_payload(entry, refresher=_refresh_entry_tokens),
                )
                child_env = os.environ.copy()
                child_env["CODEX_HOME"] = codex_home
                result = subprocess.run(
                    ["codex", "exec", "--disable", "use_linux_sandbox_bwrap", full_prompt],
                    capture_output=True, text=True, timeout=current_timeout,
                    cwd=str(config.SAHILBLOG_REPO), env=child_env,
                )
                capped = _output_shows_cap(result.stdout) or _output_shows_cap(result.stderr)
                img = None if capped else _find_latest_codex_image(
                    after_ts=before_ts,
                    images_dir=codex_home_path / "generated_images",
                )
                if img and Path(img).exists():
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img, out_path)
                    copied_image = True
            print(f"[blog_illustrator] codex exec exit={result.returncode} (attempt {attempt})")
            if result.returncode != 0:
                print(f"[blog_illustrator] codex stdout: {(result.stdout or '')[-500:]}")
                print(f"[blog_illustrator] codex stderr: {(result.stderr or '')[-500:]}")
        except subprocess.TimeoutExpired:
            print(f"[blog_illustrator] codex timed out after {current_timeout}s (attempt {attempt})")
            continue
        except Exception as exc:
            print(f"[blog_illustrator] codex execution error: {exc}")
            continue
        finally:
            pool.release_lease(credential_id)

        if _output_shows_cap(result.stdout) or _output_shows_cap(result.stderr):
            print(f"[blog_illustrator] Codex usage cap reached for account {credential_id}")
            exhausted = True
            pool.mark_exhausted_and_rotate(
                status_code=429, credential_id=credential_id,
                api_key_hint=entry.access_token, failure_reason="rate_limit",
            )
            continue

        if not copied_image:
            print(f"[blog_illustrator] no image found after codex run (attempt {attempt})")
            if _output_shows_auth_failure(result.stdout) or _output_shows_auth_failure(result.stderr):
                print(f"[blog_illustrator] auth failure for account {credential_id}; "
                      "marking for rotation (re-login required)")
                pool.mark_exhausted_and_rotate(
                    status_code=401, credential_id=credential_id,
                    api_key_hint=entry.access_token, failure_reason="auth",
                )
            continue

        size_kb = Path(out_path).stat().st_size // 1024
        print(f"[blog_illustrator] generated {size_kb}KB -> {out_path}")
        if Path(out_path).exists():
            print(f"[blog_illustrator] verified: {Path(out_path).stat().st_size} bytes")
            return out_path

    if exhausted and raise_on_cap and not pool.has_available():
        raise CodexCapExceeded("Codex image usage cap reached for all configured accounts")
    print(f"[blog_illustrator] all Codex attempts failed for {out_path}")
    return None


def _find_latest_codex_image(
    after_ts: Optional[float] = None,
    images_dir: Optional[Path] = None,
) -> Optional[str]:
    """Find the newest Codex-generated image after a timestamp (by mtime)."""
    images_dir = images_dir or Path(
        os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))
    ) / "generated_images"
    if not images_dir.exists():
        return None
    candidates: list[tuple[float, str]] = []
    for session_dir in images_dir.iterdir():
        if not session_dir.is_dir():
            continue
        for img in session_dir.glob("*"):
            if img.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            try:
                mtime = img.stat().st_mtime
            except OSError:
                continue
            if after_ts is None or mtime >= after_ts:
                candidates.append((mtime, str(img)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _generate_webp(png_path: str) -> Optional[str]:
    """Generate a WebP copy alongside the PNG for the Astro <picture> tag.

    Always overwrites: PostLayout serves the .webp first, so a stale webp left
    beside a freshly regenerated png would keep the OLD image live.
    """
    webp_path = Path(png_path).with_suffix(".webp")
    try:
        result = subprocess.run(
            [
                "node", "-e",
                f"const s=require('sharp');"
                f" s('{png_path}').webp({{quality:90,effort:6}}).toFile('{webp_path}')"
                f" .then(()=>console.log('ok'))",
            ],
            capture_output=True, timeout=60,
            cwd=str(config.SAHILBLOG_REPO),
        )
        if result.returncode == 0 and webp_path.exists():
            print(f"[blog_illustrator] webp generated: {webp_path} ({webp_path.stat().st_size} bytes)")
            return str(webp_path)
    except Exception as exc:
        print(f"[blog_illustrator] webp generation skipped: {exc}")
    return None


# ── Section parsing ─────────────────────────────────────────────

def _extract_h2_headings(body_md: str) -> list[str]:
    """Extract H2 heading texts from markdown body."""
    out = []
    for line in body_md.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            out.append(m.group(1).strip())
    return out


def _extract_diagram_spec(draft: dict) -> Optional[str]:
    """Extract Mermaid diagram code from a blueprint-format draft body."""
    body = draft.get("body_md", "") or ""
    m = re.search(r"```mermaid\n(.*?)```", body, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_section_text(body_lines: list[str], heading: str) -> str:
    """Return the body text under a given H2 heading (for fallback concepts)."""
    found = False
    lines: list[str] = []
    for line in body_lines:
        if re.match(r"^##\s+" + re.escape(heading) + r"\s*$", line):
            found = True
            continue
        if found:
            if line.startswith("## "):
                break
            lines.append(line)
    return " ".join(l.strip() for l in lines if l.strip())[:400]


# ── Public API ──────────────────────────────────────────────────

# One hero plus this hard maximum preserves the scheduler's max_images=3 bound,
# even when a direct caller supplies an unsafe config or max_sections override.
MAX_SECTION_IMAGES = 2


def _resolve_extended_traits_for_brief(brief: dict) -> Optional[dict]:
    """Deterministic additive trait blend for a blog post (extended menu).

    Derived from the brief's selection_seed so the whole post is coherent
    (one style + one blend) while varying across posts. Never overrides the
    brief's style/layout; purely additive. Returns the injected dict (with
    ``fragment``) or None when unavailable. When the LLM brief already pinned
    an explicit extended blend, it is preserved (returned unchanged).
    """
    if brief.get("extended_traits"):
        return brief["extended_traits"]
    try:
        from style_registry import pick_variation, resolve_fragment
        seed = brief.get("selection_seed") or 0
        variation = pick_variation(seed)
        frag = resolve_fragment(
            style_slug=variation["style_slug"],
            blend_slugs=variation["blend_slugs"],
            seed=seed,
        )
        if not frag:
            return None
        return {
            "style_slug": variation["style_slug"],
            "blend_slugs": variation["blend_slugs"],
            "blend_seed": seed,
            "fragment": frag,
        }
    except Exception as exc:  # noqa: BLE001 — extended menu is additive, never a blocker
        print(f"[blog_illustrator] extended traits unavailable: {exc}")
        return None


def illustrate(
    draft: dict,
    out_dir: Optional[Path] = None,
    max_sections: Optional[int] = None,
    workdir: Optional[str] = None,
    raise_on_cap: bool = False,
) -> dict:
    """Generate an art-directed hero + section image set via Codex CLI.

    One art brief (style + locked palette/motif + shared direction) drives the
    whole post; each image gets a unique, article-grounded prompt composed
    against that shared direction, so the set is consistent but not repetitive.

    Returns {hero_path: str|None, section_paths: {h2_heading: path}}.
    """
    stream = draft.get("stream", "ai")
    if max_sections is None:
        max_sections = config.BLOG_MAX_SECTION_IMAGES
    max_sections = min(max(0, max_sections), MAX_SECTION_IMAGES)
    out_path = Path(out_dir) if out_dir else Path(config.OUTPUT_DIR) / "blog_images"
    out_path.mkdir(parents=True, exist_ok=True)

    result: dict = {"hero_path": None, "section_paths": {}}

    body_md = draft.get("body_md", "")
    headings = _extract_h2_headings(body_md)[:max_sections] if max_sections > 0 else []

    # One art brief for the whole post (LLM). HARD FAIL if unavailable.
    # The fallback_brief produces generic, article-disconnected images with no
    # palette, motif, or per-section art direction. Silently using it shipped
    # dozens of terrible images to production. Now we stop and report instead.
    recent = _load_recent_styles()
    recent_fingerprints = _load_recent_concept_fingerprints()
    brief = build_art_brief(
        draft,
        headings,
        recent_styles=recent,
        recent_concept_fingerprints=recent_fingerprints,
    )
    if brief is None:
        print("[blog_illustrator] ⚠️  ART DIRECTOR FAILED — refusing to generate images.")
        print("[blog_illustrator] The LLM art brief is mandatory. Fallback brief produces")
        print("[blog_illustrator] generic images with no article-specific art direction.")
        print(f"[blog_illustrator] SKIPPED: {draft.get('title', '?')}")
        return result
    print(f"[blog_illustrator] art brief: style={brief['style']} "
          f"seed={brief.get('selection_seed', 'n/a')} "
          f"layout={brief.get('layout', '')[:80]!r} "
          f"palette={brief.get('palette','')[:60]!r} motif={brief.get('motif','')[:60]!r}")
    concept_plan = brief.get("concept_plan") if isinstance(brief.get("concept_plan"), dict) else {}
    _record_selection(brief["style"], concept_plan.get("fingerprint"))

    # Extended style menu (blog path): deterministic additive trait blend for
    # this post, derived from the same selection_seed as the brief so the
    # whole post is coherent (one style + one blend) while varying across
    # posts. Never overrides the brief's style/layout; purely additive. When
    # the LLM brief already pinned an explicit blend, keep it.
    extended = _resolve_extended_traits_for_brief(brief)
    if extended:
        brief["extended_traits"] = extended

    # P11 contract seam: select only reviewed core references, write the plan
    # and planned provenance before the unchanged legacy generator is reached.
    asset_layouts = brief.get("asset_layouts", {})
    planned_prompts = {
        "hero": compose_prompt(
            brief["hero_prompt"], brief, assigned_layout=asset_layouts.get("hero")
        )
    }
    planned_outputs = {"hero": "hero.png"}
    section_prompts = brief.get("section_prompts", {})
    body_lines = body_md.splitlines()
    for index, heading in enumerate(headings, 1):
        concept = section_prompts.get(heading) or _extract_section_text(body_lines, heading) or heading
        key = f"section-{index:02d}"
        planned_prompts[key] = compose_prompt(
            concept, brief, assigned_layout=asset_layouts.get(key)
        )
        planned_outputs[key] = f"section_{index:02d}.png"
    try:
        visual_plan = art_director.build_visual_plan_from_brief(
            draft,
            headings,
            brief,
            catalog_root=Path(config.IMAGERY_ANCHORS_DIR),
        )
        planned_manifest = art_director.build_planned_asset_manifest_from_plan(
            visual_plan,
            planned_prompts,
            planned_outputs,
            text_policy=str(brief.get("text_policy", "none")),
        )
        visual_plan_path = save_visual_plan(visual_plan, out_path / "visual-plan.json")
        asset_manifest_path = save_asset_manifest(
            planned_manifest, out_path / "asset-manifest.json"
        )
    except (AssetManifestError, CatalogIntegrityError, OSError, ValueError, VisualPlanError) as exc:
        print(f"[blog_illustrator] P11 visual contract failed — refusing generation: {exc}")
        return result
    result["visual_plan_path"] = str(visual_plan_path)
    result["asset_manifest_path"] = str(asset_manifest_path)

    # ── Hero ────────────────────────────────────────────────
    hero_out = str(out_path / "hero.png")
    hero_prompt = planned_prompts["hero"]
    gen = _generate_codex_image(hero_prompt, hero_out, raise_on_cap=raise_on_cap)
    if gen and Path(gen).exists():
        result["hero_path"] = gen
        _generate_webp(gen)
        print(f"[blog_illustrator] hero via {gen}")
    else:
        print(f"[blog_illustrator] hero generation failed for '{draft.get('title','')}'")

    # ── Sections ────────────────────────────────────────────
    if headings:
        for idx, heading in enumerate(headings, 1):
            key = f"section-{idx:02d}"
            section_out = str(out_path / planned_outputs[key])
            prompt = planned_prompts[key]
            gen_sec = _generate_codex_image(prompt, section_out, raise_on_cap=raise_on_cap)
            if gen_sec and Path(gen_sec).exists():
                result["section_paths"][heading] = gen_sec
                _generate_webp(gen_sec)
                print(f"[blog_illustrator] section {idx} via {gen_sec}")
            else:
                print(f"[blog_illustrator] section '{heading}' generation failed")

    return result
