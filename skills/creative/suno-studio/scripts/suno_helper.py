#!/usr/bin/env python3
"""Cross-platform helper for the suno-studio skill (`suno.com/create`).

Capabilities:
  1. `cdp-status` / `cdp-launch`: Detect or launch a persistent Chromium-family
     browser (Chrome, Brave, Edge, Chromium) on Linux, macOS, and Windows with
     a dedicated `chrome-debug` profile under `HERMES_HOME` so Suno login
     persists across sessions.
  2. `validate-prompt`: Validate Suno prompt specs (title length, style length,
     blocked artist/trademark hints, lyrics length, unexpanded numbers/acronyms,
     excessive or contradictory `[metatags]`).
  3. `emit-fill-js` / `emit-click-js` / `emit-harvest-js`: Generate escaped,
     IIFE-wrapped JavaScript expressions for `browser_exec` (`js(...)`) or
     `browser_console` that set React controlled inputs via native prototype
     setters, click the Create button, and harvest generated `/song/` links.
  4. `parse-harvest`: Parse raw harvested track output into structured JSON
     grouped by song title and variation index.

Stdlib only; no external dependencies or live network calls required for prompt
validation, JS code generation, or harvest parsing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_PORT = 9222
STYLE_CHAR_LIMIT = 1000
LYRICS_CHAR_LIMIT = 3000
TITLE_WORD_LIMIT = 8
MAX_TAGS_PER_SECTION = 8

_PATH_BROWSER_NAMES = (
    "brave-browser",
    "brave",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
    "microsoft-edge-stable",
    "chrome",
)

_CONTRADICTORY_TAG_PAIRS = (
    ({"calm", "quiet arrangement", "low energy", "whispered"}, {"aggressive", "explosive", "high energy", "belted"}),
    ({"instrumental", "silence"}, {"belted", "spoken word", "choir", "harmonized chorus"}),
    ({"slow down", "falling tension"}, {"building energy", "explosive"}),
)

_COMMON_ARTIST_BLOCKLIST = (
    "taylor swift",
    "drake",
    "nirvana",
    "metallica",
    "beatles",
    "the weeknd",
    "billie eilish",
    "ed sheeran",
    "kanye",
    "radiohead",
    "daft punk",
    "hans zimmer",
    "john williams",
    "james bond",
    "eminem",
    "kendrick lamar",
    "beyonce",
    "adele",
)


def hermes_home() -> Path:
    """Resolve the active Hermes home directory (`HERMES_HOME` or `~/.hermes`)."""
    env_home = os.environ.get("HERMES_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return (Path.home() / ".hermes").resolve()


def default_profile_dir() -> Path:
    """Dedicated debug profile dir (`$HERMES_HOME/chrome-debug`) for persistent Suno auth."""
    return hermes_home() / "chrome-debug"


def _mac_browser_candidates() -> list[str]:
    return [
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]


def _windows_browser_candidates() -> list[str]:
    bases = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("LOCALAPPDATA", ""),
    ]
    rels = [
        r"BraveSoftware\Brave-Browser\Application\brave.exe",
        r"Google\Chrome\Application\chrome.exe",
        r"Chromium\Application\chrome.exe",
        r"Microsoft\Edge\Application\msedge.exe",
    ]
    out: list[str] = []
    for base in bases:
        if not base:
            continue
        for rel in rels:
            out.append(str(Path(base) / rel))
    return out


def find_browser(override: str | None = None, platform: str | None = None) -> str | None:
    """Locate a Chromium-family executable across Linux, macOS, and Windows."""
    if override:
        if Path(override).exists():
            return override
        return shutil.which(override)

    for name in _PATH_BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return found

    plat = platform or sys.platform
    if plat == "darwin":
        candidates = _mac_browser_candidates()
    elif plat == "win32":
        candidates = _windows_browser_candidates()
    else:
        candidates = []

    for cand in candidates:
        if Path(cand).exists():
            return cand
    return None


def build_launch_argv(
    browser: str,
    port: int = DEFAULT_PORT,
    profile_dir: Path | None = None,
) -> list[str]:
    """Build cross-platform argv to start Chromium/Brave with remote debugging."""
    profile = profile_dir or default_profile_dir()
    return [
        browser,
        f"--remote-debugging-port={int(port)}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
    ]


def cdp_endpoint_status(
    port: int = DEFAULT_PORT,
    host: str = "127.0.0.1",
    timeout: float = 1.0,
) -> dict[str, Any] | None:
    """Return `/json/version` payload if a CDP browser is listening on `host:port`."""
    url = f"http://{host}:{int(port)}/json/version"
    req = urllib.request.Request(url, headers={"User-Agent": "suno-studio-cdp/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError):
        return None
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except (ValueError, AttributeError):
        return None
    return data if isinstance(data, dict) else None


def launch_cdp_browser(
    browser: str | None = None,
    port: int = DEFAULT_PORT,
    profile_dir: Path | None = None,
) -> dict[str, Any]:
    """Start a detached Chromium-family browser with CDP enabled on Linux, macOS, or Windows."""
    resolved = find_browser(browser)
    if not resolved:
        return {
            "ok": False,
            "error": "No Chromium-family browser (Brave, Chrome, Chromium, Edge) found on this machine.",
        }
    profile = profile_dir or default_profile_dir()
    profile.mkdir(parents=True, exist_ok=True)
    argv = build_launch_argv(resolved, port=port, profile_dir=profile)
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        )
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(argv, **kwargs)
    return {
        "ok": True,
        "pid": proc.pid,
        "browser": resolved,
        "port": int(port),
        "cdp_url": f"http://127.0.0.1:{int(port)}",
        "profile_dir": str(profile),
        "argv": argv,
    }


def validate_prompt(
    title: str,
    style: str,
    lyrics: str,
    exclude_styles: str = "",
) -> dict[str, Any]:
    """Validate a Suno Custom/Advanced prompt for length, tag rules, and common pitfalls."""
    errors: list[str] = []
    warnings: list[str] = []

    title_clean = title.strip()
    style_clean = style.strip()
    lyrics_clean = lyrics.strip()

    if not title_clean:
        errors.append("Title is empty.")
    else:
        word_count = len(title_clean.split())
        if word_count > TITLE_WORD_LIMIT:
            warnings.append(
                f"Title has {word_count} words (recommended <= {TITLE_WORD_LIMIT} words)."
            )

    if not style_clean:
        errors.append("Style prompt is empty.")
    elif len(style_clean) > STYLE_CHAR_LIMIT:
        errors.append(
            f"Style prompt is {len(style_clean)} chars (hard limit is {STYLE_CHAR_LIMIT})."
        )

    lower_style = style_clean.lower()
    for artist in _COMMON_ARTIST_BLOCKLIST:
        if re.search(rf"\b{re.escape(artist)}\b", lower_style):
            errors.append(
                f"Style prompt contains blocked artist/trademark reference {artist!r}; describe the sonic palette instead."
            )

    if not lyrics_clean:
        warnings.append("Lyrics are empty (track will render as instrumental or auto-lyrics).")
    else:
        if len(lyrics_clean) > LYRICS_CHAR_LIMIT:
            errors.append(
                f"Lyrics are {len(lyrics_clean)} chars (limit is {LYRICS_CHAR_LIMIT})."
            )

        tags_all = re.findall(r"\[([^\[\]]+)\]", lyrics_clean)
        if not tags_all:
            warnings.append(
                "Lyrics contain no structural [metatags] (e.g. [Intro], [Verse 1], [Chorus])."
            )

        # Split lyrics into sections starting at structural headers or blank-line blocks
        sections = re.split(r"\n\s*\n", lyrics_clean)
        for idx, section in enumerate(sections, start=1):
            sec_tags = [t.strip() for t in re.findall(r"\[([^\[\]]+)\]", section)]
            if len(sec_tags) > MAX_TAGS_PER_SECTION:
                warnings.append(
                    f"Section {idx} has {len(sec_tags)} metatags (recommended <= {MAX_TAGS_PER_SECTION})."
                )
            lowered = {t.lower() for t in sec_tags}
            for side_a, side_b in _CONTRADICTORY_TAG_PAIRS:
                hit_a = lowered & side_a
                hit_b = lowered & side_b
                if hit_a and hit_b:
                    warnings.append(
                        f"Section {idx} combines contradictory metatags: {sorted(hit_a)} vs {sorted(hit_b)}."
                    )

        # Strip bracketed metatags before checking lyric text for raw numbers
        lyric_lines_no_tags = re.sub(r"\[[^\[\]]+\]", "", lyrics_clean)
        raw_numbers = re.findall(r"\b\d+(?:/\d+)?\b", lyric_lines_no_tags)
        if raw_numbers:
            warnings.append(
                f"Lyrics contain unexpanded digits {sorted(set(raw_numbers))[:5]}; spell numbers out phonetically for AI vocalists."
            )

    return {
        "valid": len(errors) == 0,
        "title_words": len(title_clean.split()) if title_clean else 0,
        "style_chars": len(style_clean),
        "lyrics_chars": len(lyrics_clean),
        "exclude_styles": exclude_styles.strip(),
        "errors": errors,
        "warnings": warnings,
    }


def emit_batch_fill_js(
    title: str,
    style: str,
    lyrics: str,
    exclude_styles: str = "",
) -> str:
    """Return a self-contained IIFE JS snippet that fills Title, Lyrics, and Style on suno.com/create."""
    payload = {
        "title": title,
        "style": style,
        "lyrics": lyrics,
        "excludeStyles": exclude_styles,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    return (
        "(() => {\n"
        f"  const data = {payload_json};\n"
        "  const inputSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;\n"
        "  const taSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;\n"
        "  const setField = (el, setter, val) => {\n"
        "    if (!el || val === undefined || val === null) return false;\n"
        "    setter.call(el, String(val));\n"
        "    el.dispatchEvent(new Event('input', { bubbles: true }));\n"
        "    el.dispatchEvent(new Event('change', { bubbles: true }));\n"
        "    return true;\n"
        "  };\n"
        "  const titleEl = document.querySelector('input[placeholder*=\"Song Title\"], input[placeholder*=\"Title\"]');\n"
        "  let lyricsEl = null, styleEl = null, excludeEl = null;\n"
        "  for (const ta of document.querySelectorAll('textarea')) {\n"
        "    const ph = (ta.placeholder || '').toLowerCase();\n"
        "    const aria = (ta.getAttribute('aria-label') || '').toLowerCase();\n"
        "    if (!lyricsEl && (ph.includes('lyrics') || aria.includes('lyrics'))) lyricsEl = ta;\n"
        "    else if (!styleEl && (ph.includes('classic jazz') || ph.includes('style') || aria.includes('style'))) styleEl = ta;\n"
        "  }\n"
        "  for (const inp of document.querySelectorAll('input, textarea')) {\n"
        "    const ph = (inp.placeholder || '').toLowerCase();\n"
        "    if (ph.includes('exclude')) { excludeEl = inp; break; }\n"
        "  }\n"
        "  const status = {\n"
        "    title: setField(titleEl, inputSetter, data.title),\n"
        "    lyrics: setField(lyricsEl, taSetter, data.lyrics),\n"
        "    style: setField(styleEl, taSetter, data.style),\n"
        "    exclude: data.excludeStyles ? setField(excludeEl, excludeEl && excludeEl.tagName === 'TEXTAREA' ? taSetter : inputSetter, data.excludeStyles) : true,\n"
        "  };\n"
        "  return JSON.stringify(status);\n"
        "})()"
    )


def emit_click_create_js() -> str:
    """Return an IIFE JS snippet that locates and clicks the Suno Create button."""
    return (
        "(() => {\n"
        "  for (const btn of document.querySelectorAll('button')) {\n"
        "    const t = (btn.textContent || '').trim();\n"
        "    if (t === 'Create song' || t === 'Create' || t === 'Create songs') {\n"
        "      btn.click();\n"
        "      return JSON.stringify({ clicked: true, label: t, disabled: Boolean(btn.disabled) });\n"
        "    }\n"
        "  }\n"
        "  return JSON.stringify({ clicked: false, error: 'Create button not found' });\n"
        "})()"
    )


def emit_harvest_links_js() -> str:
    """Return an IIFE JS snippet that collects generated `/song/` links from the Suno workspace."""
    return (
        "(() => {\n"
        "  const seen = new Set();\n"
        "  const results = [];\n"
        "  for (const link of document.querySelectorAll('a[href*=\"/song/\"]')) {\n"
        "    const title = (link.textContent || '').trim();\n"
        "    const href = link.href || link.getAttribute('href') || '';\n"
        "    if (!title || !href) continue;\n"
        "    const key = title + '|' + href;\n"
        "    if (seen.has(key)) continue;\n"
        "    seen.add(key);\n"
        "    results.push(title + ' | ' + href);\n"
        "  }\n"
        "  return results.join('\\n');\n"
        "})()"
    )


def parse_harvested_links(raw_text: str) -> list[dict[str, Any]]:
    """Parse newline-delimited `Title | https://suno.com/song/<id>` lines into structured tracks."""
    tracks: list[dict[str, Any]] = []
    counts_by_title: dict[str, int] = {}
    seen_urls: set[str] = set()

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        title_part, url_part = line.rsplit("|", 1)
        title = title_part.strip()
        url = url_part.strip()
        if not title or "/song/" not in url:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        m = re.search(r"/song/([A-Za-z0-9-]+)", url)
        song_id = m.group(1) if m else ""
        variation = counts_by_title.get(title, 0) + 1
        counts_by_title[title] = variation
        tracks.append(
            {
                "title": title,
                "url": url,
                "song_id": song_id,
                "variation": variation,
            }
        )
    return tracks


def _read_text_arg(inline_val: str | None, file_val: str | None) -> str:
    if file_val:
        return Path(file_val).read_text(encoding="utf-8-sig")
    return inline_val or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="suno_helper.py",
        description="Cross-platform helper for Suno Studio browser automation and prompt validation.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("cdp-status", help="Check if a CDP browser is listening on port 9222")
    p_status.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_status.add_argument("--host", default="127.0.0.1")

    p_launch = sub.add_parser("cdp-launch", help="Launch a persistent CDP browser across Linux/macOS/Windows")
    p_launch.add_argument("--browser", default=None, help="Optional browser binary name or path")
    p_launch.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_launch.add_argument("--profile-dir", default=None, help="Custom user-data-dir path")
    p_launch.add_argument("--print-only", action="store_true", help="Print the resolved argv without spawning")

    p_val = sub.add_parser("validate-prompt", help="Validate Suno Title, Style, and Lyrics fields")
    p_val.add_argument("--title", default="")
    p_val.add_argument("--style", default="")
    p_val.add_argument("--style-file", default=None)
    p_val.add_argument("--lyrics", default="")
    p_val.add_argument("--lyrics-file", default=None)
    p_val.add_argument("--exclude-styles", default="")

    p_fill = sub.add_parser("emit-fill-js", help="Emit an escaped IIFE JS snippet to batch-fill Suno fields")
    p_fill.add_argument("--title", required=True)
    p_fill.add_argument("--style", default="")
    p_fill.add_argument("--style-file", default=None)
    p_fill.add_argument("--lyrics", default="")
    p_fill.add_argument("--lyrics-file", default=None)
    p_fill.add_argument("--exclude-styles", default="")

    sub.add_parser("emit-click-js", help="Emit an IIFE JS snippet to click Create")
    sub.add_parser("emit-harvest-js", help="Emit an IIFE JS snippet to harvest /song/ links")

    p_harvest = sub.add_parser("parse-harvest", help="Parse harvested `Title | URL` lines into JSON")
    p_harvest.add_argument("--input", default=None, help="Input file (defaults to stdin)")

    args = parser.parse_args(argv)

    if args.cmd == "cdp-status":
        info = cdp_endpoint_status(port=args.port, host=args.host)
        if info is None:
            print(json.dumps({"listening": False, "port": args.port, "host": args.host}))
            return 1
        print(
            json.dumps(
                {
                    "listening": True,
                    "port": args.port,
                    "host": args.host,
                    "browser": info.get("Browser", ""),
                    "webSocketDebuggerUrl": info.get("webSocketDebuggerUrl", ""),
                }
            )
        )
        return 0

    if args.cmd == "cdp-launch":
        profile = Path(args.profile_dir).expanduser().resolve() if args.profile_dir else default_profile_dir()
        if args.print_only:
            resolved = find_browser(args.browser) or (args.browser or "chromium")
            argv_out = build_launch_argv(resolved, port=args.port, profile_dir=profile)
            print(json.dumps({"ok": True, "argv": argv_out, "profile_dir": str(profile)}))
            return 0
        res = launch_cdp_browser(browser=args.browser, port=args.port, profile_dir=profile)
        print(json.dumps(res))
        return 0 if res.get("ok") else 1

    if args.cmd == "validate-prompt":
        style = _read_text_arg(args.style, args.style_file)
        lyrics = _read_text_arg(args.lyrics, args.lyrics_file)
        report = validate_prompt(args.title, style, lyrics, args.exclude_styles)
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 1

    if args.cmd == "emit-fill-js":
        style = _read_text_arg(args.style, args.style_file)
        lyrics = _read_text_arg(args.lyrics, args.lyrics_file)
        print(emit_batch_fill_js(args.title, style, lyrics, args.exclude_styles))
        return 0

    if args.cmd == "emit-click-js":
        print(emit_click_create_js())
        return 0

    if args.cmd == "emit-harvest-js":
        print(emit_harvest_links_js())
        return 0

    if args.cmd == "parse-harvest":
        raw = Path(args.input).read_text(encoding="utf-8-sig") if args.input else sys.stdin.read()
        tracks = parse_harvested_links(raw)
        print(json.dumps({"count": len(tracks), "tracks": tracks}, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
