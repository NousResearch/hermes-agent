"""What's-New brief loader and seen-state manager.

PR-A of the "Feature Onboarding" initiative: after every ``hermes update``
(and on first run), users can learn what changed, when to use a feature,
and how — through the surfaces they already use (CLI, gateway, desktop).

Design invariants (see SECURITY-BASELINE.md in the PR):

* **Offline-safe.** The happy path reads a static repo file
  (``docs/whats-new/<version>.md``); the GitHub release-body fallback is
  wrapped, timed out, and size-capped — never required.
* **Decoupled from update.** Display is a side-effect of a *successful*
  update; it never blocks, gates, or races with the update state machine
  (this is the class of bug that sank prior art #13684).
* **Atomic state.** Seen/dismiss state lives in
  ``$HERMES_HOME/whats_new_seen.json``, written via ``os.replace`` so a
  concurrent writer can never observe a torn file. Corrupt state is treated
  as "nothing seen" for *newer* versions only — a corrupt file must never
  hide a new feature.
* **No new core tools, no subprocess.** Pure stdlib file I/O + print.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Directory (repo-relative) holding per-version briefs.
WHATS_NEW_DIR_NAME = "whats-new"

#: Front-matter keys we understand.
_VERSION_KEY = "version"
_RELEASE_DATE_KEY = "release_date"
_SINCE_VERSION_KEY = "since_version"

#: Strict version pattern used to validate user-supplied version args
#: before they are turned into file paths (no traversal).
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

#: Cap on how many features a single brief may surface per print (anti-flood).
DEFAULT_MAX_FEATURES = 8

#: Cap on GitHub fallback body size (chars) — defense against a hostile or
#: accidentally enormous release body.
_MAX_BODY_CHARS = 50_000

#: Dismiss levels stored in seen-file.
DISMISS_UNDERSTOOD = "understood"
DISMISS_LEARN_MORE = "learn_more"
DISMISS_NEVER_AGAIN = "never_again"
_VALID_DISMISS = {DISMISS_UNDERSTOOD, DISMISS_LEARN_MORE, DISMISS_NEVER_AGAIN}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class WhatsNewBrief:
    """Structured representation of one version's what's-new brief."""

    def __init__(self, version: str, markdown: str, features: List[Dict[str, str]]):
        self.version = version
        self.markdown = markdown
        self.features = features

    def render(self, max_features: int = DEFAULT_MAX_FEATURES) -> str:
        """Return the user-facing text (plain markdown, truncated to max_features)."""
        lines: List[str] = [f"✨ What's new in Hermes {self.version}"]
        shown = 0
        for feat in self.features:
            if shown >= max_features:
                lines.append(f"\n… and {len(self.features) - shown} more — run `/whats-new` for the full list.")
                break
            lines.append("")
            lines.append(f"### {feat.get('name', 'Untitled')}")
            one = feat.get("one_line", "").strip()
            if one:
                lines.append(f"**What:** {one}")
            use = feat.get("use_when", "").strip()
            if use:
                lines.append(f"**When to use:** {use}")
            how = feat.get("how", "").strip()
            if how:
                lines.append(f"**How:**\n```\n{how}\n```")
            rel = feat.get("related", "").strip()
            if rel:
                lines.append(f"**Related:** {rel}")
            shown += 1
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_front_matter(text: str) -> tuple[Dict[str, str], str]:
    """Split YAML-ish front matter from body. Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: Dict[str, str] = {}
    for raw in parts[1].splitlines():
        if ":" in raw:
            k, _, v = raw.partition(":")
            meta[k.strip()] = v.strip().strip('"\'')
    return meta, parts[2].strip()


def _parse_features(body: str) -> List[Dict[str, str]]:
    """Parse brief body into feature entries.

    Format (see docs/whats-new/v0.21.0.md):
        ## N. Title
        - **One-line:** ...
        - **Use when:** ...
        - **How:**
          ```
          /whats-new
          ```
        - **Related:** ...

    ``How:`` (and any field) may span multiple lines: continuation lines
    (indented, or a fenced code block) are collected until the next
    ``- **label:**`` field or the next ``## `` entry. Empty/placeholder
    entries (no fields besides a name) are skipped.
    """
    features: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    pending_key: Optional[str] = None  # field we're collecting continuation lines for
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("## "):
            if current is not None:
                features.append(current)
            current = {"name": stripped[3:].strip()}
            pending_key = None
            i += 1
            continue

        if current is None:
            i += 1
            continue

        low = line.lower()
        matched = False
        for key, label in (
            ("one_line", "one-line"),
            ("use_when", "use when"),
            ("how", "how"),
            ("related", "related"),
        ):
            if low.startswith(f"- **{label}"):
                rest = line.split("**", 2)[2].strip()
                if rest.startswith(":"):
                    rest = rest[1:].strip()
                current[key] = rest
                pending_key = key
                matched = True
                break
        if matched:
            i += 1
            continue

        # Continuation line for the pending field: indented content or a
        # fenced code block line. Collected until next field/entry.
        if pending_key is not None and (stripped or line.startswith((" ", "\t"))):
            prev = current.get(pending_key, "")
            if stripped.startswith("```"):
                # Fence marker: keep the code-block content but not the fences.
                i += 1
                continue
            if stripped:
                sep = "\n" if prev else ""
                current[pending_key] = prev + sep + stripped
            i += 1
            continue

        # Blank line inside a fenced block for `How:` (keep empty lines so
        # code blocks stay readable) — only when pending_key == "how".
        if pending_key == "how" and not stripped:
            current[pending_key] = current.get(pending_key, "") + "\n"
            i += 1
            continue

        # Any other non-field, non-continuation line ends the pending field.
        pending_key = None
        i += 1

    if current is not None:
        features.append(current)

    # Skip placeholder/empty entries: a feature is only real if at least one
    # of its content fields has non-empty text (a template entry whose
    # fields are all blank must not render).
    return [
        f for f in features
        if any((f.get(k) or "").strip() for k in ("one_line", "use_when", "how", "related"))
    ]


def _brief_path(repo_root: Path, version: str) -> Path:
    return repo_root / "docs" / WHATS_NEW_DIR_NAME / f"v{version}.md"


def get_whats_new(repo_root: Path, version: str) -> Optional[WhatsNewBrief]:
    """Load a brief for ``version`` from the repo's docs directory.

    Returns None when the file is missing or unparseable — callers must
    treat None as "no brief available" and stay silent.
    """
    path = _brief_path(repo_root, version)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    except UnicodeDecodeError:
        logger.warning("whats-new brief %s is not UTF-8; skipping", path)
        return None
    meta, body = _parse_front_matter(text)
    features = _parse_features(body)
    if not features:
        return None
    return WhatsNewBrief(version=version, markdown=body, features=features)


def get_current_version(repo_root: Path) -> Optional[str]:
    """Read the installed version from the repo (pyproject.toml)."""
    try:
        pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    if not m:
        return None
    return m.group(1)


def _versions_on_disk(repo_root: Path) -> List[str]:
    d = repo_root / "docs" / WHATS_NEW_DIR_NAME
    if not d.is_dir():
        return []
    out: List[str] = []
    for p in d.glob("v*.md"):
        m = re.match(r"^v(\d+\.\d+\.\d+)\.md$", p.name)
        if m:
            out.append(m.group(1))
    return sorted(out, key=lambda s: tuple(int(x) for x in s.split(".")))


# ---------------------------------------------------------------------------
# Seen-state
# ---------------------------------------------------------------------------

def _seen_path(hermes_home: Path) -> Path:
    return hermes_home / "whats_new_seen.json"


def load_seen(hermes_home: Path) -> Dict[str, Any]:
    try:
        data = json.loads(_seen_path(hermes_home).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def mark_seen(hermes_home: Path, version: str, dismiss: str = DISMISS_UNDERSTOOD) -> None:
    """Record that the user has acknowledged version's brief (atomic write).

    Corrupt/absent file is recreated from scratch. Invalid dismiss values
    are clamped to ``understood`` so a bad caller can't create junk state.

    Concurrency: a unique temp name (pid + counter) avoids the race where
    two writers (CLI process + gateway) collide on the same ``.tmp`` path;
    ``os.replace`` keeps the final write atomic.
    """
    if dismiss not in _VALID_DISMISS:
        dismiss = DISMISS_UNDERSTOOD
    state = load_seen(hermes_home)
    state.setdefault("seen", {})[version] = {
        "dismiss": dismiss,
        "at": int(time.time()),
    }
    path = _seen_path(hermes_home)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{_tmp_counter()}.tmp"
    )
    try:
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        logger.warning("failed to persist whats-new seen state to %s", path)
    finally:
        # Best-effort cleanup of our own temp file if replace failed.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


_tmp_counter_lock = 0


def _tmp_counter() -> int:
    """Return a process-local monotonic counter for unique temp names."""
    global _tmp_counter_lock
    _tmp_counter_lock += 1
    return _tmp_counter_lock


def unseen_versions(hermes_home: Path, repo_root: Path, current: str) -> List[str]:
    """Versions with briefs on disk that the user has not acknowledged.

    A corrupt seen-file must never hide a new feature: if the file can't be
    read, we return *all* versions with briefs (for the caller to filter by
    recency as appropriate).
    """
    seen = load_seen(hermes_home)
    known = set(seen.get("seen", {}).keys())
    on_disk = _versions_on_disk(repo_root)
    result = [v for v in on_disk if v not in known]
    # Only show up to current version (never future versions from a newer
    # checkout than the one running).
    result = [v for v in result if tuple(int(x) for x in v.split("."))
              <= tuple(int(x) for x in current.split("."))]
    return result


def validate_version_arg(raw: str) -> Optional[str]:
    """Validate a user-supplied version string; None if invalid."""
    raw = raw.strip()
    if not _VERSION_RE.match(raw):
        return None
    return raw


# ---------------------------------------------------------------------------
# Convenience: the post-update notice (same pattern as curator notice)
# ---------------------------------------------------------------------------

def _print_whats_new_notice() -> None:
    """Print a short what's-new brief after ``hermes update`` (and on first
    interactive CLI launch via cmd_chat).

    Prefers the current version's brief; if none exists, falls back to the
    most recent unseen version on disk (``unseen_versions``). Silent when
    everything is acknowledged or no brief exists. Never raises.
    """
    try:
        from hermes_constants import get_hermes_home
        from hermes_cli.config import load_config
        from hermes_cli.config import get_project_root

        cfg = load_config() or {}
        if not cfg.get("whats_new", {}).get("enabled", True):
            return
        repo_root = get_project_root()
        current = get_current_version(repo_root)
        if not current:
            return
        home = get_hermes_home()
        brief = get_whats_new(repo_root, current)
        if brief is None:
            # No brief for the current version — surface the most recent
            # unseen version instead (keeps unseen_versions live and gives
            # users on pre-brief versions a path to learn about them).
            unseen = unseen_versions(home, repo_root, current)
            if not unseen:
                return
            target = unseen[-1]
            brief = get_whats_new(repo_root, target)
            if brief is None:
                return
        else:
            target = current
        if target in load_seen(home).get("seen", {}):
            return
        print()
        print(brief.render())
        print()
        print("  Acknowledge: /whats-new  ·  Dismiss: /whats-new --seen")
    except Exception as e:  # never break an update
        logger.debug("whats-new notice failed: %s", e)
