"""
hermes sync — back up and sync a Hermes profile to YOUR OWN git repository.

This is for **independent git users**. It pushes a curated subset of the
profile (skills, memory, persona, and an allow-listed slice of config) to a
git repo you control — your own GitHub/GitLab/whatever. There is no Nous
portal, object store, or content-addressing layer involved; it is plain
``git`` driven over ``subprocess``.

Subcommands:
  hermes sync init               One-time setup: choose PRIVATE (default) or
                                 PUBLIC, create/point at a remote, write the
                                 secret-excluding .gitignore, persist config.
  hermes sync push               Stage the synced subset, run a pre-push secret
                                 scan, commit, and push. Aborts if anything
                                 secret-shaped is about to be committed.
  hermes sync pull [--force]     Pull from the remote and update the local
                                 profile. Last-writer-wins; confirms before
                                 clobbering locally-modified files.
  hermes sync status            Show remote + visibility, last sync time,
                                 pending changes, dirty state.
  hermes sync share <skill>      Print the install command others run, and
                                 optionally stage+push just that skill.

Design — why a dedicated staging git dir rather than a worktree over the live
profile:

  We ``git init`` a dedicated **staging directory** at
  ``<hermes_home>/.sync-git/`` and mirror ONLY the curated subset into it.
  This is deliberately chosen over pointing a worktree-style ``GIT_DIR`` at
  the live ``~/.hermes`` because:

    1. Git metadata stays out of the live profile.
    2. The staging dir physically contains *only* the curated subset, so it is
       structurally impossible to commit a secret that lives elsewhere in the
       profile — there is nothing else there to add.
    3. The allow-listed config slice is materialized as a generated
       ``sync-config.yaml`` in the staging dir; the raw ``config.yaml`` (which
       holds model keys / secrets) is never copied in and is .gitignored as a
       second line of defense.

  Pull is the mirror image: we fetch into the staging dir, then copy the
  curated files back out into the live profile.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

# ---------------------------------------------------------------------------
# Constants — what syncs, what is excluded
# ---------------------------------------------------------------------------

#: Subdirectory under HERMES_HOME holding the dedicated sync git repo / staging
#: mirror.  Never the live profile root itself.
SYNC_DIR_NAME = ".sync-git"

#: Generated file (inside the staging dir) that holds ONLY the allow-listed
#: config keys.  The raw config.yaml is never synced.
SYNC_CONFIG_NAME = "sync-config.yaml"

#: Allow-listed config.yaml keys that are safe to sync.  Dotted paths are
#: resolved against the nested config dict.  NOTHING else from config.yaml is
#: ever written into the staging dir — model keys, provider creds, gateway
#: tokens, etc. stay local.
CONFIG_ALLOWLIST: Tuple[str, ...] = (
    "display.skin",
    "default_toolsets",
)

#: Files / trees mirrored from the live profile into the staging dir.
#:   (relative-path-under-hermes-home, kind)  where kind is "tree" or "file".
SYNCED_PATHS: Tuple[Tuple[str, str], ...] = (
    ("skills", "tree"),
    ("memories/MEMORY.md", "file"),
    ("memories/USER.md", "file"),
    ("SOUL.md", "file"),  # persona; HERMES.md is a project-level override, not profile state
)

#: .gitignore content for the staging repo.  Load-bearing: excludes every
#: known secret / ephemeral artifact.  Even though the staging dir only ever
#: receives the curated subset, this is belt-and-suspenders so that a future
#: code change (or a user manually dropping a file in) can't leak.
GITIGNORE_CONTENT = """\
# Managed by `hermes sync`. Excludes secrets and ephemera.
# DO NOT remove the secret-exclusion lines.
#
# NOTE: patterns are ANCHORED with a leading slash so they match ONLY at the
# repo root, never nested user content. An unanchored pattern like ``*cache*/``
# or ``*.lock`` would silently drop a legitimately-named user skill such as
# ``skills/response-cache-helper/`` or ``skills/poetry-lock-helper/`` — git
# applies unanchored patterns at every directory depth. The staging mirror
# already contains only the curated subset, so these root-anchored excludes are
# belt-and-suspenders for the few profile-root artifacts that could appear.

# ── Secrets (match anywhere — these names are never legitimate content) ──
.env
.env.*
auth.json
/config.yaml
*.pem
*.key
id_rsa
id_ed25519

# ── Local state / databases (root-anchored: real profile artifacts) ──
/state.db
/state.db-*
/sessions.db
/sessions.db-*
/*.sqlite
/*.sqlite3

# ── Ephemera / caches / logs (root-anchored so nested skills are safe) ──
/logs/
/*cache*/
/document_cache/
/audio_cache/
/heapdumps/
/checkpoints/
/plugins/*/data/

# ── Editor / OS junk (safe to match anywhere) ──
.DS_Store
*.swp
*.bak
*.bak.*
*.corrupt.*
__pycache__/
"""


# ---------------------------------------------------------------------------
# Console helpers — Rich if available, plain print otherwise (mirrors the
# defensive style used across hermes_cli when Rich may be unavailable).
# ---------------------------------------------------------------------------

def _console():
    try:
        from rich.console import Console

        return Console()
    except Exception:  # pragma: no cover - Rich is a hard dep, but be safe
        return None


def _print(msg: str = "", style: Optional[str] = None) -> None:
    con = _console()
    if con is not None:
        con.print(msg, style=style)
    else:  # pragma: no cover
        print(msg)


def _err(msg: str) -> None:
    _print(msg, style="bold red")


def _ok(msg: str) -> None:
    _print(msg, style="green")


def _warn(msg: str) -> None:
    _print(msg, style="yellow")


def _has_tty() -> bool:
    """True if stdin is an interactive terminal we can prompt on.

    Slash commands (TUI / desktop / messaging) run in a worker with no
    interactive stdin — prompting there hangs or auto-answers. Callers use this
    to require explicit flags (--yes / --remote / --public/--private) instead.
    """
    try:
        import sys
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _sync_dir() -> Path:
    """Dedicated staging git dir under the (profile-aware) HERMES_HOME."""
    return get_hermes_home() / SYNC_DIR_NAME


def _git(args: List[str], *, cwd: Path, check: bool = True,
         capture: bool = True) -> subprocess.CompletedProcess:
    """Run a plain ``git`` subprocess in *cwd*."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=check,
        text=True,
        capture_output=capture,
    )


def _git_available() -> bool:
    return shutil.which("git") is not None


def _gh_available() -> bool:
    return shutil.which("gh") is not None


# ---------------------------------------------------------------------------
# Config (sync: key in config.yaml)
# ---------------------------------------------------------------------------

def _read_sync_config() -> Dict[str, Any]:
    from hermes_cli.config import load_config

    cfg = load_config()
    sub = cfg.get("sync")
    return dict(sub) if isinstance(sub, dict) else {}


def _write_sync_config(updates: Dict[str, Any]) -> None:
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    sub = cfg.get("sync")
    if not isinstance(sub, dict):
        sub = {}
    sub.update(updates)
    cfg["sync"] = sub
    save_config(cfg)


def _get_nested(config: Dict[str, Any], dotted: str) -> Any:
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_nested(target: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = target
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def build_synced_config_subset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return ONLY the allow-listed keys from *config*.

    This is the single chokepoint that guarantees no non-allow-listed key
    (model creds, gateway tokens, provider api keys, ...) ever reaches the
    synced file.  Tested directly.
    """
    out: Dict[str, Any] = {}
    for dotted in CONFIG_ALLOWLIST:
        val = _get_nested(config, dotted)
        if val is not None:
            _set_nested(out, dotted, val)
    return out


# ---------------------------------------------------------------------------
# Secret scanning — reuse tools/skills_guard.py credential patterns
# ---------------------------------------------------------------------------

def _credential_pattern_ids() -> set:
    """The skills_guard pattern ids that indicate a leaked secret value.

    We intentionally restrict the pre-push gate to the credential-exposure
    category (hardcoded keys, private keys, provider tokens) rather than the
    full threat table — a synced skill is *allowed* to legitimately reference
    ``os.environ.get("FOO_API_KEY")`` etc., but must never embed the secret
    value itself.
    """
    return {
        "hardcoded_secret",
        "embedded_private_key",
        "github_token_leaked",
        "openai_key_leaked",
        "anthropic_key_leaked",
        "aws_access_key_leaked",
    }


def scan_paths_for_secrets(paths: List[Path], root: Path) -> List[Tuple[str, str, str]]:
    """Walk *paths* and return secret findings.

    Returns a list of ``(relative_path, pattern_id, matched_snippet)`` for any
    file whose content trips a credential-exposure pattern from
    ``tools.skills_guard``.  Reuses skills_guard's ``scan_file`` and its
    ``THREAT_PATTERNS`` table rather than duplicating regexes.
    """
    from tools.skills_guard import scan_file

    wanted = _credential_pattern_ids()
    findings: List[Tuple[str, str, str]] = []

    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(
                f for f in p.rglob("*")
                # Skip git internals and build junk: scanning .git/objects is
                # slow and meaningless (packed/zlib blobs), and a symlink could
                # point outside the tree — only scan real files.
                if f.is_file()
                and not f.is_symlink()
                and ".git" not in f.parts
                and "__pycache__" not in f.parts
            )
        elif p.is_file():
            files.append(p)

    for f in files:
        try:
            rel = str(f.relative_to(root))
        except ValueError:
            rel = f.name
        for finding in scan_file(f, rel):
            if finding.pattern_id in wanted:
                findings.append((rel, finding.pattern_id, finding.match))

    return findings


# ---------------------------------------------------------------------------
# Mirroring the curated subset into / out of the staging dir
# ---------------------------------------------------------------------------

def _mirror_into_staging(home: Path, staging: Path) -> List[str]:
    """Copy the curated subset from the live profile into *staging*.

    Returns the list of relative paths that were written. Stale tracked files
    that no longer exist in the profile are removed from the staging mirror so
    deletions sync too.
    """
    written: List[str] = []

    for rel, kind in SYNCED_PATHS:
        src = home / rel
        dst = staging / rel
        if kind == "tree":
            if dst.exists():
                shutil.rmtree(dst)
            if src.is_dir():
                # Only exclude VCS metadata and build junk. Do NOT use broad
                # globs like ``*cache*`` / ``*.lock`` here: those would silently
                # drop legitimately-named user skills/files (e.g. a skill called
                # ``response-cache-helper`` or a ``cache_notes.md``) from the
                # backup, causing silent data loss on restore. Secret/ephemeral
                # exclusion is the .gitignore's job at the git layer, where
                # ``*cache*/`` (trailing slash) correctly matches only dirs.
                shutil.copytree(
                    src, dst,
                    # symlinks=True copies links AS links rather than
                    # dereferencing them. Without this, a symlink inside a skill
                    # (e.g. ``key -> ~/.ssh/id_rsa``) would have its TARGET copied
                    # into the repo — a secret leak the name-based .gitignore
                    # can't catch. As a link, git stores only the target path text.
                    symlinks=True,
                    ignore=shutil.ignore_patterns(
                        ".git", "__pycache__", "*.pyc", ".DS_Store",
                    ),
                )
                written.append(rel)
        else:  # file
            if dst.exists():
                dst.unlink()
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                written.append(rel)

    # Allow-listed config subset → generated sync-config.yaml
    from hermes_cli.config import load_config
    import yaml

    subset = build_synced_config_subset(load_config())
    sync_cfg_path = staging / SYNC_CONFIG_NAME
    if subset:
        sync_cfg_path.write_text(
            "# Generated by `hermes sync` — allow-listed config keys ONLY.\n"
            "# The full config.yaml is NEVER synced (it contains secrets).\n"
            + yaml.safe_dump(subset, sort_keys=True),
            encoding="utf-8",
        )
        written.append(SYNC_CONFIG_NAME)
    elif sync_cfg_path.exists():
        sync_cfg_path.unlink()

    return written


def _restore_from_staging(home: Path, staging: Path, *, force: bool) -> List[str]:
    """Copy the curated subset from *staging* back into the live profile.

    Last-writer-wins. Without *force*, prompts before overwriting a profile
    file whose content differs from what's incoming.
    """
    restored: List[str] = []

    def _confirm_overwrite(label: str) -> bool:
        if force:
            return True
        if not _has_tty():
            # No interactive terminal (slash command): don't hang on input().
            # Conservative default — skip the overwrite and tell the user how to
            # force it. Real local content is never clobbered silently.
            _warn(f"  Local {label} differs from remote; skipping "
                  "(no terminal to confirm — re-run `hermes sync pull --force` to overwrite).")
            return False
        try:
            resp = input(f"  Local {label} differs from remote. Overwrite? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return resp in {"y", "yes"}

    def _dir_has_user_content(d: Path) -> bool:
        """True if *d* holds any real file (ignoring VCS/build junk).

        A fresh Hermes profile pre-creates an EMPTY ``skills/`` directory at
        startup, so ``dst.exists()`` is True on a brand-new machine. Treating
        that empty dir as "differs — ask permission" caused the first ``pull``
        on a new device to SKIP restoring skills entirely (non-interactive
        stdin auto-answers no). An empty local dir is not user data worth
        protecting, so we overwrite it without prompting.
        """
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts or p.suffix == ".pyc" or p.name == ".DS_Store":
                continue
            return True
        return False

    def _file_has_user_content(f: Path) -> bool:
        """True if *f* exists with real user content (not a blank/default stub).

        A fresh Hermes profile seeds a default SOUL.md (the canonical persona
        template). That default is NOT user data worth protecting, so a first
        pull on a new device should overwrite it with the synced persona
        without prompting — otherwise the keynote "new laptop, my Hermes is
        back" flow silently keeps the stock persona.
        """
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        stripped = text.strip()
        if not stripped:
            return False
        # Treat the canonical default SOUL.md template as an overwritable stub.
        try:
            from hermes_cli.default_soul import DEFAULT_SOUL_MD

            if stripped == DEFAULT_SOUL_MD.strip():
                return False
        except Exception:
            pass
        return True

    for rel, kind in SYNCED_PATHS:
        src = staging / rel
        dst = home / rel
        if kind == "tree":
            if not src.is_dir():
                continue
            # Only guard against clobbering REAL local content. An empty
            # freshly-scaffolded dir is overwritten silently.
            if _dir_has_user_content(dst) and not force:
                if not _dirs_equal(src, dst) and not _confirm_overwrite(f"{rel}/"):
                    _warn(f"  Skipped {rel}/")
                    continue
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            restored.append(rel)
        else:
            if not src.is_file():
                continue
            # Only prompt if the local file has REAL content that differs —
            # a blank/default stub is overwritten silently.
            if _file_has_user_content(dst) and not force:
                if dst.read_bytes() != src.read_bytes() and not _confirm_overwrite(rel):
                    _warn(f"  Skipped {rel}")
                    continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(rel)

    # sync-config.yaml → merge allow-listed keys back into config.yaml
    sync_cfg_path = staging / SYNC_CONFIG_NAME
    if sync_cfg_path.is_file():
        import yaml
        from hermes_cli.config import load_config, save_config

        incoming = yaml.safe_load(sync_cfg_path.read_text(encoding="utf-8")) or {}
        cfg = load_config()
        applied = False
        for dotted in CONFIG_ALLOWLIST:
            val = _get_nested(incoming, dotted)
            if val is not None:
                _set_nested(cfg, dotted, val)
                applied = True
        if applied:
            save_config(cfg)
            restored.append(f"config.yaml ({', '.join(CONFIG_ALLOWLIST)})")

    return restored


def _dirs_equal(a: Path, b: Path) -> bool:
    """Cheap structural+content comparison of two trees."""
    a_files = {str(p.relative_to(a)): p for p in a.rglob("*") if p.is_file()}
    b_files = {str(p.relative_to(b)): p for p in b.rglob("*") if p.is_file()}
    if set(a_files) != set(b_files):
        return False
    for rel, pa in a_files.items():
        if pa.read_bytes() != b_files[rel].read_bytes():
            return False
    return True


# ---------------------------------------------------------------------------
# Repo bootstrap
# ---------------------------------------------------------------------------

def _ensure_staging_repo(staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    if not (staging / ".git").exists():
        _git(["init"], cwd=staging)
        # Use a stable default branch name regardless of the host git config.
        _git(["symbolic-ref", "HEAD", "refs/heads/main"], cwd=staging, check=False)
    gi = staging / ".gitignore"
    if not gi.exists():
        gi.write_text(GITIGNORE_CONTENT, encoding="utf-8")


def _configured_remote(staging: Path) -> Optional[str]:
    res = _git(["remote", "get-url", "origin"], cwd=staging, check=False)
    if res.returncode == 0:
        return (res.stdout or "").strip()
    return None


def _set_remote(staging: Path, url: str) -> None:
    existing = _configured_remote(staging)
    if existing is None:
        _git(["remote", "add", "origin", url], cwd=staging)
    else:
        _git(["remote", "set-url", "origin", url], cwd=staging)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_sync_init(args) -> None:
    if not _git_available():
        _err("git is not installed or not on PATH. Install git first.")
        raise SystemExit(1)

    home = get_hermes_home()
    staging = _sync_dir()

    _print("\n[bold]hermes sync init[/bold] — back up your profile to a git repo you own.\n")

    # ── Visibility: PRIVATE is the safety default ────────────────────────────
    visibility = getattr(args, "visibility", None)
    if visibility not in {"private", "public"}:
        if not _has_tty():
            # Non-interactive (slash command / no TTY): default to the safe
            # choice rather than hang on input(). User can pass --public to override.
            visibility = "private"
            _print("No interactive terminal — defaulting to a PRIVATE repo. "
                   "(Pass --public to override.)")
        else:
            _print("Repository visibility:")
            _print("  [bold]1. private[/bold]  (default — recommended; your profile stays yours)")
            _print("  2. public   (anyone can read it; only do this if you intend to share)")
            try:
                choice = input("Choice [1-2] (default 1 = private): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                _warn("Cancelled.")
                return
            visibility = "public" if choice == "2" else "private"

    if visibility == "public":
        _warn(
            "\n⚠  PUBLIC repo selected. Anything pushed is world-readable. "
            "The pre-push secret scan still runs, but double-check your skills "
            "contain no embedded credentials.\n"
        )

    _ensure_staging_repo(staging)

    # ── Remote: gh-driven create, or paste an existing URL ───────────────────
    remote_url = getattr(args, "remote", None)
    if not remote_url:
        default_name = _default_repo_name()
        if not _has_tty():
            # No TTY: we can still auto-create via gh non-interactively if it's
            # available; otherwise we cannot prompt for a URL — fail clearly.
            if _gh_available():
                remote_url = _gh_create_repo(default_name, visibility, staging)
                if remote_url is None:
                    _err("gh repo create failed and no interactive terminal to paste a URL. "
                         "Re-run in a terminal, or pass `--remote <git-url>`.")
                    raise SystemExit(1)
            else:
                _err("No interactive terminal and gh CLI not available. "
                     "Pass `--remote <git-url>` (e.g. git@github.com:you/hermes-profile.git).")
                raise SystemExit(1)
        elif _gh_available():
            _print(f"GitHub CLI (gh) detected. A repo can be created for you.")
            try:
                name = input(f"Repository name [{default_name}]: ").strip() or default_name
                use_gh = input(f"Create '{name}' via gh as {visibility}? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                _warn("Cancelled.")
                return
            if use_gh in {"", "y", "yes"}:
                remote_url = _gh_create_repo(name, visibility, staging)
                if remote_url is None:
                    _err("gh repo create failed. Re-run and paste an existing remote URL instead.")
                    raise SystemExit(1)
        if not remote_url and _has_tty():
            try:
                remote_url = input(
                    "Paste the git remote URL to push to "
                    "(e.g. git@github.com:you/hermes-profile.git): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                _warn("Cancelled.")
                return
            if not remote_url:
                _err("No remote URL provided. Aborting.")
                raise SystemExit(1)

    if remote_url:
        _set_remote(staging, remote_url)

    _write_sync_config({
        "remote": remote_url,
        "visibility": visibility,
        "initialized_at": datetime.now(timezone.utc).isoformat(),
    })

    _ok(f"\n✓ Sync initialized ({visibility}).")
    _print(f"  Staging repo: {staging}")
    _print(f"  Remote:       {remote_url}")
    _print(f"  .gitignore:   excludes .env, auth.json, config.yaml, *.db, logs/, caches, ...")
    _print("\nNext: [bold]hermes sync push[/bold] to back up your profile.\n")


def _default_repo_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        profile = get_active_profile_name() or "default"
    except Exception:
        profile = "default"
    if profile in {"", "default"}:
        return "hermes-profile"
    return f"hermes-profile-{profile}"


def _gh_create_repo(name: str, visibility: str, staging: Path) -> Optional[str]:
    """Create the repo via gh and wire it as origin. Returns the remote URL."""
    flag = "--private" if visibility == "private" else "--public"
    res = subprocess.run(
        ["gh", "repo", "create", name, flag, "--source", str(staging),
         "--remote", "origin", "--disable-wiki"],
        text=True,
        capture_output=True,
    )
    if res.returncode != 0:
        _err((res.stderr or res.stdout or "gh error").strip())
        return None
    return _configured_remote(staging)


def cmd_sync_push(args) -> None:
    if not _git_available():
        _err("git is not installed or not on PATH.")
        raise SystemExit(1)

    sync_cfg = _read_sync_config()
    remote = sync_cfg.get("remote")
    if not remote:
        _err("Sync is not initialized. Run `hermes sync init` first.")
        raise SystemExit(1)

    home = get_hermes_home()
    staging = _sync_dir()
    _ensure_staging_repo(staging)
    _set_remote(staging, remote)

    _print("\n[bold]hermes sync push[/bold]\n")
    written = _mirror_into_staging(home, staging)
    if not written:
        _warn("Nothing to sync (no skills/memory/persona found).")
        return

    # ── Pre-push secret scan (load-bearing) ──────────────────────────────────
    # Scan the ENTIRE staging working tree, not just the curated SYNCED_PATHS.
    # ``git add -A`` below stages everything that isn't .gitignored, so the scan
    # must cover the same surface — otherwise a stale file left in the persistent
    # staging dir (e.g. a path dropped from SYNCED_PATHS in a future version, or
    # a manual drop) could be committed without ever being scanned. Scanning the
    # whole tree closes that gap. ``.git`` is skipped inside scan_paths_for_secrets.
    findings = scan_paths_for_secrets([staging], staging)
    if findings:
        _err("\n✗ ABORTING PUSH — secret-shaped content detected in the staged files:\n")
        for rel, pid, snippet in findings[:25]:
            _print(f"    [red]{rel}[/red]  ({pid})", style=None)
            _print(f"      {snippet}", style="dim")
        _print(
            "\nRemove the embedded secret (use an env var / .env reference instead) "
            "and re-run `hermes sync push`. Nothing was pushed.",
            style="yellow",
        )
        raise SystemExit(1)

    _git(["add", "-A"], cwd=staging)
    status = _git(["status", "--porcelain"], cwd=staging)
    if not (status.stdout or "").strip():
        _ok("✓ Already up to date — nothing changed since last push.")
        _write_sync_config({"last_sync_at": datetime.now(timezone.utc).isoformat()})
        return

    msg = getattr(args, "message", None) or (
        f"hermes sync: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    commit = _git(["commit", "-m", msg], cwd=staging, check=False)
    if commit.returncode != 0:
        _err("\n✗ git commit failed:")
        _print((commit.stderr or commit.stdout or "").strip(), style="dim")
        _print(
            "If this is a brand-new machine, git may need an identity: "
            "set `git config --global user.email you@example.com` and "
            "`git config --global user.name \"Your Name\"`, then re-run.",
            style="yellow",
        )
        raise SystemExit(1)

    branch = _current_branch(staging)
    push = _git(["push", "-u", "origin", branch], cwd=staging, check=False)
    if push.returncode != 0:
        _err("\n✗ git push failed:")
        _print((push.stderr or push.stdout or "").strip(), style="dim")
        raise SystemExit(1)

    _write_sync_config({"last_sync_at": datetime.now(timezone.utc).isoformat()})
    _ok(f"\n✓ Pushed {len(written)} item(s) to {remote} ({branch}).")


def cmd_sync_pull(args) -> None:
    if not _git_available():
        _err("git is not installed or not on PATH.")
        raise SystemExit(1)

    sync_cfg = _read_sync_config()
    remote = sync_cfg.get("remote")
    if not remote:
        _err("Sync is not initialized. Run `hermes sync init` first.")
        raise SystemExit(1)

    home = get_hermes_home()
    staging = _sync_dir()
    _ensure_staging_repo(staging)
    _set_remote(staging, remote)

    force = bool(getattr(args, "force", False))

    _print("\n[bold]hermes sync pull[/bold]\n")
    branch = _current_branch(staging)
    fetch = _git(["fetch", "origin"], cwd=staging, check=False)
    if fetch.returncode != 0:
        _err("git fetch failed:")
        _print((fetch.stderr or fetch.stdout or "").strip(), style="dim")
        raise SystemExit(1)

    # Last-writer-wins on the git side: hard-reset the staging mirror to origin.
    reset = _git(["reset", "--hard", f"origin/{branch}"], cwd=staging, check=False)
    if reset.returncode != 0:
        _err(f"Could not reset to origin/{branch}. Has anything been pushed yet?")
        _print((reset.stderr or reset.stdout or "").strip(), style="dim")
        raise SystemExit(1)

    restored = _restore_from_staging(home, staging, force=force)
    _write_sync_config({"last_sync_at": datetime.now(timezone.utc).isoformat()})
    if restored:
        _ok(f"\n✓ Pulled and updated {len(restored)} item(s) from {remote}.")
        for r in restored:
            _print(f"    {r}", style="dim")
    else:
        _ok("✓ Up to date — nothing to update locally.")


def cmd_sync_status(args) -> None:
    sync_cfg = _read_sync_config()
    remote = sync_cfg.get("remote")
    visibility = sync_cfg.get("visibility", "private")
    last_sync = sync_cfg.get("last_sync_at")

    _print("\n[bold]hermes sync status[/bold]\n")
    if not remote:
        _warn("Not initialized. Run `hermes sync init` to set up profile sync.")
        return

    _print(f"  Remote:       {remote}")
    vis_style = "yellow" if visibility == "public" else "green"
    _print(f"  Visibility:   [{vis_style}]{visibility}[/{vis_style}]")
    _print(f"  Last sync:    {last_sync or 'never'}")

    if not _git_available():
        _warn("\n  git not available — cannot compute pending changes.")
        return

    home = get_hermes_home()
    staging = _sync_dir()
    if not (staging / ".git").exists():
        _warn("\n  Staging repo missing — run `hermes sync push` to create it.")
        return
    _set_remote(staging, remote)

    # What would be pushed: mirror current profile state, then diff.
    written = _mirror_into_staging(home, staging)
    _git(["add", "-A"], cwd=staging)
    status = _git(["status", "--porcelain"], cwd=staging)
    pending = [ln for ln in (status.stdout or "").splitlines() if ln.strip()]

    _print(f"\n  Synced subset: {', '.join(rel for rel, _ in SYNCED_PATHS)}, "
           f"{SYNC_CONFIG_NAME}")
    if pending:
        _warn(f"\n  {len(pending)} change(s) pending push:")
        for ln in pending[:40]:
            _print(f"    {ln}", style="dim")
    else:
        _ok("\n  Clean — local profile matches last push.")

    # Behind/ahead vs remote
    branch = _current_branch(staging)
    _git(["fetch", "origin"], cwd=staging, check=False)
    rev = _git(["rev-list", "--left-right", "--count",
                f"origin/{branch}...HEAD"], cwd=staging, check=False)
    if rev.returncode == 0 and rev.stdout.strip():
        try:
            behind, ahead = rev.stdout.split()
            if int(behind) or int(ahead):
                _print(f"\n  vs origin/{branch}: {ahead} ahead, {behind} behind", style="dim")
        except ValueError:
            pass


def _export_skills_zip(home: Path, skill_names: List[str], out_path: Path) -> List[str]:
    """Export one or more skills into a zip at *out_path*.

    Returns the list of skill names written. Raises SystemExit(1) if any named
    skill is missing or trips the secret scanner — a zip is just as capable of
    leaking a credential as a git push, so the same pre-export scan applies.

    The zip is self-contained: each skill is stored under ``skills/<name>/`` so
    a recipient can drop it straight into their ``~/.hermes/skills/``. No git
    remote or sync init is required — this works for fully offline sharing.
    """
    import tempfile
    import zipfile

    skills_root = home / "skills"
    resolved: List[Tuple[str, Path]] = []
    for name in skill_names:
        sp = skills_root / name
        if not sp.is_dir():
            _err(f"Skill '{name}' not found at {sp}")
            raise SystemExit(1)
        resolved.append((name, sp))

    # Stage into a temp dir (so the secret scan sees exactly what will be
    # zipped, with symlinks preserved as links — never dereferenced).
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        for name, sp in resolved:
            shutil.copytree(
                sp, stage / name,
                symlinks=True,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".DS_Store"),
            )
        findings = scan_paths_for_secrets([stage], stage)
        if findings:
            _err("\n✗ ABORTING EXPORT — secret-shaped content in the skill(s):")
            for rel, pid, snippet in findings[:25]:
                _print(f"    {rel}  ({pid}): {snippet}", style="dim")
            _print(
                "\nRemove the embedded secret (use an env var / .env reference instead) "
                "and re-run. Nothing was written.",
                style="yellow",
            )
            raise SystemExit(1)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, _sp in resolved:
                base = stage / name
                for f in sorted(base.rglob("*")):
                    if f.is_symlink() or not f.is_file():
                        continue
                    # Arcname: skills/<name>/...  → drop-in for ~/.hermes/skills/
                    arc = Path("skills") / name / f.relative_to(base)
                    zf.write(f, arcname=str(arc))

    return [name for name, _ in resolved]


def cmd_sync_share(args) -> None:
    skill_names = getattr(args, "skill_name", None)
    # argparse may pass a single name (nargs default) or a list (nargs="+").
    if isinstance(skill_names, str):
        skill_names = [skill_names]
    if not skill_names:
        _err("Usage: hermes sync share <skill-name> [<skill-name> ...] [--zip [PATH]]")
        raise SystemExit(2)

    home = get_hermes_home()

    # ── Zip export path — fully standalone, NO sync init / remote required ────
    zip_arg = getattr(args, "zip", None)
    if zip_arg is not None:
        # --zip with no value → default name; --zip PATH → that path.
        if zip_arg is True or zip_arg == "":
            if len(skill_names) == 1:
                default = f"{skill_names[0]}.zip"
            else:
                default = "hermes-skills.zip"
            out_path = Path.cwd() / default
        else:
            out_path = Path(zip_arg).expanduser()
            if out_path.is_dir():
                out_path = out_path / "hermes-skills.zip"
        _print(f"\n[bold]Export skill(s) as zip:[/bold] {', '.join(skill_names)}\n")
        written = _export_skills_zip(home, skill_names, out_path)
        size_kb = out_path.stat().st_size / 1024
        _ok(f"✓ Exported {len(written)} skill(s) → {out_path} ({size_kb:.1f} KB)")
        _print(
            "\nRecipients unzip into their profile:\n"
            f"    [bold]unzip {out_path.name} -d ~/.hermes/[/bold]\n"
            "(the archive already contains skills/<name>/ paths)",
        )
        return

    # ── Git-share path (existing behaviour) — requires sync init ─────────────
    # Only single-skill git-share is supported; group sharing is zip-only.
    if len(skill_names) > 1:
        _err("Sharing multiple skills at once is only supported with --zip. "
             "For git-share, run `hermes sync share <one-skill>` per skill.")
        raise SystemExit(2)
    skill_name = skill_names[0]

    sync_cfg = _read_sync_config()
    remote = sync_cfg.get("remote")
    visibility = sync_cfg.get("visibility", "private")
    if not remote:
        _err("Sync is not initialized. Run `hermes sync init` first, "
             "or use `--zip` to export without a remote.")
        raise SystemExit(1)

    skill_path = home / "skills" / skill_name
    if not skill_path.is_dir():
        _err(f"Skill '{skill_name}' not found at {skill_path}")
        raise SystemExit(1)

    github_url = _remote_to_github_url(remote)

    _print(f"\n[bold]Share skill:[/bold] {skill_name}\n")

    if visibility == "private":
        _warn("⚠  Your sync repo is PRIVATE. Recipients need read access to the "
              "repo before they can install this skill (or use `--zip` to send a file).\n")

    if github_url:
        # Mirror tools/skills_hub.py install-command format exactly:
        #   npx skills add <github-url> --skill <skill-name>
        install_cmd = f"npx skills add {github_url} --skill {skill_name}"
        _print("Others can install this skill with:\n")
        _print(f"    [bold]{install_cmd}[/bold]\n")
    else:
        _warn(f"Remote '{remote}' is not a github.com URL — the `npx skills add` "
              "installer expects a GitHub repo. Share the repo URL directly, "
              "or use `--zip`.\n")

    if getattr(args, "push", False):
        staging = _sync_dir()
        _ensure_staging_repo(staging)
        _set_remote(staging, remote)
        # Mirror only this skill into staging, then scan + commit + push.
        dst = staging / "skills" / skill_name
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            skill_path, dst,
            symlinks=True,  # copy links as links, never deref (see _mirror_into_staging)
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".DS_Store"),
        )
        findings = scan_paths_for_secrets([dst], staging)
        if findings:
            _err("\n✗ ABORTING — secret-shaped content in the skill:")
            for rel, pid, snippet in findings[:25]:
                _print(f"    {rel}  ({pid}): {snippet}", style="dim")
            raise SystemExit(1)
        _git(["add", "--", f"skills/{skill_name}"], cwd=staging)
        status = _git(["status", "--porcelain"], cwd=staging)
        if (status.stdout or "").strip():
            commit = _git(["commit", "-m", f"hermes sync: share skill {skill_name}"],
                          cwd=staging, check=False)
            if commit.returncode != 0:
                _err("git commit failed:")
                _print((commit.stderr or commit.stdout or "").strip(), style="dim")
                _print(
                    "If this is a brand-new machine, set git identity: "
                    "`git config --global user.email you@example.com`.",
                    style="yellow",
                )
                raise SystemExit(1)
            branch = _current_branch(staging)
            push = _git(["push", "-u", "origin", branch], cwd=staging, check=False)
            if push.returncode != 0:
                _err("git push failed:")
                _print((push.stderr or push.stdout or "").strip(), style="dim")
                raise SystemExit(1)
            _ok(f"✓ Pushed skill '{skill_name}'.")
        else:
            _ok("✓ Skill already up to date in the sync repo.")


# ---------------------------------------------------------------------------
# Small git helpers
# ---------------------------------------------------------------------------

def _current_branch(staging: Path) -> str:
    res = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=staging, check=False)
    branch = (res.stdout or "").strip()
    if not branch or branch == "HEAD":
        return "main"
    return branch


def _remote_to_github_url(remote: str) -> Optional[str]:
    """Normalize a git remote (ssh or https) to an ``https://github.com/...`` URL."""
    r = remote.strip()
    if r.startswith("git@github.com:"):
        path = r[len("git@github.com:"):]
        if path.endswith(".git"):
            path = path[: -len(".git")]
        return f"https://github.com/{path}"
    if r.startswith("https://github.com/") or r.startswith("http://github.com/"):
        if r.endswith(".git"):
            r = r[: -len(".git")]
        return r.replace("http://", "https://", 1)
    if r.startswith("ssh://git@github.com/"):
        path = r[len("ssh://git@github.com/"):]
        if path.endswith(".git"):
            path = path[: -len(".git")]
        return f"https://github.com/{path}"
    return None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def cmd_sync(args) -> None:
    """Top-level dispatcher for ``hermes sync [subcommand]``."""
    sub = getattr(args, "sync_command", None)
    if sub == "init":
        cmd_sync_init(args)
    elif sub == "push":
        cmd_sync_push(args)
    elif sub == "pull":
        cmd_sync_pull(args)
    elif sub in {None, "", "status"}:
        cmd_sync_status(args)
    elif sub == "share":
        cmd_sync_share(args)
    else:
        _err(f"Unknown sync subcommand: {sub}")
        _print("Use one of: init, push, pull, status, share")
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# Slash command entry point (/sync in chat — CLI / TUI / desktop)
# ---------------------------------------------------------------------------

class _SlashArgs:
    """Lightweight args shim so the cmd_sync_* handlers can be driven from a
    parsed slash string instead of an argparse Namespace."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def handle_sync_slash(cmd: str) -> None:
    """Parse and dispatch ``/sync <subcommand> [args]`` from the chat interface.

    Mirrors the argparse surface (init/push/pull/status/share) but driven by a
    slash string. The underlying handlers are TTY-aware: in a no-terminal chat
    context they default to safe choices or print a clear "re-run with a flag"
    message rather than hanging on input(). Examples:

        /sync                 → status
        /sync status
        /sync push
        /sync push -m "backup before refactor"
        /sync pull            (skips conflicting files unless --force)
        /sync pull --force
        /sync init --public --remote git@github.com:me/repo.git
        /sync share my-skill --zip
        /sync share a b c --zip team.zip
    """
    import shlex

    try:
        parts = shlex.split(cmd.strip())
    except ValueError:
        parts = cmd.strip().split()
    if parts and parts[0].lower() == "/sync":
        parts = parts[1:]

    sub = parts[0].lower() if parts else "status"
    rest = parts[1:]

    def _flag(*names) -> bool:
        return any(n in rest for n in names)

    def _opt(*names):
        for i, a in enumerate(rest):
            if a in names and i + 1 < len(rest):
                return rest[i + 1]
        return None

    if sub == "status":
        cmd_sync_status(_SlashArgs())
    elif sub == "push":
        cmd_sync_push(_SlashArgs(message=_opt("-m", "--message")))
    elif sub == "pull":
        cmd_sync_pull(_SlashArgs(force=_flag("--force")))
    elif sub == "init":
        visibility = "public" if _flag("--public") else ("private" if _flag("--private") else None)
        cmd_sync_init(_SlashArgs(visibility=visibility, remote=_opt("--remote")))
    elif sub == "share":
        names = [a for a in rest if not a.startswith("-")]
        # drop an -o/--zip value from the name list
        zip_val = _opt("--zip", "-o", "--output")
        if zip_val:
            names = [n for n in names if n != zip_val]
        zip_present = _flag("--zip", "-o", "--output")
        cmd_sync_share(_SlashArgs(
            skill_name=names,
            zip=(zip_val if zip_val else (True if zip_present else None)),
            push=_flag("--push"),
        ))
    elif sub in {"help", "--help", "-h"}:
        _print("Usage: /sync [status|push|pull|init|share]")
    else:
        _err(f"Unknown sync subcommand: {sub}")
        _print("Use one of: status, push, pull, init, share")
