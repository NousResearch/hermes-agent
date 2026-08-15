"""Welcome banner, ASCII art, skills summary, and update check for the CLI.

Pure display functions with no HermesCLI state dependency.
"""
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
from hermes_constants import get_hermes_home
from hermes_cli._subprocess_compat import windows_hide_flags
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# rich and prompt_toolkit are imported lazily (inside the functions that use
# them) rather than at module level.  Importing this module is on the TUI
# gateway's critical startup path purely to reach the lightweight update-check
# helpers (``prefetch_update_check``); pulling rich.console + prompt_toolkit
# eagerly added ~50ms of wasted imports before ``gateway.ready`` could fire.
# Keep the type-only reference available to checkers without the runtime cost.
if TYPE_CHECKING:
    from rich.console import Console

logger = logging.getLogger(__name__)


# =========================================================================
# ANSI building blocks for conversation display
# =========================================================================

_GOLD = "\033[1;38;2;255;215;0m"  # True-color #FFD700 bold
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"


def cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's renderer."""
    from prompt_toolkit import print_formatted_text as _pt_print
    from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
    try:
        _pt_print(_PT_ANSI(text))
    except Exception:
        # prompt_toolkit needs a real console. On Windows, a redirected or
        # absent stdout (pythonw.exe, CI, `hermes ... > file`) raises
        # NoConsoleScreenBufferError from its Win32Output — display helpers
        # must never crash the caller over that, so degrade to plain print.
        print(text)


# =========================================================================
# Skin-aware color helpers
# =========================================================================

def _skin_color(key: str, fallback: str) -> str:
    """Get a color from the active skin, or return fallback."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        return get_active_skin().get_color(key, fallback)
    except Exception:
        return fallback
# =========================================================================
# ASCII Art & Branding
# =========================================================================

from hermes_cli import __version__ as VERSION, __release_date__ as RELEASE_DATE

HERMES_AGENT_LOGO = """[bold #FFD700]██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #FFD700]██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#FFBF00]███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#FFBF00]██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#CD7F32]██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#CD7F32]╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]"""

HERMES_CADUCEUS = """[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⣀⣀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣇⠸⣿⣿⠇⣸⣿⣿⣷⣦⣄⡀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⢀⣠⣴⣶⠿⠋⣩⡿⣿⡿⠻⣿⡇⢠⡄⢸⣿⠟⢿⣿⢿⣍⠙⠿⣶⣦⣄⡀⠀[/]
[#FFBF00]⠀⠀⠉⠉⠁⠶⠟⠋⠀⠉⠀⢀⣈⣁⡈⢁⣈⣁⡀⠀⠉⠀⠙⠻⠶⠈⠉⠉⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⡿⠛⢁⡈⠛⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⣿⣦⣤⣈⠁⢠⣴⣿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠻⢿⣿⣦⡉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣦⣈⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⠦⠈⠙⠿⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣤⡈⠁⢤⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠷⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠑⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⢰⡆⠈⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⠈⣡⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]"""



# =========================================================================
# Skills scanning
# =========================================================================

_available_skills_cache: Optional[tuple] = None  # (result,) once computed


def get_available_skills() -> Dict[str, List[str]]:
    """Return skills grouped by category, filtered by platform and disabled state.

    Delegates to ``_find_all_skills()`` from ``tools/skills_tool`` which already
    handles platform gating (``platforms:`` frontmatter) and respects the
    user's ``skills.disabled`` config list.

    Cached per-process: this feeds only the startup banner, whose snapshot
    is taken once anyway, and the underlying skills-tree walk costs ~100ms.
    ``prefetch_banner_data()`` uses the cache to pay that walk off-thread.
    """
    global _available_skills_cache
    if _available_skills_cache is not None:
        return _available_skills_cache[0]
    try:
        from tools.skills_tool import _find_all_skills
        all_skills = _find_all_skills()  # already filtered
    except Exception:
        return {}

    skills_by_category: Dict[str, List[str]] = {}
    for skill in all_skills:
        category = skill.get("category") or "general"
        skills_by_category.setdefault(category, []).append(skill["name"])
    _available_skills_cache = (skills_by_category,)
    return skills_by_category


# =========================================================================
# Update check
# =========================================================================

# Cache update check results for 6 hours to avoid repeated git fetches
_UPDATE_CHECK_CACHE_SECONDS = 6 * 3600

# Sentinel returned when we know an update exists but can't count commits
# (e.g. nix-built hermes — no local git history to count against).
UPDATE_AVAILABLE_NO_COUNT = -1

_UPSTREAM_REPO_URL = "https://github.com/NousResearch/hermes-agent.git"
_OFFICIAL_REPO_CANONICAL = "github.com/nousresearch/hermes-agent"


def _canonical_github_remote(url: str | None) -> str:
    """Return ``host/owner/repo`` for common GitHub remote URL forms."""
    if not url:
        return ""
    value = url.strip()
    if value.startswith("git@github.com:"):
        value = "github.com/" + value[len("git@github.com:"):]
    elif value.startswith("ssh://git@github.com/"):
        value = "github.com/" + value[len("ssh://git@github.com/"):]
    else:
        parsed = urlparse(value)
        if parsed.netloc and parsed.path:
            value = f"{parsed.netloc}{parsed.path}"
    value = value.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    return value.lower()


def _is_ssh_remote(url: str | None) -> bool:
    if not url:
        return False
    value = url.strip().lower()
    return value.startswith("git@") or value.startswith("ssh://")


def _is_official_ssh_remote(url: str | None) -> bool:
    return _is_ssh_remote(url) and _canonical_github_remote(url) == _OFFICIAL_REPO_CANONICAL


def _git_stdout(args: list[str], *, cwd: Path, timeout: int = 5) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            # git output is UTF-8; on Windows text=True defaults to the ANSI
            # code page and bytes like 0x90 (3rd byte of 🐛 in a commit
            # subject) crash the stdlib reader thread (#52649).
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(cwd),
            creationflags=windows_hide_flags(),
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def _github_compare_behind(current_rev: str, target_rev: str) -> Optional[int]:
    """Exact behind-count via the GitHub compare API for uncountable graphs.

    Shallow installer clones and ls-remote-only probes know the two tip SHAs
    but have no local history to run ``rev-list --count`` across. GitHub's
    ``GET /repos/<owner>/<repo>/compare/<current>...<target>`` knows the full
    graph regardless of local clone depth and returns ``ahead_by`` — exactly
    the behind count the local graph lost. Unauthenticated, bounded, and
    best-effort: any failure (offline, rate limit, diverged/unknown SHAs)
    returns None so callers keep the honest UPDATE_AVAILABLE_NO_COUNT.
    """
    if not (_is_full_sha(current_rev) and _is_full_sha(target_rev)):
        return None
    url = (
        "https://api.github.com/repos/nousresearch/hermes-agent/"
        f"compare/{current_rev}...{target_rev}"
    )
    try:
        import urllib.request

        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                # api.github.com 403s requests without a User-Agent.
                "User-Agent": "hermes-cli-update-check",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    ahead = payload.get("ahead_by") if isinstance(payload, dict) else None
    if isinstance(ahead, int) and not isinstance(ahead, bool) and ahead >= 0:
        return ahead
    return None


def _is_full_sha(value: Optional[str]) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(c in "0123456789abcdefABCDEF" for c in value)
    )


def _upstream_main_sha() -> Optional[str]:
    """Tip SHA of upstream main via HTTPS ls-remote (no auth, no prompts)."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", _UPSTREAM_REPO_URL, "refs/heads/main"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10,
            creationflags=windows_hide_flags(),
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    upstream_rev = result.stdout.split()[0]
    return upstream_rev or None


def _check_via_rev(local_rev: str) -> Optional[int]:
    """Compare an embedded git revision to upstream main via ls-remote.

    Returns 0 if up-to-date, the exact behind-count when the GitHub compare
    API can recover it, ``UPDATE_AVAILABLE_NO_COUNT`` if behind by an unknown
    amount, or ``None`` on failure.
    """
    upstream_rev = _upstream_main_sha()
    if not upstream_rev:
        return None
    if upstream_rev == local_rev:
        return 0
    # Behind, but ls-remote only knows tip SHAs. Try to recover the exact
    # count from the GitHub compare API before falling back to the sentinel.
    # ahead_by == 0 with differing tips means the remote tip is reachable from
    # our HEAD — a local-ahead checkout, i.e. NOT behind.
    counted = _github_compare_behind(local_rev, upstream_rev)
    return counted if counted is not None else UPDATE_AVAILABLE_NO_COUNT


def _check_via_local_git(repo_dir: Path) -> Optional[int]:
    """Count commits behind origin/main in a local checkout."""
    origin_url = _git_stdout(["remote", "get-url", "origin"], cwd=repo_dir)
    if _is_official_ssh_remote(origin_url):
        head_rev = _git_stdout(["rev-parse", "HEAD"], cwd=repo_dir)
        if not head_rev:
            return None
        # Passive probe via HTTPS ls-remote (never SSH — no hardware-key
        # prompts). Tip SHAs alone can't distinguish "behind" from a local
        # carried commit sitting AHEAD of origin/main, and misreporting an
        # ahead checkout as behind nudges the user into `hermes update`,
        # which can wipe their carried work.
        upstream_rev = _upstream_main_sha()
        if upstream_rev is None:
            return None
        if upstream_rev == head_rev:
            return 0
        # Local-ahead: the remote tip is an ancestor of HEAD. Checked against
        # the FRESH upstream SHA (not the possibly stale origin/main tracking
        # ref) so a stale ref can't fake an up-to-date report.
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", upstream_rev, "HEAD"],
            capture_output=True, timeout=5, cwd=str(repo_dir),
        )
        if ancestor.returncode == 0:
            return 0
        # Genuinely behind (or diverged). Recover the exact count via the
        # GitHub compare API; a local-only HEAD 404s there, which safely
        # degrades to the honest no-count sentinel — never a fabricated 1.
        counted = _github_compare_behind(head_rev, upstream_rev)
        return counted if counted is not None else UPDATE_AVAILABLE_NO_COUNT

    # Installer checkouts are shallow (`git clone --depth 1`). On a shallow
    # clone the history stops at a single commit, so a plain `git fetch` would
    # unshallow the repo (dragging in the whole history) and
    # `rev-list --count HEAD..origin/main` would report a huge bogus "behind"
    # number (e.g. "12492 commits behind"). Detect shallow up front: fetch with
    # --depth 1 to preserve the boundary and compare tip SHAs instead of
    # counting. Full clones (developers, Docker dev images) keep the exact
    # count path unchanged. Mirrors the desktop fix in apps/desktop/electron/main.cjs.
    shallow = _git_stdout(["rev-parse", "--is-shallow-repository"], cwd=repo_dir)
    is_shallow = shallow == "true"

    try:
        # Self-heal abandoned git lock files before fetching. A stale
        # .git/shallow.lock from a crashed fetch makes the fetch fail, the
        # exception below is swallowed, and stale refs get compared against
        # HEAD — silently degrading the passive check until a human removes
        # the lock (git never self-heals these).
        from hermes_cli.gitlock import clear_stale_git_locks

        clear_stale_git_locks(repo_dir)

        # Scope the fetch to the one branch the behind-count compares against.
        # An unscoped ``git fetch origin`` transfers every remote head (~1,400
        # on this repo — measured 3.0 s vs 0.55 s scoped) and can burn the full
        # 10 s timeout on slow links. ``cmd_update`` already scopes its fetch
        # for the same reason. Modern git updates the ``origin/main`` tracking
        # ref on a scoped fetch, so the ``HEAD..origin/main`` count below is
        # unaffected; the shallow path compares against FETCH_HEAD, which a
        # scoped fetch also updates.
        fetch_args = ["git", "fetch", "origin", "main"]
        if is_shallow:
            fetch_args += ["--depth", "1"]
        fetch_args.append("--quiet")
        subprocess.run(
            fetch_args,
            capture_output=True, timeout=10,
            cwd=str(repo_dir),
            creationflags=windows_hide_flags(),
        )
    except Exception:
        pass  # Offline or timeout — use stale refs, that's fine

    if is_shallow:
        # No history to count across the shallow boundary. `origin/main` may not
        # be a tracking ref in a `clone --depth 1`, so prefer FETCH_HEAD (just
        # updated by the fetch above) and fall back to origin/main.
        head_rev = _git_stdout(["rev-parse", "HEAD"], cwd=repo_dir)
        target_rev = (
            _git_stdout(["rev-parse", "FETCH_HEAD"], cwd=repo_dir)
            or _git_stdout(["rev-parse", "origin/main"], cwd=repo_dir)
        )
        if not head_rev or not target_rev:
            return None
        if head_rev == target_rev:
            return 0
        # Tips differ but the shallow boundary hides the history between them.
        # Recover the exact count from the GitHub compare API when possible
        # (ahead_by == 0 means local-ahead ⇒ up to date); otherwise report the
        # honest "update available, count unknown" sentinel.
        counted = _github_compare_behind(head_rev, target_rev)
        return counted if counted is not None else UPDATE_AVAILABLE_NO_COUNT

    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5,
            cwd=str(repo_dir),
            creationflags=windows_hide_flags(),
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def check_for_updates() -> Optional[int]:
    """Check whether a Hermes update is available.

    Two paths: if ``HERMES_REVISION`` is set (nix builds embed it), compare
    it to upstream main via ``git ls-remote``. Otherwise look for a local
    git checkout and count commits behind ``origin/main``.

    Returns the number of commits behind, ``UPDATE_AVAILABLE_NO_COUNT`` (-1)
    if behind but the count is unknown, ``0`` if up-to-date, or ``None`` if
    the check failed or doesn't apply. Cached for 6 hours.
    """
    hermes_home = get_hermes_home()
    cache_file = hermes_home / ".update_check"
    embedded_rev = os.environ.get("HERMES_REVISION") or None

    # Docker images have no working tree to count commits against — the
    # published image excludes `.git` (see .dockerignore) and sets no
    # HERMES_REVISION (that's nix-only). Returning None makes both the Rich
    # banner (build_welcome_banner) and the Ink badge (branding.tsx, guarded
    # on `typeof === 'number' && > 0`) show nothing. The dashboard's REST
    # `/api/hermes/update/check` endpoint short-circuits docker the same way
    # (web_server.py); mirror that here so the banner/TUI surfaces agree.
    try:
        from hermes_cli.config import detect_install_method, get_project_root
        if detect_install_method(get_project_root()) == "docker":
            return None
    except Exception:
        pass

    # Read cache — invalidate if the embedded rev OR installed version has
    # changed since the last check.
    now = time.time()
    try:
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if (
                now - cached.get("ts", 0) < _UPDATE_CHECK_CACHE_SECONDS
                and cached.get("rev") == embedded_rev
                and cached.get("ver") == VERSION
            ):
                return cached.get("behind")
    except Exception:
        pass

    if embedded_rev:
        behind = _check_via_rev(embedded_rev)
    else:
        # Prefer the running code's location over the profile-scoped path.
        # $HERMES_HOME/hermes-agent/ may be a stale copy from --clone-all;
        # Path(__file__) always resolves to the actual installed checkout.
        repo_dir = Path(__file__).parent.parent.resolve()
        if not (repo_dir / ".git").exists():
            repo_dir = hermes_home / "hermes-agent"
        if not (repo_dir / ".git").exists():
            # No git checkout and no embedded revision — can't determine
            # update status. This is the Docker path (already short-circuited
            # above) or an unsupported install without a source tree.
            behind = None
        else:
            behind = _check_via_local_git(repo_dir)

    try:
        cache_file.write_text(
            json.dumps({"ts": now, "behind": behind, "rev": embedded_rev, "ver": VERSION}),
            encoding="utf-8",
        )
    except Exception:
        pass

    return behind


def _resolve_repo_dir() -> Optional[Path]:
    """Return the active Hermes git checkout, or None if this isn't a git install.

    Prefers the running code's location over the profile-scoped path
    because ``$HERMES_HOME/hermes-agent/`` may be a stale copy carried
    over by ``--clone-all``.
    """
    repo_dir = Path(__file__).parent.parent.resolve()
    if not (repo_dir / ".git").exists():
        hermes_home = get_hermes_home()
        repo_dir = hermes_home / "hermes-agent"
    return repo_dir if (repo_dir / ".git").exists() else None


def _git_short_hash(repo_dir: Path, rev: str) -> Optional[str]:
    """Resolve a git revision to an 8-character short hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", rev],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            cwd=str(repo_dir),
            creationflags=windows_hide_flags(),
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    return value or None


_git_banner_state_cache: Optional[tuple] = None  # (state_or_None,) once computed


def get_git_banner_state(repo_dir: Optional[Path] = None) -> Optional[dict]:
    """Return upstream/local git hashes for the startup banner.

    For source installs and dev images this runs ``git rev-parse`` against
    the active checkout.  When no checkout is available — the canonical case
    is the published Docker image, which excludes ``.git`` from the build
    context — we fall back to the baked-in build SHA (see
    ``hermes_cli/build_info.py``) and return it as a frozen
    ``upstream == local`` state with ``ahead=0``.  A built image is by
    definition pinned to one commit, so "ahead" is always zero and the
    banner correctly shows ``· upstream <sha>`` with no carried-commits
    annotation.

    Cached per-process (default ``repo_dir`` only): the state costs 2-3 git
    subprocesses (~100ms) and the checkout revision cannot change under a
    running CLI in a way the banner needs to observe live. The cache also
    lets ``prefetch_banner_data()`` pay this cost off-thread before the
    banner renders.
    """
    global _git_banner_state_cache
    if repo_dir is None and _git_banner_state_cache is not None:
        return _git_banner_state_cache[0]
    state = _compute_git_banner_state(repo_dir)
    if repo_dir is None:
        _git_banner_state_cache = (state,)
    return state


def _compute_git_banner_state(repo_dir: Optional[Path] = None) -> Optional[dict]:
    repo_dir = repo_dir or _resolve_repo_dir()
    if repo_dir is None:
        # No git checkout — try the baked build SHA (Docker image path).
        try:
            from hermes_cli.build_info import get_build_sha
            baked = get_build_sha(short=8)
            if baked:
                return {"upstream": baked, "local": baked, "ahead": 0}
        except Exception:
            pass
        return None

    upstream = _git_short_hash(repo_dir, "origin/main")
    local = _git_short_hash(repo_dir, "HEAD")
    if not upstream or not local:
        # Live-git lookup failed (e.g. shallow clone without origin/main).
        # Fall back to the baked build SHA if available.
        try:
            from hermes_cli.build_info import get_build_sha
            baked = get_build_sha(short=8)
            if baked:
                return {"upstream": baked, "local": baked, "ahead": 0}
        except Exception:
            pass
        return None

    ahead = 0
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "origin/main..HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            cwd=str(repo_dir),
            creationflags=windows_hide_flags(),
        )
        if result.returncode == 0:
            ahead = int((result.stdout or "0").strip() or "0")
    except Exception:
        ahead = 0

    return {"upstream": upstream, "local": local, "ahead": max(ahead, 0)}


_RELEASE_URL_BASE = "https://github.com/NousResearch/hermes-agent/releases/tag"
_latest_release_cache: Optional[tuple] = None  # (tag, url) once resolved


def get_latest_release_tag(repo_dir: Optional[Path] = None) -> Optional[tuple]:
    """Return ``(tag, release_url)`` for the latest git tag, or None.

    Local-only — runs ``git describe --tags --abbrev=0`` against the
    Hermes checkout. Cached per-process. Release URL always points at the
    canonical NousResearch/hermes-agent repo (forks don't get a link).
    """
    global _latest_release_cache
    if _latest_release_cache is not None:
        return _latest_release_cache or None

    repo_dir = repo_dir or _resolve_repo_dir()
    if repo_dir is None:
        _latest_release_cache = ()  # falsy sentinel — skip future lookups
        return None

    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            cwd=str(repo_dir),
            creationflags=windows_hide_flags(),
        )
    except Exception:
        _latest_release_cache = ()
        return None

    if result.returncode != 0:
        _latest_release_cache = ()
        return None

    tag = (result.stdout or "").strip()
    if not tag:
        _latest_release_cache = ()
        return None

    url = f"{_RELEASE_URL_BASE}/{tag}"
    _latest_release_cache = (tag, url)
    return _latest_release_cache


def format_banner_version_label() -> str:
    """Return the version label shown in the startup banner title."""
    base = f"Hermes Agent v{VERSION} ({RELEASE_DATE})"
    state = get_git_banner_state()
    if not state:
        return base

    upstream = state["upstream"]
    local = state["local"]
    ahead = int(state.get("ahead") or 0)

    if ahead <= 0 or upstream == local:
        return f"{base} · upstream {upstream}"

    carried_word = "commit" if ahead == 1 else "commits"
    return f"{base} · upstream {upstream} · local {local} (+{ahead} carried {carried_word})"


# =========================================================================
# Non-blocking update check
# =========================================================================

_update_result: Optional[int] = None
_update_check_done = threading.Event()


def prefetch_update_check():
    """Kick off update check in a background daemon thread."""
    def _run():
        global _update_result
        _update_result = check_for_updates()
        _update_check_done.set()
    t = threading.Thread(target=_run, daemon=True)
    t.start()


_banner_data_prefetch_started = False


def prefetch_banner_data():
    """Warm the banner's subprocess/I/O-heavy inputs in a daemon thread.

    ``build_welcome_banner`` needs git state (2-4 ``git rev-parse``/
    ``describe`` subprocesses, ~130ms) and the skills index (a skills-tree
    rglob, ~110ms). Both are cached per-process by their own modules, so
    warming them here while the main thread pays the CPU-bound ``cli`` /
    prompt_toolkit imports overlaps subprocess waits and file I/O (which
    release the GIL) with import work. Idempotent; failures are irrelevant
    because the banner recomputes anything missing.
    """
    global _banner_data_prefetch_started
    if _banner_data_prefetch_started:
        return
    _banner_data_prefetch_started = True

    def _run() -> None:
        try:
            get_git_banner_state()
        except Exception:
            pass
        try:
            get_latest_release_tag()
        except Exception:
            pass
        try:
            get_available_skills()
        except Exception:
            pass

    threading.Thread(target=_run, name="banner-data-prefetch", daemon=True).start()


def get_update_result(timeout: float = 0.5) -> Optional[int]:
    """Get result of prefetched check. Returns None if not ready."""
    _update_check_done.wait(timeout=timeout)
    return _update_result


def _format_update_notice(behind: int) -> str:
    """Render the update warning line for a non-zero ``behind`` result."""
    from hermes_cli.config import get_managed_update_command, recommended_update_command
    if behind > 0:
        commits_word = "commit" if behind == 1 else "commits"
        return (
            f"[bold yellow]⚠ {behind} {commits_word} behind[/]"
            f"[dim yellow] — run [bold]{recommended_update_command()}[/bold] to update[/]"
        )
    # UPDATE_AVAILABLE_NO_COUNT: nix-built hermes; we know an update
    # exists but not by how much, and we don't know how the user
    # installed it (nix run, profile, system flake, home-manager).
    managed_cmd = get_managed_update_command()
    line = "[bold yellow]⚠ update available[/]"
    if managed_cmd:
        line += f"[dim yellow] — run [bold]{managed_cmd}[/bold][/]"
    return line


_deferred_update_notice_started = False


def _defer_update_notice(console: "Console", max_wait: float = 30.0) -> None:
    """Print the update warning once the prefetched check completes.

    Used when the banner rendered before the update prefetch finished so
    startup never blocks on git/network. Prints at most once per process.
    """
    global _deferred_update_notice_started
    if _deferred_update_notice_started:
        return
    _deferred_update_notice_started = True

    def _wait_and_print() -> None:
        try:
            if not _update_check_done.wait(timeout=max_wait):
                return
            behind = _update_result
            if behind is None or behind == 0:
                return
            console.print(_format_update_notice(behind))
        except Exception:
            pass  # never break the session over an update notice

    threading.Thread(
        target=_wait_and_print, name="update-notice", daemon=True
    ).start()


# =========================================================================
# Welcome banner
# =========================================================================

def _format_context_length(tokens: int) -> str:
    """Format a token count for display (e.g. 128000 → '128K', 1048576 → '1M')."""
    if tokens >= 1_000_000:
        val = tokens / 1_000_000
        rounded = round(val)
        if abs(val - rounded) < 0.05:
            return f"{rounded}M"
        return f"{val:.1f}M"
    elif tokens >= 1_000:
        val = tokens / 1_000
        rounded = round(val)
        if abs(val - rounded) < 0.05:
            return f"{rounded}K"
        return f"{val:.1f}K"
    return str(tokens)


def _display_toolset_name(toolset_name: str) -> str:
    """Normalize internal/legacy toolset identifiers for banner display."""
    if not toolset_name:
        return "unknown"
    return (
        toolset_name[:-6]
        if toolset_name.endswith("_tools")
        else toolset_name
    )


# =========================================================================
# Banner snapshot — warm-launch fast path
# =========================================================================
# The banner's tool panel needs the full tool registry (get_tool_definitions:
# tools/*.py discovery + every check_fn), which costs ~0.5-0.9s cold and is
# the single largest chunk of CLI time-to-banner. The tool list shown in the
# banner is a pure function of (config.yaml, .env, code checkout, enabled
# toolsets), so we snapshot the rendered inputs to disk after each launch
# and replay them on the next one when the fingerprint matches. The agent's
# REAL tool list is still computed fresh at first message (agent init) —
# the snapshot only feeds the cosmetic startup panel, and a background
# refresh re-verifies it right after the banner renders (see
# cli.show_banner), so a stale panel self-heals within one launch.

_BANNER_SNAPSHOT_VERSION = 1


def _banner_snapshot_path() -> Path:
    return get_hermes_home() / "cache" / "banner_snapshot.json"


def banner_snapshot_fingerprint() -> Optional[str]:
    """Fingerprint the inputs the banner tool panel depends on."""
    import hashlib
    parts = [f"v{_BANNER_SNAPSHOT_VERSION}"]
    try:
        from hermes_cli.config import get_config_path
        for p in (get_config_path(), get_hermes_home() / ".env"):
            try:
                st = p.stat()
                parts.append(f"{p.name}:{st.st_mtime_ns}:{st.st_size}")
            except OSError:
                parts.append(f"{p.name}:absent")
    except Exception:
        return None
    # Code checkout: version + git HEAD when available (post-update change).
    parts.append(str(VERSION))
    state = get_git_banner_state()
    if state:
        parts.append(str(state.get("local", "")))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def load_banner_snapshot(enabled_toolsets: List[str] = None) -> Optional[Dict[str, Any]]:
    """Return the stored banner snapshot when its fingerprint is current."""
    try:
        blob = json.loads(_banner_snapshot_path().read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    fp = banner_snapshot_fingerprint()
    if not fp or blob.get("fingerprint") != fp:
        return None
    if blob.get("enabled_toolsets") != sorted(enabled_toolsets or []):
        return None
    tools = blob.get("tools")
    toolset_map = blob.get("toolset_map")
    availability = blob.get("availability")
    if not isinstance(tools, list) or not isinstance(toolset_map, dict) \
            or not isinstance(availability, dict):
        return None
    if not isinstance(blob.get("skills_by_category"), dict):
        return None
    return blob


def save_banner_snapshot(
    tools: List[dict],
    enabled_toolsets: List[str],
    availability: Dict[str, Any],
    toolset_map: Dict[str, str],
) -> None:
    """Persist the banner tool panel inputs for next launch (best-effort)."""
    fp = banner_snapshot_fingerprint()
    if not fp:
        return
    payload = {
        "fingerprint": fp,
        "enabled_toolsets": sorted(enabled_toolsets or []),
        "tools": [
            {"function": {"name": t["function"]["name"]}}
            for t in tools
            if isinstance(t, dict) and t.get("function", {}).get("name")
        ],
        "toolset_map": toolset_map,
        "availability": {
            "unavailable_toolsets": availability.get("unavailable_toolsets", []),
            "lazy_tools": list(availability.get("lazy_tools", [])),
            "disabled_tools": list(availability.get("disabled_tools", [])),
        },
        "skills_by_category": get_available_skills(),
    }
    path = _banner_snapshot_path()
    try:
        import os as _os
        import tempfile as _tempfile
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = _tempfile.mkstemp(dir=str(path.parent), prefix=".banner_snap.")
        with _os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        _os.replace(tmp, path)
    except Exception:
        pass


def compute_toolset_availability(enabled_toolsets: List[str] = None) -> Dict[str, Any]:
    """Compute the banner's toolset-availability payload.

    Returns ``{"unavailable_toolsets": [...], "lazy_tools": [...],
    "disabled_tools": [...]}`` — the exact inputs ``build_welcome_banner``
    needs to annotate disabled/lazy tools. Split out so the result can be
    snapshotted to disk and replayed on the next launch without importing
    ``model_tools`` (see ``load_banner_snapshot``).
    """
    from model_tools import check_tool_availability, TOOLSET_REQUIREMENTS

    enabled_toolsets = enabled_toolsets or []
    _, unavailable_toolsets = check_tool_availability(quiet=True)
    # The availability check walks the GLOBAL toolset registry, so it includes
    # toolsets that aren't part of this agent's platform set at all (e.g.
    # `discord`, `feishu_doc` on a CLI session). Those must never surface in the
    # banner's "Available Tools" — they aren't exposed to the agent. Restrict to
    # toolsets actually enabled for this agent; a toolset that's enabled but
    # currently has unmet deps legitimately shows as disabled/lazy below.
    _enabled_ts = {str(t) for t in enabled_toolsets}
    if _enabled_ts:
        unavailable_toolsets = [
            item for item in unavailable_toolsets
            if str(item.get("id", item.get("name", ""))) in _enabled_ts
        ]
    disabled_tools = set()
    # Tools whose toolset has a check_fn are lazy-initialized (e.g. honcho,
    # homeassistant) — they show as unavailable at banner time because the
    # check hasn't run yet, but they aren't misconfigured.
    lazy_tools = set()
    for item in unavailable_toolsets:
        toolset_name = item.get("name", "")
        ts_req = TOOLSET_REQUIREMENTS.get(toolset_name, {})
        tools_in_ts = item.get("tools", [])
        if ts_req.get("check_fn"):
            lazy_tools.update(tools_in_ts)
        else:
            disabled_tools.update(tools_in_ts)
    return {
        "unavailable_toolsets": unavailable_toolsets,
        "lazy_tools": sorted(lazy_tools),
        "disabled_tools": sorted(disabled_tools),
    }


def build_welcome_banner(console: "Console", model: str, cwd: str,
                         tools: List[dict] = None,
                         enabled_toolsets: List[str] = None,
                         session_id: str = None,
                         get_toolset_for_tool=None,
                         context_length: int = None,
                         provider: str = None,
                         availability: Dict[str, Any] = None,
                         skills_by_category: Dict[str, List[str]] = None):
    """Build and print a welcome banner with caduceus on left and info on right.

    Args:
        console: Rich Console instance.
        model: Current model name.
        cwd: Current working directory.
        tools: List of tool definitions.
        enabled_toolsets: List of enabled toolset names.
        session_id: Session identifier.
        get_toolset_for_tool: Callable to map tool name -> toolset name.
        context_length: Model's context window size in tokens.
        provider: Active provider id. When ``"moa"``, ``model`` is a MoA
            preset name and the banner renders the aggregator instead of a
            bare model slug.
        availability: Optional precomputed result of
            ``compute_toolset_availability`` (e.g. replayed from the banner
            snapshot). When provided together with ``get_toolset_for_tool``,
            this function performs no ``model_tools`` import at all.
    """
    from rich.panel import Panel
    from rich.table import Table
    if get_toolset_for_tool is None:
        from model_tools import get_toolset_for_tool

    tools = tools or []
    enabled_toolsets = enabled_toolsets or []

    if availability is None:
        availability = compute_toolset_availability(enabled_toolsets)
    unavailable_toolsets = availability.get("unavailable_toolsets", [])
    lazy_tools = set(availability.get("lazy_tools", []))
    disabled_tools = set(availability.get("disabled_tools", []))
    _enabled_ts = {str(t) for t in enabled_toolsets}

    layout_table = Table.grid(padding=(0, 2))
    layout_table.add_column("left", justify="center")
    layout_table.add_column("right", justify="left")

    # Resolve skin colors once for the entire banner
    accent = _skin_color("banner_accent", "#FFBF00")
    dim = _skin_color("banner_dim", "#B8860B")
    text = _skin_color("banner_text", "#FFF8DC")
    session_color = _skin_color("session_border", "#8B8682")

    # Use skin's custom caduceus art if provided
    try:
        from hermes_cli.skin_engine import get_active_skin
        _bskin = get_active_skin()
        _hero = _bskin.banner_hero if hasattr(_bskin, 'banner_hero') and _bskin.banner_hero else HERMES_CADUCEUS
    except Exception:
        _bskin = None
        _hero = HERMES_CADUCEUS
    left_lines = ["", _hero, ""]
    if (provider or "").strip().lower() == "moa":
        # MoA virtual provider: ``model`` is a preset name. Show the preset and
        # its aggregator so the banner is meaningful instead of a bare slug.
        preset_name = model
        agg_label = ""
        try:
            from hermes_cli.config import load_config
            from hermes_cli.moa_config import normalize_moa_config

            _moa = normalize_moa_config(load_config().get("moa") or {})
            _preset = _moa.get("presets", {}).get(preset_name)
            if _preset:
                _agg = _preset.get("aggregator") or {}
                _am = str(_agg.get("model") or "")
                agg_label = _am.split("/")[-1] if "/" in _am else _am
        except Exception:
            agg_label = ""
        if len(preset_name) > 28:
            preset_name = preset_name[:25] + "..."
        agg_str = f" [dim {dim}]·[/] [dim {dim}]agg {agg_label}[/]" if agg_label else ""
        ctx_str = f" [dim {dim}]·[/] [dim {dim}]{_format_context_length(context_length)} context[/]" if context_length else ""
        left_lines.append(f"[{accent}]MoA: {preset_name}[/]{agg_str}{ctx_str} [dim {dim}]·[/] [dim {dim}]Nous Research[/]")
    else:
        if not (model or "").strip() or (model or "").strip().lower() == "unknown":
            # Unconfigured install: say so in red instead of a blank/"unknown"
            # slug — this is the single clearest place to tell the user what
            # is wrong and how to fix it.
            left_lines.append(
                f"[bold red]no model configured[/] "
                f"[dim {dim}]— run /model or hermes setup[/]"
            )
        else:
            model_short = model.split("/")[-1] if "/" in model else model
            if model_short.endswith(".gguf"):
                model_short = model_short[:-5]
            if len(model_short) > 28:
                model_short = model_short[:25] + "..."
            ctx_str = f" [dim {dim}]·[/] [dim {dim}]{_format_context_length(context_length)} context[/]" if context_length else ""
            left_lines.append(f"[{accent}]{model_short}[/]{ctx_str} [dim {dim}]·[/] [dim {dim}]Nous Research[/]")

    if os.getenv("HERMES_YOLO_MODE"):
        left_lines.append(f"[bold red]⚠ YOLO mode[/] [dim {dim}]— all approval prompts bypassed[/]")
    left_lines.append(f"[dim {dim}]{cwd}[/]")
    if session_id:
        left_lines.append(f"[dim {session_color}]Session: {session_id}[/]")
    left_content = "\n".join(left_lines)

    right_lines = [f"[bold {accent}]Available Tools[/]"]
    toolsets_dict: Dict[str, list] = {}

    for tool in tools:
        tool_name = tool["function"]["name"]
        toolset = _display_toolset_name(get_toolset_for_tool(tool_name) or "other")
        toolsets_dict.setdefault(toolset, []).append(tool_name)

    for item in unavailable_toolsets:
        toolset_id = item.get("id", item.get("name", "unknown"))
        display_name = _display_toolset_name(toolset_id)
        if display_name not in toolsets_dict:
            toolsets_dict[display_name] = []
        for tool_name in item.get("tools", []):
            if tool_name not in toolsets_dict[display_name]:
                toolsets_dict[display_name].append(tool_name)

    sorted_toolsets = sorted(toolsets_dict.keys())
    display_toolsets = sorted_toolsets[:8]
    remaining_toolsets = len(sorted_toolsets) - 8

    for toolset in display_toolsets:
        tool_names = toolsets_dict[toolset]
        colored_names = []
        for name in sorted(tool_names):
            if name in disabled_tools:
                colored_names.append(f"[red]{name}[/]")
            elif name in lazy_tools:
                colored_names.append(f"[yellow]{name}[/]")
            else:
                colored_names.append(f"[{text}]{name}[/]")

        tools_str = ", ".join(colored_names)
        if len(", ".join(sorted(tool_names))) > 45:
            short_names = []
            length = 0
            for name in sorted(tool_names):
                if length + len(name) + 2 > 42:
                    short_names.append("...")
                    break
                short_names.append(name)
                length += len(name) + 2
            colored_names = []
            for name in short_names:
                if name == "...":
                    colored_names.append("[dim]...[/]")
                elif name in disabled_tools:
                    colored_names.append(f"[red]{name}[/]")
                elif name in lazy_tools:
                    colored_names.append(f"[yellow]{name}[/]")
                else:
                    colored_names.append(f"[{text}]{name}[/]")
            tools_str = ", ".join(colored_names)

        right_lines.append(f"[dim {dim}]{toolset}:[/] {tools_str}")

    if remaining_toolsets > 0:
        right_lines.append(f"[dim {dim}](and {remaining_toolsets} more toolsets...)[/]")

    # MCP Servers section (only if configured). Probe cheaply first: the
    # full get_mcp_status() path resolves portable plugin MCP servers,
    # which JOINS the in-flight background plugin discovery (~100ms on the
    # startup path). When neither config.yaml nor the persisted plugin
    # key cache mentions any MCP server, skip the section outright.
    mcp_status = []
    try:
        from hermes_cli.config import load_config as _load_cfg
        _has_native_mcp = bool((_load_cfg() or {}).get("mcp_servers"))
    except Exception:
        _has_native_mcp = True  # can't tell — take the full path
    _has_portable_mcp = False
    if not _has_native_mcp:
        try:
            from hermes_cli.plugins import get_portable_mcp_server_names_nowait
            _has_portable_mcp = bool(get_portable_mcp_server_names_nowait())
        except Exception:
            _has_portable_mcp = True  # can't tell — take the full path
    if _has_native_mcp or _has_portable_mcp:
        try:
            from tools.mcp_tool import get_mcp_status
            mcp_status = get_mcp_status()
        except Exception:
            mcp_status = []

    if mcp_status:
        right_lines.append("")
        right_lines.append(f"[bold {accent}]MCP Servers[/]")
        for srv in mcp_status:
            status = srv.get("status")
            if srv["connected"]:
                right_lines.append(
                    f"[dim {dim}]{srv['name']}[/] [{text}]({srv['transport']})[/] "
                    f"[dim {dim}]—[/] [{text}]{srv['tools']} tool(s)[/]"
                )
            elif srv.get("disabled") or status == "disabled":
                right_lines.append(
                    f"[dim {dim}]{srv['name']}[/] [dim]({srv['transport']})[/] "
                    f"[dim {dim}]— disabled[/]"
                )
            elif status == "connecting":
                right_lines.append(
                    f"[dim {dim}]{srv['name']}[/] [dim]({srv['transport']})[/] "
                    f"[yellow]— connecting[/]"
                )
            elif status == "configured":
                right_lines.append(
                    f"[dim {dim}]{srv['name']}[/] [dim]({srv['transport']})[/] "
                    f"[dim {dim}]— configured[/]"
                )
            else:
                right_lines.append(
                    f"[red]{srv['name']}[/] [dim]({srv['transport']})[/] "
                    f"[red]— failed[/]"
                )

    right_lines.append("")
    right_lines.append(f"[bold {accent}]Available Skills[/]")
    # The skills catalog is only reachable when the `skills` toolset is enabled
    # (it exposes skill_view / skill_manage). When it's disabled — e.g. a Blank
    # Slate install — the agent literally cannot load any skill, so advertising
    # the on-disk catalog here is misleading. Reflect the real state instead.
    _skills_enabled = (not _enabled_ts) or ("skills" in _enabled_ts)
    if _skills_enabled:
        if skills_by_category is None:
            skills_by_category = get_available_skills()
        total_skills = sum(len(s) for s in skills_by_category.values())
    else:
        skills_by_category = {}
        total_skills = 0

    # Dynamically size skills display based on terminal width.
    # Rich grid with 2 columns; right column gets roughly 60% of terminal.
    _term_cols = shutil.get_terminal_size().columns
    _right_col_width = max(int(_term_cols * 0.6) - 10, 30)

    if not _skills_enabled:
        right_lines.append(f"[dim {dim}]Skills toolset disabled[/]")
    elif skills_by_category:
        for category in sorted(skills_by_category.keys()):
            skill_names = sorted(skills_by_category[category])
            # Account for "category: " prefix
            _prefix_len = len(category) + 2
            _avail = max(_right_col_width - _prefix_len, 20)
            # Accumulate skills until we run out of space
            parts, length = [], 0
            for i, name in enumerate(skill_names):
                _sep = ", " if parts else ""
                _needed = len(_sep) + len(name)
                # Estimate indicator size IF we were to add this skill then stop
                _after = len(skill_names) - (i + 1)  # remaining after adding this
                _ind_len = len(f", +{_after} more") if _after > 0 else 0
                if parts and length + _needed + _ind_len > _avail:
                    remaining = len(skill_names) - len(parts)
                    parts.append(f"+{remaining} more")
                    break
                parts.append(name)
                length += _needed
            skills_str = ", ".join(parts)
            right_lines.append(f"[dim {dim}]{category}:[/] [{text}]{skills_str}[/]")
    else:
        right_lines.append(f"[dim {dim}]No skills installed[/]")

    right_lines.append("")
    mcp_connected = sum(1 for s in mcp_status if s["connected"]) if mcp_status else 0
    summary_parts = [f"{len(tools)} tools", f"{total_skills} skills"]
    if mcp_connected:
        summary_parts.append(f"{mcp_connected} MCP servers")
    summary_parts.append("/help for commands")
    # Indicate when the codex_app_server runtime is active so users
    # understand why tool counts may not match what's actually reachable
    # (codex builds its own tool list inside the spawned subprocess).
    try:
        from hermes_cli.codex_runtime_switch import get_current_runtime
        from hermes_cli.config import load_config as _load_cfg
        if get_current_runtime(_load_cfg()) == "codex_app_server":
            right_lines.append(
                f"[bold {accent}]Runtime:[/] [{text}]codex app-server[/] "
                f"[dim {dim}](terminal/file ops/MCP run inside codex)[/]"
            )
    except Exception:
        pass
    # Show active profile name when not 'default'
    try:
        from hermes_cli.profiles import get_active_profile_name
        _profile_name = get_active_profile_name()
        if _profile_name and _profile_name != "default":
            right_lines.append(f"[bold {accent}]Profile:[/] [{text}]{_profile_name}[/]")
    except Exception:
        pass  # Never break the banner over a profiles.py bug

    right_lines.append(f"[dim {dim}]{' · '.join(summary_parts)}[/]")

    # Update check — use prefetched result if available. NEVER block the
    # banner on it: the prefetch does git/network work that rarely finishes
    # before the banner renders, so a blocking wait here just adds its full
    # timeout to every startup (500ms of the banner path pre-fix). If the
    # result isn't ready yet, defer the warning line: a daemon thread waits
    # for the prefetch and prints the same notice above the prompt when it
    # lands (prompt_toolkit's patch_stdout renders late prints safely).
    try:
        behind = get_update_result(timeout=0.05)
        if behind is None and not _update_check_done.is_set():
            _defer_update_notice(console)
        elif behind is not None and behind != 0:
            right_lines.append(_format_update_notice(behind))
    except Exception:
        pass  # Never break the banner over an update check

    right_content = "\n".join(right_lines)
    layout_table.add_row(left_content, right_content)

    title_color = _skin_color("banner_title", "#FFD700")
    border_color = _skin_color("banner_border", "#CD7F32")
    version_label = format_banner_version_label()
    release_info = get_latest_release_tag()
    if release_info:
        _tag, _url = release_info
        title_markup = f"[bold {title_color}][link={_url}]{version_label}[/link][/]"
    else:
        title_markup = f"[bold {title_color}]{version_label}[/]"
    outer_panel = Panel(
        layout_table,
        title=title_markup,
        border_style=border_color,
        padding=(0, 2),
    )

    console.print()
    term_width = shutil.get_terminal_size().columns
    if term_width >= 95:
        _logo = _bskin.banner_logo if _bskin and hasattr(_bskin, 'banner_logo') and _bskin.banner_logo else HERMES_AGENT_LOGO
        console.print(_logo)
        console.print()
    console.print(outer_panel)
