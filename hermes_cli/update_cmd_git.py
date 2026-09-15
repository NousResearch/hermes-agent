"""Git plumbing for ``hermes update``: fork/upstream sync, trampoline-git detection, lockfile/EOL churn cleanup, orphan rescue refs, parked-branch assessment, fetch-failure classification.

Split out of ``update_cmd.py``, which re-imports every name so ``hermes_cli.update_cmd.<name>``
still resolves/monkeypatches. Origin helpers are imported lazily per function (no cycle;
test patches on ``update_cmd`` stay effective).
"""

import logging
from contextlib import suppress
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("hermes_cli.update_cmd")  # log-record parity with the origin module

_ORPHAN_RESCUE_REFS_TO_KEEP = 10
_ORPHAN_RESCUE_REF_MAX_AGE_DAYS = 30

_GIT_TEXT_KW = dict(capture_output=True, text=True, encoding="utf-8", errors="replace")
_BAR = "=" * 68
_UPSTREAM_ADD_CMD = "git remote add upstream https://github.com/NousResearch/hermes-agent.git"


def _git_ok(git_cmd, args, cwd, **kw) -> bool:
    """True when ``_git_run`` exits 0; any exception counts as failure."""
    return _git_stdout(git_cmd, args, cwd, **kw) is not None


def _git_stdout(git_cmd, args, cwd, **kw) -> Optional[str]:
    """Stripped stdout of a successful ``_git_run``; ``None`` on non-zero exit or any exception."""
    from hermes_cli.update_cmd import _git_run
    with suppress(Exception):
        result = _git_run(git_cmd, args, cwd, **kw)
        if result.returncode == 0:
            return result.stdout.strip()
    return None


def _prune_orphan_rescue_refs(
    git_cmd, cwd, branch, keep=_ORPHAN_RESCUE_REFS_TO_KEEP, max_age_days=_ORPHAN_RESCUE_REF_MAX_AGE_DAYS
) -> None:
    """Expire old orphan rescue refs (``refs/hermes-update-backups/orphan-<branch>-<ts>-<sha>``).

    Each ref pins a possibly multi-GB snapshot against ``git gc``, so a repeatedly corrupted install would
    grow ``.git`` unbounded. Keep the ``keep`` newest AND drop any older than ``max_age_days`` by the
    ``YYYYMMDD-HHMMSS`` stamp (unparseable names left alone); names sort chronologically so
    ``for-each-ref`` order is creation order. Best-effort, never blocks.

    A rescue ref pins every object reachable from that commit against ``git gc`` — and in the incident shape
    those objects include a full working-tree snapshot (the autostash orphan commit), which can be multi-GB
    when the tree holds large stray files. See #87694.
    """
    from hermes_cli.update_cmd import _git_run
    with suppress(OSError):
        prefix = f"refs/hermes-update-backups/orphan-{branch}-"
        list_result = _git_run(git_cmd, ["for-each-ref", "--format=%(refname)", "--sort=refname", f"{prefix}*"], cwd)
        if list_result.returncode != 0:
            return
        refs = [line.strip() for line in list_result.stdout.splitlines() if line.strip()]
        stale = set(refs[:-keep] if keep > 0 else refs)
        if max_age_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            for ref in refs:
                with suppress(ValueError):
                    if datetime.strptime(ref[len(prefix):][:15], "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc) < cutoff:
                        stale.add(ref)
        for ref in sorted(stale):
            _git_run(git_cmd, ["update-ref", "-d", ref], cwd)


def _branch_head_label(git_cmd=None, cwd=None) -> str | None:
    """``"<branch> @ <short-sha>"`` for the checkout (``detached`` when not on a branch), or None. Never raises.

    Appended to summary lines so a checkout parked on a stale branch is visible."""
    from hermes_cli.update_cmd import _m
    try:
        cmd = list(git_cmd) if git_cmd else ["git"]
        root = cwd if cwd is not None else _m().PROJECT_ROOT

        def _rev_parse(*args):
            return subprocess.run(cmd + ["rev-parse", *args], cwd=root, **_GIT_TEXT_KW)

        branch, sha = _rev_parse("--abbrev-ref", "HEAD"), _rev_parse("--short", "HEAD")
        branch_name, sha_text = branch.stdout.strip(), sha.stdout.strip()
        if branch.returncode != 0 or sha.returncode != 0 or not sha_text or not branch_name:
            return None
        return f"{'detached' if branch_name == 'HEAD' else branch_name} @ {sha_text}"
    except Exception:
        return None


def _branch_head_suffix(git_cmd=None, cwd=None) -> str:
    """`` [<branch> @ <sha>]`` suffix for summary lines ("" when unknown)."""
    label = _branch_head_label(git_cmd, cwd)
    return f" [{label}]" if label else ""


def _assess_parked_branch_switch(git_cmd: list[str], cwd: Path, current_branch: str, target_branch: str) -> tuple[bool, str]:
    """Decide whether a parked feature branch may be auto-switched back to the update target.

    - (True, "") — tree clean and every parked commit is in ``origin/<target>`` (no ``git cherry +``).
    - (True, "unmerged:<n>") — tree clean but commits not in target; switching is safe (checkout keeps
      committed work) but caller must print a LOUD notice. Non-interactive callers (desktop, gateway
      /update, cron) can't resolve a skip, so a clean checkout must reach target.
    - (False, "disabled"|"dirty"|"unverifiable") — caller must NOT touch the branch. Dirty is the
      genuinely unsafe case: uncommitted work riding an autostash across branches.
    A config read failure must not disable the safety checks: fall through with the default."""
    from hermes_cli.update_cmd import _git_run
    try:
        from hermes_cli.config import load_config
        _update_cfg = (load_config() or {}).get("updates", {})
        if isinstance(_update_cfg, dict) and not bool(_update_cfg.get("auto_switch_parked_branch", True)):
            return False, "disabled"
    except Exception as exc:
        logger.debug("Could not read updates.auto_switch_parked_branch: %s", exc)
    status = _git_run(git_cmd, ["status", "--porcelain"], cwd)
    if status.returncode != 0:
        return False, "unverifiable"
    if status.stdout.strip():
        return False, "dirty"
    cherry = _git_run(git_cmd, ["cherry", f"origin/{target_branch}"], cwd)
    if cherry.returncode != 0:
        return False, "unverifiable"
    unmerged = [line for line in cherry.stdout.splitlines() if line.startswith("+")]
    return True, f"unmerged:{len(unmerged)}" if unmerged else ""


_PARKED_SKIP_WHY = {
    "dirty": "the working tree has uncommitted changes",
    "disabled": "updates.auto_switch_parked_branch is set to false in config.yaml",
}


def _print_parked_branch_skip_warning(git_cmd: list[str], cwd: Path, current_branch: str, target_branch: str, reason: str) -> None:
    """LOUD block: why the update was skipped on a parked branch, behind-count, fix commands."""
    behind = None
    with suppress(Exception):
        behind_text = _git_stdout(git_cmd, ["rev-list", f"HEAD..origin/{target_branch}", "--count"], cwd)
        if behind_text:
            behind = int(behind_text)
    why = _PARKED_SKIP_WHY.get(reason, f"the branch state could not be verified against origin/{target_branch}")
    print(f"\n{_BAR}\n⚠ CODE UPDATE SKIPPED — checkout is parked on '{current_branch}'")
    print(f"  Not auto-switching to {target_branch}: {why}.")
    if behind is not None and behind > 0:
        print(f"  This checkout is {behind} commit(s) BEHIND origin/{target_branch} — the code you are running is stale.")
    print(
        f"\n  To resolve, inspect the branch and switch back yourself:\n"
        f"    git -C {cwd} status\n"
        f"    git -C {cwd} checkout {target_branch} && hermes update\n"
        f"  (commit or stash your work on the branch first if you want to keep it)\n{_BAR}"
    )


def _print_parked_branch_kept_notice(current_branch: str, target_branch: str, unmerged_count: str) -> None:
    """LOUD notice when a clean parked branch with unmerged commits is auto-switched.

    Non-interactive callers can't resolve a skip, so we proceed — but the unmerged work
    (still safe on its branch) must be impossible to miss."""
    print(
        f"\n{_BAR}\n"
        f"⚠ Checkout was parked on '{current_branch}' with "
        f"{unmerged_count} commit(s) not merged into origin/{target_branch}.\n"
        f"  Switching to {target_branch} so the update can proceed — your "
        f"commit(s) are safe on '{current_branch}'.\n\n"
        f"  To pick the work back up later:\n    git checkout {current_branch}\n{_BAR}"
    )


OFFICIAL_REPO_URLS = {
    "https://github.com/NousResearch/hermes-agent.git",
    "git@github.com:NousResearch/hermes-agent.git",
    "https://github.com/NousResearch/hermes-agent",
    "git@github.com:NousResearch/hermes-agent",
}
OFFICIAL_REPO_URL = "https://github.com/NousResearch/hermes-agent.git"
SKIP_UPSTREAM_PROMPT_FILE = ".skip_upstream_prompt"


def _get_origin_url(git_cmd: list[str], cwd: Path) -> Optional[str]:
    """Get the URL of the origin remote, or None if not set."""
    return _git_stdout(git_cmd, ["remote", "get-url", "origin"], cwd)


def _is_fork(origin_url: Optional[str]) -> bool:
    """Check if the origin remote points to a fork (not the official repo)."""
    if not origin_url:
        return False

    def _norm(url: str) -> str:
        url = url.rstrip("/")
        return url[:-4] if url.endswith(".git") else url

    return _norm(origin_url) not in {_norm(official) for official in OFFICIAL_REPO_URLS}


def _has_upstream_remote(git_cmd: list[str], cwd: Path) -> bool:
    """Check if an 'upstream' remote already exists."""
    return _git_ok(git_cmd, ["remote", "get-url", "upstream"], cwd)


def _add_upstream_remote(git_cmd: list[str], cwd: Path) -> bool:
    """Add the official repo as the 'upstream' remote. Returns True on success."""
    return _git_ok(git_cmd, ["remote", "add", "upstream", OFFICIAL_REPO_URL], cwd)


def _count_commits_between(git_cmd: list[str], cwd: Path, base: str, head: str) -> int:
    """Count commits on `head` that are not on `base`. Returns -1 on error."""
    with suppress(Exception):
        count = _git_stdout(git_cmd, ["rev-list", "--count", f"{base}..{head}"], cwd)
        if count is not None:
            return int(count)
    return -1


def _should_skip_upstream_prompt() -> bool:
    """Check if user previously declined to add upstream."""
    from hermes_constants import get_hermes_home
    return (get_hermes_home() / SKIP_UPSTREAM_PROMPT_FILE).exists()


def _mark_skip_upstream_prompt():
    """Create marker file to skip future upstream prompts."""
    with suppress(Exception):
        from hermes_constants import get_hermes_home
        (get_hermes_home() / SKIP_UPSTREAM_PROMPT_FILE).touch()


def _sync_fork_with_upstream(git_cmd: list[str], cwd: Path) -> bool:
    """Push updated main to origin (sync fork); True on success."""
    return _git_ok(git_cmd, ["push", "origin", "main", "--force-with-lease"], cwd, network=True)


def _offer_upstream_remote(git_cmd: list[str], cwd: Path, *, assume_yes: bool, input_fn) -> bool:
    """Prompt to add ``upstream`` and add it; False when the user declined, the run is non-interactive, or add failed.

    ``--yes`` means "don't block", not "mutate my remotes", so a non-interactive skip is NOT persisted."""
    from hermes_cli.update_cmd import _add_upstream_remote, _mark_skip_upstream_prompt
    print(
        "\nℹ Your fork is not tracking the official Hermes repository.\n"
        "  This means you may miss updates from NousResearch/hermes-agent.\n"
    )
    if assume_yes or (input_fn is None and not (sys.stdin.isatty() and sys.stdout.isatty())):
        print(f"  Skipping upstream setup (non-interactive run).\n  Add it later with: {_UPSTREAM_ADD_CMD}")
        return False
    if input_fn is not None:
        response = input_fn("Add official repo as 'upstream' remote? [y/N]", "n").strip().lower()
    else:
        try:
            response = input("Add official repo as 'upstream' remote? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt, UnicodeDecodeError):
            print()
            response = "n"
    if response not in {"", "y", "yes"}:
        print(f"  Skipped. Run '{_UPSTREAM_ADD_CMD}' to add later.")
        _mark_skip_upstream_prompt()
        return False
    print("→ Adding upstream remote...")
    if not _add_upstream_remote(git_cmd, cwd):
        print("  ✗ Failed to add upstream remote. Skipping upstream sync.")
        return False
    print("  ✓ Added upstream: https://github.com/NousResearch/hermes-agent.git")
    return True


def _sync_with_upstream_if_needed(git_cmd: list[str], cwd: Path, *, assume_yes: bool = False, input_fn=None) -> bool:
    """Offer to add ``upstream``, compare origin/main vs upstream/main, ff-pull when strictly behind, then push origin.

    Returns True only when origin/main was actually verified against upstream/main; False when the check never
    happened, so the caller never reports "up to date" on an origin-only compare. Fetches only upstream/main:
    a bare fetch drags in thousands of auto-generated branches.

    See #97052.
    """
    from hermes_cli.update_cmd import _count_commits_between, _has_upstream_remote, _no_prompt_git_kwargs, _should_skip_upstream_prompt
    if not _has_upstream_remote(git_cmd, cwd) and (
        _should_skip_upstream_prompt() or not _offer_upstream_remote(git_cmd, cwd, assume_yes=assume_yes, input_fn=input_fn)
    ):
        return False
    print("\n→ Fetching upstream...")
    try:
        subprocess.run(git_cmd + ["fetch", "upstream", "main", "--quiet"], cwd=cwd, capture_output=True, check=True, **_no_prompt_git_kwargs())
    except subprocess.CalledProcessError:
        print("  ✗ Failed to fetch upstream. Skipping upstream sync.")
        return False
    origin_ahead = _count_commits_between(git_cmd, cwd, "upstream/main", "origin/main")
    upstream_ahead = _count_commits_between(git_cmd, cwd, "origin/main", "upstream/main")
    if origin_ahead < 0 or upstream_ahead < 0:
        print("  ✗ Could not compare branches. Skipping upstream sync.")
        return False
    if origin_ahead > 0:
        print(
            f"\nℹ Your fork has {origin_ahead} commit(s) not on upstream.\n"
            "  Skipping upstream sync to preserve your changes.\n"
            "  If you want to merge upstream changes, run:\n    git pull upstream main"
        )
        return True
    if upstream_ahead == 0:
        print("  ✓ Fork is up to date with upstream")
        return True
    print(f"\n→ Fork is {upstream_ahead} commit(s) behind upstream\n→ Pulling from upstream...")
    try:
        subprocess.run(git_cmd + ["pull", "--ff-only", "upstream", "main"], cwd=cwd, check=True, **_no_prompt_git_kwargs())
    except subprocess.CalledProcessError:
        print("  ✗ Failed to pull from upstream. You may need to resolve conflicts manually.")
        return False
    print("  ✓ Updated from upstream\n→ Syncing fork...")
    if _sync_fork_with_upstream(git_cmd, cwd):
        print("  ✓ Fork synced with upstream")
    else:
        print(
            "  ℹ Got updates from upstream but couldn't push to fork (no write access?)\n"
            "    Your local repo is updated, but your fork on GitHub may be behind."
        )
    return True


def _has_http_code(stderr: str, *codes: str) -> bool:
    return any(f"HTTP {code}" in stderr or f"returned error: {code}" in stderr for code in codes)


# Ordered (predicate, diagnosis): curl reports HTTP errors as ``unable to access '<url>': ... error: 429``,
# so rate-limit/outage checks must run BEFORE the generic "unable to access" check. An anonymous fetch
# answered with HTTP 401 ("could not read Username") is GitHub during an outage (or a renamed/private
# repo), not a user credentials problem.
_FETCH_FAILURE_RULES = (
    (lambda s: _has_http_code(s, "429") or "rate limit" in s.lower(),
     "✗ GitHub is rate limiting requests or having an outage (HTTP 429) — try again in 5 minutes."),
    (lambda s: _has_http_code(s, "500", "502", "503", "504"),
     "✗ GitHub appears to be having an outage — try again in a few minutes (https://www.githubstatus.com)."),
    (lambda s: "Could not resolve host" in s or "unable to access" in s,
     "✗ Network error — cannot reach the remote repository."),
    (lambda s: "could not read Username" in s or "terminal prompts disabled" in s,
     "✗ GitHub rejected the anonymous fetch (asked for a login) — this usually means a GitHub outage;"
     " try again in a few minutes (https://www.githubstatus.com). If it persists, check"
     " `git remote -v` points at a public repo."),
    (lambda s: "Authentication failed" in s,
     "✗ Authentication failed — check your git credentials or SSH key."),
)


def _classify_fetch_failure(stderr: str) -> str:
    """Map git-fetch stderr to a one-line diagnosis (caller also prints the raw first line)."""
    return next((message for matches, message in _FETCH_FAILURE_RULES if matches(stderr)), "✗ Failed to fetch updates from origin.")


def _print_fetch_failure(stderr: str) -> None:
    """Print the classified diagnosis plus the first raw stderr line."""
    stderr = (stderr or "").strip()
    print(_classify_fetch_failure(stderr))
    if stderr:
        print(f"  {stderr.splitlines()[0]}")


def _probe_fork_bomb(argv: list) -> Optional[bool]:
    """Run ``<argv> --version``; True/False = guard message seen/absent, None = probe itself failed."""
    try:
        result = subprocess.run(argv + ["--version"], timeout=15, **_GIT_TEXT_KW)
    except Exception:
        return None
    return "fork bomb" in ((result.stdout or "") + (result.stderr or "")).lower()


def _git_is_trampoline(git_cmd: list) -> bool:
    """Whether *git_cmd* is a broken Git-for-Windows trampoline shim.

    The ~46KB ``bin\\git.exe``/``cmd\\git.exe`` shims re-exec git-core; when they can't find it every call
    dies with the launcher's guard message (a PATH problem, not network). Never raises; unknown states
    report False so a probe failure can't block an update.

    Git for Windows ships two ~46KB shims (``bin\\git.exe``, ``cmd\\git.exe``) that re-exec the real
    ``mingw64\\libexec\\git-core\\git.exe``. See #87876.
    """
    return _probe_fork_bomb(git_cmd) is True


def _portable_git_candidates() -> list:
    """PortableGit candidates: shared root first (where the managed tree actually lives, not the
    profile-scoped HERMES_HOME), then profile home as a fallback for custom layouts.

    The Hermes-managed PortableGit tree lives under the SHARED root (``<root>/git/...``), not the
    profile-scoped HERMES_HOME (``<root>/profiles/<name>``), so a profile-scoped ``hermes update`` must look
    there (monerostar review, #87876).
    """
    from hermes_cli.update_cmd import get_default_hermes_root, get_hermes_home
    candidates = []
    with suppress(Exception):
        candidates += [root / "git" / "mingw64" / "libexec" / "git-core" / "git.exe" for root in (get_default_hermes_root(), Path(get_hermes_home()))]
    return candidates


def _locate_real_git() -> Optional[Path]:
    """Find a real Git-for-Windows ``git-core/git.exe`` (standard locations + managed PortableGit) that runs
    without the trampoline guard. None when nothing suits — callers keep the broken command and let the
    fetch-failure ZIP fallback handle it. A failed probe (None) disqualifies a candidate like a guard hit.

    The trampoline symptom is PATH-level: ``bin\\git.exe`` / ``cmd\\git.exe`` (both ~46KB shims) fail to
    re-exec git-core, while the real binary at ``mingw64\\libexec\\git-core\\git.exe`` (≈4.4MB) works when
    invoked directly (#87876).
    """
    candidates = [
        Path(r"C:\Program Files\Git\mingw64\libexec\git-core\git.exe"),
        Path(r"C:\Program Files (x86)\Git\mingw64\libexec\git-core\git.exe"),
    ] + _portable_git_candidates()
    return next((c for c in candidates if c.exists() and _probe_fork_bomb([str(c)]) is False), None)


def _ensure_non_trampoline_git(git_cmd: list) -> list:
    """Swap a broken Git-for-Windows trampoline for a real git binary so fetch/pull/checkout keep working;
    if none is found leave the command untouched (fetch-failure handler falls back to ZIP). No-op off
    Windows and when git is healthy."""
    from hermes_cli.update_cmd import _locate_real_git
    if sys.platform != "win32" or not _git_is_trampoline(git_cmd):
        return git_cmd
    real_git = _locate_real_git()
    if real_git is None:
        print("⚠ Detected a broken git trampoline and could not locate a real git binary — the update will fall back to the ZIP path.")
        return git_cmd
    print(f"⚠ Detected a broken git trampoline; switching to real git at {real_git}")
    return [str(real_git)] + list(git_cmd[1:])


def _discard_lockfile_churn(git_cmd, repo_root):
    """Restore ``package-lock.json`` files npm rewrote non-deterministically, so the update sees a clean tree
    instead of autostashing every run. Only touches lockfiles whose package.json is NOT also dirty. Best-effort."""
    from hermes_cli.update_cmd import _git_run
    with suppress(Exception):
        diff = _git_run(git_cmd, ["diff", "--name-only"], repo_root)
        if diff.returncode != 0:
            return
        changed = [line.strip() for line in diff.stdout.splitlines()]
        dirty_package_dirs = {Path(p).parent for p in changed if p.endswith("package.json")}
        dirty = [p for p in changed if p.endswith("package-lock.json") and Path(p).parent not in dirty_package_dirs]
        if not dirty:
            return
        _git_run(git_cmd, ["checkout", "--", *dirty], repo_root)
        print(f"→ Discarded npm lockfile churn ({len(dirty)} file(s))")


def _normalize_managed_eol(git_cmd, repo_root):
    """Take a managed checkout off ``core.autocrlf=true`` without leaving it dirty.

    Git for Windows sets ``autocrlf=true`` system-wide, turning LF files CRLF and breaking ``git checkout``
    on update; install.ps1 pins ``false`` but older checkouts never got it and only ``hermes update`` can
    fix them. Pin and cleanup are one operation: under ``autocrlf=true`` a CRLF tree reads clean, so pinning
    alone would expose every file as modified (whole-tree autostash). Pin only after the tree verifies clean
    under it; a checkout we can't fully normalize is left as-is. Only ``true`` rewrites LF->CRLF
    (unset/false/input leave the tree alone). Best-effort.

    Checkouts created before that landed never got the pin and cannot receive it — the bootstrap installer
    reuses its build-pinned ``install.ps1`` forever — so ``hermes update``, which ships with the checkout
    itself, is the only path left that can fix them. See #67730.
    """
    from hermes_cli.update_cmd import _git_run
    # -c, not config: evaluate the tree as it WOULD look pinned, persisting nothing.
    probe = git_cmd + ["-c", "core.autocrlf=false"]

    def _probe_run(*args, **kw):
        return subprocess.run(probe + list(args), cwd=repo_root, **_GIT_TEXT_KW, **kw)

    def _eol_only():
        """Dirty paths whose ONLY change is CRLF; None when either probe fails."""
        all_dirty = _probe_run("diff", "-z", "--name-only")
        # Files with a *content* change ignoring CRLF. ``--name-only --ignore-cr-at-eol`` still LISTS
        # CR-only files; ``--numstat`` honors the filter (no record for them). Records are
        # "<added>\\t<deleted>\\t<path>"; rename detection is off, so exactly one path.
        real_dirty = _probe_run("-c", "core.quotepath=false", "diff", "--numstat", "--ignore-cr-at-eol")
        if all_dirty.returncode != 0 or real_dirty.returncode != 0:
            return None
        return {p for p in all_dirty.stdout.split("\0") if p} - {
            parts[2]
            for parts in (line.split("\t", 2) for line in real_dirty.stdout.splitlines() if line.strip())
            if len(parts) == 3 and parts[2]
        }

    with suppress(Exception):
        if _git_run(git_cmd, ["config", "--get", "core.autocrlf"], repo_root).stdout.strip().lower() != "true":
            return
        eol_only = _eol_only()
        if eol_only is None:
            return
        if eol_only:
            # Pathspec via stdin: thousands of paths exceed the Windows argv limit.
            _probe_run("checkout", "--pathspec-from-file=-", "--pathspec-file-nul", "--",
                       input="\0".join(sorted(eol_only)), check=False)
            if _eol_only():  # still dirty: pinning would only surface churn we failed to clear
                return
            print(f"→ Normalized line-ending churn ({len(eol_only)} file(s))")
        subprocess.run(git_cmd + ["config", "core.autocrlf", "false"], cwd=repo_root, capture_output=True, check=False)
