"""Backend git operations for the desktop coding rail + review pane.

Mirrors the desktop's Electron-local git ops over the dashboard's authenticated
REST surface so a *remote* gateway acts on the right filesystem. Shells out to
system ``git`` (and ``gh`` for PRs). Reads degrade to ``None``/empty on a
non-repo; mutations raise so the renderer can toast. ``cwd`` is already hardened.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from hermes_cli._subprocess_compat import harden_git_argv, noninteractive_git_env

_GIT_TIMEOUT = 30
_GH_TIMEOUT = 30
_UNTRACKED_LINE_MAX_BYTES = 1024 * 1024
_UNTRACKED_SCAN_CAP = 500
_COMMIT_CONTEXT_DIFF_MAX_CHARS = 120_000
_COMMIT_CONTEXT_UNTRACKED_MAX = 80
_TRUNK_BRANCHES = ("main", "master")


def _run(argv: list[str], cwd: str, timeout: int, env: dict) -> subprocess.CompletedProcess | None:
    """Non-interactive subprocess (stdin nulled, prompts disabled): a credential prompt from
    ``fetch``/``push`` could never be answered from a REST request, so fail fast and surface
    the real auth error in the toast. None when the process could not run at all."""
    try:
        return subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=timeout, stdin=subprocess.DEVNULL, env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git(cwd: str, args: list[str], *, timeout: int = _GIT_TIMEOUT) -> tuple[int, str, str]:
    """(returncode, stdout, stderr) of ``git`` in ``cwd``; never raises on non-zero exit."""
    proc = _run(["git", *harden_git_argv(args)], cwd, timeout, noninteractive_git_env())
    if proc is None:
        return 1, "", "git invocation failed"
    return proc.returncode, proc.stdout, proc.stderr


def _git_out(cwd: str, args: list[str]) -> str:
    """stdout of a git command, or "" on any failure."""
    code, out, _ = _git(cwd, args)
    return out if code == 0 else ""


def _git_line(cwd: str, args: list[str]) -> str:
    """Stripped stdout of a single-value git query ("" on failure)."""
    return _git_out(cwd, args).strip()


def _git_ok(cwd: str, args: list[str]) -> None:
    """Run a git mutation, raising RuntimeError with stderr on failure."""
    code, _, err = _git(cwd, args)
    if code != 0:
        raise RuntimeError(err.strip() or f"git {' '.join(args)} failed")


def _is_dir(cwd: str) -> bool:
    try:
        return Path(cwd).is_dir()
    except OSError:
        return False


def _status_z(cwd: str, *extra: str) -> tuple[int, str]:
    """(returncode, raw) of ``git status --porcelain=v2 -z``."""
    code, raw, _ = _git(cwd, ["status", "--porcelain=v2", *extra, "-z"])
    return code, raw


def _origin_head(cwd: str) -> str:
    """Short name of origin's default branch (``main``), or "" without a remote."""
    return _git_line(cwd, ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"])


def _origin_head_abbrev(cwd: str) -> str:
    return _git_line(cwd, ["rev-parse", "--abbrev-ref", "origin/HEAD"])


def _ref_exists(cwd: str, ref: str) -> bool:
    return _git(cwd, ["rev-parse", "--verify", "--quiet", ref])[0] == 0


# ── shared helpers ───────────────────────────────────────────────────────────


def resolve_rename_path(raw: str) -> str:
    """``old => new`` (and ``dir/{old => new}/f``) → the NEW path, so a row addresses the real file."""
    path = str(raw or "").strip()
    if " => " not in path:
        return path
    head, _, tail = path.partition("{")
    if tail and "}" in tail:
        inner, _, suffix = tail.partition("}")
        _, _, to = inner.partition(" => ")
        return f"{head}{to}{suffix}".replace("//", "/")
    return path.split(" => ")[-1].strip()


def _numstat(cwd: str, args: list[str]) -> dict[str, tuple[int, int]]:
    """``git diff --numstat`` → {path: (added, removed)}; binary files (``-``) → 0."""
    counts: dict[str, tuple[int, int]] = {}
    for line in _git_out(cwd, ["diff", "--numstat", *args]).splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            added, removed = (0 if p == "-" else int(p or 0) for p in parts[:2])
            counts[resolve_rename_path(parts[2])] = (added, removed)
    return counts


def _untracked_insertions(cwd: str, rel: str) -> int:
    """Line count of an untracked file (+N for new files in the review tree). Binary/oversized → 0."""
    try:
        target = Path(cwd) / rel
        if not os.path.isfile(target) or target.stat().st_size > _UNTRACKED_LINE_MAX_BYTES:
            return 0
        data = target.read_bytes()
        if b"\0" in data:
            return 0
        lines = data.count(b"\n")
        return lines + 1 if data and not data.endswith(b"\n") else lines
    except OSError:
        return 0


def _branch_base(cwd: str) -> str | None:
    """Merge-base with the remote default branch for "all branch changes"."""
    head = _origin_head_abbrev(cwd)
    candidates = ([head] if head else []) + ["origin/main", "origin/master", "main", "master"]
    return next((b for b in (_git_line(cwd, ["merge-base", "HEAD", ref]) for ref in candidates) if b), None)


def _default_branch_name(cwd: str) -> str | None:
    """The repo's trunk name ("main"/"master"/…), preferring origin/HEAD."""
    head = _origin_head_abbrev(cwd)
    if head and head != "origin/HEAD":
        return head.split("/", 1)[-1]
    for ref in ("refs/heads/main", "refs/heads/master",
                "refs/remotes/origin/main", "refs/remotes/origin/master"):
        if _ref_exists(cwd, ref):
            return ref.split("/")[-1]
    return None


# ── porcelain v2 status parsing ──────────────────────────────────────────────


def _walk_entries(raw: str):
    """Yield (tag, xy, path) per changed file from porcelain-v2 ``-z`` output,
    skipping branch headers and rename/copy origin-path records."""
    records = iter(raw.split("\0"))
    for rec in records:
        tag = rec[0] if rec else ""
        if tag == "?":
            yield "?", "??", rec[2:]
        elif tag == "u":
            yield "u", rec.split(" ")[1], rec.split(" ", 10)[-1]
        elif tag in ("1", "2"):
            path = rec.split(" ", 8)[-1] if tag == "1" else rec.split(" ", 9)[-1]
            if tag == "2":
                next(records, None)  # rename/copy: the origin path is the next NUL record
            yield tag, rec.split(" ")[1], resolve_rename_path(path)


def _entry_staged(tag: str, xy: str) -> bool:
    """A tracked entry whose index (staged) code is set."""
    return tag in ("1", "2") and xy[0] not in (".", "?")


def _classify(tag: str, xy: str, path: str) -> dict:
    y = xy[1] if len(xy) > 1 else "."
    return {"path": path, "staged": _entry_staged(tag, xy),
            "unstaged": tag == "?" or (tag in ("1", "2") and y not in (".", "?")),
            "untracked": tag == "?", "conflicted": tag == "u"}


def _status_letter(tag: str, xy: str) -> str:
    if tag in ("?", "u"):
        return tag.upper()
    code = xy[0] if xy[0] != "." else (xy[1] if len(xy) > 1 else ".")
    return (code if code != "." else "M").upper()


# ── coding rail ──────────────────────────────────────────────────────────────


def repo_status(cwd: str) -> dict | None:
    """Compact working-tree status for the coding rail. None on a non-repo."""
    if not _is_dir(cwd):
        return None
    code, raw = _status_z(cwd, "--branch")
    if code != 0:
        return None

    branch: str | None = None
    detached = False
    ahead = behind = 0
    for rec in raw.split("\0"):
        if rec.startswith("# branch.head "):
            head = rec[len("# branch.head ") :]
            detached = head == "(detached)"
            branch = None if detached else head
        elif rec.startswith("# branch.ab "):
            for tok in rec.split()[2:]:
                if tok.startswith("+"):
                    ahead = int(tok[1:] or 0)
                elif tok.startswith("-"):
                    behind = int(tok[1:] or 0)

    files = [_classify(tag, xy, path) for tag, xy, path in _walk_entries(raw)]
    # +/- vs HEAD, then fold in untracked insertions (`git diff HEAD` ignores them, so a
    # new-file-only turn would read +0); bounded scan.
    counts = _numstat(cwd, ["HEAD"]).values()
    added = sum(a for a, _ in counts)
    added += sum(_untracked_insertions(cwd, f["path"]) for f in files[:_UNTRACKED_SCAN_CAP] if f["untracked"])
    return {
        "branch": branch, "defaultBranch": _default_branch_name(cwd), "detached": detached,
        "ahead": ahead, "behind": behind,
        **{flag: sum(f[flag] for f in files) for flag in ("staged", "unstaged", "untracked", "conflicted")},
        "changed": len(files), "added": added, "removed": sum(r for _, r in counts), "files": files[:200],
    }


# ── review pane ──────────────────────────────────────────────────────────────


def _review_result(cwd: str, files: list[dict], base: str | None) -> dict:
    """Sorted rows; untracked rows with no counts get their insertion count filled in."""
    files.sort(key=lambda f: f["path"])
    for file in files:
        if file["status"] == "?" and file["added"] == 0 and file["removed"] == 0:
            file["added"] = _untracked_insertions(cwd, file["path"])
    return {"files": files, "base": base}


def review_list(cwd: str, scope: str, base_ref: str | None) -> dict:
    """Changed files for a scope. Mirrors the Electron reviewList shapes."""
    if not _is_dir(cwd):
        return {"files": [], "base": None}
    if scope in ("branch", "lastTurn"):
        base = _branch_base(cwd) if scope == "branch" else base_ref
        if not base:
            return {"files": [], "base": None}
        rng = f"{base}...HEAD" if scope == "branch" else base
        files = [
            {"path": path, "added": a, "removed": r, "status": "M", "staged": False}
            for path, (a, r) in _numstat(cwd, [rng]).items()
        ]
        if scope == "lastTurn":
            seen = {f["path"] for f in files}
            files += [
                {"path": path, "added": 0, "removed": 0, "status": "?", "staged": False}
                for tag, _xy, path in _walk_entries(_status_z(cwd)[1])
                if tag == "?" and path not in seen
            ]
        return _review_result(cwd, files, base)

    code, raw = _status_z(cwd)
    if code != 0:
        return {"files": [], "base": None}
    staged = _numstat(cwd, ["--cached"])
    unstaged = _numstat(cwd, [])
    files = []
    for tag, xy, path in _walk_entries(raw):
        sa, sr = staged.get(path, (0, 0))
        ua, ur = unstaged.get(path, (0, 0))
        files.append({"path": path, "added": sa + ua, "removed": sr + ur,
                      "status": _status_letter(tag, xy), "staged": _entry_staged(tag, xy)})
    return _review_result(cwd, files, None)


def _all_add_diff(cwd: str, file_path: str) -> str:
    """Synthesized all-add diff for an untracked file (``--no-index`` exits non-zero by design)."""
    return _git(cwd, ["diff", "--no-index", "--", os.devnull, file_path])[1]


def review_diff(cwd: str, file_path: str, scope: str, base_ref: str | None, staged: bool) -> str:
    if not _is_dir(cwd):
        return ""
    if scope == "branch":
        base = _branch_base(cwd)
        return _git_out(cwd, ["diff", f"{base}...HEAD", "--", file_path]) if base else ""
    if scope == "lastTurn":
        return _git_out(cwd, ["diff", base_ref, "--", file_path]) if base_ref else ""
    if staged:
        return _git_out(cwd, ["diff", "--cached", "--", file_path])
    worktree = _git_out(cwd, ["diff", "--", file_path])
    return worktree if worktree.strip() else _all_add_diff(cwd, file_path)


def file_diff_vs_head(cwd: str, file_path: str) -> str:
    """Working-tree-vs-HEAD diff for one file (the preview's diff view). Unlike
    review_diff, never all-adds a clean tracked file; only a genuinely untracked one."""
    if not _is_dir(cwd):
        return ""
    head = _git_out(cwd, ["diff", "HEAD", "--", file_path])
    if head.strip():
        return head
    status = _git_out(cwd, ["status", "--porcelain", "--", file_path])
    return _all_add_diff(cwd, file_path) if status.strip().startswith("??") else ""


def review_stage(cwd: str, file_path: str | None) -> dict:
    _git_ok(cwd, ["add", "--", file_path] if file_path else ["add", "-A"])
    return {"ok": True}


def review_unstage(cwd: str, file_path: str | None) -> dict:
    _git_ok(cwd, ["reset", "-q", "HEAD", *(["--", file_path] if file_path else [])])
    return {"ok": True}


def review_revert(cwd: str, file_path: str | None) -> dict:
    """Discard changes back to the committed state (restore tracked, remove untracked)."""
    target = ["--", file_path or "."]
    _git(cwd, ["checkout", "HEAD", *target])
    _git(cwd, ["clean", "-fd", *target])
    return {"ok": True}


def review_rev_parse(cwd: str, ref: str | None) -> str | None:
    return _git_line(cwd, ["rev-parse", ref or "HEAD"]) or None


def _has_staged(raw: str) -> bool:
    return any(_entry_staged(tag, xy) for tag, xy, _ in _walk_entries(raw))


def review_commit(cwd: str, message: str, push: bool) -> dict:
    """Commit the working tree; stage everything first when nothing is staged."""
    if not _has_staged(_status_z(cwd)[1]):
        _git_ok(cwd, ["add", "-A"])
    _git_ok(cwd, ["commit", "-m", message])
    if push:
        _review_push(cwd)
    return {"ok": True}


def _review_push(cwd: str) -> None:
    if _git_line(cwd, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]):
        _git_ok(cwd, ["push"])
        return
    branch = _git_line(cwd, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch and branch != "HEAD":
        _git_ok(cwd, ["push", "-u", "origin", branch])


def review_push(cwd: str) -> dict:
    _review_push(cwd)
    return {"ok": True}


def review_commit_context(cwd: str) -> dict:
    """Diff of what WILL commit + recent subjects, for drafting a commit message."""
    code, raw = _status_z(cwd) if _is_dir(cwd) else (1, "")
    if code != 0:
        return {"diff": "", "recent": ""}
    entries = list(_walk_entries(raw))
    diff = _git_out(cwd, ["diff", "--cached"] if _has_staged(raw) else ["diff", "HEAD"])
    if len(diff) > _COMMIT_CONTEXT_DIFF_MAX_CHARS:
        omitted = len(diff) - _COMMIT_CONTEXT_DIFF_MAX_CHARS
        diff = f"{diff[:_COMMIT_CONTEXT_DIFF_MAX_CHARS]}\n# diff truncated: {omitted} chars omitted\n"
    untracked = [path for tag, _xy, path in entries if tag == "?"]
    if untracked:
        visible = untracked[:_COMMIT_CONTEXT_UNTRACKED_MAX]
        diff += "\n# New (untracked) files:\n" + "".join(f"#   {p}\n" for p in visible)
        if len(untracked) > len(visible):
            diff += f"#   ... {len(untracked) - len(visible)} more omitted\n"
    return {"diff": diff or "", "recent": _git_line(cwd, ["log", "-n", "10", "--pretty=format:%s"])}


# ── ship flow (gh) ───────────────────────────────────────────────────────────


def _gh(cwd: str, args: list[str]) -> tuple[bool, str]:
    if not shutil.which("gh"):
        return False, ""
    # GH_PROMPT_DISABLED: gh's documented kill-switch for interactive prompts.
    env = noninteractive_git_env()
    env["GH_PROMPT_DISABLED"] = "1"
    proc = _run(["gh", *args], cwd, _GH_TIMEOUT, env)
    if proc is None:
        return False, ""
    return proc.returncode == 0, proc.stdout or ""


def _gh_json(cwd: str, args: list[str]):
    """Parsed JSON stdout of a successful gh call, else None."""
    ok, out = _gh(cwd, args)
    if not ok:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def review_ship_info(cwd: str) -> dict:
    """gh availability/auth + this branch's PR. ghReady false when gh missing/unauthed."""
    if not _is_dir(cwd) or not _gh(cwd, ["auth", "status"])[0]:
        return {"ghReady": False, "pr": None}
    pr = _gh_json(cwd, ["pr", "view", "--json", "url,state,number"])
    if pr and pr.get("url"):
        return {"ghReady": True, "pr": {"url": pr["url"], "state": pr.get("state"), "number": pr.get("number")}}
    return {"ghReady": True, "pr": None}


# GraphQL asks per branch so the answer can't be crowded out like a `gh pr list`
# page. Aliases carry many branches per request; 50 stays inside GitHub's node budget.
_PR_QUERY_BRANCH_CHUNK = 50
_PR_QUERY_BRANCH_CAP = 300
_PR_NODE_FIELDS = "number state isDraft isCrossRepository title url headRefName"


def _pr_query(owner: str, name: str, branches: list[str], numbers: list[int]) -> str:
    fields = [
        f"b{i}: pullRequests(headRefName: {json.dumps(branch)}, first: 5, "
        f"orderBy: {{field: CREATED_AT, direction: DESC}}) "
        f"{{ nodes {{ {_PR_NODE_FIELDS} }} }}"
        for i, branch in enumerate(branches)
    ]
    # A PR recovered from a transcript is known by number; asking directly also
    # yields its branch, so it lands in the same by-branch map.
    fields += [f"n{i}: pullRequest(number: {n}) {{ {_PR_NODE_FIELDS} }}" for i, n in enumerate(numbers)]
    return (f"query {{ repository(owner: {json.dumps(owner)}, name: {json.dumps(name)}) {{\n"
            + "\n".join(fields) + "\n} }")


def _pr_payload(pr: dict) -> dict:
    return {"branch": str(pr.get("headRefName")), "draft": bool(pr.get("isDraft")),
            "number": int(pr.get("number") or 0), "state": str(pr.get("state") or "").lower(),
            "title": str(pr.get("title") or ""), "url": str(pr.get("url") or "")}


def _own_pr(key: str, field: dict) -> dict | None:
    """The PR a GraphQL alias resolved to. Asked-by-number (``n<i>``) → ours by construction (a
    fork PR can't come from our own transcript). Fork PRs share our branch namespace (a
    contributor's `main` would badge a trunk session with a stranger's PR), so by-branch
    lookups only count own-repo nodes."""
    if key.startswith("n"):
        return field
    return next((n for n in (field.get("nodes") or []) if n and not n.get("isCrossRepository")), None)


def review_pr_list(cwd: str, branches: list[str], numbers: list[int] = None) -> dict:
    """PRs on the given branches (plus any asked for by number) — queried per branch
    rather than paging the repo's newest PRs and hoping ours are in the page."""
    not_ready = {"ghReady": False, "prs": []}
    if not _is_dir(cwd):
        return not_ready
    wanted = list(dict.fromkeys(str(b) for b in (branches or []) if b))[:_PR_QUERY_BRANCH_CAP]
    by_number = list(dict.fromkeys(int(n) for n in (numbers or []) if n))[:_PR_QUERY_BRANCH_CAP]
    if not wanted and not by_number:
        return not_ready
    repo_ok, repo_out = _gh(cwd, ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"])
    owner, _, name = repo_out.strip().partition("/")
    if not repo_ok or not owner or not name:
        # gh missing, unauthenticated, or no GitHub remote — all "nothing to badge".
        return not_ready

    prs: list[dict] = []
    step = _PR_QUERY_BRANCH_CHUNK
    chunks = ([(wanted[i:i + step], []) for i in range(0, len(wanted), step)]
              + [([], by_number[i:i + step]) for i in range(0, len(by_number), step)])
    for branch_chunk, number_chunk in chunks:
        # A failed/malformed chunk drops its branches; the rest still resolve.
        data = _gh_json(cwd, ["api", "graphql", "-f", f"query={_pr_query(owner, name, branch_chunk, number_chunk)}"])
        repository = ((data or {}).get("data") or {}).get("repository") or {}
        for key, field in repository.items():
            pr = _own_pr(key, field) if field else None
            if pr and pr.get("headRefName"):
                prs.append(_pr_payload(pr))
    return {"ghReady": True, "prs": prs}


def review_create_pr(cwd: str) -> dict:
    """Create a PR for the current branch (push first), letting gh fill title/body."""
    try:
        _review_push(cwd)
    except RuntimeError:
        pass
    created, out = _gh(cwd, ["pr", "create", "--fill"])
    if not created:
        raise RuntimeError("gh pr create failed (is gh installed and authenticated?)")
    url = next((line for line in reversed(out.strip().splitlines()) if line.strip()), "")
    return {"url": url}


# ── worktrees & branches ─────────────────────────────────────────────────────


def worktree_list(cwd: str) -> list[dict]:
    """``git worktree list --porcelain`` -> one dict per tree (main tree first)."""
    trees: list[dict] = []
    for line in _git_out(cwd, ["worktree", "list", "--porcelain"]).split("\n"):
        if line.startswith("worktree "):
            trees.append({"path": line[9:].strip(), "branch": None, "isMain": not trees,
                          "detached": False, "locked": False})
        elif not trees:
            continue
        elif line.startswith("branch "):
            trees[-1]["branch"] = line[7:].strip().replace("refs/heads/", "", 1)
        elif line == "detached":
            trees[-1]["detached"] = True
        elif line.startswith("locked"):
            trees[-1]["locked"] = True
    return trees


def _main_root(cwd: str) -> str:
    trees = worktree_list(cwd)
    return trees[0]["path"] if trees else cwd


_BRANCH_SANITIZERS = (
    (r"\s+", "-"), (r"[^\w./-]", ""), (r"-{2,}", "-"), (r"/{2,}", "/"), (r"\.{2,}", "."), (r"^[-./]+|[-./]+$", ""),
)


def _sanitize_branch(name: str) -> str:
    value = str(name or "")
    for pattern, repl in _BRANCH_SANITIZERS:
        value = re.sub(pattern, repl, value)
    return value


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower())
    slug = re.sub(r"^-+|-+$", "", slug)[:40].rstrip("-")
    return slug or "work"


def _default_branch(cwd: str) -> str:
    return (
        _origin_head(cwd).replace("origin/", "", 1)
        or _git_line(cwd, ["config", "--get", "init.defaultBranch"])
        or next((b for b in _TRUNK_BRANCHES if _git_line(cwd, ["show-ref", "--verify", f"refs/heads/{b}"])), "")
    )


def _ensure_repo(cwd: str) -> None:
    """A new project folder may not be a repo (or has no commit to branch from);
    init it with a root commit so worktrees just work. No-op for a committed repo."""
    if _git_line(cwd, ["rev-parse", "--is-inside-work-tree"]) != "true":
        _git_ok(cwd, ["init"])
        needs_root = True
    else:
        needs_root = not _ref_exists(cwd, "HEAD")
    if needs_root:
        _git_ok(cwd, ["-c", "user.email=hermes@localhost", "-c", "user.name=Hermes",
                      "commit", "--allow-empty", "-m", "Initial commit"])


def _unique_dir(base: str) -> str:
    candidates = itertools.chain([base], (f"{base}-{n}" for n in itertools.count(2)))
    return next(c for c in candidates if not os.path.exists(c))


def _remote_of_ref(cwd: str, name: str) -> str:
    """The remote a ref belongs to ("origin" for "origin/main"), or "" when ``name`` is not a
    remote-tracking ref here. Asks git rather than assuming "origin" (mirrors the Electron op)."""
    if "/" not in name or _git(cwd, ["show-ref", "--verify", "--quiet", f"refs/remotes/{name}"])[0] != 0:
        return ""
    return name.split("/", 1)[0]


def _worktree_for_existing(root: str, raw_name: str) -> dict:
    """Check out an existing local or remote-tracking branch into a worktree (or switch the
    main tree when it IS the trunk).

    "origin/feature" is a remote-tracking ref — `git worktree add <dir> origin/feature` detaches
    HEAD. Create a local branch of the same short name tracking it, like `git switch feature`
    does (Electron-op parity).
    """
    requested = _sanitize_branch(raw_name)
    if not requested:
        raise RuntimeError("Branch name is required.")
    # "origin/feature" is a remote-tracking ref, not a branch git can check out — `git worktree add <dir>
    # origin/feature` detaches HEAD. Create a local branch with the same short name that tracks the remote
    # ref, like `git switch feature` does for a branch on exactly one remote. (Parity with the Electron op;
    # a remote gateway serves this mirror, so the desktop's convert-a-branch flow must behave identically.
    # #81724)
    remote = _remote_of_ref(root, requested)
    existing = requested.split("/", 1)[1] if remote else requested
    if not remote and existing == _default_branch(root):
        _git_ok(root, ["switch", existing])
        return {"path": root, "branch": existing, "repoRoot": root}
    target = _unique_dir(os.path.join(root, ".worktrees", _slugify(existing)))
    if remote:
        # Best-effort freshness; on failure (offline, branch gone) the last known ref is still
        # there to branch from.
        _git(root, ["fetch", remote, existing])
        _git_ok(root, ["worktree", "add", "--track", "-b", existing, target, requested])
    else:
        _git_ok(root, ["worktree", "add", target, existing])
    return {"path": target, "branch": existing, "repoRoot": root}


def worktree_add(cwd: str, options: dict) -> dict:
    _ensure_repo(cwd)
    root = _main_root(cwd)
    options = options or {}
    if options.get("existingBranch"):
        return _worktree_for_existing(root, options["existingBranch"])

    slug = _slugify(options.get("name") or f"work-{os.urandom(4).hex()}")
    branch = _sanitize_branch(options.get("branch") or "") or f"hermes/{slug}"
    target = _unique_dir(os.path.join(root, ".worktrees", slug))
    args = ["worktree", "add", "-b", branch, target]
    if options.get("base"):
        base = str(options["base"])
        # Fetch just that branch so a stale remote-tracking ref is fresh; fetch failures
        # (offline / no remote) are ignored — git uses the local ref or raises a clear error
        # below if it is entirely missing.
        if base.startswith("origin/"):
            _git(root, ["fetch", "origin", base[len("origin/"):]])
            # Branching off a remote-tracking ref auto-wires upstream tracking; the user wants
            # a standalone local branch (Electron-op parity).
            args.append("--no-track")
        args.append(base)
    code, _, err = _git(root, args)
    if code != 0:
        if "already exists" not in (err or "").lower():
            raise RuntimeError(err.strip() or "git worktree add failed")
        _git_ok(root, ["worktree", "add", target, branch])
    return {"path": target, "branch": branch, "repoRoot": root}


def worktree_remove(cwd: str, worktree_path: str, force: bool) -> dict:
    _git_ok(_main_root(cwd), ["worktree", "remove", *(["--force"] if force else []), worktree_path])
    return {"removed": worktree_path}


def _ref_names(cwd: str, *patterns: str, fmt: str = "%(refname:short)") -> list[str]:
    """Non-empty ``for-each-ref`` lines, newest commit first."""
    out = _git_out(cwd, ["for-each-ref", f"--format={fmt}", "--sort=-committerdate", *patterns])
    return [line.strip() for line in out.split("\n") if line.strip()]


def branch_list(cwd: str) -> list[dict]:
    """Branches for the convert-a-branch picker: local heads first, then remote-tracking
    refs with no local head yet (a teammate's branch without a manual checkout).

    Parity with the Electron op — a remote gateway serves this mirror for the same desktop UI (#81724).
    """
    locals_ = _ref_names(cwd, "refs/heads")
    if not locals_:
        return []
    path_by_branch = {t["branch"]: t["path"] for t in worktree_list(cwd) if t["branch"]}
    trunk = _default_branch(cwd)
    local_set = set(locals_)
    # "origin/HEAD" is a symbolic alias, not a branch; a remote branch tracked locally is
    # reachable via its head (checking out the remote ref detaches).
    remotes = [name for name in _ref_names(cwd, "refs/remotes")
               if not name.endswith("/HEAD") and name.split("/", 1)[-1] not in local_set]
    return [
        *({"name": name, "checkedOut": name in path_by_branch, "isDefault": bool(trunk and name == trunk),
           "isRemote": False, "worktreePath": path_by_branch.get(name)} for name in locals_),
        # Remote rows: no local checkout, and never the local trunk.
        *({"name": name, "checkedOut": False, "isDefault": False, "isRemote": True,
           "worktreePath": None} for name in remotes),
    ]


def branch_switch(cwd: str, branch: str) -> dict:
    target = _sanitize_branch(branch)
    if not target:
        raise RuntimeError("Branch name is required.")
    _git_ok(cwd, ["switch", target])
    return {"branch": target}


def base_branch_list(cwd: str) -> list[dict]:
    """Local heads + remote-tracking refs for the base-branch picker; the remote default
    (origin/HEAD) is flagged so the UI can preselect it."""
    lines = _ref_names(cwd, "refs/heads", "refs/remotes", fmt="%(refname:short)\t%(committerdate:iso)")
    if not lines:
        return []
    # origin/HEAD when a remote exists; else the local default so a no-remote repo still flags its trunk.
    default = _origin_head(cwd) or _default_branch(cwd)
    return [
        {"name": name, "isRemote": name.startswith("origin/"), "isDefault": bool(default and name == default)}
        for name in (line.split("\t")[0] for line in lines)
    ]
