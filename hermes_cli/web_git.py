"""Backend git operations for the desktop coding rail + Codex-style review pane.

The desktop's git affordances (coding-rail status, worktree lanes, review pane,
branch switch) run as Electron-local git on the user's machine. On a *remote*
gateway those would operate on the wrong filesystem, so this module mirrors them
over the dashboard's authenticated REST surface — the same pattern as ``/api/fs``.

Everything shells out to the system ``git`` (and ``gh`` for ship info / PRs).
Reads degrade to ``None`` / empty on a non-repo; mutations raise so the renderer
can surface a toast. Callers pass an already path-hardened ``cwd``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

_GIT_TIMEOUT = 30
_GH_TIMEOUT = 30
_MAX_BUFFER = 32 * 1024 * 1024
_UNTRACKED_LINE_MAX_BYTES = 1024 * 1024
_UNTRACKED_SCAN_CAP = 500
_COMMIT_CONTEXT_DIFF_MAX_CHARS = 120_000
_COMMIT_CONTEXT_UNTRACKED_MAX = 80
_TRUNK_BRANCHES = ("main", "master")


def _git(cwd: str, args: list[str], *, timeout: int = _GIT_TIMEOUT) -> tuple[int, str, str]:
    """Run ``git`` in ``cwd``. Returns (returncode, stdout, stderr); never raises
    on a non-zero exit (callers decide what an error means)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return 1, "", "git invocation failed"
    return proc.returncode, proc.stdout, proc.stderr


def _git_out(cwd: str, args: list[str]) -> str:
    """stdout of a git command, or "" on any failure."""
    code, out, _ = _git(cwd, args)
    return out if code == 0 else ""


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


# ── shared helpers ───────────────────────────────────────────────────────────


def resolve_rename_path(raw: str) -> str:
    """``old => new`` (and ``dir/{old => new}/f``) → the NEW path, so a row
    addresses the real file for diff/stage."""
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
    out = _git_out(cwd, ["diff", "--numstat", *args])
    counts: dict[str, tuple[int, int]] = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added = 0 if parts[0] == "-" else int(parts[0] or 0)
        removed = 0 if parts[1] == "-" else int(parts[1] or 0)
        counts[resolve_rename_path(parts[2])] = (added, removed)
    return counts


def _untracked_insertions(cwd: str, rel: str) -> int:
    """Line count of an untracked file (newlines + a final unterminated line),
    so the review tree can show +N for new files. Binary / oversized → 0."""
    try:
        target = Path(cwd) / rel
        st = target.stat()
        if not os.path.isfile(target) or st.st_size > _UNTRACKED_LINE_MAX_BYTES:
            return 0
        data = target.read_bytes()
        if b"\0" in data:
            return 0
        lines = data.count(b"\n")
        return lines + 1 if data and not data.endswith(b"\n") else lines
    except OSError:
        return 0


def _fill_untracked_counts(cwd: str, files: list[dict]) -> None:
    for file in files:
        if file["status"] == "?" and file["added"] == 0 and file["removed"] == 0:
            file["added"] = _untracked_insertions(cwd, file["path"])


def _branch_base(cwd: str) -> str | None:
    """Merge-base with the remote default branch for "all branch changes"."""
    candidates: list[str] = []
    head = _git_out(cwd, ["rev-parse", "--abbrev-ref", "origin/HEAD"]).strip()
    if head:
        candidates.append(head)
    candidates += ["origin/main", "origin/master", "main", "master"]
    for ref in candidates:
        base = _git_out(cwd, ["merge-base", "HEAD", ref]).strip()
        if base:
            return base
    return None


def _default_branch_name(cwd: str) -> str | None:
    """The repo's trunk name ("main"/"master"/…), preferring origin/HEAD."""
    head = _git_out(cwd, ["rev-parse", "--abbrev-ref", "origin/HEAD"]).strip()
    if head and head != "origin/HEAD":
        return head.split("/", 1)[-1]
    for ref in (
        "refs/heads/main",
        "refs/heads/master",
        "refs/remotes/origin/main",
        "refs/remotes/origin/master",
    ):
        code, _, _ = _git(cwd, ["rev-parse", "--verify", "--quiet", ref])
        if code == 0:
            return ref.split("/")[-1]
    return None


# ── porcelain v2 status parsing ──────────────────────────────────────────────


def _walk_entries(raw: str):
    """Yield (tag, xy, path) per changed file from ``git status --porcelain=v2 -z``,
    skipping branch headers and the rename/copy origin-path records. One walker
    feeds the rail, the review list, and the commit flow."""
    records = raw.split("\0")
    i = 0
    while i < len(records):
        rec = records[i]
        tag = rec[0] if rec else ""
        if tag == "?":
            yield "?", "??", rec[2:]
        elif tag == "u":
            yield "u", rec.split(" ")[1], rec.split(" ", 10)[-1]
        elif tag in ("1", "2"):
            xy = rec.split(" ")[1]
            path = rec.split(" ", 8)[-1] if tag == "1" else rec.split(" ", 9)[-1]
            if tag == "2":
                i += 1  # rename/copy: the origin path is the next NUL record
            yield tag, xy, resolve_rename_path(path)
        i += 1


def _entry_staged(tag: str, xy: str) -> bool:
    """A tracked entry whose index (staged) code is set."""
    return tag in ("1", "2") and xy[0] not in (".", "?")


def _classify(tag: str, xy: str, path: str) -> dict:
    y = xy[1] if len(xy) > 1 else "."
    return {
        "path": path,
        "staged": _entry_staged(tag, xy),
        "unstaged": tag == "?" or (tag in ("1", "2") and y not in (".", "?")),
        "untracked": tag == "?",
        "conflicted": tag == "u",
    }


def _status_letter(tag: str, xy: str) -> str:
    if tag in ("?", "u"):
        return tag.upper() if tag == "u" else "?"
    code = xy[0] if xy[0] != "." else (xy[1] if len(xy) > 1 else ".")
    return (code if code != "." else "M").upper()


# ── coding rail ──────────────────────────────────────────────────────────────


def repo_status(cwd: str) -> dict | None:
    """Compact working-tree status for the coding rail. None on a non-repo."""
    if not _is_dir(cwd):
        return None

    code, raw, _ = _git(cwd, ["status", "--porcelain=v2", "--branch", "-z"])
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

    # +/- vs HEAD (tracked), then fold in untracked insertions — `git diff HEAD`
    # ignores them, so a new-file-only turn would otherwise read +0 (bounded scan).
    added = removed = 0
    for a, r in _numstat(cwd, ["HEAD"]).values():
        added += a
        removed += r
    added += sum(_untracked_insertions(cwd, f["path"]) for f in files[:_UNTRACKED_SCAN_CAP] if f["untracked"])

    return {
        "branch": branch,
        "defaultBranch": _default_branch_name(cwd),
        "detached": detached,
        "ahead": ahead,
        "behind": behind,
        "staged": sum(f["staged"] for f in files),
        "unstaged": sum(f["unstaged"] for f in files),
        "untracked": sum(f["untracked"] for f in files),
        "conflicted": sum(f["conflicted"] for f in files),
        "changed": len(files),
        "added": added,
        "removed": removed,
        "files": files[:200],
    }


# ── review pane ──────────────────────────────────────────────────────────────


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
            _, raw, _ = _git(cwd, ["status", "--porcelain=v2", "-z"])
            files += [
                {"path": path, "added": 0, "removed": 0, "status": "?", "staged": False}
                for tag, _xy, path in _walk_entries(raw)
                if tag == "?" and path not in seen
            ]
        files.sort(key=lambda f: f["path"])
        _fill_untracked_counts(cwd, files)
        return {"files": files, "base": base}

    code, raw, _ = _git(cwd, ["status", "--porcelain=v2", "-z"])
    if code != 0:
        return {"files": [], "base": None}
    staged = _numstat(cwd, ["--cached"])
    unstaged = _numstat(cwd, [])

    files = []
    for tag, xy, path in _walk_entries(raw):
        sa, sr = staged.get(path, (0, 0))
        ua, ur = unstaged.get(path, (0, 0))
        files.append(
            {
                "path": path,
                "added": sa + ua,
                "removed": sr + ur,
                "status": _status_letter(tag, xy),
                "staged": _entry_staged(tag, xy),
            }
        )
    files.sort(key=lambda f: f["path"])
    _fill_untracked_counts(cwd, files)
    return {"files": files, "base": None}


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
    if worktree.strip():
        return worktree
    # Untracked: synthesize an all-add diff (exits non-zero by design).
    _, out, _ = _git(cwd, ["diff", "--no-index", "--", os.devnull, file_path])
    return out


def file_diff_vs_head(cwd: str, file_path: str) -> str:
    """Working-tree-vs-HEAD diff for one file (the preview's diff view). Unlike
    review_diff, never all-adds a clean tracked file; only a genuinely untracked one."""
    if not _is_dir(cwd):
        return ""
    head = _git_out(cwd, ["diff", "HEAD", "--", file_path])
    if head.strip():
        return head
    status = _git_out(cwd, ["status", "--porcelain", "--", file_path])
    if not status.strip().startswith("??"):
        return ""
    _, out, _ = _git(cwd, ["diff", "--no-index", "--", os.devnull, file_path])
    return out


def review_stage(cwd: str, file_path: str | None) -> dict:
    _git_ok(cwd, ["add", "--", file_path] if file_path else ["add", "-A"])
    return {"ok": True}


def review_unstage(cwd: str, file_path: str | None) -> dict:
    _git_ok(cwd, ["reset", "-q", "HEAD", "--", file_path] if file_path else ["reset", "-q", "HEAD"])
    return {"ok": True}


def review_revert(cwd: str, file_path: str | None) -> dict:
    """Discard changes back to the committed state (restore tracked, remove untracked)."""
    target = ["--", file_path] if file_path else ["--", "."]
    _git(cwd, ["checkout", "HEAD", *target])
    _git(cwd, ["clean", "-fd", *target])
    return {"ok": True}


def review_rev_parse(cwd: str, ref: str | None) -> str | None:
    out = _git_out(cwd, ["rev-parse", ref or "HEAD"]).strip()
    return out or None


def review_commit(cwd: str, message: str, push: bool) -> dict:
    """Commit the working tree; stage everything first when nothing is staged."""
    _, raw, _ = _git(cwd, ["status", "--porcelain=v2", "-z"])
    if not any(_entry_staged(tag, xy) for tag, xy, _ in _walk_entries(raw)):
        _git_ok(cwd, ["add", "-A"])
    _git_ok(cwd, ["commit", "-m", message])
    if push:
        _review_push(cwd)
    return {"ok": True}


def _review_push(cwd: str) -> None:
    upstream = _git_out(cwd, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]).strip()
    if upstream:
        _git_ok(cwd, ["push"])
        return
    branch = _git_out(cwd, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    if branch and branch != "HEAD":
        _git_ok(cwd, ["push", "-u", "origin", branch])


def review_push(cwd: str) -> dict:
    _review_push(cwd)
    return {"ok": True}


def review_commit_context(cwd: str) -> dict:
    """Diff of what WILL commit + recent subjects, for drafting a commit message."""
    if not _is_dir(cwd):
        return {"diff": "", "recent": ""}
    code, raw, _ = _git(cwd, ["status", "--porcelain=v2", "-z"])
    if code != 0:
        return {"diff": "", "recent": ""}
    entries = list(_walk_entries(raw))

    has_staged = any(_entry_staged(tag, xy) for tag, xy, _ in entries)
    diff = _git_out(cwd, ["diff", "--cached"]) if has_staged else _git_out(cwd, ["diff", "HEAD"])
    if len(diff) > _COMMIT_CONTEXT_DIFF_MAX_CHARS:
        omitted = len(diff) - _COMMIT_CONTEXT_DIFF_MAX_CHARS
        diff = f"{diff[:_COMMIT_CONTEXT_DIFF_MAX_CHARS]}\n# diff truncated: {omitted} chars omitted\n"

    untracked = [path for tag, _xy, path in entries if tag == "?"]
    if untracked:
        visible = untracked[:_COMMIT_CONTEXT_UNTRACKED_MAX]
        note = "\n# New (untracked) files:\n" + "".join(f"#   {p}\n" for p in visible)
        if len(untracked) > len(visible):
            note += f"#   ... {len(untracked) - len(visible)} more omitted\n"
        diff = f"{diff}{note}" if diff else note

    return {"diff": diff or "", "recent": _git_out(cwd, ["log", "-n", "10", "--pretty=format:%s"]).strip()}


# ── ship flow (gh) ───────────────────────────────────────────────────────────


def _gh(cwd: str, args: list[str]) -> tuple[bool, str]:
    if not shutil.which("gh"):
        return False, ""
    try:
        proc = subprocess.run(
            ["gh", *args], cwd=cwd, capture_output=True, text=True, timeout=_GH_TIMEOUT
        )
    except (OSError, subprocess.SubprocessError):
        return False, ""
    return proc.returncode == 0, proc.stdout or ""


def review_ship_info(cwd: str) -> dict:
    """gh availability/auth + this branch's PR. ghReady false when gh missing/unauthed."""
    if not _is_dir(cwd):
        return {"ghReady": False, "pr": None}
    auth_ok, _ = _gh(cwd, ["auth", "status"])
    if not auth_ok:
        return {"ghReady": False, "pr": None}
    view_ok, out = _gh(cwd, ["pr", "view", "--json", "url,state,number"])
    if not view_ok:
        return {"ghReady": True, "pr": None}
    try:
        pr = json.loads(out)
    except json.JSONDecodeError:
        return {"ghReady": True, "pr": None}
    if pr and pr.get("url"):
        return {"ghReady": True, "pr": {"url": pr["url"], "state": pr.get("state"), "number": pr.get("number")}}
    return {"ghReady": True, "pr": None}


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


def _parse_worktrees(out: str) -> list[dict]:
    trees: list[dict] = []
    cur: dict | None = None
    for line in out.split("\n"):
        if line.startswith("worktree "):
            if cur:
                trees.append(cur)
            cur = {"path": line[9:].strip(), "branch": None, "detached": False, "bare": False, "locked": False}
        elif cur is None:
            continue
        elif line.startswith("branch "):
            cur["branch"] = line[7:].strip().replace("refs/heads/", "", 1)
        elif line == "detached":
            cur["detached"] = True
        elif line == "bare":
            cur["bare"] = True
        elif line.startswith("locked"):
            cur["locked"] = True
    if cur:
        trees.append(cur)
    return trees


def worktree_list(cwd: str) -> list[dict]:
    out = _git_out(cwd, ["worktree", "list", "--porcelain"])
    if not out:
        return []
    return [
        {
            "path": tree["path"],
            "branch": tree["branch"],
            "isMain": index == 0,
            "detached": tree["detached"],
            "locked": tree["locked"],
        }
        for index, tree in enumerate(_parse_worktrees(out))
    ]


def _main_root(cwd: str) -> str:
    for tree in worktree_list(cwd):
        if tree["isMain"]:
            return tree["path"]
    return cwd


def _sanitize_branch(name: str) -> str:
    value = str(name or "")
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^\w./-]", "", value)
    value = re.sub(r"-{2,}", "-", value)
    value = re.sub(r"/{2,}", "/", value)
    value = re.sub(r"\.{2,}", ".", value)
    return re.sub(r"^[-./]+|[-./]+$", "", value)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower())
    slug = re.sub(r"^-+|-+$", "", slug)[:40].rstrip("-")
    return slug or "work"


def _default_branch(cwd: str) -> str:
    remote = _git_out(
        cwd, ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"]
    ).strip().replace("origin/", "", 1)
    if remote:
        return remote
    configured = _git_out(cwd, ["config", "--get", "init.defaultBranch"]).strip()
    if configured:
        return configured
    for branch in _TRUNK_BRANCHES:
        if _git_out(cwd, ["show-ref", "--verify", f"refs/heads/{branch}"]).strip():
            return branch
    return ""


def _ensure_repo(cwd: str) -> None:
    """A new project folder may not be a repo (or has no commit to branch from);
    init it with a root commit so worktrees just work. No-op for a committed repo."""
    inside = _git_out(cwd, ["rev-parse", "--is-inside-work-tree"]).strip()
    needs_root = False
    if inside != "true":
        _git_ok(cwd, ["init"])
        needs_root = True
    else:
        code, _, _ = _git(cwd, ["rev-parse", "--verify", "HEAD"])
        needs_root = code != 0
    if needs_root:
        _git_ok(
            cwd,
            [
                "-c",
                "user.email=hermes@localhost",
                "-c",
                "user.name=Hermes",
                "commit",
                "--allow-empty",
                "-m",
                "Initial commit",
            ],
        )


def _unique_dir(base: str) -> str:
    candidate = base
    n = 1
    while os.path.exists(candidate):
        n += 1
        candidate = f"{base}-{n}"
    return candidate


def worktree_add(cwd: str, options: dict) -> dict:
    _ensure_repo(cwd)
    root = _main_root(cwd)
    options = options or {}

    existing = _sanitize_branch(options.get("existingBranch") or "")
    if options.get("existingBranch"):
        if not existing:
            raise RuntimeError("Branch name is required.")
        if existing == _default_branch(root):
            _git_ok(root, ["switch", existing])
            return {"path": root, "branch": existing, "repoRoot": root}
        target = _unique_dir(os.path.join(root, ".worktrees", _slugify(existing)))
        _git_ok(root, ["worktree", "add", target, existing])
        return {"path": target, "branch": existing, "repoRoot": root}

    slug = _slugify(options.get("name") or f"work-{os.urandom(4).hex()}")
    branch = _sanitize_branch(options.get("branch") or "") or f"hermes/{slug}"
    target = _unique_dir(os.path.join(root, ".worktrees", slug))
    args = ["worktree", "add", "-b", branch, target]
    if options.get("base"):
        base = str(options["base"])
        # Remote-tracking branches may be stale or missing; fetch just that
        # branch so the local ref is up to date before branching. Ignore fetch
        # failures (offline / no remote) — git will use whatever local ref
        # exists, or raise a clear error below if the ref is entirely missing.
        if base.startswith("origin/"):
            remote_branch = base[len("origin/"):]
            _git(root, ["fetch", "origin", remote_branch])
        args.append(base)
    code, _, err = _git(root, args)
    if code != 0:
        if "already exists" in (err or "").lower():
            _git_ok(root, ["worktree", "add", target, branch])
        else:
            raise RuntimeError(err.strip() or "git worktree add failed")
    return {"path": target, "branch": branch, "repoRoot": root}


def worktree_remove(cwd: str, worktree_path: str, force: bool) -> dict:
    root = _main_root(cwd)
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(worktree_path)
    _git_ok(root, args)
    return {"removed": worktree_path}


def branch_list(cwd: str) -> list[dict]:
    out = _git_out(
        cwd, ["for-each-ref", "--format=%(refname:short)", "--sort=-committerdate", "refs/heads"]
    )
    if not out:
        return []
    trees = worktree_list(cwd)
    path_by_branch = {t["branch"]: t["path"] for t in trees if t["branch"]}
    trunk = _default_branch(cwd)
    return [
        {
            "name": name,
            "checkedOut": name in path_by_branch,
            "isDefault": bool(trunk and name == trunk),
            "worktreePath": path_by_branch.get(name),
        }
        for name in (line.strip() for line in out.split("\n"))
        if name
    ]


def branch_switch(cwd: str, branch: str) -> dict:
    target = _sanitize_branch(branch)
    if not target:
        raise RuntimeError("Branch name is required.")
    _git_ok(cwd, ["switch", target])
    return {"branch": target}


def base_branch_list(cwd: str) -> list[dict]:
    """Local heads + remote-tracking refs for the base-branch picker.

    The remote default (origin/HEAD) is flagged so the UI can preselect it.
    """
    out = _git_out(
        cwd,
        [
            "for-each-ref",
            "--format=%(refname:short)\t%(committerdate:iso)",
            "--sort=-committerdate",
            "refs/heads",
            "refs/remotes",
        ],
    )
    if not out:
        return []
    remote_default = _git_out(
        cwd, ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"]
    ).strip()
    local_default = _default_branch(cwd) if not remote_default else ""
    result: list[dict] = []
    for line in out.split("\n"):
        line = line.strip()
        if not line:
            continue
        name = line.split("\t")[0]
        result.append(
            {
                "name": name,
                "isRemote": name.startswith("origin/"),
                # origin/HEAD when a remote exists; otherwise the local
                # default (main/master/init.defaultBranch) so a no-remote
                # repo still flags its trunk.
                "isDefault": bool(
                    (remote_default and name == remote_default)
                    or (not remote_default and local_default and name == local_default)
                ),
            }
        )
    return result
