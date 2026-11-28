"""Machine acceptance for Kanban cards: validation, fixed evaluators and the dispatcher pass.

A card may declare one ``acceptance`` predicate at creation (``tasks.acceptance``, canonical JSON).
Completing such a card is a *submission* (see ``kanban_db_acceptance``); this module decides it.
Only fixed, evaluator-built checks exist — no shell string, argv or script path is accepted, so
the dispatcher never runs a command a model authored.

The checks prove narrow facts: ``file`` that a directory entry exists at a path (``lstat`` only), ``git_commit`` that a
branch descends from a fixed base with at least one new commit, ``pr_checks`` that a PR's
repository-required checks passed at its exact head. They do not prove the work is correct.

This is a lifecycle / command-construction boundary, not OS isolation: an agent running as the
same OS user can still edit this file or the board DB directly.
"""
from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
import secrets
import selectors
import shutil
import signal
import sqlite3
import stat
import subprocess
import tempfile
import time
from typing import Any, Optional

from hermes_cli._startup_fast import project_root_str
from hermes_cli.kanban_pr_acceptance import _PR

logger = logging.getLogger(__name__)

KINDS = ("file", "git_commit", "pr_checks", "tbd")
_KEYS = {
    "file": ({"path"}, {"exists"}),
    "git_commit": ({"repo", "branch", "base"}, set()),
    "pr_checks": ({"pr_url"}, set()),
    "tbd": (set(), set()),
}
_SHA1 = re.compile(r"[0-9a-f]{40}")
# git check-ref-format: no control chars / space / ~ ^ : ? * [ \ anywhere.
_REF_BAD_CHARS = re.compile(r"[\x00-\x20\x7f~^:?*\[\\]")

DEFAULT_TOTAL_DEADLINE = 60.0
DEFAULT_PER_CHECK = 30.0
LEASE_SECONDS = 90
BACKOFF_FIRST = 60
BACKOFF_MAX = 30 * 60
_MAX_RECEIPT_BYTES = 8 * 1024
_MAX_FAILURE_TEXT = 2048
_MAX_GIT_OUTPUT = 4096
_MAX_GIT_STDERR = 16 * 1024


# --------------------------------------------------------------------------- validation

def _abs_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"acceptance.{field} must be a non-empty absolute path string")
    if "\x00" in value or any(ord(c) < 0x20 or ord(c) == 0x7f for c in value):
        raise ValueError(f"acceptance.{field} must not contain control characters")
    if not value.startswith("/") or value.startswith("//"):
        raise ValueError(f"acceptance.{field} must be an absolute path (got {value!r})")
    if ".." in value.split("/"):
        raise ValueError(f"acceptance.{field} must not contain '..' segments")
    return os.path.normpath(value)


def validate_branch(value: Any) -> str:
    """``git check-ref-format --branch`` rules, re-implemented so validation never runs git."""
    if not isinstance(value, str) or not value:
        raise ValueError("acceptance.branch must be a non-empty string")
    bad = (
        value.startswith("-") or value.startswith("/") or value.endswith("/") or value.endswith(".")
        or value == "@" or value == "HEAD" or ".." in value or "@{" in value or "//" in value
        or _REF_BAD_CHARS.search(value)
        or any(part.startswith(".") or part.endswith(".lock") for part in value.split("/"))
    )
    if bad:
        raise ValueError(f"acceptance.branch is not a valid branch name: {value!r}")
    return value


def validate_acceptance(value: Any) -> Optional[str]:
    """Canonical JSON for one acceptance object, or None when absent. Raises ``ValueError`` on an
    unknown kind/key, a wrong type, a relative path or an unsafe ref; never degrades to "no check"."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("acceptance must be a JSON object with a 'kind' "
                         f"(one of {', '.join(KINDS)})")
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError(f"acceptance.kind must be one of {', '.join(KINDS)} (got {kind!r})")
    required, optional = _KEYS[kind]
    keys = set(value) - {"kind"}
    unknown = keys - required - optional
    if unknown:
        raise ValueError(f"acceptance ({kind}) has unknown field(s): {', '.join(sorted(unknown))}")
    missing = required - keys
    if missing:
        raise ValueError(f"acceptance ({kind}) is missing field(s): {', '.join(sorted(missing))}")
    spec: dict[str, Any] = {"kind": kind}
    if kind == "file":
        spec["path"] = _abs_path(value["path"], "path")
        exists = value.get("exists", True)
        if type(exists) is not bool:  # a JSON 0/1 is not a boolean
            raise ValueError("acceptance.exists must be a boolean (true/false)")
        spec["exists"] = exists
    elif kind == "git_commit":
        spec["repo"] = _abs_path(value["repo"], "repo")
        spec["branch"] = validate_branch(value["branch"])
        base = value["base"]
        if not isinstance(base, str) or not _SHA1.fullmatch(base):
            raise ValueError("acceptance.base must be a full 40-character lowercase hex commit id")
        spec["base"] = base
    elif kind == "pr_checks":
        url = value["pr_url"]
        if not isinstance(url, str) or not _PR.fullmatch(url):
            raise ValueError("acceptance.pr_url must be an exact https://github.com/OWNER/REPO/pull/N URL")
        spec["pr_url"] = url
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def parse_acceptance_arg(text: Optional[str]) -> Optional[str]:
    """CLI form: a JSON object, or ``none`` to clear. Returns canonical JSON or None."""
    if text is None:
        return None
    if text.strip().lower() == "none":
        return None

    def _no_constants(name: str):
        raise ValueError(f"acceptance JSON must not contain {name}")

    try:
        value = json.loads(text, parse_constant=_no_constants)
    except json.JSONDecodeError as exc:
        raise ValueError(f"acceptance must be valid JSON: {exc.msg}") from None
    return validate_acceptance(value)


def load_spec(canonical: Optional[str]) -> Optional[dict]:
    if not canonical:
        return None
    try:
        value = json.loads(canonical)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def fingerprint(value: Optional[str]) -> str:
    """Stable fingerprint of a canonical spec / contract text (``-`` = none)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32] if value else "-"


# --------------------------------------------------------------------------- trusted binaries

def trusted_binary(name: str) -> Optional[str]:
    """Absolute path of ``git``/``gh``: the PM-pinned binary, else ``/usr/bin/<name>``. Never PATH —
    the worker controls PATH entries such as its venv and workspace."""
    try:
        from pm.install import installed_package
        package = installed_package(name)
        binary = str(package.binary) if package is not None and package.binary is not None else None
        if binary and os.path.isabs(binary) and os.path.isfile(binary) and os.access(binary, os.X_OK):
            return binary
    except Exception:
        logger.debug("kanban acceptance: pm lookup for %s failed", name, exc_info=True)
    fallback = f"/usr/bin/{name}"
    return fallback if os.path.isfile(fallback) and os.access(fallback, os.X_OK) else None


# --------------------------------------------------------------------------- evaluators

def _result(verdict: str, reason: str, **receipt: Any) -> dict:
    return {"verdict": verdict, "reason": reason, "receipt": receipt}


def _kill_group_members(pgid: int) -> None:
    """SIGKILL every process still in process group ``pgid`` (read from /proc): the group kill is
    re-done member by member so no descendant survives holding the pipes. No-op off Linux."""
    try:
        entries = os.listdir("/proc")
    except OSError:
        return
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as f:
                fields = f.read().decode("ascii", "replace").rsplit(")", 1)[1].split()
        except OSError:
            continue
        if len(fields) > 2 and fields[2] == str(pgid) and fields[0] not in ("Z", "X"):
            try:
                os.kill(int(entry), signal.SIGKILL)  # windows-footgun: ok — POSIX only: walks /proc for one process group
            except OSError:
                pass


class OutputOverflow(subprocess.SubprocessError):
    """A child wrote more than its byte budget; it was killed and its output discarded."""


def run_capped(argv: list[str], *, env: dict[str, str], timeout: float, max_stdout: int, max_stderr: int,
               cwd: Optional[str] = None) -> tuple[int, bytes, bytes]:
    """Run ``argv`` (shell=False) reading stdout/stderr incrementally under byte caps and a wall
    deadline. Overflow raises :class:`OutputOverflow`, the deadline ``TimeoutExpired``; either kills
    the child's whole process group first, so partial output is never returned."""
    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        # detached-spawn: ok — awaited below against an absolute deadline; every exit path kills the whole group
        env=env, cwd=cwd, shell=False, start_new_session=True)
    end = time.monotonic() + timeout
    caps = {proc.stdout.fileno(): max_stdout, proc.stderr.fileno(): max_stderr}
    bufs = {fd: bytearray() for fd in caps}
    sel = selectors.DefaultSelector()
    try:
        for fd in caps:
            sel.register(fd, selectors.EVENT_READ)
        while sel.get_map():
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout)
            for key, _ in sel.select(remaining):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    sel.unregister(key.fd)
                    continue
                bufs[key.fd] += chunk
                if len(bufs[key.fd]) > caps[key.fd]:
                    raise OutputOverflow(f"{os.path.basename(argv[0])} exceeded its output budget")
        rc = proc.wait(timeout=max(end - time.monotonic(), 0.001))
        out_fd, err_fd = proc.stdout.fileno(), proc.stderr.fileno()
        return rc, bytes(bufs[out_fd]), bytes(bufs[err_fd])
    except BaseException:
        # The group outlives its leader: descendants holding the pipes die with it.
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # windows-footgun: ok — the group was created by start_new_session
        except ProcessLookupError:
            pass
        except OSError:
            if proc.poll() is None:
                proc.kill()
        _kill_group_members(proc.pid)
        proc.wait(timeout=10)
        raise
    finally:
        sel.close()
        proc.stdout.close()
        proc.stderr.close()


def evaluate_file(spec: dict) -> dict:
    """Checks only that a directory entry exists at ``path`` (``lstat``: a dangling symlink counts);
    the file is never opened, so nothing about its content is proven."""
    path, want = spec["path"], bool(spec.get("exists", True))
    observed_at = int(time.time())
    try:
        st = os.lstat(path)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR):
            return _result("fail" if want else "pass", "file_missing" if want else "file_absent",
                           path=path, check="lstat_entry_exists", exists=False, observed_at=observed_at)
        return _result("pending", "infra", path=path, errno=errno.errorcode.get(exc.errno, str(exc.errno)),
                       observed_at=observed_at)
    kind = ("dir" if stat.S_ISDIR(st.st_mode) else "symlink" if stat.S_ISLNK(st.st_mode)
            else "file" if stat.S_ISREG(st.st_mode) else "other")
    return _result("pass" if want else "fail", "file_present" if want else "file_unexpected",
                   path=path, check="lstat_entry_exists", exists=True, type=kind, size=int(st.st_size),
                   mtime=int(st.st_mtime), observed_at=observed_at)


def _git_env(home: str) -> dict[str, str]:
    """Minimal env: no inherited ``GIT_*``, no user/system config, no prompts, no lazy fetch."""
    return {
        "PATH": "/usr/bin:/bin", "HOME": home, "XDG_CONFIG_HOME": home, "LANG": "C", "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_LAZY_FETCH": "1", "GIT_NO_REPLACE_OBJECTS": "1", "GIT_OPTIONAL_LOCKS": "0",
        "GIT_GRAFT_FILE": os.devnull,  # info/grafts would let a worker forge ancestry
    }


_GIT_PREFIX = (
    "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "protocol.allow=never",
    "-c", "core.useReplaceRefs=false", "-c", "core.commitGraph=false", "--no-pager", "--no-optional-locks",
)


def _run_git(git: str, repo: str, args: list[str], *, home: str, timeout: float) -> tuple[int, str]:
    """``(returncode, stdout)``; shell=False, fixed argv, byte-capped output, cwd outside the repo."""
    rc, out, _ = run_capped([git, *_GIT_PREFIX, "-C", repo, *args], env=_git_env(home), cwd=home,
                            timeout=timeout, max_stdout=_MAX_GIT_OUTPUT, max_stderr=_MAX_GIT_STDERR)
    return rc, out.decode("utf-8", "replace").strip()


def evaluate_git_commit(spec: dict, *, deadline: float, per_check: float = DEFAULT_PER_CHECK) -> dict:
    repo, branch, base = spec["repo"], spec["branch"], spec["base"]
    observed_at = int(time.time())
    receipt = {"repo": repo, "branch": branch, "base": base, "observed_at": observed_at}
    try:
        if not stat.S_ISDIR(os.stat(repo).st_mode):
            return _result("fail", "repo_missing", **receipt)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR):
            return _result("fail", "repo_missing", **receipt)
        return _result("pending", "infra", **receipt)
    git = trusted_binary("git")
    if git is None:
        return _result("pending", "no_trusted_git", **receipt)
    home = tempfile.mkdtemp(prefix="hermes-acceptance-git-")
    try:
        def call(args: list[str]) -> tuple[int, str]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(git, 0)
            return _run_git(git, repo, args, home=home, timeout=min(per_check, remaining))

        rc, tip = call(["rev-parse", "--quiet", "--verify", "--end-of-options", f"refs/heads/{branch}^{{commit}}"])
        if rc == 1 and not tip:
            return _result("fail", "branch_missing", **receipt)
        if rc != 0 or not _SHA1.fullmatch(tip):
            return _result("pending", "git_error", **receipt)
        receipt["tip"] = tip
        if tip == base:
            return _result("fail", "no_new_commit", **receipt)
        rc, _ = call(["cat-file", "-e", "--end-of-options", f"{base}^{{commit}}"])
        if rc != 0:
            # The declared base is not in this repo: a spec problem for the operator, not failed work.
            return _result("pending", "base_unknown", **receipt)
        rc, _ = call(["merge-base", "--is-ancestor", "--end-of-options", base, tip])
        if rc == 0:
            return _result("pass", "descends_from_base", **receipt)
        if rc == 1:
            # "Not found" is only proof on complete history; a shallow repo hides ancestors. No fetch.
            rc, shallow = call(["rev-parse", "--is-shallow-repository"])
            if rc != 0 or shallow != "false":
                return _result("pending", "history_incomplete", **receipt)
            return _result("fail", "not_descendant", **receipt)
        return _result("pending", "git_error", **receipt)
    except subprocess.TimeoutExpired:
        return _result("pending", "deadline", **receipt)
    except OutputOverflow:
        return _result("pending", "output_overflow", **receipt)
    except (OSError, subprocess.SubprocessError):
        return _result("pending", "infra", **receipt)
    finally:
        shutil.rmtree(home, ignore_errors=True)


def evaluate_pr(pr_url: str, *, assignee: Optional[str], deadline: float, reason_prefix: str = "pr") -> dict:
    """Map a ``collect_acceptance`` receipt: a required check that definitely failed -> fail; all
    required checks proven green -> pass; anything unproven (missing/pending/stale/auth/infra/
    unverifiable_app) -> pending. Runs as the assignee's gh identity only — never ambient."""
    observed_at = int(time.time())
    if not assignee:
        return _result("pending", "no_identity", pr_url=pr_url, observed_at=observed_at)
    gh = trusted_binary("gh")
    if gh is None:
        return _result("pending", "no_trusted_gh", pr_url=pr_url, observed_at=observed_at)
    from hermes_cli.kanban_pr_acceptance import collect_acceptance
    receipt = collect_acceptance(pr_url, None, assignee=assignee, deadline=deadline, gh_binary=gh,
                                 mark_complete=True)
    classification = receipt.get("classification")
    checks = receipt.get("checks") or []
    summary = {
        "pr_url": receipt.get("pr_url") or pr_url, "head_sha": receipt.get("head_sha"),
        "classification": classification, "observed_at": observed_at,
        "checks": [{k: c.get(k) for k in ("name", "classification", "conclusion", "head_sha", "url")}
                   for c in checks][:30],
    }
    if not receipt.get("collection_complete"):
        # Collection stopped early (API error, head moved, nothing required): partial check
        # receipts prove nothing either way.
        return _result("pending", f"{reason_prefix}_{classification or 'missing'}", **summary)
    # Complete collection with the exact head re-verified: a required failure outranks any
    # other unproven check.
    failed = [c for c in checks if c.get("classification") == "failure"]
    if failed:
        summary["failed_checks"] = [str(c.get("name") or "")[:100] for c in failed][:10]
        return _result("fail", f"{reason_prefix}_checks_failed", **summary)
    if receipt.get("ok") and classification == "success":
        return _result("pass", f"{reason_prefix}_checks_passed", **summary)
    return _result("pending", f"{reason_prefix}_{classification or 'missing'}", **summary)


def evaluate_spec(spec: dict, *, assignee: Optional[str], deadline: float,
                  per_check: float = DEFAULT_PER_CHECK) -> dict:
    kind = spec.get("kind")
    if kind == "file":
        return evaluate_file(spec)
    if kind == "git_commit":
        return evaluate_git_commit(spec, deadline=deadline, per_check=per_check)
    if kind == "pr_checks":
        return evaluate_pr(spec["pr_url"], assignee=assignee, deadline=deadline)
    if kind == "tbd":
        return _result("pending", "tbd")
    return _result("pending", "unknown_kind")


def combine(results: list[dict]) -> tuple[str, str]:
    """AND of predicates: any fail -> fail; else any pending -> pending; else pass."""
    for verdict in ("fail", "pending"):
        hits = [r for r in results if r["verdict"] == verdict]
        if hits:
            return verdict, hits[0]["reason"]
    if not results:
        return "pending", "no_predicates"
    return "pass", results[0]["reason"]


# Fixed failure templates: declared fields and observed SHAs only, never file contents or gh stderr.
_FAIL_TEXT = {
    "file_missing": "expected {path} to exist; it does not",
    "file_unexpected": "expected {path} to be absent; a {type} exists there",
    "repo_missing": "repository {repo} does not exist",
    "branch_missing": "branch {branch} does not exist in {repo}",
    "no_new_commit": "branch {branch} in {repo} has no commit after base {base}",
    "not_descendant": "branch {branch} in {repo} (tip {tip}) does not descend from base {base}",
    "pr_checks_failed": "required check(s) failed on {pr_url} at head {head_sha}: {failed_checks}",
    "contract_checks_failed": "completion_contract: required check(s) failed on {pr_url} at head {head_sha}: {failed_checks}",
}
FAIL_REASONS = frozenset(_FAIL_TEXT)


def failure_text(results: list[dict]) -> str:
    parts = []
    for r in results:
        if r["verdict"] != "fail":
            continue
        fields = {k: (", ".join(map(str, v)) if isinstance(v, list) else v) for k, v in r["receipt"].items()
                  if not isinstance(v, (list, dict)) or k == "failed_checks"}
        try:
            parts.append(_FAIL_TEXT[r["reason"]].format_map(_Default(fields)))
        except (KeyError, ValueError):
            parts.append(r["reason"])
    text = ("Machine acceptance failed: " + "; ".join(parts) + ". Fix the work, then call kanban_complete "
            "again; the card returns to machine acceptance.")
    return text[:_MAX_FAILURE_TEXT]


class _Default(dict):
    def __missing__(self, key: str) -> str:
        return "?"


def bounded_receipt(results: list[dict]) -> str:
    text = json.dumps(results, ensure_ascii=False, sort_keys=True)
    if len(text.encode("utf-8")) <= _MAX_RECEIPT_BYTES:
        return text
    slim = [{"verdict": r["verdict"], "reason": r["reason"]} for r in results]
    return json.dumps(slim, sort_keys=True)


def backoff_seconds(attempts: int) -> int:
    return min(BACKOFF_FIRST * (2 ** max(attempts - 1, 0)), BACKOFF_MAX)


# --------------------------------------------------------------------------- the pass

def _estop_allows() -> bool:
    try:
        from gateway.kanban_watchers_common import _kanban_dispatch_allowed
    except ImportError:
        return True
    return _kanban_dispatch_allowed()


def _one_exit_state(pid, host, fingerprint, kind) -> Optional[str]:
    if not pid:
        # Only work that was never claimed had no worker; a claimed run without a recorded pid is
        # an unproven worker, never evidence that none exists.
        return None if kind in ("synthesized", None) else "identity_unknown"
    from hermes_cli import kanban_db as kb
    if not host:
        return "identity_unknown"
    if f"{host}:" != kb._host_prefix():
        return "worker_remote"
    if kb._worker_alive(int(pid), fingerprint):
        return "worker_alive"
    return None


def worker_exit_state(sub) -> Optional[str]:
    """None when the implementation worker AND any approving reviewer worker are proven gone (or
    never existed), else a pending reason (approver reasons prefixed ``approver_``). Only a
    host-local pid with the spawn fingerprint can be checked; anything else waits."""
    impl = _one_exit_state(sub["worker_pid"], sub["worker_host"], sub["worker_fingerprint"], sub["origin_kind"])
    if impl is not None:
        return impl
    if sub.get("approver_kind") != "worker":
        return None
    approver = _one_exit_state(sub.get("approver_pid"), sub.get("approver_host"),
                               sub.get("approver_fingerprint"), "worker")
    return f"approver_{approver}" if approver else None


def run_acceptance_pass(
    conn: sqlite3.Connection, *, board: Optional[str] = None,
    total_deadline: float = DEFAULT_TOTAL_DEADLINE, per_check: float = DEFAULT_PER_CHECK,
    failure_limit: Optional[int] = None, respect_estop: bool = True, evaluate=None,
) -> list[dict]:
    """Decide due machine-acceptance submissions on one board. Call OUTSIDE the dispatch tick lock
    and any write txn: each submission is leased in a short txn, evaluated with no txn open, then
    applied by a CAS txn that a stale lease, edited spec, moved card or reopened parent loses.
    Independent of spawn capacity and ``review_dispatch``; stopped by the emergency stop. Returns
    one ``{task_id, submission_id, verdict, reason}`` per submission it touched.

    Kill boundary: each submission is evaluated by :func:`evaluate_in_child` in its own process
    group under an absolute monotonic deadline of ``min(now + per_check, pass deadline)`` — worker
    exit proof, spec re-validation, profile resolution and ``.env`` reads, env building, lstat/stat,
    git/gh and their JSON all happen there and are SIGKILLed at the deadline (pending ``deadline``).
    Outside it, in this process: the SQLite lease and CAS apply (bounded by SQLite's busy
    handling), spawning the child and decoding its byte-capped result, and a pass's post-commit
    finalization (breaker reset, dependants, workspace cleanup, completion hook). ``evaluate`` lets
    tests run a job in-process."""
    from hermes_cli import kanban_db_acceptance as kba

    if respect_estop and not _estop_allows():
        return []
    evaluate = evaluate or evaluate_in_child
    deadline = time.monotonic() + total_deadline
    owner = f"{_owner_prefix()}:{secrets.token_hex(4)}"
    outcomes: list[dict] = []
    for sub_id in kba.due_submission_ids(conn, int(time.time())):
        if time.monotonic() >= deadline:
            break  # the rest wait for the next tick
        try:
            lease = kba.lease_submission(conn, sub_id, owner, lease_seconds=LEASE_SECONDS)
            if lease is None:
                continue
            if lease.get("cancelled"):
                outcomes.append({"task_id": lease["task_id"], "submission_id": sub_id,
                                 "verdict": "cancelled", "reason": lease["cancelled"]})
                continue
            job = lease_job(lease, deadline=min(time.monotonic() + per_check, deadline), per_check=per_check)
            try:
                results = evaluate(job)
            except Exception:
                logger.exception("kanban acceptance: evaluating submission %s failed", sub_id)
                results = [_result("pending", "evaluator_error")]
            applied = kba.apply_verdict(conn, lease, results, failure_limit=failure_limit)
        except Exception:
            # One damaged row must not stop the patrol; its lease expires and it is retried.
            logger.exception("kanban acceptance: submission %s skipped this pass", sub_id)
            continue
        if applied is None:
            continue
        outcomes.append({"task_id": lease["task_id"], "submission_id": sub_id, **applied})
        logger.info("kanban acceptance: board=%s task=%s submission=%s verdict=%s reason=%s",
                    board or "default", lease["task_id"], sub_id, applied["verdict"], applied["reason"])
    return outcomes


def _owner_prefix() -> str:
    from hermes_cli import kanban_db as kb
    return kb._claimer_id()


def lease_job(lease: dict, *, deadline: float, per_check: float) -> dict:
    """The JSON-able evaluation input for one leased submission (no DB handle crosses over)."""
    sub = lease["submission"]
    return {
        "worker": {k: sub[k] for k in ("worker_pid", "worker_host", "worker_fingerprint", "origin_kind",
                                       "approver_pid", "approver_host", "approver_fingerprint", "approver_kind")},
        "acceptance": lease["acceptance"], "assignee": lease["assignee"],
        "completion_contract": lease["completion_contract"], "deadline": deadline, "per_check": per_check,
    }


def evaluate_job(job: dict) -> list[dict]:
    """Evaluate one job against its absolute ``time.monotonic`` deadline (in-process)."""
    deadline, per_check = float(job["deadline"]), float(job["per_check"])
    gate = worker_exit_state(job["worker"])
    if gate is not None:
        return [_result("pending", gate)]
    try:  # re-validate what the DB holds: a hand-edited row must not reach an evaluator
        spec = load_spec(validate_acceptance(load_spec(job["acceptance"])))
    except ValueError:
        spec = None
    if spec is None:
        return [_result("pending", "spec_invalid")]
    if spec.get("kind") == "tbd":
        return [_result("pending", "tbd")]
    results = [evaluate_spec(spec, assignee=job["assignee"], deadline=deadline, per_check=per_check)]
    contract = job["completion_contract"]
    if contract and contract != "local-only":
        results.append(evaluate_pr(contract, assignee=job["assignee"], deadline=deadline,
                                   reason_prefix="contract"))
    return results


def _evaluate_lease(lease: dict, *, deadline: float, per_check: float) -> list[dict]:
    return evaluate_job(lease_job(lease, deadline=deadline, per_check=per_check))


_CHILD_MAX_STDOUT = 256 * 1024
_CHILD_MAX_STDERR = 64 * 1024
_VERDICTS = ("pass", "fail", "pending")


def evaluate_in_child(job: dict) -> list[dict]:
    """Run :func:`evaluate_job` in ``hermes_cli.kanban_acceptance_job`` (own session / process
    group) and kill the group at the job's deadline. Timeout, overflow, a crash or malformed output
    are pending — never pass or fail."""
    import sys
    remaining = float(job["deadline"]) - time.monotonic()
    if remaining <= 0:
        return [_result("pending", "deadline")]
    root = project_root_str()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH", "")) if p)
    try:
        rc, out, _ = run_capped([sys.executable, "-m", "hermes_cli.kanban_acceptance_job", json.dumps(job)],
                                env=env, cwd=root, timeout=remaining,
                                max_stdout=_CHILD_MAX_STDOUT, max_stderr=_CHILD_MAX_STDERR)
    except subprocess.TimeoutExpired:
        return [_result("pending", "deadline")]
    except OutputOverflow:
        return [_result("pending", "output_overflow")]
    except (OSError, subprocess.SubprocessError):
        return [_result("pending", "evaluator_error")]
    lines = out.decode("utf-8", "replace").strip().splitlines()
    try:
        results = json.loads(lines[-1]) if rc == 0 and lines else None
    except ValueError:
        results = None
    ok = isinstance(results, list) and results and all(
        isinstance(r, dict) and r.get("verdict") in _VERDICTS and isinstance(r.get("reason"), str)
        and isinstance(r.get("receipt"), dict) for r in results)
    return results if ok else [_result("pending", "evaluator_error")]
