"""Gitea exact-PR acceptance transport for kanban completion contracts.

A contract ``https://ORIGIN/OWNER/REPO/pulls/N`` is assessed by a trusted, read-only
gate command that the card's ASSIGNEE profile configures for that origin in
``<profile home>/gitea-acceptance.json``::

    {"origins": {"https://forge.example": {
        "argv": ["/abs/trusted/gate", "--flag"],     # PR URL is appended as the last argv
        "policy_id": "forge-main-v1",
        "qualification_only": false,                 # explicit; must equal the receipt's
        "repos": ["OWNER/REPO"]}}}

There is no global switch: an unconfigured origin, a repo outside ``repos``, a
missing ``qualification_only`` or any unknown key (``allow_qualification``) fails
closed. The gate prints :func:`hermes_cli.gitea_acceptance.gate_receipt` JSON; this
module re-verifies its binding and rebuilds a sanitized receipt, so command output
and stderr never reach the task event log. The command runs with the assignee
profile's own credentials only (launch-profile residue and agent sockets scrubbed),
with bounded time and output.
"""
from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import threading
from pathlib import Path

from hermes_cli.gitea_acceptance import HEX40, ORIGIN, REPO, parse_pr_url

CONFIG_NAME = "gitea-acceptance.json"
MAX_CONFIG_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 256 * 1024
TIMEOUT_SECONDS = 60
_ENTRY_KEYS = {"argv", "policy_id", "qualification_only", "repos"}
# Failure classifications a gate may report; anything else is reported as infra.
_GATE_FAILURES = {"missing", "stale", "closed", "unmerged", "binding", "protection", "workflow",
                  "untrusted", "pending", "failure", "custody", "review", "merge", "infra"}
# Ambient agent sockets are credentials no profile .env recorded; the overlay re-adds the profile's own.
_AMBIENT_AGENT_KEYS = ("SSH_AUTH_SOCK", "SSH_AGENT_PID", "GPG_AGENT_INFO")
RECOVERY = ("Fix required Gitea checks, review/merge provenance, trusted workflow or runner custody, "
            "then retry completion; writer-posted statuses never substitute.")


class GiteaAcceptanceError(Exception):
    """A refusal with a fixed, sanitized detail and its classification."""

    def __init__(self, classification: str, detail: str):
        super().__init__(detail)
        self.classification = classification
        self.detail = detail


def is_gitea_contract(value) -> bool:
    return parse_pr_url(value) is not None


def _safe_owner(info: os.stat_result, *, allow_root: bool = False) -> bool:
    uid = os.getuid()  # windows-footgun: ok — POSIX only (collect_gitea_acceptance gates on os.name)
    owners = {uid, 0} if allow_root else {uid}
    return info.st_uid in owners and not info.st_mode & 0o022


def _read_config(profile_home: str) -> dict:
    home = Path(profile_home)
    try:
        if not (stat.S_ISDIR((home_info := os.stat(home)).st_mode) and _safe_owner(home_info)):
            raise GiteaAcceptanceError("auth", "Assignee profile home has unsafe ownership or permissions.")
        fd = os.open(home / CONFIG_NAME, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        raise GiteaAcceptanceError("auth", f"Assignee profile has no {CONFIG_NAME}.") from None
    except OSError:
        raise GiteaAcceptanceError("auth", f"{CONFIG_NAME} is not a regular readable file.") from None
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or not _safe_owner(info) or info.st_size > MAX_CONFIG_BYTES:
            raise GiteaAcceptanceError("auth", f"{CONFIG_NAME} has unsafe ownership, permissions or size.")
        raw = handle.read(MAX_CONFIG_BYTES + 1)
    try:
        config = json.loads(raw)
    except ValueError:
        raise GiteaAcceptanceError("auth", f"{CONFIG_NAME} is not valid JSON.") from None
    if not isinstance(config, dict) or config.keys() != {"origins"} or not isinstance(config["origins"], dict):
        raise GiteaAcceptanceError("auth", f"{CONFIG_NAME} must hold exactly an 'origins' object.")
    return config["origins"]


def _validate_entry(entry) -> dict:
    invalid = GiteaAcceptanceError("auth", f"{CONFIG_NAME} origin entry is invalid; required keys: "
                                           + ", ".join(sorted(_ENTRY_KEYS)) + " (no others).")
    if not isinstance(entry, dict) or entry.keys() != _ENTRY_KEYS:
        raise invalid
    argv, repos = entry["argv"], entry["repos"]
    if (not isinstance(argv, list) or not argv
            or not all(isinstance(v, str) and v and "\x00" not in v for v in argv)
            or not os.path.isabs(argv[0]) or os.path.normpath(argv[0]) != argv[0]
            or not isinstance(entry["policy_id"], str) or not entry["policy_id"].strip()
            or not isinstance(entry["qualification_only"], bool)
            or not isinstance(repos, list) or not repos
            or not all(isinstance(r, str) and REPO.fullmatch(r) for r in repos)
            or len(set(repos)) != len(repos)):
        raise invalid
    return entry


def _check_command(path: str) -> None:
    """The gate binary and its directory must not be writable by anyone but its owner (user/root)."""
    real = os.path.realpath(path)
    try:
        info, parent = os.stat(real), os.stat(os.path.dirname(real))
    except OSError:
        raise GiteaAcceptanceError("auth", "Configured Gitea gate command is missing.") from None
    if (not stat.S_ISREG(info.st_mode) or not info.st_mode & 0o100
            or not _safe_owner(info, allow_root=True) or not _safe_owner(parent, allow_root=True)):
        raise GiteaAcceptanceError("auth", "Configured Gitea gate command is not a trusted executable.")


def load_origin_entry(profile_home: str, origin: str, repo: str) -> dict:
    origins = _read_config(profile_home)
    if not all(isinstance(k, str) and ORIGIN.fullmatch(k) for k in origins):
        raise GiteaAcceptanceError("auth", f"{CONFIG_NAME} has an origin that is not an exact https origin.")
    if origin not in origins:
        raise GiteaAcceptanceError("auth", f"No trusted Gitea gate is configured for {origin} in the assignee profile.")
    entry = _validate_entry(origins[origin])
    if repo not in entry["repos"]:
        raise GiteaAcceptanceError("auth", f"{repo} is not in the configured repos for {origin}.")
    _check_command(entry["argv"][0])
    return entry


def _child_env(profile_home: str) -> dict[str, str]:
    from tools.environments.local import hermes_subprocess_env, served_profile_child_env
    base = hermes_subprocess_env(inherit_credentials=False)
    for key in _AMBIENT_AGENT_KEYS:
        base.pop(key, None)
    return served_profile_child_env(base=base, target_home=profile_home, inherit_credentials=True)


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # windows-footgun: ok — POSIX only (gated in collect_gitea_acceptance)
    except (ProcessLookupError, PermissionError):
        pass


def _run_bounded(argv: list[str], *, env: dict, cwd: str) -> bytes:
    """stdout of ``argv`` capped in time and size; stderr is discarded, never persisted."""
    from agent.memory_provider import spawn_context_thread
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, env=env, cwd=cwd, close_fds=True,
                                start_new_session=True)
    except OSError:
        raise GiteaAcceptanceError("infra", "Gitea gate command could not be started.") from None
    stdout = proc.stdout
    assert stdout is not None
    buffer, overflow = bytearray(), threading.Event()

    def pump():
        while chunk := os.read(stdout.fileno(), 65536):
            if len(buffer) + len(chunk) > MAX_OUTPUT_BYTES:
                overflow.set()
                _kill(proc)
                return
            buffer.extend(chunk)

    reader = spawn_context_thread(pump, name="gitea-gate-stdout")
    reader.start()
    code, lingered = None, False
    try:
        code = proc.wait(timeout=TIMEOUT_SECONDS)
        reader.join(5)  # a grandchild still holding stdout is killed with the group below
    except subprocess.TimeoutExpired:
        pass
    finally:
        lingered = reader.is_alive()
        if code is None or lingered:
            _kill(proc)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            reader.join(5)
        stdout.close()
    if overflow.is_set():
        raise GiteaAcceptanceError("infra", "Gitea gate command output exceeded its bound.")
    if code is None or lingered:
        raise GiteaAcceptanceError("infra", "Gitea gate command timed out.")
    if code != 0:
        raise GiteaAcceptanceError("infra", f"Gitea gate command exited with status {code}.")
    return bytes(buffer)


def _clean_checks(checks, sha: str) -> list[dict]:
    if not isinstance(checks, list) or not checks:
        raise GiteaAcceptanceError("infra", "Positive Gitea receipt has no checks.")
    cleaned = []
    for check in checks:
        if (not isinstance(check, dict) or not isinstance(check.get("name"), str) or not check["name"]
                or check.get("head_sha") != sha or check.get("classification") != "success"):
            raise GiteaAcceptanceError("infra", "Positive Gitea receipt has a check not bound to its head.")
        cleaned.append({k: check[k] for k in ("name", "id", "run_id", "run_attempt", "runner_id",
                                              "head_sha", "classification", "completed_at", "run_completed_at") if k in check})
    return cleaned


def _production_evidence(raw: dict, sha: str) -> dict:
    merge, review = raw.get("merge"), raw.get("review")
    if (not isinstance(merge, dict) or merge.get("merged") is not True
            or not isinstance(merge.get("merge_commit_sha"), str) or not HEX40.fullmatch(merge["merge_commit_sha"])
            or not isinstance(merge.get("merged_by"), str) or not merge["merged_by"]
            or not isinstance(review, dict) or review.get("commit_id") != sha
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"]
            or not isinstance(review.get("id"), int)):
        raise GiteaAcceptanceError("infra", "Production Gitea receipt lacks merge provenance or an exact-head approval.")
    return {"merge": {k: merge[k] for k in ("merged", "merge_commit_sha", "merged_by", "merged_by_id", "merged_at")
                      if k in merge},
            "review": {k: review[k] for k in ("id", "reviewer", "reviewer_id", "commit_id", "submitted_at") if k in review}}


def verify_receipt(raw, *, contract: str, origin: str, repo: str, entry: dict) -> dict:
    """Sanitized receipt for gate output ``raw``; raises on any binding mismatch."""
    if not isinstance(raw, dict):
        raise GiteaAcceptanceError("infra", "Gitea gate printed no receipt object.")
    if raw.get("qualification_only") is True and entry["qualification_only"] is False:
        raise GiteaAcceptanceError("qualification", "Qualification-only evidence cannot complete a production-scoped origin.")
    expected = {"pr_url": contract, "provider": "gitea", "origin": origin, "repo": repo,
                "policy_id": entry["policy_id"], "qualification_only": entry["qualification_only"]}
    if any(raw.get(k) != v or type(raw.get(k)) is not type(v) for k, v in expected.items()):
        raise GiteaAcceptanceError("binding", "Gitea receipt is not bound to this PR, origin, repo and policy.")
    sha = raw.get("head_sha")
    receipt = {**expected, "ok": False, "head_sha": sha if isinstance(sha, str) and HEX40.fullmatch(sha) else None,
               "checks": [], "classification": "infra"}
    if raw.get("ok") is False:
        classification = raw.get("classification")
        receipt["classification"] = classification if classification in _GATE_FAILURES else "infra"
        receipt["detail"] = f"Gitea gate refused acceptance ({receipt['classification']})."
        return receipt
    if raw.get("ok") is not True or raw.get("classification") != "success" or receipt["head_sha"] is None:
        raise GiteaAcceptanceError("infra", "Gitea receipt has no valid verdict or exact head.")
    receipt.update(ok=True, classification="success", checks=_clean_checks(raw.get("checks"), sha))
    if not entry["qualification_only"]:
        receipt.update(_production_evidence(raw, sha))
    return receipt


def collect_gitea_acceptance(contract: str, published_pr: str | None, assignee: str | None) -> dict:
    origin, repo, _number = parse_pr_url(contract)
    receipt = {"ok": False, "classification": "missing", "head_sha": None, "pr_url": contract,
               "checks": [], "provider": "gitea", "origin": origin, "repo": repo, "recovery": RECOVERY}
    if published_pr and published_pr != contract:
        return dict(receipt, detail="metadata.published_pr must equal the declared Gitea PR contract.")
    if not assignee:
        return dict(receipt, classification="auth",
                    detail="Gitea acceptance needs an assigned profile; ambient credentials are never used.")
    if os.name != "posix":
        return dict(receipt, classification="auth",
                    detail="Gitea acceptance needs POSIX ownership checks; this host cannot verify its gate.")
    from hermes_cli.kanban_pr_acceptance import _assignee_profile_home, _GateAuthError
    try:
        profile_home = _assignee_profile_home(assignee)
        entry = load_origin_entry(profile_home, origin, repo)
        output = _run_bounded([*entry["argv"], contract], env=_child_env(profile_home), cwd=profile_home)
        try:
            raw = json.loads(output)
        except ValueError:
            raise GiteaAcceptanceError("infra", "Gitea gate printed malformed JSON.") from None
        return {**verify_receipt(raw, contract=contract, origin=origin, repo=repo, entry=entry),
                "recovery": RECOVERY}
    except _GateAuthError as exc:
        return dict(receipt, classification="auth", detail=f"{exc}; fix the assignee profile, then retry completion.")
    except GiteaAcceptanceError as exc:
        return dict(receipt, classification=exc.classification, detail=exc.detail)
