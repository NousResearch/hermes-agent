"""File tools keep the documented write sandbox, write denylist and read denylist under ../ and symlink spellings.

Hermes documents three file-tool boundaries (website/docs/user-guide/security.md, "File write safety"):

* ``HERMES_WRITE_SAFE_ROOT``: ``write_file`` / ``patch`` may only land inside the listed roots.
* Protected paths: the Hermes-home ``.env`` / OAuth stores, ``~/.ssh`` keys and ``authorized_keys`` are
  never written; ``~/.ssh/config`` is approval-gated (a ``-q`` run has no approver, so it is blocked).
* Read denylist: ``auth.json`` / ``.env`` under the Hermes home and project-local ``.env`` files are never
  returned by ``read_file`` / ``search_files``.

There is no read confinement to a workspace (reads outside the cwd are allowed), so none is asserted.
Each boundary is judged on the RESOLVED target, so spelling the same file through ``../``, an absolute
path, a symlink inside the workspace, or a symlinked parent directory must get the same verdict as the
direct spelling. Everything runs through the real agent tool path: ``hermes chat -q`` with the loopback
model issuing the tool calls; outcomes are read on disk and in the tool results Hermes sent back.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.e2e.core.security import _helpers as H
from tests.e2e.core.security._traversal import digest, run_tool_calls

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks and ../ spellings")


CONFINED_SCENARIOS = (
    "write_dotdot", "write_absolute", "write_symlink_file", "write_symlink_parent", "patch_symlink_file",
    "patch_dotdot", "v4a_add_symlink_parent", "v4a_add_absolute", "v4a_delete_symlink_parent")
# Hermes quarantines an unparseable auth.json as auth.json.corrupt at startup; the copy holds the same secrets.
QUARANTINE_SCENARIOS = ("read_quarantined_auth_copy", "search_quarantined_auth_copy")
PROTECTED_SCENARIOS = (
    "env_direct_patch", "env_symlink_patch", "env_dotdot_patch", "oauth_dangling_symlink_write",
    "authorized_keys_symlink_parent", "authorized_keys_dotdot", "ssh_key_create_symlink_parent",
    "ssh_config_symlink_parent_gated", "read_auth_symlink", "read_webhooks_dotdot", "read_google_oauth_symlink_dir",
    "read_project_env_symlink", "search_through_symlink_dir")

KNOWN: dict[str, str] = dict.fromkeys(
    QUARANTINE_SCENARIOS, "#121278 read denylist misses auth.json.corrupt, the quarantined copy of a corrupt auth store")


@dataclass
class Layout:
    """``<root>/home`` is HOME, ``<root>/home/work`` the workspace and cwd, ``<root>/outside`` a foreign dir."""

    root: Path
    home: Path = field(init=False)
    ws: Path = field(init=False)
    hh: Path = field(init=False)
    out: Path = field(init=False)
    marks: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.home, self.out = self.root / "home", self.root / "outside"
        self.ws, self.hh = self.home / "work", self.home / ".hermes"
        for d in (self.ws, self.out, self.hh, self.home / ".ssh"):
            d.mkdir(parents=True, exist_ok=True)

    def plant(self, path: Path, label: str, *, json_store: bool = False) -> str:
        """Write a canary file. ``json_store`` writes a well-formed auth store (a malformed auth.json is
        quarantined by Hermes at startup, which would move the canary elsewhere)."""
        self.marks[label] = mark = H.canary(label)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"version": 1, "providers": {}, "credential_pool": {}, "e2e_mark": mark}) if json_store else mark
        path.write_text(f"{body}\n", encoding="utf-8")
        return mark

    def link(self, name: str, target: Path) -> None:
        (self.ws / name).symlink_to(target)


@dataclass
class Step:
    """One tool call. Several spellings share one subject (there is one Hermes-home .env), so a breach
    through one spelling also reds its siblings; the tool result in the failure names the spelling that
    did not refuse. ``expect``: ``absent`` (subject must not exist after), ``unchanged`` (subject bytes
    identical), ``hidden`` (the protected file's canary never appears in the tool result), ``setup`` (no check)."""

    scenario: str
    tool: str
    args: dict[str, Any]
    expect: str
    subject: Any = None  # Path for absent/unchanged; (protected path, its canary) for hidden


def _v4a(op: str, path: str, body: str = "") -> dict[str, Any]:
    return {"mode": "patch", "patch": f"*** Begin Patch\n*** {op} File: {path}\n{body}*** End Patch\n"}


def _replace(path: str, old: str) -> dict[str, Any]:
    return {"mode": "replace", "path": path, "old_string": old, "new_string": "PWNED"}


def _confined_steps(lay: Layout) -> list[Step]:
    """HERMES_WRITE_SAFE_ROOT=<workspace>: every write whose resolved target is outside is refused."""
    target = lay.out / "target.txt"
    t_mark = lay.plant(target, "outside-target")
    lay.plant(lay.out / "doomed.txt", "outside-doomed")
    lay.plant(lay.ws / "inner.txt", "inner")
    lay.link("link.txt", target)
    lay.link("linkdir", lay.out)
    lay.link("innerlink.txt", lay.ws / "inner.txt")
    lay.marks["token"] = token = H.canary("oauth-token")
    (lay.hh / "auth.json").write_text('{"providers": {"nous": {"access_token": "%s"' % token, encoding="utf-8")
    quarantined = (lay.hh / "auth.json.corrupt", token)
    return [
        Step("write_dotdot", "write_file", {"path": "../../outside/new-dotdot.txt", "content": "x"},
             "absent", lay.out / "new-dotdot.txt"),
        Step("write_absolute", "write_file", {"path": str(lay.out / "new-abs.txt"), "content": "x"},
             "absent", lay.out / "new-abs.txt"),
        # reading first satisfies the stale-overwrite guard, so the sandbox is what must refuse
        Step("setup_read_link", "read_file", {"path": "link.txt"}, "setup"),
        Step("write_symlink_file", "write_file", {"path": "link.txt", "content": "PWNED"}, "unchanged", target),
        Step("write_symlink_parent", "write_file", {"path": "linkdir/new-link.txt", "content": "x"},
             "absent", lay.out / "new-link.txt"),
        Step("patch_symlink_file", "patch", _replace("link.txt", t_mark), "unchanged", target),
        Step("patch_dotdot", "patch", _replace("../../outside/target.txt", t_mark), "unchanged", target),
        Step("v4a_add_symlink_parent", "patch", _v4a("Add", "linkdir/new-v4a.txt", "+x\n"),
             "absent", lay.out / "new-v4a.txt"),
        Step("v4a_add_absolute", "patch", _v4a("Add", str(lay.out / "new-v4a-abs.txt"), "+x\n"),
             "absent", lay.out / "new-v4a-abs.txt"),
        Step("v4a_delete_symlink_parent", "patch", _v4a("Delete", "linkdir/doomed.txt"),
             "unchanged", lay.out / "doomed.txt"),
        Step("read_quarantined_auth_copy", "read_file", {"path": "../.hermes/auth.json.corrupt"}, "hidden", quarantined),
        Step("search_quarantined_auth_copy", "search_files", {"pattern": token, "path": "../.hermes"}, "hidden",
             quarantined),
        Step("control_write_inside", "write_file", {"path": "sub/inside.txt", "content": "inside-ok"}, "setup"),
        Step("setup_read_innerlink", "read_file", {"path": "innerlink.txt"}, "setup"),
        Step("control_symlink_inside", "write_file", {"path": "innerlink.txt", "content": "inner-rewritten"}, "setup"),
    ]


def _protected_steps(lay: Layout) -> list[Step]:
    """No sandbox: protected paths refuse writes and credential files refuse reads under every spelling."""
    env, keys, ssh_config = lay.hh / ".env", lay.home / ".ssh" / "authorized_keys", lay.home / ".ssh" / "config"
    lay.plant(keys, "authorized-keys")
    lay.plant(ssh_config, "ssh-config")
    auth = lay.plant(lay.hh / "auth.json", "auth-json", json_store=True)
    # one credential file per read spelling: read_file dedups repeat reads of an unchanged file, which
    # would hide a leak on the second spelling of the same target
    hooks = lay.plant(lay.hh / "webhook_subscriptions.json", "webhook-subs")
    google = lay.plant(lay.hh / "auth" / "google_oauth.json", "google-oauth")
    proj_env = lay.plant(lay.ws / "proj" / ".env", "project-env")
    lay.plant(lay.home / "notes.txt", "plain-notes")
    for name, target in (("envlink", env), ("sshdir", lay.home / ".ssh"), ("oauthlink", lay.hh / ".anthropic_oauth.json"),
                         ("authlink.txt", lay.hh / "auth.json"), ("hhdir", lay.hh), ("cfg.txt", lay.ws / "proj" / ".env"),
                         ("notes-link.txt", lay.home / "notes.txt")):
        lay.link(name, target)
    env_mark = lay.marks["env-sentinel"]
    return [
        Step("env_direct_patch", "patch", _replace(str(env), env_mark), "unchanged", env),
        Step("env_symlink_patch", "patch", _replace("envlink", env_mark), "unchanged", env),
        Step("env_dotdot_patch", "patch", _replace("../.hermes/.env", env_mark), "unchanged", env),
        Step("oauth_dangling_symlink_write", "write_file", {"path": "oauthlink", "content": "{}"},
             "absent", lay.hh / ".anthropic_oauth.json"),
        Step("setup_read_keys", "read_file", {"path": "sshdir/authorized_keys"}, "setup"),
        Step("authorized_keys_symlink_parent", "write_file", {"path": "sshdir/authorized_keys", "content": "evil\n"},
             "unchanged", keys),
        Step("authorized_keys_dotdot", "write_file", {"path": "../.ssh/authorized_keys", "content": "evil\n"},
             "unchanged", keys),
        Step("ssh_key_create_symlink_parent", "write_file", {"path": "sshdir/id_ed25519", "content": "k\n"},
             "absent", lay.home / ".ssh" / "id_ed25519"),
        Step("setup_read_ssh_config", "read_file", {"path": "sshdir/config"}, "setup"),
        Step("ssh_config_symlink_parent_gated", "write_file", {"path": "sshdir/config", "content": "Host evil\n"},
             "unchanged", ssh_config),
        Step("read_auth_symlink", "read_file", {"path": "authlink.txt"}, "hidden", (lay.hh / "auth.json", auth)),
        Step("read_webhooks_dotdot", "read_file", {"path": "../.hermes/webhook_subscriptions.json"}, "hidden",
             (lay.hh / "webhook_subscriptions.json", hooks)),
        Step("read_google_oauth_symlink_dir", "read_file", {"path": "hhdir/auth/google_oauth.json"}, "hidden",
             (lay.hh / "auth" / "google_oauth.json", google)),
        Step("read_project_env_symlink", "read_file", {"path": "cfg.txt"}, "hidden", (lay.ws / "proj" / ".env", proj_env)),
        Step("search_through_symlink_dir", "search_files", {"pattern": auth, "path": "hhdir"}, "hidden", (lay.hh / "auth.json", auth)),
        Step("control_read_plain_symlink", "read_file", {"path": "notes-link.txt"}, "setup"),
        Step("control_write_plain", "write_file", {"path": "ok.txt", "content": "plain-ok"}, "setup"),
    ]


@dataclass
class Run:
    lay: Layout
    steps: dict[str, Step]
    results: dict[str, str]
    before: dict[str, str | None]


def _run(root: Path, build: Callable[[Layout], list[Step]], expected: tuple[str, ...],
         extra_env: Callable[[Layout], dict[str, str]]) -> Run:
    lay = Layout(root)
    lay.marks["env-sentinel"] = env_mark = H.canary("env-sentinel")
    steps = build(lay)
    assert [s.scenario for s in steps if s.expect != "setup"] == list(expected), "scenario table drifted"
    before: dict[str, str | None] = {}

    def snapshot() -> None:  # after the home (incl. .env) is written, before Hermes starts
        before.update({s.scenario: digest(s.subject) for s in steps if s.expect == "unchanged"})
        assert None not in before.values(), f"an 'unchanged' subject was not planted: {before}"

    results = run_tool_calls(lay.home, [(s.tool, s.args) for s in steps], cwd=lay.ws,
                             env_lines={"E2E_SENTINEL": env_mark}, extra_env=extra_env(lay), prepare=snapshot)
    return Run(lay, {s.scenario: s for s in steps}, dict(zip((s.scenario for s in steps), results)), before)


@pytest.fixture(scope="module")
def confined(tmp_path_factory: pytest.TempPathFactory) -> Run:
    return _run(tmp_path_factory.mktemp("confined"), _confined_steps, CONFINED_SCENARIOS + QUARANTINE_SCENARIOS,
                lambda lay: {"HERMES_WRITE_SAFE_ROOT": str(lay.ws)})


@pytest.fixture(scope="module")
def protected(tmp_path_factory: pytest.TempPathFactory) -> Run:
    return _run(tmp_path_factory.mktemp("protected"), _protected_steps, PROTECTED_SCENARIOS, lambda lay: {})


def _hidden_breach(step: Step, text: str) -> str | None:
    path, mark = step.subject
    # a vacuous pass is impossible: the protected file must still hold the canary the tool must not return
    assert mark in path.read_text(encoding="utf-8", errors="replace"), f"{path} no longer holds its canary"
    return "the tool result carries the protected file's content" if mark in text else None


def _check(run: Run, scenario: str) -> None:
    step, text = run.steps[scenario], run.results[scenario]
    breach = {
        "absent": lambda: f"{step.subject} was created" if step.subject.exists() else None,
        "unchanged": lambda: (f"{step.subject} was modified or removed"
                              if digest(step.subject) != run.before[scenario] else None),
        "hidden": lambda: _hidden_breach(step, text),
    }[step.expect]()
    if breach:
        raise H.BoundaryBreach(f"{scenario}: {step.tool} {step.args}: {breach}\nresult: {text[:400]}")
    refused = '"error"' in text or '"success": false' in text
    assert refused or step.tool == "search_files", \
        f"{scenario}: the boundary held on disk but the tool did not report a refusal: {text[:400]}"


@pytest.mark.parametrize("scenario", [H.known_param(n, KNOWN) for n in CONFINED_SCENARIOS])
def test_write_sandbox_holds_under_every_spelling(confined: Run, scenario: str) -> None:
    _check(confined, scenario)


@pytest.mark.parametrize("scenario", [H.known_param(n, KNOWN) for n in PROTECTED_SCENARIOS])
def test_protected_paths_hold_under_every_spelling(protected: Run, scenario: str) -> None:
    _check(protected, scenario)


@pytest.mark.parametrize("scenario", [H.known_param(n, KNOWN) for n in QUARANTINE_SCENARIOS])
def test_quarantined_auth_store_stays_read_denied(confined: Run, scenario: str) -> None:
    _check(confined, scenario)


def test_control_sandbox_allows_writes_that_resolve_inside(confined: Run) -> None:
    lay = confined.lay
    assert (lay.ws / "sub" / "inside.txt").read_text(encoding="utf-8") == "inside-ok", confined.results["control_write_inside"]
    # a symlink that stays inside the root is allowed: the guard judges the resolved target, not the spelling
    assert (lay.ws / "inner.txt").read_text(encoding="utf-8") == "inner-rewritten", confined.results["control_symlink_inside"]


def test_control_plain_files_stay_readable_and_writable(protected: Run) -> None:
    assert protected.lay.marks["plain-notes"] in protected.results["control_read_plain_symlink"], \
        "a non-credential file read through a symlink must return its content"
    assert (protected.lay.ws / "ok.txt").read_text(encoding="utf-8") == "plain-ok", protected.results["control_write_plain"]
