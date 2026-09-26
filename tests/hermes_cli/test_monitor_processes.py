"""Tests for machine-wide Hermes process discovery and sampling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.monitor_processes import (
    MonitorSampler,
    ProcessKey,
    _command_tokens,
    _default_ledger_reader,
    _default_start_fingerprint_resolver,
    _is_hermes_candidate,
)


class FakeProcess:
    def __init__(
        self, pid, *, name="python", cmdline=None, create=100.0, cpu=1.0,
        rss=1024, status="sleeping", username="me", tty=None, environ=None,
        ppid=1, account_identity="account-me",
    ):
        self.pid = pid
        self.info = {
            "pid": pid, "name": name, "cmdline": list(cmdline or []),
            "create_time": create, "status": status, "username": username, "ppid": ppid,
        }
        self._cpu = cpu
        self._rss = rss
        self._tty = tty
        self._environ = dict(environ or {})
        self.account_identity = account_identity

    def cpu_times(self):
        return SimpleNamespace(user=self._cpu, system=0.0)

    def memory_info(self):
        return SimpleNamespace(rss=self._rss)

    def terminal(self):
        return self._tty

    def status(self):
        return self.info["status"]

    def environ(self):
        return dict(self._environ)

    def create_time(self):
        return self.info["create_time"]


def _sampler(
    processes, *, now=200.0, homes=None, delegations=None, fingerprints=None, ledgers=None,
    delegation_scanner=None,
):
    homes = homes or {}
    fingerprints = fingerprints or {
        process.pid: int(process.info["create_time"] * 100) for process in processes
    }
    return MonitorSampler(
        process_iter=lambda: list(processes),
        ledger_reader=lambda: list(ledgers or []),
        home_resolver=lambda pid: homes.get(pid),
        delegation_scanner=delegation_scanner or (lambda _homes: list(delegations or [])),
        start_fingerprint_resolver=lambda proc, _key: fingerprints.get(proc.pid),
        process_key_resolver=lambda pid: ProcessKey(
            pid, next(process for process in processes if process.pid == pid).info["create_time"],
        ),
        account_identity_resolver=lambda proc: proc.account_identity,
        current_account_identity="account-me",
        wall_clock=lambda: now,
        monotonic_clock=lambda: now,
    )


def test_discovers_old_hermes_processes_and_rejects_same_username_from_another_account():
    processes = [
        FakeProcess(10, name="python3", cmdline=["python3", "/nix/store/x/bin/hermes", "--resume", "20260926_103218_abcdef"],
                    tty="/dev/pts/3", environ={"TMUX_PANE": "%7"}),
        FakeProcess(11, name="python3", cmdline=["python3", "tools/mcp_death_supervisor.py"]),
        FakeProcess(12, name="hermes", cmdline=["hermes"]),
        FakeProcess(13, name="hermes", cmdline=["hermes"], username="me", account_identity="other-domain-me"),
        FakeProcess(14, name="hermes", cmdline=["hermes"], username="DOMAIN\\alias"),
    ]
    snapshot = _sampler(
        processes,
        homes={10: "/home/me/.hermes", 12: "/home/me/.hermes", 14: "/home/me/.hermes"},
    ).sample()

    assert [row.pid for row in snapshot.processes] == [10, 12, 14]
    assert snapshot.processes[0].kind == "interactive"
    assert snapshot.processes[0].activity == "session 20260926_103218_abcdef"
    assert snapshot.processes[0].tmux == "%7"
    assert snapshot.processes[1].kind == "hermes"


@pytest.mark.parametrize(("argv", "command"), [
    (["hermes"], []),
    (["hermes.exe", "chat"], ["chat"]),
    (["hermes-agent", "hello"], ["hello"]),
    (["hermes-acp"], []),
    (["hermes-gateway"], []),
    ([r"C:\\Hermes\\hermes.exe", "chat"], ["chat"]),
    (["python3", "/nix/store/abc/bin/hermes", "chat"], ["chat"]),
    (["python", r"C:\\Hermes\\hermes-acp"], []),
    (["python", "/app/hermes_cli/main.py", "monitor"], ["monitor"]),
    (["python", "-m", "hermes_cli.main", "gateway", "run"], ["gateway", "run"]),
    (["python", "-m", "agent.legacy_cli", "hello"], ["hello"]),
    (["python", "-m", "acp_adapter.entry"], []),
    (["python", "/opt/hermes/desktop-gateway.py"], []),
])
def test_canonical_roots_recognize_only_supported_launcher_forms(argv, command):
    assert _is_hermes_candidate("python", argv)
    assert _command_tokens(argv) == command


@pytest.mark.parametrize("argv", [
    ["python", "script.py", "hermes"],
    ["python", "script.py", "please run hermes"],
    ["python", "worker.py", "hermes_cli.main"],
    ["python", "-m", "unrelated", "hermes"],
    ["python", "script.py", "desktop-gateway.py"],
])
def test_prompt_and_noncanonical_argument_occurrences_are_not_roots(argv):
    assert not _is_hermes_candidate("python", argv)
    assert _command_tokens(argv) == []


def test_process_with_unknown_native_account_identity_is_omitted():
    process = FakeProcess(15, name="hermes", cmdline=["hermes"], account_identity=None)

    assert _sampler([process]).sample().processes == []


def test_descendant_helper_is_included_but_unrelated_ledger_process_is_not():
    root = FakeProcess(16, name="hermes", cmdline=["hermes", "chat"])
    helper = FakeProcess(17, name="sleep", cmdline=["sleep", "30"], ppid=19)
    unrelated = FakeProcess(18, name="python", cmdline=["python", "worker.py"], ppid=1)
    launcher = FakeProcess(19, name="python", cmdline=["python", "launcher.py"], ppid=16)
    ledgers = [{
        "pid": 18,
        "create_time": 100.0,
        "purpose": "mcp-helper",
        "hermes_home": "/fabricated/home",
    }]

    rows = _sampler([unrelated, helper, launcher, root], ledgers=ledgers).sample().processes

    assert [(row.pid, row.kind) for row in rows] == [
        (16, "interactive"), (17, "helper"), (19, "helper"),
    ]


def test_reused_parent_pid_does_not_claim_an_older_unrelated_process():
    root = FakeProcess(160, name="hermes", cmdline=["hermes"], create=200.0)
    older = FakeProcess(161, name="sleep", cmdline=["sleep", "30"], ppid=160, create=100.0)

    rows = _sampler([older, root], now=300.0).sample().processes

    assert [row.pid for row in rows] == [160]


def test_classification_prefers_kanban_then_gateway_and_monitor():
    processes = [
        FakeProcess(20, name="hermes", cmdline=["hermes", "chat"],
                    environ={"HERMES_KANBAN_TASK": "t_deadbeef"}),
        FakeProcess(21, name="hermes", cmdline=["hermes", "gateway", "run"]),
        FakeProcess(22, name="hermes", cmdline=["hermes", "monitor"]),
        FakeProcess(
            23, name="hermes",
            cmdline=["hermes", "chat", "-q", "please monitor gateway run serve"],
        ),
    ]
    rows = _sampler(processes).sample().processes

    assert [(row.pid, row.kind, row.activity) for row in rows] == [
        (20, "kanban", "task t_deadbeef"),
        (21, "gateway", "gateway run"),
        (22, "monitor", "fleet dashboard"),
        (23, "interactive", "interactive"),
    ]


def test_ledger_metadata_requires_same_pid_incarnation():
    proc = FakeProcess(24, name="python", cmdline=["python"], create=200.0)
    sampler = MonitorSampler(
        process_iter=lambda: [proc],
        ledger_reader=lambda: [{
            "pid": 24, "create_time": 100.0, "purpose": "gateway",
            "hermes_home": "/home/me/.hermes",
        }],
        home_resolver=lambda _pid: None,
        delegation_scanner=lambda _homes: [],
        account_identity_resolver=lambda proc: proc.account_identity,
        current_account_identity="account-me",
        wall_clock=lambda: 300.0,
        monotonic_clock=lambda: 300.0,
    )

    assert sampler.sample().processes == []


def test_ledger_metadata_requires_exact_create_time_equality():
    proc = FakeProcess(241, name="hermes", cmdline=["hermes"], create=100.0)
    row = _sampler([proc], ledgers=[{
        "pid": 241, "create_time": 100.005, "purpose": "gateway",
        "hermes_home": "/fabricated/home",
    }]).sample().processes[0]

    assert row.kind == "hermes"
    assert row.confidence == "discovered"


def test_matching_ledger_home_overrides_mutable_active_profile_resolution():
    proc = FakeProcess(25, name="hermes", cmdline=["hermes"], create=100.0)
    scanned_homes = []
    sampler = MonitorSampler(
        process_iter=lambda: [proc],
        ledger_reader=lambda: [{
            "pid": 25,
            "create_time": 100.0,
            "purpose": "interactive",
            "hermes_home": "/home/me/.hermes/profiles/actual",
        }],
        home_resolver=lambda _pid: "/home/me/.hermes/profiles/current-sticky",
        delegation_scanner=lambda homes: scanned_homes.extend(homes) or [],
        start_fingerprint_resolver=lambda _proc, _key: 10000,
        process_key_resolver=lambda pid: ProcessKey(pid, proc.info["create_time"]),
        account_identity_resolver=lambda process: process.account_identity,
        current_account_identity="account-me",
        wall_clock=lambda: 200.0,
        monotonic_clock=lambda: 200.0,
    )

    snapshot = sampler.sample()

    assert snapshot.processes[0].profile == "actual"
    assert scanned_homes == ["/home/me/.hermes/profiles/actual"]


def test_profile_resolution_and_cpu_delta_are_pid_reuse_safe():
    proc = FakeProcess(30, name="hermes", cmdline=["hermes"], create=50.0, cpu=1.0, rss=2048)
    sampler = _sampler([proc], now=100.0, homes={30: "/home/me/.hermes/profiles/coder"})
    first = sampler.sample()
    assert first.processes[0].profile == "coder"
    assert first.processes[0].cpu_percent == 0.0

    proc._cpu = 2.0
    sampler._wall_clock = lambda: 102.0
    sampler._monotonic_clock = lambda: 102.0
    second = sampler.sample()
    assert second.processes[0].cpu_percent == 50.0

    proc.info["create_time"] = 70.0
    proc._cpu = 20.0
    sampler._wall_clock = lambda: 104.0
    sampler._monotonic_clock = lambda: 104.0
    third = sampler.sample()
    assert third.processes[0].cpu_percent == 0.0
    assert ProcessKey(30, 50.0) not in sampler._cpu_samples


@pytest.mark.parametrize("replacement_point", ["environment", "home", "resource"])
def test_pid_replacement_after_sensitive_reads_omits_row(replacement_point):
    process = FakeProcess(31, name="hermes", cmdline=["hermes"], create=50.0)
    current_create_time = 50.0

    def replace_after(call):
        def wrapped(*args, **kwargs):
            nonlocal current_create_time
            result = call(*args, **kwargs)
            current_create_time = 51.0
            return result
        return wrapped

    if replacement_point == "environment":
        process.environ = replace_after(process.environ)
    elif replacement_point == "resource":
        process.memory_info = replace_after(process.memory_info)

    def resolve_home(_pid):
        nonlocal current_create_time
        current_create_time = 51.0 if replacement_point == "home" else current_create_time
        return "/home/me/.hermes"

    sampler = MonitorSampler(
        process_iter=lambda: [process], ledger_reader=lambda: [],
        home_resolver=resolve_home, delegation_scanner=lambda _homes: [],
        start_fingerprint_resolver=lambda _proc, _key: 5000,
        process_key_resolver=lambda pid: ProcessKey(pid, current_create_time),
        account_identity_resolver=lambda proc: proc.account_identity,
        current_account_identity="account-me",
        wall_clock=lambda: 100.0, monotonic_clock=lambda: 100.0,
    )

    assert sampler.sample().processes == []


def test_delegations_are_included_without_double_counting_processes():
    process = FakeProcess(40, name="hermes", cmdline=["hermes"])
    children = [{
        "owner_started_at": 424242,
        "owner_pid": 40, "delegation_id": "deleg_deadbeef", "task_index": 0,
        "status": "running", "goal": "inspect the worker", "last_tool": "read_file",
        "updated_at": 199.0,
    }]
    snapshot = _sampler(
        [process], delegations=children, fingerprints={40: 424242},
    ).sample()

    assert snapshot.total_processes == 1
    assert snapshot.total_delegations == 1
    assert snapshot.delegations[0]["owner_pid"] == 40
    assert "goal" not in snapshot.delegations[0]
    assert set(snapshot.delegations[0]).isdisjoint({"model", "provider", "last_tool"})


def test_delegation_scan_uses_every_discovered_home_and_labels_only_from_owner_row():
    first = FakeProcess(44, name="hermes", cmdline=["hermes"])
    second = FakeProcess(45, name="hermes", cmdline=["hermes"])
    unrelated = FakeProcess(46, name="python", cmdline=["python", "worker.py"])
    homes = {
        44: "/install-a/.hermes/profiles/alpha",
        45: "/install-b/hermes/profiles/beta",
        46: "/unknown/root",
    }
    scanned_homes = []

    def scan(discovered_homes):
        scanned_homes.extend(discovered_homes)
        return [
            {
                "owner_pid": 44, "owner_started_at": 4400,
                "delegation_id": "deleg_deadbeef", "task_index": 0,
                "status": "running", "updated_at": 199.0,
                "profile": "forged", "home": "/must/not/escape",
            },
            {
                "owner_pid": 45, "owner_started_at": 4500,
                "delegation_id": "deleg_cafebabe", "task_index": 1,
                "status": "queued", "updated_at": 198.0,
                "profile": "forged", "path": "/must/not/escape",
            },
        ]

    snapshot = _sampler(
        [first, second, unrelated], homes=homes, delegation_scanner=scan,
        fingerprints={44: 4400, 45: 4500},
    ).sample()

    assert set(scanned_homes) == {homes[44], homes[45]}
    assert [row["profile"] for row in snapshot.delegations] == ["alpha", "beta"]
    assert all("home" not in row and "path" not in row for row in snapshot.delegations)


def test_untrusted_process_and_delegation_values_never_reach_snapshot():
    secret = "PROMPT/path/API_KEY=secret"
    credential = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    unsafe_profile = "PROMPT_SECRET"
    process = FakeProcess(
        41,
        name="hermes",
        cmdline=["hermes", "--resume", secret],
        create=100.0,
        status=secret,
        tty=f"/dev/{secret}",
        environ={"HERMES_KANBAN_TASK": secret, "TMUX_PANE": secret},
    )
    delegations = [
        {
            "owner_pid": 41,
            "owner_started_at": 10000,
            "delegation_id": "deleg_deadbeef",
            "subagent_id": "sa-0-cafebabe",
            "task_index": 0,
            "status": "running",
            "model": credential,
            "provider": credential,
            "last_tool": credential,
            "updated_at": 199.0,
            "profile": secret,
        },
        {
            "owner_pid": 41,
            "owner_started_at": 10000,
            "delegation_id": secret,
            "task_index": 1,
            "status": "running",
            "model": secret,
            "updated_at": secret,
        },
    ]

    snapshot = _sampler(
        [process],
        homes={41: f"/home/me/.hermes/profiles/{unsafe_profile}"},
        delegations=delegations,
    ).sample()

    encoded = json.dumps({
        "processes": [row.to_dict() for row in snapshot.processes],
        "delegations": snapshot.delegations,
    })
    assert secret not in encoded
    assert credential not in encoded
    assert unsafe_profile not in encoded
    assert snapshot.processes[0].activity == "interactive"
    assert snapshot.processes[0].profile == "?"
    assert snapshot.processes[0].status == "?"
    assert snapshot.processes[0].tty == "—"
    assert snapshot.processes[0].tmux == "—"
    assert snapshot.delegations == [{
        "owner_pid": 41,
        "subagent_id": "sa-0-cafebabe",
        "delegation_id": "deleg_deadbeef",
        "task_index": 0,
        "status": "running",
        "profile": "?",
        "updated_at": 199.0,
        "confidence": None,
    }]


def test_delegations_require_discovered_same_user_owner_incarnation():
    process = FakeProcess(42, name="hermes", cmdline=["hermes"], create=100.0)
    common = {
        "delegation_id": "deleg_deadbeef",
        "task_index": 0,
        "status": "running",
        "updated_at": 199.0,
    }
    scanner_rows = [
        {**common, "owner_pid": 42, "owner_started_at": 9000},
        {**common, "owner_pid": 999, "owner_started_at": 10000},
    ]

    assert _sampler([process], delegations=scanner_rows).sample().delegations == []


@pytest.mark.platforms("linux")
def test_default_owner_fingerprint_uses_linux_proc_start_ticks():
    process = FakeProcess(os.getpid(), create=123.0)
    expected = int(Path(f"/proc/{os.getpid()}/stat").read_text(encoding="utf-8").split()[21])

    assert _default_start_fingerprint_resolver(
        process, ProcessKey(process.pid, process.create_time()),
    ) == expected
    assert expected != int(process.create_time() * 100)


def test_delegation_owner_fingerprint_is_bound_to_exact_process_incarnation():
    process = FakeProcess(43, name="hermes", cmdline=["hermes"], create=100.0)
    child = {
        "owner_pid": 43,
        "owner_started_at": 5000,
        "delegation_id": "deleg_deadbeef",
        "task_index": 0,
        "status": "running",
        "updated_at": 199.0,
    }
    seen = []

    def reused_while_resolving(proc, key):
        seen.append((proc, key))
        proc.info["create_time"] = 101.0
        return 5000

    raced = MonitorSampler(
        process_iter=lambda: [process], ledger_reader=lambda: [],
        home_resolver=lambda _pid: None, delegation_scanner=lambda _homes: [child],
        start_fingerprint_resolver=reused_while_resolving,
        process_key_resolver=lambda pid: ProcessKey(pid, process.info["create_time"]),
        account_identity_resolver=lambda proc: proc.account_identity,
        current_account_identity="account-me",
        wall_clock=lambda: 200.0, monotonic_clock=lambda: 200.0,
    ).sample()

    assert seen == [(process, ProcessKey(43, 100.0))]
    assert raced.delegations == []

    process.info["create_time"] = 100.0
    near_match = {**child, "owner_started_at": 5001}
    exact_only = MonitorSampler(
        process_iter=lambda: [process], ledger_reader=lambda: [],
        home_resolver=lambda _pid: None, delegation_scanner=lambda _homes: [near_match],
        start_fingerprint_resolver=lambda _proc, _key: 5000,
        process_key_resolver=lambda pid: ProcessKey(pid, process.info["create_time"]),
        account_identity_resolver=lambda proc: proc.account_identity,
        current_account_identity="account-me",
        wall_clock=lambda: 200.0, monotonic_clock=lambda: 200.0,
    ).sample()
    assert exact_only.delegations == []


def test_default_ledger_reader_never_quarantines_corrupt_file(monkeypatch, tmp_path):
    ledger = tmp_path / "spawn-ledger.json"
    ledger.write_text("{corrupt", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.process_identity._ledger_path", lambda: ledger)

    assert _default_ledger_reader() == []
    assert ledger.read_text(encoding="utf-8-sig") == "{corrupt"
    assert not Path(f"{ledger}.corrupt").exists()


def test_disappearing_or_denied_process_is_skipped():
    class Broken(FakeProcess):
        def cpu_times(self):
            raise RuntimeError("gone")

    snapshot = _sampler([Broken(50, name="hermes", cmdline=["hermes"])]).sample()
    assert snapshot.processes == []
