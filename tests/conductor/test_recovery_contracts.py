from __future__ import annotations

import json
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from conductor.engine import Conductor, LaunchSpec, TickResult
from conductor.launcher import TmuxLauncher
from conductor.models import CampaignPlan, Step, StepKind
from conductor.receipts import (
    canonical_receipt_bytes,
    receipt_from_launch,
    write_receipt,
)
from conductor.store import ConductorStore
from conductor.worker_exec import _seatbelt_profile


class RunningLauncher:
    def __init__(self):
        self.specs = []
        self.running = set()

    def launch(self, spec):
        self.specs.append(spec)
        self.running.add(spec.tmux_session)
        return {"pid": 7, "start_marker": "marker"}

    def is_running(self, session):
        return session in self.running

    def cleanup(self, session):
        self.running.discard(session)


def _plan(repo: Path, *, max_turns: int = 2) -> CampaignPlan:
    return CampaignPlan(
        "recovery",
        str(repo),
        ["known.txt"],
        [Step("write", StepKind.IMPLEMENTATION, "write")],
        {"command": ["writer"], "provider": "p", "model": "w"},
        {"command": ["reviewer"], "provider": "p", "model": "r"},
        {"max_conductor_turns": max_turns},
    )


def test_tick_lease_has_one_owner_no_premature_steal_and_restart_recovery(tmp_path):
    path = tmp_path / "state.sqlite"
    first = ConductorStore(path)
    second = ConductorStore(path)
    first.create_campaign(_plan(tmp_path))

    assert first.acquire_tick("recovery", "owner-a", lease_seconds=10, now=100) is True
    assert (
        second.acquire_tick("recovery", "owner-b", lease_seconds=10, now=109.999)
        is False
    )
    assert second.acquire_tick("recovery", "owner-b", lease_seconds=10, now=110) is True
    first.release_tick("recovery", "owner-a")
    assert (
        second.acquire_tick("recovery", "owner-c", lease_seconds=10, now=111) is False
    )


def test_expired_tick_reclaim_is_atomic_under_concurrency(tmp_path):
    path = tmp_path / "state.sqlite"
    seed = ConductorStore(path)
    seed.create_campaign(_plan(tmp_path))
    assert seed.acquire_tick("recovery", "crashed", lease_seconds=5, now=10)
    stores = [ConductorStore(path), ConductorStore(path)]
    barrier = threading.Barrier(3)
    results = []

    def acquire(index):
        barrier.wait()
        results.append(
            stores[index].acquire_tick(
                "recovery", f"replacement-{index}", lease_seconds=5, now=15
            )
        )

    threads = [threading.Thread(target=acquire, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    assert sorted(results) == [False, True]


def test_running_worker_polls_do_not_spend_conductor_turns(tmp_path):
    launcher = RunningLauncher()
    store = ConductorStore(tmp_path / "state.sqlite")
    store.create_campaign(_plan(tmp_path, max_turns=1))
    engine = Conductor(store, launcher)
    assert engine.tick("recovery") is TickResult.LAUNCHED_WRITER
    for _ in range(250):
        assert engine.tick("recovery") is TickResult.WAITING_STALE
    assert store.get_campaign("recovery").conductor_turns == 1
    launcher.running.clear()
    assert engine.tick("recovery") is TickResult.BLOCKED_BUDGET


def test_receipt_canonical_utf8_contract_and_external_launch_adapter(tmp_path):
    launch = {
        "worker_id": "工人",
        "campaign_id": "café",
        "step_index": 1,
        "role": "reviewer",
        "cwd": "/tmp/é",
        "tmux_session": "会话",
        "provider": "提供者",
        "model": "模型",
        "prompt_hash": "abc",
        "mutable_manifest": ["résumé.txt"],
        "nonce": "ñ",
        "receipt_path": str(tmp_path / "receipt.json"),
    }
    receipt = receipt_from_launch(
        launch, status="COMPLETE", usage={"input_tokens": 1}, note="検査済み"
    )
    expected_body = {
        key: value for key, value in receipt.items() if key != "receipt_hash"
    }
    expected = json.dumps(
        expected_body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert canonical_receipt_bytes(expected_body) == expected
    assert b"\\u" not in canonical_receipt_bytes(receipt)
    write_receipt(Path(launch["receipt_path"]), receipt)
    assert Path(launch["receipt_path"]).read_bytes() == canonical_receipt_bytes(receipt)


def test_seatbelt_profile_denies_every_protected_root_and_allows_only_run_evidence(
    tmp_path,
):
    product = str((tmp_path / "product").resolve())
    git_dir = str((tmp_path / "git-common").resolve())
    evidence = (tmp_path / "evidence").resolve()
    profile = _seatbelt_profile([product, git_dir], evidence)
    assert f'(deny file-write* (subpath "{product}"))' in profile
    assert f'(deny file-write* (subpath "{git_dir}"))' in profile
    assert profile.endswith(f'(allow file-write* (subpath "{evidence}"))')


@pytest.mark.skipif(
    sys.platform != "darwin", reason="Seatbelt adversarial test is macOS-specific"
)
def test_reviewer_hard_boundary_denies_product_and_git_writes_but_allows_receipt(
    tmp_path,
):
    repo = tmp_path / "product"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seed"], check=True)
    run_dir = tmp_path / "evidence"
    run_dir.mkdir()
    attack = tmp_path / "reviewer.py"
    attack.write_text(
        """
import json, pathlib, subprocess, sys
from conductor.receipts import receipt_from_launch, write_receipt
spec = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
repo = pathlib.Path(spec['cwd'])
product_denied = False
try:
    (repo / 'tracked.txt').write_text('mutated\\n', encoding='utf-8')
except OSError:
    product_denied = True
git_result = subprocess.run(['git', '-C', str(repo), 'add', 'tracked.txt'])
pathlib.Path(spec['output_path']).write_text(json.dumps({
    'product_denied': product_denied, 'git_denied': git_result.returncode != 0
}), encoding='utf-8')
receipt = receipt_from_launch(spec, status='COMPLETE', usage={}, worker_turns=1)
write_receipt(pathlib.Path(spec['receipt_path']), receipt)
""",
        encoding="utf-8",
    )
    launch = {
        "worker_id": "review",
        "campaign_id": "campaign",
        "step_index": 0,
        "role": "reviewer",
        "command": [sys.executable, str(attack)],
        "cwd": str(repo),
        "tmux_session": "review-session",
        "provider": "fake",
        "model": "fake",
        "prompt_path": str(run_dir / "prompt.json"),
        "prompt_hash": "hash",
        "mutable_manifest": ["tracked.txt"],
        "output_path": str(run_dir / "output.json"),
        "receipt_path": str(run_dir / "receipt.json"),
        "heartbeat_path": str(run_dir / "heartbeat.json"),
        "nonce": "nonce",
        "read_only": True,
        "protected_roots": [str(repo), str(repo / ".git")],
    }
    launch_path = run_dir / "launch.json"
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "conductor.worker_exec", str(launch_path)],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    if (
        result.returncode == 71
        and "sandbox_apply: Operation not permitted" in result.stderr
    ):
        pytest.skip(
            "the enclosing test runner sandbox forbids nested Seatbelt profiles"
        )
    assert result.returncode == 0, result.stderr
    assert tracked.read_text(encoding="utf-8") == "original\n"
    assert (
        subprocess.run(
            ["git", "-C", str(repo), "diff", "--cached", "--quiet"], check=False
        ).returncode
        == 0
    )
    assert json.loads((run_dir / "output.json").read_text()) == {
        "product_denied": True,
        "git_denied": True,
    }
    assert (run_dir / "receipt.json").is_file()


def test_tmux_launch_argv_and_metadata_parsing_are_exact(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "display-message" in argv:
            return subprocess.CompletedProcess(
                argv, 0, stdout="4321 1700000000\n", stderr=""
            )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    launcher = TmuxLauncher(tmux="/tmux", socket_path=tmp_path / "sock")
    spec = LaunchSpec(
        command=("worker",),
        cwd=str(tmp_path.resolve()),
        receipt_path=str(tmp_path / "run" / "receipt.json"),
        tmux_session="exact-session",
        worker_id="w",
        campaign_id="c",
        step_index=0,
        role="writer",
        provider="p",
        model="m",
        prompt_path="prompt",
        prompt_hash="h",
        mutable_manifest=(),
        output_path="output",
        heartbeat_path="heartbeat",
        nonce="n",
        read_only=False,
        protected_roots=(),
    )
    (tmp_path / "run").mkdir()
    metadata = launcher.launch(spec)
    launch_path = tmp_path / "run" / "launch.json"
    log_path = tmp_path / "run" / "worker.log"
    fixed_command = (
        shlex.join([sys.executable, launcher._wrapper, str(launch_path)])
        + f" > {shlex.quote(str(log_path))} 2>&1"
    )
    assert calls[0][0] == [
        "/tmux",
        "-S",
        str((tmp_path / "sock").resolve()),
        "new-session",
        "-d",
        "-s",
        "exact-session",
        "-c",
        str(tmp_path.resolve()),
        fixed_command,
    ]
    assert calls[1][0] == [
        "/tmux",
        "-S",
        str((tmp_path / "sock").resolve()),
        "display-message",
        "-p",
        "-t",
        "exact-session",
        "#{pane_pid} #{session_created}",
    ]
    assert metadata == {"pid": 4321, "start_marker": "1700000000"}
