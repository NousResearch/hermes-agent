from __future__ import annotations

import hashlib
import inspect
import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts.canary import source_artifact_publication as publication


_DIRECT_RAW = b'{"artifact":"direct-iam","observed_at_unix":1800000000}'
_HOST_RAW = b'{"artifact":"host-identity","observed_at_unix":1800000001}\n'
_WORKER = r"""
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

from scripts.canary import source_artifact_publication as publication

owner_home = Path(sys.argv[1])
checkpoint = sys.argv[2]
counter = Path(sys.argv[3])
chain_name = sys.argv[4]
delay = float(sys.argv[5])
raw = b'{"artifact":"direct-iam","observed_at_unix":1800000000}'

def validate(value):
    if value != raw:
        raise RuntimeError("invalid artifact")
    return publication._ValidatedArtifact(
        value={"artifact": "direct-iam"},
        logical_sha256=hashlib.sha256(value).hexdigest(),
    )

def collect():
    descriptor = os.open(
        counter,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(descriptor, b"x")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if delay:
        time.sleep(delay)
    return raw

def stop(name):
    if name == checkpoint:
        os.kill(os.getpid(), signal.SIGKILL)

result = publication._run_direct_iam(
    owner_home=owner_home,
    chain={"foundation_source_revision": chain_name},
    maximum=4096,
    validator=validate,
    collector=collect,
    _checkpoint=stop,
)
sys.stdout.write(json.dumps({"replayed": result.replayed}, sort_keys=True))
"""


def _owner_home(tmp_path: Path) -> Path:
    owner_home = tmp_path / "owner-home"
    trusted = owner_home / ".hermes" / "trusted"
    trusted.mkdir(parents=True, mode=0o700)
    os.chmod(trusted, 0o700)
    os.chown(trusted, -1, os.getegid())
    return owner_home


def _chain(name: str = "a") -> dict[str, Any]:
    return {
        "foundation_source_revision": name * 40,
        "pre_foundation_authority_sha256": name * 64,
    }


def _validator(expected: bytes) -> Callable[[bytes], publication._ValidatedArtifact]:
    def validate(raw: bytes) -> publication._ValidatedArtifact:
        if raw != expected:
            raise publication._SourceArtifactPublicationError(
                "test_artifact_invalid"
            )
        return publication._ValidatedArtifact(
            value={"raw_sha256": hashlib.sha256(raw).hexdigest()},
            logical_sha256=hashlib.sha256(b"logical:" + raw).hexdigest(),
        )

    return validate


def _canonical(raw: bytes) -> bool:
    value = json.loads(raw.decode("ascii"))
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        == raw
    )


def _worker(
    owner_home: Path,
    *,
    checkpoint: str,
    counter: Path,
    chain_name: str = "a" * 40,
    delay: float = 0.0,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        (
            sys.executable,
            "-c",
            _WORKER,
            str(owner_home),
            checkpoint,
            str(counter),
            chain_name,
            str(delay),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )


def _direct_run(
    owner_home: Path,
    *,
    raw: bytes = _DIRECT_RAW,
    chain: dict[str, Any] | None = None,
    collector: Callable[[], bytes] | None = None,
    checkpoint: Callable[[str], None] | None = None,
    recovery_only: bool = False,
) -> publication._PublicationResult:
    return publication._run_direct_iam(
        owner_home=owner_home,
        chain=_chain() if chain is None else chain,
        maximum=4096,
        validator=_validator(raw),
        collector=(lambda: raw) if collector is None else collector,
        _checkpoint=checkpoint,
        _recovery_only=recovery_only,
    )


def _transaction_root(owner_home: Path, chain: dict[str, Any]) -> Path:
    txid = publication._transaction_id(publication._DIRECT_KIND, chain)
    return (
        owner_home
        / ".hermes/trusted/.source-artifact-transactions/direct-iam-v1"
        / txid
    )


def test_direct_publication_is_fixed_canonical_and_replayable(
    tmp_path: Path,
) -> None:
    owner_home = _owner_home(tmp_path)
    collections = 0

    def collect() -> bytes:
        nonlocal collections
        collections += 1
        return _DIRECT_RAW

    first = _direct_run(owner_home, collector=collect)
    replay = _direct_run(
        owner_home,
        collector=lambda: pytest.fail("replay recollected live evidence"),
    )

    final = owner_home / publication._DIRECT_RELATIVE
    root = _transaction_root(owner_home, _chain())
    assert collections == 1
    assert first.replayed is False
    assert replay.replayed is True
    assert first.raw == replay.raw == final.read_bytes() == _DIRECT_RAW
    assert first.path == str(final)
    assert stat.S_IMODE(final.stat().st_mode) == 0o400
    assert final.stat().st_nlink == 1
    assert not (root / "candidate.bin").exists()
    for name in ("intent.json", "success.json"):
        path = root / name
        raw = path.read_bytes()
        assert _canonical(raw)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.stat().st_nlink == 1


def test_host_publication_uses_only_fixed_destination_and_mode(
    tmp_path: Path,
) -> None:
    owner_home = _owner_home(tmp_path)
    result = publication._run_host_identity(
        owner_home=owner_home,
        chain=_chain(),
        maximum=4096,
        validator=_validator(_HOST_RAW),
        collector=lambda: _HOST_RAW,
    )

    final = owner_home / publication._HOST_RELATIVE
    assert result.path == str(final)
    assert final.read_bytes() == _HOST_RAW
    assert stat.S_IMODE(final.stat().st_mode) == 0o600
    assert final.stat().st_nlink == 1


def test_preexisting_final_without_matching_intent_fails_closed(
    tmp_path: Path,
) -> None:
    owner_home = _owner_home(tmp_path)
    final = owner_home / publication._DIRECT_RELATIVE
    final.write_bytes(_DIRECT_RAW)
    os.chmod(final, 0o400)

    with pytest.raises(
        publication._SourceArtifactPublicationError,
        match="source_artifact_publication_final_without_intent",
    ):
        _direct_run(owner_home)
    assert final.read_bytes() == _DIRECT_RAW


def test_oversized_or_unvalidated_collection_never_becomes_candidate(
    tmp_path: Path,
) -> None:
    oversized_home = _owner_home(tmp_path / "oversized")
    with pytest.raises(
        publication._SourceArtifactPublicationError,
        match="source_artifact_publication_collector_invalid",
    ):
        publication._run_direct_iam(
            owner_home=oversized_home,
            chain=_chain(),
            maximum=8,
            validator=_validator(_DIRECT_RAW),
            collector=lambda: _DIRECT_RAW,
        )
    oversized_root = _transaction_root(oversized_home, _chain())
    assert (oversized_root / "intent.json").exists()
    assert not (oversized_root / "candidate.bin").exists()
    assert not (oversized_home / publication._DIRECT_RELATIVE).exists()

    invalid_home = _owner_home(tmp_path / "invalid")
    with pytest.raises(
        publication._SourceArtifactPublicationError,
        match="source_artifact_publication_artifact_invalid",
    ):
        publication._run_direct_iam(
            owner_home=invalid_home,
            chain=_chain(),
            maximum=4096,
            validator=lambda _raw: object(),  # type: ignore[arg-type]
            collector=lambda: _DIRECT_RAW,
        )
    invalid_root = _transaction_root(invalid_home, _chain())
    assert (invalid_root / "intent.json").exists()
    assert not (invalid_root / "candidate.bin").exists()
    assert not (invalid_home / publication._DIRECT_RELATIVE).exists()


@pytest.mark.parametrize("attack", ["symlink_ancestor", "mutable_parent"])
def test_destination_ancestry_must_be_canonical_owner_only(
    tmp_path: Path,
    attack: str,
) -> None:
    owner_home = tmp_path / "owner-home"
    owner_home.mkdir(mode=0o700)
    if attack == "symlink_ancestor":
        real_hermes = tmp_path / "real-hermes"
        trusted = real_hermes / "trusted"
        trusted.mkdir(parents=True, mode=0o700)
        os.chmod(trusted, 0o700)
        os.chown(trusted, -1, os.getegid())
        (owner_home / ".hermes").symlink_to(real_hermes, target_is_directory=True)
    else:
        trusted = owner_home / ".hermes" / "trusted"
        trusted.mkdir(parents=True, mode=0o700)
        os.chmod(trusted, 0o755)

    with pytest.raises(
        publication._SourceArtifactPublicationError,
        match="source_artifact_publication_directory_invalid",
    ):
        _direct_run(owner_home)


def test_chain_conflict_cannot_recollect_or_replace_final(
    tmp_path: Path,
) -> None:
    owner_home = _owner_home(tmp_path)
    _direct_run(owner_home, chain=_chain("a"))

    with pytest.raises(
        publication._SourceArtifactPublicationError,
        match="source_artifact_publication_final_without_intent",
    ):
        _direct_run(
            owner_home,
            chain=_chain("b"),
            collector=lambda: pytest.fail("conflicting chain recollected"),
        )
    assert (owner_home / publication._DIRECT_RELATIVE).read_bytes() == _DIRECT_RAW


def test_tamper_and_unknown_inventory_fail_closed(tmp_path: Path) -> None:
    owner_home = _owner_home(tmp_path)
    _direct_run(owner_home)
    final = owner_home / publication._DIRECT_RELATIVE
    os.chmod(final, 0o600)
    final.write_bytes(b"X" * len(_DIRECT_RAW))
    os.chmod(final, 0o400)
    with pytest.raises(publication._SourceArtifactPublicationError):
        _direct_run(owner_home)

    os.chmod(final, 0o600)
    final.write_bytes(_DIRECT_RAW)
    os.chmod(final, 0o400)
    journal = final.parent / publication._JOURNAL_DIRECTORY
    unknown = journal / "unknown-purpose"
    unknown.mkdir(mode=0o700)
    with pytest.raises(
        publication._SourceArtifactPublicationError,
        match="source_artifact_publication_inventory_invalid",
    ):
        _direct_run(owner_home)


@pytest.mark.parametrize("target", ["intent", "success", "final_nlink"])
def test_transition_divergence_or_extra_link_fails_closed(
    tmp_path: Path,
    target: str,
) -> None:
    owner_home = _owner_home(tmp_path)
    _direct_run(owner_home)
    root = _transaction_root(owner_home, _chain())
    final = owner_home / publication._DIRECT_RELATIVE
    if target == "final_nlink":
        os.link(final, tmp_path / "unexpected-hardlink")
    else:
        path = root / f"{target}.json"
        value = json.loads(path.read_bytes())
        value["transaction_id"] = "f" * 64
        path.write_bytes(
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        )
        os.chmod(path, 0o600)

    with pytest.raises(publication._SourceArtifactPublicationError):
        _direct_run(owner_home)
    assert final.read_bytes() == _DIRECT_RAW


def test_unknown_entry_in_sibling_transaction_fails_closed(
    tmp_path: Path,
) -> None:
    owner_home = _owner_home(tmp_path)
    _direct_run(owner_home)
    final = owner_home / publication._DIRECT_RELATIVE
    kind_root = (
        final.parent
        / publication._JOURNAL_DIRECTORY
        / publication._DIRECT_KIND
    )
    sibling = kind_root / ("f" * 64)
    sibling.mkdir(mode=0o700)
    (sibling / "unknown").write_bytes(b"x")

    with pytest.raises(
        publication._SourceArtifactPublicationError,
        match="source_artifact_publication_inventory_invalid",
    ):
        _direct_run(owner_home)


@pytest.mark.parametrize(
    "checkpoint,expected_before,expected_after",
    [
        ("after_intent", 0, 1),
        ("after_candidate", 1, 1),
        ("after_final_link", 1, 1),
        ("after_success_scratch_open", 1, 1),
        ("after_success", 1, 1),
        ("after_candidate_scratch_open", 1, 2),
    ],
)
def test_sigkill_recovery_is_absent_or_complete_and_bounded(
    tmp_path: Path,
    checkpoint: str,
    expected_before: int,
    expected_after: int,
) -> None:
    owner_home = _owner_home(tmp_path)
    counter = tmp_path / "collections"
    killed = _worker(
        owner_home,
        checkpoint=checkpoint,
        counter=counter,
    )
    stdout, stderr = killed.communicate(timeout=20)

    assert killed.returncode == -signal.SIGKILL
    assert stdout == b""
    assert stderr == b""
    before = counter.read_bytes() if counter.exists() else b""
    assert len(before) == expected_before
    final = owner_home / publication._DIRECT_RELATIVE
    if final.exists():
        assert final.read_bytes() == _DIRECT_RAW
    resumed = _worker(
        owner_home,
        checkpoint="none",
        counter=counter,
    )
    resumed_stdout, resumed_stderr = resumed.communicate(timeout=20)
    assert resumed.returncode == 0, resumed_stderr.decode(errors="replace")
    assert json.loads(resumed_stdout) in (
        {"replayed": False},
        {"replayed": True},
    )
    assert len(counter.read_bytes()) == expected_after
    assert final.read_bytes() == _DIRECT_RAW
    assert stat.S_IMODE(final.stat().st_mode) == 0o400
    assert final.stat().st_nlink == 1


def test_recovery_only_refuses_empty_or_intent_only_transaction(
    tmp_path: Path,
) -> None:
    empty_home = _owner_home(tmp_path / "empty")
    with pytest.raises(
        publication._SourceArtifactPublicationError,
        match="source_artifact_publication_recovery_unavailable",
    ):
        _direct_run(
            empty_home,
            collector=lambda: pytest.fail("empty recovery collected"),
            recovery_only=True,
        )

    intent_home = _owner_home(tmp_path / "intent")

    class StopAfterIntent(BaseException):
        pass

    def stop(name: str) -> None:
        if name == "after_intent":
            raise StopAfterIntent

    with pytest.raises(StopAfterIntent):
        _direct_run(intent_home, checkpoint=stop)
    with pytest.raises(
        publication._SourceArtifactPublicationError,
        match="source_artifact_publication_recovery_unavailable",
    ):
        _direct_run(
            intent_home,
            collector=lambda: pytest.fail("intent-only recovery collected"),
            recovery_only=True,
        )


def test_concurrent_callers_collect_once_and_reach_one_terminal_state(
    tmp_path: Path,
) -> None:
    owner_home = _owner_home(tmp_path)
    counter = tmp_path / "collections"
    first = _worker(
        owner_home,
        checkpoint="none",
        counter=counter,
        delay=0.25,
    )
    second = _worker(
        owner_home,
        checkpoint="none",
        counter=counter,
        delay=0.25,
    )
    first_stdout, first_stderr = first.communicate(timeout=20)
    second_stdout, second_stderr = second.communicate(timeout=20)

    assert first.returncode == second.returncode == 0
    assert first_stderr == second_stderr == b""
    assert len(counter.read_bytes()) == 1
    results = [json.loads(first_stdout), json.loads(second_stdout)]
    assert sorted(item["replayed"] for item in results) == [False, True]
    final = owner_home / publication._DIRECT_RELATIVE
    assert final.read_bytes() == _DIRECT_RAW
    kind_root = final.parent / publication._JOURNAL_DIRECTORY / publication._DIRECT_KIND
    terminals = list(kind_root.glob("*/success.json"))
    assert len(terminals) == 1


def test_private_factories_expose_no_destination_or_journal_selection() -> None:
    direct = set(inspect.signature(publication._run_direct_iam).parameters)
    host = set(inspect.signature(publication._run_host_identity).parameters)
    forbidden = {
        "output",
        "output_path",
        "candidate",
        "candidate_path",
        "journal",
        "journal_path",
    }
    assert forbidden.isdisjoint(direct)
    assert forbidden.isdisjoint(host)
    assert publication.__all__ == []
