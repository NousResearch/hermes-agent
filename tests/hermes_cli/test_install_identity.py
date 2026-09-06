from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pathlib import Path
import time

from gateway.hosted_rooms import local_authority_gateway_id
import hermes_cli.install_identity as install_identity
from hermes_cli.install_identity import read_or_create_install_id


def _race_first_install_id(
    root_value,
    minted,
    results,
    start_barrier=None,
    writer_entered=None,
    release_writer=None,
):
    root = Path(root_value)
    install_identity.uuid.uuid4 = lambda: type("FixedUuid", (), {"hex": minted})()
    if start_barrier is not None:
        start_barrier.wait(timeout=10)
    if writer_entered is not None:
        original_mkstemp = install_identity.tempfile.mkstemp

        def held_mkstemp(*args, **kwargs):
            writer_entered.set()
            assert release_writer.wait(timeout=10)
            return original_mkstemp(*args, **kwargs)

        install_identity.tempfile.mkstemp = held_mkstemp
    results.put(read_or_create_install_id(root))


def test_concurrent_first_use_returns_one_persisted_identity(tmp_path):
    with ThreadPoolExecutor(max_workers=16) as executor:
        values = list(executor.map(lambda _: read_or_create_install_id(tmp_path), range(64)))

    assert len(set(values)) == 1
    assert values[0]
    assert (tmp_path / "install_id").read_text(encoding="utf-8").strip() == values[0]


def test_independent_first_callers_return_the_single_committed_identity(tmp_path, monkeypatch):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    writer_entered = context.Event()
    release_writer = context.Event()
    winner = context.Process(
        target=_race_first_install_id,
        args=(
            str(tmp_path),
            "a" * 32,
            results,
            None,
            writer_entered,
            release_writer,
        ),
    )
    loser = context.Process(
        target=_race_first_install_id,
        args=(str(tmp_path), "b" * 32, results),
    )

    winner.start()
    assert writer_entered.wait(timeout=10)
    loser.start()
    time.sleep(0.25)
    assert loser.is_alive()
    release_writer.set()
    processes = [winner, loser]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    returned = [results.get(timeout=2) for _ in processes]
    persisted = (tmp_path / "install_id").read_text(encoding="utf-8").strip()

    assert returned == [persisted, persisted]

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        install_identity,
        "_INSTALL_ID_CACHE",
        {"root": None, "value": None},
    )
    assert local_authority_gateway_id() == f"install:{persisted}"


def test_concurrent_corrupt_file_repair_returns_one_committed_identity(tmp_path):
    (tmp_path / "install_id").write_text("corrupt\n", encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_race_first_install_id,
            args=(str(tmp_path), value, results, barrier),
        )
        for value in ("a" * 32, "b" * 32)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    returned = [results.get(timeout=2) for _ in processes]
    persisted = (tmp_path / "install_id").read_text(encoding="utf-8").strip()

    assert returned == [persisted, persisted]
