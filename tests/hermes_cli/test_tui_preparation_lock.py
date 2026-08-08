import threading
from pathlib import Path

from hermes_cli.tui_preparation_lock import tui_preparation_lock


def test_tui_preparation_lock_serializes_threads(tmp_path: Path) -> None:
    tui_dir = tmp_path / "ui-tui"
    tui_dir.mkdir()
    entered = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def first() -> None:
        with tui_preparation_lock(tui_dir):
            order.append("first-enter")
            entered.set()
            assert release.wait(timeout=5)
            order.append("first-exit")

    def second() -> None:
        assert entered.wait(timeout=5)
        with tui_preparation_lock(tui_dir):
            order.append("second-enter")

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert entered.wait(timeout=5)
    release.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert order == ["first-enter", "first-exit", "second-enter"]
