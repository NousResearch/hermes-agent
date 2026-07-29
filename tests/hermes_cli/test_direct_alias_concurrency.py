"""Concurrency regression: refreshing DIRECT_ALIASES must not break a concurrent reader.

Review finding (teknium1 on #67007): the in-place prune/update in
``_ensure_direct_aliases()`` is not synchronized against ``resolve_alias()``'s
reverse-lookup ``for alias_name, da in DIRECT_ALIASES.items()``.  Gateway
``/model`` work runs the switch under ``asyncio.to_thread``
(``gateway/slash_commands.py``), so a refresh on one worker thread can mutate
the dict while another thread is iterating it.

On the in-place-mutation implementation these tests fail with
``RuntimeError: dictionary changed size during iteration``.  They pass once the
refresh builds a NEW dict and swaps the module reference atomically.
"""
import threading

import pytest

from hermes_cli import model_switch as ms


@pytest.fixture(autouse=True)
def _restore_direct_aliases():
    saved = dict(ms.DIRECT_ALIASES)
    saved_degraded = ms._DIRECT_ALIASES_DEGRADED
    yield
    ms.DIRECT_ALIASES = dict(saved)
    ms._DIRECT_ALIASES_DEGRADED = saved_degraded


def _alias_set(n: int) -> dict:
    return {
        f"alias{i}": ms.DirectAlias(f"model-{i}", "custom", "")
        for i in range(n)
    }


def test_reverse_lookup_survives_concurrent_refresh(monkeypatch):
    """Reader iterating DIRECT_ALIASES while a refresh prunes/adds entries.

    RED on the in-place implementation:
        RuntimeError: dictionary changed size during iteration
    """
    small = _alias_set(400)
    large = _alias_set(900)
    flip = [0]

    def _alternating_loader():
        flip[0] += 1
        return (dict(large) if flip[0] % 2 else dict(small)), True

    monkeypatch.setattr(ms, "_load_direct_aliases", _alternating_loader)
    ms.DIRECT_ALIASES = dict(small)

    errors: list[BaseException] = []
    stop = threading.Event()

    def _reader():
        try:
            while not stop.is_set():
                # Miss on the direct hit -> exercises the reverse-lookup
                # `for alias_name, da in DIRECT_ALIASES.items()` scan.
                ms.resolve_alias("no-such-alias-zzz", "openrouter")
        except BaseException as exc:  # noqa: BLE001 - recording for assertion
            errors.append(exc)

    def _refresher():
        try:
            while not stop.is_set():
                ms._ensure_direct_aliases()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_reader) for _ in range(3)]
    threads += [threading.Thread(target=_refresher) for _ in range(3)]
    for t in threads:
        t.start()
    stop.wait(6.0)
    stop.set()
    for t in threads:
        t.join(timeout=10)

    assert not errors, (
        "concurrent refresh broke a reader iterating DIRECT_ALIASES: "
        f"{type(errors[0]).__name__}: {errors[0]}"
    )


def test_reader_never_sees_a_partial_alias_table(monkeypatch):
    """A reader must observe either the old table or the new one, never a torn mix.

    The in-place prune deletes removed keys BEFORE overlaying the fresh set, so
    a reader scheduled between the two steps sees a table that matches neither
    generation.  An atomic swap makes every observation a complete generation.
    """
    gen_a = {f"a{i}": ms.DirectAlias(f"m-a{i}", "custom", "") for i in range(300)}
    gen_b = {f"b{i}": ms.DirectAlias(f"m-b{i}", "custom", "") for i in range(300)}
    flip = [0]

    def _alternating_loader():
        flip[0] += 1
        return (dict(gen_b) if flip[0] % 2 else dict(gen_a)), True

    monkeypatch.setattr(ms, "_load_direct_aliases", _alternating_loader)
    ms.DIRECT_ALIASES = dict(gen_a)

    torn: list[str] = []
    errors: list[BaseException] = []
    stop = threading.Event()

    def _reader():
        try:
            while not stop.is_set():
                snap = ms.DIRECT_ALIASES
                keys = set(snap)          # single read of the module reference
                if keys and keys != set(gen_a) and keys != set(gen_b):
                    torn.append(
                        f"partial table: {len(keys)} keys "
                        f"(a={len(keys & set(gen_a))} b={len(keys & set(gen_b))})"
                    )
                    return
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def _refresher():
        try:
            while not stop.is_set():
                ms._ensure_direct_aliases()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_reader) for _ in range(2)]
    threads += [threading.Thread(target=_refresher) for _ in range(2)]
    for t in threads:
        t.start()
    stop.wait(5.0)
    stop.set()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"{type(errors[0]).__name__}: {errors[0]}"
    assert not torn, f"reader observed a torn alias table: {torn[0]}"
