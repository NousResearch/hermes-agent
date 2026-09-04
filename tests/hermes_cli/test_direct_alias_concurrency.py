"""``DIRECT_ALIASES`` is refreshed in place while readers scan it.

``_ensure_direct_aliases`` republishes the cache with ``clear()`` + ``update()``
whenever the config identity moves, and ``resolve_alias`` iterates the same dict
(``for alias_name, da in DIRECT_ALIASES.items()``) on every alias miss. Nothing
serialized the two, so a concurrent refresh could:

* raise ``RuntimeError: dictionary changed size during iteration`` out of the
  reverse-model scan, and
* answer ``None`` for an alias that exists in *every* generation, because a
  reader can land in the window between ``clear()`` and ``update()`` — the
  silent half, which no exception ever reports and which a ``try/except``
  around the scan would leave fully intact.

Both are reachable from the gateway, which resolves models off the event loop
(``asyncio.to_thread``), so ``/model`` on two sessions runs the reader on two
threadpool workers while any config write reloads the cache underneath them.

Republishing by rebinding the module attribute would be atomic and need no
lock, but callers hold this exact dict (#16767) — so the refresh has to empty
and refill it, and the two steps have to be serialized against readers instead.
"""

import threading

import pytest

import hermes_cli.model_switch as ms
from hermes_cli.model_switch import DirectAlias


# A key that is in DIRECT_ALIASES for every generation the refresher publishes.
STABLE_ALIAS = "zz-stable-alias"
# A key in neither DIRECT_ALIASES nor MODEL_ALIASES: forces resolve_alias all
# the way through the reverse-model scan, then straight out. A HIT would return
# from the .get() fast path and never reach the iteration under test.
MISSING_ALIAS = "zz-deliberate-miss"

# The two generations differ in SIZE, not just contents: a same-size rewrite
# often skips the resize that makes the iteration raise.
SMALL, LARGE = 400, 900

# Enough of each to interleave; one-vs-one frequently does not.
READERS = REFRESHERS = 3
RUN_SECONDS = 3.0


def _generation(count: int) -> dict:
    aliases = {
        f"zz-alias-{i}": DirectAlias(f"zz-model-{i}", "custom", "")
        for i in range(count)
    }
    aliases[STABLE_ALIAS] = DirectAlias("zz-stable-model", "custom", "")
    return aliases


@pytest.fixture
def contended_alias_cache(monkeypatch):
    """Point the loader at two alternating generations, reloading every call.

    ``_direct_alias_source_identity`` returning ``None`` means "source unknown,
    do not reuse the cache" — exactly the refresh-on-every-call shape the race
    needs, without depending on a real config file's mtime granularity.
    """
    monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
    monkeypatch.setattr(ms, "_DIRECT_ALIAS_IDENTITY", None)
    monkeypatch.setattr(ms, "_DIRECT_ALIAS_LOADED", None)
    monkeypatch.setattr(ms, "_direct_alias_source_identity", lambda: None)

    generations = (_generation(SMALL), _generation(LARGE))
    counter = [0]

    def _alternating_load():
        counter[0] += 1
        return dict(generations[counter[0] % 2])

    monkeypatch.setattr(ms, "_load_direct_aliases", _alternating_load)
    ms._ensure_direct_aliases()
    return generations


def _run_contended(reader):
    """Drive *reader* on several threads against a refresher storm.

    Returns whatever the reader threads raised. Exceptions are collected and
    re-asserted on the main thread: one raised inside a ``Thread`` target is
    otherwise just printed, and the test passes green.
    """
    errors = []
    stop = threading.Event()

    def _guard(fn):
        def _loop():
            try:
                while not stop.is_set():
                    fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised from main
                errors.append(exc)
                stop.set()

        return _loop

    threads = [
        threading.Thread(target=_guard(reader), daemon=True)
        for _ in range(READERS)
    ]
    threads += [
        threading.Thread(target=_guard(ms._ensure_direct_aliases), daemon=True)
        for _ in range(REFRESHERS)
    ]
    for thread in threads:
        thread.start()
    # `stop` doubles as the early exit: the first failure ends the run.
    stop.wait(RUN_SECONDS)
    stop.set()
    for thread in threads:
        thread.join(timeout=10)
    return errors


def test_reader_survives_a_concurrent_refresh(contended_alias_cache):
    """The reverse-model scan must not raise while the cache is republished."""
    errors = _run_contended(lambda: ms.resolve_alias(MISSING_ALIAS, "custom"))

    assert not errors, f"{type(errors[0]).__name__}: {errors[0]}"


def test_reader_never_sees_a_half_published_cache(contended_alias_cache):
    """An alias present in every generation must never resolve to None.

    The silent half of the race: no exception, the caller is just told the
    alias does not exist and falls through to the default provider.
    """

    def _resolve_stable():
        if ms.resolve_alias(STABLE_ALIAS, "custom") is None:
            raise AssertionError(
                f"{STABLE_ALIAS!r} resolved to None during a refresh: the "
                "reader observed the cache between clear() and update()"
            )

    errors = _run_contended(_resolve_stable)

    assert not errors, f"{type(errors[0]).__name__}: {errors[0]}"


def test_snapshot_holds_one_whole_generation(contended_alias_cache):
    """Each snapshot is one published generation, never a mix of two."""
    small, large = contended_alias_cache
    valid = ({*small}, {*large})

    def _snapshot():
        keys = {*ms._direct_alias_snapshot()}
        if keys not in valid:
            raise AssertionError(
                f"snapshot held {len(keys)} keys, matching neither the "
                f"{len(small)}-entry nor the {len(large)}-entry generation"
            )

    errors = _run_contended(_snapshot)

    assert not errors, f"{type(errors[0]).__name__}: {errors[0]}"
