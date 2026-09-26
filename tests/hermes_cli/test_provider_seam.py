"""Behavior contract for the provider-registry generation seam.

``hermes_cli.provider_seam`` binds the provider-registry containers to copy-on-write facades at
their definition sites and swaps multi-container publications in one generation reference. These
tests pin: every container is its facade (t1); copy-on-write (t2); whole-or-absent publication (t3,
t4); the base-class bypass sees the same content (t5); C-level consumers see the data (t6); no lost
update under concurrent writers (t7); publication idempotency / collision; refresh callbacks and
the call sites that trigger them; readers never block on a writer; both boot orders; and the two
user-visible regressions the seam fixes (torn iteration during registration, and a provider
registered after import missing from ``provider:model`` parsing).

No test reads source files; every assertion exercises the real modules.
"""

from __future__ import annotations

import asyncio
import collections
import copy
import json
import os
import pickle
import platform
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import providers
import hermes_cli.auth  # noqa: F401  (registers PROVIDER_REGISTRY)
import hermes_cli.models as models
import hermes_cli.models_catalog_static as mcs
import hermes_cli.providers as cli_providers
from hermes_cli import provider_seam
from hermes_cli.auth import ProviderConfig
from hermes_cli.provider_seam import GuardedDict, GuardedList, GuardedSet
from providers import ProviderProfile

REPO_ROOT = Path(__file__).resolve().parents[2]
_OVERLAYS = next(n for n, o in provider_seam.FACADES.items() if provider_seam.owner_of(n) == "hermes_cli.providers")
_OVERLAY_TYPE = type(next(iter(getattr(cli_providers, _OVERLAYS).values())))

# The nine containers the seam must own, keyed to their defining module.
EXPECTED = {
    "_REGISTRY": "providers",
    "_ALIASES": "providers",
    "PROVIDER_REGISTRY": "hermes_cli.auth",
    "_PROVIDER_MODELS": "hermes_cli.models_catalog_static",
    "CANONICAL_PROVIDERS": "hermes_cli.models_catalog_static",
    "_PROVIDER_LABELS": "hermes_cli.models_catalog_static",
    "_PROVIDER_ALIASES": "hermes_cli.models_catalog_static",
    "_KNOWN_PROVIDER_NAMES": "hermes_cli.models",
    _OVERLAYS: "hermes_cli.providers",
}


@pytest.fixture(autouse=True)
def _seam_isolation():
    generation = provider_seam.current()
    facades = dict(provider_seam.FACADES)
    callbacks = list(provider_seam._refresh_callbacks)
    yield
    provider_seam._park = None
    provider_seam._refresh_callbacks[:] = callbacks
    for name in set(provider_seam.FACADES) - set(facades):
        provider_seam.FACADES.pop(name)
    provider_seam.restore(generation)


def _delta(name: str) -> dict:
    """One pin's full seven-surface publication + parser membership."""
    return {
        "_REGISTRY": {name: ProviderProfile(name=name, aliases=(f"{name}-alias",))},
        "_ALIASES": {f"{name}-alias": name},
        "PROVIDER_REGISTRY": {name: ProviderConfig(id=name, name=name, auth_type="api_key")},
        "_PROVIDER_MODELS": {name: ["model-a"]},
        "CANONICAL_PROVIDERS": [mcs.ProviderEntry(name, name, f"{name} (direct API)")],
        "_PROVIDER_LABELS": {name: name},
        "_PROVIDER_ALIASES": {f"{name}-alias": name},
        "_KNOWN_PROVIDER_NAMES": {name, f"{name}-alias"},
        _OVERLAYS: {name: _OVERLAY_TYPE(transport="openai_chat")},
        "committed": {"test-lane": {name}},
    }


def _surfaces(g, name: str) -> dict:
    return {
        "_REGISTRY": name in g._REGISTRY,
        "_ALIASES": f"{name}-alias" in g._ALIASES,
        "PROVIDER_REGISTRY": name in g.PROVIDER_REGISTRY,
        "_PROVIDER_MODELS": name in g._PROVIDER_MODELS,
        "CANONICAL_PROVIDERS": any(e.slug == name for e in g.CANONICAL_PROVIDERS),
        "_PROVIDER_LABELS": name in g._PROVIDER_LABELS,
        "_PROVIDER_ALIASES": f"{name}-alias" in g._PROVIDER_ALIASES,
        "_KNOWN_PROVIDER_NAMES": name in g._KNOWN_PROVIDER_NAMES,
        _OVERLAYS: name in g[_OVERLAYS],
        "committed": name in g.committed.get("test-lane", ()),
    }


class _CountingLock:
    """Stand-in for the seam RLock that counts acquisitions per thread."""

    def __init__(self):
        self._lock = threading.RLock()
        self.acquired = collections.Counter()
        self.watch = None
        self.watched_attempt = threading.Event()

    def __enter__(self):
        ident = threading.get_ident()
        self.acquired[ident] += 1
        if ident == self.watch:
            self.watched_attempt.set()
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()


def _park_thread(ident_holder: dict, stage: str, parked: threading.Event, release: threading.Event):
    def park(at: str) -> None:
        if at == stage and threading.get_ident() == ident_holder.get("ident"):
            parked.set()
            assert release.wait(30), "test never released the parked publisher"

    return park


def _run_child(code: str) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


# ---------------------------------------------------------------------------
# t1 — every container is its module's facade
# ---------------------------------------------------------------------------

def test_t1_every_container_is_bound_to_its_facade():
    missing = sorted(set(EXPECTED) - set(provider_seam.FACADES))
    assert not missing, f"containers not bound to a seam facade: {missing}"
    for name, owner in EXPECTED.items():
        assert provider_seam.owner_of(name) == owner, name
    for name, facade in provider_seam.FACADES.items():
        module = sys.modules[provider_seam.owner_of(name)]
        assert getattr(module, name) is facade, f"{provider_seam.owner_of(name)}.{name} is not its facade"


# ---------------------------------------------------------------------------
# t2 — copy-on-write
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_t2_facade_write_leaves_bound_snapshot_unchanged(name):
    facade = provider_seam.FACADES[name]
    before = provider_seam.snapshot()
    marker = "__seam_t2__"
    if isinstance(facade, dict):
        facade[marker] = object()
    elif isinstance(facade, list):
        facade.append(mcs.ProviderEntry(marker, marker, marker))
    else:
        facade.add(marker)
    after = provider_seam.current()
    assert after is not before
    held = before[name]
    now = after[name]
    has = (lambda c: any(getattr(e, "slug", None) == marker for e in c)) if isinstance(facade, list) else (lambda c: marker in c)
    assert not has(held)
    assert has(now)


# ---------------------------------------------------------------------------
# t3 / t4 — whole-or-absent publication; re-entrant owner-thread read
# ---------------------------------------------------------------------------

def test_t3_parked_publication_is_whole_before_then_whole_after():
    name = "seam-t3"
    parked, release = threading.Event(), threading.Event()
    holder: dict = {}
    provider_seam._park = _park_thread(holder, "swap", parked, release)

    def publisher():
        holder["ident"] = threading.get_ident()
        provider_seam.publish(_delta(name))

    errors: list = []
    seen: list = []

    def read(iterations: int):
        for _ in range(iterations):
            try:
                for facade in provider_seam.FACADES.values():
                    for _item in facade:
                        pass
                g = provider_seam.snapshot()
                seen.append(frozenset(v for v in _surfaces(g, name).values()))
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

    pub = threading.Thread(target=publisher)
    pub.start()
    assert parked.wait(30)
    reader = threading.Thread(target=read, args=(5000,))
    reader.start()
    reader.join(60)
    assert all(s == {False} for s in seen), "a surface of the parked pin leaked before the swap"
    release.set()
    pub.join(30)
    seen.clear()
    read(5000)
    assert not errors, errors
    assert all(s == {True} for s in seen), "a surface of the pin is missing after the swap"


def test_t4_reentrant_owner_thread_read_sees_committed_generation():
    name = "seam-t4"
    observed: dict = {}

    def park(stage: str) -> None:
        if stage == "swap":
            observed["surfaces"] = _surfaces(provider_seam.snapshot(), name)
            observed["facade"] = name in mcs._PROVIDER_LABELS

    provider_seam._park = park
    g = provider_seam.publish(_delta(name))
    assert set(observed["surfaces"].values()) == {False}
    assert observed["facade"] is False
    assert set(_surfaces(g, name).values()) == {True}


# ---------------------------------------------------------------------------
# t5 — the base-class bypass reads the same content, never empty
# ---------------------------------------------------------------------------

def test_t5_bypass_returns_same_content_and_held_base_iterator_is_loud():
    registry = hermes_cli.auth.PROVIDER_REGISTRY
    assert list(dict.items(registry)) == list(registry.items())
    assert dict.__len__(registry) == len(registry) > 0
    canonical = models.CANONICAL_PROVIDERS
    assert list(list.__iter__(canonical)) == list(canonical)

    held = dict.__iter__(registry)
    next(held)
    provider_seam.publish({"PROVIDER_REGISTRY": {"seam-t5": ProviderConfig(id="seam-t5", name="t5", auth_type="api_key")}})
    with pytest.raises(RuntimeError):
        list(held)


def _describe(value) -> str:
    try:
        return repr(value)
    except Exception as exc:  # a half-built facade copy cannot even repr itself
        return f"<{type(value).__name__}: repr raised {type(exc).__name__}>"


# ---------------------------------------------------------------------------
# t6 — C-consumer matrix, AFTER a publish and an overwriting facade write
# ---------------------------------------------------------------------------

_seam_t6_dict = None
_seam_t6_list = None


def test_t6_c_consumer_matrix_after_publish_and_overwrite():
    global _seam_t6_dict, _seam_t6_list
    _seam_t6_dict = d = GuardedDict(__name__, "_seam_t6_dict", {"a": 1, "b": 2})
    _seam_t6_list = l = GuardedList(__name__, "_seam_t6_list", ["x", "y"])
    provider_seam.publish({"_seam_t6_dict": {"c": 3}, "_seam_t6_list": ["z"]})
    d["a"] = 10  # overwrite an existing key through the facade
    DATA = {"a": 10, "b": 2, "c": 3}
    LDATA = ["x", "y", "z"]
    PLAIN = {dict, list}

    checks = {
        "json.dumps(dict) [C encoder]": (lambda: json.loads(json.dumps(d)), DATA),
        "json.dumps(dict, indent)": (lambda: json.loads(json.dumps(d, indent=1)), DATA),
        "json.dumps(dict, sort_keys)": (lambda: json.loads(json.dumps(d, sort_keys=True)), DATA),
        "dict(x)": (lambda: dict(d), DATA),
        "{**x}": (lambda: {**d}, DATA),
        "f(**x)": (lambda: (lambda **kw: kw)(**d), DATA),
        "len": (lambda: len(d), 3),
        "bool": (lambda: bool(d), True),
        "in": (lambda: "c" in d, True),
        "x == DATA": (lambda: d == DATA, True),
        "DATA == x": (lambda: DATA == d, True),
        "list(x)": (lambda: list(d), list(DATA)),
        "sorted(x)": (lambda: sorted(d), sorted(DATA)),
        "for k in x": (lambda: [k for k in d], list(DATA)),
        "copy.copy": (lambda: (type(copy.copy(d)) in PLAIN, copy.copy(d)), (True, DATA)),
        "copy.deepcopy": (lambda: (type(copy.deepcopy(d)) in PLAIN, copy.deepcopy(d)), (True, DATA)),
        "pickle rt": (lambda: (type(pickle.loads(pickle.dumps(d))) in PLAIN, pickle.loads(pickle.dumps(d))), (True, DATA)),
        "x.items() list": (lambda: list(d.items()), list(DATA.items())),
        "dict.items(x) [bypass]": (lambda: list(dict.items(d)), list(DATA.items())),
        "dict.__len__(x) [bypass]": (lambda: dict.__len__(d), 3),
        "str(x)": (lambda: str(d), str(DATA)),
        "json.dumps(list) [C]": (lambda: json.loads(json.dumps(l)), LDATA),
        "json.dumps(list, indent)": (lambda: json.loads(json.dumps(l, indent=1)), LDATA),
        "''.join(list)": (lambda: "".join(l), "xyz"),
        "list + []": (lambda: l + [], LDATA),
        "[] + list": (lambda: [] + l, LDATA),
        "list * 1": (lambda: l * 1, LDATA),
        "list(x)[list]": (lambda: list(l), LDATA),
        "tuple(x)": (lambda: tuple(l), tuple(LDATA)),
        "x[0]": (lambda: l[0], "x"),
        "x[0:1]": (lambda: (type(l[0:1]), l[0:1]), (list, ["x"])),
        "sorted(list)": (lambda: sorted(l), sorted(LDATA)),
        "list == LDATA": (lambda: l == LDATA, True),
        "LDATA == list": (lambda: LDATA == l, True),
        "copy.copy(list)": (lambda: (type(copy.copy(l)) in PLAIN, copy.copy(l)), (True, LDATA)),
        "copy.deepcopy(list)": (lambda: (type(copy.deepcopy(l)) in PLAIN, copy.deepcopy(l)), (True, LDATA)),
        "pickle rt list": (lambda: (type(pickle.loads(pickle.dumps(l))) in PLAIN, pickle.loads(pickle.dumps(l))), (True, LDATA)),
        "enumerate": (lambda: [i for i, _ in enumerate(l)], [0, 1, 2]),
        "any/all": (lambda: (any(l), all(l)), (True, True)),
        "list.__iter__(x) [bypass]": (lambda: list(list.__iter__(l)), LDATA),
        "x in list": (lambda: "z" in l, True),
        "*x unpack": (lambda: [*l], LDATA),
        "f(*x)": (lambda: (lambda *a: list(a))(*l), LDATA),
        "min/max": (lambda: (min(l), max(l)), ("x", "z")),
    }
    failures = {}
    for label, (fn, expect) in checks.items():
        try:
            got = fn()
        except Exception as exc:
            got = f"{type(exc).__name__}: {exc}"
        if got != expect:
            failures[label] = _describe(got)
    print(f"t6 matrix: {len(checks)} checks on python {platform.python_version()}")
    assert not failures, f"C-consumer checks that did not see the data: {failures}"


# ---------------------------------------------------------------------------
# t7 — lost update: the compare-and-swap re-derives
# ---------------------------------------------------------------------------

_seam_t7 = None


def test_t7_second_writer_rederives_and_both_additions_survive():
    global _seam_t7
    _seam_t7 = facade = GuardedDict(__name__, "_seam_t7", {"base": 0})
    parked, release = threading.Event(), threading.Event()
    holder: dict = {}
    provider_seam._park = _park_thread(holder, "build", parked, release)

    def writer_a():
        holder["ident"] = threading.get_ident()
        facade["from_a"] = 1

    a = threading.Thread(target=writer_a)
    a.start()
    assert parked.wait(30)  # A built its plan from the shared base and is parked
    facade["from_b"] = 2  # B commits in between
    release.set()
    a.join(30)
    committed = dict(provider_seam.current()["_seam_t7"])
    assert committed == {"base": 0, "from_a": 1, "from_b": 2}
    assert dict(dict.items(facade)) == committed  # the mirror agrees


# ---------------------------------------------------------------------------
# Publication semantics
# ---------------------------------------------------------------------------

def test_publish_is_idempotent_per_committed_name():
    first = provider_seam.publish(_delta("seam-idem"))
    again = provider_seam.publish(_delta("seam-idem"))
    assert again is first
    assert provider_seam.current() is first


@pytest.mark.parametrize("names", [
    ["seam-pin-1", "seam-pin-7", "seam-pin-200"],
    [f"seam-high-{i}" for i in range(300)],
])
def test_sparse_and_high_n_each_publish_their_own_generation(names):
    generations = [provider_seam.publish(_delta(n)) for n in names]
    assert len({id(g) for g in generations}) == len(names)
    g = provider_seam.snapshot()
    assert set(names) <= g.committed["test-lane"]
    for n in names:
        assert set(_surfaces(g, n).values()) == {True}, n


def test_collision_with_a_foreign_owner_is_rejected_whole():
    existing = next(iter(hermes_cli.auth.PROVIDER_REGISTRY))
    before = provider_seam.current()
    delta = _delta("seam-collide")
    delta["PROVIDER_REGISTRY"][existing] = ProviderConfig(id=existing, name="impostor", auth_type="api_key")
    with pytest.raises(provider_seam.SeamCollision):
        provider_seam.publish(delta)
    assert provider_seam.current() is before
    assert set(_surfaces(before, "seam-collide").values()) == {False}


def test_removal_is_copy_and_swap_and_copies_are_plain():
    """Removal is supported (``patch.dict``, the plugin doctor's unload) but never mutates a
    generation a reader already holds."""
    registry = hermes_cli.auth.PROVIDER_REGISTRY
    key = next(iter(registry))
    held = provider_seam.snapshot()
    value = registry.pop(key)
    assert key not in registry and key not in dict.keys(registry)  # facade and mirror agree
    assert key in held.PROVIDER_REGISTRY
    registry[key] = value
    canonical = mcs.CANONICAL_PROVIDERS
    first = canonical[0]
    canonical.remove(first)
    assert first not in canonical and first in held.CANONICAL_PROVIDERS
    canonical.insert(0, first)
    assert list(list.__iter__(canonical)) == list(held.CANONICAL_PROVIDERS)
    models._KNOWN_PROVIDER_NAMES.discard("openrouter")
    assert "openrouter" not in models._KNOWN_PROVIDER_NAMES and "openrouter" in held._KNOWN_PROVIDER_NAMES
    assert type(copy.copy(registry)) is dict
    assert type(copy.deepcopy(models._KNOWN_PROVIDER_NAMES)) is set
    known = models._KNOWN_PROVIDER_NAMES
    assert set() | known == set(known) == known | set()
    assert type(known | set()) is set


def test_patch_dict_round_trips_through_the_facade():
    from unittest import mock

    registry = hermes_cli.auth.PROVIDER_REGISTRY
    before = dict(registry)
    with mock.patch.dict(registry, {"seam-patched": ProviderConfig(id="seam-patched", name="p", auth_type="api_key")}, clear=True):
        assert list(registry) == ["seam-patched"]
    assert dict(registry) == before == dict(dict.items(registry))


# ---------------------------------------------------------------------------
# Refresh callbacks
# ---------------------------------------------------------------------------

def test_refresh_same_thread_reentry_is_noop_and_depth_resets_after_failure():
    calls: list = []

    def cb(reason, name):
        calls.append((reason, name))
        provider_seam.refresh("typed", "nested")  # re-entry: skipped
        if name == "boom":
            raise RuntimeError("callback failure")

    provider_seam.register_refresh(cb)
    provider_seam.register_refresh(cb)  # idempotent
    provider_seam.refresh("request", "boom")
    provider_seam.refresh("picker")
    assert calls == [("request", "boom"), ("picker", None)]
    with pytest.raises(ValueError):
        provider_seam.refresh("sideways")


class _Stop(BaseException):
    """Aborts a call AT its refresh trigger (``refresh()`` only swallows ``Exception``)."""


def test_refresh_triggers_run_before_recognition():
    seen: list = []
    provider_seam.register_refresh(lambda reason, name: seen.append((reason, name)))
    models.parse_model_input("seam-typed:some-model", "openrouter")
    assert ("typed", "seam-typed") in seen

    from hermes_cli.model_switch import resolve_startup_model_route
    from hermes_cli.provider_catalog import provider_catalog

    resolve_startup_model_route("seam-startup:some-model", current_provider="openrouter")
    assert ("typed", "seam-startup") in seen
    provider_catalog()
    assert ("picker", None) in seen


@pytest.mark.parametrize("call, expected", [
    ("switch", ("typed", "seam-switch")),
    ("runtime", ("request", "seam-runtime")),
    ("picker", ("picker", None)),
])
def test_refresh_trigger_is_the_first_step(monkeypatch, call, expected):
    """The trigger fires before any resolution work (a callback that publishes the provider must
    run before the lookup that needs it); the callback aborts the call right there."""
    seen: list = []

    def cb(reason, name):
        seen.append((reason, name))
        raise _Stop

    monkeypatch.setattr(provider_seam, "_refresh_callbacks", [cb])
    from hermes_cli.model_switch import switch_model
    from hermes_cli.model_switch_providers import list_authenticated_providers
    from hermes_cli.runtime_provider import resolve_runtime_provider

    fn = {
        "switch": lambda: switch_model("seam-switch:some-model", "openrouter", "x"),
        "runtime": lambda: resolve_runtime_provider(requested="seam-runtime"),
        "picker": lambda: list_authenticated_providers(),
    }[call]
    with pytest.raises(_Stop):
        fn()
    assert seen == [expected]


# ---------------------------------------------------------------------------
# Event-loop latency with a publisher parked inside the critical section
# ---------------------------------------------------------------------------

def test_event_loop_reads_complete_while_publisher_holds_the_lock(monkeypatch):
    counting = _CountingLock()
    monkeypatch.setattr(provider_seam, "_lock", counting)
    parked, release = threading.Event(), threading.Event()
    holder: dict = {}
    provider_seam._park = _park_thread(holder, "swap", parked, release)

    def publisher():
        holder["ident"] = threading.get_ident()
        provider_seam.publish(_delta("seam-latency"))

    pub = threading.Thread(target=publisher)
    pub.start()
    assert parked.wait(30)

    timings: list = []
    reader_ident: dict = {}

    async def loop_body():
        reader_ident["ident"] = threading.get_ident()
        for _ in range(1000):
            t0 = time.perf_counter()
            models.parse_model_input("openrouter:anthropic/claude", "nous")
            hermes_cli.auth.PROVIDER_REGISTRY.get("nous")
            provider_seam.snapshot()
            timings.append(time.perf_counter() - t0)
            await asyncio.sleep(0)

    reader = threading.Thread(target=lambda: asyncio.run(loop_body()))
    reader.start()
    reader.join(60)
    assert not reader.is_alive()
    assert pub.is_alive(), "all reads completed while the publisher still held the lock"
    release.set()
    pub.join(30)
    timings.sort()
    p99 = timings[int(len(timings) * 0.99) - 1]
    print(f"event-loop read p99={p99 * 1000:.3f}ms over {len(timings)} iterations")
    assert len(timings) == 1000
    assert counting.acquired[reader_ident["ident"]] == 0
    # Spec target is < 1 ms (measured ~0.03 ms); the bound is loose so a
    # loaded CI runner cannot flake it. The lock counter above is the gate.
    assert p99 < 0.05


def test_iteration_during_registration_never_tears():
    """Before the seam: RuntimeError('dictionary changed size during iteration')."""
    registry = hermes_cli.auth.PROVIDER_REGISTRY
    labels = mcs._PROVIDER_LABELS
    it_registry, it_labels = iter(registry.items()), iter(labels)
    next(it_registry), next(it_labels)
    providers.register_provider(ProviderProfile(name="seam-tear", display_name="Seam Tear",
                                                base_url="https://example.invalid/v1", env_vars=("SEAM_TEAR_KEY",)))
    assert "seam-tear" in registry and "seam-tear" in labels
    list(it_registry), list(it_labels)  # the held iterators finish over their own generation


def test_late_registered_provider_is_recognised_by_provider_model_parsing():
    """Before the seam ``_KNOWN_PROVIDER_NAMES`` was computed once at import: a provider registered
    later was listed in the picker, but ``/model late:m`` fell through to the current provider."""
    out = _run_child("""
        import providers
        from providers import ProviderProfile
        import hermes_cli.models as models
        providers.list_providers()
        providers.register_provider(ProviderProfile(name="seamlate", display_name="Seam Late",
                                                    base_url="https://example.invalid/v1", env_vars=("SEAMLATE_KEY",)))
        from hermes_cli import provider_seam
        g = provider_seam.snapshot()
        print("SURFACES", "seamlate" in [e.slug for e in g.CANONICAL_PROVIDERS], g._PROVIDER_LABELS.get("seamlate"),
              "seamlate" in g._KNOWN_PROVIDER_NAMES, "PARSE", models.parse_model_input("seamlate:m1", "openrouter"))
    """)
    assert "SURFACES True Seam Late True PARSE ('seamlate', 'm1')" in out, out


# ---------------------------------------------------------------------------
# Boot orders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("first", ["providers", "hermes_cli.models"])
def test_both_boot_orders_bind_every_facade(first):
    out = _run_child(f"""
        import importlib, sys
        importlib.import_module({first!r})
        import providers, hermes_cli.models, hermes_cli.auth, hermes_cli.providers
        from hermes_cli import provider_seam as s
        bad = [n for n, f in s.FACADES.items() if getattr(sys.modules[s.owner_of(n)], n) is not f]
        print("FACADES", len(s.FACADES), "BAD", bad, "CANON", len(hermes_cli.models.CANONICAL_PROVIDERS) > 0)
    """)
    assert "FACADES 9 BAD [] CANON True" in out, out
