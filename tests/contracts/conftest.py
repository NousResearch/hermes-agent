"""Shared fixtures for the persistence contract suites.

Tests requesting ``backend_name``, directly or through fixtures, run against each registered assembled backend by parametrization; backend-neutral tests do not. The in-memory reference backend (``tests/contracts/_reference/``) registers here; every other backend registers through the same ``register_backend`` hook from a ``_<name>_backend.py`` module in this folder, loaded when present. The registry is typed against ``AssembledBackend`` (declared capabilities plus named repository attributes), not against the reference implementation, so a registering backend never has to subclass the test fixture.

Fixtures provided:

- ``backend_factory``: async context manager yielding a fresh assembled backend per entry (named repository attributes, declared ``PersistenceCapabilities``, deterministic teardown).
- ``backend``: one function-scoped backend instance per test, torn down deterministically.
- ``scope_a`` / ``scope_b``: two distinct-profile ``ProfileScope`` values, each opened as its own store handle on the same backend instance so isolation proofs exercise the provider, not separate assemblies.

Collection policy: a contracts run that collects backend-bound cases fails when every such case has a ``skip`` marker or a ``skipif`` marker whose first positional argument is truthy; entirely backend-neutral collections are exempt. It tests that argument directly; it does not evaluate string conditions or read ``condition=``. Use ``require_capability`` for capability-gated optional features so the skip reason names the missing capability; ``_approved_skip`` also accepts uppercase ``QUIRK`` reason text.

Skip accounting: per-backend outcome counts are collected from runtest reports (a skip raised at setup or call time counts as a skip; nothing is inferred at collection time) and written as JSON to the path in ``HERMES_CONTRACT_ACCOUNTING`` when set and reports or backend-bound collected nodes exist, keyed by backend name with collected/passed/skipped/xfailed/errored/failed counts plus each skipped node id and reason. With that artifact requested, ``pytest_sessionfinish`` fails an otherwise successful run on accounting violations, including an unapproved skip in a mapped authority case whose backend declares the capability.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, Optional, Protocol

import pytest

from persistence import ProfileScope
from persistence.core.backend import PersistenceBackend
from tests.contracts._reference import open_reference_backend
from tests.contracts._scopes import profile_scope


class AssembledBackend(PersistenceBackend, Protocol):
    """What every registered backend must present: the interface's ``PersistenceBackend`` door (``open(scope)``, the admin doors, ``reopen``, ``close``) plus the suites' epoch seam. The harness opens one handle per profile and reuses it."""

    def begin_epoch(self, scope: ProfileScope, instance_id: str) -> int:
        """Begin an authority epoch on ``scope``'s store and return it: the epoch seam suites use when the runtime family is not the subject."""
        ...


BackendFactory = Callable[[], ContextManager[AssembledBackend]]

_BACKEND_FACTORIES: dict[str, BackendFactory] = {}

# The two test scopes are two distinct profiles, the same pair on every backend.
_SCOPES = (profile_scope("profile-a"), profile_scope("profile-b"))


def register_backend(name: str, factory: BackendFactory) -> None:
    """Registration hook: one name, one zero-argument async-context-manager factory."""
    if name in _BACKEND_FACTORIES:
        raise ValueError(f"backend {name!r} already registered")
    _BACKEND_FACTORIES[name] = factory


register_backend("in_memory", open_reference_backend)

_CONTRACTS_DIR = Path(__file__).parent

# Each backend registration module beside this file: its ``register(register_backend, scopes)`` registers the backend over the two test scopes when the tree holds it.
for _module in sorted(_CONTRACTS_DIR.glob("_*_backend.py")):
    importlib.import_module(f"tests.contracts.{_module.stem}").register(register_backend, _SCOPES)


def registered_backend_factory(name: str) -> Optional[BackendFactory]:
    """The factory registered under ``name``, or ``None``, read without touching the private registry."""
    return _BACKEND_FACTORIES.get(name)


# nodeid -> backend name, built at collection time from exact callspec params
# (outcome accounting ownership; the outcomes themselves stay report-driven).
_NODE_BACKENDS: dict[str, str] = {}

# ── Authority-contract accounting ─────

# The three authority capability flags.
_AUTHORITY_FLAGS = ("authority_fences", "durable_admissions", "receipted_writes")

# Authority suite file -> the capability flag its cases exercise (the
# node-to-capability metadata). This conftest owns the map; a capability
# suite must land under its named file to be accounted.
_AUTHORITY_SUITE_FLAGS: dict[str, str] = {
    "test_runtime_contract.py": "authority_fences",
    "test_admissions_contract.py": "durable_admissions",
    "test_worker_executions_contract.py": "receipted_writes",
    "test_bindings_contract.py": "receipted_writes",
}

# nodeid -> capability flag, from the suite map at collection time.
_NODE_CAPABILITY_FLAGS: dict[str, str] = {}

# backend -> collected contract node ids, from the FINAL collection (
# denominators come from collection, never from report arrival).
_COLLECTED_NODES: dict[str, set[str]] = defaultdict(set)

# backend -> {flag: declared bool}, recorded when the backend fixture assembles
# an instance; the declaration is the assembled capabilities object itself.
_DECLARED_AUTHORITY_FLAGS: dict[str, dict[str, bool]] = {}


def _approved_skip(reason: str) -> bool:
    """Approve any reason containing 'backend does not declare' or 'QUIRK' (case-sensitive). These are the substrings used by the require_* helpers and quirk cases; provenance is not checked. Reasons containing only 'backend does not implement ...' or a lowercase 'quirk' mention stay unapproved."""
    return "backend does not declare" in reason or "QUIRK" in reason


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "contract_backend: parametrized persistence-contract case bound to one registered backend")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize every test that touches a backend over the registered backends."""
    if "backend_name" in metafunc.fixturenames:
        names = sorted(_BACKEND_FACTORIES)
        metafunc.parametrize("backend_name", names, ids=names)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Fail loudly on coroutine tests and tag backend-bound cases ``contract_backend``.

    Contract verbs are synchronous, so a coroutine test is a leftover that would never run its body; it is a hard collection error.
    """
    unmarked: list[str] = []
    for item in items:
        if _CONTRACTS_DIR not in Path(str(item.path)).parents:
            continue
        if isinstance(item, pytest.Function):
            if inspect.iscoroutinefunction(item.function):
                unmarked.append(item.nodeid)
            if "backend_name" in getattr(item, "fixturenames", ()):
                item.add_marker(pytest.mark.contract_backend)
            # Exact backend ownership for the outcome accounting: the real
            # callspec param, never a name parsed out of the display id.
            callspec = getattr(item, "callspec", None)
            if callspec is not None and "backend_name" in callspec.params:
                _NODE_BACKENDS[item.nodeid] = str(callspec.params["backend_name"])
            flag = _AUTHORITY_SUITE_FLAGS.get(item.path.name)
            if flag is not None:
                _NODE_CAPABILITY_FLAGS[item.nodeid] = flag
    if unmarked:
        raise pytest.UsageError(
            "coroutine contract tests (verbs are synchronous; they would never run):\n"
            + "\n".join(unmarked)
        )


def pytest_collection_finish(session: pytest.Session) -> None:
    """Fail a contracts collection whose backend-bound cases each have a skip marker or a truthy first positional skipif argument; backend-neutral collections are exempt.

    A case only counts as executable here if it has a ``contract_backend`` marker, no ``skip`` marker and no ``skipif`` marker with a truthy first positional argument. String conditions are not evaluated, and keyword ``condition`` arguments are not inspected. Capability-gated runtime skips still count because they execute against the declared capabilities.

    Also records the accounting denominators: per-backend collected node ids come from this final collection (post-deselection), so missing reports or nodes that disappeared after collection modification are detectable against them.
    """
    contract_items = [
        item for item in session.items if _CONTRACTS_DIR in Path(str(item.path)).parents
    ]
    if not contract_items:
        return
    for item in contract_items:
        backend = _NODE_BACKENDS.get(item.nodeid)
        if backend is not None:
            _COLLECTED_NODES[backend].add(item.nodeid)
    executable = [
        item
        for item in contract_items
        if item.get_closest_marker("contract_backend")
        and item.get_closest_marker("skip") is None
        and not any(
            mark.name == "skipif" and mark.args and bool(mark.args[0])
            for mark in item.iter_markers()
        )
    ]
    if not executable and any(
        item.get_closest_marker("contract_backend") for item in contract_items
    ):
        # Zero executable is only vacuous when the run COLLECTED backend-bound
        # cases (all statically skipped). A run of purely static policy suites
        # (test_raw_access_policy.py, the raw-access detector) has no backend cases by
        # design and must not be forced to fake one.
        raise pytest.UsageError(
            "tests/contracts collected zero executable backend cases; "
            "a contract suite must run real backend behavior, not just import"
        )


# ── outcome-based per-backend skip accounting ───────────────

_ACCOUNTING_ENV = "HERMES_CONTRACT_ACCOUNTING"


def _backend_of(nodeid: str) -> Optional[str]:
    """The backend a node id binds to, from the ownership map built at collection time (exact ``backend_name`` callspec params; never parsed out of display ids)."""
    return _NODE_BACKENDS.get(nodeid)


class _OutcomeAccounting:
    """Collects per-backend outcome counts from runtest reports; never from collection state."""

    def __init__(self) -> None:
        self.counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"collected": 0, "passed": 0, "skipped": 0, "xfailed": 0, "errored": 0, "failed": 0}
        )
        self.skips: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.neutral: dict[str, int] = {"collected": 0, "passed": 0, "skipped": 0, "xfailed": 0, "errored": 0, "failed": 0}
        self.seen: set[str] = set()
        # nodeid -> final outcome, for the collection-vs-report reconciliation.
        self.outcomes: dict[str, str] = {}

    def _bucket(self, nodeid: str) -> tuple[dict[str, int], Optional[str]]:
        backend = _backend_of(nodeid)
        if backend is None:
            return self.neutral, None
        return self.counts[backend], backend

    def record(self, report: pytest.TestReport) -> None:
        # report.fspath is rootdir-relative; resolve before comparing to the
        # absolute contracts dir (collection hooks use item.path, already absolute).
        if _CONTRACTS_DIR not in Path(str(report.fspath)).resolve().parents:
            return
        bucket, backend = self._bucket(report.nodeid)
        if report.nodeid not in self.seen:
            self.seen.add(report.nodeid)
            bucket["collected"] += 1
        if report.skipped:
            # Setup- or call-phase skip both count.
            if hasattr(report, "wasxfail"):
                bucket["xfailed"] += 1
                self.outcomes[report.nodeid] = "xfailed"
            else:
                bucket["skipped"] += 1
                self.outcomes[report.nodeid] = "skipped"
                reason = ""
                if isinstance(report.longrepr, tuple):
                    reason = str(report.longrepr[2])
                if backend is not None:
                    self.skips[backend].append({"nodeid": report.nodeid, "reason": reason})
        elif report.failed:
            key = "errored" if report.when != "call" else "failed"
            bucket[key] += 1
            self.outcomes[report.nodeid] = key
        elif report.passed and report.when == "call":
            bucket["passed"] += 1
            self.outcomes[report.nodeid] = "passed"


_accounting = _OutcomeAccounting()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Feed every phase report into the outcome accounting; skips surface at setup OR call time."""
    if report.when in ("setup", "call"):
        _accounting.record(report)


def _authority_accounting() -> dict[str, Any]:
    """Per-backend per-capability denominators from the final collection, declared flags from the assembled capabilities, and the violation list used by ``pytest_sessionfinish``.

    Violations (each a dict with a ``kind``): ``missing_report`` (a collected backend-bound node produced no outcome), ``unexpected_deselection`` (a node seen at collection-modification time vanished from the final collection), ``missing_declaration`` (authority cases were collected but no assembled instance recorded its flags), ``unapproved_skip`` (an authority-flagged case skipped on a backend that declares the flag, with a reason ``_approved_skip`` rejects), and ``zero_exercised`` (a backend declares an authority flag with no passed cases, including when no cases for that flag were collected). ``pytest_sessionfinish`` writes this list to the requested accounting artifact and fails an otherwise successful run when it is nonempty; without an artifact request it does neither.
    """
    per_capability: dict[str, dict[str, dict[str, int]]] = {}
    violations: list[dict[str, str]] = []
    final_nodes = set().union(*_COLLECTED_NODES.values()) if _COLLECTED_NODES else set()
    for nodeid in sorted(set(_NODE_BACKENDS) - final_nodes):
        violations.append({"kind": "unexpected_deselection", "nodeid": nodeid})
    for backend, nodes in sorted(_COLLECTED_NODES.items()):
        flags = per_capability.setdefault(backend, {})
        for nodeid in sorted(nodes):
            outcome = _accounting.outcomes.get(nodeid)
            if outcome is None:
                violations.append({"kind": "missing_report", "backend": backend, "nodeid": nodeid})
            flag = _NODE_CAPABILITY_FLAGS.get(nodeid)
            if flag is None:
                continue
            counts = flags.setdefault(
                flag, {"collected": 0, "passed": 0, "skipped": 0, "xfailed": 0, "errored": 0, "failed": 0}
            )
            counts["collected"] += 1
            if outcome is not None:
                counts[outcome] += 1
        declared = _DECLARED_AUTHORITY_FLAGS.get(backend, {})
        if not declared and any(n in _NODE_CAPABILITY_FLAGS for n in nodes):
            # Authority-flagged cases were collected but no assembled instance
            # ever recorded its declaration (every case statically skipped):
            # the declared-flag checks below would silently vacuously pass.
            violations.append({"kind": "missing_declaration", "backend": backend})
        for flag, value in sorted(declared.items()):
            if not value:
                continue
            counts = flags.get(flag)
            if counts is None or counts["passed"] == 0:
                # A declared capability with NO collected cases (omitted suite,
                # broken node-to-capability map) is as much a zero-exercise
                # violation as one whose cases all skipped or failed.
                violations.append({"kind": "zero_exercised", "backend": backend, "capability": flag})
        for entry in _accounting.skips.get(backend, []):
            flag = _NODE_CAPABILITY_FLAGS.get(entry["nodeid"])
            if flag is not None and declared.get(flag) and not _approved_skip(entry["reason"]):
                violations.append(
                    {"kind": "unapproved_skip", "backend": backend, "nodeid": entry["nodeid"], "reason": entry["reason"]}
                )
    return {
        "capabilities": per_capability,
        "declared_flags": dict(sorted(_DECLARED_AUTHORITY_FLAGS.items())),
        "violations": violations,
    }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write accounting JSON when HERMES_CONTRACT_ACCOUNTING names a path and reports or backend-bound collected nodes exist; fail an otherwise successful run on accounting violations.

    Enforcement is tied to the artifact request: without the env var this hook returns without computing violations, writing an artifact, or changing the exit status. With the env var, violations turn a successful exit status into failure. Collected backend-bound nodes produce an artifact even when zero reports arrived (``missing_report``); with neither reports nor backend-bound collected nodes, no artifact is written.
    """
    target = os.environ.get(_ACCOUNTING_ENV)
    if not target or not (_accounting.seen or _COLLECTED_NODES):
        return
    authority = _authority_accounting()
    payload = {
        "backends": {name: dict(counts) for name, counts in _accounting.counts.items()},
        "skips": {name: list(entries) for name, entries in _accounting.skips.items()},
        "backend_neutral": dict(_accounting.neutral),
        "authority": authority,
    }
    Path(target).write_text(json.dumps(payload, indent=1, sort_keys=True))
    if authority["violations"] and exitstatus == 0:
        session.exitstatus = 1
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        if tr is not None:
            tr.write_line(
                f"accounting violations ({len(authority['violations'])}); see {target}",
                red=True,
            )


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def backend_name() -> str:
    """Placeholder overridden by pytest_generate_tests parametrization."""
    raise RuntimeError("backend_name is parametrized; a bare request means collection is broken")


@pytest.fixture
def backend_factory(backend_name: str) -> BackendFactory:
    """The registered factory for the parametrized backend; each call yields a fresh assembled instance."""
    return _BACKEND_FACTORIES[backend_name]


@pytest.fixture
def backend(backend_name: str, backend_factory: BackendFactory):
    """One fresh assembled backend per test, deterministically torn down."""
    with backend_factory() as instance:
        _DECLARED_AUTHORITY_FLAGS.setdefault(
            backend_name,
            {flag: bool(getattr(instance.capabilities, flag)) for flag in _AUTHORITY_FLAGS},
        )
        yield instance


@pytest.fixture
def scope_a() -> ProfileScope:
    """First test scope."""
    return _SCOPES[0]


@pytest.fixture
def scope_b() -> ProfileScope:
    """Second test scope; a distinct profile from scope_a."""
    return _SCOPES[1]


def require_capability(backend: AssembledBackend, capability: str) -> None:
    """Skip (never silently pass) a capability-gated test when the backend does not declare the capability."""
    if not getattr(backend.capabilities, capability):
        pytest.skip(f"backend does not declare capability {capability!r}")


def require_fenced_config_patch(backend: AssembledBackend) -> None:
    """Skip a same-transition fenced model-config-patch case on backends whose lease fence is check-then-write: they declare ``fenced_config_patch`` False; the reference declares True."""
    if not getattr(backend, "fenced_config_patch", False):
        pytest.skip(
            "backend does not declare fenced_config_patch: its lease fence is "
            "check-then-write, not the same-transition fence"
        )


def require_failpoints(backend: AssembledBackend) -> None:
    """Skip a reference-atomicity crash case on backends without injectable failpoints.

    Only the reference backend exposes the staged-mutation injection seam; a backend without it proves crash consistency through its own transactions instead.
    """
    if not getattr(backend, "injectable_failpoints", False):
        pytest.skip(
            "backend does not declare injectable_failpoints: crash-window "
            "injection is the reference backend's surface"
        )
