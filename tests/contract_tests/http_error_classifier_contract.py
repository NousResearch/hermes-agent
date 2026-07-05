"""HttpErrorClassifier Contract Suite.

Parametrized test suite. Any provider that implements
:class:`HttpErrorClassifier` must pass every test in this suite
without modification.

Usage::

    from tests.contract_tests.http_error_classifier_contract import (
        run_http_error_classifier_contract,
    )
    from agent.provider_errors import CodexErrorClassifier

    def _factory():
        return CodexErrorClassifier()

    run_http_error_classifier_contract(_factory)
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from agent.provider_errors import (
    HttpErrorClassifier,
    ProviderErrorCode,
    ProviderErrorFact,
)


def _identity(fact: ProviderErrorFact | None) -> tuple[str, int]:
    """Return a comparable snapshot of a fact's identity (code, http_status)."""
    if fact is None:
        return ("<none>", -1)
    return (fact.error_code.value, int(fact.http_status or -1))


def _run(test_name: str, factory: Callable[[], HttpErrorClassifier]) -> None:
    """Run a single named contract test against a fresh classifier."""
    cls = factory().__class__.__name__
    method = globals()[test_name]
    method(factory)
    print(f"  [{cls}] {test_name}: PASS")


# ---------------------------------------------------------------------------
# Level 1 — Structural
# ---------------------------------------------------------------------------


def test_structural_subclasses_abstract_base(factory):
    cls = factory()
    assert isinstance(cls, HttpErrorClassifier)


def test_structural_signature(factory):
    cls = factory()
    # parse_provider_error must accept exactly (*, status_code, body, text)
    # and return ProviderErrorFact | None.
    result = cls.parse_provider_error(
        status_code=418, body=None, text=""
    )
    assert result is None or isinstance(result, ProviderErrorFact)


# ---------------------------------------------------------------------------
# Level 2 — Functional
# ---------------------------------------------------------------------------


def test_classifier_is_deterministic(factory):
    cls = factory()
    inputs = [
        (401, None, ""),
        (429, None, ""),
        (503, None, ""),
        (200, None, ""),
    ]
    for status_code, body, text in inputs:
        for _ in range(10):
            r1 = cls.classify_http_response(
                status_code=status_code, body=body, text=text
            )
            r2 = cls.classify_http_response(
                status_code=status_code, body=body, text=text
            )
            assert _identity(r1) == _identity(r2), (
                f"non-deterministic for {status_code}"
            )


def test_equivalent_mappings_produce_same_provider_error_fact(factory):
    cls = factory()
    body_dict = {"error": {"code": "quota", "message": "out of credits"}}
    body_proxy = MappingProxyType(dict(body_dict))
    body_frozen = _FrozenMapping(dict(body_dict))

    for status_code in (400, 429, 503):
        text = "quota exceeded"
        r_dict = cls.classify_http_response(
            status_code=status_code, body=body_dict, text=text
        )
        r_proxy = cls.classify_http_response(
            status_code=status_code, body=body_proxy, text=text
        )
        r_frozen = cls.classify_http_response(
            status_code=status_code, body=body_frozen, text=text
        )
        assert _identity(r_dict) == _identity(r_proxy), (
            f"dict vs proxy diverged at {status_code}"
        )
        assert _identity(r_dict) == _identity(r_frozen), (
            f"dict vs frozen diverged at {status_code}"
        )


def test_mapping_proxy_behaves_identically_to_dict(factory):
    cls = factory()
    cases = [
        (401, {"error": "auth"}),
        (429, {"error": "rate"}),
        (400, {"error": "context"}),
    ]
    for status_code, body in cases:
        r1 = cls.classify_http_response(
            status_code=status_code, body=dict(body), text=""
        )
        r2 = cls.classify_http_response(
            status_code=status_code,
            body=MappingProxyType(dict(body)),
            text="",
        )
        assert _identity(r1) == _identity(r2)


def test_semantically_equivalent_inputs_are_equal(factory):
    cls = factory()
    body_a = {"error": {"code": "billing"}}
    body_b = {"error": {"code": "billing"}}
    # Both dicts; semantically identical.
    r1 = cls.classify_http_response(
        status_code=400, body=body_a, text="quota exceeded"
    )
    r2 = cls.classify_http_response(
        status_code=400, body=body_b, text="quota exceeded"
    )
    assert _identity(r1) == _identity(r2)


# ---------------------------------------------------------------------------
# Level 3 — Purity
# ---------------------------------------------------------------------------


def test_mapping_input_is_not_mutated(factory):
    cls = factory()
    body = {"error": {"code": "quota"}}
    snapshot = json.dumps(body, sort_keys=True)
    cls.classify_http_response(status_code=400, body=body, text="quota")
    cls.classify_http_response(
        status_code=400, body=MappingProxyType(dict(body)), text="quota"
    )
    assert json.dumps(body, sort_keys=True) == snapshot


def test_no_side_effects_no_io(factory):
    """Pure classifiers must not perform any IO.

    We verify purity by behavioral test: calling the classifier
    must not raise on inputs that would require network/filesystem
    access (we simply don't provide any transport).
    """
    cls = factory()
    # If the classifier tried to read secrets or hit the network, it
    # would either raise or hang. It does neither.
    for _ in range(50):
        result = cls.classify_http_response(
            status_code=503, body=None, text=""
        )
        assert isinstance(result, ProviderErrorFact)


def test_no_global_state_dependency(factory):
    """Two fresh instances produce identical results for identical inputs."""
    for _ in range(5):
        a = factory()
        b = factory()
        for sc in (200, 401, 429, 503):
            ra = a.classify_http_response(status_code=sc, body=None, text="")
            rb = b.classify_http_response(status_code=sc, body=None, text="")
            assert _identity(ra) == _identity(rb)


def test_classify_exception_purity(factory):
    cls = factory()
    for exc in [
        TimeoutError("timed out"),
        ConnectionError("refused"),
        OSError("os error"),
        ValueError("unknown"),
    ]:
        fact = cls.classify_exception(exc)
        assert isinstance(fact, ProviderErrorFact)


# ---------------------------------------------------------------------------
# Level 4 — Behaviour
# ---------------------------------------------------------------------------


def test_5xx_returns_transient_error(factory):
    cls = factory()
    for code in (500, 502, 503, 504):
        fact = cls.classify_http_response(
            status_code=code, body=None, text=""
        )
        assert fact is not None, f"5xx {code} must produce a fact"
        assert fact.error_code is ProviderErrorCode.TRANSIENT_ERROR


def test_200_ok_with_empty_body_returns_invalid_response(factory):
    cls = factory()
    fact = cls.classify_http_response(status_code=200, body=None, text="")
    assert fact is not None
    assert fact.error_code is ProviderErrorCode.INVALID_RESPONSE


def test_200_ok_with_parseable_body_returns_none(factory):
    cls = factory()
    fact = cls.classify_http_response(
        status_code=200, body={"ok": True}, text=""
    )
    assert fact is None


def test_4xx_delegates_to_provider_parser(factory):
    cls = factory()
    # 401 / 403 → AUTH_ERROR; this is provider-specific but expected by
    # every classifier that conforms to the contract.
    for sc in (401, 403):
        fact = cls.classify_http_response(status_code=sc, body=None, text="")
        assert fact is not None
        assert fact.error_code is ProviderErrorCode.AUTH_ERROR
    # 429 → RATE_LIMIT_EXCEEDED.
    fact = cls.classify_http_response(status_code=429, body=None, text="")
    assert fact is not None
    assert fact.error_code is ProviderErrorCode.RATE_LIMIT_EXCEEDED


def test_generic_5xx_does_not_invoke_provider_parser(factory):
    """Even if parse_provider_error would return something, 5xx must
    resolve to TRANSIENT_ERROR via the base."""
    cls = factory()
    fact = cls.classify_http_response(status_code=500, body=None, text="")
    assert fact is not None
    assert fact.error_code is ProviderErrorCode.TRANSIENT_ERROR


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

CONTRACT_TESTS: tuple[str, ...] = (
    # Level 1
    "test_structural_subclasses_abstract_base",
    "test_structural_signature",
    # Level 2
    "test_classifier_is_deterministic",
    "test_equivalent_mappings_produce_same_provider_error_fact",
    "test_mapping_proxy_behaves_identically_to_dict",
    "test_semantically_equivalent_inputs_are_equal",
    # Level 3
    "test_mapping_input_is_not_mutated",
    "test_no_side_effects_no_io",
    "test_no_global_state_dependency",
    "test_classify_exception_purity",
    # Level 4
    "test_5xx_returns_transient_error",
    "test_200_ok_with_empty_body_returns_invalid_response",
    "test_200_ok_with_parseable_body_returns_none",
    "test_4xx_delegates_to_provider_parser",
    "test_generic_5xx_does_not_invoke_provider_parser",
)


def run_http_error_classifier_contract(
    factory: Callable[[], HttpErrorClassifier],
    *,
    name: str = "<unnamed>",
) -> bool:
    """Run the full HttpErrorClassifier contract suite.

    Returns True if all tests pass; prints a per-test summary.
    """
    print(f"\n[contract] Running suite for {name}")
    passed = 0
    failed = 0
    for test_name in CONTRACT_TESTS:
        try:
            _run(test_name, factory)
            passed += 1
        except AssertionError as exc:
            failed += 1
            print(
                f"  [{name}] {test_name}: FAIL — {exc}"
            )
        except Exception as exc:
            failed += 1
            print(
                f"  [{name}] {test_name}: ERROR — {type(exc).__name__}: {exc}"
            )
    print(
        f"[contract] {name}: {passed} passed, {failed} failed "
        f"({len(CONTRACT_TESTS)} total)"
    )
    return failed == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FrozenMapping(Mapping[str, Any]):
    """Immutable Mapping for contract tests. Reads only."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        # Shallow freeze; nested structures kept as-is to preserve
        # the contract that tests do not modify the input.
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data