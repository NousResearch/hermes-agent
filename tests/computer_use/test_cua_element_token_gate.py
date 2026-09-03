"""Regression tests for the element_token attach gate.

``_maybe_attach_element_token`` decides whether to send the opaque
``element_token`` alongside ``element_index`` on a cua-driver action.

Getting that gate wrong is not a soft degradation. cua-driver 0.21 REFUSES a
bare ``element_index``::

    click: bare element_index is not accepted; pass element_token,
    or snapshot_id together with element_index

so a gate that fails closed breaks every element-targeted click and leaves
agents with nothing but blind pixel coordinates.

The original gate consulted only the trycua/cua#1961 capability vocabulary.
cua-driver 0.21 stopped publishing per-tool capability sets — every tool
reports an empty set — while still accepting ``element_token`` in its input
schema, so that check alone silently disabled element targeting on 0.21.x.
The schema (``tools/list``) is the authoritative signal; the capability token
is kept as a fallback for drivers that shipped the vocabulary.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tools.computer_use.cua_backend import CuaDriverBackend


def _backend(*, schema_ok: bool, capability_ok: bool) -> CuaDriverBackend:
    """A backend whose session advertises support via one channel, both, or neither."""
    backend = CuaDriverBackend.__new__(CuaDriverBackend)
    backend._snapshot_tokens = {7: "s00000001:7"}
    session = MagicMock()
    session.supports_input_property.return_value = schema_ok
    session.supports_capability.return_value = capability_ok
    backend._session = session
    return backend


def test_schema_alone_is_enough() -> None:
    """0.21.x: schema accepts element_token, capability vocabulary is gone."""
    backend = _backend(schema_ok=True, capability_ok=False)
    args = {"element_index": 7}
    backend._maybe_attach_element_token("click", args)
    assert args["element_token"] == "s00000001:7"


def test_capability_alone_is_enough() -> None:
    """Older drivers: capability advertised, schema introspection unavailable."""
    backend = _backend(schema_ok=False, capability_ok=True)
    args = {"element_index": 7}
    backend._maybe_attach_element_token("click", args)
    assert args["element_token"] == "s00000001:7"


def test_neither_signal_leaves_args_untouched() -> None:
    """Pre-#1961 drivers reject unknown properties, so send nothing."""
    backend = _backend(schema_ok=False, capability_ok=False)
    args = {"element_index": 7}
    backend._maybe_attach_element_token("click", args)
    assert "element_token" not in args


def test_unknown_index_is_not_guessed() -> None:
    """An index with no cached token must not borrow another element's."""
    backend = _backend(schema_ok=True, capability_ok=True)
    args = {"element_index": 999}
    backend._maybe_attach_element_token("click", args)
    assert "element_token" not in args


def test_non_element_calls_are_ignored() -> None:
    """Coordinate-based actions carry no element_index and must be left alone."""
    backend = _backend(schema_ok=True, capability_ok=True)
    args = {"x": 10, "y": 20}
    backend._maybe_attach_element_token("click", args)
    assert "element_token" not in args


def test_schema_is_consulted_before_capability() -> None:
    """The schema check must be able to satisfy the gate on its own.

    Guards against a regression to capability-first evaluation, which is what
    failed closed on 0.21.x.
    """
    backend = _backend(schema_ok=True, capability_ok=False)
    args = {"element_index": 7}
    backend._maybe_attach_element_token("click", args)

    backend._session.supports_input_property.assert_called_once_with(
        "click", "element_token"
    )
    assert args["element_token"] == "s00000001:7"
