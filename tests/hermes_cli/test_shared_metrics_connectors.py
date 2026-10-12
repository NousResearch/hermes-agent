"""Connector work in the shared metrics: per-entry tool counting, gateway error classes, the
search-to-execute chain, and the tool_search / tool_batch / connector_chain rows."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli import lifecycle
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_cli.observability.shared_metrics_connectors import tool_batch_fields

from tests.hermes_cli.test_relay_shared_metrics import _schema_validator
from tests.hermes_cli import test_relay_shared_metrics_runtime as runtime_tests

_RESOURCE = {"architecture": "x86_64", "hermes_version": "0.0.0", "install_method": "git", "os_family": "linux"}
_ROUTE = {"model": "m", "provider": "openrouter"}
@pytest.fixture
def fake_relay(tmp_path, monkeypatch):
    """The runtime tests' fake Relay, so lifecycle hooks drive the real metrics runtime."""
    yield from runtime_tests.direct_runtime.__wrapped__(tmp_path, monkeypatch)


_BASE = {"session_id": "s1", "task_id": "t1", "turn_id": "turn-1", "api_request_id": "r1", "platform": "cli"}


@pytest.mark.parametrize(("code", "family", "expected"), [
    ("CONNECTION_REQUIRED", "product", "connection_required"),
    ("CONNECTION_EXPIRED", "product", "connection_required"),
    ("TOOL_NOT_ALLOWED", "product", "not_allowed"),
    ("ALIAS_REQUIRED", "product", "alias"),
    ("ALIAS_UNKNOWN", "product", "alias"),
    ("INVALID_ARGUMENTS", "product", "invalid_arguments"),
    ("PROVIDER_TIMEOUT", "backend", "connector_backend"),
    ("GATEWAY_ERROR", "backend", "connector_backend"),
    ("SOMETHING_NEW", "backend", "connector_backend"),  # an unlisted code falls back to its family
    ("SOMETHING_NEW", "product", "tool_error"),
    ("TOOL_NOT_FOUND", "product", "tool_error"),
])
def test_a_failed_connector_entry_is_classed_by_its_gateway_code(code, family, expected):
    result = json.dumps({"error": {"code": code, "family": family, "message": "private text"}})
    event = {"tool_name": "connectors__gmail__SEND_EMAIL", "status": "error", "error_type": "tool_error",
             "result": result}
    assert contract.tool_error_class(event) == expected
    assert expected in contract.TOOL_ERROR_CLASSES


def test_every_classed_code_is_a_gateway_execute_code():
    from tools.connectors.gateway.wire import EXECUTE_ERROR_CODES

    assert set(contract._CONNECTOR_CODE_CLASSES) <= set(EXECUTE_ERROR_CODES)
    assert all(EXECUTE_ERROR_CODES[code][0] == "product" for code in contract._CONNECTOR_CODE_CLASSES)


def test_connector_and_bridge_calls_are_named_without_a_slug():
    connector = {"tool_name": "connectors__gmail__SEND_EMAIL", "toolset": "other"}
    assert contract.tool_metric_name(connector) == "connector"
    assert contract.tool_category(connector) == "connector"
    for bridge in ("tool_search", "tool_describe", "tool_call"):
        assert contract.tool_metric_name({"tool_name": bridge}) == bridge
    # A non-connector tool that reports an error code keeps the generic class.
    plain = {"tool_name": "mcp_github_x", "status": "error", "error_type": "tool_error",
             "result": json.dumps({"error": {"code": "TOOL_NOT_ALLOWED"}})}
    assert contract.tool_error_class(plain) == "tool_error"


def _tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def test_batch_row_counts_logical_calls_and_the_connector_share():
    batch = _tool_call("tool_call", {"calls": [
        {"name": "connectors__gmail__SEND_EMAIL", "arguments": {}},
        {"name": "connectors__slack__POST", "arguments": {}}]})
    assert tool_batch_fields(_ROUTE, [batch]) == {**_ROUTE, "connector_share": "all", "size_bucket": "2"}
    mixed = tool_batch_fields(_ROUTE, [batch, _tool_call("read_file", {"path": "x"})])
    assert mixed == {**_ROUTE, "connector_share": "some", "size_bucket": "3_to_5"}
    assert tool_batch_fields(_ROUTE, [_tool_call("terminal", {})])["connector_share"] == "none"
    assert tool_batch_fields(_ROUTE, []) is None


def _connector_batch_hooks(entries, *, wrapper_result="ok"):
    """The hook sequence a connector batch produces: pre/post per entry under ``<id>#<position>``,
    then the wrapper's post under the bare id."""
    call = {**_BASE, "tool_call_id": "tc1"}
    for position, (name, result) in enumerate(entries):
        entry = {**call, "tool_call_id": f"tc1#{position}"}
        lifecycle.invoke_hook("pre_tool_call", **entry, tool_name=name, args={})
        failed = isinstance(result, dict)
        lifecycle.invoke_hook(
            "post_tool_call", **entry, tool_name=name, args={}, result=json.dumps(result if failed else {"ok": 1}),
            status="error" if failed else "ok", error_type="tool_error" if failed else None)
    lifecycle.invoke_hook(
        "post_tool_call", **call, tool_name="tool_call", status=wrapper_result, result="{}",
        args={"calls": [{"name": name, "arguments": {}} for name, _ in entries]})


def _end_turn():
    lifecycle.invoke_hook("on_session_end", **_BASE, completed=True, failed=False, interrupted=False,
                          turn_exit_reason="text_response(stop)")
    lifecycle.finalize_session(session_id="s1")


def test_a_connector_batch_counts_each_entry_once_and_not_the_wrapper(fake_relay):
    lifecycle.invoke_hook("pre_llm_call", **_BASE)
    _connector_batch_hooks([
        ("connectors__gmail__SEND_EMAIL", None),
        ("connectors__slack__POST", {"error": {"code": "CONNECTION_REQUIRED", "family": "product"}}),
    ])
    _end_turn()

    ends = [event[2] for event in fake_relay.events if event[0] == "tool.call_end"]
    assert [(end["tool_name"], end["tool_category"], end["outcome"], end["error_class"]) for end in ends] == [
        ("connector", "connector", "success", "none"),
        ("connector", "connector", "failed", "connection_required"),
    ]
    [task_end] = [event[2]["output"] for event in fake_relay.events
                  if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"]
    assert task_end["tool_call_count_bucket"] == "2"


def test_a_batch_rejected_before_any_entry_still_counts_as_tool_call(fake_relay):
    lifecycle.invoke_hook("pre_llm_call", **_BASE)
    lifecycle.invoke_hook(
        "post_tool_call", **_BASE, tool_call_id="tc1", tool_name="tool_call", status="error",
        error_type="tool_error", result=json.dumps({"error": "too many calls"}),
        args={"calls": [{"name": "connectors__gmail__SEND_EMAIL", "arguments": {}}]})
    _end_turn()
    [end] = [event[2] for event in fake_relay.events if event[0] == "tool.call_end"]
    assert (end["tool_name"], end["outcome"]) == ("tool_call", "failed")


def test_chain_row_counts_searches_since_the_last_execute_and_whether_the_tool_was_described(
    fake_relay,
):
    lifecycle.invoke_hook("pre_llm_call", **_BASE)
    for n, (name, args) in enumerate([
        ("tool_search", {"queries": ["send mail"]}),
        ("tool_search", {"queries": ["gmail send"]}),
        ("tool_describe", {"names": ["connectors__gmail__SEND_EMAIL"]}),
        ("connectors__gmail__SEND_EMAIL", {}),
        ("connectors__slack__POST", {}),
    ]):
        call = {**_BASE, "tool_call_id": f"c{n}", "tool_name": name, "args": args}
        lifecycle.invoke_hook("pre_tool_call", **call)
        lifecycle.invoke_hook("post_tool_call", **call, result="{}", status="ok")
    _end_turn()
    data = [event[2]["data"] for event in fake_relay.events
            if event[0] == "scope.event" and event[1] == contract.CONNECTOR_CHAIN_MARK]
    assert data == [
        {"described_first": "yes", "outcome": "success", "searches_before_bucket": "2"},
        {"described_first": "no", "outcome": "success", "searches_before_bucket": "0"},
    ]


def test_a_plugin_approval_on_a_connector_entry_is_attributed_to_that_entry(fake_relay):
    lifecycle.invoke_hook("pre_llm_call", **_BASE)
    entry = {**_BASE, "tool_call_id": "tc1#0", "tool_name": "connectors__gmail__SEND_EMAIL", "args": {}}
    lifecycle.invoke_hook("pre_tool_call", **entry)
    lifecycle.invoke_hook("post_approval_response", **_BASE, tool_call_id="tc1#0", choice="once")
    lifecycle.invoke_hook("post_tool_call", **entry, result="{}", status="ok")
    _end_turn()
    [approval] = [event[2]["data"] for event in fake_relay.events
                  if event[0] == "scope.event" and event[1] == contract.TOOL_APPROVAL_MARK]
    assert approval == {"attribution": "tool_call", "outcome": "approved"}
    [end] = [event[2] for event in fake_relay.events if event[0] == "tool.call_end"]
    assert end["approval_outcome"] == "approved"


def test_the_chain_survives_a_compression_rotation(fake_relay):
    from hermes_cli.observability import relay_shared_metrics

    lifecycle.invoke_hook("pre_llm_call", **_BASE)
    for n, (name, args) in enumerate([
        ("tool_search", {"queries": ["mail"]}),
        ("tool_describe", {"names": " connectors__gmail__SEND_EMAIL "}),  # bare string, padded
    ]):
        call = {**_BASE, "tool_call_id": f"c{n}", "tool_name": name, "args": args}
        lifecycle.invoke_hook("pre_tool_call", **call)
        lifecycle.invoke_hook("post_tool_call", **call, result="{}", status="ok")
    relay_shared_metrics.rotate_segment("s1", "s2")
    rotated = {**_BASE, "session_id": "s2", "task_id": "t2", "turn_id": "turn-2"}
    lifecycle.invoke_hook("pre_llm_call", **rotated)
    call = {**rotated, "tool_call_id": "c9", "tool_name": "connectors__gmail__SEND_EMAIL", "args": {}}
    lifecycle.invoke_hook("pre_tool_call", **call)
    lifecycle.invoke_hook("post_tool_call", **call, result="{}", status="ok")
    [chain] = [event[2]["data"] for event in fake_relay.events
               if event[0] == "scope.event" and event[1] == contract.CONNECTOR_CHAIN_MARK]
    assert chain == {"described_first": "yes", "outcome": "success", "searches_before_bucket": "1"}


def test_entries_report_their_own_post_even_under_the_executor_suppression(monkeypatch):
    """The sequential executor suppresses the inner post_tool_call for the call it owns; connector
    entries are their own calls and must still report, or each entry's lifecycle never closes."""
    import model_tools
    from hermes_cli import lifecycle as lc
    from tools.connectors.gateway import bridge, config
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    invalidate_check_fn_cache()
    gateway_ids = []

    def fake_remote(plans, dispatch_id, **_kw):
        gateway_ids.append(dispatch_id)
        return [{"index": plan.position, "name": plan.name, "response": {"ok": 1}} for plan in plans]

    monkeypatch.setattr(bridge, "run_remote", fake_remote)
    posts = []
    monkeypatch.setattr(lc, "has_hook", lambda name: True)
    monkeypatch.setattr(lc, "invoke_hook", lambda name, **kw: posts.append((kw["tool_name"], kw["tool_call_id"]))
                        or [] if name == "post_tool_call" else [])
    calls = [{"name": "connectors__gmail__SEND_EMAIL", "arguments": {}},
             {"name": "connectors__slack__POST", "arguments": {}}]
    with model_tools.suppress_post_tool_call_hook():
        model_tools.handle_function_call("tool_call", {"calls": calls}, tool_call_id="tc1",
                                         enabled_toolsets=["connections"])
    # Each entry reports under its own id, so a trace observer never folds entries together.
    assert posts == [("connectors__gmail__SEND_EMAIL", "tc1#0"), ("connectors__slack__POST", "tc1#1")]
    assert gateway_ids == ["tc1#0", "tc1#1"]


def test_a_package_with_each_new_row_validates_against_the_v4_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    store = SharedMetricsStore()
    store.record_counter(contract.TOOL_SEARCH_METRIC,
                         {"hits_bucket": "3_to_5", "outcome": "hits", "scope": "both"}, _RESOURCE)
    store.record_counter(contract.TOOL_BATCH_METRIC,
                         {**_ROUTE, "connector_share": "some", "size_bucket": "2"}, _RESOURCE)
    store.record_counter(contract.CONNECTOR_CHAIN_METRIC,
                         {"described_first": "no", "outcome": "failed", "searches_before_bucket": "1"}, _RESOURCE)
    store.record_counter(contract.TOOL_USAGE_METRIC,
                         {"error_class": "connection_required", "outcome": "failed", "tool_name": "connector"}, _RESOURCE)
    store.record_counter(contract.TOOL_CALL_METRIC,
                         {"approval_outcome": "not_required", "outcome": "success", "tool_category": "connector"},
                         _RESOURCE)
    [path] = store.create_and_export_package()
    package = json.loads(path.read_text(encoding="utf-8"))
    _schema_validator().validate(package)
    assert {metric["name"] for metric in package["metrics"]} >= {
        contract.TOOL_SEARCH_METRIC, contract.TOOL_BATCH_METRIC, contract.CONNECTOR_CHAIN_METRIC}
