"""Exact approval-request correlation must survive every button transport."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ADAPTERS = [
    "gateway/platforms/whatsapp_cloud.py",
    "gateway/platforms/qqbot/adapter.py",
    "gateway/relay/adapter.py",
    "plugins/platforms/telegram/adapter.py",
    "plugins/platforms/slack/adapter.py",
    "plugins/platforms/discord/adapter.py",
    "plugins/platforms/feishu/adapter.py",
    "plugins/platforms/teams/adapter.py",
    "plugins/platforms/matrix/adapter.py",
]


def _tree(relative: str) -> ast.AST:
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def _calls(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == name:
            yield node
        elif isinstance(fn, ast.Attribute) and fn.attr == name:
            yield node


def test_gateway_run_forwards_exact_approval_request_id_to_adapter():
    calls = list(_calls(_tree("gateway/run.py"), "send_exec_approval"))
    assert calls
    assert any(
        any(
            kw.arg == "request_id"
            and isinstance(kw.value, ast.Call)
            and isinstance(kw.value.func, ast.Attribute)
            and kw.value.func.attr == "get"
            for kw in call.keywords
        )
        for call in calls
    )


def test_api_run_approval_forwards_client_request_id_to_resolver():
    calls = list(_calls(_tree("gateway/platforms/api_server.py"), "resolve_gateway_approval"))
    assert calls
    assert any(any(kw.arg == "request_id" for kw in call.keywords) for call in calls)


def test_button_adapters_accept_and_resolve_exact_request_id():
    failures = []
    for relative in ADAPTERS:
        tree = _tree(relative)
        send_defs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "send_exec_approval"
        ]
        if not send_defs or not all(
            "request_id" in [arg.arg for arg in node.args.args] for node in send_defs
        ):
            failures.append(f"{relative}: send_exec_approval lacks request_id")

        resolver_calls = list(_calls(tree, "resolve_gateway_approval"))
        if not resolver_calls or not all(
            any(kw.arg == "request_id" for kw in call.keywords)
            for call in resolver_calls
        ):
            failures.append(f"{relative}: resolver call lacks request_id")

    assert failures == []
