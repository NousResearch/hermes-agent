import asyncio
import json
from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


class FakeContext:
    def __init__(self, results):
        self.results = dict(results)
        self.calls = []

    def dispatch_tool(self, tool_name, args):
        self.calls.append((tool_name, args))
        result = self.results.get(tool_name, {"error": f"missing {tool_name}"})
        return json.dumps(result)


class FakeAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )


class FakeKbActionsAdapter(FakeAdapter):
    async def send_kb_actions(self, chat_id, text, actions, metadata=None, reply_to=None):
        self.sent.append(
            {
                "chat_id": chat_id,
                "text": text,
                "actions": actions,
                "metadata": metadata,
                "reply_to": reply_to,
            }
        )


def _event(text="/kb"):
    return MessageEvent(
        text=text,
        message_id="m1",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id="user-1",
            user_name="tester",
            chat_type="dm",
            thread_id="topic-1",
        ),
    )


def _gateway(adapter):
    return SimpleNamespace(adapters={Platform.TELEGRAM: adapter})


def _authorized_gateway(adapter, allowed=True):
    return SimpleNamespace(
        adapters={Platform.TELEGRAM: adapter},
        _is_user_authorized=lambda _source: allowed,
    )


def _drain_scheduled_tasks():
    async def _drain():
        await asyncio.sleep(0)

    asyncio.run(_drain())


def test_kb_command_renders_attention_cockpit_with_native_adapter(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_attention_cockpit": {
                "result": {
                    "readiness": {"status": "ready"},
                    "publication": {"status": "published"},
                    "queue": {"count": 3},
                    "todos": {"count": 2},
                    "runs": {
                        "active": [{"name": "sync", "status": "running"}],
                        "recent": [{"name": "publish", "status": "ok"}],
                    },
                    "next_actions": ["Review 3 proposals"],
                }
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/today"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls == [
        (
            "mcp_kb_engine_prod_attention_cockpit",
            {
                "attention_limit": 5,
                "include_publication": True,
                "include_readiness": True,
                "run_limit": 3,
            },
        )
    ]
    assert adapter.sent
    text = adapter.sent[0]["text"]
    assert "KB Today" in text
    assert "ready" in text
    assert "published" in text
    assert "Review 3 proposals" in text
    assert adapter.sent[0]["actions"] == []
    assert adapter.sent[0]["reply_to"] == "m1"


def test_kb_command_renders_live_attention_summary_shape(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_attention_cockpit": {
                "result": {
                    "summary": {
                        "active_todo_count": 309,
                        "publication_status": "dirty",
                        "queue_item_count": 309,
                        "readiness_status": "degraded",
                        "recent_run_count": 5,
                    },
                    "sections": {
                        "publication": {
                            "summary": {"ok": True, "status": "dirty"},
                            "payload": {"git": {"changed_count": 106}},
                        },
                        "readiness": {"summary": {"ready": True, "status": "degraded"}},
                    },
                    "next_actions": ["Review prioritized queue items through workbench.queue."],
                }
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    text = adapter.sent[0]["text"]
    assert "Readiness: degraded" in text
    assert "Publication: dirty" in text
    assert "Queue: 309" in text


def test_dashboard_command_prefers_live_dashboard_packet(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_dashboard_live": {
                "result": {
                    "surface": "dashboard.live",
                    "summary": {
                        "active_run_count": 1,
                        "active_todo_count": 4,
                        "publication_status": "clean",
                        "queue_item_count": 2,
                        "readiness_status": "ready",
                    },
                    "sections": [
                        {
                            "id": "now",
                            "title": "Now",
                            "cards": [
                                {"id": "system:readiness", "title": "Readiness: ready", "detail": ""},
                                {"id": "system:runs", "title": "1 active run(s)", "detail": "sync"},
                            ],
                        },
                        {
                            "id": "queue",
                            "title": "Queue",
                            "cards": [{"id": "queue:1", "title": "Review one proposal", "detail": "low risk"}],
                        },
                    ],
                    "refresh": {"ttl_seconds": 60},
                }
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/dashboard"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls == [
        (
            "mcp_kb_engine_prod_dashboard_live",
            {
                "limit": 5,
                "include_feedback": True,
                "include_publication": True,
                "include_readiness": True,
            },
        )
    ]
    text = adapter.sent[0]["text"]
    assert "KB Dashboard" in text
    assert "Readiness: ready" in text
    assert "Publication: clean" in text
    assert "Queue 2" in text
    assert "Review one proposal" in text
    assert [action.label for action in adapter.sent[0]["actions"]] == ["Refresh", "Queue", "Runs", "Status"]


def test_plain_queue_command_is_left_for_system_queue(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext({})
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/queue"), gateway=_authorized_gateway(adapter), session_store=None)

    assert result is None
    assert ctx.calls == []
    assert adapter.sent == []


def test_kbqueue_dashboard_reviews_one_item_before_preview_and_confirm(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook
    from tools.kb_callback_registry import KbCallbackContext

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_queue_summary": {
                "result": {
                    "total": 9,
                    "items": [
                        {
                            "id": "p1",
                            "item_id": "accounts/stanford-das-lab",
                            "title": "Admit Stanford DAS Lab",
                            "kind": "proposal_entity",
                            "preview": "Would update existing entity.",
                            "raw": {"proposal_ids": ["act_1", "act_2"]},
                        }
                    ],
                }
            },
            "mcp_kb_engine_prod_queue_decision_preview": {
                "result": {"status": "preview", "ok": True, "plan": {"summary": "Reject 2 proposals."}}
            },
            "mcp_kb_engine_prod_queue_batch_decide_confirmed": {
                "result": {"status": "applied", "ok": True, "publication": {"status": "manual"}}
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kbqueue"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls == [("mcp_kb_engine_prod_queue_summary", {"scope": "proposals", "limit": 5})]
    assert adapter.sent
    text = adapter.sent[0]["text"]
    assert "KB Queue" in text
    assert "9" in text
    assert "Admit Stanford DAS Lab" in text
    assert "Tap Review N" in text
    assert [action.label for action in adapter.sent[0]["actions"]] == ["Review 1"]
    assert adapter.sent[0]["reply_to"] == "m1"

    detail = asyncio.run(
        adapter.sent[0]["actions"][0].handler(
            KbCallbackContext(
                callback_id="cb_detail",
                action_id="queue.item.1",
                actor_id="777",
                actor_name="Ada",
            )
        )
    )
    assert "Queue Item 1" in detail["text"]
    assert "Admit Stanford DAS Lab" in detail["text"]
    assert "Would update existing entity." in detail["text"]
    assert "Decision buttons apply only this item." in detail["text"]
    assert [action.label for action in detail["actions"]] == [
        "Preview approve",
        "Preview reject",
        "Preview archive",
    ]

    preview = asyncio.run(
        detail["actions"][1].handler(
            KbCallbackContext(
                callback_id="cb_preview",
                action_id="queue.preview.reject",
                actor_id="777",
                actor_name="Ada",
            )
        )
    )
    assert "Queue reject preview" in preview["text"]
    assert "Item: Admit Stanford DAS Lab" in preview["text"]
    assert preview["actions"][0].label == "Confirm reject"

    confirmed = asyncio.run(
        preview["actions"][0].handler(
            KbCallbackContext(
                callback_id="cb_confirm",
                action_id="queue.confirm.reject",
                actor_id="777",
                actor_name="Ada",
            )
        )
    )
    assert "Queue reject applied" in confirmed
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_queue_decision_preview"
    assert ctx.calls[-2][1]["proposal_ids"] == ["act_1", "act_2"]
    assert ctx.calls[-2][1]["decision"] == "reject"
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_queue_batch_decide_confirmed"
    assert ctx.calls[-1][1]["user_confirmation"]["confirmed"] is True


def test_queue_preview_failure_does_not_offer_confirm(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook
    from tools.kb_callback_registry import KbCallbackContext

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_queue_summary": {
                "result": {
                    "total": 1,
                    "items": [
                        {
                            "item_id": "accounts/acme",
                            "title": "Risky proposal",
                            "raw": {"proposal_ids": ["act_fail"]},
                        }
                    ],
                }
            },
            "mcp_kb_engine_prod_queue_decision_preview": {
                "result": {"error": "preview precondition failed"}
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kbqueue"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    detail = asyncio.run(
        adapter.sent[0]["actions"][0].handler(
            KbCallbackContext(
                callback_id="cb_detail",
                action_id="queue.item.1",
                actor_id="777",
                actor_name="Ada",
            )
        )
    )
    preview = asyncio.run(
        detail["actions"][0].handler(
            KbCallbackContext(
                callback_id="cb_preview",
                action_id="queue.preview.approve",
                actor_id="777",
                actor_name="Ada",
            )
        )
    )
    assert "Queue approve preview failed" in preview["text"]
    assert preview["actions"] == []


def test_run_command_previews_and_starts_with_confirmed_envelope(monkeypatch):
    from plugins import kb_journeys
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook
    from tools.kb_callback_registry import KbCallbackContext

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_workflow_plan_request": {
                "result": {
                    "status": "confirmation_required",
                    "schema_version": 1,
                    "tool": "workflow.start_confirmed",
                    "workflow": {"workflow_id": "update_kb", "risk": "write_broad"},
                    "request": {"args": {}, "queue_gate_limit": 0, "force": False},
                    "request_id": "wfreq_1",
                    "idempotency_key": "workflow:update_kb:test",
                    "preconditions": [],
                    "provenance": {
                        "actor": "telegram:operator",
                        "source": "Hermes Telegram",
                        "session_id": "telegram-kb-1",
                    },
                    "effect_plan": {"effects": [{"id": "workflow.sync.fetch_sources"}]},
                    "followthrough_contract": {
                        "watch_tool": "run.watch",
                        "terminal_summary_tool": "run.summary",
                    },
                }
            },
            "mcp_kb_engine_prod_workflow_start_confirmed": {
                "result": {
                    "status": "started",
                    "started": True,
                    "run": {"run_id": "gen-123", "workflow_id": "update_kb"},
                    "followthrough_contract": {"recommended_next_action": "watch_until_terminal"},
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/run kb sync"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls[0] == (
        "mcp_kb_engine_prod_workflow_plan_request",
        {
            "workflow_id": "update_kb",
            "intent": "kb sync",
            "actor": "telegram:operator",
            "source": "Hermes Telegram",
            "session_id": ctx.calls[0][1]["session_id"],
        },
    )
    assert "Workflow Preview" in adapter.sent[0]["text"]
    assert adapter.sent[0]["actions"][0].label == "Start update_kb"

    monkeypatch.setattr(kb_journeys, "_run_delivery", lambda coro: coro.close())
    started = asyncio.run(
        adapter.sent[0]["actions"][0].handler(
            KbCallbackContext(
                callback_id="cb_start",
                action_id="workflow.start.update_kb",
                chat_id="chat-1",
                thread_id="topic-1",
                actor_id="777",
                actor_name="Ada",
            )
        )
    )

    assert "Workflow start result" in started
    assert "gen-123" in started
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_workflow_start_confirmed"
    envelope = ctx.calls[-1][1]["envelope"]
    assert envelope["tool"] == "workflow.start_confirmed"
    assert envelope["plan"]["workflow_id"] == "update_kb"
    assert envelope["user_confirmation"]["confirmed"] is True
    assert envelope["user_confirmation"]["surface"] == "telegram"


def test_non_telegram_or_unknown_command_is_ignored(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    ctx = FakeContext({})
    hook = build_pre_gateway_dispatch_hook(ctx)

    event = _event("/kb")
    event.source.platform = Platform.WHATSAPP
    assert hook(event=event, gateway=_gateway(FakeAdapter()), session_store=None) is None
    assert hook(event=_event("/unknown"), gateway=_gateway(FakeAdapter()), session_store=None) is None
    assert ctx.calls == []


def test_telegram_command_respects_gateway_authorization(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_attention_cockpit": {
                "result": {"readiness": {"status": "ready"}}
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    assert hook(event=_event("/kb"), gateway=_authorized_gateway(adapter, allowed=False), session_store=None) is None
    _drain_scheduled_tasks()

    assert adapter.sent == []
    assert ctx.calls == []


def test_status_reports_noc_lane_and_reasoning(monkeypatch):
    from plugins.kb_journeys import _render_status

    monkeypatch.setenv("HERMES_KB_MODE", "staging")
    monkeypatch.setenv("HERMES_ENVIRONMENT", "staging-dev")
    monkeypatch.setenv("HERMES_KB_WORKSPACE", "/home/abcosta/Knowledge/kb-anthony-staging")
    monkeypatch.setenv("HERMES_MODEL_API_MODE", "responses")
    monkeypatch.setenv("HERMES_REASONING_EFFORT", "xhigh")

    card = _render_status({"readiness": {"status": "ready"}}, "kb_engine_staging")

    assert "Lane: staging" in card["text"]
    assert "Environment: staging-dev" in card["text"]
    assert "Reasoning: xhigh" in card["text"]
    assert "responses" in card["text"]


def test_status_reports_live_attention_summary_shape(monkeypatch):
    from plugins.kb_journeys import _render_status

    monkeypatch.setenv("HERMES_KB_MODE", "prod")
    monkeypatch.setenv("HERMES_ENVIRONMENT", "production")
    monkeypatch.setenv("HERMES_KB_WORKSPACE", "/home/abcosta/Knowledge/kb-anthony")

    card = _render_status(
        {
            "summary": {"publication_status": "dirty", "readiness_status": "degraded"},
            "sections": {
                "publication": {"summary": {"status": "dirty"}},
                "readiness": {"summary": {"status": "degraded"}},
            },
        },
        "kb_engine_prod",
    )

    assert "Readiness: degraded" in card["text"]
    assert "Publication: dirty" in card["text"]
