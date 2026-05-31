import asyncio
import json
import os
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
                "text": content,
                "actions": [],
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )


class FakeSessionStore:
    def __init__(self, session_id: str = "session-visible"):
        self._entries = {"session-key": SimpleNamespace(session_id=session_id)}

    def _ensure_loaded(self):
        return None

    def _generate_session_key(self, _source):
        return "session-key"


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


def test_kbtoday_command_renders_attention_cockpit_with_native_adapter(monkeypatch):
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

    result = hook(event=_event("/kb today"), gateway=_authorized_gateway(adapter), session_store=None)
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


def test_kb_command_renders_live_dashboard_without_calling_todo_count_queue(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_dashboard_live": {
                "result": {
                    "surface": "dashboard.live",
                    "summary": {
                        "active_todo_count": 309,
                        "active_run_count": 0,
                        "publication_status": "dirty",
                        "queue_item_count": 309,
                        "readiness_status": "degraded",
                        "recent_run_count": 0,
                    },
                    "sections": [
                        {"id": "now", "title": "Now", "cards": [{"title": "Readiness: degraded"}]},
                        {
                            "id": "queue",
                            "title": "Queue",
                            "cards": [
                                {
                                    "id": "queue:todo1",
                                    "kind": "queue",
                                    "title": "Feature Anthropic Operon",
                                    "detail": "P0 BIO activation TODO",
                                }
                            ],
                        },
                    ],
                    "next_actions": ["Review prioritized queue items through workbench.queue."],
                    "refresh": {"ttl_seconds": 60},
                }
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls[0][0] == "mcp_kb_engine_prod_dashboard_live"
    text = adapter.sent[0]["text"]
    assert "KB Cockpit" in text
    assert "Runtime: degraded" in text
    assert "Publication: dirty" in text
    assert "TODOs 309" in text
    assert "Queue 309" not in text
    assert "TODO Focus" in text
    assert "Review prioritized TODO items" in text


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

    result = hook(event=_event("/kb"), gateway=_authorized_gateway(adapter), session_store=None)
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
    assert "KB Cockpit" in text
    assert "Runtime: ready" in text
    assert "Publication: clean" in text
    assert "Proposals 2" in text
    assert "Review one proposal" in text
    assert "Commands: /kb queue" in text
    assert adapter.sent[0]["actions"] == []


def test_plain_non_kb_commands_are_left_for_system_handlers(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext({})
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    for command in [
        "/queue",
        "/dashboard",
        "/today",
        "/runs",
        "/run",
        "/review",
    ]:
        result = hook(event=_event(command), gateway=_authorized_gateway(adapter), session_store=None)
        assert result is None

    assert ctx.calls == []
    assert adapter.sent == []


def test_legacy_kb_slash_commands_are_supported_but_not_registered(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_queue_summary": {"result": {"total": 0, "items": []}},
            "mcp_kb_engine_prod_attention_cockpit": {"result": {"readiness": {"status": "ready"}}},
            "mcp_kb_engine_prod_run_health": {"result": {"active": [], "recent": []}},
            "mcp_kb_engine_prod_workflow_plan_request": {"result": {"status": "ready"}},
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    for command in ["/kbqueue", "/kbreview 1", "/kbtoday", "/kbruns", "/kbrun sync"]:
        result = hook(event=_event(command), gateway=_authorized_gateway(adapter), session_store=None)
        assert result == {"action": "skip", "reason": "kb_journeys"}

    assert adapter.sent


def test_register_exposes_single_clear_kb_menu_command():
    from plugins import kb_journeys

    class RegisterContext:
        def __init__(self):
            self.commands = {}
            self.hooks = {}

        def register_command(self, name, handler, description=""):
            self.commands[name] = {"handler": handler, "description": description}

        def register_hook(self, name, handler):
            self.hooks[name] = handler

    ctx = RegisterContext()
    kb_journeys.register(ctx)

    assert sorted(ctx.commands) == ["kb"]
    assert ctx.commands["kb"]["description"] == "KB dashboard, queue, status, reasoning, runs, and sync."
    assert ctx.commands["kb"]["handler"]("") == (
        "Use /kb in Telegram. Try: /kb queue, /kb status, /kb reasoning xhigh, /kb run sync."
    )
    assert "pre_gateway_dispatch" in ctx.hooks


def test_kb_root_queue_dashboard_is_text_first(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

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
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb queue"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls == [("mcp_kb_engine_prod_queue_summary", {"scope": "proposals", "limit": 5})]
    assert adapter.sent
    text = adapter.sent[0]["text"]
    assert "KB Queue" in text
    assert "9 pending" in text
    assert "Admit Stanford DAS Lab" in text
    assert "Target: accounts/stanford-das-lab" in text
    assert "Review: /kb queue review 1" in text
    assert "Then preview a listed action, for example: /kb queue reject 1" in text
    assert adapter.sent[0]["actions"] == []
    assert adapter.sent[0]["reply_to"] == "m1"


def test_kbqueue_review_item_can_be_opened_by_text_command(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_queue_summary": {
                "result": {
                    "total": 2,
                    "items": [
                        {
                            "id": "p1",
                            "item_id": "accounts/keio-university",
                            "title": "Keio University",
                            "kind": "proposal_entity",
                            "preview": "Admission: Keio is tied to a healthcare AI PoC.",
                            "raw": {"proposal_ids": ["act_1"]},
                        }
                    ],
                }
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb queue review 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert adapter.sent
    text = adapter.sent[0]["text"]
    assert "Queue Item 1" in text
    assert "Keio University" in text
    assert "Target: accounts/keio-university" in text
    assert "Available actions:" in text
    assert "/kb queue reject 1" in text
    assert adapter.sent[0]["actions"] == []


def test_kbqueue_review_todo_item_shows_todo_native_actions(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_queue_summary": {
                "result": {
                    "total": 1,
                    "items": [
                        {
                            "item_id": "accounts/bankinter",
                            "title": "Bankinter Innovation Foundation",
                            "kind": "proposal_entity",
                            "summary": "Review stale P1->P2 TODO: Respond to Bankinter.",
                            "raw": {"proposal_ids": ["act_todo"]},
                            "safe_actions": [
                                {
                                    "action_id": "todo_queue.complete",
                                    "label": "Complete TODO",
                                    "params": {"proposal_ids": ["act_todo"], "decision": "complete"},
                                },
                                {
                                    "action_id": "todo_queue.keep",
                                    "label": "Keep unchanged",
                                    "params": {"proposal_ids": ["act_todo"], "decision": "keep"},
                                },
                                {
                                    "action_id": "todo_queue.demote",
                                    "label": "Demote priority",
                                    "params": {"proposal_ids": ["act_todo"], "decision": "demote"},
                                },
                                {
                                    "action_id": "todo_queue.archive",
                                    "label": "Archive TODO",
                                    "params": {"proposal_ids": ["act_todo"], "decision": "archive"},
                                },
                            ],
                        }
                    ],
                }
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb queue review 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    text = adapter.sent[0]["text"]
    assert "Available actions:" in text
    assert "Complete TODO: /kb queue complete 1" in text
    assert "Keep unchanged: /kb queue keep 1" in text
    assert "Demote priority: /kb queue demote 1" in text
    assert "Archive TODO: /kb queue archive 1" in text
    assert "/kb queue approve 1" not in text


def test_kbqueue_review_item_renders_descriptor_preview_and_confirm_buttons(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_queue_summary": {
                "result": {
                    "total": 1,
                    "items": [
                        {
                            "item_id": "accounts/mistral",
                            "title": "Mistral",
                            "kind": "proposal_entity",
                            "summary": "Admission: Mistral has Nemotron Coalition licensing coordination.",
                            "raw": {"proposal_ids": ["act_2"]},
                            "safe_actions": [
                                {
                                    "packet_type": "dashboard_action_descriptor",
                                    "schema_version": 2,
                                    "action_id": "review.entity_reject",
                                    "label": "Reject",
                                    "target_kind": "proposal_queue",
                                    "target_ref": "accounts/mistral",
                                    "preview_tool": "queue.decision_preview",
                                    "confirm_tool": "queue.batch_decide_confirmed",
                                    "params": {"proposal_ids": ["act_2"], "decision": "reject"},
                                    "dashboard_owned_write": False,
                                    "requires_canonical_tool": True,
                                    "expected_result": "Preview first, then reject after confirmation.",
                                    "confirmation_copy": "Confirm Reject after reviewing the preview.",
                                }
                            ],
                        }
                    ],
                }
            },
            "mcp_kb_engine_prod_queue_decision_preview": {
                "result": {"status": "preview", "ok": True, "plan": {"summary": "Reject 1 proposal."}}
            },
            "mcp_kb_engine_prod_queue_batch_decide_confirmed": {
                "result": {"status": "applied", "ok": True}
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb queue review 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert adapter.sent[0]["actions"]
    preview_action = adapter.sent[0]["actions"][0]
    assert preview_action.label == "Preview Reject"

    preview_card = preview_action.handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(preview_card):
        preview_card = asyncio.run(preview_card)

    assert "Queue reject preview" in preview_card["text"]
    assert preview_card["actions"][0].label == "Confirm Reject"

    confirm_card = preview_card["actions"][0].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(confirm_card):
        confirm_card = asyncio.run(confirm_card)

    assert "Queue Reject Applied" in confirm_card["text"]
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_queue_decision_preview"
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_queue_batch_decide_confirmed"
    assert ctx.calls[-1][1]["user_confirmation"]["confirmed"] is True
    assert ctx.calls[-1][1]["user_confirmation"]["preview_required"] is True


def test_kbqueue_decision_can_be_previewed_and_confirmed_by_text_command(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_queue_summary": {
                "result": {
                    "total": 1,
                    "items": [
                        {
                            "item_id": "accounts/mistral",
                            "title": "Mistral",
                            "kind": "proposal_entity",
                            "preview": "Admission: Mistral has Nemotron Coalition licensing coordination.",
                            "raw": {"proposal_ids": ["act_2"]},
                        }
                    ],
                }
            },
            "mcp_kb_engine_prod_queue_decision_preview": {
                "result": {"status": "preview", "ok": True, "plan": {"summary": "Reject 1 proposal."}}
            },
            "mcp_kb_engine_prod_queue_batch_decide_confirmed": {
                "result": {
                    "status": "applied",
                    "ok": True,
                    "publication": {"status": "manual"},
                    "git": {
                        "before": {"branch": "main", "changed_count": 0},
                        "after": {"branch": "main", "changed_count": 3, "changes": ["a", "b", "c"]},
                    },
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    preview = hook(event=_event("/kb queue reject 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert preview == {"action": "skip", "reason": "kb_journeys"}
    assert "Queue reject preview" in adapter.sent[0]["text"]
    assert "To apply: /kb queue reject 1 confirm" in adapter.sent[0]["text"]
    assert adapter.sent[0]["actions"] == []

    applied = hook(event=_event("/kb queue reject 1 confirm"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert applied == {"action": "skip", "reason": "kb_journeys"}
    applied_text = adapter.sent[1]["text"]
    assert "Queue Reject Applied" in applied_text
    assert "Mistral" in applied_text
    assert "Target: accounts/mistral" in applied_text
    assert "Proposal ids: act_2" in applied_text
    assert "Git: 3 changed path(s) on main" in applied_text
    assert "{'before':" not in applied_text
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_queue_decision_preview"
    assert ctx.calls[-2][1]["proposal_ids"] == ["act_2"]
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_queue_batch_decide_confirmed"
    assert ctx.calls[-1][1]["user_confirmation"]["confirmed"] is True


def test_kbqueue_bare_reply_uses_visible_iterative_item_state(monkeypatch, tmp_path):
    from plugins import kb_journeys
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    kb_journeys._record_iterative_queue_reply_state(
        session_id="session-visible",
        response_text=(
            "Done — Huang Foundation was treated as P0.\n"
            "Archived proposal ids: act_huang1, act_huang2\n\n"
            "Next item:\n\n"
            "GTC Taipei 2026\n"
            "- Proposal: stale P1→P2 TODO\n"
            "- TODO id: todo_gtc\n"
            "- Proposal id: act_gtc\n\n"
            "Reply with: complete, keep, demote, archive, detail, or skip."
        ),
    )
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_queue_decision_preview": {
                "result": {"status": "preview", "ok": True, "plan": {"summary": "Archive GTC."}}
            },
            "mcp_kb_engine_prod_queue_batch_decide_confirmed": {
                "result": {"status": "applied", "ok": True, "git": {"after": {"changed_count": 2}}}
            },
            "mcp_kb_engine_prod_queue_summary": {
                "result": {
                    "counts": {"proposals": 1},
                    "items": [
                        {
                            "item_id": "forums/wg-agents",
                            "title": "WG Agents",
                            "kind": "proposal_entity",
                            "summary": "Review stale P1→P2 TODO: send the strategy note.",
                            "raw": {"proposal_ids": ["act_next"]},
                            "safe_actions": [
                                {
                                    "label": "Archive TODO",
                                    "params": {"decision": "archive", "proposal_ids": ["act_next"]},
                                }
                            ],
                        }
                    ],
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(
        event=_event("archive"),
        gateway=_authorized_gateway(adapter),
        session_store=FakeSessionStore("session-visible"),
    )
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls[0] == (
        "mcp_kb_engine_prod_queue_decision_preview",
        {
            "proposal_ids": ["act_gtc"],
            "decision": "archive",
            "actor": "telegram:operator",
            "source": "Hermes Telegram iterative queue",
            "note": "Previewed from Telegram iterative queue reply for GTC Taipei 2026",
        },
    )
    assert ctx.calls[1][0] == "mcp_kb_engine_prod_queue_batch_decide_confirmed"
    assert ctx.calls[1][1]["proposal_ids"] == ["act_gtc"]
    assert "act_huang1" not in json.dumps(ctx.calls)
    assert adapter.sent
    assert "GTC Taipei 2026" in adapter.sent[0]["text"]
    assert "WG Agents" in adapter.sent[0]["text"]


def test_kbqueue_bare_reply_records_options_presented_as_pending_action(monkeypatch, tmp_path):
    from plugins import kb_journeys
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    state = kb_journeys._record_iterative_queue_reply_state(
        session_id="session-options",
        response_text=(
            "Proposal 1 — Hitachi\n"
            "- Type: Create Entity\n"
            "- Path: accounts/hitachi\n"
            "- Proposal id: act_hitachi\n"
            "- Rationale: durable strategic relevance.\n\n"
            "Options presented: Approve, Reject, Archive, Details, Feedback."
        ),
    )

    assert state is not None
    assert state["proposal_ids"] == ["act_hitachi"]
    assert "reject" in state["choices"]
    assert "detail" in state["choices"]

    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_queue_decision_preview": {
                "result": {"status": "preview", "ok": True, "plan": {"summary": "Reject Hitachi."}}
            },
            "mcp_kb_engine_prod_queue_batch_decide_confirmed": {
                "result": {"status": "applied", "ok": True, "git": {"after": {"changed_count": 1}}}
            },
            "mcp_kb_engine_prod_queue_summary": {"result": {"counts": {"proposals": 0}, "items": []}},
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(
        event=_event("Reject"),
        gateway=_authorized_gateway(adapter),
        session_store=FakeSessionStore("session-options"),
    )
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls[0][0] == "mcp_kb_engine_prod_queue_decision_preview"
    assert ctx.calls[0][1]["proposal_ids"] == ["act_hitachi"]
    assert ctx.calls[0][1]["decision"] == "reject"
    assert ctx.calls[1][0] == "mcp_kb_engine_prod_queue_batch_decide_confirmed"
    assert ctx.calls[1][1]["proposal_ids"] == ["act_hitachi"]
    assert "Hitachi" in adapter.sent[0]["text"]


def test_kbqueue_pending_action_exposes_scoped_mcp_tools(monkeypatch, tmp_path):
    from plugins import kb_journeys

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    kb_journeys._record_iterative_queue_reply_state(
        session_id="session-posture",
        response_text=(
            "Next item:\n\n"
            "Hitachi\n"
            "- Proposal id: act_hitachi\n\n"
            "Reply: approve, reject, archive, detail."
        ),
    )

    assert kb_journeys.scoped_mcp_tool_allowlist_for_message(
        session_id="session-posture",
        message="Reject",
    ) == {
        "mcp_kb_engine_prod_queue_batch_decide_confirmed",
        "mcp_kb_engine_prod_queue_decide_confirmed",
        "mcp_kb_engine_prod_queue_decisions_confirmed",
        "mcp_kb_engine_prod_queue_decision_preview",
    }
    assert kb_journeys.scoped_mcp_tool_allowlist_for_message(
        session_id="session-posture",
        message="keep me posted",
    ) == set()


def test_kbqueue_todo_complete_decision_uses_queue_decision_contract(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_queue_summary": {
                "result": {
                    "total": 1,
                    "items": [
                        {
                            "item_id": "accounts/bankinter",
                            "title": "Bankinter Innovation Foundation",
                            "kind": "proposal_entity",
                            "summary": "Review stale P1->P2 TODO: Respond to Bankinter.",
                            "raw": {"proposal_ids": ["act_todo"]},
                            "safe_actions": [
                                {
                                    "action_id": "todo_queue.complete",
                                    "label": "Complete TODO",
                                    "params": {"proposal_ids": ["act_todo"], "decision": "complete"},
                                }
                            ],
                        }
                    ],
                }
            },
            "mcp_kb_engine_prod_queue_decision_preview": {
                "result": {
                    "status": "preview",
                    "ok": True,
                    "plan": {"summary": "Complete TODO for 1 TODO-backed proposal(s)."},
                }
            },
            "mcp_kb_engine_prod_queue_batch_decide_confirmed": {
                "result": {"status": "applied", "ok": True, "publication": {"status": "manual"}}
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    preview = hook(event=_event("/kb queue complete 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()
    confirmed = hook(event=_event("/kb queue complete 1 confirm"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert preview == {"action": "skip", "reason": "kb_journeys"}
    assert confirmed == {"action": "skip", "reason": "kb_journeys"}
    assert "Queue complete preview" in adapter.sent[0]["text"]
    assert "To apply: /kb queue complete 1 confirm" in adapter.sent[0]["text"]
    assert "Queue Complete Applied" in adapter.sent[1]["text"]
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_queue_decision_preview"
    assert ctx.calls[-2][1]["decision"] == "complete"
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_queue_batch_decide_confirmed"
    assert ctx.calls[-1][1]["decision"] == "complete"


def test_kbqueue_decision_supports_batch_text_commands_and_legacy_alias(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_queue_summary": {
                "result": {
                    "total": 3,
                    "items": [
                        {
                            "item_id": "accounts/keio-university",
                            "title": "Keio University",
                            "kind": "proposal_entity",
                            "preview": "Admission proposal.",
                            "raw": {"proposal_ids": ["act_1"]},
                        },
                        {
                            "item_id": "accounts/mistral",
                            "title": "Mistral",
                            "kind": "proposal_entity",
                            "preview": "Admission proposal.",
                            "raw": {"proposal_ids": ["act_2", "act_3"]},
                        },
                    ],
                }
            },
            "mcp_kb_engine_prod_queue_decision_preview": {
                "result": {"status": "preview", "ok": True, "plan": {"summary": "Reject 3 proposals."}}
            },
            "mcp_kb_engine_prod_queue_batch_decide_confirmed": {
                "result": {"status": "applied", "ok": True, "publication": {"status": "disabled"}}
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kbqueue reject 1, 2 confirm"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert "Queue Reject Applied" in adapter.sent[0]["text"]
    assert "1. Keio University" in adapter.sent[0]["text"]
    assert "2. Mistral" in adapter.sent[0]["text"]
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_queue_decision_preview"
    assert ctx.calls[-2][1]["proposal_ids"] == ["act_1", "act_2", "act_3"]
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_queue_batch_decide_confirmed"
    assert ctx.calls[-1][1]["proposal_ids"] == ["act_1", "act_2", "act_3"]


def test_queue_preview_failure_does_not_offer_confirm(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

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

    result = hook(event=_event("/kb queue approve 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert "Queue approve preview failed" in adapter.sent[0]["text"]
    assert adapter.sent[0]["actions"] == []


def test_kb_publish_previews_without_committing(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_publication_preview_commit": {
                "result": {
                    "status": "ready",
                    "ok": True,
                    "message": "Publish KB update",
                    "changed_paths": ["accounts/mistral/state.md", "_state/runtime/transactions.jsonl"],
                    "git": {"branch": "main", "head": "abc123", "upstream": "origin/main"},
                }
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb publish"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls == [("mcp_kb_engine_prod_publication_preview_commit", {"message": "Publish KB update"})]
    text = adapter.sent[0]["text"]
    assert "KB Publish Preview" in text
    assert "Changed paths: 2" in text
    assert "accounts/mistral/state.md" in text
    assert "To publish: /kb publish confirm" in text
    assert "No commit or push has been made." in text


def test_kb_publish_confirm_commits_and_pushes_after_fresh_preview(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_publication_preview_commit": {
                "result": {
                    "status": "ready",
                    "ok": True,
                    "message": "Publish KB update",
                    "changed_paths": ["accounts/mistral/state.md"],
                    "git": {"branch": "main", "head": "abc123", "upstream": "origin/main"},
                }
            },
            "mcp_kb_engine_prod_publication_commit_confirmed": {
                "result": {
                    "status": "committed",
                    "ok": True,
                    "publication": {
                        "status": "committed",
                        "changed_paths": ["accounts/mistral/state.md"],
                        "commit": "def456",
                    },
                }
            },
            "mcp_kb_engine_prod_publication_push_confirmed": {
                "result": {
                    "status": "pushed",
                    "ok": True,
                    "publication": {"status": "pushed", "branch": "main", "upstream": "origin/main"},
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb publish confirm"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert [call[0] for call in ctx.calls] == [
        "mcp_kb_engine_prod_publication_preview_commit",
        "mcp_kb_engine_prod_publication_commit_confirmed",
        "mcp_kb_engine_prod_publication_push_confirmed",
    ]
    commit_args = ctx.calls[1][1]
    assert commit_args["expected_git_head"] == "abc123"
    assert commit_args["expected_changed_paths"] == ["accounts/mistral/state.md"]
    assert commit_args["user_confirmation"]["confirmed"] is True
    assert commit_args["push"] is False
    assert ctx.calls[2][1]["user_confirmation"]["confirmed"] is True
    text = adapter.sent[0]["text"]
    assert "KB Published" in text
    assert "Committed: committed" in text
    assert "Pushed: pushed" in text
    assert "accounts/mistral/state.md" in text


def test_kb_publish_confirm_noops_when_preview_has_no_changes(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_publication_preview_commit": {
                "result": {
                    "status": "noop",
                    "ok": True,
                    "message": "Publish KB update",
                    "changed_paths": [],
                    "git": {"branch": "main", "head": "abc123"},
                }
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb publish confirm"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls == [("mcp_kb_engine_prod_publication_preview_commit", {"message": "Publish KB update"})]
    assert "Nothing to publish" in adapter.sent[0]["text"]


def test_run_command_previews_and_starts_with_confirmed_envelope(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

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
            "mcp_kb_engine_prod_run_watch": {
                "result": {
                    "status": "running",
                    "terminal": False,
                    "progress_digest": {
                        "status": "running",
                        "progress": {"current_phase": "Classifying", "current_detail": "entity_admission"},
                        "stage": {
                            "stage_id": "entity_admission",
                            "total": 1,
                            "completed": 0,
                            "failed": 0,
                        },
                        "provider": {"provider": "plugin:openai-compatible", "model": "gpt-5.5"},
                    },
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb run kb sync"), gateway=_authorized_gateway(adapter), session_store=None)
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
    assert "To start: /kb run kb sync confirm" in adapter.sent[0]["text"]
    assert adapter.sent[0]["actions"] == []

    started = hook(event=_event("/kb run kb sync confirm"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert started == {"action": "skip", "reason": "kb_journeys"}
    assert "Workflow start result" in adapter.sent[1]["text"]
    assert "Run:" not in adapter.sent[1]["text"]
    assert "gen-123" not in adapter.sent[1]["text"]
    assert "Initial progress: Classifying" in adapter.sent[1]["text"]
    assert "Stage:" not in adapter.sent[1]["text"]
    assert "Provider:" not in adapter.sent[1]["text"]
    assert "plugin:openai-compatible" not in adapter.sent[1]["text"]
    assert "gpt-5.5" not in adapter.sent[1]["text"]
    assert "watch_until_terminal" not in adapter.sent[1]["text"]
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_workflow_start_confirmed"
    assert ctx.calls[-1] == (
        "mcp_kb_engine_prod_run_watch",
        {"run_id": "gen-123", "timeout_seconds": 0, "poll_interval_seconds": 1, "timeline_limit": 5},
    )
    envelope = ctx.calls[-2][1]["envelope"]
    assert envelope["tool"] == "workflow.start_confirmed"
    assert envelope["plan"]["workflow_id"] == "update_kb"
    assert envelope["user_confirmation"]["confirmed"] is True
    assert envelope["user_confirmation"]["surface"] == "telegram"


def test_meeting_command_hands_live_notes_to_kb_workflow_without_echo(monkeypatch, tmp_path):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    meeting_file = "accounts/allen-institute/meetings/2026-05-05 - Allen.md"
    notes = "Private Telegram live notes: follow up on NeuroBase."
    plan_args = {
        "meeting_file": meeting_file,
        "source_kind": "telegram",
        "source_notes_source": "telegram",
        "source_notes_text": notes,
        "harness_id": "telegram-hermes",
        "harness_session_id": "telegram-session-1",
    }
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_workflow_plan_request": {
                "result": {
                    "status": "confirmation_required",
                    "schema_version": 1,
                    "tool": "workflow.start_confirmed",
                    "workflow": {"workflow_id": "meeting_process", "risk": "write_scoped"},
                    "request": {"args": plan_args, "queue_gate_limit": 0, "force": False},
                    "request_id": "wfreq_meeting_1",
                    "idempotency_key": "workflow:meeting_process:telegram-session-1",
                    "preconditions": [],
                    "provenance": {
                        "actor": "telegram:tester",
                        "source": "Hermes Telegram",
                        "session_id": "telegram-session-1",
                    },
                    "effect_plan": {"effects": [{"id": "workflow.meeting_process"}]},
                    "meeting_artifact_journey": {
                        "artifact_packets": [
                            {
                                "result_contract": "meeting_artifact_packet",
                                "source_kind": "telegram_live_notes",
                                "payload": {"notes_chars": len(notes)},
                            }
                        ]
                    },
                }
            },
            "mcp_kb_engine_prod_workflow_start_confirmed": {
                "result": {
                    "status": "started",
                    "started": True,
                    "run": {"run_id": "gen-meeting-1", "workflow_id": "meeting_process"},
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)
    session_store = FakeSessionStore("telegram-session-1")

    preview = hook(
        event=_event(f"/kb meeting {meeting_file} -- {notes}"),
        gateway=_authorized_gateway(adapter),
        session_store=session_store,
    )
    _drain_scheduled_tasks()

    assert preview == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls[0] == (
        "mcp_kb_engine_prod_workflow_plan_request",
        {
            "workflow_id": "meeting_process",
            "args": plan_args,
            "actor": "telegram:tester",
            "source": "Hermes Telegram",
            "session_id": "telegram-session-1",
        },
    )
    assert "Workflow Preview" in adapter.sent[0]["text"]
    assert "To start: /kb meeting confirm" in adapter.sent[0]["text"]
    assert notes not in adapter.sent[0]["text"]

    started = hook(
        event=_event("/kb meeting confirm"),
        gateway=_authorized_gateway(adapter),
        session_store=session_store,
    )
    _drain_scheduled_tasks()

    assert started == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_workflow_start_confirmed"
    envelope = ctx.calls[-2][1]["envelope"]
    assert envelope["plan"]["workflow_id"] == "meeting_process"
    assert envelope["plan"]["args"]["source_notes_text"] == notes
    assert envelope["plan"]["args"]["source_notes_source"] == "telegram"
    assert envelope["user_confirmation"]["surface"] == "telegram"
    assert notes not in adapter.sent[1]["text"]


def test_runs_command_surfaces_stalled_progress(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_run_health": {
                "result": {
                    "status": "attention_needed",
                    "runs": [
                        {
                            "run_id": "gen-stale",
                            "workflow_id": "update_kb",
                            "status": "stalled_unobserved",
                            "staleness": {"stale": True, "last_trace_age_seconds": 7200},
                            "recommended_next_action": "recover_stalled_run",
                        }
                    ],
                }
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb runs"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert "stalled_unobserved" in adapter.sent[0]["text"]
    assert "stalled 7200s" in adapter.sent[0]["text"]
    assert "recover_stalled_run" in adapter.sent[0]["text"]


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

    card = _render_status(
        {"readiness": {"status": "ready"}},
        "kb_engine_staging",
        {
            "targets": [
                {
                    "role": "primary",
                    "adapter": "plugin:openai-compatible",
                    "model": "gpt-5.5",
                    "reasoning_effort": "low",
                }
            ]
        },
    )

    assert "Lane: staging" in card["text"]
    assert "Environment: staging-dev" in card["text"]
    assert "Hermes reasoning: xhigh" in card["text"]
    assert "KB reasoning: low" in card["text"]
    assert "KB model: gpt-5.5" in card["text"]
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


def test_status_falls_back_to_kb_profile_env_when_provider_status_hidden(monkeypatch, tmp_path):
    from plugins.kb_journeys import _render_status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in (
        "HERMES_KB_MODE",
        "HERMES_ENVIRONMENT",
        "HERMES_KB_WORKSPACE",
        "HERMES_KB_LLM_PROVIDER",
        "HERMES_KB_LLM_MODEL",
        "HERMES_KB_REASONING_EFFORT",
    ):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "HERMES_KB_MODE=prod",
                "HERMES_ENVIRONMENT=production",
                "HERMES_KB_WORKSPACE=/home/abcosta/Knowledge/kb-anthony",
                "HERMES_KB_LLM_PROVIDER=plugin:openai-compatible",
                "HERMES_KB_LLM_MODEL=gpt-5.5",
                "HERMES_KB_REASONING_EFFORT=low",
                "OPENAI_API_KEY=sk-redacted",
            ]
        )
        + "\n"
    )

    card = _render_status(
        {
            "summary": {"publication_status": "clean", "readiness_status": "degraded"},
        },
        "kb_engine_prod",
        None,
    )

    assert "Lane: prod" in card["text"]
    assert "Environment: production" in card["text"]
    assert "Workspace: /home/abcosta/Knowledge/kb-anthony" in card["text"]
    assert "Hermes provider/API:" in card["text"]
    assert "OPENAI" in card["text"]
    assert "KB provider: plugin:openai-compatible" in card["text"]
    assert "KB model: gpt-5.5" in card["text"]
    assert "KB reasoning: low" in card["text"]
    assert "Readiness: degraded" in card["text"]
    assert "Publication: clean" in card["text"]


def test_kb_status_fetches_provider_status_and_shows_both_reasoning(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    monkeypatch.setenv("HERMES_REASONING_EFFORT", "xhigh")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_attention_cockpit": {
                "result": {
                    "readiness": {"status": "degraded"},
                    "publication": {"status": "clean"},
                }
            },
            "mcp_kb_engine_prod_provider_status": {
                "result": {
                    "status": "ready",
                    "targets": [
                        {
                            "role": "primary",
                            "adapter": "plugin:openai-compatible",
                            "model": "gpt-5.5",
                            "reasoning_effort": "low",
                        }
                    ],
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb status"), gateway=_authorized_gateway(adapter), session_store=None)
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
        ),
        ("mcp_kb_engine_prod_provider_status", {}),
    ]
    text = adapter.sent[0]["text"]
    assert "Hermes reasoning: xhigh" in text
    assert "KB reasoning: low" in text
    assert "KB model: gpt-5.5" in text
    assert "KB provider: plugin:openai-compatible" in text


def test_kb_status_uses_profile_env_when_provider_status_hidden(monkeypatch, tmp_path):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    for key in (
        "HERMES_KB_LLM_PROVIDER",
        "HERMES_KB_LLM_MODEL",
        "HERMES_KB_REASONING_EFFORT",
    ):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "HERMES_KB_LLM_PROVIDER=plugin:openai-compatible",
                "HERMES_KB_LLM_MODEL=gpt-5.5",
                "HERMES_KB_REASONING_EFFORT=low",
            ]
        )
        + "\n"
    )
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_attention_cockpit": {
                "result": {
                    "readiness": {"status": "degraded"},
                    "publication": {"status": "clean"},
                }
            },
            "mcp_kb_engine_prod_provider_status": {
                "error": "MCP tool is not visible in profile journey_first_strict: provider.status",
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb status"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    text = adapter.sent[0]["text"]
    assert "KB provider: plugin:openai-compatible" in text
    assert "KB model: gpt-5.5" in text
    assert "KB reasoning: low" in text
    assert "Readiness: degraded" in text
    assert "Publication: clean" in text


def test_kb_status_prefers_live_hermes_session_reasoning(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    monkeypatch.setenv("HERMES_REASONING_EFFORT", "low")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_attention_cockpit": {"result": {"readiness": {"status": "ready"}}},
            "mcp_kb_engine_prod_provider_status": {
                "result": {
                    "targets": [
                        {
                            "role": "primary",
                            "adapter": "plugin:openai-compatible",
                            "model": "gpt-5.5",
                            "reasoning_effort": "low",
                        }
                    ]
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    gateway = SimpleNamespace(
        adapters={Platform.TELEGRAM: adapter},
        _is_user_authorized=lambda _source: True,
        _resolve_session_reasoning_config=lambda **_kwargs: {"enabled": True, "effort": "xhigh"},
    )
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb status"), gateway=gateway, session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    text = adapter.sent[0]["text"]
    assert "Hermes reasoning: xhigh" in text
    assert "KB reasoning: low" in text


def test_kb_reasoning_sets_env_and_reloads_mcp(monkeypatch, tmp_path):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KB_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext({})
    adapter = FakeKbActionsAdapter()
    reload_events = []

    async def _execute_mcp_reload(event):
        reload_events.append(getattr(event, "text", ""))
        return "MCP Reload\nReconnected: kb_engine_prod"

    gateway = SimpleNamespace(
        adapters={Platform.TELEGRAM: adapter},
        _is_user_authorized=lambda _source: True,
        _execute_mcp_reload=_execute_mcp_reload,
    )
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb reasoning xhigh"), gateway=gateway, session_store=None)
    _drain_scheduled_tasks()
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    env_text = (tmp_path / ".env").read_text()
    assert "HERMES_KB_REASONING_EFFORT=xhigh" in env_text
    assert os.environ["HERMES_KB_REASONING_EFFORT"] == "xhigh"
    assert reload_events == ["/kb reasoning xhigh"]
    assert "KB reasoning set to xhigh" in adapter.sent[0]["text"]
    assert "MCP reload started" in adapter.sent[0]["text"]
    assert "MCP Reload" in adapter.sent[1]["text"]
