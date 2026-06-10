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


class SequencedFakeContext:
    def __init__(self, results):
        self.results = {key: list(value) for key, value in results.items()}
        self.calls = []

    def dispatch_tool(self, tool_name, args):
        self.calls.append((tool_name, args))
        values = self.results.get(tool_name)
        if values:
            result = values.pop(0)
        else:
            result = {"error": f"missing {tool_name}"}
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


class FailingKbActionsAdapter(FakeAdapter):
    async def send_kb_actions(self, chat_id, text, actions, metadata=None, reply_to=None):
        self.sent.append(
            {
                "chat_id": chat_id,
                "text": text,
                "actions": actions,
                "metadata": metadata,
                "reply_to": reply_to,
                "failed_native_card": True,
            }
        )
        return SimpleNamespace(success=False, error="button rendering unavailable")


def _event(text="/kb", *, user_id="user-1", user_name="tester"):
    return MessageEvent(
        text=text,
        message_id="m1",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            user_id=user_id,
            user_name=user_name,
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


def _advisory_guidance(summary="Use advisory guidance to reason about this KB action."):
    return {
        "packet_type": "kb_advisory_guidance",
        "schema_version": 1,
        "mode": "advisory_only",
        "authority": "no_mutation_authority",
        "llm_prompt": "kb.review_guidance",
        "llm_invocation": "explicit_user_request_only",
        "mutates_state": False,
        "requires_preview_before_write": True,
        "summary": summary,
        "recommended_sequence": [
            "Read the canonical KB context and evidence first.",
            "Preview with the canonical preview tool before confirmation.",
        ],
    }


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
    assert "KB" in text
    assert "kb status: runtime degraded · publication dirty" in text
    assert "TODOs 309" in text
    assert "Queue 309" not in text
    assert "Attention Review" in text
    assert "Review prioritized attention items" in text
    assert "Commands: /kb status · /kb sync · /kb review" in text


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
    assert "KB" in text
    assert "kb status: runtime ready · publication clean" in text
    assert "Proposals 2" in text
    assert "Review one proposal" in text
    assert "Commands: /kb status · /kb sync · /kb review" in text
    assert adapter.sent[0]["actions"] == []


def test_dashboard_situation_descriptor_renders_readonly_action_button(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_dashboard_live": {
                "result": {
                    "surface": "dashboard.live",
                    "summary": {
                        "active_run_count": 0,
                        "active_todo_count": 1,
                        "publication_status": "clean",
                        "readiness_status": "ready",
                    },
                    "sections": [
                        {
                            "id": "situations",
                            "title": "Situations",
                            "cards": [
                                {
                                    "id": "situation:acme-launch",
                                    "kind": "situation",
                                    "title": "Acme Launch Decision",
                                    "detail": "Needs next-step guidance.",
                                    "action_descriptors": [
                                        {
                                            "packet_type": "dashboard_action_descriptor",
                                            "schema_version": 2,
                                            "action_id": "open_situation",
                                            "label": "Open situation",
                                            "method": "object.context",
                                            "mutation": "read_only",
                                            "target_kind": "situation",
                                            "target_ref": "situations/2026-05-acme-launch-decision",
                                            "preview_tool": "object.context",
                                            "params": {
                                                "object_path": "situations/2026-05-acme-launch-decision/state.md"
                                            },
                                            "advisory_guidance": _advisory_guidance(
                                                "Use advisory guidance to reason about the Acme Launch Decision."
                                            ),
                                            "dashboard_owned_write": False,
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                    "refresh": {"ttl_seconds": 60},
                    "llm_invoked_by_read_surface": False,
                }
            },
            "mcp_kb_engine_prod_object_context": {
                "result": {
                    "title": "Acme Launch Decision",
                    "summary": "Choose the next launch note after reviewing stakeholder evidence.",
                    "target_ref": "situations/2026-05-acme-launch-decision",
                    "request": {"kind": "component_action", "route": "object.context"},
                    "outcome": {"family": "answer"},
                    "receipt": {
                        "state": "answered",
                        "durable_effect": "none",
                        "llm_invoked_by_read_surface": False,
                    },
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert adapter.sent[0]["actions"]
    action = adapter.sent[0]["actions"][0]
    assert action.label == "Open situation"
    guidance_action = adapter.sent[0]["actions"][1]
    assert guidance_action.label == "Ask LLM"

    card = action.handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(card):
        card = asyncio.run(card)

    assert "Acme Launch Decision" in card["text"]
    assert "Choose the next launch note" in card["text"]
    assert "Receipt: answered" in card["text"]
    assert "Effect: none" in card["text"]
    assert "Read-surface LLM: no" in card["text"]
    assert "Outcome: answer" in card["text"]
    assert ctx.calls[-1] == (
        "mcp_kb_engine_prod_object_context",
        {"object_path": "situations/2026-05-acme-launch-decision/state.md"},
    )

    guidance_card = guidance_action.handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(guidance_card):
        guidance_card = asyncio.run(guidance_card)

    assert "KB LLM Guidance" in guidance_card["text"]
    assert "kb.review_guidance" in guidance_card["text"]
    assert "no_mutation_authority" in guidance_card["text"]
    assert "Advisory output never confirms" in guidance_card["text"]


def test_kb_workbench_renders_guided_decision_cards(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_dashboard_live": {
                "result": {
                    "surface": "dashboard.live",
                    "summary": {
                        "active_todo_count": 4,
                        "proposal_queue": {"count": 3},
                        "publication_status": "dirty",
                        "readiness_status": "ready",
                    },
                    "sections": [
                        {
                            "id": "workbench",
                            "title": "Review Queue",
                            "cards": [
                                {
                                    "id": "proposal:accounts/acme",
                                    "title": "Review Acme proposal",
                                    "detail": "Decide whether the proposed Situation should be created.",
                                    "action_descriptors": [
                                        {
                                            "packet_type": "dashboard_action_descriptor",
                                            "schema_version": 2,
                                            "action_id": "proposal.details",
                                            "label": "Details",
                                            "method": "object.context",
                                            "mutation": "read_only",
                                            "target_kind": "situation",
                                            "target_ref": "accounts/acme",
                                            "preview_tool": "object.context",
                                            "params": {"object_path": "accounts/acme/state.md"},
                                            "advisory_guidance": _advisory_guidance(
                                                "Ask for advisory guidance before deciding on Acme."
                                            ),
                                            "dashboard_owned_write": False,
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            },
            "mcp_kb_engine_prod_object_context": {
                "result": {
                    "title": "Acme context",
                    "summary": "Canonical evidence for the proposal.",
                    "receipt": {"state": "answered", "durable_effect": "none"},
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb workbench"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    text = adapter.sent[0]["text"]
    assert "KB Review" in text
    assert "Decision Cards" in text
    assert "Review Acme proposal" in text
    assert "Buttons open or preview canonical kb-engine actions" in text
    assert [action.label for action in adapter.sent[0]["actions"]] == ["Details", "Ask LLM"]

    guidance_card = adapter.sent[0]["actions"][1].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(guidance_card):
        guidance_card = asyncio.run(guidance_card)

    assert "KB LLM Guidance" in guidance_card["text"]
    assert "Ask for advisory guidance before deciding on Acme." in guidance_card["text"]
    assert "Advisory output never confirms" in guidance_card["text"]

    detail_card = adapter.sent[0]["actions"][0].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(detail_card):
        detail_card = asyncio.run(detail_card)

    assert "Acme context" in detail_card["text"]
    assert ctx.calls[-1] == ("mcp_kb_engine_prod_object_context", {"object_path": "accounts/acme/state.md"})


def test_kb_review_opens_lifecycle_review_packet(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_lifecycle_review": {
                "result": {
                    "packet_type": "lifecycle_review.packet",
                    "workflow": "Lifecycle Review",
                    "stewardship_area": "KB Stewardship",
                    "target": "situations",
                    "mutation_performed": False,
                    "candidates": [
                        {
                            "candidate_id": "cand_1",
                            "title": "Review Acme launch lifecycle",
                            "target_ref": "situations/2026-06-acme-launch",
                            "recommended_action": "review",
                            "summary": "Check whether the launch situation needs an update.",
                            "evidence_refs": ["source:launch-review"],
                        }
                    ],
                }
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb review"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls[0] == ("mcp_kb_engine_prod_lifecycle_review", {"target": "situations", "dry_run": True})
    text = adapter.sent[0]["text"]
    assert "Lifecycle Review" in text
    assert "Review Acme launch lifecycle" in text
    assert "Mutation: none" in text


def test_kb_workbench_renders_situation_compact_rail_and_handoff_buttons(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    situation_ref = "situations/2026-05-acme-launch-decision"
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_dashboard_live": {
                "result": {
                    "surface": "dashboard.live",
                    "summary": {
                        "active_todo_count": 1,
                        "publication_status": "clean",
                        "readiness_status": "ready",
                    },
                    "sections": [
                        {
                            "id": "situations",
                            "title": "Situations",
                            "cards": [
                                {
                                    "id": "situation:acme-launch",
                                    "kind": "situation",
                                    "title": "Acme Launch Decision",
                                    "detail": "Review the launch posture before standalone TODO hygiene.",
                                    "target": situation_ref,
                                    "action_descriptors": [
                                        {
                                            "packet_type": "dashboard_action_descriptor",
                                            "schema_version": 2,
                                            "action_id": "open_situation",
                                            "label": "Open brief",
                                            "method": "object.context",
                                            "mutation": "read_only",
                                            "target_kind": "situation",
                                            "target_ref": situation_ref,
                                            "preview_tool": "object.context",
                                            "params": {"object_path": f"{situation_ref}/state.md"},
                                            "advisory_guidance": _advisory_guidance(
                                                "Use advisory guidance to decide the next Situation update."
                                            ),
                                            "dashboard_owned_write": False,
                                        },
                                        {
                                            "packet_type": "dashboard_action_descriptor",
                                            "schema_version": 2,
                                            "action_id": "propose_situation_update",
                                            "label": "Add update",
                                            "method": "situation.request_preview",
                                            "mutation": "handoff_only",
                                            "target_kind": "situation",
                                            "target_ref": situation_ref,
                                            "preview_tool": "situation.request_preview",
                                            "required_inputs": ["update_text"],
                                            "dashboard_owned_write": False,
                                        },
                                        {
                                            "packet_type": "dashboard_action_descriptor",
                                            "schema_version": 2,
                                            "action_id": "propose_child_commitment",
                                            "label": "Add commitment",
                                            "method": "situation.request_preview",
                                            "mutation": "handoff_only",
                                            "target_kind": "situation",
                                            "target_ref": situation_ref,
                                            "preview_tool": "situation.request_preview",
                                            "required_inputs": ["commitment_text"],
                                            "dashboard_owned_write": False,
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                }
            },
            "mcp_kb_engine_prod_object_context": {
                "result": {
                    "title": "Acme Launch Decision",
                    "summary": "Canonical Situation context.",
                    "receipt": {"state": "answered", "durable_effect": "none"},
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb workbench"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    text = adapter.sent[0]["text"]
    assert "kb review" in text.lower() or "KB Review" in text
    assert "Rail: Open brief, Add update, Add commitment, Ask LLM, Skip" in text
    assert "Writes: handoff-only until kb-engine returns a confirmed workflow." in text
    assert [action.label for action in adapter.sent[0]["actions"]] == [
        "Open brief",
        "Add update",
        "Add commitment",
        "Ask LLM",
    ]

    handoff_card = adapter.sent[0]["actions"][1].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(handoff_card):
        handoff_card = asyncio.run(handoff_card)

    assert "Add update" in handoff_card["text"]
    assert "This is a kb-engine handoff action, not a durable write." in handoff_card["text"]
    assert "Required input: update_text" in handoff_card["text"]
    assert "No KB state changed." in handoff_card["text"]


def test_kb_sync_command_previews_sync_workflow(monkeypatch, tmp_path):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_workflow_plan_request": {
                "result": {
                    "status": "confirmation_required",
                    "workflow": {"workflow_id": "update_kb", "risk": "write_broad"},
                    "request": {"args": {}, "force": False},
                    "effect_plan": {"effects": [{"id": "workflow.sync.fetch_sources"}]},
                    "outcome": {"family": "workflow_start_plan"},
                    "receipt": {
                        "state": "ready_to_confirm",
                        "durable_effect": "none",
                        "llm_invoked_by_read_surface": False,
                    },
                }
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb sync"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls[0] == (
        "mcp_kb_engine_prod_workflow_plan_request",
        {
            "workflow_id": "update_kb",
            "intent": "sync",
            "actor": "telegram:operator",
            "source": "Hermes Telegram",
            "session_id": ctx.calls[0][1]["session_id"],
        },
    )
    text = adapter.sent[0]["text"]
    assert "KB Sync Preview" in text
    assert "Request: /kb sync" in text
    assert "Receipt: ready_to_confirm" in text
    assert "Outcome: workflow_start_plan" in text
    assert "To start: /kb sync confirm" in text
    assert adapter.sent[0]["actions"] == []


def test_kb_sync_confirm_requires_pending_preview(monkeypatch, tmp_path):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext({})
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb sync confirm"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert "No fresh /kb sync preview" in adapter.sent[0]["text"]
    assert "No KB state changed." in adapter.sent[0]["text"]
    assert ctx.calls == []


def test_kb_sync_confirm_consumes_fresh_preview_lease(monkeypatch, tmp_path):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_workflow_plan_request": {
                "result": {
                    "status": "confirmation_required",
                    "schema_version": 1,
                    "tool": "workflow.start_confirmed",
                    "workflow": {"workflow_id": "update_kb", "risk": "write_broad"},
                    "request": {"args": {}, "queue_gate_limit": 0, "force": False},
                    "request_id": "sync_preview_1",
                    "idempotency_key": "workflow:update_kb:telegram",
                    "effect_plan": {"effects": [{"id": "workflow.sync.fetch_sources"}]},
                    "receipt": {"state": "ready_to_confirm", "durable_effect": "none"},
                }
            },
            "mcp_kb_engine_prod_workflow_start_confirmed": {
                "result": {
                    "status": "started",
                    "started": True,
                    "run": {"run_id": "gen-123", "workflow_id": "update_kb"},
                    "receipt": {"state": "workflow_running", "durable_effect": "workflow_run"},
                }
            },
            "mcp_kb_engine_prod_run_watch": {
                "result": {"status": "running", "terminal": False, "progress_digest": {"status": "running"}},
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    preview = hook(event=_event("/kb sync"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()
    confirmed = hook(event=_event("/kb sync confirm"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert preview == {"action": "skip", "reason": "kb_journeys"}
    assert confirmed == {"action": "skip", "reason": "kb_journeys"}
    assert [call[0] for call in ctx.calls].count("mcp_kb_engine_prod_workflow_plan_request") == 1
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_workflow_start_confirmed"
    envelope = ctx.calls[-2][1]["envelope"]
    assert envelope["plan"]["request_id"] == "sync_preview_1"
    assert envelope["user_confirmation"]["preview_required"] is True
    assert envelope["user_confirmation"]["preview_lease"]["preview_lease_id"].startswith("sha256:")
    assert "Workflow start result" in adapter.sent[-1]["text"]


def test_kb_sync_confirm_fails_closed_for_stale_or_wrong_actor_preview(monkeypatch, tmp_path):
    from plugins import kb_journeys
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_workflow_plan_request": {
                "result": {
                    "status": "confirmation_required",
                    "workflow": {"workflow_id": "update_kb"},
                    "request": {"args": {}, "force": False},
                    "request_id": "sync_preview_stale",
                }
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)
    store = FakeSessionStore(session_id="shared-session")

    monkeypatch.setattr(kb_journeys.time, "time", lambda: 1000.0)
    hook(event=_event("/kb sync"), gateway=_authorized_gateway(adapter), session_store=store)
    _drain_scheduled_tasks()
    wrong_actor = hook(
        event=_event("/kb sync confirm", user_id="user-2", user_name="other"),
        gateway=_authorized_gateway(adapter),
        session_store=store,
    )
    _drain_scheduled_tasks()

    assert wrong_actor == {"action": "skip", "reason": "kb_journeys"}
    assert "belongs to another Telegram user" in adapter.sent[-1]["text"]
    assert "mcp_kb_engine_prod_workflow_start_confirmed" not in [call[0] for call in ctx.calls]

    monkeypatch.setattr(kb_journeys.time, "time", lambda: 1000.0 + kb_journeys.SYNC_PREVIEW_STATE_TTL_SECONDS + 1)
    stale = hook(event=_event("/kb sync confirm"), gateway=_authorized_gateway(adapter), session_store=store)
    _drain_scheduled_tasks()

    assert stale == {"action": "skip", "reason": "kb_journeys"}
    assert "pending /kb sync preview is stale" in adapter.sent[-1]["text"]
    assert "mcp_kb_engine_prod_workflow_start_confirmed" not in [call[0] for call in ctx.calls]


def test_dashboard_validation_descriptor_renders_graph_receipt(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_dashboard_live": {
                "result": {
                    "surface": "dashboard.live",
                    "summary": {
                        "active_run_count": 0,
                        "active_todo_count": 0,
                        "publication_status": "clean",
                        "readiness_status": "ready",
                    },
                    "sections": [
                        {
                            "id": "now",
                            "title": "Now",
                            "cards": [
                                {
                                    "id": "graph:validation",
                                    "title": "Graph validation",
                                    "action_descriptors": [
                                        {
                                            "packet_type": "dashboard_action_descriptor",
                                            "schema_version": 2,
                                            "action_id": "objects.validate_graph",
                                            "label": "Validate graph",
                                            "method": "objects.validate_graph",
                                            "mutation": "read_only",
                                            "target_kind": "object_graph",
                                            "target_ref": "kb",
                                            "preview_tool": "objects.validate_graph",
                                            "params": {},
                                            "dashboard_owned_write": False,
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            },
            "mcp_kb_engine_prod_objects_validate_graph": {
                "result": {
                    "packet_type": "durable_graph_validation",
                    "schema_version": 1,
                    "status": "warning",
                    "ok": False,
                    "warning_count": 1,
                    "error_count": 0,
                    "warnings": [
                        {
                            "code": "missing_related_object",
                            "ref": "reports/2026-06-future-of-ai-for-synbio",
                            "message": "Report ref is missing.",
                        }
                    ],
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    action = next(action for action in adapter.sent[0]["actions"] if action.label == "Validate graph")

    card = action.handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(card):
        card = asyncio.run(card)

    assert "KB Graph Validation" in card["text"]
    assert "Status: warning" in card["text"]
    assert "Warnings: 1" in card["text"]
    assert "missing_related_object" in card["text"]
    assert "reports/2026-06-future-of-ai-for-synbio" in card["text"]


def test_dashboard_report_descriptor_renders_preview_confirm_receipts(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    descriptor = {
        "packet_type": "dashboard_action_descriptor",
        "schema_version": 2,
        "action_id": "report.admit",
        "label": "Admit report",
        "method": "report.admit",
        "mutation": "workspace_write",
        "target_kind": "report",
        "target_ref": "reports/2026-06-future-of-ai-for-synbio",
        "preview_tool": "report.admit_preview",
        "confirm_tool": "report.admit_confirmed",
        "params": {
            "report_ref": "reports/2026-06-future-of-ai-for-synbio",
            "event_ref": "events/2026-06-future-trends-forum",
        },
        "dashboard_owned_write": False,
        "requires_canonical_tool": True,
        "confirmation_copy": "Confirm report admission after reviewing the preview.",
    }
    preview_receipt = {
        "packet_type": "report_admission_receipt",
        "schema_version": 1,
        "status": "preview",
        "object_family": "report",
        "report_ref": "reports/2026-06-future-of-ai-for-synbio",
        "report_refs": ["reports/2026-06-future-of-ai-for-synbio"],
        "title": "Future of AI for Synbio",
        "event_ref": "events/2026-06-future-trends-forum",
        "event_role": "canonical_event",
        "situation_ref": "situations/2026-06-future-of-ai-for-synbio-prep",
        "related_objects": [
            "events/2026-06-future-trends-forum",
            "situations/2026-06-future-of-ai-for-synbio-prep",
        ],
        "source_transfers": [{"source_path": "/tmp/report.md", "destination_path": "files/report.md"}],
    }
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_dashboard_live": {
                "result": {
                    "surface": "dashboard.live",
                    "summary": {
                        "active_run_count": 0,
                        "active_todo_count": 0,
                        "publication_status": "dirty",
                        "readiness_status": "ready",
                    },
                    "sections": [
                        {
                            "id": "reports",
                            "title": "Reports",
                            "cards": [
                                {
                                    "id": "report:future-ai-synbio",
                                    "title": "Future of AI for Synbio",
                                    "action_descriptors": [descriptor],
                                }
                            ],
                        }
                    ],
                }
            },
            "mcp_kb_engine_prod_report_admit_preview": {"result": preview_receipt},
            "mcp_kb_engine_prod_report_admit_confirmed": {
                "result": {
                    **preview_receipt,
                    "status": "applied",
                    "changed_paths": [
                        "reports/2026-06-future-of-ai-for-synbio/report.yaml",
                        "situations/2026-06-future-of-ai-for-synbio-prep/state.md",
                    ],
                    "graph_validation": {
                        "packet_type": "durable_graph_validation",
                        "schema_version": 1,
                        "status": "ok",
                        "ok": True,
                    },
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    preview_action = next(action for action in adapter.sent[0]["actions"] if action.label == "Preview Admit report")

    preview_card = preview_action.handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(preview_card):
        preview_card = asyncio.run(preview_card)

    assert "Report Admission" in preview_card["text"]
    assert "Status: preview" in preview_card["text"]
    assert "Future of AI for Synbio" in preview_card["text"]
    assert "Object family: report" in preview_card["text"]
    assert "Report refs: reports/2026-06-future-of-ai-for-synbio" in preview_card["text"]
    assert "Related objects: events/2026-06-future-trends-forum" in preview_card["text"]
    assert "No durable write has been made." in preview_card["text"]
    assert preview_card["actions"][0].label == "Confirm Admit report"

    confirm_card = preview_card["actions"][0].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(confirm_card):
        confirm_card = asyncio.run(confirm_card)

    assert "Status: applied" in confirm_card["text"]
    assert "Changed paths: 2" in confirm_card["text"]
    assert "Graph validation: ok" in confirm_card["text"]
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_report_admit_preview"
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_report_admit_confirmed"
    assert ctx.calls[-1][1]["user_confirmation"]["confirmed"] is True
    assert ctx.calls[-1][1]["user_confirmation"]["preview_required"] is True
    assert ctx.calls[-1][1]["actor"] == "telegram:user-1"


def test_kb_review_lifecycle_renders_readonly_packet_and_descriptor_preview(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    descriptor = {
        "packet_type": "dashboard_action_descriptor",
        "schema_version": 2,
        "action_id": "lifecycle.archive_proposal",
        "label": "Draft archive proposal",
        "method": "lifecycle.proposal_preview",
        "mutation": "proposal_preview",
        "target_kind": "lifecycle_candidate",
        "target_ref": "situations/2026-04-old-launch",
        "preview_tool": "lifecycle.proposal_preview",
        "confirm_tool": "lifecycle.proposal_confirmed",
        "requires_canonical_tool": True,
        "params": {
            "candidate_id": "cand_1",
            "target_ref": "situations/2026-04-old-launch",
            "recommended_action": "archive",
        },
    }
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_lifecycle_review": {
                "result": {
                    "packet_type": "lifecycle_review.packet",
                    "workflow": "Lifecycle Review",
                    "stewardship_area": "KB Stewardship",
                    "target": "situations",
                    "mutation_performed": False,
                    "candidates": [
                        {
                            "candidate_id": "cand_1",
                            "title": "Old launch situation",
                            "target_ref": "situations/2026-04-old-launch",
                            "recommended_action": "archive",
                            "evidence_refs": ["source:launch-retro"],
                            "evidence_gaps": ["No active owner found"],
                            "signals": {
                                "lifecycle_signal_count": 1,
                                "closure_signal": {
                                    "signal_id": "inbox-old-launch-final",
                                    "source": "inbox",
                                    "source_ref": "inbox:old-launch-final",
                                    "observed_at": "2026-06-08T12:00:00Z",
                                    "kind": "outcome",
                                    "polarity": "complete",
                                    "confidence": "high",
                                    "summary": "Final owner confirmed the launch thread is complete.",
                                    "evidence_refs": ["inbox:old-launch-final"],
                                },
                            },
                            "action_descriptor": descriptor,
                        }
                    ],
                }
            },
            "mcp_kb_engine_prod_lifecycle_proposal_preview": {
                "result": {
                    "packet_type": "lifecycle_proposal_draft.packet",
                    "workflow": "Lifecycle Review",
                    "stewardship_area": "KB Stewardship",
                    "mutation_performed": False,
                    "expected_before_hash": "life-before-hash-1",
                    "proposal_count": 1,
                    "proposals": [
                        {
                            "proposal_id": "draft_archive_old_launch",
                            "target_ref": "situations/2026-04-old-launch",
                            "recommended_action": "archive",
                            "summary": "Draft an archive proposal for reviewer approval.",
                        }
                    ],
                }
            },
            "mcp_kb_engine_prod_lifecycle_proposal_confirmed": {
                "result": {
                    "packet_type": "lifecycle_proposal_confirm_receipt",
                    "status": "proposal_recorded",
                    "reason": "queued for proposal review",
                    "proposal_count": 1,
                    "proposal_ids": ["draft_archive_old_launch"],
                    "durable_kb_object_changed": False,
                    "apply_performed": False,
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb review lifecycle"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls == [("mcp_kb_engine_prod_lifecycle_review", {"target": "situations", "dry_run": True})]
    text = adapter.sent[0]["text"]
    assert "Lifecycle Review" in text
    assert "Stewardship: KB Stewardship" in text
    assert "Mutation: none" in text
    assert "Old launch situation" in text
    assert "Action: archive" in text
    assert "Target: situations/2026-04-old-launch" in text
    assert "Source: inbox" in text
    assert "Observed: 2026-06-08T12:00:00Z" in text
    assert "Signal: outcome / complete (high)" in text
    assert "Signal evidence refs: 1" in text
    assert "Signal summary: Final owner confirmed the launch thread is complete." in text
    assert "Evidence: source:launch-retro" in text
    assert "Gaps: No active owner found" in text
    assert [action.label for action in adapter.sent[0]["actions"]] == ["Draft archive proposal"]
    assert adapter.sent[0]["actions"][0].metadata["durable_write"] is False
    assert adapter.sent[0]["actions"][0].metadata["confirm_tool"] == "mcp_kb_engine_prod_lifecycle_proposal_confirmed"

    preview_card = adapter.sent[0]["actions"][0].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(preview_card):
        preview_card = asyncio.run(preview_card)

    assert "Lifecycle Proposal Draft" in preview_card["text"]
    assert "Proposals: 1" in preview_card["text"]
    assert "draft_archive_old_launch" in preview_card["text"]
    assert "No durable write has been made." in preview_card["text"]
    assert [action.label for action in preview_card["actions"]] == ["Confirm Draft archive proposal"]
    assert ctx.calls[-1] == (
        "mcp_kb_engine_prod_lifecycle_proposal_preview",
        {
            "candidate_id": "cand_1",
            "target_ref": "situations/2026-04-old-launch",
            "recommended_action": "archive",
        },
    )

    confirm_card = preview_card["actions"][0].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(confirm_card):
        confirm_card = asyncio.run(confirm_card)

    assert "Status: proposal_recorded" in confirm_card["text"]
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_lifecycle_proposal_confirmed"
    assert ctx.calls[-1][1]["user_confirmation"]["confirmed"] is True
    assert ctx.calls[-1][1]["user_confirmation"]["preview_required"] is True
    assert ctx.calls[-1][1]["expected_before_hash"] == "life-before-hash-1"
    assert ctx.calls[-1][1]["actor"] == "telegram:user-1"
    assert ctx.calls[-1][1]["source"] == "Hermes Telegram Action Card"


def test_kb_review_and_review_queue_prose_render_lifecycle_update_packet(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    lifecycle_update_packet = {
        "kind": "lifecycle_update.packet",
        "workflow": "Lifecycle Review",
        "stewardship_area": "KB Stewardship",
        "target": "review_queue",
        "mutation_performed": False,
        "raw_private_source_text_copied": False,
        "source_manifest": {
            "source_count": 2,
            "sources": [
                {
                    "kind": "inbox",
                    "source_ref": "source:review-mail",
                    "title": "Review queue follow-up",
                    "status": "included",
                },
                {
                    "kind": "doc",
                    "source_ref": "https://example.invalid/private-raw-source",
                    "title": "Lifecycle handoff",
                },
            ],
        },
        "surface_authority_map": {
            "Telegram": "read_surface",
            "kb_engine_prod": {"authority": "durable_semantics"},
        },
        "candidates": [
            {
                "candidate_id": "cand_1",
                "title": "Archive stale situation",
                "target_ref": "situations/2026-04-old-launch",
                "recommended_action": "draft_archive_proposal",
                "summary": "Closure evidence is strong enough to draft a lifecycle proposal.",
                "candidate_actions": [{"label": "Draft archive proposal"}],
                "evidence_refs": ["source:review-mail", "https://example.invalid/private-source"],
            }
        ],
        "source_body": "private source body should not render",
    }
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_lifecycle_review": {"result": lifecycle_update_packet},
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    prose = hook(event=_event("what is in the review queue"), gateway=_authorized_gateway(adapter), session_store=None)
    slash = hook(event=_event("/kb review"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert prose == {"action": "skip", "reason": "kb_journeys"}
    assert slash == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls == [
        ("mcp_kb_engine_prod_lifecycle_review", {"target": "situations", "dry_run": True}),
        ("mcp_kb_engine_prod_lifecycle_review", {"target": "situations", "dry_run": True}),
    ]
    for sent in adapter.sent:
        text = sent["text"]
        assert "Lifecycle Update" in text
        assert "Workflow: Lifecycle Review" in text
        assert "Mutation: none" in text
        assert "Raw private source text copied: no" in text
        assert "Sources: 2" in text
        assert "inbox - source:review-mail - Review queue follow-up" in text
        assert "Authority map: Telegram=read_surface; kb_engine_prod=durable_semantics" in text
        assert "Archive stale situation" in text
        assert "Candidate actions: Draft archive proposal, draft_archive_proposal" in text
        assert "No durable write has been made." in text
        assert "private source body" not in text
        assert "example.invalid" not in text


def test_dashboard_proposal_queue_descriptor_uses_generic_preview_confirm_with_lease(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb_engine_prod")
    descriptor = {
        "packet_type": "dashboard_action_descriptor",
        "schema_version": 2,
        "action_id": "review.entity_reject",
        "label": "Reject visible proposal",
        "method": "queue.batch_decide_confirmed",
        "mutation": "workspace_write",
        "target_kind": "proposal_queue",
        "target_ref": "accounts/acme",
        "preview_tool": "queue.decision_preview",
        "confirm_tool": "queue.batch_decide_confirmed",
        "params": {"proposal_ids": ["act_acme"], "decision": "reject"},
        "dashboard_owned_write": False,
        "requires_canonical_tool": True,
        "confirmation_copy": "Confirm Reject after reviewing the proposal preview.",
    }
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_dashboard_live": {
                "result": {
                    "surface": "dashboard.live",
                    "summary": {"publication_status": "dirty", "readiness_status": "ready"},
                    "sections": [
                        {
                            "id": "workbench",
                            "title": "Review Queue",
                            "cards": [
                                {
                                    "id": "proposal:accounts/acme",
                                    "title": "Acme proposal",
                                    "action_descriptors": [descriptor],
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
                    "summary": "Reject Acme only after review.",
                    "review_session": {
                        "review_session_id": "review_session_acme",
                        "decision_scope": "explicit_ids",
                        "cursor": {"cursor_id": "cursor_acme", "displayed_count": 1, "candidate_count": 1},
                    },
                    "preview_lease": {
                        "preview_lease_id": "lease_acme",
                        "review_session_id": "review_session_acme",
                        "cursor_id": "cursor_acme",
                        "decision_scope": "explicit_ids",
                    },
                }
            },
            "mcp_kb_engine_prod_queue_batch_decide_confirmed": {
                "result": {
                    "packet_type": "request.receipt",
                    "schema_version": 1,
                    "route": "queue.batch_decide_confirmed",
                    "status": "applied",
                    "ok": True,
                    "confirmed_count": 1,
                    "affected_ids": ["act_acme"],
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb workbench"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    preview_action = next(action for action in adapter.sent[0]["actions"] if action.label == "Preview Reject visible proposal")
    preview_card = preview_action.handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(preview_card):
        preview_card = asyncio.run(preview_card)

    assert "Scope: Selected" in preview_card["text"]
    assert preview_card["actions"][0].metadata["preview_lease"] is True
    assert preview_card["actions"][0].metadata["review_session_id"] == "review_session_acme"

    confirm_card = preview_card["actions"][0].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(confirm_card):
        confirm_card = asyncio.run(confirm_card)

    assert "KB Review Receipt" in confirm_card["text"]
    assert "Affected ids: act_acme" in confirm_card["text"]
    assert [call[0] for call in ctx.calls] == [
        "mcp_kb_engine_prod_dashboard_live",
        "mcp_kb_engine_prod_queue_decision_preview",
        "mcp_kb_engine_prod_queue_batch_decide_confirmed",
    ]
    confirm_args = ctx.calls[-1][1]
    assert confirm_args["session_id"] == "review_session_acme"
    assert confirm_args["review_session_id"] == "review_session_acme"
    assert confirm_args["cursor_id"] == "cursor_acme"
    assert confirm_args["decision_scope"] == "explicit_ids"
    assert confirm_args["user_confirmation"]["preview_lease"]["preview_lease_id"] == "lease_acme"


def test_descriptor_guidance_facets_are_advisory_and_redacted():
    from plugins import kb_journeys

    descriptor = {
        "action_id": "review.entity_reject",
        "label": "Reject",
        "target_kind": "proposal_queue",
        "target_ref": "accounts/acme",
        "advisory_guidance": {
            "packet_type": "kb_advisory_guidance",
            "schema_version": 1,
            "mode": "advisory_only",
            "authority": "no_mutation_authority",
            "llm_prompt": "kb.review_guidance",
            "mutates_state": False,
            "summary": "Use care before deciding.",
            "why": "The proposed update lacks a durable delta.",
            "recommendation": "Reject unless the evidence improves.",
            "evidence": [
                "Source body is cached at /Users/acosta/private/source.txt",
                "token=super-secret-value",
            ],
            "missing_context": "Need a concrete owner or date.",
        },
    }

    card = kb_journeys._render_descriptor_guidance(descriptor, title="KB Review LLM Guidance")
    assert [action.label for action in card["actions"]] == ["Why", "Recommend", "Evidence", "Missing Context"]
    assert "Advisory output never confirms" in card["text"]

    evidence_card = card["actions"][2].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(evidence_card):
        evidence_card = asyncio.run(evidence_card)

    assert "Evidence" in evidence_card["text"]
    assert "/Users/acosta" not in evidence_card["text"]
    assert "super-secret-value" not in evidence_card["text"]
    assert evidence_card["actions"] == []


def test_descriptor_guidance_unavailable_and_stale_states_are_non_authoritative():
    from plugins import kb_journeys

    unavailable = {
        "action_id": "review.entity_reject",
        "label": "Reject",
        "advisory_guidance": {
            "packet_type": "kb_advisory_guidance",
            "schema_version": 1,
            "status": "unavailable",
            "mutates_state": False,
            "unavailable_reason": "No guidance route is attached.",
        },
    }
    unavailable_card = kb_journeys._render_descriptor_guidance(unavailable, title="KB Review LLM Guidance")
    assert "Guidance unavailable" in unavailable_card["text"]
    assert "No guidance route" in unavailable_card["text"]
    assert unavailable_card["actions"] == []

    stale = {
        "action_id": "review.entity_reject",
        "label": "Reject",
        "advisory_guidance": {
            "packet_type": "kb_advisory_guidance",
            "schema_version": 1,
            "status": "stale",
            "mutates_state": False,
            "summary": "This recommendation came from an old queue packet.",
        },
    }
    stale_card = kb_journeys._render_descriptor_guidance(stale, title="KB Review LLM Guidance")
    assert "Status: stale" in stale_card["text"]
    assert "Advisory output never confirms" in stale_card["text"]
    assert stale_card["actions"] == []


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
            "mcp_kb_engine_prod_review_inbox": {"result": {"total": 0, "items": []}},
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
    assert ctx.commands["kb"]["description"] == "KB status, sync, and review."
    assert ctx.commands["kb"]["handler"]("") == (
        "Use /kb in Telegram. Try: /kb status, /kb sync, or /kb review."
    )
    assert "pre_gateway_dispatch" in ctx.hooks


def test_kb_help_exposes_only_three_primary_verbs(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext({})
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb help"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    text = adapter.sent[0]["text"]
    assert "/kb status" in text
    assert "/kb sync" in text
    assert "/kb review" in text
    assert "/kb queue" not in text
    assert "/kb publish" not in text
    assert "/kb runs" not in text
    assert "/kb reasoning" not in text


def test_send_card_falls_back_to_text_when_native_action_card_fails():
    from plugins.kb_journeys import _send_card
    from tools.kb_callback_registry import KbAction

    adapter = FailingKbActionsAdapter()
    asyncio.run(
        _send_card(
            adapter,
            _event("/kb"),
            {
                "title": "KB Review",
                "text": "Review proposal",
                "actions": [
                    KbAction(label="Preview Reject", action_id="preview", handler=lambda _ctx: None),
                    KbAction(label="Guidance", action_id="guidance", handler=lambda _ctx: None),
                ],
            },
        )
    )

    assert len(adapter.sent) == 2
    assert adapter.sent[0]["failed_native_card"] is True
    assert adapter.sent[1]["text"] == "Review proposal\n\nActions: Preview Reject, Guidance"
    assert adapter.sent[1]["reply_to"] == "m1"
    assert adapter.sent[1]["metadata"]["thread_id"] == "topic-1"
    assert adapter.sent[1]["metadata"]["telegram_dm_topic_reply_fallback"] is True


def test_kb_root_queue_dashboard_starts_guided_first_item(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": {
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
    assert ctx.calls == [("mcp_kb_engine_prod_review_inbox", {"scope": "proposals", "limit": 5})]
    assert adapter.sent
    text = adapter.sent[0]["text"]
    assert "KB Review" in text
    assert "1 of 9 · Visible scope" in text
    assert "Admit Stanford DAS Lab" in text
    assert "Scope: accounts/stanford-das-lab · 2 proposals · 1 visible · 9 total" in text
    assert "Rail: Details" in text
    assert "Nothing applies until kb-engine returns a preview lease and you confirm." in text
    assert "Review: /kb review 1" not in text
    assert "Text fallback:" not in text
    assert "Batch:" not in text
    assert [action.label for action in adapter.sent[0]["actions"]] == ["Details"]
    assert adapter.sent[0]["reply_to"] == "m1"


def test_kb_queue_guided_card_buttons_preview_and_skip(monkeypatch, tmp_path):
    from plugins import kb_journeys
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": {
                "result": {
                    "total": 2,
                    "items": [
                        {
                            "item_id": "accounts/mistral",
                            "title": "Mistral",
                            "kind": "proposal_entity",
                            "summary": "Admission: Mistral has licensing coordination.",
                            "raw": {"proposal_ids": ["act_2"]},
                            "safe_actions": [
                                {
                                    "packet_type": "dashboard_action_descriptor",
                                    "schema_version": 2,
                                    "action_id": "review.entity_approve",
                                    "label": "Approve",
                                    "target_kind": "proposal_queue",
                                    "target_ref": "accounts/mistral",
                                    "preview_tool": "queue.decision_preview",
                                    "confirm_tool": "queue.batch_decide_confirmed",
                                    "params": {"proposal_ids": ["act_2"], "decision": "approve"},
                                    "dashboard_owned_write": False,
                                    "requires_canonical_tool": True,
                                },
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
                                    "advisory_guidance": _advisory_guidance(
                                        "Use advisory guidance to reason about Reject before previewing."
                                    ),
                                },
                                {
                                    "packet_type": "dashboard_action_descriptor",
                                    "schema_version": 2,
                                    "action_id": "review.entity_archive",
                                    "label": "Archive",
                                    "target_kind": "proposal_queue",
                                    "target_ref": "accounts/mistral",
                                    "preview_tool": "queue.decision_preview",
                                    "confirm_tool": "queue.batch_decide_confirmed",
                                    "params": {"proposal_ids": ["act_2"], "decision": "archive"},
                                    "dashboard_owned_write": False,
                                    "requires_canonical_tool": True,
                                },
                            ],
                        },
                        {
                            "item_id": "accounts/keio-university",
                            "title": "Keio University",
                            "kind": "proposal_entity",
                            "summary": "Admission: Keio has a healthcare AI PoC.",
                            "raw": {"proposal_ids": ["act_3"]},
                        },
                    ],
                }
            },
            "mcp_kb_engine_prod_queue_decision_preview": {
                "result": {"status": "preview", "ok": True, "plan": {"summary": "Reject 1 proposal."}}
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)
    store = FakeSessionStore("session-guided")

    result = hook(event=_event("/kb queue"), gateway=_authorized_gateway(adapter), session_store=store)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    text = adapter.sent[0]["text"]
    assert "KB Review" in text
    assert "Rail: Approve, Reject, Archive, Details, Ask LLM, Skip" in text
    assert "Decision Card" not in text
    assert "Preview Reject" not in text
    assert "/kb review 1" not in text
    assert "Text fallback:" not in text
    assert [action.label for action in adapter.sent[0]["actions"]] == ["Approve", "Reject", "Archive", "Details", "Ask LLM", "Skip"]
    assert kb_journeys.scoped_mcp_tool_allowlist_for_message(
        session_id="session-guided",
        message="Reject",
    ) == {"mcp_kb_engine_prod_queue_decision_preview"}

    preview_card = next(action for action in adapter.sent[0]["actions"] if action.label == "Reject").handler(
        SimpleNamespace(actor_id="user-1", actor_name="tester")
    )
    if asyncio.iscoroutine(preview_card):
        preview_card = asyncio.run(preview_card)

    assert "Review reject preview" in preview_card["text"]
    assert preview_card["actions"][0].label == "Confirm Reject"
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_queue_decision_preview"
    assert ctx.calls[-1][1]["proposal_ids"] == ["act_2"]

    skip_card = next(action for action in adapter.sent[0]["actions"] if action.label == "Skip").handler(
        SimpleNamespace(actor_id="user-1", actor_name="tester")
    )
    if asyncio.iscoroutine(skip_card):
        skip_card = asyncio.run(skip_card)

    assert "Skipped item 1 locally. No KB state changed." in skip_card["text"]
    assert "Review Item 2" in skip_card["text"]
    assert "Keio University" in skip_card["text"]


def test_kb_queue_tasks_renders_nonproposal_review_card_and_control_preview(monkeypatch, tmp_path):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    todo_item = {
        "item_id": "todo_123",
        "title": "Confirm Acme launch contract",
        "kind": "todo",
        "summary": "Use the workbench queue.",
        "entity_path": "accounts/acme",
        "safe_actions": [
            {
                "action_id": "todo.complete",
                "label": "Complete",
                "method": "control.apply_confirmed",
                "mutation": "workspace_write",
                "requires_confirmation": True,
                "params": {"todo_id": "todo_123", "operation_id": "todo.complete"},
                "confirmed_write_route": {
                    "object_ref": {"kind": "todo", "id": "todo_123"},
                    "operation_id": "todo.complete",
                    "arguments": {"todo_id": "todo_123"},
                    "required_input": [],
                    "sequence": [
                        "control.context",
                        "control.apply_preview",
                        "control.build_confirmed_envelope",
                        "control.apply_confirmed",
                    ],
                },
            },
            {
                "action_id": "todo.delegate",
                "label": "Delegate",
                "method": "control.apply_confirmed",
                "mutation": "workspace_write",
                "requires_confirmation": True,
                "params": {"todo_id": "todo_123", "operation_id": "todo.delegate"},
                "confirmed_write_route": {
                    "object_ref": {"kind": "todo", "id": "todo_123"},
                    "operation_id": "todo.delegate",
                    "arguments": {"todo_id": "todo_123", "delegated_to": ""},
                    "required_input": ["delegated_to"],
                    "sequence": [
                        "control.context",
                        "control.apply_preview",
                        "control.build_confirmed_envelope",
                        "control.apply_confirmed",
                    ],
                },
            },
            {
                "action_id": "todo.archive",
                "label": "Archive",
                "method": "control.apply_confirmed",
                "mutation": "workspace_write",
                "requires_confirmation": True,
                "params": {"todo_id": "todo_123", "operation_id": "todo.archive"},
                "confirmed_write_route": {
                    "object_ref": {"kind": "todo", "id": "todo_123"},
                    "operation_id": "todo.archive",
                    "arguments": {"todo_id": "todo_123"},
                    "required_input": [],
                    "sequence": [
                        "control.context",
                        "control.apply_preview",
                        "control.build_confirmed_envelope",
                        "control.apply_confirmed",
                    ],
                },
            },
        ],
        "raw": {
            "review_session": {
                "packet_type": "guided_kb_review_session",
                "review_session_id": "todo_session_123",
                "surface": "todo.review",
                "scope": "tasks_queue",
                "decision_scope": "explicit_ids",
                "cursor": {
                    "cursor_id": "todo_cursor_123",
                    "displayed_count": 2,
                    "candidate_count": 2,
                    "item_ids": ["todo_123", "todo_456"],
                    "viewed_item_ids": ["todo_123", "todo_456"],
                },
                "source": "workbench.queue",
            },
            "review_target": {
                "packet_type": "guided_kb_review_target",
                "target_id": "todo_123",
                "kind": "todo",
                "title": "Confirm Acme launch contract",
                "summary": "Use the workbench queue.",
                "scope": {
                    "affected_ids": ["todo_123"],
                    "affected_count": 1,
                    "viewed_ids": ["todo_123", "todo_456"],
                    "viewed_count": 2,
                },
                "action_ids": ["todo.complete", "todo.delegate", "todo.archive"],
                "durable_action_ids": ["todo.complete", "todo.delegate", "todo.archive"],
                "policy": {
                    "semantic_owner": "kb-engine",
                    "preview_required": True,
                    "confirmed_envelope_required": True,
                    "advisory_guidance_authoritative": False,
                },
            },
        },
    }
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": {
                "result": {
                    "scope": "tasks",
                    "total": 2,
                    "items": [
                        todo_item,
                        {"item_id": "todo_456", "title": "Second TODO", "kind": "todo"},
                    ],
                }
            },
            "mcp_kb_engine_prod_control_context": {
                "result": {"packet_type": "control_context", "object": {"kind": "todo", "id": "todo_123"}}
            },
            "mcp_kb_engine_prod_control_apply_preview": {
                "result": {
                    "status": "noop",
                    "ok": True,
                    "plan": {
                        "summary": "Complete from Telegram KB Review.",
                        "operations": [
                            {
                                "operation_id": "todo.complete",
                                "arguments": {"todo_id": "todo_123"},
                            }
                        ],
                    },
                    "results": [{"operation_id": "todo.complete", "message": "dry run"}],
                }
            },
            "mcp_kb_engine_prod_control_build_confirmed_envelope": {
                "result": {"ok": True, "envelope": {"id": "env_123"}}
            },
            "mcp_kb_engine_prod_control_apply_confirmed": {
                "result": {
                    "status": "applied",
                    "ok": True,
                    "results": [{"operation_id": "todo.complete", "message": "applied"}],
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)
    store = FakeSessionStore("session-tasks")

    result = hook(event=_event("/kb review tasks"), gateway=_authorized_gateway(adapter), session_store=store)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls[0] == ("mcp_kb_engine_prod_review_inbox", {"scope": "tasks", "limit": 5})
    text = adapter.sent[0]["text"]
    assert "KB Review" in text
    assert "Confirm Acme launch contract" in text
    assert "Rail: Complete, Delegate, Archive, Details, Ask LLM, Skip" in text
    assert "Nothing applies until kb-engine previews the control route and you confirm." in text
    assert "/kb review 1" not in text
    assert "Text fallback:" not in text
    assert [action.label for action in adapter.sent[0]["actions"]] == [
        "Complete",
        "Delegate",
        "Archive",
        "Details",
        "Ask LLM",
        "Skip",
    ]

    guidance_card = next(action for action in adapter.sent[0]["actions"] if action.label == "Ask LLM").handler(
        SimpleNamespace(actor_id="user-1", actor_name="tester")
    )
    assert "Advisory only: cannot preview, confirm, or mutate KB state." in guidance_card["text"]

    delegate_card = next(action for action in adapter.sent[0]["actions"] if action.label == "Delegate").handler(
        SimpleNamespace(actor_id="user-1", actor_name="tester")
    )
    assert "needs additional input first: delegated_to" in delegate_card["text"]

    preview_card = next(action for action in adapter.sent[0]["actions"] if action.label == "Complete").handler(
        SimpleNamespace(actor_id="user-1", actor_name="tester")
    )
    assert "KB Control Preview" in preview_card["text"]
    assert "Action: Complete" in preview_card["text"]
    assert preview_card["actions"][0].label == "Confirm Complete"
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_control_context"
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_control_apply_preview"
    assert ctx.calls[-1][1]["plan"]["operations"][0]["operation_id"] == "todo.complete"

    confirm_card = preview_card["actions"][0].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    assert "KB Control Result" in confirm_card["text"]
    assert "Status: applied" in confirm_card["text"]
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_control_build_confirmed_envelope"
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_control_apply_confirmed"
    assert ctx.calls[-2][1]["user_confirmation"]["review_session_id"] == "todo_session_123"


def test_kb_queue_skip_uses_server_window_when_available(monkeypatch, tmp_path):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = SequencedFakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": [
                {
                    "result": {
                        "total": 3,
                        "offset": 0,
                        "next_offset": 1,
                        "items": [
                            {
                                "item_id": "accounts/mistral",
                                "title": "Mistral",
                                "kind": "proposal_entity",
                                "summary": "Admission: Mistral has licensing coordination.",
                                "raw": {"proposal_ids": ["act_2"]},
                            },
                            {
                                "item_id": "accounts/keio-university",
                                "title": "Keio University",
                                "kind": "proposal_entity",
                                "summary": "Admission: Keio has a healthcare AI PoC.",
                                "raw": {"proposal_ids": ["act_3"]},
                            },
                        ],
                    }
                },
                {
                    "result": {
                        "total": 3,
                        "offset": 1,
                        "next_offset": 2,
                        "items": [
                            {
                                "item_id": "accounts/keio-university",
                                "title": "Keio University",
                                "kind": "proposal_entity",
                                "summary": "Admission: Keio has a healthcare AI PoC.",
                                "raw": {"proposal_ids": ["act_3"]},
                            }
                        ],
                    }
                },
            ]
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)
    store = FakeSessionStore("session-server-skip")

    result = hook(event=_event("/kb queue"), gateway=_authorized_gateway(adapter), session_store=store)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    skip_action = next(action for action in adapter.sent[0]["actions"] if action.label == "Skip")
    skip_card = skip_action.handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(skip_card):
        skip_card = asyncio.run(skip_card)

    assert ctx.calls[-1] == ("mcp_kb_engine_prod_review_inbox", {"scope": "proposals", "limit": 5, "offset": 1})
    assert "Advanced to the next kb-engine review window" in skip_card["text"]
    assert "Keio University" in skip_card["text"]


def test_kb_review_without_args_starts_lifecycle_review(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_lifecycle_review": {
                "result": {
                    "packet_type": "lifecycle_review.packet",
                    "workflow": "Lifecycle Review",
                    "stewardship_area": "KB Stewardship",
                    "target": "situations",
                    "mutation_performed": False,
                    "candidates": [
                        {
                            "candidate_id": "cand_1",
                            "title": "Mistral",
                            "target_ref": "situations/mistral",
                            "recommended_action": "review",
                        }
                    ],
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb review"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls == [("mcp_kb_engine_prod_lifecycle_review", {"target": "situations", "dry_run": True})]
    assert "Lifecycle Review" in adapter.sent[0]["text"]
    assert "Mistral" in adapter.sent[0]["text"]
    assert "Use /kb review to list proposals." not in adapter.sent[0]["text"]
    assert "/kb review 1" not in adapter.sent[0]["text"]


def test_kbqueue_review_item_can_be_opened_by_text_command(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": {
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

    result = hook(event=_event("/kb review 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert adapter.sent
    text = adapter.sent[0]["text"]
    assert "Review Item 1" in text
    assert "Keio University" in text
    assert "Target: accounts/keio-university" in text
    assert "Fallback text actions:" in text
    assert "/kb review reject 1" in text
    assert adapter.sent[0]["actions"] == []


def test_kb_review_queue_refuses_legacy_queue_fallback_without_review_inbox(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_queue_summary": {
                "result": {
                    "total": 1,
                    "items": [{"title": "Legacy Queue Item"}],
                }
            }
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb review queue"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls == [("mcp_kb_engine_prod_review_inbox", {"scope": "proposals", "limit": 5})]
    assert "KB data is not available yet." in adapter.sent[0]["text"]
    assert "mcp_kb_engine_prod_review_inbox" in adapter.sent[0]["text"]
    assert "Legacy Queue Item" not in adapter.sent[0]["text"]


def test_kbqueue_review_todo_item_shows_todo_native_actions(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": {
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

    result = hook(event=_event("/kb review 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    text = adapter.sent[0]["text"]
    assert "Fallback text actions:" in text
    assert "Complete TODO: /kb review complete 1" in text
    assert "Keep unchanged: /kb review keep 1" in text
    assert "Demote priority: /kb review demote 1" in text
    assert "Archive TODO: /kb review archive 1" in text
    assert "/kb review approve 1" not in text


def test_kbqueue_review_item_renders_descriptor_preview_and_confirm_buttons(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": {
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
                                    "advisory_guidance": _advisory_guidance(
                                        "Use advisory guidance to reason about Reject before previewing."
                                    ),
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

    result = hook(event=_event("/kb review 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert adapter.sent[0]["actions"]
    assert [action.label for action in adapter.sent[0]["actions"]] == ["Reject", "Ask LLM"]
    guidance_action = adapter.sent[0]["actions"][1]
    assert guidance_action.label == "Ask LLM"
    guidance_card = guidance_action.handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(guidance_card):
        guidance_card = asyncio.run(guidance_card)
    assert "KB Review LLM Guidance" in guidance_card["text"]
    assert "Use advisory guidance to reason about Reject" in guidance_card["text"]
    assert "kb.review_guidance" in guidance_card["text"]
    assert "Advisory output never confirms" in guidance_card["text"]

    preview_action = next(action for action in adapter.sent[0]["actions"] if action.label == "Reject")
    assert preview_action.label == "Reject"

    preview_card = preview_action.handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(preview_card):
        preview_card = asyncio.run(preview_card)

    assert "Review reject preview" in preview_card["text"]
    assert preview_card["actions"][0].label == "Confirm Reject"

    confirm_card = preview_card["actions"][0].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(confirm_card):
        confirm_card = asyncio.run(confirm_card)

    assert "Review Reject Applied" in confirm_card["text"]
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_queue_decision_preview"
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_queue_batch_decide_confirmed"
    assert ctx.calls[-1][1]["user_confirmation"]["confirmed"] is True
    assert ctx.calls[-1][1]["user_confirmation"]["preview_required"] is True


def test_kbqueue_descriptor_confirm_carries_lease_session_and_blocks_stale_result(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": {
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
                                    "confirmation_copy": "Confirm Reject after reviewing the preview.",
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
                    "plan": {"summary": "Reject 1 proposal."},
                    "preview_lease": {
                        "preview_lease_id": "lease_123",
                        "review_session_id": "review_session_123",
                        "cursor_id": "cursor_123",
                        "decision_scope": "all_viewed",
                        "proposal_ids": ["act_2"],
                        "expires_at": "2026-06-03T12:00:00Z",
                    },
                    "review_session": {
                        "review_session_id": "review_session_123",
                        "decision_scope": "all_viewed",
                        "cursor": {
                            "cursor_id": "cursor_123",
                            "displayed_count": 1,
                            "candidate_count": 5,
                        },
                    },
                }
            },
            "mcp_kb_engine_prod_queue_batch_decide_confirmed": {
                "result": {
                    "status": "preview_lease_stale",
                    "ok": False,
                    "reason": "Preview lease expired; refresh the queue.",
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb review 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    preview_action = next(action for action in adapter.sent[0]["actions"] if action.label == "Reject")

    preview_card = preview_action.handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(preview_card):
        preview_card = asyncio.run(preview_card)

    assert "Scope: Visible" in preview_card["text"]
    assert "Review session: 1 item(s) · 1 proposal(s)" in preview_card["text"]
    assert "lease_123" not in preview_card["text"]
    assert preview_card["actions"][0].metadata["preview_lease"] is True
    assert preview_card["actions"][0].metadata["review_session_id"] == "review_session_123"

    confirm_card = preview_card["actions"][0].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(confirm_card):
        confirm_card = asyncio.run(confirm_card)

    assert "Review Reject Blocked" in confirm_card["text"]
    assert "Preview lease expired" in confirm_card["text"]
    assert "Review Reject Applied" not in confirm_card["text"]
    assert [call[0] for call in ctx.calls] == [
        "mcp_kb_engine_prod_review_inbox",
        "mcp_kb_engine_prod_queue_decision_preview",
        "mcp_kb_engine_prod_queue_batch_decide_confirmed",
    ]
    confirm_args = ctx.calls[-1][1]
    assert "preview_lease" not in confirm_args
    assert "review_session" not in confirm_args
    assert "preview_session" not in confirm_args
    assert confirm_args["user_confirmation"]["preview_lease"]["preview_lease_id"] == "lease_123"
    assert confirm_args["session_id"] == "review_session_123"
    assert confirm_args["review_session_id"] == "review_session_123"
    assert confirm_args["cursor_id"] == "cursor_123"
    assert confirm_args["decision_scope"] == "all_viewed"
    assert confirm_args["user_confirmation"]["review_session_id"] == "review_session_123"


def test_kbqueue_confirm_advances_to_backend_next_review_card(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    current_descriptor = {
        "packet_type": "dashboard_action_descriptor",
        "schema_version": 2,
        "action_id": "review.entity_reject",
        "label": "Reject",
        "target_kind": "proposal_queue",
        "target_ref": "accounts/acme",
        "preview_tool": "queue.decision_preview",
        "confirm_tool": "queue.batch_decide_confirmed",
        "params": {"proposal_ids": ["act_acme"], "decision": "reject"},
        "dashboard_owned_write": False,
        "requires_canonical_tool": True,
        "confirmation_copy": "Confirm Reject after reviewing the preview.",
    }
    next_actions = [
        {
            **current_descriptor,
            "action_id": f"review.entity_{decision}",
            "label": label,
            "params": {"proposal_ids": ["act_globex"], "decision": decision},
            "target_ref": "accounts/globex",
        }
        for decision, label in (("approve", "Approve"), ("reject", "Reject"), ("archive", "Archive"))
    ]
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": {
                "result": {
                    "total": 2,
                    "items": [
                        {
                            "item_id": "accounts/acme",
                            "title": "Acme",
                            "kind": "proposal_entity",
                            "summary": "Current proposal.",
                            "raw": {"proposal_ids": ["act_acme"]},
                            "safe_actions": [current_descriptor],
                        }
                    ],
                }
            },
            "mcp_kb_engine_prod_queue_decision_preview": {
                "result": {
                    "status": "preview",
                    "ok": True,
                    "plan": {"summary": "Reject 1 proposal."},
                    "preview_lease": {
                        "preview_lease_id": "lease_acme",
                        "review_session_id": "session_acme",
                        "cursor_id": "cursor_acme",
                        "decision_scope": "explicit_ids",
                        "proposal_ids": ["act_acme"],
                    },
                    "review_session": {
                        "review_session_id": "session_acme",
                        "decision_scope": "explicit_ids",
                        "cursor": {"cursor_id": "cursor_acme", "displayed_count": 1, "candidate_count": 2},
                    },
                }
            },
            "mcp_kb_engine_prod_queue_batch_decide_confirmed": {
                "result": {
                    "receipt": {
                        "packet_type": "request.receipt",
                        "schema_version": 1,
                        "state": "applied",
                        "route": "queue.batch_decide_confirmed",
                        "saved": True,
                        "ok": True,
                        "affected_ids": ["act_acme"],
                        "reviewed_count": 1,
                        "confirmed_count": 1,
                        "safe_message": "Applied queue decision to 1 proposal(s).",
                        "next_review": {
                            "packet_type": "guided_kb_review_next",
                            "schema_version": 1,
                            "status": "ready",
                            "reason": "next_review_target_ready",
                            "source_review_session_id": "session_acme",
                            "source_cursor_id": "cursor_acme",
                            "source_preview_lease_id": "lease_acme",
                            "scope": {"affected_ids": ["act_acme"], "viewed_ids": ["act_acme"]},
                            "review_session": {
                                "packet_type": "guided_kb_review_session",
                                "review_session_id": "session_globex",
                                "decision_scope": "explicit_ids",
                                "cursor": {
                                    "cursor_id": "cursor_globex",
                                    "displayed_count": 1,
                                    "candidate_count": 1,
                                    "item_ids": ["act_globex"],
                                    "viewed_item_ids": ["act_globex"],
                                },
                            },
                            "target": {
                                "target_id": "accounts/globex",
                                "kind": "proposal_entity",
                                "title": "Globex",
                                "summary": "Next backend-supplied proposal.",
                                "entity_path": "accounts/globex",
                                "status": "pending",
                                "proposal_ids": ["act_globex"],
                                "proposal_count": 1,
                                "safe_actions": next_actions,
                            },
                        },
                    }
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb review 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    preview_action = next(action for action in adapter.sent[0]["actions"] if action.label == "Reject")
    preview_card = preview_action.handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(preview_card):
        preview_card = asyncio.run(preview_card)
    confirm_card = preview_card["actions"][0].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(confirm_card):
        confirm_card = asyncio.run(confirm_card)

    assert confirm_card["title"] == "KB Review"
    assert "Applied queue decision to 1 proposal(s)." in confirm_card["text"]
    assert "Next review from kb-engine:" in confirm_card["text"]
    assert "Globex" in confirm_card["text"]
    assert "Rail: Approve, Reject, Archive, Details" in confirm_card["text"]
    assert [action.label for action in confirm_card["actions"][:4]] == ["Approve", "Reject", "Archive", "Details"]

    next_reject = next(action for action in confirm_card["actions"] if action.label == "Reject")
    next_preview = next_reject.handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(next_preview):
        next_preview = asyncio.run(next_preview)

    next_preview_args = ctx.calls[-1][1]
    assert next_preview_args["proposal_ids"] == ["act_globex"]
    assert next_preview_args["review_session_id"] == "session_globex"
    assert next_preview_args["cursor_id"] == "cursor_globex"
    assert next_preview_args["decision_scope"] == "explicit_ids"
    assert next_preview_args["session_id"] == "session_globex"


def test_kbqueue_receipt_renders_changed_queue_next_review_as_refresh_required():
    from plugins import kb_journeys

    card = kb_journeys._render_request_receipt_packet(
        {
            "packet_type": "request.receipt",
            "state": "blocked",
            "route": "queue.batch_decide_confirmed",
            "saved": False,
            "ok": False,
            "safe_message": "Review confirmation blocked because the preview lease did not match.",
            "next_review": {
                "packet_type": "guided_kb_review_next",
                "status": "changed_queue",
                "reason": "preview_lease_mismatch:proposal_ids_hash",
                "target": {},
                "review_session": {},
            },
        },
        ctx=None,
        target="kb_engine_prod",
    )

    assert card["title"] == "KB Review Receipt"
    assert "Next review: refresh required" in card["text"]
    assert "proposal_ids_hash" in card["text"]
    assert "Next review from kb-engine" not in card["text"]
    assert card["actions"] == []


def test_request_receipt_renders_report_reference_fields():
    from plugins import kb_journeys

    card = kb_journeys._render_request_receipt_packet(
        {
            "packet_type": "request.receipt",
            "state": "answered",
            "route": "object.context",
            "saved": False,
            "object_family": "report",
            "report_refs": ["reports/2026-06-future-of-ai-for-synbio"],
            "related_object_refs": ["events/2026-06-future-trends-forum"],
            "safe_message": "Rendered object context without mutating durable KB state.",
        },
        ctx=None,
        target="kb_engine_prod",
    )

    assert card["title"] == "KB Request Receipt"
    assert "Object family: report" in card["text"]
    assert "Report refs: reports/2026-06-future-of-ai-for-synbio" in card["text"]
    assert "Related objects: events/2026-06-future-trends-forum" in card["text"]
    assert "Rendered object context" in card["text"]
    assert card["actions"] == []


def test_semantic_write_receipt_renders_compact_prod_and_publication_statuses():
    from plugins import kb_journeys

    packet = {
        "packet_type": "semantic_write_receipt",
        "schema_version": 1,
        "prod_write_status": "prod_write_applied",
        "publication": {"status": "publication_applied"},
        "local_reconciliation": {"status": "offline_receipt_pending_reconciliation"},
        "transaction_id": "semantic-txn-123",
        "changed_paths": [
            "situations/2026-06-acme-launch/state.md",
            "https://example.invalid/private-source",
            "/Users/acosta/secrets/source.md",
        ],
        "object_ids": [
            "situations/2026-06-acme-launch",
            "anthony@example.com",
            "acct:private-login",
        ],
        "operation_ids": [
            "object.semantic_update_preview",
            "control.apply_confirmed",
        ],
        "source_body": "private source body should not render",
        "login": "anthony@example.com",
        "account": "acct:private-login",
        "token": "sk-secret-token",
    }

    card = kb_journeys._render_supported_result_packet({"result": packet})

    assert card["title"] == "Semantic Write Receipt"
    text = card["text"]
    assert "Semantic Write Receipt" in text
    assert "Prod write: prod_write_applied" in text
    assert "Publication: publication_applied" in text
    assert "Reconciliation: offline_receipt_pending_reconciliation" in text
    assert "Transaction: semantic-txn-123" in text
    assert "Changed paths: 1" in text
    assert "situations/2026-06-acme-launch/state.md" in text
    assert "Objects: situations/2026-06-acme-launch" in text
    assert "Operations: object.semantic_update_preview, control.apply_confirmed" in text
    assert "private source body" not in text
    assert "anthony@example.com" not in text
    assert "acct:private-login" not in text
    assert "example.invalid" not in text
    assert "/Users/acosta" not in text
    assert "sk-secret-token" not in text
    assert card["actions"] == []


def test_semantic_write_receipt_supports_offline_status_aliases():
    from plugins import kb_journeys

    card = kb_journeys._render_supported_result_packet(
        {
            "receipt": {
                "packet_type": "semantic_write_through_receipt",
                "status": "offline_not_prod_publication",
                "publication_status": "manual_publication_expected",
                "reconciliation_status": "offline_receipt_pending_reconciliation",
                "object_id": "accounts/mistral",
                "operation_id": "semantic.update",
                "transaction": {"id": "offline-txn-7"},
            }
        }
    )

    text = card["text"]
    assert "Prod write: offline_not_prod_publication" in text
    assert "Publication: manual_publication_expected" in text
    assert "Reconciliation: offline_receipt_pending_reconciliation" in text
    assert "Transaction: offline-txn-7" in text
    assert "Objects: accounts/mistral" in text
    assert "Operations: semantic.update" in text


def test_semantic_write_shadow_preview_renders_manifest_authority_and_receipt_statuses():
    from plugins import kb_journeys

    card = kb_journeys._render_supported_result_packet(
        {
            "result": {
                "kind": "semantic_write_through.shadow_preview",
                "shadow_mode": True,
                "mutation_performed": False,
                "raw_private_source_text_copied": False,
                "durable_write_families": ["situation.lifecycle", "report.provenance"],
                "classification": {
                    "family": "lifecycle",
                    "intent": "semantic_update",
                    "confidence": "high",
                },
                "source_manifest": {
                    "sources": [
                        {"kind": "inbox", "source_ref": "source:closeout", "title": "Closeout note"},
                        {"kind": "doc", "source_ref": "/Users/acosta/private/raw.md", "title": "Private raw"},
                    ]
                },
                "surface_authority_map": {
                    "Telegram": "read_surface",
                    "kb_engine_prod": {"authority": "durable_write_authority"},
                },
                "semantic_preview": {
                    "status": "ready",
                    "summary": "Would update lifecycle state from the canonical preview.",
                    "object_refs": ["situations/2026-06-acme-launch", "anthony@example.com"],
                    "operations": [{"operation_id": "object.semantic_update_preview"}],
                    "source_body": "private source body should not render",
                },
                "comparison": {
                    "status": "shadow_matches_preview",
                    "differences": ["candidate action wording differs"],
                    "summary": "Shadow and canonical preview agree on durable family.",
                },
                "receipt_status": {
                    "prod_write_status": "shadow_not_applied",
                    "publication": {"status": "not_published"},
                    "reconciliation_status": "not_needed",
                    "shadow_receipt_status": "shadow_preview_recorded",
                },
                "candidate_actions": [
                    {"label": "Preview canonical write", "action_id": "object.semantic_update_preview"}
                ],
                "token": "sk-secret-token",
            }
        }
    )

    assert card["title"] == "Semantic Write Shadow Preview"
    text = card["text"]
    assert "Shadow mode: yes" in text
    assert "Mutation: none" in text
    assert "Raw private source text copied: no" in text
    assert "Durable write families: situation.lifecycle, report.provenance" in text
    assert "Classification: lifecycle / semantic_update / high" in text
    assert "Sources: 2" in text
    assert "inbox - source:closeout - Closeout note" in text
    assert "Authority map: Telegram=read_surface; kb_engine_prod=durable_write_authority" in text
    assert "Preview status: ready" in text
    assert "Preview objects: situations/2026-06-acme-launch" in text
    assert "Preview operations: object.semantic_update_preview" in text
    assert "Comparison: shadow_matches_preview" in text
    assert "Differences: 1" in text
    assert "Prod write: shadow_not_applied" in text
    assert "Publication: not_published" in text
    assert "Reconciliation: not_needed" in text
    assert "Shadow receipt: shadow_preview_recorded" in text
    assert "Candidate actions: Preview canonical write, object.semantic_update_preview" in text
    assert "No durable write has been made." in text
    assert "private source body" not in text
    assert "anthony@example.com" not in text
    assert "/Users/acosta" not in text
    assert "sk-secret-token" not in text
    assert card["actions"] == []


def test_kbqueue_decision_can_be_previewed_and_confirmed_by_text_command(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": {
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
                    "receipt": {
                        "packet_type": "request.receipt",
                        "state": "applied",
                        "route": "queue.batch_decide_confirmed",
                        "saved": True,
                        "ok": True,
                        "receipt_id": "qrcpt-1",
                        "affected_ids": ["act_2"],
                        "reviewed_count": 1,
                        "confirmed_count": 1,
                        "transaction_id": "control:abc",
                        "restore_available": True,
                        "restore_hint": {
                            "preview_tool": "queue.restore_preview",
                            "confirm_tool": "queue.restore_confirmed",
                            "transaction_id": "control:abc",
                            "receipt_id": "qrcpt-1",
                            "proposal_ids": ["act_2"],
                        },
                        "safe_message": "Applied queue decision to 1 proposal(s).",
                    },
                    "publication": {"status": "manual"},
                    "git": {
                        "before": {"branch": "main", "changed_count": 0},
                        "after": {"branch": "main", "changed_count": 3, "changes": ["a", "b", "c"]},
                    },
                }
            },
            "mcp_kb_engine_prod_queue_restore_preview": {
                "result": {
                    "status": "noop",
                    "ok": True,
                    "restorable_ids": ["act_2"],
                    "incompatible_ids": [],
                    "already_restored_ids": [],
                    "review_session": {
                        "review_session_id": "restore-session-1",
                        "cursor": {"cursor_id": "restore-cursor-1", "displayed_count": 1, "candidate_count": 1},
                    },
                    "preview_lease": {
                        "preview_lease_id": "restore-lease-1",
                        "review_session_id": "restore-session-1",
                        "cursor_id": "restore-cursor-1",
                        "decision_scope": "explicit_ids",
                        "proposal_ids": ["act_2"],
                    },
                }
            },
            "mcp_kb_engine_prod_queue_restore_confirmed": {
                "result": {
                    "status": "applied",
                    "ok": True,
                    "receipt": {
                        "packet_type": "request.receipt",
                        "state": "applied",
                        "route": "queue.restore_confirmed",
                        "saved": True,
                        "ok": True,
                        "receipt_id": "qrcpt-restore-1",
                        "affected_ids": ["act_2"],
                        "restored_ids": ["act_2"],
                        "transaction_id": "control:restore",
                        "safe_message": "Restored 1 proposal(s) to the review queue.",
                    },
                }
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    preview = hook(event=_event("/kb review reject 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert preview == {"action": "skip", "reason": "kb_journeys"}
    assert "Review reject preview" in adapter.sent[0]["text"]
    assert "Confirm with the button below" in adapter.sent[0]["text"]
    assert "Text fallback: /kb review reject 1 confirm" in adapter.sent[0]["text"]
    assert adapter.sent[0]["actions"][0].label == "Confirm Reject"

    applied = hook(event=_event("/kb review reject 1 confirm"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert applied == {"action": "skip", "reason": "kb_journeys"}
    applied_text = adapter.sent[1]["text"]
    assert "KB Review Receipt" in applied_text
    assert "Applied queue decision to 1 proposal(s)." in applied_text
    assert "Affected ids: act_2" in applied_text
    assert "Counts: 1 reviewed · 1 confirmed" in applied_text
    assert adapter.sent[1]["actions"][0].label == "Preview Restore"
    restore_preview = adapter.sent[1]["actions"][0].handler(SimpleNamespace(actor_id="777", actor_name="Ada"))
    assert "Review restore preview" in restore_preview["text"]
    assert "Restorable ids: act_2" in restore_preview["text"]
    assert restore_preview["actions"][0].label == "Confirm Restore"
    restore_result = restore_preview["actions"][0].handler(SimpleNamespace(actor_id="777", actor_name="Ada"))
    assert "Restored 1 proposal(s) to the review queue." in restore_result["text"]
    assert "Restored ids: act_2" in restore_result["text"]
    assert "{'before':" not in applied_text
    assert ctx.calls[-4][0] == "mcp_kb_engine_prod_queue_decision_preview"
    assert ctx.calls[-4][1]["proposal_ids"] == ["act_2"]
    assert ctx.calls[-3][0] == "mcp_kb_engine_prod_queue_batch_decide_confirmed"
    assert ctx.calls[-3][1]["user_confirmation"]["confirmed"] is True
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_queue_restore_preview"
    assert ctx.calls[-2][1] == {
        "transaction_id": "control:abc",
        "receipt_id": "qrcpt-1",
        "proposal_ids": ["act_2"],
    }
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_queue_restore_confirmed"
    assert ctx.calls[-1][1]["review_session_id"] == "restore-session-1"
    assert ctx.calls[-1][1]["cursor_id"] == "restore-cursor-1"
    assert ctx.calls[-1][1]["user_confirmation"]["preview_lease"]["preview_lease_id"] == "restore-lease-1"
    assert "preview_lease" not in ctx.calls[-1][1]


def test_kbqueue_text_confirm_uses_preview_scope_when_queue_shifts(monkeypatch, tmp_path):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = SequencedFakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": [
                {
                    "result": {
                        "total": 2,
                        "items": [
                            {
                                "item_id": "accounts/eli-lilly",
                                "title": "Eli Lilly",
                                "kind": "proposal_entity",
                                "preview": "Project Nova situation proposal.",
                                "raw": {"proposal_ids": ["act_eli"]},
                            }
                        ],
                    }
                },
                {
                    "result": {
                        "total": 1,
                        "items": [
                            {
                                "item_id": "accounts/atomic-ai",
                                "title": "Atomic AI",
                                "kind": "proposal_entity",
                                "preview": "Different proposal now occupies index 1.",
                                "raw": {"proposal_ids": ["act_atomic"]},
                            }
                        ],
                    }
                },
            ],
            "mcp_kb_engine_prod_queue_decision_preview": [
                {"result": {"status": "preview", "ok": True, "plan": {"summary": "Reject Eli."}}},
                {"result": {"status": "preview", "ok": True, "plan": {"summary": "Reject Eli after re-preview."}}},
            ],
            "mcp_kb_engine_prod_queue_batch_decide_confirmed": [
                {"result": {"status": "applied", "ok": True, "publication": {"status": "manual"}}}
            ],
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)
    store = FakeSessionStore("session-shift")

    preview = hook(event=_event("/kb review reject 1"), gateway=_authorized_gateway(adapter), session_store=store)
    _drain_scheduled_tasks()
    applied = hook(event=_event("/kb review reject 1 confirm"), gateway=_authorized_gateway(adapter), session_store=store)
    _drain_scheduled_tasks()

    assert preview == {"action": "skip", "reason": "kb_journeys"}
    assert applied == {"action": "skip", "reason": "kb_journeys"}
    assert "Eli Lilly" in adapter.sent[1]["text"]
    assert "Atomic AI" not in adapter.sent[1]["text"]
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_queue_decision_preview"
    assert ctx.calls[-2][1]["proposal_ids"] == ["act_eli"]
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_queue_batch_decide_confirmed"
    assert ctx.calls[-1][1]["proposal_ids"] == ["act_eli"]
    assert ctx.calls[-1][1]["user_confirmation"]["proposal_ids"] == ["act_eli"]


def test_kbqueue_reject_all_previews_visible_window_only(monkeypatch, tmp_path):
    from plugins import kb_journeys
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": {
                "result": {
                    "total": 11,
                    "items": [
                        {"item_id": "a", "title": "Atomic AI", "raw": {"proposal_ids": ["act_atomic"]}},
                        {"item_id": "n", "title": "Nous Research", "raw": {"proposal_ids": ["act_nous"]}},
                        {"item_id": "p", "title": "Palantir", "raw": {"proposal_ids": ["act_palantir"]}},
                        {"item_id": "s", "title": "Stellantis", "raw": {"proposal_ids": ["act_stellantis"]}},
                        {"item_id": "t", "title": "TSMC", "raw": {"proposal_ids": ["act_tsmc"]}},
                    ],
                }
            },
            "mcp_kb_engine_prod_queue_decision_preview": {
                "result": {"status": "preview", "ok": True, "plan": {"summary": "Reject 5 shown proposals."}}
            },
            "mcp_kb_engine_prod_queue_batch_decide_confirmed": {
                "result": {"status": "applied", "ok": True}
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)
    store = FakeSessionStore("session-visible-all")

    listed = hook(event=_event("/kb queue"), gateway=_authorized_gateway(adapter), session_store=store)
    _drain_scheduled_tasks()
    preview = hook(event=_event("Reject all"), gateway=_authorized_gateway(adapter), session_store=store)
    _drain_scheduled_tasks()

    assert listed == {"action": "skip", "reason": "kb_journeys"}
    assert kb_journeys.scoped_mcp_tool_allowlist_for_message(
        session_id="session-visible-all",
        message="Reject all",
    ) == {"mcp_kb_engine_prod_queue_decision_preview"}
    assert preview == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_queue_decision_preview"
    assert ctx.calls[-1][1]["proposal_ids"] == [
        "act_atomic",
        "act_nous",
        "act_palantir",
        "act_stellantis",
        "act_tsmc",
    ]
    assert ctx.calls[-1][1]["decision_scope"] == "all_viewed"
    assert ctx.calls[-1][1]["candidate_count"] == 11
    assert ctx.calls[-1][1]["displayed_count"] == 5
    assert "Scope: visible Telegram review window only" in adapter.sent[-1]["text"]
    assert "To apply: /kb review reject 1,2,3,4,5 confirm" in adapter.sent[-1]["text"]
    assert "mcp_kb_engine_prod_queue_batch_decide_confirmed" not in [call[0] for call in ctx.calls]


def test_kbqueue_reject_all_without_visible_scope_does_not_fall_through(monkeypatch, tmp_path):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext({})
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(
        event=_event("Reject all"),
        gateway=_authorized_gateway(adapter),
        session_store=FakeSessionStore("session-empty"),
    )
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert ctx.calls == []
    assert "Run /kb review first" in adapter.sent[0]["text"]


def test_kbqueue_visible_scope_all_phrases_are_narrow_decisions():
    from plugins import kb_journeys

    assert kb_journeys._visible_scope_all_decision("Reject the five proposals you showed me") == "reject"
    assert kb_journeys._visible_scope_all_decision("Approve everything visible") == "approve"
    assert kb_journeys._visible_scope_all_decision("Review proposals") == ""


def test_kbqueue_visible_scope_without_timestamp_expires(monkeypatch, tmp_path):
    from plugins import kb_journeys

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    path = kb_journeys._queue_scope_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "session-stale": {
                    "visible": {
                        "kind": "visible_queue_window",
                        "selection": [
                            {
                                "index": 1,
                                "title": "Old Proposal",
                                "proposal_ids": ["act_old"],
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert kb_journeys._get_visible_queue_scope("session-stale") == []
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "visible" not in stored["session-stale"]


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
            "mcp_kb_engine_prod_review_inbox": {
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
            "note": "Previewed from Telegram iterative review reply for GTC Taipei 2026",
        },
    )
    assert len(ctx.calls) == 1
    assert "act_huang1" not in json.dumps(ctx.calls)
    assert adapter.sent
    assert "GTC Taipei 2026" in adapter.sent[0]["text"]
    assert "To apply: /kb review archive 1 confirm" in adapter.sent[0]["text"]


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
            "mcp_kb_engine_prod_review_inbox": {"result": {"counts": {"proposals": 0}, "items": []}},
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
    assert len(ctx.calls) == 1
    assert "Hitachi" in adapter.sent[0]["text"]
    assert "To apply: /kb review reject 1 confirm" in adapter.sent[0]["text"]


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
    ) == {"mcp_kb_engine_prod_queue_decision_preview"}
    assert kb_journeys.scoped_mcp_tool_allowlist_for_message(
        session_id="session-posture",
        message="keep me posted",
    ) == set()


def test_kbqueue_todo_complete_decision_uses_queue_decision_contract(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": {
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

    preview = hook(event=_event("/kb review complete 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()
    confirmed = hook(event=_event("/kb review complete 1 confirm"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert preview == {"action": "skip", "reason": "kb_journeys"}
    assert confirmed == {"action": "skip", "reason": "kb_journeys"}
    assert "Review complete preview" in adapter.sent[0]["text"]
    assert "Confirm with the button below" in adapter.sent[0]["text"]
    assert "Text fallback: /kb review complete 1 confirm" in adapter.sent[0]["text"]
    assert "Review Complete Applied" in adapter.sent[1]["text"]
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_queue_decision_preview"
    assert ctx.calls[-2][1]["decision"] == "complete"
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_queue_batch_decide_confirmed"
    assert ctx.calls[-1][1]["decision"] == "complete"


def test_kbqueue_decision_supports_batch_text_commands_and_legacy_alias(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": {
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

    preview = hook(event=_event("/kbqueue reject 1, 2"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()
    result = hook(event=_event("/kbqueue reject 1, 2 confirm"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert preview == {"action": "skip", "reason": "kb_journeys"}
    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert "Review Reject Applied" in adapter.sent[1]["text"]
    assert "1. Keio University" in adapter.sent[1]["text"]
    assert "2. Mistral" in adapter.sent[1]["text"]
    assert ctx.calls[-2][0] == "mcp_kb_engine_prod_queue_decision_preview"
    assert ctx.calls[-2][1]["proposal_ids"] == ["act_1", "act_2", "act_3"]
    assert ctx.calls[-1][0] == "mcp_kb_engine_prod_queue_batch_decide_confirmed"
    assert ctx.calls[-1][1]["proposal_ids"] == ["act_1", "act_2", "act_3"]


def test_queue_preview_failure_does_not_offer_confirm(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_review_inbox": {
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

    result = hook(event=_event("/kb review approve 1"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert "Review approve preview failed" in adapter.sent[0]["text"]
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
    assert ctx.calls == [
        ("mcp_kb_engine_prod_closeout_packet", {"limit": 5}),
        ("mcp_kb_engine_prod_publication_preview_commit", {"message": "Publish KB update"}),
    ]
    text = adapter.sent[0]["text"]
    assert "KB Publish Preview" in text
    assert "Changed paths: 2" in text
    assert "accounts/mistral/state.md" in text
    assert "To publish: /kb publish confirm" in text
    assert "No commit or push has been made." in text


def test_kb_publish_renders_descriptor_confirm_action_button(monkeypatch):
    from plugins.kb_journeys import build_pre_gateway_dispatch_hook

    monkeypatch.setenv("HERMES_KB_MCP_TARGET", "kb-engine-prod")
    ctx = FakeContext(
        {
            "mcp_kb_engine_prod_closeout_packet": {
                "result": {
                    "packet_type": "closeout.packet",
                    "contract_id": "kb.closeout.operation.v1",
                    "publication": {
                        "status": "dirty",
                        "manual_publication_expected": True,
                        "changed_count": 1,
                        "reason": "KB workspace has unpublished review-session changes.",
                    },
                    "action_descriptors": [
                        {
                            "packet_type": "dashboard_action_descriptor",
                            "schema_version": 2,
                            "action_id": "publication.preflight",
                            "label": "Run publication preflight",
                            "method": "publication.preflight",
                            "mutation": "read_only",
                            "target_kind": "publication",
                            "target_ref": "publication",
                            "preview_tool": "publication.preflight",
                            "confirm_tool": "",
                            "params": {},
                            "dashboard_owned_write": False,
                            "requires_canonical_tool": True,
                        },
                        {
                            "packet_type": "dashboard_action_descriptor",
                            "schema_version": 2,
                            "action_id": "publication.preview_commit",
                            "label": "Preview publication commit",
                            "method": "publication.preview_commit",
                            "mutation": "read_only",
                            "target_kind": "publication",
                            "target_ref": "publication",
                            "preview_tool": "publication.preview_commit",
                            "confirm_tool": "",
                            "params": {"message": "Publish KB update"},
                            "dashboard_owned_write": False,
                            "requires_canonical_tool": True,
                        },
                        {
                            "packet_type": "dashboard_action_descriptor",
                            "schema_version": 2,
                            "action_id": "publication.commit_confirmed",
                            "label": "Confirm publication commit",
                            "method": "publication.commit_confirmed",
                            "mutation": "workspace_write",
                            "target_kind": "publication",
                            "target_ref": "publication",
                            "preview_tool": "publication.preview_commit",
                            "confirm_tool": "publication.commit_confirmed",
                            "params": {"message": "Publish KB update"},
                            "dashboard_owned_write": False,
                            "requires_canonical_tool": True,
                            "confirmation_copy": "Confirm publication only after reviewing the commit preview.",
                        },
                    ],
                }
            },
            "mcp_kb_engine_prod_publication_preview_commit": {
                "result": {
                    "status": "ready",
                    "ok": True,
                    "message": "Publish KB update",
                    "changed_paths": ["accounts/mistral/state.md"],
                    "git": {"branch": "main", "head": "abc123", "upstream": "origin/main"},
                }
            },
            "mcp_kb_engine_prod_publication_preflight": {
                "result": {
                    "packet_type": "publication_observation",
                    "schema_version": 1,
                    "status": "dirty",
                    "publication_state": "dirty",
                    "changed_count": 1,
                    "changed_paths": ["accounts/mistral/state.md"],
                    "secret_values_exposed": False,
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
                "result": {"status": "pushed", "ok": True, "publication": {"status": "pushed"}},
            },
        }
    )
    adapter = FakeKbActionsAdapter()
    hook = build_pre_gateway_dispatch_hook(ctx)

    result = hook(event=_event("/kb publish"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert result == {"action": "skip", "reason": "kb_journeys"}
    assert adapter.sent[0]["actions"]
    text = adapter.sent[0]["text"]
    assert "Decision Card: Publication" in text
    assert "Manual publication expected." in text
    assert "KB workspace has unpublished review-session changes." in text
    assert [action.label for action in adapter.sent[0]["actions"]] == ["Run Preflight", "Confirm Publish"]

    preflight_card = adapter.sent[0]["actions"][0].handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(preflight_card):
        preflight_card = asyncio.run(preflight_card)

    assert "Publication Observation" in preflight_card["text"]
    assert "Changed paths: 1" in preflight_card["text"]

    confirm_action = adapter.sent[0]["actions"][1]
    assert confirm_action.label == "Confirm Publish"

    confirm_card = confirm_action.handler(SimpleNamespace(actor_id="user-1", actor_name="tester"))
    if asyncio.iscoroutine(confirm_card):
        confirm_card = asyncio.run(confirm_card)

    assert "KB Published" in confirm_card["text"]
    assert [call[0] for call in ctx.calls] == [
        "mcp_kb_engine_prod_closeout_packet",
        "mcp_kb_engine_prod_publication_preview_commit",
        "mcp_kb_engine_prod_publication_preflight",
        "mcp_kb_engine_prod_publication_preview_commit",
        "mcp_kb_engine_prod_publication_commit_confirmed",
        "mcp_kb_engine_prod_publication_push_confirmed",
    ]
    commit_args = ctx.calls[-2][1]
    assert commit_args["user_confirmation"]["confirmed"] is True
    assert commit_args["user_confirmation"]["preview_required"] is True
    assert commit_args["expected_changed_paths"] == ["accounts/mistral/state.md"]


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
        "mcp_kb_engine_prod_closeout_packet",
        "mcp_kb_engine_prod_publication_preview_commit",
        "mcp_kb_engine_prod_publication_commit_confirmed",
        "mcp_kb_engine_prod_publication_push_confirmed",
    ]
    commit_args = ctx.calls[2][1]
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
    assert ctx.calls == [
        ("mcp_kb_engine_prod_closeout_packet", {"limit": 5}),
        ("mcp_kb_engine_prod_publication_preview_commit", {"message": "Publish KB update"}),
    ]
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
                    "outcome": {"family": "workflow_start_plan"},
                    "receipt": {
                        "state": "ready_to_confirm",
                        "durable_effect": "none",
                        "next_step": "Confirm through workflow.start_confirmed.",
                        "llm_invoked_by_read_surface": False,
                    },
                }
            },
            "mcp_kb_engine_prod_workflow_start_confirmed": {
                "result": {
                    "status": "started",
                    "started": True,
                    "run": {"run_id": "gen-123", "workflow_id": "update_kb"},
                    "followthrough_contract": {"recommended_next_action": "watch_until_terminal"},
                    "receipt": {
                        "state": "workflow_running",
                        "durable_effect": "workflow_run",
                        "llm_invoked_by_read_surface": False,
                    },
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
    assert "Receipt: ready_to_confirm" in adapter.sent[0]["text"]
    assert "Outcome: workflow_start_plan" in adapter.sent[0]["text"]
    assert "To start: /kb run kb sync confirm" in adapter.sent[0]["text"]
    assert adapter.sent[0]["actions"] == []

    started = hook(event=_event("/kb run kb sync confirm"), gateway=_authorized_gateway(adapter), session_store=None)
    _drain_scheduled_tasks()

    assert started == {"action": "skip", "reason": "kb_journeys"}
    assert "Workflow start result" in adapter.sent[1]["text"]
    assert "Receipt: workflow_running" in adapter.sent[1]["text"]
    assert "Effect: workflow_run" in adapter.sent[1]["text"]
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
            "mcp_kb_engine_prod_status_proof": {
                "result": {
                    "kind": "kb_status_proof_packet",
                    "status": "ready",
                    "workspace": {"lane": "prod"},
                    "runtime": {"version": "0.17.68"},
                    "transport": {"status": "open"},
                    "publication": {"status": "clean"},
                    "review": {"pending_count": 3},
                    "sync": {"status": "idle"},
                    "dirty_scope": {"dirty_path_count": 0},
                    "privacy": {"secret_values_exposed": False, "raw_source_bodies_exposed": False},
                    "next_action": {"command": "/kb review"},
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
        ("mcp_kb_engine_prod_status_proof", {}),
        ("mcp_kb_engine_prod_provider_status", {}),
    ]
    text = adapter.sent[0]["text"]
    assert "Request: /kb status" in text
    assert "Lane: prod" in text
    assert "Runtime: 0.17.68" in text
    assert "Transport: open" in text
    assert "Pending review: 3" in text
    assert "Sync: idle" in text
    assert "Dirty: 0 dirty" in text
    assert "Privacy: ok" in text
    assert "Next: /kb review" in text
    assert "Hermes reasoning: xhigh" in text
    assert "KB reasoning: low" in text
    assert "KB model: plugin:openai-compatible / gpt-5.5 / low" in text


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
