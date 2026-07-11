"""Recovery decision for a routed LM Studio instance that vanished after ensure.

When Hermes stacks its own LM Studio instance (``lmstudio_unload_policy:
never``) it addresses requests to that instance's id. If the instance
disappears mid-session (server restart, manual unload) a spec-compliant
server or proxy answers with a 404. The predicate below recognizes that
case so the loop can clear the stale claim and retry with the base name.

Note: the LM Studio builds tested in this project do NOT 404 a vanished
instance — they silently strip the suffix and reroute to a resident model.
This recovery is therefore defense-in-depth for compliant servers/proxies;
the primary staleness guard is the catalog validation in
``ensure_lmstudio_model_loaded``.
"""

from __future__ import annotations

from agent.conversation_loop import _should_recover_stale_routed_lmstudio


def test_routed_404_with_substituted_model_triggers_recovery():
    assert _should_recover_stale_routed_lmstudio(
        status_code=404,
        sent_model="qwen/qwen3.6-35b-a3b:2",
        agent_model="qwen/qwen3.6-35b-a3b",
    )


def test_bare_model_404_is_not_a_routing_failure():
    # No substitution happened (sent == configured), so a 404 is a plain
    # model error, not a stale-instance one — leave it to normal handling.
    assert not _should_recover_stale_routed_lmstudio(
        status_code=404,
        sent_model="qwen/qwen3.6-35b-a3b",
        agent_model="qwen/qwen3.6-35b-a3b",
    )


def test_non_404_status_does_not_trigger_recovery():
    assert not _should_recover_stale_routed_lmstudio(
        status_code=500,
        sent_model="qwen/qwen3.6-35b-a3b:2",
        agent_model="qwen/qwen3.6-35b-a3b",
    )


def test_missing_models_do_not_trigger_recovery():
    assert not _should_recover_stale_routed_lmstudio(
        status_code=404, sent_model="", agent_model="qwen/qwen3.6-35b-a3b"
    )
    assert not _should_recover_stale_routed_lmstudio(
        status_code=404, sent_model="qwen/qwen3.6-35b-a3b:2", agent_model=""
    )
    assert not _should_recover_stale_routed_lmstudio(
        status_code=None,
        sent_model="qwen/qwen3.6-35b-a3b:2",
        agent_model="qwen/qwen3.6-35b-a3b",
    )
