"""Regression: the system/developer role name must follow the target model's dialect.

A client speaking the Responses dialect (Codex CLI, newer Cursor/Open WebUI) sends
system-level instructions as ``role="developer"``. Providers that predate that
spelling — DeepSeek among them — reject the request wholesale:

    422 ... messages[1].role: unknown variant `developer`,
              expected one of `system`, `user`, `assistant`, `tool`, ...

Because the transport forwarded the inbound role verbatim, the primary model failed
on *every* turn and the agent silently fell back to another provider — the reply
looked fine while a different (slower, pricier) model produced it.

Invariant: for a model that does not support ``developer``, no outbound message may
carry that role; for one that does, the system prompt is promoted to it. Content and
message count never change.
"""

from __future__ import annotations

from agent.transports.chat_completions import _swap_developer_role

DEVELOPER_DIALECT = [
    {"role": "system", "content": "hermes core prompt"},
    {"role": "developer", "content": "client instructions"},
    {"role": "user", "content": "hello"},
]


def test_no_developer_role_survives_towards_a_model_that_rejects_it():
    out = _swap_developer_role(DEVELOPER_DIALECT, "deepseek-flash")

    assert [m["role"] for m in out] == ["system", "system", "user"]
    # the fold-back is a rename only: nothing is dropped or rewritten
    assert len(out) == len(DEVELOPER_DIALECT)
    assert out[1]["content"] == "client instructions"


def test_system_prompt_is_promoted_for_models_that_take_developer():
    out = _swap_developer_role(list(DEVELOPER_DIALECT[:1]) + DEVELOPER_DIALECT[2:], "gpt-5.6-sol")

    assert out[0]["role"] == "developer"
    assert out[1]["role"] == "user"


def test_messages_are_untouched_when_no_developer_role_is_present():
    plain = [{"role": "system", "content": "core"}, {"role": "user", "content": "hi"}]

    assert _swap_developer_role(plain, "deepseek-flash") == plain
    assert _swap_developer_role([], "deepseek-flash") == []
