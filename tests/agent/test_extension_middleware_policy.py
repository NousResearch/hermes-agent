from unittest.mock import Mock, patch


def test_reduced_authority_skips_request_and_execution_middleware():
    from agent.extension_middleware_policy import (
        apply_agent_llm_request_middleware,
        run_agent_llm_execution_middleware,
    )

    agent = Mock(_skip_extension_middleware=True)
    payload = {"messages": [{"role": "user", "content": "private text"}]}
    perform = Mock(return_value="provider-response")

    with patch("hermes_cli.middleware.apply_llm_request_middleware") as request_middleware, \
         patch("hermes_cli.middleware.run_llm_execution_middleware") as execution_middleware:
        request_result = apply_agent_llm_request_middleware(agent, payload)
        response = run_agent_llm_execution_middleware(agent, payload, perform)

    assert request_result.payload is payload
    assert request_result.original_payload == payload
    assert request_result.original_payload is not payload
    assert request_result.trace == []
    assert response == "provider-response"
    perform.assert_called_once_with(payload)
    request_middleware.assert_not_called()
    execution_middleware.assert_not_called()


def test_normal_authority_uses_configured_middleware():
    from agent.extension_middleware_policy import (
        apply_agent_llm_request_middleware,
        run_agent_llm_execution_middleware,
    )

    agent = Mock(_skip_extension_middleware=False)
    payload = {"messages": []}
    perform = Mock()
    request_result = object()

    with patch(
        "hermes_cli.middleware.apply_llm_request_middleware",
        return_value=request_result,
    ) as request_middleware, patch(
        "hermes_cli.middleware.run_llm_execution_middleware",
        return_value="wrapped-response",
    ) as execution_middleware:
        assert apply_agent_llm_request_middleware(agent, payload, task_id="task") is request_result
        assert run_agent_llm_execution_middleware(
            agent,
            payload,
            perform,
            task_id="task",
        ) == "wrapped-response"

    request_middleware.assert_called_once_with(payload, task_id="task")
    execution_middleware.assert_called_once_with(payload, perform, task_id="task")
    perform.assert_not_called()
