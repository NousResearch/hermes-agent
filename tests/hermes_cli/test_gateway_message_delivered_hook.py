from hermes_cli.plugins import PluginManager, VALID_HOOKS


def test_gateway_message_delivered_is_a_valid_hook_with_an_exact_payload():
    payload = {
        "source": "cron",
        "execution_id": "exec-1",
        "job_id": "job-1",
        "platform": "telegram",
        "chat_id": "123",
        "thread_id": None,
        "message_id": "456",
    }
    manager = PluginManager()
    manager._hooks["gateway_message_delivered"] = [lambda **kwargs: kwargs]

    assert "gateway_message_delivered" in VALID_HOOKS
    assert manager.invoke_hook("gateway_message_delivered", **payload) == [payload]
