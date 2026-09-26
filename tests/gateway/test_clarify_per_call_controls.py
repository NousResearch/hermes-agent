import json


def test_gateway_callback_accepts_per_call_controls():
    from tools.clarify_tool import clarify_tool
    seen = {}

    def cb(question, choices, timeout=None, auto_select=True):
        seen.update(timeout=timeout, auto_select=auto_select)
        return "answer"

    result = json.loads(clarify_tool("Q?", choices=["a", "b"], callback=cb,
                                    timeout=2, auto_select=False))
    assert result["user_response"] == "answer"
    assert seen == {"timeout": 2, "auto_select": False}


def test_gateway_timeout_false_returns_timeout_result():
    from tools.clarify_tool import clarify_tool, TIMEOUT_RESPONSE

    def cb(question, choices, timeout=None, auto_select=True):
        return TIMEOUT_RESPONSE

    result = json.loads(clarify_tool("Q?", choices=["a", "b"], callback=cb,
                                    timeout=1, auto_select=False))
    assert result["user_response"] == TIMEOUT_RESPONSE
