"""The stored API-turn result keeps inline media intact while secrets are still masked (#160176)."""

from gateway.session_results import prepare_result

# A JPEG data URL whose base64 happens to contain a JWT-looking run (``eyJ...``).
IMAGE = "data:image/jpeg;base64,/9j/4AAQ" + "A" * 40 + "+VeyJNurQWERTYqwertyQWERTYgqvU+lNORnIwKkmG04U="
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XNy_IOH0d3NLQxpCw8Mk"


def _stored(messages, final_response="ok"):
    row = {"payload": {"text": "look at this photo"}}
    result = {"result": {"final_response": final_response, "messages": messages}, "usage": {}}
    _outcome, stored = prepare_result(row, final_response, "completed", result)
    return stored["result"]


def test_inline_image_data_url_survives_redaction():
    stored = _stored([{"role": "tool", "tool_call_id": "c1",
                       "content": [{"type": "image_url", "image_url": {"url": IMAGE}}]}])
    assert stored["messages"][0]["content"][0]["image_url"]["url"] == IMAGE


def test_secrets_next_to_inline_media_are_still_masked():
    leaked = f"token {JWT}"
    stored = _stored([{"role": "tool", "tool_call_id": "c1",
                       "content": [{"type": "image_url", "image_url": {"url": IMAGE}},
                                   {"type": "text", "text": leaked}]}],
                     final_response=leaked)
    assert JWT not in stored["final_response"]
    assert JWT not in stored["messages"][0]["content"][1]["text"]
    # A string that only starts like a data URL but is not pure base64 is still scrubbed.
    assert JWT not in _stored([], final_response=f"data:text/plain,{JWT}")["final_response"]
