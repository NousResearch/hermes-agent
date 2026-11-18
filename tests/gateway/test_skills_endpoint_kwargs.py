"""#157330: GET /v1/skills must not pass a kwarg the reverted signature no longer accepts."""

import inspect

from tools.skills_tool import find_all_skills


def test_find_all_skills_signature_has_no_editorial_kwarg():
    params = inspect.signature(find_all_skills).parameters
    assert "include_editorial" not in params
    assert "skip_disabled" in params


def test_skills_handler_calls_with_supported_kwargs_only():
    import gateway.platforms.api_server as api_server
    source = inspect.getsource(api_server.APIServerAdapter._handle_skills)
    assert "include_editorial" not in source
    assert "find_all_skills(skip_disabled=False)" in source


def test_find_all_skills_runs_with_the_handler_kwargs():
    # The exact call the handler makes must not raise TypeError.
    assert isinstance(find_all_skills(skip_disabled=False), list)
