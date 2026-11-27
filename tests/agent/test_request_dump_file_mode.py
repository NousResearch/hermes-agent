"""A request debug dump holds the whole prompt and conversation, even after redaction."""
import os
import stat

import pytest

from tests.agent.test_run_agent_codex_responses import _build_agent


@pytest.mark.platforms("posix")
def test_dump_api_request_debug_file_is_owner_only_under_permissive_umask(monkeypatch, tmp_path):
    """A request dump carries the whole prompt and conversation even after redaction, so a NEW
    dump is 0600 whatever the umask. It was 0600 via mkstemp until #109555 made mode-less new
    files follow the umask (0644 under 022)."""
    agent = _build_agent(monkeypatch)
    agent.logs_dir = tmp_path

    old_umask = os.umask(0o022)
    try:
        dump_file = agent._dump_api_request_debug(
            {"model": "gpt-5-codex", "input": [{"role": "user", "content": "private text"}]},
            reason="non_retryable_client_error",
        )
    finally:
        os.umask(old_umask)

    assert dump_file is not None and dump_file.exists()
    assert oct(stat.S_IMODE(dump_file.stat().st_mode)) == oct(0o600)
