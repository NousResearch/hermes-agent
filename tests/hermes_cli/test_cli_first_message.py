"""A fresh install's first CLI message reaches the model exactly as typed.

The first-contact note (the profile-build offer or the plain intro) is sent by the messaging gateway and
the TUI/desktop backend. The CLI never sent one. A one-shot ``hermes chat -q`` in a fresh home, which is
how kanban workers and scripted runs start, must not gain that note or record
``onboarding.seen.profile_build_offered`` in the profile's config.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from agent.onboarding import PROFILE_BUILD_FLAG, is_seen
from hermes_cli.config_effective import load_user_config_effective
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home

ROOT = Path(__file__).parents[2]
PROMPT = "work kanban task t_1"


def test_first_cli_message_in_a_fresh_home_reaches_the_model_unchanged(tmp_path):
    home = tmp_path / "hermes"
    with FakeLLMServer([Text("done")]) as srv:
        write_hermes_home(
            home, srv.base_url,
            extra_config="updates:\n  check: false\nmodels_dev:\n  url: http://127.0.0.1:9/api.json\n",
        )
        env = {**os.environ, "HERMES_HOME": str(home), "PYTHONUTF8": "1"}
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        for name in ("HERMES_GUEST_ONBOARDING", "HERMES_DESKTOP"):
            env.pop(name, None)
        proc = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "chat", "-q", PROMPT],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding="utf-8", timeout=240, check=False,
        )
        requests = srv.main_requests()

    assert proc.returncode == 0, proc.stderr[-3000:]
    assert requests, f"the model was never called; stderr tail: {proc.stderr[-2000:]}"
    users = [m["content"] for m in requests[0]["messages"] if m.get("role") == "user"]
    assert users == [PROMPT]
    assert not is_seen(load_user_config_effective(home / "config.yaml"), PROFILE_BUILD_FLAG)
