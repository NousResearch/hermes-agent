"""Opt-in live Codex QA server, not a no-model fixture. See the companion UI QA docs.

Credentials are resolved by the official read-only API and remain in memory.
Only the explicitly supplied isolated HERMES_HOME is written.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.auth import resolve_codex_runtime_credentials
    import hermes_yaml as yaml

    home = Path(os.environ["HERMES_HOME"]).resolve()
    auth_home = Path(os.environ["QA_AUTH_HOME"]).resolve()
    if home == auth_home or home in auth_home.parents or auth_home in home.parents:
        raise ValueError("QA and authentication homes must be separate, non-overlapping directories")
    scope = set_hermes_home_override(auth_home)
    try:
        runtime = resolve_codex_runtime_credentials(read_only=True)
    finally:
        reset_hermes_home_override(scope)
    if not runtime.get("api_key"):
        raise RuntimeError("Read-only Codex credentials unavailable; authenticate normally before live QA")
    failure = os.environ.get("QA_FAIL_CHILD") == "1"
    delegation = {"max_iterations": 8}
    if failure:
        # Real child runtime failure via normal routing; no synthetic progress events.
        delegation.update(model="qa-intentionally-unavailable-model", fallback_providers=[])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "gpt-6.1-sol", "provider": "openai-codex"},
        "delegation": delegation,
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "skills": {"auto_load": False}, "terminal": {"backend": "local"}, "max_iterations": 10,
    }), encoding="utf-8")

    from run_agent import AIAgent
    from acp_adapter.server import HermesACPAgent
    from acp_adapter.session import SessionManager
    import acp

    def factory():
        return AIAgent(
            model="gpt-6.1-sol", provider="openai-codex", api_key=runtime["api_key"],
            base_url=runtime["base_url"], api_mode="codex_responses",
            enabled_toolsets=["delegation", "terminal", "file"], max_iterations=10,
            quiet_mode=True, skip_memory=True, skip_background_review=True, skip_context_files=True,
            platform="acp", ephemeral_system_prompt=(
                "This is an isolated integration QA workspace containing synthetic files. "
                "Use only the supplied synthetic tasks. For background delegation, start the requested child, "
                "then immediately give the requested short parent acknowledgment; do not poll or wait. "
                "Never read outside this workspace."
            ),
        )

    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    asyncio.run(acp.run_agent(HermesACPAgent(SessionManager(agent_factory=factory)), use_unstable_protocol=True))


if __name__ == "__main__":
    main()
