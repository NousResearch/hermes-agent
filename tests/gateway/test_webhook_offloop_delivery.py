"""Event-loop liveness for webhook GitHub-comment delivery (Pattern A).

``_deliver_github_comment`` shells out to ``gh`` with a 30s timeout. Before
the off-loop fix it called ``subprocess.run`` inline from an ``async def``,
freezing the entire gateway event loop — every adapter, timer, and health
check — for up to the full timeout while the network call ran.

The contract under test is behavioral, not structural: while delivery is
awaiting the subprocess, OTHER coroutines on the same loop must keep
running.  A ticker coroutine sampled at the moment delivery returns proves
it — on the blocking implementation it manages ~1 tick during a 1-second
``gh`` run; off-loop it manages ~20.
"""

import asyncio
import os
import stat
import sys

import pytest

from gateway.platforms.webhook import WebhookAdapter


@pytest.fixture()
def fake_gh(tmp_path, monkeypatch):
    """A ``gh`` stub that sleeps 1s then succeeds, prepended to PATH."""
    if sys.platform.startswith("win"):
        pytest.skip("POSIX shell stub")
    gh = tmp_path / "gh"
    gh.write_text("#!/bin/bash\nsleep 1\necho posted\nexit 0\n", encoding="utf-8")
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return gh


class TestGithubCommentDeliveryOffLoop:
    @pytest.mark.asyncio
    async def test_loop_stays_live_during_gh_subprocess(self, fake_gh):
        """Concurrent coroutines keep ticking while ``gh`` runs (~1s)."""
        adapter = WebhookAdapter.__new__(WebhookAdapter)

        ticks = []
        stop = False

        async def ticker():
            while not stop:
                ticks.append(1)
                await asyncio.sleep(0.05)

        task = asyncio.create_task(ticker())
        await asyncio.sleep(0)  # let the ticker take its first turn

        result = await adapter._deliver_github_comment(
            "test body",
            {"deliver_extra": {"repo": "owner/repo", "pr_number": "1"}},
        )
        ticks_during_delivery = len(ticks)

        stop = True
        try:
            await asyncio.wait_for(task, timeout=2)
        except asyncio.TimeoutError:
            task.cancel()

        assert result.success is True
        # Blocking implementation: ~1 tick (loop frozen for the 1s gh run).
        # Off-loop implementation: ~20 ticks. 10 is a comfortable midpoint
        # that stays robust on slow CI runners.
        assert ticks_during_delivery >= 10, (
            f"event loop starved during gh delivery: only "
            f"{ticks_during_delivery} ticker turns ran in ~1s"
        )

    @pytest.mark.asyncio
    async def test_delivery_result_faithful_off_loop(self, fake_gh):
        """Off-loop offload must not change the SendResult contract."""
        adapter = WebhookAdapter.__new__(WebhookAdapter)
        result = await adapter._deliver_github_comment(
            "body",
            {"deliver_extra": {"repo": "owner/repo", "pr_number": "7"}},
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_invalid_inputs_still_rejected_before_subprocess(self):
        """Validation short-circuits stay synchronous and unchanged."""
        adapter = WebhookAdapter.__new__(WebhookAdapter)
        bad_repo = await adapter._deliver_github_comment(
            "body", {"deliver_extra": {"repo": "not a repo!", "pr_number": "1"}}
        )
        assert bad_repo.success is False
        bad_pr = await adapter._deliver_github_comment(
            "body", {"deliver_extra": {"repo": "owner/repo", "pr_number": "-2"}}
        )
        assert bad_pr.success is False
