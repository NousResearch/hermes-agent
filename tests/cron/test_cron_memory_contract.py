"""Contract pin: cron <-> persistent-memory loading.

This contract FLIPPED TWICE in August 2026 and must never flip silently again:

  * #91269 reported "cron loads MEMORY.md even though skip_memory is on".
  * PR #91384 flipped cron to ``skip_memory=True`` and denylisted the
    ``memory`` toolset ("do not load MEMORY.md into scheduled jobs").
  * PR #91447 flipped it BACK: "cron jobs now load and update persistent
    memory like every other agent" — ``skip_memory=False`` at the scheduler's
    AIAgent construction site, ``memory`` removed from the cron denylist,
    and ``agent/agent_init.py`` clarified that ``skip_memory`` skips the
    *external memory provider* path (built-in MEMORY.md/USER.md store follows
    the normal ``not skip_memory or memory-toolset-requested`` rule).

CURRENT INTENDED MATRIX (as of PR #91447, pinned here):

  default cron job          -> skip_memory=False; MEMORY.md/USER.md load into
                               the system prompt; ``memory`` toolset follows
                               normal resolution (NOT policy-denied).
  per-job enabled_toolsets  -> naming ``memory`` keeps it; skip_memory stays
                               False.
  config.yaml
  agent.disabled_toolsets:
    [memory]                -> the ONLY off-switch: ``memory`` lands in the
                               cron agent's disabled_toolsets (tool denied,
                               and agent_init treats a denylisted toolset as
                               not-requested). skip_memory itself is NOT a
                               per-job/config toggle — the scheduler always
                               passes False.

ANY future flip of this behavior MUST consciously edit this test and cite
the issue/PR that decided the flip in the module docstring above, extending
the flip history. Do not "fix" a failure here by inverting an assertion
without that citation.

Tests drive the REAL ``cron.scheduler.run_job`` path and capture the actual
kwargs the scheduler passes to AIAgent (patched at ``run_agent.AIAgent``,
matching tests/cron/test_scheduler.py's pattern). The ON direction (default
skip_memory=False, memory not denylisted, per-job memory toolset kept) is
already pinned by tests/cron/test_scheduler.py::test_run_job_*memory*; this
module pins the OFF direction and the "no per-job knob" rule.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

from cron.scheduler import run_job

@contextlib.contextmanager
def _run_job_patches(tmp_path):
    """Patch bundle so run_job runs offline; yields (fake_db, mock_agent_cls).

    Mirrors tests/cron/test_scheduler.py::_run_job_patches — every patch is
    entered via one ExitStack so none can be silently dropped.
    ``cron.scheduler._hermes_home`` is pointed at ``tmp_path`` so run_job's
    config load reads ``tmp_path/config.yaml`` (write one to exercise config
    toggles).
    """
    fake_db = MagicMock()
    fake_db.get_compression_tip.side_effect = lambda session_id: session_id
    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {"final_response": "ok"}
    base = [
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler_delivery._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state_registry.acquire", return_value=fake_db),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "test-key",
                "base_url": "https://example.invalid/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
            },
        ),
        patch("run_agent.AIAgent", return_value=mock_agent),
    ]
    with contextlib.ExitStack() as stack:
        entered = [stack.enter_context(cm) for cm in base]
        yield fake_db, entered[-1]

class TestCronMemoryContractOff:
    """Direction (b): the supported OFF switch stays off."""

    def test_config_disabled_toolsets_denies_memory(self, tmp_path):
        """agent.disabled_toolsets: [memory] in config.yaml denies the toolset.

        This is the intended user-level off-switch after #91447: the user
        denylist layers onto cron's base denylist (#25752), so the memory
        tool is denied AND agent_init treats a denylisted toolset as
        not-requested. A per-job enabled_toolsets cannot widen past it.
        """
        (tmp_path / "config.yaml").write_text(
            "agent:\n  disabled_toolsets:\n    - memory\n"
        )
        job = {
            "id": "mem-contract-off",
            "name": "t",
            "prompt": "hi",
            "enabled_toolsets": ["memory", "file"],
        }
        with _run_job_patches(tmp_path) as (_db, agent_cls):
            run_job(job)
        kwargs = agent_cls.call_args.kwargs
        assert "memory" in (kwargs["disabled_toolsets"] or []), (
            "config.yaml agent.disabled_toolsets must propagate 'memory' into "
            "the cron agent's denylist — the OFF direction of the contract"
        )
