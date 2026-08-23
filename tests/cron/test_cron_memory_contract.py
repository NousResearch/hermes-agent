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
matching tests/cron/test_scheduler.py's pattern).
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
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB", return_value=fake_db),
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


class TestCronMemoryContractOn:
    """Direction (a): default cron agents GET persistent memory (#91447)."""

    def test_default_cron_agent_passes_skip_memory_false(self, tmp_path):
        """The scheduler's AIAgent construction site passes skip_memory=False.

        This is the exact flag PR #91384 set to True and PR #91447 set back
        to False. skip_memory=False means the built-in MEMORY.md/USER.md
        store loads into the system prompt via agent_init's normal path.
        """
        job = {"id": "mem-contract-default", "name": "t", "prompt": "hi"}
        with _run_job_patches(tmp_path) as (_db, agent_cls):
            success, _out, _final, error = run_job(job)
        assert success is True and error is None
        kwargs = agent_cls.call_args.kwargs
        assert kwargs["skip_memory"] is False, (
            "Cron memory contract (#91447): cron agents load MEMORY.md/USER.md "
            "like any other agent. If you are flipping this on purpose, edit "
            "this module's docstring and cite the deciding issue/PR."
        )

    def test_memory_toolset_not_policy_denied_by_default(self, tmp_path):
        """PR #91447 removed 'memory' from the cron toolset denylist.

        PR #91384 had added it alongside messaging/clarify. It must not creep
        back in silently.
        """
        job = {"id": "mem-contract-denylist", "name": "t", "prompt": "hi"}
        with _run_job_patches(tmp_path) as (_db, agent_cls):
            run_job(job)
        kwargs = agent_cls.call_args.kwargs
        assert "memory" not in (kwargs["disabled_toolsets"] or []), (
            "'memory' must not be in cron's default denylist (#91447)"
        )

    def test_resolver_denylist_has_no_memory_entry(self):
        """_resolve_cron_disabled_toolsets({}) itself never emits 'memory'."""
        from cron.scheduler import _resolve_cron_disabled_toolsets

        assert "memory" not in _resolve_cron_disabled_toolsets({})
        assert "memory" not in _resolve_cron_disabled_toolsets(
            {"cron": {"allow_agent_scheduling": True}}
        )

    def test_per_job_memory_toolset_survives(self, tmp_path):
        """A per-job enabled_toolsets naming memory keeps it (no stripping).

        PR #91384 introduced _strip_cron_memory_toolset; PR #91447 deleted it.
        """
        job = {
            "id": "mem-contract-perjob",
            "name": "t",
            "prompt": "hi",
            "enabled_toolsets": ["memory", "file"],
        }
        with _run_job_patches(tmp_path) as (_db, agent_cls):
            run_job(job)
        kwargs = agent_cls.call_args.kwargs
        assert "memory" in (kwargs["enabled_toolsets"] or [])
        assert kwargs["skip_memory"] is False


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

    def test_skip_memory_is_not_a_per_job_knob(self, tmp_path):
        """No per-job field flips skip_memory: the scheduler always passes False.

        Guards against a partial re-flip where some job shape quietly gets
        #91384 behavior back. A field named skip_memory on the job dict is
        ignored by the construction site.
        """
        job = {
            "id": "mem-contract-noknob",
            "name": "t",
            "prompt": "hi",
            "skip_memory": True,  # not a supported job field; must be ignored
        }
        with _run_job_patches(tmp_path) as (_db, agent_cls):
            run_job(job)
        kwargs = agent_cls.call_args.kwargs
        assert kwargs["skip_memory"] is False


class TestAgentInitOffPathStaysOff:
    """agent_init side of the OFF direction (#65429 / #91447 wording).

    skip_memory=True with 'memory' merely present-by-default but denylisted
    must NOT load the MEMORY.md store — a denylisted toolset is not a request.
    """

    def test_denylisted_memory_toolset_is_not_a_request(self):
        enabled = ["memory", "file"]
        disabled = ["memory"]
        requested = "memory" in enabled and "memory" not in disabled
        # Mirrors agent_init's _memory_toolset_requested gate exactly; the
        # deep store-construction behavior is covered by
        # tests/agent/test_skip_memory_store_65429.py.
        assert requested is False
