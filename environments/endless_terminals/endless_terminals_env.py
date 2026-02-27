"""
Endless Terminals Environment for Hermes-Agent + Atropos RL.

Loads pre-generated terminal tasks from HuggingFace dataset and scores
agent performance using test execution in the agent's sandbox.

Uses hermes-agent backends (modal, docker, local) with per-task Docker images
extracted from container.def files. Tests run in the same sandbox the agent
used, following the Terminal Bench 2 pattern.

Dataset: https://huggingface.co/datasets/obiwan96/endless-terminals-train

Run:
  python environments/endless_terminals/endless_terminals_env.py process \
    --config environments/endless_terminals/default.yaml
"""

import asyncio
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

# Ensure hermes-agent root is on path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from atroposlib.envs.base import ScoredDataGroup, ScoredDataItem
from atroposlib.type_definitions import Item

from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.agent_loop import AgentResult
from environments.tool_context import ToolContext
from tools.terminal_tool import (
    register_task_env_overrides,
    clear_task_env_overrides,
    cleanup_vm,
)

logger = logging.getLogger(__name__)

# Add endless-terminals to path for imports
ENDLESS_TERMINALS_PATH = os.getenv(
    "ENDLESS_TERMINALS_PATH",
    str(Path.home() / "Desktop" / "Projects" / "endless-terminals")
)
sys.path.insert(0, ENDLESS_TERMINALS_PATH)


class EndlessTerminalsEnvConfig(HermesAgentEnvConfig):
    """Configuration for Endless Terminals environment."""

    # Dataset settings
    use_dataset: bool = Field(
        default=True,
        description="Load tasks from HuggingFace dataset (recommended). If False, generate procedurally."
    )
    dataset_name: str = Field(
        default="obiwan96/endless-terminals-train",
        description="HuggingFace dataset name"
    )
    dataset_split: str = Field(
        default="train",
        description="Dataset split to use"
    )
    dataset_cache_dir: str = Field(
        default="~/.cache/huggingface/datasets",
        description="HuggingFace datasets cache directory"
    )
    tasks_base_dir: str = Field(
        default="",
        description="Base directory containing task_* folders. If empty, uses paths from dataset."
    )

    # Test execution
    test_timeout_s: int = Field(default=60, description="Test execution timeout (seconds)")

    # Docker image fallback
    default_docker_image: str = Field(
        default="ubuntu:22.04",
        description="Default Docker image if container.def parsing fails"
    )

    # Agent defaults
    max_agent_turns: int = Field(default=32, description="Max turns for agent (increased for long traces)")

    # Evaluation settings
    num_eval_tasks: int = Field(
        default=10,
        description="Number of tasks to run during periodic evaluation"
    )


class EndlessTerminalsEnv(HermesAgentBaseEnv):
    """
    Endless Terminals environment using pre-generated HuggingFace dataset.

    Loads terminal tasks from dataset, runs agent with terminal tools,
    and scores by executing tests in the agent's sandbox using ToolContext.
    """

    name = "endless_terminals_env"
    env_config_cls = EndlessTerminalsEnvConfig

    @classmethod
    def config_init(cls) -> Tuple[EndlessTerminalsEnvConfig, List["APIServerConfig"]]:
        """
        Default configuration for Endless Terminals environment.

        This is used when no config file is provided, but note that when using
        --config, the YAML is loaded differently and this may not be called.
        """
        from atroposlib.envs.server_handling.server_manager import APIServerConfig

        env_config = EndlessTerminalsEnvConfig(
            enabled_toolsets=["terminal", "file"],
            max_agent_turns=32,
            terminal_backend="local",
            use_dataset=True,
            tasks_base_dir="",
            group_size=1,
            total_steps=1,
            use_wandb=False,
        )

        server_configs = [
            APIServerConfig(
                base_url="https://openrouter.ai/api/v1",
                model_name="anthropic/claude-sonnet-4.5",
                server_type="openai",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                health_check=False,
            )
        ]

        return env_config, server_configs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dataset = None
        self._dataset_indices = []
        self._current_index = 0

        # Metrics tracking for wandb - single buffer with dicts
        self._metrics_buffer = []

        # Debug: check server config
        if hasattr(self, 'server') and hasattr(self.server, 'servers'):
            for i, srv in enumerate(self.server.servers):
                logger.debug(f"Server {i}: model_name={getattr(srv.config, 'model_name', 'NONE')}")

    async def setup(self):
        """Load dataset from HuggingFace or local directory."""
        if not self.config.use_dataset:
            logger.info("Using procedural task generation (not implemented yet)")
            return

        # If tasks_base_dir is set, load from local directory instead of HuggingFace
        if self.config.tasks_base_dir:
            tasks_base = Path(os.path.expanduser(self.config.tasks_base_dir))

            # Resolve to absolute path if relative
            if not tasks_base.is_absolute():
                tasks_base = Path.cwd() / tasks_base

            tasks_base = tasks_base.resolve()

            if not tasks_base.exists():
                raise RuntimeError(f"tasks_base_dir not found: {tasks_base}")

            logger.info(f"Loading tasks from local directory: {tasks_base}")

            # Find all task_* directories
            task_dirs = sorted(tasks_base.glob("task_*"))
            logger.info(f"Found {len(task_dirs)} task directories")

            if not task_dirs:
                # Debug: show what's actually in the directory
                all_items = list(tasks_base.iterdir())
                logger.warning(f"Directory contains {len(all_items)} items:")
                for item in all_items[:10]:
                    logger.warning(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
                raise RuntimeError(f"No task_* directories found in {tasks_base}")

            # Create fake dataset items (just the directory paths)
            self._dataset = [
                {
                    "description": f"Task from {task_dir.name}",
                    "extra_info": {"task_dir": str(task_dir)},
                }
                for task_dir in task_dirs
            ]

            # Create shuffled indices
            self._dataset_indices = list(range(len(self._dataset)))
            random.shuffle(self._dataset_indices)
            self._current_index = 0

            logger.info(f"Loaded {len(self._dataset)} tasks from local directory")
            return

        # Otherwise, load from HuggingFace
        logger.info(f"Loading dataset from HuggingFace: {self.config.dataset_name}")

        try:
            from datasets import load_dataset

            self._dataset = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: load_dataset(
                    self.config.dataset_name,
                    split=self.config.dataset_split,
                    cache_dir=os.path.expanduser(self.config.dataset_cache_dir)
                )
            )

            # Create shuffled indices
            self._dataset_indices = list(range(len(self._dataset)))
            random.shuffle(self._dataset_indices)
            self._current_index = 0

            logger.info(f"Loaded {len(self._dataset)} tasks from HuggingFace")

        except Exception as e:
            logger.error(f"ERROR loading dataset: {e}")
            raise

    async def get_next_item(self) -> Item:
        """Sample next task from dataset."""
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")

        # Get next task (with wraparound)
        idx = self._dataset_indices[self._current_index]
        task = self._dataset[idx]

        # Advance to next task
        self._current_index += 1
        if self._current_index >= len(self._dataset_indices):
            # Reshuffle for next epoch
            random.shuffle(self._dataset_indices)
            self._current_index = 0
            logger.info("Reshuffled dataset (completed one epoch)")

        # Extract task directory path
        task_dir = task.get("extra_info", {}).get("task_dir")
        if not task_dir:
            task_dir = task.get("reward_spec", {}).get("ground_truth")

        # Resolve task directory path
        if task_dir:
            task_dir_path = Path(task_dir)
            # If tasks_base_dir is configured and path doesn't exist, reconstruct it
            if self.config.tasks_base_dir and not task_dir_path.exists():
                original_path = Path(task_dir)
                task_name = original_path.name
                task_dir_path = Path(os.path.expanduser(self.config.tasks_base_dir)) / task_name
        else:
            logger.error("No task directory path found in dataset item")
            return await self.get_next_item()

        # Verify directory exists
        if not task_dir_path.exists():
            logger.warning(f"Task dir not found: {task_dir_path}")
            logger.warning("Hint: Set tasks_base_dir to directory containing task_* folders")
            return await self.get_next_item()  # Try next task

        # Look for test file in tests/ subdirectory first, then at root
        final_test = task_dir_path / "tests" / "test_final_state.py"
        if not final_test.exists():
            final_test = task_dir_path / "test_final_state.py"

        # Verify test file exists
        if not final_test.exists():
            logger.warning(f"Missing test file in {task_dir_path} (checked tests/ and root)")
            return await self.get_next_item()

        # Parse container.def to extract Docker image
        # Check environment/ subdirectory first, then root
        container_def = task_dir_path / "environment" / "container.def"
        if not container_def.exists():
            container_def = task_dir_path / "container.def"
        docker_image = self._parse_docker_image_from_def(container_def)

        # Try to load description from instruction.md or task.json
        description = task.get("description", "")

        # First try instruction.md
        instruction_md = task_dir_path / "instruction.md"
        if not description and instruction_md.exists():
            try:
                description = instruction_md.read_text().strip()
            except Exception as e:
                logger.warning(f"Failed to load instruction.md for {task_dir_path.name}: {e}")

        # Fallback to task.json in environment/
        if not description:
            task_json = task_dir_path / "environment" / "task.json"
            if task_json.exists():
                try:
                    import json
                    task_data = json.loads(task_json.read_text())
                    description = task_data.get("description", "") or task_data.get("instruction", "")
                except Exception as e:
                    logger.warning(f"Failed to load task.json for {task_dir_path.name}: {e}")

        if not description:
            description = f"Complete the task in {task_dir_path.name}"

        return {
            "task_id": f"{task_dir_path.name}",
            "task_name": task_dir_path.name,
            "description": description,
            "task_dir": str(task_dir_path),
            "final_test": str(final_test),
            "docker_image": docker_image,
            "dataset_index": idx,
        }

    def format_prompt(self, item: Item) -> str:
        """Return the task description for the agent."""
        return str(item.get("description", ""))

    def _parse_docker_image_from_def(self, container_def_path: Path) -> str:
        """
        Parse container.def file to extract the Docker base image.

        Apptainer definition files typically look like:
            Bootstrap: docker
            From: ubuntu:22.04

        Returns the image from the "From:" line, or falls back to default.
        """
        if not container_def_path.exists():
            logger.warning(f"container.def not found at {container_def_path}, using default image")
            return self.config.default_docker_image

        try:
            content = container_def_path.read_text()
            # Look for "From: <image>" line (case-insensitive)
            match = re.search(r'^From:\s*(.+)$', content, re.MULTILINE | re.IGNORECASE)
            if match:
                image = match.group(1).strip()
                logger.info(f"Extracted Docker image from container.def: {image}")
                return image
        except Exception as e:
            logger.warning(f"Failed to parse {container_def_path}: {e}")

        logger.warning(f"Could not extract image from {container_def_path}, using default")
        return self.config.default_docker_image

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """
        Override to register per-task Docker image before running the agent.

        Follows Terminal Bench 2 pattern: register_task_env_overrides() tells
        the hermes-agent terminal backend to use a specific Docker image for
        this task_id.

        This is a copy of HermesAgentBaseEnv.collect_trajectory with Docker
        image registration added after task_id generation.
        """
        import uuid
        from environments.agent_loop import HermesAgentLoop

        task_id = str(uuid.uuid4())
        task_name = item.get("task_name", "unknown")
        docker_image = item.get("docker_image", self.config.default_docker_image)

        logger.debug(f"collect_trajectory START for {task_name}")

        # Register Docker image override for this task_id
        logger.debug(f"Registering Docker image: {docker_image}")
        register_task_env_overrides(task_id, {"modal_image": docker_image})
        logger.info(
            f"Task {task_name}: registered Docker image {docker_image} for task_id {task_id[:8]}"
        )
        logger.debug("Docker image registered")

        try:
            # Get group-level tools (resolved once in collect_trajectories)
            logger.debug("Resolving tools...")
            if self._current_group_tools is None:
                tools, valid_names = self._resolve_tools_for_group()
            else:
                tools, valid_names = self._current_group_tools
            logger.debug(f"Tools resolved: {len(tools)} tools")

            # Build initial messages
            logger.debug("Building initial messages...")
            messages: List[Dict[str, Any]] = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.append({"role": "user", "content": self.format_prompt(item)})
            logger.debug("Messages built, starting agent loop...")

            # Run the agent loop
            result: AgentResult
            if self._use_managed_server():
                # Phase 2: ManagedServer with parser
                from environments.tool_call_parsers import get_parser
                try:
                    tc_parser = get_parser(self.config.tool_call_parser)
                except KeyError:
                    logger.warning(
                        "Tool call parser '%s' not found, falling back to 'hermes'",
                        self.config.tool_call_parser,
                    )
                    tc_parser = get_parser("hermes")

                try:
                    async with self.server.managed_server(
                        tokenizer=self.tokenizer,
                        tool_call_parser=tc_parser,
                    ) as managed:
                        agent = HermesAgentLoop(
                            server=managed,
                            tool_schemas=tools,
                            valid_tool_names=valid_names,
                            max_turns=self.config.max_agent_turns,
                            task_id=task_id,
                            temperature=self.config.agent_temperature,
                            max_tokens=self.config.max_token_length,
                            extra_body=self.config.extra_body,
                        )
                        result = await agent.run(messages)
                except NotImplementedError:
                    # DummyManagedServer not allowed
                    logger.warning("ManagedServer not available. Falling back to direct server mode.")
                    agent = HermesAgentLoop(
                        server=self.server,
                        tool_schemas=tools,
                        valid_tool_names=valid_names,
                        max_turns=self.config.max_agent_turns,
                        task_id=task_id,
                        temperature=self.config.agent_temperature,
                        max_tokens=self.config.max_token_length,
                        extra_body=self.config.extra_body,
                    )
                    result = await agent.run(messages)
            else:
                # Phase 1: OpenAI server
                agent = HermesAgentLoop(
                    server=self.server,
                    tool_schemas=tools,
                    valid_tool_names=valid_names,
                    max_turns=self.config.max_agent_turns,
                    task_id=task_id,
                    temperature=self.config.agent_temperature,
                    max_tokens=self.config.max_token_length,
                    extra_body=self.config.extra_body,
                )
                result = await agent.run(messages)

            # Skip reward computation if agent produced no output
            only_system_and_user = all(
                msg.get("role") in ("system", "user") for msg in result.messages
            )
            if result.turns_used == 0 or only_system_and_user:
                logger.warning(
                    "Agent loop produced no output (turns=%d). Skipping reward.",
                    result.turns_used,
                )
                reward = 0.0
            else:
                # Compute reward using ToolContext
                ctx = ToolContext(task_id)
                try:
                    reward = await self.compute_reward(item, result, ctx)
                except Exception as e:
                    logger.error("compute_reward failed: %s", e)
                    reward = 0.0
                finally:
                    ctx.cleanup()

            # Track metrics for wandb logging
            task_metrics = {
                "test_passed": 1.0 if reward > 0.5 else 0.0,
                "reward": reward,
                "turns_used": result.turns_used,
                "finished_naturally": result.finished_naturally,
                "docker_image": docker_image,
                "num_tool_errors": len(result.tool_errors),
            }

            # Include detailed tool errors if any occurred
            if result.tool_errors:
                task_metrics["tool_errors"] = [
                    {
                        "turn": err.turn,
                        "tool": err.tool_name,
                        "error": err.error[:200],
                    }
                    for err in result.tool_errors
                ]

            self._metrics_buffer.append(task_metrics)

            # Build ScoredDataItem from ManagedServer state
            # Phase 2: real tokens/masks/logprobs from SequenceNodes
            # Phase 1: placeholder tokens
            nodes = (result.managed_state or {}).get("nodes", [])

            if nodes:
                # Phase 2: use actual node data
                # nodes[-1] contains the full accumulated trajectory from all turns
                node = nodes[-1]
                scored_item: Dict[str, Any] = {
                    "tokens": node.tokens,
                    "masks": node.masked_tokens,
                    "scores": reward,
                }
                if hasattr(node, "logprobs") and node.logprobs:
                    scored_item["logprobs"] = node.logprobs
                    scored_item["advantages"] = None
                    scored_item["ref_logprobs"] = None
            else:
                # Phase 1: create placeholder tokens
                full_text = "\n".join(
                    msg.get("content", "") for msg in result.messages if msg.get("content")
                )
                if self.tokenizer:
                    tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
                else:
                    tokens = list(range(min(len(full_text) // 4, 128)))

                scored_item = {
                    "tokens": tokens,
                    "masks": [-100] + tokens[1:],
                    "scores": reward,
                }

            # Include messages for wandb rollout display
            scored_item["messages"] = result.messages

            return scored_item, []

        finally:
            # Clean up task overrides and sandbox
            clear_task_env_overrides(task_id)
            try:
                cleanup_vm(task_id)
            except Exception as e:
                logger.debug(f"VM cleanup for {task_id[:8]}: {e}")

    async def compute_reward(
        self,
        item: Item,
        result: AgentResult,
        ctx: ToolContext
    ) -> float:
        """
        Run final tests in the agent's sandbox and return binary reward.

        Uses ToolContext to execute pytest in the SAME sandbox the agent used,
        following the Terminal Bench 2 verification pattern. No separate
        Apptainer execution needed.

        Returns 1.0 if tests pass, 0.0 otherwise.
        """
        task_name = item.get("task_name", "unknown")
        final_test_path = Path(item.get("final_test", ""))

        if not final_test_path.exists():
            logger.error(f"Task {task_name}: test file not found at {final_test_path}")
            return 0.0

        logger.info(f"Task {task_name}: running tests in sandbox...")

        try:
            # Run tests in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            reward = await loop.run_in_executor(
                None,
                self._run_tests_in_sandbox,
                final_test_path,
                ctx,
                task_name,
            )

            status = "PASS" if reward == 1.0 else "FAIL"
            logger.info(f"Task {task_name}: {status} (reward={reward})")
            return reward

        except Exception as e:
            logger.error(f"Task {task_name}: test execution failed: {e}", exc_info=True)
            return 0.0

    def _run_tests_in_sandbox(
        self,
        test_file_path: Path,
        ctx: ToolContext,
        task_name: str,
    ) -> float:
        """
        Upload test file to sandbox and execute pytest.

        Runs in thread pool (via run_in_executor) to avoid blocking the event loop
        with synchronous ToolContext calls.

        Args:
            test_file_path: Local path to test_final_state.py
            ctx: ToolContext scoped to the agent's sandbox
            task_name: For logging

        Returns:
            1.0 if tests pass, 0.0 otherwise
        """
        try:
            # Upload test file to sandbox
            test_content = test_file_path.read_text()
            ctx.write_file("/workspace/test_final_state.py", test_content)
            logger.debug(f"Task {task_name}: uploaded test file to /workspace/test_final_state.py")

            # Run pytest in the sandbox
            result = ctx.terminal(
                "cd /workspace && python -m pytest -q test_final_state.py",
                timeout=self.config.test_timeout_s,
            )

            exit_code = result.get("exit_code", -1)
            output = result.get("output", "")

            if exit_code == 0:
                logger.debug(f"Task {task_name}: tests passed")
                return 1.0
            else:
                # Log failure output (last 500 chars for debugging)
                output_preview = output[-500:] if output else "(no output)"
                logger.info(
                    f"Task {task_name}: tests failed (exit_code={exit_code})\n{output_preview}"
                )
                return 0.0

        except Exception as e:
            logger.error(f"Task {task_name}: error running tests: {e}")
            return 0.0

    async def evaluate(self):
        """
        Periodic evaluation on a fixed set of tasks.

        Runs the agent on num_eval_tasks tasks and measures performance
        without affecting training. Returns metrics for wandb logging.
        """
        if self._dataset is None:
            logger.warning("Cannot evaluate: dataset not loaded")
            return {}

        logger.info(f"Starting evaluation on {self.config.num_eval_tasks} tasks...")

        eval_metrics = {
            "rewards": [],
            "passes": [],
            "turns": [],
            "natural_finishes": [],
        }

        # Sample eval tasks randomly
        import random
        eval_indices = random.sample(range(len(self._dataset)), min(self.config.num_eval_tasks, len(self._dataset)))

        for idx in eval_indices:
            task = self._dataset[idx]

            # Build item using same logic as get_next_item
            task_dir = task.get("extra_info", {}).get("task_dir")
            if not task_dir:
                task_dir = task.get("reward_spec", {}).get("ground_truth")

            if not task_dir:
                continue

            task_dir_path = Path(task_dir)
            if self.config.tasks_base_dir and not task_dir_path.exists():
                original_path = Path(task_dir)
                task_name = original_path.name
                task_dir_path = Path(os.path.expanduser(self.config.tasks_base_dir)) / task_name

            if not task_dir_path.exists():
                continue

            # Find test file
            final_test = task_dir_path / "tests" / "test_final_state.py"
            if not final_test.exists():
                final_test = task_dir_path / "test_final_state.py"
            if not final_test.exists():
                continue

            # Parse Docker image
            container_def = task_dir_path / "environment" / "container.def"
            if not container_def.exists():
                container_def = task_dir_path / "container.def"
            docker_image = self._parse_docker_image_from_def(container_def)

            # Load description
            description = task.get("description", "")
            instruction_md = task_dir_path / "instruction.md"
            if not description and instruction_md.exists():
                try:
                    description = instruction_md.read_text().strip()
                except Exception:
                    pass

            item = {
                "description": description,
                "final_test": str(final_test),
                "docker_image": docker_image,
            }

            # Run agent on this task
            try:
                import uuid
                task_id = str(uuid.uuid4())

                # Register task environment
                from model_tools import register_task_env_overrides
                register_task_env_overrides(task_id, {"modal_image": docker_image})

                # Build messages
                messages = [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": description or "Complete the task."},
                ]

                # Get tools
                from model_tools import get_tool_definitions
                tools = get_tool_definitions(self.config.enabled_toolsets)
                valid_names = {t["function"]["name"] for t in tools}

                # Run agent
                from environments.agent_loop import HermesAgentLoop
                agent = HermesAgentLoop(
                    server=self.server,
                    tool_schemas=tools,
                    valid_tool_names=valid_names,
                    max_turns=self.config.max_agent_turns,
                    task_id=task_id,
                    temperature=self.config.agent_temperature,
                    max_tokens=self.config.max_token_length,
                    extra_body=self.config.extra_body,
                )
                result = await agent.run(messages)

                # Compute reward
                from environments.tool_context import ToolContext
                ctx = ToolContext(task_id)
                try:
                    reward = await self.compute_reward(item, result, ctx)
                except Exception as e:
                    logger.warning(f"Eval reward computation failed: {e}")
                    reward = 0.0
                finally:
                    ctx.cleanup()

                # Track metrics
                eval_metrics["rewards"].append(reward)
                eval_metrics["passes"].append(1.0 if reward > 0.5 else 0.0)
                eval_metrics["turns"].append(result.turns_used)
                eval_metrics["natural_finishes"].append(1.0 if result.finished_naturally else 0.0)

            except Exception as e:
                logger.error(f"Eval task failed: {e}")
                continue
            finally:
                # Cleanup
                from model_tools import clear_task_env_overrides, cleanup_vm
                clear_task_env_overrides(task_id)
                cleanup_vm(task_id)

        # Aggregate metrics
        if not eval_metrics["rewards"]:
            logger.warning("No eval tasks completed successfully")
            return {}

        aggregated = {
            "eval/pass_rate": sum(eval_metrics["passes"]) / len(eval_metrics["passes"]),
            "eval/avg_reward": sum(eval_metrics["rewards"]) / len(eval_metrics["rewards"]),
            "eval/avg_turns": sum(eval_metrics["turns"]) / len(eval_metrics["turns"]),
            "eval/natural_finish_rate": sum(eval_metrics["natural_finishes"]) / len(eval_metrics["natural_finishes"]),
            "eval/num_tasks": len(eval_metrics["rewards"]),
        }

        logger.info(f"Evaluation complete: pass_rate={aggregated['eval/pass_rate']:.2%}, avg_turns={aggregated['eval/avg_turns']:.1f}")
        return aggregated

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log Endless Terminals specific metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Aggregate metrics from buffer
        if self._metrics_buffer:
            # Test pass rate
            test_passes = [m["test_passed"] for m in self._metrics_buffer]
            wandb_metrics["endless_terminals/test_pass_rate"] = sum(test_passes) / len(test_passes)
            wandb_metrics["endless_terminals/num_tests_passed"] = sum(test_passes)
            wandb_metrics["endless_terminals/num_tests_total"] = len(test_passes)

            # Turns used statistics
            turns = [m["turns_used"] for m in self._metrics_buffer]
            wandb_metrics["endless_terminals/avg_turns_used"] = sum(turns) / len(turns)
            wandb_metrics["endless_terminals/max_turns_used"] = max(turns)
            wandb_metrics["endless_terminals/min_turns_used"] = min(turns)

            # Natural finish rate (did model stop on its own vs hitting max turns)
            natural_finishes = [1.0 if m["finished_naturally"] else 0.0 for m in self._metrics_buffer]
            wandb_metrics["endless_terminals/natural_finish_rate"] = sum(natural_finishes) / len(natural_finishes)

            # Tool error statistics
            total_tool_errors = sum(m["num_tool_errors"] for m in self._metrics_buffer)
            wandb_metrics["endless_terminals/total_tool_errors"] = total_tool_errors
            wandb_metrics["endless_terminals/avg_tool_errors_per_task"] = total_tool_errors / len(self._metrics_buffer)

            # Docker image distribution (count unique images used)
            docker_images = [m["docker_image"] for m in self._metrics_buffer]
            unique_images = set(docker_images)
            wandb_metrics["endless_terminals/num_unique_docker_images"] = len(unique_images)

            # Log most common errors if any
            all_errors = []
            for m in self._metrics_buffer:
                if "tool_errors" in m:
                    all_errors.extend(m["tool_errors"])

            if all_errors:
                # Count error types
                error_tools = {}
                for err in all_errors:
                    tool = err["tool"]
                    error_tools[tool] = error_tools.get(tool, 0) + 1

                # Log top 3 error-prone tools
                for i, (tool, count) in enumerate(sorted(error_tools.items(), key=lambda x: x[1], reverse=True)[:3]):
                    wandb_metrics[f"endless_terminals/errors_by_tool/{tool}"] = count

            # Clear buffer after logging
            self._metrics_buffer = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    EndlessTerminalsEnv.cli()
