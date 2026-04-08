import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


def _load_hermes_swe_module():
    module_path = (
        Path(__file__).resolve().parent.parent
        / "environments"
        / "hermes_swe_env"
        / "hermes_swe_env.py"
    )

    datasets_mod = types.ModuleType("datasets")
    datasets_mod.load_dataset = lambda *args, **kwargs: []

    atropos_mod = types.ModuleType("atroposlib")
    atropos_envs_mod = types.ModuleType("atroposlib.envs")
    atropos_envs_base_mod = types.ModuleType("atroposlib.envs.base")
    atropos_server_mod = types.ModuleType("atroposlib.envs.server_handling")
    atropos_server_manager_mod = types.ModuleType(
        "atroposlib.envs.server_handling.server_manager"
    )
    atropos_types_mod = types.ModuleType("atroposlib.type_definitions")

    class ScoredDataGroup:
        pass

    class APIServerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    atropos_envs_base_mod.ScoredDataGroup = ScoredDataGroup
    atropos_server_manager_mod.APIServerConfig = APIServerConfig
    atropos_types_mod.Item = dict

    agent_loop_mod = types.ModuleType("environments.agent_loop")
    agent_loop_mod.AgentResult = object

    hermes_base_env_mod = types.ModuleType("environments.hermes_base_env")

    class HermesAgentBaseEnv:
        pass

    class HermesAgentEnvConfig:
        pass

    hermes_base_env_mod.HermesAgentBaseEnv = HermesAgentBaseEnv
    hermes_base_env_mod.HermesAgentEnvConfig = HermesAgentEnvConfig

    tool_context_mod = types.ModuleType("environments.tool_context")
    tool_context_mod.ToolContext = object

    stubbed_modules = {
        "datasets": datasets_mod,
        "atroposlib": atropos_mod,
        "atroposlib.envs": atropos_envs_mod,
        "atroposlib.envs.base": atropos_envs_base_mod,
        "atroposlib.envs.server_handling": atropos_server_mod,
        "atroposlib.envs.server_handling.server_manager": atropos_server_manager_mod,
        "atroposlib.type_definitions": atropos_types_mod,
        "environments.agent_loop": agent_loop_mod,
        "environments.hermes_base_env": hermes_base_env_mod,
        "environments.tool_context": tool_context_mod,
    }

    spec = importlib.util.spec_from_file_location("test_hermes_swe_env_target", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubbed_modules):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class RecordingCtx:
    def __init__(self, terminal_results):
        self.writes = []
        self.terminal_calls = []
        self._terminal_results = list(terminal_results)

    def write_file(self, path, content):
        self.writes.append((path, content))
        return {}

    def terminal(self, command, timeout=180):
        self.terminal_calls.append((command, timeout))
        if self._terminal_results:
            return self._terminal_results.pop(0)
        return {"exit_code": 0, "output": ""}


@pytest.mark.asyncio
async def test_compute_reward_writes_verifier_file_for_code_with_quotes():
    module = _load_hermes_swe_module()
    env = module.HermesSweEnv.__new__(module.HermesSweEnv)
    env.reward_buffer = []

    test_code = 'assert candidate("x") == "x"'
    ctx = RecordingCtx(
        [
            {"exit_code": 0, "output": "ok"},
            {"exit_code": 0, "output": ""},
        ]
    )

    reward = await module.HermesSweEnv.compute_reward(
        env,
        {"test_code": test_code},
        result=object(),
        ctx=ctx,
    )

    assert reward == 1.0
    assert env.reward_buffer == [1.0]
    assert ctx.writes == [("/workspace/.hermes_reward_test.py", test_code)]
    assert ctx.terminal_calls == [
        ("cd /workspace && python3 .hermes_reward_test.py", 60),
        ("rm -f /workspace/.hermes_reward_test.py", 10),
    ]
    assert all("python3 -c" not in command for command, _ in ctx.terminal_calls)
    assert all(test_code not in command for command, _ in ctx.terminal_calls)


@pytest.mark.asyncio
async def test_compute_reward_does_not_shell_interpolate_injection_like_verifier():
    module = _load_hermes_swe_module()
    env = module.HermesSweEnv.__new__(module.HermesSweEnv)
    env.reward_buffer = []

    verifier = '"; touch /workspace/pwned #'
    ctx = RecordingCtx(
        [
            {"exit_code": 1, "output": "SyntaxError"},
            {"exit_code": 0, "output": ""},
            {"exit_code": 0, "output": "/workspace/solution.py\n"},
        ]
    )

    reward = await module.HermesSweEnv.compute_reward(
        env,
        {"test_code": verifier},
        result=object(),
        ctx=ctx,
    )

    assert reward == 0.1
    assert env.reward_buffer == [0.1]
    assert ctx.writes == [("/workspace/.hermes_reward_test.py", verifier)]
    assert ctx.terminal_calls == [
        ("cd /workspace && python3 .hermes_reward_test.py", 60),
        ("rm -f /workspace/.hermes_reward_test.py", 10),
        ("find /workspace -name '*.py' -newer /tmp/.start_marker 2>/dev/null | head -5", 180),
    ]
    assert all("python3 -c" not in command for command, _ in ctx.terminal_calls)
    assert all(verifier not in command for command, _ in ctx.terminal_calls)
