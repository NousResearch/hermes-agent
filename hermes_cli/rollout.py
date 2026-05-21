import argparse
import concurrent.futures
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

from run_agent import AIAgent
from hermes_cli.config import load_config
from tools.environments.local import LocalEnvironment
from hermes_constants import get_hermes_home

def cmd_rollout(args: argparse.Namespace):
    """
    GSPO/RLVR Rollout Orchestrator.
    Spawns parallel Hermes agents to generate multiple verified trajectories per prompt.
    """
    print(f"\n🚀 Starting GSPO Rollout Orchestrator")
    print(f"Prompt: {args.prompt}")
    print(f"Verifier: {args.verifier}")
    print(f"Group Size (G): {args.G}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output}\n")

    config = load_config()
    default_model = config.get("model", {}).get("default", "gemini-2.5-flash")
    default_provider = config.get("model", {}).get("provider", "gemini")

    # Set temperature for the providers
    os.environ["HERMES_TEMPERATURE"] = str(args.temperature)

    # Initialize temp dir in a safe, non-system location so write_file doesn't block it
    gspo_tmp_dir = os.path.join(get_hermes_home(), "gspo_tmp")
    os.makedirs(gspo_tmp_dir, exist_ok=True)
    temp_dirs = []
    try:
        # Create G isolated directories
        for i in range(args.G):
            d = Path(tempfile.mkdtemp(prefix=f"hermes_gspo_run{i}_", dir=gspo_tmp_dir))
            temp_dirs.append(d)

        print(f"Spawning {args.G} isolated agents...")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.G) as executor:
            futures = []
            for i, d in enumerate(temp_dirs):
                futures.append(
                    executor.submit(_run_single_trajectory, i, args.prompt, args.verifier, d, default_model, default_provider)
                )
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        # Sort results back into their original index order (optional, but clean)
        results.sort(key=lambda x: x[0])

        completions = []
        rewards = []

        for idx, trajectory, reward in results:
            if trajectory:
                # Format to match Unsloth GSPO structure
                # We expect trajectory to be a list of conversation turns (from/value dicts)
                # Unsloth/TRL completions list should just be the agent responses, but
                # ShareGPT format typically expects standard roles.
                # For GSPO, completions array holds arrays of dicts [{"role": "assistant", "content": ...}]
                
                # We extract the pure assistant text/tool_calls from the trajectory
                # The AIAgent._convert_to_trajectory_format gives us ShareGPT format:
                # {"conversations": [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}, ...]}
                
                # To match standard HF JSONL format for the completion array:
                completion_msg = ""
                
                # Format to match standard HF JSONL GSPO format
                # The _convert_to_trajectory_format produces ShareGPT style with "human", "gpt", "tool"
                # We map these to Unsloth's "user", "assistant", "tool" structure
                completion_sequence = []
                
                for msg in trajectory:
                    if isinstance(msg, dict):
                        # Map internal role strings
                        role_map = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool"}
                        role = role_map.get(msg.get("from", "user"), "user")
                        content = msg.get("value", "")
                        
                        # Only append if not the original system/user prompt (this is for the completion half)
                        if role != "user" and role != "system":
                            completion_sequence.append({"role": role, "content": content})

                completions.append(completion_sequence)
                rewards.append(reward)

        output_data = {
            "prompt": [{"role": "user", "content": args.prompt}],
            "completions": completions,
            "rewards": rewards
        }

        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(output_data) + "\n")

        print(f"\n✅ Exported {len(rewards)} sequence-level rewards to {args.output}")
        print(f"Rewards distribution: {rewards}")

    finally:
        # Cleanup disabled for debugging
        # for d in temp_dirs:
        #     shutil.rmtree(d, ignore_errors=True)
        pass

def _run_single_trajectory(idx: int, prompt: str, verifier_cmd: str, workdir: Path, model: str, provider: str) -> Tuple[int, Dict[str, Any], float]:
    """Runs a single agent, tracks the trajectory, and executes the verification."""
    print(f"  [Agent {idx}] Starting in {workdir.name}")
    
    task_id = f"gspo_run_{idx}_{workdir.name}"
    
    # We must route the terminal commands for this task to the isolated temp dir
    from tools.terminal_tool import register_task_env_overrides
    overrides = {"cwd": str(workdir)}
    register_task_env_overrides(task_id, overrides)

    agent = AIAgent(
        model=model,
        provider=provider,
        skip_context_files=True,
        skip_memory=True,
        max_iterations=10, # Keep it bounded for rollouts
        quiet_mode=True,   # Less terminal spam
        save_trajectories=False,
    )
    
    # Inject the absolute workspace path into the prompt to ensure file tools write to the correct place
    isolated_prompt = f"[System directive: You are executing in an isolated temporary workspace. Your absolute working directory is {workdir}. All file operations MUST be strictly relative to this directory.]\n\n{prompt}"
    
    try:
        result = agent.run_conversation(isolated_prompt, task_id=task_id)
        
        trajectory = agent._convert_to_trajectory_format(
            result["messages"],
            prompt,
            result["completed"]
        )
    except Exception as e:
        print(f"  [Agent {idx}] Agent crashed: {e}")
        return idx, None, 0.0

    print(f"  [Agent {idx}] Finished generation. Verifying...")
    
    # Run the verifier script in the isolated workdir
    term = LocalEnvironment(cwd=str(workdir))
    verify_result = term.execute(verifier_cmd, timeout=30)
    
    # BaseEnvironment returns "returncode", but terminal tool returns "exit_code"
    code = verify_result.get("returncode", verify_result.get("exit_code", -1))
    reward = 1.0 if code == 0 else 0.0
    print(f"  [Agent {idx}] Reward: {reward}")

    return idx, trajectory, reward
