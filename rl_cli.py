#!/usr/bin/env python3
"""
RL Training CLI Runner

Dedicated CLI runner for RL training workflows with:
- Extended timeouts for long-running training
- RL-focused system prompts
- Full toolset including RL training tools
- Special handling for 30-minute check intervals

Usage:
    python rl_cli.py "Train a model on GSM8k for math reasoning"
    python rl_cli.py --interactive
    python rl_cli.py --list-environments

Environment Variables:
    TINKER_API_KEY: API key for Tinker service (required)
    WANDB_API_KEY: API key for WandB metrics (required)
    OPENROUTER_API_KEY: API key for OpenRouter (required for agent)
"""

import asyncio
import os
import sys
from pathlib import Path

import fire
import yaml

# Load .env from ~/.hermes/.env first, then project root as dev fallback
from dotenv import load_dotenv

_hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
_user_env = _hermes_home / ".env"
_project_env = Path(__file__).parent / '.env'

if _user_env.exists():
    try:
        load_dotenv(dotenv_path=_user_env, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=_user_env, encoding="latin-1")
    print(f"✅ Loaded environment variables from {_user_env}")
elif _project_env.exists():
    try:
        load_dotenv(dotenv_path=_project_env, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=_project_env, encoding="latin-1")
    print(f"✅ Loaded environment variables from {_project_env}")

# Set terminal working directory to tinker-atropos submodule
# This ensures terminal commands run in the right context for RL work
tinker_atropos_dir = Path(__file__).parent / 'tinker-atropos'
if tinker_atropos_dir.exists():
    os.environ['TERMINAL_CWD'] = str(tinker_atropos_dir)
    os.environ['HERMES_QUIET'] = '1'  # Disable temp subdirectory creation
    print(f"📂 Terminal working directory: {tinker_atropos_dir}")
else:
    # Fall back to hermes-agent directory if submodule not found
    os.environ['TERMINAL_CWD'] = str(Path(__file__).parent)
    os.environ['HERMES_QUIET'] = '1'
    print(f"⚠️  tinker-atropos submodule not found, using: {Path(__file__).parent}")

# Import agent and tools
from run_agent import AIAgent
from model_tools import get_tool_definitions, check_toolset_requirements
from tools.rl_training_tool import check_rl_api_keys, get_missing_keys


# ============================================================================
# Config Loading
# ============================================================================

from hermes_constants import OPENROUTER_BASE_URL

DEFAULT_MODEL = "anthropic/claude-opus-4.5"
DEFAULT_BASE_URL = OPENROUTER_BASE_URL


def load_hermes_config() -> dict:
    """
    Load configuration from ~/.hermes/config.yaml.
    
    Returns:
        dict: Configuration with model, base_url, etc.
    """
    config_path = _hermes_home / 'config.yaml'
    
    config = {
        "model": DEFAULT_MODEL,
        "base_url": DEFAULT_BASE_URL,
    }
    
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            
            # Get model from config
            if "model" in file_config:
                if isinstance(file_config["model"], str):
                    config["model"] = file_config["model"]
                elif isinstance(file_config["model"], dict):
                    config["model"] = file_config["model"].get("default", DEFAULT_MODEL)
            
            # Get base_url if specified
            if "base_url" in file_config:
                config["base_url"] = file_config["base_url"]
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to load config.yaml: {e}")
    
    return config


# ============================================================================
# RL-Specific Configuration
# ============================================================================

# Extended timeouts for long-running RL operations
RL_MAX_ITERATIONS = 200  # Allow many more iterations for long workflows

# RL-focused system prompt
RL_SYSTEM_PROMPT = """You are an automated post-training engineer specializing in reinforcement learning for language models.

## Your Capabilities

You have access to RL training tools for running reinforcement learning on models through Tinker-Atropos:

1. **DISCOVER**: Use `rl_list_environments` to see available RL environments
2. **INSPECT**: Read environment files to understand how they work (verifiers, data loading, rewards)
3. **INSPECT DATA**: Use terminal to explore HuggingFace datasets and understand their format
4. **CREATE**: Copy existing environments as templates, modify for your needs
5. **CONFIGURE**: Use `rl_select_environment` and `rl_edit_config` to set up training
6. **TEST**: Always use `rl_test_inference` before full training to validate your setup
7. **TRAIN**: Use `rl_start_training` to begin, `rl_check_status` to monitor
8. **EVALUATE**: Use `rl_get_results` and analyze WandB metrics to assess performance

## Environment Files

Environment files are located in: `tinker-atropos/tinker_atropos/environments/`

Study existing environments to learn patterns. Look for:
- `load_dataset()` calls - how data is loaded
- `score_answer()` / `score()` - verification logic
- `get_next_item()` - prompt formatting
- `system_prompt` - instruction format
- `config_init()` - default configuration

## Creating New Environments

To create a new environment:
1. Read an existing environment file (e.g., gsm8k_tinker.py)
2. Use terminal to explore the target dataset format
3. Copy the environment file as a template
4. Modify the dataset loading, prompt formatting, and verifier logic
5. Test with `rl_test_inference` before training

## Important Guidelines

- **Always test before training**: Training runs take hours - verify everything works first
- **Monitor metrics**: Check WandB for reward/mean and percent_correct
- **Status check intervals**: Wait at least 30 minutes between status checks
- **Early stopping**: Stop training early if metrics look bad or stagnant
- **Iterate quickly**: Start with small total_steps to validate, then scale up

## Available Toolsets

You have access to:
- **RL tools**: Environment discovery, config management, training, testing
- **Terminal**: Run commands, inspect files, explore datasets
- **Web**: Search for information, documentation, papers
- **File tools**: Read and modify code files

When asked to train a model, follow this workflow:
1. List available environments
2. Select and configure the appropriate environment
3. Test with sample prompts
4. Start training with conservative settings
5. Monitor progress and adjust as needed
"""

# Toolsets to enable for RL workflows
RL_TOOLSETS = ["terminal", "web", "rl"]


# ============================================================================
# Helper Functions
# ============================================================================

def check_requirements():
    """Check that all required environment variables and services are available."""
    errors = []
    
    # Check API keys
    if not os.getenv("OPENROUTER_API_KEY"):
        errors.append("OPENROUTER_API_KEY not set - required for agent")
    
    missing_rl_keys = get_missing_keys()
    if missing_rl_keys:
        errors.append(f"Missing RL API keys: {', '.join(missing_rl_keys)}")
    
    if errors:
        print("❌ Missing requirements:")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease set these environment variables in your .env file or shell.")
        return False
    
    return True


def check_tinker_atropos():
    """Check if tinker-atropos submodule is properly set up."""
    tinker_path = Path(__file__).parent / "tinker-atropos"
    
    if not tinker_path.exists():
        return False, "tinker-atropos submodule not found. Run: git submodule update --init"
    
    envs_path = tinker_path / "tinker_atropos" / "environments"
    if not envs_path.exists():
        return False, f"environments directory not found at {envs_path}"
    
    env_files = list(envs_path.glob("*.py"))
    env_files = [f for f in env_files if not f.name.startswith("_")]
    
    return True, {"path": str(tinker_path), "environments_count": len(env_files)}


def list_environments_sync():
    """List available environments (synchronous wrapper)."""
    from tools.rl_training_tool import rl_list_environments
    import json
    
    async def _list():
        result = await rl_list_environments()
        return json.loads(result)
    
    return asyncio.run(_list())


# ============================================================================
# Main CLI
# ============================================================================

def main(
    task: str = None,
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    max_iterations: int = RL_MAX_ITERATIONS,
    interactive: bool = False,
    list_environments: bool = False,
    check_server: bool = False,
    verbose: bool = False,
    save_trajectories: bool = True,
):
    """
    RL Training CLI - Dedicated runner for RL training workflows.
    
    Args:
        task: The training task/goal (e.g., "Train a model on GSM8k for math")
        model: Model to use for the agent (reads from ~/.hermes/config.yaml if not provided)
        api_key: OpenRouter API key (uses OPENROUTER_API_KEY env var if not provided)
        base_url: API base URL (reads from config or defaults to OpenRouter)
        max_iterations: Maximum agent iterations (default: 200 for long workflows)
        interactive: Run in interactive mode (multiple conversations)
        list_environments: Just list available RL environments and exit
        check_server: Check if RL API server is running and exit
        verbose: Enable verbose logging
        save_trajectories: Save conversation trajectories (default: True for RL)
    
    Examples:
        # Train on a specific environment
        python rl_cli.py "Train a model on GSM8k math problems"
        
        # Interactive mode
        python rl_cli.py --interactive
        
        # List available environments
        python rl_cli.py --list-environments
        
        # Check server status
        python rl_cli.py --check-server
    """
    # Load config from ~/.hermes/config.yaml
    config = load_hermes_config()
    
    # Use config values if not explicitly provided
    if model is None:
        model = config["model"]
    if base_url is None:
        base_url = config["base_url"]
    
    print("🎯 RL Training Agent")
    print("=" * 60)
    
    # Handle setup check
    if check_server:
        print("\n🔍 Checking tinker-atropos setup...")
        ok, result = check_tinker_atropos()
        if ok:
            print("✅ tinker-atropos submodule found")
            print(f"   Path: {result.get('path')}")
            print(f"   Environments found: {result.get('environments_count', 0)}")
            
            # Also check API keys
            missing = get_missing_keys()
            if missing:
                print(f"\n⚠️  Missing API keys: {', '.join(missing)}")
                print("   Add them to ~/.hermes/.env")
            else:
                print("✅ API keys configured")
        else:
            print(f"❌ tinker-atropos not set up: {result}")
            print("\nTo set up:")
            print("  git submodule update --init")
            print("  pip install -e ./tinker-atropos")
        return
    
    # Handle environment listing
    if list_environments:
        print("\n📋 Available RL Environments:")
        print("-" * 40)
        try:
            data = list_environments_sync()
            if "error" in data:
                print(f"❌ Error: {data['error']}")
                return
            
            envs = data.get("environments", [])
            if not envs:
                print("No environments found.")
                print("\nMake sure tinker-atropos is set up:")
                print("  git submodule update --init")
                return
            
            for env in envs:
                print(f"\n  📦 {env['name']}")
                print(f"     Class: {env['class_name']}")
                print(f"     Path: {env['file_path']}")
                if env.get('description'):
                    desc = env['description'][:100] + "..." if len(env.get('description', '')) > 100 else env.get('description', '')
                    print(f"     Description: {desc}")
            
            print(f"\n📊 Total: {len(envs)} environments")
            print("\nUse `rl_select_environment(name)` to select an environment for training.")
        except Exception as e:
            print(f"❌ Error listing environments: {e}")
            print("\nMake sure tinker-atropos is set up:")
            print("  git submodule update --init")
            print("  pip install -e ./tinker-atropos")
        return
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Set default task if none provided
    if not task and not interactive:
        print("\n⚠️  No task provided. Use --interactive for interactive mode or provide a task.")
        print("\nExamples:")
        print('  python rl_cli.py "Train a model on GSM8k math problems"')
        print('  python rl_cli.py "Create an RL environment for code generation"')
        print('  python rl_cli.py --interactive')
        return
    
    # Get API key
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ No API key provided. Set OPENROUTER_API_KEY or pass --api-key")
        sys.exit(1)
    
    print(f"\n🤖 Model: {model}")
    print(f"🔧 Max iterations: {max_iterations}")
    print(f"📁 Toolsets: {', '.join(RL_TOOLSETS)}")
    print("=" * 60)
    
    # Create agent with RL configuration
    agent = AIAgent(
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_iterations=max_iterations,
        enabled_toolsets=RL_TOOLSETS,
        save_trajectories=save_trajectories,
        verbose_logging=verbose,
        quiet_mode=False,
        ephemeral_system_prompt=RL_SYSTEM_PROMPT,
    )
    
    if interactive:
        # Interactive mode - multiple conversations
        print("\n🔄 Interactive RL Training Mode")
        print("Type 'quit' or 'exit' to end the session.")
        print("Type 'status' to check active training runs.")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("\n🎯 RL Task> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ('quit', 'exit', 'q'):
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'status':
                    # Quick status check
                    from tools.rl_training_tool import rl_list_runs
                    import json
                    result = asyncio.run(rl_list_runs())
                    runs = json.loads(result)
                    if isinstance(runs, list) and runs:
                        print("\n📊 Active Runs:")
                        for run in runs:
                            print(f"  - {run['run_id']}: {run['environment']} ({run['status']})")
                    else:
                        print("\nNo active runs.")
                    continue
                
                # Run the agent
                print("\n" + "=" * 60)
                response = agent.run_conversation(user_input)
                print("\n" + "=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
    else:
        # Single task mode
        print(f"\n📝 Task: {task}")
        print("-" * 40)
        
        try:
            response = agent.run_conversation(task)
            print("\n" + "=" * 60)
            print("✅ Task completed")
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    fire.Fire(main)
