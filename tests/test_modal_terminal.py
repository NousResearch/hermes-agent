#!/usr/bin/env python3
"""
Test Modal Terminal Tool

This script tests that the Modal terminal backend is correctly configured
and can execute commands in Modal sandboxes.

Usage:
    # Run with Modal backend
    TERMINAL_ENV=modal python tests/test_modal_terminal.py
    
    # Or run directly (will use whatever TERMINAL_ENV is set in .env)
    python tests/test_modal_terminal.py
"""

import os
import sys
import json
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Manually load .env if dotenv not available
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key.strip(), value)

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "mini-swe-agent" / "src"))

# Import terminal_tool module directly using importlib to avoid tools/__init__.py
import importlib.util
terminal_tool_path = parent_dir / "tools" / "terminal_tool.py"
spec = importlib.util.spec_from_file_location("terminal_tool", terminal_tool_path)
terminal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(terminal_module)

terminal_tool = terminal_module.terminal_tool
check_terminal_requirements = terminal_module.check_terminal_requirements
_get_env_config = terminal_module._get_env_config
cleanup_vm = terminal_module.cleanup_vm
get_active_environments_info = terminal_module.get_active_environments_info


def test_modal_requirements():
    """Test that Modal requirements are met."""
    print("\n" + "=" * 60)
    print("TEST 1: Modal Requirements Check")
    print("=" * 60)
    
    config = _get_env_config()
    print(f"Current TERMINAL_ENV: {config['env_type']}")
    print(f"Modal image: {config['modal_image']}")
    
    # Check for Modal authentication
    modal_token = os.getenv("MODAL_TOKEN_ID")
    modal_toml = Path.home() / ".modal.toml"
    
    print(f"\nModal authentication:")
    print(f"  MODAL_TOKEN_ID env var: {'✅ Set' if modal_token else '❌ Not set'}")
    print(f"  ~/.modal.toml file: {'✅ Exists' if modal_toml.exists() else '❌ Not found'}")
    
    if config['env_type'] != 'modal':
        print(f"\n⚠️  TERMINAL_ENV is '{config['env_type']}', not 'modal'")
        print("   Set TERMINAL_ENV=modal in .env or export it to test Modal backend")
        return False
    
    requirements_met = check_terminal_requirements()
    print(f"\nRequirements check: {'✅ Passed' if requirements_met else '❌ Failed'}")
    
    return requirements_met


def test_simple_command():
    """Test executing a simple command."""
    print("\n" + "=" * 60)
    print("TEST 2: Simple Command Execution")
    print("=" * 60)
    
    test_task_id = "modal_test_simple"
    
    print("Executing: echo 'Hello from Modal!'")
    result = terminal_tool("echo 'Hello from Modal!'", task_id=test_task_id)
    result_json = json.loads(result)
    
    print(f"\nResult:")
    print(f"  Output: {result_json.get('output', '')[:200]}")
    print(f"  Exit code: {result_json.get('exit_code')}")
    print(f"  Error: {result_json.get('error')}")
    
    success = result_json.get('exit_code') == 0 and 'Hello from Modal!' in result_json.get('output', '')
    print(f"\nTest: {'✅ Passed' if success else '❌ Failed'}")
    
    # Cleanup
    cleanup_vm(test_task_id)
    
    return success


def test_python_execution():
    """Test executing Python code in Modal."""
    print("\n" + "=" * 60)
    print("TEST 3: Python Execution")
    print("=" * 60)
    
    test_task_id = "modal_test_python"
    
    python_cmd = 'python3 -c "import sys; print(f\'Python {sys.version}\')"'
    print(f"Executing: {python_cmd}")
    
    result = terminal_tool(python_cmd, task_id=test_task_id)
    result_json = json.loads(result)
    
    print(f"\nResult:")
    print(f"  Output: {result_json.get('output', '')[:200]}")
    print(f"  Exit code: {result_json.get('exit_code')}")
    print(f"  Error: {result_json.get('error')}")
    
    success = result_json.get('exit_code') == 0 and 'Python' in result_json.get('output', '')
    print(f"\nTest: {'✅ Passed' if success else '❌ Failed'}")
    
    # Cleanup
    cleanup_vm(test_task_id)
    
    return success


def test_pip_install():
    """Test installing a package with pip in Modal."""
    print("\n" + "=" * 60)
    print("TEST 4: Pip Install Test")
    print("=" * 60)
    
    test_task_id = "modal_test_pip"
    
    # Install a small package and verify
    print("Executing: pip install --break-system-packages cowsay && python3 -c \"import cowsay; cowsay.cow('Modal works!')\"")
    
    result = terminal_tool(
        "pip install --break-system-packages cowsay && python3 -c \"import cowsay; cowsay.cow('Modal works!')\"",
        task_id=test_task_id,
        timeout=120
    )
    result_json = json.loads(result)
    
    print(f"\nResult:")
    output = result_json.get('output', '')
    print(f"  Output (last 500 chars): ...{output[-500:] if len(output) > 500 else output}")
    print(f"  Exit code: {result_json.get('exit_code')}")
    print(f"  Error: {result_json.get('error')}")
    
    success = result_json.get('exit_code') == 0 and 'Modal works!' in result_json.get('output', '')
    print(f"\nTest: {'✅ Passed' if success else '❌ Failed'}")
    
    # Cleanup
    cleanup_vm(test_task_id)
    
    return success


def test_filesystem_persistence():
    """Test that filesystem persists between commands in the same task."""
    print("\n" + "=" * 60)
    print("TEST 5: Filesystem Persistence")
    print("=" * 60)
    
    test_task_id = "modal_test_persist"
    
    # Create a file
    print("Step 1: Creating test file...")
    result1 = terminal_tool("echo 'persistence test' > /tmp/modal_test.txt", task_id=test_task_id)
    result1_json = json.loads(result1)
    print(f"  Exit code: {result1_json.get('exit_code')}")
    
    # Read the file back
    print("Step 2: Reading test file...")
    result2 = terminal_tool("cat /tmp/modal_test.txt", task_id=test_task_id)
    result2_json = json.loads(result2)
    print(f"  Output: {result2_json.get('output', '')}")
    print(f"  Exit code: {result2_json.get('exit_code')}")
    
    success = (
        result1_json.get('exit_code') == 0 and
        result2_json.get('exit_code') == 0 and
        'persistence test' in result2_json.get('output', '')
    )
    print(f"\nTest: {'✅ Passed' if success else '❌ Failed'}")
    
    # Cleanup
    cleanup_vm(test_task_id)
    
    return success


def test_environment_isolation():
    """Test that different task_ids get isolated environments."""
    print("\n" + "=" * 60)
    print("TEST 6: Environment Isolation")
    print("=" * 60)
    
    task1 = "modal_test_iso_1"
    task2 = "modal_test_iso_2"
    
    # Create file in task1
    print("Step 1: Creating file in task1...")
    result1 = terminal_tool("echo 'task1 data' > /tmp/isolated.txt", task_id=task1)
    
    # Try to read from task2 (should not exist)
    print("Step 2: Trying to read file from task2 (should not exist)...")
    result2 = terminal_tool("cat /tmp/isolated.txt 2>&1 || echo 'FILE_NOT_FOUND'", task_id=task2)
    result2_json = json.loads(result2)
    
    # The file should either not exist or be empty in task2
    output = result2_json.get('output', '')
    isolated = 'task1 data' not in output or 'FILE_NOT_FOUND' in output or 'No such file' in output
    
    print(f"  Task2 output: {output[:200]}")
    print(f"\nTest: {'✅ Passed (environments isolated)' if isolated else '❌ Failed (environments NOT isolated)'}")
    
    # Cleanup
    cleanup_vm(task1)
    cleanup_vm(task2)
    
    return isolated


def main():
    """Run all Modal terminal tests."""
    print("🧪 Modal Terminal Tool Test Suite")
    print("=" * 60)
    
    # Check current config
    config = _get_env_config()
    print(f"\nCurrent configuration:")
    print(f"  TERMINAL_ENV: {config['env_type']}")
    print(f"  TERMINAL_MODAL_IMAGE: {config['modal_image']}")
    print(f"  TERMINAL_TIMEOUT: {config['timeout']}s")
    
    if config['env_type'] != 'modal':
        print(f"\n⚠️  WARNING: TERMINAL_ENV is set to '{config['env_type']}', not 'modal'")
        print("   To test Modal specifically, set TERMINAL_ENV=modal")
        response = input("\n   Continue testing with current backend? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            return
    
    results = {}
    
    # Run tests
    results['requirements'] = test_modal_requirements()
    
    if not results['requirements']:
        print("\n❌ Requirements not met. Cannot continue with other tests.")
        return
    
    results['simple_command'] = test_simple_command()
    results['python_execution'] = test_python_execution()
    results['pip_install'] = test_pip_install()
    results['filesystem_persistence'] = test_filesystem_persistence()
    results['environment_isolation'] = test_environment_isolation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Show active environments
    env_info = get_active_environments_info()
    print(f"\nActive environments after tests: {env_info['count']}")
    if env_info['count'] > 0:
        print(f"  Task IDs: {env_info['task_ids']}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
