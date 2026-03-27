#!/usr/bin/env python3
"""
Test script for remote agent functionality.

Run this to verify the remote agent module is working correctly.
"""

import sys
sys.path.insert(0, '/home/hermes/hermes-agent')

def test_remote_agent_module():
    """Test that the remote agent module loads correctly."""
    print("Testing remote agent module...")
    
    try:
        from tools.remote_agent import (
            get_remote_agents,
            call_remote_agent,
            load_remote_agents_config,
            refresh_remote_agents_config,
        )
        print("✓ Module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import module: {e}")
        return False
    
    # Test config loading
    print("\nTesting config loading...")
    config = load_remote_agents_config()
    print(f"  Config loaded: {config}")
    
    # Test get_remote_agents
    print("\nTesting get_remote_agents()...")
    agents = get_remote_agents()
    print(f"  Remote agents: {list(agents.keys())}")
    
    print("\n✓ All tests passed!")
    return True

def test_delegate_tool_integration():
    """Test that delegate_tool can import remote agent functions."""
    print("\nTesting delegate_tool integration...")
    
    try:
        from tools.delegate_tool import REMOTE_AGENTS_AVAILABLE
        if REMOTE_AGENTS_AVAILABLE:
            print("✓ Remote agents available in delegate_tool")
        else:
            print("✗ Remote agents NOT available in delegate_tool")
            return False
    except Exception as e:
        print(f"✗ Failed to check integration: {e}")
        return False
    
    print("✓ Delegate tool integration test passed!")
    return True

if __name__ == "__main__":
    success = True
    success &= test_remote_agent_module()
    success &= test_delegate_tool_integration()
    
    if success:
        print("\n" + "="*50)
        print("ALL TESTS PASSED!")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("SOME TESTS FAILED")
        print("="*50)
        sys.exit(1)
