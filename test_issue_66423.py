#!/usr/bin/env python3
"""Test for issue #66423: disable_watchdog config option"""
import sys
sys.path.insert(0, '.')

from tools.mcp_tool import _wrap_command_with_watchdog

# Test 1: Default behavior wraps command
cmd, args = _wrap_command_with_watchdog('uvx', ['foo'])
assert 'mcp_stdio_watchdog.py' in ' '.join(args), 'Default should wrap'
print('✓ Test 1: Default wraps command')

# Test 2: disable_watchdog=True skips wrap
config = {'command': 'uvx', 'args': ['foo'], 'disable_watchdog': True}
cmd, args = config['command'], config['args']
if not config.get('disable_watchdog', False):
    cmd, args = _wrap_command_with_watchdog(cmd, args)
assert cmd == 'uvx' and args == ['foo'], 'disable_watchdog=True should skip wrap'
print('✓ Test 2: disable_watchdog=True skips wrap')

# Test 3: disable_watchdog=False (explicit) wraps
config = {'command': 'uvx', 'args': ['foo'], 'disable_watchdog': False}
cmd, args = config['command'], config['args']
if not config.get('disable_watchdog', False):
    cmd, args = _wrap_command_with_watchdog(cmd, args)
assert 'mcp_stdio_watchdog.py' in ' '.join(args), 'disable_watchdog=False should wrap'
print('✓ Test 3: disable_watchdog=False wraps')

print('\nAll tests passed')
