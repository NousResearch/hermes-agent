"""Test-only: the daemon believes its interpreter is a launcher trampoline.

``UGW_TRAMPOLINE`` names an executable that spawns the real interpreter with the same
argv and waits for it (the shape of uv's venv ``python.exe`` on Windows), so every
``Popen([sys.executable, ...])`` child of this daemon is a launcher whose grandchild is
the real Python. No other behaviour changes.
"""
import os
import runpy
import sys

sys.executable = os.environ['UGW_TRAMPOLINE']
runpy.run_module('gateway.run', run_name='__main__')
