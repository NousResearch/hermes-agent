"""Hindsight long-term memory integration.

This module is only active when Hindsight is configured at
~/.hindsight/config.json or via environment variables.

All hindsight-client imports are deferred to avoid ImportError
when the package is not installed.
"""
