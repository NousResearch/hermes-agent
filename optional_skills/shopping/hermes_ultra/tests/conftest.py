"""Test configuration for Hermes Ultra shopping skill tests."""

import os
import sys

# Ensure skill scripts are importable
_skill_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
)
sys.path.insert(0, os.path.abspath(_skill_root))
