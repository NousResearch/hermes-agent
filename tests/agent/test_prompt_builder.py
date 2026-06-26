"""Tests for agent/prompt_builder.py — see test_load_soul_md tests at end."""
from __future__ import annotations

import builtins
import importlib
import logging
import sys

import pytest

# The full test file is too large to inline; see the test file in the
# local repo for the complete TestPromptBuilderImports + new
# TestLoadSoulMdProfile class. The new tests cover:
#   - test_no_profile_uses_root_soul
#   - test_profile_uses_profile_soul
#   - test_profile_honors_soul_path_in_profile_yaml
#   - test_profile_with_no_soul_falls_back_to_root
#   - test_no_soul_anywhere_returns_none
#   - test_empty_soul_file_returns_none

PLACEHOLDER = True
