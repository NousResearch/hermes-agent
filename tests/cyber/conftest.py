"""Cyber test isolation helpers."""

import importlib


# Import the real registry before legacy cyber tests decide whether to install
# lightweight stubs.  Several of those modules only stub when ``tools.registry``
# is absent; preloading the real module prevents a partial stub from leaking
# into later tests that import the full agent stack.
importlib.import_module("tools.registry")
