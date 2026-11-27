"""Test-support suites shipped with Hermes for out-of-tree implementations of its ports.

Installed with the package (``pyproject.toml`` package discovery), so a plugin's own test suite
imports them without a Hermes checkout. Importing a suite imports ``pytest``; nothing here is
loaded at runtime.
"""
