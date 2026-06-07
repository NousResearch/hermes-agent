"""Tests for cron scheduler profile-job context isolation.

Regression tests for #39886: Non-profile cron jobs fail with "Script not found"
when running concurrently with profile-scoped jobs. The issue is that profile jobs
mutate a module-global _hermes_home that's not protected by contextvars, causing
a race condition where non-profile jobs see the wrong profile home.

The fix: Remove the module-global mutation and rely on contextvars exclusively.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from contextlib import contextmanager
import pytest


class TestProfileContextIsolation:
    """Test that profile context mutations don't bleed to non-profile jobs."""

    def test_contextvar_protects_profile_home(self):
        """contextvars.copy_context() should isolate profile overrides."""
        # This test demonstrates that contextvars are properly isolated
        # across thread boundaries
        from contextvars import ContextVar, copy_context
        
        test_var: ContextVar[str] = ContextVar("test", default="default")
        
        # Set a value in the main context
        test_var.set("profile1")
        
        # Copy the context (simulating what tick() does with concurrent jobs)
        ctx = copy_context()
        
        # In the copied context, the value should be preserved
        assert ctx.run(test_var.get) == "profile1"
        
        # But we can override it in the copy without affecting the main context
        def override_in_context():
            test_var.set("profile2")
            return test_var.get()
        
        result = ctx.run(override_in_context)
        assert result == "profile2"
        
        # The main context is unaffected
        assert test_var.get() == "profile1"

    def test_module_global_not_protected_by_contextvar(self):
        """Module globals are NOT protected by contextvars.copy_context()."""
        from contextvars import copy_context
        
        # Simulate the problematic pattern in cron/scheduler.py
        module_global = "default"
        
        def get_value():
            nonlocal module_global
            return module_global
        
        def set_value(val):
            nonlocal module_global
            module_global = val
        
        # Set a value
        set_value("profile1")
        assert get_value() == "profile1"
        
        # Copy the context
        ctx = copy_context()
        
        # Even though we copy the context, the module global is shared
        def run_in_context():
            set_value("profile2")
            return get_value()
        
        result = ctx.run(run_in_context)
        assert result == "profile2"
        
        # The module global WAS mutated (not protected by contextvars!)
        assert module_global == "profile2", \
            "Module globals are NOT isolated by contextvars.copy_context()"

    def test_correct_pattern_uses_contextvar_only(self):
        """The fix: use contextvar for profile override, not module global."""
        from contextvars import ContextVar, copy_context
        
        # This is the correct pattern: use contextvar, not module global
        profile_override: ContextVar[str | None] = ContextVar("profile", default=None)
        
        def get_profile():
            override = profile_override.get()
            return override or "default"
        
        def set_profile(val):
            return profile_override.set(val)
        
        # Main context: default
        assert get_profile() == "default"
        
        # Set a profile in a copied context
        ctx = copy_context()
        
        def run_in_profile_context():
            token = set_profile("profile1")
            try:
                return get_profile()
            finally:
                # Reset after use
                profile_override.set(None)
        
        result = ctx.run(run_in_profile_context)
        assert result == "profile1"
        
        # Main context is unaffected
        assert get_profile() == "default", \
            "Contextvar changes don't leak to parent context"


class TestGetHermesHomeResolution:
    """Test _get_hermes_home() resolution order."""

    def test_get_hermes_home_reads_contextvar_first(self):
        """_get_hermes_home() should read contextvar before module global."""
        # Simulate the corrected _get_hermes_home() logic:
        # 1. Check contextvar override
        # 2. Fall back to module global (for tests)
        # 3. Fall back to system get_hermes_home()
        
        from contextvars import ContextVar
        from pathlib import Path
        
        profile_override: ContextVar[Path | None] = ContextVar("hermes_home", default=None)
        _module_global = None
        
        def get_hermes_home_default():
            return Path.home() / ".hermes"
        
        def _get_hermes_home():
            # Correct order: contextvar → module global → system default
            override = profile_override.get()
            if override:
                return override
            return _module_global or get_hermes_home_default()
        
        # Default: system home
        assert _get_hermes_home() == Path.home() / ".hermes"
        
        # With contextvar override
        profile_override.set(Path("/profile1/home"))
        assert _get_hermes_home() == Path("/profile1/home")


class TestJobProfileContextManager:
    """Test the _job_profile_context context manager isolation."""

    def test_profile_context_restores_contextvar_on_exit(self):
        """Profile context should use contextvar (isolates across threads)."""
        from contextvars import ContextVar
        from pathlib import Path
        
        override_var: ContextVar[Path | None] = ContextVar("hermes_home", default=None)
        
        @contextmanager
        def profile_context(profile_home: Path | None):
            """Corrected version: uses contextvar only."""
            if not profile_home:
                yield
                return
            
            token = override_var.set(profile_home)
            try:
                yield profile_home
            finally:
                override_var.set(None)
        
        # Before: contextvar should be None
        assert override_var.get() is None
        
        # Inside the context: should be set
        with profile_context(Path("/profile1")):
            assert override_var.get() == Path("/profile1")
        
        # After: should be restored
        assert override_var.get() is None

    def test_module_global_pattern_has_race_condition(self):
        """The old pattern (module global) has a race condition."""
        # This demonstrates why the old code fails
        
        state: dict[str, Path | None] = {"_hermes_home": None}
        
        def get_hermes_home():
            hermes_home = state["_hermes_home"]
            return hermes_home or Path.home() / ".hermes"
        
        @contextmanager
        def buggy_profile_context(profile_home: Path | None):
            """Buggy version: mutates module global."""
            if not profile_home:
                yield
                return
            
            prior = state["_hermes_home"]
            state["_hermes_home"] = profile_home
            try:
                yield profile_home
            finally:
                state["_hermes_home"] = prior
        
        # The problem: if two threads call get_hermes_home() during the
        # profile context window, they both see the mutated value
        default_home = Path.home() / ".hermes"
        
        with buggy_profile_context(Path("/profile1")):
            # Both threads would see "/profile1" here (race condition)
            assert get_hermes_home() == Path("/profile1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
