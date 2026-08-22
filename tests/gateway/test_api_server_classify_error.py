"""
Tests for _classify_provider_resolution_error in api_server.py.

This helper was added in #91656 to distinguish disk-full / I/O errors from
genuine provider authentication failures, so logs and user messages don't
mislead operators into chasing credentials when the disk is full.
"""

import errno

import pytest

from gateway.platforms.api_server import _classify_provider_resolution_error


class TestClassifyProviderResolutionError:
    """Verify OSError/IOError in the __cause__ chain is detected and classified distinctly."""

    def test_plain_runtime_error_is_auth_failure(self):
        """
        A plain RuntimeError with no I/O error in the chain → auth failure.
        """
        exc = RuntimeError("Invalid API key")
        log_prefix, user_prefix = _classify_provider_resolution_error(exc)
        assert log_prefix == "Provider authentication failed"
        assert user_prefix == "⚠️ Provider authentication failed"

    def test_oserror_at_root_is_io_failure(self):
        """
        RuntimeError wrapping an OSError (disk full) → I/O failure.
        """
        os_err = OSError(errno.ENOSPC, "No space left on device")
        wrapper = RuntimeError("Provider resolution failed")
        wrapper.__cause__ = os_err
        log_prefix, user_prefix = _classify_provider_resolution_error(wrapper)
        assert log_prefix == "Provider resolution failed"
        assert user_prefix == "⚠️ Provider resolution failed"

    def test_oserror_deep_in_chain(self):
        """
        OSError buried several levels deep in __cause__ chain → I/O failure.
        """
        inner = OSError(errno.EDQUOT, "Disk quota exceeded")
        middle = ValueError("Invalid config format")
        middle.__cause__ = inner
        outer = RuntimeError("Boom")
        outer.__cause__ = middle
        log_prefix, user_prefix = _classify_provider_resolution_error(outer)
        assert log_prefix == "Provider resolution failed"
        assert user_prefix == "⚠️ Provider resolution failed"

    def test_ioerror_variant(self):
        """
        IOError is treated the same as OSError (both are I/O failures).
        """
        io_err = IOError(errno.EACCES, "Permission denied")
        wrapper = RuntimeError("Failed to resolve")
        wrapper.__cause__ = io_err
        log_prefix, user_prefix = _classify_provider_resolution_error(wrapper)
        assert log_prefix == "Provider resolution failed"
        assert user_prefix == "⚠️ Provider resolution failed"

    def test_unrelated_exception_types_are_auth_failure(self):
        """
        Other exception types (ValueError, KeyError) → auth failure.
        """
        inner = KeyError("missing_key")
        outer = RuntimeError("Config error")
        outer.__cause__ = inner
        log_prefix, user_prefix = _classify_provider_resolution_error(outer)
        assert log_prefix == "Provider authentication failed"
        assert user_prefix == "⚠️ Provider authentication failed"

    def test_none_cause_chain_terminates_safely(self):
        """
        Walking the __cause__ chain when it ends with None → no crash.
        """
        exc = RuntimeError("No cause")
        # Explicitly no __cause__ set (the default)
        log_prefix, user_prefix = _classify_provider_resolution_error(exc)
        assert log_prefix == "Provider authentication failed"
        assert user_prefix == "⚠️ Provider authentication failed"
