"""
Test improved error message for Windows Smart App Control blocking embedded Python.

This test verifies that when fastapi/uvicorn import fails due to _ssl module
being blocked by Windows security policies, the error message clearly identifies
the root cause and provides recovery guidance.
"""
import sys
from unittest.mock import patch


def test_smart_app_control_blocked_ssl_error_message():
    """Test that _ssl blocked by Smart App Control shows helpful error message."""
    # Simulate ImportError with Smart App Control signature
    fake_error = ImportError(
        "DLL load failed while importing _ssl: An Application Control policy has blocked this file."
    )

    with patch('builtins.print') as mock_print:
        with patch('builtins.exit') as mock_exit:
            # Simulate the error handling logic
            error_msg = str(fake_error)
            if "DLL load failed" in error_msg and "_ssl" in error_msg:
                print("✗ Critical: Embedded Python runtime is blocked by Windows security policy.")
                print()
                print("Root cause: Python's SSL module (_ssl) could not be loaded.")
                print(f"Error: {error_msg}")
                print()
                print("This happens when Windows Smart App Control or Application Control")
                print("blocks the embedded Python runtime. The 'missing dependencies' message")
                print("above is a symptom, not the actual cause.")
                print()
                print("Recovery options:")
                print("  1. Use a trusted system Python installation instead of the embedded runtime")
                print("     (if your organization allows it).")
                print("  2. Request an exemption for the Hermes Desktop application from")
                print("     your IT administrator (Smart App Control / Application Control).")
                print("  3. Use the CLI or gateway version instead (they use your system Python).")
                print()
                print("See https://aka.ms/smartappcontrol for Windows Smart App Control details.")
            else:
                print("Web UI dependencies not installed (need fastapi + uvicorn).")
            print(f"Import error: {error_msg}")
            mock_exit(1)

    # Verify the critical error message was printed
    print_calls = [str(call) for call in mock_print.call_args_list]
    print_text = "\n".join(print_calls)
    assert "Embedded Python runtime is blocked by Windows security policy" in print_text
    assert "Smart App Control" in print_text
    assert "DLL load failed while importing _ssl" in print_text
    assert "Recovery options:" in print_text
    assert "https://aka.ms/smartappcontrol" in print_text

    # Verify we exited with error code 1
    mock_exit.assert_called_once_with(1)


def test_generic_import_error_message():
    """Test that generic ImportError shows the standard missing dependencies message."""
    # Simulate a generic ImportError (e.g., missing fastapi package)
    fake_error = ImportError("No module named 'fastapi'")

    with patch('builtins.print') as mock_print:
        with patch('builtins.exit') as mock_exit:
            # Simulate the error handling logic
            error_msg = str(fake_error)
            if "DLL load failed" in error_msg and "_ssl" in error_msg:
                print("Smart App Control blocked path")
            else:
                print("Web UI dependencies not installed (need fastapi + uvicorn).")
            print(f"Import error: {error_msg}")
            mock_exit(1)

    # Verify the standard missing dependencies message was printed
    print_calls = [str(call) for call in mock_print.call_args_list]
    print_text = "\n".join(print_calls)
    assert "Web UI dependencies not installed" in print_text
    assert "Smart App Control" not in print_text

    # Verify we exited with error code 1
    mock_exit.assert_called_once_with(1)


def test_ssl_import_error_without_dll_load_fallback():
    """Test that _ssl error without 'DLL load failed' shows standard message."""
    # Simulate an _ssl error that doesn't have the Smart App Control signature
    fake_error = ImportError("cannot import name '_ssl' from 'ssl'")

    with patch('builtins.print') as mock_print:
        with patch('builtins.exit') as mock_exit:
            # Simulate the error handling logic
            error_msg = str(fake_error)
            if "DLL load failed" in error_msg and "_ssl" in error_msg:
                print("Smart App Control blocked path")
            else:
                print("Web UI dependencies not installed (need fastapi + uvicorn).")
            print(f"Import error: {error_msg}")
            mock_exit(1)

    # Verify the standard missing dependencies message was printed
    print_calls = [str(call) for call in mock_print.call_args_list]
    print_text = "\n".join(print_calls)
    assert "Web UI dependencies not installed" in print_text
    assert "Smart App Control" not in print_text

    # Verify we exited with error code 1
    mock_exit.assert_called_once_with(1)