"""Tests for tui_gateway shell.exec security hardening.

Verifies that the shell.exec RPC method properly gates dangerous commands
and fails closed when the approval system is unavailable.

See issue #36847: Security: tui_gateway shell.exec inherits the approval
denylist bypass → arbitrary command execution.
"""

import pytest
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class TestDetectDangerousCommandImport:
    """Test that dangerous pattern detection works correctly."""

    def test_detect_dangerous_command_imported(self):
        """Verify detect_dangerous_command is available from tools.approval."""
        from tools.approval import detect_dangerous_command
        assert callable(detect_dangerous_command)

    def test_dangerous_patterns_rm_variants(self):
        """Verify rm variants are properly detected."""
        from tools.approval import detect_dangerous_command
        
        # These should be detected as dangerous
        dangerous_commands = [
            "rm -rf /",
            "rm -r /home",
            "rm -rf /etc",
        ]
        
        for cmd in dangerous_commands:
            is_dangerous, _, desc = detect_dangerous_command(cmd)
            assert is_dangerous, f"Command '{cmd}' should be detected as dangerous"

    def test_safe_commands_pass(self):
        """Verify safe commands are not flagged as dangerous."""
        from tools.approval import detect_dangerous_command
        
        safe_commands = [
            "echo hello",
            "ls -la",
            "pwd",
            "cat /etc/hostname",
            "date",
        ]
        
        for cmd in safe_commands:
            is_dangerous, _, desc = detect_dangerous_command(cmd)
            assert not is_dangerous, f"Command '{cmd}' should be safe but was flagged: {desc}"

    def test_dangerous_patterns_chmod_variants(self):
        """Verify chmod world-writable patterns are detected."""
        from tools.approval import detect_dangerous_command
        
        dangerous_commands = [
            "chmod 777 /",
            "chmod 666 /home",
            "chmod o+w /tmp",
        ]
        
        for cmd in dangerous_commands:
            is_dangerous, _, desc = detect_dangerous_command(cmd)
            assert is_dangerous, f"Command '{cmd}' should be detected as dangerous"

    def test_dangerous_patterns_deletion_variants(self):
        """Verify SQL DELETE/DROP/TRUNCATE variants are detected."""
        from tools.approval import detect_dangerous_command
        
        dangerous_commands = [
            "DROP TABLE users;",
            "DELETE FROM users;",
            "TRUNCATE TABLE logs;",
        ]
        
        for cmd in dangerous_commands:
            is_dangerous, _, desc = detect_dangerous_command(cmd)
            assert is_dangerous, f"Command '{cmd}' should be detected as dangerous"

    def test_dangerous_patterns_return_format(self):
        """Verify detect_dangerous_command returns the correct format."""
        from tools.approval import detect_dangerous_command
        
        # Dangerous command should return (True, pattern_key, description)
        is_dangerous, pattern_key, desc = detect_dangerous_command("rm -rf /")
        assert is_dangerous is True
        assert pattern_key is not None
        assert desc is not None
        assert isinstance(desc, str)
        
        # Safe command should return (False, None, None)
        is_dangerous, pattern_key, desc = detect_dangerous_command("echo hello")
        assert is_dangerous is False
        assert pattern_key is None
        assert desc is None


class TestShellExecGateSecurity:
    """Test that shell.exec properly uses approval gate."""

    def test_approval_system_required_in_workflow(self):
        """Verify the approval system is part of the shell.exec call path."""
        # Read the server.py source to verify the fix is in place
        server_path = os.path.join(os.path.dirname(__file__), '../../tui_gateway/server.py')
        with open(server_path, 'r') as f:
            content = f.read()
        
        # Check that shell.exec method has detect_dangerous_command
        assert 'shell.exec' in content
        assert 'detect_dangerous_command' in content
        
        # Check that the ImportError is handled with fail-closed (not pass)
        # The fix changes: except ImportError: pass
        # To: except ImportError as e: return _err(...)
        shell_exec_start = content.find('@method("shell.exec")')
        assert shell_exec_start != -1
        
        shell_exec_section = content[shell_exec_start:shell_exec_start + 2000]
        
        # Verify the approval gate is called
        assert 'detect_dangerous_command(cmd)' in shell_exec_section
        
        # Verify fail-closed behavior (should have return _err for ImportError)
        assert 'except ImportError' in shell_exec_section
        
        # Verify the critical fix: ImportError should be followed by return _err, not pass
        import re
        # Look for the ImportError handler block
        importerror_match = re.search(
            r'except ImportError.*?:\n(.*?)(?=\n\s{0,4}(?:except|try|def|\Z))',
            shell_exec_section,
            re.DOTALL
        )
        assert importerror_match is not None, "Could not find ImportError handler"
        
        importerror_body = importerror_match.group(1)
        # Should have return _err and 5001 error code
        assert 'return _err' in importerror_body, \
            f"ImportError handler should return error, got: {importerror_body}"
        assert '5001' in importerror_body or 'unavailable' in importerror_body, \
            f"ImportError handler should indicate approval system unavailable: {importerror_body}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
