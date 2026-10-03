import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add scripts directory to path to import the wrapper script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

import macos_mdfind

class TestMacOSSpotlight(unittest.TestCase):
    
    @patch('macos_mdfind.platform.system')
    def test_platform_check_fails_on_non_mac(self, mock_system):
        """Test that the script exits with an error on non-macOS systems."""
        mock_system.return_value = "Linux"
        with self.assertRaises(SystemExit) as cm:
            macos_mdfind.main()
        self.assertEqual(cm.exception.code, 1)

    @patch('macos_mdfind.platform.system')
    @patch('macos_mdfind.subprocess.run')
    @patch('sys.argv', ['macos_mdfind.py', '--name', 'report', '--type', 'pdf'])
    @patch('sys.stdout', new_callable=MagicMock)
    def test_mdfind_command_construction(self, mock_stdout, mock_run, mock_system):
        """Test if the mdfind command is constructed securely and correctly."""
        mock_system.return_value = "Darwin"
        mock_process = MagicMock()
        mock_process.stdout = "/path/to/report.pdf\n"
        mock_run.return_value = mock_process

        macos_mdfind.main()

        # Verify subprocess.run was called once
        mock_run.assert_called_once()
        
        # Verify the command list passed to subprocess
        called_cmd = mock_run.call_args[0][0]
        self.assertEqual(called_cmd[0], 'mdfind')
        
        # Check if the query logic is properly combined with '&&'
        query_string = called_cmd[1]
        self.assertIn('kMDItemFSName == "*report*"cd', query_string)
        self.assertIn('kMDItemFSName == "*.pdf"', query_string)
        self.assertIn('&&', query_string)

    @patch('macos_mdfind.platform.system')
    @patch('macos_mdfind.subprocess.run')
    @patch('sys.argv', ['macos_mdfind.py', '--content', 'budget', '--limit', '2'])
    @patch('sys.stdout', new_callable=MagicMock)
    def test_mdfind_limit_truncation(self, mock_stdout, mock_run, mock_system):
        """Test if the output is properly truncated to prevent token overflow."""
        mock_system.return_value = "Darwin"
        mock_process = MagicMock()
        # Simulate mdfind returning 3 results
        mock_process.stdout = "/path/1.txt\n/path/2.txt\n/path/3.txt\n"
        mock_run.return_value = mock_process

        macos_mdfind.main()

        # Extract all prints
        prints = [call.args[0] for call in mock_stdout.write.call_args_list if call.args[0] != '\n']
        
        # Output should contain exactly 2 file paths due to --limit 2, plus the truncation message
        self.assertIn('/path/1.txt', prints)
        self.assertIn('/path/2.txt', prints)
        self.assertNotIn('/path/3.txt', prints)
        
        # Check truncation warning
        truncation_msg = next((p for p in prints if "... and 1 more results" in p), None)
        self.assertIsNotNone(truncation_msg)

if __name__ == '__main__':
    unittest.main()
