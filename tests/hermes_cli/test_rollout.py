import argparse
from unittest.mock import patch, MagicMock
from hermes_cli.rollout import cmd_rollout

@patch("hermes_cli.rollout.concurrent.futures.ThreadPoolExecutor")
@patch("hermes_cli.rollout.load_config")
@patch("hermes_cli.rollout.Path")
@patch("hermes_cli.rollout.tempfile.mkdtemp")
@patch("hermes_cli.rollout.open")
def test_cmd_rollout(mock_open, mock_mkdtemp, mock_path, mock_load_config, mock_tpe):
    mock_load_config.return_value = {"model": {"default": "test-model", "provider": "test-provider"}}
    mock_mkdtemp.return_value = "/tmp/fake_dir"
    
    # Setup mock executor
    mock_executor = MagicMock()
    mock_tpe.return_value.__enter__.return_value = mock_executor
    
    # Create fake future
    mock_future = MagicMock()
    mock_future.result.return_value = (0, [{"from": "assistant", "value": "test"}], 1.0)
    
    # We patch concurrent.futures.as_completed inside the module to return our mock future
    with patch("hermes_cli.rollout.concurrent.futures.as_completed", return_value=[mock_future]):
        args = argparse.Namespace(
            prompt="Hello",
            verifier="echo success",
            G=1,
            temperature=0.7,
            output="/tmp/fake_output.jsonl"
        )
        cmd_rollout(args)
        
    mock_open.assert_called_with("/tmp/fake_output.jsonl", "a", encoding="utf-8")
