"""Quick smoke test for the validate-config CLI command."""
import sys
sys.path.insert(0, '/home/kensei/.hermes/hermes-agent')
import argparse
from hermes_cli.validate_config import run_validate_config

args = argparse.Namespace()
result = run_validate_config(args)
print(f'Exit code: {result}')
