#!/usr/bin/env python3
"""
Rate Limited Executor for Cron Jobs
Wraps API calls in a rate-limited executor that reads headers from stderr/stdout.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from typing import Dict, Optional

# Import the adaptive rate limiter - we'll need to adjust the path
sys.path.insert(0, '/home/kensei/.hermes/profiles/wesker/mcp/adaptive_rate_limiter')
from server import AdaptiveRateLimiter

class RateLimitedExecutor:
    def __init__(self, key: str, rpm: float):
        self.key = key
        self.rpm = rpm
        self.limiter = AdaptiveRateLimiter()
        # Override the default RPM for this key
        # We'll need to modify the limiter's state directly for simplicity
        # In a real implementation, we'd pass this to await_slot
    
    async def execute_command(self, cmd: list) -> int:
        """Execute a command with rate limiting."""
        # Wait for slot
        await self.limiter.await_slot(self.key, self.rpm)
        
        # Execute command and capture output
        try:
            # Run command and capture stderr for headers
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Try to parse headers from stderr (if command outputs JSON lines with headers)
            status = process.returncode if process.returncode != 0 else 200  # Assume 200 if success
            headers = {}
            
            # Look for JSON header info in stderr
            try:
                stderr_lines = stderr.decode('utf-8').strip().split('\n')
                for line in stderr_lines:
                    if line.startswith('{') and line.endswith('}'):
                        data = json.loads(line)
                        if 'status' in data:
                            status = data['status']
                        if 'headers' in data:
                            headers = data['headers']
                        break
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # No JSON headers found
            
            # Record the response
            await self.limiter.record_response_async(self.key, status, headers)
            
            # Output stdout/stderr
            if stdout:
                sys.stdout.buffer.write(stdout)
            if stderr:
                sys.stderr.buffer.write(stderr)
            
            return process.returncode
            
        except Exception as e:
            print(f"Error executing command: {e}", file=sys.stderr)
            return 1

async def main():
    parser = argparse.ArgumentParser(description='Rate limited executor for cron jobs')
    parser.add_argument('--key', required=True, help='Rate limit key')
    parser.add_argument('--rpm', type=float, required=True, help='Requests per minute')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Command to execute')
    
    if not parser.parse_args().command:
        parser.error('No command provided')
    
    args = parser.parse_args()
    
    executor = RateLimitedExecutor(args.key, args.rpm)
    exit_code = await executor.execute_command(args.command)
    sys.exit(exit_code)

if __name__ == '__main__':
    asyncio.run(main())