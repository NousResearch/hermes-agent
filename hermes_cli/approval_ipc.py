"""
File-based IPC for blocking approval waits.

This module provides synchronous waiting for user approvals in scenarios
where gateway callbacks are not available (cron, batch jobs, etc.).

Mechanism:
1. When approval is needed, write request to ~/.hermes/approvals/{session_key}.json
2. Block and poll for response file ~/.hermes/approvals/{session_key}.response
3. User responds via 'hermes approve' CLI command
4. Read response and unblock

Usage:
    from hermes_cli.approval_ipc import wait_for_approval_blocking
    
    choice = wait_for_approval_blocking(
        session_key="abc123",
        command="rm -rf /tmp/test",
        description="recursive delete",
        timeout=300
    )
    # choice is 'once', 'session', 'always', 'deny', or None (timeout)
"""

import json
import os
import time
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Approval files directory
APPROVALS_DIR = Path.home() / ".hermes" / "approvals"


def _ensure_approvals_dir():
    """Ensure the approvals directory exists."""
    APPROVALS_DIR.mkdir(parents=True, exist_ok=True)
    # Set secure permissions
    try:
        os.chmod(APPROVALS_DIR, 0o700)
    except (OSError, NotImplementedError):
        pass


def _get_request_path(session_key: str) -> Path:
    """Get path to approval request file."""
    return APPROVALS_DIR / f"{session_key}.json"


def _get_response_path(session_key: str) -> Path:
    """Get path to approval response file."""
    return APPROVALS_DIR / f"{session_key}.response"


def write_approval_request(session_key: str, command: str, description: str, 
                           pattern_keys: list = None):
    """Write an approval request to disk.
    
    Args:
        session_key: Unique session identifier
        command: The command requiring approval
        description: Human-readable description of the risk
        pattern_keys: List of pattern keys that were matched
    """
    _ensure_approvals_dir()
    
    request_data = {
        "session_key": session_key,
        "command": command,
        "description": description,
        "pattern_keys": pattern_keys or [],
        "timestamp": time.time(),
        "status": "pending"
    }
    
    request_path = _get_request_path(session_key)
    try:
        with open(request_path, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, indent=2)
        os.chmod(request_path, 0o600)
        logger.debug("Wrote approval request: %s", request_path)
    except Exception as e:
        logger.error("Failed to write approval request: %s", e)
        raise


def read_approval_response(session_key: str) -> Optional[dict]:
    """Read approval response from disk if available.
    
    Returns:
        Response dict with 'choice' key, or None if no response yet
    """
    response_path = _get_response_path(session_key)
    
    if not response_path.exists():
        return None
    
    try:
        with open(response_path, 'r', encoding='utf-8') as f:
            response = json.load(f)
        
        # Verify session key matches
        if response.get("session_key") != session_key:
            logger.warning("Response session key mismatch")
            return None
        
        return response
    except Exception as e:
        logger.warning("Failed to read approval response: %s", e)
        return None


def write_approval_response(session_key: str, choice: str):
    """Write user's approval decision to disk.
    
    Args:
        session_key: Session identifier
        choice: User's choice ('once', 'session', 'always', 'deny')
    """
    _ensure_approvals_dir()
    
    response_data = {
        "session_key": session_key,
        "choice": choice,
        "timestamp": time.time()
    }
    
    response_path = _get_response_path(session_key)
    try:
        with open(response_path, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2)
        os.chmod(response_path, 0o600)
        logger.debug("Wrote approval response: %s", response_path)
    except Exception as e:
        logger.error("Failed to write approval response: %s", e)
        raise


def cleanup_approval_files(session_key: str):
    """Remove approval request and response files for a session."""
    try:
        request_path = _get_request_path(session_key)
        if request_path.exists():
            request_path.unlink()
        
        response_path = _get_response_path(session_key)
        if response_path.exists():
            response_path.unlink()
        
        logger.debug("Cleaned up approval files for session: %s", session_key)
    except Exception as e:
        logger.warning("Failed to cleanup approval files: %s", e)


def wait_for_approval_blocking(session_key: str, command: str, 
                                description: str, timeout: int = 300,
                                poll_interval: float = 1.0) -> Optional[str]:
    """Wait synchronously for user approval response.
    
    This function blocks until:
    1. User provides approval response
    2. Timeout is reached
    
    Args:
        session_key: Unique session identifier
        command: Command requiring approval
        description: Human-readable risk description
        timeout: Maximum seconds to wait (default: 300 = 5 minutes)
        poll_interval: Seconds between polling for response (default: 1.0)
    
    Returns:
        User's choice: 'once', 'session', 'always', 'deny'
        None if timeout occurred
    
    Raises:
        TimeoutError: If timeout is reached (optional, based on config)
    """
    logger.info("Waiting for approval (timeout: %ds): %s", timeout, command[:50])
    
    # Write request to disk
    write_approval_request(session_key, command, description)
    
    start_time = time.time()
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            # Check timeout
            if elapsed >= timeout:
                logger.warning("Approval timeout after %ds", elapsed)
                cleanup_approval_files(session_key)
                return None  # Timeout
            
            # Check for response
            response = read_approval_response(session_key)
            if response:
                choice = response.get("choice")
                logger.info("Approval response received: %s (after %.1fs)", 
                          choice, elapsed)
                cleanup_approval_files(session_key)
                return choice
            
            # Wait before polling again
            time.sleep(poll_interval)
    
    except KeyboardInterrupt:
        logger.info("Approval wait interrupted by user")
        cleanup_approval_files(session_key)
        return None


def list_pending_approvals() -> list:
    """List all pending approval requests.
    
    Returns:
        List of pending approval request dicts
    """
    pending = []
    
    if not APPROVALS_DIR.exists():
        return pending
    
    try:
        for file_path in APPROVALS_DIR.glob("*.json"):
            if file_path.suffix == ".json" and not file_path.name.endswith(".response"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        request = json.load(f)
                    
                    # Check if response exists
                    session_key = request.get("session_key", "")
                    response_path = _get_response_path(session_key)
                    
                    if not response_path.exists():
                        pending.append(request)
                except Exception as e:
                    logger.warning("Failed to read approval request %s: %s", 
                                 file_path, e)
    except Exception as e:
        logger.error("Failed to list pending approvals: %s", e)
    
    return pending


# CLI command for responding to approvals
def approve_command(session_key: str, choice: str) -> bool:
    """Respond to a pending approval request.
    
    Args:
        session_key: Session identifier
        choice: User's choice ('once', 'session', 'always', 'deny')
    
    Returns:
        True if response was written successfully
    """
    # Verify request exists
    request_path = _get_request_path(session_key)
    if not request_path.exists():
        logger.error("No pending approval for session: %s", session_key)
        return False
    
    # Write response
    try:
        write_approval_response(session_key, choice)
        return True
    except Exception as e:
        logger.error("Failed to write approval response: %s", e)
        return False
