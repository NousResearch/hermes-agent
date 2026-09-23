"""
NOX V3 State Management
"""

from typing import Dict, Any
from pathlib import Path
import json
from datetime import datetime, date


# State file path
STATE_FILE = Path.home() / ".hermes" / "plugins" / "nox_v3" / "state.json"


def load_state() -> Dict[str, Any]:
    """
    Load NOX V3 state from file.

    Returns:
        State dictionary
    """
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)

            # Check if state is from today
            last_reset = state.get("last_reset")
            if last_reset:
                last_date = datetime.fromisoformat(last_reset).date()
                today = date.today()

                # Reset if from different day
                if last_date != today:
                    state = get_default_state()
                    state["last_reset"] = datetime.now().isoformat()

            return state
        except (json.JSONDecodeError, IOError, ValueError):
            # Return default state on error
            return get_default_state()

    return get_default_state()


def save_state(state: Dict[str, Any]) -> bool:
    """
    Save NOX V3 state to file.

    Args:
        state: State dictionary

    Returns:
        True if successful, False otherwise
    """
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        return True
    except IOError:
        return False


def get_default_state() -> Dict[str, Any]:
    """
    Get default NOX V3 state.

    Returns:
        Default state dictionary
    """
    return {
        "daily_token_usage": 0,
        "session_count": 0,
        "verification_count": 0,
        "verification_successes": 0,
        "verification_success_rate": 1.0,
        "last_reset": datetime.now().isoformat(),
    }


def reset_daily_usage() -> None:
    """Reset daily token usage and verification stats."""
    state = load_state()
    state["daily_token_usage"] = 0
    state["verification_count"] = 0
    state["verification_successes"] = 0
    state["verification_success_rate"] = 1.0
    state["last_reset"] = datetime.now().isoformat()
    save_state(state)


def record_token_usage(tokens: int) -> None:
    """
    Record token usage.

    Args:
        tokens: Number of tokens used
    """
    state = load_state()
    state["daily_token_usage"] += tokens
    save_state(state)


def increment_session_count() -> None:
    """Increment active session count."""
    state = load_state()
    state["session_count"] += 1
    save_state(state)


def decrement_session_count() -> None:
    """Decrement active session count."""
    state = load_state()
    state["session_count"] = max(0, state["session_count"] - 1)
    save_state(state)


def record_verification_result(success: bool) -> None:
    """
    Record verification result.

    Args:
        success: Whether verification succeeded
    """
    state = load_state()
    state["verification_count"] += 1
    if success:
        state["verification_successes"] += 1

    # Update success rate
    total = state["verification_count"]
    successes = state["verification_successes"]
    state["verification_success_rate"] = successes / total if total > 0 else 1.0

    save_state(state)


def get_daily_token_usage() -> int:
    """
    Get daily token usage.

    Returns:
        Daily token usage
    """
    state = load_state()
    return state.get("daily_token_usage", 0)


def get_session_count() -> int:
    """
    Get session count.

    Returns:
        Session count
    """
    state = load_state()
    return state.get("session_count", 0)


def get_verification_success_rate() -> float:
    """
    Get verification success rate.

    Returns:
        Verification success rate (0.0 to 1.0)
    """
    state = load_state()
    return state.get("verification_success_rate", 1.0)
