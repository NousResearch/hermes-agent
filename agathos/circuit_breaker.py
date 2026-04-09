"""Agathos Circuit Breaker — automatic provider disable on failure.

Integrates with Hermes config.yaml to dynamically manage provider_routing.
Uses Agathos provider_health for failure detection.

States:
- CLOSED: Normal operation (provider active)
- OPEN: Circuit tripped (provider disabled via provider_routing.ignore)
- HALF_OPEN: Testing provider after recovery timeout

Configuration (agathos.circuit_breaker in config.yaml):
    agathos:
      circuit_breaker:
        enabled: true
        failure_threshold: 3              # Consecutive failures to open
        error_rate_threshold: 0.50        # 50% error rate over window
        min_requests_for_rate: 5          # Min requests before rate check
        recovery_timeout: 300             # Seconds before retry (half-open)
        half_open_requests: 2             # Successful requests to close
        notify_on_open: true              # Alert when circuit opens
        notify_on_close: true             # Alert when circuit closes

Uses Hermes classes/functions:
- hermes_cli.config.load_config() — read current provider_routing
- hermes_cli.config.save_config() — write updated provider_routing
- provider_health module — get provider error rates
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto

logger = logging.getLogger("agathos.circuit_breaker")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Provider disabled
    HALF_OPEN = auto()   # Testing after timeout


# Default configuration
_DEFAULT_CIRCUIT_BREAKER_CONFIG = {
    "enabled": False,              # Disabled by default - opt-in
    "failure_threshold": 3,        # Consecutive failures
    "error_rate_threshold": 0.50,  # 50% error rate
    "min_requests_for_rate": 5,    # Min requests for rate calc
    "recovery_timeout": 300,       # 5 minutes
    "half_open_requests": 2,       # Successes to close
    "notify_on_open": True,
    "notify_on_close": True,
}


class CircuitBreaker:
    """Circuit breaker for provider failover.

    Manages provider state via Hermes config.yaml provider_routing.
    Tracks circuit state in agathos.db for persistence.

    The circuit breaker monitors provider health metrics (consecutive failures,
    error rates) and automatically disables failing providers by adding them
    to provider_routing.ignore in config.yaml.

    State transitions:
    - CLOSED -> OPEN: When failure_threshold consecutive errors or error_rate_threshold exceeded
    - OPEN -> HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN -> CLOSED: After half_open_requests successful requests
    - HALF_OPEN -> OPEN: On any failure during testing

    Attributes:
        cursor: Database cursor for agathos.db
        conn: Database connection for commits
        config: Circuit breaker configuration dict (from config.yaml)
        _hermes_available: Cached boolean for Hermes API availability

    Side effects:
        - Reads/writes provider_routing in ~/.hermes/config.yaml
        - Creates circuit_breaker_states table in agathos.db
        - Sends notifications when circuits open/close (if configured)
    """
    
    def __init__(self, cursor: sqlite3.Cursor, conn: sqlite3.Connection, config: Optional[Dict] = None):
        self.cursor = cursor
        self.conn = conn
        self.config = config or {}
        self._hermes_available: Optional[bool] = None
        self._ensure_table()
    
    def _check_hermes_available(self) -> bool:
        """Check if we can import Hermes config functions."""
        if self._hermes_available is not None:
            return self._hermes_available
        try:
            from hermes_cli.config import load_config, save_config
            self._hermes_available = True
            return True
        except ImportError:
            logger.warning("Hermes config unavailable - monitoring only")
            self._hermes_available = False
            return False
    
    def _ensure_table(self):
        """Create circuit_breaker table if not exists."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS circuit_breaker (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                state TEXT NOT NULL,
                opened_at TIMESTAMP,
                opened_reason TEXT,
                last_tested_at TIMESTAMP,
                half_open_successes INTEGER DEFAULT 0,
                consecutive_failures INTEGER DEFAULT 0,
                total_requests INTEGER DEFAULT 0,
                total_failures INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider)
            )
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_circuit_breaker_provider
            ON circuit_breaker(provider)
        """)
        self.conn.commit()
    
    def get_circuit_config(self) -> Dict[str, Any]:
        """Get circuit breaker configuration."""
        cb_config = self.config.get("circuit_breaker", {})
        return {
            "enabled": cb_config.get("enabled", _DEFAULT_CIRCUIT_BREAKER_CONFIG["enabled"]),
            "failure_threshold": cb_config.get("failure_threshold", _DEFAULT_CIRCUIT_BREAKER_CONFIG["failure_threshold"]),
            "error_rate_threshold": cb_config.get("error_rate_threshold", _DEFAULT_CIRCUIT_BREAKER_CONFIG["error_rate_threshold"]),
            "min_requests_for_rate": cb_config.get("min_requests_for_rate", _DEFAULT_CIRCUIT_BREAKER_CONFIG["min_requests_for_rate"]),
            "recovery_timeout": cb_config.get("recovery_timeout", _DEFAULT_CIRCUIT_BREAKER_CONFIG["recovery_timeout"]),
            "half_open_requests": cb_config.get("half_open_requests", _DEFAULT_CIRCUIT_BREAKER_CONFIG["half_open_requests"]),
            "notify_on_open": cb_config.get("notify_on_open", _DEFAULT_CIRCUIT_BREAKER_CONFIG["notify_on_open"]),
            "notify_on_close": cb_config.get("notify_on_close", _DEFAULT_CIRCUIT_BREAKER_CONFIG["notify_on_close"]),
        }
    
    def _get_provider_routing_from_hermes(self) -> Dict[str, Any]:
        if not self._check_hermes_available():
            return {}
        try:
            from hermes_cli.config import load_config
            hermes_config = load_config()
            return hermes_config.get("provider_routing", {})
        except Exception as e:
            logger.error("Failed to load Hermes config: %s", e)
            return {}
    
    def _save_provider_routing_to_hermes(self, routing: Dict[str, Any]) -> bool:
        if not self._check_hermes_available():
            return False
        try:
            from hermes_cli.config import load_config, save_config
            hermes_config = load_config()
            hermes_config["provider_routing"] = routing
            save_config(hermes_config)
            return True
        except Exception as e:
            logger.error("Failed to save Hermes config: %s", e)
            return False
    
    def _is_provider_ignored(self, provider: str, routing: Dict[str, Any]) -> bool:
        """Check if provider is in provider_routing.ignore list."""
        ignore_list = routing.get("ignore", [])
        return provider in ignore_list
    
    def _add_provider_to_ignore(self, provider: str, routing: Dict[str, Any]) -> Dict[str, Any]:
        """Add provider to ignore list."""
        ignore_list = routing.get("ignore", [])
        if provider not in ignore_list:
            ignore_list.append(provider)
            routing["ignore"] = ignore_list
        return routing
    
    def _remove_provider_from_ignore(self, provider: str, routing: Dict[str, Any]) -> Dict[str, Any]:
        """Remove provider from ignore list."""
        ignore_list = routing.get("ignore", [])
        if provider in ignore_list:
            ignore_list.remove(provider)
            routing["ignore"] = ignore_list
        return routing
    
    def get_circuit_state(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get circuit state for a provider."""
        self.cursor.execute("""
            SELECT * FROM circuit_breaker WHERE provider = ?
        """, (provider,))
        row = self.cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def set_circuit_state(self, provider: str, state: CircuitState, reason: str = ""):
        """Update circuit state in database."""
        state_str = state.name
        now = datetime.now().isoformat()
        
        if state == CircuitState.OPEN:
            self.cursor.execute("""
                INSERT INTO circuit_breaker 
                (provider, state, opened_at, opened_reason, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    state = excluded.state,
                    opened_at = excluded.opened_at,
                    opened_reason = excluded.opened_reason,
                    half_open_successes = 0,
                    updated_at = excluded.updated_at
            """, (provider, state_str, now, reason, now))
        elif state == CircuitState.HALF_OPEN:
            self.cursor.execute("""
                INSERT INTO circuit_breaker 
                (provider, state, last_tested_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    state = excluded.state,
                    last_tested_at = excluded.last_tested_at,
                    updated_at = excluded.updated_at
            """, (provider, state_str, now, now))
        else:  # CLOSED
            self.cursor.execute("""
                INSERT INTO circuit_breaker 
                (provider, state, updated_at, consecutive_failures, total_requests, total_failures)
                VALUES (?, ?, ?, 0, 0, 0)
                ON CONFLICT(provider) DO UPDATE SET
                    state = excluded.state,
                    consecutive_failures = 0,
                    updated_at = excluded.updated_at
            """, (provider, state_str, now))
        
        self.conn.commit()
    
    def update_failure_stats(self, provider: str, failed: bool):
        """Update failure statistics for a provider."""
        if failed:
            self.cursor.execute("""
                INSERT INTO circuit_breaker (provider, consecutive_failures, total_requests, total_failures)
                VALUES (?, 1, 1, 1)
                ON CONFLICT(provider) DO UPDATE SET
                    consecutive_failures = consecutive_failures + 1,
                    total_requests = total_requests + 1,
                    total_failures = total_failures + 1,
                    updated_at = CURRENT_TIMESTAMP
            """, (provider,))
        else:
            self.cursor.execute("""
                INSERT INTO circuit_breaker (provider, consecutive_failures, total_requests, total_failures, half_open_successes)
                VALUES (?, 0, 1, 0, 0)
                ON CONFLICT(provider) DO UPDATE SET
                    consecutive_failures = 0,
                    total_requests = total_requests + 1,
                    half_open_successes = CASE 
                        WHEN state = 'HALF_OPEN' THEN half_open_successes + 1 
                        ELSE 0 
                    END,
                    updated_at = CURRENT_TIMESTAMP
            """, (provider,))
        self.conn.commit()
    
    def check_and_transition_circuits(self) -> List[Dict[str, Any]]:
        """Check all providers and transition circuits as needed.
        
        Returns list of state transition events for notifications.
        """
        cb_config = self.get_circuit_config()
        if not cb_config["enabled"]:
            return []
        
        events = []
        
        # Get providers from provider_health data
        try:
            from . import provider_health as _provider_health
            health_data = _provider_health.run_provider_check(self.cursor, self.conn)
        except Exception as e:
            logger.error("Failed to get provider health: %s", e)
            return []
        
        routing = self._get_provider_routing_from_hermes()
        
        for provider, stats in health_data.get("providers", {}).items():
            # Update failure stats from health data
            error_rate = stats.get("error_rate", 0)
            total = stats.get("total", 0)
            failures = int(total * error_rate)
            
            # Get current circuit state
            circuit = self.get_circuit_state(provider)
            current_state = CircuitState[circuit["state"]] if circuit else CircuitState.CLOSED
            
            # State machine logic
            if current_state == CircuitState.CLOSED:
                # Check if should open
                should_open = False
                reason = ""
                
                # Check consecutive failures (from our tracking)
                consecutive = circuit.get("consecutive_failures", 0) if circuit else 0
                if consecutive >= cb_config["failure_threshold"]:
                    should_open = True
                    reason = f"{consecutive} consecutive failures"
                
                # Check error rate
                if (total >= cb_config["min_requests_for_rate"] and 
                    error_rate >= cb_config["error_rate_threshold"]):
                    should_open = True
                    reason = f"{error_rate:.1%} error rate over {total} requests"
                
                if should_open:
                    # Open the circuit
                    self.set_circuit_state(provider, CircuitState.OPEN, reason)
                    self._add_provider_to_ignore(provider, routing)
                    if self._save_provider_routing_to_hermes(routing):
                        logger.warning("Circuit OPEN for %s: %s", provider, reason)
                        events.append({
                            "provider": provider,
                            "transition": "CLOSED→OPEN",
                            "reason": reason,
                            "notify": cb_config["notify_on_open"],
                        })
                    else:
                        logger.error("Failed to update Hermes config for %s", provider)
            
            elif current_state == CircuitState.OPEN:
                # Check if recovery timeout passed
                if circuit:
                    opened_at = circuit.get("opened_at")
                    if opened_at:
                        opened_time = datetime.fromisoformat(opened_at)
                        elapsed = (datetime.now() - opened_time).total_seconds()
                        
                        if elapsed >= cb_config["recovery_timeout"]:
                            # Transition to half-open
                            self.set_circuit_state(provider, CircuitState.HALF_OPEN)
                            self._remove_provider_from_ignore(provider, routing)
                            if self._save_provider_routing_to_hermes(routing):
                                logger.info("Circuit HALF_OPEN for %s (after %.0fs)", provider, elapsed)
                                # No notification for half-open (testing phase)
            
            elif current_state == CircuitState.HALF_OPEN:
                # Check if enough successes to close
                half_open_successes = circuit.get("half_open_successes", 0) if circuit else 0
                
                if half_open_successes >= cb_config["half_open_requests"]:
                    # Close the circuit
                    self.set_circuit_state(provider, CircuitState.CLOSED)
                    logger.info("Circuit CLOSED for %s (%d successes)", provider, half_open_successes)
                    events.append({
                        "provider": provider,
                        "transition": "HALF_OPEN→CLOSED",
                        "reason": f"{half_open_successes} consecutive successes",
                        "notify": cb_config["notify_on_close"],
                    })
                elif error_rate > 0.5 and total > 0:
                    # Re-open if still failing
                    reason = f"Still failing in half-open ({error_rate:.1%} error rate)"
                    self.set_circuit_state(provider, CircuitState.OPEN, reason)
                    self._add_provider_to_ignore(provider, routing)
                    if self._save_provider_routing_to_hermes(routing):
                        logger.warning("Circuit re-OPENED for %s: %s", provider, reason)
        
        return events
    
    def record_provider_result(self, provider: str, success: bool, error_type: Optional[str] = None):
        """Record a provider request result for circuit tracking.
        
        Called by the adapter layer or message processing to update stats.
        """
        self.update_failure_stats(provider, not success)
        
        # Also update provider_health for consistency
        try:
            from . import provider_health as _provider_health
            if success:
                _provider_health.record_provider_success(self.cursor, self.conn, provider)
            else:
                _provider_health.record_provider_error(
                    self.cursor, self.conn, provider, 
                    error_type or "unknown", "Circuit breaker tracked"
                )
        except Exception as e:
            logger.debug("Failed to record provider_health update: %s", e)
    
    def get_all_circuits(self) -> List[Dict[str, Any]]:
        """Get status of all circuits."""
        self.cursor.execute("SELECT * FROM circuit_breaker ORDER BY provider")
        return [dict(row) for row in self.cursor.fetchall()]
    
    def manual_reset(self, provider: str) -> bool:
        """Manually reset a provider's circuit to CLOSED."""
        routing = self._get_provider_routing_from_hermes()
        self._remove_provider_from_ignore(provider, routing)
        
        if self._save_provider_routing_to_hermes(routing):
            self.set_circuit_state(provider, CircuitState.CLOSED)
            logger.info("Manually reset circuit for %s to CLOSED", provider)
            return True
        return False


def check_circuits(cursor: sqlite3.Cursor, conn: sqlite3.Connection, config: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Run circuit breaker checks for all providers.

    Convenience function that instantiates CircuitBreaker and runs the
    full check_and_transition_circuits() cycle. Used by Agathos daemon
    during periodic health checks.

    Args:
        cursor: Database cursor for agathos.db
        conn: Database connection for commits
        config: Agathos config dict with circuit_breaker section

    Returns:
        List of state transition event dicts, each with:
        - provider: Provider name
        - transition: State change (e.g., "CLOSED -> OPEN")
        - reason: Human-readable explanation
        - timestamp: Unix timestamp

    Side effects:
        - May update provider_routing.ignore in config.yaml
        - Inserts/updates circuit_breaker_states table
        - Commits database changes

    Connection management: Does NOT close conn (caller manages lifecycle).
    """
    breaker = CircuitBreaker(cursor, conn, config)
    try:
        return breaker.check_and_transition_circuits()
    finally:
        pass  # Don't close conn here (managed by caller)


def format_circuit_event(event: Dict[str, Any]) -> str:
    """Format a circuit state transition event for notification.

    Converts a circuit event dict into a human-readable message suitable
    for Telegram, Discord, Slack, or other notification channels.

    Args:
        event: Event dict from check_circuits with keys:
            - provider: Provider name
            - transition: State change string (e.g., "CLOSED -> OPEN")
            - reason: Explanation of why transition occurred

    Returns:
        Formatted notification string with emoji indicator:
        - "🔴" for OPEN (problem)
        - "🟢" for CLOSE/ recovery

    Example:
        "🔴 Circuit CLOSED -> OPEN for openrouter: 5 consecutive failures"
    """
    provider = event["provider"]
    transition = event["transition"]
    reason = event["reason"]

    emoji = "🔴" if "OPEN" in transition else "🟢"
    return f"{emoji} Circuit {transition} for {provider}: {reason}"
