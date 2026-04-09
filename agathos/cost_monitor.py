"""Agathos cost monitoring — integration with Hermes insights for budget alerts.

Discovers providers dynamically from session history and user profile.
No hardcoded providers — all configuration via config.yaml.

Uses Hermes insights engine for data, doesn't duplicate cost tracking.

Configuration (agathos.cost_monitoring in config.yaml):
    agathos:
      cost_monitoring:
        enabled: false           # Disabled by default - opt-in
        daily_budget: 20.00      # USD daily limit
        alert_at_percent: 80     # Alert when budget % reached
        expensive_session_threshold: 2.00  # Flag sessions above this
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Any
from pathlib import Path

logger = logging.getLogger("agathos.cost_monitor")


# Default budget settings (user overrides in config.yaml)
# Note: cost_monitoring is disabled by default in config - user must explicitly enable
_DEFAULT_DAILY_BUDGET = 20.0  # USD - default if user enables cost_monitoring
_DEFAULT_ALERT_PERCENT = 80  # Alert at 80% of budget
_DEFAULT_EXPENSIVE_SESSION_THRESHOLD = 2.0  # USD


class CostMonitor:
    """Monitor costs using Hermes insights data.

    Discovers providers dynamically from session history rather than using
    hardcoded lists. Integrates with Hermes InsightsEngine for cost data,
    avoiding duplicate cost tracking logic.

    Key capabilities:
    - Provider discovery from actual usage history
    - Budget threshold monitoring (daily limits)
    - Expensive session detection
    - Budget percentage alerts

    Attributes:
        config: Cost monitoring configuration dict from config.yaml
        _insights_engine: Cached InsightsEngine instance (lazy-loaded)
        _db: Cached SessionDB instance (lazy-loaded)

    Configuration keys read:
    - enabled: Boolean to enable/disable monitoring
    - daily_budget: Float USD amount for daily spending limit
    - alert_at_percent: Int percentage to trigger alerts (default: 80)
    - expensive_session_threshold: Float USD amount for expensive session flag

    Side effects:
        - Queries Hermes insights engine for cost data
        - May read SessionDB for provider discovery
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._insights_engine = None
        self._db = None
        
    def _get_insights_engine(self):
        """Lazy-load insights engine from Hermes."""
        if self._insights_engine is not None:
            return self._insights_engine
            
        try:
            from agent.insights import InsightsEngine
            from hermes_state import SessionDB
            
            self._db = SessionDB()
            self._insights_engine = InsightsEngine(self._db)
            return self._insights_engine
        except ImportError as e:
            logger.warning("Hermes insights unavailable: %s", e)
            return None
    
    def discover_providers(self, days: int = 7) -> Set[str]:
        """Dynamically discover providers from recent session history.
        
        Returns set of billing_provider values used in last N days.
        Respects user's actual usage patterns.
        """
        engine = self._get_insights_engine()
        if not engine:
            return set()
        
        try:
            report = engine.generate(days=days)
            if report.get("empty"):
                return set()
            
            # Extract unique providers from platform breakdown
            providers = set()
            for platform in report.get("platforms", []):
                provider = platform.get("name")
                if provider:
                    providers.add(provider)
            
            return providers
        except Exception as e:
            logger.error("Failed to discover providers: %s", e)
            return set()
    
    def get_budget_config(self) -> Dict[str, Any]:
        """Load budget configuration from Agathos config.
        
        User configures in config.yaml under argus.cost_monitoring:
        
        agathos:
          cost_monitoring:
            enabled: true
            daily_budget: 10.00
            alert_at_percent: 80
            expensive_session_threshold: 2.00
            per_provider_limits:  # Optional, discovered providers use daily_budget/n if not set
              anthropic: 5.00
              fireworks: 3.00
        """
        cm_config = self.config.get("cost_monitoring", {})
        
        # Default is disabled - user must explicitly enable in config.yaml
        return {
            "enabled": cm_config.get("enabled", False),
            "daily_budget": cm_config.get("daily_budget", _DEFAULT_DAILY_BUDGET),
            "alert_at_percent": cm_config.get("alert_at_percent", _DEFAULT_ALERT_PERCENT),
            "expensive_session_threshold": cm_config.get(
                "expensive_session_threshold", _DEFAULT_EXPENSIVE_SESSION_THRESHOLD
            ),
            "per_provider_limits": cm_config.get("per_provider_limits", {}),
        }
    
    def check_daily_budget(self) -> Dict[str, Any]:
        """Check current spend vs daily budget.
        
        Returns status dict with alerts if thresholds exceeded.
        """
        budget_config = self.get_budget_config()
        if not budget_config["enabled"]:
            return {"enabled": False}
        
        engine = self._get_insights_engine()
        if not engine:
            return {"error": "Insights engine unavailable"}
        
        try:
            # Get last 24 hours of usage
            report = engine.generate(days=1)
            if report.get("empty"):
                return {
                    "enabled": True,
                    "spent": 0.0,
                    "budget": budget_config["daily_budget"],
                    "remaining": budget_config["daily_budget"],
                    "percent_used": 0.0,
                    "alert": False,
                }
            
            overview = report.get("overview", {})
            spent = float(overview.get("estimated_cost", 0))
            budget = budget_config["daily_budget"]
            remaining = budget - spent
            percent_used = (spent / budget * 100) if budget > 0 else 0
            
            alert_threshold = budget_config["alert_at_percent"]
            alert = percent_used >= alert_threshold
            
            result = {
                "enabled": True,
                "spent": round(spent, 4),
                "budget": budget,
                "remaining": round(remaining, 4),
                "percent_used": round(percent_used, 2),
                "alert": alert,
                "alert_threshold": alert_threshold,
            }
            
            if alert:
                logger.warning(
                    "Daily budget alert: $%.2f / $%.2f (%.1f%%)",
                    spent, budget, percent_used
                )
            
            return result
            
        except Exception as e:
            logger.error("Failed to check daily budget: %s", e)
            return {"error": str(e)}
    
    def check_provider_limits(self) -> List[Dict[str, Any]]:
        """Check per-provider spend vs configured limits.
        
        Discovers providers from history, applies limits from config.
        """
        budget_config = self.get_budget_config()
        if not budget_config["enabled"]:
            return []
        
        engine = self._get_insights_engine()
        if not engine:
            return []
        
        provider_limits = budget_config.get("per_provider_limits", {})
        daily_budget = budget_config["daily_budget"]
        
        # Discover providers from recent usage
        providers = self.discover_providers(days=1)
        
        # If no explicit limits, distribute budget equally among active providers
        if not provider_limits and providers:
            per_provider = daily_budget / len(providers)
            provider_limits = {p: per_provider for p in providers}
        
        alerts = []
        
        try:
            report = engine.generate(days=1)
            if report.get("empty"):
                return []
            
            # Check each provider's spend
            for platform in report.get("platforms", []):
                provider_name = platform.get("name")
                if not provider_name:
                    continue
                
                # Get limit for this provider
                limit = provider_limits.get(provider_name)
                if not limit:
                    continue  # No limit configured for this provider
                
                spent = float(platform.get("estimated_cost", 0))
                percent_used = (spent / limit * 100) if limit > 0 else 0
                
                if percent_used >= budget_config["alert_at_percent"]:
                    alert = {
                        "provider": provider_name,
                        "spent": round(spent, 4),
                        "limit": limit,
                        "percent_used": round(percent_used, 2),
                        "alert_threshold": budget_config["alert_at_percent"],
                    }
                    alerts.append(alert)
                    logger.warning(
                        "Provider limit alert: %s $%.2f / $%.2f (%.1f%%)",
                        provider_name, spent, limit, percent_used
                    )
            
            return alerts
            
        except Exception as e:
            logger.error("Failed to check provider limits: %s", e)
            return []
    
    def check_expensive_sessions(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Identify sessions exceeding expensive session threshold.
        
        Returns list of expensive sessions for alerting.
        """
        budget_config = self.get_budget_config()
        if not budget_config["enabled"]:
            return []
        
        threshold = threshold or budget_config["expensive_session_threshold"]
        
        engine = self._get_insights_engine()
        if not engine:
            return []
        
        try:
            report = engine.generate(days=1)
            if report.get("empty"):
                return []
            
            expensive = []
            for session in report.get("top_sessions", []):
                cost = float(session.get("estimated_cost", 0))
                if cost >= threshold:
                    expensive.append({
                        "session_id": session.get("id"),
                        "cost": round(cost, 4),
                        "threshold": threshold,
                        "model": session.get("model"),
                        "duration": session.get("duration_seconds"),
                    })
            
            return expensive
            
        except Exception as e:
            logger.error("Failed to check expensive sessions: %s", e)
            return []
    
    def run_cost_checks(self) -> Dict[str, Any]:
        """Run all cost monitoring checks.
        
        Returns comprehensive cost status for alerting.
        """
        budget_config = self.get_budget_config()
        
        if not budget_config["enabled"]:
            return {"enabled": False}
        
        results: Dict[str, Any] = {
            "enabled": True,
            "timestamp": datetime.now().isoformat(),
            "daily_budget": self.check_daily_budget(),
            "provider_alerts": self.check_provider_limits(),
            "expensive_sessions": self.check_expensive_sessions(),
        }
        
        # Determine overall alert status
        daily = results.get("daily_budget", {})
        daily_alert = daily.get("alert", False) if isinstance(daily, dict) else False
        has_alert = (
            daily_alert
            or len(results["provider_alerts"]) > 0
            or len(results["expensive_sessions"]) > 0
        )
        results["has_alert"] = has_alert
        
        return results
    
    def close(self):
        """Clean up database connection."""
        if self._db:
            try:
                self._db.close()
            except Exception:
                pass


def format_cost_alert(check_results: Dict[str, Any]) -> Optional[str]:
    """Format cost check results as human-readable alert message.

    Converts cost check results into a multi-line notification message
    suitable for Telegram, Discord, Slack, or logging. Returns None if
    cost monitoring disabled or no alert conditions triggered.

    Args:
        check_results: Dict from CostMonitor.check() or check_costs() with:
            - enabled: Boolean if cost monitoring is active
            - has_alert: Boolean if any alert condition triggered
            - daily_budget: Dict with spent, budget, percent_used, alert
            - provider_alerts: List of provider-specific alert dicts
            - expensive_sessions: List of expensive session dicts

    Returns:
        Formatted alert message string with emoji header and detail lines,
        or None if no alert warranted.

    Example output:
        "💰 Cost Alert
        Daily budget: $18.50 / $20.00 (92.5%)
        Provider openrouter: $12.00 / $15.00 (80.0%)"
    """
    if not check_results.get("enabled"):
        return None
    
    if not check_results.get("has_alert"):
        return None
    
    lines = ["💰 Cost Alert"]
    
    # Daily budget alert
    daily = check_results.get("daily_budget", {})
    if daily.get("alert"):
        lines.append(
            f"Daily budget: ${daily['spent']:.2f} / ${daily['budget']:.2f} "
            f"({daily['percent_used']:.1f}%)"
        )
    
    # Provider alerts
    for provider_alert in check_results.get("provider_alerts", []):
        lines.append(
            f"Provider {provider_alert['provider']}: "
            f"${provider_alert['spent']:.2f} / ${provider_alert['limit']:.2f} "
            f"({provider_alert['percent_used']:.1f}%)"
        )
    
    # Expensive sessions
    for session in check_results.get("expensive_sessions", []):
        lines.append(
            f"Expensive session {session['session_id'][:12]}: "
            f"${session['cost']:.2f} (threshold: ${session['threshold']:.2f})"
        )
    
    return "\n".join(lines)


# Convenience function for integration
def check_costs(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Run cost monitoring checks and return alert status.

    Convenience function for Agathos main loop integration. Instantiates
    CostMonitor, runs full check cycle, and returns results.

    Args:
        config: Agathos config dict with cost_monitoring section

    Returns:
        Dict with keys:
        - enabled: Boolean if cost monitoring active
        - has_alert: Boolean if any alert condition triggered
        - daily_budget: Dict with spent, budget, percent_used, alert status
        - provider_alerts: List of provider-specific budget alerts
        - expensive_sessions: List of sessions exceeding threshold
        - total_cost: Float total USD spent (all time or windowed)

    Side effects:
        - Instantiates CostMonitor (queries insights engine)
        - Opens/closes database connections
        - May query Hermes InsightsEngine for cost data

    Resource management: Automatically closes CostMonitor on completion.
    """
    monitor = CostMonitor(config)
    try:
        return monitor.run_cost_checks()
    finally:
        monitor.close()
