"""
Automatic model deprecation detection and handling system.

This module provides mechanisms to:
1. Detect when models become unavailable from providers
2. Automatically tag deprecated models
3. Provide user-friendly error messages with alternatives
4. Support automatic model redirects for known deprecations
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from hermes_cli.models import normalize_provider

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Path to the deprecation database
_DEPRECATION_DB_PATH = Path.home() / ".hermes" / "model_deprecations.json"

# Cache TTL for model availability checks (in seconds)
_MODEL_AVAILABILITY_CACHE_TTL = 3600  # 1 hour

# Number of consecutive failures before marking a model as deprecated
_DEPRECATION_FAILURE_THRESHOLD = 3

# Time window for failure counting (in seconds)
_DEPRECATION_FAILURE_WINDOW = 86400  # 24 hours


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ModelDeprecationRecord:
    """Record of a model deprecation event."""
    
    model_id: str
    provider: str
    first_detected: datetime
    last_detected: datetime
    failure_count: int
    status: str = "pending"  # pending, confirmed, resolved
    suggested_alternatives: List[str] = field(default_factory=list)
    redirect_model: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "first_detected": self.first_detected.isoformat(),
            "last_detected": self.last_detected.isoformat(),
            "failure_count": self.failure_count,
            "status": self.status,
            "suggested_alternatives": self.suggested_alternatives,
            "redirect_model": self.redirect_model,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelDeprecationRecord":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            provider=data["provider"],
            first_detected=datetime.fromisoformat(data["first_detected"]),
            last_detected=datetime.fromisoformat(data["last_detected"]),
            failure_count=data["failure_count"],
            status=data.get("status", "pending"),
            suggested_alternatives=data.get("suggested_alternatives", []),
            redirect_model=data.get("redirect_model"),
            error_message=data.get("error_message"),
        )


@dataclass
class ModelAvailabilityCache:
    """Cache for model availability check results."""
    
    model_id: str
    provider: str
    available: bool
    last_checked: datetime
    error_message: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return (datetime.now() - self.last_checked).total_seconds() > _MODEL_AVAILABILITY_CACHE_TTL


# ============================================================================
# Known Model Deprecations (Hardcoded for immediate availability)
# ============================================================================

_KNOWN_DEPRECATIONS: Dict[str, Dict[str, Any]] = {
    "deepseek-chat": {
        "provider": "deepseek",
        "redirect_model": "deepseek-v4-pro",
        "suggested_alternatives": ["deepseek-v4-pro", "deepseek-v4-flash", "deepseek-reasoner"],
        "deprecation_date": "2025-04-01",
        "reason": "DeepSeek API no longer supports the deepseek-chat model",
    },
}


# ============================================================================
# Deprecation Database Management
# ============================================================================

class ModelDeprecationDB:
    """Database for tracking model deprecations."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the deprecation database."""
        self.db_path = db_path or _DEPRECATION_DB_PATH
        self._records: Dict[str, ModelDeprecationRecord] = {}
        self._load()
    
    def _load(self):
        """Load deprecation records from disk."""
        if not self.db_path.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            return
        
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key, record_data in data.items():
                    self._records[key] = ModelDeprecationRecord.from_dict(record_data)
        except Exception as e:
            logger.warning("Failed to load deprecation database: %s", e)
    
    def _save(self):
        """Save deprecation records to disk."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, "w", encoding="utf-8") as f:
                data = {key: record.to_dict() for key, record in self._records.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save deprecation database: %s", e)
    
    def get_record(self, model_id: str, provider: str) -> Optional[ModelDeprecationRecord]:
        """Get deprecation record for a model."""
        key = f"{provider}:{model_id}"
        return self._records.get(key)
    
    def update_record(self, record: ModelDeprecationRecord):
        """Update or create a deprecation record."""
        key = f"{record.provider}:{record.model_id}"
        self._records[key] = record
        self._save()
    
    def delete_record(self, model_id: str, provider: str):
        """Delete a deprecation record."""
        key = f"{provider}:{model_id}"
        if key in self._records:
            del self._records[key]
            self._save()
    
    def get_confirmed_deprecations(self) -> List[ModelDeprecationRecord]:
        """Get all confirmed deprecation records."""
        return [
            record for record in self._records.values()
            if record.status == "confirmed"
        ]
    
    def get_pending_deprecations(self) -> List[ModelDeprecationRecord]:
        """Get all pending deprecation records."""
        return [
            record for record in self._records.values()
            if record.status == "pending"
        ]


# Global deprecation database instance
_deprecation_db = ModelDeprecationDB()


# ============================================================================
# Model Availability Checking
# ============================================================================

class ModelAvailabilityChecker:
    """Check if models are available from their providers."""
    
    def __init__(self):
        """Initialize the availability checker."""
        self._cache: Dict[str, ModelAvailabilityCache] = {}
    
    def check_availability(
        self,
        model_id: str,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a model is available from its provider.
        
        Returns:
            Tuple of (is_available, error_message)
        """
        cache_key = f"{provider}:{model_id}"
        
        # Check cache first
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if not cached.is_expired():
                return cached.available, cached.error_message
        
        # Perform actual availability check
        is_available, error_message = self._check_model_availability(
            model_id, provider, api_key, base_url
        )
        
        # Update cache
        self._cache[cache_key] = ModelAvailabilityCache(
            model_id=model_id,
            provider=provider,
            available=is_available,
            last_checked=datetime.now(),
            error_message=error_message,
        )
        
        return is_available, error_message
    
    def _check_model_availability(
        self,
        model_id: str,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Perform the actual availability check.
        
        This is a placeholder - the actual implementation would depend on
        the specific provider's API capabilities.
        """
        try:
            # Import here to avoid circular dependencies
            from hermes_cli.models import fetch_api_models
            
            normalized_provider = normalize_provider(provider)
            
            # Try to fetch the model list from the provider
            api_models = fetch_api_models(api_key, base_url)
            
            if api_models is None:
                # Could not reach API - assume available to avoid false positives
                return True, None
            
            # Check if the model is in the list
            if model_id in api_models:
                return True, None
            else:
                return False, f"Model `{model_id}` not found in provider's model listing"
                
        except Exception as e:
            logger.debug("Error checking model availability: %s", e)
            # On error, assume available to avoid false positives
            return True, None


# Global availability checker instance
_availability_checker = ModelAvailabilityChecker()


# ============================================================================
# Deprecation Detection and Management
# ============================================================================

def record_model_failure(
    model_id: str,
    provider: str,
    error_message: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ModelDeprecationRecord:
    """
    Record a model failure for deprecation tracking.
    
    Args:
        model_id: The model ID that failed
        provider: The provider name
        error_message: The error message received
        api_key: Optional API key for availability checking
        base_url: Optional base URL for the provider
        
    Returns:
        The updated or created deprecation record
    """
    normalized_provider = normalize_provider(provider)
    now = datetime.now()
    
    # Get existing record or create new one
    record = _deprecation_db.get_record(model_id, normalized_provider)
    
    if record is None:
        # Check if this is a known deprecation
        known_info = _KNOWN_DEPRECATIONS.get(model_id, {})
        record = ModelDeprecationRecord(
            model_id=model_id,
            provider=normalized_provider,
            first_detected=now,
            last_detected=now,
            failure_count=1,
            status="pending",
            suggested_alternatives=known_info.get("suggested_alternatives", []),
            redirect_model=known_info.get("redirect_model"),
            error_message=error_message,
        )
    else:
        # Update existing record
        record.last_detected = now
        record.failure_count += 1
        record.error_message = error_message
        
        # Check if threshold is reached
        time_since_first = (now - record.first_detected).total_seconds()
        if (record.failure_count >= _DEPRECATION_FAILURE_THRESHOLD and
            time_since_first <= _DEPRECATION_FAILURE_WINDOW):
            record.status = "confirmed"
            
            # If we have known deprecation info, use it
            known_info = _KNOWN_DEPRECATIONS.get(model_id, {})
            if known_info and not record.suggested_alternatives:
                record.suggested_alternatives = known_info.get("suggested_alternatives", [])
            if known_info and not record.redirect_model:
                record.redirect_model = known_info.get("redirect_model")
    
    _deprecation_db.update_record(record)
    return record


def get_deprecation_info(model_id: str, provider: str) -> Optional[Dict[str, Any]]:
    """
    Get deprecation information for a model.
    
    Returns:
        Dictionary with deprecation info or None if model is not deprecated
    """
    normalized_provider = normalize_provider(provider)
    
    # Check known deprecations first
    if model_id in _KNOWN_DEPRECATIONS:
        info = _KNOWN_DEPRECATIONS[model_id].copy()
        info["source"] = "known"
        return info
    
    # Check database
    record = _deprecation_db.get_record(model_id, normalized_provider)
    if record and record.status == "confirmed":
        return {
            "model_id": record.model_id,
            "provider": record.provider,
            "source": "detected",
            "redirect_model": record.redirect_model,
            "suggested_alternatives": record.suggested_alternatives,
            "error_message": record.error_message,
            "first_detected": record.first_detected.isoformat(),
        }
    
    return None


def get_deprecation_message(model_id: str, provider: str) -> Optional[str]:
    """
    Get a user-friendly deprecation message for a model.
    
    Returns:
        Formatted deprecation message or None if model is not deprecated
    """
    info = get_deprecation_info(model_id, provider)
    if not info:
        return None
    
    message_parts = [f"⚠️  Model `{model_id}` has been deprecated."]
    
    if info.get("redirect_model"):
        message_parts.append(
            f"Please use `{info['redirect_model']}` instead."
        )
    
    if info.get("suggested_alternatives"):
        alternatives = ", ".join(f"`{alt}`" for alt in info["suggested_alternatives"])
        message_parts.append(f"Alternative models: {alternatives}")
    
    if info.get("reason"):
        message_parts.append(f"Reason: {info['reason']}")
    
    return " ".join(message_parts)


def should_redirect_model(model_id: str, provider: str) -> Optional[str]:
    """
    Check if a deprecated model should be redirected to an alternative.
    
    Returns:
        The target model ID for redirection, or None if no redirect
    """
    info = get_deprecation_info(model_id, provider)
    if info and info.get("redirect_model"):
        return info["redirect_model"]
    return None


def validate_model_with_deprecation_check(
    model_id: str,
    provider: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate a model with deprecation checking.
    
    This extends the standard model validation to include deprecation
    detection and user-friendly error messages.
    
    Returns:
        Dictionary with validation results including deprecation info
    """
    # Import here to avoid circular dependencies
    from hermes_cli.models import validate_requested_model
    
    # Perform standard validation
    result = validate_requested_model(
        model_id, provider, api_key=api_key, base_url=base_url
    )
    
    # Check for deprecation
    deprecation_info = get_deprecation_info(model_id, provider)
    if deprecation_info:
        result["is_deprecated"] = True
        result["deprecation_info"] = deprecation_info
        
        # Add deprecation message to existing message
        deprecation_message = get_deprecation_message(model_id, provider)
        if deprecation_message:
            existing_message = result.get("message", "")
            if existing_message:
                result["message"] = f"{deprecation_message}\n\n{existing_message}"
            else:
                result["message"] = deprecation_message
        
        # Suggest redirect if available
        redirect_model = should_redirect_model(model_id, provider)
        if redirect_model:
            result["suggested_redirect"] = redirect_model
    else:
        result["is_deprecated"] = False
    
    return result


# ============================================================================
# Public API
# ============================================================================

def get_all_deprecations() -> Dict[str, Any]:
    """Get all known and detected deprecations."""
    known = {
        model_id: {**info, "source": "known"}
        for model_id, info in _KNOWN_DEPRECATIONS.items()
    }
    
    detected = {
        f"{record.provider}:{record.model_id}": record.to_dict()
        for record in _deprecation_db.get_confirmed_deprecations()
    }
    
    return {
        "known": known,
        "detected": detected,
    }


def clear_deprecation_cache():
    """Clear the deprecation database cache."""
    global _deprecation_db
    _deprecation_db = ModelDeprecationDB()


def add_known_deprecation(
    model_id: str,
    provider: str,
    redirect_model: Optional[str] = None,
    suggested_alternatives: Optional[List[str]] = None,
    reason: Optional[str] = None,
):
    """
    Add a known deprecation to the hardcoded list.
    
    This is primarily for development/testing purposes.
    """
    _KNOWN_DEPRECATIONS[model_id] = {
        "provider": provider,
        "redirect_model": redirect_model,
        "suggested_alternatives": suggested_alternatives or [],
        "reason": reason,
    }