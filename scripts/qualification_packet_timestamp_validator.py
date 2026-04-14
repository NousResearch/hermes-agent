#!/usr/bin/env python3
"""Qualification packet timestamp validator (v1).

Reusable module for validating and backfilling timestamp fields in qualification packets.
Integrates with model rollout gate runner and other qualification packet producers.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple


def parse_iso_timestamp(raw: Any) -> Optional[dt.datetime]:
    """Parse ISO 8601 timestamp string to datetime object."""
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(text)
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def validate_qualification_packet_timestamps(packet: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Validate qualification packet timestamp completeness.
    
    Args:
        packet: Qualification packet dictionary
    
    Returns:
        Tuple of (is_valid, missing_fields, invalid_fields)
        - is_valid: True if all required timestamps are present and valid
        - missing_fields: List of missing timestamp field names
        - invalid_fields: List of timestamp fields with invalid format
    """
    missing = []
    invalid = []
    
    # Check evaluated_at
    evaluated_at = packet.get("evaluated_at")
    if not evaluated_at:
        missing.append("evaluated_at")
    elif not parse_iso_timestamp(evaluated_at):
        invalid.append("evaluated_at")
    
    # Check scorecard.scored_at
    scorecard = packet.get("scorecard", {})
    if isinstance(scorecard, dict):
        scored_at = scorecard.get("scored_at")
        if not scored_at:
            missing.append("scorecard.scored_at")
        elif not parse_iso_timestamp(scored_at):
            invalid.append("scorecard.scored_at")
    else:
        # No scorecard at all
        missing.append("scorecard.scored_at")
    
    # Check scorecard.cost.provider_evidence_updated_at
    cost = scorecard.get("cost", {}) if isinstance(scorecard, dict) else {}
    if isinstance(cost, dict):
        provider_updated = cost.get("provider_evidence_updated_at")
        if not provider_updated:
            missing.append("scorecard.cost.provider_evidence_updated_at")
        elif not parse_iso_timestamp(provider_updated):
            invalid.append("scorecard.cost.provider_evidence_updated_at")
    else:
        # No cost section
        missing.append("scorecard.cost.provider_evidence_updated_at")
    
    is_valid = len(missing) == 0 and len(invalid) == 0
    return is_valid, missing, invalid


def is_timestamp_complete(packet: Dict[str, Any]) -> bool:
    """Check if qualification packet has all required timestamps.
    
    Args:
        packet: Qualification packet dictionary
    
    Returns:
        True if all required timestamps are present and valid
    """
    is_valid, missing, invalid = validate_qualification_packet_timestamps(packet)
    return is_valid


def get_missing_timestamp_fields(packet: Dict[str, Any]) -> List[str]:
    """Get list of missing timestamp fields in qualification packet.
    
    Args:
        packet: Qualification packet dictionary
    
    Returns:
        List of missing timestamp field names
    """
    _, missing, _ = validate_qualification_packet_timestamps(packet)
    return missing


def get_invalid_timestamp_fields(packet: Dict[str, Any]) -> List[str]:
    """Get list of invalid timestamp fields in qualification packet.
    
    Args:
        packet: Qualification packet dictionary
    
    Returns:
        List of timestamp field names with invalid format
    """
    _, _, invalid = validate_qualification_packet_timestamps(packet)
    return invalid


def backfill_missing_timestamps(packet: Dict[str, Any], force_fresh: bool = False) -> Dict[str, Any]:
    """Backfill missing timestamp fields in qualification packet.
    
    Args:
        packet: Qualification packet dictionary
        force_fresh: If True, update all timestamps to current time
    
    Returns:
        Updated packet with backfilled timestamps
    
    Note:
        This function imports the existing backfill utility to avoid code duplication.
    """
    try:
        from qualification_packet_timestamp_backfill import backfill_qualification_packet
        return backfill_qualification_packet(packet, force_fresh=force_fresh)
    except ImportError:
        # Fallback implementation if import fails
        return _simple_backfill_timestamps(packet, force_fresh=force_fresh)


def _simple_backfill_timestamps(packet: Dict[str, Any], force_fresh: bool = False) -> Dict[str, Any]:
    """Simple fallback implementation for timestamp backfill.
    
    Used when the main backfill utility cannot be imported.
    """
    result = packet.copy()
    now_iso = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    
    # Backfill evaluated_at if missing or forced
    if force_fresh or not result.get("evaluated_at"):
        result["evaluated_at"] = now_iso
    
    # Ensure scorecard exists
    if "scorecard" not in result:
        result["scorecard"] = {}
    
    scorecard = result["scorecard"]
    if not isinstance(scorecard, dict):
        scorecard = {}
        result["scorecard"] = scorecard
    
    # Backfill scored_at if missing or forced
    if force_fresh or not scorecard.get("scored_at"):
        scorecard["scored_at"] = now_iso
    
    # Ensure cost section exists
    if "cost" not in scorecard:
        scorecard["cost"] = {}
    
    cost = scorecard["cost"]
    if not isinstance(cost, dict):
        cost = {}
        scorecard["cost"] = cost
    
    # Backfill provider_evidence_updated_at if missing or forced
    if force_fresh or not cost.get("provider_evidence_updated_at"):
        cost["provider_evidence_updated_at"] = now_iso
    
    return result


def ensure_qualification_packet_timestamps(
    packet: Dict[str, Any], 
    allow_backfill: bool = False,
    force_fresh: bool = False
) -> Tuple[Dict[str, Any], bool, List[str]]:
    """Ensure qualification packet has required timestamps.
    
    Args:
        packet: Qualification packet dictionary
        allow_backfill: If True, backfill missing timestamps
        force_fresh: If True, update all timestamps to current time
    
    Returns:
        Tuple of (updated_packet, was_backfilled, backfilled_fields)
        - updated_packet: Packet with timestamps (backfilled if needed)
        - was_backfilled: True if timestamps were backfilled
        - backfilled_fields: List of field names that were backfilled
    
    Raises:
        ValueError: If timestamps are missing and backfill not allowed
    """
    is_valid, missing, invalid = validate_qualification_packet_timestamps(packet)
    
    if is_valid and not force_fresh:
        return packet, False, []
    
    if not allow_backfill and not force_fresh:
        raise ValueError(f"Missing required timestamp fields: {', '.join(missing)}")
    
    # Backfill timestamps
    backfilled = backfill_missing_timestamps(packet, force_fresh=force_fresh)
    was_backfilled = backfilled != packet
    
    # Determine which fields were backfilled
    backfilled_fields = []
    
    # Check evaluated_at
    if packet.get("evaluated_at") != backfilled.get("evaluated_at"):
        backfilled_fields.append("evaluated_at")
    
    # Check scored_at
    packet_scorecard = packet.get("scorecard", {})
    backfilled_scorecard = backfilled.get("scorecard", {})
    if isinstance(packet_scorecard, dict) and isinstance(backfilled_scorecard, dict):
        if packet_scorecard.get("scored_at") != backfilled_scorecard.get("scored_at"):
            backfilled_fields.append("scored_at")
    
    # Check provider_evidence_updated_at
    packet_cost = packet_scorecard.get("cost", {}) if isinstance(packet_scorecard, dict) else {}
    backfilled_cost = backfilled_scorecard.get("cost", {}) if isinstance(backfilled_scorecard, dict) else {}
    if isinstance(packet_cost, dict) and isinstance(backfilled_cost, dict):
        if packet_cost.get("provider_evidence_updated_at") != backfilled_cost.get("provider_evidence_updated_at"):
            backfilled_fields.append("provider_evidence_updated_at")
    
    return backfilled, was_backfilled, backfilled_fields


if __name__ == "__main__":
    import json
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python qualification_packet_timestamp_validator.py <packet.json>")
        sys.exit(1)
    
    try:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            packet = json.load(f)
        
        is_valid, missing, invalid = validate_qualification_packet_timestamps(packet)
        
        result = {
            "schema": "clawd.qualification_packet_timestamp_validation.result.v1",
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "is_valid": is_valid,
            "missing_fields": missing,
            "invalid_fields": invalid,
            "is_timestamp_complete": is_timestamp_complete(packet),
            "missing_timestamp_fields": get_missing_timestamp_fields(packet),
            "invalid_timestamp_fields": get_invalid_timestamp_fields(packet),
        }
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        if not is_valid:
            sys.exit(1)
            
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)