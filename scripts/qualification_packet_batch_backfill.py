#!/usr/bin/env python3
"""Batch migration script for legacy qualification packets.

Finds and backfills qualification packets missing required timestamps:
- evaluated_at
- scorecard.scored_at
- scorecard.cost.provider_evidence_updated_at

Usage:
  # Dry-run (preview)
  python qualification_packet_batch_backfill.py --directory state/ --dry-run --json
  
  # Apply migration
  python qualification_packet_batch_backfill.py --directory state/ --apply --validate
  
  # Force fresh timestamps (migration scenario)
  python qualification_packet_batch_backfill.py --directory state/ --force-fresh --apply --validate
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path to import backfill utility
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from qualification_packet_timestamp_backfill import (
        backfill_qualification_packet,
        validate_against_schema,
        SCHEMA_VERSION,
        DEFAULT_SCHEMA_PATH,
    )
except ImportError:
    print("Error: Could not import qualification_packet_timestamp_backfill module")
    print("Make sure scripts/qualification_packet_timestamp_backfill.py exists")
    sys.exit(1)


def _is_qualification_packet(data: Dict[str, Any]) -> bool:
    """Check if data looks like a qualification packet."""
    # Check for qualification packet signature fields
    if not isinstance(data, dict):
        return False
    
    # Check for qualification_id or schema_version indicating qualification packet
    has_qualification_id = "qualification_id" in data
    has_schema_version = "schema_version" in data and isinstance(data["schema_version"], str)
    
    # Check for model qualification packet schema
    if has_schema_version:
        schema_version = data["schema_version"]
        if "qualification" in schema_version.lower() or "clawd.model_qualification" in schema_version:
            return True
    
    # Check for qualification structure
    has_qualification_section = "qualification" in data and isinstance(data["qualification"], dict)
    has_model_section = "model" in data and isinstance(data["model"], dict)
    
    return (has_qualification_id or has_qualification_section) and has_model_section


def _check_timestamp_completeness(packet: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check if qualification packet has all required timestamps.
    
    Returns:
        Tuple of (is_legacy, missing_fields)
    """
    missing_fields = []
    
    # Check evaluated_at
    evaluated_at = packet.get("evaluated_at")
    if not evaluated_at or not _parse_iso_timestamp(evaluated_at):
        missing_fields.append("evaluated_at")
    
    # Check scorecard.scored_at
    scorecard = packet.get("scorecard", {})
    if isinstance(scorecard, dict):
        scored_at = scorecard.get("scored_at")
        if not scored_at or not _parse_iso_timestamp(scored_at):
            missing_fields.append("scorecard.scored_at")
    
    # Check scorecard.cost.provider_evidence_updated_at
    cost = scorecard.get("cost", {}) if isinstance(scorecard, dict) else {}
    if isinstance(cost, dict):
        provider_updated = cost.get("provider_evidence_updated_at")
        if not provider_updated or not _parse_iso_timestamp(provider_updated):
            missing_fields.append("scorecard.cost.provider_evidence_updated_at")
    
    return len(missing_fields) > 0, missing_fields


def _parse_iso_timestamp(timestamp_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not timestamp_str:
        return None
    
    try:
        # Handle various ISO formats
        timestamp_str = timestamp_str.replace("Z", "+00:00")
        
        # Try with timezone
        try:
            return datetime.fromisoformat(timestamp_str)
        except ValueError:
            # Try without timezone
            return datetime.fromisoformat(timestamp_str.replace("+00:00", ""))
    except (ValueError, AttributeError):
        return None


def find_legacy_qualification_packets(directory: Path) -> List[Tuple[Path, List[str]]]:
    """Find qualification packets missing required timestamps.
    
    Returns:
        List of (file_path, missing_fields) tuples
    """
    legacy_packets = []
    
    print(f"Scanning directory: {directory}")
    
    for json_file in directory.rglob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if _is_qualification_packet(data):
                is_legacy, missing_fields = _check_timestamp_completeness(data)
                if is_legacy:
                    legacy_packets.append((json_file, missing_fields))
                    print(f"  Found legacy packet: {json_file.relative_to(directory)}")
                    print(f"    Missing: {', '.join(missing_fields)}")
                    
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # Skip non-JSON files or invalid JSON
            continue
        except Exception as e:
            print(f"  Error reading {json_file}: {e}")
            continue
    
    return legacy_packets


def backfill_qualification_packet_file(
    file_path: Path,
    force_fresh: bool = False,
    validate: bool = True,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Backfill timestamps in a qualification packet file.
    
    Returns:
        Dictionary with backfill results
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            packet = json.load(f)
        
        # Check if already has all timestamps
        is_legacy, missing_fields = _check_timestamp_completeness(packet)
        if not is_legacy and not force_fresh:
            return {
                "path": str(file_path),
                "success": True,
                "backfilled_fields": [],
                "message": "Already has all required timestamps",
                "dry_run": dry_run
            }
        
        # Backfill timestamps
        backfilled = backfill_qualification_packet(packet, force_fresh=force_fresh)
        
        # Validate schema if requested
        validation_result = None
        if validate:
            validation_ok, validation_error, validation_details = validate_against_schema(
                backfilled, DEFAULT_SCHEMA_PATH
            )
            validation_result = {
                "valid": validation_ok,
                "error": validation_error,
                "details": validation_details
            }
            if not validation_ok:
                return {
                    "path": str(file_path),
                    "success": False,
                    "error": f"Schema validation failed: {validation_error}",
                    "dry_run": dry_run
                }
        
        # Determine which fields were backfilled
        backfilled_fields = []
        if packet.get("evaluated_at") != backfilled.get("evaluated_at"):
            backfilled_fields.append("evaluated_at")
        
        packet_scorecard = packet.get("scorecard", {})
        backfilled_scorecard = backfilled.get("scorecard", {})
        if isinstance(packet_scorecard, dict) and isinstance(backfilled_scorecard, dict):
            if packet_scorecard.get("scored_at") != backfilled_scorecard.get("scored_at"):
                backfilled_fields.append("scored_at")
            
            packet_cost = packet_scorecard.get("cost", {})
            backfilled_cost = backfilled_scorecard.get("cost", {})
            if isinstance(packet_cost, dict) and isinstance(backfilled_cost, dict):
                if packet_cost.get("provider_evidence_updated_at") != backfilled_cost.get("provider_evidence_updated_at"):
                    backfilled_fields.append("provider_evidence_updated_at")
        
        # Write back if not dry-run
        if not dry_run:
            # Create backup
            backup_path = file_path.with_suffix(file_path.suffix + ".backup")
            shutil.copy2(file_path, backup_path)
            
            # Write backfilled packet
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(backfilled, f, indent=2)
        
        return {
            "path": str(file_path),
            "success": True,
            "backfilled_fields": backfilled_fields,
            "validation": validation_result,
            "dry_run": dry_run,
            "backup_created": not dry_run
        }
        
    except Exception as e:
        return {
            "path": str(file_path),
            "success": False,
            "error": str(e),
            "dry_run": dry_run
        }


def migrate_legacy_packets(
    directory: Path,
    apply: bool = False,
    force_fresh: bool = False,
    validate: bool = True,
    backup: bool = True
) -> Dict[str, Any]:
    """Migrate legacy qualification packets in batch.
    
    Args:
        directory: Directory to scan for qualification packets
        apply: If True, actually write changes (otherwise dry-run)
        force_fresh: If True, update all timestamps (migration scenario)
        validate: If True, validate schema after backfill
        backup: If True, create backup before modifying (only when apply=True)
    
    Returns:
        Dictionary with migration results
    """
    legacy_packets = find_legacy_qualification_packets(directory)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "directory": str(directory),
        "apply": apply,
        "force_fresh": force_fresh,
        "validate": validate,
        "backup": backup,
        "total_found": len(legacy_packets),
        "migrated": 0,
        "failed": 0,
        "errors": [],
        "details": []
    }
    
    if not legacy_packets:
        print(f"No legacy qualification packets found in {directory}")
        return results
    
    print(f"\nFound {len(legacy_packets)} legacy qualification packet(s)")
    print(f"Mode: {'DRY-RUN' if not apply else 'APPLY'}")
    print(f"Force fresh: {force_fresh}")
    print(f"Validate: {validate}")
    
    for i, (packet_path, missing_fields) in enumerate(legacy_packets, 1):
        print(f"\n[{i}/{len(legacy_packets)}] Processing: {packet_path.relative_to(directory)}")
        print(f"  Missing fields: {', '.join(missing_fields)}")
        
        try:
            result = backfill_qualification_packet_file(
                packet_path,
                force_fresh=force_fresh,
                validate=validate,
                dry_run=not apply
            )
            
            results["details"].append(result)
            
            if result["success"]:
                if not apply:
                    print(f"  ✓ Would backfill: {', '.join(result['backfilled_fields'])}")
                else:
                    results["migrated"] += 1
                    print(f"  ✓ Backfilled: {', '.join(result['backfilled_fields'])}")
                    if result.get("backup_created"):
                        print(f"    Backup created: {packet_path.name}.backup")
            else:
                results["failed"] += 1
                results["errors"].append({
                    "path": str(packet_path),
                    "error": result.get("error", "Unknown error")
                })
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "path": str(packet_path),
                "error": str(e)
            })
            print(f"  ✗ Error: {e}")
    
    print(f"\n{'='*60}")
    print("Migration Summary:")
    print(f"  Total found: {results['total_found']}")
    print(f"  {'Would migrate' if not apply else 'Migrated'}: {results['migrated']}")
    print(f"  Failed: {results['failed']}")
    
    if results['errors']:
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error['path']}: {error['error']}")
        if len(results['errors']) > 5:
            print(f"  ... and {len(results['errors']) - 5} more errors")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch migration script for legacy qualification packets"
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path.cwd(),
        help="Directory to scan for qualification packets (default: current directory)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes (default: dry-run)"
    )
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help="Update all timestamps to current time (migration scenario)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate schema after backfill (default: True)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Disable schema validation"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup before modifying (default: True, only when --apply)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Disable backup creation"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Check directory exists
    if not args.directory.exists():
        print(f"Error: Directory does not exist: {args.directory}")
        sys.exit(1)
    
    # Run migration
    results = migrate_legacy_packets(
        directory=args.directory,
        apply=args.apply,
        force_fresh=args.force_fresh,
        validate=args.validate,
        backup=args.backup
    )
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Exit code based on results
    if results["failed"] > 0:
        sys.exit(1)
    elif not args.apply and results["total_found"] > 0:
        print(f"\nNote: Run with --apply to actually migrate {results['total_found']} packet(s)")
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()