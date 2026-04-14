#!/usr/bin/env python3
"""Qualification packet timestamp backfill/emission utility (v1).

Ensures qualification packets have required timestamp fields at the producer layer:
- evaluated_at: when the qualification was evaluated
- scored_at: when the scorecard was generated  
- provider_evidence_updated_at: when provider evidence was last updated

Design goals:
- Source-side truth production (not routing-side exceptions)
- Fail-closed preservation (missing timestamps = conservative rejection)
- Operator-visible backfill decisions
- Schema-compliant output
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "model_qualification_packet.schema.json"

SCHEMA_VERSION = "clawd.model_qualification_packet.v1"


def now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def load_json_file(path: Path) -> Any:
    """Load JSON file with UTF-8 encoding."""
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def backfill_qualification_packet(packet: Dict[str, Any], force_fresh: bool = False) -> Dict[str, Any]:
    """
    Backfill missing timestamp fields in a qualification packet.
    
    Args:
        packet: Qualification packet dictionary
        force_fresh: If True, always update timestamps to current time
    
    Returns:
        Updated packet with backfilled timestamps
    """
    # Create a deep copy to avoid mutating input
    import copy
    result = copy.deepcopy(packet)
    
    # Track what we backfilled
    backfilled_fields = []
    
    # 1. Ensure schema_version is present
    if "schema_version" not in result:
        result["schema_version"] = SCHEMA_VERSION
    
    # 2. Backfill evaluated_at if missing or forced
    evaluated_at = parse_iso_timestamp(result.get("evaluated_at"))
    if evaluated_at is None or force_fresh:
        result["evaluated_at"] = now_iso()
        if evaluated_at is None:
            backfilled_fields.append("evaluated_at")
    
    # 3. Backfill scored_at in scorecard if missing or forced
    scorecard = result.get("scorecard", {})
    if isinstance(scorecard, dict):
        scored_at = parse_iso_timestamp(scorecard.get("scored_at"))
        if scored_at is None or force_fresh:
            if "scorecard" not in result:
                result["scorecard"] = {}
            result["scorecard"]["scored_at"] = now_iso()
            if scored_at is None:
                backfilled_fields.append("scored_at")
    
    # 4. Backfill provider_evidence_updated_at in cost section if missing or forced
    cost = scorecard.get("cost", {}) if isinstance(scorecard, dict) else {}
    if isinstance(cost, dict):
        provider_evidence_updated_at = parse_iso_timestamp(cost.get("provider_evidence_updated_at"))
        if provider_evidence_updated_at is None or force_fresh:
            # Ensure nested structure exists
            if "scorecard" not in result:
                result["scorecard"] = {}
            if "cost" not in result["scorecard"]:
                result["scorecard"]["cost"] = {}
            result["scorecard"]["cost"]["provider_evidence_updated_at"] = now_iso()
            if provider_evidence_updated_at is None:
                backfilled_fields.append("provider_evidence_updated_at")
    
    # Add backfill metadata
    if backfilled_fields:
        if "source_refs" not in result:
            result["source_refs"] = []
        
        backfill_ref = {
            "ref_id": f"src_timestamp_backfill_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            "path": str(SCRIPT_PATH),
            "locator": "# timestamp_backfill",
            "content_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000"  # Placeholder
        }
        result["source_refs"].append(backfill_ref)
        
        # Add decision reference
        if "decision_refs" not in result:
            result["decision_refs"] = []
        result["decision_refs"].append("backfill:qualification_packet_timestamp_backfill_v1")
    
    return result


def validate_against_schema(packet: Dict[str, Any], schema_path: Path) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate packet against JSON schema.
    
    Returns:
        Tuple of (is_valid, error_message, validation_details)
    """
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError:
        return True, "jsonschema not available, skipping validation", None
    
    try:
        schema = load_json_file(schema_path)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        errors = list(validator.iter_errors(packet))
        if errors:
            error_details = []
            for error in errors:
                error_details.append({
                    "path": list(error.path),
                    "message": error.message,
                    "validator": error.validator,
                    "validator_value": error.validator_value,
                })
            return False, "schema_validation_failed", {"errors": error_details}
        return True, None, None
    except Exception as exc:
        return False, f"schema_validation_error: {exc}", None


def backfill_qualification_packet_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    force_fresh: bool = False,
    dry_run: bool = False,
    validate: bool = True,
    schema_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Backfill timestamps in a qualification packet file.
    
    Returns:
        Dictionary with backfill results
    """
    if schema_path is None:
        schema_path = DEFAULT_SCHEMA_PATH
    
    # Load input packet
    try:
        packet = load_json_file(input_path)
    except Exception as exc:
        return {
            "path": str(input_path),
            "success": False,
            "error": f"Failed to load input file: {exc}",
            "dry_run": dry_run
        }
    
    # Backfill timestamps
    backfilled = backfill_qualification_packet(packet, force_fresh=force_fresh)
    
    # Validate if requested
    validation_ok = True
    validation_error = None
    validation_details = None
    
    if validate:
        validation_ok, validation_error, validation_details = validate_against_schema(
            backfilled, schema_path
        )
    
    # Prepare result
    result = {
        "schema": "clawd.qualification_packet_timestamp_backfill.result.v1",
        "timestamp": now_iso(),
        "input_path": str(input_path),
        "output_path": str(output_path) if output_path else str(input_path),
        "backfilled_fields": [],
        "validation": {
            "requested": validate,
            "valid": validation_ok,
            "error": validation_error,
            "details": validation_details,
        },
        "dry_run": dry_run,
        "packet": backfilled,
    }
    
    # Extract backfilled fields from packet comparison
    original_evaluated_at = parse_iso_timestamp(packet.get("evaluated_at"))
    backfilled_evaluated_at = parse_iso_timestamp(backfilled.get("evaluated_at"))
    if original_evaluated_at is None and backfilled_evaluated_at is not None:
        result["backfilled_fields"].append("evaluated_at")
    
    # Handle scorecard safely
    original_scorecard = packet.get("scorecard", {})
    backfilled_scorecard = backfilled.get("scorecard", {})
    
    original_scored_at = parse_iso_timestamp(original_scorecard.get("scored_at") if isinstance(original_scorecard, dict) else None)
    backfilled_scored_at = parse_iso_timestamp(backfilled_scorecard.get("scored_at") if isinstance(backfilled_scorecard, dict) else None)
    if original_scored_at is None and backfilled_scored_at is not None:
        result["backfilled_fields"].append("scored_at")
    
    # Handle cost section safely
    original_cost = original_scorecard.get("cost", {}) if isinstance(original_scorecard, dict) else {}
    backfilled_cost = backfilled_scorecard.get("cost", {}) if isinstance(backfilled_scorecard, dict) else {}
    
    original_provider_updated = parse_iso_timestamp(
        original_cost.get("provider_evidence_updated_at") if isinstance(original_cost, dict) else None
    )
    backfilled_provider_updated = parse_iso_timestamp(
        backfilled_cost.get("provider_evidence_updated_at") if isinstance(backfilled_cost, dict) else None
    )
    if original_provider_updated is None and backfilled_provider_updated is not None:
        result["backfilled_fields"].append("provider_evidence_updated_at")
    
    # Write output if not dry run
    if not dry_run and validation_ok:
        actual_output_path = output_path if output_path else input_path
        try:
            atomic_write_json(actual_output_path, backfilled)
            result["written"] = True
        except Exception as exc:
            result["written"] = False
            result["write_error"] = str(exc)
            result["success"] = False
            return result
    
    result["success"] = validation_ok
    return result


def backfill_qualification_packet_batch(
    file_patterns: List[str],
    force_fresh: bool = False,
    validate: bool = True,
    dry_run: bool = False,
    schema_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Backfill multiple qualification packets.
    
    Args:
        file_patterns: List of file patterns (glob patterns)
        force_fresh: If True, force all timestamps to current time
        validate: If True, validate against schema
        dry_run: If True, don't write changes
        schema_path: Path to JSON schema
    
    Returns:
        Dictionary with batch results
    """
    if schema_path is None:
        schema_path = DEFAULT_SCHEMA_PATH
    
    # Expand glob patterns
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern, recursive=True))
    
    results = {
        "schema": "clawd.qualification_packet_timestamp_backfill.batch_result.v1",
        "timestamp": now_iso(),
        "patterns": file_patterns,
        "files_found": len(all_files),
        "processed": 0,
        "success": 0,
        "failed": 0,
        "details": []
    }
    
    for file_path_str in all_files:
        file_path = Path(file_path_str)
        try:
            result = backfill_qualification_packet_file(
                input_path=file_path,
                output_path=None,  # Overwrite input
                force_fresh=force_fresh,
                dry_run=dry_run,
                validate=validate,
                schema_path=schema_path
            )
            
            results["processed"] += 1
            if result.get("success", False):
                results["success"] += 1
            else:
                results["failed"] += 1
            
            results["details"].append(result)
            
        except Exception as e:
            results["processed"] += 1
            results["failed"] += 1
            results["details"].append({
                "path": file_path_str,
                "success": False,
                "error": str(e)
            })
    
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill timestamp fields in qualification packets"
    )
    
    # Input mode: single file or batch
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input qualification packet JSON file")
    input_group.add_argument("--batch", nargs="+", help="Batch mode: one or more file patterns (glob)")
    
    parser.add_argument("--output", help="Output file (default: overwrite input)")
    parser.add_argument("--force-fresh", action="store_true", 
                       help="Force all timestamps to current time (not just missing ones)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be backfilled without writing")
    parser.add_argument("--validate", action="store_true", default=True,
                       help="Validate against schema after backfill (default: True)")
    parser.add_argument("--no-validate", action="store_false", dest="validate",
                       help="Disable schema validation")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH),
                       help="Path to model qualification packet JSON schema")
    parser.add_argument("--json", action="store_true",
                       help="Output JSON result")
    
    args = parser.parse_args()
    
    # Handle batch mode
    if args.batch:
        results = backfill_qualification_packet_batch(
            file_patterns=args.batch,
            force_fresh=args.force_fresh,
            validate=args.validate,
            dry_run=args.dry_run,
            schema_path=Path(args.schema).expanduser().resolve()
        )
        
        # Output results
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(f"Batch timestamp backfill completed:")
            print(f"  Patterns: {', '.join(args.batch)}")
            print(f"  Files found: {results['files_found']}")
            print(f"  Processed: {results['processed']}")
            print(f"  Success: {results['success']}")
            print(f"  Failed: {results['failed']}")
            if args.dry_run:
                print(f"  Mode: DRY RUN (no changes written)")
            
            # Show summary of backfilled fields
            backfilled_counts = {}
            for detail in results["details"]:
                if detail.get("success", False):
                    for field in detail.get("backfilled_fields", []):
                        backfilled_counts[field] = backfilled_counts.get(field, 0) + 1
            
            if backfilled_counts:
                print(f"  Backfilled fields:")
                for field, count in backfilled_counts.items():
                    print(f"    {field}: {count} file(s)")
            
            # Show errors if any
            if results["failed"] > 0:
                print(f"  Errors:")
                for detail in results["details"]:
                    if not detail.get("success", False):
                        error = detail.get("error", "Unknown error")
                        print(f"    {detail.get('path', 'Unknown')}: {error}")
        
        return 0 if results["failed"] == 0 else 1
    
    # Handle single file mode
    else:
        input_path = Path(args.input).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve() if args.output else None
        
        result = backfill_qualification_packet_file(
            input_path=input_path,
            output_path=output_path,
            force_fresh=args.force_fresh,
            dry_run=args.dry_run,
            validate=args.validate,
            schema_path=Path(args.schema).expanduser().resolve()
        )
        
        # Output result
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"Timestamp backfill completed:")
            print(f"  Input: {result['input_path']}")
            print(f"  Output: {result['output_path']}")
            if result['backfilled_fields']:
                print(f"  Backfilled fields: {', '.join(result['backfilled_fields'])}")
            else:
                print(f"  No fields needed backfill")
            if args.validate:
                if result['validation']['valid']:
                    print(f"  Validation: PASS")
                else:
                    print(f"  Validation: FAIL - {result['validation']['error']}")
            if args.dry_run:
                print(f"  Mode: DRY RUN (no changes written)")
            else:
                print(f"  Written: {'YES' if result.get('written', False) else 'NO'}")
        
        return 0 if result.get('success', False) else 1



if __name__ == "__main__":
    sys.exit(main())