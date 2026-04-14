#!/usr/bin/env python3
"""
Record a critique packet in a ledger JSONL file.
"""
import argparse
import json
import datetime as dt
import sys
import pathlib

def record_critique_packet(packet_path: pathlib.Path, packet: dict, root: pathlib.Path, validation_status: str = "generated") -> None:
    """
    Record packet generation in a ledger JSONL file.
    """
    ledger_path = packet_path.parent / "critique_packet_ledger.jsonl"
    entry = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "packet_path": str(packet_path.relative_to(root)),
        "packet_id": packet.get("packet_id"),
        "task_id": packet.get("task_id"),
        "validation_status": validation_status,
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    sys.stdout.write(f"Recorded packet to ledger: {ledger_path}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-path", required=True, help="Path to generated packet JSON file")
    parser.add_argument("--packet-json", required=True, help="Packet JSON string")
    parser.add_argument("--root", required=True, help="Root directory of the workspace")
    parser.add_argument("--validation-status", default="generated", help="Validation status")
    args = parser.parse_args()
    packet_path = pathlib.Path(args.packet_path).resolve()
    root = pathlib.Path(args.root).resolve()
    packet = json.loads(args.packet_json)
    record_critique_packet(packet_path, packet, root, args.validation_status)

if __name__ == "__main__":
    main()