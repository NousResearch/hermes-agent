"""Bounded inspection of verified merge evidence, never merge execution."""
import json

from .merge_controller import MergeReceipt


def recent_merge_receipts(ledger):
    rows = ledger._connection.execute(
        "SELECT receipt_json FROM merge_attempts WHERE status = 'completed' "
        "AND receipt_json IS NOT NULL ORDER BY updated_at DESC LIMIT 10"
    ).fetchall()
    return [MergeReceipt.from_payload(json.loads(row[0])).to_payload() for row in rows]
