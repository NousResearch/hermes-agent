"""
approval_queue.py - SQLite Approval Queue System
==================================================
Functions:
- SQLite-based approval queue for transactions
- Status: pending, approved, rejected, sent, failed
- Queue management (add, approve, reject, list)
- Transaction lifecycle tracking

SAFETY:
- ALL transactions must go through this queue
- No transaction executes without explicit approval

Usage:
    python -m custom_tools.approval_queue add --plan plan.json
    python -m custom_tools.approval_queue list
    python -m custom_tools.approval_queue approve --id 1
    python -m custom_tools.approval_queue reject --id 1
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from pathlib import Path



# Database path
DB_DIR = Path(os.getenv("APPROVAL_DB_DIR", ".data"))
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "approval_queue.db"

# Valid statuses
STATUSES = ("pending", "approved", "rejected", "sent", "failed")


def _get_db() -> sqlite3.Connection:
    """Get database connection, creating tables if needed."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS approval_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL DEFAULT 'pending',
            contract_address TEXT NOT NULL,
            chain TEXT NOT NULL DEFAULT 'ethereum',
            wallet_label TEXT NOT NULL,
            from_address TEXT NOT NULL,
            mint_function TEXT,
            quantity INTEGER DEFAULT 1,
            total_value_wei TEXT,
            gas_limit INTEGER,
            tx_data TEXT,
            tx_hash TEXT,
            tx_receipt TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            approved_at TEXT,
            sent_at TEXT,
            approved_by TEXT DEFAULT 'manual'
        )
    """)
    conn.commit()
    return conn



def add_to_queue(preview: dict) -> int:
    """
    Add a mint transaction to the approval queue.
    
    Args:
        preview: Transaction preview from mint_planner
    
    Returns:
        Queue entry ID
    """
    conn = _get_db()
    now = datetime.utcnow().isoformat()
    
    cursor = conn.execute("""
        INSERT INTO approval_queue 
        (status, contract_address, chain, wallet_label, from_address,
         mint_function, quantity, total_value_wei, gas_limit, tx_data,
         created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "pending",
        preview.get("contract", ""),
        preview.get("chain", "ethereum"),
        preview.get("from_wallet", ""),
        preview.get("from_address", ""),
        preview.get("mint_function", ""),
        preview.get("quantity", 1),
        str(preview.get("tx_data", {}).get("value", 0)),
        preview.get("estimated_gas", 200000),
        json.dumps(preview.get("tx_data", {}), default=str),
        now,
        now,
    ))
    conn.commit()
    entry_id = cursor.lastrowid
    conn.close()
    
    print(f"  Added to queue: ID #{entry_id} [PENDING]")
    return entry_id


def approve(entry_id: int, approved_by: str = "manual") -> dict:
    """Approve a pending transaction."""
    conn = _get_db()
    now = datetime.utcnow().isoformat()
    
    # Check current status
    row = conn.execute(
        "SELECT * FROM approval_queue WHERE id = ?", (entry_id,)
    ).fetchone()
    
    if not row:
        conn.close()
        raise ValueError(f"Entry #{entry_id} not found")
    
    if row["status"] != "pending":
        conn.close()
        raise ValueError(f"Entry #{entry_id} is '{row['status']}', not pending")
    
    conn.execute("""
        UPDATE approval_queue 
        SET status = 'approved', approved_at = ?, updated_at = ?, approved_by = ?
        WHERE id = ?
    """, (now, now, approved_by, entry_id))
    conn.commit()
    conn.close()
    
    print(f"  Approved: ID #{entry_id}")
    return {"id": entry_id, "status": "approved", "approved_by": approved_by}



def reject(entry_id: int, reason: str = "") -> dict:
    """Reject a pending transaction."""
    conn = _get_db()
    now = datetime.utcnow().isoformat()
    
    row = conn.execute(
        "SELECT * FROM approval_queue WHERE id = ?", (entry_id,)
    ).fetchone()
    
    if not row:
        conn.close()
        raise ValueError(f"Entry #{entry_id} not found")
    
    conn.execute("""
        UPDATE approval_queue 
        SET status = 'rejected', error_message = ?, updated_at = ?
        WHERE id = ?
    """, (reason, now, entry_id))
    conn.commit()
    conn.close()
    
    print(f"  Rejected: ID #{entry_id}")
    return {"id": entry_id, "status": "rejected", "reason": reason}


def mark_sent(entry_id: int, tx_hash: str) -> dict:
    """Mark transaction as sent."""
    conn = _get_db()
    now = datetime.utcnow().isoformat()
    
    conn.execute("""
        UPDATE approval_queue 
        SET status = 'sent', tx_hash = ?, sent_at = ?, updated_at = ?
        WHERE id = ?
    """, (tx_hash, now, now, entry_id))
    conn.commit()
    conn.close()
    
    return {"id": entry_id, "status": "sent", "tx_hash": tx_hash}


def mark_failed(entry_id: int, error: str) -> dict:
    """Mark transaction as failed."""
    conn = _get_db()
    now = datetime.utcnow().isoformat()
    
    conn.execute("""
        UPDATE approval_queue 
        SET status = 'failed', error_message = ?, updated_at = ?
        WHERE id = ?
    """, (error, now, entry_id))
    conn.commit()
    conn.close()
    
    return {"id": entry_id, "status": "failed", "error": error}



def get_entry(entry_id: int) -> dict:
    """Get a specific queue entry."""
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM approval_queue WHERE id = ?", (entry_id,)
    ).fetchone()
    conn.close()
    
    if not row:
        raise ValueError(f"Entry #{entry_id} not found")
    
    return dict(row)


def list_queue(status: str = None, limit: int = 50) -> list:
    """List queue entries, optionally filtered by status."""
    conn = _get_db()
    
    if status:
        if status not in STATUSES:
            raise ValueError(f"Invalid status: {status}")
        rows = conn.execute(
            "SELECT * FROM approval_queue WHERE status = ? ORDER BY id DESC LIMIT ?",
            (status, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM approval_queue ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    
    conn.close()
    return [dict(row) for row in rows]


def get_approved_pending_execution() -> list:
    """Get all approved transactions waiting for execution."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT * FROM approval_queue WHERE status = 'approved' ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Approval Queue Manager")
    sub = parser.add_subparsers(dest="command")
    
    # List
    list_p = sub.add_parser("list", help="List queue entries")
    list_p.add_argument("--status", choices=STATUSES, help="Filter by status")
    
    # Approve
    approve_p = sub.add_parser("approve", help="Approve entry")
    approve_p.add_argument("--id", type=int, required=True, help="Entry ID")
    
    # Reject
    reject_p = sub.add_parser("reject", help="Reject entry")
    reject_p.add_argument("--id", type=int, required=True, help="Entry ID")
    reject_p.add_argument("--reason", default="", help="Rejection reason")
    
    args = parser.parse_args()
    
    if args.command == "list":
        entries = list_queue(status=args.status)
        for e in entries:
            print(f"  #{e['id']} [{e['status']}] {e['contract_address']} "
                  f"wallet={e['wallet_label']} qty={e['quantity']}")
        if not entries:
            print("  Queue is empty.")
    elif args.command == "approve":
        result = approve(args.id)
        print(json.dumps(result, indent=2))
    elif args.command == "reject":
        result = reject(args.id, args.reason)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
