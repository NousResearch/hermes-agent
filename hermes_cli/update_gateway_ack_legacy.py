"""Protected-gateway acknowledgments for the per-home fallback marker."""

import json


def _marker_path():
    from hermes_cli.update_cmd_fleet import _fleet_restart_pending_marker_path
    return _fleet_restart_pending_marker_path()


def _read_marker(path):
    try:
        text = path.read_text(encoding="utf-8-sig")
        fields = {}
        for line in text.splitlines():
            key, sep, value = line.partition("=")
            if not sep or key in fields:
                return None
            fields[key] = value
        fields["inventory"] = json.loads(fields.get("inventory", "null"))
        return fields
    except (OSError, UnicodeError, ValueError):
        return None


def fallback_record():
    from hermes_cli.update_host_obligation import host_obligation_present
    return None if host_obligation_present() else _read_marker(_marker_path())


def acknowledge_fallback(token, receipt):
    """Mutate the specific fallback inventory under its marker lock."""
    from hermes_cli.update_host_obligation import (
        _ack_ledger_path, _protected_ack_token, _read_ack_ledger, _receipt_gateway_hashes,
        _receipt_snapshot_hash, _write_record, host_obligation_present, replace_bytes,
    )
    from hermes_cli.update_lock import marker_mutex
    marker = _marker_path()
    with marker_mutex(marker):
        if host_obligation_present():
            return False
        record = _read_marker(marker)
        inventory = record.get("inventory") if record else None
        if not isinstance(inventory, dict) or inventory.get("version") != 1 or not isinstance(inventory.get("runtimes"), list):
            return False
        rows = inventory["runtimes"]
        matches = [row for row in rows if isinstance(row, dict)
                   and row.get("kind") == "gateway" and row.get("protected_unmapped")
                   and row.get("incident_id") and _protected_ack_token(record, row, receipt, rows.count(row)) == token]
        if not matches or any(row != matches[0] for row in matches):
            return False
        row = matches[0]
        receipt_id = receipt.get("update_id")
        receipt_sha = (receipt.get("post_update") or {}).get("sha") if isinstance(receipt.get("post_update") or {}, dict) else None
        if isinstance(receipt_id, str) and receipt_id:
            ledger_path = _ack_ledger_path(marker)
            ledger = _read_ack_ledger(ledger_path)
            snapshot = _receipt_snapshot_hash(receipt)
            hashes = (_receipt_gateway_hashes(receipt, row.get("pid"), row["incident_id"])
                      if receipt_sha == record.get("expected_sha") else [])
            ledger.extend({"receipt_id": receipt_id, "receipt_hash": snapshot, "row_hash": value,
                           "incident_id": row["incident_id"]} for value in hashes)
            ledger.append({"receipt_id": receipt_id, "receipt_hash": snapshot, "incident_id": row["incident_id"],
                           "pid": row["pid"], "recovery_context": True})
            _write_record(ledger_path, {"version": 1, "entries": ledger})
        record["inventory"] = {**inventory, "runtimes": [item for item in rows if item != row]}
        lines = [f"{key}={json.dumps(value) if key == 'inventory' else value}" for key, value in record.items()]
        replace_bytes(marker, ("\n".join(lines) + "\n").encode("utf-8"))
        return True


def merge_fallback_inventory(expected_sha, additions):
    """Merge into the current fallback marker without erasing concurrent debts."""
    from hermes_cli.update_host_obligation import host_obligation_present, replace_bytes
    from hermes_cli.update_lock import marker_mutex
    marker = _marker_path()
    with marker_mutex(marker):
        if host_obligation_present():
            return False
        record = _read_marker(marker)
        if record is None or record.get("expected_sha", "") != expected_sha:
            return False
        inventory = record.get("inventory")
        if inventory is None:
            inventory = {"version": 1, "runtimes": []}
        if (not isinstance(inventory, dict) or inventory.get("version") != 1
                or not isinstance(inventory.get("runtimes"), list)):
            return False
        rows = list(inventory["runtimes"])
        rows.extend(row for row in additions if row not in rows)
        record["inventory"] = {**inventory, "runtimes": rows}
        lines = [f"{key}={json.dumps(value) if key == 'inventory' else value}" for key, value in record.items()]
        replace_bytes(marker, ("\n".join(lines) + "\n").encode("utf-8"))
        return True

def _acknowledged_receipt_has_unresolved_debt(original: dict, receipt: dict) -> bool:
    if receipt == original:
        return False
    restart = receipt.get("gateway_restart") or {}
    return bool(receipt.get("unrecorded_gateway_obligations") or (isinstance(restart, dict) and (restart.get("incomplete") or restart.get("phase_error")
                                                or restart.get("failed_units")))
                or any(isinstance(row, dict) and row.get("step") == "gateway_restart"
                       for row in receipt.get("followups") or []))
