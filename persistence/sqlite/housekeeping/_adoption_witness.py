"""Compare the actual transcript, context and accounting before retiring a source."""

import json
import math

from .._schema_ddl import _sql_json_extract
from ..conversation._store.sql_common import QUEUED_PROMPT_METADATA_KEY
from ._adoption_records import AdoptionConflict, activity_digest


# Physical row/display indexes are rebuilt on import; every other stored field can carry
# history or sidecars. Legacy rows without logical UIDs may acquire them on import.
_ADOPTION_PHYSICAL_FIELDS = {"id", "session_id", "display_identity", "display_order"}
_ADOPTION_OPTIONAL_UID_FIELDS = {"message_uid", "absorbed_message_uids", "tool_call_uids", "tool_call_uid"}
_ADOPTION_JSON_FIELDS = {"tool_calls", "reasoning_details", "codex_reasoning_items", "codex_message_items",
                         "display_metadata", "absorbed_message_uids", "tool_call_uids"}


def _adoption_value(field, value):
    json_text = value
    structured_content = field == "content" and isinstance(value, str) and value.startswith("\x00json:")
    if structured_content:
        json_text = value[len("\x00json:"):]
    if (field in _ADOPTION_JSON_FIELDS or structured_content) and isinstance(json_text, str):
        try:
            decoded = json.loads(json_text)
        except (ValueError, TypeError):
            return value  # Malformed stored bytes must still compare exactly.
        if field == "tool_calls" and decoded == []:
            return None
        # Canonical JSON retains type distinctions such as true versus 1.
        return ("json", json.dumps(decoded, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
    return value


def _adoption_history_covers(donor_conn, target_conn, session_id, *, retired_witness=None):
    """Does the target retain every donor row in storage order, regardless of physical row ids?"""
    if (donor_conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None
            or target_conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None):
        return False
    donor = donor_conn.execute("SELECT * FROM messages WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
    target = target_conn.execute("SELECT * FROM messages WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
    if len(donor) > len(target):
        return False
    allowed_retired_rows = set()
    if retired_witness is not None:
        source_state = donor_conn.execute(
            "SELECT ended_at, end_reason, archived FROM sessions WHERE id=?", (session_id,)).fetchone()
        count = retired_witness.get("source_count")
        witness = retired_witness.get("target_active_digest")
        if count is None or witness is None:
            raise AdoptionConflict("Saved adoption has no activity witness; inspect history before recovery")
        if count != len(donor) or activity_digest(target_conn, session_id, count) != witness:
            raise AdoptionConflict("Saved adoption activity witness differs; inspect history before recovery")
        if (source_state and source_state[0] is not None and source_state[1] == "adopted_by_profile"
                and source_state[2] in (0, 1)):
            expression = _sql_json_extract('display_metadata', '$.' + QUEUED_PROMPT_METADATA_KEY)
            allowed_retired_rows = {row[0] for row in donor_conn.execute(
                "SELECT id FROM messages WHERE session_id=? AND role='user' AND active=0 "
                f"AND COALESCE({expression}, 0) = 1", (session_id,))}
    for source, destination in zip(donor, target):
        src, dst = dict(source), dict(destination)
        if src.keys() != dst.keys():
            return False
        for field, value in src.items():
            if field in _ADOPTION_PHYSICAL_FIELDS or (field in _ADOPTION_OPTIONAL_UID_FIELDS and value is None):
                continue
            if _adoption_value(field, value) != _adoption_value(field, dst[field]):
                if (field == "active" and value == 0 and dst[field] == 1
                        and src["id"] in allowed_retired_rows):
                    continue
                return False
    return True


# Accounting is an aggregate rather than an event log. An existing target covers
# a donor attribution only when the original first_seen witness is retained and
# every nonnegative cumulative amount and last_seen includes the donor's range.
_ADOPTION_TOTALS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
                    "reasoning_tokens", "estimated_cost_usd", "actual_cost_usd", "api_call_count")
_ADOPTION_USAGE_KEY = ("session_id", "model", "billing_provider", "billing_base_url", "billing_mode", "task")
_ADOPTION_BILLING_CONTEXT = ("billing_provider", "billing_base_url", "billing_mode",
                             "cost_status", "cost_source", "pricing_version")


def _adoption_context(conn, session_id):
    row = conn.execute(
        "SELECT s.*, sp.prompt AS resolved_prompt, tp.prompt AS resolved_tools "
        "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash=s.system_prompt_hash "
        "LEFT JOIN system_prompts tp ON tp.hash=s.tool_names WHERE s.id=?", (session_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    if data["system_prompt_hash"] is not None and data["resolved_prompt"] is None:
        raise ValueError("Donor or target prompt reference cannot be resolved")
    pin = data["tool_names"]
    if pin is not None and not isinstance(pin, str):
        raise ValueError("Donor or target tool pin has an invalid representation")
    if pin and len(pin) == 64 and all(ch in "0123456789abcdef" for ch in pin) and data["resolved_tools"] is None:
        raise ValueError("Donor or target tool pin cannot be resolved")
    data["prompt_body"] = data["resolved_prompt"] if data["system_prompt_hash"] is not None else data["system_prompt"]
    data["tool_body"] = data["resolved_tools"] if data["resolved_tools"] is not None else pin
    return data


def _adoption_amount_covers(source, target):
    if source is None:
        return True
    if type(source) not in (int, float) or type(target) not in (int, float):
        return False
    if not math.isfinite(source) or not math.isfinite(target):
        return False
    # A negative total can be a correction/credit; a larger number does not prove it survived.
    return target == source if source < 0 else target >= source


def _adoption_accounting_covers(donor_conn, target_conn, session_id):
    source, target = _adoption_context(donor_conn, session_id), _adoption_context(target_conn, session_id)
    if source is None or target is None:
        return False
    for field in ("prompt_body", "tool_body", "created_source", *_ADOPTION_BILLING_CONTEXT):
        if source[field] is not None and source[field] != target[field]:
            return False
    # A changed chosen title cannot prove that the donor's title/provenance survived.
    if source["title"] is not None:
        if source["title"] != target["title"]:
            return False
        # Legacy NULL provenance carries user-level protection, never a derived title.
        source_title = "user" if source["title_source"] is None else source["title_source"]
        target_title = "user" if target["title_source"] is None else target["title_source"]
        if source_title != target_title:
            return False
    if any(not _adoption_amount_covers(source[field], target[field]) for field in _ADOPTION_TOTALS):
        return False
    source_rows = donor_conn.execute("SELECT * FROM session_model_usage WHERE session_id=?", (session_id,)).fetchall()
    target_rows = target_conn.execute("SELECT * FROM session_model_usage WHERE session_id=?", (session_id,)).fetchall()
    by_key = {tuple(row[k] for k in _ADOPTION_USAGE_KEY): dict(row) for row in target_rows}
    for row in source_rows:
        donor = dict(row)
        local = by_key.get(tuple(donor[k] for k in _ADOPTION_USAGE_KEY))
        if local is None or donor["first_seen"] != local["first_seen"]:
            return False
        origin = donor["first_seen"]
        if origin is not None and (type(origin) not in (int, float) or not math.isfinite(origin)):
            return False
        if any(not _adoption_amount_covers(donor[field], local[field]) for field in _ADOPTION_TOTALS):
            return False
        # Without an origin witness, larger aggregates cannot prove they extend this donor.
        if origin is None and any(donor[field] != local[field] for field in _ADOPTION_TOTALS):
            return False
        end = donor["last_seen"]
        if end is not None and not _adoption_amount_covers(end, local["last_seen"]):
            return False
        if any(donor[field] is not None and donor[field] != local[field] for field in ("cost_status", "cost_source")):
            return False
    return True


def _copy_adoption_portable_state(db, donor_conn, target_conn, session_id):
    """Copy attribution and the resolved tool pin only for a newly inserted target."""
    context = _adoption_context(donor_conn, session_id)
    if context is None:
        raise ValueError("Donor session disappeared during adoption")
    # These are trusted, guarded donor provenance, not fields accepted from
    # generic external JSON imports; neither transfers routing or authority.
    target_conn.execute("UPDATE sessions SET created_source=?,title_source=? WHERE id=?",
                        (context["created_source"], context["title_source"], session_id))
    pin = context["tool_body"]
    if pin is not None:
        target_conn.execute("UPDATE sessions SET tool_names=? WHERE id=?",
                            (db._store_system_prompt(target_conn, pin), session_id))
    source_columns = [row[1] for row in donor_conn.execute("PRAGMA table_info(session_model_usage)")]
    target_columns = [row[1] for row in target_conn.execute("PRAGMA table_info(session_model_usage)")]
    if set(source_columns) != set(target_columns):
        raise ValueError("Usage schemas differ; adoption cannot preserve attribution")
    columns = ", ".join(source_columns)
    rows = donor_conn.execute(f"SELECT {columns} FROM session_model_usage WHERE session_id=?", (session_id,)).fetchall()
    if rows:
        placeholders = ", ".join("?" for _ in source_columns)
        target_conn.executemany(f"INSERT INTO session_model_usage({columns}) VALUES ({placeholders})",
                                [tuple(row) for row in rows])
