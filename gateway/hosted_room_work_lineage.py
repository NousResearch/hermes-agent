"""Bind passive work provenance to actually retained canonical authority spans."""

from gateway import hosted_room_passive_lineage as lineage
from gateway.hosted_room_authority_history import at_sequence


def source_prefix_locked(conn, room_id, authority, seq):
    from gateway.hosted_room_work_records import WorkRecordPrefixError
    try:
        history, digest = lineage.source_locked(conn, room_id, authority)
        spans, _, _ = lineage.descriptor(history, gateway_id=authority["gateway_id"], epoch=authority["epoch"])
        count = 0
        for event in conn.execute("SELECT * FROM hosted_room_events WHERE room_id=? AND seq<=? ORDER BY seq", (room_id, seq)):
            count += 1
            if event["seq"] != count:
                raise WorkRecordPrefixError("work record history is not contiguous")
            lineage.event_span(spans, event)
        if count != seq or lineage.status(spans, seq) != "verified":
            raise WorkRecordPrefixError("work record history is not verified")
        return spans, digest
    except (lineage.PassiveLineageError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise WorkRecordPrefixError(str(exc)) from exc


def target_prefix_locked(conn, row, record):
    from gateway.hosted_room_work_records import WorkRecordPrefixError
    try:
        spans = lineage.replica_history_locked(conn, row)
        if (record["lineage_sha256"] != row["lineage_sha256"]
                or lineage.status(spans, record["history"]["seq"]) != "verified"):
            raise WorkRecordPrefixError("work record lineage prefix is not verified")
        validate_provenance(record, spans)
        return spans
    except (lineage.PassiveLineageError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise WorkRecordPrefixError(str(exc)) from exc


def validate_provenance(record, spans):
    from gateway.hosted_room_work_records import WorkRecordError
    # Execution generation remains an independent per-task counter.
    for task in record["tasks"]:
        at_sequence(spans, task["source_event_seq"])
    for receipt in record["receipts"]:
        if not any(s.gateway_id == receipt["home_install_id"] == receipt["authority_gateway_id"]
                   and s.epoch == receipt["authority_epoch"] for s in spans):
            raise WorkRecordError("work record receipt has no original authority span")


def task_origins(record, spans):
    return {task["task_id"]: {"gateway_id": origin.gateway_id, "epoch": origin.epoch}
            for task in record["tasks"] for origin in [at_sequence(spans, task["source_event_seq"])]}
