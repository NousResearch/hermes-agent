"""Freshness checks with a durable check receipt; never sends or publishes."""
import json
import sqlite3
from datetime import datetime, timezone


def prepare_delivery(artifact_id):
    import x_manager as xm
    with sqlite3.connect(str(xm.DB_PATH)) as conn:
        conn.row_factory=sqlite3.Row
        row=conn.execute('SELECT * FROM x_manager_artifacts WHERE id=?',(artifact_id,)).fetchone()
        if row is None or row['status'] != xm.STATUS_PENDING:
            raise xm.XManagerError('artifact is absent or no longer pending')
        context=json.loads(row['context'])
        artifact=xm.XArtifact(row['id'],row['lane'],row['brand'],row['body'],xm.ArgumentPack(row['claim'],row['evidence'],row['mechanism'],row['position'],context),row['status'],row['created_at'])
        xm._validate_artifact(artifact)
        context['delivery_checked_at']=datetime.now(timezone.utc).isoformat()
        conn.execute('UPDATE x_manager_artifacts SET context=? WHERE id=?',(json.dumps(context),artifact_id))
    return artifact


def expiring_report_bytes(body, artifacts, *, generated):
    """Bind a static review's lifetime to its oldest source, not rendering time.

    The generic cron delivery owner checks this receipt at actual dispatch and
    bounds delayed sends/retries. No X imports or database access in cron.
    ``*.expiry.html`` is required so missing receipts fail closed as well.
    """
    import hashlib
    from datetime import timedelta

    expires = min(
        (datetime.fromisoformat(source['created_at'].replace('Z', '+00:00'))
         + timedelta(hours=6)
         for artifact in artifacts for source in artifact.pack.context['sources']),
        default=generated + timedelta(hours=6),
    )
    if not generated < expires:
        raise ValueError('review sources expired while rendering')
    payload = body.encode('utf-8')
    receipt = dict(version=1, not_before=generated.isoformat(),
                   expires_at=expires.isoformat(),
                   sha256=hashlib.sha256(payload).hexdigest())
    return ('<!-- hermes-delivery-expiry ' + json.dumps(receipt, sort_keys=True)
            + ' -->\n').encode('utf-8') + payload
