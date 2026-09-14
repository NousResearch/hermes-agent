"""Recover historical source identities without inventing provenance or age."""
import json
import re
from urllib.parse import urlsplit


def historical_source_ids(raw_context):
    """Return recoverable identities, or None when any source needs review."""
    try:
        context = json.loads(raw_context)
        if not isinstance(context, dict):
            return None
        sources = context.get('sources')
        if sources is None:
            # Old quote/reply artifacts predate the common sources envelope.
            sources = [{'id': context.get('tweet_id'), 'url': context.get('source_url')}]
        if not isinstance(sources, list) or not sources:
            return None
        recovered = set()
        for source in sources:
            if not isinstance(source, dict):
                return None
            identity, url = source.get('id'), source.get('url')
            if url:
                parsed = urlsplit(url)
                if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
                    return None
                if parsed.hostname in {'x.com', 'www.x.com', 'twitter.com', 'www.twitter.com'}:
                    match = re.fullmatch(r'/(?:[A-Za-z0-9_]{1,15}|i/web)/status/([0-9]+)', parsed.path)
                    if not match or (identity is not None and identity != match[1]):
                        return None
                    identity = match[1]
            if not isinstance(identity, str) or not identity.strip():
                return None
            recovered.add(identity)
        return recovered
    except (ValueError, TypeError):
        return None


def backfill_source_history(conn):
    """Idempotent, transactional; ambiguous history stays visible for review."""
    conn.execute('CREATE TABLE IF NOT EXISTS x_manager_history_review (artifact_id TEXT PRIMARY KEY, reason TEXT NOT NULL)')
    for artifact_id, raw_context, created_at in conn.execute('SELECT id, context, created_at FROM x_manager_artifacts ORDER BY created_at, id').fetchall():
        identities = historical_source_ids(raw_context)
        if identities is None:
            conn.execute('INSERT OR REPLACE INTO x_manager_history_review VALUES (?, ?)', (artifact_id, 'source identity unavailable; human review required'))
            continue
        conn.execute('DELETE FROM x_manager_history_review WHERE artifact_id=?', (artifact_id,))
        for identity in identities:
            conn.execute('INSERT OR IGNORE INTO x_manager_sources VALUES (?, ?, ?)', (identity, artifact_id, created_at))
