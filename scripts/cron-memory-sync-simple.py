#!/usr/bin/env python3
"""
Cron memory/social sync - counts-only output, no secrets

Usage:
    python scripts/cron-memory-sync-simple.py

This script implements the Bitwarden-safe cron sync pattern:
- Scans Hermes state.db for session/message counts
- Extracts confirmed public X posts from lm-twitterer activity
- Stores URL-only artifacts in Ebbinghaus (idempotent)
- Saves abstracted policy facts
- Runs two-tier secret/path guards
- Verifies post-write for violations
- Reports counts only (no raw candidate text)

Reference: ebbinghaus-memory:references/ebbinghaus-schema.md
"""
import sqlite3
import json
import re
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

# Configuration
STATE_DB = Path.home() / ".hermes" / "state.db"
LM_TWITTERER_ACTIVITY = Path.home() / ".hermes" / "lm-twitterer" / "activity.jsonl"
EBBINGHAUS_DB = Path.home() / ".hermes" / "ebbinghaus_memory.db"  # Canonical path

# Secret markers for broad exclusion
SECRET_PATTERNS = [
    r'(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|APIKEY|AUTH_TOKEN|CT0|PRIVATE_KEY)',
    r'(?:Bearer\s+[A-Za-z0-9_\-=]+)',
    r'(?:sk-[A-Za-z0-9]{20,})',
    r'(?:ghp_[A-Za-z0-9]{36})',
    r'(?:\.env\b)',
    r'(?:oauth_token[=:])',
    r'(?:session_token[=:])',
]

# Path markers (Windows-safe)
bs = chr(92)  # backslash
PATH_MARKERS = [
    'C:' + bs + 'Users',
    'C:/Users',
    '/home/',
    bs + 'Users' + bs,
    bs + '.hermes' + bs,
]

# Compile broad exclusion regex for candidate intake. This intentionally flags
# generic words such as "secret" and ".env" so raw social/X text is excluded
# before it can be stored.
BROAD_EXCLUSION = re.compile('|'.join(SECRET_PATTERNS), re.IGNORECASE)

# Narrow raw-value guard for post-write verification/remediation. Policy facts
# may safely mention categories such as credentials, tokens, or .env contents;
# verification should only flag actual value-looking material or local paths.
RAW_VALUE_PATTERNS = [
    r'(?:Bearer\s+[A-Za-z0-9_\-=]{10,})',
    r'(?:sk-[A-Za-z0-9]{20,})',
    r'(?:ghp_[A-Za-z0-9]{36})',
    r'(?:xox[baprs]-[A-Za-z0-9-]{10,})',
    r'(?:\b(?:[A-Z0-9]+_)*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|APIKEY|AUTH_TOKEN|CT0|PRIVATE_KEY)(?:_[A-Z0-9]+)*\b\s*[=:]\s*\S{6,})',
    r'(?:oauth_token[=:]\S{6,})',
    r'(?:session_token[=:]\S{6,})',
]
RAW_VALUE_GUARD = re.compile('|'.join(RAW_VALUE_PATTERNS), re.IGNORECASE)

def check_intake_guard(text):
    """Broadly exclude raw candidate text before storage."""
    if BROAD_EXCLUSION.search(text):
        return True
    for marker in PATH_MARKERS:
        if marker in text:
            return True
    return False

def check_scoped_guard(text):
    """Check saved content for raw value-looking material or local paths."""
    if RAW_VALUE_GUARD.search(text):
        return True
    for marker in PATH_MARKERS:
        if marker in text:
            return True
    return False

def _encoded_payload(content, tags):
    """Build the required encoded/cues fields for direct SQLite inserts."""
    tokens = re.findall(r"[A-Za-z0-9_ぁ-んァ-ン一-龥]{2,}", " ".join([content, tags]).lower())
    counts = Counter(tokens)
    cues = [token for token, _ in counts.most_common(12)]
    encoded = {
        "version": 1,
        "kind": "cue_encoding",
        "summary": content[:280],
        "cue_vector": {token: counts[token] for token in cues},
        "cues": cues,
        "length": len(content),
    }
    return json.dumps(encoded, ensure_ascii=False), " ".join(cues)

def extract_x_posts():
    """Extract confirmed public X posts from lm-twitterer activity"""
    if not LM_TWITTERER_ACTIVITY.exists():
        return [], 0, 0
    
    valid_posts = []
    total_records = 0
    excluded = 0
    
    with open(LM_TWITTERER_ACTIVITY, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total_records += 1
            try:
                record = json.loads(line)
            except:
                excluded += 1
                continue
            
            # Only confirmed public posts
            if record.get('action') != 'post':
                excluded += 1
                continue
            if not record.get('ok'):
                excluded += 1
                continue
            if record.get('dry_run'):
                excluded += 1
                continue
            if not record.get('posted'):
                excluded += 1
                continue
            
            url = record.get('url', '')
            if not url or not url.startswith('https://x.com/i/web/status/'):
                excluded += 1
                continue
            
            valid_posts.append({
                'url': url,
                'text': record.get('tweet_text', ''),
                'timestamp': record.get('timestamp', ''),
            })
    
    return valid_posts, total_records, excluded

def get_session_counts():
    """Get session and message counts from state.db"""
    if not STATE_DB.exists():
        return 0, 0
    
    conn = sqlite3.connect(str(STATE_DB))
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        message_count = cursor.fetchone()[0]
    except:
        session_count = 0
        message_count = 0
    finally:
        conn.close()
    
    return session_count, message_count

def save_to_ebbinghaus(posts):
    """Save URL-only X artifacts to Ebbinghaus (idempotent)"""
    if not EBBINGHAUS_DB.exists():
        return 0, 0
    
    conn = sqlite3.connect(str(EBBINGHAUS_DB))
    cursor = conn.cursor()
    
    saved = 0
    excluded = 0
    
    for post in posts:
        # Store URL only (safer than full text)
        content = f"X post: {post['url']}"
        
        if check_scoped_guard(content) or check_intake_guard(post['text']):
            excluded += 1
            continue
        
        # Check for existing content (idempotency)
        cursor.execute("SELECT memory_id FROM memories WHERE content = ?", (content,))
        if cursor.fetchone():
            continue  # Already exists
        
        try:
            now = time.time()
            encoded, cues = _encoded_payload(content, 'x-post-artifact,lm-twitterer')
            cursor.execute("""
                INSERT INTO memories (
                    content, encoded, cues, tags, source, salience, valence, strength,
                    session_id, created_at, updated_at, last_rehearsed_at
                )
                VALUES (?, ?, ?, ?, ?, 0.5, 0.0, 1.5, '', ?, ?, ?)
            """, (content, encoded, cues, 'x-post-artifact,lm-twitterer', 'lm-twitterer', now, now, now))
            if cursor.rowcount > 0:
                saved += 1
        except Exception as e:
            excluded += 1
    
    conn.commit()
    conn.close()
    return saved, excluded

def save_policy_facts():
    """Save abstracted policy facts (idempotent)"""
    if not EBBINGHAUS_DB.exists():
        return 0
    
    policy_facts = [
        "User manages environment variables in Bitwarden.",
        "User prefers AES-256-GCM memory vault keys escrowed in Bitwarden.",
        "Memory/social sync excludes secrets: passwords, API keys, tokens, paths, .env contents.",
        "X posts stored as URL-only artifacts for safety.",
        "Cron sync reports counts only, no raw candidate text.",
    ]
    
    conn = sqlite3.connect(str(EBBINGHAUS_DB))
    cursor = conn.cursor()
    
    saved = 0
    for fact in policy_facts:
        # Check for existing content
        cursor.execute("SELECT memory_id FROM memories WHERE content = ?", (fact,))
        if cursor.fetchone():
            continue  # Already exists
        
        try:
            now = time.time()
            encoded, cues = _encoded_payload(fact, 'policy-fact,cron-sync')
            cursor.execute("""
                INSERT INTO memories (
                    content, encoded, cues, tags, source, salience, valence, strength,
                    session_id, created_at, updated_at, last_rehearsed_at
                )
                VALUES (?, ?, ?, ?, ?, 0.8, 0.0, 1.8, '', ?, ?, ?)
            """, (fact, encoded, cues, 'policy-fact,cron-sync', 'cron-sync', now, now, now))
            if cursor.rowcount > 0:
                saved += 1
        except:
            pass
    
    conn.commit()
    conn.close()
    return saved

def verify_post_write():
    """Verify no scoped guard violations in newly saved rows (last hour)"""
    if not EBBINGHAUS_DB.exists():
        return 0, 0
    
    conn = sqlite3.connect(str(EBBINGHAUS_DB))
    cursor = conn.cursor()
    
    one_hour_ago = time.time() - 3600
    
    try:
        cursor.execute("""
            SELECT memory_id, content FROM memories 
            WHERE created_at >= ?
        """, (one_hour_ago,))
        rows = cursor.fetchall()
        
        violations = 0
        for row_id, content in rows:
            if check_scoped_guard(content):
                violations += 1
        
        return len(rows), violations
    finally:
        conn.close()

# Main execution
if __name__ == '__main__':
    print("Starting cron memory/social sync...", flush=True)
    
    # Get session counts
    session_count, message_count = get_session_counts()
    
    # Extract X posts
    x_posts, x_total, x_excluded_intake = extract_x_posts()
    valid_x_count = len(x_posts)
    
    # Save X artifacts (URL-only)
    x_saved, x_excluded_guard = save_to_ebbinghaus(x_posts)
    
    # Save policy facts
    policy_saved = save_policy_facts()
    
    # Post-write verification
    verified_rows, violations = verify_post_write()
    
    # Calculate totals
    total_synced = session_count + message_count + x_total
    total_saved = x_saved + policy_saved
    total_excluded = x_excluded_intake + x_excluded_guard
    
    # Determine residual risk
    if violations > 0:
        residual_risk = "high"
    elif total_excluded > 100:
        residual_risk = "medium"
    else:
        residual_risk = "low"
    
    # Report counts only (no raw data)
    result = {
        "synced_sessions": session_count,
        "synced_messages": message_count,
        "synced_x_records": x_total,
        "valid_x_posts": valid_x_count,
        "x_artifacts_saved": x_saved,
        "policy_facts_saved": policy_saved,
        "total_saved": total_saved,
        "excluded_intake": x_excluded_intake,
        "excluded_guard": x_excluded_guard,
        "total_excluded": total_excluded,
        "verified_rows": verified_rows,
        "guard_violations": violations,
        "residual_risk": residual_risk,
        "timestamp": datetime.now().isoformat(),
    }
    
    print(json.dumps(result, indent=2, ensure_ascii=False))