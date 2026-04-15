#!/usr/bin/env python3
"""
Title Sessions: Generate short conversation titles for CLI sessions.

This script automatically generates concise, descriptive titles for untitled
CLI sessions stored in ~/.hermes/state.db. Titles are 3-8 words and capture
the main topic of the conversation.

Usage:
    python title_sessions.py              # Title the most recent session
    python title_sessions.py --backfill    # Title sessions from last 4 days
    python title_sessions.py backfill       # Same as --backfill
"""

import sqlite3
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, List, Tuple


def get_db_path() -> str:
    """Get the path to the state.db file."""
    return os.path.expanduser("~/.hermes/state.db")


def get_first_user_message(conn: sqlite3.Connection, session_id: str) -> Optional[str]:
    """
    Get the first user message from a session.

    Args:
        conn: SQLite database connection
        session_id: Session ID to fetch message from

    Returns:
        First user message content, or None if not found
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT content FROM messages
        WHERE session_id = ? AND role = 'user'
        ORDER BY timestamp ASC
        LIMIT 1
        """,
        (session_id,)
    )
    result = cursor.fetchone()
    return result[0] if result else None


def generate_title(message: str) -> str:
    """
    Generate a short title from a user message using heuristics.

    Args:
        message: The user message to summarize

    Returns:
        A short title (3-8 words) capturing the topic
    """
    # Clean up the message
    # Remove workspace tags like [Workspace: /path/to/dir]
    import re
    message = re.sub(r'\[Workspace: [^\]]+\]\s*', '', message)
    
    # Remove common prefixes
    message = re.sub(r'^(Note:|WARNING:|INFO:)\s*', '', message)
    
    # Use a simple heuristic-based approach
    words = message.split()
    
    # Take first few meaningful words
    meaningful_words = []
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'has', 'been', 'have', 'will', 'with',
        'this', 'that', 'from', 'they', 'them', 'their', 'there', 'where', 'when',
        'which', 'while', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once'
    }
    
    for word in words:
        # Clean punctuation from word
        word = re.sub(r'[^\w]', '', word)
        
        # Skip very short words and common stop words
        if len(word) > 2 and word.lower() not in stop_words:
            meaningful_words.append(word)
        if len(meaningful_words) >= 6:
            break
    
    if not meaningful_words:
        # Fallback to first few words
        meaningful_words = [re.sub(r'[^\w]', '', w) for w in words[:5] if len(w) > 2]
    
    title = ' '.join(meaningful_words)
    
    # Capitalize first letter of each word
    title = ' '.join(word.capitalize() for word in title.split())
    
    # Truncate to reasonable length
    if len(title) > 50:
        title = title[:47] + '...'
    
    return title


def get_untitled_sessions(conn: sqlite3.Connection, backfill: bool = False) -> List[Tuple[str, str]]:
    """
    Get sessions that need titles.

    Args:
        conn: SQLite database connection
        backfill: If True, get sessions from last 4 days. If False, get only the most recent.

    Returns:
        List of (session_id, first_message) tuples
    """
    cursor = conn.cursor()

    if backfill:
        # Get sessions from last 4 days that need titles
        four_days_ago = (datetime.now() - timedelta(days=4)).timestamp()
        cursor.execute(
            """
            SELECT id FROM sessions
            WHERE (title IS NULL OR title = 'CLI Session')
            AND (title_source IS NULL OR title_source = 'auto')
            AND started_at >= ?
            ORDER BY started_at DESC
            """,
            (four_days_ago,)
        )
    else:
        # Get only the most recent session that needs a title
        cursor.execute(
            """
            SELECT id FROM sessions
            WHERE (title IS NULL OR title = 'CLI Session')
            AND (title_source IS NULL OR title_source = 'auto')
            ORDER BY started_at DESC
            LIMIT 1
            """
        )

    sessions = cursor.fetchall()
    results = []

    for (session_id,) in sessions:
        message = get_first_user_message(conn, session_id)
        if message:
            results.append((session_id, message))

    return results


def update_session_title(conn: sqlite3.Connection, session_id: str, title: str) -> bool:
    """
    Update a session's title and mark it as auto-generated.
    Handles duplicate titles by appending a counter.

    Args:
        conn: SQLite database connection
        session_id: Session ID to update
        title: New title to set

    Returns:
        True if successful, False otherwise
    """
    cursor = conn.cursor()
    
    # Check if title already exists
    cursor.execute(
        """
        SELECT COUNT(*) FROM sessions WHERE title = ? AND id != ?
        """,
        (title, session_id)
    )
    count = cursor.fetchone()[0]
    
    # If title exists, append a counter
    if count > 0:
        base_title = title
        counter = 2
        while True:
            new_title = f"{base_title} ({counter})"
            cursor.execute(
                """
                SELECT COUNT(*) FROM sessions WHERE title = ? AND id != ?
                """,
                (new_title, session_id)
            )
            if cursor.fetchone()[0] == 0:
                title = new_title
                break
            counter += 1
    
    try:
        cursor.execute(
            """
            UPDATE sessions
            SET title = ?, title_source = 'auto'
            WHERE id = ?
            """,
            (title, session_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"Error updating session {session_id}: {e}")
        return False


def main():
    """Main entry point."""
    # Parse arguments
    backfill = False
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--backfill', 'backfill']:
            backfill = True

    db_path = get_db_path()
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    
    try:
        sessions = get_untitled_sessions(conn, backfill=backfill)
        
        if not sessions:
            if backfill:
                print("No untitled sessions found from the last 4 days.")
            else:
                print("No untitled sessions found.")
            sys.exit(0)

        print(f"Found {len(sessions)} session(s) to title.")
        
        for session_id, message in sessions:
            print(f"\nProcessing session {session_id}...")
            print(f"First message: {message[:100]}...")
            
            title = generate_title(message)
            print(f"Generated title: {title}")
            
            if update_session_title(conn, session_id, title):
                print(f"✓ Updated session {session_id}")
            else:
                print(f"✗ Failed to update session {session_id}")

        print(f"\nCompleted: {len(sessions)} session(s) titled.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()