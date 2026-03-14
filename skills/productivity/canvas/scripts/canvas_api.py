#!/usr/bin/env python3
"""Canvas LMS API CLI for Hermes Agent.

A thin CLI wrapper around the Canvas REST API.
Authenticates using a personal access token from environment variables.

Usage:
  python canvas_api.py list_courses [--per-page N] [--enrollment-state STATE]
  python canvas_api.py list_assignments COURSE_ID [--per-page N] [--order-by FIELD]
  python canvas_api.py get_assignment COURSE_ID ASSIGNMENT_ID
  python canvas_api.py submit_assignment COURSE_ID ASSIGNMENT_ID --type TYPE [--body TEXT] [--url URL] [--file PATH]
  python canvas_api.py sync_assignments COURSE_ID [--per-page N]
  python canvas_api.py mark_done COURSE_ID ASSIGNMENT_ID [--notes TEXT]
  python canvas_api.py list_pending [COURSE_ID]
  python canvas_api.py list_done [COURSE_ID]
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

import requests


def _load_hermes_env_value(key: str) -> str:
    """Load a value from ~/.hermes/.env if not already in os.environ."""
    val = os.environ.get(key, "")
    if val:
        return val
    # Fallback: read directly from the .env file (python-dotenv may not be
    # installed or load_dotenv may not have run for this process).
    env_path = os.path.join(
        os.environ.get("HERMES_HOME", os.path.join(os.path.expanduser("~"), ".hermes")),
        ".env",
    )
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    if k.strip() == key:
                        return v.strip().strip("\"'")
    except FileNotFoundError:
        pass
    return ""


CANVAS_API_TOKEN = _load_hermes_env_value("CANVAS_API_TOKEN")
CANVAS_BASE_URL = _load_hermes_env_value("CANVAS_BASE_URL").rstrip("/")


def _check_config():
    """Validate required environment variables are set."""
    missing = []
    if not CANVAS_API_TOKEN:
        missing.append("CANVAS_API_TOKEN")
    if not CANVAS_BASE_URL:
        missing.append("CANVAS_BASE_URL")
    if missing:
        print(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Set them in ~/.hermes/.env or export them in your shell.\n"
            "See the canvas skill SKILL.md for setup instructions.",
            file=sys.stderr,
        )
        sys.exit(1)


def _headers():
    return {"Authorization": f"Bearer {CANVAS_API_TOKEN}"}


def _db_path():
    hermes_home = os.environ.get(
        "HERMES_HOME", os.path.join(os.path.expanduser("~"), ".hermes")
    )
    return os.path.join(hermes_home, "canvas_assignments.db")


def _get_db():
    """Return an open SQLite connection, initializing schema if needed."""
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER NOT NULL,
            course_id INTEGER NOT NULL,
            name TEXT,
            due_at TEXT,
            points_possible REAL,
            submission_types TEXT,
            html_url TEXT,
            last_synced TEXT NOT NULL,
            local_done INTEGER NOT NULL DEFAULT 0,
            done_at TEXT,
            done_notes TEXT,
            PRIMARY KEY (id, course_id)
        )
    """)
    conn.commit()
    return conn


def _is_google_assignment(a: dict) -> bool:
    """Check if an assignment uses Google Assignments (LTI)."""
    url = (a.get("external_tool_tag_attributes") or {}).get("url", "")
    return "external_tool" in a.get("submission_types", []) and "google" in url.lower()


def _google_assignments_url(a: dict) -> str:
    """Return the Google Assignments URL if applicable, else empty string."""
    if _is_google_assignment(a):
        return (a.get("external_tool_tag_attributes") or {}).get("url", "")
    return ""


def _paginated_get(url, params=None, max_items=200):
    """Fetch all pages up to max_items, following Canvas Link headers."""
    results = []
    while url and len(results) < max_items:
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        resp.raise_for_status()
        results.extend(resp.json())
        params = None  # params are included in the Link URL for subsequent pages
        url = None
        link = resp.headers.get("Link", "")
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
    return results[:max_items]


# =========================================================================
# Commands
# =========================================================================


def list_courses(args):
    """List enrolled courses."""
    _check_config()
    url = f"{CANVAS_BASE_URL}/api/v1/courses"
    params = {"per_page": args.per_page}
    if args.enrollment_state:
        params["enrollment_state"] = args.enrollment_state
    try:
        courses = _paginated_get(url, params)
    except requests.HTTPError as e:
        print(f"API error: {e.response.status_code} {e.response.text}", file=sys.stderr)
        sys.exit(1)
    output = [
        {
            "id": c["id"],
            "name": c.get("name", ""),
            "course_code": c.get("course_code", ""),
            "enrollment_term_id": c.get("enrollment_term_id"),
            "start_at": c.get("start_at"),
            "end_at": c.get("end_at"),
            "workflow_state": c.get("workflow_state", ""),
        }
        for c in courses
    ]
    print(json.dumps(output, indent=2))


def list_assignments(args):
    """List assignments for a course."""
    _check_config()
    url = f"{CANVAS_BASE_URL}/api/v1/courses/{args.course_id}/assignments"
    params = {"per_page": args.per_page}
    if args.order_by:
        params["order_by"] = args.order_by
    try:
        assignments = _paginated_get(url, params)
    except requests.HTTPError as e:
        print(f"API error: {e.response.status_code} {e.response.text}", file=sys.stderr)
        sys.exit(1)
    output = [
        {
            "id": a["id"],
            "name": a.get("name", ""),
            "description": (a.get("description") or "")[:500],
            "due_at": a.get("due_at"),
            "points_possible": a.get("points_possible"),
            "submission_types": a.get("submission_types", []),
            "html_url": a.get("html_url", ""),
            "course_id": a.get("course_id"),
        }
        for a in assignments
    ]
    print(json.dumps(output, indent=2))


def get_assignment(args):
    """Fetch a single assignment with full details, attached files, and links."""
    _check_config()
    url = f"{CANVAS_BASE_URL}/api/v1/courses/{args.course_id}/assignments/{args.assignment_id}"
    try:
        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(
            f"API error: {e.response.status_code} {e.response.text}",
            file=sys.stderr,
        )
        sys.exit(1)
    a = resp.json()

    raw_attachments = a.get("attachments") or []
    attachments = [
        {
            "display_name": att.get("display_name", ""),
            "url": att.get("url", ""),
            "content_type": att.get("content-type", ""),
            "size": att.get("size"),
        }
        for att in raw_attachments
    ]

    output = {
        "id": a["id"],
        "name": a.get("name", ""),
        "course_id": a.get("course_id"),
        "description": a.get("description") or "",
        "due_at": a.get("due_at"),
        "points_possible": a.get("points_possible"),
        "submission_types": a.get("submission_types", []),
        "html_url": a.get("html_url", ""),
        "attachments": attachments,
        "external_tool_url": (a.get("external_tool_tag_attributes") or {}).get(
            "url", ""
        ),
        "google_assignments": _is_google_assignment(a),
        "google_assignments_url": _google_assignments_url(a),
        "locked_for_user": a.get("locked_for_user", False),
        "lock_explanation": a.get("lock_explanation", ""),
    }
    print(json.dumps(output, indent=2))


def submit_assignment(args):
    """Submit an assignment (text, URL, or file upload)."""
    _check_config()

    # --- Validate argument combinations ---
    if args.type == "online_url" and not args.url:
        print(
            "Error: --url is required for submission type 'online_url'",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.type == "online_upload" and not args.file:
        print(
            "Error: --file is required for submission type 'online_upload'",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.type == "online_upload" and args.file and not os.path.isfile(args.file):
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    # --- Fetch assignment to verify submission type is allowed ---
    assign_url = (
        f"{CANVAS_BASE_URL}/api/v1/courses/{args.course_id}"
        f"/assignments/{args.assignment_id}"
    )
    try:
        resp = requests.get(assign_url, headers=_headers(), timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(
            f"API error fetching assignment: {e.response.status_code} {e.response.text}",
            file=sys.stderr,
        )
        sys.exit(1)
    a = resp.json()
    allowed_types = a.get("submission_types", [])

    # --- Detect Google Assignments (LTI) ---
    if _is_google_assignment(a):
        google_url = _google_assignments_url(a)
        print(
            json.dumps(
                {
                    "error": "google_assignments",
                    "message": (
                        "This assignment uses Google Assignments (an LTI tool). "
                        "Submission must be done through the Google Assignments "
                        "interface, not through the Canvas API."
                    ),
                    "google_assignments_url": google_url,
                    "html_url": a.get("html_url", ""),
                },
                indent=2,
            )
        )
        sys.exit(1)

    # --- Check requested type is allowed ---
    if args.type not in allowed_types:
        print(
            json.dumps(
                {
                    "error": "submission_type_not_allowed",
                    "message": (
                        f"Submission type '{args.type}' is not allowed for this "
                        f"assignment."
                    ),
                    "allowed_types": allowed_types,
                },
                indent=2,
            )
        )
        sys.exit(1)

    # --- Assignments with no_submission / not_graded ---
    if "no_submission" in allowed_types or "not_graded" in allowed_types:
        print(
            json.dumps(
                {
                    "error": "no_submission_required",
                    "message": (
                        "This assignment has no submission button. "
                        "Use 'mark_done' to record it locally instead."
                    ),
                    "allowed_types": allowed_types,
                },
                indent=2,
            )
        )
        sys.exit(1)

    # --- Dry-run: preview what would be submitted ---
    if args.dry_run:
        preview = {
            "dry_run": True,
            "assignment_name": a.get("name", ""),
            "course_id": args.course_id,
            "assignment_id": args.assignment_id,
            "submission_type": args.type,
            "body": args.body if args.type == "online_text_entry" else None,
            "url": args.url if args.type == "online_url" else None,
            "file": args.file if args.type == "online_upload" else None,
            "file_name": (
                os.path.basename(args.file)
                if args.type == "online_upload" and args.file
                else None
            ),
            "file_size_bytes": (
                os.path.getsize(args.file)
                if args.type == "online_upload" and args.file
                else None
            ),
            "html_url": a.get("html_url", ""),
        }
        print(json.dumps(preview, indent=2))
        return

    # --- Build submission payload ---
    if args.type == "online_upload":
        # Step 1: Request upload slot
        file_name = os.path.basename(args.file)
        file_size = os.path.getsize(args.file)
        slot_url = (
            f"{CANVAS_BASE_URL}/api/v1/courses/{args.course_id}"
            f"/assignments/{args.assignment_id}/submissions/self/files"
        )
        try:
            slot_resp = requests.post(
                slot_url,
                headers=_headers(),
                json={"name": file_name, "size": file_size},
                timeout=30,
            )
            slot_resp.raise_for_status()
        except requests.HTTPError as e:
            print(
                f"API error requesting upload slot: "
                f"{e.response.status_code} {e.response.text}",
                file=sys.stderr,
            )
            sys.exit(1)
        slot = slot_resp.json()
        upload_url = slot["upload_url"]
        upload_params = slot.get("upload_params", {})

        # Step 2: Upload the file (multipart)
        try:
            with open(args.file, "rb") as fh:
                files_payload = {"file": (file_name, fh)}
                upload_resp = requests.post(
                    upload_url,
                    data=upload_params,
                    files=files_payload,
                    timeout=120,
                    allow_redirects=True,
                )
                upload_resp.raise_for_status()
        except requests.HTTPError as e:
            print(
                f"API error uploading file: "
                f"{e.response.status_code} {e.response.text}",
                file=sys.stderr,
            )
            sys.exit(1)
        upload_result = upload_resp.json()
        file_id = upload_result.get("id")
        if not file_id:
            print(
                "Error: file upload did not return a file ID", file=sys.stderr
            )
            sys.exit(1)

        # Step 3: Submit with the uploaded file_id
        submission_payload = {
            "submission_type": "online_upload",
            "file_ids": [file_id],
        }
    elif args.type == "online_text_entry":
        submission_payload = {
            "submission_type": "online_text_entry",
            "body": args.body,
        }
    elif args.type == "online_url":
        submission_payload = {
            "submission_type": "online_url",
            "url": args.url,
        }

    # --- Final submission POST ---
    sub_url = (
        f"{CANVAS_BASE_URL}/api/v1/courses/{args.course_id}"
        f"/assignments/{args.assignment_id}/submissions"
    )
    try:
        sub_resp = requests.post(
            sub_url,
            headers=_headers(),
            json={"submission": submission_payload},
            timeout=30,
        )
        sub_resp.raise_for_status()
    except requests.HTTPError as e:
        print(
            f"API error submitting: {e.response.status_code} {e.response.text}",
            file=sys.stderr,
        )
        sys.exit(1)
    s = sub_resp.json()
    print(
        json.dumps(
            {
                "success": True,
                "submission_id": s.get("id"),
                "submitted_at": s.get("submitted_at"),
                "workflow_state": s.get("workflow_state", ""),
                "submission_type": s.get("submission_type", ""),
            },
            indent=2,
        )
    )


def sync_assignments(args):
    """Fetch assignments from Canvas and upsert into local SQLite DB."""
    _check_config()
    url = f"{CANVAS_BASE_URL}/api/v1/courses/{args.course_id}/assignments"
    params = {"per_page": args.per_page}
    try:
        assignments = _paginated_get(url, params)
    except requests.HTTPError as e:
        print(
            f"API error: {e.response.status_code} {e.response.text}",
            file=sys.stderr,
        )
        sys.exit(1)
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_db()
    count = 0
    for a in assignments:
        conn.execute(
            """
            INSERT INTO assignments (id, course_id, name, due_at, points_possible,
                submission_types, html_url, last_synced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id, course_id) DO UPDATE SET
                name=excluded.name,
                due_at=excluded.due_at,
                points_possible=excluded.points_possible,
                submission_types=excluded.submission_types,
                html_url=excluded.html_url,
                last_synced=excluded.last_synced
            """,
            (
                a["id"],
                a.get("course_id", args.course_id),
                a.get("name", ""),
                a.get("due_at"),
                a.get("points_possible"),
                json.dumps(a.get("submission_types", [])),
                a.get("html_url", ""),
                now,
            ),
        )
        count += 1
    conn.commit()
    conn.close()
    print(
        json.dumps(
            {"synced": count, "course_id": int(args.course_id), "synced_at": now},
            indent=2,
        )
    )


def mark_done(args):
    """Mark an assignment as done in the local DB."""
    conn = _get_db()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "UPDATE assignments SET local_done=1, done_at=?, done_notes=? "
        "WHERE id=? AND course_id=?",
        (now, args.notes or "", int(args.assignment_id), int(args.course_id)),
    )
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        print(
            json.dumps(
                {
                    "error": "not_found",
                    "message": (
                        "Assignment not found in local DB. "
                        "Run sync_assignments first."
                    ),
                }
            )
        )
        sys.exit(1)
    print(
        json.dumps(
            {
                "success": True,
                "assignment_id": int(args.assignment_id),
                "done_at": now,
            },
            indent=2,
        )
    )


def list_pending(args):
    """List assignments not marked done from local DB."""
    conn = _get_db()
    query = "SELECT * FROM assignments WHERE local_done=0"
    params = []
    if args.course_id:
        query += " AND course_id=?"
        params.append(int(args.course_id))
    query += " ORDER BY due_at ASC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    output = [dict(r) for r in rows]
    for row in output:
        row["submission_types"] = json.loads(row["submission_types"] or "[]")
    print(json.dumps(output, indent=2))


def list_done(args):
    """List assignments marked done from local DB."""
    conn = _get_db()
    query = "SELECT * FROM assignments WHERE local_done=1"
    params = []
    if args.course_id:
        query += " AND course_id=?"
        params.append(int(args.course_id))
    query += " ORDER BY done_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    output = [dict(r) for r in rows]
    for row in output:
        row["submission_types"] = json.loads(row["submission_types"] or "[]")
    print(json.dumps(output, indent=2))


# =========================================================================
# CLI parser
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Canvas LMS API CLI for Hermes Agent"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- list_courses ---
    p = sub.add_parser("list_courses", help="List enrolled courses")
    p.add_argument("--per-page", type=int, default=50, help="Results per page (default 50)")
    p.add_argument(
        "--enrollment-state",
        default="",
        help="Filter by enrollment state (active, invited_or_pending, completed)",
    )
    p.set_defaults(func=list_courses)

    # --- list_assignments ---
    p = sub.add_parser("list_assignments", help="List assignments for a course")
    p.add_argument("course_id", help="Canvas course ID")
    p.add_argument("--per-page", type=int, default=50, help="Results per page (default 50)")
    p.add_argument(
        "--order-by",
        default="",
        help="Order by field (due_at, name, position)",
    )
    p.set_defaults(func=list_assignments)

    # --- get_assignment ---
    p = sub.add_parser("get_assignment", help="Fetch full details for a single assignment")
    p.add_argument("course_id", help="Canvas course ID")
    p.add_argument("assignment_id", help="Canvas assignment ID")
    p.set_defaults(func=get_assignment)

    # --- submit_assignment ---
    p = sub.add_parser("submit_assignment", help="Submit an assignment")
    p.add_argument("course_id", help="Canvas course ID")
    p.add_argument("assignment_id", help="Canvas assignment ID")
    p.add_argument(
        "--type",
        required=True,
        choices=["online_text_entry", "online_url", "online_upload"],
        help="Submission type",
    )
    p.add_argument(
        "--body",
        default="Assignment completed",
        help="Text body for online_text_entry submissions (default: 'Assignment completed')",
    )
    p.add_argument(
        "--url",
        default=None,
        help="URL for online_url submissions",
    )
    p.add_argument(
        "--file",
        default=None,
        help="Local file path for online_upload submissions",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be submitted without writing to Canvas",
    )
    p.set_defaults(func=submit_assignment)

    # --- sync_assignments ---
    p = sub.add_parser("sync_assignments", help="Sync assignments to local DB")
    p.add_argument("course_id", help="Canvas course ID")
    p.add_argument("--per-page", type=int, default=50, help="Results per page (default 50)")
    p.set_defaults(func=sync_assignments)

    # --- mark_done ---
    p = sub.add_parser("mark_done", help="Mark an assignment as done locally")
    p.add_argument("course_id", help="Canvas course ID")
    p.add_argument("assignment_id", help="Canvas assignment ID")
    p.add_argument("--notes", default="", help="Optional completion notes")
    p.set_defaults(func=mark_done)

    # --- list_pending ---
    p = sub.add_parser("list_pending", help="List pending (not done) assignments from local DB")
    p.add_argument("course_id", nargs="?", default=None, help="Optional course ID filter")
    p.set_defaults(func=list_pending)

    # --- list_done ---
    p = sub.add_parser("list_done", help="List completed assignments from local DB")
    p.add_argument("course_id", nargs="?", default=None, help="Optional course ID filter")
    p.set_defaults(func=list_done)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
