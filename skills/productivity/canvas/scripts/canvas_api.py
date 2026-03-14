#!/usr/bin/env python3
"""Canvas LMS API CLI for Hermes Agent.

A thin CLI wrapper around the Canvas REST API.
Authenticates using a personal access token from environment variables.

Usage:
  python canvas_api.py list_courses [--per-page N] [--enrollment-state STATE]
  python canvas_api.py list_assignments COURSE_ID [--per-page N] [--order-by FIELD]
"""

import argparse
import json
import os
import sys

import requests

CANVAS_API_TOKEN = os.environ.get("CANVAS_API_TOKEN", "")
CANVAS_BASE_URL = os.environ.get("CANVAS_BASE_URL", "").rstrip("/")


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
