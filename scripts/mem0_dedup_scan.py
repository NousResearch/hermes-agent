#!/usr/bin/env python3
"""Monthly mem0 near-duplicate scanner — reports (and optionally consolidates) near-duplicates.

Scans a Qdrant collection for pairs of memories belonging to the same
user_id that have cosine similarity >= 0.92. Prints a report to stdout.

Without --consolidate: reports only (does NOT delete anything).
With --consolidate (alone): dry-run — prints what WOULD be deleted but does nothing.
With --consolidate --yes: actually deletes the identified duplicates via Qdrant API.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MEM0_CONFIG = Path.home() / ".hermes" / "mem0.json"
SIMILARITY_THRESHOLD = 0.92
SCROLL_LIMIT = 100

# Module-level constants — overridden by CLI flags in main()
QDRANT_URL = "http://localhost:6333"
COLLECTION = "hermes_memories"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_mem0_config():
    """Return parsed mem0.json as a dict, or {} if unavailable."""
    try:
        with open(MEM0_CONFIG) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not read {MEM0_CONFIG}: {exc}", file=sys.stderr)
        return {}


def _config_user_id(cfg: dict) -> str | None:
    return cfg.get("user_id") or None


def _config_collection(cfg: dict) -> str | None:
    try:
        return cfg["oss"]["vector_store"]["config"]["collection_name"] or None
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def api_get(path, timeout=60):
    req = urllib.request.Request(f"{QDRANT_URL}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def api_post(path, body, timeout=120):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{QDRANT_URL}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def api_delete(path, body=None, timeout=30):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        f"{QDRANT_URL}{path}",
        data=data,
        method="DELETE",
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    """Cosine similarity between two equal-length lists of floats."""
    dot = 0.0
    mag_a = 0.0
    mag_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        mag_a += x * x
        mag_b += y * y
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / ((mag_a ** 0.5) * (mag_b ** 0.5))


# ---------------------------------------------------------------------------
# Collection fetching
# ---------------------------------------------------------------------------

def scroll_all_points():
    """Fetch all points from the collection with vectors and payloads."""
    points = []
    offset = None

    while True:
        body = {
            "limit": SCROLL_LIMIT,
            "with_vector": True,
            "with_payload": True,
        }
        if offset is not None:
            body["offset"] = offset

        try:
            result = api_post(f"/collections/{COLLECTION}/points/scroll", body)
        except urllib.error.URLError as exc:
            print(f"[ERROR] Qdrant scroll failed: {exc}", file=sys.stderr)
            sys.exit(1)

        batch = result.get("result", {}).get("points", [])
        points.extend(batch)

        next_offset = result.get("result", {}).get("next_page_offset")
        if next_offset is None or not batch:
            break
        offset = next_offset

    return points


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def find_duplicates(points, target_user_id):
    """Return list of (score, point_a, point_b) tuples for near-duplicate pairs."""
    # Filter to target user and points that have vectors
    eligible = []
    for p in points:
        payload = p.get("payload") or {}
        vec = p.get("vector")
        if not isinstance(vec, list) or not vec:
            continue
        # mem0 stores user_id directly in payload
        if payload.get("user_id") != target_user_id:
            continue
        eligible.append(p)

    print(
        f"Comparing {len(eligible)} points for user '{target_user_id}' "
        f"({len(points)} total in collection)...",
        file=sys.stderr,
    )

    pairs = []
    n = len(eligible)
    for i in range(n):
        for j in range(i + 1, n):
            a = eligible[i]
            b = eligible[j]
            score = cosine_similarity(a["vector"], b["vector"])
            if score >= SIMILARITY_THRESHOLD:
                pairs.append((score, a, b))

    # Sort highest similarity first
    pairs.sort(key=lambda t: t[0], reverse=True)
    return pairs, len(eligible)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_pairs(pairs):
    """Group near-duplicate pairs WITHOUT transitive union-find chaining.

    We deliberately do NOT use union-find here because transitivity creates
    unsafe groups: if A~B and B~C but A and C have similarity 0.4 (below
    threshold), merging all three into one group would cause A or C to be
    deleted even though they are not duplicates of each other.

    Instead, each directly-similar pair becomes its own two-member group.
    If the same ID appears in multiple pairs it will appear in multiple groups,
    and the consolidation step will handle it safely pair-by-pair, always
    keeping the higher-ranked member of each direct pair.
    """
    point_map = {}
    edge_scores = {}
    # Each pair becomes an independent two-member group keyed by a frozenset
    pair_groups = []

    for score, a, b in pairs:
        aid, bid = str(a["id"]), str(b["id"])
        point_map[aid] = a
        point_map[bid] = b
        edge_scores[(aid, bid)] = score
        pair_groups.append([aid, bid])

    return [
        {
            "members": pids,
            "points": {pid: point_map[pid] for pid in pids},
            "edge_scores": {(a, b): s for (a, b), s in edge_scores.items() if a in pids and b in pids},
        }
        for pids in pair_groups
    ]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def extract_text(point):
    """Pull the memory text from a point's payload."""
    payload = point.get("payload") or {}
    # mem0 stores the text in 'data' field
    return (
        payload.get("data")
        or payload.get("text")
        or payload.get("memory")
        or payload.get("content")
        or "<no text in payload>"
    )


def print_report(groups, pairs, total_memories):
    print()
    print("=" * 72)
    print("  mem0 Near-Duplicate Memory Report")
    print(f"  Collection: {COLLECTION}  |  Threshold: {SIMILARITY_THRESHOLD}")
    print("=" * 72)

    if not groups:
        print("\nNo near-duplicate pairs found.")
    else:
        for idx, group in enumerate(groups, 1):
            members = group["members"]
            edge_scores = group["edge_scores"]
            points = group["points"]

            # Find max score within group for display
            group_edges = [
                (s, a, b) for (a, b), s in edge_scores.items()
                if a in members and b in members
            ]
            max_score = max((s for s, _, _ in group_edges), default=0.0)

            print(f"\n--- Group {idx} (max similarity: {max_score:.4f}) ---")
            for pid in members:
                pt = points[pid]
                text = extract_text(pt)
                print(f"  ID:   {pid}")
                print(f"  Text: {text[:200]}{'...' if len(text) > 200 else ''}")
                print()

            # Show pairwise scores within the group
            if group_edges:
                print("  Pairwise similarities:")
                for score, aid, bid in sorted(group_edges, reverse=True):
                    print(f"    {aid} <-> {bid}: {score:.4f}")

    print()
    print("=" * 72)
    print(f"  Summary: Found {len(groups)} duplicate group(s) across {total_memories} total memories")
    print(f"  (Examined {len(pairs)} near-duplicate pair(s) at threshold >= {SIMILARITY_THRESHOLD})")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Consolidation (--consolidate flag)
# ---------------------------------------------------------------------------

def pick_keeper(group: dict) -> str:
    """Choose which point to keep in a duplicate group.

    Strategy: prefer the point with the longest text; break ties by most
    recently updated (latest updated_at timestamp string, lexicographic).
    """
    points = group["points"]
    members = group["members"]

    def rank(pid):
        pt = points[pid]
        payload = pt.get("payload") or {}
        text = extract_text(pt)
        updated_at = payload.get("updated_at", "")
        return (len(text), updated_at)

    return max(members, key=rank)


def delete_points(point_ids: list) -> bool:
    """Delete a list of Qdrant points by ID. Returns True on success."""
    if not point_ids:
        return True
    body = {"points": point_ids}
    try:
        result = api_post(f"/collections/{COLLECTION}/points/delete", body)
        status = result.get("result", {}).get("status", "unknown")
        return status == "acknowledged"
    except urllib.error.URLError as exc:
        print(f"[ERROR] Qdrant delete failed: {exc}", file=sys.stderr)
        return False


def consolidate_groups(groups: list, dry_run: bool = True) -> tuple[int, int]:
    """Keep the best memory per group and delete the rest.

    When dry_run=True (the default), only prints what WOULD be deleted without
    touching Qdrant. Pass dry_run=False (requires --yes on the CLI) to actually
    execute the deletes.

    Returns (kept_count, deleted_count).
    """
    kept = 0
    deleted = 0

    for idx, group in enumerate(groups, 1):
        members = group["members"]
        keeper_id = pick_keeper(group)
        to_delete = [pid for pid in members if pid != keeper_id]

        keeper_text = extract_text(group["points"][keeper_id])
        prefix = "[DRY-RUN] " if dry_run else ""
        print(f"\n[GROUP {idx}] {prefix}Keeping {keeper_id}")
        print(f"  Text: {keeper_text[:160]}{'...' if len(keeper_text) > 160 else ''}")
        print(f"  {prefix}Would delete {len(to_delete)} duplicate(s): {to_delete}")

        if dry_run:
            kept += 1
            deleted += len(to_delete)
        else:
            ok = delete_points(to_delete)
            if ok:
                kept += 1
                deleted += len(to_delete)
                print(f"  [OK] Deleted {len(to_delete)} point(s).")
            else:
                print(f"  [FAIL] Delete did not succeed for group {idx}.", file=sys.stderr)

    return kept, deleted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    cfg = _load_mem0_config()
    cfg_user_id = _config_user_id(cfg)
    cfg_collection = _config_collection(cfg) or "hermes_memories"

    parser = argparse.ArgumentParser(
        description=(
            "Scan a mem0 Qdrant collection for near-duplicate memory pairs. "
            "By default reports only. Use --consolidate to auto-delete duplicates."
        )
    )
    parser.add_argument(
        "--user",
        default=cfg_user_id,
        help=(
            "user_id to filter on. If omitted and not set in mem0.json, "
            "all user_ids found in the collection are scanned."
        ),
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant base URL (default: QDRANT_URL env var, or http://localhost:6333).",
    )
    parser.add_argument(
        "--collection",
        default=cfg_collection,
        help=(
            f"Collection name to scan "
            f"(default: from mem0.json oss.vector_store.config.collection_name, "
            f"or 'hermes_memories')."
        ),
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help=(
            "After finding duplicate groups, show what WOULD be kept/deleted "
            "(dry-run by default). Add --yes to actually execute the deletes."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Required together with --consolidate to actually execute deletes. "
            "Without this flag, --consolidate only prints a dry-run preview."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold for duplicate detection (default: {SIMILARITY_THRESHOLD}).",
    )
    args = parser.parse_args()

    # --yes without --consolidate is meaningless; warn and exit
    if args.yes and not args.consolidate:
        print("[ERROR] --yes requires --consolidate to be specified.", file=sys.stderr)
        sys.exit(1)

    dry_run = args.consolidate and not args.yes

    # Wire CLI flags into module-level constants used by the rest of the script
    globals()["QDRANT_URL"] = args.qdrant_url
    globals()["COLLECTION"] = args.collection
    if args.threshold != SIMILARITY_THRESHOLD:
        globals()["SIMILARITY_THRESHOLD"] = args.threshold

    print(f"Qdrant URL:  {args.qdrant_url}", file=sys.stderr)
    print(f"Collection:  {args.collection}", file=sys.stderr)
    if args.user:
        print(f"User filter: {args.user!r}", file=sys.stderr)
    else:
        print("User filter: (all users in collection)", file=sys.stderr)

    if args.consolidate:
        if dry_run:
            print("[CONSOLIDATE DRY-RUN] Will show what would be deleted. Re-run with --yes to execute.", file=sys.stderr)
        else:
            print("[CONSOLIDATE MODE] Duplicates will be permanently deleted.", file=sys.stderr)

    # Verify Qdrant is reachable — root endpoint returns JSON
    try:
        info = api_get("/")
        version = info.get("version", "unknown")
        print(f"Qdrant {version} is up.", file=sys.stderr)
    except urllib.error.URLError as exc:
        print(f"[ERROR] Cannot reach Qdrant at {args.qdrant_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    points = scroll_all_points()
    print(f"Fetched {len(points)} total points.", file=sys.stderr)

    # Determine which user_ids to scan
    if args.user:
        user_ids = [args.user]
    else:
        user_ids = sorted({
            p.get("payload", {}).get("user_id")
            for p in points
            if p.get("payload", {}).get("user_id")
        })
        if not user_ids:
            print("No points with a user_id found in collection.", file=sys.stderr)
            print_report([], [], 0)
            return

    print(f"User IDs to scan: {user_ids}", file=sys.stderr)

    all_groups = []
    all_pairs = []
    total_memories = 0

    for uid in user_ids:
        pairs, count = find_duplicates(points, uid)
        groups = group_pairs(pairs)
        all_groups.extend(groups)
        all_pairs.extend(pairs)
        total_memories += count

    print_report(all_groups, all_pairs, total_memories)

    if args.consolidate and all_groups:
        print("\n" + "=" * 72)
        if dry_run:
            print("  Consolidation Preview (DRY-RUN — nothing will be deleted)")
        else:
            print("  Consolidation Pass")
        print("=" * 72)
        kept, deleted = consolidate_groups(all_groups, dry_run=dry_run)
        if dry_run:
            print(f"\nDry-run complete: {kept} group(s) would be resolved, {deleted} duplicate(s) would be deleted.")
            print("Re-run with --consolidate --yes to actually execute the deletes.")
        else:
            print(f"\nConsolidation complete: {kept} group(s) resolved, {deleted} duplicate(s) deleted.")
    elif args.consolidate and not all_groups:
        print("\nNo duplicate groups to consolidate.")


if __name__ == "__main__":
    main()
