#!/usr/bin/env python3
"""Monthly mem0 near-duplicate scanner — reports (and optionally consolidates) near-duplicates.

Scans a Qdrant collection for pairs of memories belonging to the same
user_id that have cosine similarity >= 0.92. Prints a report to stdout.

Without --consolidate: reports only (does NOT delete anything).
With --consolidate (alone): dry-run — prints what WOULD be deleted but does nothing.
With --consolidate --yes: actually deletes the identified duplicates via Qdrant API.

Supports both HTTP Qdrant (url=...) and embedded/local-path Qdrant (path=...).
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Profile-aware Hermes home resolution
# ---------------------------------------------------------------------------

try:
    from hermes_constants import get_hermes_home as _get_hermes_home  # noqa: F401
    _hermes_home: Path = _get_hermes_home()
except Exception:  # noqa: BLE001  — script may be run standalone outside the venv
    _hermes_home = Path.home() / ".hermes"


def get_hermes_home() -> Path:
    """Return the active Hermes home directory.

    When run inside the hermes-agent venv, delegates to the canonical
    ``hermes_constants.get_hermes_home()`` (which respects HERMES_HOME and
    active-profile overrides).  When run standalone, falls back to
    ``~/.hermes`` so the script still works outside the project.
    """
    try:
        from hermes_constants import get_hermes_home as _ghh  # noqa: F401
        return _ghh()
    except Exception:  # noqa: BLE001
        return Path.home() / ".hermes"


MEM0_CONFIG = get_hermes_home() / "mem0.json"
SCROLL_LIMIT = 100

# ---------------------------------------------------------------------------
# Runtime configuration — populated by main() after CLI parsing.
# Functions read from this namespace instead of bare module globals so that
# the script can be imported and tested without side-effects.
# ---------------------------------------------------------------------------

class _Cfg:  # noqa: N801  (intentionally lowercase-ish for internal use)
    """Mutable runtime config set once in main() and read by all helpers."""
    qdrant_url: str | None = None   # None → embedded (local-path) mode
    qdrant_path: str | None = None  # Set when running embedded Qdrant
    collection: str = "hermes_memories"
    threshold: float = 0.92


_cfg = _Cfg()

# ---------------------------------------------------------------------------
# Default values for argparse — these are the initial values before CLI parsing.
# All runtime code reads from _cfg, not from these constants.
# ---------------------------------------------------------------------------
_DEFAULT_THRESHOLD = _cfg.threshold
_DEFAULT_COLLECTION = _cfg.collection


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_mem0_config() -> dict:
    """Return parsed mem0.json as a dict, or {} if unavailable."""
    config_path = get_hermes_home() / "mem0.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not read {config_path}: {exc}", file=sys.stderr)
        return {}


def _config_user_id(cfg: dict) -> str | None:
    return cfg.get("user_id") or None


def _config_collection(cfg: dict) -> str | None:
    try:
        return cfg["oss"]["vector_store"]["config"]["collection_name"] or None
    except (KeyError, TypeError):
        return None


def _config_qdrant_path(cfg: dict) -> str | None:
    """Return the local storage path if Qdrant is configured in embedded mode."""
    try:
        vs_config = cfg["oss"]["vector_store"]["config"]
        return vs_config.get("path") or None
    except (KeyError, TypeError):
        return None


def _config_qdrant_url(cfg: dict) -> str | None:
    """Return the HTTP URL if Qdrant is configured in server mode."""
    try:
        vs_config = cfg["oss"]["vector_store"]["config"]
        # Qdrant HTTP config uses 'url' or 'host'/'port'
        url = vs_config.get("url")
        if url:
            return url
        host = vs_config.get("host")
        port = vs_config.get("port", 6333)
        if host:
            return f"http://{host}:{port}"
        return None
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Qdrant mode detection
# ---------------------------------------------------------------------------

def _is_embedded_mode() -> bool:
    """Return True when the script should use qdrant-client with path= (embedded)."""
    return _cfg.qdrant_path is not None


def _require_qdrant_client():
    """Import qdrant-client or exit with a helpful message."""
    try:
        from qdrant_client import QdrantClient  # noqa: F401
        return QdrantClient
    except ImportError:
        print(
            "[ERROR] qdrant-client is not installed. "
            "Install it with: pip install qdrant-client",
            file=sys.stderr,
        )
        sys.exit(1)


def _get_qdrant_client():
    """Return a QdrantClient configured for the active mode (embedded or HTTP)."""
    QdrantClient = _require_qdrant_client()
    if _cfg.qdrant_path:
        return QdrantClient(path=_cfg.qdrant_path)
    return QdrantClient(url=_cfg.qdrant_url)


# ---------------------------------------------------------------------------
# HTTP helpers (used only in HTTP mode)
# ---------------------------------------------------------------------------

def api_get(path, timeout=60):
    req = urllib.request.Request(f"{_cfg.qdrant_url}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def api_post(path, body, timeout=120):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{_cfg.qdrant_url}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def api_delete_points(point_ids: list) -> bool:
    """Delete points by ID via HTTP API. Returns True on success."""
    if not point_ids:
        return True
    body = {"points": point_ids}
    try:
        result = api_post(f"/collections/{_cfg.collection}/points/delete", body)
        status = result.get("result", {}).get("status", "unknown")
        return status == "acknowledged"
    except urllib.error.URLError as exc:
        print(f"[ERROR] Qdrant delete failed: {exc}", file=sys.stderr)
        return False


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

def scroll_all_points() -> list:
    """Fetch all points from the collection with vectors and payloads."""
    if _is_embedded_mode():
        return _scroll_all_points_embedded()
    return _scroll_all_points_http()


def _scroll_all_points_embedded() -> list:
    """Use qdrant-client SDK to scroll all points in embedded (local-path) mode."""
    client = _get_qdrant_client()
    points = []
    offset = None

    while True:
        try:
            result, next_offset = client.scroll(
                collection_name=_cfg.collection,
                scroll_filter=None,
                limit=SCROLL_LIMIT,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Qdrant scroll failed: {exc}", file=sys.stderr)
            sys.exit(1)

        # Convert ScoredPoint / Record objects to plain dicts
        for record in result:
            vec = record.vector
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            points.append({
                "id": str(record.id),
                "vector": vec,
                "payload": dict(record.payload or {}),
            })

        if next_offset is None or not result:
            break
        offset = next_offset

    return points


def _scroll_all_points_http() -> list:
    """Use raw HTTP API to scroll all points (HTTP/cloud Qdrant mode)."""
    points = []
    offset = None

    while True:
        body: dict = {
            "limit": SCROLL_LIMIT,
            "with_vector": True,
            "with_payload": True,
        }
        if offset is not None:
            body["offset"] = offset

        try:
            result = api_post(f"/collections/{_cfg.collection}/points/scroll", body)
        except urllib.error.URLError as exc:
            print(f"[ERROR] Qdrant scroll failed: {exc}", file=sys.stderr)
            sys.exit(1)

        batch = result.get("result", {}).get("points", [])
        # Normalise IDs to str for consistency
        for pt in batch:
            pt["id"] = str(pt["id"])
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
        # mem0 stores user_id directly in payload (may be int or str)
        raw_uid = payload.get("user_id")
        if raw_uid is None or str(raw_uid) != str(target_user_id):
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
            if score >= _cfg.threshold:
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
    and the consolidation step handles it safely by building a single
    deduplicated delete-set before issuing any deletes.
    """
    point_map = {}
    edge_scores = {}
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
    print(f"  Collection: {_cfg.collection}  |  Threshold: {_cfg.threshold}")
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
    print(f"  (Examined {len(pairs)} near-duplicate pair(s) at threshold >= {_cfg.threshold})")
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
    if _is_embedded_mode():
        client = _get_qdrant_client()
        try:
            from qdrant_client.models import PointIdsList
            client.delete(
                collection_name=_cfg.collection,
                points_selector=PointIdsList(points=point_ids),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Qdrant embedded delete failed: {exc}", file=sys.stderr)
            return False
    return api_delete_points(point_ids)


def consolidate_groups(groups: list, dry_run: bool = True) -> tuple[int, int]:
    """Keep the best memory per group and delete the rest.

    Builds a single deduplicated set of IDs to delete before executing any
    deletes, preventing the same ID from being submitted for deletion multiple
    times when it appears in overlapping pairs.

    When dry_run=True (the default), only prints what WOULD be deleted without
    touching Qdrant. Pass dry_run=False (requires --yes on the CLI) to actually
    execute the deletes.

    Returns (resolved_groups_count, deleted_count).
    """
    # --- Phase 1: determine keeper and losers for every group without touching Qdrant ---
    plan: list[dict] = []           # {keeper_id, to_delete, group_idx, group}
    ids_to_delete: set[str] = set() # deduplicated across all groups
    keepers: set[str] = set()       # IDs elected as keeper in a prior group

    for idx, group in enumerate(groups, 1):
        members = group["members"]
        # Skip this group if any member is already slated for deletion by a
        # higher-similarity group.  Without this guard, a transitive chain
        # A~B~C could delete both A and B even when A and C are not similar,
        # causing permanent information loss.
        if any(m in ids_to_delete for m in members):
            continue
        keeper_id = pick_keeper(group)
        to_delete = [pid for pid in members if pid != keeper_id]
        # Don't delete a point that was elected keeper in a prior group —
        # otherwise a transitive chain (A,B) then (B,C) could demote B,
        # causing A's information to be lost.
        to_delete = [pid for pid in to_delete if pid not in keepers]
        if not to_delete:
            continue
        keepers.add(keeper_id)
        ids_to_delete.update(to_delete)
        plan.append({
            "keeper_id": keeper_id,
            "to_delete": to_delete,
            "group_idx": idx,
            "group": group,
        })

    prefix = "[DRY-RUN] " if dry_run else ""

    # --- Phase 2: print the plan ---
    for entry in plan:
        keeper_id = entry["keeper_id"]
        to_delete = entry["to_delete"]
        group = entry["group"]
        keeper_text = extract_text(group["points"][keeper_id])
        print(f"\n[GROUP {entry['group_idx']}] {prefix}Keeping {keeper_id}")
        print(f"  Text: {keeper_text[:160]}{'...' if len(keeper_text) > 160 else ''}")
        print(f"  {prefix}Would delete {len(to_delete)} duplicate(s): {to_delete}")

    deleted_count = len(ids_to_delete)

    # --- Phase 3: execute (if not dry-run) using the deduplicated set ---
    if not dry_run:
        ok = delete_points(list(ids_to_delete))
        if not ok:
            print(
                f"  [FAIL] Batch delete of {deleted_count} point(s) did not fully succeed.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  [OK] Deleted {deleted_count} point(s) in one batch.")

    return len(plan), deleted_count


# ---------------------------------------------------------------------------
# Qdrant connectivity check
# ---------------------------------------------------------------------------

def verify_qdrant_connection(qdrant_url: str | None, qdrant_path: str | None) -> None:
    """Verify Qdrant is reachable; exit(1) on failure."""
    if qdrant_path:
        # Embedded mode — the path must already exist (it is created by mem0/Qdrant on
        # first write, but if it is absent at scan time there are no memories to read).
        p = Path(qdrant_path)
        if not p.exists():
            print(
                f"[ERROR] Embedded Qdrant path '{qdrant_path}' does not exist. "
                f"Has mem0 written any memories yet?",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Embedded Qdrant at: {qdrant_path}", file=sys.stderr)
        return

    # HTTP mode — hit the root endpoint
    try:
        info = api_get("/")
        version = info.get("version", "unknown")
        print(f"Qdrant {version} is up at {qdrant_url}", file=sys.stderr)
    except urllib.error.URLError as exc:
        print(f"[ERROR] Cannot reach Qdrant at {qdrant_url}: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    cfg = _load_mem0_config()
    cfg_user_id = _config_user_id(cfg)
    cfg_collection = _config_collection(cfg) or "hermes_memories"

    # Detect default Qdrant mode from mem0.json
    cfg_qdrant_path = _config_qdrant_path(cfg)
    cfg_qdrant_url = _config_qdrant_url(cfg)

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
        default=os.environ.get("QDRANT_URL", cfg_qdrant_url),
        help=(
            "Qdrant HTTP base URL (e.g. http://localhost:6333). "
            "Takes precedence over --qdrant-path. "
            "Default: QDRANT_URL env var, then mem0.json url config."
        ),
    )
    parser.add_argument(
        "--qdrant-path",
        default=cfg_qdrant_path,
        help=(
            "Local filesystem path for embedded Qdrant storage. "
            "Used when Qdrant runs in embedded (no server) mode. "
            "Default: from mem0.json oss.vector_store.config.path, "
            "or ~/.hermes/mem0_qdrant."
        ),
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
        default=_DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold for duplicate detection (default: {_DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Safety limit on number of points to compare (default: 2000). Use --max-points 0 to disable.",
    )
    args = parser.parse_args()

    # --yes without --consolidate is meaningless; warn and exit
    if args.yes and not args.consolidate:
        print("[ERROR] --yes requires --consolidate to be specified.", file=sys.stderr)
        sys.exit(1)

    dry_run = args.consolidate and not args.yes

    # ---------------------------------------------------------------------------
    # Resolve Qdrant mode: --qdrant-url takes precedence over --qdrant-path.
    # If neither is given, fall back to the default embedded path.
    # ---------------------------------------------------------------------------
    resolved_url: str | None = args.qdrant_url
    resolved_path: str | None = None

    if resolved_url:
        # Explicit HTTP URL provided — use HTTP mode regardless of path config
        resolved_path = None
    else:
        # No URL: use embedded mode
        resolved_path = args.qdrant_path or str(get_hermes_home() / "mem0_qdrant")

    # Wire resolved settings into the module-level _cfg object used by helpers
    _cfg.qdrant_url = resolved_url
    _cfg.qdrant_path = resolved_path
    _cfg.collection = args.collection
    _cfg.threshold = args.threshold

    # Logging
    if resolved_path:
        print(f"Qdrant mode: embedded path ({resolved_path})", file=sys.stderr)
    else:
        print(f"Qdrant mode: HTTP ({resolved_url})", file=sys.stderr)
    print(f"Collection:  {args.collection}", file=sys.stderr)
    if args.user is not None:
        print(f"User filter: {args.user!r}", file=sys.stderr)
    else:
        print("User filter: (all users in collection)", file=sys.stderr)

    if args.consolidate:
        if dry_run:
            print(
                "[CONSOLIDATE DRY-RUN] Will show what would be deleted. "
                "Re-run with --yes to execute.",
                file=sys.stderr,
            )
        else:
            print("[CONSOLIDATE MODE] Duplicates will be permanently deleted.", file=sys.stderr)

    verify_qdrant_connection(resolved_url, resolved_path)

    points = scroll_all_points()
    print(f"Fetched {len(points)} total points.", file=sys.stderr)

    if args.max_points and len(points) > args.max_points:
        print(
            f"ERROR: Collection has {len(points)} points, exceeding --max-points limit of {args.max_points}.",
            file=sys.stderr,
        )
        print(
            "O(n²) comparison would be very slow. Use --max-points 0 to disable this check, or filter by --user.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine which user_ids to scan
    if args.user is not None:
        user_ids = [args.user]
    else:
        user_ids = sorted({
            str(p.get("payload", {}).get("user_id"))
            for p in points
            if p.get("payload", {}).get("user_id") is not None
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
            print(
                f"\nDry-run complete: {kept} group(s) would be resolved, "
                f"{deleted} duplicate(s) would be deleted."
            )
            print("Re-run with --consolidate --yes to actually execute the deletes.")
        else:
            print(f"\nConsolidation complete: {kept} group(s) resolved, {deleted} duplicate(s) deleted.")
    elif args.consolidate and not all_groups:
        print("\nNo duplicate groups to consolidate.")


if __name__ == "__main__":
    main()
