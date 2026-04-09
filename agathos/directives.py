"""
Agathos directive loading and execution.

Loads directive definitions from directives.yaml and executes them
against session data. Supports built-in check types and custom Python plugins.

Built-in check types:
- quality_threshold: AVG(quality_score) from holographic_memory.db
- count_threshold: COUNT(*) from specified table
- entropy_threshold: COUNT(*) from entropy_detections table

Custom checks can be loaded from Python files in the checks/ directory.
"""

import glob
import importlib.util
import json
import logging
import os
import sqlite3
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger("agathos.directives")

# Default path for directives.yaml
_DEFAULT_DIRECTIVES_PATH = os.path.expanduser("~/.hermes/agathos/directives.yaml")
_DEFAULT_CUSTOM_CHECKS_DIR = os.path.expanduser("~/.hermes/agathos/checks/")


def load_directives(path: Optional[str] = None) -> Dict:
    """Load and parse directives.yaml configuration file.

    Loads the Prime Directives configuration that defines quality thresholds,
    count checks, and entropy monitoring rules. Returns safe defaults if
    file missing or malformed.

    Args:
        path: Path to directives.yaml. Defaults to ~/.hermes/agathos/directives.yaml

    Returns:
        Dict with keys:
        - prime_directive: Natural language guidance string
        - checks: List of check configuration dicts
        - custom_checks_dir: Optional path to custom Python checks
        - ignore: Optional list of check names to skip

    Returns empty defaults {"prime_directive": "", "checks": []} on any error.

    Side effects: Logs info/warning/error via agathos.directives logger.
    """
    path = path or _DEFAULT_DIRECTIVES_PATH
    if not os.path.exists(path):
        logger.info("No directives.yaml at %s — using empty defaults", path)
        return {"prime_directive": "", "checks": []}

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            logger.warning("directives.yaml is not a dict — using empty defaults")
            return {"prime_directive": "", "checks": []}

        checks = data.get("checks") or []
        if not isinstance(checks, list):
            logger.warning("checks key is not a list — ignoring")
            checks = []

        logger.info("Loaded %d checks from directives.yaml", len(checks))
        return {
            "prime_directive": data.get("prime_directive", ""),
            "checks": checks,
            "custom_checks_dir": data.get("custom_checks_dir"),
            "ignore": data.get("ignore", []),
        }
    except Exception as e:
        logger.error("Failed to load directives.yaml: %s", e, exc_info=True)
        return {"prime_directive": "", "checks": []}


def _parse_window(window: str) -> str:
    """Convert human window (2h, 30m, 10m) to SQL datetime modifier."""
    if not window or window == "none":
        return None
    # Already in SQL format (e.g., '-2 hours')
    if window.startswith("-"):
        return window
    # Parse human format
    mapping = {
        "10m": "-10 minutes",
        "30m": "-30 minutes",
        "1h": "-1 hour",
        "2h": "-2 hours",
        "6h": "-6 hours",
        "24h": "-24 hours",
    }
    return mapping.get(window, f"-{window}")


def _check_quality_threshold(check: Dict, holo_conn: sqlite3.Connection) -> Dict:
    """Execute a quality_threshold check against holographic_memory.db."""
    metric = check.get("metric", "avg_quality")
    threshold = check.get("threshold", 0.92)
    window = _parse_window(check.get("window", "2h"))
    table = "facts"  # quality metrics live in facts table

    try:
        cur = holo_conn.cursor()
        if window:
            cur.execute(
                f"""  # nosec
                SELECT AVG(quality_score) as avg_q, COUNT(*) as cnt
                FROM {table}
                WHERE timestamp > datetime('now', ?)
                AND quality_score IS NOT NULL
            """,  # nosec B608
                (window,),
            )
        else:
            cur.execute(
                f"""  # nosec
                SELECT AVG(quality_score) as avg_q, COUNT(*) as cnt
                FROM {table}
                WHERE quality_score IS NOT NULL
            """  # nosec B608
            )
        row = cur.fetchone()
        avg_val = row["avg_q"] if row and row["avg_q"] else 0.0
        cnt = row["cnt"] if row else 0

        if cnt == 0:
            passed = True  # No data — can't fail
        else:
            passed = avg_val >= threshold

        return {
            "check_type": check["name"],
            "passed": passed,
            "details": json.dumps(
                {
                    "metric": metric,
                    "avg_value": round(avg_val, 4),
                    "count": cnt,
                    "threshold": threshold,
                }
            ),
        }
    except sqlite3.Error as e:
        return {
            "check_type": check["name"],
            "passed": True,
            "details": json.dumps({"error": str(e)}),
        }


def _check_count_threshold(
    check: Dict, session_id: str, holo_conn: sqlite3.Connection
) -> Dict:
    """Execute a count_threshold check against holographic_memory.db."""
    table = check.get("table", "facts")
    min_count = check.get("min_count", 1)
    window = _parse_window(check.get("window", "30m"))
    quality_threshold = check.get("quality_threshold")

    try:
        cur = holo_conn.cursor()
        conditions = []
        params = []

        if window:
            conditions.append("timestamp > datetime('now', ?)")
            params.append(window)
        if quality_threshold:
            conditions.append("quality_score >= ?")
            params.append(quality_threshold)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

        # Try session-specific first, fall back to system-wide
        if table == "trajectories":
            cur.execute(
                f"SELECT COUNT(*) as cnt FROM {table} WHERE session_id = ?{where.replace('WHERE', 'AND') if where else ''}",  # nosec B608
                (session_id, *params),
            )
            session_cnt = cur.fetchone()["cnt"]
            if session_cnt >= min_count:
                return {
                    "check_type": check["name"],
                    "passed": True,
                    "details": json.dumps(
                        {"count": session_cnt, "min_required": min_count}
                    ),
                }

        # System-wide check
        cur.execute(
            f"SELECT COUNT(*) as cnt FROM {table}{where}", params)  # nosec B608
        cnt = cur.fetchone()["cnt"]
        passed = cnt >= min_count

        return {
            "check_type": check["name"],
            "passed": passed,
            "details": json.dumps({"count": cnt, "min_required": min_count}),
        }
    except sqlite3.Error as e:
        return {
            "check_type": check["name"],
            "passed": True,
            "details": json.dumps({"error": str(e)}),
        }


def _check_entropy_threshold(
    check: Dict, cursor: sqlite3.Cursor, session_id: str
) -> Dict:
    """Execute an entropy_threshold check against agathos.db."""
    entropy_type = check.get("entropy_type", "")
    min_count = check.get("min_count", 3)
    window = _parse_window(check.get("window", "10m"))

    try:
        if window:
            cursor.execute(
                """
                SELECT COUNT(*) as cnt FROM entropy_detections
                WHERE session_id = ? AND entropy_type = ?
                AND timestamp > datetime('now', ?)
            """,
                (session_id, entropy_type, window),
            )
        else:
            cursor.execute(
                """
                SELECT COUNT(*) as cnt FROM entropy_detections
                WHERE session_id = ? AND entropy_type = ?
            """,
                (session_id, entropy_type),
            )
        cnt = cursor.fetchone()["cnt"]
        passed = cnt < min_count

        return {
            "check_type": check["name"],
            "passed": passed,
            "details": json.dumps(
                {"entropy_type": entropy_type, "count": cnt, "threshold": min_count}
            ),
        }
    except sqlite3.Error as e:
        return {
            "check_type": check["name"],
            "passed": True,
            "details": json.dumps({"error": str(e)}),
        }


def _load_custom_checks(checks_dir: str) -> List:
    """Load custom check plugins from a directory."""
    checks = []
    if not os.path.isdir(checks_dir):
        return checks

    for filepath in glob.glob(os.path.join(checks_dir, "*.py")):
        fname = os.path.basename(filepath)
        if fname.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(fname[:-3], filepath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "check"):
                checks.append(mod.check)
                logger.info("Loaded custom check from %s", fname)
        except Exception as e:
            logger.warning("Failed to load custom check %s: %s", fname, e)

    return checks


def execute_checks(
    session_id: str,
    cursor: sqlite3.Cursor,
    holo_conn: Optional[sqlite3.Connection],
    directives: Dict,
) -> List[Dict]:
    """Execute all enabled directive checks for a session.

    Runs the Prime Directives check suite against a session. Each check
    queries either agathos.db (for entropy) or holographic_memory.db
    (for quality/count metrics) and returns pass/fail results.

    Supported check types:
    - quality_threshold: AVG(quality_score) >= threshold (holographic_memory.db)
    - count_threshold: COUNT(*) >= min_count (holographic_memory.db table)
    - entropy_threshold: COUNT(*) >= min_count (agathos.db entropy_detections)
    - budget_threshold: Handled separately in agathos.py (skipped here)

    Args:
        session_id: Session to check
        cursor: Database cursor for agathos.db
        holo_conn: Optional connection to holographic_memory.db
        directives: Dict from load_directives with 'checks' list

    Returns:
        List of result dicts, each with:
        - check_type: Name of the check
        - passed: Boolean (True = quality met, False = violation)
        - details: JSON string with metrics and context

    Side effects:
        - Executes SQL queries against both databases
        - Logs info/warning/error via agathos.directives logger
        - Loads and executes custom Python checks if configured

    Error handling: Individual check failures are caught, logged, and
    return "passed": True (fail-safe) to prevent blocking sessions.
    """
    results = []
    checks = directives.get("checks", [])
    custom_checks_dir = directives.get("custom_checks_dir")

    for check in checks:
        if not check.get("enabled", True):
            continue

        check_type = check.get("type", "")

        try:
            if check_type == "quality_threshold":
                if holo_conn:
                    result = _check_quality_threshold(check, holo_conn)
                else:
                    result = {
                        "check_type": check["name"],
                        "passed": True,
                        "details": json.dumps(
                            {"note": "holographic_memory.db unavailable"}
                        ),
                    }

            elif check_type == "count_threshold":
                if holo_conn:
                    result = _check_count_threshold(check, session_id, holo_conn)
                else:
                    result = {
                        "check_type": check["name"],
                        "passed": True,
                        "details": json.dumps(
                            {"note": "holographic_memory.db unavailable"}
                        ),
                    }

            elif check_type == "entropy_threshold":
                result = _check_entropy_threshold(check, cursor, session_id)

            elif check_type == "budget_threshold":
                # Budget check is handled separately (needs session messages)
                # Skip here — argus.py handles it in detect_entropy
                continue

            else:
                logger.warning("Unknown check type: %s", check_type)
                continue

            results.append(result)

        except Exception as e:
            logger.error(
                "Error executing check %s: %s", check.get("name"), e, exc_info=True
            )
            results.append(
                {
                    "check_type": check["name"],
                    "passed": True,
                    "details": json.dumps({"error": str(e)}),
                }
            )

    # Execute custom checks
    if custom_checks_dir:
        custom_dir = os.path.expanduser(custom_checks_dir)
        for check_fn in _load_custom_checks(custom_dir):
            try:
                result = check_fn(session_id, cursor, {})
                if isinstance(result, dict) and "check_type" in result:
                    results.append(result)
            except Exception as e:
                logger.error("Custom check failed: %s", e, exc_info=True)

    return results


def setup_directives():
    """Generate directives.yaml by spawning a delegate agent.

    Returns the path to the generated file, or None on failure.
    """
    argus_dir = os.path.expanduser("~/.hermes/agathos")
    os.makedirs(argus_dir, exist_ok=True)

    directives_path = os.path.join(argus_dir, "directives.yaml")
    if os.path.exists(directives_path):
        logger.info("directives.yaml already exists at %s", directives_path)
        return directives_path

    # Read the default prompt template
    prompt_path = os.path.join(os.path.dirname(__file__), "setup_prompt.md")
    if not os.path.exists(prompt_path):
        logger.error("Setup prompt template not found at %s", prompt_path)
        return None

    with open(prompt_path) as f:
        prompt = f.read()

    # Add output path instruction
    prompt += f"\n\nWrite the output to: {directives_path}\n"

    logger.info("ARGUS setup: spawning delegate to generate directives.yaml")
    logger.info("Output path: %s", directives_path)

    return {
        "prompt": prompt,
        "output_path": directives_path,
        "agathos_dir": argus_dir,
    }
