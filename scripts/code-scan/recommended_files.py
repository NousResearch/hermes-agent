"""Deterministic relevance ranking for UA recommended files.

The scanner output is metadata-only; this module does not read project source and
never calls LLM/provider APIs.  It converts scan, graph analytics, and severity
artifacts into a stable reading plan with source/security-aware scoring.
"""
from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Any, Optional

BUCKET_ORDER = {
    "project identity": 0,
    "entrypoints": 1,
    "auth/security": 2,
    "data/API": 3,
    "backend/serverless": 4,
    "DB/RLS": 5,
    "runtime/deployment": 6,
    "tests": 7,
    "docs/process": 8,
    "other source": 9,
    "deprioritized assets": 10,
}

LOCKFILE_NAMES = {
    "package-lock.json",
    "npm-shrinkwrap.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lockb",
    "poetry.lock",
    "pdm.lock",
    "pipfile.lock",
    "cargo.lock",
    "composer.lock",
    "gemfile.lock",
    "go.sum",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".bmp",
    ".tiff",
    ".svg",
}

BINARY_LIKE_EXTENSIONS = {
    ".pdf",
    ".zip",
    ".gz",
    ".tgz",
    ".tar",
    ".7z",
    ".rar",
    ".wasm",
    ".dll",
    ".exe",
    ".so",
    ".dylib",
    ".jar",
    ".class",
    ".pyc",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
}

STATIC_ARTIFACT_EXTENSIONS = {
    ".map",
    ".min.js",
    ".bundle.js",
    ".chunk.js",
    ".css.map",
}

GENERATED_DIR_MARKERS = {
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".next",
    ".nuxt",
    "out",
    "target",
    "vendor",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}

SOURCE_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".kts",
    ".swift",
    ".rb",
    ".php",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".sql",
}

CONFIG_NAMES = {
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    "vercel.json",
    "netlify.toml",
    "fly.toml",
    "railway.json",
    "render.yaml",
    "wrangler.toml",
    "vite.config.ts",
    "vite.config.js",
    "next.config.js",
    "next.config.mjs",
    "next.config.ts",
    "tsconfig.json",
    "pyproject.toml",
    "requirements.txt",
    "package.json",
}

MANUAL_NAMES = {
    "agents.md",
    "agent.md",
    "readme.md",
    "readme",
    "contributing.md",
    "security.md",
    "architecture.md",
    "docs.md",
}

ENTRYPOINT_NAMES = {
    "main.py",
    "app.py",
    "server.py",
    "index.py",
    "main.ts",
    "main.tsx",
    "main.js",
    "main.jsx",
    "index.ts",
    "index.tsx",
    "index.js",
    "index.jsx",
    "app.ts",
    "app.tsx",
    "app.js",
    "app.jsx",
    "server.ts",
    "server.js",
}

SEVERITY_WEIGHTS = {
    "critical": 100,
    "high": 75,
    "medium": 45,
    "low": 20,
    "info": 8,
    "unknown": 10,
}


def _path_parts(path: str) -> list[str]:
    return [part.lower() for part in PurePosixPath(path.replace("\\", "/")).parts]


def _basename(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).name.lower()


def _suffix(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).suffix.lower()


def _has_any(path: str, needles: tuple[str, ...]) -> bool:
    lowered = path.lower()
    return any(needle in lowered for needle in needles)


def _is_static_artifact(path: str) -> bool:
    lowered = path.lower()
    return any(lowered.endswith(ext) for ext in STATIC_ARTIFACT_EXTENSIONS)


def _is_generated(path: str) -> bool:
    parts = set(_path_parts(path))
    return bool(parts & GENERATED_DIR_MARKERS) or _is_static_artifact(path)


def _is_lockfile(path: str) -> bool:
    return _basename(path) in LOCKFILE_NAMES


def _is_image(path: str) -> bool:
    return _suffix(path) in IMAGE_EXTENSIONS


def _is_binary_like(path: str, language: str) -> bool:
    language_l = language.lower()
    if language_l in {"binary", "image", "png", "jpg", "jpeg", "gif", "webp"}:
        return True
    return _suffix(path) in BINARY_LIKE_EXTENSIONS


def _bucket_for(path: str) -> str:
    base = _basename(path)
    lowered = path.lower().replace("\\", "/")
    parts = set(_path_parts(path))

    if base == "agents.md" or base == "package.json" or base == "pyproject.toml":
        return "project identity"
    if "test" in lowered or "spec" in lowered or "tests" in parts or "__tests__" in parts:
        return "tests"
    if _has_any(lowered, ("/auth/", "auth", "security", "session", "login", "oauth", "jwt", "permission", "role")):
        return "auth/security"
    if lowered.startswith("supabase/functions/") or "/functions/" in lowered and "supabase" in lowered:
        return "backend/serverless"
    if lowered.startswith("supabase/migrations/") or "/migrations/" in lowered or "rls" in lowered:
        return "DB/RLS"
    if _has_any(lowered, ("/api/", "api/", "route.ts", "route.js", "controller", "endpoint", "graphql", "trpc")):
        return "data/API"
    if base in ENTRYPOINT_NAMES or base.startswith("run_") or "src/main" in lowered or "src/index" in lowered:
        return "entrypoints"
    if base in CONFIG_NAMES or ".github/workflows/" in lowered or "docker" in base or base.endswith((".toml", ".yaml", ".yml")):
        return "runtime/deployment"
    if base in MANUAL_NAMES or lowered.startswith("docs/") or "/docs/" in lowered:
        return "docs/process"
    if _suffix(path) in SOURCE_EXTENSIONS:
        return "other source"
    return "deprioritized assets"


def _line_bonus(lines: int) -> int:
    if lines <= 0:
        return 0
    return min(20, int(math.log2(lines + 1) * 3))


def _score_scan_file(path: str, lines: int, language: str) -> tuple[int, dict[str, int], list[str], str]:
    bucket = _bucket_for(path)
    details: dict[str, int] = {"base": 10, "line_bonus": _line_bonus(lines)}
    reasons = ["listed in scan"]

    bucket_boosts = {
        "project identity": 90,
        "entrypoints": 80,
        "auth/security": 95,
        "data/API": 85,
        "backend/serverless": 82,
        "DB/RLS": 88,
        "runtime/deployment": 65,
        "tests": 50,
        "docs/process": 45,
        "other source": 30,
        "deprioritized assets": 0,
    }
    details["bucket_boost"] = bucket_boosts[bucket]
    if bucket != "deprioritized assets":
        reasons.append(f"bucket: {bucket}")

    penalty = 0
    if _is_lockfile(path):
        penalty -= 220
        reasons.append("deprioritized lockfile")
    if _is_image(path) or _is_binary_like(path, language):
        penalty -= 240
        reasons.append("deprioritized image/binary-like asset")
    if _is_generated(path):
        penalty -= 160
        reasons.append("deprioritized generated output")
    if lines > 3000 and bucket in {"deprioritized assets", "runtime/deployment", "docs/process"}:
        penalty -= 80
        reasons.append("deprioritized oversized static artifact")
    if penalty:
        details["penalty"] = penalty

    score = sum(details.values())
    return score, details, reasons, bucket


def _collect_hub_scores(analytics: Optional[dict]) -> dict[str, int]:
    hub_scores: dict[str, int] = {}
    if not analytics:
        return hub_scores
    hubs = analytics.get("hub_nodes", [])
    if not isinstance(hubs, list):
        return hub_scores
    for hub in hubs:
        if not isinstance(hub, dict):
            continue
        node_id = str(hub.get("node_id") or "")
        if not node_id.startswith("file:"):
            continue
        path = node_id[len("file:"):]
        try:
            degree = int(hub.get("degree", 0))
        except (TypeError, ValueError):
            degree = 0
        hub_scores[path] = max(hub_scores.get(path, 0), min(70, 35 + degree * 3))
    return hub_scores


def _collect_severity_scores(severity: Optional[dict]) -> dict[str, tuple[int, list[str]]]:
    severity_scores: dict[str, tuple[int, list[str]]] = {}
    if not severity:
        return severity_scores
    items = severity.get("items", [])
    if not isinstance(items, list):
        return severity_scores
    for item in items:
        if not isinstance(item, dict):
            continue
        path = str(item.get("file") or item.get("path") or "")
        if not path:
            continue
        level = str(item.get("severity") or "unknown").lower()
        weight = SEVERITY_WEIGHTS.get(level, SEVERITY_WEIGHTS["unknown"])
        old_weight, old_reasons = severity_scores.get(path, (0, []))
        reasons = old_reasons + [f"severity: {level}"]
        severity_scores[path] = (max(old_weight, weight), reasons)
    return severity_scores


def build_recommended_files(
    scan: Optional[dict],
    graph: Optional[dict],
    severity: Optional[dict],
    analytics: Optional[dict],
) -> list[dict]:
    """Build deterministic source/security-aware file recommendations.

    ``graph`` is accepted for API compatibility with build_context_bundle.py;
    the current scorer only needs graph analytics hub metadata.
    """
    del graph
    candidates: dict[str, dict[str, Any]] = {}

    if scan and isinstance(scan.get("files"), list):
        for file_item in scan.get("files", []):
            if not isinstance(file_item, dict):
                continue
            path = str(file_item.get("path") or "")
            if not path:
                continue
            lines = int(file_item.get("lines") or 0)
            language = str(file_item.get("language") or "")
            score, details, reasons, bucket = _score_scan_file(path, lines, language)
            candidates[path] = {
                "path": path,
                "reason_parts": reasons,
                "lines": lines,
                "language": language,
                "bucket": bucket,
                "score": score,
                "score_details": details,
            }

    hub_scores = _collect_hub_scores(analytics)
    for path, boost in hub_scores.items():
        entry = candidates.setdefault(path, {
            "path": path,
            "reason_parts": [],
            "lines": "unknown",
            "language": "",
            "bucket": _bucket_for(path),
            "score": 10,
            "score_details": {"base": 10},
        })
        entry["score"] += boost
        entry["score_details"]["hub_boost"] = boost
        entry["reason_parts"].append(f"hub node (score={boost})")

    severity_scores = _collect_severity_scores(severity)
    for path, (boost, reasons) in severity_scores.items():
        entry = candidates.setdefault(path, {
            "path": path,
            "reason_parts": [],
            "lines": "unknown",
            "language": "",
            "bucket": _bucket_for(path),
            "score": 10,
            "score_details": {"base": 10},
        })
        entry["score"] += boost
        entry["score_details"]["severity_boost"] = max(entry["score_details"].get("severity_boost", 0), boost)
        entry["reason_parts"].extend(reasons)

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -int(item.get("score", 0)),
            BUCKET_ORDER.get(str(item.get("bucket")), 99),
            str(item.get("path", "")),
        ),
    )

    recommended: list[dict] = []
    for item in ranked:
        reason_parts = []
        seen_reasons = set()
        for reason in item.pop("reason_parts", []):
            if reason not in seen_reasons:
                reason_parts.append(reason)
                seen_reasons.add(reason)
        item["reason"] = "; ".join(reason_parts) or "ranked by deterministic relevance"
        recommended.append(item)
    return recommended
