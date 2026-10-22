#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic X Analytics importer for Hermes.

Standard-library only. Raw analytics and normalized snapshots are sensitive by
default. No Git, network, browser, or publishing side effects are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

VERSION = "1.0.0"

EXIT_OK = 0
EXIT_VALIDATION = 2
EXIT_USAGE = 3
EXIT_INTERNAL = 4

DATE_FORMATS = (
    "%a, %b %d, %Y",
    "%a %b %d %Y",
    "%m/%d/%Y",
    "%Y-%m-%d",
)

SCHEMAS: dict[str, dict[str, Any]] = {
    "overview": {
        "required": [
            "date",
            "impressions",
            "likes",
            "engagements",
            "bookmarks",
            "shares",
            "new_follows",
            "unfollows",
            "replies",
            "reposts",
            "profile_visits",
        ],
        "optional": ["create_post", "video_views", "media_views"],
        "string": [],
        "date": ["date"],
        "number": [
            "impressions",
            "likes",
            "engagements",
            "bookmarks",
            "shares",
            "new_follows",
            "unfollows",
            "replies",
            "reposts",
            "profile_visits",
            "create_post",
            "video_views",
            "media_views",
        ],
    },
    "content": {
        "required": [
            "post_id",
            "date",
            "post_text",
            "post_link",
            "impressions",
            "likes",
            "engagements",
            "bookmarks",
            "shares",
            "new_follows",
            "replies",
            "reposts",
            "profile_visits",
        ],
        "optional": [
            "detail_expands",
            "url_clicks",
            "hashtag_clicks",
            "permalink_clicks",
        ],
        "string": ["post_id", "post_text", "post_link"],
        "date": ["date"],
        "number": [
            "impressions",
            "likes",
            "engagements",
            "bookmarks",
            "shares",
            "new_follows",
            "replies",
            "reposts",
            "profile_visits",
            "detail_expands",
            "url_clicks",
            "hashtag_clicks",
            "permalink_clicks",
        ],
    },
    "video": {
        "required": [
            "date",
            "views",
            "watch_time_ms",
            "completion_rate",
            "average_watch_time_ms",
            "estimated_revenue",
        ],
        "optional": [],
        "string": [],
        "date": ["date"],
        "number": [
            "views",
            "watch_time_ms",
            "completion_rate",
            "average_watch_time_ms",
            "estimated_revenue",
        ],
    },
}

ALIASES: dict[str, set[str]] = {
    "date": {"date", "day"},
    "impressions": {"impressions", "impression"},
    "likes": {"likes", "like"},
    "engagements": {"engagements", "engagement"},
    "bookmarks": {"bookmarks", "bookmark"},
    "shares": {"shares", "share"},
    "new_follows": {"new_follows", "new_follow", "follows", "new_followers"},
    "unfollows": {"unfollows", "unfollow", "lost_follows", "lost_followers"},
    "replies": {"replies", "reply"},
    "reposts": {"reposts", "repost", "retweets", "retweet"},
    "profile_visits": {"profile_visits", "profile_visit"},
    "create_post": {"create_post", "posts_created", "post_creations"},
    "video_views": {"video_views", "video_view"},
    "media_views": {"media_views", "media_view"},
    "post_id": {"post_id", "tweet_id", "id"},
    "post_text": {"post_text", "tweet_text", "text", "post"},
    "post_link": {"post_link", "tweet_link", "permalink", "url"},
    "detail_expands": {"detail_expands", "detail_expansions"},
    "url_clicks": {"url_clicks", "link_clicks"},
    "hashtag_clicks": {"hashtag_clicks", "hashtag_click"},
    "permalink_clicks": {"permalink_clicks", "permalink_click"},
    "views": {"views", "video_views"},
    "watch_time_ms": {"watch_time_ms", "watch_time_milliseconds"},
    "completion_rate": {"completion_rate", "video_completion_rate"},
    "average_watch_time_ms": {
        "average_watch_time_ms",
        "avg_watch_time_ms",
        "average_watch_time_milliseconds",
    },
    "estimated_revenue": {"estimated_revenue", "revenue", "estimated_earnings"},
}

PRIVACY_CLASSIFICATIONS = {
    "import_status": "public-safe",
    "schema": "public-safe",
    "date_coverage": "public-safe",
    "validation_summary": "public-safe",
    "methodology": "public-safe",
    "source_hashes": "private",
    "source_paths": "private",
    "raw_rows": "private",
    "normalized_rows": "private",
    "post_ids": "private",
    "post_text": "private",
    "post_links": "private",
    "lane_results": "private",
    "exact_account_metrics": "requires-approval",
    "exact_post_metrics": "requires-approval",
    "monetization_metrics": "requires-approval",
    "revenue_metrics": "requires-approval",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "schema": {
        "allow_unknown_columns": True,
        "fail_on_unknown_columns": False,
        "allow_blank_optional_numbers": True,
    },
    "partial_day": {
        "lookback_days": 14,
        "minimum_prior_days": 5,
        "relative_to_prior_median": 0.35,
        "absolute_impressions_floor": 1000,
        "include_partial_in_statistics": False,
    },
    "reconciliation": {
        "max_export_mtime_gap_seconds": 900,
        "content_end_date_tolerance_days": 1,
        "minimum_content_date_coverage_ratio": 0.95,
    },
    "classification": {
        "fallback_lane": "Unclassified",
        "minimum_confident_samples": 5,
        "lanes": [],
    },
    "statistics": {
        "metrics": [
            "impressions",
            "engagements",
            "likes",
            "bookmarks",
            "replies",
            "reposts",
            "new_follows",
            "engagement_rate",
            "bookmark_rate",
        ],
        "outlier_iqr_multiplier": 1.5,
        "minimum_group_size": 3,
    },
    "privacy": {
        "classifications": PRIVACY_CLASSIFICATIONS,
        "public_safe_fields": [
            "import_status",
            "schema",
            "date_coverage",
            "validation_summary",
            "methodology",
        ],
    },
    "monetization": {
        "threshold_impressions": None,
        "window_days": 90,
    },
}


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    row: int | None = None
    field: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.row is not None:
            result["row"] = self.row
        if self.field is not None:
            result["field"] = self.field
        return result


@dataclass
class ParsedExport:
    kind: str
    path: Path
    sha256: str
    size_bytes: int
    modified_utc: str
    header_row: int
    original_headers: list[str]
    mapping: dict[str, int]
    unknown_columns: list[str]
    rows: list[dict[str, Any]]
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def source_summary(self) -> dict[str, Any]:
        dates = sorted(
            {row["date"] for row in self.rows if isinstance(row.get("date"), str)}
        )
        return {
            "kind": self.kind,
            "filename": self.path.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "modified_utc": self.modified_utc,
            "header_row": self.header_row,
            "detected_columns": sorted(self.mapping),
            "unknown_columns": self.unknown_columns,
            "row_count": len(self.rows),
            "date_start": dates[0] if dates else None,
            "date_end": dates[-1] if dates else None,
        }


class ImportFailure(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is None:
        return config
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ImportFailure(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ImportFailure(
            f"Config is not valid JSON at line {exc.lineno}, column {exc.colno}."
        ) from exc
    if not isinstance(data, dict):
        raise ImportFailure("Config root must be a JSON object.")
    return deep_merge(config, data)


def normalize_header(value: str) -> str:
    value = value.lstrip("\ufeff").strip().casefold()
    value = value.replace("%", " percent ")
    value = re.sub(r"\([^)]*\)", lambda m: " " + m.group(0)[1:-1] + " ", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def canonical_for_header(value: str) -> str | None:
    normalized = normalize_header(value)
    for canonical, aliases in ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_date_value(value: str) -> date | None:
    cleaned = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def parse_number(value: str, *, allow_blank: bool) -> float | int | None:
    cleaned = value.strip()
    if not cleaned:
        return None if allow_blank else None
    cleaned = cleaned.replace(",", "").replace("$", "").strip()
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if is_percent:
        number /= 100.0
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number if math.isfinite(number) else None


def detect_header(rows: Sequence[Sequence[str]], kind: str) -> tuple[int, dict[str, int], list[str]]:
    schema = SCHEMAS[kind]
    required = set(schema["required"])
    best_index = -1
    best_mapping: dict[str, int] = {}
    best_unknown: list[str] = []
    best_score = -1

    for index, row in enumerate(rows[:10]):
        mapping: dict[str, int] = {}
        unknown: list[str] = []
        duplicate_canonical = False
        for column_index, header in enumerate(row):
            canonical = canonical_for_header(header)
            if canonical is None:
                if header.strip():
                    unknown.append(header.strip())
                continue
            if canonical in mapping:
                duplicate_canonical = True
                continue
            mapping[canonical] = column_index
        score = len(required.intersection(mapping))
        if duplicate_canonical:
            score -= 3
        if score > best_score:
            best_score = score
            best_index = index
            best_mapping = mapping
            best_unknown = unknown

    return best_index, best_mapping, best_unknown


def infer_kind(path: Path, rows: Sequence[Sequence[str]]) -> str:
    filename = path.name.casefold()
    if "account_overview" in filename:
        return "overview"
    if "analytics_content" in filename:
        return "content"
    if "video_overview" in filename:
        return "video"

    candidates = []
    for kind in SCHEMAS:
        index, mapping, _ = detect_header(rows, kind)
        required = set(SCHEMAS[kind]["required"])
        candidates.append((len(required.intersection(mapping)), kind, index))
    candidates.sort(reverse=True)
    if not candidates or candidates[0][0] == 0:
        raise ImportFailure(f"Could not infer export type for {path.name}.")
    return candidates[0][1]


def read_csv_rows(path: Path) -> list[list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [list(row) for row in csv.reader(handle)]
    except UnicodeDecodeError as exc:
        raise ImportFailure(f"{path.name} is not valid UTF-8 CSV data.") from exc
    except OSError as exc:
        raise ImportFailure(f"Could not read {path.name}: {exc}") from exc


def parse_export(path: Path, expected_kind: str | None, config: Mapping[str, Any]) -> ParsedExport:
    if not path.exists() or not path.is_file():
        raise ImportFailure(f"Input file not found: {path}")

    raw_rows = read_csv_rows(path)
    if not raw_rows:
        raise ImportFailure(f"{path.name} is empty.")

    kind = expected_kind or infer_kind(path, raw_rows)
    if kind not in SCHEMAS:
        raise ImportFailure(f"Unsupported export kind: {kind}")

    header_index, mapping, unknown = detect_header(raw_rows, kind)
    schema = SCHEMAS[kind]
    required = set(schema["required"])
    issues: list[Issue] = []

    if header_index < 0:
        raise ImportFailure(f"Could not find a header row in {path.name}.")

    missing = sorted(required.difference(mapping))
    for field_name in missing:
        issues.append(
            Issue(
                "error",
                "missing_required_column",
                f"Required column '{field_name}' was not found.",
                field=field_name,
            )
        )

    if unknown:
        severity = (
            "error"
            if config["schema"].get("fail_on_unknown_columns", False)
            else "warning"
        )
        for column in unknown:
            issues.append(
                Issue(
                    severity,
                    "unknown_column",
                    f"Unknown column detected: {column}",
                )
            )

    original_headers = raw_rows[header_index]
    normalized_rows: list[dict[str, Any]] = []
    allow_blank_optional = bool(
        config["schema"].get("allow_blank_optional_numbers", True)
    )

    for source_row_number, raw in enumerate(
        raw_rows[header_index + 1 :], start=header_index + 2
    ):
        if not any(cell.strip() for cell in raw):
            continue
        if kind == "video" and raw:
            first_cell = normalize_header(raw[0])
            if first_cell in {"your_videos", "uploaded_on"}:
                break
        record: dict[str, Any] = {"_source_row": source_row_number}
        row_has_error = False

        for canonical, column_index in mapping.items():
            value = raw[column_index] if column_index < len(raw) else ""
            if canonical in schema["string"]:
                record[canonical] = value.strip()
                if canonical in required and not value.strip() and canonical != "post_text":
                    issues.append(
                        Issue(
                            "error",
                            "blank_required_value",
                            f"Required field '{canonical}' is blank.",
                            row=source_row_number,
                            field=canonical,
                        )
                    )
                    row_has_error = True
            elif canonical in schema["date"]:
                parsed_date = parse_date_value(value)
                if parsed_date is None:
                    issues.append(
                        Issue(
                            "error",
                            "invalid_date",
                            f"Field '{canonical}' is not a supported date.",
                            row=source_row_number,
                            field=canonical,
                        )
                    )
                    row_has_error = True
                else:
                    record[canonical] = parsed_date.isoformat()
            elif canonical in schema["number"]:
                number = parse_number(
                    value,
                    allow_blank=canonical not in required and allow_blank_optional,
                )
                if number is None and canonical in required:
                    issues.append(
                        Issue(
                            "error",
                            "invalid_number",
                            f"Field '{canonical}' is blank or not numeric.",
                            row=source_row_number,
                            field=canonical,
                        )
                    )
                    row_has_error = True
                elif number is not None and number < 0:
                    issues.append(
                        Issue(
                            "error",
                            "negative_metric",
                            f"Field '{canonical}' cannot be negative.",
                            row=source_row_number,
                            field=canonical,
                        )
                    )
                    row_has_error = True
                else:
                    record[canonical] = 0 if number is None else number

        if len(raw) != len(original_headers):
            issues.append(
                Issue(
                    "warning",
                    "row_width_mismatch",
                    "Row column count does not match the detected header.",
                    row=source_row_number,
                )
            )
        if not row_has_error:
            normalized_rows.append(record)

    parsed = ParsedExport(
        kind=kind,
        path=path.resolve(),
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        modified_utc=datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(),
        header_row=header_index + 1,
        original_headers=original_headers,
        mapping=mapping,
        unknown_columns=unknown,
        rows=normalized_rows,
        issues=issues,
    )
    validate_semantics(parsed)
    return parsed


def validate_semantics(export: ParsedExport) -> None:
    if not export.rows:
        export.issues.append(
            Issue("error", "no_valid_rows", "No valid data rows were found.")
        )
        return

    if export.kind == "overview":
        dates = [row["date"] for row in export.rows]
        duplicates = sorted(key for key, count in Counter(dates).items() if count > 1)
        for duplicate in duplicates:
            export.issues.append(
                Issue(
                    "error",
                    "duplicate_overview_date",
                    f"Overview contains duplicate date {duplicate}.",
                    field="date",
                )
            )

    if export.kind == "content":
        ids = [str(row.get("post_id", "")).strip() for row in export.rows]
        duplicates = sorted(key for key, count in Counter(ids).items() if key and count > 1)
        for duplicate in duplicates:
            export.issues.append(
                Issue(
                    "error",
                    "duplicate_post_id",
                    "Content contains a duplicate post ID.",
                    field="post_id",
                )
            )

    for row in export.rows:
        impressions = row.get("impressions")
        engagements = row.get("engagements")
        if isinstance(impressions, (int, float)) and isinstance(
            engagements, (int, float)
        ):
            if engagements > impressions and impressions >= 0:
                export.issues.append(
                    Issue(
                        "warning",
                        "engagements_exceed_impressions",
                        "Engagements exceed impressions on a row. Verify X export semantics.",
                        row=row.get("_source_row"),
                    )
                )


def validation_report(exports: Sequence[ParsedExport]) -> dict[str, Any]:
    issues = [issue.as_dict() for export in exports for issue in export.issues]
    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "version": VERSION,
        "generated_at": utc_now(),
        "status": "failed" if errors else "passed_with_warnings" if warnings else "passed",
        "error_count": errors,
        "warning_count": warnings,
        "sources": [export.source_summary() for export in exports],
        "issues": issues,
    }


def identify_partial_days(
    overview_rows: Sequence[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    if not overview_rows:
        return {
            "partial_dates": [],
            "latest_complete_date": None,
            "confidence": "low",
            "reason": "No overview rows were available.",
        }

    ordered = sorted(overview_rows, key=lambda row: row["date"])
    latest = ordered[-1]
    prior = ordered[:-1]
    settings = config["partial_day"]
    lookback = int(settings.get("lookback_days", 14))
    prior_values = [
        float(row.get("impressions", 0))
        for row in prior[-lookback:]
        if float(row.get("impressions", 0)) > 0
    ]
    minimum_prior = int(settings.get("minimum_prior_days", 5))
    floor = float(settings.get("absolute_impressions_floor", 1000))
    ratio = float(settings.get("relative_to_prior_median", 0.35))

    partial_dates: list[str] = []
    confidence = "low"
    reason = "Insufficient prior days for robust partial-day detection."
    threshold = floor
    prior_median = None

    if len(prior_values) >= minimum_prior:
        prior_median = statistics.median(prior_values)
        threshold = max(floor, prior_median * ratio)
        if float(latest.get("impressions", 0)) < threshold:
            partial_dates.append(latest["date"])
            confidence = "high" if len(prior_values) >= 10 else "medium"
            reason = (
                "Latest overview day is below the configured fraction of the "
                "recent median."
            )
        else:
            confidence = "medium"
            reason = "Latest overview day is not materially below the recent median."
    else:
        latest_date = date.fromisoformat(latest["date"])
        if (
            latest_date >= date.today()
            and float(latest.get("impressions", 0)) < floor
        ):
            partial_dates.append(latest["date"])
            reason = "Latest day is current and below the absolute floor."

    complete_dates = [
        row["date"] for row in ordered if row["date"] not in partial_dates
    ]
    return {
        "partial_dates": partial_dates,
        "latest_complete_date": complete_dates[-1] if complete_dates else None,
        "confidence": confidence,
        "reason": reason,
        "threshold_impressions": round(threshold, 4),
        "prior_median_impressions": (
            round(float(prior_median), 4) if prior_median is not None else None
        ),
        "prior_sample_size": len(prior_values),
    }


def classify_post_type(row: Mapping[str, Any]) -> dict[str, str]:
    text = str(row.get("post_text", "")).strip()
    link = str(row.get("post_link", "")).strip()

    if not text:
        return {
            "value": "unknown",
            "confidence": "low",
            "method": "blank_text",
        }
    if re.match(r"^RT\s+@", text, flags=re.IGNORECASE):
        return {
            "value": "repost",
            "confidence": "high",
            "method": "rt_prefix",
        }
    if text.startswith("@"):
        return {
            "value": "reply",
            "confidence": "medium",
            "method": "leading_mention",
        }

    status_urls = re.findall(
        r"https?://(?:www\.)?(?:x|twitter)\.com/[^/\s]+/status/\d+",
        text,
        flags=re.IGNORECASE,
    )
    if status_urls and all(url.rstrip("/") != link.rstrip("/") for url in status_urls):
        return {
            "value": "quote",
            "confidence": "medium",
            "method": "embedded_status_url",
        }

    return {
        "value": "original",
        "confidence": "medium",
        "method": "default_non_reply_text",
    }


def compile_lane_rules(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for raw in config["classification"].get("lanes", []):
        if not isinstance(raw, Mapping) or not raw.get("name"):
            continue
        try:
            include = [re.compile(p, re.IGNORECASE) for p in raw.get("include_any", [])]
            exclude = [re.compile(p, re.IGNORECASE) for p in raw.get("exclude_any", [])]
        except re.error as exc:
            raise ImportFailure(
                f"Invalid lane regex in '{raw.get('name')}': {exc}"
            ) from exc
        rules.append(
            {
                "name": str(raw["name"]),
                "priority": int(raw.get("priority", 0)),
                "include": include,
                "exclude": exclude,
                "post_types": set(raw.get("post_types", [])),
            }
        )
    return sorted(rules, key=lambda item: (-item["priority"], item["name"]))


def classify_lane(
    row: Mapping[str, Any],
    post_type: Mapping[str, str],
    rules: Sequence[Mapping[str, Any]],
    fallback: str,
) -> dict[str, Any]:
    text = str(row.get("post_text", ""))
    matches: list[dict[str, Any]] = []

    for rule in rules:
        allowed_types = rule["post_types"]
        if allowed_types and post_type["value"] not in allowed_types:
            continue
        if any(pattern.search(text) for pattern in rule["exclude"]):
            continue
        include_matches = [
            pattern.pattern for pattern in rule["include"] if pattern.search(text)
        ]
        if include_matches:
            matches.append(
                {
                    "name": rule["name"],
                    "priority": rule["priority"],
                    "matched_patterns": include_matches,
                }
            )

    if not matches:
        return {
            "value": fallback,
            "confidence": "low",
            "method": "fallback",
            "candidates": [],
            "matched_patterns": [],
        }

    top_priority = matches[0]["priority"]
    top = [match for match in matches if match["priority"] == top_priority]
    if len(top) > 1:
        return {
            "value": "Uncertain",
            "confidence": "low",
            "method": "priority_tie",
            "candidates": [match["name"] for match in top],
            "matched_patterns": sorted(
                {pattern for match in top for pattern in match["matched_patterns"]}
            ),
        }

    winner = top[0]
    return {
        "value": winner["name"],
        "confidence": "high" if len(winner["matched_patterns"]) >= 2 else "medium",
        "method": "deterministic_regex",
        "candidates": [match["name"] for match in matches],
        "matched_patterns": winner["matched_patterns"],
    }


def normalize_posts(
    rows: Sequence[dict[str, Any]],
    partial: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rules = compile_lane_rules(config)
    fallback = str(config["classification"].get("fallback_lane", "Other"))
    partial_dates = set(partial.get("partial_dates", []))
    normalized: list[dict[str, Any]] = []

    for row in rows:
        post = {key: value for key, value in row.items() if key != "_source_row"}
        impressions = float(post.get("impressions", 0))
        engagements = float(post.get("engagements", 0))
        bookmarks = float(post.get("bookmarks", 0))
        post["engagement_rate"] = engagements / impressions if impressions else 0.0
        post["bookmark_rate"] = bookmarks / impressions if impressions else 0.0
        post["is_partial_day"] = post.get("date") in partial_dates
        post_type = classify_post_type(post)
        lane = classify_lane(post, post_type, rules, fallback)
        post["post_type"] = post_type
        post["lane"] = lane
        normalized.append(post)
    return normalized


def quantile(values: Sequence[float], percentile: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1 - weight) + clean[upper] * weight


def metric_summary(values: Sequence[float], iqr_multiplier: float) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "sample_size": 0,
            "total": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "minimum": None,
            "maximum": None,
            "outlier_count": 0,
        }
    p25 = quantile(clean, 0.25)
    p75 = quantile(clean, 0.75)
    assert p25 is not None and p75 is not None
    iqr = p75 - p25
    upper = p75 + iqr_multiplier * iqr
    lower = p25 - iqr_multiplier * iqr
    outliers = [value for value in clean if value < lower or value > upper]
    return {
        "sample_size": len(clean),
        "total": sum(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p25": p25,
        "p75": p75,
        "p90": quantile(clean, 0.90),
        "p95": quantile(clean, 0.95),
        "minimum": min(clean),
        "maximum": max(clean),
        "iqr": iqr,
        "outlier_lower_bound": lower,
        "outlier_upper_bound": upper,
        "outlier_count": len(outliers),
    }


def confidence_for_sample(size: int) -> str:
    if size >= 30:
        return "high"
    if size >= 10:
        return "medium"
    return "low"


def summarize_group(
    posts: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    metric_names = config["statistics"].get("metrics", [])
    multiplier = float(config["statistics"].get("outlier_iqr_multiplier", 1.5))
    summary = {
        metric: metric_summary(
            [
                float(post.get(metric, 0))
                for post in posts
                if isinstance(post.get(metric, 0), (int, float))
            ],
            multiplier,
        )
        for metric in metric_names
    }
    return {
        "sample_size": len(posts),
        "confidence": confidence_for_sample(len(posts)),
        "metrics": summary,
    }


def calculate_statistics(
    posts: Sequence[dict[str, Any]], partial: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    include_partial = bool(
        config["partial_day"].get("include_partial_in_statistics", False)
    )
    eligible = [
        post for post in posts if include_partial or not post.get("is_partial_day")
    ]
    by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for post in eligible:
        by_lane[post["lane"]["value"]].append(post)
        by_type[post["post_type"]["value"]].append(post)

    impressions = [float(post.get("impressions", 0)) for post in eligible]
    impression_summary = metric_summary(
        impressions,
        float(config["statistics"].get("outlier_iqr_multiplier", 1.5)),
    )
    upper = impression_summary.get("outlier_upper_bound")
    outlier_ids = [
        post.get("post_id")
        for post in eligible
        if upper is not None and float(post.get("impressions", 0)) > float(upper)
    ]

    return {
        "methodology": {
            "partial_days_excluded": not include_partial,
            "percentiles": ["p25", "p50", "p75", "p90", "p95"],
            "outlier_method": "Tukey IQR",
            "outlier_iqr_multiplier": config["statistics"].get(
                "outlier_iqr_multiplier", 1.5
            ),
        },
        "overall": summarize_group(eligible, config),
        "by_lane": {
            name: summarize_group(group, config)
            for name, group in sorted(by_lane.items())
        },
        "by_post_type": {
            name: summarize_group(group, config)
            for name, group in sorted(by_type.items())
        },
        "viral_outliers": {
            "sample_size": len(eligible),
            "count": len(outlier_ids),
            "post_ids": outlier_ids,
        },
        "partial_day": partial,
    }


def parse_content_filename_range(filename: str) -> tuple[str | None, str | None]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", filename)
    return (match.group(1), match.group(2)) if match else (None, None)


def reconcile_exports(
    overview: ParsedExport | None,
    content: ParsedExport | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    status = "passed"

    def add(level: str, code: str, message: str, details: Any = None) -> None:
        nonlocal status
        checks.append(
            {
                "level": level,
                "code": code,
                "message": message,
                "details": details,
            }
        )
        if level == "error":
            status = "failed"
        elif level == "warning" and status == "passed":
            status = "passed_with_warnings"

    if overview is None or content is None:
        add(
            "warning",
            "missing_pair",
            "Overview and content were not both supplied, so cross-export reconciliation is limited.",
        )
        return {"status": status, "checks": checks}

    overview_dates = sorted({row["date"] for row in overview.rows})
    content_dates = sorted({row["date"] for row in content.rows})
    if not overview_dates or not content_dates:
        add("error", "empty_date_coverage", "One export has no valid dates.")
        return {"status": status, "checks": checks}

    tolerance_days = int(
        config["reconciliation"].get("content_end_date_tolerance_days", 1)
    )
    overview_start = date.fromisoformat(overview_dates[0])
    overview_end = date.fromisoformat(overview_dates[-1])
    content_start = date.fromisoformat(content_dates[0])
    content_end = date.fromisoformat(content_dates[-1])
    signed_end_gap = (content_end - overview_end).days

    if signed_end_gap > tolerance_days:
        add(
            "error",
            "content_after_overview",
            "Content contains publish dates later than the overview export coverage.",
            {
                "overview_end": str(overview_end),
                "content_end": str(content_end),
                "days": signed_end_gap,
            },
        )
    elif abs(signed_end_gap) > tolerance_days:
        add(
            "warning",
            "end_date_mismatch",
            "Overview and content end dates differ beyond the configured tolerance.",
            {
                "overview_end": str(overview_end),
                "content_end": str(content_end),
                "days": abs(signed_end_gap),
            },
        )
    else:
        add(
            "info",
            "end_dates_aligned",
            "Overview and content end dates align within tolerance.",
            {"days": abs(signed_end_gap)},
        )

    if content_start < overview_start:
        add(
            "warning",
            "content_starts_before_overview",
            "Content contains publish dates earlier than the overview coverage.",
            {
                "overview_start": str(overview_start),
                "content_start": str(content_start),
            },
        )

    filename_start, filename_end = parse_content_filename_range(content.path.name)
    if filename_start and filename_start != content_dates[0]:
        add(
            "warning",
            "filename_content_start_mismatch",
            "Content filename start date differs from the earliest valid content row.",
            {"filename_start": filename_start, "row_start": content_dates[0]},
        )
    if filename_end and filename_end != content_dates[-1]:
        add(
            "warning",
            "filename_content_end_mismatch",
            "Content filename end date differs from the latest valid content row.",
            {"filename_end": filename_end, "row_end": content_dates[-1]},
        )
    if filename_end:
        filename_end_date = date.fromisoformat(filename_end)
        if (filename_end_date - overview_end).days > tolerance_days:
            add(
                "error",
                "filename_after_overview",
                "Content filename claims coverage later than the overview export.",
                {"filename_end": filename_end, "overview_end": str(overview_end)},
            )

    mtime_gap = abs(
        datetime.fromisoformat(overview.modified_utc).timestamp()
        - datetime.fromisoformat(content.modified_utc).timestamp()
    )
    max_gap = int(
        config["reconciliation"].get("max_export_mtime_gap_seconds", 900)
    )
    if mtime_gap > max_gap:
        add(
            "warning",
            "mtime_gap",
            "Source modification times suggest the files may not be from the same export run.",
            {"seconds": round(mtime_gap, 3)},
        )
    else:
        add(
            "info",
            "mtime_aligned",
            "Source modification times are consistent with one export run.",
            {"seconds": round(mtime_gap, 3)},
        )

    overview_date_set = set(overview_dates)
    content_date_set = set(content_dates)
    overlapping = sorted(overview_date_set.intersection(content_date_set))
    content_date_coverage_ratio = (
        len(overlapping) / len(content_date_set) if content_date_set else 0.0
    )
    minimum_coverage = float(
        config["reconciliation"].get("minimum_content_date_coverage_ratio", 0.95)
    )
    if content_date_coverage_ratio < minimum_coverage:
        add(
            "warning",
            "low_content_date_coverage",
            "Some content publish dates are outside or missing from overview coverage.",
            {
                "covered_dates": len(overlapping),
                "content_dates": len(content_date_set),
                "coverage_ratio": content_date_coverage_ratio,
            },
        )
    else:
        add(
            "info",
            "content_date_coverage",
            "Content publish dates are covered by overview dates.",
            {
                "covered_dates": len(overlapping),
                "content_dates": len(content_date_set),
                "coverage_ratio": content_date_coverage_ratio,
            },
        )

    add(
        "info",
        "metric_scopes_not_directly_comparable",
        "Content post metrics are not summed against daily overview metrics because the exports use different scopes.",
    )

    return {
        "status": status,
        "checks": checks,
        "coverage": {
            "overview_date_start": overview_dates[0],
            "overview_date_end": overview_dates[-1],
            "content_date_start": content_dates[0],
            "content_date_end": content_dates[-1],
            "overlap_days": len(overlapping),
            "content_date_coverage_ratio": content_date_coverage_ratio,
            "content_filename_start": filename_start,
            "content_filename_end": filename_end,
            "metric_reconciliation": "not_performed_non_comparable_scopes",
        },
    }


def account_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "impressions",
        "likes",
        "engagements",
        "bookmarks",
        "shares",
        "new_follows",
        "unfollows",
        "replies",
        "reposts",
        "profile_visits",
        "video_views",
        "media_views",
    ]
    totals = {
        metric: sum(float(row.get(metric, 0)) for row in rows)
        for metric in metric_names
    }
    totals["net_follows"] = totals["new_follows"] - totals["unfollows"]
    totals["engagement_rate"] = (
        totals["engagements"] / totals["impressions"]
        if totals["impressions"]
        else 0.0
    )
    return totals


def calculate_monetization(
    overview_rows: Sequence[Mapping[str, Any]],
    video_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    threshold = config["monetization"].get("threshold_impressions")
    revenue_total = sum(float(row.get("estimated_revenue", 0)) for row in video_rows)
    if threshold is None and not video_rows:
        return None

    ordered = sorted(overview_rows, key=lambda row: row["date"])
    window_days = int(config["monetization"].get("window_days", 90))
    window_rows = ordered[-window_days:] if ordered else []
    window_impressions = sum(
        float(row.get("impressions", 0)) for row in window_rows
    )
    result: dict[str, Any] = {
        "privacy": "requires-approval",
        "window_days": window_days,
        "window_impressions": window_impressions,
        "estimated_revenue": revenue_total,
    }
    if threshold is not None:
        threshold_value = float(threshold)
        result["threshold_impressions"] = threshold_value
        result["remaining_impressions"] = max(
            0.0, threshold_value - window_impressions
        )
    return result


def make_import_id(exports: Sequence[ParsedExport]) -> tuple[str, str]:
    components = sorted(f"{export.kind}:{export.sha256}" for export in exports)
    combined = hashlib.sha256("\n".join(components).encode("utf-8")).hexdigest()
    return combined[:20], combined


def latest_snapshot(output_dir: Path, exclude: Path | None = None) -> Path | None:
    snapshot_dir = output_dir / "snapshots"
    if not snapshot_dir.exists():
        return None
    candidates = [
        path
        for path in snapshot_dir.glob("*.json")
        if exclude is None or path.resolve() != exclude.resolve()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def compare_snapshot_dicts(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    left_overview = {
        row["date"]: row for row in left.get("data", {}).get("overview", [])
    }
    right_overview = {
        row["date"]: row for row in right.get("data", {}).get("overview", [])
    }
    overlap_dates = sorted(set(left_overview).intersection(right_overview))

    metrics = [
        "impressions",
        "engagements",
        "likes",
        "bookmarks",
        "new_follows",
        "unfollows",
    ]
    matched_period = {}
    for metric in metrics:
        left_total = sum(float(left_overview[d].get(metric, 0)) for d in overlap_dates)
        right_total = sum(float(right_overview[d].get(metric, 0)) for d in overlap_dates)
        matched_period[metric] = {
            "left": left_total,
            "right": right_total,
            "delta": right_total - left_total,
        }

    left_posts = {
        str(row.get("post_id")): row
        for row in left.get("data", {}).get("content", [])
        if row.get("post_id")
    }
    right_posts = {
        str(row.get("post_id")): row
        for row in right.get("data", {}).get("content", [])
        if row.get("post_id")
    }
    common_ids = sorted(set(left_posts).intersection(right_posts))
    post_metric_deltas: dict[str, float] = defaultdict(float)
    for post_id in common_ids:
        for metric in metrics:
            post_metric_deltas[metric] += float(right_posts[post_id].get(metric, 0)) - float(
                left_posts[post_id].get(metric, 0)
            )

    union_dates = set(left_overview).union(right_overview)
    overlap_ratio = len(overlap_dates) / len(union_dates) if union_dates else 0.0
    confidence = "high" if overlap_ratio >= 0.95 else "medium" if overlap_ratio >= 0.75 else "low"

    return {
        "version": VERSION,
        "generated_at": utc_now(),
        "left_import_id": left.get("import_id"),
        "right_import_id": right.get("import_id"),
        "confidence": confidence,
        "overview_overlap": {
            "days": len(overlap_dates),
            "union_days": len(union_dates),
            "overlap_ratio": overlap_ratio,
            "date_start": overlap_dates[0] if overlap_dates else None,
            "date_end": overlap_dates[-1] if overlap_dates else None,
            "matched_period_metrics": matched_period,
        },
        "content_match": {
            "matched_post_ids": len(common_ids),
            "new_post_ids": len(set(right_posts).difference(left_posts)),
            "removed_post_ids": len(set(left_posts).difference(right_posts)),
            "matched_post_metric_deltas": dict(post_metric_deltas),
        },
    }


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_manifest_entries(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "manifests" / "imports.jsonl"
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                entries.append(value)
    return entries


def already_imported(output_dir: Path, combined_hash: str) -> dict[str, Any] | None:
    for entry in reversed(read_manifest_entries(output_dir)):
        if (
            entry.get("combined_source_hash") == combined_hash
            and entry.get("status") == "complete"
        ):
            return entry
    return None


def render_analysis_markdown(snapshot: Mapping[str, Any]) -> str:
    validation = snapshot["validation"]
    partial = snapshot["partial_day"]
    stats = snapshot["statistics"]
    totals = snapshot.get("account_totals", {})
    lines = [
        "# X Analytics Import Report",
        "",
        f"- Import ID: `{snapshot['import_id']}`",
        f"- Importer version: `{snapshot['version']}`",
        f"- Generated: `{snapshot['generated_at']}`",
        f"- Validation: **{validation['status']}**",
        f"- Reconciliation: **{snapshot['reconciliation']['status']}**",
        f"- Latest complete date: `{partial.get('latest_complete_date')}`",
        f"- Partial dates: `{', '.join(partial.get('partial_dates', [])) or 'none'}`",
        "",
        "## Local account totals",
        "",
        "> Privacy class: requires approval before public use.",
        "",
    ]
    for key in [
        "impressions",
        "engagements",
        "likes",
        "bookmarks",
        "net_follows",
        "engagement_rate",
    ]:
        value = totals.get(key)
        if value is not None:
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")
    lines.extend(
        [
            "",
            "## Statistical summary",
            "",
            f"- Eligible post sample size: {stats['overall']['sample_size']}",
            f"- Confidence: {stats['overall']['confidence']}",
            f"- Viral outliers by IQR: {stats['viral_outliers']['count']}",
            "",
            "## Lane summary",
            "",
            "| Lane | n | Confidence | Median impressions | p90 impressions |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for lane, result in stats["by_lane"].items():
        impressions = result["metrics"].get("impressions", {})
        lines.append(
            f"| {lane} | {result['sample_size']} | {result['confidence']} | "
            f"{impressions.get('median')} | {impressions.get('p90')} |"
        )
    lines.extend(
        [
            "",
            "## Source attribution",
            "",
            "Every source is identified by filename, SHA-256 hash, row count, schema, and date coverage in the normalized snapshot.",
            "",
            "## Interpretation boundary",
            "",
            "This report contains deterministic calculations. Any strategic recommendation added by an agent must cite this snapshot, state its sample size, account for outliers, and preserve the privacy classifications.",
            "",
        ]
    )
    return "\n".join(lines)


def render_public_safe_markdown(snapshot: Mapping[str, Any]) -> str:
    sources = snapshot["validation"]["sources"]
    lines = [
        "# X Analytics Import Validation",
        "",
        "This report contains only fields classified as public-safe.",
        "",
        f"- Import status: **{snapshot['validation']['status']}**",
        f"- Reconciliation status: **{snapshot['reconciliation']['status']}**",
        f"- Importer version: `{snapshot['version']}`",
        "",
        "## Data coverage",
        "",
    ]
    for source in sources:
        lines.append(
            f"- {source['kind']}: {source['row_count']} rows, "
            f"{source.get('date_start')} to {source.get('date_end')}"
        )
    lines.extend(
        [
            "",
            "## Methodology",
            "",
            "- Inputs were schema-detected and validated.",
            "- Duplicate IDs and duplicate overview dates were checked.",
            "- Partial current-day data was detected using a configurable recent-median rule.",
            "- Statistics use medians, percentiles, sample sizes, and IQR outlier detection.",
            "- Exact metrics, post content, source hashes, lane findings, revenue, and monetization data are excluded.",
            "",
        ]
    )
    rendered = "\n".join(lines)
    forbidden_tokens = [
        "post_text",
        "post_link",
        "estimated_revenue",
        "window_impressions",
        "source_hash",
    ]
    for token in forbidden_tokens:
        if token in rendered:
            raise ImportFailure(f"Public-safe report leakage guard found token: {token}")
    return rendered


def build_snapshot(
    exports: Sequence[ParsedExport],
    config: Mapping[str, Any],
    import_id: str,
    combined_hash: str,
) -> dict[str, Any]:
    by_kind = {export.kind: export for export in exports}
    overview_export = by_kind.get("overview")
    content_export = by_kind.get("content")
    video_export = by_kind.get("video")
    overview_rows = [
        {key: value for key, value in row.items() if key != "_source_row"}
        for row in (overview_export.rows if overview_export else [])
    ]
    video_rows = [
        {key: value for key, value in row.items() if key != "_source_row"}
        for row in (video_export.rows if video_export else [])
    ]

    partial = identify_partial_days(overview_rows, config)
    content_rows = normalize_posts(
        content_export.rows if content_export else [], partial, config
    )
    validation = validation_report(exports)
    reconciliation = reconcile_exports(
        overview_export, content_export, config
    )
    statistics_result = calculate_statistics(content_rows, partial, config)
    totals = account_totals(overview_rows)
    monetization = calculate_monetization(overview_rows, video_rows, config)

    return {
        "schema_version": "2.0",
        "version": VERSION,
        "import_id": import_id,
        "combined_source_hash": combined_hash,
        "generated_at": utc_now(),
        "privacy": {
            "default": "private",
            "classifications": config["privacy"]["classifications"],
        },
        "sources": [export.source_summary() for export in exports],
        "validation": validation,
        "reconciliation": reconciliation,
        "partial_day": partial,
        "account_totals": totals,
        "statistics": statistics_result,
        "monetization": monetization,
        "data": {
            "overview": overview_rows,
            "content": content_rows,
            "video": video_rows,
        },
    }


def save_import_artifacts(
    snapshot: Mapping[str, Any],
    output_dir: Path,
    *,
    previous_snapshot: Path | None,
) -> dict[str, str]:
    import_id = str(snapshot["import_id"])
    paths = {
        "validation_report": output_dir / "reports" / f"{import_id}-validation.json",
        "reconciliation_report": output_dir / "reports" / f"{import_id}-reconciliation.json",
        "classification_report": output_dir / "reports" / f"{import_id}-classification.json",
        "analysis_report": output_dir / "reports" / f"{import_id}-analysis.md",
        "public_safe_report": output_dir / "reports" / f"{import_id}-public-safe.md",
        "normalized_snapshot": output_dir / "snapshots" / f"{import_id}.json",
        "manifest": output_dir / "manifests" / f"{import_id}.json",
    }
    atomic_write_json(paths["validation_report"], snapshot["validation"])
    atomic_write_json(paths["reconciliation_report"], snapshot["reconciliation"])
    atomic_write_json(
        paths["classification_report"],
        {
            "version": VERSION,
            "import_id": import_id,
            "rows": [
                {
                    "post_id": post.get("post_id"),
                    "post_type": post.get("post_type"),
                    "lane": post.get("lane"),
                }
                for post in snapshot["data"]["content"]
            ],
        },
    )
    atomic_write_text(paths["analysis_report"], render_analysis_markdown(snapshot))
    atomic_write_text(
        paths["public_safe_report"], render_public_safe_markdown(snapshot)
    )
    atomic_write_json(paths["normalized_snapshot"], snapshot)

    comparison_path: Path | None = None
    if previous_snapshot and previous_snapshot.exists():
        prior = json.loads(previous_snapshot.read_text(encoding="utf-8"))
        comparison = compare_snapshot_dicts(prior, snapshot)
        comparison_path = (
            output_dir
            / "reports"
            / f"{prior.get('import_id', 'prior')}-to-{import_id}-comparison.json"
        )
        atomic_write_json(comparison_path, comparison)

    manifest = {
        "schema_version": "2.0",
        "version": VERSION,
        "import_id": import_id,
        "combined_source_hash": snapshot["combined_source_hash"],
        "created_at": utc_now(),
        "status": "complete",
        "source_files": [
            {
                "kind": source["kind"],
                "filename": source["filename"],
                "sha256": source["sha256"],
            }
            for source in snapshot["sources"]
        ],
        "artifacts": {
            key: str(path)
            for key, path in paths.items()
            if key != "manifest"
        },
        "comparison_report": str(comparison_path) if comparison_path else None,
    }
    atomic_write_json(paths["manifest"], manifest)
    append_jsonl(output_dir / "manifests" / "imports.jsonl", manifest)
    return {key: str(value) for key, value in paths.items()} | {
        "comparison_report": str(comparison_path) if comparison_path else ""
    }


def inspect_inputs(
    inputs: Sequence[tuple[str | None, Path]], config: Mapping[str, Any]
) -> dict[str, Any]:
    exports = [parse_export(path, kind, config) for kind, path in inputs]
    return {
        "version": VERSION,
        "generated_at": utc_now(),
        "sources": [export.source_summary() for export in exports],
        "validation_summary": validation_report(exports),
    }


def load_snapshot(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ImportFailure(f"Snapshot not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ImportFailure(f"Snapshot is invalid JSON: {path}") from exc
    if not isinstance(value, dict) or "data" not in value:
        raise ImportFailure(f"File is not a normalized snapshot: {path}")
    return value


def rebuild_classification(
    snapshot_path: Path, output_dir: Path, config: Mapping[str, Any], dry_run: bool
) -> dict[str, Any]:
    snapshot = load_snapshot(snapshot_path)
    content = snapshot.get("data", {}).get("content", [])
    partial = snapshot.get("partial_day", {})
    rebuilt = normalize_posts(content, partial, config)
    result = dict(snapshot)
    result["version"] = VERSION
    result["classification_rebuilt_at"] = utc_now()
    result["data"] = dict(snapshot["data"])
    result["data"]["content"] = rebuilt
    result["statistics"] = calculate_statistics(rebuilt, partial, config)
    old_id = str(snapshot.get("import_id", "snapshot"))
    config_hash = hashlib.sha256(
        json.dumps(config["classification"], sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    output_path = output_dir / "snapshots" / f"{old_id}-reclassified-{config_hash}.json"
    report_path = output_dir / "reports" / f"{old_id}-reclassified-{config_hash}.json"
    report = {
        "version": VERSION,
        "source_snapshot": str(snapshot_path),
        "classification_config_hash": config_hash,
        "rows": [
            {
                "post_id": post.get("post_id"),
                "post_type": post.get("post_type"),
                "lane": post.get("lane"),
            }
            for post in rebuilt
        ],
    }
    if not dry_run:
        atomic_write_json(output_path, result)
        atomic_write_json(report_path, report)
    return {
        "status": "dry_run" if dry_run else "complete",
        "snapshot": str(output_path),
        "report": str(report_path),
        "row_count": len(rebuilt),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, normalize, classify, compare, and report on X Analytics CSV exports."
    )
    parser.add_argument(
        "mode",
        choices=[
            "inspect",
            "validate-only",
            "dry-run",
            "full-import",
            "incremental-import",
            "compare-snapshots",
            "rebuild-classification",
        ],
    )
    parser.add_argument("--overview", type=Path)
    parser.add_argument("--content", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path.cwd() / "x-analytics-artifacts"
    )
    parser.add_argument("--snapshot-a", type=Path)
    parser.add_argument("--snapshot-b", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--version", action="version", version=VERSION)
    return parser


def collect_inputs(args: argparse.Namespace) -> list[tuple[str | None, Path]]:
    inputs = []
    if args.overview:
        inputs.append(("overview", args.overview))
    if args.content:
        inputs.append(("content", args.content))
    if args.video:
        inputs.append(("video", args.video))
    return inputs


def print_result(payload: Mapping[str, Any], machine_readable: bool) -> None:
    if machine_readable:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    status = payload.get("status") or payload.get("validation_summary", {}).get("status")
    print(f"status: {status}")
    for key in ["import_id", "row_count", "snapshot", "report"]:
        if payload.get(key) is not None:
            print(f"{key}: {payload[key]}")
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, Mapping):
        for name, path in artifacts.items():
            if path:
                print(f"{name}: {path}")


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    config = load_config(args.config)
    output_dir = args.output_dir.resolve()

    if args.mode == "compare-snapshots":
        if not args.snapshot_a or not args.snapshot_b:
            raise ImportFailure(
                "compare-snapshots requires --snapshot-a and --snapshot-b."
            )
        result = compare_snapshot_dicts(
            load_snapshot(args.snapshot_a), load_snapshot(args.snapshot_b)
        )
        return EXIT_OK, {"status": "complete", "comparison": result}

    if args.mode == "rebuild-classification":
        if not args.snapshot:
            raise ImportFailure("rebuild-classification requires --snapshot.")
        return EXIT_OK, rebuild_classification(
            args.snapshot, output_dir, config, dry_run=False
        )

    inputs = collect_inputs(args)
    if not inputs:
        raise ImportFailure("At least one of --overview, --content, or --video is required.")

    if args.mode == "inspect":
        result = inspect_inputs(inputs, config)
        result["status"] = result["validation_summary"]["status"]
        return (
            EXIT_VALIDATION
            if result["validation_summary"]["error_count"]
            else EXIT_OK,
            result,
        )

    exports = [parse_export(path, kind, config) for kind, path in inputs]
    validation = validation_report(exports)

    if args.mode == "validate-only":
        result = {
            "status": validation["status"],
            "validation": validation,
        }
        return (
            EXIT_VALIDATION if validation["error_count"] else EXIT_OK,
            result,
        )

    if validation["error_count"]:
        return EXIT_VALIDATION, {
            "status": "validation_failed",
            "validation": validation,
        }

    import_id, combined_hash = make_import_id(exports)
    previous = latest_snapshot(output_dir)

    if args.mode == "incremental-import" and not args.force:
        duplicate = already_imported(output_dir, combined_hash)
        if duplicate:
            return EXIT_OK, {
                "status": "already_imported",
                "import_id": duplicate.get("import_id"),
                "manifest": duplicate,
            }

    snapshot = build_snapshot(exports, config, import_id, combined_hash)

    if args.mode == "dry-run":
        return EXIT_OK, {
            "status": "dry_run",
            "import_id": import_id,
            "validation": snapshot["validation"],
            "reconciliation": snapshot["reconciliation"],
            "partial_day": snapshot["partial_day"],
            "statistics_summary": {
                "sample_size": snapshot["statistics"]["overall"]["sample_size"],
                "confidence": snapshot["statistics"]["overall"]["confidence"],
                "viral_outliers": snapshot["statistics"]["viral_outliers"]["count"],
            },
            "writes": [],
        }

    duplicate = already_imported(output_dir, combined_hash)
    if duplicate and not args.force:
        return EXIT_OK, {
            "status": "already_imported",
            "import_id": duplicate.get("import_id"),
            "manifest": duplicate,
        }

    artifacts = save_import_artifacts(
        snapshot, output_dir, previous_snapshot=previous
    )
    return EXIT_OK, {
        "status": "complete",
        "import_id": import_id,
        "artifacts": artifacts,
        "validation_status": snapshot["validation"]["status"],
        "reconciliation_status": snapshot["reconciliation"]["status"],
        "privacy_default": "private",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code, result = run(args)
    except ImportFailure as exc:
        print_result({"status": "error", "error": str(exc)}, args.json)
        return EXIT_USAGE
    except Exception as exc:
        print_result(
            {
                "status": "internal_error",
                "error": f"{type(exc).__name__}: {exc}",
            },
            args.json,
        )
        return EXIT_INTERNAL
    print_result(result, args.json)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
