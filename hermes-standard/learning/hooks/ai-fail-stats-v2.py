#!/usr/bin/env python3
import datetime
import json
import os
import sys


DEFAULT_STATS_DIR = os.path.expanduser("~/.claude/ai-fail-stats")
MINIMAL_KEYWORDS = {
    "version": 2,
    "repeat_warn_threshold": 3,
    "targets": {
        "hermes": ["fuck you hermes"],
        "claude": ["fuck you ai"],
    },
    "generic_curse": [],
    "jargon_markers": ["ภาษาคน"],
    "disabled": [],
}


def get_stats_dir():
    configured = os.environ.get("AI_FAIL_STATS_DIR") or DEFAULT_STATS_DIR
    return os.path.expanduser(configured)


def _as_clean_list(value):
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = str(item).strip().lower()
        if text:
            cleaned.append(text)
    return cleaned


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    return None


def _merge_target_phrase(targets, name, phrase):
    name = str(name).strip().lower()
    phrase = str(phrase).strip().lower()
    if not name or not phrase:
        return
    targets.setdefault(name, [])
    if phrase not in targets[name]:
        targets[name].append(phrase)


def normalize_keywords(data):
    if not isinstance(data, dict):
        data = MINIMAL_KEYWORDS

    disabled = set(_as_clean_list(data.get("disabled")))
    targets = {}
    raw_targets = data.get("targets")

    if isinstance(raw_targets, dict):
        for name, phrases in raw_targets.items():
            target_name = str(name).strip().lower()
            for phrase in _as_clean_list(phrases):
                if phrase not in disabled:
                    _merge_target_phrase(targets, target_name, phrase)
    elif isinstance(raw_targets, list):
        for name in raw_targets:
            target_name = str(name).strip().lower()
            if target_name == "ai":
                target_name = "claude"
            for prefix in ("fuck you", "fuck u", "f u"):
                phrase = "%s %s" % (prefix, str(name).strip().lower())
                if phrase not in disabled:
                    _merge_target_phrase(targets, target_name, phrase)

    generic_source = data.get("generic_curse")
    if generic_source is None:
        generic_source = data.get("keywords")
    generic_curse = [
        phrase for phrase in _as_clean_list(generic_source) if phrase not in disabled
    ]
    jargon_markers = [
        phrase for phrase in _as_clean_list(data.get("jargon_markers")) if phrase not in disabled
    ]

    try:
        threshold = int(data.get("repeat_warn_threshold", 3))
    except Exception:
        threshold = 3

    return {
        "version": 2,
        "repeat_warn_threshold": threshold,
        "targets": targets,
        "generic_curse": generic_curse,
        "jargon_markers": jargon_markers,
        "disabled": sorted(disabled),
    }


def load_keywords(stats_dir=None, fallback_path=None):
    base_dir = os.path.expanduser(stats_dir or get_stats_dir())
    local_path = os.path.join(base_dir, "curse-keywords.json")
    bundled_path = fallback_path or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "curse-keywords.json")
    )

    for path in (local_path, bundled_path):
        data = _read_json(path)
        if data is not None:
            return normalize_keywords(data)
    return normalize_keywords(MINIMAL_KEYWORDS)


def _target_category(target):
    if target == "hermes":
        return "hermes-fail"
    if target == "claude":
        return "ai-fail"
    return "target:%s" % target


def detect_hits(prompt, keywords=None):
    low = (prompt or "").lower()
    book = normalize_keywords(keywords or load_keywords())
    hits = []

    curse_hit = None
    for target, phrases in book.get("targets", {}).items():
        for phrase in phrases:
            if phrase and phrase in low:
                curse_hit = {
                    "category": _target_category(target),
                    "phrase": phrase,
                    "target": target,
                }
                break
        if curse_hit:
            break

    if curse_hit is None:
        for phrase in book.get("generic_curse", []):
            if phrase and phrase in low:
                curse_hit = {
                    "category": "curse-generic",
                    "phrase": phrase,
                    "target": "-",
                }
                break

    if curse_hit:
        hits.append(curse_hit)

    for phrase in book.get("jargon_markers", []):
        if phrase and phrase in low:
            hits.append({"category": "jargon", "phrase": phrase, "target": "-"})
            break

    return hits


def _load_counts(path):
    data = _read_json(path)
    if data is None:
        return {}
    counts = {}
    for key, value in data.items():
        try:
            counts[str(key)] = int(value)
        except Exception:
            counts[str(key)] = 0
    return counts


def _host_name():
    try:
        return os.uname().nodename
    except Exception:
        return "unknown"


def _utc_timestamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _total_counts(counts):
    total = 0
    for value in counts.values():
        try:
            total += int(value)
        except Exception:
            pass
    return total


def _format_counts(counts):
    order = ["jargon", "hermes-fail", "ai-fail", "curse-generic"]
    seen = set()
    parts = []
    for category in order:
        seen.add(category)
        parts.append("%s %s" % (category, int(counts.get(category, 0))))
    for category in sorted(counts):
        if category not in seen:
            parts.append("%s %s" % (category, int(counts.get(category, 0))))
    return " · ".join(parts)


def write_hits(hits, cwd, stats_dir=None):
    output_dir = os.path.expanduser(stats_dir or get_stats_dir())
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log.jsonl")
    counts_path = os.path.join(output_dir, "counts.json")
    counts = _load_counts(counts_path)
    host = _host_name()
    ts = _utc_timestamp()

    with open(log_path, "a", encoding="utf-8") as handle:
        for hit in hits:
            row = {
                "ts": ts,
                "host": host,
                "cwd": cwd,
                "category": hit["category"],
                "phrase": hit["phrase"],
                "target": hit.get("target") or "-",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[hit["category"]] = int(counts.get(hit["category"], 0)) + 1

    with open(counts_path, "w", encoding="utf-8") as handle:
        json.dump(counts, handle, ensure_ascii=False, indent=2)

    return counts, host


def build_response(counts, host):
    total = _total_counts(counts)
    count_text = _format_counts(counts)
    return {
        "systemMessage": "📊 บันทึกแล้ว (%s) · สะสมเครื่องนี้ %s ครั้ง — %s"
        % (host, total, count_text),
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "[สถิติ AI พลาด] เจ้าของเพิ่งตำหนิเรื่องเดิม (เครื่อง %s สะสม %s ครั้ง: %s). อย่าทำผิดซ้ำ — โดยเฉพาะต้องพูดภาษาคน แปลศัพท์เทคนิคทันที"
            % (host, total, count_text),
        },
    }


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    if not isinstance(data, dict):
        return 0

    hits = detect_hits(data.get("prompt") or "")
    if not hits:
        return 0

    try:
        counts, host = write_hits(hits, data.get("cwd") or "")
        print(json.dumps(build_response(counts, host), ensure_ascii=False))
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
