# /// script
# requires-python = ">=3.10"
# dependencies = ["ruamel.yaml>=0.18"]
# ///
"""Parse epic files and deterministically generate or refresh sprint-status.yaml.

Prints ONLY JSON to stdout — argparse failures included. Errors are emitted as
JSON to stdout with a non-zero exit code. Writes are atomic (temp file, fsync,
``os.replace``) and the original file bytes are restored atomically if
post-write validation fails.

Subcommands:
  generate  Parse epics, merge with any existing status file, write the result.
            --dry-run reports (including drift: in_sync, illegal, orphans)
            without writing. --fresh ignores existing statuses for a pristine
            rebuild; --set key=status applies explicit, user-confirmed statuses
            on top — the repair path is allowed to downgrade.
  status    Summarize an existing status file: counts, risks, open action
            items, and the next recommended action. No writes.
  validate  Report whether an existing status file is structurally valid:
            parseable, recognized keys, legal statuses, well-formed
            action_items. No writes; exit 0 whether valid or not.

The LLM decides *which* files are epics (discovery is judgment); this script
owns everything after that decision: parsing, key derivation, ordering, status
preservation, story-file detection, action-item carry-over, and validation.
Legacy v6 statuses (drafted, contexted) are normalized on read everywhere, so
they merge and count by their modern meaning and are reported, never reset.
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

EPIC_RE = re.compile(r"^#{1,3}\s*Epic\s+(\d+)\s*:?\s*(.*?)\s*#*\s*$", re.IGNORECASE)
STORY_RE = re.compile(
    r"^#{2,4}\s*Story\s+(\d+)\.(\d+[a-z]?)\s*:?\s*(.*?)\s*#*\s*$", re.IGNORECASE
)
# Heading lines that mention Epic/Story but failed the strict patterns above.
SUSPECT_RE = re.compile(r"^#{1,4}\s.*\b(?:epic|story)\b", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s{0,3}(?:```|~~~)")

# The key grammar for sprint-status.yaml. The trailing [a-z]? matches
# split-story keys like 2-6a-...; bmad-retrospective's sprint_status.py reads
# the same file with the same grammar.
EPIC_KEY_RE = re.compile(r"^epic-(\d+)$")
RETRO_KEY_RE = re.compile(r"^epic-(\d+)-retrospective$")
STORY_KEY_RE = re.compile(r"^(\d+)-(\d+)([a-z]?)-.+")

STORY_RANK = {"backlog": 0, "ready-for-dev": 1, "in-progress": 2, "review": 3, "done": 4}
EPIC_RANK = {"backlog": 0, "in-progress": 1, "done": 2}
RETRO_RANK = {"optional": 0, "done": 1}
RANKS = {"epic": EPIC_RANK, "story": STORY_RANK, "retro": RETRO_RANK}
ACTION_STATUSES = ("open", "in-progress", "done")

# v6 wrote these; they still exist in the wild (v6-shims/bmad-create-story
# actively writes 'contexted'). Normalized on every read so no subcommand ever
# treats a valid legacy file as illegal or resets its progress.
LEGACY_STATUS = {"drafted": "ready-for-dev", "contexted": "in-progress"}

STALE_DAYS_DEFAULT = 7
DATE_FORMAT = "%m-%d-%Y %H:%M"
# Hand-edited files drift toward ISO stamps; accept them rather than silently
# disabling the staleness check.
STAMP_FORMATS = (DATE_FORMAT, "%Y-%m-%d %H:%M", "%Y-%m-%d")

# Kept byte-identical (modulo the leading "# ") with the STATUS DEFINITIONS
# block in sprint-status-template.yaml; test_sprint_plan.py asserts the two
# never drift.
HEADER_COMMENT = """\
STATUS DEFINITIONS:
==================
Epic Status:
  - backlog: Epic not yet started
  - in-progress: Epic actively being worked on
  - done: All stories in epic completed

Story Status:
  - backlog: Story only exists in epic file
  - ready-for-dev: Story file created, ready for development
  - in-progress: Developer actively working on implementation
  - review: Implementation complete, ready for review
  - done: Story completed

Retrospective Status:
  - optional: Can be completed but not required
  - done: Retrospective has been completed

Action Item Status:
  - open: Committed during a retrospective, not yet addressed
  - in-progress: Actively being worked on
  - done: Completed

WORKFLOW NOTES:
===============
- Epic transitions to 'in-progress' automatically when its first story starts (via build's sprint sync)
- Stories can be worked in parallel if team capacity allows
- Developer typically creates the next story after the previous one is 'done' to incorporate learnings
- Dev moves story to 'review', then runs code-review (fresh context, different LLM recommended)
- Retrospective appends its action items to action_items; the status view surfaces open ones
"""


def _fail(message, **extra):
    print(json.dumps({"ok": False, "error": message, **extra}, default=str))
    sys.exit(1)


class JsonArgumentParser(argparse.ArgumentParser):
    """Emit argparse failures on the JSON-only stdout contract, not usage text.

    Built with ``add_help=False`` everywhere: the built-in help action prints
    plain usage to stdout with exit 0, which would break the machine consumer
    this script serves. ``-h`` therefore routes through ``error()`` as an
    ordinary unrecognized argument; the skill's SKILL.md carries the usage a
    human needs.
    """

    def error(self, message):
        print(json.dumps({"ok": False, "error": f"argument error: {message}"}))
        sys.exit(2)


def _slug(text, maxlen=60):
    # Unicode-aware: a non-Latin title must keep its own characters in the key
    # rather than every such story collapsing onto one shared placeholder.
    slug = re.sub(r"[^\w]+", "-", str(text).lower(), flags=re.UNICODE).strip("-")
    slug = slug[:maxlen].strip("-")
    if not slug:
        # Nothing sluggable (punctuation/emoji only): a short content hash keeps
        # the key deterministic and distinct instead of a bare "untitled".
        slug = hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:8]
    return slug


def classify_key(key):
    """Return (kind, epic_num) for a recognized key, else None."""
    m = RETRO_KEY_RE.match(key)
    if m:
        return "retro", int(m.group(1))
    m = EPIC_KEY_RE.match(key)
    if m:
        return "epic", int(m.group(1))
    m = STORY_KEY_RE.match(key)
    if m:
        return "story", int(m.group(1))
    return None


def _story_sort_key(key):
    m = STORY_KEY_RE.match(key)
    if not m:
        return (10**9, 10**9, "", key)
    return (int(m.group(1)), int(m.group(2)), m.group(3), key)


def _normalize(raw):
    """Map a raw status through the legacy vocabulary. Returns (status, was_legacy)."""
    status = LEGACY_STATUS.get(raw, raw)
    return status, raw in LEGACY_STATUS


def parse_epics(paths):
    """Return (entries, warnings). Entries are (key, kind, epic_num) in file order."""
    epics = {}  # epic_num -> [story keys in order]
    warnings = []
    for path in paths:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            _fail(f"cannot read epic file {path}: {exc}")
        in_fence = False
        for lineno, line in enumerate(lines, 1):
            if FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            epic_m = EPIC_RE.match(line)
            if epic_m:
                epics.setdefault(int(epic_m.group(1)), [])
                continue
            story_m = STORY_RE.match(line)
            if story_m:
                epic_num = int(story_m.group(1))
                story_num = story_m.group(2)
                key = f"{epic_num}-{story_num}-{_slug(story_m.group(3))}"
                stories = epics.setdefault(epic_num, [])
                if key in stories:
                    warnings.append(f"duplicate story heading '{key}' at {path}:{lineno}")
                else:
                    stories.append(key)
                continue
            if SUSPECT_RE.match(line):
                warnings.append(f"unparsed Epic/Story-like heading at {path}:{lineno}: {line.strip()}")
    entries = []
    for epic_num in sorted(epics):
        entries.append((f"epic-{epic_num}", "epic", epic_num))
        for story_key in epics[epic_num]:
            entries.append((story_key, "story", epic_num))
        entries.append((f"epic-{epic_num}-retrospective", "retro", epic_num))
    return entries, warnings


def _make_yaml():
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    # Pin the emitter to the indentation the sprint-status template ships with.
    # Without this, ruamel re-dumps block sequences at its own default offset and
    # every write silently de-indents pre-existing, untouched action_items.
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.encoding = "utf-8"
    return yaml


def _load_existing(path):
    yaml = _make_yaml()
    if not Path(path).exists():
        return yaml, None
    try:
        with io.open(path, "r", encoding="utf-8") as fh:
            data = yaml.load(fh)
    except Exception as exc:
        _fail(f"existing status file is not valid YAML: {exc}", status_file=str(path))
    if data is not None and not isinstance(data, dict):
        _fail(
            f"existing status file is valid YAML but not a mapping (got {type(data).__name__})",
            status_file=str(path),
        )
    return yaml, data


def _merge_status(kind, computed, existing_raw, key, warnings, report):
    """Return the higher-ranked of computed/existing; never downgrade."""
    rank = RANKS[kind]
    if existing_raw is None:
        return computed
    existing, was_legacy = _normalize(existing_raw)
    if was_legacy:
        report["legacy_mapped"].append({"key": key, "from": existing_raw, "to": existing})
    if existing not in rank:
        warnings.append(f"illegal status '{existing_raw}' on '{key}' replaced with '{computed}'")
        report["illegal"].append({"key": key, "status": existing_raw})
        return computed
    return existing if rank[existing] >= rank[computed] else computed


def build_status(entries, existing_data, stories_dir, warnings):
    """Return (development_status CommentedMap, merge report dict)."""
    existing_status = {}
    if existing_data is not None:
        existing_status = dict(existing_data.get("development_status") or {})
    report = {
        "new_entries": [],
        "preserved": 0,
        "changed": 0,
        "upgraded_from_disk": [],
        "dropped_orphans": [],
        "legacy_mapped": [],
        "illegal": [],
    }
    # One directory scan instead of a stat() per story.
    story_files = set()
    if stories_dir and Path(stories_dir).is_dir():
        story_files = {p.name for p in Path(stories_dir).glob("*.md")}
    dev = CommentedMap()
    first_epic = True
    for key, kind, _epic_num in entries:
        default = {"epic": "backlog", "story": "backlog", "retro": "optional"}[kind]
        computed = default
        if kind == "story" and f"{key}.md" in story_files:
            computed = "ready-for-dev"
        merged = _merge_status(kind, computed, existing_status.get(key), key, warnings, report)
        if key not in existing_status:
            report["new_entries"].append(key)
        elif merged == _normalize(existing_status[key])[0]:
            report["preserved"] += 1
        else:
            report["changed"] += 1
        if kind == "story" and computed == "ready-for-dev" and existing_status.get(key) in (None, "backlog"):
            report["upgraded_from_disk"].append(key)
        dev[key] = merged
        if kind == "epic" and not first_epic:
            dev.yaml_set_comment_before_after_key(key, before="\n")
        if kind == "epic":
            first_epic = False
    computed_keys = {key for key, _, _ in entries}
    # Old statuses ride along so the LLM can transplant them after a rename —
    # the values would otherwise be destroyed by the write.
    report["dropped_orphans"] = [
        {"key": k, "status": existing_status[k]}
        for k in existing_status
        if k not in computed_keys
    ]
    report["in_sync"] = (
        not report["new_entries"]
        and not report["dropped_orphans"]
        and not report["illegal"]
        and not report["legacy_mapped"]
        and report["changed"] == 0
    )
    return dev, report


def _counts(dev):
    counts = {}
    for value in dev.values():
        counts[value] = counts.get(value, 0) + 1
    return counts


def _dump_bytes(yaml, doc):
    """Serialize before any file is touched, so a dump failure cannot leave a
    partial file anywhere."""
    buf = io.BytesIO()
    yaml.dump(doc, buf)
    return buf.getvalue()


def _atomic_write(path, payload, mode=None):
    """Replace ``path``'s contents with ``payload`` atomically.

    Temp file alongside the target, fsynced, taking the target's permission
    bits (mkstemp creates 0600, which would silently narrow the file), then
    renamed over it. ``path`` is resolved through symlinks first: renaming onto
    a symlink would detach the link and leave the real file stale.
    """
    path = os.path.realpath(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".sprint-status-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _parse_sets(pairs, valid_keys):
    """Validate --set key=status pairs against the generated plan and vocabulary."""
    parsed = []
    for pair in pairs:
        key, sep, status = pair.partition("=")
        if not sep or not key or not status:
            _fail(f"--set expects key=status, got '{pair}'")
        if key not in valid_keys:
            _fail(f"--set key '{key}' is not in the generated plan", valid_keys=sorted(valid_keys))
        kind, _ = classify_key(key)
        if status not in RANKS[kind]:
            _fail(
                f"--set status '{status}' is not legal for {kind} '{key}'",
                legal=sorted(RANKS[kind]),
            )
        parsed.append((key, status))
    return parsed


def cmd_generate(args):
    entries, warnings = parse_epics(args.epic_file)
    if not entries:
        _fail("no epics or stories parsed from the given epic files", epic_files=args.epic_file)
    yaml, existing = _load_existing(args.status_file)
    status_path = Path(args.status_file)
    original_bytes = status_path.read_bytes() if status_path.exists() else None
    original_mode = (os.stat(status_path).st_mode & 0o777) if status_path.exists() else None

    merge_source = None if args.fresh else existing
    dev, report = build_status(entries, merge_source, args.stories_dir, warnings)

    # Explicit, user-confirmed statuses (the fix flow). Applied last: repair is
    # the one path allowed to downgrade.
    explicit = _parse_sets(args.set or [], set(dev.keys()))
    for key, status in explicit:
        dev[key] = status
    report["explicit_set"] = [f"{k}={s}" for k, s in explicit]

    def _meta(field, arg_value, default):
        if arg_value is not None:
            return arg_value
        if existing is not None and existing.get(field):
            return str(existing[field])
        return default

    generated = args.date
    if existing is not None and existing.get("generated"):
        generated = str(existing["generated"])

    if existing is not None and not args.fresh:
        # Round-trip the existing document: unknown top-level keys and their
        # comments survive; only the managed fields and development_status are
        # replaced.
        doc = existing
    else:
        doc = CommentedMap()
        doc.yaml_set_start_comment(HEADER_COMMENT)
    doc["generated"] = generated
    doc["last_updated"] = args.date
    doc["project"] = args.project
    doc["project_key"] = _meta("project_key", args.project_key, "NOKEY")
    doc["tracking_system"] = _meta("tracking_system", args.tracking_system, "file-system")
    doc["story_location"] = _meta("story_location", args.story_location, args.stories_dir)
    doc["development_status"] = dev
    if "action_items" not in doc and existing is not None and existing.get("action_items") is not None:
        doc["action_items"] = existing["action_items"]
        doc.yaml_set_comment_before_after_key(
            "action_items",
            before="\nAction items committed during retrospectives (section created by the retrospective workflow)",
        )

    result = {
        "ok": True,
        "action": "generate",
        "status_file": str(args.status_file),
        "dry_run": bool(args.dry_run),
        "fresh": bool(args.fresh),
        "epics": sum(1 for _, kind, _ in entries if kind == "epic"),
        "stories": sum(1 for _, kind, _ in entries if kind == "story"),
        "counts": _counts(dev),
        "generated": generated,
        "last_updated": args.date,
        "warnings": warnings,
        **report,
    }

    if args.dry_run:
        print(json.dumps(result, default=str))
        return

    try:
        payload = _dump_bytes(yaml, doc)
        _atomic_write(args.status_file, payload, original_mode)
        verify_yaml = _make_yaml()
        with io.open(args.status_file, "r", encoding="utf-8") as fh:
            reread = verify_yaml.load(fh)
        if dict(reread.get("development_status") or {}) != {k: v for k, v in dev.items()}:
            raise ValueError("development_status mismatch after write")
        for field in ("generated", "last_updated", "project"):
            if str(reread.get(field)) != str(doc[field]):
                raise ValueError(f"{field} mismatch after write")
    except Exception as exc:
        if original_bytes is not None:
            try:
                _atomic_write(args.status_file, original_bytes, original_mode)
                restored = True
            except Exception:
                restored = False
        else:
            Path(args.status_file).unlink(missing_ok=True)
            restored = True
        _fail(f"write or validation failed, original {'restored' if restored else 'NOT restored'}: {exc}",
              restored=restored)
    print(json.dumps(result, default=str))


def _parse_stamp(value):
    from datetime import datetime

    for fmt in STAMP_FORMATS:
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    return None


def cmd_status(args):
    from datetime import timedelta

    _, data = _load_existing(args.status_file)
    if data is None:
        _fail("status file does not exist — run sprint planning to generate it",
              status_file=str(args.status_file))
    dev = dict(data.get("development_status") or {})
    if not dev:
        _fail("development_status missing or empty — re-run sprint planning",
              status_file=str(args.status_file))

    warnings = []
    counts = {"story": {}, "epic": {}, "retro": {}}
    by_status = {}
    legacy_mapped, illegal, unrecognized = [], [], []
    epic_nums, story_epic_nums = set(), set()
    epic_status, retro_status = {}, {}
    for key, raw in dev.items():
        key = str(key)
        parsed = classify_key(key)
        if parsed is None:
            unrecognized.append({"key": key, "status": raw})
            continue
        kind, epic_num = parsed
        status, was_legacy = _normalize(raw)
        if was_legacy:
            legacy_mapped.append({"key": key, "from": raw, "to": status})
        if status not in RANKS[kind]:
            illegal.append({"key": key, "status": raw})
            continue
        counts[kind][status] = counts[kind].get(status, 0) + 1
        if kind == "story":
            by_status.setdefault(status, []).append(key)
            story_epic_nums.add(epic_num)
        elif kind == "epic":
            epic_nums.add(epic_num)
            epic_status[epic_num] = status
        else:
            retro_status[epic_num] = status
    for stories in by_status.values():
        stories.sort(key=_story_sort_key)

    action_items = data.get("action_items") or []
    open_items = []
    for i, item in enumerate(action_items):
        if not isinstance(item, dict):
            warnings.append(f"action_items[{i}] is not a mapping and was skipped: {item!r}")
            continue
        status = item.get("status")
        if status not in ACTION_STATUSES:
            warnings.append(f"action_items[{i}] has a missing or unknown status ({status!r})")
            continue
        if status in ("open", "in-progress"):
            open_items.append({k: item.get(k) for k in ("epic", "action", "owner", "status")})

    risks = []
    stamp = data.get("last_updated") or data.get("generated")
    if args.date and stamp:
        now, then = _parse_stamp(args.date), _parse_stamp(stamp)
        if now is None or then is None:
            warnings.append(
                f"timestamp format not recognized (--date {args.date!r}, file {stamp!r}); "
                "staleness check skipped"
            )
        elif now - then > timedelta(days=args.stale_days):
            risks.append(f"sprint-status.yaml may be stale (last updated {stamp})")
    for stories in by_status.values():
        for key in stories:
            num = classify_key(key)[1]
            if num not in epic_nums:
                risks.append(f"orphaned story '{key}' has no epic-{num} entry")
    for num, status in epic_status.items():
        if status == "in-progress" and num not in story_epic_nums:
            risks.append(f"in-progress epic 'epic-{num}' has no stories")
    if by_status.get("review"):
        risks.append(f"{len(by_status['review'])} story(ies) in review — run bmad-code-review")
    if unrecognized:
        risks.append(f"{len(unrecognized)} unrecognized key(s) in development_status — run validate")

    recommendation = None
    if by_status.get("in-progress"):
        recommendation = {"skill": "bmad-build", "story_key": by_status["in-progress"][0],
                          "reason": "resume the in-progress story"}
    elif by_status.get("review"):
        recommendation = {"skill": "bmad-code-review", "story_key": by_status["review"][0],
                          "reason": "review the completed implementation"}
    elif by_status.get("ready-for-dev"):
        recommendation = {"skill": "bmad-build", "story_key": by_status["ready-for-dev"][0],
                          "reason": "start the next ready story"}
    elif by_status.get("backlog"):
        recommendation = {"skill": "bmad-build", "story_key": by_status["backlog"][0],
                          "reason": "start the first backlog story"}
    else:
        optional_retros = sorted(num for num, status in retro_status.items() if status == "optional")
        if optional_retros:
            recommendation = {"skill": "bmad-retrospective", "story_key": None,
                              "reason": f"all stories done — epic-{optional_retros[0]}-retrospective is still open"}

    print(json.dumps({
        "ok": True, "action": "status", "status_file": str(args.status_file),
        "project": data.get("project"), "project_key": data.get("project_key"),
        "tracking_system": data.get("tracking_system"),
        "generated": data.get("generated"), "last_updated": data.get("last_updated"),
        "stories": counts["story"], "epics": counts["epic"], "retrospectives": counts["retro"],
        "legacy_mapped": legacy_mapped, "illegal": illegal, "unrecognized": unrecognized,
        "open_action_items": open_items, "risks": risks, "warnings": warnings,
        "recommendation": recommendation,
        "all_done": recommendation is None,
    }, default=str))


def cmd_validate(args):
    problems = []
    legacy_mapped = []
    path = Path(args.status_file)
    if not path.exists():
        print(json.dumps({
            "ok": True, "action": "validate", "status_file": str(args.status_file),
            "valid": False, "problems": ["status file does not exist"], "legacy_mapped": [],
        }))
        return
    yaml = _make_yaml()
    try:
        with io.open(args.status_file, "r", encoding="utf-8") as fh:
            data = yaml.load(fh)
    except Exception as exc:
        print(json.dumps({
            "ok": True, "action": "validate", "status_file": str(args.status_file),
            "valid": False, "problems": [f"not valid YAML: {exc}"], "legacy_mapped": [],
        }, default=str))
        return
    if not isinstance(data, dict):
        problems.append(f"top level is not a mapping (got {type(data).__name__})")
    else:
        for field in ("generated", "last_updated", "project", "development_status"):
            if data.get(field) is None:
                problems.append(f"missing required key '{field}'")
        for field in ("generated", "last_updated"):
            value = data.get(field)
            if value is not None and _parse_stamp(value) is None:
                problems.append(f"'{field}' timestamp {str(value)!r} does not match '{DATE_FORMAT}'")
        dev = data.get("development_status")
        if dev is not None and not isinstance(dev, dict):
            problems.append("development_status is not a mapping")
        elif dev:
            for key, raw in dev.items():
                parsed = classify_key(str(key))
                if parsed is None:
                    problems.append(f"unrecognized key '{key}' (expected epic-N, N-M-slug, or epic-N-retrospective)")
                    continue
                kind, _ = parsed
                status, was_legacy = _normalize(raw)
                if was_legacy:
                    legacy_mapped.append({"key": str(key), "from": raw, "to": status})
                if status not in RANKS[kind]:
                    problems.append(f"illegal {kind} status {str(raw)!r} on '{key}'")
        elif isinstance(data.get("development_status"), dict):
            problems.append("development_status is empty")
        items = data.get("action_items")
        if items is not None:
            if not isinstance(items, list):
                problems.append("action_items is not a list")
            else:
                for i, item in enumerate(items):
                    if not isinstance(item, dict):
                        problems.append(f"action_items[{i}] is not a mapping")
                    elif item.get("status") not in ACTION_STATUSES:
                        problems.append(
                            f"action_items[{i}] has a missing or unknown status ({item.get('status')!r})"
                        )
    print(json.dumps({
        "ok": True, "action": "validate", "status_file": str(args.status_file),
        "valid": not problems, "problems": problems, "legacy_mapped": legacy_mapped,
    }, default=str))


def build_parser():
    parser = JsonArgumentParser(prog="sprint_plan.py", add_help=False)
    sub = parser.add_subparsers(dest="command", required=True, parser_class=JsonArgumentParser)

    gen = sub.add_parser("generate", add_help=False)
    gen.add_argument("--epic-file", action="append", required=True)
    gen.add_argument("--status-file", required=True)
    gen.add_argument("--stories-dir", required=True)
    gen.add_argument("--project", required=True)
    gen.add_argument("--date", required=True)
    gen.add_argument("--project-key", default=None)
    gen.add_argument("--tracking-system", default=None)
    gen.add_argument("--story-location", default=None)
    gen.add_argument("--dry-run", action="store_true")
    gen.add_argument("--fresh", action="store_true")
    gen.add_argument("--set", action="append", metavar="KEY=STATUS")
    gen.set_defaults(func=cmd_generate)

    st = sub.add_parser("status", add_help=False)
    st.add_argument("--status-file", required=True)
    st.add_argument("--date", default=None)
    st.add_argument("--stale-days", type=int, default=STALE_DAYS_DEFAULT)
    st.set_defaults(func=cmd_status)

    val = sub.add_parser("validate", add_help=False)
    val.add_argument("--status-file", required=True)
    val.set_defaults(func=cmd_validate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    if sys.platform == "win32":
        # Piped output on Windows defaults to a legacy code page, not UTF-8.
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
