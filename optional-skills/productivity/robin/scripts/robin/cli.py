from __future__ import annotations

import argparse
import json
from datetime import date
from typing import NoReturn

from robin.config import load_config, load_index, save_index, topics_path
from robin.doctor import run_doctor
from robin.entry_ops import (
    EntryOperationError,
    append_entry_to_file,
    delete_entry,
    find_duplicate_candidates,
    duplicate_payload,
    move_entry,
    remove_new_media_if_present,
)
from robin.index import ensure_entry_in_index, rebuild_index
from robin.media import copy_image_to_media, is_video_url
from robin.parser import RobinEntryParseError, SEPARATOR, load_all_entries, load_topic_entries, topic_slug, topic_to_filename
from robin.review_logic import mark_surfaced, pick_best_candidate, rate_item
from robin.search_logic import filter_by_tags, search_entries
from robin.serializer import build_media_entry, build_text_entry, generate_entry_id, serialize_entry


def _error(message: str, *, as_json: bool) -> NoReturn:
    if as_json:
        print(json.dumps({"error": message}, indent=2))
    else:
        print(f"ERROR: {message}")
    raise SystemExit(1)


def _duplicate_error(matches: list, *, as_json: bool, media_source_to_cleanup: str = "", explicit_state_dir: str | None = None) -> NoReturn:
    remove_new_media_if_present(explicit_state_dir, media_source_to_cleanup)
    payload = duplicate_payload(matches)
    message = "Duplicate Robin entry found. Rerun with --allow-duplicate if this is intentional."
    if as_json:
        print(json.dumps({"error": message, "duplicates": payload}, indent=2))
    else:
        print(f"ERROR: {message}")
        for entry in payload:
            location = f"{entry['topic']} / {entry['id']}"
            detail = entry.get("source") or entry.get("media_source") or entry.get("description") or "No detail"
            print(f"  - {location}: {detail}")
    raise SystemExit(1)


def _emit_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


def _recall_value(value: str) -> str:
    return value.strip() if value.strip() else "Not provided"


def _saved_on_value(value: str) -> str:
    raw = _recall_value(value)
    if raw == "Not provided":
        return raw

    try:
        saved_on = date.fromisoformat(raw)
    except ValueError:
        return raw

    delta_days = (date.today() - saved_on).days
    if delta_days >= 0:
        return f"{raw} ({delta_days} days ago)"
    return f"{raw} (in {abs(delta_days)} days)"


def _print_recall(entry) -> None:
    source = entry.source.strip() or entry.media_source.strip()
    print("📚 Robin Recall")
    print()
    print(f"Topic: {_recall_value(entry.topic)}")
    print(f"Type: {_recall_value(entry.entry_type)}")
    print(f"Source: {_recall_value(source)}")
    print(f"Creator: {_recall_value(entry.creator)}")
    print(f"Saved on: {_saved_on_value(entry.date_added)}")
    print()
    print("Description:")
    print(_recall_value(entry.description))
    print()
    print("Body:")
    print(_recall_value(entry.body))


def _add_to_topic(config: dict, explicit_state_dir: str | None, topic: str, entry_text: str) -> str:
    base = topics_path(config, explicit_state_dir)
    base.mkdir(parents=True, exist_ok=True)
    filepath = base / topic_to_filename(topic)
    append_entry_to_file(filepath, entry_text)
    return filepath.name


def _validate_serialized_entry(entry_text: str, *, as_json: bool) -> None:
    # The topic file writer appends a trailing newline after serialization, so
    # validate against entry_text + "\n" to catch a body that ends with a
    # standalone "***" line as well as separator occurrences in the middle.
    if SEPARATOR in entry_text + "\n":
        _error(
            "Entry content cannot contain a standalone '***' line because Robin uses it as an internal separator.",
            as_json=as_json,
        )


def _add_state_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--state-dir",
        help="Directory containing robin-config.json and robin-review-index.json",
    )


def add_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Add an entry to Robin's commonplace book")
    _add_state_dir_arg(parser)
    parser.add_argument("--topic", required=True, help="Topic name")
    parser.add_argument("--entry-type", choices=["text", "image", "video"], default="text", help="Entry type")
    parser.add_argument("--content", default="", help="Content to file")
    parser.add_argument("--description", required=True, help="2-3 sentence context to store with the entry")
    parser.add_argument("--source", help="Source URL")
    parser.add_argument("--media-path", help="Local image file path for image entries or text entry attachments")
    parser.add_argument("--media-url", help="Remote video URL for video entries")
    parser.add_argument("--creator", help="Required for media entries")
    parser.add_argument("--published-at", help="Required for media entries")
    parser.add_argument("--summary", help="Required for media entries")
    parser.add_argument("--note", help="Robin note")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--allow-duplicate", action="store_true", help="Save even if Robin finds a likely duplicate")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    config = load_config(args.state_dir)
    index = load_index(args.state_dir)

    topic = args.topic.strip()
    date_added = str(date.today())
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    entry_type = args.entry_type

    if entry_type == "text":
        has_text_payload = any(
            [
                args.content.strip(),
                (args.note or "").strip(),
                (args.source or "").strip(),
                (args.media_path or "").strip(),
            ]
        )
        if not has_text_payload:
            _error("Text entries require --content, --note, --source, or --media-path.", as_json=args.json)
        if args.media_url:
            _error("Text entries do not accept --media-url. Attach local images with --media-path.", as_json=args.json)
        entry_id = generate_entry_id(date_added) if args.media_path else None
        media_source = ""
        if args.media_path and entry_id:
            try:
                media_source = copy_image_to_media(config, args.state_dir, topic, entry_id, args.media_path)
            except ValueError as exc:
                _error(str(exc), as_json=args.json)
            except OSError as exc:
                _error(f"Failed to copy image into vault: {exc}", as_json=args.json)
        entry = build_text_entry(
            topic=topic,
            content=args.content.strip(),
            description=args.description.strip(),
            source=args.source.strip() if args.source else None,
            media_source=media_source,
            note=args.note.strip() if args.note else None,
            tags=tags,
            date_added=date_added,
            entry_id=entry_id,
        )
    else:
        missing = [
            flag
            for flag, value in (
                ("--creator", args.creator),
                ("--published-at", args.published_at),
                ("--summary", args.summary),
            )
            if not (value or "").strip()
        ]
        if missing:
            _error(f"Media entries require {', '.join(missing)}.", as_json=args.json)

        entry_id = generate_entry_id(date_added)
        if entry_type == "image":
            if not args.media_path:
                _error("Image entries require --media-path.", as_json=args.json)
            if args.media_url:
                _error("Image entries do not accept --media-url.", as_json=args.json)
            try:
                media_source = copy_image_to_media(config, args.state_dir, topic, entry_id, args.media_path)
            except ValueError as exc:
                _error(str(exc), as_json=args.json)
            except OSError as exc:
                _error(f"Failed to copy image into vault: {exc}", as_json=args.json)
        else:
            if args.media_path:
                _error("Uploaded or local video files are not supported. Pass a video URL with --media-url.", as_json=args.json)
            if not args.media_url:
                _error("Video entries require --media-url.", as_json=args.json)
            if not is_video_url(args.media_url):
                _error("Video entries require a valid http(s) URL.", as_json=args.json)
            media_source = args.media_url.strip()

        entry = build_media_entry(
            topic=topic,
            media_kind=entry_type,
            media_source=media_source,
            description=args.description.strip(),
            creator=args.creator.strip(),
            published_at=args.published_at.strip(),
            summary=args.summary.strip(),
            content=args.content.strip(),
            source=args.source.strip() if args.source else None,
            note=args.note.strip() if args.note else None,
            tags=tags,
            date_added=date_added,
            entry_id=entry_id,
        )

    serialized_entry = serialize_entry(entry)
    _validate_serialized_entry(serialized_entry, as_json=args.json)
    if not args.allow_duplicate:
        try:
            matches = find_duplicate_candidates(config, args.state_dir, entry)
        except RobinEntryParseError as exc:
            remove_new_media_if_present(args.state_dir, entry.media_source)
            _error(str(exc), as_json=args.json)
        if matches:
            _duplicate_error(
                matches,
                as_json=args.json,
                media_source_to_cleanup=entry.media_source,
                explicit_state_dir=args.state_dir,
            )
    try:
        filename = _add_to_topic(config, args.state_dir, topic, serialized_entry)
    except OSError as exc:
        _error(f"Failed to write topic file: {exc}", as_json=args.json)
    ensure_entry_in_index(entry, index)
    try:
        save_index(index, args.state_dir)
    except OSError as exc:
        _error(f"Failed to write review index: {exc}", as_json=args.json)

    if args.json:
        print(
            json.dumps(
                {
                    "id": entry.entry_id,
                    "topic": entry.topic,
                    "filename": filename,
                    "entry_type": entry.entry_type,
                    "media_source": entry.media_source,
                    "description": entry.description,
                },
                indent=2,
            )
        )
        return

    print(f"✓ Filed {entry.entry_id} under [{topic}]({filename})")


def review_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Robin review system")
    _add_state_dir_arg(parser)
    parser.add_argument("--status", action="store_true", help="Show review status")
    parser.add_argument("--rate", nargs=2, metavar=("ID", "RATING"), help="Rate an item by stable entry id")
    parser.add_argument("--active-review", action="store_true", help="Mark surfaced item as awaiting a rating")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    config = load_config(args.state_dir)
    index = load_index(args.state_dir)

    if args.rate:
        entry_id, rating = args.rate
        try:
            rating_value = int(rating)
        except ValueError:
            _error("Rating must be a number between 1 and 5.", as_json=args.json)
        try:
            item = rate_item(index, entry_id, rating_value)
        except KeyError:
            _error(f"Item '{entry_id}' not found in index", as_json=args.json)
        except ValueError as exc:
            _error(str(exc), as_json=args.json)
        try:
            save_index(index, args.state_dir)
        except OSError as exc:
            _error(f"Failed to write review index: {exc}", as_json=args.json)
        if args.json:
            _emit_json(item)
            return
        print(f"✓ Rated {entry_id}: {item['rating']}/5")
        return

    if args.status:
        index_items = index.get("items", {})
        total = len(index_items)
        min_items = config.get("min_items_before_review", 30)
        rated = sum(1 for item in index_items.values() if item.get("rating") is not None)
        if args.json:
            print(json.dumps({
                "total_items": total,
                "rated": rated,
                "unrated": total - rated,
                "min_items_before_review": min_items,
                "ready": total >= min_items,
            }, indent=2))
            return
        print("Review status:")
        print(f"  Total items:   {total}")
        print(f"  Rated:         {rated}")
        print(f"  Unrated:       {total - rated}")
        print(f"  Min to review: {min_items}")
        print(f"  Ready:         {'YES' if total >= min_items else 'NO'}")
        return

    total = len(index.get("items", {}))
    min_items = config.get("min_items_before_review", 30)
    if total < min_items:
        if args.json:
            _emit_json(
                {
                    "status": "skip",
                    "reason": "not_enough_items",
                    "total_items": total,
                    "min_items_before_review": min_items,
                }
            )
            return
        print(f"SKIP: {total} items (need {min_items})")
        return

    try:
        entries = load_all_entries(config, args.state_dir)
    except RobinEntryParseError as exc:
        _error(str(exc), as_json=args.json)
    try:
        candidate = pick_best_candidate(index, entries, config)
    except ValueError as exc:
        _error(f"Review index contains an invalid timestamp: {exc}", as_json=args.json)
    if candidate is None:
        if args.json:
            _emit_json({"status": "skip", "reason": "no_eligible_items"})
            return
        print("SKIP: No eligible items (all recently surfaced or not indexed)")
        return

    item, entry = candidate
    try:
        item = mark_surfaced(index, entry.entry_id, awaiting_rating=args.active_review)
    except KeyError:
        _error(f"Item '{entry.entry_id}' not found in index", as_json=args.json)
    try:
        save_index(index, args.state_dir)
    except OSError as exc:
        _error(f"Failed to write review index: {exc}", as_json=args.json)
    if args.json:
        _emit_json({
            "status": "ok",
            "id": entry.entry_id,
            "topic": entry.topic,
            "date_added": entry.date_added,
            "entry_type": entry.entry_type,
            "media_kind": entry.media_kind,
            "media_source": entry.media_source,
            "source": entry.source,
            "description": entry.description,
            "creator": entry.creator,
            "published_at": entry.published_at,
            "summary": entry.summary,
            "tags": entry.tags,
            "body": entry.body,
            "rating": item.get("rating"),
            "times_surfaced": item.get("times_surfaced", 0),
        })
        return

    _print_recall(entry)


def search_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search Robin's commonplace book")
    _add_state_dir_arg(parser)
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--topic", help="Filter by topic name")
    parser.add_argument("--tags", help="Comma-separated tag filter")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    config = load_config(args.state_dir)
    index = load_index(args.state_dir)

    if args.topic:
        topic_path = topics_path(config, args.state_dir) / topic_to_filename(args.topic)
        if topic_path.exists():
            try:
                entries = load_topic_entries(topic_path)
            except RobinEntryParseError as exc:
                _error(str(exc), as_json=args.json)
        else:
            entries = []
        heading = f"Topic '{args.topic}': {len(entries)} entries"
    else:
        try:
            entries = load_all_entries(config, args.state_dir)
        except RobinEntryParseError as exc:
            _error(str(exc), as_json=args.json)
        heading = f"Total: {len(entries)} entries"

    if args.tags:
        tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
        if not tags:
            _error("Provide at least one non-empty tag.", as_json=args.json)
        entries = filter_by_tags(entries, tags)
        heading = f"Tags [{', '.join(tags)}]: {len(entries)} results"
    elif args.query:
        entries = search_entries(entries, args.query)
        heading = f"Query '{args.query}': {len(entries)} results"

    index_items = index.get("items", {})
    if args.json:
        print(json.dumps({
            "count": len(entries),
            "entries": [
                {
                    "id": entry.entry_id,
                    "topic": entry.topic,
                    "date_added": entry.date_added,
                    "entry_type": entry.entry_type,
                    "media_kind": entry.media_kind,
                    "media_source": entry.media_source,
                    "source": entry.source,
                    "description": entry.description,
                    "creator": entry.creator,
                    "published_at": entry.published_at,
                    "summary": entry.summary,
                    "tags": entry.tags,
                    "rating": index_items.get(entry.entry_id, {}).get("rating"),
                    "body": entry.body,
                }
                for entry in entries
            ],
        }, indent=2))
        return

    print(heading)
    print()
    for entry in entries:
        rating = index_items.get(entry.entry_id, {}).get("rating")
        print(f"[{entry.topic}.md] {entry.entry_id} / {entry.date_added}  ★{rating or '—'}")
        if entry.entry_type != "text":
            print(f"  Type: {entry.entry_type}")
        if entry.media_source:
            print(f"  Media: {entry.media_source}")
        if entry.source:
            print(f"  Source: {entry.source}")
        if entry.creator:
            print(f"  Creator: {entry.creator}")
        if entry.published_at:
            print(f"  Published: {entry.published_at}")
        if entry.summary:
            print(f"  Summary: {entry.summary}")
        if entry.description:
            print(f"  Description: {entry.description}")
        if entry.tags:
            print(f"  Tags: {', '.join(entry.tags)}")
        body = entry.body.replace("\n", " ").strip()
        print(f"  {body[:200]}{'...' if len(body) > 200 else ''}")
        print()


def topics_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="List all Robin topics")
    _add_state_dir_arg(parser)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    config = load_config(args.state_dir)
    index = load_index(args.state_dir)
    try:
        entries = load_all_entries(config, args.state_dir)
    except RobinEntryParseError as exc:
        _error(str(exc), as_json=args.json)

    topics: dict[str, dict] = {}
    index_items = index.get("items", {})
    for entry in entries:
        topic_stats = topics.setdefault(
            entry.topic,
            {"topic": entry.topic, "filename": f"{entry.topic}.md", "entries": 0, "rated": 0, "unrated": 0},
        )
        topic_stats["entries"] += 1
        if index_items.get(entry.entry_id, {}).get("rating") is None:
            topic_stats["unrated"] += 1
        else:
            topic_stats["rated"] += 1

    ordered_topics = [topics[key] for key in sorted(topics)]
    if args.json:
        print(json.dumps(ordered_topics, indent=2))
        return

    if not ordered_topics:
        print("No topics yet. Start filing things with Robin!")
        return

    total_entries = sum(topic["entries"] for topic in ordered_topics)
    print(f"{len(ordered_topics)} topics, {total_entries} total entries\n")
    visual_cap = 10
    for topic in ordered_topics:
        rated_visual = min(topic["rated"], visual_cap)
        unrated_visual = min(max(visual_cap - rated_visual, 0), topic["unrated"])
        stars = "★" * rated_visual + "☆" * unrated_visual
        overflow = topic["entries"] - (rated_visual + unrated_visual)
        if overflow > 0:
            stars = f"{stars}+{overflow}"
        print(f"  {topic['topic']}")
        print(f"    {topic['entries']} entries  {stars}  {topic['rated']} rated / {topic['unrated']} unrated")
        print()


def entries_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Move or delete Robin entries")
    _add_state_dir_arg(parser)
    parser.add_argument("--delete", metavar="ID", help="Delete an entry by stable entry id")
    parser.add_argument("--move", metavar="ID", help="Move an entry by stable entry id")
    parser.add_argument("--topic", help="Destination topic for --move")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    if bool(args.delete) == bool(args.move):
        _error("Pass exactly one of --delete ID or --move ID.", as_json=args.json)
    if args.delete and args.topic:
        _error("--topic is only valid with --move.", as_json=args.json)
    if args.move and not (args.topic or "").strip():
        _error("--move requires --topic.", as_json=args.json)

    config = load_config(args.state_dir)
    index = load_index(args.state_dir)

    try:
        if args.delete:
            payload = delete_entry(config, args.state_dir, index, args.delete)
        else:
            payload = move_entry(config, args.state_dir, index, args.move, args.topic or "")
        save_index(index, args.state_dir)
    except RobinEntryParseError as exc:
        _error(str(exc), as_json=args.json)
    except EntryOperationError as exc:
        _error(str(exc), as_json=args.json)
    except OSError as exc:
        _error(f"Failed to update entry files: {exc}", as_json=args.json)

    if args.json:
        _emit_json(payload)
        return
    if payload["status"] == "deleted":
        print(f"✓ Deleted {payload['id']} from {payload['filename']}")
    else:
        print(f"✓ Moved {payload['id']} from {payload['from_topic']} to {payload['to_topic']}")


def _print_doctor_report(payload: dict) -> None:
    print(f"Robin doctor: {'OK' if payload['ok'] else 'ISSUES FOUND'}")
    print(f"Errors: {payload['errors']}  Warnings: {payload['warnings']}")
    if not payload["diagnostics"]:
        print("No issues found.")
        return

    print()
    for item in payload["diagnostics"]:
        location = f" ({item['path']})" if item.get("path") else ""
        print(f"[{item['level'].upper()}] {item['code']}: {item['message']}{location}")


def doctor_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check a Robin library for common health issues")
    _add_state_dir_arg(parser)
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    payload = run_doctor(args.state_dir)
    if args.json:
        _emit_json(payload)
    else:
        _print_doctor_report(payload)
    raise SystemExit(0 if payload["ok"] else 1)


def reindex_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rebuild Robin's review index")
    _add_state_dir_arg(parser)
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    config = load_config(args.state_dir)
    old_index = load_index(args.state_dir)
    if not args.json:
        print("Scanning topic files...")
    try:
        entries = load_all_entries(config, args.state_dir)
    except RobinEntryParseError as exc:
        _error(str(exc), as_json=args.json)
    new_index = rebuild_index(entries, old_index)
    try:
        save_index(new_index, args.state_dir)
    except OSError as exc:
        _error(f"Failed to write review index: {exc}", as_json=args.json)

    rated = sum(1 for item in new_index["items"].values() if item.get("rating") is not None)
    if args.json:
        print(json.dumps({
            "entries_found": len(entries),
            "items_indexed": len(new_index["items"]),
            "rated": rated,
            "unrated": len(new_index["items"]) - rated,
        }, indent=2))
        return

    print(f"Found {len(entries)} entries")
    print(f"✓ Index rebuilt: {len(new_index['items'])} items, {rated} rated, {len(new_index['items']) - rated} unrated")
