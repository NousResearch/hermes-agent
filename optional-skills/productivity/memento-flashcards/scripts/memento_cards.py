#!/usr/bin/env python3
"""Memento card storage, spaced-repetition engine, and CSV I/O.

Stdlib-only. All output is JSON for agent parsing.
Data file: $HERMES_HOME/skills/productivity/memento-flashcards/data/cards.json
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
DATA_DIR = _HERMES_HOME / "skills" / "productivity" / "memento-flashcards" / "data"
CARDS_FILE = DATA_DIR / "cards.json"

RETIRED_SENTINEL = "9999-12-31T23:59:59+00:00"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _empty_store() -> dict:
    return {"cards": [], "version": 1}


def _load() -> dict:
    if not CARDS_FILE.exists():
        return _empty_store()
    try:
        with open(CARDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "cards" not in data:
            return _empty_store()
        return data
    except (json.JSONDecodeError, OSError):
        return _empty_store()


def _save(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, CARDS_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _out(obj: object) -> None:
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


# ── Subcommands ──────────────────────────────────────────────────────────────

def cmd_add(args: argparse.Namespace) -> None:
    data = _load()
    now = _now()
    card = {
        "id": str(uuid.uuid4()),
        "question": args.question,
        "answer": args.answer,
        "collection": args.collection or "General",
        "status": "learning",
        "ease_streak": 0,
        "next_review_at": _iso(now),
        "created_at": _iso(now),
        "video_id": None,
        "last_user_answer": None,
    }
    data["cards"].append(card)
    _save(data)
    _out({"ok": True, "card": card})


def cmd_add_quiz(args: argparse.Namespace) -> None:
    data = _load()
    now = _now()

    try:
        questions = json.loads(args.questions)
    except json.JSONDecodeError as exc:
        _out({"ok": False, "error": f"Invalid JSON for --questions: {exc}"})
        sys.exit(1)

    # Dedup: skip if cards with this video_id already exist
    existing_ids = {c["video_id"] for c in data["cards"] if c.get("video_id")}
    if args.video_id in existing_ids:
        existing = [c for c in data["cards"] if c.get("video_id") == args.video_id]
        _out({"ok": True, "skipped": True, "reason": "duplicate_video_id", "existing_count": len(existing), "cards": existing})
        return

    created = []
    for qa in questions:
        card = {
            "id": str(uuid.uuid4()),
            "question": qa["question"],
            "answer": qa["answer"],
            "collection": args.collection or "Quiz",
            "status": "learning",
            "ease_streak": 0,
            "next_review_at": _iso(now),
            "created_at": _iso(now),
            "video_id": args.video_id,
            "last_user_answer": None,
        }
        data["cards"].append(card)
        created.append(card)

    _save(data)
    _out({"ok": True, "created_count": len(created), "cards": created})


def cmd_due(args: argparse.Namespace) -> None:
    data = _load()
    now = _now()
    due = []
    for card in data["cards"]:
        if card["status"] == "retired":
            continue
        review_at = _parse_iso(card["next_review_at"])
        if review_at <= now:
            if args.collection and card["collection"] != args.collection:
                continue
            due.append(card)
    _out({"ok": True, "count": len(due), "cards": due})


def cmd_rate(args: argparse.Namespace) -> None:
    data = _load()
    now = _now()
    card = None
    for c in data["cards"]:
        if c["id"] == args.id:
            card = c
            break
    if not card:
        _out({"ok": False, "error": f"Card not found: {args.id}"})
        sys.exit(1)

    rating = args.rating
    user_answer = getattr(args, "user_answer", None)
    if user_answer is not None:
        card["last_user_answer"] = user_answer

    if rating == "retire":
        card["status"] = "retired"
        card["next_review_at"] = RETIRED_SENTINEL
        card["ease_streak"] = 0
    elif rating == "hard":
        card["next_review_at"] = _iso(now + timedelta(days=1))
        card["ease_streak"] = 0
    elif rating == "good":
        card["next_review_at"] = _iso(now + timedelta(days=3))
        card["ease_streak"] = 0
    elif rating == "easy":
        card["next_review_at"] = _iso(now + timedelta(days=7))
        card["ease_streak"] = card.get("ease_streak", 0) + 1
        if card["ease_streak"] >= 3:
            card["status"] = "retired"

    _save(data)
    _out({"ok": True, "card": card})


def cmd_list(args: argparse.Namespace) -> None:
    data = _load()
    cards = data["cards"]
    if args.collection:
        cards = [c for c in cards if c["collection"] == args.collection]
    if args.status:
        cards = [c for c in cards if c["status"] == args.status]
    _out({"ok": True, "count": len(cards), "cards": cards})


def cmd_stats(args: argparse.Namespace) -> None:
    data = _load()
    now = _now()
    total = len(data["cards"])
    learning = sum(1 for c in data["cards"] if c["status"] == "learning")
    retired = sum(1 for c in data["cards"] if c["status"] == "retired")
    due_now = 0
    for c in data["cards"]:
        if c["status"] != "retired" and _parse_iso(c["next_review_at"]) <= now:
            due_now += 1

    collections: dict[str, int] = {}
    for c in data["cards"]:
        name = c["collection"]
        collections[name] = collections.get(name, 0) + 1

    _out({
        "ok": True,
        "total": total,
        "learning": learning,
        "retired": retired,
        "due_now": due_now,
        "collections": collections,
    })


def cmd_export(args: argparse.Namespace) -> None:
    data = _load()
    output_path = Path(args.output).expanduser()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        for card in data["cards"]:
            writer.writerow([card["question"], card["answer"], card["collection"]])
    _out({"ok": True, "exported": len(data["cards"]), "path": str(output_path)})


def cmd_import(args: argparse.Namespace) -> None:
    data = _load()
    now = _now()
    file_path = Path(args.file).expanduser()

    if not file_path.exists():
        _out({"ok": False, "error": f"File not found: {file_path}"})
        sys.exit(1)

    created = 0
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            question = row[0].strip()
            answer = row[1].strip()
            collection = row[2].strip() if len(row) >= 3 and row[2].strip() else (args.collection or "Imported")
            if not question or not answer:
                continue
            card = {
                "id": str(uuid.uuid4()),
                "question": question,
                "answer": answer,
                "collection": collection,
                "status": "learning",
                "ease_streak": 0,
                "next_review_at": _iso(now),
                "created_at": _iso(now),
                "video_id": None,
                "last_user_answer": None,
            }
            data["cards"].append(card)
            created += 1

    _save(data)
    _out({"ok": True, "imported": created})


def cmd_delete(args: argparse.Namespace) -> None:
    data = _load()
    original = len(data["cards"])
    data["cards"] = [c for c in data["cards"] if c["id"] != args.id]
    removed = original - len(data["cards"])
    if removed == 0:
        _out({"ok": False, "error": f"Card not found: {args.id}"})
        sys.exit(1)
    _save(data)
    _out({"ok": True, "deleted": args.id})


def cmd_delete_collection(args: argparse.Namespace) -> None:
    data = _load()
    original = len(data["cards"])
    data["cards"] = [c for c in data["cards"] if c["collection"] != args.collection]
    removed = original - len(data["cards"])
    _save(data)
    _out({"ok": True, "deleted_count": removed, "collection": args.collection})


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Memento flashcard manager")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Create one card")
    p_add.add_argument("--question", required=True)
    p_add.add_argument("--answer", required=True)
    p_add.add_argument("--collection", default="General")

    p_quiz = sub.add_parser("add-quiz", help="Batch-add quiz cards")
    p_quiz.add_argument("--video-id", required=True)
    p_quiz.add_argument("--questions", required=True, help="JSON array of {question, answer}")
    p_quiz.add_argument("--collection", default="Quiz")

    p_due = sub.add_parser("due", help="List due cards")
    p_due.add_argument("--collection", default=None)

    p_rate = sub.add_parser("rate", help="Rate a card")
    p_rate.add_argument("--id", required=True)
    p_rate.add_argument("--rating", required=True, choices=["easy", "good", "hard", "retire"])
    p_rate.add_argument("--user-answer", default=None)

    p_list = sub.add_parser("list", help="List cards")
    p_list.add_argument("--collection", default=None)
    p_list.add_argument("--status", default=None, choices=["learning", "retired"])

    sub.add_parser("stats", help="Show statistics")

    p_export = sub.add_parser("export", help="Export cards to CSV")
    p_export.add_argument("--output", required=True)

    p_import = sub.add_parser("import", help="Import cards from CSV")
    p_import.add_argument("--file", required=True)
    p_import.add_argument("--collection", default="Imported")

    p_del = sub.add_parser("delete", help="Delete one card")
    p_del.add_argument("--id", required=True)

    p_delcol = sub.add_parser("delete-collection", help="Delete all cards in a collection")
    p_delcol.add_argument("--collection", required=True)

    args = parser.parse_args()
    cmd_map = {
        "add": cmd_add,
        "add-quiz": cmd_add_quiz,
        "due": cmd_due,
        "rate": cmd_rate,
        "list": cmd_list,
        "stats": cmd_stats,
        "export": cmd_export,
        "import": cmd_import,
        "delete": cmd_delete,
        "delete-collection": cmd_delete_collection,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
