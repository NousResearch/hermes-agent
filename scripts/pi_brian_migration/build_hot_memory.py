#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STRONG_USER_PATTERNS = [
	("morning digest", "8:00 am"),
	("morning digest", "8:00 am et"),
	("morning digest", "not a raw link dump"),
	("morning digest", "same content and formatting"),
	("morning digest", "same style as before"),
]

STRONG_MEMORY_KEYWORDS = [
	"brian@poseidon",
	"poseidon",
	"research.briankeefe.dev",
	"grubhub",
	"telegram_bot_token",
	"trigger",
	"opencode",
	"home assistant",
	"homelab",
]

HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean(text: str) -> str:
	text = HTML_TAG_RE.sub("", text)
	text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
	text = re.sub(r"\s+", " ", text).strip()
	return text


def load_json(path: Path) -> object:
	return json.loads(path.read_text(encoding="utf-8"))


def draft_from_flat_memories(flat_store: dict[str, object]) -> tuple[list[str], list[str], list[str]]:
	user: list[str] = []
	memory: list[str] = []
	manual: list[str] = []
	for key, raw_entry in flat_store.items():
		if not isinstance(raw_entry, dict):
			continue
		value = clean(str(raw_entry.get("value") or ""))
		category = str(raw_entry.get("category") or "general")
		if not value:
			continue
		if category == "preference":
			user.append(f"- {key}: {value}")
		elif category == "location":
			manual.append(f"- review location memory `{key}` before import: {value}")
		else:
			memory.append(f"- {key}: {value}")
	return user, memory, manual


def draft_from_mem0(memories: list[dict[str, object]]) -> tuple[list[str], list[str], list[str]]:
	user: list[str] = []
	memory: list[str] = []
	manual: list[str] = []
	seen: set[str] = set()
	for item in memories:
		text = clean(str(item.get("memory") or ""))
		if not text:
			continue
		lower = text.lower()

		if text.lower().startswith("grubhub_browser_status:"):
			candidate = "- Grubhub browser flow on poseidon is known-good and can extract restaurant cards plus restaurant details."
			if candidate not in seen:
				memory.append(candidate)
				seen.add(candidate)
			continue

		if lower.startswith("user:"):
			if any(all(part in lower for part in pair) for pair in STRONG_USER_PATTERNS):
				candidate = "- Morning digest: weekday 8:00 AM America/New_York, concise parsed readable format, preserve prior style, avoid raw link dumps."
				if candidate not in seen:
					user.append(candidate)
					seen.add(candidate)
			elif "ssh to brian@poseidon" in lower or ("poseidon" in lower and "docker socket" in lower):
				candidate = "- For host/container diagnostics, prefer ssh to brian@poseidon when the local shell lacks Docker socket access."
				if candidate not in seen:
					memory.append(candidate)
					seen.add(candidate)
			elif "missing telegram_bot_token" in lower and "research" in lower:
				candidate = "- Research delivery back to Telegram depends on TELEGRAM_BOT_TOKEN being present in the runtime environment."
				if candidate not in seen:
					memory.append(candidate)
					seen.add(candidate)
			continue

		metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
		category = str(metadata.get("category") or "")
		if text.lower().startswith("delivery_address:"):
			manual.append(f"- review sensitive explicit memory before import: {text}")
			continue
		if category == "preference":
			candidate = f"- {text}"
			if candidate not in seen:
				user.append(candidate)
				seen.add(candidate)
		elif category:
			candidate = f"- {text}"
			if candidate not in seen:
				memory.append(candidate)
				seen.add(candidate)
	return user, memory, manual


def compact(lines: list[str], max_chars: int) -> list[str]:
	out: list[str] = []
	used = 0
	for line in lines:
		addition = len(line) + 1
		if used + addition > max_chars:
			break
		out.append(line)
		used += addition
	return out


def render(title: str, lines: list[str]) -> str:
	body = "\n".join(lines) if lines else "- draft pending manual curation"
	return f"# {title}\n\n{body}\n"


def dedupe_manual(lines: list[str]) -> list[str]:
	seen_delivery_address = False
	out: list[str] = []
	for line in lines:
		if "delivery_address" in line:
			if seen_delivery_address:
				continue
			seen_delivery_address = True
		if line not in out:
			out.append(line)
	return out


def main() -> int:
	parser = argparse.ArgumentParser(description="Build Hermes hot-memory drafts from pi-brian exports")
	parser.add_argument("--mem0-export", required=True, help="Path to mem0-export.json")
	parser.add_argument("--flat-memory", help="Optional path to flat memories.json export")
	parser.add_argument("--output-dir", required=True, help="Directory to write USER/MEMORY draft files")
	args = parser.parse_args()

	mem0_export = Path(args.mem0_export).expanduser().resolve()
	output_dir = Path(args.output_dir).expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	memories = load_json(mem0_export)
	if not isinstance(memories, list):
		raise SystemExit("mem0 export must be a JSON list")

	user_lines: list[str] = []
	memory_lines: list[str] = []
	manual_lines: list[str] = []

	if args.flat_memory:
		flat_payload = load_json(Path(args.flat_memory).expanduser().resolve())
		if isinstance(flat_payload, dict):
			u, m, manual = draft_from_flat_memories(flat_payload)
			user_lines.extend(u)
			memory_lines.extend(m)
			manual_lines.extend(manual)

	u, m, manual = draft_from_mem0(memories)  # type: ignore[arg-type]
	user_lines.extend(u)
	memory_lines.extend(m)
	manual_lines.extend(manual)

	user_lines = compact(list(dict.fromkeys(user_lines)), 1200)
	memory_lines = compact(list(dict.fromkeys(memory_lines)), 1900)
	manual_lines = dedupe_manual(manual_lines)

	user_path = output_dir / "USER.draft.md"
	memory_path = output_dir / "MEMORY.draft.md"
	manual_path = output_dir / "MANUAL_REVIEW.md"
	summary_path = output_dir / "summary.json"

	user_path.write_text(render("USER draft", user_lines), encoding="utf-8")
	memory_path.write_text(render("MEMORY draft", memory_lines), encoding="utf-8")
	manual_path.write_text(render("Manual review", manual_lines), encoding="utf-8")
	summary_path.write_text(
		json.dumps(
			{
				"user_draft_chars": len(user_path.read_text(encoding="utf-8")),
				"memory_draft_chars": len(memory_path.read_text(encoding="utf-8")),
				"manual_review_count": len(manual_lines),
				"user_path": str(user_path),
				"memory_path": str(memory_path),
				"manual_review_path": str(manual_path),
			},
			indent=2,
		)
		+ "\n",
		encoding="utf-8",
	)

	print(summary_path.read_text(encoding="utf-8"))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
