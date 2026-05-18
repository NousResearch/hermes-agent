#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def fetch_json(base_url: str, path: str, *, query: dict[str, str] | None = None) -> object:
	url = f"{base_url.rstrip('/')}{path}"
	if query:
		url = f"{url}?{urlencode(query)}"
	with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=15) as response:
		return json.loads(response.read().decode("utf-8"))


def unwrap_results(payload: object) -> list[dict[str, object]]:
	if isinstance(payload, dict):
		results = payload.get("results")
		return results if isinstance(results, list) else []
	if isinstance(payload, list):
		return payload  # type: ignore[return-value]
	return []


def render_markdown(items: list[dict[str, object]]) -> str:
	lines = ["# Self-hosted Mem0 export", ""]
	for index, item in enumerate(items, 1):
		memory = str(item.get("memory") or "").strip()
		metadata = item.get("metadata")
		created_at = item.get("created_at") or ""
		lines.append(f"## {index}. {item.get('id', '')}")
		if created_at:
			lines.append(f"- created_at: {created_at}")
		if metadata:
			lines.append(f"- metadata: `{json.dumps(metadata, ensure_ascii=False)}`")
		lines.append("")
		lines.append(memory)
		lines.append("")
	return "\n".join(lines).rstrip() + "\n"


def main() -> int:
	parser = argparse.ArgumentParser(description="Export pi-brian self-hosted Mem0 data for Hermes migration")
	parser.add_argument("--base-url", required=True, help="Self-hosted Mem0 base URL, e.g. http://127.0.0.1:8000")
	parser.add_argument("--user-id", required=True, help="Mem0 user scope to export")
	parser.add_argument("--output-dir", required=True, help="Directory to write export files into")
	args = parser.parse_args()

	output_dir = Path(args.output_dir).expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	payload = fetch_json(args.base_url, "/memories", query={"user_id": args.user_id})
	items = unwrap_results(payload)

	json_path = output_dir / "mem0-export.json"
	md_path = output_dir / "mem0-export.md"
	json_path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
	md_path.write_text(render_markdown(items), encoding="utf-8")

	print(json.dumps({"count": len(items), "json": str(json_path), "markdown": str(md_path)}, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
