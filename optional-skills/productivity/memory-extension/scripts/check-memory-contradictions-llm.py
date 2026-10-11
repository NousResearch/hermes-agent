#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pass 2 — LLM-based contradiction detection (direct API call).

Usage:
    python check-memory-contradictions-llm.py [HERMES_HOME] [--out REPORT.md] [--dry-run]

Reads MEMORY.md + USER.md + extended/*.md, sends them to the configured LLM
provider (default model from config.yaml), and writes a markdown report listing
contradictions with sources. Exit 0 if none, 1 if found, 2 on error.

The report is a DETECTION aid: resolution is always human. The script never
edits memory files.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

PROMPT_TEMPLATE = """You are a memory coherence auditor. Here is the persistent memory of an AI agent (index MEMORY.md/USER.md + detail files extended/).

Find CONTRADICTIONS between facts: two statements that cannot both be true (e.g. "OS: Windows 11" vs "OS: Windows 10", "never launch X" vs "launch X for tests", two different versions of the same tool, two different dates for the same event).

Rules:
- Report ONLY real contradictions, not mere phrasing differences, nor compatible pairs (e.g. "HIBP check shows no breach" vs "SFR leak" are compatible if the leak is not in HIBP).
- A nuance (e.g. "registry says Win10, Win32_OperatingSystem is authoritative") is NOT a contradiction if explicitly resolved.
- Cite the exact source (MEMORY.md, USER.md, extended/<file>.md) for each fact.
- Reply STRICTLY with a single valid JSON array, with NO other text before or after (no commentary, no explanation, no thinking). Format:
[{{"fait_a": "...", "source_a": "...", "fait_b": "...", "source_b": "...", "raison": "..."}}]
- If none: [] (and nothing else)

MEMORY TO AUDIT:
{memory}"""

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 400
MAX_RETRIES = 6
RETRY_BASE_DELAY = 20  # seconds — DeepSeek peak-hour latency


def load_config(hermes_home: Path) -> dict:
    """Return (provider, base_url, model) from config.yaml + auth.json."""
    cfg_path = hermes_home / "config.yaml"
    auth_path = hermes_home / "auth.json"
    provider = "nous"
    base_url = "https://inference-api.nousresearch.com/v1"
    model = "deepseek/deepseek-v4-flash"
    token = None

    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        model_cfg = cfg.get("model") or {}
        provider = model_cfg.get("provider") or provider
        base_url = model_cfg.get("base_url") or base_url
        model = model_cfg.get("model") or model
        # model may be "provider/model" — keep the full id for the API
        if "/" not in model and provider:
            model = f"{provider}/{model}"

    if auth_path.exists():
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
        prov = auth.get("providers", {}).get(provider, {})
        token = prov.get("access_token") or prov.get("api_key")
        if not token:
            # fall back to active_provider
            active = auth.get("active_provider")
            if active:
                prov = auth.get("providers", {}).get(active, {})
                token = prov.get("access_token") or prov.get("api_key")
                if token:
                    provider = active

    if not token:
        raise KeyError(f"no access token found for provider '{provider}' in auth.json")

    return {"provider": provider, "base_url": base_url, "model": model, "token": token}


def collect_memory(hermes_home: Path) -> str:
    mem_dir = hermes_home / "memories"
    parts = []
    for idx in ("MEMORY.md", "USER.md"):
        p = mem_dir / idx
        if p.exists():
            parts.append(f"### {idx}\n{p.read_text(encoding='utf-8')}")
    ext = mem_dir / "extended"
    if ext.is_dir():
        for f in sorted(ext.glob("*.md")):
            if f.name == "README.md":
                continue
            parts.append(f"### extended/{f.name}\n{f.read_text(encoding='utf-8')}")
    if not parts:
        raise FileNotFoundError(f"no memory found in {mem_dir}")
    return "\n\n".join(parts)


def chunk_memory(memory: str) -> list:
    """Split memory into overlapping chunks to stay under server timeouts."""
    if len(memory) <= CHUNK_SIZE:
        return [memory]
    chunks = []
    start = 0
    while start < len(memory):
        end = min(start + CHUNK_SIZE, len(memory))
        if end < len(memory):
            nl = memory.rfind(chr(10), start + CHUNK_OVERLAP, end)
            if nl != -1:
                end = nl + 1
        chunks.append(memory[start:end])
        if end >= len(memory):
            break
        start = max(start, end - CHUNK_OVERLAP)
    return chunks


def parse_api_response(raw: str) -> dict:
    """Parse the raw HTTP body. Raises JSONDecodeError on non-JSON so the
    caller can retry (server latency bodies are often plain text)."""
    return json.loads(raw)


def extract_json_array(text: str):
    """Best-effort parse of a chatty LLM reply into a JSON list.
    Tries: direct load, fenced ``` block, then every [..] span
    (longest first). Returns None when no valid array is found."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1]
            if t.startswith("json"):
                t = t[4:]
            t = t.strip()
    try:
        v = json.loads(t)
        if isinstance(v, list):
            return v
    except (json.JSONDecodeError, ValueError):
        pass
    starts = [i for i, c in enumerate(t) if c == "["]
    ends = [i for i, c in enumerate(t) if c == "]"]
    for s in starts:
        for e in reversed(ends):
            if e <= s:
                break
            try:
                v = json.loads(t[s:e + 1])
                if isinstance(v, list):
                    return v
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def call_llm_once(cfg: dict, prompt: str) -> list:
    body = json.dumps({
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.1,
    }).encode("utf-8")
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url, data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['token']}",
            "User-Agent": UA,
        },
    )
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                raw = r.read().decode("utf-8", errors="replace")
            try:
                resp = parse_api_response(raw)
            except json.JSONDecodeError:
                print(f"  [retry {attempt}/{MAX_RETRIES}] non-JSON response ({raw[:120]!r}) — waiting {RETRY_BASE_DELAY * attempt}s...")
                time.sleep(RETRY_BASE_DELAY * attempt)
                continue
            content = resp["choices"][0]["message"]["content"]
            found = extract_json_array(content)
            if found is not None:
                return found
            print(f"  [retry {attempt}/{MAX_RETRIES}] non-JSON LLM reply — waiting {RETRY_BASE_DELAY * attempt}s...")
            print("    excerpt:", repr(content[:200]), file=sys.stderr)
            time.sleep(RETRY_BASE_DELAY * attempt)
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode()[:200]}"
            if e.code == 429 or e.code >= 500:  # 500/502/503/504/520/524 + rate-limit
                print(f"  [retry {attempt}/{MAX_RETRIES}] {e.code} (server latency) — waiting {RETRY_BASE_DELAY * attempt}s...")
                time.sleep(RETRY_BASE_DELAY * attempt)
                continue
            print(f"ERROR API {last_err}", file=sys.stderr)
            sys.exit(2)
        except (TimeoutError, urllib.error.URLError) as e:
            last_err = str(e)
            print(f"  [retry {attempt}/{MAX_RETRIES}] timeout — waiting {RETRY_BASE_DELAY * attempt}s...")
            time.sleep(RETRY_BASE_DELAY * attempt)
    print(f"ERROR: failed after {MAX_RETRIES} attempts ({last_err})", file=sys.stderr)
    sys.exit(2)
def call_llm(cfg: dict, memory: str, dry_run: bool = False) -> list:
    if dry_run:
        print("[dry-run] no API call — memory collected:", len(memory), "chars")
        return []
    chunks = chunk_memory(memory)
    print(f"[pass LLM] {len(chunks)} chunk(s) — {len(memory)} chars")
    all_found = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[pass LLM] chunk {i}/{len(chunks)} ({len(chunk)} chars)...")
        prompt = PROMPT_TEMPLATE.format(memory=chunk)
        found = call_llm_once(cfg, prompt)
        all_found.extend(found)
    seen = set()
    dedup = []
    for c in all_found:
        key = (c.get("fait_a", "")[:60], c.get("fait_b", "")[:60])
        if key not in seen:
            seen.add(key)
            dedup.append(c)
    return dedup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hermes_home", nargs="?", default=os.environ.get("HERMES_HOME", ""))
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    hermes_home = Path(args.hermes_home) if args.hermes_home else Path.home() / ".hermes"
    out_path = Path(args.out) if args.out else hermes_home / "memories" / "contradictions-report.md"

    memory = collect_memory(hermes_home)
    if args.dry_run:
        cfg = {"model": "dry-run", "base_url": "", "token": ""}
    else:
        cfg = load_config(hermes_home)
    contradictions = call_llm(cfg, memory, args.dry_run)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if contradictions:
        lines = [
            "# Memory contradiction report",
            "",
            f"_Generated {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')} — model {cfg['model']}_",
            "",
            f"**{len(contradictions)} contradiction(s) detected.** Resolve manually:",
            "",
        ]
        for i, c in enumerate(contradictions, 1):
            lines += [
                f"## {i}. {c.get('raison', 'Contradiction')}",
                "",
                f"- **Fact A:** {c.get('fait_a', '?')}",
                f"  - Source: {c.get('source_a', '?')}",
                f"- **Fact B:** {c.get('fait_b', '?')}",
                f"  - Source: {c.get('source_b', '?')}",
                "",
            ]
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"FOUND: {len(contradictions)} contradiction(s) -> {out_path}")
        return 1
    else:
        out_path.write_text(
            "# Memory contradiction report\n\n"
            f"_Generated {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')} — model {cfg['model']}_\n\n"
            "OK: no contradiction detected.\n",
            encoding="utf-8",
        )
        print(f"OK: no contradiction -> {out_path}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
