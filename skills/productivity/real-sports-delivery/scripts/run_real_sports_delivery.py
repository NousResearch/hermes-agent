#!/usr/bin/env python3
"""Hermes-side orchestration for the Real Sports extractor.

This helper keeps the extractor repo as the source of truth and uses Hermes
for the delivery/orchestration layer:

1. optionally run `npm run extract`
2. locate and validate `ai-ready.json`
3. invoke `hermes chat` with a focused delivery prompt
4. let Hermes use its configured tools to post the result

Usage:
  python run_real_sports_delivery.py --extractor-dir ~/src/real-extractor
  python run_real_sports_delivery.py --extractor-dir ~/src/real-extractor --skip-extract
  python run_real_sports_delivery.py --extractor-dir ~/src/real-extractor --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_STALE_MS = 36 * 60 * 60 * 1000
DEFAULT_TOOLSETS = "file,terminal,send_message"


@dataclass(frozen=True)
class AiReady:
    path: Path
    payload: dict


def _die(message: str, code: int = 1) -> None:
    print(f"[real-sports] {message}", file=sys.stderr)
    raise SystemExit(code)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _extractor_dir_from_args(args: argparse.Namespace) -> Path:
    raw = args.extractor_dir or os.getenv("REALSPORTS_EXTRACTOR_DIR", "")
    if not raw:
        _die(
            "Set REALSPORTS_EXTRACTOR_DIR or pass --extractor-dir /absolute/path/to/real-extractor."
        )
    extractor_dir = Path(raw).expanduser().resolve()
    if not extractor_dir.is_dir():
        _die(f"Extractor directory not found: {extractor_dir}")
    return extractor_dir


def _auth_file(extractor_dir: Path) -> Path:
    return extractor_dir / ".auth" / "realsports-auth.json"


def _ai_ready_dir(extractor_dir: Path, target_date: Optional[str] = None) -> Path:
    if target_date:
        return extractor_dir / "data" / "normalized" / target_date
    return extractor_dir / "data" / "normalized"


def _load_ai_ready(path: Path) -> AiReady:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _die(f"Missing ai-ready JSON: {path}")
    return AiReady(path=path, payload=payload)


def _latest_ai_ready(extractor_dir: Path, target_date: Optional[str] = None) -> AiReady:
    base = _ai_ready_dir(extractor_dir, target_date)
    if target_date:
        path = base / "ai-ready.json"
        if path.is_file():
            return _load_ai_ready(path)

        fallback = extractor_dir / "ai-ready.json"
        if fallback.is_file():
            ai = _load_ai_ready(fallback)
            if ai.payload.get("targetDate") == target_date:
                return ai
        _die(f"Missing ai-ready JSON for {target_date}: expected {path} (or compatible root ai-ready.json)")

    candidates = sorted(
        base.glob("*/ai-ready.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return _load_ai_ready(candidates[0])

    fallback = extractor_dir / "ai-ready.json"
    if fallback.is_file():
        return _load_ai_ready(fallback)

    _die(f"No ai-ready.json files found under {base} or at {fallback}")


def _validate_ai_ready(ai: AiReady, expected_date: Optional[str], max_stale_ms: int) -> None:
    payload = ai.payload
    target_date = payload.get("targetDate")
    extracted_at = payload.get("extractedAt")
    if not isinstance(target_date, str) or not target_date:
        _die(f"ai-ready.json is missing a valid targetDate: {ai.path}")
    if expected_date and target_date != expected_date:
        _die(
            f"ai-ready.json targetDate {target_date!r} does not match expected {expected_date!r}: {ai.path}"
        )
    if not isinstance(extracted_at, str) or not extracted_at:
        _die(f"ai-ready.json is missing a valid extractedAt timestamp: {ai.path}")

    try:
        extracted_ts = datetime.fromisoformat(extracted_at.replace("Z", "+00:00"))
    except ValueError:
        _die(f"ai-ready.json extractedAt is not ISO-8601: {extracted_at!r}")

    if extracted_ts.tzinfo is None:
        extracted_ts = extracted_ts.replace(tzinfo=timezone.utc)
    age_ms = int((datetime.now(timezone.utc) - extracted_ts.astimezone(timezone.utc)).total_seconds() * 1000)
    if age_ms > max_stale_ms:
        hours = max_stale_ms / 3_600_000
        _die(
            f"ai-ready.json looks stale ({age_ms / 3_600_000:.1f}h old, max {hours:.1f}h): {ai.path}"
        )


def _run(cmd: list[str], cwd: Path, env: Optional[dict[str, str]] = None) -> None:
    result = subprocess.run(cmd, cwd=str(cwd), env=env, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _run_extract(extractor_dir: Path) -> None:
    auth_path = _auth_file(extractor_dir)
    if not auth_path.is_file():
        _die(
            f"Missing auth file: {auth_path}. Run `npm run auth` in the extractor repo first."
        )
    print(f"[real-sports] Running extraction in {extractor_dir}")
    _run(["npm", "run", "extract"], cwd=extractor_dir)


def _hermes_prompt(extractor_dir: Path, ai: AiReady, toolsets: str) -> str:
    return (
        "You are Hermes orchestrating the Real Sports delivery workflow. "
        "The extractor repo is the source of truth; do not modify it. "
        f"Read the delivery input from `{ai.path}` and generate the final delivery-ready picks output. "
        "Use the file tool to inspect the JSON, summarize only what is present, and do not invent picks. "
        "If the send_message tool is available in this session, post the final result to the configured Discord target. "
        "If the messaging tool is unavailable or posting fails, explain exactly what is missing. "
        f"Extractor repo: `{extractor_dir}`. "
        f"Available toolsets requested for this run: `{toolsets}`."
    )


def _invoke_hermes(args: argparse.Namespace, extractor_dir: Path, ai: AiReady) -> None:
    hermes_bin = args.hermes_bin or os.getenv("HERMES_BIN", "hermes")
    toolsets = args.toolsets or os.getenv("REALSPORTS_TOOLSETS", DEFAULT_TOOLSETS)
    prompt = _hermes_prompt(extractor_dir, ai, toolsets)
    cmd = [hermes_bin, "chat", "-t", toolsets]
    if args.model:
        cmd.extend(["-m", args.model])
    if args.provider:
        cmd.extend(["--provider", args.provider])
    cmd.extend(["-q", prompt])

    print("[real-sports] Invoking Hermes delivery session:")
    print("[real-sports] ", " ".join(cmd[:4]), "...", sep="")
    _run(cmd, cwd=_repo_root())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes-side orchestration for the Real Sports extractor")
    parser.add_argument("--extractor-dir", help="Path to the real-extractor checkout")
    parser.add_argument("--date", help="Target date folder under data/normalized/YYYY-MM-DD")
    parser.add_argument("--skip-extract", action="store_true", help="Reuse existing ai-ready.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the Hermes command without running it")
    parser.add_argument("--max-stale-ms", type=int, default=DEFAULT_STALE_MS, help="Maximum allowed age for ai-ready.json")
    parser.add_argument("--hermes-bin", help="Hermes executable name or path (default: hermes)")
    parser.add_argument("--toolsets", help=f"Hermes toolsets to request (default: {DEFAULT_TOOLSETS})")
    parser.add_argument("--model", help="Optional Hermes model override")
    parser.add_argument("--provider", help="Optional Hermes provider override")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    extractor_dir = _extractor_dir_from_args(args)

    if not args.skip_extract:
        _run_extract(extractor_dir)

    ai = _latest_ai_ready(extractor_dir, args.date)
    _validate_ai_ready(ai, args.date, args.max_stale_ms)

    if args.dry_run:
        toolsets = args.toolsets or os.getenv("REALSPORTS_TOOLSETS", DEFAULT_TOOLSETS)
        print("[real-sports] Dry run OK")
        print(f"[real-sports] ai-ready: {ai.path}")
        print("[real-sports] Hermes prompt:")
        print(_hermes_prompt(extractor_dir, ai, toolsets))
        return

    _invoke_hermes(args, extractor_dir, ai)


if __name__ == "__main__":
    main()
