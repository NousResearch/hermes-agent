"""Fetch a normalized timestamped transcript for a YouTube URL or video ID."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

SKILL_ROOT = Path(__file__).resolve().parents[1]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

from utils.provider import TranscriptProviderError, fetch_transcript
from utils.youtube import extract_video_id


def run(
    user_input: str,
    *,
    languages: Sequence[str] | None = None,
    provider: Callable[..., list[dict[str, float | str]]] = fetch_transcript,
) -> list[dict[str, float | str]] | dict[str, str]:
    """Return normalized segments or a structured error without raising."""
    try:
        video_id = extract_video_id(user_input)
        return provider(video_id, languages=languages)
    except ValueError:
        return {
            "error": "InvalidYouTubeURL",
            "message": "The provided input is not a valid YouTube URL or video ID.",
        }
    except TranscriptProviderError as exc:
        return exc.as_dict()
    except Exception:
        return {
            "error": "TranscriptFetchFailed",
            "message": "Transcript retrieval failed. Try again later or provide a timestamped transcript.",
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="YouTube URL or 11-character video ID")
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        help="Preferred transcript language code; may be repeated",
    )
    args = parser.parse_args(argv)

    result = run(args.video, languages=args.languages)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 1 if isinstance(result, dict) and "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
