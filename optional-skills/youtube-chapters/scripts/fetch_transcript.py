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

from utils.provider import DEFAULT_LANGUAGES, TranscriptProviderError, fetch_transcript, list_transcripts
from utils.youtube import extract_video_id


def parse_languages(value: str | None) -> list[str]:
    """Parse comma-separated language codes in preference order."""
    languages = [language.strip() for language in (value or "").split(",") if language.strip()]
    return languages or list(DEFAULT_LANGUAGES)


def run(
    user_input: str,
    *,
    languages: Sequence[str] | None = None,
    list_only: bool = False,
    cookies: str | None = None,
    cookies_from_browser: str | None = None,
    provider: Callable[..., Any] = fetch_transcript,
    list_provider: Callable[..., dict[str, Any]] = list_transcripts,
) -> list[dict[str, float | str]] | dict[str, Any]:
    """Return normalized segments or a structured error without raising."""
    try:
        video_id = extract_video_id(user_input)
        if list_only:
            return list_provider(
                video_id,
                cookies=cookies,
                cookies_from_browser=cookies_from_browser,
            )
        return provider(
            video_id,
            languages=languages,
            cookies=cookies,
            cookies_from_browser=cookies_from_browser,
        )
    except ValueError:
        return {
            "error": "InvalidYouTubeURL",
            "message": "The provided input is not a valid YouTube URL or video ID.",
        }
    except TranscriptProviderError as exc:
        return exc.as_dict()
    except Exception as exc:
        return {
            "error": "TranscriptFetchFailed",
            "message": "Transcript retrieval failed. Try again later or provide a timestamped transcript.",
            "detail": type(exc).__name__,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="YouTube URL or 11-character video ID")
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated transcript language codes in preference order (default: tr,en)",
    )
    parser.add_argument(
        "--list-transcripts",
        action="store_true",
        help="List available transcript tracks instead of fetching transcript text",
    )
    parser.add_argument("--cookies", help="Path to a private Netscape-format cookies file")
    parser.add_argument("--cookies-from-browser", help="Browser name to load private cookies from")
    args = parser.parse_args(argv)

    result = run(
        args.video,
        languages=parse_languages(args.languages),
        list_only=args.list_transcripts,
        cookies=args.cookies,
        cookies_from_browser=args.cookies_from_browser,
    )
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 1 if isinstance(result, dict) and "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
