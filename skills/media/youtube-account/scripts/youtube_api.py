"""Read-only YouTube Data API commands for the active Hermes profile."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
TOKEN_PATH = HERMES_HOME / "youtube_token.json"


def service():
    if not TOKEN_PATH.exists():
        raise FileNotFoundError("No youtube_token.json in the active Hermes profile")
    credentials = Credentials.from_authorized_user_file(str(TOKEN_PATH))
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        TOKEN_PATH.write_text(credentials.to_json(), encoding="utf-8")
    return build("youtube", "v3", credentials=credentials, cache_discovery=False)


def channel(youtube) -> dict:
    response = (
        youtube.channels()
        .list(part="snippet,contentDetails,statistics", mine=True, maxResults=1)
        .execute()
    )
    items = response.get("items", [])
    if not items:
        return {"channel": None}
    item = items[0]
    snippet = item.get("snippet", {})
    return {
        "channel": {
            "id": item.get("id"),
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "custom_url": snippet.get("customUrl"),
            "published_at": snippet.get("publishedAt"),
            "statistics": item.get("statistics", {}),
            "related_playlists": item.get("contentDetails", {}).get(
                "relatedPlaylists", {}
            ),
        }
    }


def subscriptions(youtube, maximum: int) -> dict:
    response = (
        youtube.subscriptions()
        .list(
            part="snippet",
            mine=True,
            maxResults=maximum,
            order="alphabetical",
        )
        .execute()
    )
    return {
        "subscriptions": [
            {
                "id": item.get("id"),
                "channel_id": item.get("snippet", {})
                .get("resourceId", {})
                .get("channelId"),
                "title": item.get("snippet", {}).get("title"),
            }
            for item in response.get("items", [])
        ],
        "next_page_token": response.get("nextPageToken"),
    }


def _is_channel_not_found(error: HttpError) -> bool:
    if getattr(error.resp, "status", None) != 404:
        return False
    try:
        payload = json.loads(error.content.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    details = payload.get("error", {}).get("errors", [])
    return any(detail.get("reason") == "channelNotFound" for detail in details)


def playlists(youtube, maximum: int) -> dict:
    try:
        response = (
            youtube.playlists()
            .list(part="snippet,status,contentDetails", mine=True, maxResults=maximum)
            .execute()
        )
    except HttpError as error:
        # YouTube returns 404/channelNotFound for valid Google identities that
        # have never created a YouTube channel. Treat that as an empty account,
        # while preserving every other API error.
        if not _is_channel_not_found(error):
            raise
        response = {"items": []}
    return {
        "playlists": [
            {
                "id": item.get("id"),
                "title": item.get("snippet", {}).get("title"),
                "description": item.get("snippet", {}).get("description"),
                "privacy": item.get("status", {}).get("privacyStatus"),
                "item_count": item.get("contentDetails", {}).get("itemCount"),
            }
            for item in response.get("items", [])
        ],
        "next_page_token": response.get("nextPageToken"),
    }


def playlist_items(youtube, playlist_id: str, maximum: int) -> dict:
    response = (
        youtube.playlistItems()
        .list(part="snippet,contentDetails", playlistId=playlist_id, maxResults=maximum)
        .execute()
    )
    return {
        "items": [
            {
                "id": item.get("id"),
                "video_id": item.get("contentDetails", {}).get("videoId"),
                "title": item.get("snippet", {}).get("title"),
                "channel_title": item.get("snippet", {}).get("videoOwnerChannelTitle"),
                "published_at": item.get("contentDetails", {}).get("videoPublishedAt"),
            }
            for item in response.get("items", [])
        ],
        "next_page_token": response.get("nextPageToken"),
    }


def _bounded(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 50:
        raise argparse.ArgumentTypeError("--max must be between 1 and 50")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="Read private YouTube account data")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("channel")

    subscription_parser = commands.add_parser("subscriptions")
    subscription_parser.add_argument("--max", type=_bounded, default=25)

    playlist_parser = commands.add_parser("playlists")
    playlist_parser.add_argument("--max", type=_bounded, default=25)

    item_parser = commands.add_parser("playlist-items")
    item_parser.add_argument("playlist_id")
    item_parser.add_argument("--max", type=_bounded, default=25)
    args = parser.parse_args()

    try:
        youtube = service()
        if args.command == "channel":
            result = channel(youtube)
        elif args.command == "subscriptions":
            result = subscriptions(youtube, args.max)
        elif args.command == "playlists":
            result = playlists(youtube, args.max)
        else:
            result = playlist_items(youtube, args.playlist_id, args.max)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
