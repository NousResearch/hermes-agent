"""Strict Gmail unread-metadata MCP server.

The server exposes exactly one read-only operation.  It lists unread message IDs,
then fetches each message with Gmail's ``format=metadata`` endpoint, which does
not return body parts, raw MIME or attachments.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
_METADATA_HEADERS = ["From", "Subject"]
_METADATA_QUERY = "is:unread"
_MAX_PAGE_SIZE = 25


class GmailMetadataPolicyError(ValueError):
    """The requested mailbox read falls outside the narrow local policy."""


class GmailMetadataReader:
    """Return normalised unread metadata for explicitly allowed Gmail accounts."""

    def __init__(
        self,
        *,
        allowed_accounts: Iterable[str],
        service_factory: Callable[[str], Any],
    ) -> None:
        self._allowed_accounts = {account.casefold() for account in allowed_accounts}
        self._service_factory = service_factory

    def search_gmail_messages(
        self, *, query: str, user_google_email: str, page_size: int = 10
    ) -> str:
        account = user_google_email.casefold()
        if account not in self._allowed_accounts:
            raise GmailMetadataPolicyError("requested Gmail account is not allowlisted")
        if query != _METADATA_QUERY:
            raise GmailMetadataPolicyError("only is:unread Gmail metadata is allowed")
        if not 1 <= page_size <= _MAX_PAGE_SIZE:
            raise GmailMetadataPolicyError(f"page_size must be between 1 and {_MAX_PAGE_SIZE}")

        service = self._service_factory(user_google_email)
        listed = service.users().messages().list(
            userId="me", q=_METADATA_QUERY, maxResults=page_size
        ).execute()
        message_refs = listed.get("messages", []) if isinstance(listed, Mapping) else []
        rows: list[dict[str, str]] = []
        for ref in message_refs:
            if not isinstance(ref, Mapping) or not isinstance(ref.get("id"), str):
                continue
            raw = service.users().messages().get(
                userId="me",
                id=ref["id"],
                format="metadata",
                metadataHeaders=_METADATA_HEADERS,
            ).execute()
            rows.append(_normalise_metadata(raw))
        return json.dumps({"messages": rows}, separators=(",", ":"))


def _normalise_metadata(raw: Mapping[str, object]) -> dict[str, str]:
    payload = raw.get("payload")
    headers = payload.get("headers", []) if isinstance(payload, Mapping) else []
    header_values = {
        str(header.get("name", "")).casefold(): str(header.get("value", ""))
        for header in headers
        if isinstance(header, Mapping)
    }
    internal_date = raw.get("internalDate", "")
    try:
        received_at = datetime.fromtimestamp(int(str(internal_date)) / 1000, UTC).isoformat()
    except (TypeError, ValueError, OverflowError):
        received_at = ""
    return {
        "id": str(raw.get("id", "")),
        "from": header_values.get("from", ""),
        "subject": header_values.get("subject", ""),
        "received_at": received_at,
        "preview": str(raw.get("snippet", "")),
    }


def _credential_path(account: str, root: Path) -> Path:
    return root / f"{account}.json"


def _google_service_factory(*, credentials_root: Path) -> Callable[[str], Any]:
    def build_service(account: str) -> Any:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        credential_path = _credential_path(account, credentials_root)
        if not credential_path.is_file():
            raise GmailMetadataPolicyError("no OAuth credential exists for requested Gmail account")
        credentials = Credentials.from_authorized_user_file(
            str(credential_path), scopes=[GMAIL_READONLY_SCOPE]
        )
        if not credentials.valid:
            if not credentials.expired or not credentials.refresh_token:
                raise GmailMetadataPolicyError("Gmail OAuth credential is not valid")
            credentials.refresh(Request())
            _write_refreshed_credentials(credential_path, credentials.to_json())
        return build("gmail", "v1", credentials=credentials, cache_discovery=False)

    return build_service


def _write_refreshed_credentials(path: Path, contents: str) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(contents)
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run strict Gmail unread-metadata MCP")
    parser.add_argument("--allow-account", action="append", required=True)
    parser.add_argument(
        "--credentials-root",
        type=Path,
        default=Path("/home/kensei/.google_workspace_mcp/credentials"),
    )
    args = parser.parse_args(argv)

    from fastmcp import FastMCP

    reader = GmailMetadataReader(
        allowed_accounts=args.allow_account,
        service_factory=_google_service_factory(credentials_root=args.credentials_root),
    )
    server = FastMCP("gmail-metadata-readonly")

    @server.tool()
    def search_gmail_messages(
        query: str, user_google_email: str, page_size: int = 10
    ) -> str:
        return reader.search_gmail_messages(
            query=query, user_google_email=user_google_email, page_size=page_size
        )

    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
