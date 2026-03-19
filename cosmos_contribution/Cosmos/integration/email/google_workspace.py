"""
cosmos Google Workspace Integration

"Good news, everyone! I can manage your Google Workspace email!"

Full Google Workspace / Gmail integration including:
- Email management (Gmail API)
- Calendar integration
- Mailbox filtering
- Admin functions (with proper permissions)
"""

import asyncio
import base64
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from loguru import logger


class GoogleAuthType(Enum):
    """Google authentication types."""
    OAUTH_USER = "oauth_user"  # User-based OAuth
    SERVICE_ACCOUNT = "service_account"  # Service account with domain-wide delegation


@dataclass
class GoogleConfig:
    """Google Workspace configuration."""
    client_id: str = ""
    client_secret: str = ""
    project_id: str = ""
    service_account_file: str = ""  # Path to service account JSON
    delegated_user: str = ""  # User to impersonate (for service account)
    redirect_uri: str = "http://localhost:8400/callback"
    auth_type: GoogleAuthType = GoogleAuthType.OAUTH_USER

    # Required scopes by feature
    SCOPES = {
        "gmail_read": ["https://www.googleapis.com/auth/gmail.readonly"],
        "gmail_modify": ["https://www.googleapis.com/auth/gmail.modify"],
        "gmail_send": ["https://www.googleapis.com/auth/gmail.send"],
        "gmail_full": ["https://mail.google.com/"],
        "calendar": ["https://www.googleapis.com/auth/calendar"],
        "admin_users": ["https://www.googleapis.com/auth/admin.directory.user"],
        "admin_groups": ["https://www.googleapis.com/auth/admin.directory.group"],
    }


@dataclass
class GmailMessage:
    """Gmail message structure."""
    id: str = ""
    thread_id: str = ""
    label_ids: List[str] = field(default_factory=list)
    subject: str = ""
    sender: str = ""
    sender_name: str = ""
    recipients: List[str] = field(default_factory=list)
    cc: List[str] = field(default_factory=list)
    date: Optional[datetime] = None
    snippet: str = ""
    body: str = ""
    body_type: str = "text"
    is_read: bool = True
    is_starred: bool = False
    has_attachments: bool = False
    attachments: List[Dict] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class GmailLabel:
    """Gmail label."""
    id: str
    name: str
    type: str = "user"  # user or system
    message_list_visibility: str = "show"
    label_list_visibility: str = "labelShow"
    messages_total: int = 0
    messages_unread: int = 0


class GoogleWorkspaceIntegration:
    """
    Google Workspace / Gmail Integration.

    Prerequisites:
    1. Create project in Google Cloud Console
    2. Enable Gmail API
    3. Configure OAuth consent screen
    4. Create OAuth 2.0 credentials (or service account)
    5. For admin features, enable Admin SDK and configure domain-wide delegation

    Setup Guide:
    1. Go to https://console.cloud.google.com
    2. Create new project or select existing
    3. Enable APIs: Gmail API, Admin SDK (if needed)
    4. Go to "APIs & Services" > "Credentials"
    5. Create OAuth 2.0 Client ID (type: Desktop or Web application)
    6. Download the credentials JSON file
    7. For service account: Create service account, enable domain-wide delegation
    """

    GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"
    ADMIN_API_BASE = "https://admin.googleapis.com/admin/directory/v1"
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"

    def __init__(self, config: Optional[GoogleConfig] = None):
        """Initialize Google Workspace integration."""
        self.config = config or GoogleConfig()
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._http_client = None

    async def _get_http_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(timeout=30.0)
            except ImportError:
                raise ImportError("httpx required: pip install httpx")
        return self._http_client

    def get_auth_url(self, scopes: List[str] = None) -> str:
        """
        Get OAuth2 authorization URL.

        Args:
            scopes: List of permission scopes

        Returns:
            Authorization URL for user to visit
        """
        if scopes is None:
            scopes = (
                self.config.SCOPES["gmail_modify"] +
                self.config.SCOPES["gmail_send"]
            )

        scope_str = " ".join(scopes)

        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": scope_str,
            "access_type": "offline",
            "prompt": "consent",
            "state": "cosmos_google",
        }

        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.AUTH_URL}?{query}"

    async def exchange_code_for_token(self, auth_code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        client = await self._get_http_client()

        data = {
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "code": auth_code,
            "redirect_uri": self.config.redirect_uri,
            "grant_type": "authorization_code",
        }

        response = await client.post(self.TOKEN_URL, data=data)

        if response.status_code == 200:
            token_data = response.json()
            self._access_token = token_data.get("access_token")
            self._refresh_token = token_data.get("refresh_token")
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires = datetime.now() + timedelta(seconds=expires_in)
            logger.info("Google OAuth token obtained successfully")
            return token_data
        else:
            raise Exception(f"Token exchange failed: {response.text}")

    async def authenticate_service_account(self, user_email: str = None) -> bool:
        """
        Authenticate using service account with domain-wide delegation.

        Args:
            user_email: User to impersonate (required for domain-wide delegation)
        """
        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request as GoogleRequest

            credentials = service_account.Credentials.from_service_account_file(
                self.config.service_account_file,
                scopes=self.config.SCOPES["gmail_full"],
            )

            if user_email:
                credentials = credentials.with_subject(user_email)

            credentials.refresh(GoogleRequest())

            self._access_token = credentials.token
            self._token_expires = credentials.expiry

            logger.info(f"Service account authentication successful for {user_email or 'default'}")
            return True

        except ImportError:
            logger.error("google-auth required: pip install google-auth")
            return False
        except Exception as e:
            logger.error(f"Service account authentication failed: {e}")
            return False

    async def _refresh_access_token(self) -> bool:
        """Refresh the access token."""
        if not self._refresh_token:
            return False

        client = await self._get_http_client()

        data = {
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "refresh_token": self._refresh_token,
            "grant_type": "refresh_token",
        }

        response = await client.post(self.TOKEN_URL, data=data)

        if response.status_code == 200:
            token_data = response.json()
            self._access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires = datetime.now() + timedelta(seconds=expires_in)
            return True

        return False

    async def _make_request(
        self,
        method: str,
        url: str,
        data: Dict = None,
        params: Dict = None,
    ) -> Dict[str, Any]:
        """Make authenticated request to Google API."""
        if not self._access_token:
            raise Exception("Not authenticated. Call authenticate first.")

        if self._token_expires and datetime.now() >= self._token_expires:
            if not await self._refresh_access_token():
                raise Exception("Token expired and refresh failed")

        client = await self._get_http_client()
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        if method == "GET":
            response = await client.get(url, headers=headers, params=params)
        elif method == "POST":
            response = await client.post(url, headers=headers, json=data)
        elif method == "PATCH":
            response = await client.patch(url, headers=headers, json=data)
        elif method == "DELETE":
            response = await client.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code >= 400:
            raise Exception(f"API error ({response.status_code}): {response.text}")

        if response.status_code == 204 or not response.text:
            return {}

        return response.json()

    # ========== Gmail Operations ==========

    async def list_messages(
        self,
        user: str = "me",
        query: str = None,
        label_ids: List[str] = None,
        max_results: int = 50,
        include_spam_trash: bool = False,
    ) -> List[GmailMessage]:
        """
        List Gmail messages.

        Args:
            user: User ID or "me" for authenticated user
            query: Gmail search query (same as search box)
            label_ids: Filter by labels
            max_results: Maximum messages to return
            include_spam_trash: Include spam and trash

        Returns:
            List of GmailMessage objects
        """
        params = {"maxResults": max_results}

        if query:
            params["q"] = query
        if label_ids:
            params["labelIds"] = ",".join(label_ids)
        if include_spam_trash:
            params["includeSpamTrash"] = "true"

        url = f"{self.GMAIL_API_BASE}/users/{user}/messages"
        result = await self._make_request("GET", url, params=params)

        messages = []
        for msg_ref in result.get("messages", []):
            # Get full message details
            msg = await self.get_message(msg_ref["id"], user)
            messages.append(msg)

        return messages

    async def get_message(
        self,
        message_id: str,
        user: str = "me",
        format: str = "full",
    ) -> GmailMessage:
        """
        Get a single Gmail message.

        Args:
            message_id: Message ID
            user: User ID or "me"
            format: "minimal", "full", "raw", or "metadata"
        """
        url = f"{self.GMAIL_API_BASE}/users/{user}/messages/{message_id}"
        params = {"format": format}

        result = await self._make_request("GET", url, params=params)
        return self._parse_message(result)

    async def send_message(
        self,
        to: List[str],
        subject: str,
        body: str,
        cc: List[str] = None,
        bcc: List[str] = None,
        body_type: str = "text",
        user: str = "me",
    ) -> str:
        """
        Send an email message.

        Args:
            to: List of recipient addresses
            subject: Email subject
            body: Email body
            cc: CC recipients
            bcc: BCC recipients
            body_type: "text" or "html"
            user: Sender user ID or "me"

        Returns:
            Sent message ID
        """
        # Create MIME message
        if body_type == "html":
            msg = MIMEMultipart("alternative")
            msg.attach(MIMEText(body, "html"))
        else:
            msg = MIMEText(body)

        msg["To"] = ", ".join(to)
        msg["Subject"] = subject

        if cc:
            msg["Cc"] = ", ".join(cc)
        if bcc:
            msg["Bcc"] = ", ".join(bcc)

        # Encode message
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

        url = f"{self.GMAIL_API_BASE}/users/{user}/messages/send"
        data = {"raw": raw}

        result = await self._make_request("POST", url, data=data)

        logger.info(f"Email sent to {', '.join(to)}")
        return result.get("id", "")

    async def modify_message(
        self,
        message_id: str,
        add_labels: List[str] = None,
        remove_labels: List[str] = None,
        user: str = "me",
    ) -> bool:
        """Modify message labels."""
        url = f"{self.GMAIL_API_BASE}/users/{user}/messages/{message_id}/modify"
        data = {}

        if add_labels:
            data["addLabelIds"] = add_labels
        if remove_labels:
            data["removeLabelIds"] = remove_labels

        await self._make_request("POST", url, data=data)
        return True

    async def trash_message(self, message_id: str, user: str = "me") -> bool:
        """Move message to trash."""
        url = f"{self.GMAIL_API_BASE}/users/{user}/messages/{message_id}/trash"
        await self._make_request("POST", url)
        return True

    async def delete_message(self, message_id: str, user: str = "me") -> bool:
        """Permanently delete message."""
        url = f"{self.GMAIL_API_BASE}/users/{user}/messages/{message_id}"
        await self._make_request("DELETE", url)
        return True

    async def mark_as_read(
        self,
        message_id: str,
        is_read: bool = True,
        user: str = "me",
    ) -> bool:
        """Mark message as read/unread."""
        if is_read:
            return await self.modify_message(message_id, remove_labels=["UNREAD"], user=user)
        else:
            return await self.modify_message(message_id, add_labels=["UNREAD"], user=user)

    # ========== Labels ==========

    async def list_labels(self, user: str = "me") -> List[GmailLabel]:
        """List all Gmail labels."""
        url = f"{self.GMAIL_API_BASE}/users/{user}/labels"
        result = await self._make_request("GET", url)

        labels = []
        for label_data in result.get("labels", []):
            labels.append(GmailLabel(
                id=label_data.get("id", ""),
                name=label_data.get("name", ""),
                type=label_data.get("type", "user"),
                message_list_visibility=label_data.get("messageListVisibility", "show"),
                label_list_visibility=label_data.get("labelListVisibility", "labelShow"),
                messages_total=label_data.get("messagesTotal", 0),
                messages_unread=label_data.get("messagesUnread", 0),
            ))

        return labels

    async def create_label(
        self,
        name: str,
        user: str = "me",
        background_color: str = None,
        text_color: str = None,
    ) -> GmailLabel:
        """Create a new label."""
        url = f"{self.GMAIL_API_BASE}/users/{user}/labels"
        data = {"name": name}

        if background_color or text_color:
            data["color"] = {}
            if background_color:
                data["color"]["backgroundColor"] = background_color
            if text_color:
                data["color"]["textColor"] = text_color

        result = await self._make_request("POST", url, data=data)

        return GmailLabel(
            id=result.get("id", ""),
            name=result.get("name", ""),
        )

    # ========== Filters ==========

    async def list_filters(self, user: str = "me") -> List[Dict[str, Any]]:
        """List Gmail filters."""
        url = f"{self.GMAIL_API_BASE}/users/{user}/settings/filters"
        result = await self._make_request("GET", url)
        return result.get("filter", [])

    async def create_filter(
        self,
        criteria: Dict[str, Any],
        action: Dict[str, Any],
        user: str = "me",
    ) -> Dict[str, Any]:
        """
        Create a Gmail filter.

        Example criteria:
        {
            "from": "alerts@example.com",
            "to": "me",
            "subject": "urgent",
            "hasAttachment": True,
            "query": "is:important"
        }

        Example action:
        {
            "addLabelIds": ["Label_123"],
            "removeLabelIds": ["INBOX"],
            "forward": "other@example.com"
        }
        """
        url = f"{self.GMAIL_API_BASE}/users/{user}/settings/filters"
        data = {
            "criteria": criteria,
            "action": action,
        }

        result = await self._make_request("POST", url, data=data)
        logger.info(f"Created Gmail filter: {criteria}")
        return result

    async def delete_filter(self, filter_id: str, user: str = "me") -> bool:
        """Delete a Gmail filter."""
        url = f"{self.GMAIL_API_BASE}/users/{user}/settings/filters/{filter_id}"
        await self._make_request("DELETE", url)
        return True

    # ========== Admin Functions ==========

    async def list_users(self, domain: str = None) -> List[Dict[str, Any]]:
        """
        List users in the domain (requires Admin SDK).

        Requires: admin.directory.user.readonly or admin.directory.user scope
        """
        url = f"{self.ADMIN_API_BASE}/users"
        params = {}

        if domain:
            params["domain"] = domain
        else:
            params["customer"] = "my_customer"

        try:
            result = await self._make_request("GET", url, params=params)
            return result.get("users", [])
        except Exception as e:
            logger.error(f"Failed to list users (requires Admin SDK permission): {e}")
            return []

    # ========== Helper Methods ==========

    def _parse_message(self, data: Dict[str, Any]) -> GmailMessage:
        """Parse API response to GmailMessage."""
        msg = GmailMessage(
            id=data.get("id", ""),
            thread_id=data.get("threadId", ""),
            label_ids=data.get("labelIds", []),
            snippet=data.get("snippet", ""),
        )

        # Parse headers
        payload = data.get("payload", {})
        headers = payload.get("headers", [])

        for header in headers:
            name = header.get("name", "").lower()
            value = header.get("value", "")
            msg.headers[name] = value

            if name == "from":
                # Parse "Name <email>" format
                if "<" in value:
                    parts = value.split("<")
                    msg.sender_name = parts[0].strip().strip('"')
                    msg.sender = parts[1].rstrip(">")
                else:
                    msg.sender = value
            elif name == "to":
                msg.recipients = [r.strip() for r in value.split(",")]
            elif name == "cc":
                msg.cc = [r.strip() for r in value.split(",")]
            elif name == "subject":
                msg.subject = value
            elif name == "date":
                msg.date = self._parse_date(value)

        # Check labels
        msg.is_read = "UNREAD" not in msg.label_ids
        msg.is_starred = "STARRED" in msg.label_ids

        # Parse body
        msg.body, msg.body_type = self._extract_body(payload)

        # Check attachments
        parts = payload.get("parts", [])
        for part in parts:
            if part.get("filename"):
                msg.has_attachments = True
                msg.attachments.append({
                    "filename": part.get("filename"),
                    "mime_type": part.get("mimeType"),
                    "attachment_id": part.get("body", {}).get("attachmentId"),
                })

        return msg

    def _extract_body(self, payload: Dict) -> tuple:
        """Extract body from message payload."""
        body = ""
        body_type = "text"

        # Check direct body
        if "body" in payload and payload["body"].get("data"):
            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
            body_type = "html" if payload.get("mimeType") == "text/html" else "text"
            return body, body_type

        # Check parts
        parts = payload.get("parts", [])
        for part in parts:
            mime_type = part.get("mimeType", "")

            if mime_type == "text/plain" and not body:
                if part.get("body", {}).get("data"):
                    body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
                    body_type = "text"

            elif mime_type == "text/html":
                if part.get("body", {}).get("data"):
                    body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
                    body_type = "html"
                    break  # Prefer HTML

            # Handle multipart
            if "parts" in part:
                nested_body, nested_type = self._extract_body(part)
                if nested_body:
                    body = nested_body
                    body_type = nested_type

        return body, body_type

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse email date string."""
        from email.utils import parsedate_to_datetime

        try:
            return parsedate_to_datetime(date_str)
        except Exception:
            return None

    @staticmethod
    def get_setup_guide() -> str:
        """Get setup instructions for Google Workspace integration."""
        return """
# Google Workspace / Gmail Integration Setup Guide

## Prerequisites
- Google Cloud Platform account
- Google Workspace domain (for admin features)
- Domain administrator access (for service account delegation)

## Step 1: Create Google Cloud Project

1. Go to https://console.cloud.google.com
2. Click "Select a project" > "New Project"
3. Name your project (e.g., "cosmos Integration")
4. Click "Create"

## Step 2: Enable Required APIs

1. Go to "APIs & Services" > "Library"
2. Search and enable:
   - Gmail API
   - Admin SDK API (for admin features)
   - Google Calendar API (if needed)

## Step 3: Configure OAuth Consent Screen

1. Go to "APIs & Services" > "OAuth consent screen"
2. Select "Internal" (for Workspace users) or "External"
3. Fill in app information:
   - App name: "cosmos Integration"
   - User support email: your email
   - Developer contact: your email
4. Add scopes:
   - https://www.googleapis.com/auth/gmail.modify
   - https://www.googleapis.com/auth/gmail.send
5. Save and continue

## Step 4a: Create OAuth 2.0 Credentials (User Auth)

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. Application type: "Desktop app" or "Web application"
4. Name: "cosmos OAuth Client"
5. For Web app, add redirect URI: http://localhost:8400/callback
6. Click "Create"
7. Download the credentials JSON file

## Step 4b: Create Service Account (App Auth - Optional)

For automated/server-side access:

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "Service account"
3. Name: "cosmos Service Account"
4. Click "Create and Continue"
5. Grant role: "Editor" (or custom role with Gmail permissions)
6. Click "Done"
7. Click on the service account > "Keys" tab
8. Add Key > Create new key > JSON
9. Download the key file

## Step 5: Configure Domain-Wide Delegation (Service Account)

For service account to access user mailboxes:

1. Go to Google Workspace Admin Console (admin.google.com)
2. Navigate to Security > API Controls > Domain-wide delegation
3. Click "Add new"
4. Enter Client ID from service account
5. Add OAuth scopes:
   - https://www.googleapis.com/auth/gmail.modify
   - https://www.googleapis.com/auth/gmail.send
   - https://mail.google.com/
6. Click "Authorize"

## Step 6: Configure cosmos

Add to your environment or config:

```bash
# For OAuth (user authentication)
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8400/callback

# For Service Account
GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/service-account.json
GOOGLE_DELEGATED_USER=admin@yourdomain.com
```

## Step 7: Test Connection

```python
from Cosmos.integration.email.google_workspace import GoogleWorkspaceIntegration, GoogleConfig

# OAuth flow
config = GoogleConfig(
    client_id="your-client-id",
    client_secret="your-client-secret",
)
integration = GoogleWorkspaceIntegration(config)

# Get auth URL for user to visit
print(integration.get_auth_url())

# After user authorizes, exchange code
await integration.exchange_code_for_token(auth_code)

# List messages
messages = await integration.list_messages()
```

## Security Notes

- Store credentials securely (environment variables or secret manager)
- Use minimum required OAuth scopes
- For service accounts, limit domain-wide delegation scopes
- Enable 2FA on admin accounts
- Monitor API usage in Cloud Console
- Rotate service account keys regularly
"""


# Global instance (requires configuration)
google_workspace_integration = GoogleWorkspaceIntegration()
