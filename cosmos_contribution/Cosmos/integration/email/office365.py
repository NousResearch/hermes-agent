"""
cosmos Office 365 Integration

"Good news, everyone! I can manage your Microsoft 365 email!"

Full Microsoft 365 / Office 365 integration including:
- Email management
- Calendar integration
- Mailbox filtering
- Security & Compliance
- Admin functions (with proper permissions)
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from loguru import logger


class O365AuthType(Enum):
    """Office 365 authentication types."""
    DELEGATED = "delegated"  # User-based OAuth
    APPLICATION = "application"  # App-only with client credentials
    DEVICE_CODE = "device_code"  # Device code flow for CLI


@dataclass
class O365Config:
    """Office 365 configuration."""
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""  # For application auth
    redirect_uri: str = "http://localhost:8400/callback"
    auth_type: O365AuthType = O365AuthType.DELEGATED

    # Required scopes by feature
    SCOPES = {
        "mail_read": ["Mail.Read", "Mail.ReadBasic"],
        "mail_write": ["Mail.ReadWrite", "Mail.Send"],
        "mailbox_settings": ["MailboxSettings.ReadWrite"],
        "calendar": ["Calendars.ReadWrite"],
        "contacts": ["Contacts.ReadWrite"],
        "user": ["User.Read", "User.ReadBasic.All"],
        "admin": ["Directory.Read.All", "User.Read.All"],
        "security": ["SecurityEvents.Read.All", "ThreatIndicators.Read.All"],
    }


@dataclass
class EmailMessage:
    """Email message structure."""
    id: str = ""
    subject: str = ""
    sender: str = ""
    sender_name: str = ""
    recipients: List[str] = field(default_factory=list)
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    body: str = ""
    body_type: str = "text"  # text or html
    received_datetime: Optional[datetime] = None
    sent_datetime: Optional[datetime] = None
    is_read: bool = False
    importance: str = "normal"
    has_attachments: bool = False
    attachments: List[Dict] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class MailFolder:
    """Mail folder structure."""
    id: str
    display_name: str
    parent_folder_id: str = ""
    child_folder_count: int = 0
    total_item_count: int = 0
    unread_item_count: int = 0


class Office365Integration:
    """
    Microsoft 365 / Office 365 Integration.

    Prerequisites:
    1. Register an app in Azure AD (Entra ID)
    2. Configure API permissions (Microsoft Graph)
    3. Create client secret (for app-only auth)
    4. Set up redirect URI for delegated auth

    Setup Guide:
    1. Go to Azure Portal > Azure Active Directory > App registrations
    2. Click "New registration"
    3. Name your app (e.g., "cosmos Email Integration")
    4. Select account type (single tenant or multi-tenant)
    5. Set redirect URI to http://localhost:8400/callback
    6. Under "API Permissions", add Microsoft Graph permissions:
       - Mail.ReadWrite
       - Mail.Send
       - MailboxSettings.ReadWrite
       - User.Read
    7. Create a client secret under "Certificates & secrets"
    8. Grant admin consent if using application permissions
    """

    GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"
    AUTH_URL = "https://login.microsoftonline.com"

    def __init__(self, config: Optional[O365Config] = None):
        """Initialize Office 365 integration."""
        self.config = config or O365Config()
        self._access_token: Optional[str] = None
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
                self.config.SCOPES["mail_read"] +
                self.config.SCOPES["mail_write"] +
                self.config.SCOPES["user"]
            )

        scope_str = " ".join([f"https://graph.microsoft.com/{s}" for s in scopes])

        params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "redirect_uri": self.config.redirect_uri,
            "response_mode": "query",
            "scope": scope_str + " offline_access",
            "state": "cosmos_o365",
        }

        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.AUTH_URL}/{self.config.tenant_id}/oauth2/v2.0/authorize?{query}"

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

        response = await client.post(
            f"{self.AUTH_URL}/{self.config.tenant_id}/oauth2/v2.0/token",
            data=data,
        )

        if response.status_code == 200:
            token_data = response.json()
            self._access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires = datetime.now() + timedelta(seconds=expires_in)
            return token_data
        else:
            raise Exception(f"Token exchange failed: {response.text}")

    async def authenticate_app_only(self) -> bool:
        """Authenticate using client credentials (app-only)."""
        client = await self._get_http_client()

        data = {
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials",
        }

        response = await client.post(
            f"{self.AUTH_URL}/{self.config.tenant_id}/oauth2/v2.0/token",
            data=data,
        )

        if response.status_code == 200:
            token_data = response.json()
            self._access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires = datetime.now() + timedelta(seconds=expires_in)
            logger.info("Office 365 app-only authentication successful")
            return True
        else:
            logger.error(f"Authentication failed: {response.text}")
            return False

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Dict = None,
        params: Dict = None,
    ) -> Dict[str, Any]:
        """Make authenticated request to Graph API."""
        if not self._access_token:
            raise Exception("Not authenticated. Call authenticate first.")

        if self._token_expires and datetime.now() >= self._token_expires:
            # Token expired, re-authenticate
            await self.authenticate_app_only()

        client = await self._get_http_client()
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        url = f"{self.GRAPH_API_BASE}{endpoint}"

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

        if response.status_code == 204:
            return {}

        return response.json()

    # ========== Mail Operations ==========

    async def list_messages(
        self,
        folder: str = "inbox",
        user: str = "me",
        top: int = 50,
        filter_query: str = None,
        select: List[str] = None,
    ) -> List[EmailMessage]:
        """
        List email messages.

        Args:
            folder: Folder name (inbox, sentitems, drafts, etc.)
            user: User ID or "me" for current user
            top: Number of messages to retrieve
            filter_query: OData filter query
            select: Fields to select

        Returns:
            List of EmailMessage objects
        """
        params = {"$top": top}

        if filter_query:
            params["$filter"] = filter_query

        if select:
            params["$select"] = ",".join(select)
        else:
            params["$select"] = "id,subject,from,toRecipients,receivedDateTime,isRead,importance,hasAttachments,bodyPreview"

        endpoint = f"/users/{user}/mailFolders/{folder}/messages"
        result = await self._make_request("GET", endpoint, params=params)

        messages = []
        for msg_data in result.get("value", []):
            messages.append(self._parse_message(msg_data))

        return messages

    async def get_message(
        self,
        message_id: str,
        user: str = "me",
        include_headers: bool = False,
    ) -> EmailMessage:
        """Get a single email message with full details."""
        endpoint = f"/users/{user}/messages/{message_id}"

        params = {}
        if include_headers:
            params["$select"] = "*"
            params["$expand"] = "singleValueExtendedProperties($filter=id eq 'String 0x007D')"

        result = await self._make_request("GET", endpoint, params=params)
        message = self._parse_message(result)

        # Get headers if requested
        if include_headers:
            headers_endpoint = f"/users/{user}/messages/{message_id}?$select=internetMessageHeaders"
            headers_result = await self._make_request("GET", headers_endpoint)
            if "internetMessageHeaders" in headers_result:
                message.headers = {
                    h["name"]: h["value"]
                    for h in headers_result["internetMessageHeaders"]
                }

        return message

    async def send_message(
        self,
        to: List[str],
        subject: str,
        body: str,
        cc: List[str] = None,
        bcc: List[str] = None,
        body_type: str = "text",
        user: str = "me",
        save_to_sent: bool = True,
    ) -> bool:
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
            save_to_sent: Save to sent items

        Returns:
            Success status
        """
        message = {
            "subject": subject,
            "body": {
                "contentType": "HTML" if body_type == "html" else "Text",
                "content": body,
            },
            "toRecipients": [
                {"emailAddress": {"address": addr}} for addr in to
            ],
        }

        if cc:
            message["ccRecipients"] = [
                {"emailAddress": {"address": addr}} for addr in cc
            ]

        if bcc:
            message["bccRecipients"] = [
                {"emailAddress": {"address": addr}} for addr in bcc
            ]

        endpoint = f"/users/{user}/sendMail"
        data = {
            "message": message,
            "saveToSentItems": save_to_sent,
        }

        await self._make_request("POST", endpoint, data=data)
        logger.info(f"Email sent to {', '.join(to)}")
        return True

    async def move_message(
        self,
        message_id: str,
        destination_folder_id: str,
        user: str = "me",
    ) -> bool:
        """Move a message to another folder."""
        endpoint = f"/users/{user}/messages/{message_id}/move"
        data = {"destinationId": destination_folder_id}
        await self._make_request("POST", endpoint, data=data)
        return True

    async def delete_message(
        self,
        message_id: str,
        user: str = "me",
        permanent: bool = False,
    ) -> bool:
        """Delete a message."""
        if permanent:
            endpoint = f"/users/{user}/messages/{message_id}"
            await self._make_request("DELETE", endpoint)
        else:
            # Move to deleted items
            await self.move_message(message_id, "deleteditems", user)
        return True

    async def mark_as_read(
        self,
        message_id: str,
        is_read: bool = True,
        user: str = "me",
    ) -> bool:
        """Mark message as read/unread."""
        endpoint = f"/users/{user}/messages/{message_id}"
        data = {"isRead": is_read}
        await self._make_request("PATCH", endpoint, data=data)
        return True

    # ========== Folder Operations ==========

    async def list_folders(self, user: str = "me") -> List[MailFolder]:
        """List all mail folders."""
        endpoint = f"/users/{user}/mailFolders"
        result = await self._make_request("GET", endpoint)

        folders = []
        for folder_data in result.get("value", []):
            folders.append(MailFolder(
                id=folder_data.get("id", ""),
                display_name=folder_data.get("displayName", ""),
                parent_folder_id=folder_data.get("parentFolderId", ""),
                child_folder_count=folder_data.get("childFolderCount", 0),
                total_item_count=folder_data.get("totalItemCount", 0),
                unread_item_count=folder_data.get("unreadItemCount", 0),
            ))

        return folders

    async def create_folder(
        self,
        name: str,
        parent_folder_id: str = None,
        user: str = "me",
    ) -> MailFolder:
        """Create a new mail folder."""
        endpoint = f"/users/{user}/mailFolders"
        if parent_folder_id:
            endpoint = f"/users/{user}/mailFolders/{parent_folder_id}/childFolders"

        data = {"displayName": name}
        result = await self._make_request("POST", endpoint, data=data)

        return MailFolder(
            id=result.get("id", ""),
            display_name=result.get("displayName", ""),
        )

    # ========== Rules & Filters ==========

    async def list_rules(self, user: str = "me") -> List[Dict[str, Any]]:
        """List mailbox rules."""
        endpoint = f"/users/{user}/mailFolders/inbox/messageRules"
        result = await self._make_request("GET", endpoint)
        return result.get("value", [])

    async def create_rule(
        self,
        name: str,
        conditions: Dict[str, Any],
        actions: Dict[str, Any],
        user: str = "me",
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a mailbox rule.

        Example conditions:
        {
            "senderContains": ["alert@example.com"],
            "subjectContains": ["urgent"],
            "importance": "high"
        }

        Example actions:
        {
            "moveToFolder": "folder-id",
            "markAsRead": True,
            "forwardTo": [{"emailAddress": {"address": "other@example.com"}}]
        }
        """
        endpoint = f"/users/{user}/mailFolders/inbox/messageRules"
        data = {
            "displayName": name,
            "sequence": 1,
            "isEnabled": enabled,
            "conditions": conditions,
            "actions": actions,
        }

        result = await self._make_request("POST", endpoint, data=data)
        logger.info(f"Created mailbox rule: {name}")
        return result

    # ========== Security & Admin ==========

    async def get_message_trace(
        self,
        sender: str = None,
        recipient: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> List[Dict[str, Any]]:
        """
        Get message trace (requires Exchange admin permissions).

        Note: This uses Exchange Online PowerShell or Security & Compliance API.
        For full message trace, consider using Microsoft Defender for Office 365 API.
        """
        # Graph API has limited message trace - would need Security API or PowerShell
        logger.warning("Full message trace requires Security & Compliance Center access")
        return []

    async def get_security_alerts(self) -> List[Dict[str, Any]]:
        """Get security alerts (requires SecurityEvents.Read.All permission)."""
        endpoint = "/security/alerts"
        try:
            result = await self._make_request("GET", endpoint)
            return result.get("value", [])
        except Exception as e:
            logger.error(f"Failed to get security alerts: {e}")
            return []

    # ========== Helper Methods ==========

    def _parse_message(self, data: Dict[str, Any]) -> EmailMessage:
        """Parse API response to EmailMessage."""
        from_data = data.get("from", {}).get("emailAddress", {})

        return EmailMessage(
            id=data.get("id", ""),
            subject=data.get("subject", ""),
            sender=from_data.get("address", ""),
            sender_name=from_data.get("name", ""),
            recipients=[
                r.get("emailAddress", {}).get("address", "")
                for r in data.get("toRecipients", [])
            ],
            cc=[
                r.get("emailAddress", {}).get("address", "")
                for r in data.get("ccRecipients", [])
            ],
            body=data.get("body", {}).get("content", data.get("bodyPreview", "")),
            body_type=data.get("body", {}).get("contentType", "text").lower(),
            received_datetime=self._parse_datetime(data.get("receivedDateTime")),
            sent_datetime=self._parse_datetime(data.get("sentDateTime")),
            is_read=data.get("isRead", False),
            importance=data.get("importance", "normal"),
            has_attachments=data.get("hasAttachments", False),
            categories=data.get("categories", []),
        )

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def get_setup_guide() -> str:
        """Get setup instructions for Office 365 integration."""
        return """
# Office 365 / Microsoft 365 Integration Setup Guide

## Prerequisites
- Azure AD (Entra ID) tenant
- Global Admin or Application Admin permissions
- Microsoft 365 subscription with Exchange Online

## Step 1: Register Azure AD Application

1. Go to https://portal.azure.com
2. Navigate to "Azure Active Directory" > "App registrations"
3. Click "New registration"
4. Fill in details:
   - Name: "cosmos Email Integration"
   - Supported account types: Single tenant (or multi-tenant if needed)
   - Redirect URI: Web - http://localhost:8400/callback
5. Click "Register"
6. Note down:
   - Application (client) ID
   - Directory (tenant) ID

## Step 2: Configure API Permissions

1. In your app registration, go to "API permissions"
2. Click "Add a permission" > "Microsoft Graph"
3. Add these permissions:

   **Delegated Permissions** (for user-based access):
   - Mail.Read
   - Mail.ReadWrite
   - Mail.Send
   - MailboxSettings.ReadWrite
   - User.Read

   **Application Permissions** (for app-only/daemon access):
   - Mail.Read
   - Mail.ReadWrite
   - Mail.Send
   - User.Read.All

4. Click "Grant admin consent" (requires admin)

## Step 3: Create Client Secret

1. Go to "Certificates & secrets"
2. Click "New client secret"
3. Add description and expiration
4. Copy the secret value immediately (shown only once)

## Step 4: Configure cosmos

Add to your environment or config:

```bash
# Required
O365_TENANT_ID=your-tenant-id
O365_CLIENT_ID=your-client-id
O365_CLIENT_SECRET=your-client-secret

# Optional
O365_REDIRECT_URI=http://localhost:8400/callback
```

## Step 5: Test Connection

```python
from Cosmos.integration.email.office365 import Office365Integration, O365Config

config = O365Config(
    tenant_id="your-tenant-id",
    client_id="your-client-id",
    client_secret="your-client-secret",
)

integration = Office365Integration(config)

# For app-only auth
await integration.authenticate_app_only()

# List messages
messages = await integration.list_messages(user="user@yourdomain.com")
```

## Security Notes

- Store credentials securely (environment variables or key vault)
- Use minimum required permissions
- Enable conditional access policies
- Monitor sign-in logs in Azure AD
- Rotate client secrets regularly
"""


# Global instance (requires configuration)
office365_integration = Office365Integration()
