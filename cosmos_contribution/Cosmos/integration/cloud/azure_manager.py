"""
cosmos Azure Manager

"Good news, everyone! I can manage your entire Azure estate!"

Comprehensive Azure / Entra ID management for sysadmins.

Capabilities:
- Entra ID (Azure AD) user and group management
- Azure resource management
- VM operations
- Storage management
- Network configuration
- Security & Compliance
- Cost management
- Monitoring & Alerts
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from loguru import logger


class AzureAuthMethod(Enum):
    """Azure authentication methods."""
    SERVICE_PRINCIPAL = "service_principal"
    MANAGED_IDENTITY = "managed_identity"
    CLI = "cli"
    DEVICE_CODE = "device_code"
    INTERACTIVE = "interactive"


class ResourceState(Enum):
    """Azure resource states."""
    RUNNING = "running"
    STOPPED = "stopped"
    DEALLOCATED = "deallocated"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


@dataclass
class AzureConfig:
    """Azure configuration."""
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    subscription_id: str = ""
    auth_method: AzureAuthMethod = AzureAuthMethod.SERVICE_PRINCIPAL

    # Management API endpoints
    MANAGEMENT_URL = "https://management.azure.com"
    GRAPH_URL = "https://graph.microsoft.com/v1.0"
    AUTH_URL = "https://login.microsoftonline.com"


@dataclass
class AzureResource:
    """Azure resource representation."""
    id: str
    name: str
    type: str
    location: str
    resource_group: str
    tags: Dict[str, str] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    state: ResourceState = ResourceState.UNKNOWN


@dataclass
class EntraUser:
    """Entra ID (Azure AD) user."""
    id: str
    display_name: str
    user_principal_name: str
    mail: str = ""
    job_title: str = ""
    department: str = ""
    account_enabled: bool = True
    created_datetime: Optional[datetime] = None
    last_sign_in: Optional[datetime] = None
    assigned_licenses: List[str] = field(default_factory=list)
    member_of: List[str] = field(default_factory=list)


@dataclass
class EntraGroup:
    """Entra ID group."""
    id: str
    display_name: str
    description: str = ""
    mail: str = ""
    group_types: List[str] = field(default_factory=list)
    member_count: int = 0


class AzureManager:
    """
    Comprehensive Azure management.

    Prerequisites:
    1. Azure subscription
    2. App registration in Entra ID with appropriate permissions
    3. Service principal with required role assignments

    Required Permissions (by feature):
    - Resource Management: Reader, Contributor, or Owner on subscription
    - Entra ID: User.Read.All, Group.Read.All, Directory.Read.All
    - VM Management: Virtual Machine Contributor
    - Storage: Storage Account Contributor
    - Network: Network Contributor
    """

    def __init__(self, config: Optional[AzureConfig] = None):
        """Initialize Azure manager."""
        self.config = config or AzureConfig()
        self._access_token: Optional[str] = None
        self._graph_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._http_client = None

    async def _get_http_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(timeout=60.0)
            except ImportError:
                raise ImportError("httpx required: pip install httpx")
        return self._http_client

    async def authenticate(self) -> bool:
        """
        Authenticate with Azure.

        Returns:
            Success status
        """
        if self.config.auth_method == AzureAuthMethod.SERVICE_PRINCIPAL:
            return await self._auth_service_principal()
        elif self.config.auth_method == AzureAuthMethod.CLI:
            return await self._auth_cli()
        else:
            logger.error(f"Auth method {self.config.auth_method} not implemented")
            return False

    async def _auth_service_principal(self) -> bool:
        """Authenticate using service principal."""
        client = await self._get_http_client()

        # Get management token
        data = {
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "scope": "https://management.azure.com/.default",
            "grant_type": "client_credentials",
        }

        response = await client.post(
            f"{self.config.AUTH_URL}/{self.config.tenant_id}/oauth2/v2.0/token",
            data=data,
        )

        if response.status_code == 200:
            token_data = response.json()
            self._access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires = datetime.now() + timedelta(seconds=expires_in)
        else:
            logger.error(f"Management auth failed: {response.text}")
            return False

        # Get Graph token
        data["scope"] = "https://graph.microsoft.com/.default"
        response = await client.post(
            f"{self.config.AUTH_URL}/{self.config.tenant_id}/oauth2/v2.0/token",
            data=data,
        )

        if response.status_code == 200:
            token_data = response.json()
            self._graph_token = token_data.get("access_token")
            logger.info("Azure authentication successful")
            return True
        else:
            logger.error(f"Graph auth failed: {response.text}")
            return False

    async def _auth_cli(self) -> bool:
        """Authenticate using Azure CLI credentials."""
        try:
            import subprocess
            result = subprocess.run(
                ["az", "account", "get-access-token", "--output", "json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                token_data = json.loads(result.stdout)
                self._access_token = token_data.get("accessToken")
                self._token_expires = datetime.fromisoformat(
                    token_data.get("expiresOn", "").replace(" ", "T")
                )

                # Get Graph token
                result = subprocess.run(
                    ["az", "account", "get-access-token",
                     "--resource", "https://graph.microsoft.com",
                     "--output", "json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    token_data = json.loads(result.stdout)
                    self._graph_token = token_data.get("accessToken")
                    logger.info("Azure CLI authentication successful")
                    return True

        except Exception as e:
            logger.error(f"CLI auth failed: {e}")

        return False

    async def _make_request(
        self,
        method: str,
        url: str,
        token_type: str = "management",
        data: Dict = None,
        params: Dict = None,
    ) -> Dict[str, Any]:
        """Make authenticated Azure API request."""
        token = self._access_token if token_type == "management" else self._graph_token

        if not token:
            raise Exception("Not authenticated")

        client = await self._get_http_client()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        if method == "GET":
            response = await client.get(url, headers=headers, params=params)
        elif method == "POST":
            response = await client.post(url, headers=headers, json=data)
        elif method == "PATCH":
            response = await client.patch(url, headers=headers, json=data)
        elif method == "PUT":
            response = await client.put(url, headers=headers, json=data)
        elif method == "DELETE":
            response = await client.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code >= 400:
            raise Exception(f"API error ({response.status_code}): {response.text}")

        if response.status_code == 204 or not response.text:
            return {}

        return response.json()

    # ========== Resource Management ==========

    async def list_subscriptions(self) -> List[Dict[str, Any]]:
        """List all subscriptions."""
        url = f"{self.config.MANAGEMENT_URL}/subscriptions?api-version=2022-12-01"
        result = await self._make_request("GET", url)
        return result.get("value", [])

    async def list_resource_groups(
        self,
        subscription_id: str = None,
    ) -> List[Dict[str, Any]]:
        """List resource groups in subscription."""
        sub_id = subscription_id or self.config.subscription_id
        url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/resourcegroups?api-version=2022-09-01"
        result = await self._make_request("GET", url)
        return result.get("value", [])

    async def list_resources(
        self,
        resource_group: str = None,
        resource_type: str = None,
        subscription_id: str = None,
    ) -> List[AzureResource]:
        """List resources with optional filtering."""
        sub_id = subscription_id or self.config.subscription_id

        if resource_group:
            url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/resourceGroups/{resource_group}/resources?api-version=2022-09-01"
        else:
            url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/resources?api-version=2022-09-01"

        if resource_type:
            url += f"&$filter=resourceType eq '{resource_type}'"

        result = await self._make_request("GET", url)

        resources = []
        for r in result.get("value", []):
            resources.append(AzureResource(
                id=r.get("id", ""),
                name=r.get("name", ""),
                type=r.get("type", ""),
                location=r.get("location", ""),
                resource_group=r.get("id", "").split("/resourceGroups/")[1].split("/")[0] if "/resourceGroups/" in r.get("id", "") else "",
                tags=r.get("tags", {}),
            ))

        return resources

    async def create_resource_group(
        self,
        name: str,
        location: str,
        tags: Dict[str, str] = None,
        subscription_id: str = None,
    ) -> Dict[str, Any]:
        """Create a resource group."""
        sub_id = subscription_id or self.config.subscription_id
        url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/resourcegroups/{name}?api-version=2022-09-01"

        data = {
            "location": location,
            "tags": tags or {},
        }

        result = await self._make_request("PUT", url, data=data)
        logger.info(f"Created resource group: {name}")
        return result

    async def delete_resource_group(
        self,
        name: str,
        subscription_id: str = None,
    ) -> bool:
        """Delete a resource group."""
        sub_id = subscription_id or self.config.subscription_id
        url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/resourcegroups/{name}?api-version=2022-09-01"

        await self._make_request("DELETE", url)
        logger.info(f"Deleted resource group: {name}")
        return True

    # ========== Virtual Machine Management ==========

    async def list_vms(
        self,
        resource_group: str = None,
        subscription_id: str = None,
    ) -> List[Dict[str, Any]]:
        """List virtual machines."""
        sub_id = subscription_id or self.config.subscription_id

        if resource_group:
            url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/virtualMachines?api-version=2023-09-01"
        else:
            url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/providers/Microsoft.Compute/virtualMachines?api-version=2023-09-01"

        result = await self._make_request("GET", url)
        return result.get("value", [])

    async def get_vm_status(
        self,
        vm_name: str,
        resource_group: str,
        subscription_id: str = None,
    ) -> Dict[str, Any]:
        """Get VM instance view (status)."""
        sub_id = subscription_id or self.config.subscription_id
        url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/virtualMachines/{vm_name}/instanceView?api-version=2023-09-01"

        result = await self._make_request("GET", url)
        return result

    async def start_vm(
        self,
        vm_name: str,
        resource_group: str,
        subscription_id: str = None,
    ) -> bool:
        """Start a VM."""
        sub_id = subscription_id or self.config.subscription_id
        url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/virtualMachines/{vm_name}/start?api-version=2023-09-01"

        await self._make_request("POST", url)
        logger.info(f"Started VM: {vm_name}")
        return True

    async def stop_vm(
        self,
        vm_name: str,
        resource_group: str,
        deallocate: bool = True,
        subscription_id: str = None,
    ) -> bool:
        """Stop a VM."""
        sub_id = subscription_id or self.config.subscription_id
        action = "deallocate" if deallocate else "powerOff"
        url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/virtualMachines/{vm_name}/{action}?api-version=2023-09-01"

        await self._make_request("POST", url)
        logger.info(f"Stopped VM: {vm_name}")
        return True

    async def restart_vm(
        self,
        vm_name: str,
        resource_group: str,
        subscription_id: str = None,
    ) -> bool:
        """Restart a VM."""
        sub_id = subscription_id or self.config.subscription_id
        url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/virtualMachines/{vm_name}/restart?api-version=2023-09-01"

        await self._make_request("POST", url)
        logger.info(f"Restarted VM: {vm_name}")
        return True

    # ========== Entra ID (Azure AD) Management ==========

    async def list_users(
        self,
        filter_query: str = None,
        top: int = 100,
    ) -> List[EntraUser]:
        """List Entra ID users."""
        params = {"$top": top}
        if filter_query:
            params["$filter"] = filter_query

        url = f"{self.config.GRAPH_URL}/users"
        result = await self._make_request("GET", url, token_type="graph", params=params)

        users = []
        for u in result.get("value", []):
            users.append(EntraUser(
                id=u.get("id", ""),
                display_name=u.get("displayName", ""),
                user_principal_name=u.get("userPrincipalName", ""),
                mail=u.get("mail", ""),
                job_title=u.get("jobTitle", ""),
                department=u.get("department", ""),
                account_enabled=u.get("accountEnabled", True),
            ))

        return users

    async def get_user(self, user_id: str) -> EntraUser:
        """Get a specific user."""
        url = f"{self.config.GRAPH_URL}/users/{user_id}"
        u = await self._make_request("GET", url, token_type="graph")

        return EntraUser(
            id=u.get("id", ""),
            display_name=u.get("displayName", ""),
            user_principal_name=u.get("userPrincipalName", ""),
            mail=u.get("mail", ""),
            job_title=u.get("jobTitle", ""),
            department=u.get("department", ""),
            account_enabled=u.get("accountEnabled", True),
        )

    async def create_user(
        self,
        display_name: str,
        user_principal_name: str,
        password: str,
        mail_nickname: str = None,
        force_change_password: bool = True,
    ) -> EntraUser:
        """Create a new Entra ID user."""
        url = f"{self.config.GRAPH_URL}/users"

        data = {
            "displayName": display_name,
            "userPrincipalName": user_principal_name,
            "mailNickname": mail_nickname or user_principal_name.split("@")[0],
            "accountEnabled": True,
            "passwordProfile": {
                "password": password,
                "forceChangePasswordNextSignIn": force_change_password,
            },
        }

        result = await self._make_request("POST", url, token_type="graph", data=data)
        logger.info(f"Created user: {user_principal_name}")

        return EntraUser(
            id=result.get("id", ""),
            display_name=display_name,
            user_principal_name=user_principal_name,
        )

    async def disable_user(self, user_id: str) -> bool:
        """Disable a user account."""
        url = f"{self.config.GRAPH_URL}/users/{user_id}"
        data = {"accountEnabled": False}

        await self._make_request("PATCH", url, token_type="graph", data=data)
        logger.info(f"Disabled user: {user_id}")
        return True

    async def enable_user(self, user_id: str) -> bool:
        """Enable a user account."""
        url = f"{self.config.GRAPH_URL}/users/{user_id}"
        data = {"accountEnabled": True}

        await self._make_request("PATCH", url, token_type="graph", data=data)
        logger.info(f"Enabled user: {user_id}")
        return True

    async def reset_user_password(
        self,
        user_id: str,
        new_password: str,
        force_change: bool = True,
    ) -> bool:
        """Reset user password."""
        url = f"{self.config.GRAPH_URL}/users/{user_id}"
        data = {
            "passwordProfile": {
                "password": new_password,
                "forceChangePasswordNextSignIn": force_change,
            }
        }

        await self._make_request("PATCH", url, token_type="graph", data=data)
        logger.info(f"Reset password for user: {user_id}")
        return True

    async def list_groups(self, top: int = 100) -> List[EntraGroup]:
        """List Entra ID groups."""
        url = f"{self.config.GRAPH_URL}/groups"
        params = {"$top": top}

        result = await self._make_request("GET", url, token_type="graph", params=params)

        groups = []
        for g in result.get("value", []):
            groups.append(EntraGroup(
                id=g.get("id", ""),
                display_name=g.get("displayName", ""),
                description=g.get("description", ""),
                mail=g.get("mail", ""),
                group_types=g.get("groupTypes", []),
            ))

        return groups

    async def add_user_to_group(self, user_id: str, group_id: str) -> bool:
        """Add user to a group."""
        url = f"{self.config.GRAPH_URL}/groups/{group_id}/members/$ref"
        data = {
            "@odata.id": f"{self.config.GRAPH_URL}/directoryObjects/{user_id}"
        }

        await self._make_request("POST", url, token_type="graph", data=data)
        logger.info(f"Added user {user_id} to group {group_id}")
        return True

    async def remove_user_from_group(self, user_id: str, group_id: str) -> bool:
        """Remove user from a group."""
        url = f"{self.config.GRAPH_URL}/groups/{group_id}/members/{user_id}/$ref"

        await self._make_request("DELETE", url, token_type="graph")
        logger.info(f"Removed user {user_id} from group {group_id}")
        return True

    # ========== Storage Management ==========

    async def list_storage_accounts(
        self,
        resource_group: str = None,
        subscription_id: str = None,
    ) -> List[Dict[str, Any]]:
        """List storage accounts."""
        sub_id = subscription_id or self.config.subscription_id

        if resource_group:
            url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/resourceGroups/{resource_group}/providers/Microsoft.Storage/storageAccounts?api-version=2023-01-01"
        else:
            url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/providers/Microsoft.Storage/storageAccounts?api-version=2023-01-01"

        result = await self._make_request("GET", url)
        return result.get("value", [])

    # ========== Network Management ==========

    async def list_virtual_networks(
        self,
        resource_group: str = None,
        subscription_id: str = None,
    ) -> List[Dict[str, Any]]:
        """List virtual networks."""
        sub_id = subscription_id or self.config.subscription_id

        if resource_group:
            url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/resourceGroups/{resource_group}/providers/Microsoft.Network/virtualNetworks?api-version=2023-09-01"
        else:
            url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/providers/Microsoft.Network/virtualNetworks?api-version=2023-09-01"

        result = await self._make_request("GET", url)
        return result.get("value", [])

    async def list_network_security_groups(
        self,
        resource_group: str = None,
        subscription_id: str = None,
    ) -> List[Dict[str, Any]]:
        """List network security groups."""
        sub_id = subscription_id or self.config.subscription_id

        if resource_group:
            url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/resourceGroups/{resource_group}/providers/Microsoft.Network/networkSecurityGroups?api-version=2023-09-01"
        else:
            url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/providers/Microsoft.Network/networkSecurityGroups?api-version=2023-09-01"

        result = await self._make_request("GET", url)
        return result.get("value", [])

    # ========== Monitoring & Alerts ==========

    async def get_metrics(
        self,
        resource_id: str,
        metric_names: List[str],
        timespan: str = "PT1H",
        interval: str = "PT5M",
    ) -> Dict[str, Any]:
        """Get resource metrics."""
        url = f"{self.config.MANAGEMENT_URL}{resource_id}/providers/Microsoft.Insights/metrics"

        params = {
            "api-version": "2023-10-01",
            "metricnames": ",".join(metric_names),
            "timespan": timespan,
            "interval": interval,
        }

        result = await self._make_request("GET", url, params=params)
        return result

    async def list_activity_logs(
        self,
        filter_query: str = None,
        subscription_id: str = None,
    ) -> List[Dict[str, Any]]:
        """List activity log events."""
        sub_id = subscription_id or self.config.subscription_id
        url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/providers/Microsoft.Insights/eventtypes/management/values"

        params = {"api-version": "2015-04-01"}
        if filter_query:
            params["$filter"] = filter_query

        result = await self._make_request("GET", url, params=params)
        return result.get("value", [])

    # ========== Cost Management ==========

    async def get_cost_summary(
        self,
        scope: str = None,
        timeframe: str = "MonthToDate",
        subscription_id: str = None,
    ) -> Dict[str, Any]:
        """Get cost summary."""
        sub_id = subscription_id or self.config.subscription_id
        scope = scope or f"/subscriptions/{sub_id}"

        url = f"{self.config.MANAGEMENT_URL}{scope}/providers/Microsoft.CostManagement/query?api-version=2023-08-01"

        data = {
            "type": "ActualCost",
            "timeframe": timeframe,
            "dataset": {
                "granularity": "Daily",
                "aggregation": {
                    "totalCost": {
                        "name": "Cost",
                        "function": "Sum"
                    }
                },
                "grouping": [
                    {"type": "Dimension", "name": "ServiceName"}
                ]
            }
        }

        result = await self._make_request("POST", url, data=data)
        return result

    # ========== Security ==========

    async def list_security_alerts(
        self,
        subscription_id: str = None,
    ) -> List[Dict[str, Any]]:
        """List security alerts from Microsoft Defender."""
        sub_id = subscription_id or self.config.subscription_id
        url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/providers/Microsoft.Security/alerts?api-version=2022-01-01"

        result = await self._make_request("GET", url)
        return result.get("value", [])

    async def get_secure_score(
        self,
        subscription_id: str = None,
    ) -> Dict[str, Any]:
        """Get Azure Secure Score."""
        sub_id = subscription_id or self.config.subscription_id
        url = f"{self.config.MANAGEMENT_URL}/subscriptions/{sub_id}/providers/Microsoft.Security/secureScores/ascScore?api-version=2020-01-01"

        result = await self._make_request("GET", url)
        return result

    @staticmethod
    def get_setup_guide() -> str:
        """Get Azure setup instructions."""
        return """
# Azure / Entra ID Integration Setup Guide

## Prerequisites
- Azure subscription
- Global Administrator or appropriate admin role in Entra ID
- Azure CLI installed (optional, for CLI auth)

## Step 1: Create App Registration in Entra ID

1. Go to https://portal.azure.com
2. Navigate to "Azure Active Directory" (Entra ID) > "App registrations"
3. Click "New registration"
4. Fill in details:
   - Name: "cosmos Azure Manager"
   - Supported account types: Single tenant
   - Redirect URI: Leave blank (for service principal)
5. Click "Register"
6. Note down:
   - Application (client) ID
   - Directory (tenant) ID

## Step 2: Create Client Secret

1. In your app registration, go to "Certificates & secrets"
2. Click "New client secret"
3. Add description and expiration
4. Copy the secret value immediately

## Step 3: Configure API Permissions

### For Resource Management:
1. Go to "API permissions"
2. Add permission > "Azure Service Management"
   - user_impersonation (Delegated) or
   - For service principal: No additional permissions needed

### For Entra ID Management:
1. Add permission > "Microsoft Graph"
2. Add these Application permissions:
   - User.Read.All
   - User.ReadWrite.All (for user management)
   - Group.Read.All
   - Group.ReadWrite.All (for group management)
   - Directory.Read.All
3. Click "Grant admin consent"

## Step 4: Assign Azure RBAC Roles

1. Go to your subscription in Azure Portal
2. Navigate to "Access control (IAM)"
3. Click "Add" > "Add role assignment"
4. Assign roles to your app:
   - Reader (minimum for viewing)
   - Contributor (for resource management)
   - Virtual Machine Contributor (for VM operations)
   - Storage Account Contributor (for storage)

## Step 5: Configure cosmos

Add to your environment:

```bash
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_SUBSCRIPTION_ID=your-subscription-id
```

## Step 6: Test Connection

```python
from Cosmos.integration.cloud.azure_manager import AzureManager, AzureConfig

config = AzureConfig(
    tenant_id="your-tenant-id",
    client_id="your-client-id",
    client_secret="your-client-secret",
    subscription_id="your-subscription-id",
)

manager = AzureManager(config)
await manager.authenticate()

# List resources
resources = await manager.list_resources()

# List users
users = await manager.list_users()
```

## Alternative: Azure CLI Authentication

If you have Azure CLI installed and logged in:

```python
config = AzureConfig(auth_method=AzureAuthMethod.CLI)
manager = AzureManager(config)
await manager.authenticate()  # Uses 'az' CLI credentials
```

## Security Best Practices

1. Use least-privilege principle for permissions
2. Set short expiration for client secrets
3. Use Managed Identity when running in Azure
4. Enable audit logging for the app
5. Use Conditional Access policies
6. Rotate secrets regularly
"""


# Global instance (requires configuration)
azure_manager = AzureManager()
