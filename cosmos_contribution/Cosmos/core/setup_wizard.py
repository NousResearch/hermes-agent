"""
cosmos Setup Wizard - Comprehensive Configuration.

"I've built a machine that lets you choose your own fate!"

This module provides a step-by-step interactive setup for all cosmos systems.
It saves configuration to a local .env file and provides detailed setup guides.
"""

import os
import json
import argparse
import asyncio
import sys
from pathlib import Path
from typing import   Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv
from loguru import logger

try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False


class UseCase(Enum):
    """Primary use case profiles."""
    PERSONAL = "personal"           # Home automation, personal productivity
    DEVELOPER = "developer"         # Software development, CI/CD
    SECURITY = "security"           # Security research, pentesting
    ENTERPRISE = "enterprise"       # Business operations, cloud management
    SYSADMIN = "sysadmin"          # Server management, monitoring
    HEALTH = "health"              # Health tracking, wellness
    TRADING = "trading"            # Financial intelligence, trading
    FULL = "full"                  # Everything enabled


@dataclass
class IntegrationGuide:
    """Setup guide for an integration."""
    name: str
    description: str
    required_permissions: list[str]
    setup_steps: list[str]
    env_vars: dict[str, str]
    documentation_url: str
    estimated_time: str = "5-10 minutes"


class SetupWizard:
    """Comprehensive setup wizard for all cosmos integrations."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.env_file = project_root / ".env"
        self.config: dict[str, str] = {}
        self.selected_use_cases: list[UseCase] = []
        self.guides_shown: list[str] = []

        # Load existing config if present
        if self.env_file.exists():
            load_dotenv(self.env_file)
            self._load_existing()

    def _load_existing(self):
        """Load existing configuration from .env file."""
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, _, value = line.partition('=')
                        self.config[key.strip()] = value.strip()

    def _ask(self, question: str, default: str = "", secret: bool = False) -> str:
        """Standard input with optional secret masking."""
        existing = self.config.get(question.replace(' ', '_').upper(), '')
        if existing and not secret:
            default = existing

        if HAS_QUESTIONARY and secret:
            return questionary.password(f"{question} [{default}]: ").ask() or default

        prompt = f"  {question}"
        if default and not secret:
            prompt += f" [{default}]"
        prompt += ": "

        val = input(prompt).strip()
        return val if val else default

    def _ask_bool(self, question: str, default: bool = True) -> bool:
        """Ask a yes/no question."""
        default_str = "Y/n" if default else "y/N"
        val = input(f"  {question} ({default_str}): ").strip().lower()
        if not val:
            return default
        return val.startswith('y')

    def _ask_select(self, question: str, options: list[str], default: str = None) -> str:
        """Ask user to select from options."""
        if HAS_QUESTIONARY:
            return questionary.select(question, choices=options, default=default).ask()

        print(f"\n  {question}")
        for i, opt in enumerate(options, 1):
            marker = " *" if opt == default else ""
            print(f"    {i}. {opt}{marker}")

        while True:
            val = input(f"  Enter choice (1-{len(options)}) [{options.index(default)+1 if default else 1}]: ").strip()
            if not val and default:
                return default
            try:
                idx = int(val) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                if val in options:
                    return val
            print("  Invalid choice, please try again.")

    def _ask_multi(self, question: str, options: list[str], defaults: list[str] = None) -> list[str]:
        """Ask user to select multiple options."""
        if HAS_QUESTIONARY:
            return questionary.checkbox(question, choices=options, default=defaults).ask()

        print(f"\n  {question} (comma-separated numbers)")
        for i, opt in enumerate(options, 1):
            marker = " *" if defaults and opt in defaults else ""
            print(f"    {i}. {opt}{marker}")

        val = input("  Enter choices: ").strip()
        if not val and defaults:
            return defaults

        selected = []
        for part in val.split(','):
            try:
                idx = int(part.strip()) - 1
                if 0 <= idx < len(options):
                    selected.append(options[idx])
            except ValueError:
                pass
        return selected or defaults or []

    def _print_header(self, title: str, emoji: str = ""):
        """Print a section header."""
        print(f"\n{'='*60}")
        print(f"{emoji}  {title}")
        print('='*60)

    def _print_subheader(self, title: str):
        """Print a subsection header."""
        print(f"\n  --- {title} ---")

    def _print_guide(self, guide: IntegrationGuide):
        """Print detailed setup guide for an integration."""
        if guide.name in self.guides_shown:
            return

        self.guides_shown.append(guide.name)

        print(f"\n  {'='*50}")
        print(f"  SETUP GUIDE: {guide.name}")
        print(f"  {'='*50}")
        print(f"  {guide.description}")
        print(f"  Estimated time: {guide.estimated_time}")
        print(f"\n  Required Permissions:")
        for perm in guide.required_permissions:
            print(f"    - {perm}")
        print(f"\n  Setup Steps:")
        for i, step in enumerate(guide.setup_steps, 1):
            print(f"    {i}. {step}")
        print(f"\n  Documentation: {guide.documentation_url}")
        print(f"  {'-'*50}\n")

    # =========================================================================
    # INTEGRATION GUIDES
    # =========================================================================

    def _get_azure_guide(self) -> IntegrationGuide:
        return IntegrationGuide(
            name="Microsoft Azure / Entra ID",
            description="Full Azure cloud and Entra ID (Azure AD) management including VMs, storage, users, groups, and security.",
            required_permissions=[
                "Azure Subscription with Owner or Contributor role",
                "Entra ID: User Administrator (for user management)",
                "Entra ID: Global Reader (minimum for read-only)",
                "Microsoft Graph API permissions (for advanced features)",
            ],
            setup_steps=[
                "Go to Azure Portal: https://portal.azure.com",
                "Navigate to 'App registrations' in Entra ID",
                "Click 'New registration' and name it 'cosmos'",
                "Set redirect URI to 'http://localhost:8080/callback' (Web)",
                "Copy the 'Application (client) ID' - this is your AZURE_CLIENT_ID",
                "Copy the 'Directory (tenant) ID' - this is your AZURE_TENANT_ID",
                "Go to 'Certificates & secrets' > 'New client secret'",
                "Copy the secret value immediately - this is your AZURE_CLIENT_SECRET",
                "Go to 'API permissions' and add required Microsoft Graph permissions",
                "Grant admin consent for the permissions",
                "For Subscription access: Go to Subscriptions > Your Sub > IAM > Add role assignment",
                "Assign 'Contributor' role to the cosmos app",
            ],
            env_vars={
                "AZURE_TENANT_ID": "Your Entra ID Tenant ID",
                "AZURE_CLIENT_ID": "Your App Registration Client ID",
                "AZURE_CLIENT_SECRET": "Your App Client Secret",
                "AZURE_SUBSCRIPTION_ID": "Your Azure Subscription ID",
            },
            documentation_url="https://learn.microsoft.com/en-us/azure/active-directory/develop/quickstart-register-app",
            estimated_time="15-20 minutes"
        )

    def _get_aws_guide(self) -> IntegrationGuide:
        return IntegrationGuide(
            name="Amazon Web Services (AWS)",
            description="Full AWS management including EC2, S3, IAM, VPC, and CloudWatch.",
            required_permissions=[
                "IAM User with programmatic access",
                "EC2: AmazonEC2FullAccess (or specific actions)",
                "S3: AmazonS3FullAccess (or specific buckets)",
                "IAM: IAMFullAccess (for user management)",
                "CloudWatch: CloudWatchReadOnlyAccess",
                "Cost Explorer: ce:GetCostAndUsage (for cost tracking)",
            ],
            setup_steps=[
                "Go to AWS Console: https://console.aws.amazon.com",
                "Navigate to IAM > Users > Add users",
                "Create user 'cosmos-integration' with programmatic access",
                "Attach policies directly or create a custom policy",
                "For full access: Attach 'AdministratorAccess' (not recommended for production)",
                "For limited access: Create custom policy with specific permissions",
                "Download or copy the Access Key ID and Secret Access Key",
                "Optionally configure MFA for additional security",
                "Set default region (e.g., us-east-1, eu-west-1)",
            ],
            env_vars={
                "AWS_ACCESS_KEY_ID": "Your IAM Access Key ID",
                "AWS_SECRET_ACCESS_KEY": "Your IAM Secret Access Key",
                "AWS_DEFAULT_REGION": "Default AWS region (e.g., us-east-1)",
            },
            documentation_url="https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html",
            estimated_time="10-15 minutes"
        )

    def _get_gcp_guide(self) -> IntegrationGuide:
        return IntegrationGuide(
            name="Google Cloud Platform (GCP)",
            description="Google Cloud management including Compute Engine, Cloud Storage, and IAM.",
            required_permissions=[
                "Service Account with appropriate roles",
                "Compute Engine: roles/compute.admin",
                "Cloud Storage: roles/storage.admin",
                "IAM: roles/iam.serviceAccountAdmin",
            ],
            setup_steps=[
                "Go to Google Cloud Console: https://console.cloud.google.com",
                "Select or create a project",
                "Navigate to IAM & Admin > Service Accounts",
                "Click 'Create Service Account'",
                "Name it 'cosmos-integration'",
                "Grant required roles (Editor for full access)",
                "Click 'Create Key' > JSON",
                "Save the JSON key file securely",
                "Set GOOGLE_APPLICATION_CREDENTIALS to the file path",
            ],
            env_vars={
                "GOOGLE_APPLICATION_CREDENTIALS": "Path to service account JSON key",
                "GCP_PROJECT_ID": "Your Google Cloud Project ID",
            },
            documentation_url="https://cloud.google.com/iam/docs/creating-managing-service-accounts",
            estimated_time="10-15 minutes"
        )

    def _get_office365_guide(self) -> IntegrationGuide:
        return IntegrationGuide(
            name="Microsoft 365 / Office 365",
            description="Full email, calendar, and collaboration integration via Microsoft Graph API.",
            required_permissions=[
                "Azure AD App Registration",
                "Mail.Read, Mail.Send, Mail.ReadWrite (for email)",
                "Calendars.ReadWrite (for calendar)",
                "User.Read (for profile)",
                "MailboxSettings.ReadWrite (for rules)",
                "Admin consent may be required for app-only access",
            ],
            setup_steps=[
                "Go to Azure Portal > Entra ID > App registrations",
                "Create new registration 'cosmos-O365'",
                "Add redirect URI: http://localhost:8080/o365/callback",
                "Go to API permissions > Add permission > Microsoft Graph",
                "Add Delegated permissions: Mail.ReadWrite, Calendars.ReadWrite, etc.",
                "For app-only: Add Application permissions and grant admin consent",
                "Go to Certificates & secrets > New client secret",
                "Copy the client ID, tenant ID, and secret",
                "For MX record management: Add Exchange Online permissions",
            ],
            env_vars={
                "O365_CLIENT_ID": "Azure AD App Client ID",
                "O365_CLIENT_SECRET": "Azure AD App Client Secret",
                "O365_TENANT_ID": "Azure AD Tenant ID",
            },
            documentation_url="https://learn.microsoft.com/en-us/graph/auth-register-app-v2",
            estimated_time="15-20 minutes"
        )

    def _get_google_workspace_guide(self) -> IntegrationGuide:
        return IntegrationGuide(
            name="Google Workspace / Gmail",
            description="Gmail, Google Calendar, Drive, and Admin SDK integration.",
            required_permissions=[
                "Google Cloud Project with Gmail API enabled",
                "OAuth 2.0 credentials or Service Account",
                "Gmail API scopes for email access",
                "Admin SDK for domain-wide management (optional)",
            ],
            setup_steps=[
                "Go to Google Cloud Console > APIs & Services",
                "Enable Gmail API, Calendar API, and Admin SDK API",
                "Go to Credentials > Create Credentials > OAuth 2.0 Client ID",
                "Select 'Desktop app' or 'Web application'",
                "Download the credentials JSON file",
                "For domain-wide delegation: Create a Service Account",
                "In Google Admin Console: Security > API controls > Domain-wide delegation",
                "Add the service account client ID with required scopes",
                "Scopes needed: https://www.googleapis.com/auth/gmail.modify",
            ],
            env_vars={
                "GOOGLE_OAUTH_CREDENTIALS": "Path to OAuth credentials JSON",
                "GOOGLE_SERVICE_ACCOUNT": "Path to service account JSON (optional)",
                "GOOGLE_DELEGATED_USER": "User email for domain-wide delegation",
            },
            documentation_url="https://developers.google.com/gmail/api/quickstart/python",
            estimated_time="15-20 minutes"
        )

    def _get_health_providers_guide(self) -> IntegrationGuide:
        return IntegrationGuide(
            name="Health Tracking Providers",
            description="Integration with Fitbit, Oura Ring, WHOOP, and Apple Health for comprehensive health tracking.",
            required_permissions=[
                "Fitbit: Developer account with OAuth application",
                "Oura: Personal access token from cloud.ouraring.com",
                "WHOOP: API access (requires paid plan)",
                "Apple Health: Export from iOS Health app",
            ],
            setup_steps=[
                "FITBIT:",
                "  - Go to dev.fitbit.com and create an application",
                "  - Set OAuth 2.0 Application Type to 'Personal'",
                "  - Set Callback URL to http://localhost:8080/fitbit/callback",
                "  - Copy Client ID and Client Secret",
                "",
                "OURA:",
                "  - Go to cloud.ouraring.com/personal-access-tokens",
                "  - Create a new personal access token",
                "  - Copy the token (shown only once)",
                "",
                "WHOOP:",
                "  - Contact WHOOP developer support for API access",
                "  - Requires WHOOP membership",
                "",
                "APPLE HEALTH:",
                "  - Export health data from iOS Health app",
                "  - Place export.xml in the configured directory",
            ],
            env_vars={
                "FITBIT_CLIENT_ID": "Fitbit OAuth Client ID",
                "FITBIT_CLIENT_SECRET": "Fitbit OAuth Client Secret",
                "OURA_ACCESS_TOKEN": "Oura Ring Personal Access Token",
                "WHOOP_CLIENT_ID": "WHOOP API Client ID",
                "WHOOP_CLIENT_SECRET": "WHOOP API Client Secret",
            },
            documentation_url="https://dev.fitbit.com/build/reference/web-api/developer-guide/getting-started/",
            estimated_time="20-30 minutes"
        )

    # =========================================================================
    # MAIN WIZARD FLOW
    # =========================================================================

    async def run(self):
        """Run the complete setup wizard."""
        print("\n" + "="*60)
        print("  cosmos COMPREHENSIVE SETUP WIZARD")
        print("  'Good news, everyone! Let's configure your fate!'")
        print("="*60)

        # Step 1: Determine use cases
        await self._setup_use_cases()

        # Step 2: Privacy & Networking
        await self._setup_networking()

        # Step 3: Hardware Profile
        await self._setup_hardware()

        # Step 4: AI Providers
        await self._setup_ai_providers()

        # Step 5: Cloud Integrations (based on use case)
        if any(uc in [UseCase.ENTERPRISE, UseCase.SYSADMIN, UseCase.FULL]):
            await self._setup_cloud_providers()

        # Step 6: Email Integrations
        if any(uc in [UseCase.ENTERPRISE, UseCase.PERSONAL, UseCase.FULL]):
            await self._setup_email_integrations()

        # Step 7: Security Tools
        if any(uc in [UseCase.SECURITY, UseCase.SYSADMIN, UseCase.FULL]):
            await self._setup_security_tools()

        # Step 8: Health Tracking
        if any(uc in [UseCase.HEALTH, UseCase.PERSONAL, UseCase.FULL]):
            await self._setup_health_tracking()

        # Step 9: Cognitive Engines
        await self._setup_cognitive_engines()

        # Step 10: External Integrations
        await self._setup_external_integrations()

        # Step 11: Financial Intelligence
        if any(uc in [UseCase.TRADING, UseCase.FULL]):
            await self._setup_financial()

        # Step 12: Sysadmin Tools
        if any(uc in [UseCase.SYSADMIN, UseCase.ENTERPRISE, UseCase.FULL]):
            await self._setup_sysadmin_tools()

        # Save and finish
        self._save()
        self._print_summary()

    async def _setup_use_cases(self):
        """Determine primary use cases to optimize configuration."""
        self._print_header("USE CASE SELECTION", "🎯")
        print("  Select your primary use cases to optimize the setup.\n")

        use_case_options = [
            "Personal - Home automation, personal productivity",
            "Developer - Software development, CI/CD, coding assistance",
            "Security - Security research, penetration testing, defense",
            "Enterprise - Business operations, cloud management",
            "Sysadmin - Server management, monitoring, log analysis",
            "Health - Health tracking, wellness, nutrition",
            "Trading - Financial intelligence, trading, crypto",
            "Full - Enable everything (advanced users)",
        ]

        selected = self._ask_multi(
            "Select use cases (you can choose multiple):",
            use_case_options,
            defaults=["Personal - Home automation, personal productivity"]
        )

        # Map selections to UseCase enum
        for sel in selected:
            if "Personal" in sel:
                self.selected_use_cases.append(UseCase.PERSONAL)
            elif "Developer" in sel:
                self.selected_use_cases.append(UseCase.DEVELOPER)
            elif "Security" in sel:
                self.selected_use_cases.append(UseCase.SECURITY)
            elif "Enterprise" in sel:
                self.selected_use_cases.append(UseCase.ENTERPRISE)
            elif "Sysadmin" in sel:
                self.selected_use_cases.append(UseCase.SYSADMIN)
            elif "Health" in sel:
                self.selected_use_cases.append(UseCase.HEALTH)
            elif "Trading" in sel:
                self.selected_use_cases.append(UseCase.TRADING)
            elif "Full" in sel:
                self.selected_use_cases = list(UseCase)
                break

        self.config["cosmos_USE_CASES"] = ",".join(uc.value for uc in self.selected_use_cases)
        print(f"\n  Selected: {', '.join(uc.value for uc in self.selected_use_cases)}")

    async def _setup_networking(self):
        """Configure privacy and networking settings."""
        self._print_header("PRIVACY & NETWORKING", "🛡️")

        is_isolated = self._ask_bool("Enable ISOLATED MODE? (Disables P2P swarm)", default=False)
        self.config["cosmos_ISOLATED"] = "true" if is_isolated else "false"

        if not is_isolated:
            self.config["cosmos_P2P_PORT"] = self._ask("P2P Swarm Port", "9999")
            self.config["cosmos_DISCOVERY_PORT"] = self._ask("UDP Discovery Port", "8888")

    async def _setup_hardware(self):
        """Configure hardware profile."""
        self._print_header("HARDWARE PROFILE", "💻")

        profiles = ["minimal", "cpu_only", "low_vram", "medium_vram", "high_vram", "max"]
        profile = self._ask_select(
            "Select hardware profile:",
            profiles,
            default="medium_vram"
        )
        self.config["cosmos_HARDWARE_PROFILE"] = profile

        if profile == "max":
            print("\n  MAX profile detected - enabling all GPU acceleration")
            self.config["ENABLE_CUDA"] = "true"
            self.config["ENABLE_TENSOR_RT"] = "true"
            self.config["MAX_GPU_MEMORY"] = self._ask("Max GPU memory (GB)", "24")

    async def _setup_ai_providers(self):
        """Configure AI model providers."""
        self._print_header("AI MODEL PROVIDERS", "🤖")

        # Anthropic
        if self._ask_bool("Configure Anthropic (Claude)?", default=True):
            self.config["ANTHROPIC_API_KEY"] = self._ask("Anthropic API Key", secret=True)

        # OpenAI
        if self._ask_bool("Configure OpenAI (GPT)?", default=False):
            self.config["OPENAI_API_KEY"] = self._ask("OpenAI API Key", secret=True)
            self.config["OPENAI_ORG_ID"] = self._ask("OpenAI Organization ID (optional)")

        # Google Gemini
        if self._ask_bool("Configure Google Gemini (2.0 Flash, 1.5 Pro)?", default=False):
            print("\n  Get your API key from: https://makersuite.google.com/app/apikey")
            self.config["GEMINI_API_KEY"] = self._ask("Gemini API Key", secret=True)
            print("  Available models: gemini-2.0-flash, gemini-2.0-flash-lite, gemini-1.5-pro, gemini-1.5-flash")

        # DeepInfra (for local models)
        if self._ask_bool("Configure DeepInfra (DeepSeek, Llama, etc.)?", default=False):
            self.config["DEEPINFRA_API_KEY"] = self._ask("DeepInfra API Key", secret=True)

        # xAI Grok
        if self._ask_bool("Configure xAI (Grok)?", default=False):
            self.config["XAI_API_KEY"] = self._ask("xAI API Key", secret=True)

        # Local Ollama
        if self._ask_bool("Configure Ollama (local models)?", default=False):
            self.config["OLLAMA_HOST"] = self._ask("Ollama Host", "http://localhost:11434")

        # Parallel AI
        self.config["ENABLE_PARALLEL_AI"] = "true" if self._ask_bool(
            "Enable Parallel AI (Multi-model consensus)?", default=True
        ) else "false"

    async def _setup_cloud_providers(self):
        """Configure cloud provider integrations."""
        self._print_header("CLOUD PROVIDERS", "☁️")

        # Azure
        if self._ask_bool("Configure Microsoft Azure?", default=False):
            self._print_guide(self._get_azure_guide())

            self.config["AZURE_TENANT_ID"] = self._ask("Azure Tenant ID")
            self.config["AZURE_CLIENT_ID"] = self._ask("Azure Client ID")
            self.config["AZURE_CLIENT_SECRET"] = self._ask("Azure Client Secret", secret=True)
            self.config["AZURE_SUBSCRIPTION_ID"] = self._ask("Azure Subscription ID")

            if self._ask_bool("Enable Entra ID (Azure AD) management?", default=True):
                self.config["ENABLE_ENTRA_ID"] = "true"

        # AWS
        if self._ask_bool("Configure Amazon Web Services (AWS)?", default=False):
            self._print_guide(self._get_aws_guide())

            self.config["AWS_ACCESS_KEY_ID"] = self._ask("AWS Access Key ID")
            self.config["AWS_SECRET_ACCESS_KEY"] = self._ask("AWS Secret Access Key", secret=True)
            self.config["AWS_DEFAULT_REGION"] = self._ask("AWS Default Region", "us-east-1")

            if self._ask_bool("Enable Cost Explorer?", default=True):
                self.config["ENABLE_AWS_COST_EXPLORER"] = "true"

        # GCP
        if self._ask_bool("Configure Google Cloud Platform (GCP)?", default=False):
            self._print_guide(self._get_gcp_guide())

            self.config["GCP_PROJECT_ID"] = self._ask("GCP Project ID")
            self.config["GOOGLE_APPLICATION_CREDENTIALS"] = self._ask(
                "Path to service account JSON",
                "./gcp-credentials.json"
            )

    async def _setup_email_integrations(self):
        """Configure email provider integrations."""
        self._print_header("EMAIL INTEGRATIONS", "📧")

        # Office 365
        if self._ask_bool("Configure Microsoft 365 / Office 365?", default=False):
            self._print_guide(self._get_office365_guide())

            self.config["O365_CLIENT_ID"] = self._ask("O365 Client ID")
            self.config["O365_CLIENT_SECRET"] = self._ask("O365 Client Secret", secret=True)
            self.config["O365_TENANT_ID"] = self._ask("O365 Tenant ID")

            if self._ask_bool("Enable mailbox filtering and rules?", default=True):
                self.config["ENABLE_O365_MAIL_RULES"] = "true"

            if self._ask_bool("Enable header analysis for threat detection?", default=True):
                self.config["ENABLE_EMAIL_HEADER_ANALYSIS"] = "true"

        # Google Workspace
        if self._ask_bool("Configure Google Workspace / Gmail?", default=False):
            self._print_guide(self._get_google_workspace_guide())

            self.config["GOOGLE_OAUTH_CREDENTIALS"] = self._ask(
                "Path to OAuth credentials JSON",
                "./google-oauth.json"
            )

            if self._ask_bool("Use service account for domain-wide delegation?", default=False):
                self.config["GOOGLE_SERVICE_ACCOUNT"] = self._ask(
                    "Path to service account JSON",
                    "./google-service-account.json"
                )
                self.config["GOOGLE_DELEGATED_USER"] = self._ask("Delegated user email")

    async def _setup_security_tools(self):
        """Configure security research tools."""
        self._print_header("SECURITY TOOLS", "🔒")

        print("  Configuring defensive security tools for research.\n")

        # Vulnerability Scanner
        if self._ask_bool("Enable Vulnerability Scanner?", default=True):
            self.config["ENABLE_VULN_SCANNER"] = "true"
            if self._ask_bool("  Enable aggressive port scanning?", default=False):
                self.config["VULN_SCAN_AGGRESSIVE"] = "true"

        # Header Analyzer
        if self._ask_bool("Enable Email Header Analyzer?", default=True):
            self.config["ENABLE_HEADER_ANALYZER"] = "true"

        # Threat Intelligence
        if self._ask_bool("Enable Threat Intelligence feeds?", default=True):
            self.config["ENABLE_THREAT_INTEL"] = "true"
            if self._ask_bool("  Configure VirusTotal API?", default=False):
                self.config["VIRUSTOTAL_API_KEY"] = self._ask("VirusTotal API Key", secret=True)
            if self._ask_bool("  Configure AbuseIPDB?", default=False):
                self.config["ABUSEIPDB_API_KEY"] = self._ask("AbuseIPDB API Key", secret=True)

        # EDR
        if self._ask_bool("Enable EDR (Endpoint Detection & Response)?", default=True):
            self.config["ENABLE_EDR"] = "true"
            self.config["EDR_QUARANTINE_PATH"] = self._ask(
                "Quarantine directory",
                "./quarantine"
            )

        # Forensics
        if self._ask_bool("Enable Digital Forensics tools?", default=True):
            self.config["ENABLE_FORENSICS"] = "true"

        # Log Parser
        if self._ask_bool("Enable Security Log Parser?", default=True):
            self.config["ENABLE_LOG_PARSER"] = "true"

        # Recon (with warning)
        print("\n  ⚠️  Reconnaissance tools are for authorized testing only!")
        if self._ask_bool("Enable Recon tools?", default=False):
            self.config["ENABLE_RECON"] = "true"

    async def _setup_health_tracking(self):
        """Configure health tracking integrations."""
        self._print_header("HEALTH TRACKING", "🏥")

        if self._ask_bool("Enable Health Dashboard?", default=True):
            self.config["cosmos_HEALTH_ENABLED"] = "true"
            self.config["cosmos_HEALTH_PORT"] = self._ask("Health Dashboard Port", "8081")

        if self._ask_bool("Configure health providers?", default=False):
            self._print_guide(self._get_health_providers_guide())

            if self._ask_bool("  Configure Fitbit?", default=False):
                self.config["FITBIT_CLIENT_ID"] = self._ask("Fitbit Client ID")
                self.config["FITBIT_CLIENT_SECRET"] = self._ask("Fitbit Client Secret", secret=True)

            if self._ask_bool("  Configure Oura Ring?", default=False):
                self.config["OURA_ACCESS_TOKEN"] = self._ask("Oura Access Token", secret=True)

            if self._ask_bool("  Configure WHOOP?", default=False):
                self.config["WHOOP_CLIENT_ID"] = self._ask("WHOOP Client ID")
                self.config["WHOOP_CLIENT_SECRET"] = self._ask("WHOOP Client Secret", secret=True)

        if self._ask_bool("Enable DeepSeek OCR for document parsing?", default=True):
            self.config["ENABLE_HEALTH_OCR"] = "true"

    async def _setup_cognitive_engines(self):
        """Configure cognitive and experimental features."""
        self._print_header("COGNITIVE ENGINES", "🧠")

        self.config["ENABLE_TOM"] = "true" if self._ask_bool(
            "Enable Theory of Mind (User simulation)?", default=True
        ) else "false"

        self.config["ENABLE_CAUSAL"] = "true" if self._ask_bool(
            "Enable Causal Reasoning?", default=True
        ) else "false"

        self.config["ENABLE_VIDEO_FLOW"] = "true" if self._ask_bool(
            "Enable Advanced Video Flow Analysis?", default=False
        ) else "false"

        self._print_subheader("Experimental Features")

        if self._ask_bool("Enable Quantum-Inspired Search?", default=False):
            self.config["ENABLE_QUANTUM_SEARCH"] = "true"

        if self._ask_bool("Enable Planetary Memory (Shared learning)?", default=False):
            self.config["ENABLE_PLANETARY_MEMORY"] = "true"
            self.config["PLANETARY_USE_P2P"] = "true" if self._ask_bool(
                "  Allow P2P Skill Sharing?", default=False
            ) else "false"

        if self._ask_bool("Enable Dream Catcher (Offline learning)?", default=True):
            self.config["ENABLE_DREAM_CATCHER"] = "true"

        if self._ask_bool("Enable Affective Computing (Emotion-aware)?", default=False):
            self.config["ENABLE_AFFECTIVE_COMPUTING"] = "true"

        if self._ask_bool("Enable Bio Interface Support?", default=False):
            self.config["ENABLE_BIO_INTERFACE"] = "true"
            if self._ask_bool("  Use Mock Provider for testing?", default=True):
                self.config["BIO_INTERFACE_PROVIDER"] = "mock"

    async def _setup_external_integrations(self):
        """Configure external service integrations."""
        self._print_header("EXTERNAL INTEGRATIONS", "🔗")

        # GitHub
        if self._ask_bool("Configure GitHub integration?", default=False):
            self.config["GITHUB_TOKEN"] = self._ask("GitHub Personal Access Token", secret=True)
            self.config["GITHUB_REPO"] = self._ask("Default Repository (user/repo)")

        # X (Twitter)
        if self._ask_bool("Configure X (Twitter) integration?", default=False):
            self.config["X_API_KEY"] = self._ask("X API Key", secret=True)
            self.config["X_API_SECRET"] = self._ask("X API Secret", secret=True)
            self.config["X_ACCESS_TOKEN"] = self._ask("X Access Token", secret=True)
            self.config["X_ACCESS_SECRET"] = self._ask("X Access Secret", secret=True)

        # Discord
        if self._ask_bool("Configure Discord integration?", default=False):
            self.config["DISCORD_TOKEN"] = self._ask("Discord Bot Token", secret=True)
            self.config["DISCORD_WEBHOOK_URL"] = self._ask("Discord Webhook URL (optional)")

        # n8n Workflows
        if self._ask_bool("Configure n8n workflow integration?", default=False):
            self.config["N8N_WEBHOOK_URL"] = self._ask("n8n Webhook URL")
            self.config["N8N_API_KEY"] = self._ask("n8n API Key (optional)", secret=True)

        # Database
        if self._ask_bool("Configure SQL Database?", default=False):
            db_type = self._ask_select("Database Type", ["sqlite", "postgres", "mysql"], "sqlite")
            self.config["DB_TYPE"] = db_type
            if db_type == "sqlite":
                self.config["DB_CONNECTION_STRING"] = self._ask("SQLite file path", "cosmos_data.db")
            else:
                self.config["DB_CONNECTION_STRING"] = self._ask("Connection string")

        # Remotion
        if self._ask_bool("Enable Remotion Video Generation?", default=False):
            self.config["REMOTION_WORKSPACE"] = self._ask("Remotion Workspace Path", "./remotion_workspace")

        # Web Scraping
        if self._ask_bool("Enable Universal Scraper (Crawlee)?", default=False):
            self.config["ENABLE_CRAWLEE"] = "true"

    async def _setup_financial(self):
        """Configure financial intelligence tools."""
        self._print_header("FINANCIAL INTELLIGENCE", "📈")

        if self._ask_bool("Enable DexScreener (On-chain tracking)?", default=False):
            self.config["ENABLE_DEXSCREENER"] = "true"

        if self._ask_bool("Enable Polymarket (Prediction markets)?", default=False):
            self.config["ENABLE_POLYMARKET"] = "true"

        if self._ask_bool("Enable Bags.fm (Social trading)?", default=False):
            self.config["BAGS_API_KEY"] = self._ask("Bags.fm API Key", secret=True)

        if self._ask_bool("Enable TradFi Agent (Stocks & Forex)?", default=False):
            self.config["ENABLE_TRADFI"] = "true"
            self.config["ALPHAVANTAGE_API_KEY"] = self._ask("Alpha Vantage API Key (optional)")

        # Solana Trading (Danger Zone)
        self._print_subheader("SOLANA TRADING (DANGER ZONE)")
        print("  WARNING: Private keys in .env files are RISKY!")
        print("  RECOMMENDATION: Use a BURNER wallet with small amounts only.\n")

        if self._ask_bool("Enable Solana Trading?", default=False):
            self.config["SOLANA_PRIVATE_KEY"] = self._ask("Solana Private Key (Base58)", secret=True)
            self.config["SOLANA_RPC_URL"] = self._ask(
                "Solana RPC URL",
                "https://api.mainnet-beta.solana.com"
            )

            if self._ask_bool("Enable advanced trading (sniping, whale tracking)?", default=False):
                self.config["HELIUS_API_KEY"] = self._ask("Helius API Key", secret=True)
                self.config["ENABLE_SNIPER"] = "true"
                self.config["ENABLE_WHALE_WATCH"] = "true"

    async def _setup_sysadmin_tools(self):
        """Configure system administration tools."""
        self._print_header("SYSADMIN TOOLS", "🔧")

        if self._ask_bool("Enable System Monitor?", default=True):
            self.config["ENABLE_SYSTEM_MONITOR"] = "true"
            self.config["MONITOR_INTERVAL_SECONDS"] = self._ask("Monitor interval (seconds)", "30")

        if self._ask_bool("Enable Service Manager?", default=True):
            self.config["ENABLE_SERVICE_MANAGER"] = "true"

        if self._ask_bool("Enable Log Analyzer?", default=True):
            self.config["ENABLE_LOG_ANALYZER"] = "true"

        if self._ask_bool("Enable Network Tools?", default=True):
            self.config["ENABLE_NETWORK_TOOLS"] = "true"

        if self._ask_bool("Enable Backup Manager?", default=True):
            self.config["ENABLE_BACKUP_MANAGER"] = "true"
            self.config["BACKUP_DESTINATION"] = self._ask("Backup destination", "./backups")

        if self._ask_bool("Enable WSL Bridge (Windows only)?", default=True):
            self.config["ENABLE_WSL_BRIDGE"] = "true"

    def _save(self):
        """Save configuration to .env file."""
        with open(self.env_file, "w") as f:
            f.write("# cosmos Environment Configuration\n")
            f.write(f"# Generated by Setup Wizard\n")
            f.write(f"# Use cases: {self.config.get('cosmos_USE_CASES', 'personal')}\n\n")

            # Group by category for readability
            categories = {
                "Core": ["cosmos_", "ENABLE_"],
                "AI Providers": ["ANTHROPIC_", "OPENAI_", "GEMINI_", "DEEPINFRA_", "XAI_", "OLLAMA_"],
                "Cloud - Azure": ["AZURE_", "ENTRA_"],
                "Cloud - AWS": ["AWS_"],
                "Cloud - GCP": ["GCP_", "GOOGLE_APPLICATION"],
                "Email - O365": ["O365_"],
                "Email - Google": ["GOOGLE_OAUTH", "GOOGLE_SERVICE", "GOOGLE_DELEGATED"],
                "Security": ["VULN_", "EDR_", "VIRUSTOTAL_", "ABUSEIPDB_"],
                "Health": ["FITBIT_", "OURA_", "WHOOP_", "HEALTH_"],
                "Financial": ["SOLANA_", "HELIUS_", "BAGS_", "ALPHAVANTAGE_", "DEXSCREENER", "POLYMARKET", "TRADFI"],
                "External": ["GITHUB_", "X_", "DISCORD_", "N8N_", "DB_", "REMOTION_", "CRAWLEE"],
                "Sysadmin": ["MONITOR_", "BACKUP_", "WSL_"],
            }

            written_keys = set()

            for category, prefixes in categories.items():
                category_items = []
                for key, value in self.config.items():
                    if any(prefix in prefixes) and key not in written_keys:
                        category_items.append((key, value))
                        written_keys.add(key)

                if category_items:
                    f.write(f"\n# {category}\n")
                    for key, value in sorted(category_items):
                        if value:
                            # Mask secrets in comments
                            f.write(f"{key}={value}\n")

            # Write dict remaining keys
            remaining = [(k, v) for k, v in self.config.items() if k not in written_keys and v]
            if remaining:
                f.write("\n# Other\n")
                for key, value in sorted(remaining):
                    f.write(f"{key}={value}\n")

    def _print_summary(self):
        """Print setup summary."""
        print("\n" + "="*60)
        print("  SETUP COMPLETE!")
        print("="*60)

        print(f"\n  Configuration saved to: {self.env_file}")
        print(f"\n  Use Cases: {', '.join(uc.value for uc in self.selected_use_cases)}")

        # Count configured integrations
        configured = []
        if self.config.get("AZURE_CLIENT_ID"):
            configured.append("Azure")
        if self.config.get("AWS_ACCESS_KEY_ID"):
            configured.append("AWS")
        if self.config.get("GCP_PROJECT_ID"):
            configured.append("GCP")
        if self.config.get("O365_CLIENT_ID"):
            configured.append("Office 365")
        if self.config.get("GOOGLE_OAUTH_CREDENTIALS"):
            configured.append("Google Workspace")
        if self.config.get("GITHUB_TOKEN"):
            configured.append("GitHub")
        if self.config.get("DISCORD_TOKEN"):
            configured.append("Discord")

        if configured:
            print(f"\n  Configured Integrations: {', '.join(configured)}")

        # Enabled features
        enabled = []
        if self.config.get("ENABLE_EDR") == "true":
            enabled.append("EDR")
        if self.config.get("ENABLE_VULN_SCANNER") == "true":
            enabled.append("Vulnerability Scanner")
        if self.config.get("ENABLE_FORENSICS") == "true":
            enabled.append("Forensics")
        if self.config.get("cosmos_HEALTH_ENABLED") == "true":
            enabled.append("Health Dashboard")
        if self.config.get("ENABLE_PARALLEL_AI") == "true":
            enabled.append("Parallel AI")

        if enabled:
            print(f"\n  Enabled Features: {', '.join(enabled)}")

        print("\n  Next Steps:")
        print("    1. Review the .env file and add dict missing credentials")
        print("    2. Run 'python -m cosmos' to start cosmos")
        print("    3. Access the web dashboard at http://localhost:8080")

        if any(uc in [UseCase.SECURITY, UseCase.SYSADMIN]):
            print("\n  Security Tools:")
            print("    - Vulnerability Scanner: cosmos scan <target>")
            print("    - EDR: cosmos edr --start")
            print("    - Log Parser: cosmos logs <path>")

        if UseCase.HEALTH in self.selected_use_cases:
            print(f"\n  Health Dashboard: http://localhost:{self.config.get('cosmos_HEALTH_PORT', '8081')}")

        print("\n" + "="*60 + "\n")

    @staticmethod
    def get_quick_guide(integration: str) -> str:
        """Get quick setup guide for a specific integration."""
        guides = {
            "azure": """
AZURE / ENTRA ID QUICK SETUP
3. Copy: Application ID (CLIENT_ID), Directory ID (TENANT_ID)
4. Create client secret and copy immediately
5. Go to Subscriptions > Your Sub > IAM > Add Contributor role
6. Add to .env:
   AZURE_TENANT_ID=your_tenant_id
   AZURE_CLIENT_ID=your_client_id
   AZURE_CLIENT_SECRET=your_secret
   AZURE_SUBSCRIPTION_ID=your_sub_id
""",
            "aws": """
AWS QUICK SETUP
3. Attach AdministratorAccess (or specific policies)
4. Download/copy Access Key ID and Secret
5. Add to .env:
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_DEFAULT_REGION=us-east-1
""",
            "office365": """
OFFICE 365 QUICK SETUP
3. Add API permissions: Mail.ReadWrite, Calendars.ReadWrite
4. Grant admin consent
5. Create client secret
6. Add to .env:
   O365_CLIENT_ID=your_client_id
   O365_CLIENT_SECRET=your_secret
   O365_TENANT_ID=your_tenant_id
""",
            "google": """
GOOGLE WORKSPACE QUICK SETUP
1. Go to Google Cloud Console > APIs & Services
2. Enable Gmail API, Calendar API
3. Create OAuth 2.0 Client ID
4. Download credentials.json
5. Add to .env:
   GOOGLE_OAUTH_CREDENTIALS=path/to/credentials.json
"""
        }
        return guides.get(name, "No guide found for that integration.")

def main():
    parser = argparse.ArgumentParser(description="cosmos Setup Wizard")
    parser.add_argument("--guide", type=str, help="Show quick guide for integration (azure, aws, office365, google)")
    parser.add_argument("--root", type=str, default=".", help="Project root directory")

    args = parser.parse_args()

    if args.guide:
        print(SetupWizard.get_quick_guide(args.guide))
    else:
        wizard = SetupWizard(Path(args.root))
        asyncio.run(wizard.run())

if __name__ == "__main__":
    main()
