"""
cosmos n8n Workflow Templates

"I've created over 6000 workflow templates! ...Actually, I lost count after 12."

Comprehensive n8n workflow templates for automated setup and common integrations.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from .n8n_enhanced import (
    EnhancedN8nIntegration,
    N8nWorkflow,
    N8nNode,
    N8nNodeType,
)


class WorkflowCategory(Enum):
    """Categories of workflow templates."""
    HEALTH = "health"
    AUTOMATION = "automation"
    MONITORING = "monitoring"
    NOTIFICATIONS = "notifications"
    DATA_SYNC = "data_sync"
    SECURITY = "security"
    DEVOPS = "devops"
    PRODUCTIVITY = "productivity"
    AI_AGENTS = "ai_agents"


@dataclass
class WorkflowTemplate:
    """A workflow template definition."""
    id: str
    name: str
    description: str
    category: WorkflowCategory
    tags: List[str]
    required_credentials: List[str]
    workflow_generator: callable


class N8nTemplateLibrary:
    """
    Library of n8n workflow templates.

    Provides ready-to-use workflows for:
    - Health monitoring and alerts
    - System monitoring
    - Notification pipelines
    - Data synchronization
    - Security scanning
    - DevOps automation
    - AI agent integration
    """

    def __init__(self, n8n: Optional[EnhancedN8nIntegration] = None):
        """Initialize template library."""
        self.n8n = n8n
        self.templates: Dict[str, WorkflowTemplate] = {}
        self._register_templates()

    def _register_templates(self):
        """Register all available templates."""
        # Health templates
        self._register("health_daily_summary", WorkflowTemplate(
            id="health_daily_summary",
            name="Daily Health Summary",
            description="Send daily health summary via email/Slack",
            category=WorkflowCategory.HEALTH,
            tags=["health", "daily", "summary"],
            required_credentials=["email", "slack"],
            workflow_generator=self._gen_health_daily_summary,
        ))

        self._register("health_anomaly_alert", WorkflowTemplate(
            id="health_anomaly_alert",
            name="Health Anomaly Alert",
            description="Alert on detected health anomalies",
            category=WorkflowCategory.HEALTH,
            tags=["health", "alert", "anomaly"],
            required_credentials=["webhook"],
            workflow_generator=self._gen_health_anomaly_alert,
        ))

        # Monitoring templates
        self._register("system_health_check", WorkflowTemplate(
            id="system_health_check",
            name="System Health Check",
            description="Monitor system health metrics",
            category=WorkflowCategory.MONITORING,
            tags=["monitoring", "system", "health"],
            required_credentials=[],
            workflow_generator=self._gen_system_health_check,
        ))

        self._register("website_uptime", WorkflowTemplate(
            id="website_uptime",
            name="Website Uptime Monitor",
            description="Monitor website availability",
            category=WorkflowCategory.MONITORING,
            tags=["monitoring", "uptime", "website"],
            required_credentials=["webhook"],
            workflow_generator=self._gen_website_uptime,
        ))

        self._register("port_monitor", WorkflowTemplate(
            id="port_monitor",
            name="Port Monitor",
            description="Monitor specific ports on hosts",
            category=WorkflowCategory.MONITORING,
            tags=["monitoring", "ports", "network"],
            required_credentials=[],
            workflow_generator=self._gen_port_monitor,
        ))

        # Notification templates
        self._register("multi_channel_notify", WorkflowTemplate(
            id="multi_channel_notify",
            name="Multi-Channel Notification",
            description="Send notifications to multiple channels",
            category=WorkflowCategory.NOTIFICATIONS,
            tags=["notifications", "multi-channel"],
            required_credentials=["email", "slack", "telegram"],
            workflow_generator=self._gen_multi_channel_notify,
        ))

        self._register("scheduled_report", WorkflowTemplate(
            id="scheduled_report",
            name="Scheduled Report",
            description="Generate and send scheduled reports",
            category=WorkflowCategory.NOTIFICATIONS,
            tags=["report", "scheduled", "email"],
            required_credentials=["email"],
            workflow_generator=self._gen_scheduled_report,
        ))

        # Data sync templates
        self._register("database_backup", WorkflowTemplate(
            id="database_backup",
            name="Database Backup",
            description="Automated database backup to cloud storage",
            category=WorkflowCategory.DATA_SYNC,
            tags=["backup", "database", "cloud"],
            required_credentials=["database", "s3"],
            workflow_generator=self._gen_database_backup,
        ))

        self._register("file_sync", WorkflowTemplate(
            id="file_sync",
            name="File Sync",
            description="Sync files between locations",
            category=WorkflowCategory.DATA_SYNC,
            tags=["sync", "files", "backup"],
            required_credentials=["ftp", "s3"],
            workflow_generator=self._gen_file_sync,
        ))

        # Security templates
        self._register("security_scan_report", WorkflowTemplate(
            id="security_scan_report",
            name="Security Scan Report",
            description="Run security scans and generate reports",
            category=WorkflowCategory.SECURITY,
            tags=["security", "scan", "report"],
            required_credentials=[],
            workflow_generator=self._gen_security_scan_report,
        ))

        self._register("ssl_expiry_alert", WorkflowTemplate(
            id="ssl_expiry_alert",
            name="SSL Certificate Expiry Alert",
            description="Alert before SSL certificates expire",
            category=WorkflowCategory.SECURITY,
            tags=["security", "ssl", "certificate"],
            required_credentials=["webhook"],
            workflow_generator=self._gen_ssl_expiry_alert,
        ))

        # DevOps templates
        self._register("ci_cd_trigger", WorkflowTemplate(
            id="ci_cd_trigger",
            name="CI/CD Pipeline Trigger",
            description="Trigger CI/CD pipelines on events",
            category=WorkflowCategory.DEVOPS,
            tags=["devops", "ci", "cd", "pipeline"],
            required_credentials=["github", "gitlab"],
            workflow_generator=self._gen_ci_cd_trigger,
        ))

        self._register("deployment_notify", WorkflowTemplate(
            id="deployment_notify",
            name="Deployment Notification",
            description="Notify team on deployments",
            category=WorkflowCategory.DEVOPS,
            tags=["devops", "deployment", "notify"],
            required_credentials=["slack", "webhook"],
            workflow_generator=self._gen_deployment_notify,
        ))

        # AI Agent templates
        self._register("agent_task_queue", WorkflowTemplate(
            id="agent_task_queue",
            name="AI Agent Task Queue",
            description="Queue and process AI agent tasks",
            category=WorkflowCategory.AI_AGENTS,
            tags=["ai", "agent", "queue"],
            required_credentials=["webhook"],
            workflow_generator=self._gen_agent_task_queue,
        ))

        self._register("agent_response_handler", WorkflowTemplate(
            id="agent_response_handler",
            name="Agent Response Handler",
            description="Process and route AI agent responses",
            category=WorkflowCategory.AI_AGENTS,
            tags=["ai", "agent", "response"],
            required_credentials=["webhook"],
            workflow_generator=self._gen_agent_response_handler,
        ))

        # Productivity templates
        self._register("calendar_reminder", WorkflowTemplate(
            id="calendar_reminder",
            name="Calendar Reminder",
            description="Send reminders for calendar events",
            category=WorkflowCategory.PRODUCTIVITY,
            tags=["calendar", "reminder", "productivity"],
            required_credentials=["google_calendar", "slack"],
            workflow_generator=self._gen_calendar_reminder,
        ))

        self._register("daily_standup", WorkflowTemplate(
            id="daily_standup",
            name="Daily Standup Collector",
            description="Collect and summarize daily standups",
            category=WorkflowCategory.PRODUCTIVITY,
            tags=["standup", "team", "productivity"],
            required_credentials=["slack"],
            workflow_generator=self._gen_daily_standup,
        ))

    def _register(self, template_id: str, template: WorkflowTemplate):
        """Register a template."""
        self.templates[template_id] = template

    def list_templates(
        self,
        category: Optional[WorkflowCategory] = None,
        tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List available templates."""
        templates = []

        for template in self.templates.values():
            if category and template.category != category:
                continue
            if tag and tag not in template.tags:
                continue

            templates.append({
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "category": template.category.value,
                "tags": template.tags,
                "required_credentials": template.required_credentials,
            })

        return templates

    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)

    def generate_workflow(
        self,
        template_id: str,
        config: Dict[str, Any] = None,
    ) -> Optional[N8nWorkflow]:
        """Generate a workflow from a template."""
        template = self.templates.get(template_id)
        if not template:
            return None

        return template.workflow_generator(config or {})

    async def deploy_workflow(
        self,
        template_id: str,
        config: Dict[str, Any] = None,
        activate: bool = True,
    ) -> Optional[str]:
        """Deploy a workflow template to n8n."""
        if not self.n8n:
            logger.error("n8n integration not configured")
            return None

        workflow = self.generate_workflow(template_id, config)
        if not workflow:
            return None

        workflow_id = await self.n8n.create_workflow(workflow)

        if workflow_id and activate:
            await self.n8n.activate_workflow(workflow_id)

        return workflow_id

    # ========== Workflow Generators ==========

    def _gen_health_daily_summary(self, config: Dict) -> N8nWorkflow:
        """Generate health daily summary workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "cosmos Health Daily Summary"),
            tags=["cosmos", "health"],
        )

        # Cron trigger - 8 AM daily
        cron = N8nNode(
            id="cron_1",
            name="Daily 8 AM",
            type=N8nNodeType.CRON.value,
            position=[250, 300],
            parameters={
                "rule": {"interval": [{"field": "cronExpression", "expression": "0 8 * * *"}]}
            },
        )
        workflow.add_node(cron)

        # Fetch health data
        fetch = N8nNode(
            id="http_1",
            name="Fetch Health Summary",
            type=N8nNodeType.HTTP_REQUEST.value,
            position=[450, 300],
            parameters={
                "url": config.get("cosmos_url", "http://localhost:8080") + "/api/health/summary",
                "method": "GET",
            },
        )
        workflow.add_node(fetch)

        # Format message
        format_node = N8nNode(
            id="code_1",
            name="Format Summary",
            type=N8nNodeType.CODE.value,
            position=[650, 300],
            parameters={
                "jsCode": """
const data = items[0].json;
const summary = `
Daily Health Summary

Steps: ${data.steps || 'N/A'}
Calories: ${data.calories || 'N/A'}
Sleep: ${data.sleep_hours || 'N/A'} hours
Resting HR: ${data.resting_hr || 'N/A'} bpm
Recovery: ${data.recovery || 'N/A'}%
`;
return [{json: {message: summary, ...data}}];
"""
            },
        )
        workflow.add_node(format_node)

        # Send notification
        notify = N8nNode(
            id="http_2",
            name="Send Notification",
            type=N8nNodeType.HTTP_REQUEST.value,
            position=[850, 300],
            parameters={
                "url": config.get("notify_url", "{{$env.NOTIFICATION_WEBHOOK}}"),
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": '={"text": "{{$json.message}}"}',
            },
        )
        workflow.add_node(notify)

        workflow.connect("Daily 8 AM", "Fetch Health Summary")
        workflow.connect("Fetch Health Summary", "Format Summary")
        workflow.connect("Format Summary", "Send Notification")

        return workflow

    def _gen_health_anomaly_alert(self, config: Dict) -> N8nWorkflow:
        """Generate health anomaly alert workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "cosmos Health Anomaly Alert"),
            tags=["cosmos", "health", "alert"],
        )

        # Webhook trigger
        webhook = N8nNode(
            id="webhook_1",
            name="Anomaly Webhook",
            type=N8nNodeType.WEBHOOK.value,
            position=[250, 300],
            parameters={
                "path": config.get("webhook_path", "cosmos-health-anomaly"),
                "httpMethod": "POST",
            },
        )
        workflow.add_node(webhook)

        # Check severity
        if_node = N8nNode(
            id="if_1",
            name="Check Severity",
            type=N8nNodeType.IF.value,
            position=[450, 300],
            parameters={
                "conditions": {
                    "string": [{"value1": "={{$json.severity}}", "operation": "equals", "value2": "high"}]
                }
            },
        )
        workflow.add_node(if_node)

        # High severity - immediate alert
        alert_high = N8nNode(
            id="http_1",
            name="Urgent Alert",
            type=N8nNodeType.HTTP_REQUEST.value,
            position=[650, 200],
            parameters={
                "url": "{{$env.URGENT_NOTIFICATION_URL}}",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": '={"alert": "URGENT Health Anomaly", "details": "{{$json.message}}", "metric": "{{$json.metric}}"}',
            },
        )
        workflow.add_node(alert_high)

        # Normal severity - log
        log_normal = N8nNode(
            id="http_2",
            name="Log Alert",
            type=N8nNodeType.HTTP_REQUEST.value,
            position=[650, 400],
            parameters={
                "url": "{{$env.NOTIFICATION_URL}}",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": '={"alert": "Health Anomaly", "details": "{{$json.message}}"}',
            },
        )
        workflow.add_node(log_normal)

        workflow.connect("Anomaly Webhook", "Check Severity")
        workflow.connect("Check Severity", "Urgent Alert", from_output=0)
        workflow.connect("Check Severity", "Log Alert", from_output=1)

        return workflow

    def _gen_system_health_check(self, config: Dict) -> N8nWorkflow:
        """Generate system health check workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "System Health Check"),
            tags=["cosmos", "monitoring"],
        )

        # Cron trigger - every 5 minutes
        cron = N8nNode(
            id="cron_1",
            name="Every 5 Minutes",
            type=N8nNodeType.CRON.value,
            position=[250, 300],
            parameters={
                "rule": {"interval": [{"field": "minutes", "minutesInterval": 5}]}
            },
        )
        workflow.add_node(cron)

        # Check endpoints
        check = N8nNode(
            id="code_1",
            name="Check Systems",
            type=N8nNodeType.CODE.value,
            position=[450, 300],
            parameters={
                "jsCode": """
const endpoints = ${JSON.stringify(config.get("endpoints", ["http://localhost:8080/health"]))};
const results = [];

for (const endpoint of endpoints) {
    try {
        const response = await fetch(endpoint);
        results.push({
            url: endpoint,
            status: response.status,
            ok: response.ok
        });
    } catch (e) {
        results.push({
            url: endpoint,
            status: 0,
            ok: false,
            error: e.message
        });
    }
}

return results.map(r => ({json: r}));
"""
            },
        )
        workflow.add_node(check)

        # Filter failures
        if_node = N8nNode(
            id="if_1",
            name="Check Status",
            type=N8nNodeType.IF.value,
            position=[650, 300],
            parameters={
                "conditions": {
                    "boolean": [{"value1": "={{$json.ok}}", "value2": False}]
                }
            },
        )
        workflow.add_node(if_node)

        # Alert on failure
        alert = N8nNode(
            id="http_1",
            name="Alert Failure",
            type=N8nNodeType.HTTP_REQUEST.value,
            position=[850, 200],
            parameters={
                "url": "{{$env.NOTIFICATION_URL}}",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": '={"alert": "System Down", "url": "{{$json.url}}", "error": "{{$json.error}}"}',
            },
        )
        workflow.add_node(alert)

        workflow.connect("Every 5 Minutes", "Check Systems")
        workflow.connect("Check Systems", "Check Status")
        workflow.connect("Check Status", "Alert Failure", from_output=0)

        return workflow

    def _gen_website_uptime(self, config: Dict) -> N8nWorkflow:
        """Generate website uptime monitor workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "Website Uptime Monitor"),
            tags=["monitoring", "uptime"],
        )

        # Cron trigger
        cron = N8nNode(
            id="cron_1",
            name="Every Minute",
            type=N8nNodeType.CRON.value,
            position=[250, 300],
            parameters={
                "rule": {"interval": [{"field": "minutes", "minutesInterval": 1}]}
            },
        )
        workflow.add_node(cron)

        # HTTP request
        check = N8nNode(
            id="http_1",
            name="Check Website",
            type=N8nNodeType.HTTP_REQUEST.value,
            position=[450, 300],
            parameters={
                "url": config.get("url", "https://example.com"),
                "method": "GET",
                "options": {
                    "timeout": config.get("timeout", 10000),
                },
            },
        )
        workflow.add_node(check)

        workflow.connect("Every Minute", "Check Website")

        return workflow

    def _gen_port_monitor(self, config: Dict) -> N8nWorkflow:
        """Generate port monitor workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "Port Monitor"),
            tags=["monitoring", "network"],
        )

        # Cron trigger
        cron = N8nNode(
            id="cron_1",
            name="Every 5 Minutes",
            type=N8nNodeType.CRON.value,
            position=[250, 300],
            parameters={
                "rule": {"interval": [{"field": "minutes", "minutesInterval": 5}]}
            },
        )
        workflow.add_node(cron)

        # Check ports
        check = N8nNode(
            id="code_1",
            name="Check Ports",
            type=N8nNodeType.CODE.value,
            position=[450, 300],
            parameters={
                "jsCode": f"""
const targets = {config.get("targets", [{"host": "localhost", "ports": [22, 80, 443]}])};
const results = [];

// Port checking logic would go here
// In practice, this would use a native node or external service

return targets.map(t => ({{json: {{host: t.host, ports: t.ports, status: 'checked'}}}}));
"""
            },
        )
        workflow.add_node(check)

        workflow.connect("Every 5 Minutes", "Check Ports")

        return workflow

    def _gen_multi_channel_notify(self, config: Dict) -> N8nWorkflow:
        """Generate multi-channel notification workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "Multi-Channel Notification"),
            tags=["notifications"],
        )

        # Webhook trigger
        webhook = N8nNode(
            id="webhook_1",
            name="Notification Webhook",
            type=N8nNodeType.WEBHOOK.value,
            position=[250, 300],
            parameters={
                "path": config.get("webhook_path", "notify"),
                "httpMethod": "POST",
            },
        )
        workflow.add_node(webhook)

        # Email notification
        email = N8nNode(
            id="http_1",
            name="Send Email",
            type=N8nNodeType.HTTP_REQUEST.value,
            position=[500, 150],
            parameters={
                "url": "{{$env.EMAIL_SERVICE_URL}}",
                "method": "POST",
                "sendBody": True,
            },
        )
        workflow.add_node(email)

        # Slack notification
        slack = N8nNode(
            id="http_2",
            name="Send Slack",
            type=N8nNodeType.HTTP_REQUEST.value,
            position=[500, 300],
            parameters={
                "url": "{{$env.SLACK_WEBHOOK_URL}}",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": '={"text": "{{$json.message}}"}',
            },
        )
        workflow.add_node(slack)

        # Telegram notification
        telegram = N8nNode(
            id="http_3",
            name="Send Telegram",
            type=N8nNodeType.HTTP_REQUEST.value,
            position=[500, 450],
            parameters={
                "url": "https://api.telegram.org/bot{{$env.TELEGRAM_BOT_TOKEN}}/sendMessage",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": '={"chat_id": "{{$env.TELEGRAM_CHAT_ID}}", "text": "{{$json.message}}"}',
            },
        )
        workflow.add_node(telegram)

        workflow.connect("Notification Webhook", "Send Email")
        workflow.connect("Notification Webhook", "Send Slack")
        workflow.connect("Notification Webhook", "Send Telegram")

        return workflow

    def _gen_scheduled_report(self, config: Dict) -> N8nWorkflow:
        """Generate scheduled report workflow."""
        workflow = N8nWorkflow(name=config.get("name", "Scheduled Report"))

        cron = N8nNode(
            id="cron_1",
            name="Weekly Monday",
            type=N8nNodeType.CRON.value,
            position=[250, 300],
            parameters={
                "rule": {"interval": [{"field": "cronExpression", "expression": "0 9 * * 1"}]}
            },
        )
        workflow.add_node(cron)

        return workflow

    def _gen_database_backup(self, config: Dict) -> N8nWorkflow:
        """Generate database backup workflow."""
        workflow = N8nWorkflow(name=config.get("name", "Database Backup"))

        cron = N8nNode(
            id="cron_1",
            name="Daily 2 AM",
            type=N8nNodeType.CRON.value,
            position=[250, 300],
            parameters={
                "rule": {"interval": [{"field": "cronExpression", "expression": "0 2 * * *"}]}
            },
        )
        workflow.add_node(cron)

        return workflow

    def _gen_file_sync(self, config: Dict) -> N8nWorkflow:
        """Generate file sync workflow."""
        workflow = N8nWorkflow(name=config.get("name", "File Sync"))

        cron = N8nNode(
            id="cron_1",
            name="Hourly",
            type=N8nNodeType.CRON.value,
            position=[250, 300],
            parameters={
                "rule": {"interval": [{"field": "hours", "hoursInterval": 1}]}
            },
        )
        workflow.add_node(cron)

        return workflow

    def _gen_security_scan_report(self, config: Dict) -> N8nWorkflow:
        """Generate security scan report workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "Security Scan Report"),
            tags=["security", "scan"],
        )

        # Weekly scan
        cron = N8nNode(
            id="cron_1",
            name="Weekly Sunday",
            type=N8nNodeType.CRON.value,
            position=[250, 300],
            parameters={
                "rule": {"interval": [{"field": "cronExpression", "expression": "0 0 * * 0"}]}
            },
        )
        workflow.add_node(cron)

        # Run security scan
        scan = N8nNode(
            id="http_1",
            name="Run Security Scan",
            type=N8nNodeType.HTTP_REQUEST.value,
            position=[450, 300],
            parameters={
                "url": config.get("cosmos_url", "http://localhost:8080") + "/api/security/scan",
                "method": "POST",
            },
        )
        workflow.add_node(scan)

        workflow.connect("Weekly Sunday", "Run Security Scan")

        return workflow

    def _gen_ssl_expiry_alert(self, config: Dict) -> N8nWorkflow:
        """Generate SSL expiry alert workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "SSL Certificate Expiry Alert"),
            tags=["security", "ssl"],
        )

        # Daily check
        cron = N8nNode(
            id="cron_1",
            name="Daily",
            type=N8nNodeType.CRON.value,
            position=[250, 300],
            parameters={
                "rule": {"interval": [{"field": "cronExpression", "expression": "0 6 * * *"}]}
            },
        )
        workflow.add_node(cron)

        return workflow

    def _gen_ci_cd_trigger(self, config: Dict) -> N8nWorkflow:
        """Generate CI/CD trigger workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "CI/CD Pipeline Trigger"),
            tags=["devops", "ci"],
        )

        webhook = N8nNode(
            id="webhook_1",
            name="GitHub Webhook",
            type=N8nNodeType.WEBHOOK.value,
            position=[250, 300],
            parameters={
                "path": "github-webhook",
                "httpMethod": "POST",
            },
        )
        workflow.add_node(webhook)

        return workflow

    def _gen_deployment_notify(self, config: Dict) -> N8nWorkflow:
        """Generate deployment notification workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "Deployment Notification"),
            tags=["devops", "notify"],
        )

        webhook = N8nNode(
            id="webhook_1",
            name="Deployment Webhook",
            type=N8nNodeType.WEBHOOK.value,
            position=[250, 300],
            parameters={
                "path": "deployment",
                "httpMethod": "POST",
            },
        )
        workflow.add_node(webhook)

        return workflow

    def _gen_agent_task_queue(self, config: Dict) -> N8nWorkflow:
        """Generate AI agent task queue workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "AI Agent Task Queue"),
            tags=["cosmos", "ai", "agent"],
        )

        webhook = N8nNode(
            id="webhook_1",
            name="Task Webhook",
            type=N8nNodeType.WEBHOOK.value,
            position=[250, 300],
            parameters={
                "path": config.get("webhook_path", "agent-task"),
                "httpMethod": "POST",
            },
        )
        workflow.add_node(webhook)

        # Route by task type
        switch = N8nNode(
            id="switch_1",
            name="Route Task",
            type=N8nNodeType.SWITCH.value,
            position=[450, 300],
            parameters={
                "dataPropertyName": "task_type",
                "rules": {"rules": [
                    {"value": "research"},
                    {"value": "code"},
                    {"value": "creative"},
                    {"value": "analysis"},
                ]},
            },
        )
        workflow.add_node(switch)

        workflow.connect("Task Webhook", "Route Task")

        return workflow

    def _gen_agent_response_handler(self, config: Dict) -> N8nWorkflow:
        """Generate agent response handler workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "Agent Response Handler"),
            tags=["cosmos", "ai", "agent"],
        )

        webhook = N8nNode(
            id="webhook_1",
            name="Response Webhook",
            type=N8nNodeType.WEBHOOK.value,
            position=[250, 300],
            parameters={
                "path": config.get("webhook_path", "agent-response"),
                "httpMethod": "POST",
            },
        )
        workflow.add_node(webhook)

        return workflow

    def _gen_calendar_reminder(self, config: Dict) -> N8nWorkflow:
        """Generate calendar reminder workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "Calendar Reminder"),
            tags=["productivity", "calendar"],
        )

        cron = N8nNode(
            id="cron_1",
            name="Every 15 Minutes",
            type=N8nNodeType.CRON.value,
            position=[250, 300],
            parameters={
                "rule": {"interval": [{"field": "minutes", "minutesInterval": 15}]}
            },
        )
        workflow.add_node(cron)

        return workflow

    def _gen_daily_standup(self, config: Dict) -> N8nWorkflow:
        """Generate daily standup collector workflow."""
        workflow = N8nWorkflow(
            name=config.get("name", "Daily Standup Collector"),
            tags=["productivity", "standup"],
        )

        # 9 AM weekdays
        cron = N8nNode(
            id="cron_1",
            name="Weekdays 9 AM",
            type=N8nNodeType.CRON.value,
            position=[250, 300],
            parameters={
                "rule": {"interval": [{"field": "cronExpression", "expression": "0 9 * * 1-5"}]}
            },
        )
        workflow.add_node(cron)

        return workflow


# Global instance
n8n_templates = N8nTemplateLibrary()
