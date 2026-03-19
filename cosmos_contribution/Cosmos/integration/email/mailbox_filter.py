"""
cosmos Mailbox Filter

"Good news, everyone! I can sort your email faster than you can say 'spam'!"

Universal mailbox filtering that works across email providers.
"""

import re
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from loguru import logger


class FilterConditionType(Enum):
    """Types of filter conditions."""
    FROM_CONTAINS = "from_contains"
    FROM_EQUALS = "from_equals"
    FROM_DOMAIN = "from_domain"
    TO_CONTAINS = "to_contains"
    TO_EQUALS = "to_equals"
    SUBJECT_CONTAINS = "subject_contains"
    SUBJECT_MATCHES = "subject_matches"  # Regex
    BODY_CONTAINS = "body_contains"
    HAS_ATTACHMENT = "has_attachment"
    ATTACHMENT_NAME = "attachment_name"
    HEADER_EXISTS = "header_exists"
    HEADER_CONTAINS = "header_contains"
    SIZE_GREATER_THAN = "size_greater_than"
    SIZE_LESS_THAN = "size_less_than"
    IS_SPAM = "is_spam"
    IS_PHISHING = "is_phishing"
    CUSTOM = "custom"


class FilterActionType(Enum):
    """Types of filter actions."""
    MOVE_TO_FOLDER = "move_to_folder"
    APPLY_LABEL = "apply_label"
    MARK_AS_READ = "mark_as_read"
    MARK_AS_UNREAD = "mark_as_unread"
    STAR = "star"
    DELETE = "delete"
    ARCHIVE = "archive"
    FORWARD = "forward"
    REPLY = "reply"
    CATEGORIZE = "categorize"
    SET_PRIORITY = "set_priority"
    BLOCK_SENDER = "block_sender"
    QUARANTINE = "quarantine"
    ALERT = "alert"
    CUSTOM = "custom"


@dataclass
class FilterCondition:
    """A single filter condition."""
    type: FilterConditionType
    value: Any
    negate: bool = False  # NOT condition

    def evaluate(self, email: Dict[str, Any]) -> bool:
        """Evaluate condition against an email."""
        result = False

        if self.type == FilterConditionType.FROM_CONTAINS:
            from_addr = email.get("from", "").lower()
            result = self.value.lower() in from_addr

        elif self.type == FilterConditionType.FROM_EQUALS:
            from_addr = email.get("from", "").lower()
            result = self.value.lower() == from_addr

        elif self.type == FilterConditionType.FROM_DOMAIN:
            from_addr = email.get("from", "").lower()
            if "@" in from_addr:
                domain = from_addr.split("@")[-1]
                result = domain == self.value.lower()

        elif self.type == FilterConditionType.TO_CONTAINS:
            to_addrs = " ".join(email.get("to", [])).lower()
            result = self.value.lower() in to_addrs

        elif self.type == FilterConditionType.TO_EQUALS:
            to_addrs = [addr.lower() for addr in email.get("to", [])]
            result = self.value.lower() in to_addrs

        elif self.type == FilterConditionType.SUBJECT_CONTAINS:
            subject = email.get("subject", "").lower()
            result = self.value.lower() in subject

        elif self.type == FilterConditionType.SUBJECT_MATCHES:
            subject = email.get("subject", "")
            try:
                result = bool(re.search(self.value, subject, re.I))
            except re.error:
                result = False

        elif self.type == FilterConditionType.BODY_CONTAINS:
            body = email.get("body", "").lower()
            result = self.value.lower() in body

        elif self.type == FilterConditionType.HAS_ATTACHMENT:
            result = email.get("has_attachments", False) == self.value

        elif self.type == FilterConditionType.ATTACHMENT_NAME:
            attachments = email.get("attachments", [])
            for att in attachments:
                if self.value.lower() in att.get("filename", "").lower():
                    result = True
                    break

        elif self.type == FilterConditionType.HEADER_EXISTS:
            headers = email.get("headers", {})
            result = self.value.lower() in [k.lower() for k in headers.keys()]

        elif self.type == FilterConditionType.HEADER_CONTAINS:
            headers = email.get("headers", {})
            header_name, search_value = self.value.split(":", 1) if ":" in self.value else (self.value, "")
            header_value = headers.get(header_name, "").lower()
            result = search_value.lower() in header_value

        elif self.type == FilterConditionType.SIZE_GREATER_THAN:
            size = email.get("size", 0)
            result = size > self.value

        elif self.type == FilterConditionType.SIZE_LESS_THAN:
            size = email.get("size", 0)
            result = size < self.value

        elif self.type == FilterConditionType.IS_SPAM:
            # Check various spam indicators
            headers = email.get("headers", {})
            spam_indicators = [
                "x-spam-flag" in [k.lower() for k in headers.keys()],
                headers.get("X-Spam-Status", "").lower().startswith("yes"),
                "SPAM" in email.get("label_ids", []),
            ]
            result = any(spam_indicators)

        elif self.type == FilterConditionType.IS_PHISHING:
            # Basic phishing detection
            result = self._check_phishing_indicators(email)

        # Apply negation
        return not result if self.negate else result

    def _check_phishing_indicators(self, email: Dict[str, Any]) -> bool:
        """Check for phishing indicators."""
        indicators = 0
        from_addr = email.get("from", "").lower()
        subject = email.get("subject", "").lower()
        body = email.get("body", "").lower()
        headers = email.get("headers", {})

        # Check for spoofed sender
        reply_to = headers.get("Reply-To", "").lower()
        if reply_to and "@" in reply_to and "@" in from_addr:
            reply_domain = reply_to.split("@")[-1]
            from_domain = from_addr.split("@")[-1]
            if reply_domain != from_domain:
                indicators += 2

        # Check authentication
        auth_results = headers.get("Authentication-Results", "").lower()
        if "spf=fail" in auth_results:
            indicators += 2
        if "dkim=fail" in auth_results:
            indicators += 2
        if "dmarc=fail" in auth_results:
            indicators += 2

        # Check for urgency keywords
        urgency_keywords = ["urgent", "immediate action", "verify your account", "suspended", "compromised"]
        for keyword in urgency_keywords:
            if keyword in subject or keyword in body:
                indicators += 1

        # Check for credential harvesting keywords
        cred_keywords = ["password", "login", "credentials", "verify your identity", "click here to confirm"]
        for keyword in cred_keywords:
            if keyword in body:
                indicators += 1

        return indicators >= 3


@dataclass
class FilterAction:
    """A single filter action."""
    type: FilterActionType
    value: Any = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterRule:
    """A complete filter rule with conditions and actions."""
    id: str
    name: str
    description: str = ""
    conditions: List[FilterCondition] = field(default_factory=list)
    actions: List[FilterAction] = field(default_factory=list)
    match_all: bool = True  # AND vs OR for conditions
    enabled: bool = True
    priority: int = 0  # Higher = processed first
    created_at: datetime = field(default_factory=datetime.now)
    last_matched: Optional[datetime] = None
    match_count: int = 0
    tags: List[str] = field(default_factory=list)

    def matches(self, email: Dict[str, Any]) -> bool:
        """Check if email matches this rule."""
        if not self.enabled or not self.conditions:
            return False

        if self.match_all:
            return all(c.evaluate(email) for c in self.conditions)
        else:
            return any(c.evaluate(email) for c in self.conditions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "conditions": [
                {"type": c.type.value, "value": c.value, "negate": c.negate}
                for c in self.conditions
            ],
            "actions": [
                {"type": a.type.value, "value": a.value, "params": a.additional_params}
                for a in self.actions
            ],
            "match_all": self.match_all,
            "enabled": self.enabled,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "match_count": self.match_count,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterRule":
        """Create rule from dictionary."""
        conditions = [
            FilterCondition(
                type=FilterConditionType(c["type"]),
                value=c["value"],
                negate=c.get("negate", False),
            )
            for c in data.get("conditions", [])
        ]

        actions = [
            FilterAction(
                type=FilterActionType(a["type"]),
                value=a.get("value"),
                additional_params=a.get("params", {}),
            )
            for a in data.get("actions", [])
        ]

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            conditions=conditions,
            actions=actions,
            match_all=data.get("match_all", True),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 0),
            tags=data.get("tags", []),
        )


class MailboxFilter:
    """
    Universal mailbox filtering engine.

    Works with Office 365, Google Workspace, or any email source.
    Can create native rules on providers or process locally.
    """

    def __init__(self, data_dir: str = "./data/filters"):
        """Initialize mailbox filter."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.rules: Dict[str, FilterRule] = {}
        self._action_handlers: Dict[FilterActionType, Callable] = {}
        self._rule_counter = 0

        # Load saved rules
        self._load_rules()

        # Register default action handlers
        self._register_default_handlers()

    def _load_rules(self):
        """Load rules from disk."""
        rules_file = self.data_dir / "rules.json"
        if rules_file.exists():
            try:
                data = json.loads(rules_file.read_text())
                for rule_data in data.get("rules", []):
                    rule = FilterRule.from_dict(rule_data)
                    self.rules[rule.id] = rule
                logger.info(f"Loaded {len(self.rules)} filter rules")
            except Exception as e:
                logger.error(f"Error loading rules: {e}")

    def _save_rules(self):
        """Save rules to disk."""
        rules_file = self.data_dir / "rules.json"
        data = {
            "rules": [r.to_dict() for r in self.rules.values()],
            "updated_at": datetime.now().isoformat(),
        }
        rules_file.write_text(json.dumps(data, indent=2))

    def _register_default_handlers(self):
        """Register default action handlers."""
        self._action_handlers[FilterActionType.ALERT] = self._action_alert
        self._action_handlers[FilterActionType.CATEGORIZE] = self._action_categorize

    def _generate_rule_id(self) -> str:
        """Generate unique rule ID."""
        self._rule_counter += 1
        return f"rule_{datetime.now().strftime('%Y%m%d')}_{self._rule_counter:04d}"

    def register_action_handler(
        self,
        action_type: FilterActionType,
        handler: Callable[[Dict[str, Any], FilterAction], bool],
    ):
        """
        Register a custom action handler.

        Handler receives (email, action) and returns success status.
        """
        self._action_handlers[action_type] = handler

    # ========== Rule Management ==========

    def create_rule(
        self,
        name: str,
        conditions: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        description: str = "",
        match_all: bool = True,
        priority: int = 0,
        tags: List[str] = None,
    ) -> FilterRule:
        """
        Create a new filter rule.

        Args:
            name: Rule name
            conditions: List of condition dicts with 'type' and 'value'
            actions: List of action dicts with 'type' and optional 'value'
            description: Rule description
            match_all: True for AND, False for OR
            priority: Higher priority rules processed first
            tags: Tags for organization

        Example:
            rule = filter.create_rule(
                name="Spam Filter",
                conditions=[
                    {"type": "from_domain", "value": "spam.com"},
                    {"type": "subject_contains", "value": "urgent", "negate": True},
                ],
                actions=[
                    {"type": "move_to_folder", "value": "Spam"},
                    {"type": "mark_as_read"},
                ],
            )
        """
        parsed_conditions = [
            FilterCondition(
                type=FilterConditionType(c["type"]),
                value=c.get("value"),
                negate=c.get("negate", False),
            )
            for c in conditions
        ]

        parsed_actions = [
            FilterAction(
                type=FilterActionType(a["type"]),
                value=a.get("value"),
                additional_params=a.get("params", {}),
            )
            for a in actions
        ]

        rule = FilterRule(
            id=self._generate_rule_id(),
            name=name,
            description=description,
            conditions=parsed_conditions,
            actions=parsed_actions,
            match_all=match_all,
            priority=priority,
            tags=tags or [],
        )

        self.rules[rule.id] = rule
        self._save_rules()

        logger.info(f"Created filter rule: {name} ({rule.id})")
        return rule

    def update_rule(self, rule_id: str, **updates) -> Optional[FilterRule]:
        """Update an existing rule."""
        if rule_id not in self.rules:
            return None

        rule = self.rules[rule_id]

        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)

        self._save_rules()
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self._save_rules()
            return True
        return False

    def get_rule(self, rule_id: str) -> Optional[FilterRule]:
        """Get a rule by ID."""
        return self.rules.get(rule_id)

    def list_rules(
        self,
        enabled_only: bool = False,
        tag: str = None,
    ) -> List[FilterRule]:
        """List all rules."""
        rules = list(self.rules.values())

        if enabled_only:
            rules = [r for r in rules if r.enabled]

        if tag:
            rules = [r for r in rules if tag in r.tags]

        # Sort by priority (highest first)
        rules.sort(key=lambda r: r.priority, reverse=True)

        return rules

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            self._save_rules()
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            self._save_rules()
            return True
        return False

    # ========== Email Processing ==========

    def process_email(
        self,
        email: Dict[str, Any],
        execute_actions: bool = True,
    ) -> List[tuple]:
        """
        Process an email through all rules.

        Args:
            email: Email data dictionary
            execute_actions: Whether to execute matched actions

        Returns:
            List of (rule_id, actions) tuples for matched rules
        """
        matches = []
        rules = self.list_rules(enabled_only=True)

        for rule in rules:
            if rule.matches(email):
                rule.last_matched = datetime.now()
                rule.match_count += 1

                logger.debug(f"Email matched rule: {rule.name}")

                if execute_actions:
                    for action in rule.actions:
                        self._execute_action(email, action)

                matches.append((rule.id, rule.actions))

        if matches:
            self._save_rules()

        return matches

    def process_batch(
        self,
        emails: List[Dict[str, Any]],
        execute_actions: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Process multiple emails.

        Returns:
            Dictionary mapping email_id to list of matched rule_ids
        """
        results = {}

        for email in emails:
            email_id = email.get("id", "")
            matches = self.process_email(email, execute_actions)
            results[email_id] = [m[0] for m in matches]

        return results

    def _execute_action(self, email: Dict[str, Any], action: FilterAction) -> bool:
        """Execute a filter action."""
        handler = self._action_handlers.get(action.type)

        if handler:
            try:
                return handler(email, action)
            except Exception as e:
                logger.error(f"Action execution failed ({action.type}): {e}")
                return False
        else:
            logger.warning(f"No handler for action type: {action.type}")
            return False

    # ========== Default Action Handlers ==========

    def _action_alert(self, email: Dict[str, Any], action: FilterAction) -> bool:
        """Alert action handler."""
        logger.warning(
            f"[ALERT] Email filter triggered: "
            f"From: {email.get('from')} Subject: {email.get('subject')}"
        )
        return True

    def _action_categorize(self, email: Dict[str, Any], action: FilterAction) -> bool:
        """Categorize action handler."""
        category = action.value
        email.setdefault("categories", []).append(category)
        return True

    # ========== Pre-built Rule Templates ==========

    def create_spam_filter(self, blocked_domains: List[str]) -> FilterRule:
        """Create a spam filter for specific domains."""
        conditions = [
            {"type": "from_domain", "value": domain}
            for domain in blocked_domains
        ]

        return self.create_rule(
            name="Domain Block List",
            conditions=conditions,
            actions=[
                {"type": "move_to_folder", "value": "Spam"},
                {"type": "mark_as_read"},
            ],
            match_all=False,  # OR - match any domain
            priority=100,
            tags=["spam", "security"],
        )

    def create_phishing_filter(self) -> FilterRule:
        """Create a phishing detection filter."""
        return self.create_rule(
            name="Phishing Detection",
            conditions=[
                {"type": "is_phishing", "value": True},
            ],
            actions=[
                {"type": "quarantine"},
                {"type": "alert"},
            ],
            priority=200,
            tags=["security", "phishing"],
        )

    def create_priority_sender_filter(
        self,
        senders: List[str],
        folder: str = "Priority",
    ) -> FilterRule:
        """Create filter for priority senders."""
        conditions = [
            {"type": "from_contains", "value": sender}
            for sender in senders
        ]

        return self.create_rule(
            name="Priority Senders",
            conditions=conditions,
            actions=[
                {"type": "move_to_folder", "value": folder},
                {"type": "star"},
            ],
            match_all=False,
            priority=50,
            tags=["priority"],
        )

    def create_attachment_filter(
        self,
        extensions: List[str],
        action: str = "quarantine",
    ) -> FilterRule:
        """Create filter for suspicious attachments."""
        conditions = [
            {"type": "has_attachment", "value": True},
        ]

        # Add extension checks
        for ext in extensions:
            conditions.append({"type": "attachment_name", "value": f".{ext}"})

        return self.create_rule(
            name="Suspicious Attachments",
            description=f"Filter emails with {', '.join(extensions)} attachments",
            conditions=conditions,
            actions=[
                {"type": action},
                {"type": "alert"},
            ],
            match_all=False,  # Any suspicious extension
            priority=150,
            tags=["security", "attachments"],
        )

    # ========== Provider Sync ==========

    async def sync_to_office365(
        self,
        integration,
        user: str = "me",
    ) -> int:
        """
        Sync rules to Office 365 mailbox rules.

        Returns number of rules synced.
        """
        synced = 0

        for rule in self.list_rules(enabled_only=True):
            try:
                conditions = self._convert_conditions_to_o365(rule.conditions)
                actions = self._convert_actions_to_o365(rule.actions)

                await integration.create_rule(
                    name=rule.name,
                    conditions=conditions,
                    actions=actions,
                    user=user,
                )
                synced += 1

            except Exception as e:
                logger.error(f"Failed to sync rule {rule.name} to O365: {e}")

        logger.info(f"Synced {synced} rules to Office 365")
        return synced

    def _convert_conditions_to_o365(self, conditions: List[FilterCondition]) -> Dict:
        """Convert conditions to Office 365 format."""
        o365_conditions = {}

        for c in conditions:
            if c.type == FilterConditionType.FROM_CONTAINS:
                o365_conditions.setdefault("senderContains", []).append(c.value)
            elif c.type == FilterConditionType.SUBJECT_CONTAINS:
                o365_conditions.setdefault("subjectContains", []).append(c.value)
            elif c.type == FilterConditionType.HAS_ATTACHMENT:
                o365_conditions["hasAttachments"] = c.value

        return o365_conditions

    def _convert_actions_to_o365(self, actions: List[FilterAction]) -> Dict:
        """Convert actions to Office 365 format."""
        o365_actions = {}

        for a in actions:
            if a.type == FilterActionType.MOVE_TO_FOLDER:
                o365_actions["moveToFolder"] = a.value
            elif a.type == FilterActionType.MARK_AS_READ:
                o365_actions["markAsRead"] = True
            elif a.type == FilterActionType.DELETE:
                o365_actions["delete"] = True

        return o365_actions

    async def sync_to_gmail(
        self,
        integration,
        user: str = "me",
    ) -> int:
        """
        Sync rules to Gmail filters.

        Returns number of rules synced.
        """
        synced = 0

        for rule in self.list_rules(enabled_only=True):
            try:
                criteria = self._convert_conditions_to_gmail(rule.conditions)
                action = self._convert_actions_to_gmail(rule.actions)

                await integration.create_filter(
                    criteria=criteria,
                    action=action,
                    user=user,
                )
                synced += 1

            except Exception as e:
                logger.error(f"Failed to sync rule {rule.name} to Gmail: {e}")

        logger.info(f"Synced {synced} rules to Gmail")
        return synced

    def _convert_conditions_to_gmail(self, conditions: List[FilterCondition]) -> Dict:
        """Convert conditions to Gmail filter criteria."""
        criteria = {}

        for c in conditions:
            if c.type == FilterConditionType.FROM_CONTAINS:
                criteria["from"] = c.value
            elif c.type == FilterConditionType.TO_CONTAINS:
                criteria["to"] = c.value
            elif c.type == FilterConditionType.SUBJECT_CONTAINS:
                criteria["subject"] = c.value
            elif c.type == FilterConditionType.HAS_ATTACHMENT:
                criteria["hasAttachment"] = c.value

        return criteria

    def _convert_actions_to_gmail(self, actions: List[FilterAction]) -> Dict:
        """Convert actions to Gmail filter actions."""
        gmail_actions = {}

        for a in actions:
            if a.type == FilterActionType.APPLY_LABEL:
                gmail_actions.setdefault("addLabelIds", []).append(a.value)
            elif a.type == FilterActionType.ARCHIVE:
                gmail_actions.setdefault("removeLabelIds", []).append("INBOX")
            elif a.type == FilterActionType.MARK_AS_READ:
                gmail_actions.setdefault("removeLabelIds", []).append("UNREAD")
            elif a.type == FilterActionType.FORWARD:
                gmail_actions["forward"] = a.value

        return gmail_actions


# Global instance
mailbox_filter = MailboxFilter()
