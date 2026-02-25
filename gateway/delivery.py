"""
Delivery routing for cron job outputs and agent responses.

Routes messages to the appropriate destination based on:
- Explicit targets (e.g., "telegram:123456789")
- Platform home channels (e.g., "telegram" → home channel)
- Origin (back to where the job was created)
- Local (always saved to files)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

MAX_PLATFORM_OUTPUT = 4000
TRUNCATED_VISIBLE = 3800

from .config import GatewayConfig, Platform
from .session import SessionSource


@dataclass
class DeliveryTarget:
    """
    A single delivery target.

    Represents where a message should be sent:
    - "origin" → back to source
    - "local" → save to local files
    - "telegram" → Telegram home channel
    - "telegram:123456" → specific Telegram chat
    """

    platform: Platform
    chat_id: Optional[str] = None  # None means use home channel
    is_origin: bool = False
    is_explicit: bool = False  # True if chat_id was explicitly specified

    @classmethod
    def parse(cls, target: str, origin: Optional[SessionSource] = None) -> "DeliveryTarget":
        """
        Parse a delivery target string.

        Formats:
        - "origin" → back to source
        - "local" → local files only
        - "telegram" → Telegram home channel
        - "telegram:123456" → specific Telegram chat
        """
        target = target.strip().lower()

        if target == "origin":
            if origin:
                return cls(
                    platform=origin.platform,
                    chat_id=origin.chat_id,
                    is_origin=True,
                )
            else:
                # Fallback to local if no origin
                return cls(platform=Platform.LOCAL, is_origin=True)

        if target == "local":
            return cls(platform=Platform.LOCAL)

        # Check for platform:chat_id format
        if ":" in target:
            platform_str, chat_id = target.split(":", 1)
            try:
                platform = Platform(platform_str)
                return cls(platform=platform, chat_id=chat_id, is_explicit=True)
            except ValueError:
                # Unknown platform, treat as local
                return cls(platform=Platform.LOCAL)

        # Just a platform name (use home channel)
        try:
            platform = Platform(target)
            return cls(platform=platform)
        except ValueError:
            # Unknown platform, treat as local
            return cls(platform=Platform.LOCAL)

    def to_string(self) -> str:
        """Convert back to string format."""
        if self.is_origin:
            return "origin"
        if self.platform == Platform.LOCAL:
            return "local"
        if self.chat_id:
            return f"{self.platform.value}:{self.chat_id}"
        return self.platform.value


class DeliveryRouter:
    """
    Routes messages to appropriate destinations.

    Handles the logic of resolving delivery targets and dispatching
    messages to the right platform adapters.
    """

    def __init__(self, config: GatewayConfig, adapters: Dict[Platform, Any] = None):
        """
        Initialize the delivery router.

        Args:
            config: Gateway configuration
            adapters: Dict mapping platforms to their adapter instances
        """
        self.config = config
        self.adapters = adapters or {}
        self.output_dir = Path.home() / ".hermes" / "cron" / "output"

    def resolve_targets(
        self, deliver: Union[str, List[str]], origin: Optional[SessionSource] = None
    ) -> List[DeliveryTarget]:
        """
        Resolve delivery specification to concrete targets.

        Args:
            deliver: Delivery spec - "origin", "telegram", ["local", "discord"], etc.
            origin: The source where the request originated (for "origin" target)

        Returns:
            List of resolved delivery targets
        """
        if isinstance(deliver, str):
            deliver = [deliver]

        targets = []
        seen_platforms = set()

        for target_str in deliver:
            target = DeliveryTarget.parse(target_str, origin)

            # Resolve home channel if needed
            if target.chat_id is None and target.platform != Platform.LOCAL:
                home = self.config.get_home_channel(target.platform)
                if home:
                    target.chat_id = home.chat_id
                else:
                    # No home channel configured, skip this platform
                    continue

            # Deduplicate
            key = (target.platform, target.chat_id)
            if key not in seen_platforms:
                seen_platforms.add(key)
                targets.append(target)

        # Always include local if configured
        if self.config.always_log_local:
            local_key = (Platform.LOCAL, None)
            if local_key not in seen_platforms:
                targets.append(DeliveryTarget(platform=Platform.LOCAL))

        return targets

    async def deliver(
        self,
        content: str,
        targets: List[DeliveryTarget],
        job_id: Optional[str] = None,
        job_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Deliver content to all specified targets.

        Args:
            content: The message/output to deliver
            targets: List of delivery targets
            job_id: Optional job ID (for cron jobs)
            job_name: Optional job name
            metadata: Additional metadata to include

        Returns:
            Dict with delivery results per target
        """
        results = {}

        for target in targets:
            try:
                if target.platform == Platform.LOCAL:
                    result = self._deliver_local(content, job_id, job_name, metadata)
                else:
                    result = await self._deliver_to_platform(target, content, metadata)

                results[target.to_string()] = {"success": True, "result": result}
            except Exception as e:
                results[target.to_string()] = {"success": False, "error": str(e)}

        return results

    def _deliver_local(
        self, content: str, job_id: Optional[str], job_name: Optional[str], metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Save content to local files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if job_id:
            output_path = self.output_dir / job_id / f"{timestamp}.md"
        else:
            output_path = self.output_dir / "misc" / f"{timestamp}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the output document
        lines = []
        if job_name:
            lines.append(f"# {job_name}")
        else:
            lines.append("# Delivery Output")

        lines.append("")
        lines.append(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if job_id:
            lines.append(f"**Job ID:** {job_id}")

        if metadata:
            for key, value in metadata.items():
                lines.append(f"**{key}:** {value}")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(content)

        output_path.write_text("\n".join(lines))

        return {"path": str(output_path), "timestamp": timestamp}

    def _save_full_output(self, content: str, job_id: str) -> Path:
        """Save full cron output to disk and return the file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path.home() / ".hermes" / "cron" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{job_id}_{timestamp}.txt"
        path.write_text(content)
        return path

    async def _deliver_to_platform(
        self, target: DeliveryTarget, content: str, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Deliver content to a messaging platform."""
        adapter = self.adapters.get(target.platform)

        if not adapter:
            raise ValueError(f"No adapter configured for {target.platform.value}")

        if not target.chat_id:
            raise ValueError(f"No chat ID for {target.platform.value} delivery")

        # Guard: truncate oversized cron output to stay within platform limits
        if len(content) > MAX_PLATFORM_OUTPUT:
            job_id = (metadata or {}).get("job_id", "unknown")
            saved_path = self._save_full_output(content, job_id)
            logger.info("Cron output truncated (%d chars) — full output: %s", len(content), saved_path)
            content = content[:TRUNCATED_VISIBLE] + f"\n\n... [truncated, full output saved to {saved_path}]"

        return await adapter.send(target.chat_id, content, metadata=metadata)


def parse_deliver_spec(
    deliver: Optional[Union[str, List[str]]], origin: Optional[SessionSource] = None, default: str = "origin"
) -> Union[str, List[str]]:
    """
    Normalize a delivery specification.

    If None or empty, returns the default.
    """
    if not deliver:
        return default
    return deliver


def build_delivery_context_for_tool(config: GatewayConfig, origin: Optional[SessionSource] = None) -> Dict[str, Any]:
    """
    Build context for the schedule_cronjob tool to understand delivery options.

    This is passed to the tool so it can validate and explain delivery targets.
    """
    connected = config.get_connected_platforms()

    options = {
        "origin": {
            "description": "Back to where this job was created",
            "available": origin is not None,
        },
        "local": {
            "description": "Save to local files only",
            "available": True,
        },
    }

    for platform in connected:
        home = config.get_home_channel(platform)
        options[platform.value] = {
            "description": f"{platform.value.title()} home channel",
            "available": True,
            "home_channel": home.to_dict() if home else None,
        }

    return {
        "origin": origin.to_dict() if origin else None,
        "options": options,
        "always_log_local": config.always_log_local,
    }
