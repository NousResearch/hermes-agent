"""World Monitor OSINT Hermes plugin."""

from __future__ import annotations

import json

from . import core
from .cli import register_cli, worldmonitor_osint_command
from . import dev_server


def _dev_json_handler(fn):
    def handler(values=None, **kwargs):
        payload = values if isinstance(values, dict) else {}
        payload.update(kwargs)
        return json.dumps(fn(payload), ensure_ascii=False, indent=2, default=str)

    return handler


_dev_status_handler = _dev_json_handler(lambda _v: dev_server.dev_status())
_dev_start_handler = _dev_json_handler(dev_server.start_dev)
_dev_stop_handler = _dev_json_handler(lambda v: dev_server.stop_dev(pid=v.get("pid")))

_TOOLS = (
    ("worldmonitor_status", core.STATUS_SCHEMA, core.handle_status, "🌐"),
    ("worldmonitor_snapshot", core.SNAPSHOT_SCHEMA, core.handle_snapshot, "📡"),
    ("worldmonitor_free_crawl", core.FREE_CRAWL_SCHEMA, core.handle_free_crawl, "🕸️"),
    ("worldmonitor_country_brief", core.COUNTRY_BRIEF_SCHEMA, core.handle_country_brief, "🗺️"),
    ("worldmonitor_fusion_report", core.FUSION_SCHEMA, core.handle_fusion_report, "🧬"),
    ("worldmonitor_dev_status", dev_server.DEV_STATUS_SCHEMA, _dev_status_handler, "🖥️"),
    ("worldmonitor_dev_start", dev_server.DEV_START_SCHEMA, _dev_start_handler, "▶️"),
    ("worldmonitor_dev_stop", dev_server.DEV_STOP_SCHEMA, _dev_stop_handler, "⏹️"),
)


def register(ctx) -> None:
    """Register World Monitor OSINT tools, slash command, and CLI."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="worldmonitor_osint",
            schema=schema,
            handler=handler,
            check_fn=core.check_available,
            emoji=emoji,
        )

    ctx.register_command(
        "worldmonitor-osint",
        handler=core.handle_slash,
        description="Real-time OSINT via World Monitor + Shinka fusion.",
        args_hint="[status|snapshot|fusion]",
    )
    ctx.register_cli_command(
        name="worldmonitor-osint",
        help="World Monitor real-time OSINT and fusion reports",
        setup_fn=register_cli,
        handler_fn=worldmonitor_osint_command,
        description=(
            "Bridge to koala73/worldmonitor API for real-time risk/news snapshots "
            "and ShinkaEvolve MILSPEC fusion reports with e-Gov primary sources."
        ),
    )
