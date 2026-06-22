"""dgx plugin — manage NVIDIA DGX Spark inference endpoints from Hermes Agent.

CLI subcommands: setup, status, models, use, endpoint, pull, rm, ps,
                 push, doctor, watch, formation, nim, node

Agent tools: dgx_gpu_status, dgx_pull_model

Note: there is deliberately no agent-callable "run arbitrary command on the
DGX" tool. Free-form remote shell belongs to the host terminal tool (which
routes through the dangerous-command approval gate); duplicating it here
would let a model run unguarded commands on the GPU host over SSH.
"""

from __future__ import annotations

from plugins.dgx.cli import dgx_command, register_cli as _register_dgx_cli
from plugins.dgx.tools import (
    DGX_GPU_STATUS_SCHEMA,
    DGX_PULL_MODEL_SCHEMA,
    handle_dgx_gpu_status,
    handle_dgx_pull_model,
)

_TOOLS = (
    ("dgx_gpu_status",  DGX_GPU_STATUS_SCHEMA,  handle_dgx_gpu_status,  "🖥️"),
    ("dgx_pull_model",  DGX_PULL_MODEL_SCHEMA,  handle_dgx_pull_model,  "📥"),
)


def _dgx_configured() -> bool:
    """True iff a DGX host is configured.

    Used as the tools' ``check_fn`` so an enabled-but-unconfigured plugin does
    not expose ``dgx_gpu_status`` / ``dgx_pull_model`` to the model — they would
    otherwise try to SSH to ``host=None`` and return a confusing error.
    """
    try:
        from plugins.dgx._dgx_config import load_dgx_config
        dgx = load_dgx_config()
        node = dgx.get("_active_node", dgx)
        return bool(node.get("host"))
    except Exception:
        return False


def register(ctx) -> None:
    ctx.register_cli_command(
        name="dgx",
        help="NVIDIA DGX Spark endpoint management",
        setup_fn=_register_dgx_cli,
        handler_fn=dgx_command,
        description=(
            "Manage local GPU inference endpoints on a DGX Spark. "
            "See: hermes dgx setup"
        ),
    )
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="dgx",
            schema=schema,
            handler=handler,
            emoji=emoji,
            check_fn=_dgx_configured,
        )
