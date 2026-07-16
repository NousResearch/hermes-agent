"""desktop-dashboard plugin — デスクトップ常駐ダッシュボード (時計/天気/Xトレンド/AIトレンド)"""

from __future__ import annotations

from .dashboard import (
    DASHBOARD_SCHEMA,
    check_available,
    handle_dashboard,
    handle_slash_dashboard,
    register_cli,
    cli_main,
)


def register(ctx) -> None:
    """Hermes plugin entry point."""
    # ツール登録
    ctx.register_tool(
        name=DASHBOARD_SCHEMA["name"],
        toolset="desktop-dashboard",
        schema=DASHBOARD_SCHEMA,
        handler=lambda args, **kw: handle_dashboard(**args),
        check_fn=check_available,
        description=DASHBOARD_SCHEMA["description"],
    )

    # Slash command
    ctx.register_command(
        "desk-widget",
        handler=handle_slash_dashboard,
        description="デスクトップ常駐ダッシュボード (時計/天気/Xトレンド/AIトレンド) を起動",
    )

    # CLI command
    ctx.register_cli_command(
        name="desk-widget",
        help="デスクトップダッシュボードを起動/停止/状態確認",
        setup_fn=register_cli,
        handler_fn=cli_main,
        description="デスクトップ常駐ダッシュボード (時計/天気/Xトレンド/AIトレンド) を起動/停止/状態確認",
    )