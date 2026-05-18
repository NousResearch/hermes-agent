"""Google Drive artifact pipeline plugin.

Registers only operator-facing CLI surfaces. The pipeline reuses the existing
google-workspace Drive primitives and adds orchestration on top.
"""

from __future__ import annotations

from plugins.google_drive_pipeline.cli import (
    google_drive_pipeline_command,
    register_cli,
)


def register(ctx) -> None:
    ctx.register_cli_command(
        name="google-drive-pipeline",
        help="Inspect and operate the Google Drive artifact pipeline",
        setup_fn=register_cli,
        handler_fn=google_drive_pipeline_command,
        description=(
            "Operator CLI for publishing artifacts into Google Drive. "
            "Resolves target folders, handles duplicate naming policies, "
            "applies sharing, and returns canonical Drive links."
        ),
    )
