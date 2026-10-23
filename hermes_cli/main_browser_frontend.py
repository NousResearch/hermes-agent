"""Select the shared Desktop browser frontend for the existing Hermes server.

The server retains all authentication, profiles, APIs and WebSocket ownership.
No second listener, backend child, credential store or proxy is introduced.
"""
from __future__ import annotations

import os
from pathlib import Path


def configure_browser_frontend(args, project_root: Path) -> None:
    """Build (unless explicitly skipped) and select the Desktop renderer bundle."""
    from hermes_cli.source_build import (
        prepare_source_dependencies, run_source_script, source_build_env, source_frontends,
    )
    from hermes_cli.web_build_limits import apply_web_build_limits

    dist = Path(os.environ.get("HERMES_WEB_DIST", project_root / "apps/desktop/dist/browser")).expanduser()
    if "HERMES_WEB_DIST" not in os.environ and not getattr(args, "skip_build", False):
        env = source_build_env()
        apply_web_build_limits(env)
        prepare_source_dependencies(project_root, (*source_frontends(project_root), "apps/desktop"), env=env)
        run_source_script(project_root, "scripts/build/desktop-browser.mjs", env=env,
                          label="Building the Desktop browser UI")
    if not (dist / "index.html").is_file():
        raise SystemExit(
            f"Desktop browser UI not built at {dist}. "
            "Run npm run build:browser --workspace apps/desktop, then restart with --skip-build.")
    # Existing internal launch hand-off; no new environment/config namespace.
    os.environ["HERMES_WEB_DIST"] = str(dist)
    os.environ.pop("HERMES_SERVE_HEADLESS", None)
