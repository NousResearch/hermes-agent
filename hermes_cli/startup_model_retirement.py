"""Best-effort startup notices for configured retired models."""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def warn_retired_models() -> None:
    """Print model-independent retirement guidance without blocking startup."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        logger.debug("Could not load config for model-retirement warning", exc_info=True)
        return

    try:
        from hermes_cli.xai_retirement import (
            MIGRATION_GUIDE_URL,
            RETIREMENT_DATE,
            find_retired_xai_refs,
            format_issue,
        )

        retired_xai_refs = find_retired_xai_refs(config)
        if retired_xai_refs:
            sys.stderr.write(
                f"\033[33m⚠ xAI retires {len(retired_xai_refs)} model(s) "
                f"in your config on {RETIREMENT_DATE}:\033[0m\n"
            )
            for ref in retired_xai_refs:
                sys.stderr.write(f"  \033[33m⚠\033[0m {format_issue(ref)}\n")
            sys.stderr.write(f"  \033[2mMigration guide: {MIGRATION_GUIDE_URL}\033[0m\n")
            sys.stderr.write("  \033[2mRun 'hermes doctor' for details.\033[0m\n\n")
    except Exception:
        logger.debug("Could not check xAI model retirements", exc_info=True)

    try:
        from hermes_cli.model_retirement import (
            find_retired_primary_model,
            format_retirement_notice,
        )

        issue = find_retired_primary_model(config)
        if issue is not None:
            sys.stderr.write(f"\033[33m⚠ {format_retirement_notice(issue)}\033[0m\n\n")
    except Exception:
        logger.debug("Could not check primary-model retirements", exc_info=True)


def log_retired_primary_model(target_logger: logging.Logger) -> None:
    """Write a model-independent retirement warning to an initialized service log."""
    try:
        from hermes_cli.config import load_config
        from hermes_cli.model_retirement import (
            find_retired_primary_model,
            format_retirement_notice,
        )

        issue = find_retired_primary_model(load_config())
        if issue is not None:
            target_logger.warning(format_retirement_notice(issue))
    except Exception:
        target_logger.debug("Could not check primary-model retirement", exc_info=True)
