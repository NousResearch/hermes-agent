"""Call-site API for process-level shared-metrics facts.

Setup completion, slash commands, compression, model switches, fallbacks and extension installs
happen outside any single tool or model call, so their producers call these functions directly.
Every function takes RAW values (normalization lives in shared_metrics_fields), is a no-op unless
shared metrics are enabled, and never raises into the caller.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from . import shared_metrics_contract as contract
from . import shared_metrics_fields as fields_

logger = logging.getLogger(__name__)


def _emit(mark: str, build: Callable[..., dict[str, str] | None], **raw: Any) -> None:
    try:
        from .relay_shared_metrics import enabled, record_process_mark

        if not enabled():
            return
        data = build(**raw)
        if data is not None:
            record_process_mark(mark, data)
    except Exception:
        logger.debug("Shared-metrics %s not recorded", mark, exc_info=True)


def record_setup_completed(*, surface: str, provider: str | None) -> None:
    _emit(contract.SETUP_COMPLETED_MARK, fields_.setup_completed_fields, surface=surface, provider=provider)


def record_slash_command(*, command: str, surface: str) -> None:
    _emit(contract.SLASH_COMMAND_MARK, fields_.slash_command_fields, command=command, surface=surface)


def record_compression(
    *, trigger: str, outcome: str, tokens_before: int | None, context_length: int | None
) -> None:
    _emit(
        contract.COMPRESSION_MARK, fields_.compression_fields, trigger=trigger, outcome=outcome,
        tokens_before=tokens_before, context_length=context_length,
    )


def record_model_switch(*, from_provider: str | None, to_provider: str | None, surface: str) -> None:
    _emit(
        contract.MODEL_SWITCH_MARK, fields_.model_switch_fields,
        from_provider=from_provider, to_provider=to_provider, surface=surface,
    )


def record_fallback(*, from_provider: str | None, to_provider: str | None, reason: Any) -> None:
    _emit(
        contract.FALLBACK_MARK, fields_.fallback_fields,
        from_provider=from_provider, to_provider=to_provider, reason=reason,
    )


def record_extension_install(*, kind: str, source: str, name: str | None, outcome: str) -> None:
    _emit(
        contract.EXTENSION_INSTALL_MARK, fields_.extension_install_fields,
        kind=kind, source=source, name=name, outcome=outcome,
    )
