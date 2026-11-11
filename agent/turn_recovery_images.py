"""Image-size recovery rung shared by the classified turn recovery chain."""
from __future__ import annotations

import logging
from typing import Any

from agent.error_classifier import FailoverReason
from agent.turn_retry_state import TurnRetryState

logger = logging.getLogger(__name__)


def _recover_oversized_images(agent: Any, api_error: Exception, classified: Any,
                              retry: TurnRetryState, api_messages: Any) -> bool:
    from agent.turn_recovery import _image_error_max_dimension, _vlines

    if classified.reason != FailoverReason.image_too_large or retry.image_shrink_retry_attempted:
        return False
    retry.image_shrink_retry_attempted = True
    if agent._try_shrink_image_parts_in_messages(
        api_messages, max_dimension=_image_error_max_dimension(api_error) or 8000
    ):
        _vlines(agent, "📐 Image(s) exceeded provider size limit — shrank and retrying...")
        return True
    logger.info(
        "image-shrink recovery: no data-URL image parts found "
        "or shrink didn't reduce size; surfacing original error."
    )
    return False
