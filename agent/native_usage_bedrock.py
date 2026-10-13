"""Validate Converse JSON token counts before botocore's integer parser."""

from collections.abc import Mapping
import json
from typing import Any

from agent.native_usage import NativeUsageCounts


def validate_bedrock_usage_response(
    response_dict: Mapping[str, Any], **kwargs: Any
) -> None:
    if response_dict["status_code"] >= 300:
        return
    try:
        data = json.loads(response_dict["body"])
    except (ValueError, UnicodeDecodeError):
        return
    if isinstance(data, dict) and data.get("usage") is not None:
        NativeUsageCounts.read(data["usage"], kind="bedrock")
