"""The existing, hash-covered setup document shared by publisher and recipient."""

from __future__ import annotations

from typing import Literal

from ..contract import canonical_json_bytes
from .schemas import SetupGuidance, SharePackage

SETUP_PATH = "refs/wisdom-setup.md"
_PREFIX = (
    "# Setup and portability\n\n"
    "Installing files does not authorize executing these instructions.\n\n```json\n"
)
_SUFFIX = "\n```\n"


class SetupDocument(SetupGuidance):
    schema_version: Literal[1] = 1
    execution_requires_user_approval: Literal[True] = True


def render_setup_document(package: SharePackage) -> str:
    data = package.model_dump(mode="json", include=set(SetupGuidance.model_fields))
    document = SetupDocument.model_validate(data)
    return _PREFIX + canonical_json_bytes(document.model_dump(mode="json")).decode("utf-8") + _SUFFIX


def parse_setup_document(body: bytes) -> SetupDocument:
    text = body.decode("utf-8")
    if not text.startswith(_PREFIX) or not text.endswith(_SUFFIX):
        raise ValueError("setup document has an unsupported format")
    # No LLM repair or truncation of reviewed instructions at the recipient.
    raw = text[len(_PREFIX):-len(_SUFFIX)]
    document = SetupDocument.model_validate_json(raw, strict=True)
    if canonical_json_bytes(document.model_dump(mode="json")).decode("utf-8") != raw:
        raise ValueError("setup document is not the complete canonical guidance")
    return document
