"""PEP 610"""

from __future__ import annotations

import json
from typing import Any

from pip._vendor.packaging.direct_url import (
    ArchiveInfo,
    DirectUrlValidationError,
    DirInfo,
    VcsInfo,
)
from pip._vendor.packaging.direct_url import (
    DirectUrl as PackagingDirectUrl,
)

__all__ = [
    "ArchiveInfo",
    "DirInfo",
    "DirectUrl",
    "DirectUrlValidationError",
    "DIRECT_URL_METADATA_NAME",
    "VcsInfo",
]

DIRECT_URL_METADATA_NAME = "direct_url.json"


class DirectUrl(PackagingDirectUrl):
    def to_dict_compat(self) -> dict[str, Any]:
        return dict(super().to_dict(generate_legacy_hash=True))

    @classmethod
    def from_json(cls, s: str) -> DirectUrl:
        return cls.from_dict(json.loads(s))

    def to_json(self) -> str:
        return json.dumps(self.to_dict_compat(), sort_keys=True)

    def is_local_editable(self) -> bool:
        return bool(self.dir_info and self.dir_info.editable)
