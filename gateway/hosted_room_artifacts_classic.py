"""Original #104198 classic custody identity; no producer/runtime activation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from gateway.hosted_room_artifacts import RoomArtifactError

def encoded(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def identifier(value) -> str:
    if not isinstance(value, str) or not value or len(value) > 160 or any(ord(c) < 32 for c in value):
        raise RoomArtifactError("Invalid classic export identity")
    return value


@dataclass(frozen=True)
class ClassicExportScope:
    export_id: str
    execution_generation: int

    def as_mapping(self):
        return {"kind": "classic", "export_id": self.export_id,
                "execution_generation": self.execution_generation}

    @property
    def key(self):
        return hashlib.sha256(encoded(self.as_mapping()).encode()).hexdigest()

    @property
    def lineage_json(self):
        return encoded({"kind": "classic", "export_id": self.export_id})

    @property
    def lineage_key(self):
        return hashlib.sha256(self.lineage_json.encode()).hexdigest()
