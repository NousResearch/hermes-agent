from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Entry:
    entry_id: str
    topic: str
    date_added: str
    entry_type: str = "text"
    media_kind: str = ""
    media_source: str = ""
    source: str = ""
    description: str = ""
    creator: str = ""
    published_at: str = ""
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    body: str = ""
