"""Write ``~/.hermes/people/<slug>.md`` files from the People DB.

Schema intentionally converges with notes-extract People entries (#32720):
YAML front matter + Facts / Commitments / Topics sections. Phase 1 fills a
**Messages** summary block; notes-extract fences remain for sibling skill.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from plugins.people.store import PeopleMessageStore


def default_people_dir() -> Path:
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(hermes_home) / "people"


def write_people_markdown(
    store: "PeopleMessageStore",
    *,
    people_dir: Optional[os.PathLike | str] = None,
    message_preview: int = 12,
) -> int:
    """Rewrite People markdown files for all people rows. Returns count written."""
    out_dir = Path(people_dir) if people_dir else default_people_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for person in store.list_people():
        person_id = person["person_id"]
        slug = person["slug"] or person_id
        name = person["display_name"] or slug
        updated = time.strftime("%Y-%m-%d", time.localtime(person["updated_at"] or time.time()))
        msgs = store.messages_for_person(person_id, limit=message_preview)
        lines = [
            "---",
            "type: person",
            f"id: {person_id}",
            f"name: {name}",
            "aliases: []",
            f"updated: {updated}",
            "---",
            "",
            "## Facts",
            "<!-- notes-extract:begin facts -->",
            "<!-- notes-extract:end facts -->",
            "",
            "## Commitments",
            "<!-- notes-extract:begin commitments -->",
            "<!-- notes-extract:end commitments -->",
            "",
            "## Topics",
            "<!-- notes-extract:begin topics -->",
            "<!-- notes-extract:end topics -->",
            "",
            "## Messages",
            "<!-- people-store:begin messages -->",
        ]
        if not msgs:
            lines.append("_No messages ingested yet._")
        else:
            for m in reversed(list(msgs)):
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(m["ts"]))
                direction = m["direction"] or "?"
                channel = m["channel"] or "?"
                body = (m["body"] or "").replace("\n", " ").strip()
                if len(body) > 200:
                    body = body[:197] + "..."
                lines.append(f"- [{ts}] ({channel}/{direction}) {body}")
        lines.append("<!-- people-store:end messages -->")
        lines.append("")
        path = out_dir / f"{slug}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
    return written
