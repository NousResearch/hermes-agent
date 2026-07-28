"""Find masters sharing a theme part in the way PowerPoint refuses to open.

Reports only; the fix is to move <p:notesMasterIdLst> back to directly after
<p:sldIdLst> in ppt/presentation.xml.
"""


from __future__ import annotations

import posixpath
import re
from typing import Mapping

from . import part_text

THEME_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme"

_MASTER_RE = re.compile(
    r"^ppt/(?P<group>slideMasters|notesMasters|handoutMasters)/"
    r"(?:slide|notes|handout)Master(?P<num>\d+)\.xml$"
)
_GROUP_ORDER = {"slideMasters": 0, "notesMasters": 1, "handoutMasters": 2}

_RELATIONSHIP_RE = re.compile(
    r"<Relationship\b[^>]*?(?:/>|>.*?</Relationship\s*>)", re.DOTALL
)


def _sort_key(name: str) -> tuple[int, int]:
    m = _MASTER_RE.match(name)
    assert m is not None
    return (_GROUP_ORDER[m.group("group")], int(m.group("num")))


def _rels_path(part: str) -> str:
    directory, base = posixpath.split(part)
    return f"{directory}/_rels/{base}.rels"


def _resolve(rels_path: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    part_dir = posixpath.dirname(posixpath.dirname(rels_path))
    return posixpath.normpath(posixpath.join(part_dir, target))


def _theme_rel(files: Mapping[str, bytes], master: str):
    rels_path = _rels_path(master)
    rels = files.get(rels_path)
    if rels is None:
        return None
    for element in _RELATIONSHIP_RE.findall(part_text(rels)):
        if f'Type="{THEME_REL_TYPE}"' not in element:
            continue
        target = re.search(r'\bTarget="([^"]+)"', element)
        if target is None:
            continue
        return rels_path, element, _resolve(rels_path, target.group(1))
    return None


def _masters(files: Mapping[str, bytes]) -> list[str]:
    return sorted((n for n in files if _MASTER_RE.match(n)), key=_sort_key)


_PRESENTATION = "ppt/presentation.xml"
_NOTES_MASTERS = "ppt/notesMasters/"
_IGNORABLE_RE = re.compile(r"<!--.*?-->|<\?.*?\?>", re.DOTALL)
_AFTER_SLDIDLST_RE = re.compile(
    r"<p:sldIdLst\b(?:[^>]*/>|[^>]*>.*?</p:sldIdLst\s*>)\s*(<[^>\s/]+)", re.DOTALL
)


def _notes_master_share_is_inert(files: Mapping[str, bytes]) -> bool:
    data = files.get(_PRESENTATION)
    if data is None:
        return False
    match = _AFTER_SLDIDLST_RE.search(_IGNORABLE_RE.sub("", part_text(data)))
    return match is not None and match.group(1) == "<p:notesMasterIdLst"


def _shares(files: Mapping[str, bytes]):
    owner: dict[str, str] = {}
    for master in _masters(files):
        found = _theme_rel(files, master)
        if found is None:
            continue
        rels_path, element, theme = found
        if theme not in files:
            continue  
        if theme in owner:
            yield master, rels_path, element, theme, owner[theme]
        else:
            owner[theme] = master


def _is_inert(master: str, inert_notes: bool) -> bool:
    return inert_notes and master.startswith(_NOTES_MASTERS)


def find_shared_master_themes(files: Mapping[str, bytes]) -> list[str]:
    return [
        f"{master} shares {theme} with {first}"
        for master, _, _, theme, first in _shares(files)
    ]


def live_shared_master_themes(files: Mapping[str, bytes]) -> list[str]:
    inert_notes = _notes_master_share_is_inert(files)
    return [
        f"{master} shares {theme} with {first}"
        for master, _, _, theme, first in _shares(files)
        if not _is_inert(master, inert_notes)
    ]
