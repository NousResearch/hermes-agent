"""Friendly @mention tags for hosted-room members (#89720).

A room member is canonical under its immutable ``handle`` (a technical profile id such as
``default-device-a1b2c3d4``). The renamed, human-facing tag is the ``display_name`` frozen in
the room descriptor — the same name the Desktop derives its autocomplete tags from
(``apps/desktop/src/plugins/hermes-bots/data.ts::mentionNameForms`` and
``tools/bot_mode_probe.alias_forms``): the slugified form (``"Dr. Foo"`` → ``dr-foo``, what
autocomplete inserts) and the collapsed form (``drfoo``).

Resolution rules, shared by every mention consumer (Discussion policy, lead routing, the auto
router):

* the canonical handle always wins, exactly as before — a friendly name never captures another
  member's handle;
* an alias shared by two members is ambiguous: it addresses neither and is reported, never
  silently resolved to one member nor turned into an ``@all`` broadcast;
* reserved tags (``all``, ``everyone``, ``user``, ``default``, ``hermes``) are never derived aliases,
  but one may still be a member's canonical handle (the desktop seats a local Bot as ``default``);
* an unknown tag keeps whatever each caller already did with it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Advertised by both ``groups.capabilities`` surfaces: tags resolve from the frozen display name.
MENTION_ALIAS_FEATURE = "room_mention_aliases_v1"

#: A tag the group composer reserves for a command or a fixed addressee; never a member alias.
RESERVED_TAGS = frozenset({"all", "everyone", "user", "default", "hermes"})

_MENTIONS_ALL = frozenset({"all", "everyone"})

#: One ``@tag`` in room text; the same charset the Desktop inserts.
MENTION_RE = re.compile(r"@([A-Za-z0-9][A-Za-z0-9._:-]*)", re.IGNORECASE)

_TAG_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_STRIP_RE = re.compile(r"[^a-z0-9_-]+")


def mention_tags_for_name(value: Any) -> tuple[str, ...]:
    """Tag forms of one friendly name, in autocomplete order (slug, then collapsed)."""
    name = str(value or "").strip().lower()
    if not name:
        return ()
    slug = _STRIP_RE.sub("-", name).strip("-")
    collapsed = _STRIP_RE.sub("", name)
    return tuple(
        form for form in dict.fromkeys((slug, collapsed))
        if form and _TAG_RE.fullmatch(form) and form not in RESERVED_TAGS)


def _field(member: Any, name: str) -> str:
    return str(getattr(member, name, "") or "") if not isinstance(member, Mapping) else str(member.get(name) or "")


def _identity(member: Any) -> str:
    """A member's immutable roster identity: its member id, else its canonical handle.

    Prompt rosters (the auto router) may omit ``member_id``; the handle is unique and immutable
    there, so it distinguishes two members that froze the same display name.
    """
    return (_field(member, "member_id") or _field(member, "handle")).casefold()


def member_mention_tags(member: Any) -> tuple[str, ...]:
    """Mention forms of *member*'s frozen display name (empty when it has none)."""
    return mention_tags_for_name(_field(member, "display_name"))


def preferred_mention_tag(member: Any, members: Sequence[Any] | None = None) -> str | None:
    """The tag autocomplete would insert for *member*, or None when no form resolves to it alone.

    *members* is the full frozen roster the prompt describes. A form another member's handle (or a
    second member's name) claims is never advertised: the model must not be told to use a tag that
    resolves to somebody else or to nobody.
    """
    forms = member_mention_tags(member)
    if not forms:
        return None
    roster = tuple(members) if members is not None else (member,)
    index = MemberMentions.build(roster)
    return next((form for form in forms if index.aliases.get(form) is member), None)


def roster_mention_line(member: Any, *, name: str | None = None,
                        members: Sequence[Any] | None = None) -> str:
    """One model-facing roster line: ``- @handle (Friendly Name, also @tag)``.

    The canonical handle stays first; the frozen friendly tag is named only when it resolves to
    exactly this member across the full *members* roster, so a duplicated name or a tag spelling a
    peer's handle is never advertised.
    """
    handle = _field(member, "handle")
    label = (name if name is not None else _field(member, "display_name")).strip()
    roster = tuple(members) if members is not None else (member,)
    index = MemberMentions.build(roster)
    tag = next((form for form in mention_tags_for_name(label) if index.aliases.get(form) is member), None)
    if tag and tag != handle.casefold():
        detail = f"{label}, also @{tag}" if label else f"also @{tag}"
    else:
        detail = label if label and label != handle else ""
    return f"- @{handle}" + (f" ({detail})" if detail else "")


@dataclass(frozen=True)
class MemberMentions:
    """One roster's tag resolution: canonical handles first, then unique display-name aliases."""

    canonical: Mapping[str, Any]
    aliases: Mapping[str, Any]
    ambiguous: Mapping[str, tuple[str, ...]]

    @classmethod
    def build(cls, members: Iterable[Any]) -> MemberMentions:
        roster = tuple(members)
        canonical = {_field(member, "handle").casefold(): member for member in roster}
        owners: dict[str, list[Any]] = {}
        for member in roster:
            for form in member_mention_tags(member):
                owners.setdefault(form, []).append(member)
        aliases: dict[str, Any] = {}
        ambiguous: dict[str, tuple[str, ...]] = {}
        for form, candidates in owners.items():
            if form in canonical:
                continue  # the handle keeps its tag; the alias never steals it
            ids = tuple(dict.fromkeys(_identity(member) for member in candidates))
            if len(ids) == 1:
                aliases[form] = candidates[0]
            else:
                ambiguous[form] = ids
        return cls(canonical, aliases, ambiguous)

    def member(self, tag: Any) -> Any | None:
        """The member *tag* addresses, or None (unknown, ambiguous, or reserved)."""
        key = str(tag or "").strip().lstrip("@").casefold()
        if not key:
            return None
        # A canonical handle always wins, even when the word is reserved for aliases: the desktop
        # seats a local default Bot under ``default``, and a roster may freeze ``hermes``/``user``.
        if (member := self.canonical.get(key)) is not None:
            return member
        if key in RESERVED_TAGS:
            return None
        return self.aliases.get(key)

    def is_ambiguous(self, tag: Any) -> bool:
        return str(tag or "").strip().lstrip("@").casefold() in self.ambiguous

    def tags(self) -> dict[str, Any]:
        """``tag -> member`` for canonical handles and unique aliases (ambiguous aliases dropped)."""
        return {**self.aliases, **self.canonical}


@dataclass(frozen=True)
class MentionResolution:
    """Members a set of texts addresses, plus the ambiguous aliases it recognized."""

    members: tuple[Any, ...]
    ambiguous: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.members)


def mention_index(members: Sequence[Any]) -> MemberMentions:
    return MemberMentions.build(members)


def resolve_mention_tags(texts: Iterable[Any], members: Sequence[Any], *, default_all: bool = True) -> MentionResolution:
    """Resolve plain ``@tag`` mentions against one frozen roster.

    ``@all``/``@everyone`` still address everyone. With no addressing tag at all,
    ``default_all`` keeps the caller's old "everyone" default; a recognized-but-ambiguous alias
    is not "no mention", so it never falls back to a broadcast.
    """
    index = MemberMentions.build(members)
    mentioned: dict[str, Any] = {}
    ambiguous: set[str] = set()
    everyone = False
    for text in texts:
        for match in MENTION_RE.finditer(str(text or "")):
            tag = match.group(1).casefold()
            if tag in _MENTIONS_ALL:
                everyone = True
            elif (member := index.member(tag)) is not None:
                mentioned[_field(member, "member_id")] = member
            elif index.is_ambiguous(tag):
                ambiguous.add(tag)
    if everyone:
        return MentionResolution(tuple(members), ())
    if not mentioned and not ambiguous and default_all:
        return MentionResolution(tuple(members), ())
    picked = tuple(member for member in members if _field(member, "member_id") in mentioned)
    return MentionResolution(picked, tuple(sorted(ambiguous)))


def has_addressing_mention(texts: Iterable[Any], members: Sequence[Any]) -> bool:
    """Whether the texts explicitly address somebody (canonical, alias, ambiguous, or @all)."""
    index = MemberMentions.build(members)
    for text in texts:
        for match in MENTION_RE.finditer(str(text or "")):
            tag = match.group(1).casefold()
            if tag in _MENTIONS_ALL or index.member(tag) is not None or index.is_ambiguous(tag):
                return True
    return False


__all__ = [
    "MENTION_ALIAS_FEATURE", "MENTION_RE", "RESERVED_TAGS", "MemberMentions", "MentionResolution",
    "has_addressing_mention", "member_mention_tags", "mention_index", "mention_tags_for_name",
    "preferred_mention_tag", "resolve_mention_tags", "roster_mention_line"]
