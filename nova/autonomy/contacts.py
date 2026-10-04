"""The tenant's contact list: who the business says its customers and its own people are.

``contacts.yaml`` in the bundle, edited in the Control Centre (Settings → Contacts)::

    internal:  [telegram:1001, slack:C0TEAM]   # our own staff and team channels
    customers: [telegram:42, telegram:77]       # known customers

Each entry is a send target as the runtime writes one, ``platform:chat``. Triage answers
the recipient question from this list and the gateway's channel directory, so a message
only counts as going to a customer when the business has said so — never because the agent
said so, and never because the tone of the message sounded like it.

The list is compiled into the policy of agents whose triage asks about the recipient, so a
change takes effect on the next apply, which the editor runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from nova.errors import SpecError

CONTACTS_FILE = "contacts.yaml"
LISTS = ("internal", "customers")
#: Enough for a small business's customers; a larger book belongs in a CRM lookup.
MAX_ENTRIES = 20000

_ENTRY = re.compile(r"^([A-Za-z][A-Za-z0-9_-]{0,31}):(\S{1,200})$")


def normalize(entry: Any, *, where: str) -> str:
    """``platform:chat`` with the platform lowercased; refuses anything else with a sentence."""
    match = _ENTRY.match(str(entry).strip()) if isinstance(entry, (str, int)) else None
    if match is None:
        raise SpecError(f"{where}: {entry!r} is not a send target such as telegram:12345 or slack:C0TEAM",
                        field=where)
    return f"{match.group(1).lower()}:{match.group(2)}"


@dataclass(frozen=True)
class Contacts:
    internal: tuple[str, ...] = ()
    customers: tuple[str, ...] = ()

    @classmethod
    def parse(cls, data: Any, *, source: str = CONTACTS_FILE) -> "Contacts":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise SpecError(f"{source} must be a mapping with internal and customers lists")
        unknown = sorted(set(data) - set(LISTS))
        if unknown:
            raise SpecError(f"{source} has unknown keys {unknown}; expected {list(LISTS)}")
        lists: dict[str, tuple[str, ...]] = {}
        for name in LISTS:
            raw = data.get(name) or []
            if not isinstance(raw, list):
                raise SpecError(f"{source}: {name} must be a list", field=name)
            entries = [normalize(item, where=f"{name}[{i}]") for i, item in enumerate(raw)]
            lists[name] = tuple(dict.fromkeys(entries))  # duplicates collapse, order kept
        both = sorted(set(lists["internal"]) & set(lists["customers"]))
        if both:
            raise SpecError(f"{source}: {', '.join(both[:5])} listed as both internal and customer; "
                            "pick one")
        if sum(len(v) for v in lists.values()) > MAX_ENTRIES:
            raise SpecError(f"{source} holds more than {MAX_ENTRIES} entries; use a CRM lookup instead")
        return cls(**lists)

    @classmethod
    def from_lines(cls, internal: Iterable[str], customers: Iterable[str]) -> "Contacts":
        """From the editor: one entry per line, blanks and ``#`` comments ignored."""
        def clean(lines: Iterable[str]) -> list[str]:
            return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

        return cls.parse({"internal": clean(internal), "customers": clean(customers)})

    def to_dict(self) -> dict[str, list[str]]:
        return {"internal": list(self.internal), "customers": list(self.customers)}

    def compiled(self) -> dict[str, list[str]]:
        return {"internal": sorted(self.internal), "customers": sorted(self.customers)}


def load_contacts(root: Path) -> Contacts:
    """The bundle's contact list. Empty when absent; a malformed file is refused at load."""
    path = Path(root) / CONTACTS_FILE
    if not path.is_file():
        return Contacts()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SpecError(f"could not be read: {exc}", source=path) from exc
    try:
        return Contacts.parse(data)
    except SpecError as exc:
        raise SpecError(str(exc), source=path) from None


def write_contacts(root: Path, contacts: Contacts):
    """Replace the bundle's contact list, validated, all or nothing. ``(bundle, changed files)``."""
    from nova.spec.writer import edit

    def mutate(staged) -> None:
        staged.write_yaml(CONTACTS_FILE, contacts.to_dict())

    return edit(root, mutate)
