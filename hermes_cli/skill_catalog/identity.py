"""Literal registration identity, shared by the original object and host descriptor."""

import hashlib
import inspect
import re
from dataclasses import asdict, dataclass

from hermes_cli.skill_catalog.provider import SkillCatalogProvider

_NAME = re.compile(r"[a-z][a-z0-9-]{0,47}", re.ASCII)


def plain_text(value, limit: int, *, empty: bool = False) -> str:
    if type(value) is not str or not (0 if empty else 1) <= len(value) <= limit:
        raise ValueError("Invalid catalog text.")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ValueError("Invalid UTF-8 text.") from None
    if any(ord(c) < 32 or 127 <= ord(c) < 160 for c in value):
        raise ValueError("Control characters are not permitted.")
    return value


def source_id(plugin_key: str, name: str) -> str:
    plain_text(plugin_key, 1024)
    if len(plugin_key.encode("utf-8")) > 1024:
        raise ValueError("Plugin key is too long.")
    if type(name) is not str or not _NAME.fullmatch(name):
        raise ValueError("Catalog name must be an exact lowercase identifier.")
    return f"plugin.{hashlib.sha256(plugin_key.encode('utf-8')).hexdigest()}.{name}"


@dataclass(frozen=True)
class Descriptor:
    schema_version: int
    name: str
    label: str
    implementation_revision: str

    @classmethod
    def from_wire(cls, value: dict):
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Invalid catalog descriptor.")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("Unsupported catalog descriptor.")
        source_id("validation", value["name"])
        plain_text(value["label"], 80)
        plain_text(value["implementation_revision"], 256)
        return cls(**value)

    def to_wire(self) -> dict:
        return asdict(self)


def describe(provider) -> Descriptor:
    cls = type(provider)
    if type(cls) is not type or not issubclass(cls, SkillCatalogProvider):
        raise ValueError("Catalog must subclass SkillCatalogProvider without a custom metaclass.")
    if cls.__getattribute__ is not object.__getattribute__ or any(
        "__getattr__" in vars(base) or "__class__" in vars(base) for base in cls.__mro__[:-1]
    ):
        raise ValueError("Catalog must not override attribute access.")
    return Descriptor.from_wire(dict(schema_version=1, **{
        key: inspect.getattr_static(provider, key, None)
        for key in ("name", "label", "implementation_revision")
    }))
