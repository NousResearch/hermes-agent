from dataclasses import dataclass
from typing import Any
from typing import cast

import pytest

from hermes_cli.kanban_card_classes import BlindCard, ResolverWiringError


@dataclass(frozen=True)
class _Ceiling:
    enforced: bool = True


class _WiredResolver:
    def __init__(self, wired=True, ceiling=None):
        self.wired = wired
        self.ceiling = _Ceiling() if ceiling is None else ceiling
        self.calls = []

    def verify_runtime_wiring(self):
        self.calls.append("probe")
        return self.wired

    def resolve_ceiling(self, **kwargs):
        self.calls.append(kwargs)
        return self.ceiling


def test_single_blind_operation_uses_verified_enforced_ceiling():
    resolver = _WiredResolver()
    card = BlindCard("implementation", {"read": ["src/**"], "write": ["src/**"]}, resolver)

    assert card.execute(lambda ceiling: ceiling.enforced) is True
    assert resolver.calls[0] == "probe"
    assert resolver.calls[-1]["card_class"] == "single_blind"
    assert resolver.calls[-1]["role"] == "junior-dev"


def test_construction_fails_closed_when_runtime_wiring_is_not_verified():
    resolver = _WiredResolver(wired=False)

    with pytest.raises(ResolverWiringError, match="not enforced end-to-end"):
        BlindCard("implementation", {}, resolver)

    assert resolver.calls == ["probe"]


def test_importable_shape_without_runtime_probe_is_rejected():
    class ImportableOnly:
        def resolve_ceiling(self, **kwargs):
            return _Ceiling()

    with pytest.raises(ResolverWiringError, match="importability is insufficient"):
        BlindCard("implementation", {}, cast(Any, ImportableOnly()))


def test_execution_rechecks_wiring_and_never_falls_back():
    resolver = _WiredResolver()
    card = BlindCard("implementation", {}, resolver)
    resolver.wired = False
    invoked = False

    def operation(_ceiling):
        nonlocal invoked
        invoked = True

    with pytest.raises(ResolverWiringError):
        card.execute(operation)
    assert invoked is False


def test_unenforced_resolution_is_rejected():
    resolver = _WiredResolver(ceiling=_Ceiling(enforced=False))
    card = BlindCard("implementation", {}, resolver)

    with pytest.raises(ResolverWiringError, match="without runtime enforcement"):
        card.execute(lambda ceiling: ceiling)
