import pytest

from hermes_cli.providers import normalize_provider, resolve_provider_full


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("veniceai", "venice"),
        ("venice-ai", "venice"),
        ("venice.ai", "venice"),
    ],
)
def test_normalize_provider_venice_aliases(alias, expected):
    assert normalize_provider(alias) == expected


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("veniceai", "venice"),
        ("venice-ai", "venice"),
        ("venice.ai", "venice"),
    ],
)
def test_resolve_provider_full_accepts_venice_aliases(alias, expected):
    resolved = resolve_provider_full(alias)

    assert resolved is not None
    assert resolved.id == expected
