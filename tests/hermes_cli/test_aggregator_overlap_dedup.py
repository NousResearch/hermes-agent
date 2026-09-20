"""Aggregator-overlap dedup in the model picker (#45954 / #47077 / #56145).

A user's LOCAL endpoint (litellm-proxy on localhost) is the more-specific deployment for the
models it serves, so official aggregator rows drop those overlapping names (#45954). A REMOTE
user-defined endpoint is a deliberate second route — two aggregating routes reselling the same
open-weight model must both stay discoverable, or the official row is gutted to an unrecognizable
stub (#56145: a custom API's 447-model catalog hid ~370 of Kilo's 382 models). And whatever the
dedup hides, ``total_models`` keeps reporting the provider's real catalog size — the picker sorts
rows and labels them with it.
"""
import pytest

from hermes_cli.inventory import _strip_aggregator_overlaps, _user_endpoint_is_local


def _aggregator_row(models, *, total=None, slug="openrouter"):
    return {"slug": slug, "name": slug, "is_current": False, "is_user_defined": False,
            "models": list(models), "total_models": total if total is not None else len(models),
            "source": "built-in"}


def _custom_row(models, api_url, *, slug="custom:cline-api"):
    return {"slug": slug, "name": slug, "is_current": False, "is_user_defined": True,
            "models": list(models), "total_models": len(models), "source": "user-config",
            "api_url": api_url}


def test_remote_custom_endpoint_keeps_aggregator_models():
    """#56145: a remote custom API reselling the same models is a second real route, not a
    more-specific deployment — the aggregator row must keep the overlapping names selectable."""
    rows = [
        _custom_row(["nex-agi/nex-n2.5-pro:free", "cline/only"], "https://api.cline.example/v1"),
        _aggregator_row(["nex-agi/nex-n2.5-pro:free", "openrouter/only"], total=382),
    ]
    _strip_aggregator_overlaps(rows)
    assert rows[1]["models"] == ["nex-agi/nex-n2.5-pro:free", "openrouter/only"]
    assert rows[1]["total_models"] == 382


def test_local_proxy_still_strips_aggregator_overlap():
    """#45954 stays fixed: a local proxy serves its models itself, so picking the aggregator row
    for one of them would silently route the call away from the user's deployment."""
    rows = [
        _custom_row(["my/deployed-model"], "http://localhost:4000/v1"),
        _aggregator_row(["my/deployed-model", "openrouter/other"]),
    ]
    _strip_aggregator_overlaps(rows)
    assert rows[1]["models"] == ["openrouter/other"]


def test_strip_keeps_real_total_models():
    """Whatever the dedup hides, total_models keeps reporting the real catalog size — the picker
    sorts rows and labels them by it, and a deduped count makes a 382-model aggregator look like
    a 12-model one (#56145)."""
    rows = [
        _custom_row([f"shared/model-{i}" for i in range(370)], "http://127.0.0.1:4000/v1"),
        _aggregator_row([f"shared/model-{i}" for i in range(370)] + ["kilo/only-12"],
                        total=382, slug="kilocode"),
    ]
    _strip_aggregator_overlaps(rows)
    assert rows[1]["models"] == ["kilo/only-12"]
    assert rows[1]["total_models"] == 382


def test_private_lan_endpoint_counts_as_local():
    """A proxy on the user's LAN is still the user's own deployment."""
    rows = [
        _custom_row(["lan/model"], "http://192.168.1.5:8000/v1"),
        _aggregator_row(["lan/model", "openrouter/other"]),
    ]
    _strip_aggregator_overlaps(rows)
    assert rows[1]["models"] == ["openrouter/other"]


def test_user_defined_row_never_stripped():
    """is_user_defined rows are the source of user_models, and is_routing_aggregator() is True
    for every custom:* slug — the guard must keep the dedup from emptying the user's own row."""
    rows = [
        _custom_row(["a/model", "b/model"], "http://localhost:4000/v1", slug="custom:proxy-a"),
        _custom_row(["a/model"], "http://localhost:5000/v1", slug="custom:proxy-b"),
    ]
    _strip_aggregator_overlaps(rows)
    assert rows[1]["models"] == ["a/model"]


@pytest.mark.parametrize("url,expected", [
    ("http://localhost:4000/v1", True),
    ("http://127.0.0.1:4000/v1", True),
    ("http://[::1]:4000/v1", True),
    ("http://10.0.0.5:11434/v1", True),
    ("http://172.16.0.9/v1", True),
    ("http://192.168.1.5:8000/v1", True),
    ("https://api.cline.example/v1", False),
    ("https://litellm.mycompany.com/v1", False),
    ("", False),
    (None, False),
    ("not a url", False),
])
def test_user_endpoint_is_local(url, expected):
    assert _user_endpoint_is_local(url) is expected
