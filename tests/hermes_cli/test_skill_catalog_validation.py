"""Appendix A plain-data boundary; no provider, filesystem or token authority."""
import copy
import importlib
from types import SimpleNamespace

import pytest

from hermes_cli.skill_catalog.identity import Descriptor, source_id


@pytest.fixture
def api():
    return importlib.import_module("hermes_cli.skill_catalog.validation")


@pytest.fixture
def owner():
    return SimpleNamespace(source_id=source_id("owner", "team"), plugin_key="owner",
                           descriptor=Descriptor(1, "team", "Team", "impl-1"))


def entry() -> dict:
    return dict(id="Skill-1", name="Skill", description="A\npreview", source=source_id("other", "other"),
                identifier=source_id("other", "other") + ":Skill-1", trust_level="trusted", repo=None,
                path=None, tags=["tag"], provider=dict(id="other", plugin_id="other", label="Other",
                version="1+build", author=None, managed_state="not_installed", scan_verdict="unknown",
                actions=["install"]))


def scan() -> dict:
    return dict(verdict="dangerous", policy="allow", summary="", findings=[], complete=True)


def review() -> dict:
    return dict(review_id="r" * 16, action="install", identifier=entry()["identifier"], version="1",
                installation_revision=None, target_relative_path="skills/team", summary="",
                files=[dict(path="SKILL.md", size=0, change="add")], preview_complete=True, scan=scan(),
                acknowledgements_required=[], policy_revision="policy", content_digest="digest",
                setup_guidance=[], confirmable=True, review_token="t" * 16, expires_at_ms=2_000_000_900_000)


def operation() -> dict:
    return dict(operation_id="o" * 16, identifier=entry()["identifier"], action="install",
                state="completed", outcome="installed", managed_state="installed", message="",
                installation_revision="revision")


def result(api, method, data, owner=None):
    return api.validate_result(method, {"ok": True, "data": data}, entry=owner)


def rejects(api, method, data, *, request=False, code=None):
    with pytest.raises(api.CatalogError) as caught:
        if request:
            api.validate_request(method, data)
        else:
            result(api, method, data)
    assert caught.value.code == (code or ("invalid_request" if request else "invalid_response"))


@pytest.mark.parametrize("method,payload,expected", [
    ("availability", {}, {}), ("search", {}, dict(query="", limit=20, cursor=None)),
    ("inspect", {"id": "X"}, dict(id="X", version=None)),
    ("scan", {"id": "X", "version": "1+build"}, dict(id="X", version="1+build")),
    ("installed", {}, dict(limit=20, cursor=None)),
    ("check", {"ids": []}, dict(ids=[], cursor=None)),
    ("review", {"action": "uninstall", "id": "X"}, dict(action="uninstall", id="X", version=None)),
    ("reviewed", {"review_token": "t" * 16}, {"review_token": "t" * 16}),
    ("operation", {"operation_id": "o" * 16}, {"operation_id": "o" * 16}),
    ("confirm", dict(action="install", id="X", review_token="t" * 16, acknowledgements=[], admission=None),
     dict(action="install", id="X", review_token="t" * 16, acknowledgements=[], admission=None)),
])
def test_requests_and_defaults(api, method, payload, expected):
    before = copy.deepcopy(payload)
    assert api.validate_request(method, payload) == expected
    assert payload == before


@pytest.mark.parametrize("method,payload,code", [
    ("search", {"context": {}}, "invalid_request"),
    ("search", {"plugin_key": "owner"}, "invalid_request"),
    ("search", {"limit": True}, "invalid_request"), ("search", {"limit": 1.0}, "invalid_request"),
    ("search", {"limit": "20"}, "invalid_request"), ("search", {"limit": 0}, "invalid_request"),
    ("search", {"limit": 101}, "invalid_request"), ("search", {"query": None}, "invalid_request"),
    ("search", {"query": "x" * 257}, "invalid_request"),
    ("search", {"cursor": "x" * 15}, "invalid_request"),
    ("inspect", {"id": "../x"}, "invalid_identifier"),
    ("inspect", {"id": "a:b"}, "invalid_identifier"),
    ("inspect", {"id": "a%20"}, "invalid_identifier"),
    ("inspect", {"id": "@x"}, "invalid_identifier"),
    ("inspect", {"id": "x" * 193}, "invalid_identifier"),
    ("inspect", {"id": "X", "version": ""}, "invalid_request"),
    ("review", {"id": "X", "action": "uninstall", "version": "1"}, "invalid_request"),
    ("check", {"ids": ["x"] * 101}, "invalid_request"),
    ("confirm", dict(action="install", id="X", review_token="t" * 16,
                     acknowledgements=["a", "a"], admission=None), "invalid_request"),
    ("unknown", {}, "invalid_request"),
])
def test_invalid_requests(api, method, payload, code):
    rejects(api, method, payload, request=True, code=code)


def test_admission(api):
    admission = dict(registration_id="r" * 16, generation="g" * 16, review_id="v" * 16,
                     reservation_id="s" * 16, expected_path_revision=0, expected_installation_revision=None)
    payload = dict(action="install", id="X", review_token="t" * 16, acknowledgements=[], admission=admission)
    assert api.validate_request("confirm", payload)["admission"] == admission
    for key, value in [("expected_path_revision", True), ("expected_path_revision", 2**53),
                       ("generation", "g"), ("unexpected", 1)]:
        bad = copy.deepcopy(payload)
        bad["admission"][key] = value
        rejects(api, "confirm", bad, request=True)


@pytest.mark.parametrize("method,data", [
    ("availability", dict(state="ready", message="")),
    ("search", dict(items=[entry()], next_cursor=None)),
    ("inspect", dict(entry=entry(), skill_md="# Skill\n", files=[], preview_complete=True)),
    ("scan", dict(entry=entry(), scan=scan())),
    ("installed", dict(items=[dict(identifier=entry()["identifier"], name="Skill", target_relative_path="team",
        installed_version=None, latest_version="1", managed_state="installed", installation_revision=None,
        actions=[])], next_cursor=None)),
    ("check", dict(items=[dict(identifier=entry()["identifier"], installed_version=None, latest_version=None,
        state="unavailable", installation_revision=None)], next_cursor=None)),
    ("review", review()), ("confirm", operation()), ("operation", operation()),
    ("reviewed", dict(kind="unstarted", review=review(), operation=None)),
    ("reviewed", dict(kind="completed", review=review(), operation=operation())),
])
def test_all_success_methods(api, owner, method, data):
    original = copy.deepcopy(data)
    actual = result(api, method, data, owner)
    assert isinstance(actual, dict)
    assert data == original


def test_stamps_and_filters_only_after_validation(api, owner):
    value = entry()
    value["extra"] = {"html": "<b>not executable</b>"}
    value["provider"]["extra"] = True
    actual = result(api, "search", dict(items=[value], next_cursor=None, extra=1), owner)["items"][0]
    assert actual["source"] == owner.source_id
    assert actual["identifier"] == owner.source_id + ":Skill-1"
    assert actual["trust_level"] == "community"
    assert actual["provider"]["id"] == "team"
    assert actual["provider"]["plugin_id"] == "owner"
    assert actual["provider"]["label"] == "Team"
    assert "extra" not in actual and "extra" not in actual["provider"]
    value["provider"]["version"] = False
    rejects(api, "search", dict(items=[value], next_cursor=None))
    assert result(api, "review", review(), owner)["identifier"] == owner.source_id + ":Skill-1"


@pytest.mark.parametrize("field,value", [
    ("preview_complete", False), ("confirmable", 1), ("review_id", None), ("version", None),
    ("policy_revision", None), ("content_digest", None), ("review_token", None),
    ("expires_at_ms", None), ("expires_at_ms", True), ("installation_revision", "existing"),
    ("target_relative_path", "../escape"), ("acknowledgements_required", ["missing"]),
])
def test_review_cross_fields(api, field, value):
    data = review()
    data[field] = value
    rejects(api, "review", data)


@pytest.mark.parametrize("path", ["/root", "a/", "a//b", "a/./b", "a/../b", "C:drive", "a\\b",
                                   "a\x00b", "a/" + "é" * 128])
def test_invalid_display_paths(api, path):
    data = review()
    data["files"][0]["path"] = path
    rejects(api, "review", data)


def test_acknowledgement_policy_and_completeness(api):
    data = review()
    finding = dict(id="f", severity="high", category="network", file=None, line=None,
                   description="Read\nthis", acknowledgement_required=True)
    data["scan"]["findings"] = [finding]
    data["scan"]["policy"] = "acknowledge"
    data["acknowledgements_required"] = ["f"]
    assert result(api, "review", data)["confirmable"]  # dangerous is not an unconditional block
    for mutate in [lambda d: d["scan"].update(complete=False),
                   lambda d: d["scan"].update(policy="allow"),
                   lambda d: d["scan"].update(policy="block"),
                   lambda d: d["scan"]["findings"].append(copy.deepcopy(finding)),
                   lambda d: d.update(acknowledgements_required=["f", "f"])]:
        bad = copy.deepcopy(data)
        mutate(bad)
        rejects(api, "review", bad)
    data.update(confirmable=False, review_id=None, version=None, review_token=None, expires_at_ms=None,
                policy_revision=None, content_digest=None)
    data["scan"].update(policy="block", complete=False)
    assert result(api, "review", data)["review_token"] is None


@pytest.mark.parametrize("state,outcome,valid", [
    ("pending", None, True), ("unknown", None, True), ("completed", "updated", True),
    ("failed", "refused", True), ("completed", "refused", False), ("failed", "installed", False),
    ("pending", "installed", False), ("completed", None, False),
])
def test_operation_cross_fields(api, state, outcome, valid):
    data = operation()
    data.update(state=state, outcome=outcome)
    if valid:
        assert result(api, "operation", data)["state"] == state
    else:
        rejects(api, "operation", data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), (1, 2), {1: "value"}, object(), "\ud800"])
def test_non_plain_json_even_in_unknown_fields(api, value):
    rejects(api, "availability", dict(state="ready", message="", extra=value))


def test_bounds_required_fields_and_strict_types(api):
    for key in entry():
        data = entry()
        del data[key]
        rejects(api, "search", dict(items=[data], next_cursor=None))
    for key, value in [("tags", ["t"] * 33), ("name", ""), ("repo", "repo"),
                       ("description", "bad\x01text")]:
        data = entry()
        data[key] = value
        rejects(api, "search", dict(items=[data], next_cursor=None))
    for key, value in [("size", True), ("size", -1), ("size", 2**53), ("change", "write")]:
        data = review()
        data["files"][0][key] = value
        rejects(api, "review", data)
    rejects(api, "inspect", dict(entry=entry(), skill_md="é" * 131073, files=[], preview_complete=False))
    rejects(api, "availability", dict(state="ready", message="", extra="x" * (8 * 1024 * 1024)))
    rejects(api, "search", {"query": "x", "extra": "x" * (256 * 1024)}, request=True)


@pytest.mark.parametrize("code,status", [
    ("invalid_request", 400), ("invalid_identifier", 400), ("not_found", 404),
    ("provider_unavailable", 503), ("ownership_state_unavailable", 503),
    ("consent_required", 403), ("permission_denied", 403), ("sign_in_required", 401),
    ("policy_denied", 403), ("unsupported", 422), ("review_required", 409),
    ("acknowledgement_required", 409), ("review_expired", 409), ("review_changed", 409),
    ("ownership_conflict", 409), ("busy", 429), ("operation_in_progress", 409),
    ("timeout", 504), ("outcome_unknown", 202), ("provider_error", 502), ("invalid_response", 502),
])
def test_named_errors(api, code, status):
    error = api.CatalogError(code)
    assert error.status == status
    assert error.payload["error"]["code"] == code
    with pytest.raises(api.CatalogError) as caught:
        api.validate_result("availability", {"ok": False, "error": {"code": code, "message": "Safe"}})
    assert caught.value.status == status


def test_additional_schema_boundaries(api):
    # Each consumed nested value is validated before unknown fields disappear.
    for key, value in [("line", 0), ("line", 2147483648), ("line", False),
                       ("severity", "severe"), ("category", ""), ("acknowledgement_required", 1)]:
        data = review()
        finding = dict(id="f", severity="high", category="network", file=None, line=None,
                       description="", acknowledgement_required=False)
        finding[key] = value
        data["scan"]["findings"] = [finding]
        rejects(api, "review", data)
    for collection, values in [("files", [dict(path="SKILL.md", size=0, change="add")] * 2001),
                               ("setup_guidance", [dict(id="s", summary="Setup", detail="")] * 101)]:
        data = review()
        data[collection] = values
        rejects(api, "review", data)
    for policy in ("block", "unavailable"):
        data = review()
        data["scan"]["policy"] = policy
        rejects(api, "review", data)
    data = review()
    data["review_id"] = data["review_token"]
    rejects(api, "review", data)
    data = review()
    data.update(action="update", installation_revision=None)
    rejects(api, "review", data)
    data["installation_revision"] = "revision"
    assert result(api, "review", data)["action"] == "update"
    rejects(api, "reviewed", dict(kind="unstarted", review=review(), operation=operation()))
    rejects(api, "reviewed", dict(kind="completed", review=review(), operation=None))
    rejects(api, "reviewed", dict(kind="pending", review=review(), operation=operation()))
    data = entry()
    data["provider"]["actions"] = ["install", "install"]
    rejects(api, "search", dict(items=[data], next_cursor=None))
    data = entry()
    data["provider"]["plugin_id"] = "é" * 513
    rejects(api, "search", dict(items=[data], next_cursor=None))
    data = entry()
    data["provider"]["version"] = "v" * 65
    rejects(api, "search", dict(items=[data], next_cursor=None))


def test_plain_types_do_not_invoke_hooks(api):
    class EvilDict(dict):
        def items(self):
            raise AssertionError("Must not inspect custom mappings")

    class EvilString(str):
        def encode(self, *args, **kwargs):
            raise AssertionError("Must not inspect custom strings")

    rejects(api, "search", EvilDict(), request=True)
    rejects(api, "search", {"query": EvilString("query")}, request=True)
    cyclic = []
    cyclic.append(cyclic)
    rejects(api, "availability", dict(state="ready", message="", extra=cyclic))


def test_exact_grammar_and_inclusive_bounds(api):
    assert api.validate_request("inspect", {"id": "A" + "x" * 191, "version": "V" + "x" * 63})["version"]
    for limit in (1, 100):
        assert api.validate_request("search", {"limit": limit})["limit"] == limit
    for length in (16, 256):
        assert api.validate_request("search", {"cursor": "x" * length})["cursor"] == "x" * length
    for bad in ("a/b", "x\n", " x", "x ", "é", "a:b", "a%", "a@"):
        rejects(api, "inspect", {"id": bad}, request=True, code="invalid_identifier")
    data = review()
    data["files"][0]["size"] = 9007199254740991
    assert result(api, "review", data)["files"][0]["size"] == 9007199254740991
    data = entry()
    data["name"] = "é" * 256
    data["description"] = "<b>HTML remains text</b>\n\t"
    actual = result(api, "search", dict(items=[data], next_cursor=None))
    assert actual["items"][0]["description"] == data["description"]


def test_error_messages_are_safe(api):
    for code in ("invalid_response", "provider_error"):
        with pytest.raises(api.CatalogError) as caught:
            api.validate_result("availability", dict(ok=False, error=dict(code=code, message="raw exception")))
        assert "raw exception" not in str(caught.value)
    assert api.CatalogError("busy", "bad\x00message").message == "Catalog request failed."


@pytest.mark.parametrize("envelope", [None, {}, {"ok": 1, "data": {}}, {"ok": True},
    {"ok": False, "error": {"code": "invented", "message": "x"}},
    {"ok": False, "error": {"code": "busy", "message": None}}])
def test_invalid_result_envelopes(api, envelope):
    with pytest.raises(api.CatalogError) as caught:
        api.validate_result("availability", envelope)
    assert caught.value.code == "invalid_response"
