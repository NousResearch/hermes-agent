"""Appendix A plain-data validation, without provider or filesystem authority.

The caller supplies only a trusted registry entry, never a client Context. Token
freshness, durable consent bindings and filesystem admission belong to their
respective stateful boundaries; this module checks the wire facts only.
"""

import json
import math
import re

_STATUSES = {
    "invalid_request": 400, "invalid_identifier": 400, "not_found": 404,
    "provider_unavailable": 503, "ownership_state_unavailable": 503,
    "consent_required": 403, "permission_denied": 403, "sign_in_required": 401,
    "policy_denied": 403, "unsupported": 422, "review_required": 409,
    "acknowledgement_required": 409, "review_expired": 409, "review_changed": 409,
    "ownership_conflict": 409, "busy": 429, "operation_in_progress": 409,
    "timeout": 504, "outcome_unknown": 202, "provider_error": 502, "invalid_response": 502,
}
_MAX_INT = 9007199254740991
_ID = r"[A-Za-z0-9][A-Za-z0-9._~-]{0,191}"
_SOURCE = r"plugin\.[0-9a-f]{64}\.[a-z][a-z0-9-]{0,47}"


class CatalogError(Exception):
    """Named safe failure, ready for the HTTP Result error envelope."""

    def __init__(self, code, message=""):
        if type(code) is not str or code not in _STATUSES:
            code = "provider_error"
        try:
            _text(message, 2048, descriptive=True)
        except (ValueError, UnicodeError):
            message = "Catalog request failed."
        self.code = code
        self.message = message
        self.status = _STATUSES[code]
        self.payload = {"ok": False, "error": {"code": code, "message": message}}
        super().__init__(message)


def _require(condition):
    if not condition:
        raise ValueError("Invalid catalog data.")


def _text(value, maximum, minimum=0, *, descriptive=False):
    _require(type(value) is str and minimum <= len(value) <= maximum)
    value.encode("utf-8")
    allowed = "\n\t" if descriptive else ""
    _require(not any((ord(c) < 32 or 127 <= ord(c) < 160) and c not in allowed for c in value))
    return value


def _pattern(value, pattern):
    _require(type(value) is str and re.fullmatch(pattern, value, flags=re.ASCII) is not None)
    return value


def _integer(value, minimum=0, maximum=_MAX_INT):
    _require(type(value) is int and minimum <= value <= maximum)
    return value


def _boolean(value):
    _require(type(value) is bool)
    return value


def _null(value):
    _require(value is None)


def _nullable(check):
    return lambda value: None if value is None else check(value)


def _enum(choices):
    allowed = frozenset(choices.split("|"))
    return lambda value: _pattern(value, "(?:" + "|".join(re.escape(c) for c in allowed) + ")")


def _list(check, maximum, *, unique=False, key=None):
    def validate(value):
        _require(type(value) is list and len(value) <= maximum)
        values = [check(item) for item in value]
        if unique:
            keys = [item[key] if key else item for item in values]
            _require(len(set(keys)) == len(keys))
        return values
    return validate


def _object(value, fields, *, strict=False, defaults=None) -> dict:
    _require(type(value) is dict)
    defaults = {} if defaults is None else defaults
    _require(set(fields) - set(defaults) <= set(value))
    if strict:
        _require(set(value) <= set(fields))
    return {name: check(value[name] if name in value else defaults[name]) for name, check in fields.items()}


def _plain(value, stack):
    """Reject Python objects/subclasses before JSON serialization invokes them."""
    kind = type(value)
    if value is None or kind in (bool, int):
        return
    if kind is float:
        _require(math.isfinite(value))
        return
    if kind is str:
        value.encode("utf-8")
        return
    _require(kind in (list, dict) and id(value) not in stack)
    stack.add(id(value))
    try:
        if kind is dict:
            for key, item in value.items():
                _require(type(key) is str)
                key.encode("utf-8")
                _plain(item, stack)
        else:
            for item in value:
                _plain(item, stack)
    finally:
        stack.remove(id(value))


def _payload(value, maximum):
    _plain(value, set())
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    _require(len(encoded) <= maximum)


def _path(value):
    _text(value, 1024, 1)
    _require(len(value.encode("utf-8")) <= 1024 and "\\" not in value and ":" not in value)
    _require(all(segment not in ("", ".", "..") and len(segment.encode("utf-8")) <= 255
                 for segment in value.split("/")))
    return value


def _plugin_key(value):
    _text(value, 1024, 1)
    _require(len(value.encode("utf-8")) <= 1024)
    return value


def _reference(value):
    return _pattern(value, _SOURCE + ":" + _ID)


def _stamp_reference(value, owner):
    return value if owner is None else owner.source_id + ":" + value.split(":", 1)[1]


_item_id = lambda value: _pattern(value, _ID)
_token = lambda value: _pattern(value, r"[A-Za-z0-9_-]{16,256}")
_version = lambda value: _pattern(value, r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}")
_revision = lambda value: _text(value, 256, 1)
_description = lambda value: _text(value, 4096, descriptive=True)
_message = lambda value: _text(value, 2048, descriptive=True)
_action = _enum("install|update|uninstall")
_actions = _list(_action, 3, unique=True)
_managed = _enum("not_installed|installed|removed|unmanaged_local|pending|unavailable")
_verdict = _enum("safe|caution|dangerous|unknown")
_availability = _enum("ready|hidden|sign_in_required|unavailable")


def _request_id(value):
    try:
        return _item_id(value)
    except ValueError:
        raise CatalogError("invalid_identifier", "Invalid catalog item identifier.") from None


def _admission(value):
    if value is None:
        return None  # Only the stateful dispatcher can authorize completed replay.
    return _object(value, dict(registration_id=_token, generation=_token, review_id=_token,
        reservation_id=_token, expected_path_revision=_integer,
        expected_installation_revision=_nullable(_revision)), strict=True)


_REQUESTS = {
    "availability": {},
    "search": dict(query=lambda v: _text(v, 256, descriptive=True),
                   limit=lambda v: _integer(v, 1, 100), cursor=_nullable(_token)),
    "inspect": dict(id=_request_id, version=_nullable(_version)),
    "scan": dict(id=_request_id, version=_nullable(_version)),
    "installed": dict(limit=lambda v: _integer(v, 1, 100), cursor=_nullable(_token)),
    "check": dict(ids=_list(_request_id, 100), cursor=_nullable(_token)),
    "review": dict(action=_action, id=_request_id, version=_nullable(_version)),
    "reviewed": dict(review_token=_token),
    "confirm": dict(action=_action, id=_request_id, review_token=_token,
                    acknowledgements=_list(_item_id, 1000, unique=True), admission=_admission),
    "operation": dict(operation_id=_token),
}
_DEFAULTS = dict(query="", limit=20, version=None, cursor=None)


def validate_request(method, payload) -> dict:
    """Validate a request without Context; reject all client-supplied authority."""
    try:
        _require(type(method) is str and method in _REQUESTS)
        _payload(payload, 256 * 1024)
        fields = _REQUESTS[method]
        result = _object(payload, fields, strict=True,
                         defaults={key: value for key, value in _DEFAULTS.items() if key in fields})
        if method == "review" and result["action"] == "uninstall":
            _require(result["version"] is None)
        return result
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise CatalogError("invalid_request", "Invalid catalog request.") from None


def _entry(value, owner):
    provider = lambda v: _object(v, dict(
        id=lambda x: _pattern(x, r"[a-z][a-z0-9-]{0,47}"), plugin_id=_plugin_key,
        label=lambda x: _text(x, 80, 1), version=_nullable(_version),
        author=_nullable(lambda x: _text(x, 256, 1)), managed_state=_managed,
        scan_verdict=_verdict, actions=_actions))
    data = _object(value, dict(id=_item_id, name=lambda v: _text(v, 256, 1),
        description=_description, source=lambda v: _pattern(v, _SOURCE), identifier=_reference,
        trust_level=lambda v: _text(v, 256, 1), repo=_null, path=_null,
        tags=_list(lambda v: _text(v, 64, 1), 32), provider=provider))
    _require(data["identifier"].split(":", 1)[1] == data["id"])
    data["trust_level"] = "community"
    if owner is not None:
        data.update(source=owner.source_id, identifier=owner.source_id + ":" + data["id"])
        data["provider"].update(id=owner.descriptor.name, plugin_id=owner.plugin_key,
                                 label=owner.descriptor.label)
    return data


def _file(value):
    return _object(value, dict(path=_path, size=_integer, change=_enum("add|replace|remove|unchanged")))


def _finding(value):
    return _object(value, dict(id=_item_id, severity=_enum("info|low|medium|high|critical"),
        category=lambda v: _text(v, 64, 1), file=_nullable(_path),
        line=_nullable(lambda v: _integer(v, 1, 2147483647)), description=_description,
        acknowledgement_required=_boolean))


def _scan(value):
    return _object(value, dict(verdict=_verdict, policy=_enum("allow|acknowledge|block|unavailable"),
        summary=_description, findings=_list(_finding, 1000, unique=True, key="id"), complete=_boolean))


def _setup(value):
    return _object(value, dict(id=_item_id, summary=lambda v: _text(v, 256, 1, descriptive=True),
                               detail=_description))


def _installation(value, owner):
    data = _object(value, dict(identifier=_reference, name=lambda v: _text(v, 256, 1),
        target_relative_path=_path, installed_version=_nullable(_version), latest_version=_nullable(_version),
        managed_state=_managed, installation_revision=_nullable(_revision), actions=_actions))
    data["identifier"] = _stamp_reference(data["identifier"], owner)
    if data["managed_state"] in ("installed", "pending") and data["actions"]:
        _require(data["installation_revision"] is not None)
    return data


def _update(value, owner):
    data = _object(value, dict(identifier=_reference, installed_version=_nullable(_version),
        latest_version=_nullable(_version), state=_enum("current|update_available|removed|pending|unavailable|blocked"),
        installation_revision=_nullable(_revision)))
    data["identifier"] = _stamp_reference(data["identifier"], owner)
    return data


def _review(value, owner):
    data = _object(value, dict(review_id=_nullable(_token), action=_action, identifier=_reference,
        version=_nullable(_version), installation_revision=_nullable(_revision), target_relative_path=_path,
        summary=_description, files=_list(_file, 2000), preview_complete=_boolean, scan=_scan,
        acknowledgements_required=_list(_item_id, 1000, unique=True), policy_revision=_nullable(_revision),
        content_digest=_nullable(_revision), setup_guidance=_list(_setup, 100, unique=True, key="id"),
        confirmable=_boolean, review_token=_nullable(_token),
        expires_at_ms=_nullable(lambda v: _integer(v, 1))))
    required = {f["id"] for f in data["scan"]["findings"] if f["acknowledgement_required"]}
    _require(set(data["acknowledgements_required"]) == required)
    policy = data["scan"]["policy"]
    if policy == "allow":
        _require(not required)
    if policy == "acknowledge":
        _require(bool(required))
    if data["action"] == "install":
        _require(data["installation_revision"] is None)
    if data["confirmable"]:
        _require(data["preview_complete"] and data["scan"]["complete"] and policy in ("allow", "acknowledge"))
        _require(all(data[key] is not None for key in (
            "review_id", "version", "policy_revision", "content_digest", "review_token", "expires_at_ms")))
        _require(data["review_id"] != data["review_token"])
        if data["action"] != "install":
            _require(data["installation_revision"] is not None)
    else:
        _require(data["review_token"] is None and data["expires_at_ms"] is None)
    data["identifier"] = _stamp_reference(data["identifier"], owner)
    return data


def _operation(value, owner):
    data = _object(value, dict(operation_id=_token, identifier=_reference, action=_action,
        state=_enum("pending|completed|failed|unknown"),
        outcome=_nullable(_enum("installed|updated|uninstalled|unchanged|refused")),
        managed_state=_managed, message=_message, installation_revision=_nullable(_revision)))
    state, outcome = data["state"], data["outcome"]
    if state in ("pending", "unknown"):
        _require(outcome is None)
    elif state == "failed":
        _require(outcome == "refused")
    else:
        _require(outcome is not None and outcome != "refused")
    data["identifier"] = _stamp_reference(data["identifier"], owner)
    return data


def _reviewed(value, owner):
    data = _object(value, dict(kind=_enum("unstarted|pending|completed"),
        review=lambda v: _review(v, owner), operation=_nullable(lambda v: _operation(v, owner))))
    operation = data["operation"]
    if data["kind"] == "unstarted":
        _require(operation is None)
    else:
        _require(operation is not None)
        _require(operation["identifier"] == data["review"]["identifier"]
                 and operation["action"] == data["review"]["action"])
        states = ("pending", "unknown") if data["kind"] == "pending" else ("completed", "failed")
        _require(operation["state"] in states)
    return data


def _page(value, check):
    return _object(value, dict(items=_list(check, 100), next_cursor=_nullable(_token)))


_RESULTS = {
    "availability": lambda v, e: _object(v, dict(state=_availability, message=_message)),
    "search": lambda v, e: _page(v, lambda item: _entry(item, e)),
    "inspect": lambda v, e: _object(v, dict(entry=lambda x: _entry(x, e), skill_md=_description_md,
                                            files=_list(_file, 2000), preview_complete=_boolean)),
    "scan": lambda v, e: _object(v, dict(entry=lambda x: _entry(x, e), scan=_scan)),
    "installed": lambda v, e: _page(v, lambda item: _installation(item, e)),
    "check": lambda v, e: _page(v, lambda item: _update(item, e)),
    "review": _review, "reviewed": _reviewed, "confirm": _operation, "operation": _operation,
}


def _description_md(value):
    _text(value, 256 * 1024, descriptive=True)
    _require(len(value.encode("utf-8")) <= 256 * 1024)
    return value


def validate_result(method, result, entry=None) -> dict:
    """Consume provider Result, return sanitized unwrapped success or raise."""
    try:
        _require(type(method) is str and method in _RESULTS)
        _payload(result, 8 * 1024 * 1024)
        envelope = _object(result, dict(ok=_boolean))
        if not envelope["ok"]:
            error = _object(result.get("error"), dict(code=_enum("|".join(_STATUSES)), message=_message))
            # These messages are core-authored even when a provider names the code.
            message = "Catalog provider failed." if error["code"] in ("provider_error", "invalid_response") else error["message"]
            raise CatalogError(error["code"], message)
        _require("data" in result)
        return _RESULTS[method](result["data"], entry)
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise CatalogError("invalid_response", "Invalid catalog provider response.") from None
