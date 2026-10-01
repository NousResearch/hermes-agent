"""Responsibility declarations; employee contracts on native Hermes."""

from __future__ import annotations
from datetime import datetime
import hashlib
import os
from pathlib import PurePosixPath
import stat
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import hermes_yaml as yaml
from cron.jobs import parse_duration, parse_schedule
from responsibilities.secrets import parse_secret_reference
from responsibilities.delivery import parse_explicit_schedule_delivery
from responsibilities.filesystem import (_read_regular_file_fd)


from responsibilities.common import (
    MAX_SCHEDULE_DECLARATION_BYTES,
    MAX_WEBHOOK_ACK_BODY_BYTES,
    MAX_WEBHOOK_ACK_CONTENT_TYPE_LENGTH,
    MAX_WEBHOOK_DECLARATION_BYTES,
    SCHEDULES_DIRNAME,
    WEBHOOKS_DIRNAME,
    _ACK_MEDIA_TYPE_RE,
    _ALLOWED_DECLARATION_FIELDS,
    _ALLOWED_WEBHOOK_FIELDS,
    _HMAC_PREFIX_RE,
    _SCRIPT_SUFFIXES,
    _VERIFY_HEADER_SOURCE_RE,
    _VERIFY_SIGNED_INPUTS,
    validate_responsibility_name
)

def _declared_scope(
    prefix: str,
    parsed: Mapping[str, Any],
    *,
    legacy_prompt: bool,
    hint: str,
) -> tuple[str | None, str | None]:
    """Resolve a declaration's scope, honoring the retired 'prompt' spelling.

    Files written before the rename keep compiling on the scan path
    (``legacy_prompt=True``); mutations must use 'scope'.
    """

    if "prompt" in parsed:
        if "scope" in parsed:
            return None, (
                f"{prefix}: set 'scope' only, not both 'scope' and 'prompt'"
            )
        if not legacy_prompt:
            return None, (
                f"{prefix}: 'prompt' is now 'scope' — {hint}; rename the field"
            )
        scope = parsed.get("prompt")
    else:
        scope = parsed.get("scope")
    if not isinstance(scope, str) or not scope.strip():
        return None, f"{prefix}: 'scope' is required"
    return scope, None

def _declared_report(
    prefix: str,
    parsed: Mapping[str, Any],
    *,
    legacy_deliver: bool,
) -> tuple[Any, bool, str | None]:
    """Resolve a declaration's report target, honoring the retired 'deliver'
    spelling.

    Files written before the rename keep compiling on the scan path
    (``legacy_deliver=True``); mutations must use 'report'. Returns
    (value, present, error).
    """

    if "deliver" in parsed:
        if "report" in parsed:
            return None, False, (
                f"{prefix}: set 'report' only, not both 'report' and 'deliver'"
            )
        if not legacy_deliver:
            return None, False, (
                f"{prefix}: 'deliver' is now 'report' — the run's reporting "
                "line; rename the field"
            )
        return parsed["deliver"], True, None
    if "report" in parsed:
        return parsed["report"], True, None
    return None, False, None

def validate_schedule_declaration(
    filename: str, raw: bytes, *, legacy: bool = False
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one schedules/<name>.yaml declaration.

    Returns (declaration, None) on success or (None, error) on failure. The
    declaration is normalized to exactly the declared fields plus a
    ``content_hash`` of the raw bytes for reconciliation. ``legacy=True`` is
    the scan path: files written before the 'prompt'/'deliver' renames or the
    report-required rule keep compiling; mutations enforce all three.
    """

    if not filename.endswith(".yaml"):
        return None, (
            f"schedules/{filename}: declarations must be <name>.yaml files"
        )
    stem = filename[: -len(".yaml")]
    name_error = validate_responsibility_name(stem)
    if name_error:
        return None, f"schedules/{filename}: {name_error}"
    if len(raw) > MAX_SCHEDULE_DECLARATION_BYTES:
        return None, (
            f"schedules/{filename}: exceeds {MAX_SCHEDULE_DECLARATION_BYTES} bytes"
        )
    try:
        parsed = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        return None, f"schedules/{filename}: not valid YAML ({exc})"
    if not isinstance(parsed, dict):
        return None, f"schedules/{filename}: must be a YAML mapping"
    unknown = set(parsed) - _ALLOWED_DECLARATION_FIELDS - {"prompt", "deliver"}
    if unknown:
        # Keys may be heterogeneous YAML types (1: x), which sorted() cannot
        # order directly; normalize for display.
        unknown_display = sorted(str(key) for key in unknown)
        return None, (
            f"schedules/{filename}: unknown fields {unknown_display}; "
            "allowed: schedule, scope, report, repeat, script, timezone"
        )
    schedule = parsed.get("schedule")
    if not isinstance(schedule, str) or not schedule.strip():
        return None, f"schedules/{filename}: 'schedule' is required"
    try:
        parsed_schedule = parse_schedule(schedule, bare_duration_is_once=True)
    except ValueError as exc:
        return None, f"schedules/{filename}: invalid schedule: {exc}"
    scope, scope_error = _declared_scope(
        f"schedules/{filename}",
        parsed,
        legacy_prompt=legacy,
        hint="the slice of the responsibility this run carries",
    )
    if scope_error:
        return None, scope_error
    # The normalized declaration crosses the unversioned worker/controller
    # scan boundary and feeds native cron rows, so the scope keeps the
    # pre-rename wire key; only the file field and run messages say "scope".
    declaration: dict[str, Any] = {
        "schedule": schedule.strip(),
        "prompt": scope,
    }
    report, report_present, report_error = _declared_report(
        f"schedules/{filename}", parsed, legacy_deliver=legacy
    )
    if report_error:
        return None, report_error
    if not report_present:
        if not legacy:
            return None, (
                f"schedules/{filename}: 'report' is required — 'muted' or "
                "one exact Slack or Telegram target returned by "
                "send_message(action='list') or session_search; the current "
                "conversation is listed first"
            )
    else:
        report_error = _validate_report(
            filename,
            report,
            directory="schedules",
            legacy=legacy,
        )
        if report_error:
            return None, report_error
        # Same wire-compatibility rule as scope: stored definitions and the
        # scan boundary keep the pre-rename 'deliver' key.
        declaration["deliver"] = report
    if "repeat" in parsed:
        repeat = parsed["repeat"]
        if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1:
            return None, (
                f"schedules/{filename}: 'repeat' must be a positive integer"
            )
        declaration["repeat"] = repeat
    if "script" in parsed:
        script = parsed["script"]
        script_error = validate_declared_script_path(script)
        if script_error:
            return None, f"schedules/{filename}: {script_error}"
        declaration["script"] = str(script).strip()
    if "timezone" in parsed:
        declared_timezone = parsed["timezone"]
        if not isinstance(declared_timezone, str) or not declared_timezone.strip():
            return None, (
                f"schedules/{filename}: 'timezone' must be an IANA zone "
                "name such as Europe/Berlin"
            )
        declared_timezone = declared_timezone.strip()
        try:
            ZoneInfo(declared_timezone)
        except (ValueError, ZoneInfoNotFoundError):
            return None, (
                f"schedules/{filename}: unknown timezone "
                f"{declared_timezone!r}; use an IANA zone name such as "
                "Europe/Berlin"
            )
        applicability_error = _declared_timezone_applicability_error(
            schedule.strip(), parsed_schedule
        )
        if applicability_error:
            return None, f"schedules/{filename}: {applicability_error}"
        declaration["timezone"] = declared_timezone
    declaration["content_hash"] = hashlib.sha256(raw).hexdigest()
    return declaration, None

def _declared_timezone_applicability_error(
    schedule_text: str, parsed_schedule: Mapping[str, Any]
) -> str | None:
    """Reject 'timezone' on schedules it cannot move.

    Intervals and durations measure elapsed time and offset-qualified
    timestamps already fix their instant; only cron expressions and naive
    date-times have a wall clock for the zone to pin.
    """

    kind = parsed_schedule.get("kind")
    if kind == "interval":
        return (
            "'timezone' has no effect on an interval — it measures "
            "elapsed time; remove the field"
        )
    if kind != "once":
        return None
    try:
        parse_duration(schedule_text)
    except ValueError:
        pass
    else:
        return (
            "'timezone' has no effect on a duration — it measures "
            "elapsed time; remove the field"
        )
    try:
        moment = datetime.fromisoformat(schedule_text.replace("Z", "+00:00"))
    except ValueError:
        return "'timezone' requires a cron expression or a naive ISO date-time; remove it from relative schedules"
    if moment.tzinfo is not None:
        return (
            "'timezone' conflicts with the schedule's explicit UTC "
            "offset; remove one"
        )
    return None

def validate_declared_script_path(value: Any) -> str | None:
    """Validate a guard-script reference: a top-level package scripts/ file."""

    if not isinstance(value, str) or not value.strip():
        return (
            "'script' must name a package file under scripts/ "
            "(scripts/<name>.sh, .bash, or .py)"
        )
    parts = PurePosixPath(value.strip()).parts
    if (
        len(parts) != 2
        or parts[0] != "scripts"
        or parts[1] in {"", ".", ".."}
        or not parts[1].endswith(_SCRIPT_SUFFIXES)
    ):
        return (
            "'script' must name a package file under scripts/ "
            "(scripts/<name>.sh, .bash, or .py)"
        )
    return None

def _scan_schedule_declarations(
    package_fd: int, flags: int
) -> tuple[dict[str, Any], dict[str, str]]:
    """Read schedules/ inside an open package: (declarations, per-file errors)."""

    declarations: dict[str, Any] = {}
    errors: dict[str, str] = {}
    try:
        info = os.stat(SCHEDULES_DIRNAME, dir_fd=package_fd, follow_symlinks=False)
    except FileNotFoundError:
        return declarations, errors
    except OSError as exc:
        errors[SCHEDULES_DIRNAME] = f"schedules/: {exc}"
        return declarations, errors
    if not stat.S_ISDIR(info.st_mode):
        errors[SCHEDULES_DIRNAME] = "schedules/: not a directory"
        return declarations, errors
    try:
        schedules_fd = os.open(SCHEDULES_DIRNAME, flags, dir_fd=package_fd)
    except OSError as exc:
        errors[SCHEDULES_DIRNAME] = f"schedules/: {exc}"
        return declarations, errors
    try:
        for entry in sorted(os.scandir(schedules_fd), key=lambda item: item.name):
            try:
                if not entry.is_file(follow_symlinks=False):
                    errors[entry.name] = (
                        f"schedules/{entry.name}: not a regular file"
                    )
                    continue
                raw = _read_regular_file_fd(
                    schedules_fd,
                    entry.name,
                    max_bytes=MAX_SCHEDULE_DECLARATION_BYTES,
                )
            except (OSError, ValueError) as exc:
                errors[entry.name] = f"schedules/{entry.name}: {exc}"
                continue
            declaration, error = validate_schedule_declaration(
                entry.name, raw, legacy=True
            )
            if error is not None:
                errors[entry.name] = error
                continue
            declarations[entry.name[: -len(".yaml")]] = declaration
    finally:
        os.close(schedules_fd)
    return declarations, errors

def _validate_report(
    filename: str,
    report: Any,
    *,
    directory: str,
    legacy: bool,
) -> str | None:
    if isinstance(report, str) and report in {"local", "muted"}:
        return None
    if isinstance(report, str) and report == "origin":
        # Files written before 'origin' was retired keep compiling on the
        # scan path; mutations must declare an exact target.
        if legacy:
            return None
        return (
            f"{directory}/{filename}: 'report' no longer accepts 'origin' — "
            "use 'muted' or one exact Slack or Telegram target returned by "
            "send_message(action='list'); the current conversation is "
            "listed first"
        )
    try:
        if not isinstance(report, str):
            raise ValueError("Schedule delivery target is invalid")
        channel, _target = parse_explicit_schedule_delivery(report)
        raw_channel, _separator, _raw_target = report.partition(":")
        if raw_channel != channel:
            raise ValueError("Delivery channel must use its canonical spelling")
    except ValueError:
        return (
            f"{directory}/{filename}: 'report' must be 'muted' or one exact "
            "Slack or Telegram target returned by send_message(action='list') "
            "or session_search"
        )
    return None

def _normalize_dot_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or any(not part for part in normalized.split(".")):
        return None
    return normalized

def _validate_handshake(filename: str, value: Any) -> str | None:
    prefix = f"webhooks/{filename}: 'handshake'"
    if isinstance(value, str):
        return (
            None
            if _normalize_dot_path(value) is not None
            else f"{prefix} must be a non-empty dot-path"
        )
    if not isinstance(value, dict):
        return f"{prefix} must be a dot-path string or mapping"
    unknown = set(value) - {"method", "when", "respond", "secret", "encoding", "prefix"}
    if unknown:
        return f"{prefix} has unknown fields {sorted(str(key) for key in unknown)}"
    method = value.get("method", "GET")
    if method not in {"GET", "POST"}:
        return f"{prefix}.method must be GET or POST"
    when = value.get("when")
    if when is not None:
        if not isinstance(when, str) or "=" not in when:
            return f"{prefix}.when must be field=value"
        when_path, expected = when.split("=", 1)
        if _normalize_dot_path(when_path) is None or not expected.strip():
            return f"{prefix}.when must be field=value"
    respond = value.get("respond")
    computed = 0
    if isinstance(respond, str):
        if _normalize_dot_path(respond) is None:
            return f"{prefix}.respond must be a non-empty dot-path"
    elif not isinstance(respond, dict) or not respond:
        return f"{prefix}.respond is required and must be a dot-path or mapping"
    else:
        for key, path in respond.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(path, str)
                or not path
            ):
                return f"{prefix}.respond keys and values must be non-empty strings"
            if path.startswith("hmac_sha256(") and path.endswith(")"):
                computed += 1
                if (
                    _normalize_dot_path(path[len("hmac_sha256(") : -1])
                    is None
                ):
                    return f"{prefix}.respond contains an invalid hmac_sha256 path"
            elif _normalize_dot_path(path) is None:
                return f"{prefix}.respond contains an invalid dot-path"
    if computed > 1:
        return f"{prefix}.respond permits exactly one hmac_sha256 primitive"
    if computed:
        secret = value.get("secret")
        if not isinstance(secret, str) or not secret:
            return f"{prefix}.secret is required for hmac_sha256"
        try:
            parse_secret_reference(secret)
        except ValueError:
            return (
                f"{prefix}.secret must be a credential:// reference from the "
                "connection store, never a literal secret value"
            )
    if not computed and "secret" in value:
        return f"{prefix}.secret is allowed only with hmac_sha256"
    for field in ("encoding", "prefix"):
        if field in value and not computed:
            return f"{prefix}.{field} is allowed only with hmac_sha256"
    if "encoding" in value and value["encoding"] not in ("hex", "base64"):
        return f"{prefix}.encoding must be hex or base64"
    literal = value.get("prefix")
    if literal is not None and (
        not isinstance(literal, str)
        or not literal
        or _HMAC_PREFIX_RE.fullmatch(literal) is None
        or "(" in literal
        or ")" in literal
    ):
        return (
            f"{prefix}.prefix must be a short literal of printable "
            "characters (at most 16)"
        )
    return None

def _validate_verify(filename: str, value: Any) -> str | None:
    prefix = f"webhooks/{filename}: 'verify'"
    if not isinstance(value, dict):
        return f"{prefix} must be a mapping"
    unknown = set(value) - {"secret", "header", "timestamp", "signature", "encoding"}
    if unknown:
        return f"{prefix} has unknown fields {sorted(str(key) for key in unknown)}"
    secret = value.get("secret")
    if not isinstance(secret, str) or not secret:
        return f"{prefix}.secret is required"
    try:
        parse_secret_reference(secret)
    except ValueError:
        return (
            f"{prefix}.secret must be a env: reference to a configured secret, never a literal secret value"
        )
    for field in ("header", "timestamp"):
        source = value.get(field)
        if field == "timestamp" and source is None:
            continue
        if (
            not isinstance(source, str)
            or not source.strip()
            or len(source) > 100
            or _VERIFY_HEADER_SOURCE_RE.fullmatch(source.strip()) is None
        ):
            article = (
                "is required and " if field == "header" else ""
            )
            return (
                f"{prefix}.{field} {article}must be a header name, "
                "optionally with a .field selector for structured headers"
            )
    signature = value.get("signature")
    if not isinstance(signature, str) or not signature.strip():
        return f"{prefix}.signature is required"
    signature = signature.strip()
    marker = "hmac_sha256("
    if marker not in signature or not signature.endswith(")"):
        return f"{prefix}.signature must be [<prefix>]hmac_sha256(<signed-input>)"
    literal = signature[: signature.index(marker)]
    signed_input = signature[signature.index(marker) + len(marker) : -1]
    if literal and (
        _HMAC_PREFIX_RE.fullmatch(literal) is None
        or "(" in literal
        or ")" in literal
    ):
        return (
            f"{prefix}.signature prefix must be a short literal of "
            "printable characters (at most 16)"
        )
    if signed_input not in _VERIFY_SIGNED_INPUTS:
        return (
            f"{prefix}.signature signed input must be one of: "
            f"{', '.join(sorted(_VERIFY_SIGNED_INPUTS))}"
        )
    if "timestamp" in signed_input and value.get("timestamp") is None:
        return f"{prefix}.timestamp is required when the signed input uses it"
    if "timestamp" not in signed_input and value.get("timestamp") is not None:
        return f"{prefix}.timestamp is allowed only when the signed input uses it"
    if "encoding" in value and value["encoding"] not in ("hex", "base64"):
        return f"{prefix}.encoding must be hex or base64"
    return None

def _normalize_verify(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "secret": str(value["secret"]).strip(),
        "header": str(value["header"]).strip(),
        "signature": str(value["signature"]).strip(),
    }
    if value.get("timestamp") is not None:
        normalized["timestamp"] = str(value["timestamp"]).strip()
    if "encoding" in value:
        normalized["encoding"] = str(value["encoding"])
    return normalized

def _validate_ack(filename: str, value: Any) -> str | None:
    prefix = f"webhooks/{filename}: 'ack'"
    if not isinstance(value, dict):
        return f"{prefix} must be a mapping"
    unknown = set(value) - {"status", "content_type", "body"}
    if unknown:
        return f"{prefix} has unknown fields {sorted(str(key) for key in unknown)}"
    status = value.get("status", 200)
    if isinstance(status, bool) or not isinstance(status, int) or not (
        200 <= status <= 299
    ):
        return f"{prefix}.status must be a 2xx integer"
    body = value.get("body", "")
    if not isinstance(body, str):
        return f"{prefix}.body must be static text"
    try:
        body_bytes = len(body.encode("utf-8"))
    except UnicodeError:
        return f"{prefix}.body must be static text"
    if body_bytes > MAX_WEBHOOK_ACK_BODY_BYTES:
        return f"{prefix}.body exceeds {MAX_WEBHOOK_ACK_BODY_BYTES} bytes"
    content_type = value.get("content_type")
    if body:
        if status in {204, 205}:
            return f"{prefix}.status {status} does not allow a body"
        if not isinstance(content_type, str) or not content_type.strip():
            return f"{prefix}.content_type is required when body is set"
        if len(content_type) > MAX_WEBHOOK_ACK_CONTENT_TYPE_LENGTH:
            return (
                f"{prefix}.content_type exceeds "
                f"{MAX_WEBHOOK_ACK_CONTENT_TYPE_LENGTH} characters"
            )
        if _ACK_MEDIA_TYPE_RE.fullmatch(content_type.strip()) is None:
            return (
                f"{prefix}.content_type must be a bare type/subtype "
                "media type without parameters"
            )
    elif content_type is not None:
        return f"{prefix}.content_type is allowed only with a body"
    return None

def _normalize_ack(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {"status": int(value.get("status", 200))}
    body = str(value.get("body", ""))
    if body:
        normalized["body"] = body
        normalized["content_type"] = str(value["content_type"]).strip()
    return normalized

def _normalize_handshake(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    normalized = dict(value)
    when = normalized.get("when")
    if isinstance(when, str):
        path, expected = when.split("=", 1)
        normalized["when"] = f"{path.strip()}={expected.strip()}"
    respond = normalized.get("respond")
    if isinstance(respond, str):
        normalized["respond"] = respond.strip()
    elif isinstance(respond, dict):
        normalized_respond: dict[str, str] = {}
        for key, raw_path in respond.items():
            path = str(raw_path).strip()
            if path.startswith("hmac_sha256(") and path.endswith(")"):
                path = (
                    "hmac_sha256("
                    f"{path[len('hmac_sha256(') : -1].strip()}"
                    ")"
                )
            normalized_respond[str(key)] = path
        normalized["respond"] = normalized_respond
    return normalized

def validate_webhook_declaration(
    filename: str, raw: bytes, *, legacy: bool = False
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one webhooks/<name>.yaml declaration.

    ``legacy=True`` is the scan path: files written before the
    'prompt'/'deliver' renames or the report-required rule keep compiling;
    mutations enforce all three.
    """

    if not filename.endswith(".yaml"):
        return None, f"webhooks/{filename}: declarations must be <name>.yaml files"
    stem = filename[: -len(".yaml")]
    name_error = validate_responsibility_name(stem)
    if name_error:
        return None, f"webhooks/{filename}: {name_error}"
    if len(raw) > MAX_WEBHOOK_DECLARATION_BYTES:
        return (
            None,
            f"webhooks/{filename}: exceeds {MAX_WEBHOOK_DECLARATION_BYTES} bytes",
        )
    try:
        parsed = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        return None, f"webhooks/{filename}: not valid YAML ({exc})"
    if not isinstance(parsed, dict):
        return None, f"webhooks/{filename}: must be a YAML mapping"
    unknown = set(parsed) - _ALLOWED_WEBHOOK_FIELDS - {"prompt", "deliver"}
    if unknown:
        return None, (
            f"webhooks/{filename}: unknown fields "
            f"{sorted(str(key) for key in unknown)}; "
            "allowed: scope, key, report, handshake, ack, verify"
        )
    scope, scope_error = _declared_scope(
        f"webhooks/{filename}",
        parsed,
        legacy_prompt=legacy,
        hint="what a delivery means and which slice handles it",
    )
    if scope_error:
        return None, scope_error
    # Same wire-compatibility rule as schedule declarations: the scope keeps
    # the pre-rename key across the scan boundary and in stored definitions.
    declaration: dict[str, Any] = {"prompt": scope}
    if "key" in parsed:
        key = parsed["key"]
        normalized_key = _normalize_dot_path(key)
        if normalized_key is None:
            return None, f"webhooks/{filename}: 'key' must be a dot-path"
        declaration["key"] = normalized_key
    report, report_present, report_error = _declared_report(
        f"webhooks/{filename}", parsed, legacy_deliver=legacy
    )
    if report_error:
        return None, report_error
    if not report_present:
        if not legacy:
            return None, (
                f"webhooks/{filename}: 'report' is required — 'muted' or "
                "one exact Slack or Telegram target returned by "
                "send_message(action='list') or session_search; the current "
                "conversation is listed first"
            )
    else:
        error = _validate_report(
            filename,
            report,
            directory="webhooks",
            legacy=legacy,
        )
        if error:
            return None, error
        # Wire key stays 'deliver' across the scan boundary and stored rows.
        declaration["deliver"] = report
    if "handshake" in parsed:
        error = _validate_handshake(filename, parsed["handshake"])
        if error:
            return None, error
        declaration["handshake"] = _normalize_handshake(parsed["handshake"])
    if "ack" in parsed:
        error = _validate_ack(filename, parsed["ack"])
        if error:
            return None, error
        declaration["ack"] = _normalize_ack(parsed["ack"])
    if "verify" in parsed:
        error = _validate_verify(filename, parsed["verify"])
        if error:
            return None, error
        declaration["verify"] = _normalize_verify(parsed["verify"])
    declaration["content_hash"] = hashlib.sha256(raw).hexdigest()
    return declaration, None

def _scan_webhook_declarations(
    package_fd: int, flags: int
) -> tuple[dict[str, Any], dict[str, str]]:
    declarations: dict[str, Any] = {}
    errors: dict[str, str] = {}
    try:
        info = os.stat(WEBHOOKS_DIRNAME, dir_fd=package_fd, follow_symlinks=False)
    except FileNotFoundError:
        return declarations, errors
    except OSError as exc:
        return declarations, {WEBHOOKS_DIRNAME: f"webhooks/: {exc}"}
    if not stat.S_ISDIR(info.st_mode):
        return declarations, {WEBHOOKS_DIRNAME: "webhooks/: not a directory"}
    try:
        directory_fd = os.open(WEBHOOKS_DIRNAME, flags, dir_fd=package_fd)
    except OSError as exc:
        return declarations, {WEBHOOKS_DIRNAME: f"webhooks/: {exc}"}
    try:
        for entry in sorted(os.scandir(directory_fd), key=lambda item: item.name):
            try:
                if not entry.is_file(follow_symlinks=False):
                    errors[entry.name] = f"webhooks/{entry.name}: not a regular file"
                    continue
                raw = _read_regular_file_fd(
                    directory_fd,
                    entry.name,
                    max_bytes=MAX_WEBHOOK_DECLARATION_BYTES,
                )
            except (OSError, ValueError) as exc:
                errors[entry.name] = f"webhooks/{entry.name}: {exc}"
                continue
            declaration, error = validate_webhook_declaration(
                entry.name, raw, legacy=True
            )
            if error is not None:
                errors[entry.name] = error
            else:
                declarations[entry.name[: -len(".yaml")]] = declaration
    finally:
        os.close(directory_fd)
    return declarations, errors
