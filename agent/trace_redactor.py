"""Trace redaction hook — credential/PII scrubber for persisted traces.

Implements the redaction hook contract of specs/trace-schema.md S4 against the
normative schema specs/trace-v1.schema.json. Detectors are loaded from YAML
config (default: ``~/.hermes/trace_redaction.yaml``, falling back to the
packaged default ``specs/trace_redaction_default.yaml``); nothing about the
detector set is hardcoded here beyond marker/span formatting.

Contract (per S4.1):

    redact(value, *, context) -> value of the same shape

- Numbers, booleans, and nulls always pass through unchanged.
- Fields flagged always-safe (structural join keys: tool name, exit_code,
  duration_ms, etc.) pass through without detector evaluation.
- Everything else is scanned by the configured detectors, deny-by-default.
- Matches are replaced by the structure-preserving marker (S4.3):
  whole-scalar redaction yields the marker object
  ``{"redacted": true, "reason": ..., "sha256": <64-hex>}``; an embedded span
  yields the compact string ``«REDACTED:<reason>:sha256:<first-12-hex>…»``.
- Fail-closed: any redactor exception replaces the affected value with
  ``{"redacted": true, "reason": "redactor_error", "sha256": null}`` and the
  incident is logged. A redactor failure must never crash a run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

VALID_REASONS = ("credential", "pii", "redactor_error")

SPAN_PLACEHOLDER_FMT = "«REDACTED:{reason}:sha256:{digest12}…»"

# Dot-separated path segments and exact key names whose string values are
# redacted whole regardless of content (the schema's `password_field`
# detector). The actual list lives in config under `value_key_detectors`;
# these are only the hard structural always-safe key names from S4.2 item 1.
ALWAYS_SAFE_KEYS = frozenset(
    {
        "role",
        "tool_call_id",
        "name",
        "index",
        "id",
        "exit_code",
        "duration_ms",
        "is_terminal",
    }
)

# Field names on a redaction marker itself — never scan inside a marker.
_MARKER_KEYS = frozenset({"redacted", "reason", "sha256"})

_DEFAULT_CONFIG_PATH = "~/.hermes/trace_redaction.yaml"
_PACKAGED_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "specs" / "trace_redaction_default.yaml"


class RedactionError(Exception):
    """Internal signalling; never escapes redact() (fail-closed)."""


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def make_marker(reason: str, sha256: Optional[str]) -> Dict[str, Any]:
    if reason not in VALID_REASONS:
        reason = "redactor_error"
        sha256 = None
    return {"redacted": True, "reason": reason, "sha256": sha256}


def _is_marker(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and obj.get("redacted") is True
        and "reason" in obj
        and "sha256" in obj
    )


@dataclass
class RedactionContext:
    """Carries the field path and always-safe flag into the hook."""

    path: str = ""
    always_safe: bool = False
    # Running log of redaction events for the record-level redaction_log.
    log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class _RegexDetector:
    name: str
    reason: str
    pattern: "re.Pattern[str]"
    span_group: Optional[str] = None  # named group to redact instead of whole match
    sentinel: "Optional[re.Pattern[str]]" = None  # cheap pre-filter; None = always run
    sentinel_min_len: int = 0


@dataclass
class _ValueKeyDetector:
    name: str
    reason: str
    keys: frozenset  # lowercased key names
    substrings: Tuple[str, ...]  # lowercased key-path substrings


class TraceRedactor:
    """Pluggable credential/PII redactor driven by a YAML detector config."""

    def __init__(self, config: Dict[str, Any], *, config_source: str = "explicit") -> None:
        self.config_source = config_source
        reasons = config.get("detector_reasons") or {}
        self._regex_detectors: List[_RegexDetector] = []
        self._value_key_detectors: List[_ValueKeyDetector] = []
        self._luhn_max_digits = int(config.get("luhn_max_digits", 19))
        # Minimum digit count before Luhn is even attempted; below this a
        # number is too short to be a card (schema says 13-19 digits).
        self._luhn_check_digit_floor = int(config.get("luhn_check_digit_floor", 4))
        flags = re.DOTALL | re.MULTILINE

        for name, spec in (config.get("detectors") or {}).items():
            reason = reasons.get(name)
            if reason not in ("credential", "pii"):
                raise RedactionError(
                    "detector %r missing valid reason mapping in detector_reasons" % name
                )
            sentinel_re: "Optional[re.Pattern[str]]" = None
            sentinel_min_len = 0
            sentinel_spec = spec.get("sentinel")
            if sentinel_spec:
                sentinel_re = re.compile(sentinel_spec["check"], flags)
                sentinel_min_len = int(sentinel_spec.get("min_len", 0))
            patterns = spec.get("patterns") or []
            for pat in patterns:
                try:
                    compiled = re.compile(pat, flags)
                except re.error as exc:
                    raise RedactionError(
                        "detector %r has invalid pattern %r: %s" % (name, pat, exc)
                    ) from exc
                self._regex_detectors.append(
                    _RegexDetector(
                        name=name,
                        reason=reason,
                        pattern=compiled,
                        span_group=spec.get("span_group"),
                        sentinel=sentinel_re,
                        sentinel_min_len=sentinel_min_len,
                    )
                )

        for name, spec in (config.get("value_key_detectors") or {}).items():
            reason = reasons.get(name)
            if reason not in ("credential", "pii"):
                raise RedactionError(
                    "value-key detector %r missing valid reason mapping" % name
                )
            keys = frozenset(k.lower() for k in (spec.get("keys") or []))
            substrings = tuple(s.lower() for s in (spec.get("substrings") or []))
            self._value_key_detectors.append(
                _ValueKeyDetector(name=name, reason=reason, keys=keys, substrings=substrings)
            )

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    @classmethod
    def from_default_config(cls, path: Optional[str] = None) -> "TraceRedactor":
        """Load config from (in order) explicit path, env var HERMES_TRACE_REDACTION_CONFIG,
        ~/.hermes/trace_redaction.yaml, or the packaged default shipped with the schema."""
        import yaml  # pyyaml, a declared dependency

        candidates = []
        if path:
            candidates.append(path)
        env_path = os.environ.get("HERMES_TRACE_REDACTION_CONFIG")
        if env_path:
            candidates.append(env_path)
        candidates.append(_DEFAULT_CONFIG_PATH)
        candidates.append(str(_PACKAGED_DEFAULT_CONFIG))

        last_err: Optional[Exception] = None
        for cand in candidates:
            p = Path(os.path.expanduser(cand))
            if not p.is_file():
                continue
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise RedactionError("redaction config %s is not a mapping" % p)
                return cls(data, config_source=str(p))
            except Exception as exc:  # broken config file — try the next one
                logger.warning("trace redactor: failed to load config %s: %s", p, exc)
                last_err = exc
        if last_err is not None:
            raise RedactionError("no usable redaction config found: %s" % last_err)
        raise RedactionError("no redaction config found in %s" % candidates)

    # ------------------------------------------------------------------
    # Public hook
    # ------------------------------------------------------------------
    def redact(self, value: Any, *, context: Optional[RedactionContext] = None) -> Any:
        """Hook entrypoint. Never raises; fail-closed on any internal error."""
        ctx = context or RedactionContext()
        try:
            return self._redact(value, ctx)
        except Exception as exc:  # fail closed, per schema S4.1
            logger.exception(
                "trace redactor failure at path %r; replacing value with redactor_error marker",
                ctx.path,
            )
            ctx.log.append(
                {"path": ctx.path or "<root>", "reason": "redactor_error", "sha256": None}
            )
            return make_marker("redactor_error", None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _redact(self, value: Any, ctx: RedactionContext) -> Any:
        if ctx.always_safe:
            return value
        if value is None or isinstance(value, bool) or isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            return self._redact_string(value, ctx)
        if isinstance(value, list):
            out = []
            for i, item in enumerate(value):
                out.append(
                    self._redact(
                        item,
                        RedactionContext(
                            path="%s[%d]" % (ctx.path, i) if ctx.path else "[%d]" % i,
                            always_safe=False,
                            log=ctx.log,
                        ),
                    )
                )
            return out
        if isinstance(value, dict):
            if _is_marker(value):
                return dict(value)  # already redacted; pass through intact
            out: Dict[str, Any] = {}
            for k, v in value.items():
                key_str = str(k)
                child_path = "%s.%s" % (ctx.path, key_str) if ctx.path else key_str
                if key_str.lower() in ALWAYS_SAFE_KEYS:
                    out[k] = v  # always-safe structural keys: untouched, not recursed
                    continue
                # Whole-value redaction when the key matches a value-key detector.
                vkd = self._match_value_key(key_str, child_path)
                if vkd is not None and isinstance(v, str) and v != "":
                    digest = _sha256_hex(v)
                    out[k] = make_marker(vkd.reason, digest)
                    ctx.log.append(
                        {"path": child_path, "reason": vkd.reason, "sha256": digest}
                    )
                    continue
                out[k] = self._redact(
                    v, RedactionContext(path=child_path, always_safe=False, log=ctx.log)
                )
            return out
        # Non-JSON types (shouldn't occur in traces) — stringify then scan.
        return self._redact_string(str(value), ctx)

    def _match_value_key(self, key: str, path: str) -> Optional[_ValueKeyDetector]:
        kl = key.lower()
        pl = path.lower()
        for det in self._value_key_detectors:
            if kl in det.keys:
                return det
            if any(sub in pl for sub in det.substrings):
                return det
        return None

    def _redact_string(self, s: str, ctx: RedactionContext) -> Any:
        if s == "":
            return s
        try:
            parsed = json.loads(s)
        except (ValueError, TypeError):
            parsed = None
        else:
            # The string is itself a JSON document (e.g. a tool result or a
            # message carrying serialized JSON). Redact as a structure so
            # key-shaped credentials ("password": "...") are caught, then
            # re-serialize compactly to keep shape for downstream assembly.
            if isinstance(parsed, (dict, list)):
                redacted = self._redact(
                    parsed,
                    RedactionContext(path=ctx.path, always_safe=False, log=ctx.log),
                )
                if redacted != parsed:
                    return json.dumps(redacted, separators=(",", ":"), ensure_ascii=False)
                return s

        matches: List[Tuple[int, int, str, str]] = []  # start, end, reason, span_text
        for det in self._iter_applicable_detectors(s):
            for m in det.pattern.finditer(s):
                if det.span_group and det.span_group in m.groupdict():
                    start, end = m.span(det.span_group)
                else:
                    start, end = m.span(0)
                span_text = s[start:end]
                matches.append((start, end, det.reason, span_text))

        # Card numbers: digit-run scan with Luhn validation (not expressible
        # well as a single regex, and the schema demands Luhn-valid hits).
        # Runs only when a 13+ digit run is present.
        if self._has_card_candidate(s):
            matches.extend(self._luhn_matches(s))

        if not matches:
            return s

        matches.sort(key=lambda t: (t[0], -(t[1] - t[0])))
        non_overlapping: List[Tuple[int, int, str, str]] = []
        last_end = -1
        for start, end, reason, span_text in matches:
            if start < last_end:
                continue
            non_overlapping.append((start, end, reason, span_text))
            last_end = end

        # Whole-string single match -> marker object (schema-permitted type change).
        if (
            len(non_overlapping) == 1
            and non_overlapping[0][0] == 0
            and non_overlapping[0][1] == len(s)
        ):
            _, _, reason, span_text = non_overlapping[0]
            digest = _sha256_hex(span_text)
            ctx.log.append({"path": ctx.path or "<root>", "reason": reason, "sha256": digest})
            return make_marker(reason, digest)

        out_parts: List[str] = []
        cursor = 0
        for start, end, reason, span_text in non_overlapping:
            digest = _sha256_hex(span_text)
            out_parts.append(s[cursor:start])
            out_parts.append(SPAN_PLACEHOLDER_FMT.format(reason=reason, digest12=digest[:12]))
            ctx.log.append({"path": ctx.path or "<root>", "reason": reason, "sha256": digest})
            cursor = end
        out_parts.append(s[cursor:])
        return "".join(out_parts)

    def _iter_applicable_detectors(self, s: str) -> List[_RegexDetector]:
        out: List[_RegexDetector] = []
        for det in self._regex_detectors:
            if det.sentinel is not None:
                if len(s) < det.sentinel_min_len or not det.sentinel.search(s):
                    continue
            out.append(det)
        return out

    _CARD_CANDIDATE_RE = re.compile(r"[0-9][0-9 \-]{11,}[0-9]")

    def _has_card_candidate(self, s: str) -> bool:
        # Cheap gate: only spend the Luhn scan when a run capable of holding
        # 13+ digits exists (13 digits => 11 separator-or-digit chars between
        # the first and last digit, minimum).
        return bool(self._CARD_CANDIDATE_RE.search(s))

    def _luhn_matches(self, s: str) -> List[Tuple[int, int, str, str]]:
        out: List[Tuple[int, int, str, str]] = []
        # Candidate runs: digits possibly interrupted by spaces/dashes.
        for m in re.finditer(r"[0-9][0-9 \-]{8,}[0-9]", s):
            run = m.group(0)
            digits = re.sub(r"\D", "", run)
            if len(digits) < 13 or len(digits) > self._luhn_max_digits:
                continue
            if _luhn_ok(digits):
                out.append((m.start(), m.end(), "pii", run))
        return out


def _luhn_ok(digits: str) -> bool:
    total = 0
    reverse = digits[::-1]
    for i, ch in enumerate(reverse):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ----------------------------------------------------------------------
# Module-level convenience: a lazily-built shared redactor.
# ----------------------------------------------------------------------
_SHARED: Optional[TraceRedactor] = None


def get_default_redactor() -> TraceRedactor:
    global _SHARED
    if _SHARED is None:
        _SHARED = TraceRedactor.from_default_config()
    return _SHARED


def redact(value: Any, *, context: Optional[RedactionContext] = None) -> Any:
    """Convenience wrapper matching the S4.1 hook signature exactly."""
    return get_default_redactor().redact(value, context=context)
