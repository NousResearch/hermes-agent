"""Compresr tool-output compression plugin for Hermes.

Compresses large tool outputs (verbose grep/read_file/execute_code dumps) as
they arrive, on the ``transform_tool_result`` hook, using Compresr's
query-specific tool-output API. The compressed text is passed through unchanged
and a footer points the agent at the full verbatim original in Hermes's managed
cache, recoverable with a plain ``read_file``/``search_files``. Fail-open.

Per-turn complement to ``plugins/context_engine/compresr`` (compaction time):
this pre-filter shrinks each output once so compaction has less to summarize.

Requires ``COMPRESR_API_KEY`` in ``.env``; the plugin stays inert until listed
in ``plugins.enabled``. See ``README.md`` for the full config/env reference.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from hermes_constants import get_hermes_home
from plugins.compresr_common import (
    sanitize_secret as _sanitize_secret,
    secure_base_url as _secure_base_url,
)

from . import cache
from .client import CompresrToolOutputClient, DEFAULT_TOOL_OUTPUT_MODEL
from .compress import FOOTER_MARKER, compress_tool_output, count_tokens

try:
    from agent.redact import redact_sensitive_text as _redact
except Exception:
    def _redact(s: str) -> str:  # type: ignore[misc]
        return s

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.compresr.ai/api"
_DEFAULT_MIN_TOKENS = 1500
_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_CACHE_MB = 256
_DEFAULT_TARGET_RATIO = 2.0
_COOLDOWN_SECONDS = 30.0                # back-off after an API error before retrying
_MAX_QUERY_CHARS = 600                  # cap on the derived query sent to the API
_CACHE_ID_LEN = 32                      # hex chars of the content hash used as cache id (128 bits)
_OK_STATUSES = ("", "ok", "success")    # statuses we compress; others pass through
_FALLBACK_QUERY = (
    "Preserve the facts, paths, identifiers, errors, and results in this tool "
    "output that are needed to continue the task."
)
# Args we mine, in priority order, to reconstruct the tool's intent as a query.
_QUERY_ARG_KEYS = ("query", "pattern", "command", "q", "search", "regex", "url")
_PATH_ARG_KEYS = ("file_path", "path", "file", "filename", "directory")


# Full path segment every recovery target carries (host + container-translated).
# Matching the whole segment avoids misreading an incidental mention as a recovery
# read. Kept in sync with cache._CACHE_SUBDIR.
_CACHE_PATH_MARKER = "cache/compresr/tool-output"

# Tool → key holding the plain-text payload inside a JSON envelope. Unwrapping
# compresses the real content (not JSON syntax) and caches readable text.
_UNWRAPPABLE_JSON_TOOLS: Dict[str, str] = {
    "read_file": "content",
    "terminal": "output",
    "execute_code": "output",
    "search_files": "matches_text",
}

# Tools whose unwrapped payload is already line-numbered ("N|code"). We cache a
# de-numbered copy so a recovery read_file re-adds exactly one clean gutter.
_NUMBERED_JSON_TOOLS = {"read_file"}
_LINE_GUTTER_RE = re.compile(r"^\d+\|")


def _max_recoverable_line_length() -> int:
    """The per-line cap the recovery read_file/search_files apply. Any cached
    line longer than this is silently clamped on recovery (with no offset to
    reach the tail), so an output containing one can't be recovered byte-exact.
    """
    try:
        from tools.tool_output_limits import get_max_line_length

        return int(get_max_line_length())
    except Exception:
        return 2000


def _has_unrecoverable_long_line(text: str) -> bool:
    limit = _max_recoverable_line_length()
    return any(len(ln) > limit for ln in text.split("\n"))


def _strip_line_gutter(text: str) -> str:
    return "\n".join(_LINE_GUTTER_RE.sub("", ln) for ln in text.split("\n"))


def _is_fully_guttered(text: str) -> bool:
    """True only if every non-blank line carries an ``N|`` gutter, so stripping
    can't mangle a payload that merely happens to contain a ``|``."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return bool(lines) and all(_LINE_GUTTER_RE.match(ln) for ln in lines)


def _try_unwrap_json_tool_result(
    tool_name: str, result: str
) -> Tuple[Optional[str], Optional[Callable[[str], str]]]:
    key = _UNWRAPPABLE_JSON_TOOLS.get(tool_name)
    if key is None:
        return None, None
    if not result.lstrip().startswith("{"):
        return None, None
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, ValueError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    inner = parsed.get(key)
    if not isinstance(inner, str) or not inner.strip():
        return None, None

    def splice(new_inner: str) -> str:
        out = dict(parsed)
        out[key] = new_inner
        return json.dumps(out, ensure_ascii=False)

    return inner, splice


def _read_config_block() -> Dict[str, Any]:
    """Best-effort read of the ``compresr:`` block from config.yaml. Never raises."""
    cfg_path = get_hermes_home() / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml

        with open(cfg_path, encoding="utf-8-sig") as f:
            data = yaml.safe_load(f) or {}
        block = data.get("compresr", {})
        return block if isinstance(block, dict) else {}
    except Exception as e:  # pragma: no cover - config is optional
        logger.debug("tool_output_compresr: could not read config block: %s", e)
        return {}


def _as_bool(v: Any) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on")


def _as_int(v: Any, default: int) -> int:
    """Tolerant int coercion — a malformed config/env value (e.g. a typo'd
    ``COMPRESR_TOOL_OUTPUT_TIMEOUT``) falls back to *default* instead of raising
    in ``__init__`` and silently disabling the whole plugin."""
    try:
        return int(v)
    except (TypeError, ValueError):
        logger.warning("tool_output_compresr: invalid numeric value %r, using %s", v, default)
        return default


def _as_float(v: Any, default: float) -> float:
    """Tolerant float coercion (see :func:`_as_int`)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        logger.warning("tool_output_compresr: invalid numeric value %r, using %s", v, default)
        return default


class ToolOutputCompressor:
    """Holds config + clients and implements the transform hook."""

    def __init__(self) -> None:
        cfg = _read_config_block()

        def _opt(env_key: str, cfg_key: str, default: Any) -> Any:
            val = os.environ.get(env_key)
            if val is not None and val != "":
                return val
            if cfg_key in cfg and cfg[cfg_key] not in (None, ""):
                return cfg[cfg_key]
            return default

        self.api_key = _sanitize_secret(
            os.environ.get("COMPRESR_API_KEY", ""), "COMPRESR_API_KEY"
        )
        self.base_url = _secure_base_url(
            str(_opt("COMPRESR_BASE_URL", "base_url", _DEFAULT_BASE_URL)).rstrip("/"),
            _DEFAULT_BASE_URL,
        )
        self.enabled = _as_bool(
            _opt("COMPRESR_TOOL_OUTPUT_ENABLED", "tool_output_enabled", "")
        )
        self.model = str(
            _opt("COMPRESR_TOOL_OUTPUT_MODEL", "tool_output_model", DEFAULT_TOOL_OUTPUT_MODEL)
        )
        self.min_tokens = _as_int(
            _opt("COMPRESR_TOOL_OUTPUT_MIN_TOKENS", "tool_output_min_tokens", _DEFAULT_MIN_TOKENS),
            _DEFAULT_MIN_TOKENS,
        )
        self.timeout = max(
            1,
            _as_int(
                _opt("COMPRESR_TOOL_OUTPUT_TIMEOUT", "tool_output_timeout", _DEFAULT_TIMEOUT),
                _DEFAULT_TIMEOUT,
            ),
        )
        self.max_cache_mb = _as_int(
            _opt(
                "COMPRESR_TOOL_OUTPUT_MAX_CACHE_MB",
                "tool_output_max_cache_mb",
                _DEFAULT_MAX_CACHE_MB,
            ),
            _DEFAULT_MAX_CACHE_MB,
        )
        self.target_ratio = _as_float(
            _opt(
                "COMPRESR_TOOL_OUTPUT_TARGET_RATIO",
                "tool_output_target_ratio",
                _DEFAULT_TARGET_RATIO,
            ),
            _DEFAULT_TARGET_RATIO,
        )

        self._client = CompresrToolOutputClient(
            api_key=self.api_key, base_url=self.base_url, model=self.model, timeout=self.timeout
        )
        # Cumulative stats for /usage and benchmarking.
        self.calls = 0
        self.errors = 0
        self.tokens_in = 0       # original tokens across compressed outputs
        self.tokens_saved = 0
        self.recoveries = 0
        self._cooldown_until = 0.0

    # -- helpers -----------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.api_key)

    @staticmethod
    def _derive_query(tool_name: str, args: Any) -> str:
        """Reconstruct the tool's intent as a Compresr query.

        For search-class tools (grep/web_search) the args ARE an excellent query;
        for read/exec tools they are a weaker but still useful proxy.
        """
        if isinstance(args, dict):
            for k in _QUERY_ARG_KEYS:
                v = args.get(k)
                if isinstance(v, str) and v.strip():
                    return f"{tool_name}: {v.strip()}"[:_MAX_QUERY_CHARS]
            for k in _PATH_ARG_KEYS:
                v = args.get(k)
                if isinstance(v, str) and v.strip():
                    return f"Relevant content of {v.strip()} for the current task"[:_MAX_QUERY_CHARS]
        if tool_name:
            return f"Relevant output of the {tool_name} call for the current task"
        return _FALLBACK_QUERY

    @staticmethod
    def _is_recovery_read(args: Any) -> bool:
        """True if *args* points at a file under our compresr cache.

        A recovery ``read_file`` (or ``search_files``) targeting a cached
        original must NOT be re-compressed: the cached file is the verbatim
        source and is >= min_tokens but carries no FOOTER_MARKER, so without
        this guard the hook re-fires and returns a lossy summary instead of the
        exact original the agent asked to recover.

        Backend-agnostic: the cache path always contains the segment
        ``cache/compresr/tool-output`` — true for the host path AND the
        container-translated ``/root/.hermes/...`` path. We first check for that
        full segment (which remote/translated paths still carry), then fall back
        to a resolved comparison against the real cache root.
        """
        if not isinstance(args, dict):
            return False
        try:
            cache_root = str(cache.get_cache_root().resolve())
        except Exception:
            cache_root = ""
        for v in args.values():
            if not isinstance(v, str) or not v:
                continue
            if _CACHE_PATH_MARKER in v:
                return True
            if cache_root:
                try:
                    if str(Path(v).resolve()).startswith(cache_root):
                        return True
                except (OSError, ValueError):
                    pass
        return False

    @staticmethod
    def _cache_id(content: str) -> str:
        # Hash the CONTENT (not the tool_call_id) so two different outputs can
        # never collide onto one cache file; identical outputs safely dedupe.
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:_CACHE_ID_LEN]

    # -- the hook ----------------------------------------------------------

    def on_transform_tool_result(
        self,
        tool_name: str = "",
        args: Any = None,
        result: Any = None,
        task_id: str = "",
        tool_call_id: str = "",
        status: str = "",
        **_: Any,
    ) -> Optional[str]:
        """Return a compressed replacement string, or None to leave unchanged.

        Fail-open: any error, cooldown, small output, or error-status result
        returns None so the model sees the original.
        """
        if not self.active or not isinstance(result, str):
            return None
        if status and status not in _OK_STATUSES:
            return None  # don't mangle error results
        if FOOTER_MARKER in result:
            return None  # already compressed by us — idempotent
        if self._is_recovery_read(args):
            # The agent is recovering a cached original via read_file/search_files;
            # return it verbatim rather than re-compressing it into a lossy summary.
            self.recoveries += 1
            return None
        if count_tokens(result) < self.min_tokens:
            return None  # too small to be worth a round-trip
        now = time.monotonic()
        if now < self._cooldown_until:
            return None

        # Redact secrets in tool args (curl -H 'Authorization: Bearer ...' etc)
        # before the query string leaves the process to a third-party API.
        query = _redact(self._derive_query(tool_name, args))

        inner_text, splice = _try_unwrap_json_tool_result(tool_name, result)
        compress_target = inner_text if inner_text is not None else result
        if inner_text is not None and count_tokens(inner_text) < self.min_tokens:
            return None

        cache_content = compress_target
        if (
            inner_text is not None
            and tool_name in _NUMBERED_JSON_TOOLS
            and _is_fully_guttered(inner_text)
        ):
            cache_content = _strip_line_gutter(inner_text)

        if _has_unrecoverable_long_line(cache_content):
            return None

        # Redact outbound + cached content. The cache is a durable secret-bearing
        # artefact on disk; content-address AFTER redaction so identical
        # redacted content dedupes.
        compress_target = _redact(compress_target)
        cache_content = _redact(cache_content)

        cache_id = self._cache_id(compress_target)
        try:
            out, info = compress_tool_output(
                query=query,
                content=compress_target,
                cache_content=cache_content,
                tool_name=tool_name,
                cache_id=cache_id,
                client=self._client,
                task_id=task_id,
                max_cache_mb=self.max_cache_mb,
                target_ratio=self.target_ratio,
            )
        except Exception as e:  # compress is already fail-open, but be defensive
            self.errors += 1
            self._cooldown_until = time.monotonic() + _COOLDOWN_SECONDS
            logger.warning("tool_output_compresr: hook error (%s)", e)
            return None

        if info.get("error"):
            # Genuine failures (API/transport error, empty output, cache-write
            # failure) arm the anti-hammer cooldown. A benign "no net win" uses
            # skipped_reason instead, so it can't suppress all later outputs.
            self.errors += 1
            self._cooldown_until = time.monotonic() + _COOLDOWN_SECONDS
        if not info.get("called_api"):
            return None  # API failed → leave the original output unchanged
        if not info.get("shortened"):
            # The API didn't produce a meaningfully shorter output (or the cache
            # write / backend-visibility check failed) — leave the original
            # unchanged rather than spend context on a footer for no gain.
            return None
        saved = max(0, info.get("base_tokens", 0) - info.get("out_tokens", 0))
        self.calls += 1
        self.tokens_in += info.get("base_tokens", 0)
        self.tokens_saved += saved
        logger.info(
            "tool_output_compresr: %s %d→%d tokens (saved %d, unwrapped=%s, cache=%s)",
            tool_name,
            info.get("base_tokens", 0),
            info.get("out_tokens", 0),
            saved,
            splice is not None,
            info.get("cache_path", "?"),
        )
        if splice is not None:
            # Fold the footer into the JSON envelope's inner text so the result
            # stays valid JSON for the tool that produced it.
            return splice(out)
        return out

    def get_status(self) -> Dict[str, Any]:
        return {
            "plugin": "tool_output_compresr",
            "active": self.active,
            "model": self.model,
            "min_tokens": self.min_tokens,
            "max_cache_mb": self.max_cache_mb,
            "calls": self.calls,
            "errors": self.errors,
            "tokens_in": self.tokens_in,
            "tokens_saved": self.tokens_saved,
            "recoveries": self.recoveries,
        }


def register(ctx: Any) -> None:
    """Plugin entry point — called by the Hermes plugin loader."""
    try:
        cache.ensure_cache_root()
    except Exception as e:
        logger.warning("tool_output_compresr: could not initialize cache root: %s", e)
    compressor = ToolOutputCompressor()
    ctx.register_hook("transform_tool_result", compressor.on_transform_tool_result)
    if not compressor.api_key:
        logger.info(
            "tool_output_compresr: loaded but COMPRESR_API_KEY is unset — inactive."
        )
    elif not compressor.enabled:
        logger.info(
            "tool_output_compresr: loaded but disabled — set compresr.tool_output_enabled: true."
        )
