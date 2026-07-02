"""LCM configuration with defaults and env var overrides."""
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - optional fallback for minimal installs
    yaml = None


def _parse_pattern_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_env(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _parse_float_env(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _parse_bool_env(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _config_bool_disabled(value) -> bool:
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            return True
        try:
            return float(normalized) == 0
        except ValueError:
            return False
    return False


def _hermes_compression_threshold(default: float) -> float:
    """Read lcm.context_threshold or Hermes compression.threshold from config.yaml.

    Priority when no ``LCM_CONTEXT_THRESHOLD`` env var is set:
      1. ``lcm.context_threshold`` (LCM-specific override in config.yaml)
      2. ``compression.threshold`` (Hermes global setting, unless compression disabled)

    Hermes gateways may load ``~/.hermes/config.yaml`` without exporting every
    setting into the process environment. The ``lcm.context_threshold`` key lets
    operators tune LCM compaction independently of the Hermes compression setting.
    Disabled Hermes compression should not leak its threshold into LCM.
    """
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    cfg_path = home / "config.yaml"
    try:
        text = cfg_path.read_text()
        if yaml is not None:
            cfg = yaml.safe_load(text) or {}
            # lcm.context_threshold takes priority over compression.threshold
            lcm_val = (cfg.get("lcm") or {}).get("context_threshold")
            if lcm_val is not None:
                return float(lcm_val)
            compression = cfg.get("compression") or {}
            if _config_bool_disabled(compression.get("enabled")):
                return default
            comp_val = compression.get("threshold")
            if comp_val is not None:
                return float(comp_val)
            return default

        in_lcm = False
        lcm_indent = None
        in_compression = False
        comp_indent = None
        compression_disabled = False
        threshold_value = None
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith((" ", "\t")):
                stripped = line.strip()
                in_lcm = stripped == "lcm:"
                in_compression = stripped == "compression:"
                lcm_indent = None
                comp_indent = None
                continue
            indent = len(line) - len(line.lstrip(" \t"))
            if in_lcm:
                if lcm_indent is None:
                    lcm_indent = indent
                if indent == lcm_indent and ":" in line:
                    key, raw_value = line.strip().split(":", 1)
                    if key == "context_threshold":
                        return float(raw_value.strip().strip("'\""))
                continue
            if in_compression:
                if comp_indent is None:
                    comp_indent = indent
                if indent != comp_indent or ":" not in line:
                    continue
                key, raw_value = line.strip().split(":", 1)
                value = raw_value.strip().strip("'\"")
                if key == "enabled" and _config_bool_disabled(value):
                    compression_disabled = True
                elif key == "threshold":
                    threshold_value = value
        if compression_disabled or threshold_value is None:
            return default
        return float(threshold_value)
    except Exception:
        return default


def _hermes_compression_float(key: str, default: float) -> float:
    """Read ``compression.<key>`` (a float) from ~/.hermes/config.yaml.

    Used for the P2 calibration knobs (``skew_floor``, ``calibration_hard_frac``)
    so the LCM engine — a process-global singleton — sources them ONCE from config
    at construction, rather than having each agent_init mutate the shared instance
    (which would let one agent's config silently change another's calibration —
    Greptile PR #111). Returns ``default`` on any read/parse failure or absence.
    """
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    cfg_path = home / "config.yaml"
    try:
        if yaml is None:
            return default
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        compression = cfg.get("compression") or {}
        val = compression.get(key)
        if val is None:
            return default
        val = float(val)
        return val if 0.0 < val <= 1.0 else default
    except Exception:
        return default


def _hermes_lcm_value(key: str):
    """Read ``lcm.<key>`` from ~/.hermes/config.yaml; None on absence/failure."""
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    cfg_path = home / "config.yaml"
    try:
        if yaml is None:
            return None
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        return (cfg.get("lcm") or {}).get(key)
    except Exception:
        return None


def _hermes_lcm_int(key: str, default: int) -> int:
    """Read ``lcm.<key>`` as a non-negative int; ``default`` on any failure."""
    val = _hermes_lcm_value(key)
    if val is None:
        return default
    try:
        parsed = int(val)
        return parsed if parsed >= 0 else default
    except (TypeError, ValueError):
        return default


def _lcm_config_bool(env_key: str, cfg_key: str, default: bool) -> bool:
    """Resolve a boolean knob: env override → lcm.<cfg_key> config → default.

    Fail-safe: only explicit falsy values ("0", "false", "no", "off") disable;
    unrecognized/garbage values keep the default (same doctrine as
    LCM_IDENTIFIER_FIDELITY — never silently drop a production behavior on a
    typo).
    """
    falsy = {"0", "false", "no", "off"}
    truthy = {"1", "true", "yes", "on"}

    raw_env = os.environ.get(env_key)
    if raw_env is not None:
        lowered = raw_env.strip().lower()
        if lowered in falsy:
            return False
        if lowered in truthy:
            return True
        return default

    val = _hermes_lcm_value(cfg_key)
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    lowered = str(val).strip().lower()
    if lowered in falsy:
        return False
    if lowered in truthy:
        return True
    return default


def _hermes_auxiliary_compression_timeout_ms(default: int) -> int:
    """Read Hermes auxiliary.compression.timeout when no LCM override is present.

    Hermes uses seconds for the auxiliary compression timeout, while LCM stores
    the summary timeout in milliseconds. Aligning the default keeps LCM summary
    calls from timing out earlier than the host compression route unless
    ``LCM_SUMMARY_TIMEOUT_MS`` is explicitly configured.
    """
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    cfg_path = home / "config.yaml"
    try:
        text = cfg_path.read_text()
        if yaml is not None:
            cfg = yaml.safe_load(text) or {}
            auxiliary = cfg.get("auxiliary") or {}
            compression = auxiliary.get("compression") or {}
            value = compression.get("timeout")
            if value is None:
                return default
            return int(float(value) * 1000)

        in_auxiliary = False
        in_compression = False
        auxiliary_indent = None
        compression_indent = None
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" \t"))
            stripped = line.strip()
            if indent == 0:
                in_auxiliary = stripped == "auxiliary:"
                in_compression = False
                auxiliary_indent = None
                compression_indent = None
                continue
            if not in_auxiliary:
                continue
            if auxiliary_indent is None:
                auxiliary_indent = indent
            if indent == auxiliary_indent:
                if stripped == "compression:":
                    in_compression = True
                    compression_indent = None
                    continue
                in_compression = False
                compression_indent = None
                continue
            if not in_compression:
                continue
            if compression_indent is None:
                compression_indent = indent
            if indent != compression_indent or ":" not in stripped:
                continue
            key, raw_value = stripped.split(":", 1)
            if key == "timeout":
                return int(float(raw_value.strip().strip("'\"")) * 1000)
        return default
    except Exception:
        return default


@dataclass
class LCMConfig:
    """All tunables for the LCM engine."""

    # -- Fresh tail: recent messages never compacted ---
    fresh_tail_count: int = 32
    # Token-budgeted fresh tail (compression.target_ratio support).
    # When enabled and a context window is known, the fresh tail is sized
    # dynamically: keep the most recent messages whose estimated tokens fit
    # ``target_ratio × threshold_tokens`` (floored at ``fresh_tail_count``
    # messages, capped at ``fresh_tail_max_tokens``). Disabled or degenerate
    # inputs reproduce the legacy fixed-count tail exactly.
    fresh_tail_token_budget_enabled: bool = True
    # 0 = derive from target_ratio × threshold_tokens; >0 = explicit budget.
    fresh_tail_token_budget: int = 0
    # Hard cap on the derived/explicit budget (guards 1M-window models).
    fresh_tail_max_tokens: int = 60_000
    # Fleet-standard kept-tail ratio, sourced from compression.target_ratio
    # (config.yaml) exactly like skew_floor — the LCM engine is a process-global
    # singleton, so this is read ONCE at construction (Greptile #111 discipline).
    # NOTE: the fleet key is compression.target_ratio; an ``lcm.target_ratio``
    # key is intentionally NOT read.
    target_ratio: float = 0.20

    # -- Compaction thresholds ---
    # Max source tokens in a leaf chunk before summarization triggers
    leaf_chunk_tokens: int = 20_000
    # Fraction of context window that triggers compaction (0.0–1.0)
    context_threshold: float = 0.35
    # Max condensation depth (-1 = unlimited, 0 = leaf only)
    incremental_max_depth: int = 3
    # How many same-depth summaries trigger condensation
    condensation_fanin: int = 4
    # When enabled, leaf compaction may use a larger working chunk size based on backlog pressure
    dynamic_leaf_chunk_enabled: bool = False
    # Upper bound for the working dynamic leaf chunk threshold
    dynamic_leaf_chunk_max: int = 40_000
    # When enabled, suppress follow-on condensation after a leaf pass unless
    # debt/pressure says the extra churn is worth it
    cache_friendly_condensation_enabled: bool = False
    # Minimum number of same-depth fanin groups before one follow-on
    # condensation pass is allowed in cache-friendly mode
    cache_friendly_min_debt_groups: int = 2
    # When enabled, turns can persist raw-backlog maintenance debt and use
    # later bounded catch-up passes to reduce it.
    deferred_maintenance_enabled: bool = False
    # Maximum extra leaf passes a debt-triggered later turn may spend on
    # catch-up work.
    deferred_maintenance_max_passes: int = 4
    # Disabled at 0.0. When set, only bypass cache-friendly/deferred polite
    # gates once prompt pressure reaches this fraction of the context window.
    critical_budget_pressure_ratio: float = 0.0

    # -- P2 "compact on the truth" calibration (shared via ContextEngine ABC) ---
    # Lower clamp on the measured real/rough skew (never scale an estimate below
    # this fraction). Sourced from compression.skew_floor.
    skew_floor: float = 0.7
    # Raw-rough window fraction at which compaction fires regardless of skew
    # (dense-paste / 413 ceiling). Sourced from compression.calibration_hard_frac.
    calibration_hard_frac: float = 0.95

    # -- Escalation ---
    # L2 bullet budget as fraction of L1
    l2_budget_ratio: float = 0.50
    # L3 deterministic truncate token limit
    l3_truncate_tokens: int = 512

    # -- Assembly guardrails ---
    # Hard cap for the assembled active context (0 = disabled)
    max_assembly_tokens: int = 0
    # Reserve this many tokens from the model context window before assembly
    # (0 = disabled). Effective cap becomes context_length - reserve_tokens_floor.
    reserve_tokens_floor: int = 0

    # -- Session and message filtering ---
    # Sessions to exclude from LCM storage entirely.
    ignore_session_patterns: list[str] = field(default_factory=list)
    # Sessions that may read carried-over LCM state but never write new data.
    stateless_session_patterns: list[str] = field(default_factory=list)
    # Per-message regex patterns; matching messages are skipped before LCM storage.
    ignore_message_patterns: list[str] = field(default_factory=list)
    # Diagnostics: where each pattern list came from.
    ignore_session_patterns_source: str = "default"
    stateless_session_patterns_source: str = "default"
    ignore_message_patterns_source: str = "default"

    # -- Summary instructions ---
    # Custom instructions injected into all summarization prompts
    custom_instructions: str = ""

    # -- Pre-compaction extraction ---
    # Extract decisions/commitments to files before compaction
    extraction_enabled: bool = False
    # Model for extraction (empty = fall back to summary_model)
    extraction_model: str = ""
    # Directory for daily extraction files (empty = auto: ~/.hermes/lcm-extractions/)
    extraction_output_path: str = ""

    # -- Sensitive-pattern handling ---
    # Disabled by default. When enabled, named patterns redact matching secrets
    # before LCM storage, FTS indexing, summarization, or externalization.
    sensitive_patterns_enabled: bool = False
    # Named pattern catalog entries to apply when sensitive handling is enabled.
    sensitive_patterns: list[str] = field(
        default_factory=lambda: ["api_key", "bearer_token", "password_assignment", "private_key"]
    )
    # Diagnostics: where the sensitive pattern list came from.
    sensitive_patterns_source: str = "default"

    # -- Large tool-output externalization ---
    # When enabled, oversized tool results are written to plugin-managed storage
    # and replaced with compact references in pre-compaction serializer input.
    large_output_externalization_enabled: bool = False
    # Character threshold above which tool results are externalized.
    large_output_externalization_threshold_chars: int = 12_000
    # Explicit storage directory for externalized payloads (empty = auto under hermes home).
    large_output_externalization_path: str = ""
    # When enabled, already-externalized summarized tool-result transcript rows may
    # be rewritten to compact GC placeholders after successful leaf compaction.
    large_output_transcript_gc_enabled: bool = False

    # -- Models ---
    summary_model: str = ""       # empty = use Hermes auxiliary model
    # Optional fallback summary models tried after summary_model/task default.
    summary_fallback_models: list[str] = field(default_factory=list)
    # Consecutive failed summary calls before a route is skipped temporarily.
    summary_circuit_breaker_failure_threshold: int = 2
    # Seconds to skip an open summary route before allowing a retry.
    summary_circuit_breaker_cooldown_seconds: int = 300
    expansion_model: str = ""     # empty = fall back to summary_model / Hermes auxiliary model
    # Serialized summary/raw/child-source/externalized context budget fed to lcm_expand_query's auxiliary LLM before it returns a bounded answer.
    expansion_context_tokens: int = 32_000

    # -- Timeouts ---
    summary_timeout_ms: int = 60_000
    expansion_timeout_ms: int = 120_000

    # -- Storage ---
    database_path: str = ""       # empty = HERMES_HOME/lcm.db; LCM_DATABASE_PATH may override
    # Optional per-row AEAD for raw message columns. When enabled, cryptography
    # must be installed and the profile-local key file is created with 0600.
    encryption_enabled: bool = False
    encryption_key_path: str = ""  # empty = HERMES_HOME/lcm-row.key
    # Retention policy reported by status/doctor. Cleanup remains explicit.
    retention_ttl_days: int = 14
    retention_max_bytes: int = 1024 * 1024 * 1024

    # -- Session carry-over ---
    # Depth retained after /new (-1 = all, 0 = nothing, 2 = keep d2+)
    new_session_retain_depth: int = 2
    # Safety gate: destructive `/lcm doctor clean apply` workflow is disabled by default.
    doctor_clean_apply_enabled: bool = False

    # -- Lifecycle GC ---
    # Enables automatic pruning of lifecycle rows for sessions that never
    # ingested any messages or nodes (gateway restart orphans, ephemeral
    # cron ticks, etc.).  Runs at session-start when the lifecycle table
    # exceeds ``empty_lifecycle_gc_threshold`` rows.
    empty_lifecycle_gc_enabled: bool = True
    # Number of lifecycle rows at which the GC pass fires.  Default 200
    # so fresh installs skip the work until enough churn has occurred.
    empty_lifecycle_gc_threshold: int = 200
    # Age guard for automatic lifecycle GC. Startup GC must not delete
    # recently-bound empty rows because another live engine may not have
    # ingested its first message yet. Set to 0 only in trusted/test
    # environments that intentionally want immediate empty-row pruning.
    empty_lifecycle_gc_max_age_hours: float | None = 24.0

    @classmethod
    def from_env(cls) -> "LCMConfig":
        """Build config from environment variables (LCM_ prefix)."""
        c = cls()
        _int = _parse_int_env
        _float = _parse_float_env
        _str = lambda key, default: os.environ.get(key, default)

        c.fresh_tail_count = _int("LCM_FRESH_TAIL_COUNT", c.fresh_tail_count)
        c.fresh_tail_token_budget_enabled = _lcm_config_bool(
            "LCM_FRESH_TAIL_TOKEN_BUDGET_ENABLED",
            "fresh_tail_token_budget_enabled",
            c.fresh_tail_token_budget_enabled,
        )
        c.fresh_tail_token_budget = _int(
            "LCM_FRESH_TAIL_TOKEN_BUDGET",
            _hermes_lcm_int("fresh_tail_token_budget", c.fresh_tail_token_budget),
        )
        c.fresh_tail_max_tokens = _int(
            "LCM_FRESH_TAIL_MAX_TOKENS",
            _hermes_lcm_int("fresh_tail_max_tokens", c.fresh_tail_max_tokens),
        )
        c.target_ratio = _float(
            "LCM_TARGET_RATIO",
            _hermes_compression_float("target_ratio", c.target_ratio),
        )
        # Same (0, 1] range guard as the config-file path — an out-of-range
        # env override must not defeat the clamp (Greptile PR review).
        if not (0.0 < c.target_ratio <= 1.0):
            c.target_ratio = 0.20
        c.leaf_chunk_tokens = _int("LCM_LEAF_CHUNK_TOKENS", c.leaf_chunk_tokens)
        c.context_threshold = _float(
            "LCM_CONTEXT_THRESHOLD",
            _hermes_compression_threshold(c.context_threshold),
        )
        c.incremental_max_depth = _int("LCM_INCREMENTAL_MAX_DEPTH", c.incremental_max_depth)
        c.condensation_fanin = _int("LCM_CONDENSATION_FANIN", c.condensation_fanin)
        c.dynamic_leaf_chunk_enabled = _parse_bool_env(
            "LCM_DYNAMIC_LEAF_CHUNK_ENABLED", c.dynamic_leaf_chunk_enabled
        )
        c.dynamic_leaf_chunk_max = _int("LCM_DYNAMIC_LEAF_CHUNK_MAX", c.dynamic_leaf_chunk_max)
        c.cache_friendly_condensation_enabled = _parse_bool_env(
            "LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED",
            c.cache_friendly_condensation_enabled,
        )
        c.cache_friendly_min_debt_groups = _int(
            "LCM_CACHE_FRIENDLY_MIN_DEBT_GROUPS",
            c.cache_friendly_min_debt_groups,
        )
        c.deferred_maintenance_enabled = _parse_bool_env(
            "LCM_DEFERRED_MAINTENANCE_ENABLED",
            c.deferred_maintenance_enabled,
        )
        c.deferred_maintenance_max_passes = _int(
            "LCM_DEFERRED_MAINTENANCE_MAX_PASSES",
            c.deferred_maintenance_max_passes,
        )
        c.critical_budget_pressure_ratio = _float(
            "LCM_CRITICAL_BUDGET_PRESSURE_RATIO",
            c.critical_budget_pressure_ratio,
        )
        # P2 calibration knobs: env override → compression.<key> config → default.
        c.skew_floor = _float(
            "LCM_SKEW_FLOOR",
            _hermes_compression_float("skew_floor", c.skew_floor),
        )
        c.calibration_hard_frac = _float(
            "LCM_CALIBRATION_HARD_FRAC",
            _hermes_compression_float("calibration_hard_frac", c.calibration_hard_frac),
        )
        c.l2_budget_ratio = _float("LCM_L2_BUDGET_RATIO", c.l2_budget_ratio)
        c.l3_truncate_tokens = _int("LCM_L3_TRUNCATE_TOKENS", c.l3_truncate_tokens)
        c.max_assembly_tokens = _int("LCM_MAX_ASSEMBLY_TOKENS", c.max_assembly_tokens)
        c.reserve_tokens_floor = _int("LCM_RESERVE_TOKENS_FLOOR", c.reserve_tokens_floor)
        c.custom_instructions = _str("LCM_CUSTOM_INSTRUCTIONS", c.custom_instructions)
        c.extraction_enabled = _parse_bool_env("LCM_EXTRACTION_ENABLED", c.extraction_enabled)
        c.extraction_model = _str("LCM_EXTRACTION_MODEL", c.extraction_model)
        c.extraction_output_path = _str("LCM_EXTRACTION_OUTPUT_PATH", c.extraction_output_path)
        c.sensitive_patterns_enabled = _parse_bool_env(
            "LCM_SENSITIVE_PATTERNS_ENABLED",
            c.sensitive_patterns_enabled,
        )
        raw_sensitive_patterns = os.environ.get("LCM_SENSITIVE_PATTERNS")
        if raw_sensitive_patterns is not None:
            c.sensitive_patterns = _parse_pattern_list(raw_sensitive_patterns)
            c.sensitive_patterns_source = "env"
        c.large_output_externalization_enabled = _parse_bool_env(
            "LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED",
            c.large_output_externalization_enabled,
        )
        c.large_output_externalization_threshold_chars = _int(
            "LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS",
            c.large_output_externalization_threshold_chars,
        )
        c.large_output_externalization_path = _str(
            "LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH",
            c.large_output_externalization_path,
        )
        c.large_output_transcript_gc_enabled = _parse_bool_env(
            "LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED",
            c.large_output_transcript_gc_enabled,
        )
        c.summary_model = _str("LCM_SUMMARY_MODEL", c.summary_model)
        raw_summary_fallback_models = os.environ.get("LCM_SUMMARY_FALLBACK_MODELS")
        if raw_summary_fallback_models is not None:
            c.summary_fallback_models = _parse_pattern_list(raw_summary_fallback_models)
        c.summary_circuit_breaker_failure_threshold = _int(
            "LCM_SUMMARY_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
            c.summary_circuit_breaker_failure_threshold,
        )
        c.summary_circuit_breaker_cooldown_seconds = _int(
            "LCM_SUMMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS",
            c.summary_circuit_breaker_cooldown_seconds,
        )
        c.expansion_model = _str("LCM_EXPANSION_MODEL", c.expansion_model)
        c.expansion_context_tokens = _int("LCM_EXPANSION_CONTEXT_TOKENS", c.expansion_context_tokens)
        c.summary_timeout_ms = _int(
            "LCM_SUMMARY_TIMEOUT_MS",
            _hermes_auxiliary_compression_timeout_ms(c.summary_timeout_ms),
        )
        c.expansion_timeout_ms = _int("LCM_EXPANSION_TIMEOUT_MS", c.expansion_timeout_ms)
        c.database_path = _str("LCM_DATABASE_PATH", c.database_path)
        c.encryption_enabled = _parse_bool_env(
            "LCM_ENCRYPTION_ENABLED",
            c.encryption_enabled,
        )
        c.encryption_key_path = _str("LCM_ENCRYPTION_KEY_PATH", c.encryption_key_path) or ""
        c.retention_ttl_days = _int("LCM_RETENTION_TTL_DAYS", c.retention_ttl_days)
        c.retention_max_bytes = _int("LCM_RETENTION_MAX_BYTES", c.retention_max_bytes)
        c.new_session_retain_depth = _int("LCM_NEW_SESSION_RETAIN_DEPTH", c.new_session_retain_depth)
        c.doctor_clean_apply_enabled = _parse_bool_env(
            "LCM_DOCTOR_CLEAN_APPLY_ENABLED",
            c.doctor_clean_apply_enabled,
        )

        c.empty_lifecycle_gc_enabled = _parse_bool_env(
            "LCM_EMPTY_LIFECYCLE_GC_ENABLED",
            c.empty_lifecycle_gc_enabled,
        )
        c.empty_lifecycle_gc_threshold = _int(
            "LCM_EMPTY_LIFECYCLE_GC_THRESHOLD",
            c.empty_lifecycle_gc_threshold,
        )
        raw_max_age = os.environ.get("LCM_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS")
        if raw_max_age is not None:
            try:
                c.empty_lifecycle_gc_max_age_hours = float(raw_max_age)
            except (TypeError, ValueError):
                pass

        raw_ignore = os.environ.get("LCM_IGNORE_SESSION_PATTERNS")
        if raw_ignore is not None:
            c.ignore_session_patterns = _parse_pattern_list(raw_ignore)
            c.ignore_session_patterns_source = "env"

        raw_stateless = os.environ.get("LCM_STATELESS_SESSION_PATTERNS")
        if raw_stateless is not None:
            c.stateless_session_patterns = _parse_pattern_list(raw_stateless)
            c.stateless_session_patterns_source = "env"

        raw_ignore_messages = os.environ.get("LCM_IGNORE_MESSAGE_PATTERNS")
        if raw_ignore_messages is not None:
            c.ignore_message_patterns = _parse_pattern_list(raw_ignore_messages)
            c.ignore_message_patterns_source = "env"

        return c
