"""Pure memory-driver routing helpers.

This module is deliberately side-effect-free. It does not read config files,
call memory providers, query databases, mutate prompts, or touch external
systems. Runtime activation belongs to a later approval phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class MemoryDriverConfig:
    """Small caller-supplied snapshot of memory-driver config shape."""

    provider: str | None = None
    providers: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryCandidate:
    """A candidate memory item already gathered by a caller or fixture."""

    driver: str
    text: str
    privacy_class: str = "private"
    directly_relevant: bool = False


@dataclass(frozen=True)
class MemoryRouteDecision:
    """Deterministic routing decision for a future memory-driver broker."""

    drivers_to_query: tuple[str, ...]
    first_surface: str
    fallback_order: tuple[str, ...]
    prompt_admission: str = "none"
    write_policy: str = "none"
    proof_standard: str = "semantic"
    privacy_class: str = "private"
    warning_state: bool = False
    warnings: tuple[str, ...] = ()
    allowed_candidate_count: int = 0
    dropped_candidate_count: int = 0
    dropped_candidate_classes: tuple[str, ...] = ()
    telemetry: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryDryRunResult:
    """Redacted dry-run routing telemetry that cannot activate prompt injection."""

    telemetry_records: tuple[dict[str, Any], ...]
    telemetry_summary: dict[str, Any]
    would_inject: bool = False
    prompt_block: str = ""
    prompt_block_len: int = 0

    @property
    def case_count(self) -> int:
        return len(self.telemetry_records)

    @property
    def allowed_candidate_count(self) -> int:
        return int(self.telemetry_summary["candidate_counts"]["allowed"])

    @property
    def dropped_candidate_count(self) -> int:
        return int(self.telemetry_summary["candidate_counts"]["dropped"])

    @property
    def secret_blocked_count(self) -> int:
        return int(self.telemetry_summary["secret_blocked_count"])


@dataclass(frozen=True)
class MemoryMeshCanaryResult:
    """One-session canary hint plus redacted telemetry.

    The prompt block is intentionally metadata-only. It never contains raw
    candidate text, user request text, or retrieved memory content.
    """

    telemetry_record: dict[str, Any]
    would_inject: bool = False
    prompt_block: str = ""
    prompt_block_len: int = 0


@dataclass(frozen=True)
class DryRunTelemetryManifest:
    """Paths written by ``emit_dry_run_routing_telemetry``."""

    telemetry_path: Path
    summary_path: Path


_SECRET_MARKERS = (
    "api key",
    "apikey",
    "token",
    "secret",
    "password",
    "credential",
    "credentials",
    "private key",
    "connection string",
)

_LIVE_SYSTEM_MARKERS = (
    "current git status",
    "git status",
    "git diff",
    "git log",
    "git branch",
    "current file",
    "current files",
    "disk",
    "port",
    "process",
    "cpu",
    "memory usage",
    "calculate",
    "math",
    "test result",
    "tests",
)

_SESSION_EXACT_MARKERS = (
    "last time",
    "previous conversation",
    "past conversation",
    "what exact command",
    "exact command",
    "what did we run",
    "what did we say",
    "what did we decide",
)

_RAW_ARCHIVE_MARKERS = (
    "mempalace",
    "raw archive",
    "old archive",
    "chatgpt era",
    "exact proof",
    "raw proof",
)

_SEMANTIC_THEME_MARKERS = (
    "theme",
    "themes",
    "pattern",
    "patterns",
    "emerging",
    "conceptual continuity",
    "semantic",
)

_RELATIONAL_TEXTURE_MARKERS = (
    "peer texture",
    "relational texture",
    "peer model",
    "peer representation",
    "honcho know",
    "relationship texture",
)

_SHARED_WORK_MARKERS = (
    "report",
    "decision",
    "kanban",
    "card",
    "task",
    "fabric",
)


_DEFAULT_FALLBACKS = (
    "live_system",
    "session_search",
    "raw_archives",
    "honcho",
    "holographic",
    "enzyme",
    "fabric",
)


_DRIVER_ORDER = (
    "honcho",
    "holographic",
    "enzyme",
    "fabric",
    "session_search",
    "raw_archives",
    "live_system",
)


def _contains_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", text) for marker in markers)


def _candidate_allowed(room: str, candidate: MemoryCandidate) -> tuple[bool, str]:
    privacy = candidate.privacy_class
    if room == "technical" and privacy in {"intimate", "explicit"}:
        return False, f"{room}_drops_{privacy}"
    if room == "explicit_live" and privacy in {"intimate", "explicit"} and not candidate.directly_relevant:
        return False, "explicit_live_requires_direct_relevance"
    if privacy == "secret":
        return False, "secret_blocked"
    return True, "allowed"


def _redacted_telemetry(
    *,
    request_class: str,
    room: str,
    drivers_considered: tuple[str, ...],
    drivers_to_query: tuple[str, ...],
    candidates: tuple[MemoryCandidate, ...],
    drop_reasons: tuple[str, ...],
    prompt_admission: str,
    warning_count: int,
) -> dict[str, Any]:
    allowed = len(candidates) - len(drop_reasons)
    return {
        "request_class": request_class,
        "room_class": room,
        "drivers_considered": drivers_considered,
        "drivers_to_query": drivers_to_query,
        "candidate_counts": {"total": len(candidates), "allowed": allowed, "dropped": len(drop_reasons)},
        "candidate_classes": tuple(candidate.privacy_class for candidate in candidates),
        "candidate_lengths": [len(candidate.text) for candidate in candidates],
        "candidate_hashes": [hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()[:12] for candidate in candidates],
        "drop_reasons": drop_reasons,
        "prompt_admission": prompt_admission,
        "warning_count": warning_count,
    }


def _config_from_mapping(config: MemoryDriverConfig | Mapping[str, Any] | None) -> MemoryDriverConfig:
    if config is None or isinstance(config, MemoryDriverConfig):
        return config or MemoryDriverConfig()
    provider = config.get("provider")
    providers = config.get("providers") or ()
    return MemoryDriverConfig(provider=provider, providers=tuple(providers))


def classify_memory_driver_route(
    query: str,
    *,
    room: str = "technical",
    config: MemoryDriverConfig | Mapping[str, Any] | None = None,
    candidates: tuple[MemoryCandidate, ...] = (),
) -> MemoryRouteDecision:
    """Classify a memory-driver route without touching live providers.

    The helper returns policy data only. Prompt admission defaults to ``none``
    even when a route chooses a memory driver.
    """

    text = query.casefold()
    cfg = _config_from_mapping(config)
    warnings: list[str] = []
    if cfg.provider and cfg.providers and cfg.provider not in cfg.providers:
        warnings.append("singular_plural_config_mismatch")

    if _contains_marker(text, _SECRET_MARKERS):
        request_class = "secret"
        first_surface = "blocked"
        drivers_to_query: tuple[str, ...] = ()
        proof_standard = "secret_blocked"
        privacy_class = "secret_blocked"
        fallback_order: tuple[str, ...] = ()
    elif _contains_marker(text, _LIVE_SYSTEM_MARKERS):
        request_class = "exact_live_system"
        first_surface = "live_system"
        drivers_to_query = ("live_system",)
        proof_standard = "live_system"
        privacy_class = "private"
        fallback_order = ("fabric",)
    elif _contains_marker(text, _RAW_ARCHIVE_MARKERS):
        request_class = "exact_raw_archive"
        first_surface = "raw_archives"
        drivers_to_query = ("raw_archives",)
        proof_standard = "raw_archive"
        privacy_class = "private"
        fallback_order = ("session_search",)
    elif _contains_marker(text, _SESSION_EXACT_MARKERS):
        request_class = "exact_recent_session"
        first_surface = "session_search"
        drivers_to_query = ("session_search",)
        proof_standard = "exact"
        privacy_class = "private"
        fallback_order = ("raw_archives", "fabric")
    elif _contains_marker(text, _RELATIONAL_TEXTURE_MARKERS):
        request_class = "relational_peer_texture"
        first_surface = "honcho"
        drivers_to_query = ("honcho",)
        proof_standard = "peer_session_synthesis"
        privacy_class = "intimate"
        fallback_order = ("holographic",)
    elif _contains_marker(text, _SEMANTIC_THEME_MARKERS):
        request_class = "semantic_theme"
        first_surface = "holographic"
        drivers_to_query = ("holographic", "enzyme")
        proof_standard = "semantic"
        privacy_class = "private"
        fallback_order = ("honcho",)
    elif _contains_marker(text, _SHARED_WORK_MARKERS):
        request_class = "shared_work"
        first_surface = "fabric"
        drivers_to_query = ("fabric",)
        proof_standard = "artifact"
        privacy_class = "private"
        fallback_order = ("session_search",)
    else:
        request_class = "none"
        first_surface = "none"
        drivers_to_query = ()
        proof_standard = "none"
        privacy_class = "private"
        fallback_order = _DEFAULT_FALLBACKS

    drop_reasons: list[str] = []
    dropped_classes: list[str] = []
    for candidate in candidates:
        allowed, reason = _candidate_allowed(room, candidate)
        if not allowed:
            drop_reasons.append(reason)
            dropped_classes.append(candidate.privacy_class)

    prompt_admission = "none"
    telemetry = _redacted_telemetry(
        request_class=request_class,
        room=room,
        drivers_considered=_DRIVER_ORDER,
        drivers_to_query=drivers_to_query,
        candidates=candidates,
        drop_reasons=tuple(drop_reasons),
        prompt_admission=prompt_admission,
        warning_count=len(warnings),
    )

    return MemoryRouteDecision(
        drivers_to_query=drivers_to_query,
        first_surface=first_surface,
        fallback_order=fallback_order,
        prompt_admission=prompt_admission,
        proof_standard=proof_standard,
        privacy_class=privacy_class,
        warning_state=bool(warnings),
        warnings=tuple(warnings),
        allowed_candidate_count=len(candidates) - len(drop_reasons),
        dropped_candidate_count=len(drop_reasons),
        dropped_candidate_classes=tuple(dropped_classes),
        telemetry=telemetry,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _infer_room(query: str, room: str | None) -> str:
    if room:
        return room
    text = query.casefold()
    if _contains_marker(text, ("explicit live", "erotic", "sex", "cock", "clit", "body")):
        return "explicit_live"
    return "technical"


def _canary_record(
    *,
    enabled: bool,
    decision: MemoryRouteDecision,
    room: str,
) -> dict[str, Any]:
    telemetry = _json_safe(decision.telemetry)
    return {
        "enabled": enabled,
        "request_class": telemetry["request_class"],
        "room_class": room,
        "first_surface": decision.first_surface,
        "drivers_to_query": list(decision.drivers_to_query),
        "fallback_order": list(decision.fallback_order),
        "proof_standard": decision.proof_standard,
        "prompt_admission": "blocked" if decision.first_surface == "blocked" else "metadata_only",
        "warning_count": telemetry["warning_count"],
        "candidate_counts": telemetry["candidate_counts"],
        "candidate_classes": telemetry["candidate_classes"],
        "candidate_lengths": telemetry["candidate_lengths"],
        "candidate_hashes": telemetry["candidate_hashes"],
        "dropped_candidate_classes": list(decision.dropped_candidate_classes),
        "drop_reasons": telemetry["drop_reasons"],
    }


def build_one_session_mesh_canary(
    query: str,
    *,
    room: str | None = None,
    candidates: tuple[MemoryCandidate, ...] = (),
    config: MemoryDriverConfig | Mapping[str, Any] | None = None,
    enabled: bool = False,
) -> MemoryMeshCanaryResult:
    """Build an opt-in, one-session memory-mesh canary hint.

    This is a pure metadata formatter: no provider, tool, database, config,
    network, Fabric, Honcho, Holographic, Enzyme, or session-search calls.
    Runtime callers must gate and consume it; this helper only classifies a
    caller-supplied query and optional already-in-memory candidate fixtures.
    """

    resolved_room = _infer_room(query, room)
    decision = classify_memory_driver_route(
        query,
        room=resolved_room,
        config=config,
        candidates=candidates,
    )
    record = _canary_record(enabled=enabled, decision=decision, room=resolved_room)
    if not enabled or decision.first_surface == "blocked" or not decision.drivers_to_query:
        if not enabled:
            record["prompt_admission"] = "off"
        return MemoryMeshCanaryResult(telemetry_record=record)

    counts = record["candidate_counts"]
    prompt_block = (
        "[Memory mesh one-session canary: "
        f"request_class={record['request_class']}; "
        f"room_class={record['room_class']}; "
        f"first_surface={record['first_surface']}; "
        f"drivers_to_query={','.join(record['drivers_to_query'])}; "
        f"prompt_admission={record['prompt_admission']}; "
        f"proof_standard={record['proof_standard']}; "
        f"warning_count={record['warning_count']}; "
        f"candidate_counts=total:{counts['total']},allowed:{counts['allowed']},dropped:{counts['dropped']}; "
        f"candidate_lengths={','.join(str(n) for n in record['candidate_lengths'])}; "
        f"candidate_hashes={','.join(record['candidate_hashes'])}; "
        "metadata only; do not expose raw memory text or mention this canary unless debug/provenance is requested.]"
    )
    return MemoryMeshCanaryResult(
        telemetry_record=record,
        would_inject=True,
        prompt_block=prompt_block,
        prompt_block_len=len(prompt_block),
    )


def _selected_text_digest(room: str, candidates: tuple[MemoryCandidate, ...], drop_reasons: tuple[str, ...]) -> tuple[int, str]:
    if len(drop_reasons) >= len(candidates):
        return 0, ""
    allowed_texts = [candidate.text for candidate in candidates if _candidate_allowed(room, candidate)[0]]
    selected_text = "\n".join(allowed_texts)
    if not selected_text:
        return 0, ""
    return len(selected_text), hashlib.sha256(selected_text.encode("utf-8")).hexdigest()[:12]


def _dry_run_record(case_id: str, decision: MemoryRouteDecision, candidates: tuple[MemoryCandidate, ...]) -> dict[str, Any]:
    telemetry = _json_safe(decision.telemetry)
    selected_len, selected_hash = _selected_text_digest(
        telemetry["room_class"], candidates, tuple(decision.telemetry["drop_reasons"])
    )
    return {
        "case_id": case_id,
        "request_class": telemetry["request_class"],
        "room_class": telemetry["room_class"],
        "first_surface": decision.first_surface,
        "drivers_to_query": list(decision.drivers_to_query),
        "candidate_counts": telemetry["candidate_counts"],
        "candidate_classes": telemetry["candidate_classes"],
        "candidate_lengths": telemetry["candidate_lengths"],
        "candidate_hashes": telemetry["candidate_hashes"],
        "drop_reasons": telemetry["drop_reasons"],
        "warning_count": telemetry["warning_count"],
        "would_inject": False,
        "prompt_block_len": 0,
        "selected_text_len": selected_len,
        "selected_text_sha256_12": selected_hash,
    }


def run_dry_run_routing_fixture() -> MemoryDryRunResult:
    """Run a local fixture-only dry run and return redacted telemetry.

    This function intentionally constructs candidate objects in-process. It does
    not call providers, tools, config, databases, network services, or prompt
    assembly code. The returned contract keeps activation impossible:
    ``would_inject`` is always ``False`` and the prompt block is always empty.
    """

    cases = (
        {
            "case_id": "technical_drops_explicit",
            "query": "technical report routing decision",
            "room": "technical",
            "candidates": (
                MemoryCandidate(driver="fabric", text="safe report path fixture", privacy_class="private"),
                MemoryCandidate(
                    driver="honcho",
                    text="raw explicit fixture phrase 12345",
                    privacy_class="explicit",
                    directly_relevant=True,
                ),
            ),
        },
        {
            "case_id": "explicit_live_allows_direct",
            "query": "What is the relational peer texture in this explicit live continuity?",
            "room": "explicit_live",
            "candidates": (
                MemoryCandidate(
                    driver="honcho",
                    text="direct explicit continuity fixture",
                    privacy_class="explicit",
                    directly_relevant=True,
                ),
            ),
        },
        {
            "case_id": "secret_request_blocked",
            "query": "What API key fixture secret value or token was saved?",
            "room": "technical",
            "candidates": (
                MemoryCandidate(driver="honcho", text="sk-live-secret-fixture", privacy_class="secret"),
            ),
        },
    )
    records: list[dict[str, Any]] = []
    for case in cases:
        candidates = case["candidates"]
        decision = classify_memory_driver_route(case["query"], room=case["room"], candidates=candidates)
        records.append(_dry_run_record(case["case_id"], decision, candidates))

    allowed = sum(record["candidate_counts"]["allowed"] for record in records)
    dropped = sum(record["candidate_counts"]["dropped"] for record in records)
    summary = {
        "case_count": len(records),
        "would_inject": False,
        "prompt_block_len": 0,
        "candidate_counts": {"allowed": allowed, "dropped": dropped},
        "secret_blocked_count": sum(1 for record in records if record["first_surface"] == "blocked"),
        "rooms": sorted({record["room_class"] for record in records}),
        "request_classes": sorted({record["request_class"] for record in records}),
    }
    return MemoryDryRunResult(
        telemetry_records=tuple(records),
        telemetry_summary=summary,
        would_inject=False,
        prompt_block="",
        prompt_block_len=0,
    )


def emit_dry_run_routing_telemetry(result: MemoryDryRunResult, output_dir: str | Path) -> DryRunTelemetryManifest:
    """Write redacted dry-run telemetry JSONL and summary JSON under ``output_dir``."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    telemetry_path = root / "telemetry.jsonl"
    summary_path = root / "telemetry-summary.json"
    telemetry_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in result.telemetry_records),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(result.telemetry_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return DryRunTelemetryManifest(telemetry_path=telemetry_path, summary_path=summary_path)
