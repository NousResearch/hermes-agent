from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

EvidencePackStatus = Literal[
    "disabled",
    "available",
    "degraded",
]

EvidencePackDegradeReason = Literal[
    "factory_missing",
    "factory_error",
    "invalid_config",
    "storage_unavailable",
]

EvidencePackEngineFactory = Callable[..., Any]


class EvidencePackStorageUnavailable(RuntimeError):
    """Typed signal that the borrowed storage cannot serve the engine.

    Raised by the canonical ``build_evidence_pack_engine`` factory when
    no usable ``SessionDB`` (or structurally equivalent state_meta
    backing) has been supplied. ``build_objective_services`` catches
    this exception explicitly so it can surface a precise
    ``storage_unavailable`` degrade reason instead of the generic
    ``factory_error`` fallback.
    """

    def __init__(self, message: str = "evidence_pack storage unavailable") -> None:
        super().__init__(message)


@dataclass(frozen=True)
class ObjectiveServices:
    session_id: str
    sources: Any | None = None
    storage: Any | None = None
    audit_sink: Any | None = None
    evidence_pack_engine: Any | None = None
    evidence_pack_status: EvidencePackStatus = "disabled"
    evidence_pack_degrade_reason: EvidencePackDegradeReason | None = None
    evidence_pack_error_type: str | None = None

    @property
    def evidence_pack_enabled(self) -> bool:
        return self.evidence_pack_status == "available"


def build_objective_services(
    *,
    session_id: str,
    config: Mapping[str, Any] | None,
    sources: Any | None = None,
    storage: Any | None = None,
    audit_sink: Any | None = None,
    evidence_pack_engine_factory: EvidencePackEngineFactory | None = None,
) -> ObjectiveServices:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required")

    def make_services(
        *,
        evidence_pack_engine: Any | None = None,
        evidence_pack_status: EvidencePackStatus = "disabled",
        evidence_pack_degrade_reason: EvidencePackDegradeReason | None = None,
        evidence_pack_error_type: str | None = None,
    ) -> ObjectiveServices:
        return ObjectiveServices(
            session_id=session_id,
            sources=sources,
            storage=storage,
            audit_sink=audit_sink,
            evidence_pack_engine=evidence_pack_engine,
            evidence_pack_status=evidence_pack_status,
            evidence_pack_degrade_reason=evidence_pack_degrade_reason,
            evidence_pack_error_type=evidence_pack_error_type,
        )

    def invalid_config_services() -> ObjectiveServices:
        return make_services(evidence_pack_degrade_reason="invalid_config")

    if config is None:
        return make_services()
    if not isinstance(config, Mapping):
        return invalid_config_services()
    if not config:
        return make_services()

    goals_config = config.get("goals")
    if goals_config is None:
        return make_services()
    if not isinstance(goals_config, Mapping):
        return invalid_config_services()

    evidence_pack_config = goals_config.get("evidence_pack")
    if evidence_pack_config is None:
        return make_services()
    if not isinstance(evidence_pack_config, Mapping):
        return invalid_config_services()

    enabled = evidence_pack_config.get("enabled")
    if enabled is None or enabled is False:
        return make_services()
    if enabled is not True:
        return invalid_config_services()

    if evidence_pack_engine_factory is None:
        return make_services(
            evidence_pack_status="degraded",
            evidence_pack_degrade_reason="factory_missing",
        )

    try:
        evidence_pack_engine = evidence_pack_engine_factory(
            session_id=session_id,
            config=config,
            sources=sources,
            storage=storage,
            audit_sink=audit_sink,
        )
    except EvidencePackStorageUnavailable as exc:
        # Typed storage failure surfaces as a precise degrade reason so
        # the CLI can show "storage unavailable" instead of the generic
        # factory error envelope.
        return make_services(
            evidence_pack_status="degraded",
            evidence_pack_degrade_reason="storage_unavailable",
            evidence_pack_error_type=type(exc).__name__,
        )
    except Exception as exc:
        return make_services(
            evidence_pack_status="degraded",
            evidence_pack_degrade_reason="factory_error",
            evidence_pack_error_type=type(exc).__name__,
        )

    return make_services(
        evidence_pack_engine=evidence_pack_engine,
        evidence_pack_status="available",
    )