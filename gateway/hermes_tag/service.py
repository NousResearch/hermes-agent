"""Profile-local service construction and shadow-admission boundary."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bridge import identity_from_session_source, surface_from_session_source
from .config import HermesTagConfig, database_path
from .enforcement import LeaseAuthority
from .errors import ConfigurationError
from .kernel import HermesTagKernel
from .ledger import HermesTagLedger
from .middleware import AdmissionResult
from .model import ContinuityMode, ExternalIdentity, SurfaceRef
from .policy import PolicyRule

logger = logging.getLogger(__name__)

SecretResolver = Callable[[str], bytes | str | None]


@dataclass(frozen=True, slots=True)
class ShadowAdmissionOutcome:
    admission: AdmissionResult | None
    error_class: str | None = None


class HermesTagService:
    """One process/profile service.

    Construction is additive. `enabled=false` returns a disabled service and
    touches no state. Shadow admission may fail open only at this observation
    boundary; effect authorization never does.
    """

    def __init__(
        self,
        *,
        config: HermesTagConfig,
        profile: str,
        ledger: HermesTagLedger | None,
        kernel: HermesTagKernel | None,
    ) -> None:
        self.config = config
        self.profile = profile
        self.ledger = ledger
        self.kernel = kernel

    @property
    def enabled(self) -> bool:
        return self.kernel is not None

    @classmethod
    def build(
        cls,
        *,
        hermes_home: str | Path,
        profile: str,
        raw_config: Mapping[str, Any] | None,
        secret_resolver: SecretResolver | None = None,
        rules: Iterable[PolicyRule] = (),
    ) -> "HermesTagService":
        config = HermesTagConfig.from_mapping(raw_config)
        if not config.enabled:
            return cls(config=config, profile=profile, ledger=None, kernel=None)

        path = database_path(Path(hermes_home), profile, config)
        ledger = HermesTagLedger(path)
        ledger.initialize()

        authority: LeaseAuthority | None = None
        secret_ref = config.leases.signing_secret_ref
        if secret_ref is not None:
            if secret_resolver is None:
                raise ConfigurationError(
                    "signing_secret_ref is configured but no secret resolver is available"
                )
            resolved = secret_resolver(secret_ref)
            if resolved is None:
                raise ConfigurationError("Hermes Tag signing secret could not be resolved")
            if isinstance(resolved, str):
                resolved = resolved.encode("utf-8")
            if not isinstance(resolved, bytes):
                raise ConfigurationError("secret resolver returned an invalid value")
            try:
                authority = LeaseAuthority(
                    resolved,
                    ttl_seconds=config.leases.ttl_seconds,
                    clock_skew_seconds=config.leases.clock_skew_seconds,
                )
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc

        kernel = HermesTagKernel(
            ledger,
            config,
            rules=rules,
            lease_authority=authority,
        )
        return cls(config=config, profile=profile, ledger=ledger, kernel=kernel)

    def require_kernel(self) -> HermesTagKernel:
        if self.kernel is None:
            raise ConfigurationError("Hermes Tag is disabled")
        return self.kernel

    def admit(
        self,
        identity: ExternalIdentity,
        surface: SurfaceRef,
        *,
        event_id: str | None = None,
        project_id: str | None = None,
        continuity_mode: ContinuityMode | None = None,
        explicit_continuity_id: str | None = None,
    ) -> AdmissionResult:
        return self.require_kernel().admit_turn(
            identity,
            surface,
            event_id=event_id,
            project_id=project_id,
            continuity_mode=continuity_mode,
            explicit_continuity_id=explicit_continuity_id,
        )

    def admit_session_source(
        self,
        source: object,
        *,
        event_id: str | None = None,
        project_id: str | None = None,
        continuity_mode: ContinuityMode | None = None,
        explicit_continuity_id: str | None = None,
    ) -> AdmissionResult:
        surface = surface_from_session_source(source, profile=self.profile)
        identity = identity_from_session_source(source, surface=surface)
        return self.admit(
            identity,
            surface,
            event_id=event_id,
            project_id=project_id,
            continuity_mode=continuity_mode,
            explicit_continuity_id=explicit_continuity_id,
        )

    def shadow_admit_session_source(
        self,
        source: object,
        *,
        event_id: str | None = None,
        project_id: str | None = None,
    ) -> ShadowAdmissionOutcome:
        if not self.enabled:
            return ShadowAdmissionOutcome(admission=None)
        try:
            result = self.admit_session_source(
                source,
                event_id=event_id,
                project_id=project_id,
            )
            return ShadowAdmissionOutcome(admission=result)
        except Exception as exc:
            if not self.config.shadow:
                raise
            logger.warning(
                "Hermes Tag shadow admission failed closed to observation only: %s",
                type(exc).__name__,
            )
            return ShadowAdmissionOutcome(
                admission=None, error_class=type(exc).__name__
            )
