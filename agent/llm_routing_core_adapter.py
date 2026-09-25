from __future__ import annotations

from llm_routing_core import (
    Candidate,
    RecoveryCoordinator,
    RecoveryEngine,
    RecoveryPolicy,
    RecoveryState,
    RoutingPlan,
    RoutingState,
)


class HermesOpenRouterRecovery:
    """Hermes adapter for the llm-routing-core OpenRouter recovery policy."""

    def __init__(self, state_path: str):
        self.engine = RecoveryEngine(
            RecoveryPolicy(
                max_attempts=4,
                backoff_seconds=(60, 120, 180),
                credential_cooldown_seconds=24 * 60 * 60,
            )
        )
        self.state = RoutingState(state_path)

    def new_recovery_state(self) -> RecoveryState:
        """Create state for one ordered model-recovery sequence."""
        return RecoveryState()

    def coordinator(
        self,
        credential: str,
        models: tuple[str, ...],
    ) -> RecoveryCoordinator:
        return RecoveryCoordinator(
            plan=RoutingPlan(
                provider="openrouter",
                credentials=(credential,),
                models=models,
            ),
            recovery=self.engine,
            state=self.state,
        )

    def openrouter_models(
        self,
        current_model: str,
        fallback_chain: list[dict],
    ) -> tuple[str, ...]:
        """Return the ordered OpenRouter model sequence from Hermes fallback config."""
        models = [current_model]
        for entry in fallback_chain:
            if not isinstance(entry, dict):
                continue
            model = str(entry.get("model", "")).strip()
            if model and model not in models:
                models.append(model)
        return tuple(models)

    def candidate(
        self,
        credential: str,
        model: str,
    ) -> Candidate:
        return Candidate(
            provider="openrouter",
            credential=credential,
            model=model,
        )

    def failure(
        self,
        recovery_state: RecoveryState,
        credential: str,
        model: str,
    ):
        return self.engine.failure(
            recovery_state,
            self.candidate(credential, model),
        )

    def credential_exhausted(
        self,
        credential: str,
    ):
        """Apply the core credential cooldown after every configured model is exhausted."""
        coordinator = self.coordinator(credential, ())
        return coordinator.credential_exhausted(credential)

    def success(
        self,
        recovery_state: RecoveryState,
        credential: str,
        model: str,
    ):
        return self.engine.success(
            recovery_state,
            self.candidate(credential, model),
        )
