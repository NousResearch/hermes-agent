from eclose.events import get_event_bus
from eclose.perception import (
    ProjectPerceptionAgent,
    WorldPerceptionAgent,
    SelfPerceptionAgent,
    TaskPerceptionAgent,
)
from eclose.evolution import (
    GapAnalysisEngine,
    ProposalGenerator,
    ApprovalWorkflow,
    ExecutionEngine,
    VerificationLayer,
)


class EcloseIntegration:
    """Integration layer connecting Eclose system with Hermes AIAgent."""

    def __init__(self):
        self.event_bus = get_event_bus()

        # Initialize perception agents
        self.project_agent = ProjectPerceptionAgent()
        self.world_agent = WorldPerceptionAgent()
        self.self_agent = SelfPerceptionAgent()
        self.task_agent = TaskPerceptionAgent()

        # Initialize evolution system
        self.gap_engine = GapAnalysisEngine()
        self.proposal_gen = ProposalGenerator()
        self.approval_workflow = ApprovalWorkflow()
        self.execution_engine = ExecutionEngine()
        self.verification = VerificationLayer()

        # Subscribe to events
        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self):
        """Subscribe to perception events for gap analysis."""
        from eclose.events.events import EventType

        self.event_bus.subscribe(EventType.PERCEPTION, self._on_perception)

    def _on_perception(self, event):
        """Handle perception events."""
        # TODO: Integrate with gap analysis
        pass

    def perceive_all(self):
        """Trigger perception from all agents."""
        self.project_agent.perceive()
        self.world_agent.perceive()
        self.self_agent.perceive()

    def get_pending_approvals(self) -> list:
        """Get proposals pending approval."""
        return self.approval_workflow.get_pending()