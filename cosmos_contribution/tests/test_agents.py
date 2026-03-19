"""
cosmos Agent System Tests

Comprehensive tests for:
- Agent base functionality
- Specialist agents (Code, Reasoning, Research, Creative)
- Swarm orchestration
- Multi-agent coordination
- User avatar modeling
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_backend():
    """Create mock LLM backend for testing."""
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value=AsyncMock(
        text="Mock LLM response",
        confidence=0.85,
        tokens_used=100,
    ))
    return mock


class TestBaseAgent:
    """Tests for base agent functionality."""

    def test_agent_creation(self):
        """Test agent initialization."""
        from cosmos.agents.base_agent import BaseAgent, AgentCapability

        agent = BaseAgent(
            name="TestAgent",
            capabilities=[AgentCapability.REASONING],
        )

        assert agent.name == "TestAgent"
        assert AgentCapability.REASONING in agent.capabilities
        assert agent.agent_id is not None

    def test_agent_status(self):
        """Test agent status reporting."""
        from cosmos.agents.base_agent import BaseAgent, AgentStatus

        agent = BaseAgent(name="TestAgent")

        assert agent.status == AgentStatus.IDLE

        status = agent.get_status()
        assert "name" in status
        assert "status" in status

    @pytest.mark.asyncio
    async def test_agent_execution(self, mock_llm_backend):
        """Test agent task execution."""
        from cosmos.agents.base_agent import BaseAgent

        agent = BaseAgent(name="TestAgent", llm_backend=mock_llm_backend)

        result = await agent.execute("Test task")

        assert result is not None
        assert hasattr(result, "success")

    def test_capability_check(self):
        """Test capability checking."""
        from cosmos.agents.base_agent import BaseAgent, AgentCapability

        agent = BaseAgent(
            name="CodeAgent",
            capabilities=[AgentCapability.CODE_GENERATION, AgentCapability.CODE_ANALYSIS],
        )

        assert agent.has_capability(AgentCapability.CODE_GENERATION)
        assert not agent.has_capability(AgentCapability.CREATIVE_WRITING)


class TestSpecialistAgents:
    """Tests for specialist agent implementations."""

    @pytest.mark.asyncio
    async def test_code_agent_creation(self, mock_llm_backend):
        """Test code agent initialization."""
        from cosmos.agents.specialist_agents import create_code_agent

        agent = create_code_agent(llm_backend=mock_llm_backend)

        assert agent is not None
        assert "code" in agent.name.lower()

    @pytest.mark.asyncio
    async def test_reasoning_agent_creation(self, mock_llm_backend):
        """Test reasoning agent initialization."""
        from cosmos.agents.specialist_agents import create_reasoning_agent

        agent = create_reasoning_agent(llm_backend=mock_llm_backend)

        assert agent is not None
        assert "reason" in agent.name.lower()

    @pytest.mark.asyncio
    async def test_research_agent_creation(self, mock_llm_backend):
        """Test research agent initialization."""
        from cosmos.agents.specialist_agents import create_research_agent

        agent = create_research_agent(llm_backend=mock_llm_backend)

        assert agent is not None
        assert "research" in agent.name.lower()

    @pytest.mark.asyncio
    async def test_creative_agent_creation(self, mock_llm_backend):
        """Test creative agent initialization."""
        from cosmos.agents.specialist_agents import create_creative_agent

        agent = create_creative_agent(llm_backend=mock_llm_backend)

        assert agent is not None
        assert "creative" in agent.name.lower()

    @pytest.mark.asyncio
    async def test_code_agent_task(self, mock_llm_backend):
        """Test code agent executing a coding task."""
        from cosmos.agents.specialist_agents import create_code_agent

        agent = create_code_agent(llm_backend=mock_llm_backend)

        result = await agent.execute("Write a function to calculate factorial")

        assert result is not None
        assert hasattr(result, "output")


class TestSwarmOrchestrator:
    """Tests for swarm orchestration."""

    def test_orchestrator_creation(self):
        """Test swarm orchestrator initialization."""
        from cosmos.agents.swarm_orchestrator import SwarmOrchestrator

        orchestrator = SwarmOrchestrator(max_concurrent=5)

        assert orchestrator.max_concurrent == 5
        assert orchestrator.active_agent_count == 0

    @pytest.mark.asyncio
    async def test_agent_spawning(self, mock_llm_backend):
        """Test spawning agents from the orchestrator."""
        from cosmos.agents.swarm_orchestrator import SwarmOrchestrator
        from cosmos.agents.specialist_agents import create_code_agent

        orchestrator = SwarmOrchestrator()
        orchestrator.register_agent_factory("code", lambda: create_code_agent(llm_backend=mock_llm_backend))
        orchestrator.llm_backend = mock_llm_backend

        agent = await orchestrator.spawn_agent("code")

        assert agent is not None
        assert orchestrator.active_agent_count == 1

    @pytest.mark.asyncio
    async def test_task_submission(self, mock_llm_backend):
        """Test submitting tasks to the swarm."""
        from cosmos.agents.swarm_orchestrator import SwarmOrchestrator

        orchestrator = SwarmOrchestrator()
        orchestrator.llm_backend = mock_llm_backend

        task_id = await orchestrator.submit_task(
            description="Test task",
            context={"key": "value"},
        )

        assert task_id is not None

    @pytest.mark.asyncio
    async def test_task_routing(self, mock_llm_backend):
        """Test automatic task routing to appropriate agents."""
        from cosmos.agents.swarm_orchestrator import SwarmOrchestrator

        orchestrator = SwarmOrchestrator()
        orchestrator.llm_backend = mock_llm_backend

        # Code-related task should route to code agent
        agent_type = orchestrator._route_task("Write a Python function")
        assert agent_type == "code"

        # Reasoning task
        agent_type = orchestrator._route_task("Analyze this problem logically")
        assert agent_type == "reasoning"

    def test_swarm_status(self):
        """Test getting swarm status."""
        from cosmos.agents.swarm_orchestrator import SwarmOrchestrator

        orchestrator = SwarmOrchestrator()
        status = orchestrator.get_swarm_status()

        assert "active_agents" in status
        assert "pending_tasks" in status

    @pytest.mark.asyncio
    async def test_handoff_protocol(self, mock_llm_backend):
        """Test agent handoff for complex tasks."""
        from cosmos.agents.swarm_orchestrator import SwarmOrchestrator

        orchestrator = SwarmOrchestrator()
        orchestrator.llm_backend = mock_llm_backend

        # Simulate handoff scenario
        handoff_data = await orchestrator.create_handoff(
            from_agent="research",
            to_agent="code",
            context={"research_results": "sample data"},
        )

        assert handoff_data is not None


class TestMultiAgentCoordination:
    """Tests for multi-agent coordination."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self, mock_llm_backend):
        """Test parallel task execution across agents."""
        from cosmos.agents.swarm_orchestrator import SwarmOrchestrator

        orchestrator = SwarmOrchestrator(max_concurrent=3)
        orchestrator.llm_backend = mock_llm_backend

        tasks = [
            "Task 1: Research topic",
            "Task 2: Write code",
            "Task 3: Analyze results",
        ]

        results = await orchestrator.execute_parallel(tasks)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_subtask_decomposition(self, mock_llm_backend):
        """Test decomposing main task into subtasks."""
        from cosmos.agents.swarm_orchestrator import SwarmOrchestrator

        orchestrator = SwarmOrchestrator()
        orchestrator.llm_backend = mock_llm_backend

        main_task = "Build a web scraper that extracts data and saves to database"

        results = await orchestrator.execute_with_subtasks(
            main_task=main_task,
            subtasks=[
                "Research web scraping libraries",
                "Write the scraping code",
                "Implement database storage",
            ],
        )

        assert len(results) >= 1


class TestUserAvatar:
    """Tests for user avatar modeling."""

    def test_avatar_creation(self):
        """Test user avatar initialization."""
        from cosmos.agents.user_avatar import UserAvatar

        avatar = UserAvatar(user_id="test_user")

        assert avatar.user_id == "test_user"
        assert avatar.preferences is not None

    @pytest.mark.asyncio
    async def test_preference_learning(self):
        """Test learning user preferences from interactions."""
        from cosmos.agents.user_avatar import UserAvatar

        avatar = UserAvatar(user_id="test_user")

        # Record some interactions
        await avatar.record_interaction(
            action="code_request",
            feedback="positive",
            context={"language": "python"},
        )

        await avatar.record_interaction(
            action="explanation_request",
            feedback="positive",
            context={"detail_level": "detailed"},
        )

        # Check learned preferences
        prefs = avatar.get_preferences()
        assert prefs is not None

    @pytest.mark.asyncio
    async def test_response_personalization(self):
        """Test personalizing responses based on user model."""
        from cosmos.agents.user_avatar import UserAvatar

        avatar = UserAvatar(user_id="test_user")

        # Set some preferences
        avatar.preferences = {
            "verbosity": 0.8,
            "code_preference": 0.9,
            "explanation_depth": 0.7,
        }

        suggestion = avatar.suggest_response_style("code generation task")

        assert suggestion is not None
        assert "temperature" in suggestion or "verbosity" in suggestion


class TestMetaCognition:
    """Tests for meta-cognition agent."""

    def test_metacognition_creation(self):
        """Test meta-cognition agent initialization."""
        from cosmos.agents.meta_cognition import MetaCognitionAgent

        agent = MetaCognitionAgent()
        assert agent is not None

    @pytest.mark.asyncio
    async def test_self_reflection(self, mock_llm_backend):
        """Test self-reflection capability."""
        from cosmos.agents.meta_cognition import MetaCognitionAgent

        agent = MetaCognitionAgent(llm_backend=mock_llm_backend)

        reflection = await agent.reflect_on_performance(
            task="Code generation",
            result={"success": True, "confidence": 0.7},
            feedback=None,
        )

        assert reflection is not None

    @pytest.mark.asyncio
    async def test_capability_gap_detection(self, mock_llm_backend):
        """Test detecting capability gaps."""
        from cosmos.agents.meta_cognition import MetaCognitionAgent

        agent = MetaCognitionAgent(llm_backend=mock_llm_backend)

        gaps = await agent.detect_capability_gaps(
            failed_tasks=[
                {"task": "Advanced math proof", "error": "Unable to complete"},
                {"task": "Complex reasoning", "error": "Low confidence"},
            ]
        )

        assert isinstance(gaps, list)

    @pytest.mark.asyncio
    async def test_improvement_proposals(self, mock_llm_backend):
        """Test generating improvement proposals."""
        from cosmos.agents.meta_cognition import MetaCognitionAgent

        agent = MetaCognitionAgent(llm_backend=mock_llm_backend)

        proposals = await agent.propose_improvements(
            current_metrics={
                "task_success": 0.75,
                "user_satisfaction": 0.8,
                "efficiency": 0.6,
            }
        )

        assert proposals is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
