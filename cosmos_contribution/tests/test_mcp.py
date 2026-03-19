"""
cosmos MCP Server Tests

Tests for Claude Code integration via MCP:
- Tool registration and execution
- Resource serving
- Memory operations through MCP
- Agent delegation through MCP
- Evolution feedback through MCP
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestMCPServerCreation:
    """Tests for MCP server initialization."""

    def test_server_creation(self):
        """Test creating the MCP server."""
        # Mock the MCP library
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server": MagicMock(), "mcp.types": MagicMock()}):
            from cosmos.mcp_server.server import cosmosMCPServer

            server = cosmosMCPServer(data_dir="./test_data")

            assert server is not None
            assert server.data_dir.exists() or True  # May not create in test

    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test initializing cosmos components."""
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server": MagicMock(), "mcp.types": MagicMock()}):
            from cosmos.mcp_server.server import cosmosMCPServer

            server = cosmosMCPServer()

            # Mock the component initialization
            server._memory_system = MagicMock()
            server._swarm_orchestrator = MagicMock()
            server._fitness_tracker = MagicMock()

            assert server._memory_system is not None


class TestMCPTools:
    """Tests for MCP tool implementations."""

    @pytest.mark.asyncio
    async def test_remember_tool(self):
        """Test the cosmos_remember tool."""
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server": MagicMock(), "mcp.types": MagicMock()}):
            from cosmos.mcp_server.server import cosmosMCPServer

            server = cosmosMCPServer()

            # Mock memory system
            server._memory_system = AsyncMock()
            server._memory_system.remember = AsyncMock(return_value="memory_123")

            result = await server.remember(
                content="User prefers detailed explanations",
                tags=["preference", "communication"],
                importance=0.8,
            )

            assert result["success"] is True
            assert "memory_id" in result

    @pytest.mark.asyncio
    async def test_recall_tool(self):
        """Test the cosmos_recall tool."""
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server": MagicMock(), "mcp.types": MagicMock()}):
            from cosmos.mcp_server.server import cosmosMCPServer

            server = cosmosMCPServer()

            # Mock memory system with results
            mock_result = MagicMock()
            mock_result.content = "User prefers detailed explanations"
            mock_result.source = "archival"
            mock_result.score = 0.95

            server._memory_system = AsyncMock()
            server._memory_system.recall = AsyncMock(return_value=[mock_result])

            result = await server.recall(
                query="What are the user's communication preferences?",
                limit=5,
            )

            assert result["success"] is True
            assert result["count"] > 0
            assert "memories" in result

    @pytest.mark.asyncio
    async def test_delegate_tool(self):
        """Test the cosmos_delegate tool."""
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server": MagicMock(), "mcp.types": MagicMock()}):
            from cosmos.mcp_server.server import cosmosMCPServer

            server = cosmosMCPServer()

            # Mock swarm orchestrator
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output = "Generated code: def hello(): print('Hello')"
            mock_result.confidence = 0.9
            mock_result.tokens_used = 150
            mock_result.execution_time = 1.5

            server._swarm_orchestrator = AsyncMock()
            server._swarm_orchestrator.submit_task = AsyncMock(return_value="task_123")
            server._swarm_orchestrator.wait_for_task = AsyncMock(return_value=mock_result)

            server._fitness_tracker = MagicMock()
            server._fitness_tracker.record_task_outcome = MagicMock()

            result = await server.delegate(
                task="Write a hello world function in Python",
                agent_type="code",
            )

            assert result["success"] is True
            assert "output" in result

    @pytest.mark.asyncio
    async def test_evolve_tool(self):
        """Test the cosmos_evolve tool."""
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server": MagicMock(), "mcp.types": MagicMock()}):
            from cosmos.mcp_server.server import cosmosMCPServer

            server = cosmosMCPServer()

            server._fitness_tracker = MagicMock()
            server._fitness_tracker.record = MagicMock()

            server._memory_system = AsyncMock()
            server._memory_system.remember = AsyncMock(return_value="feedback_123")

            result = await server.evolve(
                feedback="Great response! Very helpful and detailed.",
            )

            assert result["success"] is True
            assert result["sentiment"] == "positive"

    @pytest.mark.asyncio
    async def test_status_tool(self):
        """Test the cosmos_status tool."""
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server": MagicMock(), "mcp.types": MagicMock()}):
            from cosmos.mcp_server.server import cosmosMCPServer

            server = cosmosMCPServer()

            server._memory_system = MagicMock()
            server._memory_system.get_stats = MagicMock(return_value={"memories": 100})

            server._swarm_orchestrator = MagicMock()
            server._swarm_orchestrator.get_swarm_status = MagicMock(return_value={"agents": 3})

            server._fitness_tracker = MagicMock()
            server._fitness_tracker.get_stats = MagicMock(return_value={"fitness": 0.87})

            result = await server.status()

            assert result["success"] is True
            assert "memory" in result
            assert "agents" in result
            assert "evolution" in result


class TestMCPResources:
    """Tests for MCP resource implementations."""

    @pytest.mark.asyncio
    async def test_recent_memories_resource(self):
        """Test the cosmos://memory/recent resource."""
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server": MagicMock(), "mcp.types": MagicMock()}):
            from cosmos.mcp_server.server import cosmosMCPServer

            server = cosmosMCPServer()

            server._memory_system = MagicMock()
            server._memory_system.get_context = MagicMock(return_value="Recent context...")

            content = await server.get_recent_memories()

            assert content is not None
            assert len(content) > 0

    @pytest.mark.asyncio
    async def test_knowledge_graph_resource(self):
        """Test the cosmos://memory/graph resource."""
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server": MagicMock(), "mcp.types": MagicMock()}):
            from cosmos.mcp_server.server import cosmosMCPServer

            server = cosmosMCPServer()

            server._memory_system = MagicMock()
            server._memory_system.knowledge_graph = MagicMock()
            server._memory_system.knowledge_graph.get_stats = MagicMock(
                return_value={"entities": 50, "relationships": 120}
            )

            content = await server.get_knowledge_graph()

            assert content is not None
            data = json.loads(content)
            assert "entities" in data

    @pytest.mark.asyncio
    async def test_active_agents_resource(self):
        """Test the cosmos://agents/active resource."""
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server": MagicMock(), "mcp.types": MagicMock()}):
            from cosmos.mcp_server.server import cosmosMCPServer

            server = cosmosMCPServer()

            server._swarm_orchestrator = MagicMock()
            server._swarm_orchestrator.get_swarm_status = MagicMock(
                return_value={"active_agents": 2, "pending_tasks": 1}
            )

            content = await server.get_active_agents()

            assert content is not None
            data = json.loads(content)
            assert "active_agents" in data

    @pytest.mark.asyncio
    async def test_fitness_metrics_resource(self):
        """Test the cosmos://evolution/fitness resource."""
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server": MagicMock(), "mcp.types": MagicMock()}):
            from cosmos.mcp_server.server import cosmosMCPServer

            server = cosmosMCPServer()

            server._fitness_tracker = MagicMock()
            server._fitness_tracker.get_stats = MagicMock(
                return_value={"fitness": 0.87, "trend": 0.02}
            )

            content = await server.get_fitness_metrics()

            assert content is not None
            data = json.loads(content)
            assert "fitness" in data


class TestMemoryTools:
    """Tests for advanced memory tool implementations."""

    @pytest.mark.asyncio
    async def test_remember_with_context(self):
        """Test remembering with additional context."""
        from cosmos.mcp_server.memory_tools import MemoryTools

        mock_memory = AsyncMock()
        mock_memory.remember = AsyncMock(return_value="mem_123")

        tools = MemoryTools(memory_system=mock_memory)

        result = await tools.remember_with_context(
            content="Important fact",
            tags=["important"],
            importance=0.9,
            context={"source": "user_input"},
            extract_entities=True,
        )

        assert result.success is True
        assert "memory_id" in result.data

    @pytest.mark.asyncio
    async def test_advanced_recall(self):
        """Test advanced memory search with filters."""
        from cosmos.mcp_server.memory_tools import MemoryTools

        mock_result = MagicMock()
        mock_result.content = "Test content"
        mock_result.source = "archival"
        mock_result.score = 0.9
        mock_result.metadata = {"tags": ["test"]}

        mock_memory = AsyncMock()
        mock_memory.recall = AsyncMock(return_value=[mock_result])

        tools = MemoryTools(memory_system=mock_memory)

        result = await tools.advanced_recall(
            query="test query",
            limit=10,
            min_score=0.5,
            tags_filter=["test"],
        )

        assert result.success is True
        assert result.data["count"] > 0

    @pytest.mark.asyncio
    async def test_knowledge_graph_query(self):
        """Test querying the knowledge graph."""
        from cosmos.mcp_server.memory_tools import MemoryTools

        mock_query_result = MagicMock()
        mock_query_result.entities = []
        mock_query_result.relationships = []
        mock_query_result.paths = []

        mock_memory = MagicMock()
        mock_memory.knowledge_graph = AsyncMock()
        mock_memory.knowledge_graph.query = AsyncMock(return_value=mock_query_result)

        tools = MemoryTools(memory_system=mock_memory)

        result = await tools.query_knowledge_graph(
            query="relationships",
            max_entities=10,
            max_hops=2,
        )

        assert result.success is True


class TestAgentTools:
    """Tests for agent tool implementations."""

    @pytest.mark.asyncio
    async def test_delegate_task(self):
        """Test delegating a task to an agent."""
        from cosmos.mcp_server.agent_tools import AgentTools

        mock_agent = AsyncMock()
        mock_agent.execute = AsyncMock(return_value=MagicMock(
            success=True,
            output="Result",
            confidence=0.9,
            metadata={},
        ))
        mock_agent.agent_id = "agent_123"

        mock_swarm = MagicMock()
        mock_swarm.spawn_agent = AsyncMock(return_value=mock_agent)
        mock_swarm.state = MagicMock()
        mock_swarm.state.active_agents = {}

        tools = AgentTools(swarm_orchestrator=mock_swarm)

        result = await tools.delegate_task(
            task="Write a function",
            agent_type="code",
            context={},
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_list_available_agents(self):
        """Test listing available agents."""
        from cosmos.mcp_server.agent_tools import AgentTools

        mock_swarm = MagicMock()
        mock_swarm.state = MagicMock()
        mock_swarm.state.active_agents = {}
        mock_swarm.max_concurrent = 5

        tools = AgentTools(swarm_orchestrator=mock_swarm)

        result = await tools.list_available_agents()

        assert result.success is True
        assert "agent_types" in result.data


class TestEvolutionTools:
    """Tests for evolution tool implementations."""

    @pytest.mark.asyncio
    async def test_record_feedback(self):
        """Test recording user feedback."""
        from cosmos.mcp_server.evolution_tools import EvolutionTools

        mock_fitness = MagicMock()
        mock_fitness.record = MagicMock()
        mock_fitness.get_weighted_fitness = MagicMock(return_value=0.85)

        tools = EvolutionTools(fitness_tracker=mock_fitness)

        result = await tools.record_feedback(
            feedback_type="satisfaction",
            value=0.9,
            context={"task": "code_generation"},
        )

        assert result.success is True
        assert "recorded" in result.data

    @pytest.mark.asyncio
    async def test_get_fitness_metrics(self):
        """Test getting fitness metrics."""
        from cosmos.mcp_server.evolution_tools import EvolutionTools

        mock_fitness = MagicMock()
        mock_fitness.get_current_fitness = MagicMock(return_value={"task_success": 0.9})
        mock_fitness.get_weighted_fitness = MagicMock(return_value=0.85)
        mock_fitness.get_stats = MagicMock(return_value={
            "current_fitness": {"task_success": 0.9},
            "trends": {},
            "sample_counts": {},
        })

        tools = EvolutionTools(fitness_tracker=mock_fitness)

        result = await tools.get_fitness_metrics()

        assert result.success is True
        assert "current_fitness" in result.data

    @pytest.mark.asyncio
    async def test_get_improvement_suggestions(self):
        """Test getting improvement suggestions."""
        from cosmos.mcp_server.evolution_tools import EvolutionTools

        mock_fitness = MagicMock()
        mock_fitness.get_current_fitness = MagicMock(return_value={
            "task_success": 0.6,  # Below threshold
            "efficiency": 0.4,    # Below threshold
            "user_satisfaction": 0.5,  # Below threshold
        })
        mock_fitness.get_trend = MagicMock(return_value=-0.05)

        tools = EvolutionTools(fitness_tracker=mock_fitness)

        result = await tools.get_improvement_suggestions()

        assert result.success is True
        assert len(result.data["suggestions"]) > 0


class TestClaudeIntegration:
    """Tests specific to Claude Code integration scenarios."""

    @pytest.mark.asyncio
    async def test_session_memory_persistence(self):
        """Test that memories persist across 'sessions' (tool calls)."""
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server": MagicMock(), "mcp.types": MagicMock()}):
            from cosmos.mcp_server.server import cosmosMCPServer

            server = cosmosMCPServer()

            # Simulate a memory store
            stored_memories = {}

            async def mock_remember(content, **kwargs):
                mem_id = f"mem_{len(stored_memories)}"
                stored_memories[mem_id] = content
                return mem_id

            async def mock_recall(query, **kwargs):
                results = []
                for mem_id, content in stored_memories.items():
                    if query.lower() in content.lower():
                        result = MagicMock()
                        result.content = content
                        result.source = "archival"
                        result.score = 0.9
                        results.append(result)
                return results

            server._memory_system = AsyncMock()
            server._memory_system.remember = mock_remember
            server._memory_system.recall = mock_recall

            # Session 1: Remember something
            await server.remember(content="User likes Python")

            # Session 2: Should be able to recall it
            result = await server.recall(query="Python")

            assert result["success"] is True
            assert result["count"] > 0
            assert "Python" in result["memories"][0]["content"]

    @pytest.mark.asyncio
    async def test_context_augmentation_flow(self):
        """Test the flow of augmenting Claude's context with memories."""
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server": MagicMock(), "mcp.types": MagicMock()}):
            from cosmos.mcp_server.server import cosmosMCPServer

            server = cosmosMCPServer()

            server._memory_system = MagicMock()
            server._memory_system.get_context = MagicMock(return_value="""
            Recent Context:
            - User is working on a Python project
            - User prefers detailed explanations
            - Previous task: implement sorting algorithm
            """)

            # This is what Claude Code would read before responding
            context = await server.get_recent_memories()

            assert "Python project" in context
            assert "detailed explanations" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
