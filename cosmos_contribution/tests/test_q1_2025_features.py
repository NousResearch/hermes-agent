"""
Tests for cosmos Q1 2025 (v0.2.0) Features

Tests:
- Episodic Memory Timeline
- Semantic Memory Layers
- Memory Sharing
- Enhanced Knowledge Graph (temporal edges, entity resolution)
- Hybrid Search v2
- Context Compression
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Episodic Memory imports
from cosmos.memory.episodic_memory import (
    EpisodicMemory,
    Episode,
    Session,
    EventType,
    TimelineQuery,
)

# Semantic Layers imports
from cosmos.memory.semantic_layers import (
    SemanticLayerSystem,
    SemanticConcept,
    AbstractionLevel,
)

# Memory Sharing imports
from cosmos.memory.memory_sharing import (
    MemorySharing,
    ExportFormat,
    MergeStrategy,
)

# Knowledge Graph V2 imports
from cosmos.memory.knowledge_graph_v2 import (
    KnowledgeGraphV2,
    TemporalEdge,
    EntityResolutionCandidate,
)

# Hybrid Search V2 imports
from cosmos.rag.hybrid_search_v2 import (
    HybridSearchV2,
    QueryIntent,
    QueryAnalysis,
)

# Context Compression imports
from cosmos.rag.context_compression import (
    ContextCompressor,
    CompressionLevel,
    ContentPriority,
    ContextBlock,
    ContextBudget,
)


# ============================================================================
# Episodic Memory Tests
# ============================================================================

class TestEpisodicMemory:
    """Tests for episodic memory timeline system."""

    @pytest.fixture
    def episodic_memory(self, tmp_path):
        """Create a test episodic memory system."""
        return EpisodicMemory(data_dir=str(tmp_path / "episodic"))

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, episodic_memory):
        """Test session start/end lifecycle."""
        await episodic_memory.initialize()

        # Start session
        session = await episodic_memory.start_session(title="Test Session")
        assert session is not None
        assert session.title == "Test Session"
        assert session.is_active

        # End session
        ended = await episodic_memory.end_session(summary="Completed successfully")
        assert ended is not None
        assert not ended.is_active
        assert ended.summary == "Completed successfully"

    @pytest.mark.asyncio
    async def test_episode_recording(self, episodic_memory):
        """Test recording episodes."""
        await episodic_memory.initialize()

        # Record episode
        episode = await episodic_memory.record_episode(
            event_type=EventType.CONVERSATION,
            content="User asked about Python",
            importance=0.8,
            tags=["programming", "python"],
        )

        assert episode is not None
        assert episode.event_type == EventType.CONVERSATION
        assert episode.content == "User asked about Python"
        assert episode.importance == 0.8
        assert "python" in episode.tags

    @pytest.mark.asyncio
    async def test_timeline_query(self, episodic_memory):
        """Test querying timeline."""
        await episodic_memory.initialize()

        # Record multiple episodes
        for i in range(5):
            await episodic_memory.record_episode(
                event_type=EventType.CONVERSATION,
                content=f"Episode {i}",
                importance=0.5 + i * 0.1,
            )

        # Query timeline
        query = TimelineQuery(
            min_importance=0.7,
            event_types=[EventType.CONVERSATION],
            limit=10,
        )
        results = await episodic_memory.query_timeline(query)

        assert len(results) > 0
        assert all(ep.importance >= 0.7 for ep in results)

    @pytest.mark.asyncio
    async def test_on_this_day(self, episodic_memory):
        """Test 'on this day' functionality."""
        await episodic_memory.initialize()

        # This will be empty for new system
        results = await episodic_memory.get_on_this_day()
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_milestone_marking(self, episodic_memory):
        """Test milestone marking."""
        await episodic_memory.initialize()

        milestone = await episodic_memory.mark_milestone(
            content="Completed first project",
            metadata={"project": "test"},
        )

        assert milestone.event_type == EventType.MILESTONE
        assert milestone.importance == 1.0
        assert "milestone" in milestone.tags


# ============================================================================
# Semantic Layers Tests
# ============================================================================

class TestSemanticLayers:
    """Tests for semantic memory layer system."""

    @pytest.fixture
    def semantic_system(self, tmp_path):
        """Create a test semantic layer system."""
        return SemanticLayerSystem(data_dir=str(tmp_path / "semantic"))

    @pytest.mark.asyncio
    async def test_add_instance(self, semantic_system):
        """Test adding instances to semantic memory."""
        await semantic_system.initialize()

        instance = await semantic_system.add_instance(
            content="Python is a programming language",
            context="Learning about programming",
        )

        assert instance is not None
        assert instance.level == AbstractionLevel.INSTANCE

    @pytest.mark.asyncio
    async def test_concept_hierarchy(self, semantic_system):
        """Test concept hierarchy retrieval."""
        await semantic_system.initialize()

        # Add some instances
        await semantic_system.add_instance("JavaScript is for web development")
        await semantic_system.add_instance("TypeScript extends JavaScript")

        # Query hierarchy
        for concept_id in semantic_system.concepts:
            hierarchy = await semantic_system.get_concept_hierarchy(concept_id)
            assert "concept" in hierarchy
            break

    @pytest.mark.asyncio
    async def test_domain_creation(self, semantic_system):
        """Test domain cluster creation."""
        await semantic_system.initialize()

        domain = await semantic_system.create_domain(
            name="Programming",
            description="Programming languages and tools",
        )

        assert domain is not None
        assert domain.name == "Programming"

    @pytest.mark.asyncio
    async def test_semantic_query(self, semantic_system):
        """Test semantic querying."""
        await semantic_system.initialize()

        # Add instances
        await semantic_system.add_instance("Machine learning uses neural networks")
        await semantic_system.add_instance("Deep learning is a subset of ML")

        # Query
        results = await semantic_system.query_semantic(
            query="neural networks",
            min_level=AbstractionLevel.INSTANCE,
        )

        assert isinstance(results, list)


# ============================================================================
# Memory Sharing Tests
# ============================================================================

class TestMemorySharing:
    """Tests for memory export/import system."""

    @pytest.fixture
    def memory_sharing(self, tmp_path):
        """Create a test memory sharing system."""
        return MemorySharing(
            data_dir=str(tmp_path / "data"),
            backup_dir=str(tmp_path / "backups"),
        )

    @pytest.fixture
    def setup_mock_callbacks(self, memory_sharing):
        """Setup mock callbacks for testing."""
        async def get_memories():
            return [
                {"id": "mem1", "content": "Test memory 1", "tags": ["test"], "created_at": datetime.now().isoformat()},
                {"id": "mem2", "content": "Test memory 2", "tags": ["test"], "created_at": datetime.now().isoformat()},
            ]

        async def store_memory(mem, **kwargs):
            return True

        async def get_conversations():
            return [{"id": "conv1", "content": "Hello", "timestamp": datetime.now().isoformat()}]

        async def get_entities():
            return [{"id": "e1", "name": "Test Entity", "type": "concept"}]

        async def get_sessions():
            return [{"id": "s1", "started_at": datetime.now().isoformat()}]

        memory_sharing.get_memories_fn = get_memories
        memory_sharing.store_memory_fn = store_memory
        memory_sharing.get_conversations_fn = get_conversations
        memory_sharing.get_entities_fn = get_entities
        memory_sharing.get_sessions_fn = get_sessions

        return memory_sharing

    @pytest.mark.asyncio
    async def test_export_memories(self, setup_mock_callbacks, tmp_path):
        """Test memory export."""
        sharing = setup_mock_callbacks
        export_path = tmp_path / "export.json.gz"

        manifest = await sharing.export_memories(
            output_path=str(export_path),
            format=ExportFormat.COMPRESSED_JSON,
        )

        assert manifest is not None
        assert manifest.total_memories == 2
        assert export_path.exists()

    @pytest.mark.asyncio
    async def test_backup_creation(self, setup_mock_callbacks):
        """Test backup creation."""
        sharing = setup_mock_callbacks

        backup_info = await sharing.create_backup(name="test_backup")

        assert backup_info is not None
        assert "test_backup" in backup_info.id
        assert Path(backup_info.path).exists()

    @pytest.mark.asyncio
    async def test_list_backups(self, setup_mock_callbacks):
        """Test listing backups."""
        sharing = setup_mock_callbacks

        # Create a backup first
        await sharing.create_backup(name="backup1")

        backups = await sharing.list_backups()
        assert len(backups) >= 1


# ============================================================================
# Knowledge Graph V2 Tests
# ============================================================================

class TestKnowledgeGraphV2:
    """Tests for enhanced knowledge graph with temporal edges."""

    @pytest.fixture
    def graph_v2(self, tmp_path):
        """Create a test knowledge graph v2."""
        return KnowledgeGraphV2(data_dir=str(tmp_path / "graph"))

    @pytest.mark.asyncio
    async def test_temporal_relationship(self, graph_v2):
        """Test temporal relationship tracking."""
        await graph_v2.initialize()

        # Add entities
        e1 = await graph_v2.add_entity("Alice", "person")
        e2 = await graph_v2.add_entity("Bob", "person")

        # Add temporal relationship
        edge = await graph_v2.add_temporal_relationship(
            source="Alice",
            target="Bob",
            relation_type="knows",
            weight=0.8,
            evidence="They met at work",
        )

        assert edge is not None
        assert edge.is_active
        assert edge.current_weight == 0.8

    @pytest.mark.asyncio
    async def test_relationship_evolution(self, graph_v2):
        """Test tracking relationship evolution over time."""
        await graph_v2.initialize()

        await graph_v2.add_entity("Company", "organization")
        await graph_v2.add_entity("Employee", "person")

        # Add and update relationship multiple times
        await graph_v2.add_temporal_relationship("Employee", "Company", "works_at", weight=0.5)
        await graph_v2.add_temporal_relationship("Employee", "Company", "works_at", weight=0.7)
        await graph_v2.add_temporal_relationship("Employee", "Company", "works_at", weight=0.9)

        # Get evolution
        evolution = await graph_v2.get_relationship_evolution("Employee", "Company")

        assert len(evolution) > 0
        assert evolution[0]["weight_history"]

    @pytest.mark.asyncio
    async def test_end_relationship(self, graph_v2):
        """Test ending a relationship."""
        await graph_v2.initialize()

        await graph_v2.add_entity("Project", "concept")
        await graph_v2.add_entity("Developer", "person")

        await graph_v2.add_temporal_relationship("Developer", "Project", "works_on")

        # End the relationship
        ended = await graph_v2.end_relationship("Developer", "Project", "works_on")

        assert ended is not None
        assert not ended.is_active
        assert ended.end_time is not None

    @pytest.mark.asyncio
    async def test_entity_resolution_candidates(self, graph_v2):
        """Test finding entity resolution candidates."""
        await graph_v2.initialize()

        # Add similar entities
        await graph_v2.add_entity("Python Programming", "concept")
        await graph_v2.add_entity("Python Language", "concept")

        # Find candidates
        candidates = await graph_v2.find_resolution_candidates(min_similarity=0.5)

        # Results depend on similarity calculation
        assert isinstance(candidates, list)

    @pytest.mark.asyncio
    async def test_3d_graph_data(self, graph_v2):
        """Test 3D visualization data generation."""
        await graph_v2.initialize()

        await graph_v2.add_entity("Node1", "concept")
        await graph_v2.add_entity("Node2", "concept")
        await graph_v2.add_temporal_relationship("Node1", "Node2", "relates_to")

        data = graph_v2.get_3d_graph_data()

        assert "nodes" in data
        assert "links" in data
        assert "metadata" in data


# ============================================================================
# Hybrid Search V2 Tests
# ============================================================================

class TestHybridSearchV2:
    """Tests for advanced hybrid search with intent classification."""

    @pytest.fixture
    def search_v2(self):
        """Create a test hybrid search v2."""
        return HybridSearchV2(max_hops=3, min_confidence=0.3)

    def test_query_intent_classification(self, search_v2):
        """Test query intent classification."""
        # Factual query
        analysis = asyncio.run(search_v2._analyze_query("What is Python?"))
        assert analysis.intent == QueryIntent.FACTUAL

        # Procedural query
        analysis = asyncio.run(search_v2._analyze_query("How do I install NumPy?"))
        assert analysis.intent == QueryIntent.PROCEDURAL

        # Comparative query
        analysis = asyncio.run(search_v2._analyze_query("What's the difference between lists and tuples?"))
        assert analysis.intent == QueryIntent.COMPARATIVE

        # Causal query
        analysis = asyncio.run(search_v2._analyze_query("Why does this error happen?"))
        assert analysis.intent == QueryIntent.CAUSAL

    def test_entity_extraction(self, search_v2):
        """Test entity extraction from query."""
        entities = search_v2._extract_entities("How do I use TensorFlow with Python?")
        assert "TensorFlow" in entities or "Python" in entities

    def test_constraint_extraction(self, search_v2):
        """Test constraint extraction."""
        constraints = search_v2._extract_constraints("What happened after 2020?")
        assert constraints  # Should have time constraint

    def test_needs_multi_hop(self, search_v2):
        """Test multi-hop detection."""
        # Complex query should need multi-hop
        analysis = asyncio.run(search_v2._analyze_query(
            "Why does Python's GIL affect performance in multi-threaded applications?"
        ))
        assert search_v2._needs_multi_hop(analysis)


# ============================================================================
# Context Compression Tests
# ============================================================================

class TestContextCompression:
    """Tests for context compression system."""

    @pytest.fixture
    def compressor(self):
        """Create a test context compressor."""
        return ContextCompressor(default_budget=4000)

    def test_token_estimation(self, compressor):
        """Test token count estimation."""
        text = "This is a test sentence with some words."
        tokens = compressor.estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)  # Should be less than characters

    @pytest.mark.asyncio
    async def test_compress_context_no_compression(self, compressor):
        """Test compression when not needed."""
        blocks = [
            ContextBlock(
                id="b1",
                content="Short content",
                source="test",
                priority=ContentPriority.HIGH,
            )
        ]

        budget = ContextBudget(total_tokens=1000)
        result = await compressor.compress_context(blocks, budget)

        assert result.compression_ratio == 1.0
        assert result.blocks_included == 1

    @pytest.mark.asyncio
    async def test_compress_context_with_compression(self, compressor):
        """Test compression when needed."""
        # Create blocks that exceed budget
        blocks = [
            ContextBlock(
                id=f"b{i}",
                content="A" * 1000,  # Long content
                source="test",
                priority=ContentPriority.MEDIUM,
            )
            for i in range(10)
        ]

        budget = ContextBudget(total_tokens=500)
        result = await compressor.compress_context(
            blocks, budget, compression_level=CompressionLevel.AGGRESSIVE
        )

        assert result.compression_ratio < 1.0
        assert result.compressed_tokens <= budget.available_tokens

    def test_redundancy_removal(self, compressor):
        """Test redundancy removal."""
        text = "This is important. This is important. Really important. Very important."
        cleaned = compressor._remove_redundancy(text)
        assert len(cleaned) <= len(text)

    def test_intelligent_truncation(self, compressor):
        """Test intelligent truncation at sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        truncated = compressor._truncate_intelligently(text, target_length=40)

        # Should end at sentence boundary
        assert truncated.endswith(".") or truncated.endswith("...")

    def test_budget_creation(self, compressor):
        """Test context budget creation."""
        budget = compressor.create_budget(
            total_tokens=4000,
            system_prompt="You are a helpful assistant.",
            conversation="User: Hello\nAssistant: Hi!",
        )

        assert budget.total_tokens == 4000
        assert budget.system_prompt_tokens > 0
        assert budget.conversation_tokens > 0
        assert budget.available_tokens < 4000


# ============================================================================
# Integration Tests
# ============================================================================

class TestQ1Integration:
    """Integration tests for Q1 2025 features working together."""

    @pytest.mark.asyncio
    async def test_episodic_with_semantic(self, tmp_path):
        """Test episodic and semantic memory working together."""
        episodic = EpisodicMemory(data_dir=str(tmp_path / "episodic"))
        semantic = SemanticLayerSystem(data_dir=str(tmp_path / "semantic"))

        await episodic.initialize()
        await semantic.initialize()

        # Record an episode
        episode = await episodic.record_episode(
            event_type=EventType.CONVERSATION,
            content="User learned about machine learning",
            importance=0.8,
        )

        # Also add to semantic memory
        instance = await semantic.add_instance(
            content="Machine learning is a subset of AI",
            context="Learning session",
        )

        assert episode is not None
        assert instance is not None

    @pytest.mark.asyncio
    async def test_search_with_compression(self, tmp_path):
        """Test search results being compressed."""
        search = HybridSearchV2()
        compressor = ContextCompressor()

        # Simulate search results
        blocks = [
            ContextBlock(
                id=f"result{i}",
                content=f"Search result {i} with some content about the topic.",
                source="search",
                priority=ContentPriority.HIGH,
                relevance_score=0.9 - i * 0.1,
            )
            for i in range(5)
        ]

        # Compress for context window
        budget = ContextBudget(total_tokens=500)
        result = await compressor.compress_context(blocks, budget)

        assert result.blocks_included > 0
        assert len(result.context) > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
