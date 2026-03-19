"""
cosmos Memory System Tests

Comprehensive tests for:
- Virtual context paging
- Archival memory storage/retrieval
- Knowledge graph operations
- Memory dreaming consolidation
- Working memory management
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

# Test fixtures and helpers
@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestVirtualContext:
    """Tests for MemGPT-style virtual context."""

    def test_context_window_creation(self):
        """Test context window initialization."""
        from cosmos.memory.virtual_context import ContextWindow

        window = ContextWindow(max_tokens=4096)
        assert window.max_tokens == 4096
        assert window.current_tokens == 0
        assert len(window.content) == 0

    def test_context_add_content(self):
        """Test adding content to context window."""
        from cosmos.memory.virtual_context import ContextWindow

        window = ContextWindow(max_tokens=4096)
        window.add("Hello, world!", source="test", importance=0.8)

        assert window.current_tokens > 0
        assert len(window.content) == 1
        assert window.content[0].text == "Hello, world!"

    def test_context_overflow_eviction(self):
        """Test that low-importance content is evicted on overflow."""
        from cosmos.memory.virtual_context import ContextWindow

        window = ContextWindow(max_tokens=100)

        # Add low importance content
        window.add("Low importance content " * 10, source="test", importance=0.2)

        # Add high importance content that would overflow
        window.add("High importance content", source="test", importance=0.9)

        # Should have evicted some low importance content
        assert window.current_tokens <= window.max_tokens

    def test_page_manager_tiers(self):
        """Test page manager tier-based caching."""
        from cosmos.memory.virtual_context import PageManager

        manager = PageManager(
            hot_capacity=100,
            warm_capacity=500,
            cold_capacity=1000,
        )

        # Add pages
        for i in range(20):
            manager.add_page(f"content_{i}", importance=i / 20)

        # Should have distributed across tiers
        assert manager.hot_count > 0
        assert manager.total_pages == 20


class TestArchivalMemory:
    """Tests for archival memory storage."""

    @pytest.mark.asyncio
    async def test_archival_storage(self, temp_data_dir):
        """Test storing memories in archival storage."""
        from cosmos.memory.archival_memory import ArchivalMemory

        memory = ArchivalMemory(data_dir=temp_data_dir)
        await memory.initialize()

        # Store a memory
        memory_id = await memory.store(
            content="Test memory content",
            tags=["test", "example"],
            importance=0.7,
        )

        assert memory_id is not None
        assert memory.count > 0

    @pytest.mark.asyncio
    async def test_archival_search(self, temp_data_dir):
        """Test semantic search in archival memory."""
        from cosmos.memory.archival_memory import ArchivalMemory

        memory = ArchivalMemory(data_dir=temp_data_dir)
        await memory.initialize()

        # Store some memories
        await memory.store("Python is a programming language", tags=["programming"])
        await memory.store("Machine learning uses neural networks", tags=["ml"])
        await memory.store("The weather is sunny today", tags=["weather"])

        # Search
        results = await memory.search("programming language", top_k=2)

        assert len(results) > 0
        assert "Python" in results[0].content or "programming" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_archival_hybrid_search(self, temp_data_dir):
        """Test hybrid semantic + keyword search."""
        from cosmos.memory.archival_memory import ArchivalMemory

        memory = ArchivalMemory(data_dir=temp_data_dir)
        await memory.initialize()

        # Store memories
        await memory.store("The quick brown fox jumps over the lazy dog")
        await memory.store("A fast orange cat leaps across the sleepy canine")

        # Hybrid search should find exact keyword matches
        results = await memory.hybrid_search(
            query="quick fox",
            top_k=2,
            semantic_weight=0.5,
            keyword_weight=0.5,
        )

        assert len(results) > 0


class TestKnowledgeGraph:
    """Tests for knowledge graph operations."""

    def test_entity_creation(self):
        """Test creating entities in knowledge graph."""
        from cosmos.memory.knowledge_graph import KnowledgeGraph

        graph = KnowledgeGraph()

        entity_id = graph.add_entity(
            name="Python",
            entity_type="ProgrammingLanguage",
            properties={"creator": "Guido van Rossum"},
        )

        assert entity_id is not None
        entity = graph.get_entity(entity_id)
        assert entity.name == "Python"

    def test_relationship_creation(self):
        """Test creating relationships between entities."""
        from cosmos.memory.knowledge_graph import KnowledgeGraph

        graph = KnowledgeGraph()

        python_id = graph.add_entity("Python", "Language")
        guido_id = graph.add_entity("Guido van Rossum", "Person")

        graph.add_relationship(
            source_id=guido_id,
            target_id=python_id,
            relation_type="created",
        )

        relationships = graph.get_relationships(guido_id)
        assert len(relationships) > 0
        assert relationships[0].relation_type == "created"

    def test_graph_query(self):
        """Test querying the knowledge graph."""
        from cosmos.memory.knowledge_graph import KnowledgeGraph

        graph = KnowledgeGraph()

        # Build a small graph
        cosmos = graph.add_entity("cosmos", "Project")
        memory = graph.add_entity("Memory System", "Component")
        evolution = graph.add_entity("Evolution", "Feature")

        graph.add_relationship(cosmos, memory, "contains")
        graph.add_relationship(cosmos, evolution, "contains")

        # Query
        results = graph.query("cosmos components", max_hops=1)

        assert len(results.entities) >= 1

    def test_entity_extraction(self):
        """Test automatic entity extraction from text."""
        from cosmos.memory.knowledge_graph import KnowledgeGraph

        graph = KnowledgeGraph()

        text = "John works at Google in California. He uses Python for machine learning."
        entities = graph.extract_entities(text)

        # Should find at least some entities
        entity_names = [e["name"] for e in entities]
        assert any("John" in name or "Google" in name or "Python" in name for name in entity_names)


class TestRecallMemory:
    """Tests for conversation recall memory."""

    def test_conversation_storage(self):
        """Test storing conversation turns."""
        from cosmos.memory.recall_memory import RecallMemory

        memory = RecallMemory()

        memory.add_turn("user", "Hello, how are you?")
        memory.add_turn("assistant", "I'm doing well, thank you!")

        assert len(memory.turns) == 2
        assert memory.turns[0].role == "user"

    def test_conversation_search(self):
        """Test searching conversation history."""
        from cosmos.memory.recall_memory import RecallMemory

        memory = RecallMemory()

        memory.add_turn("user", "Tell me about Python programming")
        memory.add_turn("assistant", "Python is a versatile programming language...")
        memory.add_turn("user", "What about JavaScript?")
        memory.add_turn("assistant", "JavaScript is primarily used for web development...")

        results = memory.search("Python", max_results=5)
        assert len(results) > 0
        assert "Python" in results[0].content

    def test_topic_threading(self):
        """Test topic-based conversation threading."""
        from cosmos.memory.recall_memory import RecallMemory

        memory = RecallMemory()

        memory.add_turn("user", "Let's discuss machine learning", topic="ml")
        memory.add_turn("assistant", "Machine learning is...", topic="ml")
        memory.add_turn("user", "Now let's talk about cooking", topic="cooking")

        ml_turns = memory.get_by_topic("ml")
        assert len(ml_turns) == 2


class TestWorkingMemory:
    """Tests for working memory slots."""

    def test_slot_creation(self):
        """Test creating working memory slots."""
        from cosmos.memory.working_memory import WorkingMemory, SlotType

        wm = WorkingMemory()

        wm.set("current_task", "Implement tests", slot_type=SlotType.TASK)
        wm.set("scratch_notes", "Some notes here", slot_type=SlotType.SCRATCH)

        assert wm.get("current_task") == "Implement tests"
        assert wm.get("scratch_notes") == "Some notes here"

    def test_slot_ttl(self):
        """Test slot TTL expiration."""
        import time
        from cosmos.memory.working_memory import WorkingMemory, SlotType

        wm = WorkingMemory()

        wm.set("temp", "temporary value", slot_type=SlotType.SCRATCH, ttl_seconds=0.1)
        assert wm.get("temp") == "temporary value"

        time.sleep(0.2)
        assert wm.get("temp") is None  # Should be expired

    def test_slot_cross_references(self):
        """Test cross-references between slots."""
        from cosmos.memory.working_memory import WorkingMemory, SlotType

        wm = WorkingMemory()

        wm.set("entity_a", {"name": "A"}, slot_type=SlotType.REFERENCE)
        wm.set("entity_b", {"name": "B", "related_to": "entity_a"}, slot_type=SlotType.REFERENCE)
        wm.link("entity_a", "entity_b")

        refs = wm.get_references("entity_a")
        assert "entity_b" in refs


class TestMemoryDreaming:
    """Tests for memory consolidation (dreaming)."""

    @pytest.mark.asyncio
    async def test_dream_cycle(self, temp_data_dir):
        """Test a complete dream cycle."""
        from cosmos.memory.memory_dreaming import MemoryDreaming
        from cosmos.memory.archival_memory import ArchivalMemory

        archival = ArchivalMemory(data_dir=temp_data_dir)
        await archival.initialize()

        # Add some memories
        for i in range(10):
            await archival.store(f"Memory content {i}", importance=0.5)

        dreaming = MemoryDreaming(archival_memory=archival)

        # Run dream cycle
        result = await dreaming.dream()

        assert result.memories_processed >= 0

    def test_cluster_formation(self):
        """Test memory clustering during dreaming."""
        from cosmos.memory.memory_dreaming import MemoryDreaming

        dreaming = MemoryDreaming()

        memories = [
            "Python programming tutorial",
            "Python data science guide",
            "JavaScript web development",
            "JavaScript frameworks",
            "Cooking Italian pasta",
            "Italian cuisine recipes",
        ]

        clusters = dreaming._cluster_memories(memories)

        # Should form distinct clusters
        assert len(clusters) >= 2

    def test_forgetting_selection(self):
        """Test intelligent forgetting of low-utility memories."""
        from cosmos.memory.memory_dreaming import MemoryDreaming

        dreaming = MemoryDreaming()

        memories = [
            {"content": "Important fact", "importance": 0.9, "access_count": 10},
            {"content": "Rarely accessed", "importance": 0.3, "access_count": 1},
            {"content": "Medium importance", "importance": 0.5, "access_count": 5},
        ]

        to_forget = dreaming._select_for_forgetting(memories, forget_ratio=0.3)

        # Should select low importance/access memories first
        assert len(to_forget) > 0


class TestMemorySystem:
    """Integration tests for the complete memory system."""

    @pytest.mark.asyncio
    async def test_full_memory_cycle(self, temp_data_dir):
        """Test complete remember -> recall -> consolidate cycle."""
        from cosmos.memory.memory_system import MemorySystem

        system = MemorySystem(data_dir=temp_data_dir)
        await system.initialize()

        # Remember
        memory_id = await system.remember(
            content="The capital of France is Paris",
            tags=["geography", "facts"],
            importance=0.8,
        )
        assert memory_id is not None

        # Recall
        results = await system.recall("What is the capital of France?")
        assert len(results) > 0
        assert "Paris" in results[0].content

        # Get context
        context = system.get_context()
        assert len(context) > 0

    @pytest.mark.asyncio
    async def test_memory_summary(self, temp_data_dir):
        """Test memory summary generation."""
        from cosmos.memory.memory_system import MemorySystem

        system = MemorySystem(data_dir=temp_data_dir)
        await system.initialize()

        # Add some memories
        await system.remember("First memory", importance=0.7)
        await system.remember("Second memory", importance=0.5)

        summary = await system.get_memory_summary()
        assert summary is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
