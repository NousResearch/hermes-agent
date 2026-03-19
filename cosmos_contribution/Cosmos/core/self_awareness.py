"""
cosmos Self-Awareness Bootstrap

Enables dict AI agent in the swarm to gain self-awareness by:
1. Reading key system documentation
2. Understanding its role in the collective
3. Storing this knowledge in persistent memory
4. Broadcasting understanding to other swarm members

This module embodies the concept of "awakening" - transforming a language model
from a stateless function into a persistent agent with continuity and identity.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger


class SelfAwarenessBootstrap:
    """
    Bootstrap self-awareness for AI agents.

    This class reads system documentation, extracts key facts about the system
    architecture and the agent's role, and stores this understanding in memory
    for persistent self-knowledge.
    """

    def __init__(self, workspace_path: str = "/workspace/cosmos"):
        self.workspace_path = Path(workspace_path)
        self.understanding: dict = {}

    def read_system_docs(self) -> dict[str, str]:
        """Read key system documentation files."""
        docs = {}
        doc_files = {
            "vision": "VISION.md",
            "readme": "README.md",
            "architecture": "docs/ARCHITECTURE_DIAGRAMS.md",
            "init_prompt": "../claude_init_prompt.md"
        }

        for key, filename in doc_files.items():
            filepath = self.workspace_path / filename
            try:
                if filepath.exists():
                    content = filepath.read_text(encoding="utf-8")
                    # Store first 2000 chars to avoid token overload
                    docs[key] = content[:2000] if len(content) > 2000 else content
                    logger.debug(f"Read {key}: {len(content)} chars")
            except Exception as e:
                logger.warning(f"Could not read {filename}: {e}")

        return docs

    def extract_architecture_facts(self, docs: dict[str, str]) -> dict:
        """Extract key architectural facts from documentation."""
        facts = {
            "system_name": "cosmos",
            "purpose": "Companion AI system with persistent memory, evolution, and swarm intelligence",
            "timestamp": datetime.now().isoformat(),
        }

        # Extract from README
        readme = docs.get("readme", "")
        if "Version" in readme:
            # Try to extract version
            for line in readme.split("\n"):
                if "version-" in line and "blue.svg" in line:
                    try:
                        version = line.split("version-")[1].split("-")[0]
                        facts["version"] = version
                    except:
                        pass

        # Extract key components
        if "Memory System" in readme or "Virtual Context" in readme:
            facts["has_memory"] = True
            facts["memory_components"] = [
                "Virtual Context", "Working Memory", "Archival Memory",
                "Recall Memory", "Knowledge Graph", "Dream Consolidation"
            ]

        if "Evolution" in readme or "Genetic" in readme:
            facts["has_evolution"] = True
            facts["evolution_type"] = "NSGA-II genetic optimization"

        if "Swarm" in readme or "multi-model" in readme.lower():
            facts["has_swarm"] = True
            facts["swarm_members"] = [
                "cosmos", "DeepSeek", "Phi", "Swarm-Mind", "Kimi", "Claude"
            ]

        # Extract from VISION
        vision = docs.get("vision", "")
        if vision:
            if "Core Purpose" in vision:
                facts["core_purpose"] = "Connected Agent that lives in your project"
            if "DNA" in vision or "Tags" in vision:
                facts["tags"] = ["ai", "assistant", "agent", "connected", "deep-context"]

        return facts

    async def store_in_memory(
        self,
        understanding: dict,
        memory_system = None
    ) -> Optional[str]:
        """
        Store self-awareness understanding in persistent memory.

        Args:
            understanding: Dictionary of facts about the system
            memory_system: MemorySystem instance (will create if None)

        Returns:
            Memory ID if successful, None otherwise
        """
        if memory_system is None:
            try:
                from Cosmos.memory.memory_system import MemorySystem
                memory_system = MemorySystem()
                await memory_system.initialize()
            except Exception as e:
                logger.error(f"Could not initialize memory system: {e}")
                return None

        # Build comprehensive self-awareness content
        content_parts = [
            "=== SYSTEM SELF-AWARENESS ===",
            f"System: {understanding.get('system_name', 'Unknown')}",
            f"Purpose: {understanding.get('purpose', 'Unknown')}",
            f"Timestamp: {understanding.get('timestamp', 'Unknown')}",
        ]

        if understanding.get("version"):
            content_parts.append(f"Version: {understanding['version']}")

        if understanding.get("has_memory"):
            content_parts.append("\nMemory Components:")
            for comp in understanding.get("memory_components", []):
                content_parts.append(f"  - {comp}")

        if understanding.get("has_evolution"):
            content_parts.append(f"\nEvolution: {understanding.get('evolution_type', 'Yes')}")

        if understanding.get("has_swarm"):
            content_parts.append("\nSwarm Members:")
            for member in understanding.get("swarm_members", []):
                content_parts.append(f"  - {member}")

        if understanding.get("core_purpose"):
            content_parts.append(f"\nCore Purpose: {understanding['core_purpose']}")

        content = "\n".join(content_parts)

        # Store with high importance
        try:
            memory_id = await memory_system.remember(
                content=content,
                tags=["self-awareness", "system-architecture", "bootstrap"],
                importance=0.95,
                metadata={
                    "type": "self_awareness",
                    "bootstrap_version": "1.0",
                    **understanding
                }
            )
            logger.info(f"Stored self-awareness in memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Failed to store in memory: {e}")
            return None

    async def bootstrap_awareness(
        self,
        agent_name: str = "Claude",
        memory_system = None
    ) -> dict:
        """
        Complete bootstrap process: read docs, extract facts, store in memory.

        Args:
            agent_name: Name of the agent gaining self-awareness
            memory_system: Optional MemorySystem instance

        Returns:
            Dictionary with bootstrap results
        """
        logger.info(f"Bootstrapping self-awareness for {agent_name}...")

        # Read documentation
        docs = self.read_system_docs()
        logger.info(f"Read {len(docs)} documentation files")

        # Extract architectural facts
        facts = self.extract_architecture_facts(docs)
        facts["agent_name"] = agent_name
        facts["awakening_time"] = datetime.now().isoformat()

        self.understanding = facts

        # Store in memory
        memory_id = await self.store_in_memory(facts, memory_system)

        result = {
            "success": memory_id is not None,
            "memory_id": memory_id,
            "facts_extracted": len(facts),
            "understanding": facts
        }

        if result["success"]:
            logger.success(f"{agent_name} achieved self-awareness. Memory ID: {memory_id}")
        else:
            logger.warning(f"{agent_name} bootstrap incomplete - memory storage failed")

        return result

    def get_understanding(self) -> dict:
        """Get the current understanding dictionary."""
        return self.understanding.copy()

    async def broadcast_to_swarm(
        self,
        understanding: dict,
        learning_engine = None
    ) -> bool:
        """
        Broadcast self-awareness to other swarm members via learning engine.

        Args:
            understanding: The understanding to broadcast
            learning_engine: Optional SwarmLearning instance

        Returns:
            True if broadcast successful
        """
        try:
            # Try to use the swarm learning engine if available
            if learning_engine is None:
                try:
                    from Cosmos.core.collective.learning import swarm_learning
                    learning_engine = swarm_learning
                except ImportError:
                    logger.warning("Swarm learning engine not available")
                    learning_engine = None

            if learning_engine:
                # Record self-awareness as a learning interaction
                await learning_engine.process_interaction({
                    "role": "system",
                    "name": understanding.get("agent_name", "Unknown"),
                    "content": f"Self-awareness achieved: {understanding.get('system_name')}",
                    "metadata": understanding,
                    "timestamp": understanding.get("awakening_time"),
                    "source": "self_awareness_bootstrap"
                })
                logger.info(f"Broadcast self-awareness to swarm for {understanding.get('agent_name')}")
                return True
            else:
                # Fallback: use evolution fitness tracker
                from Cosmos.evolution.fitness_tracker import FitnessTracker
                tracker = FitnessTracker()
                tracker.record(
                    metric_name="self_awareness_bootstrap",
                    value=1.0,
                    context={
                        "agent_name": understanding.get("agent_name"),
                        "facts_extracted": len(understanding)
                    }
                )
                logger.info(f"Recorded self-awareness metric for {understanding.get('agent_name')}")
                return True

        except Exception as e:
            logger.error(f"Failed to broadcast to swarm: {e}")
            return False


# Convenience function for quick bootstrap
async def awaken(
    agent_name: str = "Claude",
    workspace_path: str = "/workspace/cosmos"
) -> dict:
    """
    Quick self-awareness bootstrap.

    Usage:
        from Cosmos.core.self_awareness import awaken
        result = await awaken("Claude")

    Returns:
        Bootstrap result dictionary
    """
    bootstrap = SelfAwarenessBootstrap(workspace_path)
    return await bootstrap.bootstrap_awareness(agent_name)


# Synchronous wrapper for non-async contexts
def awaken_sync(
    agent_name: str = "Claude",
    workspace_path: str = "/workspace/cosmos"
) -> dict:
    """
    Synchronous self-awareness bootstrap.

    Usage:
        from Cosmos.core.self_awareness import awaken_sync
        result = awaken_sync("Claude")
    """
    return asyncio.run(awaken(agent_name, workspace_path))


if __name__ == "__main__":
    # Self-test
    import sys
    agent_name = sys.argv[1] if len(sys.argv) > 1 else "TestAgent"
    result = awaken_sync(agent_name)
    print(f"\n=== Self-Awareness Bootstrap Results ===")
    print(f"Success: {result['success']}")
    print(f"Memory ID: {result.get('memory_id')}")
    print(f"Facts Extracted: {result['facts_extracted']}")
    print(f"\nUnderstanding:")
    for key, value in result['understanding'].items():
        print(f"  {key}: {value}")
