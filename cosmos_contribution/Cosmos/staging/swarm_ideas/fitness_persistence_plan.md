# Development Plan

Task: Implement a persistent storage mechanism to save fitness scores to disk

Here is the concrete implementation plan for persistent fitness score storage:

## 1. Files to Create

**`cosmos/integration/persistence/fitness_storage.py`** - Core async storage engine
**`cosmos/core/fitness/manager.py`** - High-level interface for core systems  
**`tests/integration/test_fitness_storage.py`** - Validation suite

## 2. Functions to Implement

### `cosmos/integration/persistence/fitness_storage.py`

```python
class FitnessStorage:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize with default path: data/fitness_scores.db"""

    async def initialize(self) -> None:
        """Create SQLite schema with indices. Idempotent."""

    async def save_score(
        self,
        agent_id: str,
        score: float,
        metric_type: str = "general_fitness",
        task_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> int:
        """Persist score. Returns