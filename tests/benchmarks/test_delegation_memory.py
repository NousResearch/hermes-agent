from benchmarks.judge import HeuristicJudge
from benchmarks.runner import run_delegation_memory


class DelegationBackend:
    def __init__(self):
        self.memories = []

    def reset(self):
        self.memories = []

    def store(self, content: str, category: str = "factual", scope: str = "global", importance: float = 0.5):
        self.memories.append(content)

    def recall(self, query: str, top_k: int = 10, scope=None):
        # Simple: return all stored memories (most recently stored first)
        return list(reversed(self.memories[:top_k]))


def test_delegation_memory_passes_when_result_contains_answer():
    backend = DelegationBackend()
    judge = HeuristicJudge(model="heuristic")
    scenarios = [
        {
            "id": "dm_test_01",
            "delegation_task": "Investigate deployment region.",
            "delegation_result": "Recommendation: deploy production in us-east-1.",
            "query": "What region did the delegated deployment investigation recommend?",
            "gold_answer": "us-east-1",
            "difficulty": "easy",
        }
    ]

    result = run_delegation_memory(backend, scenarios, judge)
    # The result is stored second, so with reverse ordering it comes first
    assert result.correct == 1
    assert result.score == 1.0


def test_delegation_memory_fails_when_result_does_not_contain_answer():
    backend = DelegationBackend()
    judge = HeuristicJudge(model="heuristic")
    scenarios = [
        {
            "id": "dm_test_02",
            "delegation_task": "Investigate deployment region.",
            "delegation_result": "Recommendation: the team discussed several possible regions.",
            "query": "What region did the delegated deployment investigation recommend?",
            "gold_answer": "us-east-1",
            "difficulty": "easy",
        }
    ]

    result = run_delegation_memory(backend, scenarios, judge)
    assert result.correct == 0
    assert result.score == 0.0
