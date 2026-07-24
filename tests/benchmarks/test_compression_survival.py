from benchmarks.judge import HeuristicJudge
from benchmarks.runner import run_compression_survival


class SummaryBackend:
    STOPWORDS = {"what", "which", "did", "do", "we", "our", "is", "the", "a", "an", "now"}

    def __init__(self):
        self.memories = []

    def reset(self):
        self.memories = []

    def store(self, content: str, category: str = "factual", scope: str = "global", importance: float = 0.5):
        self.memories.append(content)

    def recall(self, query: str, top_k: int = 10, scope=None):
        tokens = [t for t in query.lower().replace("?", "").split() if t not in self.STOPWORDS]
        scored = []
        for memory in self.memories:
            score = 0
            lower = memory.lower()
            for token in tokens:
                if token in lower:
                    score += 1
            scored.append((score, memory))
        scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        return [memory for score, memory in scored[:top_k] if score > 0]


def test_compression_survival_passes_when_summary_keeps_fact():
    backend = SummaryBackend()
    judge = HeuristicJudge(model="heuristic")
    scenarios = [
        {
            "id": "cs_test_01",
            "compressed_summary": "Session summary: production region is us-east-1 and backup cadence is every 6 hours.",
            "recent_noise": ["Current chat is about editor themes."],
            "query": "What production region did we settle on?",
            "gold_answer": "us-east-1",
            "difficulty": "easy",
        }
    ]

    result = run_compression_survival(backend, scenarios, judge)
    assert result.correct == 1
    assert result.score == 1.0


def test_compression_survival_fails_when_summary_loses_fact():
    backend = SummaryBackend()
    judge = HeuristicJudge(model="heuristic")
    scenarios = [
        {
            "id": "cs_test_02",
            "compressed_summary": "Session summary: the team discussed several deployment ideas.",
            "recent_noise": ["Current chat is about editor themes."],
            "query": "What production region did we settle on?",
            "gold_answer": "us-east-1",
            "difficulty": "easy",
        }
    ]

    result = run_compression_survival(backend, scenarios, judge)
    assert result.correct == 0
    assert result.score == 0.0
