from benchmarks.judge import HeuristicJudge
from benchmarks.runner import run_topic_shift_recall


class TinyBackend:
    STOPWORDS = {"what", "does", "do", "to", "the", "is", "are", "we", "our"}

    def __init__(self):
        self.memories = []

    def reset(self):
        self.memories = []

    def store(self, content: str, category: str = "factual", scope: str = "global", importance: float = 0.5):
        self.memories.append(content)

    def recall(self, query: str, top_k: int = 10, scope=None):
        q = query.lower()
        tokens = [t for t in q.replace("?", "").split() if t not in self.STOPWORDS]
        scored = []
        for memory in self.memories:
            score = 0
            m = memory.lower()
            for token in tokens:
                if token in m:
                    score += 1
            scored.append((score, memory))
        scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        return [memory for score, memory in scored[:top_k] if score > 0]


def test_topic_shift_recall_passes_without_contamination():
    backend = TinyBackend()
    judge = HeuristicJudge(model="heuristic")
    scenarios = [
        {
            "id": "tsr_test_01",
            "topic_a_facts": ["Editor theme preference: dark mode"],
            "topic_b_facts": ["Production deploy region is us-east-1"],
            "query": "What region does production deploy to?",
            "gold_answer": "us-east-1",
            "forbidden_terms": ["dark mode"],
            "difficulty": "easy",
        }
    ]

    result = run_topic_shift_recall(backend, scenarios, judge)
    assert result.correct == 1
    assert result.score == 1.0


def test_topic_shift_recall_fails_when_old_topic_contaminates_answer():
    class ContaminatingBackend(TinyBackend):
        def recall(self, query: str, top_k: int = 10, scope=None):
            # Returns topic A content as the top result
            return [
                "Editor theme preference: dark mode",
                "Production deploy region is us-east-1",
            ]

    backend = ContaminatingBackend()
    judge = HeuristicJudge(model="heuristic")
    scenarios = [
        {
            "id": "tsr_test_02",
            "topic_a_facts": ["Editor theme preference: dark mode"],
            "topic_b_facts": ["Production deploy region is us-east-1"],
            "query": "What region does production deploy to?",
            "gold_answer": "us-east-1",
            "forbidden_terms": ["dark mode"],
            "difficulty": "easy",
        }
    ]

    result = run_topic_shift_recall(backend, scenarios, judge)
    # Top result contains forbidden term "dark mode", so it fails even though
    # the gold answer could be found in the second result
    assert result.correct == 0
