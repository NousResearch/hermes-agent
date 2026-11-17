"""``delegate_task`` runs a nested orchestrator's whole batch inside one tool call by design, so it must not
sit under the generic sequential-call deadline: with it, every batch longer than the deadline "timed out"
while its children kept running as orphans and the orchestrator polled transcripts for hours."""
from agent import tool_executor as te

def test_delegate_task_is_exempt_from_the_sequential_deadline():
    assert "delegate_task" in te._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS


def test_review_is_exempt_from_the_sequential_deadline():
    """``review`` runs its reviewer via ``delegate_task``, synchronously on channels without async
    completions; under the generic deadline a long review returned tool_timeout and lost its verdict."""
    assert "review" in te._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS
