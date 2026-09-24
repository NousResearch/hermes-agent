"""Turning a model provider's error text into something an operator can act on.

Portable on purpose: the same classification explains a failed task, the model-access card
and a preflight probe, and the words a provider uses ("Operation not allowed",
"AccessDeniedException", "on-demand throughput isn't supported") do not depend on which
runtime made the call.

Every category carries a headline (what happened, in the customer's words), an ``owner``
(who can fix it — a NOVA operator, the AWS account admin, or nobody but time) and a remedy.
The raw text always travels beside it: a summary that replaced the evidence would make the
next incident harder, not easier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelError:
    category: str
    headline: str
    remedy: str
    owner: str
    raw: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


#: Ordered: the first match wins, so the specific patterns sit above the generic ones.
#: "Operation not allowed" is what Bedrock returns when the *account* is not entitled to a
#: model, which reads like a permissions error and is not one — the IAM grant can be exactly
#: right and the call still fails, so it is matched before the access-denied family.
_RULES: tuple[tuple[tuple[str, ...], str, str, str, str], ...] = (
    (
        ("operation not allowed", "use case", "not authorized to access this model",
         "model access", "account is not authorized"),
        "account_not_authorized",
        "The cloud account is not authorized to use this model yet",
        "Submit the provider's model-access request (for Anthropic on Bedrock: the use-case "
        "form in the Bedrock console), or wait for the provider's support team.",
        "account admin",
    ),
    (
        ("accessdenied", "access denied", "not authorized to perform", "403", "forbidden"),
        "permission_denied",
        "The runtime's role is not allowed to call this model",
        "Grant invoke permission on the exact model and inference-profile resources to the "
        "runtime role, then re-apply the infrastructure plan.",
        "operator",
    ),
    (
        ("on-demand throughput", "inference profile", "isn't supported", "is not supported"),
        "needs_inference_profile",
        "This model must be called through an inference profile",
        "Use the region-prefixed inference-profile id (for example eu.<model-id>) as the "
        "deployment's model.",
        "operator",
    ),
    (
        ("model identifier is invalid", "could not resolve", "model not found",
         "resourcenotfound", "404", "does not exist"),
        "model_not_found",
        "The configured model id does not exist here",
        "Check the model id and region in deployment.yaml against the provider's catalogue.",
        "operator",
    ),
    (
        # Above the throttling family: both arrive as a 429, but a per-day quota does not
        # clear with a retry a minute later, and treating it as throttling spends every
        # retry the task has on a wall that stays up until the quota resets.
        ("perday", "per-day", "per day", "daily limit", "daily quota"),
        "quota_exhausted",
        "The provider's daily quota for this model is used up",
        "Wait for the quota to reset (usually midnight in the provider's time zone), add "
        "credits or a paid plan, or switch the deployment to another model.",
        "account admin",
    ),
    (
        ("overloaded", "high demand", "temporarily unavailable", "503", "529"),
        "overloaded",
        "The model provider is temporarily overloaded",
        "Nothing to change — the retry will try again. If it persists, switch model or provider.",
        "nobody",
    ),
    (
        ("throttl", "too many requests", "rate exceeded", "429", "servicequota",
         "resource_exhausted", "quota exceeded"),
        "throttled",
        "The provider is rate-limiting requests",
        "Retry later, or request a higher quota for this model.",
        "account admin",
    ),
    (
        ("payment", "credit", "billing", "402", "insufficient"),
        "billing",
        "The provider refused the call for billing reasons",
        "Check the provider account's billing status.",
        "account admin",
    ),
    (
        ("timed out", "timeout", "connection", "name resolution", "unreachable", "ssl"),
        "network",
        "The runtime could not reach the model provider",
        "Check outbound networking (NAT or VPC endpoint) from the runtime to the provider.",
        "operator",
    ),
)


def classify_model_error(text: str) -> ModelError:
    """The best explanation for one provider error. Never raises; unknown text is 'error'."""
    raw = " ".join(str(text or "").split())
    lowered = raw.lower()
    for needles, category, headline, remedy, owner in _RULES:
        if any(needle in lowered for needle in needles):
            return ModelError(category, headline, remedy, owner, raw)
    return ModelError(
        "error",
        "The model provider returned an error",
        "Open the details for the provider's own message.",
        "operator",
        raw,
    )


#: What the runtime writes when a worker ends without reporting an outcome. It is the
#: symptom every model failure produces in a dispatched task, so it gets a plain reading
#: of its own and, where the model record explains it, that cause beside it.
_PROTOCOL_VIOLATION = ("protocol violation", "without calling kanban_complete")


def summarize_task_error(text: str) -> dict:
    """``{headline, detail}`` for a task's last error, in operator language."""
    raw = " ".join(str(text or "").split())
    lowered = raw.lower()
    if not raw:
        return {"headline": "", "detail": ""}
    if any(needle in lowered for needle in _PROTOCOL_VIOLATION):
        return {
            "headline": "The agent stopped without reporting a result",
            "detail": "Its run ended before it could mark the work done or blocked, so the "
                      "board counts the attempt as failed.",
        }
    if "timed out" in lowered or "max_runtime" in lowered or "runtime cap" in lowered:
        return {
            "headline": "The work ran past its time limit",
            "detail": "The dispatcher stopped it at the agent's maximum task runtime.",
        }
    if lowered.startswith("nova budget:"):
        return {
            "headline": "Monthly budget reached",
            "detail": "This agent's (or the whole tenant's) budget for the month is spent, so "
                      "it takes no new work. Raise the budget in the agent's limits or the "
                      "deployment, or wait for the month to turn, then retry.",
        }
    if "nova delegation policy" in lowered:
        return {
            "headline": "Handed to an agent that may not take it",
            "detail": "No declared hand-off covers this assignment. Allow it in the delegating "
                      "agent's Hand-offs, or give the task to another agent, then retry.",
        }
    model = classify_model_error(raw)
    if model.category != "error":
        return {"headline": model.headline, "detail": model.remedy}
    return {"headline": "The work failed", "detail": ""}


#: Block reasons NOVA itself records (nova-outcome, nova-policy). They mean the run failed
#: for a reason a person must fix — not that the agent is waiting on a question — so the
#: dashboard treats them as failures to retry, not decisions to make.
NOVA_BLOCK_PREFIXES = ("the ai model refused the request", "nova delegation policy:", "nova budget:")


def is_nova_failure_block(reason: str) -> bool:
    return str(reason or "").strip().lower().startswith(NOVA_BLOCK_PREFIXES)
