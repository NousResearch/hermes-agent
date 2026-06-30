from typing import Any

from .nodes import (
    tax_data_extractor_node,
    tax_field_mapper_node,
    tax_optimizer_node,
    tax_validator_node,
)
from .state import TaxPipelineState


def route_after_validation(state: TaxPipelineState) -> str:
    """Route unresolved states to human review, otherwise continue to AfA."""
    if state.get("requires_clarification", False):
        return "human_review_node"
    return "tax_optimizer_node"


def human_review_node(state: TaxPipelineState) -> dict[str, Any]:
    """Dashboard handoff point for structured ClarificationRequest objects."""
    return {}


def build_tax_pipeline_graph(checkpointer: Any | None = None) -> Any:
    """Build the LangGraph workflow.

    LangGraph is intentionally imported lazily because it is not a core Hermes
    dependency in this checkout. Install it before running this graph.
    """
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph is required to run the tax pipeline graph.") from exc

    workflow = StateGraph(TaxPipelineState)
    workflow.add_node("tax_data_extractor_node", tax_data_extractor_node)
    workflow.add_node("tax_field_mapper_node", tax_field_mapper_node)
    workflow.add_node("tax_validator_node", tax_validator_node)
    workflow.add_node("tax_optimizer_node", tax_optimizer_node)
    workflow.add_node("human_review_node", human_review_node)

    workflow.set_entry_point("tax_data_extractor_node")
    workflow.add_edge("tax_data_extractor_node", "tax_field_mapper_node")
    workflow.add_edge("tax_field_mapper_node", "tax_validator_node")
    workflow.add_conditional_edges(
        "tax_validator_node",
        route_after_validation,
        {
            "human_review_node": "human_review_node",
            "tax_optimizer_node": "tax_optimizer_node",
        },
    )
    workflow.add_edge("human_review_node", "tax_validator_node")
    workflow.add_edge("tax_optimizer_node", END)

    return workflow.compile(checkpointer=checkpointer or MemorySaver())
