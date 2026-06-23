"""
DAG Attribution Prompt Helpers
==============================

Utilities for building prompts that ask a language model to attribute a set of
observed outcomes (e.g. tool-call results, user feedback signals) back to the
specific nodes in an agent execution DAG that caused them.

Typical usage
-------------
>>> from hermes_agent.prompts.dag_attribution import build_attribution_prompt
>>> prompt = build_attribution_prompt(dag_nodes, observations)
>>> # send `prompt` to your LLM of choice
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Data-model
# ---------------------------------------------------------------------------


@dataclass
class DagNode:
    """A single node in an agent execution DAG.

    Parameters
    ----------
    node_id:
        Unique identifier for this node (e.g. ``"tool_call_3"``).
    label:
        Human-readable name / description of what this node does.
    inputs:
        Mapping of input parameter names to their values at execution time.
    output:
        The value produced by this node, if any.
    parent_ids:
        IDs of nodes whose outputs fed into this node.
    metadata:
        Arbitrary extra information (timestamps, model name, …).
    """

    node_id: str
    label: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    output: Optional[Any] = None
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """An observed outcome that should be attributed to one or more DAG nodes.

    Parameters
    ----------
    observation_id:
        Unique identifier for this observation.
    description:
        Natural-language description of what was observed.
    sentiment:
        Optional valence signal: ``"positive"``, ``"negative"``, or ``"neutral"``.
    raw_value:
        The raw signal value (e.g. a numeric score, a boolean flag, …).
    """

    observation_id: str
    description: str
    sentiment: Optional[str] = None
    raw_value: Optional[Any] = None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_PREAMBLE = """\
You are an expert at analysing agentic AI execution traces.
You will be given a description of a Directed Acyclic Graph (DAG) that \
represents the steps an AI agent took to complete a task, followed by a list \
of observations (outcomes, feedback signals, or errors).

Your job is to attribute each observation to the DAG node or nodes most \
responsible for causing it.  Be precise and concise.  If an observation \
cannot be attributed to any node, say so explicitly.
"""

_ATTRIBUTION_INSTRUCTIONS = """\
For each observation listed below, output a JSON object with the following keys:
  - "observation_id": the ID of the observation
  - "attributed_node_ids": a list of node IDs (may be empty if no node is responsible)
  - "reasoning": a brief explanation (1-3 sentences)

Return a JSON array containing one object per observation.  Do not include \
any text outside the JSON array.
"""


def _format_node(node: DagNode) -> str:
    """Render a single :class:`DagNode` as a compact text block."""
    lines: List[str] = [
        f"Node ID : {node.node_id}",
        f"Label   : {node.label}",
    ]
    if node.parent_ids:
        lines.append(f"Parents : {', '.join(node.parent_ids)}")
    else:
        lines.append("Parents : (none — root node)")
    if node.inputs:
        lines.append("Inputs  :")
        for k, v in node.inputs.items():
            lines.append(f"  {k}: {v!r}")
    if node.output is not None:
        lines.append(f"Output  : {node.output!r}")
    if node.metadata:
        lines.append("Metadata:")
        for k, v in node.metadata.items():
            lines.append(f"  {k}: {v!r}")
    return "\n".join(lines)


def _format_observation(obs: Observation) -> str:
    """Render a single :class:`Observation` as a compact text block."""
    lines: List[str] = [
        f"Observation ID : {obs.observation_id}",
        f"Description    : {obs.description}",
    ]
    if obs.sentiment is not None:
        lines.append(f"Sentiment      : {obs.sentiment}")
    if obs.raw_value is not None:
        lines.append(f"Raw value      : {obs.raw_value!r}")
    return "\n".join(lines)


def build_attribution_prompt(
    nodes: Sequence[DagNode],
    observations: Sequence[Observation],
    *,
    include_system_preamble: bool = True,
    extra_context: Optional[str] = None,
) -> str:
    """Build a full attribution prompt string.

    Parameters
    ----------
    nodes:
        All nodes in the execution DAG, in topological order (parents before
        children) if possible.
    observations:
        The outcomes / feedback signals to attribute.
    include_system_preamble:
        When *True* (default) the returned string starts with the system-level
        instructions.  Set to *False* if you are injecting the preamble via a
        separate ``system`` message in a chat API.
    extra_context:
        Optional free-text block inserted between the DAG description and the
        observations section (e.g. the original user request).

    Returns
    -------
    str
        A ready-to-send prompt string.
    """
    parts: List[str] = []

    if include_system_preamble:
        parts.append(_SYSTEM_PREAMBLE.strip())
        parts.append("")

    # --- DAG section --------------------------------------------------------
    parts.append("## Execution DAG\n")
    if not nodes:
        parts.append("(no nodes provided)")
    else:
        for i, node in enumerate(nodes):
            parts.append(f"### Node {i + 1}\n{_format_node(node)}")
    parts.append("")

    # --- Optional extra context ---------------------------------------------
    if extra_context:
        parts.append("## Additional Context\n")
        parts.append(extra_context.strip())
        parts.append("")

    # --- Observations section -----------------------------------------------
    parts.append("## Observations\n")
    if not observations:
        parts.append("(no observations provided)")
    else:
        for i, obs in enumerate(observations):
            parts.append(f"### Observation {i + 1}\n{_format_observation(obs)}")
    parts.append("")

    # --- Instructions -------------------------------------------------------
    parts.append("## Instructions\n")
    parts.append(_ATTRIBUTION_INSTRUCTIONS.strip())

    return "\n".join(parts)


def build_system_message() -> str:
    """Return just the system preamble, suitable for a ``system`` role message.

    Use this together with ``build_attribution_prompt(include_system_preamble=False)``
    when calling a chat-completion API that accepts a separate system message.
    """
    return _SYSTEM_PREAMBLE.strip()


# ---------------------------------------------------------------------------
# Convenience: parse LLM response
# ---------------------------------------------------------------------------


def parse_attribution_response(
    response_text: str,
) -> List[Dict[str, Any]]:
    """Best-effort parse of the JSON array returned by the LLM.

    Parameters
    ----------
    response_text:
        Raw text returned by the language model.

    Returns
    -------
    list of dict
        Each dict has keys ``observation_id``, ``attributed_node_ids``, and
        ``reasoning``.  Returns an empty list if parsing fails.
    """
    import json
    import re

    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", response_text).strip()
    # Find the outermost JSON array
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return data
