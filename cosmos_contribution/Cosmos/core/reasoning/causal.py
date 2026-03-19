"""
cosmos Causal Reasoning Engine.

"You changed the outcome by measuring it!"

This module implements Causal Inference capabilities (The "Why" and "What If").
It moves beyond correlation to understanding cause-and-effect relationships via Causal Graphs.

Features:
1. Causal Graph Construction: Builds a D.A.G. of events and outcomes.
2. Intervention Modeling (Do-Calculus): Simulates "What happens if I force X?".
3. Counterfactual Reasoning: Analyzes "What would have happened if X hadn't occurred?".
"""

import asyncio
from typing import   Set, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
import networkx as nx

from Cosmos.core.nexus import nexus, Signal, SignalType

@dataclass
class CausalNode:
    name: str
    description: str
    variable_type: str = "binary" # binary, continuous, categorical

class CausalGraph:
    """
    Directed Acyclic Graph representing causal dependencies.
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_causality(self, cause: str, effect: str, strength: float = 1.0, lag: float = 0.0):
        """Register that A causes B."""
        self.graph.add_edge(cause, effect, weight=strength, lag=lag)

    def get_causes(self, effect: str) -> list[str]:
        """Backward tracing (Diagnosis)."""
        if effect not in self.graph:
            return []
        return list(self.graph.predecessors(effect))

    def get_effects(self, cause: str) -> list[str]:
        """Forward tracing (Prediction)."""
        if cause not in self.graph:
            return []
        return list(self.graph.successors(cause))

class CausalEngine:
    """
    The inference engine for causal reasoning.
    """
    def __init__(self):
        self.causal_graph = CausalGraph()
        self._init_base_knowledge()
        
        # Subscribe to learn new causal links
        nexus.subscribe(SignalType.TASK_FAILED, self._learn_from_failure)

    def _init_base_knowledge(self):
        """Bootstrap with some common sense causality."""
        # Software Engineering Causal Priors
        self.causal_graph.add_causality("High_Memory_Usage", "OOM_Kill", 0.9)
        self.causal_graph.add_causality("Missing_Dependency", "Import_Error", 1.0)
        self.causal_graph.add_causality("API_Rate_Limit", "Request_Failure", 1.0)
        self.causal_graph.add_causality("Network_Partition", "Connection_Timeout", 1.0)

    async def _learn_from_failure(self, signal: Signal):
        """
        Infer causality from failures.
        If we took Action A and Result B happened, there might be a link.
        """
        payload = signal.payload
        action = payload.get("action_type")
        error = payload.get("error_type")
        
        if action and error:
            # We suspect a link, but don't know for sure.
            # In a real system, we'd use statistical significance testing.
            # Here, we just log the hypothesis.
            logger.info(f"Causal Inference: Hypothesizing link {action} -> {error}")
            # self.causal_graph.add_causality(action, error, strength=0.5)

    def simulate_intervention(self, intervention_node: str) -> dict[str, float]:
        """
        Simulate Do(X).
        Returns the probability changes in downstream nodes.
        """
        impacts = {}
        if intervention_node not in self.causal_graph.graph:
            return impacts
            
        descendants = nx.descendants(self.causal_graph.graph, intervention_node)
        for node in descendants:
            # Simplified propagation logic
            paths = list(nx.all_simple_paths(self.causal_graph.graph, intervention_node, node))
            max_strength = 0.0
            for path in paths:
                # Multiply weights along path
                strength = 1.0
                for i in range(len(path)-1):
                    strength *= self.causal_graph.graph[path[i]][path[i+1]]['weight']
                max_strength = max(max_strength, strength)
            
            impacts[node] = max_strength
            
        return impacts

    def generate_counterfactual(self, observed_effect: str, original_cause: str, alternative_cause: str) -> str:
        """
        \"If I had done Y instead of X, would Z have happened?\"
        """
        # 1. Abduction: Explain the past (Why did Z happen?)
        causes = self.causal_graph.get_causes(observed_effect)
        if original_cause not in causes:
            return f"Counterfactual unclear: {original_cause} is not a known cause of {observed_effect}."
            
        # 2. Prediction: Simulate the alternative world
        alt_impacts = self.simulate_intervention(alternative_cause)
        
        if observed_effect in alt_impacts and alt_impacts[observed_effect] > 0.5:
            return f"Even if you had done {alternative_cause}, {observed_effect} would likely still have occurred."
        else:
            return f"If you had done {alternative_cause}, {observed_effect} might have been avoided."

# Global Instance
causal_engine = CausalEngine()
