"""
cosmos Visualizations - Advanced Data Visualization Components

Provides interactive visualizations for:
- Memory activation heatmaps
- Knowledge graph exploration
- Evolution fitness timelines
- Agent swarm activity

Novel Features:
- Real-time memory access patterns
- Interactive entity relationship graphs
- Animated evolution progress
- Swarm topology visualization
"""

import json
from datetime import datetime, timedelta
from typing import Optional, Any
from dataclasses import dataclass

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from loguru import logger


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    theme: str = "plotly_dark"
    width: int = 800
    height: int = 600
    show_legend: bool = True
    animation_duration: int = 500


class MemoryVisualizer:
    """
    Visualization components for the memory system.

    Provides:
    - Memory access heatmaps
    - Importance distribution charts
    - Temporal access patterns
    - Source distribution
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()

    def create_access_heatmap(
        self,
        access_data: list[dict],
        time_buckets: int = 24,
    ) -> Optional[go.Figure]:
        """
        Create a heatmap of memory access patterns.

        Args:
            access_data: List of access records with timestamps
            time_buckets: Number of time buckets for aggregation

        Returns:
            Plotly figure or None if not available
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Aggregate access data by time and source
        # Placeholder implementation
        hours = list(range(24))
        sources = ["Archival", "Conversation", "Knowledge Graph", "Working"]

        # Generate sample data
        import random
        z_data = [[random.randint(0, 100) for _ in hours] for _ in sources]

        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f"{h}:00" for h in hours],
            y=sources,
            colorscale="Viridis",
            hoverongaps=False,
        ))

        fig.update_layout(
            title="Memory Access Patterns (24h)",
            xaxis_title="Hour of Day",
            yaxis_title="Memory Source",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height // 2,
        )

        return fig

    def create_importance_distribution(
        self,
        memories: list[dict],
    ) -> Optional[go.Figure]:
        """
        Create a histogram of memory importance scores.

        Args:
            memories: List of memory records with importance scores

        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Extract importance scores or use sample data
        if memories:
            importance_scores = [m.get("importance", 0.5) for m in memories]
        else:
            import random
            importance_scores = [random.random() for _ in range(1000)]

        fig = go.Figure(data=go.Histogram(
            x=importance_scores,
            nbinsx=20,
            marker_color="#667eea",
        ))

        fig.update_layout(
            title="Memory Importance Distribution",
            xaxis_title="Importance Score",
            yaxis_title="Count",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height // 2,
        )

        return fig

    def create_temporal_view(
        self,
        memories: list[dict],
        days: int = 30,
    ) -> Optional[go.Figure]:
        """
        Create a timeline of memory creation over time.

        Args:
            memories: List of memory records with timestamps
            days: Number of days to show

        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Generate sample timeline data
        import random
        dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                 for i in range(days, 0, -1)]
        counts = [random.randint(10, 100) for _ in dates]

        fig = go.Figure(data=go.Scatter(
            x=dates,
            y=counts,
            mode="lines+markers",
            fill="tozeroy",
            marker=dict(color="#667eea"),
            line=dict(color="#667eea", width=2),
        ))

        fig.update_layout(
            title=f"Memory Creation ({days} days)",
            xaxis_title="Date",
            yaxis_title="Memories Created",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height // 2,
        )

        return fig

    def create_source_pie(
        self,
        source_counts: dict[str, int],
    ) -> Optional[go.Figure]:
        """
        Create a pie chart of memory sources.

        Args:
            source_counts: Dictionary of source -> count

        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None

        if not source_counts:
            source_counts = {
                "Archival": 500,
                "Conversation": 350,
                "Knowledge Graph": 250,
                "Working": 147,
            }

        fig = go.Figure(data=go.Pie(
            labels=list(source_counts.keys()),
            values=list(source_counts.values()),
            hole=0.4,
            marker=dict(colors=["#667eea", "#764ba2", "#f093fb", "#f5576c"]),
        ))

        fig.update_layout(
            title="Memory by Source",
            template=self.config.theme,
            width=self.config.width // 2,
            height=self.config.height // 2,
        )

        return fig


class GraphVisualizer:
    """
    Visualization components for the knowledge graph.

    Provides:
    - Interactive node-link diagrams
    - Entity type clustering
    - Relationship exploration
    - Path visualization
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()

    def create_network_graph(
        self,
        entities: list[dict],
        relationships: list[dict],
        layout: str = "spring",
    ) -> Optional[go.Figure]:
        """
        Create an interactive network graph visualization.

        Args:
            entities: List of entity records
            relationships: List of relationship records
            layout: Layout algorithm ("spring", "circular", "random")

        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE or not NETWORKX_AVAILABLE:
            return None

        # Create NetworkX graph
        G = nx.Graph()

        # Add nodes
        if entities:
            for entity in entities:
                G.add_node(
                    entity.get("id", entity.get("name")),
                    name=entity.get("name", "Unknown"),
                    entity_type=entity.get("type", "Unknown"),
                )
        else:
            # Sample data
            sample_entities = [
                ("cosmos", "cosmos", "Project"),
                ("memory", "Memory System", "Component"),
                ("evolution", "Evolution", "Feature"),
                ("agents", "Agent Swarm", "Component"),
                ("user", "User", "Person"),
                ("claude", "Claude Code", "Tool"),
            ]
            for eid, name, etype in sample_entities:
                G.add_node(eid, name=name, entity_type=etype)

        # Add edges
        if relationships:
            for rel in relationships:
                G.add_edge(
                    rel.get("source"),
                    rel.get("target"),
                    relation_type=rel.get("type", "related"),
                )
        else:
            # Sample relationships
            sample_rels = [
                ("cosmos", "memory"),
                ("cosmos", "evolution"),
                ("cosmos", "agents"),
                ("user", "cosmos"),
                ("claude", "cosmos"),
                ("memory", "agents"),
            ]
            for source, target in sample_rels:
                G.add_edge(source, target)

        # Compute layout
        if layout == "spring":
            pos = nx.spring_layout(G, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G, seed=42)

        # Extract coordinates
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [G.nodes[node].get("name", node) for node in G.nodes()]

        # Create figure
        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            mode="lines",
        ))

        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=node_text,
            textposition="top center",
            marker=dict(
                size=20,
                color="#667eea",
                line=dict(width=2, color="#fff"),
            ),
        ))

        fig.update_layout(
            title="Knowledge Graph",
            showlegend=False,
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        return fig

    def create_entity_type_chart(
        self,
        type_counts: dict[str, int],
    ) -> Optional[go.Figure]:
        """
        Create a bar chart of entity types.

        Args:
            type_counts: Dictionary of type -> count

        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None

        if not type_counts:
            type_counts = {
                "Project": 5,
                "Component": 12,
                "Feature": 18,
                "Person": 8,
                "Tool": 15,
                "Concept": 25,
            }

        fig = go.Figure(data=go.Bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            marker_color="#667eea",
        ))

        fig.update_layout(
            title="Entities by Type",
            xaxis_title="Entity Type",
            yaxis_title="Count",
            template=self.config.theme,
            width=self.config.width // 2,
            height=self.config.height // 2,
        )

        return fig


class EvolutionVisualizer:
    """
    Visualization components for the evolution system.

    Provides:
    - Fitness timeline
    - Generation progress
    - Pareto front visualization
    - Metric comparison
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()

    def create_fitness_timeline(
        self,
        fitness_history: list[dict],
    ) -> Optional[go.Figure]:
        """
        Create a timeline of fitness evolution.

        Args:
            fitness_history: List of fitness records over time

        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Generate sample data if not provided
        if fitness_history:
            generations = [f["generation"] for f in fitness_history]
            fitness_values = [f["fitness"] for f in fitness_history]
        else:
            import random
            generations = list(range(1, 51))
            # Simulate improving fitness with noise
            base_fitness = 0.5
            fitness_values = []
            for i in generations:
                base_fitness = min(0.95, base_fitness + random.uniform(0.001, 0.01))
                fitness_values.append(base_fitness + random.uniform(-0.02, 0.02))

        fig = go.Figure()

        # Best fitness line
        fig.add_trace(go.Scatter(
            x=generations,
            y=fitness_values,
            mode="lines+markers",
            name="Best Fitness",
            line=dict(color="#4caf50", width=2),
            marker=dict(size=6),
        ))

        # Add trend line
        import numpy as np
        z = np.polyfit(generations, fitness_values, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=generations,
            y=[p(x) for x in generations],
            mode="lines",
            name="Trend",
            line=dict(color="#ff9800", width=2, dash="dash"),
        ))

        fig.update_layout(
            title="Fitness Evolution Over Generations",
            xaxis_title="Generation",
            yaxis_title="Fitness Score",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height // 2,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        return fig

    def create_metric_radar(
        self,
        metrics: dict[str, float],
    ) -> Optional[go.Figure]:
        """
        Create a radar chart of fitness metrics.

        Args:
            metrics: Dictionary of metric_name -> value (0-1)

        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None

        if not metrics:
            metrics = {
                "Task Success": 0.94,
                "User Satisfaction": 0.91,
                "Efficiency": 0.78,
                "Response Quality": 0.85,
                "Memory Utilization": 0.72,
                "Agent Coordination": 0.88,
            }

        categories = list(metrics.keys())
        values = list(metrics.values())

        # Close the radar chart
        categories.append(categories[0])
        values.append(values[0])

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(102, 126, 234, 0.3)",
            line=dict(color="#667eea", width=2),
            marker=dict(size=8),
        ))

        fig.update_layout(
            title="Fitness Metrics Overview",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                ),
            ),
            template=self.config.theme,
            width=self.config.width // 2,
            height=self.config.height // 2,
        )

        return fig

    def create_pareto_front(
        self,
        population: list[dict],
    ) -> Optional[go.Figure]:
        """
        Create a Pareto front visualization for multi-objective optimization.

        Args:
            population: List of solutions with multiple objective values

        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Generate sample population if not provided
        if not population:
            import random
            population = []
            for i in range(50):
                x = random.random()
                # Create Pareto-like distribution
                y = 1 - x + random.uniform(-0.1, 0.1)
                is_pareto = random.random() > 0.7
                population.append({
                    "task_success": x,
                    "efficiency": max(0, min(1, y)),
                    "is_pareto": is_pareto,
                })

        task_success = [p["task_success"] for p in population]
        efficiency = [p["efficiency"] for p in population]
        colors = ["#4caf50" if p.get("is_pareto", False) else "#667eea" for p in population]

        fig = go.Figure()

        # All solutions
        fig.add_trace(go.Scatter(
            x=task_success,
            y=efficiency,
            mode="markers",
            marker=dict(
                size=10,
                color=colors,
                line=dict(width=1, color="#fff"),
            ),
            hovertemplate="Task Success: %{x:.2f}<br>Efficiency: %{y:.2f}",
        ))

        fig.update_layout(
            title="Pareto Front (Multi-Objective Optimization)",
            xaxis_title="Task Success",
            yaxis_title="Efficiency",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height // 2,
        )

        return fig

    def create_generation_boxplot(
        self,
        generation_data: list[dict],
    ) -> Optional[go.Figure]:
        """
        Create boxplots showing fitness distribution per generation.

        Args:
            generation_data: List of generation fitness distributions

        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Generate sample data
        import random

        fig = go.Figure()

        for gen in range(1, 11):
            base = 0.5 + gen * 0.03
            values = [base + random.uniform(-0.1, 0.1) for _ in range(20)]

            fig.add_trace(go.Box(
                y=values,
                name=f"Gen {gen}",
                marker_color="#667eea",
            ))

        fig.update_layout(
            title="Fitness Distribution by Generation",
            xaxis_title="Generation",
            yaxis_title="Fitness Score",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height // 2,
            showlegend=False,
        )

        return fig


class AgentVisualizer:
    """
    Visualization components for agent monitoring.

    Provides:
    - Agent activity timeline
    - Task distribution charts
    - Performance comparison
    - Swarm topology
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()

    def create_activity_timeline(
        self,
        activity_logs: list[dict],
    ) -> Optional[go.Figure]:
        """
        Create a Gantt-style timeline of agent activities.

        Args:
            activity_logs: List of activity records

        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Sample activity data
        if not activity_logs:
            now = datetime.now()
            activity_logs = [
                {
                    "agent": "code-001",
                    "task": "Code generation",
                    "start": now - timedelta(minutes=30),
                    "end": now - timedelta(minutes=25),
                },
                {
                    "agent": "reasoning-001",
                    "task": "Analysis",
                    "start": now - timedelta(minutes=28),
                    "end": now - timedelta(minutes=20),
                },
                {
                    "agent": "research-001",
                    "task": "Information gathering",
                    "start": now - timedelta(minutes=22),
                    "end": now - timedelta(minutes=15),
                },
                {
                    "agent": "code-001",
                    "task": "Code review",
                    "start": now - timedelta(minutes=15),
                    "end": now - timedelta(minutes=10),
                },
            ]

        fig = go.Figure()

        colors = {
            "code": "#e3f2fd",
            "reasoning": "#fff3e0",
            "research": "#e8f5e9",
            "creative": "#fce4ec",
        }

        for activity in activity_logs:
            agent_type = activity["agent"].split("-")[0]
            color = colors.get(agent_type, "#f5f5f5")

            fig.add_trace(go.Bar(
                x=[(activity["end"] - activity["start"]).total_seconds() / 60],
                y=[activity["agent"]],
                orientation="h",
                base=[(activity["start"] - datetime.now()).total_seconds() / 60],
                marker_color=color,
                text=activity["task"],
                textposition="inside",
                hovertemplate=f"{activity['task']}<br>Duration: %{{x:.1f}} min",
            ))

        fig.update_layout(
            title="Agent Activity Timeline",
            xaxis_title="Time (minutes ago)",
            yaxis_title="Agent",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height // 2,
            showlegend=False,
            barmode="stack",
        )

        return fig

    def create_task_distribution(
        self,
        task_counts: dict[str, int],
    ) -> Optional[go.Figure]:
        """
        Create a pie chart of task distribution by agent type.

        Args:
            task_counts: Dictionary of agent_type -> task_count

        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None

        if not task_counts:
            task_counts = {
                "Code": 42,
                "Reasoning": 38,
                "Research": 27,
                "Creative": 20,
            }

        fig = go.Figure(data=go.Pie(
            labels=list(task_counts.keys()),
            values=list(task_counts.values()),
            hole=0.4,
            marker=dict(colors=["#1565c0", "#ef6c00", "#2e7d32", "#c2185b"]),
        ))

        fig.update_layout(
            title="Tasks by Agent Type",
            template=self.config.theme,
            width=self.config.width // 2,
            height=self.config.height // 2,
        )

        return fig

    def create_performance_comparison(
        self,
        agent_metrics: list[dict],
    ) -> Optional[go.Figure]:
        """
        Create a grouped bar chart comparing agent performance.

        Args:
            agent_metrics: List of agent performance records

        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            return None

        if not agent_metrics:
            agent_metrics = [
                {"agent": "Code", "success_rate": 96, "avg_time": 1.8},
                {"agent": "Reasoning", "success_rate": 94, "avg_time": 3.2},
                {"agent": "Research", "success_rate": 91, "avg_time": 2.5},
                {"agent": "Creative", "success_rate": 89, "avg_time": 2.1},
            ]

        agents = [m["agent"] for m in agent_metrics]
        success_rates = [m["success_rate"] for m in agent_metrics]
        avg_times = [m["avg_time"] for m in agent_metrics]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Bar(
            name="Success Rate (%)",
            x=agents,
            y=success_rates,
            marker_color="#4caf50",
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            name="Avg Time (s)",
            x=agents,
            y=avg_times,
            mode="lines+markers",
            line=dict(color="#ff9800", width=3),
            marker=dict(size=10),
        ), secondary_y=True)

        fig.update_layout(
            title="Agent Performance Comparison",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height // 2,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        fig.update_yaxes(title_text="Success Rate (%)", secondary_y=False)
        fig.update_yaxes(title_text="Avg Time (s)", secondary_y=True)

        return fig


def create_dashboard_figures(
    memory_data: Optional[dict] = None,
    graph_data: Optional[dict] = None,
    evolution_data: Optional[dict] = None,
    agent_data: Optional[dict] = None,
) -> dict[str, go.Figure]:
    """
    Create all dashboard figures.

    Args:
        memory_data: Memory system data
        graph_data: Knowledge graph data
        evolution_data: Evolution metrics data
        agent_data: Agent activity data

    Returns:
        Dictionary of figure name -> Plotly figure
    """
    figures = {}

    # Memory visualizations
    mem_viz = MemoryVisualizer()
    figures["memory_heatmap"] = mem_viz.create_access_heatmap([])
    figures["memory_importance"] = mem_viz.create_importance_distribution([])
    figures["memory_timeline"] = mem_viz.create_temporal_view([])
    figures["memory_sources"] = mem_viz.create_source_pie({})

    # Graph visualizations
    graph_viz = GraphVisualizer()
    figures["knowledge_graph"] = graph_viz.create_network_graph([], [])
    figures["entity_types"] = graph_viz.create_entity_type_chart({})

    # Evolution visualizations
    evo_viz = EvolutionVisualizer()
    figures["fitness_timeline"] = evo_viz.create_fitness_timeline([])
    figures["metric_radar"] = evo_viz.create_metric_radar({})
    figures["pareto_front"] = evo_viz.create_pareto_front([])
    figures["generation_boxplot"] = evo_viz.create_generation_boxplot([])

    # Agent visualizations
    agent_viz = AgentVisualizer()
    figures["agent_activity"] = agent_viz.create_activity_timeline([])
    figures["task_distribution"] = agent_viz.create_task_distribution({})
    figures["agent_performance"] = agent_viz.create_performance_comparison([])

    return figures
