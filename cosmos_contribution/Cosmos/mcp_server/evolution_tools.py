"""
cosmos Evolution Tools - MCP Tool Implementations for Evolution Operations

Provides evolution and feedback capabilities:
- Fitness tracking and feedback
- Evolution metrics
- System improvement triggers
"""

from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime

from loguru import logger


@dataclass
class EvolutionToolResult:
    """Result from an evolution tool operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        result = {"success": self.success}
        if self.data is not None:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        return result


class EvolutionTools:
    """
    Evolution tool implementations for the MCP server.

    Provides access to the evolution system for feedback and improvement.
    """

    def __init__(self, fitness_tracker, genetic_optimizer=None, behavior_mutator=None):
        self.fitness = fitness_tracker
        self.optimizer = genetic_optimizer
        self.mutator = behavior_mutator

    async def record_feedback(
        self,
        feedback_type: str,
        value: float,
        context: Optional[dict] = None,
    ) -> EvolutionToolResult:
        """
        Record explicit feedback for system improvement.

        Args:
            feedback_type: Type of feedback ("satisfaction", "quality", "speed")
            value: Feedback value (0-1 scale)
            context: Optional context about the feedback

        Returns:
            EvolutionToolResult
        """
        try:
            # Map feedback types to internal metrics
            metric_map = {
                "satisfaction": "user_satisfaction",
                "quality": "task_success",
                "speed": "efficiency",
                "helpfulness": "user_satisfaction",
                "accuracy": "task_success",
            }

            metric_name = metric_map.get(feedback_type, feedback_type)
            self.fitness.record(metric_name, value)

            return EvolutionToolResult(
                success=True,
                data={
                    "recorded": True,
                    "metric": metric_name,
                    "value": value,
                    "current_fitness": self.fitness.get_weighted_fitness(),
                },
            )

        except Exception as e:
            logger.error(f"Record feedback failed: {e}")
            return EvolutionToolResult(success=False, error=str(e))

    async def record_task_result(
        self,
        success: bool,
        tokens_used: int = 0,
        time_seconds: float = 0.0,
        user_rating: Optional[float] = None,
    ) -> EvolutionToolResult:
        """
        Record a task completion result for fitness tracking.

        Args:
            success: Whether the task was successful
            tokens_used: Number of tokens consumed
            time_seconds: Task execution time
            user_rating: Optional user rating (0-1)

        Returns:
            EvolutionToolResult
        """
        try:
            self.fitness.record_task_outcome(
                success=success,
                tokens_used=tokens_used,
                time_seconds=time_seconds,
                user_feedback=user_rating,
            )

            return EvolutionToolResult(
                success=True,
                data={
                    "recorded": True,
                    "current_fitness": self.fitness.get_current_fitness(),
                },
            )

        except Exception as e:
            logger.error(f"Record task result failed: {e}")
            return EvolutionToolResult(success=False, error=str(e))

    async def get_fitness_metrics(self) -> EvolutionToolResult:
        """
        Get current fitness metrics and trends.

        Returns:
            EvolutionToolResult with fitness data
        """
        try:
            stats = self.fitness.get_stats()

            return EvolutionToolResult(
                success=True,
                data={
                    "current_fitness": self.fitness.get_current_fitness(),
                    "weighted_fitness": self.fitness.get_weighted_fitness(),
                    "metrics": stats.get("current_fitness", {}),
                    "trends": stats.get("trends", {}),
                    "sample_counts": stats.get("sample_counts", {}),
                },
            )

        except Exception as e:
            logger.error(f"Get fitness metrics failed: {e}")
            return EvolutionToolResult(success=False, error=str(e))

    async def get_leaderboard(self, top_k: int = 10) -> EvolutionToolResult:
        """
        Get top performing configurations/genomes.

        Returns:
            EvolutionToolResult with leaderboard
        """
        try:
            leaderboard = self.fitness.get_leaderboard(top_k)

            return EvolutionToolResult(
                success=True,
                data={
                    "leaderboard": [
                        {"genome_id": gid, "fitness": score}
                        for gid, score in leaderboard
                    ],
                },
            )

        except Exception as e:
            logger.error(f"Get leaderboard failed: {e}")
            return EvolutionToolResult(success=False, error=str(e))

    async def trigger_evolution(self, generations: int = 1) -> EvolutionToolResult:
        """
        Trigger evolution cycle.

        Args:
            generations: Number of generations to evolve

        Returns:
            EvolutionToolResult with evolution results
        """
        try:
            if self.optimizer is None:
                return EvolutionToolResult(
                    success=False,
                    error="Genetic optimizer not configured",
                )

            # Run evolution
            result = await self.optimizer.run(generations=generations)

            return EvolutionToolResult(
                success=True,
                data={
                    "generations_run": result.generations_run,
                    "best_fitness": result.best_genome.total_fitness(),
                    "duration_seconds": result.duration_seconds,
                },
            )

        except Exception as e:
            logger.error(f"Trigger evolution failed: {e}")
            return EvolutionToolResult(success=False, error=str(e))

    async def get_behavior_params(self) -> EvolutionToolResult:
        """
        Get current behavioral parameters.

        Returns:
            EvolutionToolResult with behavior configuration
        """
        try:
            if self.mutator is None:
                return EvolutionToolResult(
                    success=False,
                    error="Behavior mutator not configured",
                )

            return EvolutionToolResult(
                success=True,
                data={
                    "behavior_params": self.mutator.get_behavior_params(),
                    "team_config": self.mutator.get_team_config(),
                    "generation": self.mutator.generation,
                },
            )

        except Exception as e:
            logger.error(f"Get behavior params failed: {e}")
            return EvolutionToolResult(success=False, error=str(e))

    async def evolve_behavior(self) -> EvolutionToolResult:
        """
        Trigger behavior evolution.

        Returns:
            EvolutionToolResult with new behavior configuration
        """
        try:
            if self.mutator is None:
                return EvolutionToolResult(
                    success=False,
                    error="Behavior mutator not configured",
                )

            self.mutator.evolve_generation()

            return EvolutionToolResult(
                success=True,
                data={
                    "new_generation": self.mutator.generation,
                    "new_behavior": self.mutator.get_behavior_params(),
                    "new_team": self.mutator.get_team_config(),
                },
            )

        except Exception as e:
            logger.error(f"Evolve behavior failed: {e}")
            return EvolutionToolResult(success=False, error=str(e))

    async def get_improvement_suggestions(self) -> EvolutionToolResult:
        """
        Get AI-generated improvement suggestions based on metrics.

        Returns:
            EvolutionToolResult with suggestions
        """
        try:
            current = self.fitness.get_current_fitness()
            trends = {name: self.fitness.get_trend(name) for name in current.keys()}

            suggestions = []

            # Analyze metrics and generate suggestions
            if current.get("task_success", 0) < 0.7:
                suggestions.append({
                    "area": "task_success",
                    "issue": "Low task success rate",
                    "suggestion": "Consider enabling cascade inference or switching to a more capable model",
                })

            if current.get("efficiency", 0) < 0.5:
                suggestions.append({
                    "area": "efficiency",
                    "issue": "High token usage",
                    "suggestion": "Enable speculative decoding or use a smaller draft model",
                })

            if current.get("user_satisfaction", 0) < 0.6:
                suggestions.append({
                    "area": "user_satisfaction",
                    "issue": "Low user satisfaction",
                    "suggestion": "Review recent feedback and adjust response style or verbosity",
                })

            # Check trends
            for metric, trend in trends.items():
                if trend < -0.1:
                    suggestions.append({
                        "area": metric,
                        "issue": f"Declining {metric}",
                        "suggestion": f"Investigate recent changes affecting {metric}",
                    })

            return EvolutionToolResult(
                success=True,
                data={
                    "suggestions": suggestions,
                    "current_metrics": current,
                    "trends": trends,
                },
            )

        except Exception as e:
            logger.error(f"Get improvement suggestions failed: {e}")
            return EvolutionToolResult(success=False, error=str(e))
