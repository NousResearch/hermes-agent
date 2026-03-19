"""
cosmos Swarm Health Advisor

Multi-agent health advisory system using specialist agents for
nutrition, fitness, sleep, and stress management.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

from .models import (
    HealthMetricReading,
    DailySummary,
    HealthRecommendation,
    UserHealthProfile,
    MealEntry,
    Recipe,
    NutrientInfo,
)
from .analysis import HealthAnalysisEngine, HealthInsight

logger = logging.getLogger(__name__)


@dataclass
class AdvisorContext:
    """Context for advisor agents."""
    user_profile: UserHealthProfile
    recent_summaries: List[DailySummary]
    recent_meals: List[MealEntry]
    active_goals: List[Dict[str, Any]]
    insights: List[HealthInsight]
    focus_area: Optional[str] = None

    def to_prompt_context(self) -> str:
        """Generate context string for LLM prompt."""
        lines = ["# User Health Context\n"]

        # Profile info
        if self.user_profile.age:
            lines.append(f"- Age: {self.user_profile.age}")
        if self.user_profile.gender:
            lines.append(f"- Gender: {self.user_profile.gender}")
        if self.user_profile.fitness_level:
            lines.append(f"- Fitness Level: {self.user_profile.fitness_level}")

        # Dietary info
        if self.user_profile.dietary_restrictions:
            lines.append(f"- Dietary Restrictions: {', '.join(self.user_profile.dietary_restrictions)}")
        if self.user_profile.food_allergies:
            lines.append(f"- Allergies: {', '.join(self.user_profile.food_allergies)}")

        # Targets
        if self.user_profile.daily_calorie_target:
            lines.append(f"- Daily Calorie Target: {self.user_profile.daily_calorie_target}")
        if self.user_profile.daily_steps_target:
            lines.append(f"- Daily Steps Target: {self.user_profile.daily_steps_target}")
        if self.user_profile.sleep_target_hours:
            lines.append(f"- Sleep Target: {self.user_profile.sleep_target_hours} hours")

        # Recent health data
        if self.recent_summaries:
            lines.append("\n## Recent Health Data (Last 7 Days)")
            for summary in self.recent_summaries[:7]:
                lines.append(f"\n### {summary.date}")
                lines.append(f"- Steps: {summary.total_steps}")
                lines.append(f"- Sleep: {summary.sleep_duration_hours:.1f} hours (score: {summary.sleep_score or 'N/A'})")
                lines.append(f"- Recovery: {summary.recovery_score or 'N/A'}")
                if summary.avg_heart_rate:
                    lines.append(f"- Avg HR: {summary.avg_heart_rate:.0f} bpm")
                lines.append(f"- Calories burned: {summary.total_calories_burned}")

        # Active insights
        if self.insights:
            lines.append("\n## Current Insights")
            for insight in self.insights[:5]:
                lines.append(f"- [{insight.category}] {insight.title}: {insight.message}")

        # Focus area
        if self.focus_area:
            lines.append(f"\n## Focus Area: {self.focus_area}")

        return "\n".join(lines)


@dataclass
class AdvisorResponse:
    """Response from an advisor agent."""
    agent_name: str
    category: str
    recommendations: List[HealthRecommendation]
    confidence: float
    reasoning: str


class HealthAdvisorAgent(ABC):
    """Base class for health advisor specialist agents."""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.llm_backend = None

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    @abstractmethod
    async def generate_recommendations(
        self,
        context: AdvisorContext,
    ) -> List[HealthRecommendation]:
        """Generate recommendations based on context."""
        pass

    def set_llm_backend(self, backend):
        """Set the LLM backend for this agent."""
        self.llm_backend = backend


class NutritionAdvisorAgent(HealthAdvisorAgent):
    """Specialist agent for nutrition and diet advice."""

    def __init__(self):
        super().__init__("NutritionAdvisor", "nutrition")

    @property
    def system_prompt(self) -> str:
        return """You are a nutrition specialist AI advisor. Your role is to:

1. Analyze dietary patterns and nutritional intake
2. Identify nutritional deficiencies or excesses
3. Suggest meal improvements and recipe ideas
4. Consider dietary restrictions and allergies
5. Provide practical, actionable nutrition advice

Focus on evidence-based nutrition science. Be specific with recommendations.
Consider the user's lifestyle, preferences, and health goals.
Always explain the reasoning behind your suggestions."""

    async def generate_recommendations(
        self,
        context: AdvisorContext,
    ) -> List[HealthRecommendation]:
        """Generate nutrition recommendations."""
        recommendations = []

        # Analyze recent nutrition
        if context.recent_meals:
            total_cals = sum(m.total_nutrients.calories for m in context.recent_meals[-7:])
            avg_daily_cals = total_cals / 7 if context.recent_meals else 0

            target = context.user_profile.daily_calorie_target or 2000

            if avg_daily_cals < target * 0.8:
                recommendations.append(HealthRecommendation(
                    category="nutrition",
                    title="Increase Daily Calorie Intake",
                    description=f"You're averaging {avg_daily_cals:.0f} calories, below your target of {target}.",
                    reasoning="Consistently low calorie intake can lead to nutrient deficiencies and fatigue.",
                    priority=4,
                    actionable_steps=[
                        "Add a healthy mid-morning snack",
                        "Include more healthy fats (avocado, nuts, olive oil)",
                        "Don't skip meals",
                    ],
                    related_metrics=["calories_consumed"],
                ))
            elif avg_daily_cals > target * 1.2:
                recommendations.append(HealthRecommendation(
                    category="nutrition",
                    title="Review Calorie Intake",
                    description=f"You're averaging {avg_daily_cals:.0f} calories, above your target of {target}.",
                    reasoning="Excess calorie intake over time can lead to unwanted weight gain.",
                    priority=3,
                    actionable_steps=[
                        "Track portion sizes more carefully",
                        "Choose lower-calorie alternatives for snacks",
                        "Increase vegetable portions at meals",
                    ],
                    related_metrics=["calories_consumed"],
                ))

        # Protein recommendations
        if context.recent_meals:
            total_protein = sum(m.total_nutrients.protein_g for m in context.recent_meals[-7:])
            avg_daily_protein = total_protein / 7

            protein_target = context.user_profile.daily_protein_target or 50

            if avg_daily_protein < protein_target * 0.8:
                recommendations.append(HealthRecommendation(
                    category="nutrition",
                    title="Increase Protein Intake",
                    description=f"You're averaging {avg_daily_protein:.0f}g protein, below the recommended {protein_target}g.",
                    reasoning="Adequate protein is essential for muscle maintenance, satiety, and metabolic health.",
                    priority=4,
                    actionable_steps=[
                        "Include protein with every meal",
                        "Add Greek yogurt or eggs to breakfast",
                        "Consider a protein shake post-workout",
                    ],
                    related_metrics=["protein"],
                ))

        # Hydration reminder (generic, as we don't track water intake in detail)
        recommendations.append(HealthRecommendation(
            category="nutrition",
            title="Stay Hydrated",
            description="Ensure you're drinking adequate water throughout the day.",
            reasoning="Proper hydration supports energy levels, cognitive function, and physical performance.",
            priority=2,
            actionable_steps=[
                "Aim for 8 glasses (2L) of water daily",
                "Carry a water bottle",
                "Drink a glass of water with each meal",
            ],
            related_metrics=["water_intake"],
            confidence=0.7,
        ))

        return recommendations


class FitnessAdvisorAgent(HealthAdvisorAgent):
    """Specialist agent for fitness and exercise advice."""

    def __init__(self):
        super().__init__("FitnessAdvisor", "activity")

    @property
    def system_prompt(self) -> str:
        return """You are a fitness specialist AI advisor. Your role is to:

1. Analyze activity levels and exercise patterns
2. Identify opportunities for improvement
3. Suggest workout routines and activity modifications
4. Consider recovery and injury prevention
5. Provide progressive, achievable fitness goals

Focus on sustainable fitness practices. Consider the user's current fitness level.
Emphasize both cardiovascular and strength training benefits."""

    async def generate_recommendations(
        self,
        context: AdvisorContext,
    ) -> List[HealthRecommendation]:
        """Generate fitness recommendations."""
        recommendations = []

        # Analyze step count
        if context.recent_summaries:
            avg_steps = sum(s.total_steps for s in context.recent_summaries) / len(context.recent_summaries)
            target_steps = context.user_profile.daily_steps_target or 10000

            if avg_steps < target_steps * 0.6:
                recommendations.append(HealthRecommendation(
                    category="activity",
                    title="Increase Daily Movement",
                    description=f"You're averaging {avg_steps:.0f} steps, significantly below your goal of {target_steps}.",
                    reasoning="Regular walking improves cardiovascular health, mood, and metabolic function.",
                    priority=4,
                    actionable_steps=[
                        "Take a 15-minute walk after lunch",
                        "Use stairs instead of elevators",
                        "Park farther from destinations",
                        "Set hourly movement reminders",
                    ],
                    related_metrics=["steps", "active_minutes"],
                ))
            elif avg_steps < target_steps:
                recommendations.append(HealthRecommendation(
                    category="activity",
                    title="Close the Gap on Steps",
                    description=f"You're averaging {avg_steps:.0f} steps - just {target_steps - avg_steps:.0f} more to hit your goal!",
                    reasoning="You're close to your goal. Small changes can help you reach it consistently.",
                    priority=2,
                    actionable_steps=[
                        "Add a short evening walk",
                        "Take walking meetings when possible",
                    ],
                    related_metrics=["steps"],
                ))

            # Active minutes analysis
            avg_active = sum(s.active_minutes for s in context.recent_summaries) / len(context.recent_summaries)

            if avg_active < 30:
                recommendations.append(HealthRecommendation(
                    category="activity",
                    title="Increase Active Exercise Time",
                    description=f"You're averaging {avg_active:.0f} minutes of activity. WHO recommends 150 minutes/week.",
                    reasoning="Regular moderate-intensity activity significantly reduces disease risk and improves longevity.",
                    priority=4,
                    actionable_steps=[
                        "Schedule 3-4 workout sessions per week",
                        "Try a mix of cardio and strength training",
                        "Start with 20-minute sessions and gradually increase",
                    ],
                    related_metrics=["active_minutes"],
                ))

        # Recovery-based recommendations
        if context.recent_summaries:
            recovery_scores = [s.recovery_score for s in context.recent_summaries if s.recovery_score]
            if recovery_scores:
                avg_recovery = sum(recovery_scores) / len(recovery_scores)

                if avg_recovery < 50:
                    recommendations.append(HealthRecommendation(
                        category="activity",
                        title="Prioritize Recovery",
                        description=f"Your recovery score is averaging {avg_recovery:.0f}. Consider reducing training intensity.",
                        reasoning="Low recovery scores indicate your body needs more rest to adapt to training stress.",
                        priority=4,
                        actionable_steps=[
                            "Take 1-2 complete rest days",
                            "Focus on sleep quality",
                            "Include active recovery (light walking, stretching)",
                            "Consider reducing workout intensity this week",
                        ],
                        related_metrics=["recovery_score", "strain"],
                    ))

        return recommendations


class SleepAdvisorAgent(HealthAdvisorAgent):
    """Specialist agent for sleep optimization advice."""

    def __init__(self):
        super().__init__("SleepAdvisor", "sleep")

    @property
    def system_prompt(self) -> str:
        return """You are a sleep specialist AI advisor. Your role is to:

1. Analyze sleep patterns and quality
2. Identify factors affecting sleep
3. Suggest sleep hygiene improvements
4. Consider circadian rhythm optimization
5. Provide practical sleep enhancement strategies

Focus on evidence-based sleep science. Consider the user's lifestyle constraints.
Prioritize actionable changes that can be implemented immediately."""

    async def generate_recommendations(
        self,
        context: AdvisorContext,
    ) -> List[HealthRecommendation]:
        """Generate sleep recommendations."""
        recommendations = []

        if context.recent_summaries:
            # Sleep duration analysis
            sleep_data = [s for s in context.recent_summaries if s.sleep_duration_hours > 0]

            if sleep_data:
                avg_sleep = sum(s.sleep_duration_hours for s in sleep_data) / len(sleep_data)
                target_sleep = context.user_profile.sleep_target_hours or 8

                if avg_sleep < 6:
                    recommendations.append(HealthRecommendation(
                        category="sleep",
                        title="Critical: Increase Sleep Duration",
                        description=f"You're averaging only {avg_sleep:.1f} hours of sleep. Adults need 7-9 hours.",
                        reasoning="Chronic sleep deprivation impairs cognitive function, immune system, and increases disease risk.",
                        priority=5,
                        actionable_steps=[
                            "Set a consistent bedtime 8 hours before wake time",
                            "Create a wind-down routine 30 minutes before bed",
                            "Eliminate screens 1 hour before sleep",
                            "Keep bedroom cool (65-68°F / 18-20°C)",
                        ],
                        related_metrics=["sleep_duration", "sleep_score"],
                    ))
                elif avg_sleep < 7:
                    recommendations.append(HealthRecommendation(
                        category="sleep",
                        title="Improve Sleep Duration",
                        description=f"You're averaging {avg_sleep:.1f} hours of sleep, slightly below optimal.",
                        reasoning="Most adults benefit from 7-9 hours. Small improvements can significantly impact energy and health.",
                        priority=3,
                        actionable_steps=[
                            "Try going to bed 30 minutes earlier",
                            "Limit caffeine after 2pm",
                            "Avoid alcohol close to bedtime",
                        ],
                        related_metrics=["sleep_duration"],
                    ))

                # Sleep quality analysis
                sleep_scores = [s.sleep_score for s in sleep_data if s.sleep_score]
                if sleep_scores:
                    avg_score = sum(sleep_scores) / len(sleep_scores)

                    if avg_score < 70:
                        recommendations.append(HealthRecommendation(
                            category="sleep",
                            title="Improve Sleep Quality",
                            description=f"Your sleep quality score is averaging {avg_score:.0f}. Let's improve it.",
                            reasoning="Sleep quality matters as much as duration for physical and mental recovery.",
                            priority=4,
                            actionable_steps=[
                                "Maintain consistent wake and sleep times",
                                "Create a dark, quiet sleep environment",
                                "Avoid large meals close to bedtime",
                                "Get morning sunlight exposure",
                            ],
                            related_metrics=["sleep_score", "sleep_deep", "sleep_rem"],
                        ))

        return recommendations


class StressAdvisorAgent(HealthAdvisorAgent):
    """Specialist agent for stress management advice."""

    def __init__(self):
        super().__init__("StressAdvisor", "stress")

    @property
    def system_prompt(self) -> str:
        return """You are a stress management specialist AI advisor. Your role is to:

1. Identify stress indicators from health data
2. Suggest stress reduction techniques
3. Promote mental wellness practices
4. Consider work-life balance
5. Provide practical coping strategies

Focus on sustainable stress management. Consider both acute and chronic stress.
Emphasize mind-body connection and holistic wellbeing."""

    async def generate_recommendations(
        self,
        context: AdvisorContext,
    ) -> List[HealthRecommendation]:
        """Generate stress management recommendations."""
        recommendations = []

        # Analyze HRV as stress indicator
        if context.recent_summaries:
            hrv_data = [s.hrv_avg for s in context.recent_summaries if s.hrv_avg]

            if hrv_data:
                avg_hrv = sum(hrv_data) / len(hrv_data)

                if avg_hrv < 40:
                    recommendations.append(HealthRecommendation(
                        category="stress",
                        title="Address Elevated Stress",
                        description=f"Your HRV is averaging {avg_hrv:.0f}ms, indicating elevated stress or fatigue.",
                        reasoning="Low HRV often correlates with stress, poor recovery, or overtraining.",
                        priority=4,
                        actionable_steps=[
                            "Practice 5-10 minutes of deep breathing daily",
                            "Try a guided meditation app",
                            "Take breaks during intense work",
                            "Spend time in nature if possible",
                        ],
                        related_metrics=["heart_rate_variability", "stress"],
                    ))

            # Resting heart rate as stress indicator
            rhr_data = [s.resting_heart_rate for s in context.recent_summaries if s.resting_heart_rate]

            if rhr_data:
                avg_rhr = sum(rhr_data) / len(rhr_data)

                if avg_rhr > 75:
                    recommendations.append(HealthRecommendation(
                        category="stress",
                        title="Monitor Elevated Resting Heart Rate",
                        description=f"Your resting heart rate is averaging {avg_rhr:.0f} bpm.",
                        reasoning="Elevated RHR can indicate stress, poor recovery, or dehydration.",
                        priority=3,
                        actionable_steps=[
                            "Ensure adequate hydration",
                            "Practice relaxation techniques",
                            "Limit stimulants (caffeine, energy drinks)",
                            "Consider if recent life changes are adding stress",
                        ],
                        related_metrics=["resting_heart_rate"],
                    ))

        # General stress management recommendation
        recommendations.append(HealthRecommendation(
            category="stress",
            title="Daily Stress Management Practice",
            description="Building a regular stress management routine improves overall wellbeing.",
            reasoning="Proactive stress management prevents accumulation and improves resilience.",
            priority=2,
            actionable_steps=[
                "Set aside 10 minutes for mindfulness or meditation",
                "Practice gratitude journaling",
                "Connect with friends or family",
                "Engage in a hobby you enjoy",
            ],
            related_metrics=["stress", "heart_rate_variability"],
            confidence=0.7,
        ))

        return recommendations


class SwarmHealthAdvisor:
    """
    Multi-agent health advisory system.

    Coordinates specialist agents to provide comprehensive,
    personalized health recommendations using swarm consensus.
    """

    def __init__(self):
        self.agents: Dict[str, HealthAdvisorAgent] = {
            "nutrition": NutritionAdvisorAgent(),
            "fitness": FitnessAdvisorAgent(),
            "sleep": SleepAdvisorAgent(),
            "stress": StressAdvisorAgent(),
        }

        self.analysis_engine = HealthAnalysisEngine()

    def set_llm_backends(self, backends: Dict[str, Any]):
        """Set LLM backends for agents."""
        for name, agent in self.agents.items():
            if name in backends:
                agent.set_llm_backend(backends[name])

    async def generate_recommendations(
        self,
        user_profile: UserHealthProfile,
        summaries: List[DailySummary],
        meals: Optional[List[MealEntry]] = None,
        goals: Optional[List[Dict[str, Any]]] = None,
        focus_area: Optional[str] = None,
    ) -> List[HealthRecommendation]:
        """
        Generate recommendations using multi-agent consensus.

        Args:
            user_profile: User's health profile
            summaries: Recent daily summaries
            meals: Recent meal entries
            goals: Active health goals
            focus_area: Specific area to focus on

        Returns:
            Prioritized list of health recommendations
        """
        # Generate insights for context
        if summaries:
            insights = self.analysis_engine.generate_insights(summaries)
        else:
            insights = []

        # Build context
        context = AdvisorContext(
            user_profile=user_profile,
            recent_summaries=summaries or [],
            recent_meals=meals or [],
            active_goals=goals or [],
            insights=insights,
            focus_area=focus_area,
        )

        # Determine which agents to consult
        if focus_area:
            target_agents = [self.agents.get(focus_area)] if focus_area in self.agents else []
        else:
            target_agents = list(self.agents.values())

        # Generate recommendations from each agent in parallel
        tasks = [
            agent.generate_recommendations(context)
            for agent in target_agents
            if agent
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all recommendations
        all_recommendations: List[HealthRecommendation] = []

        for result in results:
            if isinstance(result, list):
                all_recommendations.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Agent error: {result}")

        # Deduplicate and prioritize
        recommendations = self._deduplicate_recommendations(all_recommendations)
        recommendations = self._prioritize_recommendations(recommendations)

        return recommendations

    def _deduplicate_recommendations(
        self,
        recommendations: List[HealthRecommendation],
    ) -> List[HealthRecommendation]:
        """Remove duplicate recommendations."""
        seen_titles = set()
        unique = []

        for rec in recommendations:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                unique.append(rec)

        return unique

    def _prioritize_recommendations(
        self,
        recommendations: List[HealthRecommendation],
    ) -> List[HealthRecommendation]:
        """Sort recommendations by priority and confidence."""
        return sorted(
            recommendations,
            key=lambda r: (r.priority, r.confidence),
            reverse=True,
        )

    async def get_recommendations_for_category(
        self,
        category: str,
        user_profile: UserHealthProfile,
        summaries: List[DailySummary],
    ) -> List[HealthRecommendation]:
        """Get recommendations for a specific category."""
        return await self.generate_recommendations(
            user_profile=user_profile,
            summaries=summaries,
            focus_area=category,
        )
