"""
cosmos Health MCP Tools

MCP tool implementations for health tracking, nutrition, and recommendations.
Provides Claude with comprehensive health management capabilities.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_health_modules_loaded = False
_provider_manager = None
_analysis_engine = None
_nutrition_manager = None
_ocr_parser = None
_swarm_advisor = None


def _ensure_health_modules():
    """Lazy load health modules."""
    global _health_modules_loaded, _provider_manager, _analysis_engine
    global _nutrition_manager, _ocr_parser, _swarm_advisor

    if _health_modules_loaded:
        return

    try:
        from Cosmos.health.providers import HealthProviderManager, MockHealthProvider
        from Cosmos.health.analysis import HealthAnalysisEngine
        from Cosmos.health.nutrition import NutritionManager
        from Cosmos.health.models import UserHealthProfile

        # Initialize managers
        _provider_manager = HealthProviderManager()
        _provider_manager.register_provider(MockHealthProvider())

        _analysis_engine = HealthAnalysisEngine()
        _nutrition_manager = NutritionManager()

        _health_modules_loaded = True
        logger.info("Health modules loaded successfully")

    except ImportError as e:
        logger.warning(f"Health modules not available: {e}")


@dataclass
class HealthToolResult:
    """Result from a health tool operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> dict:
        result = {"success": self.success}
        if self.data is not None:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        if self.message:
            result["message"] = self.message
        return result


class HealthTools:
    """
    Health tool implementations for MCP server.

    Provides tools for:
    - Health summaries and metrics
    - Trend analysis
    - Nutrition tracking
    - Recipe suggestions
    - Goal management
    - Document parsing
    - AI recommendations
    """

    def __init__(self):
        _ensure_health_modules()

    # ============================================
    # Health Summary and Metrics
    # ============================================

    async def cosmos_health_summary(
        self,
        date_str: Optional[str] = None,
    ) -> HealthToolResult:
        """
        Get daily health summary.

        Args:
            date_str: Date in YYYY-MM-DD format (default: today)

        Returns:
            Daily health summary with wellness score
        """
        _ensure_health_modules()

        if not _provider_manager:
            return HealthToolResult(
                success=False,
                error="Health system not initialized",
            )

        try:
            target_date = date.fromisoformat(date_str) if date_str else date.today()

            # Connect providers if needed
            await _provider_manager.connect_all()

            # Get summary
            summary = await _provider_manager.get_daily_summary(target_date)

            # Calculate wellness score
            wellness = _analysis_engine.calculate_wellness_score(summary)

            return HealthToolResult(
                success=True,
                data={
                    "date": target_date.isoformat(),
                    "summary": summary.to_dict(),
                    "wellness_score": wellness.to_dict(),
                },
                message=f"Wellness score: {wellness.overall}/100",
            )

        except Exception as e:
            logger.error(f"Health summary error: {e}")
            return HealthToolResult(success=False, error=str(e))

    async def cosmos_health_trend(
        self,
        metric: str,
        days: int = 7,
    ) -> HealthToolResult:
        """
        Analyze health metric trends.

        Args:
            metric: Metric type (heart_rate, steps, sleep_duration, etc.)
            days: Number of days to analyze (7, 14, or 30)

        Returns:
            Trend analysis with direction and insights
        """
        _ensure_health_modules()

        if not _provider_manager:
            return HealthToolResult(
                success=False,
                error="Health system not initialized",
            )

        try:
            from Cosmos.health.models import MetricType

            # Parse metric type
            try:
                metric_type = MetricType(metric)
            except ValueError:
                return HealthToolResult(
                    success=False,
                    error=f"Unknown metric type: {metric}. Valid types: {[m.value for m in MetricType]}",
                )

            # Fetch data
            end = date.today()
            start = end - timedelta(days=days)

            await _provider_manager.connect_all()
            readings = await _provider_manager.get_metrics([metric_type], start, end)

            # Get summaries for analysis
            summaries = []
            for d in range(days):
                day = end - timedelta(days=d)
                summary = await _provider_manager.get_daily_summary(day)
                summaries.append(summary)

            # Analyze trends
            _analysis_engine.load_data(readings, summaries)
            trends = _analysis_engine.analyze_trends([metric_type], days=days)

            if not trends:
                return HealthToolResult(
                    success=True,
                    data={"trends": []},
                    message="Not enough data for trend analysis",
                )

            trend = trends[0]
            return HealthToolResult(
                success=True,
                data={
                    "metric": metric,
                    "period_days": days,
                    "trend": trend.to_dict(),
                },
                message=trend.description,
            )

        except Exception as e:
            logger.error(f"Health trend error: {e}")
            return HealthToolResult(success=False, error=str(e))

    # ============================================
    # Nutrition Tracking
    # ============================================

    async def cosmos_nutrition_log(
        self,
        meal_type: str,
        foods: List[dict],
        notes: str = "",
    ) -> HealthToolResult:
        """
        Log a meal.

        Args:
            meal_type: Type of meal (breakfast, lunch, dinner, snack)
            foods: List of {food_id, servings} or {name, calories, protein_g, carbs_g, fat_g}
            notes: Optional notes

        Returns:
            Logged meal with nutritional totals
        """
        _ensure_health_modules()

        if not _nutrition_manager:
            return HealthToolResult(
                success=False,
                error="Nutrition system not initialized",
            )

        try:
            from Cosmos.health.models import MealType

            # Parse meal type
            try:
                mt = MealType(meal_type)
            except ValueError:
                return HealthToolResult(
                    success=False,
                    error=f"Invalid meal type. Use: breakfast, lunch, dinner, or snack",
                )

            # Handle custom foods (not in database)
            processed_foods = []
            for food in foods:
                if "food_id" in food:
                    processed_foods.append(food)
                elif "name" in food:
                    # Create a custom food entry
                    from Cosmos.health.models import FoodItem, NutrientInfo
                    custom = FoodItem(
                        name=food["name"],
                        nutrients=NutrientInfo(
                            calories=food.get("calories", 0),
                            protein_g=food.get("protein_g", 0),
                            carbs_g=food.get("carbs_g", 0),
                            fat_g=food.get("fat_g", 0),
                        ),
                    )
                    _nutrition_manager.add_food(custom)
                    processed_foods.append({
                        "food_id": custom.id,
                        "servings": food.get("servings", 1),
                    })

            # Log the meal
            meal = _nutrition_manager.log_meal(mt, processed_foods, notes)

            return HealthToolResult(
                success=True,
                data=meal.to_dict(),
                message=f"Logged {meal_type}: {int(meal.total_nutrients.calories)} calories",
            )

        except Exception as e:
            logger.error(f"Nutrition log error: {e}")
            return HealthToolResult(success=False, error=str(e))

    async def cosmos_nutrition_search(
        self,
        query: str,
        limit: int = 10,
    ) -> HealthToolResult:
        """
        Search the food database.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching foods with nutritional info
        """
        _ensure_health_modules()

        if not _nutrition_manager:
            return HealthToolResult(
                success=False,
                error="Nutrition system not initialized",
            )

        try:
            foods = _nutrition_manager.search_foods(query, limit=limit)

            return HealthToolResult(
                success=True,
                data={
                    "query": query,
                    "count": len(foods),
                    "foods": [f.to_dict() for f in foods],
                },
            )

        except Exception as e:
            logger.error(f"Food search error: {e}")
            return HealthToolResult(success=False, error=str(e))

    async def cosmos_nutrition_daily(
        self,
        date_str: Optional[str] = None,
    ) -> HealthToolResult:
        """
        Get daily nutrition summary.

        Args:
            date_str: Date in YYYY-MM-DD format (default: today)

        Returns:
            Daily nutrition totals and meal breakdown
        """
        _ensure_health_modules()

        if not _nutrition_manager:
            return HealthToolResult(
                success=False,
                error="Nutrition system not initialized",
            )

        try:
            target_date = date.fromisoformat(date_str) if date_str else date.today()
            daily = _nutrition_manager.calculate_daily_nutrition(target_date)

            return HealthToolResult(
                success=True,
                data=daily.to_dict(),
                message=f"Daily: {int(daily.totals.calories)} / {int(daily.targets.get('calories', 2000))} calories",
            )

        except Exception as e:
            logger.error(f"Daily nutrition error: {e}")
            return HealthToolResult(success=False, error=str(e))

    # ============================================
    # Recipe Suggestions
    # ============================================

    async def cosmos_recipe_suggest(
        self,
        meal_type: Optional[str] = None,
        max_calories: Optional[int] = None,
        restrictions: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> HealthToolResult:
        """
        Get recipe suggestions.

        Args:
            meal_type: Type of meal (optional)
            max_calories: Maximum calories per serving
            restrictions: Dietary restrictions (vegetarian, gluten-free, etc.)
            query: Search query

        Returns:
            List of matching recipes
        """
        _ensure_health_modules()

        if not _nutrition_manager:
            return HealthToolResult(
                success=False,
                error="Nutrition system not initialized",
            )

        try:
            recipes = _nutrition_manager.search_recipes(
                query=query,
                tags=restrictions,
                max_calories=max_calories,
            )

            return HealthToolResult(
                success=True,
                data={
                    "count": len(recipes),
                    "recipes": [r.to_dict() for r in recipes],
                },
            )

        except Exception as e:
            logger.error(f"Recipe suggest error: {e}")
            return HealthToolResult(success=False, error=str(e))

    # ============================================
    # Goal Management
    # ============================================

    async def cosmos_goal_create(
        self,
        name: str,
        metric_type: str,
        target_value: float,
        unit: str,
        deadline: Optional[str] = None,
        description: str = "",
    ) -> HealthToolResult:
        """
        Create a health goal.

        Args:
            name: Goal name
            metric_type: Type of metric (steps, weight, sleep_duration, etc.)
            target_value: Target value to achieve
            unit: Unit of measurement
            deadline: Target date (YYYY-MM-DD)
            description: Goal description

        Returns:
            Created goal
        """
        _ensure_health_modules()

        try:
            from Cosmos.health.models import HealthGoal, MetricType

            # Parse metric type
            try:
                mt = MetricType(metric_type)
            except ValueError:
                return HealthToolResult(
                    success=False,
                    error=f"Unknown metric type: {metric_type}",
                )

            goal = HealthGoal(
                name=name,
                description=description,
                metric_type=mt,
                target_value=target_value,
                unit=unit,
                target_date=date.fromisoformat(deadline) if deadline else None,
            )

            # Would store this in a persistent store
            return HealthToolResult(
                success=True,
                data=goal.to_dict(),
                message=f"Created goal: {name} - Target {target_value} {unit}",
            )

        except Exception as e:
            logger.error(f"Goal create error: {e}")
            return HealthToolResult(success=False, error=str(e))

    async def cosmos_goal_update(
        self,
        goal_id: str,
        current_value: Optional[float] = None,
        status: Optional[str] = None,
    ) -> HealthToolResult:
        """
        Update a health goal.

        Args:
            goal_id: Goal ID
            current_value: Current progress value
            status: New status (active, completed, paused, failed)

        Returns:
            Updated goal
        """
        _ensure_health_modules()

        try:
            # Would update in persistent store
            return HealthToolResult(
                success=True,
                data={
                    "goal_id": goal_id,
                    "current_value": current_value,
                    "status": status,
                },
                message=f"Goal {goal_id} updated",
            )

        except Exception as e:
            logger.error(f"Goal update error: {e}")
            return HealthToolResult(success=False, error=str(e))

    # ============================================
    # Document Parsing
    # ============================================

    async def cosmos_document_parse(
        self,
        image_path: str,
        doc_type: str = "lab_result",
    ) -> HealthToolResult:
        """
        Parse a health document using DeepSeek OCR.

        Args:
            image_path: Path to the document image
            doc_type: Type of document (lab_result, prescription, nutrition_label)

        Returns:
            Parsed document data
        """
        _ensure_health_modules()

        try:
            from Cosmos.health.ocr_parser import DeepSeekOCRParser
            from Cosmos.health.models import DocumentType

            # Parse document type
            try:
                dt = DocumentType(doc_type)
            except ValueError:
                return HealthToolResult(
                    success=False,
                    error=f"Unknown document type: {doc_type}. Use: lab_result, prescription, nutrition_label, medical_report",
                )

            # Check if file exists
            path = Path(image_path)
            if not path.exists():
                return HealthToolResult(
                    success=False,
                    error=f"File not found: {image_path}",
                )

            # Parse the document
            async with DeepSeekOCRParser() as parser:
                result = await parser.parse_document(path, dt)

            return HealthToolResult(
                success=result.success,
                data=result.to_dict(),
                error=result.error,
                message=f"Parsed {doc_type} with {result.confidence*100:.0f}% confidence" if result.success else None,
            )

        except Exception as e:
            logger.error(f"Document parse error: {e}")
            return HealthToolResult(success=False, error=str(e))

    # ============================================
    # AI Recommendations
    # ============================================

    async def cosmos_health_recommend(
        self,
        focus_area: Optional[str] = None,
    ) -> HealthToolResult:
        """
        Get AI health recommendations.

        Args:
            focus_area: Area to focus on (activity, sleep, nutrition, recovery, stress)

        Returns:
            Personalized health recommendations
        """
        _ensure_health_modules()

        if not _provider_manager or not _analysis_engine:
            return HealthToolResult(
                success=False,
                error="Health system not initialized",
            )

        try:
            # Get recent summaries
            end = date.today()
            summaries = []

            await _provider_manager.connect_all()

            for d in range(7):
                day = end - timedelta(days=d)
                summary = await _provider_manager.get_daily_summary(day)
                summaries.append(summary)

            # Generate insights
            insights = _analysis_engine.generate_insights(summaries)

            # Filter by focus area if specified
            if focus_area:
                insights = [i for i in insights if i.category == focus_area]

            # Convert to recommendations
            recommendations = [
                {
                    "category": i.category,
                    "title": i.title,
                    "description": i.message,
                    "priority": i.priority,
                    "actionable": i.actionable,
                    "related_metrics": i.related_metrics,
                }
                for i in insights
            ]

            return HealthToolResult(
                success=True,
                data={
                    "focus_area": focus_area,
                    "count": len(recommendations),
                    "recommendations": recommendations,
                },
                message=f"Generated {len(recommendations)} recommendations",
            )

        except Exception as e:
            logger.error(f"Health recommend error: {e}")
            return HealthToolResult(success=False, error=str(e))


# Tool definitions for MCP registration
HEALTH_TOOL_DEFINITIONS = [
    {
        "name": "cosmos_health_summary",
        "description": "Get daily health summary with wellness score. Includes heart rate, steps, sleep, recovery, and more.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (default: today)",
                },
            },
        },
    },
    {
        "name": "cosmos_health_trend",
        "description": "Analyze health metric trends over time. Identifies patterns in heart rate, sleep, activity, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "Metric type: heart_rate, steps, sleep_duration, heart_rate_variability, recovery_score, etc.",
                },
                "days": {
                    "type": "integer",
                    "description": "Analysis period (7, 14, or 30 days)",
                    "default": 7,
                },
            },
            "required": ["metric"],
        },
    },
    {
        "name": "cosmos_nutrition_log",
        "description": "Log a meal with nutritional tracking. Automatically calculates calories, protein, carbs, and fat.",
        "input_schema": {
            "type": "object",
            "properties": {
                "meal_type": {
                    "type": "string",
                    "enum": ["breakfast", "lunch", "dinner", "snack"],
                    "description": "Type of meal",
                },
                "foods": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "food_id": {"type": "string"},
                            "servings": {"type": "number"},
                            "name": {"type": "string"},
                            "calories": {"type": "number"},
                            "protein_g": {"type": "number"},
                            "carbs_g": {"type": "number"},
                            "fat_g": {"type": "number"},
                        },
                    },
                    "description": "Foods in the meal (use food_id for database items, or name+nutrients for custom)",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes",
                },
            },
            "required": ["meal_type", "foods"],
        },
    },
    {
        "name": "cosmos_nutrition_search",
        "description": "Search the food database for nutritional information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (food name)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "cosmos_recipe_suggest",
        "description": "Get recipe suggestions based on dietary needs and preferences.",
        "input_schema": {
            "type": "object",
            "properties": {
                "meal_type": {
                    "type": "string",
                    "description": "Type of meal",
                },
                "max_calories": {
                    "type": "integer",
                    "description": "Maximum calories per serving",
                },
                "restrictions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Dietary restrictions (vegetarian, gluten-free, etc.)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
        },
    },
    {
        "name": "cosmos_goal_create",
        "description": "Create a new health goal to track progress.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Goal name",
                },
                "metric_type": {
                    "type": "string",
                    "description": "Metric type (steps, weight, sleep_duration, etc.)",
                },
                "target_value": {
                    "type": "number",
                    "description": "Target value",
                },
                "unit": {
                    "type": "string",
                    "description": "Unit of measurement",
                },
                "deadline": {
                    "type": "string",
                    "description": "Target date (YYYY-MM-DD)",
                },
                "description": {
                    "type": "string",
                    "description": "Goal description",
                },
            },
            "required": ["name", "metric_type", "target_value", "unit"],
        },
    },
    {
        "name": "cosmos_document_parse",
        "description": "Parse health documents (lab results, prescriptions, nutrition labels) using AI OCR.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the document image",
                },
                "doc_type": {
                    "type": "string",
                    "enum": ["lab_result", "prescription", "nutrition_label", "medical_report"],
                    "description": "Type of document",
                    "default": "lab_result",
                },
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "cosmos_health_recommend",
        "description": "Get personalized AI health recommendations based on your data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "focus_area": {
                    "type": "string",
                    "enum": ["activity", "sleep", "nutrition", "recovery", "stress", "heart"],
                    "description": "Area to focus recommendations on",
                },
            },
        },
    },
]
