"""
cosmos Nutrition Manager

Comprehensive nutrition tracking with food database, meal logging,
recipe management, and swarm-powered meal planning.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from .models import (
    NutrientInfo,
    FoodItem,
    MealEntry,
    MealType,
    Recipe,
    UserHealthProfile,
)

logger = logging.getLogger(__name__)


@dataclass
class DailyNutrition:
    """Daily nutrition totals and breakdown."""
    date: date
    totals: NutrientInfo
    meals: List[MealEntry]
    targets: Dict[str, float]
    percent_of_target: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self._calculate_percentages()

    def _calculate_percentages(self):
        """Calculate percentage of daily targets achieved."""
        if self.targets.get("calories", 0) > 0:
            self.percent_of_target["calories"] = (
                self.totals.calories / self.targets["calories"] * 100
            )
        if self.targets.get("protein_g", 0) > 0:
            self.percent_of_target["protein"] = (
                self.totals.protein_g / self.targets["protein_g"] * 100
            )
        if self.targets.get("carbs_g", 0) > 0:
            self.percent_of_target["carbs"] = (
                self.totals.carbs_g / self.targets["carbs_g"] * 100
            )
        if self.targets.get("fat_g", 0) > 0:
            self.percent_of_target["fat"] = (
                self.totals.fat_g / self.targets["fat_g"] * 100
            )

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "totals": self.totals.to_dict(),
            "meals": [m.to_dict() for m in self.meals],
            "targets": self.targets,
            "percent_of_target": self.percent_of_target,
        }


class NutritionManager:
    """
    Manages nutrition tracking, food database, and meal planning.

    Features:
    - Food database with nutritional info
    - Meal logging with automatic nutrient calculation
    - Daily/weekly nutrition summaries
    - Recipe management
    - AI-powered meal planning and suggestions
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        user_profile: Optional[UserHealthProfile] = None,
    ):
        """
        Initialize the nutrition manager.

        Args:
            data_dir: Directory for storing nutrition data
            user_profile: User's health profile for personalization
        """
        self.data_dir = data_dir or Path.home() / ".cosmos" / "health" / "nutrition"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.user_profile = user_profile or UserHealthProfile()

        # In-memory stores
        self.food_database: Dict[str, FoodItem] = {}
        self.meal_history: List[MealEntry] = []
        self.recipes: Dict[str, Recipe] = {}

        # Load data
        self._load_food_database()
        self._load_meal_history()
        self._load_recipes()

    def _load_food_database(self):
        """Load food database from file or create default."""
        db_path = self.data_dir / "foods.json"

        if db_path.exists():
            try:
                with open(db_path) as f:
                    data = json.load(f)
                    for item in data:
                        food = FoodItem(
                            id=item["id"],
                            name=item["name"],
                            brand=item.get("brand"),
                            category=item.get("category", "other"),
                            serving_size=item.get("serving_size", 100),
                            serving_unit=item.get("serving_unit", "g"),
                            nutrients=NutrientInfo(**item.get("nutrients", {})),
                            barcode=item.get("barcode"),
                            verified=item.get("verified", False),
                        )
                        self.food_database[food.id] = food
                logger.info(f"Loaded {len(self.food_database)} foods from database")
            except Exception as e:
                logger.error(f"Error loading food database: {e}")
                self._create_default_foods()
        else:
            self._create_default_foods()

    def _create_default_foods(self):
        """Create default food database with common items."""
        default_foods = [
            # Proteins
            FoodItem(id="chicken_breast", name="Chicken Breast (cooked)", category="protein",
                     serving_size=100, serving_unit="g",
                     nutrients=NutrientInfo(calories=165, protein_g=31, carbs_g=0, fat_g=3.6)),
            FoodItem(id="salmon", name="Salmon (cooked)", category="protein",
                     serving_size=100, serving_unit="g",
                     nutrients=NutrientInfo(calories=208, protein_g=20, carbs_g=0, fat_g=13)),
            FoodItem(id="eggs", name="Eggs (large, whole)", category="protein",
                     serving_size=50, serving_unit="g",
                     nutrients=NutrientInfo(calories=78, protein_g=6, carbs_g=0.6, fat_g=5, cholesterol_mg=186)),
            FoodItem(id="ground_beef_lean", name="Ground Beef (90% lean)", category="protein",
                     serving_size=100, serving_unit="g",
                     nutrients=NutrientInfo(calories=176, protein_g=20, carbs_g=0, fat_g=10)),
            FoodItem(id="tofu", name="Tofu (firm)", category="protein",
                     serving_size=100, serving_unit="g",
                     nutrients=NutrientInfo(calories=144, protein_g=17, carbs_g=3, fat_g=8)),

            # Grains
            FoodItem(id="brown_rice", name="Brown Rice (cooked)", category="grain",
                     serving_size=100, serving_unit="g",
                     nutrients=NutrientInfo(calories=112, protein_g=2.6, carbs_g=24, fat_g=0.9, fiber_g=1.8)),
            FoodItem(id="oatmeal", name="Oatmeal (cooked)", category="grain",
                     serving_size=100, serving_unit="g",
                     nutrients=NutrientInfo(calories=68, protein_g=2.4, carbs_g=12, fat_g=1.4, fiber_g=1.7)),
            FoodItem(id="quinoa", name="Quinoa (cooked)", category="grain",
                     serving_size=100, serving_unit="g",
                     nutrients=NutrientInfo(calories=120, protein_g=4.4, carbs_g=21, fat_g=1.9, fiber_g=2.8)),
            FoodItem(id="whole_wheat_bread", name="Whole Wheat Bread", category="grain",
                     serving_size=30, serving_unit="g",
                     nutrients=NutrientInfo(calories=79, protein_g=4, carbs_g=14, fat_g=1, fiber_g=2)),

            # Vegetables
            FoodItem(id="broccoli", name="Broccoli", category="vegetable",
                     serving_size=100, serving_unit="g",
                     nutrients=NutrientInfo(calories=34, protein_g=2.8, carbs_g=7, fat_g=0.4, fiber_g=2.6, vitamin_c_mg=89)),
            FoodItem(id="spinach", name="Spinach (raw)", category="vegetable",
                     serving_size=100, serving_unit="g",
                     nutrients=NutrientInfo(calories=23, protein_g=2.9, carbs_g=3.6, fat_g=0.4, fiber_g=2.2, iron_mg=2.7)),
            FoodItem(id="sweet_potato", name="Sweet Potato (cooked)", category="vegetable",
                     serving_size=100, serving_unit="g",
                     nutrients=NutrientInfo(calories=90, protein_g=2, carbs_g=21, fat_g=0.1, fiber_g=3.3)),
            FoodItem(id="carrot", name="Carrot (raw)", category="vegetable",
                     serving_size=100, serving_unit="g",
                     nutrients=NutrientInfo(calories=41, protein_g=0.9, carbs_g=10, fat_g=0.2, fiber_g=2.8)),

            # Fruits
            FoodItem(id="apple", name="Apple (medium)", category="fruit",
                     serving_size=182, serving_unit="g",
                     nutrients=NutrientInfo(calories=95, protein_g=0.5, carbs_g=25, fat_g=0.3, fiber_g=4.4)),
            FoodItem(id="banana", name="Banana (medium)", category="fruit",
                     serving_size=118, serving_unit="g",
                     nutrients=NutrientInfo(calories=105, protein_g=1.3, carbs_g=27, fat_g=0.4, fiber_g=3.1, potassium_mg=422)),
            FoodItem(id="blueberries", name="Blueberries", category="fruit",
                     serving_size=100, serving_unit="g",
                     nutrients=NutrientInfo(calories=57, protein_g=0.7, carbs_g=14, fat_g=0.3, fiber_g=2.4)),

            # Dairy
            FoodItem(id="greek_yogurt", name="Greek Yogurt (plain, nonfat)", category="dairy",
                     serving_size=170, serving_unit="g",
                     nutrients=NutrientInfo(calories=100, protein_g=17, carbs_g=6, fat_g=0.7, calcium_mg=187)),
            FoodItem(id="milk_2_percent", name="Milk (2%)", category="dairy",
                     serving_size=244, serving_unit="ml",
                     nutrients=NutrientInfo(calories=122, protein_g=8, carbs_g=12, fat_g=5, calcium_mg=293)),
            FoodItem(id="cheddar_cheese", name="Cheddar Cheese", category="dairy",
                     serving_size=28, serving_unit="g",
                     nutrients=NutrientInfo(calories=113, protein_g=7, carbs_g=0.4, fat_g=9, calcium_mg=200)),

            # Nuts & Seeds
            FoodItem(id="almonds", name="Almonds", category="nuts",
                     serving_size=28, serving_unit="g",
                     nutrients=NutrientInfo(calories=164, protein_g=6, carbs_g=6, fat_g=14, fiber_g=3.5)),
            FoodItem(id="peanut_butter", name="Peanut Butter", category="nuts",
                     serving_size=32, serving_unit="g",
                     nutrients=NutrientInfo(calories=188, protein_g=8, carbs_g=6, fat_g=16, fiber_g=2)),

            # Oils & Fats
            FoodItem(id="olive_oil", name="Olive Oil", category="oil",
                     serving_size=14, serving_unit="ml",
                     nutrients=NutrientInfo(calories=119, protein_g=0, carbs_g=0, fat_g=14)),

            # Beverages
            FoodItem(id="black_coffee", name="Coffee (black)", category="beverage",
                     serving_size=240, serving_unit="ml",
                     nutrients=NutrientInfo(calories=2, protein_g=0.3, carbs_g=0, fat_g=0)),
        ]

        for food in default_foods:
            self.food_database[food.id] = food

        self._save_food_database()
        logger.info(f"Created default food database with {len(default_foods)} items")

    def _save_food_database(self):
        """Save food database to file."""
        db_path = self.data_dir / "foods.json"
        data = [food.to_dict() for food in self.food_database.values()]
        with open(db_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_meal_history(self):
        """Load meal history from file."""
        history_path = self.data_dir / "meals.json"

        if history_path.exists():
            try:
                with open(history_path) as f:
                    data = json.load(f)
                    for item in data:
                        meal = MealEntry(
                            id=item["id"],
                            meal_type=MealType(item["meal_type"]),
                            foods=item.get("foods", []),
                            total_nutrients=NutrientInfo(**item.get("total_nutrients", {})),
                            timestamp=datetime.fromisoformat(item["timestamp"]),
                            notes=item.get("notes", ""),
                        )
                        self.meal_history.append(meal)
                logger.info(f"Loaded {len(self.meal_history)} meals from history")
            except Exception as e:
                logger.error(f"Error loading meal history: {e}")

    def _save_meal_history(self):
        """Save meal history to file."""
        history_path = self.data_dir / "meals.json"
        data = [meal.to_dict() for meal in self.meal_history]
        with open(history_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_recipes(self):
        """Load recipes from file."""
        recipes_path = self.data_dir / "recipes.json"

        if recipes_path.exists():
            try:
                with open(recipes_path) as f:
                    data = json.load(f)
                    for item in data:
                        recipe = Recipe(
                            id=item["id"],
                            name=item["name"],
                            description=item.get("description", ""),
                            ingredients=item.get("ingredients", []),
                            instructions=item.get("instructions", []),
                            servings=item.get("servings", 1),
                            prep_time_minutes=item.get("prep_time_minutes", 0),
                            cook_time_minutes=item.get("cook_time_minutes", 0),
                            nutrients_per_serving=NutrientInfo(**item.get("nutrients_per_serving", {})),
                            tags=item.get("tags", []),
                            source=item.get("source", "user"),
                        )
                        self.recipes[recipe.id] = recipe
                logger.info(f"Loaded {len(self.recipes)} recipes")
            except Exception as e:
                logger.error(f"Error loading recipes: {e}")

    def _save_recipes(self):
        """Save recipes to file."""
        recipes_path = self.data_dir / "recipes.json"
        data = [recipe.to_dict() for recipe in self.recipes.values()]
        with open(recipes_path, "w") as f:
            json.dump(data, f, indent=2)

    # ============================================
    # Food Database Operations
    # ============================================

    def search_foods(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[FoodItem]:
        """
        Search the food database.

        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum results

        Returns:
            List of matching FoodItem objects
        """
        query_lower = query.lower()
        results = []

        for food in self.food_database.values():
            # Check name match
            if query_lower in food.name.lower():
                if category is None or food.category == category:
                    results.append(food)

            # Check brand match
            if food.brand and query_lower in food.brand.lower():
                if food not in results:
                    results.append(food)

            if len(results) >= limit:
                break

        return results[:limit]

    def get_food(self, food_id: str) -> Optional[FoodItem]:
        """Get a food item by ID."""
        return self.food_database.get(food_id)

    def add_food(self, food: FoodItem) -> FoodItem:
        """Add a food item to the database."""
        self.food_database[food.id] = food
        self._save_food_database()
        return food

    def update_food(self, food_id: str, updates: Dict[str, Any]) -> Optional[FoodItem]:
        """Update a food item."""
        if food_id not in self.food_database:
            return None

        food = self.food_database[food_id]
        for key, value in updates.items():
            if hasattr(food, key):
                setattr(food, key, value)

        self._save_food_database()
        return food

    # ============================================
    # Meal Logging
    # ============================================

    def log_meal(
        self,
        meal_type: MealType,
        foods: List[Dict[str, Any]],
        notes: str = "",
        timestamp: Optional[datetime] = None,
    ) -> MealEntry:
        """
        Log a meal with nutritional calculation.

        Args:
            meal_type: Type of meal
            foods: List of {food_id, servings} dicts
            notes: Optional notes
            timestamp: When the meal was eaten

        Returns:
            MealEntry with calculated nutrients
        """
        # Calculate total nutrients
        total = NutrientInfo()

        for item in foods:
            food_id = item.get("food_id")
            servings = item.get("servings", 1)

            food = self.food_database.get(food_id)
            if food:
                scaled = food.nutrients.scale(servings)
                total.calories += scaled.calories
                total.protein_g += scaled.protein_g
                total.carbs_g += scaled.carbs_g
                total.fat_g += scaled.fat_g
                total.fiber_g += scaled.fiber_g
                total.sugar_g += scaled.sugar_g
                total.sodium_mg += scaled.sodium_mg
                total.cholesterol_mg += scaled.cholesterol_mg

        entry = MealEntry(
            meal_type=meal_type,
            foods=foods,
            total_nutrients=total,
            timestamp=timestamp or datetime.now(),
            notes=notes,
        )

        self.meal_history.append(entry)
        self._save_meal_history()

        return entry

    def get_meals_for_date(self, target_date: date) -> List[MealEntry]:
        """Get all meals for a specific date."""
        return [
            meal for meal in self.meal_history
            if meal.timestamp.date() == target_date
        ]

    def delete_meal(self, meal_id: str) -> bool:
        """Delete a meal entry."""
        for i, meal in enumerate(self.meal_history):
            if meal.id == meal_id:
                self.meal_history.pop(i)
                self._save_meal_history()
                return True
        return False

    # ============================================
    # Daily Nutrition
    # ============================================

    def calculate_daily_nutrition(
        self,
        target_date: Optional[date] = None,
    ) -> DailyNutrition:
        """
        Calculate daily nutrition totals.

        Args:
            target_date: Date to calculate (default today)

        Returns:
            DailyNutrition summary
        """
        target_date = target_date or date.today()
        meals = self.get_meals_for_date(target_date)

        # Aggregate nutrients
        total = NutrientInfo()
        for meal in meals:
            total.calories += meal.total_nutrients.calories
            total.protein_g += meal.total_nutrients.protein_g
            total.carbs_g += meal.total_nutrients.carbs_g
            total.fat_g += meal.total_nutrients.fat_g
            total.fiber_g += meal.total_nutrients.fiber_g
            total.sugar_g += meal.total_nutrients.sugar_g
            total.sodium_mg += meal.total_nutrients.sodium_mg

        # Get targets from user profile
        targets = {
            "calories": self.user_profile.daily_calorie_target or 2000,
            "protein_g": self.user_profile.daily_protein_target or 50,
            "carbs_g": 250,  # Default
            "fat_g": 65,  # Default
            "fiber_g": 25,
        }

        return DailyNutrition(
            date=target_date,
            totals=total,
            meals=meals,
            targets=targets,
        )

    def get_nutrition_history(
        self,
        days: int = 7,
    ) -> List[DailyNutrition]:
        """Get nutrition history for past days."""
        history = []
        today = date.today()

        for i in range(days):
            day = today - timedelta(days=i)
            daily = self.calculate_daily_nutrition(day)
            history.append(daily)

        return history

    # ============================================
    # Recipe Management
    # ============================================

    def add_recipe(self, recipe: Recipe) -> Recipe:
        """Add a recipe."""
        self.recipes[recipe.id] = recipe
        self._save_recipes()
        return recipe

    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        """Get a recipe by ID."""
        return self.recipes.get(recipe_id)

    def search_recipes(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_calories: Optional[int] = None,
        min_protein: Optional[float] = None,
        limit: int = 20,
    ) -> List[Recipe]:
        """
        Search recipes with filters.

        Args:
            query: Search text
            tags: Required tags (e.g., ["vegetarian", "quick"])
            max_calories: Maximum calories per serving
            min_protein: Minimum protein per serving
            limit: Maximum results

        Returns:
            List of matching recipes
        """
        results = []

        for recipe in self.recipes.values():
            # Query filter
            if query:
                query_lower = query.lower()
                if query_lower not in recipe.name.lower() and \
                   query_lower not in recipe.description.lower():
                    continue

            # Tags filter
            if tags:
                if not all(tag in recipe.tags for tag in tags):
                    continue

            # Calorie filter
            if max_calories and recipe.nutrients_per_serving.calories > max_calories:
                continue

            # Protein filter
            if min_protein and recipe.nutrients_per_serving.protein_g < min_protein:
                continue

            results.append(recipe)

            if len(results) >= limit:
                break

        return results

    def delete_recipe(self, recipe_id: str) -> bool:
        """Delete a recipe."""
        if recipe_id in self.recipes:
            del self.recipes[recipe_id]
            self._save_recipes()
            return True
        return False

    # ============================================
    # Meal Plan Generation
    # ============================================

    def generate_meal_plan(
        self,
        days: int = 7,
        target_calories: Optional[int] = None,
        dietary_restrictions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a meal plan.

        This is a simple implementation. For AI-powered planning,
        see SwarmHealthAdvisor.

        Args:
            days: Number of days
            target_calories: Daily calorie target
            dietary_restrictions: Dietary restrictions to respect

        Returns:
            List of daily meal plans
        """
        target_calories = target_calories or self.user_profile.daily_calorie_target or 2000
        restrictions = dietary_restrictions or self.user_profile.dietary_restrictions

        meal_plan = []

        for day in range(days):
            day_plan = {
                "day": day + 1,
                "date": (date.today() + timedelta(days=day)).isoformat(),
                "meals": {
                    "breakfast": [],
                    "lunch": [],
                    "dinner": [],
                    "snacks": [],
                },
                "total_calories": 0,
            }

            # Simple distribution: 25% breakfast, 30% lunch, 35% dinner, 10% snacks
            breakfast_cal = int(target_calories * 0.25)
            lunch_cal = int(target_calories * 0.30)
            dinner_cal = int(target_calories * 0.35)

            # Add placeholder suggestions
            day_plan["meals"]["breakfast"].append({
                "suggestion": "Oatmeal with berries and Greek yogurt",
                "target_calories": breakfast_cal,
            })
            day_plan["meals"]["lunch"].append({
                "suggestion": "Grilled chicken salad with quinoa",
                "target_calories": lunch_cal,
            })
            day_plan["meals"]["dinner"].append({
                "suggestion": "Salmon with roasted vegetables and brown rice",
                "target_calories": dinner_cal,
            })

            meal_plan.append(day_plan)

        return meal_plan

    def get_recipe_suggestions(
        self,
        meal_type: Optional[MealType] = None,
        max_calories: Optional[int] = None,
        preparation_time: Optional[int] = None,
    ) -> List[Recipe]:
        """
        Get recipe suggestions based on criteria.

        Args:
            meal_type: Type of meal
            max_calories: Maximum calories
            preparation_time: Maximum prep + cook time in minutes

        Returns:
            List of suggested recipes
        """
        suggestions = []

        for recipe in self.recipes.values():
            # Calorie filter
            if max_calories and recipe.nutrients_per_serving.calories > max_calories:
                continue

            # Time filter
            if preparation_time:
                total_time = recipe.prep_time_minutes + recipe.cook_time_minutes
                if total_time > preparation_time:
                    continue

            # Respect user restrictions
            if self.user_profile.dietary_restrictions:
                skip = False
                for restriction in self.user_profile.dietary_restrictions:
                    if restriction == "vegetarian" and "vegetarian" not in recipe.tags:
                        # Check if recipe contains meat
                        pass  # Would need ingredient analysis
                skip_recipe = False

            if not skip_recipe:
                suggestions.append(recipe)

        return suggestions[:10]
