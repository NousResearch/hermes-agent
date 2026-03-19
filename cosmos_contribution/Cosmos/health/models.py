"""
cosmos Health Data Models

Comprehensive data models for health tracking including metrics, nutrition,
documents, and AI recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional, List, Dict, Any
import uuid


class MetricType(Enum):
    """Types of health metrics that can be tracked."""
    # Cardiovascular
    HR = "heart_rate"
    HRV = "heart_rate_variability"
    BP_SYSTOLIC = "blood_pressure_systolic"
    BP_DIASTOLIC = "blood_pressure_diastolic"
    SPO2 = "blood_oxygen"

    # Brain/EEG
    EEG_ALPHA = "eeg_alpha"
    EEG_BETA = "eeg_beta"
    EEG_THETA = "eeg_theta"
    EEG_DELTA = "eeg_delta"
    EEG_GAMMA = "eeg_gamma"

    # Stress/Emotional
    GSR = "galvanic_skin_response"
    STRESS = "stress_level"

    # Activity
    STEPS = "steps"
    CALORIES_BURNED = "calories_burned"
    ACTIVE_MINUTES = "active_minutes"
    DISTANCE = "distance"
    FLOORS = "floors_climbed"

    # Sleep
    SLEEP_DURATION = "sleep_duration"
    SLEEP_DEEP = "sleep_deep"
    SLEEP_REM = "sleep_rem"
    SLEEP_LIGHT = "sleep_light"
    SLEEP_AWAKE = "sleep_awake"
    SLEEP_SCORE = "sleep_score"

    # Body
    WEIGHT = "weight"
    BODY_FAT = "body_fat"
    BMI = "bmi"
    MUSCLE_MASS = "muscle_mass"
    HYDRATION = "hydration"

    # Recovery
    RECOVERY_SCORE = "recovery_score"
    READINESS_SCORE = "readiness_score"
    STRAIN = "strain"

    # Nutrition
    CALORIES_CONSUMED = "calories_consumed"
    PROTEIN = "protein"
    CARBS = "carbohydrates"
    FAT = "fat"
    FIBER = "fiber"
    WATER = "water_intake"


class AlertSeverity(Enum):
    """Severity levels for health alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MealType(Enum):
    """Types of meals."""
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class GoalStatus(Enum):
    """Status of a health goal."""
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"


class DocumentType(Enum):
    """Types of health documents that can be parsed."""
    LAB_RESULT = "lab_result"
    PRESCRIPTION = "prescription"
    NUTRITION_LABEL = "nutrition_label"
    MEDICAL_REPORT = "medical_report"


@dataclass
class HealthMetricReading:
    """A single health metric reading with timestamp and metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_type: MetricType = MetricType.HR
    value: float = 0.0
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "manual"  # Provider name or 'manual'
    quality: float = 1.0  # 0-1 confidence in reading
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "quality": self.quality,
            "metadata": self.metadata,
        }


@dataclass
class DailySummary:
    """Aggregated daily health statistics."""
    date: date = field(default_factory=date.today)

    # Cardiovascular
    avg_heart_rate: Optional[float] = None
    resting_heart_rate: Optional[float] = None
    max_heart_rate: Optional[float] = None
    hrv_avg: Optional[float] = None

    # Activity
    total_steps: int = 0
    total_calories_burned: int = 0
    active_minutes: int = 0
    distance_km: float = 0.0
    floors_climbed: int = 0

    # Sleep
    sleep_duration_hours: float = 0.0
    sleep_score: Optional[int] = None
    deep_sleep_hours: float = 0.0
    rem_sleep_hours: float = 0.0

    # Scores
    recovery_score: Optional[int] = None
    readiness_score: Optional[int] = None
    wellness_score: Optional[int] = None

    # Nutrition
    calories_consumed: int = 0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    water_ml: int = 0

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "avg_heart_rate": self.avg_heart_rate,
            "resting_heart_rate": self.resting_heart_rate,
            "max_heart_rate": self.max_heart_rate,
            "hrv_avg": self.hrv_avg,
            "total_steps": self.total_steps,
            "total_calories_burned": self.total_calories_burned,
            "active_minutes": self.active_minutes,
            "distance_km": self.distance_km,
            "floors_climbed": self.floors_climbed,
            "sleep_duration_hours": self.sleep_duration_hours,
            "sleep_score": self.sleep_score,
            "deep_sleep_hours": self.deep_sleep_hours,
            "rem_sleep_hours": self.rem_sleep_hours,
            "recovery_score": self.recovery_score,
            "readiness_score": self.readiness_score,
            "wellness_score": self.wellness_score,
            "calories_consumed": self.calories_consumed,
            "protein_g": self.protein_g,
            "carbs_g": self.carbs_g,
            "fat_g": self.fat_g,
            "water_ml": self.water_ml,
        }


@dataclass
class HealthGoal:
    """A health goal with progress tracking."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    metric_type: MetricType = MetricType.STEPS
    target_value: float = 0.0
    current_value: float = 0.0
    unit: str = ""
    start_date: date = field(default_factory=date.today)
    target_date: Optional[date] = None
    status: GoalStatus = GoalStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def progress_percent(self) -> float:
        if self.target_value == 0:
            return 0.0
        return min(100.0, (self.current_value / self.target_value) * 100)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metric_type": self.metric_type.value,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "unit": self.unit,
            "progress_percent": self.progress_percent,
            "start_date": self.start_date.isoformat(),
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "status": self.status.value,
        }


@dataclass
class HealthAlert:
    """A health alert for anomalies or important notifications."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    message: str = ""
    severity: AlertSeverity = AlertSeverity.INFO
    metric_type: Optional[MetricType] = None
    triggered_value: Optional[float] = None
    threshold_value: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "metric_type": self.metric_type.value if self.metric_type else None,
            "triggered_value": self.triggered_value,
            "threshold_value": self.threshold_value,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class NutrientInfo:
    """Nutritional information for a food item."""
    calories: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    sugar_g: float = 0.0
    sodium_mg: float = 0.0
    cholesterol_mg: float = 0.0
    saturated_fat_g: float = 0.0
    trans_fat_g: float = 0.0
    potassium_mg: float = 0.0
    vitamin_a_iu: float = 0.0
    vitamin_c_mg: float = 0.0
    calcium_mg: float = 0.0
    iron_mg: float = 0.0

    def to_dict(self) -> dict:
        return {
            "calories": self.calories,
            "protein_g": self.protein_g,
            "carbs_g": self.carbs_g,
            "fat_g": self.fat_g,
            "fiber_g": self.fiber_g,
            "sugar_g": self.sugar_g,
            "sodium_mg": self.sodium_mg,
            "cholesterol_mg": self.cholesterol_mg,
            "saturated_fat_g": self.saturated_fat_g,
            "trans_fat_g": self.trans_fat_g,
            "potassium_mg": self.potassium_mg,
            "vitamin_a_iu": self.vitamin_a_iu,
            "vitamin_c_mg": self.vitamin_c_mg,
            "calcium_mg": self.calcium_mg,
            "iron_mg": self.iron_mg,
        }

    def scale(self, factor: float) -> "NutrientInfo":
        """Scale all nutrients by a factor (for serving size adjustments)."""
        return NutrientInfo(
            calories=self.calories * factor,
            protein_g=self.protein_g * factor,
            carbs_g=self.carbs_g * factor,
            fat_g=self.fat_g * factor,
            fiber_g=self.fiber_g * factor,
            sugar_g=self.sugar_g * factor,
            sodium_mg=self.sodium_mg * factor,
            cholesterol_mg=self.cholesterol_mg * factor,
            saturated_fat_g=self.saturated_fat_g * factor,
            trans_fat_g=self.trans_fat_g * factor,
            potassium_mg=self.potassium_mg * factor,
            vitamin_a_iu=self.vitamin_a_iu * factor,
            vitamin_c_mg=self.vitamin_c_mg * factor,
            calcium_mg=self.calcium_mg * factor,
            iron_mg=self.iron_mg * factor,
        )


@dataclass
class FoodItem:
    """A food item in the database."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    brand: Optional[str] = None
    category: str = "other"
    serving_size: float = 100.0
    serving_unit: str = "g"
    nutrients: NutrientInfo = field(default_factory=NutrientInfo)
    barcode: Optional[str] = None
    verified: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "brand": self.brand,
            "category": self.category,
            "serving_size": self.serving_size,
            "serving_unit": self.serving_unit,
            "nutrients": self.nutrients.to_dict(),
            "barcode": self.barcode,
            "verified": self.verified,
        }


@dataclass
class MealEntry:
    """A logged meal entry."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    meal_type: MealType = MealType.SNACK
    foods: List[Dict[str, Any]] = field(default_factory=list)  # food_id, servings
    total_nutrients: NutrientInfo = field(default_factory=NutrientInfo)
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""
    photo_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "meal_type": self.meal_type.value,
            "foods": self.foods,
            "total_nutrients": self.total_nutrients.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
            "photo_path": self.photo_path,
        }


@dataclass
class Recipe:
    """A recipe with nutritional information."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    ingredients: List[Dict[str, Any]] = field(default_factory=list)  # food_id, amount, unit
    instructions: List[str] = field(default_factory=list)
    servings: int = 1
    prep_time_minutes: int = 0
    cook_time_minutes: int = 0
    nutrients_per_serving: NutrientInfo = field(default_factory=NutrientInfo)
    tags: List[str] = field(default_factory=list)  # vegetarian, keto, etc.
    image_url: Optional[str] = None
    source: str = "user"  # user, ai_generated, imported

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "ingredients": self.ingredients,
            "instructions": self.instructions,
            "servings": self.servings,
            "prep_time_minutes": self.prep_time_minutes,
            "cook_time_minutes": self.cook_time_minutes,
            "nutrients_per_serving": self.nutrients_per_serving.to_dict(),
            "tags": self.tags,
            "image_url": self.image_url,
            "source": self.source,
        }


@dataclass
class LabResult:
    """A parsed lab test result."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    test_name: str = ""
    value: float = 0.0
    unit: str = ""
    reference_range_low: Optional[float] = None
    reference_range_high: Optional[float] = None
    status: str = "normal"  # normal, low, high, critical
    test_date: date = field(default_factory=date.today)
    lab_name: Optional[str] = None
    notes: str = ""
    confidence: float = 1.0  # OCR confidence

    @property
    def is_abnormal(self) -> bool:
        if self.reference_range_low and self.value < self.reference_range_low:
            return True
        if self.reference_range_high and self.value > self.reference_range_high:
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "test_name": self.test_name,
            "value": self.value,
            "unit": self.unit,
            "reference_range_low": self.reference_range_low,
            "reference_range_high": self.reference_range_high,
            "status": self.status,
            "test_date": self.test_date.isoformat(),
            "lab_name": self.lab_name,
            "notes": self.notes,
            "is_abnormal": self.is_abnormal,
            "confidence": self.confidence,
        }


@dataclass
class Prescription:
    """A parsed prescription."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    medication_name: str = ""
    dosage: str = ""
    frequency: str = ""
    route: str = "oral"  # oral, topical, injection, etc.
    prescriber: Optional[str] = None
    prescribed_date: date = field(default_factory=date.today)
    refills_remaining: int = 0
    pharmacy: Optional[str] = None
    instructions: str = ""
    warnings: List[str] = field(default_factory=list)
    confidence: float = 1.0  # OCR confidence

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "medication_name": self.medication_name,
            "dosage": self.dosage,
            "frequency": self.frequency,
            "route": self.route,
            "prescriber": self.prescriber,
            "prescribed_date": self.prescribed_date.isoformat(),
            "refills_remaining": self.refills_remaining,
            "pharmacy": self.pharmacy,
            "instructions": self.instructions,
            "warnings": self.warnings,
            "confidence": self.confidence,
        }


@dataclass
class HealthRecommendation:
    """An AI-generated health recommendation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: str = ""  # nutrition, fitness, sleep, stress
    title: str = ""
    description: str = ""
    reasoning: str = ""
    priority: int = 1  # 1-5, 5 being most important
    actionable_steps: List[str] = field(default_factory=list)
    related_metrics: List[str] = field(default_factory=list)
    confidence: float = 0.8
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    dismissed: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "reasoning": self.reasoning,
            "priority": self.priority,
            "actionable_steps": self.actionable_steps,
            "related_metrics": self.related_metrics,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "dismissed": self.dismissed,
        }


@dataclass
class UserHealthProfile:
    """User health profile with preferences and restrictions."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Basic info
    birth_date: Optional[date] = None
    gender: Optional[str] = None
    height_cm: Optional[float] = None

    # Dietary
    dietary_restrictions: List[str] = field(default_factory=list)  # vegetarian, vegan, gluten-free
    food_allergies: List[str] = field(default_factory=list)
    disliked_foods: List[str] = field(default_factory=list)

    # Medical
    medications: List[str] = field(default_factory=list)
    medical_conditions: List[str] = field(default_factory=list)

    # Goals
    daily_calorie_target: Optional[int] = None
    daily_protein_target: Optional[float] = None
    daily_steps_target: int = 10000
    target_weight_kg: Optional[float] = None
    sleep_target_hours: float = 8.0

    # Preferences
    preferred_cuisines: List[str] = field(default_factory=list)
    fitness_level: str = "moderate"  # sedentary, light, moderate, active, very_active
    timezone: str = "UTC"

    # Connected providers
    connected_providers: List[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def age(self) -> Optional[int]:
        if self.birth_date:
            today = date.today()
            return today.year - self.birth_date.year - (
                (today.month, today.day) < (self.birth_date.month, self.birth_date.day)
            )
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "age": self.age,
            "gender": self.gender,
            "height_cm": self.height_cm,
            "dietary_restrictions": self.dietary_restrictions,
            "food_allergies": self.food_allergies,
            "disliked_foods": self.disliked_foods,
            "medications": self.medications,
            "medical_conditions": self.medical_conditions,
            "daily_calorie_target": self.daily_calorie_target,
            "daily_protein_target": self.daily_protein_target,
            "daily_steps_target": self.daily_steps_target,
            "target_weight_kg": self.target_weight_kg,
            "sleep_target_hours": self.sleep_target_hours,
            "preferred_cuisines": self.preferred_cuisines,
            "fitness_level": self.fitness_level,
            "timezone": self.timezone,
            "connected_providers": self.connected_providers,
        }
