"""
cosmos Health Dashboard Server

FastAPI server for health tracking dashboard with WebSocket support
for real-time metric streaming.
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ..models import (
    MetricType,
    HealthMetricReading,
    DailySummary,
    HealthGoal,
    GoalStatus,
    MealType,
    MealEntry,
    FoodItem,
    NutrientInfo,
    UserHealthProfile,
)
from ..providers import HealthProviderManager, MockHealthProvider
from ..analysis import HealthAnalysisEngine

logger = logging.getLogger(__name__)

# Configuration
HEALTH_PORT = int(os.getenv("cosmos_HEALTH_PORT", "8081"))
HEALTH_ENABLED = os.getenv("cosmos_HEALTH_ENABLED", "true").lower() == "true"

# Paths
DASHBOARD_DIR = Path(__file__).parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"

# Initialize FastAPI
app = FastAPI(
    title="cosmos Health Dashboard",
    description="Comprehensive health tracking with AI-powered insights",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = None
if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global state
provider_manager = HealthProviderManager()
analysis_engine = HealthAnalysisEngine()
user_profile = UserHealthProfile()
goals: Dict[str, HealthGoal] = {}
meal_history: List[MealEntry] = []
food_database: Dict[str, FoodItem] = {}


# ============================================
# WebSocket Manager
# ============================================

class HealthWebSocketManager:
    """Manages WebSocket connections for real-time health updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Health WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Health WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for conn in dead:
            self.disconnect(conn)

    async def send_metric_update(self, reading: HealthMetricReading):
        """Send a metric update to all clients."""
        await self.broadcast({
            "type": "metric_update",
            "data": reading.to_dict(),
            "timestamp": datetime.now().isoformat(),
        })


ws_manager = HealthWebSocketManager()


# ============================================
# Request/Response Models
# ============================================

class DateRangeRequest(BaseModel):
    start_date: str
    end_date: str


class GoalCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    metric_type: str
    target_value: float
    unit: str
    target_date: Optional[str] = None


class GoalUpdateRequest(BaseModel):
    current_value: Optional[float] = None
    status: Optional[str] = None


class MealLogRequest(BaseModel):
    meal_type: str
    foods: List[Dict[str, Any]]
    notes: Optional[str] = ""


class FoodSearchRequest(BaseModel):
    query: str
    limit: int = 20


# ============================================
# Startup/Shutdown Events
# ============================================

@app.on_event("startup")
async def startup():
    """Initialize providers on startup."""
    # Register mock provider for demo
    mock_provider = MockHealthProvider()
    provider_manager.register_provider(mock_provider)

    # Connect to providers
    await provider_manager.connect_all()

    # Subscribe to real-time updates
    def on_data(packet):
        asyncio.create_task(ws_manager.broadcast({
            "type": "bio_data",
            "data": {
                "signal_type": packet.signal_type,
                "value": packet.processed_value,
                "timestamp": packet.timestamp.isoformat(),
            },
        }))

    provider_manager.add_subscriber(on_data)

    # Load sample food database
    _load_sample_foods()

    logger.info("Health Dashboard initialized")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    await provider_manager.disconnect_all()


def _load_sample_foods():
    """Load sample food database."""
    sample_foods = [
        FoodItem(
            id="chicken_breast",
            name="Chicken Breast",
            category="protein",
            serving_size=100,
            serving_unit="g",
            nutrients=NutrientInfo(
                calories=165,
                protein_g=31,
                carbs_g=0,
                fat_g=3.6,
            ),
        ),
        FoodItem(
            id="brown_rice",
            name="Brown Rice",
            category="grain",
            serving_size=100,
            serving_unit="g",
            nutrients=NutrientInfo(
                calories=112,
                protein_g=2.6,
                carbs_g=24,
                fat_g=0.9,
                fiber_g=1.8,
            ),
        ),
        FoodItem(
            id="broccoli",
            name="Broccoli",
            category="vegetable",
            serving_size=100,
            serving_unit="g",
            nutrients=NutrientInfo(
                calories=34,
                protein_g=2.8,
                carbs_g=7,
                fat_g=0.4,
                fiber_g=2.6,
                vitamin_c_mg=89,
            ),
        ),
        FoodItem(
            id="salmon",
            name="Salmon",
            category="protein",
            serving_size=100,
            serving_unit="g",
            nutrients=NutrientInfo(
                calories=208,
                protein_g=20,
                carbs_g=0,
                fat_g=13,
            ),
        ),
        FoodItem(
            id="apple",
            name="Apple",
            category="fruit",
            serving_size=182,
            serving_unit="g",
            nutrients=NutrientInfo(
                calories=95,
                protein_g=0.5,
                carbs_g=25,
                fat_g=0.3,
                fiber_g=4.4,
            ),
        ),
        FoodItem(
            id="eggs",
            name="Eggs",
            category="protein",
            serving_size=50,
            serving_unit="g",
            nutrients=NutrientInfo(
                calories=78,
                protein_g=6,
                carbs_g=0.6,
                fat_g=5,
                cholesterol_mg=186,
            ),
        ),
        FoodItem(
            id="oatmeal",
            name="Oatmeal",
            category="grain",
            serving_size=40,
            serving_unit="g",
            nutrients=NutrientInfo(
                calories=150,
                protein_g=5,
                carbs_g=27,
                fat_g=3,
                fiber_g=4,
            ),
        ),
        FoodItem(
            id="greek_yogurt",
            name="Greek Yogurt",
            category="dairy",
            serving_size=170,
            serving_unit="g",
            nutrients=NutrientInfo(
                calories=100,
                protein_g=17,
                carbs_g=6,
                fat_g=0.7,
                calcium_mg=187,
            ),
        ),
    ]

    for food in sample_foods:
        food_database[food.id] = food


# ============================================
# Page Routes
# ============================================

@app.get("/", response_class=HTMLResponse)
@app.get("/health", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Main health dashboard page."""
    if templates:
        return templates.TemplateResponse("dashboard.html", {"request": request})
    return HTMLResponse("<h1>Health Dashboard</h1><p>Templates not found. API available at /api/health/*</p>")


@app.get("/nutrition", response_class=HTMLResponse)
async def nutrition_page(request: Request):
    """Nutrition tracking page."""
    if templates:
        return templates.TemplateResponse("nutrition.html", {"request": request})
    return HTMLResponse("<h1>Nutrition</h1><p>Templates not found.</p>")


@app.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request):
    """Document parsing page."""
    if templates:
        return templates.TemplateResponse("documents.html", {"request": request})
    return HTMLResponse("<h1>Documents</h1><p>Templates not found.</p>")


@app.get("/insights", response_class=HTMLResponse)
async def insights_page(request: Request):
    """AI insights page."""
    if templates:
        return templates.TemplateResponse("insights.html", {"request": request})
    return HTMLResponse("<h1>Insights</h1><p>Templates not found.</p>")


# ============================================
# Health API Endpoints
# ============================================

@app.get("/api/health/summary")
async def get_health_summary(date_str: Optional[str] = None):
    """Get daily health summary."""
    target_date = date.fromisoformat(date_str) if date_str else date.today()

    summary = await provider_manager.get_daily_summary(target_date)
    wellness = analysis_engine.calculate_wellness_score(summary)

    return JSONResponse({
        "date": target_date.isoformat(),
        "summary": summary.to_dict(),
        "wellness_score": wellness.to_dict(),
    })


@app.get("/api/health/metrics/{metric_type}")
async def get_metrics(
    metric_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Get historical metrics for a specific type."""
    try:
        mt = MetricType(metric_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown metric type: {metric_type}")

    end = date.fromisoformat(end_date) if end_date else date.today()
    start = date.fromisoformat(start_date) if start_date else end - timedelta(days=7)

    readings = await provider_manager.get_metrics([mt], start, end)

    return JSONResponse({
        "metric_type": metric_type,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "count": len(readings),
        "readings": [r.to_dict() for r in readings],
    })


@app.get("/api/health/trends")
async def get_trends(days: int = 7):
    """Get trend analysis for all metrics."""
    # Load recent data
    end = date.today()
    start = end - timedelta(days=days)

    readings = await provider_manager.get_metrics(
        provider_manager.get_supported_metrics(), start, end
    )
    summaries = []
    for d in range(days):
        day = end - timedelta(days=d)
        summary = await provider_manager.get_daily_summary(day)
        summaries.append(summary)

    analysis_engine.load_data(readings, summaries)
    trends = analysis_engine.analyze_trends(days=days)

    return JSONResponse({
        "period_days": days,
        "trends": [t.to_dict() for t in trends],
    })


@app.get("/api/health/alerts")
async def get_alerts():
    """Get active health alerts."""
    # Detect anomalies in recent data
    end = date.today()
    start = end - timedelta(days=14)

    readings = await provider_manager.get_metrics(
        provider_manager.get_supported_metrics(), start, end
    )
    analysis_engine.load_data(readings, [])

    anomalies = analysis_engine.detect_anomalies()
    alerts = analysis_engine.create_alerts_from_anomalies(anomalies)

    return JSONResponse({
        "count": len(alerts),
        "alerts": [a.to_dict() for a in alerts],
    })


@app.get("/api/health/insights")
async def get_insights():
    """Get AI-generated health insights."""
    end = date.today()
    start = end - timedelta(days=7)

    summaries = []
    for d in range(7):
        day = end - timedelta(days=d)
        summary = await provider_manager.get_daily_summary(day)
        summaries.append(summary)

    insights = analysis_engine.generate_insights(summaries)

    return JSONResponse({
        "count": len(insights),
        "insights": [i.to_dict() for i in insights],
    })


@app.get("/api/health/providers")
async def get_providers():
    """Get status of connected health providers."""
    return JSONResponse(provider_manager.get_status())


# ============================================
# Goals API
# ============================================

@app.get("/api/goals")
async def list_goals():
    """List all health goals."""
    return JSONResponse({
        "count": len(goals),
        "goals": [g.to_dict() for g in goals.values()],
    })


@app.post("/api/goals")
async def create_goal(request: GoalCreateRequest):
    """Create a new health goal."""
    try:
        metric = MetricType(request.metric_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown metric type: {request.metric_type}")

    goal = HealthGoal(
        name=request.name,
        description=request.description,
        metric_type=metric,
        target_value=request.target_value,
        unit=request.unit,
        target_date=date.fromisoformat(request.target_date) if request.target_date else None,
    )

    goals[goal.id] = goal

    return JSONResponse({
        "success": True,
        "goal": goal.to_dict(),
    })


@app.put("/api/goals/{goal_id}")
async def update_goal(goal_id: str, request: GoalUpdateRequest):
    """Update a health goal."""
    if goal_id not in goals:
        raise HTTPException(status_code=404, detail="Goal not found")

    goal = goals[goal_id]

    if request.current_value is not None:
        goal.current_value = request.current_value
        goal.updated_at = datetime.now()

    if request.status:
        try:
            goal.status = GoalStatus(request.status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown status: {request.status}")

    return JSONResponse({
        "success": True,
        "goal": goal.to_dict(),
    })


@app.delete("/api/goals/{goal_id}")
async def delete_goal(goal_id: str):
    """Delete a health goal."""
    if goal_id not in goals:
        raise HTTPException(status_code=404, detail="Goal not found")

    del goals[goal_id]

    return JSONResponse({"success": True})


# ============================================
# Nutrition API
# ============================================

@app.get("/api/nutrition/foods")
async def search_foods(query: str, limit: int = 20):
    """Search food database."""
    query_lower = query.lower()
    results = [
        food.to_dict()
        for food in food_database.values()
        if query_lower in food.name.lower()
    ][:limit]

    return JSONResponse({
        "query": query,
        "count": len(results),
        "foods": results,
    })


@app.get("/api/nutrition/foods/{food_id}")
async def get_food(food_id: str):
    """Get food item by ID."""
    if food_id not in food_database:
        raise HTTPException(status_code=404, detail="Food not found")

    return JSONResponse(food_database[food_id].to_dict())


@app.post("/api/nutrition/meals")
async def log_meal(request: MealLogRequest):
    """Log a meal."""
    try:
        meal_type = MealType(request.meal_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown meal type: {request.meal_type}")

    # Calculate total nutrients
    total = NutrientInfo()
    for item in request.foods:
        food_id = item.get("food_id")
        servings = item.get("servings", 1)

        if food_id in food_database:
            food = food_database[food_id]
            scaled = food.nutrients.scale(servings)
            total.calories += scaled.calories
            total.protein_g += scaled.protein_g
            total.carbs_g += scaled.carbs_g
            total.fat_g += scaled.fat_g
            total.fiber_g += scaled.fiber_g

    entry = MealEntry(
        meal_type=meal_type,
        foods=request.foods,
        total_nutrients=total,
        notes=request.notes,
    )

    meal_history.append(entry)

    return JSONResponse({
        "success": True,
        "meal": entry.to_dict(),
    })


@app.get("/api/nutrition/daily")
async def get_daily_nutrition(date_str: Optional[str] = None):
    """Get daily nutrition totals."""
    target_date = date.fromisoformat(date_str) if date_str else date.today()

    # Filter meals for the target date
    day_meals = [
        m for m in meal_history
        if m.timestamp.date() == target_date
    ]

    # Aggregate nutrients
    total = NutrientInfo()
    for meal in day_meals:
        total.calories += meal.total_nutrients.calories
        total.protein_g += meal.total_nutrients.protein_g
        total.carbs_g += meal.total_nutrients.carbs_g
        total.fat_g += meal.total_nutrients.fat_g
        total.fiber_g += meal.total_nutrients.fiber_g

    return JSONResponse({
        "date": target_date.isoformat(),
        "meals_count": len(day_meals),
        "meals": [m.to_dict() for m in day_meals],
        "totals": total.to_dict(),
        "targets": {
            "calories": user_profile.daily_calorie_target or 2000,
            "protein_g": user_profile.daily_protein_target or 50,
        },
    })


# ============================================
# Document Parsing API
# ============================================

@app.post("/api/documents/parse")
async def parse_document(
    file: UploadFile = File(...),
    doc_type: str = Form("lab_result"),
):
    """Parse a health document using DeepSeek OCR."""
    # Save uploaded file temporarily
    content = await file.read()

    # In production, this would call the DeepSeek OCR parser
    # For now, return a placeholder response
    return JSONResponse({
        "success": True,
        "document_type": doc_type,
        "filename": file.filename,
        "message": "Document parsing requires DeepSeek OCR integration. See ocr_parser.py for implementation.",
        "results": [],
    })


# ============================================
# Recommendations API
# ============================================

@app.get("/api/recommendations")
async def get_recommendations(focus_area: Optional[str] = None):
    """Get AI health recommendations."""
    # Get insights as base for recommendations
    end = date.today()
    summaries = []
    for d in range(7):
        day = end - timedelta(days=d)
        summary = await provider_manager.get_daily_summary(day)
        summaries.append(summary)

    insights = analysis_engine.generate_insights(summaries)

    # Filter by focus area if specified
    if focus_area:
        insights = [i for i in insights if i.category == focus_area]

    # Convert insights to recommendations
    recommendations = []
    for insight in insights:
        if insight.actionable:
            recommendations.append({
                "category": insight.category,
                "title": insight.title,
                "description": insight.message,
                "priority": insight.priority,
                "related_metrics": insight.related_metrics,
            })

    return JSONResponse({
        "count": len(recommendations),
        "recommendations": recommendations,
    })


# ============================================
# User Profile API
# ============================================

@app.get("/api/profile")
async def get_profile():
    """Get user health profile."""
    return JSONResponse(user_profile.to_dict())


@app.put("/api/profile")
async def update_profile(updates: Dict[str, Any]):
    """Update user health profile."""
    global user_profile

    for key, value in updates.items():
        if hasattr(user_profile, key):
            setattr(user_profile, key, value)

    user_profile.updated_at = datetime.now()

    return JSONResponse({
        "success": True,
        "profile": user_profile.to_dict(),
    })


# ============================================
# WebSocket Endpoint
# ============================================

@app.websocket("/ws/health")
async def health_websocket(websocket: WebSocket):
    """WebSocket for real-time health metrics."""
    await ws_manager.connect(websocket)

    try:
        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Health Dashboard",
            "providers": provider_manager.get_status(),
        })

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)

                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

                elif data.get("type") == "subscribe_metric":
                    metric = data.get("metric")
                    await websocket.send_json({
                        "type": "subscribed",
                        "metric": metric,
                    })

            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


# ============================================
# Health Check
# ============================================

@app.get("/api/health/status")
async def health_status():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "version": "1.0.0",
        "providers_connected": sum(
            1 for p in provider_manager.providers.values() if p.status.connected
        ),
        "websocket_clients": len(ws_manager.active_connections),
    })


def main():
    """Run the health dashboard server."""
    if not HEALTH_ENABLED:
        logger.warning("Health dashboard is disabled")
        return

    host = os.getenv("cosmos_HEALTH_HOST", "0.0.0.0")

    logger.info(f"Starting cosmos Health Dashboard on {host}:{HEALTH_PORT}")

    uvicorn.run(
        "cosmos.health.dashboard.server:app",
        host=host,
        port=HEALTH_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
