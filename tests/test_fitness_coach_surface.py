from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import fitness_coach_surface as surface


def _fake_workspace():
    return {
        "coach_workspace_id": "coach-1",
        "public_token": "coach-token",
        "profile_id": "profile-1",
        "display_name": "Jean",
        "status": "active",
        "timezone": "America/Caracas",
        "dietary_preferences": [],
        "allergies": [],
        "equipment_available": ["bodyweight"],
    }


def _fake_plan():
    return {
        "daily_plan_id": "daily-plan-profile-1-2026-06-02",
        "profile_id": "profile-1",
        "plan_date": "2026-06-02",
        "target_calories": 2200,
        "target_protein_g": 160,
        "target_carbs_g": 250,
        "target_fat_g": 70,
        "target_water_ml": 2500,
        "meals": [
            {"name": "Desayuno", "calories": 550, "protein_g": 40, "carbs_g": 62, "fat_g": 18, "notes": "Avena y huevos", "substitutions": ["arepa", "yogur"]},
            {"name": "Almuerzo", "calories": 770, "protein_g": 56, "carbs_g": 88, "fat_g": 25, "notes": "Pollo arroz vegetales", "substitutions": ["papa", "pescado"]},
        ],
        "routine": {
            "routine_id": "routine-1",
            "routine_title": "Fuerza base",
            "day_title": "Empuje",
            "exercises": [
                {"order_index": 1, "exercise_id": "exercise-push-up", "exercise_name": "Push-up", "sets": 3, "target_reps_min": 8, "target_reps_max": 12, "rest_seconds": 60, "primary_muscles": ["chest"], "equipment": ["bodyweight"], "instructions": ["Mantén el cuerpo recto"]}
            ],
        },
        "recommendations": [],
    }


def test_coach_today_renders_navigation_and_records_open(monkeypatch):
    events = []
    monkeypatch.setattr(surface, "_workspace", lambda token: _fake_workspace())
    monkeypatch.setattr(surface, "_daily_plan", lambda profile_id, day=None: _fake_plan())
    monkeypatch.setattr(surface, "_nutrition_summary", lambda profile_id, day: {"calories": 300, "protein_g": 30, "carbs_g": 20, "fat_g": 10, "water_ml": 500, "entries": 1})
    monkeypatch.setattr(surface, "_progress", lambda profile_id: {"latest_weight_kg": 80})
    monkeypatch.setattr(surface, "_record_event", lambda workspace, event_type, **kwargs: events.append((event_type, kwargs)) or {"ok": True})

    html = surface.render_today("coach-token")

    assert "Tu plan de hoy" in html
    assert "Ver rutina" in html
    assert "/w/coach-token/coach/nutrition/today" in html
    assert "300" in html
    assert events[0][0] == "coach_opened"


def test_routine_and_nutrition_pages_render_cards(monkeypatch):
    monkeypatch.setattr(surface, "_workspace", lambda token: _fake_workspace())
    monkeypatch.setattr(surface, "_daily_plan", lambda profile_id, day=None: _fake_plan())
    monkeypatch.setattr(surface, "_nutrition_summary", lambda profile_id, day: {"calories": 300, "protein_g": 30, "carbs_g": 20, "fat_g": 10, "water_ml": 500, "entries": 1})
    monkeypatch.setattr(surface, "_record_event", lambda *args, **kwargs: {"ok": True})

    routine_html = surface.render_routine("coach-token")
    nutrition_html = surface.render_nutrition("coach-token")

    assert "Routine Player" in routine_html
    assert "Push-up" in routine_html
    assert "Marcar set/completado" in routine_html
    assert "Daily Regimen" in nutrition_html
    assert "Desayuno" in nutrition_html
    assert "Reportar comida" in nutrition_html


def test_public_routes_and_event_endpoint(monkeypatch):
    app = FastAPI()
    app.include_router(surface.router)
    client = TestClient(app)
    events = []

    monkeypatch.setattr(surface, "render_today", lambda token: f"<html>today {token}</html>")
    monkeypatch.setattr(surface, "render_routine", lambda token: f"<html>routine {token}</html>")
    monkeypatch.setattr(surface, "render_nutrition", lambda token: f"<html>nutrition {token}</html>")
    monkeypatch.setattr(surface, "render_progress", lambda token: f"<html>progress {token}</html>")
    monkeypatch.setattr(surface, "_workspace", lambda token: _fake_workspace())
    monkeypatch.setattr(surface, "_record_event", lambda workspace, event_type, comment=None, metadata=None: events.append((event_type, comment, metadata)) or {"ok": True})

    assert client.get("/w/coach-token/coach").status_code == 200
    assert "routine coach-token" in client.get("/w/coach-token/coach/routine/today").text
    assert "nutrition coach-token" in client.get("/w/coach-token/coach/nutrition/today").text
    assert "progress coach-token" in client.get("/w/coach-token/coach/progress").text

    response = client.post("/w/coach-token/coach/event", data={"event_type": "routine_started", "comment": "listo"}, follow_redirects=False)
    assert response.status_code == 303
    assert events == [("routine_started", "listo", {})]


def test_create_or_get_coach_workspace_uses_existing_or_inserts(monkeypatch):
    inserted = []
    monkeypatch.setattr(surface.sql, "one", lambda query, **kwargs: None)
    monkeypatch.setattr(surface.sql, "statement_one", lambda statement, **kwargs: inserted.append(statement) or {"coach_workspace_id": "coach-new", "public_token": "fixed-token"})
    monkeypatch.setattr(surface, "_token", lambda: "fixed-token")
    monkeypatch.setattr(surface, "_base_url", lambda: "https://zeus-sandbox.kidu.app")

    row = surface.create_or_get_coach_workspace("profile-1")

    assert row["public_token"] == "fixed-token"
    assert "fitness.coach_workspaces" in inserted[0]
    assert "https://zeus-sandbox.kidu.app/w/fixed-token/coach" in inserted[0]
