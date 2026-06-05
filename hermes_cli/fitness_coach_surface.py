"""Token-scoped Fitness Coach public surface.

This module renders the visual complement for Fitness Coach Core: a daily hub,
routine player, nutrition plan, progress view, and narrow event endpoints. The
chat agent remains the primary UX; these pages are public-token surfaces backed
by the local Agent Core `fitness` schema.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import secrets
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from hermes_cli import agent_core_sql as sql

router = APIRouter()

SAFE_PUBLIC_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
ALLOWED_COACH_EVENT_TYPES = {
    "coach_opened",
    "routine_started",
    "set_completed",
    "workout_finished",
    "meal_photo_requested",
    "meal_logged",
    "barcode_scanned",
    "checkin_submitted",
    "sleep_plan_acknowledged",
}


def _user() -> str:
    return sql.runtime_env().get("FITNESS_DB_RUNTIME_USER", "fitness_runtime")


def _q(value: Any) -> str:
    return sql.quote_literal(value)


def _j(value: Any) -> str:
    return sql.quote_jsonb(value)


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        return default


def _today_iso() -> str:
    return date.today().isoformat()


def _base_url() -> str:
    return (
        sql.runtime_env().get("WORKSPACE_PUBLIC_BASE_URL")
        or sql.runtime_env().get("AGENT_PUBLIC_BASE_URL")
        or os.getenv("WORKSPACE_PUBLIC_BASE_URL")
        or "https://zeus-sandbox.kidu.app"
    ).rstrip("/")


def _token() -> str:
    return secrets.token_urlsafe(18)


def validate_public_token(public_token: str) -> str:
    token = str(public_token or "").strip()
    if not SAFE_PUBLIC_TOKEN_RE.fullmatch(token):
        raise HTTPException(status_code=404, detail="Coach workspace not found")
    return token


def _workspace(public_token: str) -> dict[str, Any]:
    public_token = validate_public_token(public_token)
    row = sql.one(
        f"""
        SELECT cw.*, p.display_name, p.timezone, p.dietary_preferences, p.allergies, p.equipment_available,
               cw.expires_at IS NOT NULL AND cw.expires_at < now() AS is_expired
        FROM fitness.coach_workspaces cw
        JOIN fitness.profiles p ON p.profile_id=cw.profile_id
        WHERE cw.public_token={_q(public_token)}
        """,
        user=_user(),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Coach workspace not found")
    if row.get("is_expired") or row.get("status") in {"expired", "revoked"}:
        raise HTTPException(status_code=410, detail="Coach workspace expired")
    return row


def create_or_get_coach_workspace(profile_id: str, *, public_token: str | None = None) -> dict[str, Any]:
    existing = sql.one(
        f"SELECT * FROM fitness.coach_workspaces WHERE profile_id={_q(profile_id)} AND status='active' ORDER BY created_at DESC",
        user=_user(),
    )
    if existing:
        return existing
    token = validate_public_token(public_token) if public_token else _token()
    workspace_id = "coach-" + hashlib.sha1(f"{profile_id}:{token}".encode()).hexdigest()[:18]
    public_url = f"{_base_url()}/w/{token}/coach"
    return sql.statement_one(
        f"""
        INSERT INTO fitness.coach_workspaces (coach_workspace_id, public_token, profile_id, status, public_url, metadata, created_at, updated_at)
        VALUES ({_q(workspace_id)}, {_q(token)}, {_q(profile_id)}, 'active', {_q(public_url)}, {_j({'source': 'fitness_coach_surface'})}, now(), now())
        ON CONFLICT (public_token) DO UPDATE SET updated_at=now()
        RETURNING *
        """,
        user=_user(),
    )


def _active_goal(profile_id: str) -> dict[str, Any] | None:
    return sql.one(
        f"SELECT * FROM fitness.goals WHERE profile_id={_q(profile_id)} AND status='active' ORDER BY start_date DESC, updated_at DESC",
        user=_user(),
    )


def _nutrition_summary(profile_id: str, day: str) -> dict[str, Any]:
    return sql.one(
        f"""
        SELECT COALESCE(sum(calories),0) AS calories, COALESCE(sum(protein_g),0) AS protein_g,
               COALESCE(sum(carbs_g),0) AS carbs_g, COALESCE(sum(fat_g),0) AS fat_g,
               COALESCE(sum(water_ml),0) AS water_ml, count(*) AS entries
        FROM fitness.nutrition_logs
        WHERE profile_id={_q(profile_id)} AND occurred_at::date={_q(day)}::date
        """,
        user=_user(),
    ) or {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "water_ml": 0, "entries": 0}


def _latest_metrics(profile_id: str) -> dict[str, Any] | None:
    return sql.one(
        f"SELECT * FROM fitness.body_metrics WHERE profile_id={_q(profile_id)} ORDER BY measured_at DESC",
        user=_user(),
    )


def _routine_for_day(profile_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    routine = sql.one(
        f"SELECT * FROM fitness.routines WHERE profile_id={_q(profile_id)} AND status='active' ORDER BY updated_at DESC",
        user=_user(),
    )
    if not routine:
        return None, None, []
    days = sql.rows(
        f"SELECT * FROM fitness.routine_days WHERE routine_id={_q(routine['routine_id'])} ORDER BY day_index",
        user=_user(),
    )
    if not days:
        return routine, None, []
    selected = days[(date.today().isoweekday() - 1) % len(days)]
    exercises = sql.rows(
        f"""
        SELECT re.*, e.name AS exercise_name, e.primary_muscles, e.secondary_muscles, e.equipment, e.instructions, e.media_refs
        FROM fitness.routine_exercises re
        LEFT JOIN fitness.exercises e ON e.exercise_id=re.exercise_id
        WHERE re.routine_day_id={_q(selected.get('routine_day_id'))}
        ORDER BY re.order_index
        """,
        user=_user(),
    )
    return routine, selected, exercises


def _default_meals(goal: dict[str, Any] | None) -> list[dict[str, Any]]:
    calories = _num(goal.get("target_calories") if goal else None, 2200)
    protein = _num(goal.get("target_protein_g") if goal else None, 150)
    carbs = _num(goal.get("target_carbs_g") if goal else None, 230)
    fat = _num(goal.get("target_fat_g") if goal else None, 70)
    splits = [
        ("Desayuno", 0.25, "Proteína + carbohidrato fácil + fruta"),
        ("Almuerzo", 0.35, "Proteína magra + carbohidrato principal + vegetales"),
        ("Merienda", 0.15, "Proteína rápida o yogur/fruta según hambre"),
        ("Cena", 0.25, "Proteína + vegetales + carbohidrato moderado si entrenaste"),
    ]
    meals = []
    for name, pct, note in splits:
        meals.append({
            "name": name,
            "calories": round(calories * pct),
            "protein_g": round(protein * pct),
            "carbs_g": round(carbs * pct),
            "fat_g": round(fat * pct),
            "notes": note,
            "substitutions": ["pollo/pescado/huevos", "arroz/papa/arepa/avena", "ensalada/vegetales"],
        })
    return meals


def generate_daily_plan(profile_id: str, day: str | None = None) -> dict[str, Any]:
    plan_date = day or _today_iso()
    goal = _active_goal(profile_id)
    routine, routine_day, exercises = _routine_for_day(profile_id)
    target_calories = _num(goal.get("target_calories") if goal else None, 2200)
    target_protein = _num(goal.get("target_protein_g") if goal else None, 150)
    target_carbs = _num(goal.get("target_carbs_g") if goal else None, 230)
    target_fat = _num(goal.get("target_fat_g") if goal else None, 70)
    target_water = _num(goal.get("target_water_ml") if goal else None, 2500)
    routine_payload = {
        "routine_id": routine.get("routine_id") if routine else None,
        "routine_title": routine.get("title") if routine else "Caminata ligera + movilidad",
        "day_title": routine_day.get("title") if routine_day else "Día base",
        "exercises": exercises,
    }
    recommendations = [
        {"title": "Prioridad del día", "body": "Cumplir proteína y registrar comida con foto o barcode."},
        {"title": "Sueño", "body": "Planifica 7.5–8.5 horas de sueño; evita cafeína tarde."},
    ]
    daily_plan_id = f"daily-plan-{profile_id}-{plan_date}"
    return sql.statement_one(
        f"""
        INSERT INTO fitness.daily_plans (daily_plan_id, profile_id, plan_date, plan_type, title, target_calories,
          target_protein_g, target_carbs_g, target_fat_g, target_water_ml, meals, routine, recommendations, status, metadata, created_at, updated_at)
        VALUES ({_q(daily_plan_id)}, {_q(profile_id)}, {_q(plan_date)}::date, 'daily_regimen', {_q('Régimen diario')},
          {_q(target_calories)}, {_q(target_protein)}, {_q(target_carbs)}, {_q(target_fat)}, {_q(target_water)},
          {_j(_default_meals(goal))}, {_j(routine_payload)}, {_j(recommendations)}, 'generated', {_j({'generated_by':'fitness_coach_surface'})}, now(), now())
        ON CONFLICT (profile_id, plan_date, plan_type) DO UPDATE SET target_calories=EXCLUDED.target_calories,
          target_protein_g=EXCLUDED.target_protein_g, target_carbs_g=EXCLUDED.target_carbs_g, target_fat_g=EXCLUDED.target_fat_g,
          target_water_ml=EXCLUDED.target_water_ml, meals=EXCLUDED.meals, routine=EXCLUDED.routine, recommendations=EXCLUDED.recommendations,
          status='generated', updated_at=now()
        RETURNING *
        """,
        user=_user(),
    )


def _daily_plan(profile_id: str, day: str | None = None) -> dict[str, Any]:
    plan_date = day or _today_iso()
    row = sql.one(
        f"SELECT * FROM fitness.daily_plans WHERE profile_id={_q(profile_id)} AND plan_date={_q(plan_date)}::date AND plan_type='daily_regimen'",
        user=_user(),
    )
    return row or generate_daily_plan(profile_id, plan_date)


def _record_event(workspace: dict[str, Any], event_type: str, *, comment: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if event_type not in ALLOWED_COACH_EVENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported coach event type")
    return sql.statement_one(
        f"""
        INSERT INTO fitness.coach_events (coach_workspace_id, profile_id, event_type, actor_type, actor_ref, comment, metadata, occurred_at)
        VALUES ({_q(workspace.get('coach_workspace_id'))}, {_q(workspace.get('profile_id'))}, {_q(event_type)}, 'user', {_q(workspace.get('display_name'))}, {_q(comment)}, {_j(metadata or {})}, now())
        RETURNING *
        """,
        user=_user(),
    )


def _progress(profile_id: str) -> dict[str, Any]:
    return sql.one(
        f"""
        SELECT count(DISTINCT nl.occurred_at::date) AS logged_days,
               avg(nl.calories) AS avg_calories,
               avg(nl.protein_g) AS avg_protein_g,
               (SELECT count(*) FROM fitness.workout_sessions ws WHERE ws.profile_id={_q(profile_id)} AND ws.started_at >= now() - interval '7 days') AS sessions_7d,
               (SELECT weight_kg FROM fitness.body_metrics bm WHERE bm.profile_id={_q(profile_id)} ORDER BY measured_at DESC LIMIT 1) AS latest_weight_kg
        FROM fitness.nutrition_logs nl
        WHERE nl.profile_id={_q(profile_id)} AND nl.occurred_at >= now() - interval '14 days'
        """,
        user=_user(),
    ) or {}


def _meal_cards(meals: list[dict[str, Any]]) -> str:
    parts = []
    for meal in meals:
        subs = ", ".join(str(x) for x in (meal.get("substitutions") or []))
        parts.append(f"""
        <article class="meal-card">
          <div><span class="eyebrow">{_e(meal.get('name'))}</span><h3>{_e(meal.get('calories'))} kcal</h3></div>
          <p>{_e(meal.get('notes'))}</p>
          <div class="macro-row"><span>P {_e(meal.get('protein_g'))}g</span><span>C {_e(meal.get('carbs_g'))}g</span><span>G {_e(meal.get('fat_g'))}g</span></div>
          <p class="muted small">Sustituciones: {_e(subs)}</p>
        </article>
        """)
    return "".join(parts)


def _exercise_cards(exercises: list[dict[str, Any]]) -> str:
    if not exercises:
        return "<article class='exercise-card'><h3>Caminata + movilidad</h3><p>25–35 min zona cómoda, luego 8 min de movilidad general.</p></article>"
    parts = []
    for item in exercises:
        muscles = ", ".join(item.get("primary_muscles") or [])
        equipment = ", ".join(item.get("equipment") or [])
        instructions = item.get("instructions") or []
        first_instruction = instructions[0] if instructions else "Ejecuta con control y técnica limpia."
        parts.append(f"""
        <article class="exercise-card">
          <div class="exercise-top"><span class="index">{_e(item.get('order_index'))}</span><div><h3>{_e(item.get('exercise_name') or item.get('exercise_id'))}</h3><p>{_e(muscles)} · {_e(equipment)}</p></div></div>
          <div class="prescription"><span>{_e(item.get('sets') or '—')} sets</span><span>{_e(item.get('target_reps_min') or '—')}-{_e(item.get('target_reps_max') or '—')} reps</span><span>{_e(item.get('rest_seconds') or 60)}s descanso</span></div>
          <p>{_e(first_instruction)}</p>
          <form method="post" action="event"><input type="hidden" name="event_type" value="set_completed"><input type="hidden" name="metadata" value='{_e(json.dumps({'exercise_id': item.get('exercise_id')}))}'><button>Marcar set/completado</button></form>
        </article>
        """)
    return "".join(parts)


def _layout(title: str, body: str, token: str, active: str = "today") -> str:
    nav = [("today", "Hoy", f"/w/{token}/coach"), ("routine", "Rutina", f"/w/{token}/coach/routine/today"), ("nutrition", "Dieta", f"/w/{token}/coach/nutrition/today"), ("progress", "Progreso", f"/w/{token}/coach/progress")]
    links = "".join(f"<a class='{ 'active' if key == active else ''}' href='{href}'>{label}</a>" for key, label, href in nav)
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title><style>
:root{{--bg:#07110b;--panel:#101c14;--panel2:#142419;--ink:#eef8ef;--muted:#9fb2a4;--line:rgba(238,248,239,.13);--green:#79e2a0;--lime:#c8ff72;--orange:#ffb86b;--red:#ff8b82;--shadow:rgba(0,0,0,.42)}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 10% 0%,rgba(121,226,160,.22),transparent 30%),radial-gradient(circle at 90% 12%,rgba(200,255,114,.12),transparent 28%),var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}} a{{color:inherit;text-decoration:none}} .shell{{width:min(1180px,calc(100% - 24px));margin:0 auto;padding:22px 0 38px}} .top{{display:flex;justify-content:space-between;align-items:center;gap:14px;margin-bottom:24px}} .brand{{font-weight:950;letter-spacing:-.04em;font-size:20px}} .brand span{{color:var(--green)}} .nav{{display:flex;gap:8px;flex-wrap:wrap;background:rgba(255,255,255,.05);border:1px solid var(--line);padding:6px;border-radius:999px}} .nav a{{padding:10px 14px;border-radius:999px;color:var(--muted);font-weight:850;font-size:14px}} .nav a.active{{background:var(--green);color:#07110b}} .hero{{display:grid;grid-template-columns:minmax(0,1fr) minmax(280px,.38fr);gap:18px;margin-bottom:18px}} .card,.metric,.meal-card,.exercise-card{{background:linear-gradient(180deg,rgba(255,255,255,.07),rgba(255,255,255,.035));border:1px solid var(--line);border-radius:28px;box-shadow:0 24px 70px var(--shadow)}} .card{{padding:clamp(22px,4vw,42px)}} h1{{font-size:clamp(42px,7vw,82px);line-height:.9;letter-spacing:-.07em;margin:10px 0 16px;max-width:11ch}} h2{{font-size:clamp(26px,4vw,44px);letter-spacing:-.05em;margin:0 0 16px}} h3{{margin:0 0 8px;font-size:22px;letter-spacing:-.03em}} p{{line-height:1.5}} .muted{{color:var(--muted)}} .small{{font-size:13px}} .eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:.13em;color:var(--green);font-weight:950}} .grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}} .metric{{padding:18px}} .metric strong{{display:block;font-size:28px;letter-spacing:-.04em;margin-top:8px}} .macro-row,.prescription{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}} .macro-row span,.prescription span{{border:1px solid var(--line);background:rgba(255,255,255,.06);border-radius:999px;padding:8px 10px;color:var(--muted);font-weight:800;font-size:13px}} .meal-grid,.exercise-list{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}} .meal-card,.exercise-card{{padding:20px}} .exercise-top{{display:flex;gap:14px;align-items:flex-start}} .index{{display:grid;place-items:center;min-width:38px;height:38px;border-radius:14px;background:var(--green);color:#07110b;font-weight:950}} button,.button{{border:0;border-radius:999px;background:var(--green);color:#07110b;font-weight:950;min-height:42px;padding:0 16px;cursor:pointer;display:inline-flex;align-items:center;justify-content:center}} .secondary{{background:rgba(255,255,255,.08);color:var(--ink);border:1px solid var(--line)}} form{{margin-top:12px}} textarea,input[type=text]{{width:100%;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.06);color:var(--ink);padding:12px;font:inherit}} .bar{{height:12px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden}} .bar span{{display:block;height:100%;background:linear-gradient(90deg,var(--green),var(--lime));border-radius:inherit}} @media(max-width:820px){{.top,.hero{{display:grid;grid-template-columns:1fr}}.grid,.meal-grid,.exercise-list{{grid-template-columns:1fr}}.nav{{border-radius:24px}}}}
</style></head><body><main class="shell"><div class="top"><a class="brand" href="/w/{_e(token)}/coach">Fitness <span>Coach</span></a><nav class="nav">{links}</nav></div>{body}</main></body></html>"""


def render_today(public_token: str) -> str:
    ws = _workspace(public_token)
    profile_id = ws["profile_id"]
    day = _today_iso()
    plan = _daily_plan(profile_id, day)
    summary = _nutrition_summary(profile_id, day)
    progress = _progress(profile_id)
    consumed = _num(summary.get("calories"))
    target = _num(plan.get("target_calories"), 2200)
    pct = max(0, min(100, int((consumed / target) * 100))) if target else 0
    routine = plan.get("routine") or {}
    body = f"""
    <section class="hero"><div class="card"><span class="eyebrow">{_e(datetime.now().strftime('%d/%m/%Y'))}</span><h1>Tu plan de hoy</h1><p class="muted">Rutina: {_e(routine.get('routine_title'))} · {_e(routine.get('day_title'))}. Objetivo: {_e(round(target))} kcal y {_e(plan.get('target_protein_g'))}g proteína.</p><div class="bar"><span style="width:{pct}%"></span></div><p class="small muted">Registrado hoy: {_e(round(consumed))} kcal de {_e(round(target))}.</p><p><a class="button" href="/w/{_e(public_token)}/coach/routine/today">Ver rutina</a> <a class="button secondary" href="/w/{_e(public_token)}/coach/nutrition/today">Ver dieta</a></p></div><aside class="card"><span class="eyebrow">Check-in rápido</span><h2>¿Cómo vas?</h2><form method="post" action="/w/{_e(public_token)}/coach/event"><input type="hidden" name="event_type" value="checkin_submitted"><textarea name="comment" placeholder="Energía, hambre, sueño, bloqueo..."></textarea><button>Enviar check-in</button></form></aside></section>
    <section class="grid"><div class="metric"><span class="eyebrow">Calorías</span><strong>{_e(round(consumed))}/{_e(round(target))}</strong></div><div class="metric"><span class="eyebrow">Proteína</span><strong>{_e(round(_num(summary.get('protein_g'))))}/{_e(round(_num(plan.get('target_protein_g'),150)))}g</strong></div><div class="metric"><span class="eyebrow">Peso</span><strong>{_e(progress.get('latest_weight_kg') or '—')} kg</strong></div></section>
    """
    _record_event(ws, "coach_opened", metadata={"page": "today"})
    return _layout("Fitness Coach - Hoy", body, public_token, "today")


def render_routine(public_token: str) -> str:
    ws = _workspace(public_token)
    plan = _daily_plan(ws["profile_id"], _today_iso())
    routine = plan.get("routine") or {}
    exercises = routine.get("exercises") or []
    body = f"""
    <section class="card"><span class="eyebrow">Routine Player</span><h1>{_e(routine.get('day_title') or 'Rutina')}</h1><p class="muted">{_e(routine.get('routine_title'))}. Marca sets al completar; Zeus puede usar esos eventos para ajustar el seguimiento.</p><form method="post" action="/w/{_e(public_token)}/coach/event"><input type="hidden" name="event_type" value="routine_started"><button>Iniciar rutina</button></form></section>
    <section class="exercise-list" style="margin-top:14px">{_exercise_cards(exercises)}</section>
    """
    _record_event(ws, "coach_opened", metadata={"page": "routine"})
    return _layout("Fitness Coach - Rutina", body, public_token, "routine")


def render_nutrition(public_token: str) -> str:
    ws = _workspace(public_token)
    plan = _daily_plan(ws["profile_id"], _today_iso())
    summary = _nutrition_summary(ws["profile_id"], _today_iso())
    meals = plan.get("meals") or []
    remaining = max(0, _num(plan.get("target_calories"), 2200) - _num(summary.get("calories")))
    body = f"""
    <section class="card"><span class="eyebrow">Daily Regimen</span><h1>Dieta de hoy</h1><p class="muted">Meta: {_e(round(_num(plan.get('target_calories'),2200)))} kcal · proteína {_e(round(_num(plan.get('target_protein_g'),150)))}g · agua {_e(round(_num(plan.get('target_water_ml'),2500)))} ml. Restan aprox. {_e(round(remaining))} kcal según lo registrado.</p><form method="post" action="/w/{_e(public_token)}/coach/event"><input type="hidden" name="event_type" value="meal_photo_requested"><input type="text" name="comment" placeholder="Ej: almuerzo estimado 450g, subo foto por WhatsApp"><button>Reportar comida</button></form></section>
    <section class="meal-grid" style="margin-top:14px">{_meal_cards(meals)}</section>
    """
    _record_event(ws, "coach_opened", metadata={"page": "nutrition"})
    return _layout("Fitness Coach - Dieta", body, public_token, "nutrition")


def render_progress(public_token: str) -> str:
    ws = _workspace(public_token)
    progress = _progress(ws["profile_id"])
    latest = _latest_metrics(ws["profile_id"])
    body = f"""
    <section class="card"><span class="eyebrow">Progress</span><h1>Progreso</h1><p class="muted">Resumen de los últimos registros. Mientras más fotos, barcodes, peso y sets registres, mejores serán las recomendaciones.</p></section>
    <section class="grid" style="margin-top:14px"><div class="metric"><span class="eyebrow">Días con comida</span><strong>{_e(progress.get('logged_days') or 0)}</strong></div><div class="metric"><span class="eyebrow">Kcal promedio</span><strong>{_e(round(_num(progress.get('avg_calories'))))}</strong></div><div class="metric"><span class="eyebrow">Sesiones 7d</span><strong>{_e(progress.get('sessions_7d') or 0)}</strong></div></section>
    <section class="card" style="margin-top:14px"><h2>Cierre recomendado</h2><p class="muted">Último peso registrado: {_e((latest or {}).get('weight_kg') or '—')} kg. Para mejorar recuperación, apunta a 7.5–8.5 horas de sueño y deja una ventana sin pantallas/cafeína antes de dormir.</p></section>
    """
    _record_event(ws, "coach_opened", metadata={"page": "progress"})
    return _layout("Fitness Coach - Progreso", body, public_token, "progress")


async def _form(request: Request) -> dict[str, str]:
    data = await request.form()
    return {str(k): str(v) for k, v in data.items() if v is not None}


@router.get("/w/{public_token}/coach", response_class=HTMLResponse)
async def coach_today(public_token: str) -> HTMLResponse:
    return HTMLResponse(render_today(public_token))


@router.get("/w/{public_token}/coach/routine/today", response_class=HTMLResponse)
async def coach_routine_today(public_token: str) -> HTMLResponse:
    return HTMLResponse(render_routine(public_token))


@router.get("/w/{public_token}/coach/nutrition/today", response_class=HTMLResponse)
async def coach_nutrition_today(public_token: str) -> HTMLResponse:
    return HTMLResponse(render_nutrition(public_token))


@router.get("/w/{public_token}/coach/progress", response_class=HTMLResponse)
async def coach_progress(public_token: str) -> HTMLResponse:
    return HTMLResponse(render_progress(public_token))


@router.post("/w/{public_token}/coach/event")
async def coach_event(public_token: str, request: Request) -> RedirectResponse:
    ws = _workspace(public_token)
    form = await _form(request)
    event_type = form.get("event_type") or "coach_event"
    comment = form.get("comment")
    metadata: dict[str, Any] = {}
    if form.get("metadata"):
        try:
            metadata = json.loads(form["metadata"])
        except json.JSONDecodeError:
            metadata = {"raw_metadata": form["metadata"]}
    _record_event(ws, event_type, comment=comment, metadata=metadata)
    return RedirectResponse(url=f"/w/{public_token}/coach", status_code=303)
