#!/usr/bin/env python3
"""Publish static Fitness Coach Surface pages into Zeus public delivery sandbox."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hermes_cli import fitness_coach_surface as surface

EVENT_JS_TEMPLATE = """
<script>
(() => {{
  const token = {token_json};
  const deliverableId = {deliverable_json};
  const allowed = new Set(['routine_started','set_completed','workout_finished','meal_photo_requested','meal_logged','barcode_scanned','checkin_submitted','sleep_plan_acknowledged']);
  document.querySelectorAll('form').forEach(form => {{
    form.addEventListener('submit', async (event) => {{
      const fd = new FormData(form);
      const eventType = fd.get('event_type') || form.dataset.eventType;
      if (!eventType || !allowed.has(String(eventType))) return;
      event.preventDefault();
      let metadata = {{ page: location.pathname }};
      try {{ if (fd.get('metadata')) metadata = {{...metadata, ...JSON.parse(fd.get('metadata'))}}; }} catch (_) {{ metadata.raw_metadata = fd.get('metadata'); }}
      const button = form.querySelector('button');
      const original = button ? button.textContent : '';
      if (button) button.textContent = 'Registrando…';
      try {{
        const res = await fetch('/api/events', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{ event_type: eventType, deliverable_id: deliverableId, token, actor_type: 'user', comment: fd.get('comment') || null, metadata }})
        }});
        if (!res.ok) throw new Error('event rejected');
        if (button) button.textContent = 'Registrado ✓';
      }} catch (err) {{
        localStorage.setItem('coach-event-' + Date.now(), JSON.stringify({{eventType, comment: fd.get('comment') || null, metadata}}));
        if (button) button.textContent = 'Guardado local ✓';
      }}
      setTimeout(() => {{ if (button) button.textContent = original || 'Enviar'; }}, 1800);
    }});
  }});
  const timerMount = document.querySelector('[data-coach-timer]');
  if (timerMount) {{
    let seconds = 75, running = false, id = null;
    const draw = () => timerMount.textContent = String(Math.floor(seconds/60)).padStart(2,'0') + ':' + String(seconds%60).padStart(2,'0');
    window.coachTimerStart = () => {{ if (running) return; running = true; id = setInterval(() => {{ seconds = Math.max(0, seconds - 1); draw(); if (!seconds) {{ clearInterval(id); running=false; }} }}, 1000); }};
    window.coachTimerReset = () => {{ seconds = 75; running=false; if (id) clearInterval(id); draw(); }};
    draw();
  }}
}})();
</script>
"""


def inject_static_interactions(html: str, token: str, deliverable_id: str) -> str:
    if "Routine Player" in html and "data-coach-timer" not in html:
        html = html.replace(
            '</section>\n    <section class="exercise-list"',
            '<div class="card" style="margin-top:14px"><span class="eyebrow">Timer descanso</span><h2 data-coach-timer>01:15</h2><button onclick="coachTimerStart()">Iniciar timer</button> <button class="secondary" onclick="coachTimerReset()">Reset</button></div></section>\n    <section class="exercise-list"',
        )
    js = EVENT_JS_TEMPLATE.format(token_json=json.dumps(token), deliverable_json=json.dumps(deliverable_id))
    return html.replace("</body></html>", js + "</body></html>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-id", default="jean-garcia")
    parser.add_argument("--public-root", default="/home/jean/zeus-runtime/delivery-sandbox/public")
    parser.add_argument("--deliverable-id", default="coach-surface-jean")
    args = parser.parse_args()

    workspace = surface.create_or_get_coach_workspace(args.profile_id)
    token = surface.validate_public_token(workspace["public_token"])
    public_root = Path(args.public_root).resolve()
    root = public_root / "w" / token
    if not root.resolve().is_relative_to(public_root / "w"):
        raise SystemExit(f"Unsafe coach public token path: {token!r}")
    pages = {
        "coach/index.html": surface.render_today(token),
        "coach/routine/today/index.html": surface.render_routine(token),
        "coach/nutrition/today/index.html": surface.render_nutrition(token),
        "coach/progress/index.html": surface.render_progress(token),
    }
    for rel, html in pages.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(inject_static_interactions(html, token, args.deliverable_id), encoding="utf-8")
    (root / "workspace.json").write_text(
        json.dumps({"deliverable_id": args.deliverable_id, "source_id": args.deliverable_id, "profile_id": args.profile_id, "token": token, "kind": "fitness_coach_surface"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    comments = root / "comments.json"
    if not comments.exists():
        comments.write_text("[]\n", encoding="utf-8")
    print(json.dumps({"ok": True, "token": token, "url": f"https://zeus-sandbox.kidu.app/w/{token}/coach", "root": str(root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
