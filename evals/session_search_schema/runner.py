"""Live A/B runner: session_search schema variants, extracted from git refs.

For each arm, ``tools/session_search_tool.py`` is extracted from a git ref
(``git show <ref>:tools/session_search_tool.py``) and imported as its own
module. A minimal agent loop (OpenRouter, tools API) then runs the shared
task battery against a freshly seeded temp session DB. The ONLY variable
between arms is that module — schema text, response hints, tool behavior.

Usage:
  python3 evals/session_search_schema/runner.py \
      --base origin/main --cand HEAD \
      --model qwen/qwen3-coder-30b-a3b-instruct --reps 3

  # limit to one task
  ... --tasks t2_scroll

Results append to results/<label>/<model-slug>.jsonl (resume-safe: completed
(task, arm, rep) cells are skipped on re-run). Summarize with report.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO_ROOT))

from tasks import SYSTEM, TASKS  # noqa: E402

ALLOWED_KEYS = {
    "query", "role_filter", "limit", "session_id", "around_message_id",
    "window", "sort", "profile", "detail",
}


def _load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY not found (env or ~/.hermes/.env)")


def extract_arm(ref: str, workdir: Path, name: str) -> Path:
    """Extract tools/session_search_tool.py from a git ref."""
    out = subprocess.run(
        ["git", "show", f"{ref}:tools/session_search_tool.py"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"git show {ref}: {out.stderr.strip()}")
    path = workdir / f"ss_arm_{name}.py"
    path.write_text(out.stdout)
    return path


def load_arm(path: Path, name: str, work_db_path: Path):
    """Import an arm module and make profile resolution hermetic."""
    from hermes_state import SessionDB

    spec = importlib.util.spec_from_file_location(f"ss_arm_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ss_arm_{name}"] = mod
    spec.loader.exec_module(mod)

    def _fake_resolve_profile_db(profile):
        if profile is None or not str(profile).strip():
            return None
        if str(profile).strip().lower() == "work":
            return SessionDB(db_path=work_db_path, read_only=True)
        raise ValueError(f"profile '{profile}' does not exist")

    def _fake_locate_session_db(session_id):
        try:
            db = SessionDB(db_path=work_db_path, read_only=True)
            row = db._conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row:
                return db, "work"
            db.close()
        except Exception:
            pass
        return None, None

    mod._resolve_profile_db = _fake_resolve_profile_db
    mod._locate_session_db = _fake_locate_session_db
    return mod


def build_tools(arm_mod):
    s = arm_mod.SESSION_SEARCH_SCHEMA
    return [{
        "type": "function",
        "function": {
            "name": s["name"],
            "description": s["description"],
            "parameters": s["parameters"],
        },
    }]


def exec_tool(arm_mod, args, main_db_path: Path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=main_db_path)
    try:
        kwargs, bad = {}, []
        for k, v in args.items():
            if k in ALLOWED_KEYS:
                kwargs[k] = v
            else:
                bad.append(k)
        if bad:
            return json.dumps({
                "success": False,
                "error": f"unexpected parameter(s): {', '.join(bad)}",
            }), True
        return arm_mod.session_search(db=db, **kwargs), False
    except Exception as e:  # noqa: BLE001 — tool errors go back to the model
        return json.dumps({
            "success": False, "error": f"{type(e).__name__}: {e}",
        }), True
    finally:
        try:
            db.close()
        except Exception:
            pass


def run_one(client, model, arm_name, arm_mod, task_id, prompt, oracle,
            main_db_path: Path, max_iters: int = 8):
    tools = build_tools(arm_mod)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
    calls, bad_calls = [], 0
    first_prompt_tokens, total_tokens = None, 0
    final = ""
    t0 = time.time()
    for _ in range(max_iters):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools,
            temperature=0.2, max_tokens=2000,
        )
        u = getattr(resp, "usage", None)
        if u:
            if first_prompt_tokens is None:
                first_prompt_tokens = u.prompt_tokens
            total_tokens += (u.total_tokens or 0)
        msg = resp.choices[0].message
        tcs = msg.tool_calls or []
        if not tcs:
            final = msg.content or ""
            break
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in tcs
            ],
        })
        for tc in tcs:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args, bad_calls = {}, bad_calls + 1
            calls.append(args)
            if tc.function.name != "session_search":
                out, was_err = json.dumps(
                    {"success": False, "error": "unknown tool"}), True
            else:
                out, was_err = exec_tool(arm_mod, args, main_db_path)
            if was_err:
                bad_calls += 1
            if len(out) > 30000:
                out = out[:30000] + "...[truncated]"
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": out})
    return {
        "task": task_id, "arm": arm_name, "model": model,
        "ok": bool(oracle(final)) if final else False,
        "n_tool_calls": len(calls), "bad_calls": bad_calls,
        "first_prompt_tokens": first_prompt_tokens,
        "total_tokens": total_tokens,
        "wall_s": round(time.time() - t0, 1),
        "calls": calls, "final": final[:2000],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="git ref for the baseline arm")
    ap.add_argument("--cand", required=True, help="git ref for the candidate arm")
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--label", default="ab")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=_load_api_key())

    with tempfile.TemporaryDirectory(prefix="ss_abeval_") as td:
        tdir = Path(td)
        from fixtures import seed
        dbdir = tdir / "dbs"
        seed(dbdir)
        main_db = dbdir / "state.db"
        work_db = dbdir / "state_work.db"

        arms = {
            "base": load_arm(extract_arm(args.base, tdir, "base"), "base", work_db),
            "cand": load_arm(extract_arm(args.cand, tdir, "cand"), "cand", work_db),
        }

        outdir = EVAL_DIR / "results" / args.label
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / (re.sub(r"[^\w.-]", "_", args.model) + ".jsonl")
        done = set()
        if outpath.exists():
            for line in outpath.read_text().splitlines():
                try:
                    r = json.loads(line)
                    done.add((r["task"], r["arm"], r["rep"]))
                except Exception:
                    pass

        with open(outpath, "a", encoding="utf-8") as f:
            for task_id, (prompt, oracle, _note) in TASKS.items():
                if args.tasks and task_id not in args.tasks:
                    continue
                for rep in range(args.reps):
                    for arm_name, arm_mod in arms.items():
                        if (task_id, arm_name, rep) in done:
                            continue
                        for attempt in range(3):
                            try:
                                r = run_one(client, args.model, arm_name,
                                            arm_mod, task_id, prompt, oracle,
                                            main_db)
                                # Provider noise: zero tool calls AND empty
                                # final → one retry, identical on both arms.
                                if (not r["final"].strip()
                                        and r["n_tool_calls"] == 0
                                        and attempt < 2):
                                    print(f"NOISE-RETRY {task_id} {arm_name} "
                                          f"rep{rep}")
                                    continue
                                r["rep"] = rep
                                r["base_ref"] = args.base
                                r["cand_ref"] = args.cand
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                                f.flush()
                                print(f"{task_id} {arm_name} rep{rep}: "
                                      f"ok={r['ok']} calls={r['n_tool_calls']} "
                                      f"bad={r['bad_calls']} "
                                      f"ptok={r['first_prompt_tokens']}")
                                break
                            except Exception as e:  # noqa: BLE001
                                print(f"RETRY {task_id} {arm_name} rep{rep}: {e}")
                                traceback.print_exc()
                                time.sleep(5 * (attempt + 1))
        print("done ->", outpath)


if __name__ == "__main__":
    main()
