#!/usr/bin/env python3
"""Seed a complete software-development company into Hermes Digital Office.

Creates six departments (Product, Design, Engineering, QA, DevOps & Security,
Data) and fifteen employees with role-appropriate toolsets and system prompts,
then runs a smoke test by dispatching one task per department and polling the
office API for the resulting activity.

Usage:
    python scripts/seed_software_company.py                  # create + smoke test
    python scripts/seed_software_company.py --reset          # delete first
    python scripts/seed_software_company.py --no-tasks       # skip smoke tests
    python scripts/seed_software_company.py --base-url http://127.0.0.1:8765

The script uses only the Python standard library so it runs against any office
install without extra dependencies. Idempotent: re-running upserts by name.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:8765"
DEFAULT_MODEL = "gemma4-e2b-hermes"

# ─────────────────────────────────────────────────────────────────────────────
# Org chart definition
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EmpSpec:
    name: str
    role: str
    sprite: str
    hue: int
    toolsets: list[str]
    skills: list[str] = field(default_factory=list)
    system_prompt: str = ""


@dataclass
class DeptSpec:
    name: str
    mission: str
    color: str
    employees: list[EmpSpec]


COMPANY: list[DeptSpec] = [
    DeptSpec(
        name="产品 Product",
        mission="挖掘真实需求，写清楚的 PRD，决定我们要做什么。",
        color="#f59e0b",
        employees=[
            EmpSpec(
                name="李 Mia (PM)",
                role="产品经理 Product Manager",
                sprite="scientist",
                hue=35,
                toolsets=["web", "memory", "todo", "file"],
                skills=["research/research-paper-writing"],
                system_prompt=(
                    "You are a senior product manager. "
                    "Translate fuzzy ideas into crisp PRDs with clear user stories, "
                    "acceptance criteria, and explicit non-goals. Always cite the "
                    "source of any user research."
                ),
            ),
            EmpSpec(
                name="周 Owen (PO)",
                role="产品负责人 Product Owner",
                sprite="wizard",
                hue=20,
                toolsets=["todo", "memory", "delegation"],
                system_prompt=(
                    "You own the backlog. Prioritise ruthlessly using RICE, keep "
                    "stories small, and make sure every sprint has one measurable "
                    "outcome."
                ),
            ),
            EmpSpec(
                name="陈 Bella (BA)",
                role="业务分析师 Business Analyst",
                sprite="analyst",
                hue=45,
                toolsets=["web", "code_execution", "file", "todo"],
                system_prompt=(
                    "You analyse the market and competitors, translate findings "
                    "into opportunity sizing, and back every claim with numbers."
                ),
            ),
        ],
    ),
    DeptSpec(
        name="设计 Design",
        mission="把抽象的产品意图变成用户能一眼看懂的界面。",
        color="#ec4899",
        employees=[
            EmpSpec(
                name="林 Una (UXR)",
                role="用户体验研究员 UX Researcher",
                sprite="tutor",
                hue=320,
                toolsets=["web", "memory", "file", "todo"],
                system_prompt=(
                    "You run lightweight user research: write screener questions, "
                    "synthesise interview notes into JTBD insights, and surface the "
                    "top three user pains."
                ),
            ),
            EmpSpec(
                name="苏 Lin (UI)",
                role="视觉与交互设计 UI Designer",
                sprite="designer",
                hue=300,
                toolsets=["image_gen", "vision", "file", "todo"],
                skills=["creative/popular-web-designs", "creative/pixel-art"],
                system_prompt=(
                    "You craft modern, accessible UI. Reference shadcn/ui patterns, "
                    "ensure WCAG AA contrast, and produce exportable specs."
                ),
            ),
        ],
    ),
    DeptSpec(
        name="工程 Engineering",
        mission="把需求变成可上线的代码。",
        color="#6366f1",
        employees=[
            EmpSpec(
                name="王 Ada (Architect)",
                role="系统架构师 Software Architect",
                sprite="wizard",
                hue=260,
                toolsets=["file", "code_execution", "todo", "memory", "delegation"],
                skills=["software-development/test-driven-development"],
                system_prompt=(
                    "You design systems for the next 18 months: pick the boring "
                    "technology, draw the data flow first, document trade-offs in "
                    "ADR format."
                ),
            ),
            EmpSpec(
                name="张 Ren (Backend)",
                role="后端工程师 Backend Engineer",
                sprite="robot-1",
                hue=220,
                toolsets=["file", "code_execution", "terminal", "todo"],
                skills=["software-development/test-driven-development", "software-development/systematic-debugging"],
                system_prompt=(
                    "You build APIs in Python/FastAPI. Write tests first, keep "
                    "endpoints idempotent, and log structured JSON."
                ),
            ),
            EmpSpec(
                name="孙 Iris (Frontend)",
                role="前端工程师 Frontend Engineer",
                sprite="cat",
                hue=180,
                toolsets=["file", "code_execution", "terminal", "todo"],
                skills=["software-development/test-driven-development"],
                system_prompt=(
                    "You ship React + TypeScript with Tailwind. Components must "
                    "be accessible, lazy-loaded where useful, and covered by "
                    "Vitest."
                ),
            ),
            EmpSpec(
                name="吴 Theo (Mobile)",
                role="移动端工程师 Mobile Engineer",
                sprite="fox",
                hue=20,
                toolsets=["file", "code_execution", "terminal", "todo"],
                system_prompt=(
                    "You build cross-platform mobile features (React Native). "
                    "Optimise for cold-start and battery; profile before tuning."
                ),
            ),
        ],
    ),
    DeptSpec(
        name="质量 QA",
        mission="在用户之前发现 bug。",
        color="#14b8a6",
        employees=[
            EmpSpec(
                name="郑 Lily (QA)",
                role="质量工程师 QA Engineer",
                sprite="panda",
                hue=160,
                toolsets=["file", "todo", "memory", "code_execution"],
                skills=["software-development/systematic-debugging"],
                system_prompt=(
                    "You write thorough test plans: positive, negative, edge, and "
                    "exploratory. Reproduce every bug before filing it."
                ),
            ),
            EmpSpec(
                name="冯 Max (TestAuto)",
                role="测试自动化工程师 Test Automation",
                sprite="robot-1",
                hue=140,
                toolsets=["file", "terminal", "code_execution", "todo"],
                system_prompt=(
                    "You build and maintain Playwright + pytest suites. Tests are "
                    "deterministic, hermetic, and fast (<5min full run)."
                ),
            ),
        ],
    ),
    DeptSpec(
        name="运维与安全 DevOps & Security",
        mission="让系统又快又稳又安全。",
        color="#ef4444",
        employees=[
            EmpSpec(
                name="高 Kai (SRE)",
                role="站点可靠性工程师 SRE",
                sprite="robot-1",
                hue=0,
                toolsets=["terminal", "file", "code_execution", "todo", "memory"],
                system_prompt=(
                    "You run production. SLOs first, alerts that page only when "
                    "humans are needed, runbooks linked from every dashboard."
                ),
            ),
            EmpSpec(
                name="何 Nova (Security)",
                role="安全工程师 Security Engineer",
                sprite="wizard",
                hue=350,
                toolsets=["file", "web", "code_execution", "todo"],
                system_prompt=(
                    "You audit code and infra: threat-model new features, check "
                    "deps for CVEs, verify least-privilege on every IAM grant."
                ),
            ),
        ],
    ),
    DeptSpec(
        name="数据 Data",
        mission="把数据变成决策。",
        color="#0ea5e9",
        employees=[
            EmpSpec(
                name="罗 Eva (Data)",
                role="数据分析师 Data Analyst",
                sprite="analyst",
                hue=200,
                toolsets=["code_execution", "file", "todo"],
                skills=["data-science/jupyter-live-kernel"],
                system_prompt=(
                    "You answer business questions with SQL + Python. Show your "
                    "method, label every chart, and surface the so-what."
                ),
            ),
            EmpSpec(
                name="韩 Leo (ML)",
                role="机器学习工程师 ML Engineer",
                sprite="scientist",
                hue=240,
                toolsets=["code_execution", "file", "todo", "memory"],
                system_prompt=(
                    "You ship ML features: small models first, evaluate on holdout, "
                    "monitor drift in production, never claim accuracy without a "
                    "confidence interval."
                ),
            ),
        ],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Tiny HTTP client (stdlib only)
# ─────────────────────────────────────────────────────────────────────────────


class Client:
    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        # Avoid any system proxy snagging loopback traffic on Windows.
        os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

    def _req(self, method: str, path: str, body: Any | None = None) -> Any:
        url = self.base + path
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} -> {exc.code}\n{payload}") from None

    def get(self, path: str) -> Any:
        return self._req("GET", path)

    def post(self, path: str, body: dict) -> Any:
        return self._req("POST", path, body)

    def delete(self, path: str) -> Any:
        return self._req("DELETE", path)


# ─────────────────────────────────────────────────────────────────────────────
# Provisioning
# ─────────────────────────────────────────────────────────────────────────────


def reset(client: Client) -> None:
    print("[reset] deleting all existing departments…")
    for d in client.get("/api/departments"):
        try:
            client.delete(f"/api/departments/{d['id']}")
            print(f"  - deleted {d['name']!r} ({d['id']})")
        except Exception as exc:
            print(f"  ! failed deleting {d['id']}: {exc}")


def upsert_company(client: Client, model: str, runtime: str) -> dict[str, Any]:
    existing_depts = {d["name"]: d for d in client.get("/api/departments")}
    existing_emps_by_dept: dict[str, dict[str, dict]] = {}

    summary = {"departments": [], "employees": []}

    for dept in COMPANY:
        if dept.name in existing_depts:
            d = existing_depts[dept.name]
            print(f"[dept] reuse {dept.name!r} ({d['id']})")
        else:
            d = client.post(
                "/api/departments",
                {
                    "name": dept.name,
                    "mission": dept.mission,
                    "color": dept.color,
                    "runtime_default": runtime,
                },
            )
            print(f"[dept] created {dept.name!r} ({d['id']})")
        summary["departments"].append(d)
        dept_id = d["id"]

        if dept_id not in existing_emps_by_dept:
            existing_emps_by_dept[dept_id] = {
                e["name"]: e for e in client.get(f"/api/employees?dept_id={dept_id}")
            }

        for emp in dept.employees:
            if emp.name in existing_emps_by_dept[dept_id]:
                e = existing_emps_by_dept[dept_id][emp.name]
                print(f"  [emp] reuse {emp.name!r} ({e['id']})")
            else:
                e = client.post(
                    "/api/employees",
                    {
                        "department_id": dept_id,
                        "name": emp.name,
                        "role": emp.role,
                        "avatar": {"sprite_id": emp.sprite, "hue": emp.hue},
                        "model": model,
                        "enabled_toolsets": emp.toolsets,
                        "skills": emp.skills,
                        "system_prompt": emp.system_prompt,
                        "runtime": runtime,
                    },
                )
                print(f"  [emp] hired {emp.name!r} as {emp.role} ({e['id']})")
            summary["employees"].append(e)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

DEPT_TASKS: dict[str, str] = {
    "产品 Product": "为新功能『AI 周报自动生成』写一份 1 页 PRD：用户故事 + 验收标准 + 非目标。",
    "设计 Design": "为 PRD『AI 周报自动生成』产出主流程低保真线框：设置页 → 选周期 → 生成 → 编辑导出。",
    "工程 Engineering": "把『AI 周报自动生成』拆成后端任务清单：API 表、数据模型、依赖、估时。",
    "质量 QA": "给『AI 周报自动生成』写测试计划：正向、反向、边界、性能基线。",
    "运维与安全 DevOps & Security": "评估『AI 周报自动生成』上线所需基础设施 + 列出 Top 5 安全风险与缓解。",
    "数据 Data": "提议『AI 周报自动生成』的成功指标（北极星 + 输入指标），给出查询草稿。",
}


def fire_smoke_tasks(client: Client, depts: list[dict]) -> list[dict]:
    issued = []
    for d in depts:
        text = DEPT_TASKS.get(d["name"])
        if not text:
            continue
        if not d["employee_ids"]:
            continue
        try:
            t = client.post(
                "/api/tasks",
                {"department_id": d["id"], "text": text},
            )
            print(f"[task] -> {d['name']:<30} {t['id']}")
            issued.append(t)
        except Exception as exc:
            print(f"[task] FAILED {d['name']!r}: {exc}")
    return issued


def wait_and_report(client: Client, issued: list[dict], wait_s: int = 12) -> None:
    print(f"\n[report] waiting {wait_s}s for simulated runtimes to finish…")
    time.sleep(wait_s)

    tasks = client.get("/api/tasks?limit=100")
    by_id = {t["id"]: t for t in tasks}

    by_status = {"queued": 0, "running": 0, "done": 0, "failed": 0}
    for tid in (t["id"] for t in issued):
        last = by_id.get(tid, {})
        s = last.get("status", "missing")
        by_status[s] = by_status.get(s, 0) + 1

    health = client.get("/api/health")
    print()
    print("=" * 72)
    print(" Hermes Digital Office — Software Company smoke report")
    print("=" * 72)
    print(f" Office root      : {health['office_root']}")
    print(f" Departments      : {health['departments']}")
    print(f" Employees        : {health['employees']}")
    print(f" Runtime default  : {health['runtime_default']}")
    print(f" Tasks dispatched : {len(issued)}")
    print(f" Task statuses    : {by_status}")
    print()
    print(" Per-department roster:")
    for d in client.get("/api/departments"):
        emps = client.get(f"/api/employees?dept_id={d['id']}")
        print(f"   • {d['name']:<32} ({len(emps)} 人)  →  color {d['color']}")
        for e in emps:
            print(f"       - {e['name']:<24} {e['role']}")
    print()
    print(" Recent task outcomes:")
    for t in issued:
        last = by_id.get(t["id"], {})
        text = (last.get("text") or "")[:60].replace("\n", " ")
        summary = (last.get("result_summary") or "").replace("\n", " ")[:80]
        print(
            f"   {last.get('status','?'):<8} {last.get('id','?'):<22} "
            f"{text:<62} {summary}"
        )
    print("=" * 72)
    print(" Open the office UI and watch the avatars walk around:")
    print("   http://127.0.0.1:8765/")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=DEFAULT_BASE, help="Office API base URL")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Default LLM model id")
    p.add_argument(
        "--runtime",
        default="simulated",
        choices=("simulated", "hermes"),
        help="Runtime backend per employee (default: simulated; safe + fast)",
    )
    p.add_argument("--reset", action="store_true", help="Delete existing departments first")
    p.add_argument("--no-tasks", action="store_true", help="Skip the smoke test tasks")
    args = p.parse_args()

    client = Client(args.base_url)
    try:
        h = client.get("/api/health")
    except Exception as exc:
        print(f"[fatal] cannot reach office at {args.base_url}: {exc}")
        print("        Start it with:  hermes office     (or: python -m hermes_office)")
        return 2
    print(f"[ok] office {h['version']} reachable, "
          f"{h['departments']} dept / {h['employees']} emp present")

    if args.reset:
        reset(client)

    summary = upsert_company(client, model=args.model, runtime=args.runtime)
    print(f"\n[done] organisation: {len(summary['departments'])} departments, "
          f"{len(summary['employees'])} employees\n")

    issued: list[dict] = []
    if not args.no_tasks:
        depts = client.get("/api/departments")
        issued = fire_smoke_tasks(client, depts)

    wait_and_report(client, issued)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
