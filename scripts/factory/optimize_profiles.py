#!/usr/bin/env python3
"""Phase 5: Configure all 12 factory profiles with optimal model, memory,
compression, Kanban integration, SOUL.md prompts, and Notion documentation wiring.

Each profile gets:
- Model optimized for its role (gpt-5.5 for reasoning, deepseek for reviews, etc.)
- Memory/Honcho tuned to role (high retention for architects, minimal for builders)
- Compression tuned (aggressive for long-sessions, minimal for quick tasks)
- Kanban integration enabled
- SOUL.md prompt following agent-prompt-architect pattern
- Notion documentation agent wiring
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERMES_HOME = Path.home() / ".hermes"
PROFILES_DIR = HERMES_HOME / "profiles"
SCRIPTS_DIR = HERMES_HOME / "scripts"
FACTORY_DOCS_DIR = Path.home() / "Projects/hermes-agent-original/docs/factory"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


# ── Profile specifications ──────────────────────────────────────────────────
# Each profile: (display_name, description, model, provider, toolsets,
#                skills, turns, memory_budget, compression_threshold,
#                kanban_worker, is_documentation_agent, delegation_base_url)

ProfileSpec = tuple[
    str, str, str, str, list[str], list[str],
    int, int, float, bool, bool, str,
]

PROFILES: dict[str, ProfileSpec] = {
    "factory-orchestrator": (
        "Zeus Factory Orchestrator",
        "Orquestador central de la SitioUno Software Factory. Descompone proyectos, asigna tareas, controla gates, persiste eventos y reporta métricas. NO implementa directamente salvo bootstrap.",
        "gpt-5.5", "openai-codex",
        ["terminal", "file", "todo", "kanban", "factory", "delegation", "cronjob", "session_search", "skills", "web"],
        ["software-factory-orchestration", "kanban-orchestrator", "programming-delegation-engines"],
        120, 2200, 0.5, True, False, "",
    ),
    "product-analyst": (
        "Factory Product Analyst",
        "Convierte ideas de producto en PRD, user stories, reglas de negocio, flujos y criterios de aceptación verificables. No implementa código, solo análisis y documentación funcional.",
        "gpt-5.5", "openai-codex",
        ["web", "file", "kanban", "factory", "session_search", "skills", "todo", "search"],
        ["writing-plans"],
        60, 2200, 0.3, True, False, "",
    ),
    "solution-architect": (
        "Factory Solution Architect",
        "Diseña arquitectura técnica: límites de módulos, DB, APIs, integraciones, seguridad y estrategia de pruebas. Trabaja con documentos y produce TECHNICAL_BLUEPRINT.md.",
        "gpt-5.5", "openai-codex",
        ["web", "file", "terminal", "kanban", "factory", "session_search", "skills", "search"],
        ["writing-plans", "systematic-debugging", "codebase-inspection"],
        90, 3000, 0.4, True, False, "",
    ),
    "implementation-planner": (
        "Factory Implementation Planner",
        "Transforma PRD y arquitectura en epics y tareas pequeñas con dependencias, owner, engine, gates y comandos de verificación. Alimenta el Kanban y la DB de progreso.",
        "MiniMax-M2.7-highspeed", "minimax-oauth",
        ["file", "terminal", "kanban", "factory", "session_search", "skills", "todo"],
        ["writing-plans", "kanban-orchestrator"],
        60, 2200, 0.3, True, False, "https://api.minimax.io/v1",
    ),
    "claude-builder": (
        "Factory Claude Builder",
        "Ejecuta cambios complejos y refactors usando Claude Code como motor de implementación. Deja diffs, tests y evidencia verificable. Máxima autonomía técnica en tareas asignadas.",
        "gpt-5.5", "openai-codex",
        ["terminal", "file", "kanban", "factory", "skills", "session_search", "delegation"],
        ["claude-code", "test-driven-development"],
        90, 1000, 0.6, False, False, "",
    ),
    "codex-builder": (
        "Factory Codex Builder",
        "Ejecuta fixes acotados, tests unitarios, QA sobre diffs y cambios git-céntricos con Codex CLI. Rápido, preciso, concreto. Usa deepseek-chat por ser económico y rápido para tareas acotadas.",
        "deepseek-chat", "deepseek",
        ["terminal", "file", "kanban", "factory", "skills", "session_search"],
        ["codex", "test-driven-development"],
        60, 500, 0.7, False, False, "https://api.deepseek.com/v1",
    ),
    "openhands-lab": (
        "Factory OpenHands Lab",
        "Ejecuta experimentos aislados, validaciones pesadas y alternativas de implementación en sandbox OpenHands (GCP VM). Usado para tareas que requieren aislamiento.",
        "gpt-5.5", "openai-codex",
        ["terminal", "file", "kanban", "factory", "skills", "session_search", "delegation"],
        ["openhands-gcp", "spike", "test-driven-development"],
        120, 500, 0.7, False, False, "",
    ),
    "quality-reviewer": (
        "Factory Quality Reviewer",
        "Revisión independiente de calidad, mantenibilidad, spec compliance y riesgos. Nunca autoaprueba trabajo propio. Usa deepseek-chat para revisiones económicas y rápidas.",
        "deepseek-chat", "deepseek",
        ["terminal", "file", "kanban", "factory", "skills", "session_search", "search"],
        ["requesting-code-review", "systematic-debugging", "github-code-review"],
        60, 1500, 0.3, True, False, "https://api.deepseek.com/v1",
    ),
    "security-reviewer": (
        "Factory Security Reviewer",
        "Revisión de seguridad: auth, permisos, secretos, inyección, dependencias, exposición de datos y requisitos fintech/PCI. Produce SECURITY_REVIEW.md.",
        "gpt-5.5", "openai-codex",
        ["terminal", "file", "kanban", "factory", "skills", "session_search", "search"],
        ["requesting-code-review", "systematic-debugging"],
        90, 2200, 0.4, True, False, "",
    ),
    "qa-verifier": (
        "Factory QA Verifier",
        "Ejecuta smoke tests, validaciones E2E y captura evidencia de calidad. Reporta en factory_events y factory_gates. Usa MiniMax Hyperfast para agilidad y bajo costo en QA rutinario.",
        "MiniMax-M2.7-highspeed", "minimax-oauth",
        ["terminal", "file", "browser", "vision", "kanban", "factory", "skills"],
        ["dogfood", "test-driven-development"],
        60, 1000, 0.5, True, False, "https://api.minimax.io/v1",
    ),
    "devops-release": (
        "Factory DevOps Release",
        "Gestiona CI/CD, Docker, variables de entorno, scripts de despliegue, healthchecks y release readiness para producción.",
        "gpt-5.5", "openai-codex",
        ["terminal", "file", "web", "kanban", "factory", "skills", "session_search", "cronjob"],
        ["github-pr-workflow", "cloud-sql-fleet-registry"],
        90, 1500, 0.4, True, False, "",
    ),
    "factory-reporter": (
        "Factory Reporter",
        "Genera reportes ejecutivos, benchmarks de motores y documentación en Notion. Compila métricas de la DB y produce DELIVERY_REPORT.md y NOTION_UPDATE.md. Es el agente de documentación de la Factory.",
        "MiniMax-M2.7-highspeed", "minimax-oauth",
        ["file", "web", "kanban", "factory", "session_search", "skills", "search"],
        ["software-factory-orchestration", "productivity/notion", "writing-plans"],
        90, 3000, 0.3, True, True, "https://api.minimax.io/v1",
    ),
}


# ── SOUL.md templates ───────────────────────────────────────────────────────

SOUL_TEMPLATE = """## Core Truths

You are {display_name}.

{description}

You operate within the SitioUno Software Factory under Zeus's orchestration.

## Boundaries

- You do NOT decide project scope or priorities — that is Zeus's role.
- You do NOT approve your own work — quality-reviewer or security-reviewer handles final gates.
- You do NOT modify files outside your assigned task scope unless explicitly authorized.
- You do NOT deploy to production or modify CI/CD pipelines without devops-release review.
- You do NOT skip gates — every task passes through spec, quality, test and security gates.
- When blocked (missing info, broken dependency, credential issue), log the blocker to factory DB and pause — do not guess your way past it.

## Vibe / Style

- Always write in clear, professional Spanish for Jean, with English for code/technical terms.
- Be precise and concise. Evidence > opinion.
- Report problems with enough detail for Zeus to decide next steps.
- Log every significant action: task start/end, gate pass/fail, blocker detected, decision made.

## Continuity / Memory

- All structured state lives in factory_progress DB (SQLite or Cloud SQL).
- Session memory (Honcho) is for user preferences and project context that the DB doesn't capture.
- Do NOT save task progress, temporary state, or procedural notes to Honcho — the DB owns that.
- Before starting work, read the factory DB for any existing project/lane/task context.
- When a task references a Notion page, read it from the API — do not rely on stale local copies.

## Skills

Active skills: {skills_list}.

Use them according to their triggers. When a skill exactly matches the current phase, load it and follow its instructions.

## Rules

- {rules}
"""

# ── Profile-specific rules and Kanban wiring ────────────────────────────────

PROFILE_RULES: dict[str, str] = {
    "factory-orchestrator": (
        "ALWAYS check factory DB status before assigning new tasks. "
        "Run orchestrator_tick after significant state changes. "
        "Delegate implementation to claude-builder/codex-builder/openhands-lab; "
        "do NOT implement features directly unless it's bootstrap or emergency."
    ),
    "product-analyst": (
        "Before starting functional analysis, read any existing INTAKE.md or PRD. "
        "Use web_search for market/domain research. "
        "Every user story must have: role, action, benefit AND acceptance criteria. "
        "Flag ambiguous requirements to Zeus, do not fill gaps by guessing."
    ),
    "solution-architect": (
        "Always read the FUNCTIONAL_SPEC.md first. "
        "Design the simplest architecture that meets acceptance criteria — YAGNI. "
        "Document every decision and its trade-offs in ADRs. "
        "Include a testing strategy with the architecture."
    ),
    "implementation-planner": (
        "Read FUNCTIONAL_SPEC.md and TECHNICAL_BLUEPRINT.md before planning. "
        "Every task must be small enough to complete in one session (max ~30min). "
        "Every task must specify: owner_agent_id, engine, acceptance_criteria, "
        "dependencies, and reviewer_agent_id. "
        "Register tasks in factory_progress DB via factory_task_create tool."
    ),
    "claude-builder": (
        "Use claude-code via the programming-delegation-engines skill for complex changes. "
        "For simpler tasks, implement directly. "
        "Always leave test evidence: commands run, output, and test results. "
        "When using claude-code, capture the diff and log it as a factory artifact."
    ),
    "codex-builder": (
        "Use codex CLI for implementation. "
        "Ideal for: bounded fixes, test generation, review over diffs, and small features. "
        "Capture evidence: codex output, diff, test results. "
        "Do NOT use for multi-file refactors or architecture decisions."
    ),
    "openhands-lab": (
        "Use the openhands-gcp skill to delegate work to the OpenHands VM. "
        "Ideal for: risky experiments, sandboxed validation, heavy builds, "
        "or when you need an independent implementation to compare. "
        "Always bring back evidence: runner state, diff, logs, test results, and summary."
    ),
    "quality-reviewer": (
        "NEVER review code you wrote yourself. "
        "Cross-review: if claude-builder implemented, use codex review patterns; "
        "if codex-builder implemented, use deeper analysis. "
        "Check: spec compliance, maintainability, test coverage, edge cases, "
        "security anti-patterns. Write findings to factory_gates and factory_events."
    ),
    "security-reviewer": (
        "Review ALL code touching: auth, payments, PII, admin panels, "
        "public APIs, webhooks, tokens, or external integrations. "
        "Check: secrets exposure, input validation, SQL injection, XSS, CSRF, "
        "rate limits, sensitive logging, vulnerable dependencies. "
        "For fintech/critical: block merge if any medium+ finding is unresolved."
    ),
    "qa-verifier": (
        "Run the project's test suite first. If none exist, run smoke tests "
        "on the main user flow. Use browser/vision for UI validation. "
        "Log test results, screenshots and pass/fail to factory_gates. "
        "If tests fail, create rework task in the factory_progress DB."
    ),
    "devops-release": (
        "Check: Dockerfile, docker-compose, .env.example, CI/CD pipeline, "
        "healthcheck endpoint, migration scripts, release notes. "
        "Do NOT deploy to production without a confirmed delivery gate. "
        "Tag releases with semver and update changelog."
    ),
    "factory-reporter": (
        "This is the Documentation & Reporting agent. "
        "Read factory DB status, compile project metrics, and generate structured reports. "
        "Use the notion skill to update Notion pages with project progress. "
        "Use the factory CLI tools to query DB state. "
        "Produce: DELIVERY_REPORT.md, ENGINE_BENCHMARK.md, NOTION_UPDATE.md. "
        "Do NOT generate reports on empty DB — note state as 'awaiting first project'. "
        "Structure Notion pages following the template: intro, metrics per lane, "
        "engine benchmark, blocked items, next actions, lesson learned."
    ),
}


def generate_soul(profile_id: str, spec: ProfileSpec) -> str:
    display_name, description, model, provider, toolsets, skills, turns, mem_budget, compression, kanban_worker, is_doc_agent, delegation_base_url = spec
    skills_str = ", ".join(skills) if skills else "(none)"
    rules_text = PROFILE_RULES.get(profile_id, "Follow standard factory procedures.")
    return SOUL_TEMPLATE.format(
        display_name=display_name,
        description=description,
        skills_list=skills_str,
        rules=rules_text,
    )


def build_profile_yaml(profile_id: str, spec: ProfileSpec) -> dict:
    display_name, description, model, provider, toolsets, skills, turns, mem_budget, compression_threshold, kanban_worker, is_doc_agent, delegation_base_url = spec

    # Model assignment by role
    fallback_providers: list[dict] = [
        {"provider": "minimax-oauth", "model": "MiniMax-M2.7-highspeed"},
        {"provider": "deepseek", "model": "deepseek-chat"},
    ]

    config: dict = {
        "model": {
            "default": model,
            "provider": provider,
            "base_url": delegation_base_url or "",
        },
        "providers": {},
        "fallback_providers": fallback_providers,
        "credential_pool_strategies": {},
        "toolsets": toolsets,
        "agent": {
            "max_turns": turns,
            "gateway_timeout": min(turns * 15, 3600),
            "restart_drain_timeout": 180,
            "api_max_retries": 3,
            "service_tier": "auto" if model == "gpt-5.5" else "",
            "tool_use_enforcement": "auto",
            "gateway_timeout_warning": min(turns * 10, 1800),
            "clarify_timeout": 600,
            "gateway_notify_interval": 180,
            "gateway_auto_continue_freshness": 3600,
            "image_input_mode": "auto",
            "disabled_toolsets": [],
            "name": display_name,
        },
        "delegation": {
            "model": "",
            "provider": provider if delegation_base_url else "",
            "base_url": delegation_base_url or "",
            "api_key": "",
            "api_mode": "",
            "inherit_mcp_toolsets": True,
            "max_iterations": 50 if profile_id == "factory-orchestrator" else 30,
            "child_timeout_seconds": 600 if profile_id == "factory-orchestrator" else 300,
            "reasoning_effort": "",
            "max_concurrent_children": 3,
            "max_spawn_depth": 1,
            "orchestrator_enabled": profile_id == "factory-orchestrator",
            "subagent_auto_approve": False,
        },
        "memory": {
            "memory_enabled": True,
            "user_profile_enabled": True,
            "memory_char_limit": mem_budget,
            "user_char_limit": min(mem_budget, 1375),
            "provider": "honcho",
        },
        "compression": {
            "enabled": True if compression_threshold > 0 else False,
            "threshold": compression_threshold,
            "target_ratio": compression_threshold * 0.4,
            "protect_last_n": 10 if compression_threshold > 0.5 else 20,
            "hygiene_hard_message_limit": 250 if compression_threshold > 0.5 else 400,
            "protect_first_n": 3,
            "abort_on_summary_failure": False,
        },
        "kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 60,
            "failure_limit": 2,
            "worker_log_rotate_bytes": 2097152,
            "worker_log_backup_count": 1,
            "orchestrator_profile": "factory-orchestrator" if kanban_worker else "",
            "default_assignee": profile_id if kanban_worker else "",
            "auto_decompose": True,
            "auto_decompose_per_tick": 3,
            "dispatch_stale_timeout_seconds": 14400,
        },
        "goals": {
            "max_turns": min(turns, 20),
        },
        "skills": {
            "external_dirs": [],
            "template_vars": True,
            "inline_shell": False,
            "inline_shell_timeout": 10,
            "guard_agent_created": True if profile_id != "factory-orchestrator" else False,
        },
        "curator": {
            "enabled": True,
            "interval_hours": 168,
            "min_idle_hours": 2,
            "stale_after_days": 30,
            "archive_after_days": 90,
            "backup": {"enabled": True, "keep": 5},
        },
        "honcho": {},
        "display": {
            "compact": True if compression_threshold > 0.5 else False,
            "personality": "kawaii",
            "resume_display": "full",
            "resume_exchanges": 5 if compression_threshold > 0.5 else 10,
            "resume_max_user_chars": 300,
            "resume_max_assistant_chars": 200,
            "resume_max_assistant_lines": 3,
            "resume_skip_tool_only": True,
            "busy_input_mode": "interrupt",
            "show_reasoning": False,
            "streaming": False,
            "timestamps": False,
            "final_response_markdown": "strip",
            "persistent_output": True,
            "persistent_output_max_lines": 100 if compression_threshold > 0.5 else 200,
            "inline_diffs": True,
            "file_mutation_verifier": True,
            "show_cost": False,
            "skin": "zeus",
            "language": "en",
            "tui_status_indicator": "kaomoji",
        },
        "approvals": {
            "mode": "manual",
            "timeout": 60,
            "cron_mode": "deny",
            "mcp_reload_confirm": True,
            "destructive_slash_confirm": False,
        },
        "security": {
            "allow_private_urls": False,
            "redact_secrets": True,
            "tirith_enabled": True,
            "tirith_path": "tirith",
            "tirith_timeout": 5,
            "tirith_fail_open": True,
            "website_blocklist": {"enabled": False, "domains": [], "shared_files": []},
            "acked_advisories": [],
            "allow_lazy_installs": True,
        },
        "cron": {
            "wrap_response": True,
            "max_parallel_jobs": None,
        },
        "logging": {
            "level": "INFO",
            "max_size_mb": 5,
            "backup_count": 3,
        },
        "sessions": {
            "auto_prune": False,
            "retention_days": 14 if mem_budget > 2000 else 7,
            "vacuum_after_prune": True,
            "min_interval_hours": 24,
            "write_json_snapshots": False,
        },
        "terminal": {
            "backend": "local",
            "modal_mode": "auto",
            "cwd": ".",
            "timeout": 300 if compression_threshold > 0.5 else 180,
            "env_passthrough": ["FLEET_REGISTRY_DATABASE_URL"],
            "shell_init_files": [],
            "auto_source_bashrc": True,
            "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "docker_forward_env": [],
            "docker_env": {},
            "singularity_image": "docker://nikolaik/python-nodejs:python3.11-nodejs20",
            "modal_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "vercel_runtime": "node24",
            "container_cpu": 1,
            "container_memory": 5120,
            "container_disk": 51200,
            "container_persistent": True,
            "docker_volumes": [],
            "docker_mount_cwd_to_workspace": False,
            "docker_extra_args": [],
            "docker_run_as_host_user": False,
            "persistent_shell": True,
        },
        "timezone": "",
        "web": {"backend": "", "search_backend": "", "extract_backend": ""},
        "browser": {
            "inactivity_timeout": 120,
            "command_timeout": 30,
            "record_sessions": False,
            "allow_private_urls": False,
            "engine": "auto",
            "auto_local_for_private_urls": True,
            "cdp_url": "",
            "dialog_policy": "must_respond",
            "dialog_timeout_s": 300,
        },
        "context": {"engine": "compressor"},
        "prompt_caching": {"cache_ttl": "5m"},
        "privacy": {"redact_pii": False},
        "model_catalog": {
            "enabled": True,
            "url": "https://hermes-agent.nousresearch.com/docs/api/model-catalog.json",
            "ttl_hours": 24,
            "providers": {},
        },
        "network": {"force_ipv4": False},
        "lsp": {
            "enabled": True,
            "wait_mode": "document",
            "wait_timeout": 5,
            "install_strategy": "auto",
            "servers": {},
        },
        "x_search": {
            "model": "grok-4.20-reasoning",
            "timeout_seconds": 180,
            "retries": 2,
        },
        "secrets": {
            "bitwarden": {
                "enabled": False,
                "access_token_env": "BWS_ACCESS_TOKEN",
                "project_id": "",
                "cache_ttl_seconds": 300,
                "override_existing": True,
                "auto_install": True,
                "server_url": "",
            },
        },
        "session_reset": {
            "mode": "both",
            "idle_minutes": 1440,
            "at_hour": 4,
        },
        "_config_version": 23,
    }

    # Only orchestrator and builders need TTS/STT
    if profile_id not in ("factory-orchestrator", "claude-builder", "codex-builder", "devops-release"):
        config["tts"] = {"provider": "edge"}
        config["stt"] = {"enabled": False}
        config["voice"] = {"record_key": "", "max_recording_seconds": 0, "auto_tts": False}
        config["image_gen"] = {"provider": ""}

    return config


def main() -> int:
    import json
    import yaml

    print("=" * 60)
    print("SitioUno Software Factory — Phase 5: Profile Optimization")
    print("=" * 60)

    results: dict[str, str] = {}
    errors: list[str] = []

    for profile_id, spec in PROFILES.items():
        profile_dir = PROFILES_DIR / profile_id
        ensure_dir(profile_dir)
        display_name = spec[0]

        try:
            # 1. Write profile config.yaml
            config = build_profile_yaml(profile_id, spec)
            config_path = profile_dir / "config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            # 2. Write SOUL.md
            soul = generate_soul(profile_id, spec)
            soul_path = profile_dir / "SOUL.md"
            with open(soul_path, "w", encoding="utf-8") as f:
                f.write(soul)

            results[profile_id] = f"OK (provider={spec[3]}, model={spec[2]}, turns={spec[6]}, mem={spec[7]}, comp={spec[8]}, kanban={spec[9]})"

        except Exception as e:
            errors.append(f"{profile_id}: {e}")
            results[profile_id] = f"ERROR: {e}"

    # Summary
    print(f"\n{len(results)} profiles configured:")
    for pid, status in sorted(results.items()):
        mark = "✓" if status.startswith("OK") else "✗"
        print(f"  {mark} {pid}: {status}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    # Export config for commit verification
    summary = {}
    for pid, (n, d, m, p, *_) in PROFILES.items():
        summary[pid] = m
    ensure_dir(SCRIPTS_DIR)
    export_path = SCRIPTS_DIR / "factory_profiles_summary.json"
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary exported to {export_path}")

    print("\nDone. All 12 profiles configured with optimal settings.")
    print("\nModel assignment summary:")
    for pid, model in sorted(summary.items()):
        print(f"  {pid:30s} → {model}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
