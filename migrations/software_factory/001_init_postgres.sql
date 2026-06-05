-- SitioUno Software Factory schema for Cloud SQL / PostgreSQL
-- Idempotent migration. Safe to run multiple times.

CREATE SCHEMA IF NOT EXISTS software_factory;

CREATE TABLE IF NOT EXISTS software_factory.factory_projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    repo_path TEXT,
    repo_remote TEXT,
    base_branch TEXT,
    status TEXT NOT NULL DEFAULT 'intake',
    autonomy_level INTEGER NOT NULL DEFAULT 3,
    methodology TEXT NOT NULL DEFAULT 'dual_lane',
    risk_level TEXT NOT NULL DEFAULT 'medium',
    human_owner TEXT,
    summary TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS software_factory.factory_agents (
    agent_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL,
    preferred_engine TEXT,
    toolsets_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    skills_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    greenlight_required_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS software_factory.factory_lanes (
    lane_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES software_factory.factory_projects(project_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    methodology TEXT NOT NULL,
    kanban_board TEXT,
    branch TEXT,
    worktree_path TEXT,
    status TEXT NOT NULL DEFAULT 'planned',
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS software_factory.factory_tasks (
    task_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES software_factory.factory_projects(project_id) ON DELETE CASCADE,
    lane_id TEXT REFERENCES software_factory.factory_lanes(lane_id) ON DELETE SET NULL,
    kanban_id TEXT,
    title TEXT NOT NULL,
    description TEXT,
    phase TEXT NOT NULL DEFAULT 'intake',
    status TEXT NOT NULL DEFAULT 'todo',
    owner_agent_id TEXT REFERENCES software_factory.factory_agents(agent_id) ON DELETE SET NULL,
    reviewer_agent_id TEXT REFERENCES software_factory.factory_agents(agent_id) ON DELETE SET NULL,
    engine TEXT NOT NULL DEFAULT 'zeus',
    priority INTEGER NOT NULL DEFAULT 100,
    dependencies_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    acceptance_criteria_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_required BOOLEAN NOT NULL DEFAULT true,
    evidence_status TEXT NOT NULL DEFAULT 'missing',
    risk_level TEXT NOT NULL DEFAULT 'medium',
    branch TEXT,
    worktree_path TEXT,
    result_summary TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS software_factory.factory_events (
    event_id BIGSERIAL PRIMARY KEY,
    project_id TEXT REFERENCES software_factory.factory_projects(project_id) ON DELETE CASCADE,
    lane_id TEXT REFERENCES software_factory.factory_lanes(lane_id) ON DELETE SET NULL,
    task_id TEXT REFERENCES software_factory.factory_tasks(task_id) ON DELETE SET NULL,
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS software_factory.factory_gates (
    gate_id BIGSERIAL PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES software_factory.factory_projects(project_id) ON DELETE CASCADE,
    lane_id TEXT REFERENCES software_factory.factory_lanes(lane_id) ON DELETE SET NULL,
    task_id TEXT REFERENCES software_factory.factory_tasks(task_id) ON DELETE SET NULL,
    gate_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    reviewer TEXT,
    evidence_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS software_factory.factory_artifacts (
    artifact_id BIGSERIAL PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES software_factory.factory_projects(project_id) ON DELETE CASCADE,
    lane_id TEXT REFERENCES software_factory.factory_lanes(lane_id) ON DELETE SET NULL,
    task_id TEXT REFERENCES software_factory.factory_tasks(task_id) ON DELETE SET NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    checksum TEXT,
    created_by TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS software_factory.factory_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES software_factory.factory_projects(project_id) ON DELETE CASCADE,
    lane_id TEXT REFERENCES software_factory.factory_lanes(lane_id) ON DELETE SET NULL,
    task_id TEXT REFERENCES software_factory.factory_tasks(task_id) ON DELETE SET NULL,
    engine TEXT,
    duration_seconds INTEGER,
    files_changed INTEGER,
    lines_added INTEGER,
    lines_removed INTEGER,
    tests_total INTEGER,
    tests_passed INTEGER,
    tests_failed INTEGER,
    review_findings_count INTEGER,
    rework_count INTEGER,
    cost_estimate_usd NUMERIC(12,4),
    spec_score NUMERIC(5,2),
    quality_score NUMERIC(5,2),
    security_score NUMERIC(5,2),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS software_factory.factory_engine_benchmarks (
    benchmark_id BIGSERIAL PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES software_factory.factory_projects(project_id) ON DELETE CASCADE,
    lane_id TEXT REFERENCES software_factory.factory_lanes(lane_id) ON DELETE SET NULL,
    task_group_id TEXT,
    task_id TEXT REFERENCES software_factory.factory_tasks(task_id) ON DELETE SET NULL,
    engine TEXT NOT NULL,
    success BOOLEAN NOT NULL DEFAULT false,
    score_spec NUMERIC(5,2),
    score_quality NUMERIC(5,2),
    score_tests NUMERIC(5,2),
    score_speed NUMERIC(5,2),
    score_cost NUMERIC(5,2),
    score_maintainability NUMERIC(5,2),
    winner BOOLEAN NOT NULL DEFAULT false,
    reviewer_notes TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS software_factory.factory_method_benchmarks (
    benchmark_id BIGSERIAL PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES software_factory.factory_projects(project_id) ON DELETE CASCADE,
    methodology TEXT NOT NULL,
    success BOOLEAN NOT NULL DEFAULT false,
    score_spec NUMERIC(5,2),
    score_architecture NUMERIC(5,2),
    score_quality NUMERIC(5,2),
    score_tests NUMERIC(5,2),
    score_security NUMERIC(5,2),
    score_speed NUMERIC(5,2),
    score_overhead NUMERIC(5,2),
    winner BOOLEAN NOT NULL DEFAULT false,
    reviewer_notes TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS software_factory.factory_decisions (
    decision_id BIGSERIAL PRIMARY KEY,
    project_id TEXT REFERENCES software_factory.factory_projects(project_id) ON DELETE CASCADE,
    lane_id TEXT REFERENCES software_factory.factory_lanes(lane_id) ON DELETE SET NULL,
    task_id TEXT REFERENCES software_factory.factory_tasks(task_id) ON DELETE SET NULL,
    context TEXT NOT NULL,
    options_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    chosen_option TEXT NOT NULL,
    rationale TEXT NOT NULL,
    risk TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS software_factory.factory_runs (
    run_id BIGSERIAL PRIMARY KEY,
    project_id TEXT REFERENCES software_factory.factory_projects(project_id) ON DELETE CASCADE,
    lane_id TEXT REFERENCES software_factory.factory_lanes(lane_id) ON DELETE SET NULL,
    task_id TEXT REFERENCES software_factory.factory_tasks(task_id) ON DELETE SET NULL,
    agent_id TEXT REFERENCES software_factory.factory_agents(agent_id) ON DELETE SET NULL,
    engine TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    command TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    log_path TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS software_factory.factory_blockers (
    blocker_id BIGSERIAL PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES software_factory.factory_projects(project_id) ON DELETE CASCADE,
    lane_id TEXT REFERENCES software_factory.factory_lanes(lane_id) ON DELETE SET NULL,
    task_id TEXT REFERENCES software_factory.factory_tasks(task_id) ON DELETE SET NULL,
    blocker_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    summary TEXT NOT NULL,
    resolution TEXT,
    owner TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_factory_lanes_project ON software_factory.factory_lanes(project_id);
CREATE INDEX IF NOT EXISTS idx_factory_tasks_project_lane ON software_factory.factory_tasks(project_id, lane_id);
CREATE INDEX IF NOT EXISTS idx_factory_tasks_status ON software_factory.factory_tasks(status);
CREATE INDEX IF NOT EXISTS idx_factory_events_project_created ON software_factory.factory_events(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_factory_gates_project_status ON software_factory.factory_gates(project_id, status);
CREATE INDEX IF NOT EXISTS idx_factory_artifacts_project ON software_factory.factory_artifacts(project_id);
CREATE INDEX IF NOT EXISTS idx_factory_runs_status ON software_factory.factory_runs(status);
CREATE INDEX IF NOT EXISTS idx_factory_blockers_status ON software_factory.factory_blockers(status);

INSERT INTO software_factory.factory_agents (agent_id, display_name, role, preferred_engine, toolsets_json, skills_json, greenlight_required_json)
VALUES
('factory-orchestrator', 'Factory Orchestrator', 'Intake, routing, gates, metrics, reports', 'zeus', '["kanban","delegation","terminal","file","cronjob","skills","web"]', '["software-factory-orchestration","kanban-orchestrator","programming-delegation-engines"]', '["merge","deploy","destructive","credential-change"]'),
('product-analyst', 'Product Analyst', 'Functional analysis, PRD, acceptance criteria', 'zeus', '["file","web","session_search","skills"]', '["writing-plans"]', '["publish"]'),
('solution-architect', 'Solution Architect', 'Architecture, boundaries, integration design', 'claude_code', '["terminal","file","web","skills"]', '["writing-plans","codebase-inspection"]', '["architecture-approval"]'),
('implementation-planner', 'Implementation Planner', 'Epics, stories, dependencies, task graph', 'zeus', '["kanban","file","skills"]', '["writing-plans","software-factory-orchestration"]', '[]'),
('claude-builder', 'Claude Builder', 'Complex implementation and refactors', 'claude_code', '["terminal","file","web","skills"]', '["claude-code","test-driven-development"]', '["push","merge"]'),
('codex-builder', 'Codex Builder', 'Bounded fixes, tests, QA on diffs', 'codex', '["terminal","file","web","skills"]', '["codex","test-driven-development"]', '["push","merge"]'),
('openhands-lab', 'OpenHands Lab', 'Sandbox experiments and independent validation', 'openhands', '["terminal","file","web","skills"]', '["openhands-gcp"]', '["external-write"]'),
('quality-reviewer', 'Quality Reviewer', 'Independent spec and quality gate', 'codex', '["terminal","file","web","skills"]', '["requesting-code-review","github-code-review"]', '["approve-merge"]'),
('security-reviewer', 'Security Reviewer', 'Security and fintech/PII gates', 'codex', '["terminal","file","web","skills"]', '["requesting-code-review","systematic-debugging"]', '["security-waiver"]'),
('qa-verifier', 'QA Verifier', 'Smoke tests and evidence capture', 'zeus', '["terminal","file","browser","vision","skills"]', '["dogfood"]', '["waive-tests"]'),
('devops-release', 'DevOps Release', 'CI, environments, release readiness', 'claude_code', '["terminal","file","web","skills"]', '["github-pr-workflow"]', '["deploy","credential-change"]'),
('factory-reporter', 'Factory Reporter', 'Executive reports and benchmarks', 'zeus', '["file","session_search","skills"]', '["software-factory-orchestration"]', '[]')
ON CONFLICT (agent_id) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    role = EXCLUDED.role,
    preferred_engine = EXCLUDED.preferred_engine,
    toolsets_json = EXCLUDED.toolsets_json,
    skills_json = EXCLUDED.skills_json,
    greenlight_required_json = EXCLUDED.greenlight_required_json,
    updated_at = now();
