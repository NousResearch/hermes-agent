import type { Page, Route } from "@playwright/test";

const jsonHeaders = {
  "access-control-allow-origin": "*",
  "content-type": "application/json",
};

export async function mockDashboardBackend(page: Page) {
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    const response = responseForPath(url.pathname);
    await fulfillJson(route, response.body, response.status);
  });
}

function responseForPath(pathname: string): { body: unknown; status?: number } {
  if (pathname === "/api/auth/me") {
    return {
      body: {
        error: "AUTH_NOT_REQUIRED",
      },
      status: 401,
    };
  }

  if (pathname === "/api/profiles") {
    return {
      body: {
        profiles: [
          {
            name: "default",
            path: "/tmp/hermes-default-profile",
            is_default: true,
            model: null,
            provider: null,
            has_env: false,
            skill_count: 0,
            gateway_running: false,
            description: "Dashboard test profile",
            description_auto: false,
            distribution_name: null,
            distribution_version: null,
            distribution_source: null,
            has_alias: false,
          },
        ],
      },
    };
  }

  if (pathname === "/api/profiles/active") {
    return {
      body: {
        active: "default",
        current: "default",
      },
    };
  }

  if (pathname === "/api/dashboard/plugins") {
    return { body: [] };
  }

  if (pathname === "/api/dashboard/snapshot") {
    return { body: operatingRuntimeDashboardSnapshots()[0] };
  }

  if (pathname === "/api/dashboard/snapshots") {
    return { body: { snapshots: operatingRuntimeDashboardSnapshots() } };
  }

  if (pathname === "/api/status") {
    return {
      body: {
        active_sessions: 0,
        auth_required: false,
        auth_providers: [],
        config_path: "/tmp/hermes-config.yaml",
        config_version: 1,
        env_path: "/tmp/hermes.env",
        gateway_exit_reason: null,
        gateway_health_url: null,
        gateway_pid: null,
        gateway_platforms: {},
        gateway_running: false,
        gateway_state: null,
        gateway_updated_at: null,
        hermes_home: "/tmp/hermes",
        latest_config_version: 1,
        release_date: "2026-07-17",
        version: "test",
      },
    };
  }

  if (pathname === "/api/config") {
    return {
      body: {
        dashboard: {
          show_token_analytics: false,
        },
      },
    };
  }

  if (pathname === "/api/dashboard/themes") {
    return {
      body: {
        active: "default",
        themes: [],
      },
    };
  }

  if (pathname === "/api/dashboard/font") {
    return {
      body: {
        font: "theme",
      },
    };
  }

  if (pathname === "/api/hermes-os/summary") {
    return { body: hermesOsSummary() };
  }

  if (pathname === "/api/operating-runtime/evidence") {
    return {
      body: {
        evidence: operatingRuntimeEvidence(),
      },
    };
  }

  if (pathname === "/api/operating-runtime/audit") {
    return {
      body: {
        audit: [
          {
            id: "audit-test-readiness",
            action: "evaluate-autonomy-readiness",
            actor: "Hermes",
            allowed: false,
            approval: "explicit",
            reason: "Visual-test runtime keeps high-risk autonomy gated.",
            created_at: "2026-07-17T00:00:00.000Z",
          },
        ],
      },
    };
  }

  if (pathname === "/api/operating-runtime/summary") {
    return {
      body: {
        db_path: "/tmp/hermes/operating_runtime.db",
        evidence_count: operatingRuntimeEvidence().length,
        audit_count: 1,
        workbench_count: 1,
        ready_count: 4,
        gated_count: 5,
        latest_audit: null,
      },
    };
  }

  if (pathname === "/api/operating-runtime/readiness-check") {
    return {
      body: {
        decision: {
          allowed: false,
          approval: "explicit",
          reason: "Visual-test runtime keeps high-risk actions gated.",
        },
        audit: {
          id: "audit-test-click",
          action: "evaluate-autonomy-readiness",
          actor: "Hermes operator",
          allowed: false,
          approval: "explicit",
          reason: "Visual-test runtime keeps high-risk actions gated.",
          created_at: "2026-07-17T00:00:00.000Z",
        },
        evidence: {
          id: "evidence-test-click",
          kind: "autonomy",
          subject: "V30 readiness check",
          state: "gated",
          owner: "operator",
          detail: "Visual-test runtime keeps high-risk actions gated.",
          updated_at: "2026-07-17T00:00:00.000Z",
        },
      },
    };
  }

  if (pathname === "/api/operating-runtime/workbench") {
    return {
      body: {
        workbench: [
          {
            id: "workbench-test",
            title: "Visual test workbench item",
            status: "planned",
            owner: "Hermes",
            approval: "explicit",
            artifacts: [],
            report: "",
            updated_at: "2026-07-17T00:00:00.000Z",
          },
        ],
      },
    };
  }

  return { body: { ok: true } };
}

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    body: JSON.stringify(body),
    headers: jsonHeaders,
    status,
  });
}

function hermesOsSummary() {
  return {
    project_path: "/Users/hq/Workspace/projects/nous-hermes-agent",
    panels: [
      {
        panel_id: "architecture-score",
        title: "Architecture Score",
        data: { score: 92, blocked: false },
      },
      {
        panel_id: "architecture-gaps",
        title: "Architecture Gaps",
        data: { gaps: [] },
      },
      {
        panel_id: "work-graph-summary",
        title: "Work Graph",
        data: {
          node_count: 12,
          blocked_count: 0,
          approval_count: 1,
          assignment_count: 4,
        },
      },
      {
        panel_id: "runtime-delegation",
        title: "Runtime Delegation",
        data: {
          provider: "official-hermes-agent",
          available: true,
          mode: "test",
        },
      },
      {
        panel_id: "agent-assignments",
        title: "Agent Assignments",
        data: {
          assignments_by_agent: {
            planner: 2,
            builder: 2,
          },
          fallback_count: 0,
        },
      },
      {
        panel_id: "task-backlog",
        title: "Task Backlog",
        data: {
          task_count: 2,
          blocked_count: 0,
          tasks: [
            { id: "task-1", title: "Validate dashboard system", status: "ready" },
            { id: "task-2", title: "Document agent plan", status: "pending" },
          ],
        },
      },
      {
        panel_id: "templates",
        title: "Templates",
        data: {
          template_count: 3,
          compile_failure_count: 0,
        },
      },
      {
        panel_id: "dry-run-execution",
        title: "Dry Run",
        data: {
          batch_count: 1,
        },
      },
    ],
  };
}

function operatingRuntimeEvidence() {
  return [
    {
      id: "snapshot-hermes",
      kind: "snapshot",
      subject: "Hermes dashboard",
      state: "ready",
      owner: "Hermes",
      detail: "Dashboard metadata is available.",
      updated_at: "2026-07-17T00:00:00.000Z",
    },
    {
      id: "snapshot-media-engine",
      kind: "snapshot",
      subject: "Media Engine",
      state: "gated",
      owner: "Media Engine",
      detail: "Project-owned snapshot endpoint pending.",
      updated_at: "2026-07-17T00:00:00.000Z",
    },
    {
      id: "memory-decisions",
      kind: "memory",
      subject: "Decision records",
      state: "stored",
      owner: "Hermes",
      detail: "Decisions are persisted by the runtime store.",
      updated_at: "2026-07-17T00:00:00.000Z",
    },
    {
      id: "permission-deploy",
      kind: "permission",
      subject: "Deploy production",
      state: "gated",
      owner: "admin",
      detail: "Explicit approval is required.",
      updated_at: "2026-07-17T00:00:00.000Z",
    },
    {
      id: "model-local",
      kind: "model",
      subject: "Local Codex routing",
      state: "ready",
      owner: "Hermes",
      detail: "Local-first routing is available when reachable.",
      updated_at: "2026-07-17T00:00:00.000Z",
    },
    {
      id: "loop-dry-run",
      kind: "loop",
      subject: "Loop dry run",
      state: "ready",
      owner: "Hermes",
      detail: "Dry-run loop execution can be recorded safely.",
      updated_at: "2026-07-17T00:00:00.000Z",
    },
    {
      id: "business-command",
      kind: "business",
      subject: "TLC business command",
      state: "warning",
      owner: "TLC Capital Group OS",
      detail: "Needs live scorecard feeds.",
      updated_at: "2026-07-17T00:00:00.000Z",
    },
    {
      id: "workbench-plan",
      kind: "workbench",
      subject: "Plan to approval workflow",
      state: "ready",
      owner: "Hermes",
      detail: "Plans can be recorded with approvals.",
      updated_at: "2026-07-17T00:00:00.000Z",
    },
    {
      id: "quality-production",
      kind: "quality",
      subject: "Production URL checks",
      state: "gated",
      owner: "Operations",
      detail: "Needs deployed URL checks.",
      updated_at: "2026-07-17T00:00:00.000Z",
    },
    {
      id: "autonomy-kill-switch",
      kind: "autonomy",
      subject: "Kill switch",
      state: "gated",
      owner: "Operations",
      detail: "Must be wired before scheduled autonomy.",
      updated_at: "2026-07-17T00:00:00.000Z",
    },
  ];
}

function operatingRuntimeDashboardSnapshots() {
  return [
    {
      source: {
        id: "nous-hermes-agent.dashboard",
        label: "Nous Hermes Agent",
        owner: "Hermes",
        category: "control-plane",
        projectName: "Nous Hermes Agent",
        url: "/",
        healthUrl: "/api/status",
      },
      health: {
        state: "healthy",
        score: 88,
        message: "Hermes dashboard snapshot endpoint is live.",
        checkedAt: "2026-07-17T00:00:00.000Z",
        freshness: "fresh",
      },
      cost: {
        period: "unknown",
        known: false,
        message: "Hermes dashboard cost telemetry is not attached yet.",
      },
      capacity: {
        known: true,
        used: 1,
        limit: 1,
        pressure: "low",
        message: "Dashboard API is serving the standard snapshot contract.",
      },
      queue: { queued: 0, running: 1, failed: 0, blocked: 0, stale: 0, completed: 0 },
      actions: [],
      deployment: {
        environment: "local",
        status: "current",
        version: "test",
        message: "Served by test backend.",
      },
      updatedAt: "2026-07-17T00:00:00.000Z",
    },
    {
      source: {
        id: "khashi-vc.roc",
        label: "Khashi VC ROC",
        owner: "Khashi VC",
        category: "research-operations",
        url: "https://roc.tlccapitalgroup.com/",
        healthUrl: "https://roc.tlccapitalgroup.com/readyz",
      },
      health: {
        state: "degraded",
        score: 68,
        message: "Health URL is registered; live check was not requested.",
        checkedAt: "2026-07-17T00:00:00.000Z",
        freshness: "unknown",
      },
      cost: { period: "unknown", known: false, message: "Project has not exposed a standard cost signal." },
      capacity: { known: false, pressure: "unknown", message: "Project has not exposed a standard capacity signal." },
      queue: { queued: 0, running: 30, failed: 0, blocked: 0, stale: 0, completed: 0 },
      actions: [
        {
          id: "khashi-vc.roc-snapshot-url",
          title: "Add a project-owned dashboard snapshot endpoint.",
          owner: "Khashi VC",
          severity: "normal",
          sourceDashboardId: "khashi-vc.roc",
          source: "DashboardSnapshot contract",
        },
      ],
      deployment: { environment: "production", status: "unknown", message: "Deployment status needs a project deployment signal." },
      updatedAt: "2026-07-17T00:00:00.000Z",
    },
  ];
}
