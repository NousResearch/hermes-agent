import { AlertTriangle, CheckCircle2, Compass, Layers, Route, ShieldAlert } from "lucide-react";
import { DashboardSection } from "./shell";
import { HealthBadge, KpiCard, MetricGrid, StatusPill, type DashboardTone } from "./metrics";
import type { DashboardModuleContract, DashboardOperationalStatus, DashboardSnapshotContract } from "./contracts";
import { summarizeDashboardSnapshot } from "./contracts";
import { assessDashboardArchitecture, type DashboardArchitectureAssessment } from "./strategy";
import { groupDashboardModulesByWorkspace } from "./workspaces";

function toneFromSeverity(status: DashboardOperationalStatus): DashboardTone {
  if (status === "healthy") return "success";
  if (status === "watch") return "warning";
  if (status === "degraded" || status === "blocked") return "critical";
  return "unknown";
}

function moduleSourceSummary(module: DashboardModuleContract) {
  if (!module.dataSources.length) return "No declared sources";
  const failing = module.dataSources.filter((source) => source.status === "blocked" || source.status === "degraded").length;
  return failing ? `${failing} source issue${failing === 1 ? "" : "s"}` : `${module.dataSources.length} source${module.dataSources.length === 1 ? "" : "s"}`;
}

export function DashboardWorkspaceOverview({
  snapshot,
  assessment = assessDashboardArchitecture(snapshot),
}: {
  snapshot: DashboardSnapshotContract;
  assessment?: DashboardArchitectureAssessment;
}) {
  const summary = summarizeDashboardSnapshot(snapshot);
  const workspaceGroups = groupDashboardModulesByWorkspace(snapshot.modules);

  return (
    <div className="space-y-4">
      <MetricGrid columns={4}>
        <KpiCard label="Architecture Status" value={summary.status} tone={toneFromSeverity(summary.status)} icon={ShieldAlert} />
        <KpiCard label="Workspace Coverage" value={`${assessment.workspaceCoveragePercent}%`} detail="Six-workspace IA coverage" tone={assessment.workspaceCoveragePercent >= 80 ? "success" : "warning"} icon={Compass} />
        <KpiCard label="Modules" value={summary.moduleCount} detail={`${summary.degradedSourceCount} degraded sources`} tone={summary.degradedSourceCount ? "warning" : "success"} icon={Layers} />
        <KpiCard label="Alerts" value={summary.alertCount} detail={`${summary.criticalAlertCount} blocked`} tone={summary.criticalAlertCount ? "critical" : summary.alertCount ? "warning" : "success"} icon={AlertTriangle} />
      </MetricGrid>

      <DashboardSection title="Workspace Map" description="Use this to collapse dashboard sprawl into the six Hermes operating workspaces.">
        <div className="grid gap-3 xl:grid-cols-3">
          {workspaceGroups.map((workspace) => (
            <article key={workspace.id} className="rounded-lg border border-border bg-background p-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h3 className="text-sm font-semibold text-foreground">{workspace.label}</h3>
                  <p className="mt-1 text-xs text-muted-foreground">{workspace.primaryQuestion}</p>
                </div>
                <StatusPill tone={workspace.modules.length ? "success" : "unknown"}>{workspace.modules.length}</StatusPill>
              </div>
              <div className="mt-3 space-y-2">
                {workspace.modules.length ? workspace.modules.map((module) => (
                  <div key={module.id} className="rounded-md border border-border p-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-sm font-medium text-foreground">{module.label}</span>
                      <HealthBadge label={module.status} tone={toneFromSeverity(module.status)} />
                    </div>
                    <p className="mt-1 text-xs text-muted-foreground">{module.primaryQuestion}</p>
                    <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      <span>{moduleSourceSummary(module)}</span>
                      {module.route ? <span className="inline-flex items-center gap-1"><Route className="h-3 w-3" />{module.route}</span> : null}
                    </div>
                  </div>
                )) : (
                  <div className="rounded-md border border-dashed border-border p-3 text-sm text-muted-foreground">
                    No modules assigned yet.
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>
      </DashboardSection>

      <DashboardSection title="Capability Gaps" description="The checklist that separates visual inspiration from a dashboard that tells the truth.">
        <div className="grid gap-3 lg:grid-cols-2">
          {assessment.capabilities.map((capability) => (
            <article key={capability.id} className="rounded-lg border border-border bg-background p-3">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold text-foreground">{capability.label}</h3>
                <HealthBadge label={capability.status} tone={toneFromSeverity(capability.status)} />
              </div>
              <p className="mt-2 text-sm text-muted-foreground">{capability.gap}</p>
              <p className="mt-2 text-sm text-foreground"><CheckCircle2 className="mr-1 inline h-4 w-4" />{capability.nextAction}</p>
            </article>
          ))}
        </div>
      </DashboardSection>
    </div>
  );
}
