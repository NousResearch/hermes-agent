import { CheckCircle2, CircleAlert, GalleryVerticalEnd, GitBranch, Layers3 } from "lucide-react";
import { KpiCard, MetricGrid, StatusPill } from "./metrics";
import { DashboardSection } from "./shell";
import { assessDashboardPrototypeSet, type DashboardPrototypeDataRequirement, type DashboardPrototypeSet, type DashboardPrototypeVariant } from "./prototype-lab";

function stateTone(state: DashboardPrototypeDataRequirement["currentState"]) {
  if (state === "available") return "success";
  if (state === "missing") return "critical";
  if (state === "partial") return "warning";
  return "unknown";
}

function statusTone(status: DashboardPrototypeVariant["status"]) {
  if (status === "approved" || status === "promoted") return "success";
  if (status === "rejected") return "critical";
  if (status === "review") return "warning";
  return "info";
}

export function DashboardPrototypeReview({ prototypeSets }: { prototypeSets: DashboardPrototypeSet[] }) {
  const assessments = prototypeSets.map(assessDashboardPrototypeSet);
  const variants = prototypeSets.flatMap((set) => set.variants);
  const missingRequired = assessments.flatMap((assessment) => assessment.missingRequirements);
  const readyForReview = assessments.filter((assessment) => assessment.readyForReview).length;
  const workspaceFocusCount = new Set(variants.flatMap((variant) => variant.workspaceFocus)).size;

  return (
    <div className="space-y-5">
      <MetricGrid columns={4}>
        <KpiCard label="Prototype Sets" value={prototypeSets.length} detail="proving dashboards first" icon={GalleryVerticalEnd} tone="info" />
        <KpiCard label="Variants" value={variants.length} detail="3 per dashboard target" icon={Layers3} tone="success" />
        <KpiCard label="Ready Review" value={readyForReview} detail="operator questions + variants present" icon={CheckCircle2} tone={readyForReview === prototypeSets.length ? "success" : "warning"} />
        <KpiCard label="Missing Data" value={missingRequired.length} detail="required requirements not available" icon={CircleAlert} tone={missingRequired.length ? "warning" : "success"} />
        <KpiCard label="Workspace Focus" value={workspaceFocusCount} detail="six-workspace coverage" icon={GitBranch} tone={workspaceFocusCount >= 6 ? "success" : "warning"} />
      </MetricGrid>

      <DashboardSection
        id="overview"
        title="Promotion Gate"
        description="A variant should not move into production until workflow, data requirements, and promotion actions are explicit."
      >
        <div className="grid gap-3 lg:grid-cols-2">
          {assessments.map((assessment) => (
            <article key={assessment.projectId} className="rounded-lg border border-border bg-card p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="text-sm font-semibold text-foreground">{assessment.dashboardName}</h3>
                  <p className="mt-1 text-sm text-muted-foreground">{assessment.projectId}</p>
                </div>
                <StatusPill tone={assessment.readyForReview ? "success" : "warning"}>
                  {assessment.readyForReview ? "review-ready" : "needs setup"}
                </StatusPill>
              </div>
              <div className="mt-4 grid gap-2">
                {assessment.selectedVariant ? (
                  <div className="rounded-md border border-border bg-background p-3">
                    <p className="text-xs font-semibold uppercase text-muted-foreground">Selected Direction</p>
                    <p className="mt-1 text-sm font-semibold text-foreground">{assessment.selectedVariant.name}</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {prototypeSets.find((set) => set.projectId === assessment.projectId)?.selectionRationale}
                    </p>
                  </div>
                ) : null}
                {assessment.promotionActions.length ? assessment.promotionActions.map((action) => (
                  <div key={action} className="flex gap-2 rounded-md border border-border bg-background p-2 text-sm text-muted-foreground">
                    <CircleAlert className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
                    <span>{action}</span>
                  </div>
                )) : (
                  <div className="flex gap-2 rounded-md border border-border bg-background p-2 text-sm text-muted-foreground">
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" />
                    <span>Ready for visual review. Select a variant only after the dashboard owner signs off on workflow and data requirements.</span>
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>
      </DashboardSection>

      {prototypeSets.map((set) => (
        <DashboardSection key={set.id} id={set.id} title={set.dashboardName} description={set.objective}>
          <div className="mb-4 flex flex-wrap gap-2">
            {set.operatorQuestions.map((question) => (
              <span key={question} className="rounded-full border border-border bg-background px-3 py-1 text-xs text-muted-foreground">
                {question}
              </span>
            ))}
          </div>

          <div className="grid gap-3 xl:grid-cols-3">
            {set.variants.map((variant) => (
              <article key={variant.id} className="flex min-h-[440px] flex-col rounded-lg border border-border bg-card">
                <div className="border-b border-border bg-muted/30 p-4">
                  <div className="flex items-start justify-between gap-3">
                    <h3 className="text-sm font-semibold text-foreground">{variant.name}</h3>
                    <StatusPill tone={statusTone(variant.status)}>{variant.status}</StatusPill>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {variant.workspaceFocus.map((workspace) => (
                      <StatusPill key={workspace} tone="info">{workspace}</StatusPill>
                    ))}
                  </div>
                </div>

                <div className="grid flex-1 gap-4 p-4">
                  <div>
                    <h4 className="text-xs font-semibold uppercase text-muted-foreground">Operator Workflow</h4>
                    <p className="mt-2 text-sm text-muted-foreground">{variant.operatorWorkflow}</p>
                  </div>

                  <div>
                    <h4 className="text-xs font-semibold uppercase text-muted-foreground">Reference Notes</h4>
                    <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-muted-foreground">
                      {variant.referenceNotes.map((note) => <li key={note}>{note}</li>)}
                    </ul>
                  </div>

                  <div>
                    <h4 className="text-xs font-semibold uppercase text-muted-foreground">Data Requirements</h4>
                    <div className="mt-2 space-y-2">
                      {variant.dataRequirements.map((requirement) => (
                        <div key={requirement.id} className="flex items-center justify-between gap-3 rounded-md border border-border bg-background p-2">
                          <div className="min-w-0">
                            <p className="truncate text-sm text-foreground">{requirement.label}</p>
                            <p className="text-xs text-muted-foreground">{requirement.required ? "required" : "optional"} · {requirement.owner}</p>
                          </div>
                          <StatusPill tone={stateTone(requirement.currentState)}>{requirement.currentState}</StatusPill>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {variant.previewEvidence?.length || variant.promotedComponents?.length ? (
                  <div className="border-t border-border p-4">
                    {variant.previewEvidence?.length ? (
                      <div>
                        <h4 className="text-xs font-semibold uppercase text-muted-foreground">Preview Evidence</h4>
                        <div className="mt-2 space-y-2">
                          {variant.previewEvidence.map((evidence) => (
                            <div key={evidence.id} className="rounded-md border border-border bg-background p-2 text-sm text-muted-foreground">
                              <span className="font-medium text-foreground">{evidence.label}</span>
                              <span className="block text-xs">{evidence.kind} · {evidence.path}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                    {variant.promotedComponents?.length ? (
                      <div className="mt-4">
                        <h4 className="text-xs font-semibold uppercase text-muted-foreground">Promotion Targets</h4>
                        <div className="mt-2 flex flex-wrap gap-2">
                          {variant.promotedComponents.map((component) => (
                            <StatusPill key={component} tone="info">{component}</StatusPill>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </article>
            ))}
          </div>
        </DashboardSection>
      ))}
    </div>
  );
}
