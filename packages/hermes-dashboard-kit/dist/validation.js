import { HERMES_DASHBOARD_WORKSPACES } from "./workspaces";
function isKnownWorkspace(workspace) {
    return HERMES_DASHBOARD_WORKSPACES.some((item) => item.id === workspace);
}
function pushIssue(issues, severity, code, path, message) {
    issues.push({ severity, code, path, message });
}
function validateDataSource(source, path, issues) {
    if (!source.id)
        pushIssue(issues, "error", "data-source.id.missing", `${path}.id`, "Data source must have an id.");
    if (!source.label)
        pushIssue(issues, "error", "data-source.label.missing", `${path}.label`, "Data source must have a label.");
    if (!source.owner)
        pushIssue(issues, "error", "data-source.owner.missing", `${path}.owner`, "Data source must declare an owner.");
    if (!source.status)
        pushIssue(issues, "error", "data-source.status.missing", `${path}.status`, "Data source must declare status.");
    if (source.freshnessSeconds !== undefined && source.freshnessSeconds < 0) {
        pushIssue(issues, "error", "data-source.freshness.invalid", `${path}.freshnessSeconds`, "Freshness seconds cannot be negative.");
    }
    if ((source.status === "degraded" || source.status === "blocked") && !source.failureMode) {
        pushIssue(issues, "warning", "data-source.failure-mode.missing", `${path}.failureMode`, "Degraded or blocked sources should explain the failure mode.");
    }
}
function validateModule(module, index, issues) {
    const path = `modules[${index}]`;
    if (!module.id)
        pushIssue(issues, "error", "module.id.missing", `${path}.id`, "Module must have an id.");
    if (!module.label)
        pushIssue(issues, "error", "module.label.missing", `${path}.label`, "Module must have a label.");
    if (!isKnownWorkspace(module.workspace)) {
        pushIssue(issues, "error", "module.workspace.unknown", `${path}.workspace`, `Module workspace must be one of: ${HERMES_DASHBOARD_WORKSPACES.map((item) => item.id).join(", ")}.`);
    }
    if (!module.primaryQuestion) {
        pushIssue(issues, "error", "module.primary-question.missing", `${path}.primaryQuestion`, "Module must state the operator question it answers.");
    }
    if (!module.dataSources.length) {
        pushIssue(issues, "warning", "module.data-sources.empty", `${path}.dataSources`, "Module should declare at least one data source.");
    }
    module.dataSources.forEach((source, sourceIndex) => validateDataSource(source, `${path}.dataSources[${sourceIndex}]`, issues));
}
export function validateDashboardSnapshot(snapshot) {
    const issues = [];
    if (!snapshot.id)
        pushIssue(issues, "error", "snapshot.id.missing", "id", "Snapshot must have an id.");
    if (!snapshot.projectId)
        pushIssue(issues, "error", "snapshot.project-id.missing", "projectId", "Snapshot must have a project id.");
    if (!snapshot.generatedAt)
        pushIssue(issues, "error", "snapshot.generated-at.missing", "generatedAt", "Snapshot must have a generated timestamp.");
    if (!snapshot.modules.length)
        pushIssue(issues, "error", "snapshot.modules.empty", "modules", "Snapshot must include dashboard modules.");
    snapshot.modules.forEach((module, index) => validateModule(module, index, issues));
    const moduleIds = new Set();
    snapshot.modules.forEach((module, index) => {
        if (module.id && moduleIds.has(module.id)) {
            pushIssue(issues, "error", "module.id.duplicate", `modules[${index}].id`, `Duplicate module id: ${module.id}.`);
        }
        moduleIds.add(module.id);
    });
    const errorCount = issues.filter((issue) => issue.severity === "error").length;
    const warningCount = issues.filter((issue) => issue.severity === "warning").length;
    return {
        valid: errorCount === 0,
        issueCount: issues.length,
        errorCount,
        warningCount,
        issues,
    };
}
export function validateDashboardAdoptionRegistry(registry) {
    const issues = [];
    if (registry.schemaVersion !== 1) {
        pushIssue(issues, "error", "registry.schema-version.unsupported", "schemaVersion", "Adoption registry schemaVersion must be 1.");
    }
    if (!registry.source?.package) {
        pushIssue(issues, "error", "registry.source.package.missing", "source.package", "Adoption registry must declare the source package.");
    }
    if (!registry.dashboards?.length) {
        pushIssue(issues, "error", "registry.dashboards.empty", "dashboards", "Adoption registry must include at least one dashboard.");
    }
    registry.dashboards.forEach((dashboard, index) => {
        const path = `dashboards[${index}]`;
        if (!dashboard.project)
            pushIssue(issues, "error", "adoption.project.missing", `${path}.project`, "Dashboard adoption entry must have a project.");
        if (!dashboard.name)
            pushIssue(issues, "error", "adoption.name.missing", `${path}.name`, "Dashboard adoption entry must have a name.");
        if (!dashboard.type)
            pushIssue(issues, "error", "adoption.type.missing", `${path}.type`, "Dashboard adoption entry must have a type.");
        if (!dashboard.status)
            pushIssue(issues, "error", "adoption.status.missing", `${path}.status`, "Dashboard adoption entry must have a status.");
        if (!dashboard.targetState)
            pushIssue(issues, "error", "adoption.target-state.missing", `${path}.targetState`, "Dashboard adoption entry must describe target state.");
        if (dashboard.contractCoveragePercent === undefined) {
            pushIssue(issues, "warning", "adoption.contract-coverage.missing", `${path}.contractCoveragePercent`, "Track contract coverage percentage for this dashboard.");
        }
        if (dashboard.workspaceCoveragePercent === undefined) {
            pushIssue(issues, "warning", "adoption.workspace-coverage.missing", `${path}.workspaceCoveragePercent`, "Track workspace coverage percentage for this dashboard.");
        }
        if (dashboard.contractCoveragePercent !== undefined && (dashboard.contractCoveragePercent < 0 || dashboard.contractCoveragePercent > 100)) {
            pushIssue(issues, "error", "adoption.contract-coverage.invalid", `${path}.contractCoveragePercent`, "Contract coverage must be between 0 and 100.");
        }
        if (dashboard.workspaceCoveragePercent !== undefined && (dashboard.workspaceCoveragePercent < 0 || dashboard.workspaceCoveragePercent > 100)) {
            pushIssue(issues, "error", "adoption.workspace-coverage.invalid", `${path}.workspaceCoveragePercent`, "Workspace coverage must be between 0 and 100.");
        }
    });
    const errorCount = issues.filter((issue) => issue.severity === "error").length;
    const warningCount = issues.filter((issue) => issue.severity === "warning").length;
    return {
        valid: errorCount === 0,
        issueCount: issues.length,
        errorCount,
        warningCount,
        issues,
    };
}
//# sourceMappingURL=validation.js.map