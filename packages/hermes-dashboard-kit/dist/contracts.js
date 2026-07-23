export function clampReadinessPercent(value) {
    if (!Number.isFinite(value))
        return 0;
    return Math.max(0, Math.min(100, Math.round(value)));
}
export function severityRank(status) {
    if (status === "blocked")
        return 4;
    if (status === "degraded")
        return 3;
    if (status === "watch")
        return 2;
    if (status === "unknown")
        return 1;
    return 0;
}
export function worstSeverity(statuses) {
    return statuses.reduce((worst, status) => (severityRank(status) > severityRank(worst) ? status : worst), "healthy");
}
export function summarizeDashboardSnapshot(snapshot) {
    const sourceStatuses = snapshot.modules.flatMap((module) => module.dataSources.map((source) => source.status));
    const status = worstSeverity([
        snapshot.readiness?.status ?? "unknown",
        ...snapshot.modules.map((module) => module.status),
        ...snapshot.alerts.map((alert) => alert.severity),
        ...sourceStatuses,
    ]);
    return {
        projectId: snapshot.projectId,
        generatedAt: snapshot.generatedAt,
        status,
        moduleCount: snapshot.modules.length,
        alertCount: snapshot.alerts.length,
        criticalAlertCount: snapshot.alerts.filter((alert) => alert.severity === "blocked").length,
        degradedSourceCount: sourceStatuses.filter((sourceStatus) => severityRank(sourceStatus) >= severityRank("degraded")).length,
        readinessPercent: snapshot.readiness ? clampReadinessPercent(snapshot.readiness.readinessPercent) : undefined,
    };
}
//# sourceMappingURL=contracts.js.map