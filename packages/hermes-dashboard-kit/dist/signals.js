export function dashboardToneForHealth(state) {
    if (state === "healthy")
        return "success";
    if (state === "degraded")
        return "warning";
    if (state === "critical")
        return "critical";
    return "unknown";
}
export function dashboardToneForSeverity(severity) {
    if (severity === "critical")
        return "critical";
    if (severity === "high")
        return "warning";
    if (severity === "low")
        return "neutral";
    return "info";
}
export function dashboardFreshness(checkedAt, now = Date.now()) {
    if (!checkedAt)
        return "unknown";
    const timestamp = Date.parse(checkedAt);
    if (Number.isNaN(timestamp))
        return "unknown";
    const ageMs = now - timestamp;
    if (ageMs <= 15 * 60 * 1000)
        return "fresh";
    if (ageMs <= 60 * 60 * 1000)
        return "aging";
    return "stale";
}
export function dashboardHealthScore(snapshot) {
    if (typeof snapshot.health.score === "number")
        return snapshot.health.score;
    if (snapshot.health.state === "healthy")
        return 85;
    if (snapshot.health.state === "degraded")
        return 65;
    if (snapshot.health.state === "critical")
        return 35;
    return 55;
}
//# sourceMappingURL=signals.js.map