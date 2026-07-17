export function dashboardPluginHasSignal(plugin, signal) {
    return plugin.signals.includes(signal);
}
export function dashboardPluginRequiresAdmin(plugin) {
    return plugin.commands.some((command) => command.permission === "admin" || command.riskLevel === "high");
}
//# sourceMappingURL=marketplace.js.map