export interface DashboardThemeTokenSet {
    background: string;
    foreground: string;
    card: string;
    muted: string;
    border: string;
    primary: string;
    accent: string;
    warning: string;
    critical: string;
    success: string;
}
export interface DashboardThemeProfile {
    id: string;
    label: string;
    domain: string;
    density: "compact" | "balanced" | "spacious";
    tone: "executive" | "research" | "publishing" | "analytics" | "system";
    tokens: DashboardThemeTokenSet;
    notes: string[];
}
export declare const dashboardThemeProfiles: DashboardThemeProfile[];
export declare function dashboardThemeById(id: string): DashboardThemeProfile | undefined;
export declare function dashboardThemeCssVariables(theme: DashboardThemeProfile): {
    readonly "--hdk-background": string;
    readonly "--hdk-foreground": string;
    readonly "--hdk-card": string;
    readonly "--hdk-muted": string;
    readonly "--hdk-border": string;
    readonly "--hdk-primary": string;
    readonly "--hdk-accent": string;
    readonly "--hdk-warning": string;
    readonly "--hdk-critical": string;
    readonly "--hdk-success": string;
};
//# sourceMappingURL=themes.d.ts.map