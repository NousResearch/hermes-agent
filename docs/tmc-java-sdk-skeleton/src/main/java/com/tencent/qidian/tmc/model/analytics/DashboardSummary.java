package com.tencent.qidian.tmc.model.analytics;

/**
 * Dashboard summary item.
 */
public class DashboardSummary {
    private String dashboardId;
    private String dashboardName;

    public String getDashboardId() {
        return dashboardId;
    }

    public void setDashboardId(String dashboardId) {
        this.dashboardId = dashboardId;
    }

    public String getDashboardName() {
        return dashboardName;
    }

    public void setDashboardName(String dashboardName) {
        this.dashboardName = dashboardName;
    }
}
