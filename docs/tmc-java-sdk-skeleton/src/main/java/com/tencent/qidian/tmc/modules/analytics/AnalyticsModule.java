package com.tencent.qidian.tmc.modules.analytics;

import com.tencent.qidian.tmc.model.analytics.DashboardListRequest;
import com.tencent.qidian.tmc.model.analytics.DashboardListResponse;
import com.tencent.qidian.tmc.model.common.AsyncTaskStatusResponse;

/**
 * Analytics APIs.
 */
public interface AnalyticsModule {
    DashboardListResponse listDashboards(DashboardListRequest request);

    AsyncTaskStatusResponse queryAsyncResult(String taskId);
}
