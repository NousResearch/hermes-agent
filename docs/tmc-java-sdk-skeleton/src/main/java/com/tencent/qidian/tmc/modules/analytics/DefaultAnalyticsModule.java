package com.tencent.qidian.tmc.modules.analytics;

import com.tencent.qidian.tmc.core.TmcHttpClient;
import com.tencent.qidian.tmc.model.analytics.DashboardListRequest;
import com.tencent.qidian.tmc.model.analytics.DashboardListResponse;
import com.tencent.qidian.tmc.model.common.AsyncTaskStatusResponse;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Analytics domain APIs mapped from the official TMC OpenAPI documentation.
 */
public class DefaultAnalyticsModule implements AnalyticsModule {
    /** Official path: query dashboard/panel list. */
    public static final String LIST_PANELS_PATH = "/apiserver/openapi/panels";
    /** Official path: query async analysis result. */
    public static final String QUERY_ANALYSIS_RESULT_PATH = "/apiserver/openapi/analysis/query";

    private final TmcHttpClient httpClient;

    public DefaultAnalyticsModule(TmcHttpClient httpClient) {
        this.httpClient = httpClient;
    }

    @Override
    public DashboardListResponse listDashboards(DashboardListRequest request) {
        Map<String, Object> query = new LinkedHashMap<String, Object>();
        query.put("pageNo", request.getPageNo());
        query.put("pageSize", request.getPageSize());
        return httpClient.get(LIST_PANELS_PATH, query, DashboardListResponse.class);
    }

    @Override
    public AsyncTaskStatusResponse queryAsyncResult(String taskId) {
        Map<String, Object> query = new LinkedHashMap<String, Object>();
        query.put("taskId", taskId);
        return httpClient.get(QUERY_ANALYSIS_RESULT_PATH, query, AsyncTaskStatusResponse.class);
    }
}
