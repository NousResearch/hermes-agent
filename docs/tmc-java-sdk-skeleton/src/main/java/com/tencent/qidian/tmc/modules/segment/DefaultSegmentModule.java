package com.tencent.qidian.tmc.modules.segment;

import com.tencent.qidian.tmc.core.TmcHttpClient;
import com.tencent.qidian.tmc.model.common.AsyncTaskStatusResponse;
import com.tencent.qidian.tmc.model.segment.SegmentCreateRequest;
import com.tencent.qidian.tmc.model.segment.SegmentCreateResponse;
import com.tencent.qidian.tmc.model.segment.SegmentListRequest;
import com.tencent.qidian.tmc.model.segment.SegmentListResponse;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Segment/crowd domain APIs mapped from the official TMC OpenAPI documentation.
 */
public class DefaultSegmentModule implements SegmentModule {
    /** Official path: import crowd create v2. */
    public static final String CREATE_IMPORT_SEGMENT_V2_PATH = "/cdp-crowd/import/v2";
    /** Official path: query crowd list. */
    public static final String QUERY_SEGMENT_LIST_PATH = "/cdp-crowd/queryList";
    /** Common async task query path used by import-style APIs. */
    public static final String TASK_QUERY_PATH = "/openapi/task/status";

    private final TmcHttpClient httpClient;

    public DefaultSegmentModule(TmcHttpClient httpClient) {
        this.httpClient = httpClient;
    }

    @Override
    public SegmentCreateResponse createImportSegment(SegmentCreateRequest request) {
        return httpClient.post(CREATE_IMPORT_SEGMENT_V2_PATH, request, SegmentCreateResponse.class);
    }

    @Override
    public SegmentListResponse list(SegmentListRequest request) {
        Map<String, Object> query = new LinkedHashMap<String, Object>();
        query.put("pageNo", request.getPageNo());
        query.put("pageSize", request.getPageSize());
        return httpClient.get(QUERY_SEGMENT_LIST_PATH, query, SegmentListResponse.class);
    }

    @Override
    public AsyncTaskStatusResponse queryTask(String taskId) {
        Map<String, Object> query = new LinkedHashMap<String, Object>();
        query.put("taskId", taskId);
        return httpClient.get(TASK_QUERY_PATH, query, AsyncTaskStatusResponse.class);
    }
}
