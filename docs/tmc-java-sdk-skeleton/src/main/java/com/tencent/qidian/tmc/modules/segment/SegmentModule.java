package com.tencent.qidian.tmc.modules.segment;

import com.tencent.qidian.tmc.model.common.AsyncTaskStatusResponse;
import com.tencent.qidian.tmc.model.segment.SegmentCreateRequest;
import com.tencent.qidian.tmc.model.segment.SegmentCreateResponse;
import com.tencent.qidian.tmc.model.segment.SegmentListRequest;
import com.tencent.qidian.tmc.model.segment.SegmentListResponse;

/**
 * Segment APIs.
 */
public interface SegmentModule {
    SegmentCreateResponse createImportSegment(SegmentCreateRequest request);

    SegmentListResponse list(SegmentListRequest request);

    AsyncTaskStatusResponse queryTask(String taskId);
}
