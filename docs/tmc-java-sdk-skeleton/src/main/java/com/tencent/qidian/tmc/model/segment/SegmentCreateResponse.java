package com.tencent.qidian.tmc.model.segment;

/**
 * Segment create response.
 */
public class SegmentCreateResponse {
    private String segmentId;
    private String taskId;

    public String getSegmentId() {
        return segmentId;
    }

    public void setSegmentId(String segmentId) {
        this.segmentId = segmentId;
    }

    public String getTaskId() {
        return taskId;
    }

    public void setTaskId(String taskId) {
        this.taskId = taskId;
    }
}
