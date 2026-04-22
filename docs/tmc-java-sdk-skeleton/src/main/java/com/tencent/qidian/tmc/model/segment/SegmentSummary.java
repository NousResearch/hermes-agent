package com.tencent.qidian.tmc.model.segment;

/**
 * Segment summary item.
 */
public class SegmentSummary {
    private String segmentId;
    private String segmentName;
    private String status;

    public String getSegmentId() {
        return segmentId;
    }

    public void setSegmentId(String segmentId) {
        this.segmentId = segmentId;
    }

    public String getSegmentName() {
        return segmentName;
    }

    public void setSegmentName(String segmentName) {
        this.segmentName = segmentName;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }
}
