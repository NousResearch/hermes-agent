package com.tencent.qidian.tmc.model.segment;

/**
 * Segment create request.
 */
public class SegmentCreateRequest {
    private String segmentName;
    private String sourceFileId;

    public String getSegmentName() {
        return segmentName;
    }

    public void setSegmentName(String segmentName) {
        this.segmentName = segmentName;
    }

    public String getSourceFileId() {
        return sourceFileId;
    }

    public void setSourceFileId(String sourceFileId) {
        this.sourceFileId = sourceFileId;
    }
}
