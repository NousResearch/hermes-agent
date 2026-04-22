package com.tencent.qidian.tmc.model.tag;

/**
 * Tag create response.
 */
public class TagCreateResponse {
    private Long tagId;
    private Boolean success;

    public Long getTagId() {
        return tagId;
    }

    public void setTagId(Long tagId) {
        this.tagId = tagId;
    }

    public Boolean getSuccess() {
        return success;
    }

    public void setSuccess(Boolean success) {
        this.success = success;
    }
}
