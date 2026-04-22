package com.tencent.qidian.tmc.model.tag;

/**
 * Tag summary item.
 */
public class TagSummary {
    private Long tagId;
    private String tagName;
    private String tagCode;

    public Long getTagId() {
        return tagId;
    }

    public void setTagId(Long tagId) {
        this.tagId = tagId;
    }

    public String getTagName() {
        return tagName;
    }

    public void setTagName(String tagName) {
        this.tagName = tagName;
    }

    public String getTagCode() {
        return tagCode;
    }

    public void setTagCode(String tagCode) {
        this.tagCode = tagCode;
    }
}
