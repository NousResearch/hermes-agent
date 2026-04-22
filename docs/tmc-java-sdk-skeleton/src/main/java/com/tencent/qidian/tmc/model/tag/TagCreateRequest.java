package com.tencent.qidian.tmc.model.tag;

/**
 * Tag create request.
 */
public class TagCreateRequest {
    private String tagName;
    private String tagCode;

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
