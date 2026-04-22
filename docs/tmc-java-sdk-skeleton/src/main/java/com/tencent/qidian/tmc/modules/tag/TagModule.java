package com.tencent.qidian.tmc.modules.tag;

import com.tencent.qidian.tmc.model.tag.TagCreateRequest;
import com.tencent.qidian.tmc.model.tag.TagCreateResponse;
import com.tencent.qidian.tmc.model.tag.TagListRequest;
import com.tencent.qidian.tmc.model.tag.TagListResponse;

/**
 * Tag APIs.
 */
public interface TagModule {
    TagCreateResponse create(TagCreateRequest request);

    TagListResponse list(TagListRequest request);
}
