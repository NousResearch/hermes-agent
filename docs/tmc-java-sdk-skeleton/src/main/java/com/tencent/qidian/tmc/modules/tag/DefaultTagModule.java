package com.tencent.qidian.tmc.modules.tag;

import com.tencent.qidian.tmc.core.TmcHttpClient;
import com.tencent.qidian.tmc.model.tag.TagCreateRequest;
import com.tencent.qidian.tmc.model.tag.TagCreateResponse;
import com.tencent.qidian.tmc.model.tag.TagListRequest;
import com.tencent.qidian.tmc.model.tag.TagListResponse;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Tag domain APIs mapped from the official TMC OpenAPI documentation.
 */
public class DefaultTagModule implements TagModule {
    /** Official path: create external tag definition. */
    public static final String CREATE_EXTERNAL_TAG_PATH = "/cdp-tag/inner-api/open/cdp/tag/tagDefine/external/create";
    /** Official path: query tag definitions. */
    public static final String QUERY_TAG_LIST_PATH = "/cdp-tag/inner-api/open/cdp/tag/tagDefine/queryList";

    private final TmcHttpClient httpClient;

    public DefaultTagModule(TmcHttpClient httpClient) {
        this.httpClient = httpClient;
    }

    @Override
    public TagCreateResponse create(TagCreateRequest request) {
        return httpClient.post(CREATE_EXTERNAL_TAG_PATH, request, TagCreateResponse.class);
    }

    @Override
    public TagListResponse list(TagListRequest request) {
        Map<String, Object> query = new LinkedHashMap<String, Object>();
        query.put("pageNo", request.getPageNo());
        query.put("pageSize", request.getPageSize());
        return httpClient.get(QUERY_TAG_LIST_PATH, query, TagListResponse.class);
    }
}
