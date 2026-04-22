package com.tencent.qidian.tmc.modules.customer;

import com.tencent.qidian.tmc.core.TmcHttpClient;
import com.tencent.qidian.tmc.model.customer.CustomerCreateRequest;
import com.tencent.qidian.tmc.model.customer.CustomerCreateResponse;
import com.tencent.qidian.tmc.model.customer.CustomerListRequest;
import com.tencent.qidian.tmc.model.customer.CustomerListResponse;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Customer domain APIs mapped from the official TMC OpenAPI documentation.
 */
public class DefaultCustomerModule implements CustomerModule {
    /** Official path: create/update customer entity. */
    public static final String CREATE_PATH = "/cdp-entity/user/create";
    /** Official path: query customer list. */
    public static final String QUERY_LIST_PATH = "/cdp-entity/user/queryList";

    private final TmcHttpClient httpClient;

    public DefaultCustomerModule(TmcHttpClient httpClient) {
        this.httpClient = httpClient;
    }

    @Override
    public CustomerCreateResponse create(CustomerCreateRequest request) {
        return httpClient.post(CREATE_PATH, request, CustomerCreateResponse.class);
    }

    @Override
    public CustomerListResponse list(CustomerListRequest request) {
        Map<String, Object> query = new LinkedHashMap<String, Object>();
        query.put("pageNo", request.getPageNo());
        query.put("pageSize", request.getPageSize());
        query.put("keyword", request.getKeyword());
        return httpClient.get(QUERY_LIST_PATH, query, CustomerListResponse.class);
    }
}
