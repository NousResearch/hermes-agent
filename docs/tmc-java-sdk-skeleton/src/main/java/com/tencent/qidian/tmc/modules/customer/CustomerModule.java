package com.tencent.qidian.tmc.modules.customer;

import com.tencent.qidian.tmc.model.customer.CustomerCreateRequest;
import com.tencent.qidian.tmc.model.customer.CustomerCreateResponse;
import com.tencent.qidian.tmc.model.customer.CustomerListRequest;
import com.tencent.qidian.tmc.model.customer.CustomerListResponse;

/**
 * Customer APIs.
 */
public interface CustomerModule {
    CustomerCreateResponse create(CustomerCreateRequest request);

    CustomerListResponse list(CustomerListRequest request);
}
