package com.tencent.qidian.tmc.model.customer;

/**
 * Customer create response.
 */
public class CustomerCreateResponse {
    private String customerId;
    private Boolean success;

    public String getCustomerId() {
        return customerId;
    }

    public void setCustomerId(String customerId) {
        this.customerId = customerId;
    }

    public Boolean getSuccess() {
        return success;
    }

    public void setSuccess(Boolean success) {
        this.success = success;
    }
}
