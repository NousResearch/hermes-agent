package com.tencent.qidian.tmc.model.customer;

/**
 * Customer summary item.
 */
public class CustomerSummary {
    private String customerId;
    private String customerName;
    private String mobile;

    public String getCustomerId() {
        return customerId;
    }

    public void setCustomerId(String customerId) {
        this.customerId = customerId;
    }

    public String getCustomerName() {
        return customerName;
    }

    public void setCustomerName(String customerName) {
        this.customerName = customerName;
    }

    public String getMobile() {
        return mobile;
    }

    public void setMobile(String mobile) {
        this.mobile = mobile;
    }
}
