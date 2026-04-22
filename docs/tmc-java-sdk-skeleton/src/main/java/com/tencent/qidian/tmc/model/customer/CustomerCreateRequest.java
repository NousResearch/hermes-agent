package com.tencent.qidian.tmc.model.customer;

/**
 * Customer create request.
 */
public class CustomerCreateRequest {
    /** Customer unique identifier. */
    private String customerId;
    /** Customer display name. */
    private String customerName;
    /** Mobile phone number. */
    private String mobile;

    public static Builder builder() {
        return new Builder();
    }

    public String getCustomerId() {
        return customerId;
    }

    public String getCustomerName() {
        return customerName;
    }

    public String getMobile() {
        return mobile;
    }

    public static final class Builder {
        private final CustomerCreateRequest target = new CustomerCreateRequest();

        public Builder customerId(String customerId) {
            target.customerId = customerId;
            return this;
        }

        public Builder customerName(String customerName) {
            target.customerName = customerName;
            return this;
        }

        public Builder mobile(String mobile) {
            target.mobile = mobile;
            return this;
        }

        public CustomerCreateRequest build() {
            return target;
        }
    }
}
