package com.tencent.qidian.tmc.model.customer;

/**
 * Customer list request.
 */
public class CustomerListRequest {
    private Integer pageNo;
    private Integer pageSize;
    private String keyword;

    public Integer getPageNo() {
        return pageNo;
    }

    public void setPageNo(Integer pageNo) {
        this.pageNo = pageNo;
    }

    public Integer getPageSize() {
        return pageSize;
    }

    public void setPageSize(Integer pageSize) {
        this.pageSize = pageSize;
    }

    public String getKeyword() {
        return keyword;
    }

    public void setKeyword(String keyword) {
        this.keyword = keyword;
    }
}
