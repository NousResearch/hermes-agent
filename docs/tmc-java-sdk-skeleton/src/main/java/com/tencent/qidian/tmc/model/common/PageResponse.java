package com.tencent.qidian.tmc.model.common;

import java.util.List;

/**
 * Common page result model.
 *
 * @param <T> record type
 */
public class PageResponse<T> {
    private List<T> records;
    private Long total;
    private Integer pageNo;
    private Integer pageSize;

    public List<T> getRecords() {
        return records;
    }

    public void setRecords(List<T> records) {
        this.records = records;
    }

    public Long getTotal() {
        return total;
    }

    public void setTotal(Long total) {
        this.total = total;
    }

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
}
