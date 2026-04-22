package com.tencent.qidian.tmc.model.common;

/**
 * Common API response envelope.
 *
 * @param <T> typed data payload
 */
public class ApiResponse<T> {
    private Integer code;
    private String message;
    private T data;
    private String requestId;

    public Integer getCode() {
        return code;
    }

    public void setCode(Integer code) {
        this.code = code;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public T getData() {
        return data;
    }

    public void setData(T data) {
        this.data = data;
    }

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }
}
