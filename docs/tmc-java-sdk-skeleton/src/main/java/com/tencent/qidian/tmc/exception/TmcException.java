package com.tencent.qidian.tmc.exception;

/**
 * Base SDK exception.
 */
public class TmcException extends RuntimeException {
    private final Integer code;
    private final String requestId;

    public TmcException(String message) {
        this(message, null, null, null);
    }

    public TmcException(String message, Throwable cause) {
        this(message, null, null, cause);
    }

    public TmcException(String message, Integer code, String requestId, Throwable cause) {
        super(message, cause);
        this.code = code;
        this.requestId = requestId;
    }

    public Integer getCode() {
        return code;
    }

    public String getRequestId() {
        return requestId;
    }
}
