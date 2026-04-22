package com.tencent.qidian.tmc.exception;

/**
 * Thrown when remote request is invalid.
 */
public class TmcBadRequestException extends TmcException {
    public TmcBadRequestException(String message, Integer code, String requestId) {
        super(message, code, requestId, null);
    }
}
