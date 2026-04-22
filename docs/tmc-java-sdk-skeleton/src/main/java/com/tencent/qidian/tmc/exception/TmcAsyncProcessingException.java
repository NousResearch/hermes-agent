package com.tencent.qidian.tmc.exception;

/**
 * Thrown when asynchronous webhook processing fails.
 */
public class TmcAsyncProcessingException extends TmcException {
    public TmcAsyncProcessingException(String message, Throwable cause) {
        super(message, cause);
    }
}
