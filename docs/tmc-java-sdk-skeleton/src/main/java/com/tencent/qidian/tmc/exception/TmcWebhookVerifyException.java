package com.tencent.qidian.tmc.exception;

/**
 * Thrown when webhook verification fails.
 */
public class TmcWebhookVerifyException extends TmcException {
    public TmcWebhookVerifyException(String message) {
        super(message);
    }
}
