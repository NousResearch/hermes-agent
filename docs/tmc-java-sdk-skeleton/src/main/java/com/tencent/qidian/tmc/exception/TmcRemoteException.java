package com.tencent.qidian.tmc.exception;

/**
 * Thrown when remote API returns non-success status.
 */
public class TmcRemoteException extends TmcException {
    public TmcRemoteException(String message, Integer code, String requestId) {
        super(message, code, requestId, null);
    }
}
