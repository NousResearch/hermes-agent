package com.tencent.qidian.tmc.exception;

/**
 * Thrown when request signing or credentials are invalid.
 */
public class TmcAuthException extends TmcException {
    public TmcAuthException(String message) {
        super(message);
    }
}
