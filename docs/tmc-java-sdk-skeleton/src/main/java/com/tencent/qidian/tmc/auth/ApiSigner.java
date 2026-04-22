package com.tencent.qidian.tmc.auth;

/**
 * Signs standard TMC OpenAPI requests.
 */
public interface ApiSigner {
    /**
     * Sign by corporationId + timestamp using secret key.
     */
    String sign(String corporationId, String timestamp, String secretKey);
}
