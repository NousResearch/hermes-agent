package com.tencent.qidian.tmc.auth;

import java.nio.charset.StandardCharsets;
import java.util.Base64;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

/**
 * Default HmacSHA256 signer for TMC OpenAPI.
 */
public class HmacSha256ApiSigner implements ApiSigner {
    private static final String HMAC_SHA256 = "HmacSHA256";

    @Override
    public String sign(String corporationId, String timestamp, String secretKey) {
        String payload = corporationId + timestamp;
        return hmacBase64(payload, secretKey);
    }

    protected String hmacBase64(String payload, String secretKey) {
        try {
            Mac mac = Mac.getInstance(HMAC_SHA256);
            mac.init(new SecretKeySpec(secretKey.getBytes(StandardCharsets.UTF_8), HMAC_SHA256));
            byte[] digest = mac.doFinal(payload.getBytes(StandardCharsets.UTF_8));
            return Base64.getEncoder().encodeToString(digest);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to sign request.", e);
        }
    }
}
