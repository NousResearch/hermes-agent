package com.tencent.qidian.tmc.auth;

/**
 * Default HmacSHA256 verifier for webhook callbacks.
 */
public class HmacSha256WebhookSigner extends HmacSha256ApiSigner implements WebhookSigner {
    @Override
    public boolean verify(String url, String curTime, String providedSign, String secretKey) {
        if (url == null || curTime == null || providedSign == null || secretKey == null) {
            return false;
        }
        String expected = hmacBase64(url + "&" + curTime, secretKey);
        return expected.equals(providedSign);
    }

    /**
     * Generates expected callback sign from url and cur_time.
     */
    public String signCallback(String url, String curTime, String secretKey) {
        return hmacBase64(url + "&" + curTime, secretKey);
    }
}
