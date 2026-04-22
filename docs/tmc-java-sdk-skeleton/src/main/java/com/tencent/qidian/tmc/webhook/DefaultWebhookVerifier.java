package com.tencent.qidian.tmc.webhook;

import com.tencent.qidian.tmc.auth.WebhookSigner;
import com.tencent.qidian.tmc.exception.TmcWebhookVerifyException;

/**
 * Default verifier based on query params and signer.
 */
public class DefaultWebhookVerifier implements WebhookVerifier {
    private final String secretKey;
    private final WebhookSigner webhookSigner;

    public DefaultWebhookVerifier(String secretKey, WebhookSigner webhookSigner) {
        this.secretKey = secretKey;
        this.webhookSigner = webhookSigner;
    }

    @Override
    public void verify(WebhookRequest request) {
        String curTime = request.getQueryParams().get("cur_time");
        String sign = request.getQueryParams().get("sign");
        boolean passed = webhookSigner.verify(request.getRequestUrl(), curTime, sign, secretKey);
        if (!passed) {
            throw new TmcWebhookVerifyException("Webhook signature verification failed.");
        }
    }
}
