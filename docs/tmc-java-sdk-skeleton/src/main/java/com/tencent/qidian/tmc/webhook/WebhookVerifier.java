package com.tencent.qidian.tmc.webhook;

/**
 * Verifies webhook request identity.
 */
public interface WebhookVerifier {
    void verify(WebhookRequest request);
}
