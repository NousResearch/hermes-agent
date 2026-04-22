package com.tencent.qidian.tmc.webhook;

import java.util.List;

/**
 * Normalizes raw payloads to unified webhook events.
 */
public interface WebhookEventNormalizer {
    TmcWebhookEvent normalize(RawWebhookPayload payload);

    List<TmcWebhookEvent> normalizeAll(RawWebhookPayload payload);
}
