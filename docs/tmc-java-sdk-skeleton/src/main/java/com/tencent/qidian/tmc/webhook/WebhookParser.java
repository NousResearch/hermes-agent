package com.tencent.qidian.tmc.webhook;

/**
 * Parses raw body into a raw webhook payload.
 */
public interface WebhookParser {
    RawWebhookPayload parse(String rawBody);
}
