package com.tencent.qidian.tmc.webhook;

/**
 * Handles unknown or unsupported events.
 */
public interface UnknownWebhookEventHandler {
    void handle(TmcWebhookEvent event);
}
