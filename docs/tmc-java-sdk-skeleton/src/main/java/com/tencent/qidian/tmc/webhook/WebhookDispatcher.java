package com.tencent.qidian.tmc.webhook;

/**
 * Dispatches normalized webhook events.
 */
public interface WebhookDispatcher {
    void dispatch(TmcWebhookEvent event);
}
