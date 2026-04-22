package com.tencent.qidian.tmc.webhook;

/**
 * Generic handler for normalized webhook events.
 *
 * @param <E> event type
 */
public interface WebhookEventHandler<E extends TmcWebhookEvent> {
    String supportEventType();

    void handle(E event);
}
